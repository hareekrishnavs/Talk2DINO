#!/usr/bin/env python3
"""Shared VOC2012 V20/V21 matched evaluator: E3 vs matched k11 vs k12,
T=320 finite-step, on the committed VOC2012 dataset-source split.

Processes each of the 1449 canonical validation images exactly once:
one shared backbone/snapshot pass, one shared top-12 graph, one matched
k11 prefix, one k11 propagation, one k12 propagation -- producing three
shared, foreground-only (20-channel) stitched score tensors (E3, k11,
k12) per image. V20 and V21 are then derived from those SAME three
tensors at finalization only: V20 argmaxes them directly (no background
channel); V21 prepends the canonical constant background channel first.
Never a second backbone pass for V20 or V21, never independently
selected/recomputed foreground scores.

Reuses (imports unmodified, never reimplements) the exact finite-step
kernel and E3/graph primitives already verified by the COCO-Object
protocol-confirmation evaluator and the matched k11/k12 power evaluator:
``models.dinotext.cover_dr.coco_object_evaluator`` for the shared
snapshot/graph/propagation/stitching orchestration and the canonical
background-channel formula, and
``models.dinotext.cover_dr.matched_power_evaluator.finalize_prediction``
for V20's background-free finalization.

Requires an already-verified VOC2012 dataset-source manifest, verified
via ``verify_voc2012_dataset.py verify-manifest`` BEFORE any CUDA/model
initialization. Never modifies the dataset. Never trains, fine-tunes, or
pre-extracts target-dataset features -- the existing frozen COCO-2017-
trained Talk2DINO bridge checkpoint (vitb_mlp_infonce) is loaded
read-only.
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

from src.k11_k12_stability_report import write_checkpoint_atomically
from src.matched_k11_k12_identity import MatchedK11K12Error
from src.matched_k11_k12_identity import load_identity as load_matched_identity
from src.native_edge_support_checkpoint import parse_strict_json_document
from src.voc2012_dataset_identity import Voc2012DatasetIdentityError
from src.voc2012_dataset_identity import load_identity as load_voc2012_source_identity
from src.voc2012_dataset_manifest import canonical_validation_ids, resolve_dataset_root
from src.voc2012_matched_evaluator_checkpoint import (
    resume_dataset_index,
    validate_checkpoint_against_canonical_order,
    validate_checkpoint_against_per_image_stats,
    validate_checkpoint_structure,
)
from src.voc2012_matched_evaluator_identity import (
    Voc2012MatchedEvaluatorIdentityError,
    load_identity,
    repository_root,
    validate_static_configuration,
)
from src.voc2012_matched_evaluator_report import VARIANT_NAMES, verify_record

VOC2012_REAL_DATA_ROOT_ENV = "VOC2012_REAL_DATA_ROOT"

# The authoritative per-image-stats NPZ member schema, derived from
# VARIANT_NAMES (the same six-variant list src.voc2012_matched_evaluator_report
# already owns) -- never a second, independently-declared list of variant
# names. Two exact (never prefix/substring-guessed) per-protocol name sets
# are the single source of truth for both the closed member-set check and
# the per-array shape/dtype contract below.
V20_PER_IMAGE_STATS_ARRAY_NAMES = frozenset(
    {"label_v20"}
    | {f"{stat}_{variant}" for variant in VARIANT_NAMES if variant.startswith("v20_") for stat in ("intersect", "union", "pred")}
)
V21_PER_IMAGE_STATS_ARRAY_NAMES = frozenset(
    {"label_v21"}
    | {f"{stat}_{variant}" for variant in VARIANT_NAMES if variant.startswith("v21_") for stat in ("intersect", "union", "pred")}
)
EXPECTED_PER_IMAGE_STATS_ARRAY_NAMES = (
    frozenset({"dataset_indices"}) | V20_PER_IMAGE_STATS_ARRAY_NAMES | V21_PER_IMAGE_STATS_ARRAY_NAMES
)


def _validate_per_image_stats_array(
    name: str,
    array: np.ndarray,
    *,
    image_count: int,
    v20_class_count: int,
    v21_class_count: int,
) -> None:
    """Sole authority for the per-image-stats NPZ array shape/dtype contract.
    Invoked identically by the atomic writer (before serialization) and by
    _load_per_image_stats (after loading) so the two can never drift apart.
    Read-only: never casts, reshapes, squeezes, truncates, or pads an array
    that fails the contract -- it is always rejected, never coerced."""
    if array.dtype != np.int64:
        raise Voc2012MatchedEvaluatorIdentityError(
            f"per-image-stats array {name!r} must have exact dtype int64, got {array.dtype}"
        )
    if name == "dataset_indices":
        if array.ndim != 1:
            raise Voc2012MatchedEvaluatorIdentityError(
                f"per-image-stats array {name!r} must be exactly 1-dimensional, got ndim={array.ndim}"
            )
        if array.shape[0] != image_count:
            raise Voc2012MatchedEvaluatorIdentityError(
                f"per-image-stats array {name!r} has {array.shape[0]} rows, expected exactly image_count={image_count}"
            )
        return
    if name in V20_PER_IMAGE_STATS_ARRAY_NAMES:
        expected_columns = v20_class_count
    elif name in V21_PER_IMAGE_STATS_ARRAY_NAMES:
        expected_columns = v21_class_count
    else:
        raise Voc2012MatchedEvaluatorIdentityError(f"per-image-stats array {name!r} is not part of the authoritative schema")
    if array.ndim != 2:
        raise Voc2012MatchedEvaluatorIdentityError(
            f"per-image-stats array {name!r} must be exactly 2-dimensional, got ndim={array.ndim}"
        )
    if array.shape[0] != image_count:
        raise Voc2012MatchedEvaluatorIdentityError(
            f"per-image-stats array {name!r} has {array.shape[0]} rows, expected exactly image_count={image_count}"
        )
    if array.shape[1] != expected_columns:
        raise Voc2012MatchedEvaluatorIdentityError(
            f"per-image-stats array {name!r} has {array.shape[1]} columns, expected exactly {expected_columns}"
        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise Voc2012MatchedEvaluatorIdentityError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _resolve_data_root(raw: Path | None) -> Path:
    if raw is not None:
        return raw
    env_value = os.environ.get(VOC2012_REAL_DATA_ROOT_ENV)
    if env_value is None or not env_value.strip():
        raise Voc2012MatchedEvaluatorIdentityError(
            f"--data-root was not given and {VOC2012_REAL_DATA_ROOT_ENV} is unset/empty/whitespace-only"
        )
    return Path(env_value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shared VOC2012 V20/V21 matched evaluator: E3 vs matched k11 vs k12, T=320 finite-step."
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    parser.add_argument("--identity", type=Path, default=None)
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help=f"VOC2012 dataset root (VOCdevkit or VOC2012 directory); falls back to ${VOC2012_REAL_DATA_ROOT_ENV} if omitted",
    )
    parser.add_argument("--source-manifest", type=Path, required=True, help="verified VOC2012 dataset-source manifest JSON")
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


def _verify_source_manifest_before_cuda(root: Path, manifest_path: Path, data_root: Path) -> Mapping[str, Any]:
    """Run verify_voc2012_dataset.py verify-manifest BEFORE any CUDA/model
    initialization, as an actual subprocess invocation of the real,
    independent verifier CLI -- never a reimplemented or partial check."""
    verify_script = root / "verify_voc2012_dataset.py"
    proc = subprocess.run(
        [sys.executable, str(verify_script), "verify-manifest", "--repo-root", str(root), "--data-root", str(data_root), "--manifest", str(manifest_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise Voc2012MatchedEvaluatorIdentityError(
            f"VOC2012 source manifest verify-manifest failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return parse_strict_json_document(manifest_path, label="voc2012 source manifest")


def _run_evaluation(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    data_root = _resolve_data_root(args.data_root)

    identity = load_identity(args.identity, repo_root=root)
    validate_static_configuration(repo_root=root, identity_path=args.identity, check_git=True)
    identity_path = args.identity if args.identity is not None else root / "evaluation_identities/e12_voc2012_matched_evaluator.toml"
    identity_sha256 = _sha256_file(identity_path)

    manifest = _verify_source_manifest_before_cuda(root, args.source_manifest, data_root)
    source_manifest_sha256 = _sha256_file(args.source_manifest)

    image_count = identity["run_modes"][f"{args.run_mode}_image_count"]
    schema_name = identity["run_modes"][f"{args.run_mode}_schema_name"]
    v20_class_count = identity["v20_protocol"]["class_count"]
    v21_class_count = identity["v21_protocol"]["class_count"]
    bg_thresh = identity["background_protocol"]["bg_thresh"]

    if not args.overwrite and args.result.exists():
        raise Voc2012MatchedEvaluatorIdentityError(f"--result {args.result} already exists; pass --overwrite for explicit resume/overwrite behavior")
    for path in (args.result, args.checkpoint, args.per_image_stats, _per_image_stats_npz_path(args.per_image_stats)):
        path.parent.mkdir(parents=True, exist_ok=True)

    # The bridge/checkpoint SHA256 bound into every checkpoint/result
    # document is the FIXED identity-pinned value for weights/vitb_mlp_infonce.pth
    # (already byte-verified against disk by validate_static_configuration
    # above) -- never a self-hash of the checkpoint/result document being
    # written, which would be circular.
    bridge_checkpoint_sha256 = identity["model_and_checkpoint"]["projection_checkpoint_sha256"]

    # CPU-only, before any CUDA/model work: real dataset-source identity
    # binding and the canonical validation-split image order. Computed
    # before Phase A below so a resumed run's checkpoint/per-image-stats
    # can be checked against the canonical order without ever
    # constructing the mmseg dataset.
    source_identity = load_voc2012_source_identity(repo_root=root)
    if source_identity["identity"]["name"] != identity["parent_identities"]["voc2012_source_identity_name"]:
        raise Voc2012MatchedEvaluatorIdentityError("VOC2012 source parent identity name mismatch")
    voc_root = resolve_dataset_root(data_root, source_identity)
    canonical_ids = canonical_validation_ids(voc_root, source_identity)
    if len(canonical_ids) < image_count:
        raise Voc2012MatchedEvaluatorIdentityError(f"canonical split has only {len(canonical_ids)} images, fewer than the requested {image_count}")
    expected_image_ids = canonical_ids[:image_count]
    image_order_digest = _image_order_digest(expected_image_ids)
    if manifest.get("image_order_digest") is not None and manifest["image_order_digest"] != _image_order_digest(canonical_ids):
        raise Voc2012MatchedEvaluatorIdentityError("source manifest image_order_digest disagrees with the freshly-resolved canonical split order")

    # Phase A: every resume-artifact check that can run without the mmseg
    # dataset, torch, the model, or the bridge checkpoint -- so a
    # malformed/inconsistent resume artifact is rejected before any of
    # that work begins, never after. A fresh (non-resume) run never
    # touches --per-image-stats here. An inconsistent presence of the two
    # paired resume artifacts (one exists, the other doesn't) fails
    # closed rather than silently falling back to a fresh run.
    phase_a_checkpoint: dict[str, Any] | None = None
    phase_a_stats: dict[str, Any] | None = None
    if args.resume:
        checkpoint_exists = args.checkpoint.exists()
        stats_exists = args.per_image_stats.exists()
        if checkpoint_exists != stats_exists:
            raise Voc2012MatchedEvaluatorIdentityError(
                "--resume requires --checkpoint and --per-image-stats to be consistently present or "
                f"consistently absent together; checkpoint_exists={checkpoint_exists}, per_image_stats_exists={stats_exists}"
            )
        if checkpoint_exists:
            phase_a_checkpoint = parse_strict_json_document(args.checkpoint, label="checkpoint")
            validate_checkpoint_structure(
                phase_a_checkpoint, identity=identity, identity_sha256=identity_sha256,
                run_mode=args.run_mode, source_manifest_sha256=source_manifest_sha256,
            )
            resume_dataset_index(phase_a_checkpoint)  # raises if already complete
            validate_checkpoint_against_canonical_order(phase_a_checkpoint, expected_image_ids, image_order_digest=image_order_digest)
            phase_a_stats = _load_per_image_stats(args.per_image_stats)
            validate_checkpoint_against_per_image_stats(phase_a_checkpoint, phase_a_stats["manifest"], identity=identity)

    matched_identity_path = root / identity["parent_identities"]["matched_identity_path"]
    try:
        matched_identity = load_matched_identity(matched_identity_path, repo_root=root)
    except MatchedK11K12Error as error:
        raise Voc2012MatchedEvaluatorIdentityError(f"matched parent identity failed validation: {error}") from error
    if matched_identity["identity"]["name"] != identity["parent_identities"]["matched_identity_name"]:
        raise Voc2012MatchedEvaluatorIdentityError("matched parent identity name mismatch")
    alpha = matched_identity["propagation"]["alpha"]
    steps = matched_identity["propagation"]["steps"]
    affinity_power = matched_identity["graph"]["affinity_power"]
    crop = tuple(matched_identity["geometry"]["crop"])
    stride = tuple(matched_identity["geometry"]["stride"])

    from mmcv import Config as MMCVConfig
    from mmseg.datasets import build_dataset
    import main  # noqa: F401  -- registers the custom FloatImage transform, side-effect only

    mac = identity["model_and_checkpoint"]
    v20_dataset_config_path = root / identity["v20_protocol"]["dataset_config_relative_path"]
    v21_dataset_config_path = root / identity["v21_protocol"]["dataset_config_relative_path"]
    v20_cfg = MMCVConfig.fromfile(str(v20_dataset_config_path))
    v21_cfg = MMCVConfig.fromfile(str(v21_dataset_config_path))
    v20_cfg.data.test.data_root = str(voc_root)
    v21_cfg.data.test.data_root = str(voc_root)
    dataset_v20 = build_dataset(v20_cfg.data.test)
    dataset_v21 = build_dataset(v21_cfg.data.test)

    from src.voc2012_matched_evaluator_identity import (
        SUPPORTED_V20_CLASS_COUNT, SUPPORTED_V21_CLASS_COUNT,
    )

    def _class_digest(classes: tuple[str, ...]) -> str:
        return hashlib.sha256(json.dumps(list(classes), ensure_ascii=True).encode("utf-8")).hexdigest()

    v20_classes = dataset_v20.CLASSES
    v21_classes = dataset_v21.CLASSES
    if type(v20_classes) is not tuple or len(v20_classes) != SUPPORTED_V20_CLASS_COUNT:
        raise Voc2012MatchedEvaluatorIdentityError("live V20 dataset.CLASSES has an unexpected shape")
    if type(v21_classes) is not tuple or len(v21_classes) != SUPPORTED_V21_CLASS_COUNT or v21_classes[0] != "background":
        raise Voc2012MatchedEvaluatorIdentityError("live V21 dataset.CLASSES has an unexpected shape")
    if v21_classes[1:] != v20_classes:
        raise Voc2012MatchedEvaluatorIdentityError("live V21 dataset.CLASSES[1:] disagrees with live V20 dataset.CLASSES -- V20/V21 no longer share foreground order")
    live_v20_class_names_digest = _class_digest(v20_classes)
    live_v21_class_names_digest = _class_digest(v21_classes)

    v20_infos = dataset_v20.img_infos if hasattr(dataset_v20, "img_infos") else dataset_v20.data_infos
    v21_infos = dataset_v21.img_infos if hasattr(dataset_v21, "img_infos") else dataset_v21.data_infos
    if len(v20_infos) < image_count or len(v21_infos) < image_count:
        raise Voc2012MatchedEvaluatorIdentityError("mmseg V20/V21 dataset has fewer images than the requested run mode")
    for i in range(image_count):
        v20_id = Path(str(v20_infos[i]["filename"])).stem
        v21_id = Path(str(v21_infos[i]["filename"])).stem
        if v20_id != expected_image_ids[i] or v21_id != expected_image_ids[i]:
            raise Voc2012MatchedEvaluatorIdentityError(
                f"mmseg dataset image order at index {i} ({v20_id!r}/{v21_id!r}) disagrees with the canonical VOC2012 split order ({expected_image_ids[i]!r})"
            )

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise Voc2012MatchedEvaluatorIdentityError("--device cuda requested but CUDA is not available")

    git_commit = _git(root, "rev-parse", "HEAD")
    git_branch = _git(root, "branch", "--show-current")

    from utils.config import load_config
    from utils.logger import get_logger
    from models import build_model
    from mmcv.runner import CheckpointLoader
    from segmentation.evaluation import build_dinotext_seg_inference
    from torch.utils.data import Subset

    # Built from the V21 (with-background) eval config: build_dinotext_seg_inference
    # derives its text query as dataset.CLASSES[1:] whenever CLASSES[0]=='background'
    # -- i.e. V20's own 20-class CLASSES verbatim -- so this single inference object's
    # text_embedding/model are exactly what V20 needs too. Never built twice.
    eval_config_path = root / mac["v21_eval_config"]["eval_config_relative_path"]
    cfg = load_config(str(eval_config_path))

    model = build_model(cfg.model)
    checkpoint_path = root / mac["projection_checkpoint_relative_path"]
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

    inference = build_dinotext_seg_inference(model, Subset(dataset_v21, range(len(dataset_v21))), cfg, str(v21_dataset_config_path))
    inference.reset_evaluation_state()
    align_corners = inference.align_corners

    live_class_count = getattr(inference, "num_classes", None)
    if live_class_count != v21_class_count:
        raise Voc2012MatchedEvaluatorIdentityError(
            f"live inference.num_classes ({live_class_count}) disagrees with the registered V21 class count ({v21_class_count})"
        )

    from models.dinotext.cover_dr.coco_object_evaluator import (
        WindowOperationTelemetryE3,
        aggregate_telemetry,
        stitch_one_image_with_e3,
    )
    from models.dinotext.cover_dr.coco_object_evaluator import finalize_prediction as finalize_prediction_v21
    from models.dinotext.cover_dr.matched_power_evaluator import finalize_prediction as finalize_prediction_v20
    from diagnostics.run_k11_k12_stability import _extract_prepared_image

    started_at = time.monotonic()

    next_dataset_index = 0
    completed_image_ids: list[str] = []
    dataset_indices: list[int] = []
    image_ids: list[str] = []
    rows: dict[str, list[np.ndarray]] = {f"label_v20": [], f"label_v21": []}
    for variant in VARIANT_NAMES:
        rows[f"intersect_{variant}"] = []
        rows[f"union_{variant}"] = []
        rows[f"pred_{variant}"] = []
    windows_processed_total = 0

    if phase_a_checkpoint is not None:
        # Already fully loaded and validated in Phase A, before dataset/
        # torch/model/CUDA work -- never reloaded or revalidated here.
        next_dataset_index = phase_a_checkpoint["next_dataset_index"]
        completed_image_ids = list(phase_a_checkpoint["completed_image_ids"])
        windows_processed_total = phase_a_checkpoint["windows_processed_total"]
        dataset_indices = list(int(i) for i in phase_a_stats["manifest"]["dataset_indices"])
        image_ids = list(phase_a_stats["manifest"]["image_ids"])
        arrays = phase_a_stats["arrays"]
        for key in rows:
            rows[key] = list(arrays[key])

    def _checkpoint_payload(*, complete: bool) -> dict[str, Any]:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return {
            "schema": identity["checkpoint"]["schema_name"],
            "run_mode": args.run_mode,
            "identity": identity["identity"]["name"],
            "identity_sha256": identity_sha256,
            "matched_identity_sha256": identity["parent_identities"]["matched_identity_sha256"],
            "voc2012_source_identity_sha256": identity["parent_identities"]["voc2012_source_identity_sha256"],
            "source_manifest_sha256": source_manifest_sha256,
            "bridge_checkpoint_sha256": bridge_checkpoint_sha256,
            "git_commit": git_commit,
            "v20_class_count": v20_class_count,
            "v21_class_count": v21_class_count,
            "live_v20_class_names_digest": live_v20_class_names_digest,
            "live_v21_class_names_digest": live_v21_class_names_digest,
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
        validate_checkpoint_structure(
            payload, identity=identity, identity_sha256=identity_sha256, run_mode=args.run_mode,
            v20_class_count=v20_class_count, v21_class_count=v21_class_count, source_manifest_sha256=source_manifest_sha256,
        )
        write_checkpoint_atomically(args.checkpoint, payload)

    foreground_class_count = v20_class_count

    while next_dataset_index < image_count:
        dataset_index = next_dataset_index
        prepared = _extract_prepared_image(dataset_v21, dataset_index, canonical_image_id=expected_image_ids[dataset_index])
        if prepared.image_id != expected_image_ids[dataset_index]:
            raise Voc2012MatchedEvaluatorIdentityError(
                f"dataset[{dataset_index}] image_id {prepared.image_id!r} disagrees with the canonical order {expected_image_ids[dataset_index]!r}"
            )

        image_tensor = prepared.image_tensor
        if args.device == "cuda":
            image_tensor = image_tensor.cuda()
        batched = image_tensor.unsqueeze(0)

        from segmentation.evaluation.sliding_window_geometry import SlidingWindowPlan, SpatialSize

        plan = SlidingWindowPlan.build(
            image_size=SpatialSize(prepared.inference_height, prepared.inference_width),
            crop_size=SpatialSize(*crop), stride=SpatialSize(*stride),
        )
        stitched = stitch_one_image_with_e3(
            inference, batched, plan, class_count=foreground_class_count, alpha=alpha, steps=steps, affinity_power=affinity_power,
        )

        pred_v20_e3 = finalize_prediction_v20(stitched.e3_stitched, prepared.img_metas, align_corners=align_corners)
        pred_v20_k11 = finalize_prediction_v20(stitched.k11_stitched, prepared.img_metas, align_corners=align_corners)
        pred_v20_k12 = finalize_prediction_v20(stitched.k12_stitched, prepared.img_metas, align_corners=align_corners)
        pred_v21_e3 = finalize_prediction_v21(stitched.e3_stitched, prepared.img_metas, align_corners=align_corners, bg_thresh=bg_thresh)
        pred_v21_k11 = finalize_prediction_v21(stitched.k11_stitched, prepared.img_metas, align_corners=align_corners, bg_thresh=bg_thresh)
        pred_v21_k12 = finalize_prediction_v21(stitched.k12_stitched, prepared.img_metas, align_corners=align_corners, bg_thresh=bg_thresh)

        preds = {
            "v20_e3": pred_v20_e3, "v20_k11": pred_v20_k11, "v20_k12": pred_v20_k12,
            "v21_e3": pred_v21_e3, "v21_k11": pred_v21_k11, "v21_k12": pred_v21_k12,
        }
        preds_np = {name: t[0].detach().cpu().numpy() for name, t in preds.items()}

        label_v20 = None
        label_v21 = None
        for variant in VARIANT_NAMES:
            dataset = dataset_v20 if variant.startswith("v20_") else dataset_v21
            pre_eval = dataset.pre_eval(preds_np[variant], dataset_index)[0]
            intersect, union, area_pred, area_label = (t.numpy() for t in pre_eval)
            rows[f"intersect_{variant}"].append(intersect)
            rows[f"union_{variant}"].append(union)
            rows[f"pred_{variant}"].append(area_pred)
            if variant.startswith("v20_"):
                if label_v20 is None:
                    label_v20 = area_label
                elif not np.array_equal(label_v20, area_label):
                    raise Voc2012MatchedEvaluatorIdentityError(f"dataset[{dataset_index}]: V20 ground-truth area diverged between E3/k11/k12 variants")
            else:
                if label_v21 is None:
                    label_v21 = area_label
                elif not np.array_equal(label_v21, area_label):
                    raise Voc2012MatchedEvaluatorIdentityError(f"dataset[{dataset_index}]: V21 ground-truth area diverged between E3/k11/k12 variants")

        rows["label_v20"].append(label_v20)
        rows["label_v21"].append(label_v21)

        dataset_indices.append(dataset_index)
        image_ids.append(prepared.image_id)
        windows_processed_total += stitched.window_count
        completed_image_ids.append(prepared.image_id)
        next_dataset_index = dataset_index + 1

        _write_per_image_stats_atomically(
            args.per_image_stats, schema_name=identity["artifacts"]["per_image_stats_manifest_schema_name"],
            v20_class_count=v20_class_count, v21_class_count=v21_class_count,
            live_v20_class_names_digest=live_v20_class_names_digest, live_v21_class_names_digest=live_v21_class_names_digest,
            dataset_indices=dataset_indices, image_ids=image_ids, rows=rows,
        )
        _write_checkpoint_self_validated(complete=False)

        del image_tensor, batched, stitched, preds, preds_np
        if args.device == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.monotonic() - started_at

    if len(image_ids) != image_count or next_dataset_index != image_count or dataset_indices != list(range(image_count)):
        raise Voc2012MatchedEvaluatorIdentityError("processed images do not form the exact canonical full range")

    from models.dinotext.cover_dr import compute_full_precision_metrics

    def _metrics(variant: str) -> dict[str, float]:
        label_key = "label_v20" if variant.startswith("v20_") else "label_v21"
        fraction = compute_full_precision_metrics(
            list(zip(rows[f"intersect_{variant}"], rows[f"union_{variant}"], rows[f"pred_{variant}"], rows[label_key]))
        )
        return {name: 100.0 * value for name, value in fraction.items()}

    metrics = {variant: _metrics(variant) for variant in VARIANT_NAMES}

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

    result: dict[str, Any] = {
        "schema": schema_name,
        "run_mode": args.run_mode,
        "identity": identity["identity"]["name"],
        "identity_sha256": identity_sha256,
        "matched_identity_sha256": identity["parent_identities"]["matched_identity_sha256"],
        "voc2012_source_identity_sha256": identity["parent_identities"]["voc2012_source_identity_sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "bridge_checkpoint_sha256": bridge_checkpoint_sha256,
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
        "v20_class_count": v20_class_count,
        "v21_class_count": v21_class_count,
        "background_class_index": identity["v21_protocol"]["background_class_index"],
        "bg_thresh": bg_thresh,
        "live_v20_class_names_digest": live_v20_class_names_digest,
        "live_v21_class_names_digest": live_v21_class_names_digest,
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
    for variant in VARIANT_NAMES:
        result[f"metrics_{variant}"] = metrics[variant]
    result["delta_mIoU_v20_k11_minus_k12_percentage_points"] = metrics["v20_k11"]["mIoU"] - metrics["v20_k12"]["mIoU"]
    result["delta_mIoU_v20_k11_minus_e3_percentage_points"] = metrics["v20_k11"]["mIoU"] - metrics["v20_e3"]["mIoU"]
    result["delta_mIoU_v20_k12_minus_e3_percentage_points"] = metrics["v20_k12"]["mIoU"] - metrics["v20_e3"]["mIoU"]
    result["delta_mIoU_v21_k11_minus_k12_percentage_points"] = metrics["v21_k11"]["mIoU"] - metrics["v21_k12"]["mIoU"]
    result["delta_mIoU_v21_k11_minus_e3_percentage_points"] = metrics["v21_k11"]["mIoU"] - metrics["v21_e3"]["mIoU"]
    result["delta_mIoU_v21_k12_minus_e3_percentage_points"] = metrics["v21_k12"]["mIoU"] - metrics["v21_e3"]["mIoU"]

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
        f"VOC2012 MATCHED EVALUATOR {args.run_mode.upper()} PASS images={len(image_ids)} "
        f"mIoU_v20_e3={metrics['v20_e3']['mIoU']:.6f} mIoU_v20_k11={metrics['v20_k11']['mIoU']:.6f} mIoU_v20_k12={metrics['v20_k12']['mIoU']:.6f} "
        f"mIoU_v21_e3={metrics['v21_e3']['mIoU']:.6f} mIoU_v21_k11={metrics['v21_k11']['mIoU']:.6f} mIoU_v21_k12={metrics['v21_k12']['mIoU']:.6f} "
        f"-> {args.result}"
    )
    return 0


def _write_per_image_stats_atomically(
    manifest_path: Path, *, schema_name: str, v20_class_count: int, v21_class_count: int,
    live_v20_class_names_digest: str, live_v21_class_names_digest: str,
    dataset_indices: list[int], image_ids: list[str], rows: dict[str, list[np.ndarray]],
) -> None:
    save_arrays = {"dataset_indices": np.asarray(dataset_indices, dtype=np.int64)}
    for key, values in rows.items():
        save_arrays[key] = np.stack(values).astype(np.int64)
    for name, array in save_arrays.items():
        _validate_per_image_stats_array(
            name,
            array,
            image_count=len(image_ids),
            v20_class_count=v20_class_count,
            v21_class_count=v21_class_count,
        )

    npz_path = _per_image_stats_npz_path(manifest_path)
    temp_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    np.savez(temp_npz, allow_pickle=False, **save_arrays)
    written = temp_npz if temp_npz.suffix == ".npz" else temp_npz.with_suffix(temp_npz.suffix + ".npz")
    os.replace(written, npz_path)
    npz_sha256 = _sha256_file(npz_path)

    manifest = {
        "schema": schema_name, "npz_filename": npz_path.name, "npz_sha256": npz_sha256,
        "v20_class_count": v20_class_count, "v21_class_count": v21_class_count,
        "live_v20_class_names_digest": live_v20_class_names_digest, "live_v21_class_names_digest": live_v21_class_names_digest,
        "image_count": len(image_ids), "dataset_indices": list(dataset_indices),
        "image_ids": list(image_ids), "image_order_digest": _image_order_digest(list(image_ids)),
    }
    write_checkpoint_atomically(manifest_path, manifest)


def _load_per_image_stats(manifest_path: Path) -> dict[str, Any]:
    manifest = parse_strict_json_document(manifest_path, label="per-image-stats manifest")
    npz_path = _per_image_stats_npz_path(manifest_path)
    try:
        if manifest["npz_sha256"] != _sha256_file(npz_path):
            raise Voc2012MatchedEvaluatorIdentityError("per-image-stats NPZ SHA256 does not match its manifest; refusing to resume")
        # allow_pickle=False both structurally rejects object-dtype arrays
        # (numpy refuses to load pickled objects) and prevents any aliasing
        # trick via pickled references.
        with np.load(npz_path, allow_pickle=False) as data:
            observed_names = frozenset(data.files)
            if observed_names != EXPECTED_PER_IMAGE_STATS_ARRAY_NAMES:
                missing = sorted(EXPECTED_PER_IMAGE_STATS_ARRAY_NAMES - observed_names)
                unexpected = sorted(observed_names - EXPECTED_PER_IMAGE_STATS_ARRAY_NAMES)
                raise Voc2012MatchedEvaluatorIdentityError(
                    "per-image-stats NPZ member set disagrees with the authoritative schema"
                    f" (missing={missing}, unexpected={unexpected})"
                )
            arrays = {key: data[key] for key in data.files}
            for name, array in arrays.items():
                _validate_per_image_stats_array(
                    name,
                    array,
                    image_count=manifest["image_count"],
                    v20_class_count=manifest["v20_class_count"],
                    v21_class_count=manifest["v21_class_count"],
                )
    except Voc2012MatchedEvaluatorIdentityError:
        raise
    except (ValueError, OSError) as error:
        raise Voc2012MatchedEvaluatorIdentityError(f"cannot load per-image-stats NPZ {npz_path}: {error}") from error
    return {"manifest": manifest, "arrays": arrays}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run_evaluation(args)
    except (Voc2012MatchedEvaluatorIdentityError, Voc2012DatasetIdentityError, MatchedK11K12Error, ValueError, OSError) as error:
        print(f"VOC2012 MATCHED EVALUATOR FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
