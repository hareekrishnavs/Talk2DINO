#!/usr/bin/env python3
"""Dedicated single-GPU matched k11-vs-k12 finite-step (T=320) power
evaluator.

Computes both the k=11 and k=12 finite-step variants from exactly the same
per-window E3 patch snapshot and canonical top-12 graph selection, using
the exact finite-step kernel already verified by GPU stability-gate job
20300858 (``models.dinotext.cover_dr.matched_power_evaluator``, which
imports the kernel unmodified from ``finite_step_regime.py`` -- never
copied or reimplemented here). Requires an accepted k11/k12 stability-gate
result, passed explicitly via ``--stability-result``, before any dataset or
model construction is attempted.

This is NOT ``main.py --eval``: it drives its own bounded per-image loop,
never calls ``multi_gpu_test``, and never monkeypatches production
inference. It reuses the exact same entry points production evaluation and
the stability-gate harness already use for dataset/model construction
(``_build_dataset_only``/``_build_inference`` from
``diagnostics.run_k11_k12_stability``, never reimplemented here), and
mmseg's own ``dataset.pre_eval`` for full-precision per-image sufficient
statistics (never a reimplementation of ``intersect_and_union``).

Checkpoint reading/writing and validation go exclusively through
:mod:`src.k11_k12_power_evaluation_checkpoint` -- the same shared
loader/validator ``verify_k11_k12_power_evaluation.py verify-checkpoint``
uses. This module never reimplements checkpoint invariant checks.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
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

from src.k11_k12_power_evaluation_checkpoint import (
    CHECKPOINT_SCHEMA_NAME,
    parse_strict_json_document,
    resume_dataset_index,
    validate_checkpoint_against_artifact,
    validate_checkpoint_against_canonical_order,
    validate_checkpoint_structure,
)
from src.k11_k12_power_evaluation_identity import (
    K11K12PowerEvaluationError,
    RUN_MODE_IMAGE_COUNT_KEYS,
    RUN_MODE_SCHEMA_KEYS,
    load_identity,
    repository_root,
    validate_static_configuration,
    validate_stability_result_binding,
)
from src.k11_k12_power_evaluation_report import verify_record
from src.k11_k12_stability_report import write_checkpoint_atomically
from diagnostics.run_k11_k12_stability import (
    PreparedDiagnosticImage,
    _build_inference,
    _extract_prepared_image,
    _git,
    _reject_tracked_output_path,
    _sha256_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Matched k11-vs-k12 finite-step (T=320) power evaluator"
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    parser.add_argument("--identity", type=Path, default=None)
    parser.add_argument(
        "--stability-result", type=Path, required=True,
        help="path to a passing k11/k12 stability-gate result JSON (e.g. "
        "/scratch/haree/e12_k11_k12_stability/result-20300858.json); no default is provided",
    )
    parser.add_argument("--run-mode", required=True, choices=sorted(RUN_MODE_IMAGE_COUNT_KEYS))
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


def _expected_image_ids(dataset: Any, image_count: int) -> list[str]:
    """Cheap, pipeline-free canonical image-order derivation: reads only
    ``dataset.img_infos``/``dataset.data_infos`` (identity index only,
    never dimensions -- see docs/matched_k11_k12_power_evaluation.md),
    never runs the model/pipeline. Used only to compute the expected
    image-order digest before any image is actually processed."""
    infos = dataset.img_infos if hasattr(dataset, "img_infos") else dataset.data_infos
    if len(infos) < image_count:
        raise K11K12PowerEvaluationError(
            f"dataset has only {len(infos)} images, fewer than the requested {image_count}"
        )
    return [str(infos[i]["filename"]) for i in range(image_count)]


def resolve_and_validate_class_count(inference: Any, dataset: Any, *, expected_class_count: int) -> int:
    """Resolve the class count exactly once, from the live production
    object (``inference.num_classes``) -- never a hardcoded module
    constant -- and validate it against every other authoritative source
    available: the identity's registered class count, and (where present)
    the live dataset's own class metadata length. Callers pass the single
    validated result explicitly into every downstream consumer; this is
    never re-resolved inside a loop.
    """
    live_class_count = getattr(inference, "num_classes", None)
    if type(live_class_count) is bool or not isinstance(live_class_count, int):
        raise K11K12PowerEvaluationError("inference.num_classes must be an exact non-boolean integer")
    if live_class_count <= 0:
        raise K11K12PowerEvaluationError("inference.num_classes must be positive")
    if live_class_count != expected_class_count:
        raise K11K12PowerEvaluationError(
            f"live inference.num_classes ({live_class_count}) disagrees with the class count registered "
            f"in the authoritative identity ({expected_class_count})"
        )
    dataset_classes = getattr(dataset, "CLASSES", None)
    if dataset_classes is not None and len(dataset_classes) != live_class_count:
        raise K11K12PowerEvaluationError(
            f"live inference.num_classes ({live_class_count}) disagrees with dataset.CLASSES length "
            f"({len(dataset_classes)})"
        )
    return live_class_count


def _write_per_image_stats_atomically(
    manifest_path: Path,
    *,
    schema_name: str,
    class_count: int,
    dataset_indices: list[int],
    image_ids: list[str],
    label: np.ndarray,
    intersect_k11: np.ndarray,
    union_k11: np.ndarray,
    pred_k11: np.ndarray,
    intersect_k12: np.ndarray,
    union_k12: np.ndarray,
    pred_k12: np.ndarray,
) -> str:
    """Atomically (re)write the per-image sufficient-statistics artifact:
    a compact NPZ of exact integer arrays (``allow_pickle=False``, no
    Python object arrays) plus a small JSON manifest carrying the image
    IDs, schema, and the NPZ's own SHA256. GT (``label``) is stored once,
    shared between variants, since it is architecturally identical for
    both -- never duplicated or allowed to silently diverge. ``class_count``
    is the single value :func:`resolve_and_validate_class_count` already
    validated -- never recomputed here."""
    arrays = {
        "label": label, "intersect_k11": intersect_k11, "union_k11": union_k11, "pred_k11": pred_k11,
        "intersect_k12": intersect_k12, "union_k12": union_k12, "pred_k12": pred_k12,
    }
    for name, array in arrays.items():
        if array.shape[0] != len(image_ids):
            raise K11K12PowerEvaluationError(f"per-image-stats array {name!r} row count disagrees with image_ids length")
        if array.shape[1] != class_count:
            raise K11K12PowerEvaluationError(
                f"per-image-stats array {name!r} has {array.shape[1]} columns, expected class_count={class_count}"
            )

    npz_path = _per_image_stats_npz_path(manifest_path)
    temp_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    np.savez(
        temp_npz,
        dataset_indices=np.asarray(dataset_indices, dtype=np.int64),
        label=label.astype(np.int64),
        intersect_k11=intersect_k11.astype(np.int64),
        union_k11=union_k11.astype(np.int64),
        pred_k11=pred_k11.astype(np.int64),
        intersect_k12=intersect_k12.astype(np.int64),
        union_k12=union_k12.astype(np.int64),
        pred_k12=pred_k12.astype(np.int64),
        allow_pickle=False,
    )
    # np.savez appends .npz if the target doesn't already end with it
    written = temp_npz if temp_npz.suffix == ".npz" else temp_npz.with_suffix(temp_npz.suffix + ".npz")
    os.replace(written, npz_path)
    npz_sha256 = _sha256_file(npz_path)

    manifest = {
        "schema": schema_name,
        "npz_filename": npz_path.name,
        "npz_sha256": npz_sha256,
        "class_count": class_count,
        "image_count": len(image_ids),
        "dataset_indices": dataset_indices,
        "image_ids": image_ids,
        "image_order_digest": _image_order_digest(image_ids),
    }
    write_checkpoint_atomically(manifest_path, manifest)
    return npz_sha256


def _load_per_image_stats(manifest_path: Path) -> dict[str, Any]:
    manifest = parse_strict_json_document(manifest_path, label="per-image-stats manifest")
    npz_path = _per_image_stats_npz_path(manifest_path)
    if manifest["npz_sha256"] != _sha256_file(npz_path):
        raise K11K12PowerEvaluationError("per-image-stats NPZ SHA256 does not match its manifest; refusing to resume")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return {"manifest": manifest, "arrays": arrays}


def _operation_telemetry_for_window_count(window_count: int) -> dict[str, int]:
    """Derive the exact aggregate operation telemetry for ``window_count``
    completed windows directly from the fixed per-window contract that
    ``models.dinotext.cover_dr.WindowOperationTelemetry`` itself already
    enforces (fail-closed) on every window actually processed -- rather
    than summing an in-memory list of per-window records.

    This is deliberately resume-safe: ``window_count`` comes from
    ``windows_processed_total``, which is already correctly cumulative
    across a resume (persisted in the checkpoint after every image); a
    raw per-window record list, by contrast, would be reset to empty on
    every process invocation and so would silently under-report telemetry
    for any run that was interrupted and resumed. Since every window's
    telemetry is contractually identical by construction, deriving the
    total from the count alone is exact, not an approximation.
    """
    from models.dinotext.cover_dr import WindowOperationTelemetry, aggregate_telemetry

    if window_count <= 0:
        raise K11K12PowerEvaluationError("_operation_telemetry_for_window_count requires window_count >= 1")
    canonical_window = WindowOperationTelemetry(
        backbone_snapshot_calls=1, dino_feature_extractions=1, topk_selection_calls=1,
        graph_normalizations=2, finite_step_propagations=2, k11_updates=320, k12_updates=320,
        sigmoid_calls=2, interpolation_calls=2,
    )
    return aggregate_telemetry([canonical_window] * window_count)


def _run_evaluation(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)

    # --- STABILITY-GATE BINDING: fully validated BEFORE any dataset/model
    # construction or CUDA initialization. ---
    identity = load_identity(args.identity, repo_root=root)
    validate_static_configuration(repo_root=root, identity_path=args.identity, check_git=True)
    identity_path = args.identity if args.identity is not None else root / "evaluation_identities/e12_k11_k12_power_evaluation.toml"
    identity_sha256 = _sha256_file(identity_path)
    binding = validate_stability_result_binding(args.stability_result, identity=identity, repo_root=root, check_git=True)

    image_count = identity["run_modes"][RUN_MODE_IMAGE_COUNT_KEYS[args.run_mode]]
    schema_name = identity["run_modes"][RUN_MODE_SCHEMA_KEYS[args.run_mode]]
    registered_class_count = identity["metrics"]["class_count"]

    if not args.overwrite and args.result.exists():
        raise K11K12PowerEvaluationError(f"--result {args.result} already exists; pass --overwrite for explicit resume/overwrite behavior")
    for path in (args.result, args.checkpoint, args.per_image_stats, _per_image_stats_npz_path(args.per_image_stats)):
        try:
            _reject_tracked_output_path(root, path)
        except Exception as error:  # noqa: BLE001 -- reuses the stability-gate helper's own
            # K11K12StabilityGateError; re-raised as this CLI's own error type so main()'s
            # single except clause remains the one fail-closed reporting boundary.
            raise K11K12PowerEvaluationError(str(error)) from error
    for path in (args.result, args.checkpoint, args.per_image_stats):
        path.parent.mkdir(parents=True, exist_ok=True)

    # --- PHASE A checkpoint validation: needs only the checkpoint document
    # and the per-image-stats artifact already on disk -- no dataset,
    # model, or CUDA. Everything checkable this early is checked here, so
    # a corrupted/tampered/inconsistent checkpoint fails closed before any
    # expensive construction, and -- critically -- before any dataset
    # index could be skipped or duplicated. ---
    phase_a_checkpoint: dict[str, Any] | None = None
    phase_a_stats: dict[str, Any] | None = None
    if args.resume and args.checkpoint.exists():
        phase_a_checkpoint = parse_strict_json_document(args.checkpoint, label="checkpoint")
        validate_checkpoint_structure(
            phase_a_checkpoint, identity=identity, identity_sha256=identity_sha256,
            run_mode=args.run_mode,
            stability_result_sha256=binding["stability_result_sha256"],
            finite_step_kernel_sha256=binding["finite_step_kernel_sha256"],
        )
        resume_dataset_index(phase_a_checkpoint)  # raises if already complete
        phase_a_stats = _load_per_image_stats(args.per_image_stats)
        validate_checkpoint_against_artifact(
            phase_a_checkpoint,
            stats_manifest=phase_a_stats["manifest"],
            stats_arrays=phase_a_stats["arrays"],
        )

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise K11K12PowerEvaluationError("--device cuda requested but CUDA is not available")

    from src.matched_k11_k12_identity import load_identity as load_matched_identity
    from src.e3_evaluation_identity import load_identity as load_e3_identity

    matched_identity = load_matched_identity(root / identity["parent_identity"]["matched_identity_path"], repo_root=root)
    e3_identity = load_e3_identity(root / matched_identity["parent_identities"]["e3_identity_path"], repo_root=root)

    alpha = matched_identity["propagation"]["alpha"]
    steps = matched_identity["propagation"]["steps"]
    affinity_power = matched_identity["graph"]["affinity_power"]
    crop = tuple(matched_identity["geometry"]["crop"])
    stride = tuple(matched_identity["geometry"]["stride"])
    # matched_identity["graph"]["maximum_rank"] registers the top-k rank
    # (12) that models.dinotext.cover_dr.matched_power_evaluator.
    # process_one_window builds its canonical graph with. That value is
    # intentionally NOT threaded through as a runtime parameter -- it is a
    # defining, structural characteristic of "k11 from k12" (the matched
    # kernel this evaluator binds to via finite_step_kernel_sha256 already
    # hard-requires graph12.k == 12, see build_matched_k11_from_k12), not
    # a tunable value -- so re-plumbing it risks the verified graph
    # construction logic for no behavioral benefit. Instead, fail closed
    # here if the registered identity value were ever anything other than
    # what the (unmodified, already-verified) kernel requires.
    if matched_identity["graph"]["maximum_rank"] != 12:
        raise K11K12PowerEvaluationError(
            "matched_identity.graph.maximum_rank must be exactly 12 -- the evaluator's graph "
            "construction is hard-bound to k=12 (see build_matched_k11_from_k12)"
        )
    # NOTE: matched_identity["geometry"]["align_corners"] (=true) describes
    # the PER-WINDOW patch-grid interpolation inside
    # inference.model.masks_from_patch_scores, which is hardcoded
    # align_corners=True internally by that function and is never read as
    # a parameter anywhere in this script. The FINAL crop-to-ori_shape
    # rescale (finalize_prediction, below) uses a different, separate
    # production value -- DINOTextSegInference.align_corners (=False) --
    # read directly off the live `inference` object, never off this
    # identity, so it can never silently drift from whatever production
    # inference actually uses.

    git_commit = _git(root, "rev-parse", "HEAD")
    git_branch = _git(root, "branch", "--show-current")

    inference, dataset = _build_inference(root, e3_identity, args.device, log_dir=args.result.parent)
    align_corners = inference.align_corners
    class_count = resolve_and_validate_class_count(
        inference, dataset, expected_class_count=registered_class_count
    )
    expected_image_ids = _expected_image_ids(dataset, image_count)
    image_order_digest = _image_order_digest(expected_image_ids)

    from models.dinotext.cover_dr import (
        compute_full_precision_metrics,
        finalize_prediction,
        stitch_one_image,
    )
    from segmentation.evaluation.sliding_window_geometry import SlidingWindowPlan, SpatialSize

    started_at = time.monotonic()

    # --- resume state ---
    next_dataset_index = 0
    completed_image_ids: list[str] = []
    dataset_indices: list[int] = []
    image_ids: list[str] = []
    label_rows: list[np.ndarray] = []
    intersect_k11_rows: list[np.ndarray] = []
    union_k11_rows: list[np.ndarray] = []
    pred_k11_rows: list[np.ndarray] = []
    intersect_k12_rows: list[np.ndarray] = []
    union_k12_rows: list[np.ndarray] = []
    pred_k12_rows: list[np.ndarray] = []
    windows_processed_total = 0

    if phase_a_checkpoint is not None:
        checkpoint = phase_a_checkpoint
        stats = phase_a_stats
        # --- PHASE B: dataset-dependent validation, only possible now
        # that the real dataset exists. ---
        if checkpoint["class_count"] != class_count:
            raise K11K12PowerEvaluationError("checkpoint.class_count disagrees with the live inference class count")
        validate_checkpoint_against_canonical_order(
            checkpoint, expected_image_ids, image_order_digest=image_order_digest
        )

        next_dataset_index = checkpoint["next_dataset_index"]
        completed_image_ids = list(checkpoint["completed_image_ids"])
        dataset_indices = list(int(i) for i in stats["manifest"]["dataset_indices"])
        image_ids = list(stats["manifest"]["image_ids"])
        arrays = stats["arrays"]
        label_rows = list(arrays["label"])
        intersect_k11_rows = list(arrays["intersect_k11"])
        union_k11_rows = list(arrays["union_k11"])
        pred_k11_rows = list(arrays["pred_k11"])
        intersect_k12_rows = list(arrays["intersect_k12"])
        union_k12_rows = list(arrays["union_k12"])
        pred_k12_rows = list(arrays["pred_k12"])
        windows_processed_total = checkpoint["windows_processed_total"]

    def _checkpoint_payload(*, complete: bool) -> dict[str, Any]:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return {
            "schema": CHECKPOINT_SCHEMA_NAME,
            "run_mode": args.run_mode,
            "identity": identity["identity"]["name"],
            "identity_sha256": identity_sha256,
            "matched_identity_sha256": identity["parent_identity"]["matched_identity_sha256"],
            "stability_result_sha256": binding["stability_result_sha256"],
            "finite_step_kernel_sha256": binding["finite_step_kernel_sha256"],
            "git_commit": git_commit,
            "class_count": class_count,
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
            payload, identity=identity, identity_sha256=identity_sha256,
            run_mode=args.run_mode, stability_result_sha256=binding["stability_result_sha256"],
            finite_step_kernel_sha256=binding["finite_step_kernel_sha256"], class_count=class_count,
        )
        write_checkpoint_atomically(args.checkpoint, payload)

    # --- main per-image loop ---
    while next_dataset_index < image_count:
        dataset_index = next_dataset_index
        prepared: PreparedDiagnosticImage = _extract_prepared_image(
            dataset, dataset_index, canonical_image_id=expected_image_ids[dataset_index],
        )
        if prepared.image_id != expected_image_ids[dataset_index]:
            raise K11K12PowerEvaluationError(
                f"dataset[{dataset_index}] image_id {prepared.image_id!r} disagrees with the "
                f"precomputed canonical order {expected_image_ids[dataset_index]!r}"
            )

        image_tensor = prepared.image_tensor
        if args.device == "cuda":
            image_tensor = image_tensor.cuda()
        batched = image_tensor.unsqueeze(0)

        plan = SlidingWindowPlan.build(
            image_size=SpatialSize(prepared.inference_height, prepared.inference_width),
            crop_size=SpatialSize(*crop),
            stride=SpatialSize(*stride),
        )
        stitched = stitch_one_image(
            inference, batched, plan,
            class_count=class_count, alpha=alpha, steps=steps, affinity_power=affinity_power,
        )

        pred_k11 = finalize_prediction(stitched.k11_stitched, prepared.img_metas, align_corners=align_corners)
        pred_k12 = finalize_prediction(stitched.k12_stitched, prepared.img_metas, align_corners=align_corners)
        pred_k11_np = pred_k11[0].detach().cpu().numpy()
        pred_k12_np = pred_k12[0].detach().cpu().numpy()

        pre_eval_k11 = dataset.pre_eval(pred_k11_np, dataset_index)[0]
        pre_eval_k12 = dataset.pre_eval(pred_k12_np, dataset_index)[0]
        intersect_k11, union_k11, area_pred_k11, area_label_k11 = (t.numpy() for t in pre_eval_k11)
        intersect_k12, union_k12, area_pred_k12, area_label_k12 = (t.numpy() for t in pre_eval_k12)
        if not np.array_equal(area_label_k11, area_label_k12):
            raise K11K12PowerEvaluationError(
                f"dataset[{dataset_index}]: ground-truth area arrays diverged between k11 and k12 variants"
            )

        dataset_indices.append(dataset_index)
        image_ids.append(prepared.image_id)
        label_rows.append(area_label_k11)
        intersect_k11_rows.append(intersect_k11)
        union_k11_rows.append(union_k11)
        pred_k11_rows.append(area_pred_k11)
        intersect_k12_rows.append(intersect_k12)
        union_k12_rows.append(union_k12)
        pred_k12_rows.append(area_pred_k12)
        windows_processed_total += stitched.window_count
        completed_image_ids.append(prepared.image_id)
        next_dataset_index = dataset_index + 1

        _write_per_image_stats_atomically(
            args.per_image_stats,
            schema_name=identity["artifacts"]["per_image_stats_manifest_schema_name"],
            class_count=class_count,
            dataset_indices=dataset_indices,
            image_ids=image_ids,
            label=np.stack(label_rows),
            intersect_k11=np.stack(intersect_k11_rows),
            union_k11=np.stack(union_k11_rows),
            pred_k11=np.stack(pred_k11_rows),
            intersect_k12=np.stack(intersect_k12_rows),
            union_k12=np.stack(union_k12_rows),
            pred_k12=np.stack(pred_k12_rows),
        )
        _write_checkpoint_self_validated(complete=False)

        # release this image's GPU state before moving to the next image
        del image_tensor, batched, stitched, pred_k11, pred_k12
        if args.device == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.monotonic() - started_at

    # --- FINAL COMPLETION CONTRACT: every one of these must hold before
    # any final result is written or PASS is printed. Failing any of them
    # raises (never marks complete, never prints PASS, never overwrites a
    # previous valid --result). ---
    if len(image_ids) != image_count:
        raise K11K12PowerEvaluationError(f"processed {len(image_ids)} images, expected {image_count}")
    if next_dataset_index != image_count:
        raise K11K12PowerEvaluationError(f"next_dataset_index {next_dataset_index} != expected image count {image_count}")
    if len(completed_image_ids) != image_count:
        raise K11K12PowerEvaluationError(f"completed_image_ids has {len(completed_image_ids)} entries, expected {image_count}")
    if dataset_indices != list(range(image_count)):
        raise K11K12PowerEvaluationError("processed dataset indices are not the exact canonical full range [0, image_count)")
    row_counts = {
        "label": len(label_rows), "intersect_k11": len(intersect_k11_rows), "union_k11": len(union_k11_rows),
        "pred_k11": len(pred_k11_rows), "intersect_k12": len(intersect_k12_rows), "union_k12": len(union_k12_rows),
        "pred_k12": len(pred_k12_rows),
    }
    if any(count != image_count for count in row_counts.values()):
        raise K11K12PowerEvaluationError(f"per-image row counts disagree with the expected image count: {row_counts}")

    pre_eval_k11_all = list(zip(intersect_k11_rows, union_k11_rows, pred_k11_rows, label_rows))
    pre_eval_k12_all = list(zip(intersect_k12_rows, union_k12_rows, pred_k12_rows, label_rows))
    metrics_k11_fraction = compute_full_precision_metrics(pre_eval_k11_all)
    metrics_k12_fraction = compute_full_precision_metrics(pre_eval_k12_all)
    metrics_k11 = {name: 100.0 * value for name, value in metrics_k11_fraction.items()}
    metrics_k12 = {name: 100.0 * value for name, value in metrics_k12_fraction.items()}
    delta_miou = metrics_k11["mIoU"] - metrics_k12["mIoU"]

    # windows_processed_total is correctly cumulative across any resume;
    # operation_telemetry is derived from it directly (never from a
    # per-run-ephemeral list of window records -- see
    # _operation_telemetry_for_window_count's docstring for why that would
    # silently under-report after a resume).
    telemetry_totals = _operation_telemetry_for_window_count(windows_processed_total)

    def _peak_gpu_memory_bytes() -> int:
        return int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0

    npz_sha256 = _sha256_file(_per_image_stats_npz_path(args.per_image_stats))
    per_image_manifest_sha256 = _sha256_file(args.per_image_stats)
    per_image_manifest = parse_strict_json_document(args.per_image_stats, label="per-image-stats manifest")
    if per_image_manifest["image_count"] != image_count:
        raise K11K12PowerEvaluationError(
            f"per-image-stats artifact reports {per_image_manifest['image_count']} images, expected {image_count}"
        )

    result = {
        "schema": schema_name,
        "run_mode": args.run_mode,
        "identity": identity["identity"]["name"],
        "identity_sha256": identity_sha256,
        "matched_identity_sha256": identity["parent_identity"]["matched_identity_sha256"],
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
        "metrics_k11": metrics_k11,
        "metrics_k12": metrics_k12,
        "delta_mIoU_percentage_points": delta_miou,
        "metric_unit": "percent_0_100",
        "metric_source": identity["metrics"]["precision_source"],
        "per_image_stats_manifest_path": str(args.per_image_stats),
        "per_image_stats_manifest_sha256": per_image_manifest_sha256,
        "per_image_stats_npz_sha256": npz_sha256,
        "stability_result_sha256": binding["stability_result_sha256"],
        "stability_schema": binding["stability_schema"],
        "gate_classification": binding["gate_classification"],
        "gate_git_commit": binding["gate_git_commit"],
        "gate_identity_sha256": binding["gate_identity_sha256"],
        "finite_step_kernel_sha256": binding["finite_step_kernel_sha256"],
        "graph_construction_sha256": binding["graph_construction_sha256"],
        "operation_telemetry": telemetry_totals,
        "phase_runtime_seconds": {"total": elapsed},
        "peak_gpu_memory_bytes": _peak_gpu_memory_bytes(),
        "resumed_from_checkpoint": args.resume,
        "source_git_branch": git_branch,
        "failure_reason": None,
    }

    # --- SELF-VERIFICATION: write to a temporary sibling path, read it
    # back, and run it through the exact same validator the standalone
    # verifier uses (verify_record) BEFORE ever touching --result or
    # printing PASS. A previous valid --result is never overwritten by a
    # result that fails this check. ---
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
        f"K11/K12 POWER EVALUATION {args.run_mode.upper()} PASS images={len(image_ids)} "
        f"mIoU_k11={metrics_k11['mIoU']:.6f} mIoU_k12={metrics_k12['mIoU']:.6f} "
        f"delta_mIoU={delta_miou:.6f} -> {args.result}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from models.dinotext.cover_dr import MatchedPowerEvaluatorError
    from src.k11_k12_stability_gate_identity import K11K12StabilityGateError

    # K11K12StabilityGateError is caught too: this CLI reuses several
    # validation helpers from diagnostics.run_k11_k12_stability
    # (_build_dataset_only, _build_inference, _extract_prepared_image,
    # _reject_tracked_output_path, ...) that raise that type directly --
    # never reimplemented here, so their failures must fail closed through
    # this same single reporting boundary rather than as an uncaught
    # traceback.
    #
    # ValueError is caught too, deliberately, alongside those three
    # domain-specific types -- never widened to Exception/BaseException.
    # Several already-verified, unmodified building blocks this evaluator
    # calls (never reimplemented here) use plain ValueError as their own
    # legitimate validation contract: compute_full_precision_metrics
    # rejects non-finite/degenerate/out-of-range sufficient statistics
    # this way, and write_checkpoint_atomically's own strict
    # allow_nan=False JSON serialization raises it too. Those are real,
    # reachable data-validation failures (e.g. a pathological all-ignored
    # ground truth), not programming errors, so they must fail closed
    # through this exact same boundary -- never as an uncaught traceback.
    # KeyboardInterrupt, SystemExit, and MemoryError are untouched by this:
    # none of them are ValueError subclasses, so none of them are caught
    # here.
    try:
        return _run_evaluation(args)
    except (K11K12PowerEvaluationError, MatchedPowerEvaluatorError, K11K12StabilityGateError, ValueError) as error:
        print(f"K11/K12 POWER EVALUATION FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
