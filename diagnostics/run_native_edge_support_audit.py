#!/usr/bin/env python3
"""Native cross-view directed-edge support audit evaluator.

For every sliding-window crop of an image, builds the same shared,
immutable k12 finite-step output already produced by the matched/
stitching-control evaluators (one backbone/snapshot pass, one directed
top-12 graph build per window, reused unmodified from
``models.dinotext.cover_dr`` -- never rebuilt for a comparison), then
measures how many OTHER windows of the same image natively (unrounded,
unquantized) contain an exact counterpart of each edge's endpoints and how
many of those independently reconstruct the same directed edge. This is a
read-only structural reachability measurement: it never deletes,
reweights, or edits a graph edge, and never renders a segmentation-
accuracy or pruning-efficacy verdict.

This is NOT ``main.py --eval``: it drives its own bounded per-image loop.
Checkpoint reading/writing and validation go exclusively through
:mod:`src.native_edge_support_checkpoint`.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OVS_ROOT = _REPO_ROOT / "src/open_vocabulary_segmentation"
for _path in (_REPO_ROOT, _OVS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.native_edge_support_checkpoint import (
    AFFINITY_RANK_KEYS,
    CHECKPOINT_SCHEMA_NAME,
    CROP_EDGE_BAND_KEYS,
    CROSS_TAB_TABLE_NAMES,
    DISPLACEMENT_BAND_KEYS,
    FUNNEL_KEYS,
    IMAGE_EDGE_BAND_KEYS,
    SUPPORT_COUNT_HISTOGRAM_KEYS,
    SUPPORT_FRACTION_HISTOGRAM_KEYS,
    UNDEFINED_REASON_KEYS,
    parse_strict_json_document,
    resume_dataset_index,
    validate_checkpoint_against_artifact,
    validate_checkpoint_against_canonical_order,
    validate_checkpoint_structure,
    validate_funnel_invariants,
)
from src.native_edge_support_identity import (
    RUN_MODE_IMAGE_COUNT_KEYS,
    RUN_MODE_SCHEMA_KEYS,
    NativeEdgeSupportAuditIdentityError,
    load_identity,
    validate_static_configuration,
)
from src.native_edge_support_report import classify_reachability, verify_record
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

GRID_SIZE = 32
CROSS_TAB_ROW_KEYS = ("defined_and_misclassified", "defined_and_correct", "undefined_and_misclassified", "undefined_and_correct")


def _empty_funnel() -> dict[str, int]:
    return {key: 0 for key in FUNNEL_KEYS}


def _empty_counter(keys: tuple[str, ...]) -> dict[str, int]:
    return {key: 0 for key in keys}


def _empty_cross_tabs() -> dict[str, dict[str, int]]:
    return {name: _empty_counter(CROSS_TAB_ROW_KEYS) for name in CROSS_TAB_TABLE_NAMES}


def _add_counter(total: dict[str, int], delta: dict[str, int]) -> None:
    for key, value in delta.items():
        total[key] += value


def _process_one_image(
    inference: Any, image_tensor: Any, plan: Any, prepared: Any, dataset: Any, dataset_index: int, *,
    alpha: float, steps: int, affinity_power: float, k: int, patch_size: int, align_corners: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Process one image's full native-edge-support audit: one shared
    per-window pass, cross-view support for every row, and GT reachability
    diagnostics computed only after every row's support record is frozen.
    Returns ``(per_image_funnel_plus_stats, telemetry_delta)``. All
    per-window/per-row state is local to this call and is released when it
    returns -- nothing image-scoped survives beyond the compact aggregate
    counters returned here."""
    from models.dinotext.cover_dr.native_edge_support import (
        build_edge_records_for_row,
        build_row_correctness_record,
        build_row_summary,
        compute_dense_adjacency,
        compute_window_support,
        crop_edge_band,
        native_window_pair_offset,
        process_one_window_for_audit,
    )
    from models.dinotext.cover_dr.stitching_control import UNIFORM_PROBABILITY, StitchAccumulator, build_stitch_weight, finalize_prediction

    windows = plan.windows
    window_count = len(windows)

    contexts = []
    telemetry_delta = {
        "sample_pulls": 1, "window_enumerations": 0, "backbone_snapshot_calls": 0, "graph_builds": 0,
        "propagation_calls": 0, "probability_interpolation_calls": 0,
    }
    for window in windows:
        context, telemetry = process_one_window_for_audit(
            inference, image_tensor, window, k=k, alpha=alpha, steps=steps, affinity_power=affinity_power,
            need_diagnostics=True,
        )
        contexts.append(context)
        telemetry_delta["window_enumerations"] += 1
        telemetry_delta["backbone_snapshot_calls"] += telemetry.backbone_snapshot_calls
        telemetry_delta["graph_builds"] += telemetry.graph_builds
        telemetry_delta["propagation_calls"] += telemetry.propagation_calls
        telemetry_delta["probability_interpolation_calls"] += telemetry.probability_interpolation_calls

    window_origins = tuple(context.origin for context in contexts)
    graphs = tuple(context.graph for context in contexts)
    dense_adjacencies = tuple(compute_dense_adjacency(g) for g in graphs)

    clamped_windows = sum(1 for context in contexts if context.is_clamped)
    non_clamped_windows = window_count - clamped_windows
    aligned_window_pairs = 0
    unaligned_window_pairs = 0
    for a in range(window_count):
        for b in range(a + 1, window_count):
            offset = native_window_pair_offset(
                window_origins[a][0], window_origins[a][1], window_origins[b][0], window_origins[b][1],
                patch_size=patch_size,
            )
            if offset is None:
                unaligned_window_pairs += 1
            else:
                aligned_window_pairs += 1

    h_img, w_img = plan.image_size.as_tuple()
    accumulator = StitchAccumulator(
        UNIFORM_PROBABILITY, class_count=inference.num_classes, image_hw=(h_img, w_img),
        expected_window_count=window_count, device=image_tensor.device,
    )
    for window, context in zip(windows, contexts):
        h, w = context.probability_crop.shape[-2:]
        weight = build_stitch_weight(h, w, kind="uniform", dtype=context.probability_crop.dtype, device=context.probability_crop.device)
        accumulator.add_window(window, context.probability_crop, weight=weight)
    stitched = accumulator.finalize()
    canonical_label = finalize_prediction(stitched, prepared.img_metas, align_corners=align_corners)[0]

    gt_label_map = dataset.get_gt_seg_map_by_idx(dataset_index)
    ignore_index = dataset.ignore_index
    ori_hw = tuple(int(v) for v in gt_label_map.shape[:2])

    import torch

    gt_label_tensor = torch.as_tensor(gt_label_map, device=canonical_label.device)

    funnel = _empty_funnel()
    funnel["images"] = 1
    funnel["windows"] = window_count
    funnel["graph_rows"] = window_count * 1024
    funnel["directed_edges"] = window_count * 1024 * 12
    funnel["aligned_window_pairs"] = aligned_window_pairs
    funnel["unaligned_window_pairs"] = unaligned_window_pairs
    funnel["clamped_windows"] = clamped_windows
    funnel["non_clamped_windows"] = non_clamped_windows

    undefined_reason_counts = _empty_counter(UNDEFINED_REASON_KEYS)
    support_count_histogram = _empty_counter(SUPPORT_COUNT_HISTOGRAM_KEYS)
    support_fraction_histogram = _empty_counter(SUPPORT_FRACTION_HISTOGRAM_KEYS)
    crop_edge_band_histogram = _empty_counter(CROP_EDGE_BAND_KEYS)
    image_edge_band_histogram = _empty_counter(IMAGE_EDGE_BAND_KEYS)
    displacement_band_histogram = _empty_counter(DISPLACEMENT_BAND_KEYS)
    affinity_rank_histogram = _empty_counter(AFFINITY_RANK_KEYS)
    cross_tabs = _empty_cross_tabs()
    ignored_gt_count = 0

    from models.dinotext.cover_dr.native_edge_support import SUPPORT_FRACTION_BUCKETS

    for source_index, (window, context) in enumerate(zip(windows, contexts)):
        observer_count_tensor, support_count_tensor, reverse_support_count_tensor, aligned_mask = compute_window_support(
            source_index, window_origins=window_origins, graphs=graphs, dense_adjacencies=dense_adjacencies,
            grid_size=GRID_SIZE, patch_size=patch_size,
        )
        if window_count > 1:
            funnel["rows_with_other_crop"] += 1024
        aligned_observer_window_count = sum(1 for a in aligned_mask if a)
        if aligned_observer_window_count > 0:
            funnel["rows_with_aligned_observer"] += 1024

        graph = context.graph
        observer_count_rows = observer_count_tensor.tolist()
        support_count_rows = support_count_tensor.tolist()
        reverse_support_count_rows = reverse_support_count_tensor.tolist()
        neighbor_indices_rows = graph.neighbor_indices.tolist()
        weight_rows = graph.transition_weights.tolist()
        source_origin_row, source_origin_col = window_origins[source_index]

        for row_index in range(1024):
            edges = build_edge_records_for_row(
                window_index=source_index, source_node=row_index,
                neighbor_indices_row=neighbor_indices_rows[row_index], weights_row=weight_rows[row_index],
                observer_count_row=observer_count_rows[row_index], support_count_row=support_count_rows[row_index],
                window_count=window_count,
            )
            row_summary = build_row_summary(edges)

            patch_row, patch_col = divmod(row_index, GRID_SIZE)
            for edge in edges:
                if edge.defined:
                    funnel["edges_with_observer"] += 1
                    support_count_histogram[str(edge.support_count)] += 1
                    bucket = SUPPORT_FRACTION_BUCKETS[0] if edge.support_fraction == 0.0 else SUPPORT_FRACTION_BUCKETS[
                        min(int(math.ceil(edge.support_fraction * 10)), 10)
                    ]
                    support_fraction_histogram[bucket] += 1
                    if edge.support_count == edge.observer_count:
                        funnel["unanimous_support_edges"] += 1
                    if edge.support_count == 0 and reverse_support_count_rows[row_index][edge.neighbor_rank] > 0:
                        funnel["direction_reversal_only_cases"] += 1
                else:
                    undefined_reason_counts[edge.undefined_reason] += 1

                dest_row, dest_col = divmod(edge.destination_node, GRID_SIZE)
                displacement = round(math.sqrt((dest_row - patch_row) ** 2 + (dest_col - patch_col) ** 2))
                displacement_band_histogram[crop_edge_band(displacement)] += 1
                affinity_rank_histogram[str(edge.neighbor_rank)] += 1

            if row_summary.any_defined:
                funnel["rows_any_defined"] += 1
                if row_summary.least_support_tied:
                    funnel["rows_tied_for_least_support"] += 1
                else:
                    funnel["rows_with_unique_least_support"] += 1
            if row_summary.all_defined:
                funnel["rows_all_defined"] += 1
            if row_summary.any_defined_zero_support:
                funnel["rows_any_defined_zero_support"] += 1

            crop_distance = min(patch_row, patch_col, GRID_SIZE - 1 - patch_row, GRID_SIZE - 1 - patch_col)
            crop_edge_band_histogram[crop_edge_band(crop_distance)] += 1

            centre_row = source_origin_row + patch_row * patch_size + patch_size // 2
            centre_col = source_origin_col + patch_col * patch_size + patch_size // 2
            image_distance = min(centre_row, centre_col, h_img - 1 - centre_row, w_img - 1 - centre_col) // patch_size
            image_edge_band_histogram[crop_edge_band(max(image_distance, 0))] += 1

            correctness = build_row_correctness_record(
                window_index=source_index, source_node=row_index,
                window_origin_row=window_origins[source_index][0], window_origin_col=window_origins[source_index][1],
                patch_row=patch_row, patch_col=patch_col, patch_size=patch_size,
                source_patch_scores_row=context.patch_scores[row_index], canonical_stitched_label_map=canonical_label,
                gt_label_map=gt_label_tensor, ignore_index=ignore_index, processed_hw=(h_img, w_img), ori_hw=ori_hw,
            )
            if correctness.gt_ignored:
                ignored_gt_count += 1
                continue

            misclassified_source = correctness.source_row_misclassified
            misclassified_stitched = correctness.canonical_stitched_misclassified
            if misclassified_source:
                funnel["misclassified_rows"] += 1
                if row_summary.any_defined:
                    funnel["misclassified_rows_any_defined"] += 1
                if row_summary.all_defined:
                    funnel["misclassified_rows_all_defined"] += 1
            elif row_summary.any_defined:
                funnel["correct_rows_any_defined"] += 1

            defined_key_source = "defined" if row_summary.any_defined else "undefined"
            all_defined_key_source = "defined" if row_summary.all_defined else "undefined"
            correctness_key_source = "misclassified" if misclassified_source else "correct"
            correctness_key_stitched = "misclassified" if misclassified_stitched else "correct"
            cross_tabs["support_defined_vs_source_correct"][f"{defined_key_source}_and_{correctness_key_source}"] += 1
            cross_tabs["all_defined_vs_source_correct"][f"{all_defined_key_source}_and_{correctness_key_source}"] += 1
            cross_tabs["support_defined_vs_stitched_correct"][f"{defined_key_source}_and_{correctness_key_stitched}"] += 1
            cross_tabs["all_defined_vs_stitched_correct"][f"{all_defined_key_source}_and_{correctness_key_stitched}"] += 1

    validate_funnel_invariants(funnel)

    return (
        {
            "funnel": funnel, "undefined_reason_counts": undefined_reason_counts,
            "support_count_histogram": support_count_histogram, "support_fraction_histogram": support_fraction_histogram,
            "crop_edge_band_histogram": crop_edge_band_histogram, "image_edge_band_histogram": image_edge_band_histogram,
            "displacement_band_histogram": displacement_band_histogram, "affinity_rank_histogram": affinity_rank_histogram,
            "correctness_cross_tabs": cross_tabs, "ignored_gt_count": ignored_gt_count,
        },
        telemetry_delta,
    )


def _write_per_image_stats_atomically(
    manifest_path: Path, *, schema_name: str, class_count: int, dataset_indices: list[int],
    image_ids: list[str], per_image_funnel: list[dict[str, int]],
) -> None:
    manifest = {
        "schema": schema_name, "class_count": class_count, "image_count": len(image_ids),
        "dataset_indices": dataset_indices, "image_ids": image_ids,
        "image_order_digest": _image_order_digest(image_ids), "per_image_funnel": per_image_funnel,
    }
    write_checkpoint_atomically(manifest_path, manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Native cross-view directed-edge support audit evaluator.")
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

    identity = load_identity(args.identity, repo_root=root)
    validate_static_configuration(repo_root=root, identity_path=args.identity, check_git=True)
    identity_path = args.identity if args.identity is not None else root / "evaluation_identities/e12_native_edge_support_audit.toml"
    identity_sha256 = _sha256_file(identity_path)

    power_identity_path = root / identity["parent_identity"]["power_evaluation_identity_path"]
    power_identity = load_power_identity(power_identity_path, repo_root=root)
    validate_stability_result_binding(args.stability_result, identity=power_identity, repo_root=root, check_git=True)

    image_count = identity["run_modes"][RUN_MODE_IMAGE_COUNT_KEYS[args.run_mode]]
    schema_name = identity["run_modes"][RUN_MODE_SCHEMA_KEYS[args.run_mode]]
    registered_class_count = identity["dataset"]["classes"]

    if not args.overwrite and args.result.exists():
        raise NativeEdgeSupportAuditIdentityError(f"--result {args.result} already exists; pass --overwrite for explicit resume/overwrite behavior")
    for path in (args.result, args.checkpoint, args.per_image_stats):
        try:
            _reject_tracked_output_path(root, path)
        except Exception as error:  # noqa: BLE001 -- re-raised as this CLI's own error type
            raise NativeEdgeSupportAuditIdentityError(str(error)) from error
    for path in (args.result, args.checkpoint, args.per_image_stats):
        path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and args.checkpoint.exists():
        phase_a_checkpoint = parse_strict_json_document(args.checkpoint, label="checkpoint")
        validate_checkpoint_structure(phase_a_checkpoint, identity=identity, identity_sha256=identity_sha256, run_mode=args.run_mode)
        resume_dataset_index(phase_a_checkpoint)
        phase_a_stats = parse_strict_json_document(args.per_image_stats, label="per-image-stats manifest")
        validate_checkpoint_against_artifact(phase_a_checkpoint, stats_manifest=phase_a_stats)

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise NativeEdgeSupportAuditIdentityError("--device cuda requested but CUDA is not available")

    from src.matched_k11_k12_identity import load_identity as load_matched_identity
    from src.e3_evaluation_identity import load_identity as load_e3_identity

    matched_identity = load_matched_identity(root / identity["parent_identity"]["matched_identity_path"], repo_root=root)
    e3_identity = load_e3_identity(root / matched_identity["parent_identities"]["e3_identity_path"], repo_root=root)

    alpha = matched_identity["propagation"]["alpha"]
    steps = matched_identity["propagation"]["steps"]
    affinity_power = matched_identity["graph"]["affinity_power"]
    k = identity["propagation"]["k"]
    patch_size = identity["geometry"]["patch_size"][0]

    git_commit = _git(root, "rev-parse", "HEAD")
    git_branch = _git(root, "branch", "--show-current")

    inference, dataset = _build_inference(root, e3_identity, args.device, log_dir=args.result.parent)
    align_corners = inference.align_corners
    class_count = resolve_and_validate_class_count(inference, dataset, expected_class_count=registered_class_count)
    expected_image_ids = _expected_image_ids(dataset, image_count)
    image_order_digest = _image_order_digest(expected_image_ids)

    from segmentation.evaluation.sliding_window_geometry import SlidingWindowPlan, SpatialSize

    started_at = time.monotonic()

    dataset_indices: list[int] = []
    image_ids: list[str] = []
    per_image_funnel_rows: list[dict[str, int]] = []
    completed_image_ids: list[str] = []
    windows_processed_total = 0

    running_funnel = _empty_funnel()
    running_undefined_reason_counts = _empty_counter(UNDEFINED_REASON_KEYS)
    running_support_count_histogram = _empty_counter(SUPPORT_COUNT_HISTOGRAM_KEYS)
    running_support_fraction_histogram = _empty_counter(SUPPORT_FRACTION_HISTOGRAM_KEYS)
    running_crop_edge_band_histogram = _empty_counter(CROP_EDGE_BAND_KEYS)
    running_image_edge_band_histogram = _empty_counter(IMAGE_EDGE_BAND_KEYS)
    running_displacement_band_histogram = _empty_counter(DISPLACEMENT_BAND_KEYS)
    running_affinity_rank_histogram = _empty_counter(AFFINITY_RANK_KEYS)
    running_cross_tabs = _empty_cross_tabs()
    running_ignored_gt_count = 0
    next_dataset_index = 0

    telemetry_totals = {
        "sample_pulls": 0, "window_enumerations": 0, "backbone_snapshot_calls": 0, "graph_builds": 0,
        "propagation_calls": 0, "probability_interpolation_calls": 0, "cross_view_comparisons_total": 0,
    }

    if args.resume and args.checkpoint.exists():
        checkpoint = parse_strict_json_document(args.checkpoint, label="checkpoint")
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256, run_mode=args.run_mode, class_count=class_count)
        validate_checkpoint_against_canonical_order(checkpoint, expected_image_ids, image_order_digest=image_order_digest)
        next_dataset_index = resume_dataset_index(checkpoint)
        stats = parse_strict_json_document(args.per_image_stats, label="per-image-stats manifest")
        validate_checkpoint_against_artifact(checkpoint, stats_manifest=stats)
        dataset_indices = list(stats["dataset_indices"])
        image_ids = list(stats["image_ids"])
        per_image_funnel_rows = list(stats["per_image_funnel"])
        completed_image_ids = list(checkpoint["completed_image_ids"])
        windows_processed_total = checkpoint["windows_processed_total"]
        running_funnel = dict(checkpoint["funnel"])
        running_undefined_reason_counts = dict(checkpoint["undefined_reason_counts"])
        running_support_count_histogram = dict(checkpoint["support_count_histogram"])
        running_support_fraction_histogram = dict(checkpoint["support_fraction_histogram"])
        running_crop_edge_band_histogram = dict(checkpoint["crop_edge_band_histogram"])
        running_image_edge_band_histogram = dict(checkpoint["image_edge_band_histogram"])
        running_displacement_band_histogram = dict(checkpoint["displacement_band_histogram"])
        running_affinity_rank_histogram = dict(checkpoint["affinity_rank_histogram"])
        running_cross_tabs = {name: dict(table) for name, table in checkpoint["correctness_cross_tabs"].items()}
        running_ignored_gt_count = checkpoint["ignored_gt_count"]
        telemetry_totals["sample_pulls"] = len(completed_image_ids)
        telemetry_totals["window_enumerations"] = windows_processed_total
        telemetry_totals["backbone_snapshot_calls"] = windows_processed_total
        telemetry_totals["graph_builds"] = windows_processed_total
        telemetry_totals["propagation_calls"] = windows_processed_total
        telemetry_totals["probability_interpolation_calls"] = windows_processed_total

    def _checkpoint_payload(*, complete: bool) -> dict[str, Any]:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return {
            "schema": CHECKPOINT_SCHEMA_NAME, "run_mode": args.run_mode, "identity": identity["identity"]["name"],
            "identity_sha256": identity_sha256,
            "stitching_control_identity_sha256": identity["parent_identity"]["stitching_control_identity_sha256"],
            "power_evaluation_identity_sha256": identity["parent_identity"]["power_evaluation_identity_sha256"],
            "matched_identity_sha256": identity["parent_identity"]["matched_identity_sha256"],
            "e3_identity_sha256": identity["parent_identity"]["e3_identity_sha256"],
            "git_commit": git_commit, "class_count": class_count, "image_count_expected": image_count,
            "image_order_digest": image_order_digest, "next_dataset_index": next_dataset_index,
            "completed_image_ids": completed_image_ids, "images_completed_count": len(completed_image_ids),
            "windows_processed_total": windows_processed_total, "rows_processed_total": windows_processed_total * 1024,
            "edges_processed_total": windows_processed_total * 1024 * 12, "funnel": running_funnel,
            "undefined_reason_counts": running_undefined_reason_counts,
            "support_count_histogram": running_support_count_histogram,
            "support_fraction_histogram": running_support_fraction_histogram,
            "crop_edge_band_histogram": running_crop_edge_band_histogram,
            "image_edge_band_histogram": running_image_edge_band_histogram,
            "displacement_band_histogram": running_displacement_band_histogram,
            "affinity_rank_histogram": running_affinity_rank_histogram,
            "correctness_cross_tabs": running_cross_tabs, "ignored_gt_count": running_ignored_gt_count,
            "complete": complete, "created_at_utc": now, "updated_at_utc": now,
        }

    def _write_checkpoint_self_validated(*, complete: bool) -> None:
        payload = _checkpoint_payload(complete=complete)
        validate_checkpoint_structure(payload, identity=identity, identity_sha256=identity_sha256, run_mode=args.run_mode, class_count=class_count)
        write_checkpoint_atomically(args.checkpoint, payload)

    while next_dataset_index < image_count:
        dataset_index = next_dataset_index
        prepared = _extract_prepared_image(dataset, dataset_index, canonical_image_id=expected_image_ids[dataset_index])
        if prepared.image_id != expected_image_ids[dataset_index]:
            raise NativeEdgeSupportAuditIdentityError(
                f"dataset[{dataset_index}] image_id {prepared.image_id!r} disagrees with the precomputed canonical order"
            )

        image_tensor = prepared.image_tensor.to(device=args.device).unsqueeze(0)
        h_img, w_img = prepared.inference_height, prepared.inference_width
        plan = SlidingWindowPlan.build(
            image_size=SpatialSize(h_img, w_img),
            crop_size=SpatialSize(*identity["geometry"]["crop"]),
            stride=SpatialSize(*identity["geometry"]["stride"]),
        )

        image_stats, telemetry_delta = _process_one_image(
            inference, image_tensor, plan, prepared, dataset, dataset_index,
            alpha=alpha, steps=steps, affinity_power=affinity_power, k=k, patch_size=patch_size, align_corners=align_corners,
        )

        dataset_indices.append(dataset_index)
        image_ids.append(prepared.image_id)
        per_image_funnel_rows.append(image_stats["funnel"])
        windows_processed_total += image_stats["funnel"]["windows"]
        completed_image_ids.append(prepared.image_id)
        next_dataset_index = dataset_index + 1

        _add_counter(running_funnel, image_stats["funnel"])
        _add_counter(running_undefined_reason_counts, image_stats["undefined_reason_counts"])
        _add_counter(running_support_count_histogram, image_stats["support_count_histogram"])
        _add_counter(running_support_fraction_histogram, image_stats["support_fraction_histogram"])
        _add_counter(running_crop_edge_band_histogram, image_stats["crop_edge_band_histogram"])
        _add_counter(running_image_edge_band_histogram, image_stats["image_edge_band_histogram"])
        _add_counter(running_displacement_band_histogram, image_stats["displacement_band_histogram"])
        _add_counter(running_affinity_rank_histogram, image_stats["affinity_rank_histogram"])
        for table_name in CROSS_TAB_TABLE_NAMES:
            _add_counter(running_cross_tabs[table_name], image_stats["correctness_cross_tabs"][table_name])
        running_ignored_gt_count += image_stats["ignored_gt_count"]

        for name in ("sample_pulls", "window_enumerations", "backbone_snapshot_calls", "graph_builds", "propagation_calls", "probability_interpolation_calls"):
            telemetry_totals[name] += telemetry_delta[name]
        telemetry_totals["cross_view_comparisons_total"] += image_stats["funnel"]["directed_edges"] * max(image_stats["funnel"]["windows"] - 1, 0)

        _write_per_image_stats_atomically(
            args.per_image_stats, schema_name=identity["run_modes"]["per_image_stats_manifest_schema_name"],
            class_count=class_count, dataset_indices=dataset_indices, image_ids=image_ids,
            per_image_funnel=per_image_funnel_rows,
        )
        _write_checkpoint_self_validated(complete=False)

        del image_tensor
        if args.device == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.monotonic() - started_at

    if len(image_ids) != image_count or next_dataset_index != image_count or len(completed_image_ids) != image_count:
        raise NativeEdgeSupportAuditIdentityError("processed image count does not equal the expected image count")
    if dataset_indices != list(range(image_count)):
        raise NativeEdgeSupportAuditIdentityError("processed dataset indices are not the exact canonical full range [0, image_count)")

    decision_output, decision_rationale = classify_reachability(
        running_funnel, identity, ignored_gt_count=running_ignored_gt_count,
    )

    result = {
        "schema": schema_name, "run_mode": args.run_mode, "identity": identity["identity"]["name"],
        "identity_sha256": identity_sha256,
        "stitching_control_identity_sha256": identity["parent_identity"]["stitching_control_identity_sha256"],
        "power_evaluation_identity_sha256": identity["parent_identity"]["power_evaluation_identity_sha256"],
        "matched_identity_sha256": identity["parent_identity"]["matched_identity_sha256"],
        "e3_identity_sha256": identity["parent_identity"]["e3_identity_sha256"],
        "git_commit": git_commit, "complete": True, "final": False,
        "device": args.device, "gpu_model": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "torch_version": torch.__version__, "cuda_version": getattr(torch.version, "cuda", None) or "none",
        "image_count_expected": image_count, "image_count_processed": len(image_ids),
        "image_order_digest": image_order_digest, "windows_processed_total": windows_processed_total,
        "rows_processed_total": windows_processed_total * 1024, "edges_processed_total": windows_processed_total * 1024 * 12,
        "class_count": class_count, "funnel": running_funnel, "undefined_reason_counts": running_undefined_reason_counts,
        "support_count_histogram": running_support_count_histogram,
        "support_fraction_histogram": running_support_fraction_histogram,
        "crop_edge_band_histogram": running_crop_edge_band_histogram,
        "image_edge_band_histogram": running_image_edge_band_histogram,
        "displacement_band_histogram": running_displacement_band_histogram,
        "affinity_rank_histogram": running_affinity_rank_histogram,
        "correctness_cross_tabs": running_cross_tabs, "ignored_gt_count": running_ignored_gt_count,
        "ranking_definition": " -> ".join(identity["ranking"]["criteria_in_order"]),
        "decision_output": decision_output, "decision_rationale": decision_rationale,
        "per_image_stats_manifest_path": str(args.per_image_stats),
        "per_image_stats_manifest_sha256": _sha256_file(args.per_image_stats),
        "operation_telemetry": telemetry_totals,
        "phase_runtime_seconds": {"total": elapsed, "shared": elapsed},
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

    misclassified = running_funnel["misclassified_rows"]
    coverage = running_funnel["misclassified_rows_any_defined"] / misclassified if misclassified > 0 else float("nan")
    print(
        f"NATIVE EDGE SUPPORT AUDIT {args.run_mode.upper()} PASS images={len(image_ids)} "
        f"windows={windows_processed_total} misclassified_rows={misclassified} "
        f"support_defined_coverage_among_misclassified={coverage:.6f} -> {args.result}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from models.dinotext.cover_dr.native_edge_support import NativeEdgeSupportError
    from models.dinotext.cover_dr.stitching_control import StitchingControlError
    from src.k11_k12_power_evaluation_identity import K11K12PowerEvaluationError

    try:
        return _run_evaluation(args)
    except (NativeEdgeSupportAuditIdentityError, NativeEdgeSupportError, StitchingControlError, K11K12PowerEvaluationError, ValueError) as error:
        print(f"NATIVE EDGE SUPPORT AUDIT FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
