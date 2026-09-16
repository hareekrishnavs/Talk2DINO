"""DFP Phase 1B: the BALP development-image processor and 350-image
development-run orchestration. Implements exactly E22's canonical
inference integration (Section B3), GT isolation (B4), development-
only population enforcement (B5), metric sufficient statistics (B6),
and fresh/resume/finalize orchestration on top of the atomic
checkpoint/result artifacts (B8/B9).

Reuses the repository's already-committed, unmodified production
providers (`src.dfp_phase1b_calibration_production`) and canonical
Phase 1A primitives (graph, recurrence, stitching, final rescale) --
never a second, independently-derived inference path, and never any
masked-forward/component/pullback/E19/E20/E21 dependency. Per window:
one model forward, one graph, one 320-step k12 recurrence, from which
ALL FIVE horizon snapshots are read; recurrence/model/graph are never
re-run per horizon or per BALP candidate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import src.dfp_phase1b_lightweight_boundary as balp
import src.dfp_phase1b_lightweight_boundary_artifact as art
import src.dfp_phase1b_lightweight_boundary_identity as e22


class DfpPhase1bLightweightBoundaryExecuteError(ValueError):
    """Raised on any BALP development-runner invariant violation.
    Always fail closed. Never raised for, and never catches,
    KeyboardInterrupt / SystemExit / MemoryError -- those always
    propagate unmodified."""


VARIANT_IDS: tuple[str, ...] = tuple(e22.CANONICAL_CONTROLS) + tuple(balp.CANDIDATE_IDS)


@dataclass(frozen=True)
class BalpContracts:
    """The minimal contract bundle BALP's inference integration needs
    -- unlike the full masked-forward `ComponentContractsBundle`, BALP
    needs no component/masked_forward contract at all (it constructs no
    component and runs no masked recurrence)."""

    recurrence: Any
    stitching: Any
    pullback: Any
    final_resolution: Any


@dataclass(frozen=True)
class ExecutionProviders:
    """Injectable real-or-synthetic providers. Production callers build
    these via `dfp_phase1b_calibration_production`'s real builders;
    tests inject narrow fake stand-ins. This module never constructs a
    provider itself beyond what is passed in."""

    dataset_resources: Any
    image_preparer: Any
    inference: Any
    protocol_identity: Mapping[str, Any]
    contracts: Any
    class_count: int
    affinity_power: float
    windows_per_image: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class ImageBalpOutcome:
    """Everything computed for one development image BEFORE any GT
    access: five control label maps, 16 candidate label maps, and the
    intermediate masks needed for preservation-fraction bookkeeping."""

    canonical_id: str
    ori_shape: tuple[int, int]
    kernel_result: balp.BalpKernelResult


def process_development_image(
    inference: Any, prepared: Any, canonical_id: str, *, protocol_identity: Mapping[str, Any],
    contracts: Any, class_count: int, affinity_power: float, selected_windows_for_image: Sequence[int],
) -> ImageBalpOutcome:
    """Steps B3.2-B3.11: prepare -> per-window (forward, graph, one
    320-step recurrence, capture T0/T16/T32/T64/T320) -> stitch each
    horizon across windows -> final crop/rescale to ori_shape -> run
    BALP once. Never reads GT. Never runs masked-forward, builds a
    component, or computes a pullback/membership."""
    import torch

    # Import order matters here without being a hidden side effect: these
    # `src.dfp_phase1a_*` modules are genuinely required below, and (as an
    # unavoidable consequence of their OWN normal operation, not a
    # contrived "import just to mutate sys.path" hack) importing any of
    # them is what makes the top-level `models`/`segmentation` packages --
    # which live under `src/open_vocabulary_segmentation/`, off the
    # default Python path -- importable at all. Importing them before
    # `models.dinotext.cover_dr.graph` makes that resolution deterministic
    # from this function's own source order alone, on the very first call
    # in a fresh process, regardless of what any other module happened to
    # import earlier. No new `sys.path` mutation is added here or anywhere
    # in this module.
    from src.dfp_phase1a_horizon_stitching import WindowHorizonCaptures, stitch_dfp_phase1a_horizons
    from src.dfp_phase1a_recurrence_capture import capture_dfp_phase1a_trajectory
    from src.dfp_phase1a_stitching_pullback import build_stitching_pullback_plan
    from src.dfp_phase1a_final_rescale_pullback import apply_final_rescale_forward
    from models.dinotext.cover_dr.graph import build_directed_topk_graph

    image_tensor = prepared.image_tensor
    img_shape = (prepared.inference_height, prepared.inference_width)
    ori_shape = prepared.ori_shape

    pullback_plan = build_stitching_pullback_plan(img_shape, contract=contracts.pullback)
    selected_set = set(selected_windows_for_image)

    window_captures = []
    for window in pullback_plan.plan.windows:
        if window.index not in selected_set:
            continue
        crop_rows, crop_cols = window.crop_slice
        crop = image_tensor[:, :, crop_rows, crop_cols]
        with torch.no_grad():
            snapshot = inference.model.generate_patch_snapshot(crop, inference.text_embedding)
        s0 = snapshot.unary_scores[0]
        if s0.shape[1] != class_count:
            raise DfpPhase1bLightweightBoundaryExecuteError(
                f"image {canonical_id} window {window.index}: snapshot class dimension {s0.shape[1]} disagrees with class_count {class_count}"
            )
        dino_features = snapshot.dino_features[0]
        graph12 = build_directed_topk_graph(dino_features, k=12, affinity_power=affinity_power)
        trace = capture_dfp_phase1a_trajectory(graph12, s0, protocol_identity=protocol_identity)
        if trace.steps_completed != contracts.recurrence.steps:
            raise DfpPhase1bLightweightBoundaryExecuteError(
                f"image {canonical_id} window {window.index}: recurrence completed {trace.steps_completed} updates, expected {contracts.recurrence.steps}"
            )
        window_captures.append(WindowHorizonCaptures(window_index=window.index, captures=trace.snapshots))

    if not window_captures:
        raise DfpPhase1bLightweightBoundaryExecuteError(f"image {canonical_id}: no selected windows to process")

    stitched = stitch_dfp_phase1a_horizons(
        pullback_plan.plan, tuple(window_captures), contract=contracts.stitching, class_count=class_count,
    )
    scores_final: dict[int, Any] = {}
    for horizon, tensor in stitched.items():
        rescaled = apply_final_rescale_forward(tensor, img_shape=img_shape, ori_shape=ori_shape, contract=contracts.final_resolution)
        scores_final[horizon] = rescaled.squeeze(0).detach()

    observed = frozenset(scores_final.keys())
    if not frozenset(balp.HORIZON_SET).issubset(observed):
        raise DfpPhase1bLightweightBoundaryExecuteError(
            f"image {canonical_id}: stitched horizons {sorted(observed)} do not cover the frozen set {balp.HORIZON_SET}"
        )
    balp_scores = {h: scores_final[h] for h in balp.HORIZON_SET}
    reference_shape = balp_scores[0].shape
    for h in balp.HORIZON_SET[1:]:
        if balp_scores[h].shape != reference_shape:
            raise DfpPhase1bLightweightBoundaryExecuteError(f"image {canonical_id}: horizon {h} shape {balp_scores[h].shape} disagrees with horizon 0 shape {reference_shape}")

    kernel_result = balp.compute_balp(balp_scores, class_count=class_count)
    return ImageBalpOutcome(canonical_id=canonical_id, ori_shape=tuple(ori_shape), kernel_result=kernel_result)


# =======================================================================
# Metric sufficient-statistic accumulation (B6). GT is read ONLY here,
# strictly after `process_development_image` has already returned
# every prediction. Reuses the existing bound `ConfusionStatistics`
# machinery -- never a new metric formula, never an averaged per-image
# mIoU.
# =======================================================================


def compute_confusion_matrix(prediction: np.ndarray, ground_truth: np.ndarray, *, class_count: int, ignore_index: int) -> np.ndarray:
    """row=GT class, col=predicted class, evaluated pixels only
    (GT != ignore_index) -- the exact sufficient statistic
    `src.dfp_phase1a_analysis.ConfusionStatistics`/`compute_semantic_metrics`
    already consume; never a new formula."""
    valid = ground_truth != ignore_index
    gt_valid = ground_truth[valid].astype(np.int64)
    pred_valid = prediction[valid].astype(np.int64)
    cm = np.zeros((class_count, class_count), dtype=np.int64)
    if gt_valid.size:
        idx = gt_valid * class_count + pred_valid
        counts = np.bincount(idx, minlength=class_count * class_count)
        cm = counts.reshape(class_count, class_count)
    return cm


@dataclass(frozen=True)
class PerImageMetricRecord:
    per_variant_confusion: Mapping[str, list]           # variant -> [class_count,class_count] nested list
    per_variant_scalars: Mapping[str, Mapping[str, int]]  # variant -> 6 scalar fields


def accumulate_image_metrics(
    outcome: ImageBalpOutcome, *, ground_truth: np.ndarray, ignore_index: int, class_count: int,
) -> PerImageMetricRecord:
    """GT-isolation boundary: everything above this function has
    already finalized every prediction; only THIS function ever reads
    `ground_truth`. Ignore mask is used only here, for confusion-
    statistic accumulation -- never for any prediction/boundary/
    stability/late-flip/mask/arbitration computation."""
    kr = outcome.kernel_result
    labels_by_variant: dict[str, np.ndarray] = {}
    for control in e22.CANONICAL_CONTROLS:
        labels_by_variant[control] = kr.control_labels[control].detach().cpu().numpy()
    for cid in balp.CANDIDATE_IDS:
        labels_by_variant[cid] = kr.candidate_labels[cid].detach().cpu().numpy()

    y320 = labels_by_variant["T320"]
    boundary_union_np = kr.boundary_union.detach().cpu().numpy()

    per_variant_confusion: dict[str, list] = {}
    per_variant_scalars: dict[str, dict[str, int]] = {}
    for variant, label_map in labels_by_variant.items():
        cm = compute_confusion_matrix(label_map, ground_truth, class_count=class_count, ignore_index=ignore_index)
        per_variant_confusion[variant] = cm.tolist()
        evaluated = int((ground_truth != ignore_index).sum())
        ignored = int((ground_truth == ignore_index).sum())
        changed_vs_t320 = int((label_map != y320).sum())
        if variant in balp.CANDIDATE_IDS:
            pmask = kr.preservation_masks[variant].detach().cpu().numpy()
            preservation_count = int(pmask.sum())
            boundary_preserved = int((pmask & boundary_union_np).sum())
            non_boundary_preserved = int((pmask & ~boundary_union_np).sum())
        else:
            preservation_count = 0
            boundary_preserved = 0
            non_boundary_preserved = 0
        per_variant_scalars[variant] = {
            "candidate_versus_t320_changed_pixel_count": changed_vs_t320,
            "preservation_mask_pixel_count": preservation_count,
            "boundary_preservation_pixel_count": boundary_preserved,
            "non_boundary_preservation_pixel_count": non_boundary_preserved,
            "evaluated_pixel_count": evaluated,
            "ignored_pixel_count": ignored,
        }
    return PerImageMetricRecord(per_variant_confusion=per_variant_confusion, per_variant_scalars=per_variant_scalars)


# =======================================================================
# 350-image development-run orchestration (fresh/resume/finalize).
# =======================================================================


@dataclass(frozen=True)
class DevelopmentImageProvider:
    """One development image's already-prepared inputs, decoupled from
    any real dataset access -- production callers resolve `prepared`
    and `ground_truth` via the existing verified manifest-order
    adapter and dataset pipeline; tests inject synthetic stand-ins."""

    prepared: Any
    ground_truth: np.ndarray
    selected_windows: Sequence[int]


def execute_development(
    output_root: Path, *, providers: ExecutionProviders, development_ids: Sequence[str], confirmation_ids: Sequence[str],
    identity_bindings_without_digest: Mapping[str, str], ignore_index: int,
    image_provider_fn: Callable[[str], DevelopmentImageProvider],
    recorder: "ProfilingRecorder | None" = None,
    image_processor_fn: Callable[..., ImageBalpOutcome] = process_development_image,
    metrics_fn: Callable[..., "PerImageMetricRecord"] = None,
) -> Mapping[str, Any]:
    """The single execution entry point for fresh and resume alike.
    Processes exactly the REMAINING development images (in frozen
    order), checkpointing after every one; never reprocesses a
    completed image; never touches a confirmation ID or any of the
    other 4,500 images (only `development_ids` is ever iterated)."""
    development_ids = tuple(development_ids)
    confirmation_ids = tuple(confirmation_ids)
    if len(development_ids) != e22.DEVELOPMENT_COUNT:
        raise DfpPhase1bLightweightBoundaryExecuteError(f"development_ids must have exactly {e22.DEVELOPMENT_COUNT} entries, observed {len(development_ids)}")

    # B5 population enforcement: the caller-supplied population must be
    # BYTE-IDENTICAL, in order, to E22's own frozen 350/150 split -- never
    # merely self-consistent. This rejects a reordered, shortened,
    # resampled, or substituted-dataset population even though such a
    # population could otherwise pass every downstream relational check
    # (which only verifies internal consistency against whatever the
    # caller passed in).
    canonical_split = e22.load_and_verify_split(repo_root=e22.repository_root())
    if development_ids != tuple(canonical_split["development_image_ids"]):
        raise DfpPhase1bLightweightBoundaryExecuteError(
            "development_ids does not exactly match E22's frozen, ordered 350-image development split"
        )
    if confirmation_ids != tuple(canonical_split["confirmation_image_ids"]):
        raise DfpPhase1bLightweightBoundaryExecuteError(
            "confirmation_ids does not exactly match E22's frozen, ordered 150-image confirmation split"
        )

    existing = art.load_full_checkpoint(output_root, variant_ids=VARIANT_IDS)
    # Orphan-successor reconciliation: MUST run before any dataset/model/graph/recurrence/
    # image-processor/GT-provider access. Repairs the one remaining crash window where a new
    # generation was fully installed (atomic rename succeeded) but the current pointer never
    # advanced to it -- without this, every retry re-derives the same already-occupied
    # generation number and fails forever. See art.reconcile_orphan_successor.
    adopted = art.reconcile_orphan_successor(
        output_root, current=existing, variant_ids=VARIANT_IDS,
        identity_bindings_without_digest=identity_bindings_without_digest,
        development_ids=development_ids, confirmation_ids=confirmation_ids,
    )
    if adopted is not None:
        existing = adopted

    if existing is not None and existing["final"]:
        raise DfpPhase1bLightweightBoundaryExecuteError("execute rejected: this checkpoint is already finalized")

    if existing is not None and existing["complete"] and not existing["final"]:
        # complete-but-unfinalized recovery: no image processing, install the result only.
        return _finalize_only(output_root, existing, providers=providers, development_ids=development_ids, confirmation_ids=confirmation_ids)

    completed_ids: list[str] = list(existing["progress"]["completed_image_ids"]) if existing is not None else []
    # Every completed image installs its OWN new, immutable checkpoint
    # generation (see `art.install_checkpoint_generation`) -- generation
    # numbers are monotonic per install, not per resume-session.
    next_generation = int(existing["progress"]["generation"]) + 1 if existing is not None else 0
    if existing is not None:
        arrays = {
            "aggregate_confusion": existing["aggregate_confusion"],
            "per_image_reconstruction_statistics": existing["per_image_reconstruction_statistics"],
        }
    else:
        arrays = art.empty_checkpoint_arrays(variant_ids=VARIANT_IDS, class_count=providers.class_count)

    remaining_ids = list(development_ids[len(completed_ids):])
    reopened = existing
    for canonical_id in remaining_ids:
        if canonical_id in confirmation_ids:
            raise DfpPhase1bLightweightBoundaryExecuteError(f"{canonical_id} is a confirmation ID -- never processed by the development runner")
        image_input = image_provider_fn(canonical_id)
        t_start = time.perf_counter_ns()
        outcome = image_processor_fn(
            providers.inference, image_input.prepared, canonical_id, protocol_identity=providers.protocol_identity,
            contracts=providers.contracts, class_count=providers.class_count, affinity_power=providers.affinity_power,
            selected_windows_for_image=image_input.selected_windows,
        )
        # GT-isolation boundary: every prediction above this line is already finalized.
        active_metrics_fn = metrics_fn if metrics_fn is not None else accumulate_image_metrics
        record = active_metrics_fn(outcome, ground_truth=image_input.ground_truth, ignore_index=ignore_index, class_count=providers.class_count)
        t_end = time.perf_counter_ns()
        if recorder is not None:
            recorder.record_image(canonical_id, t_end - t_start)

        arrays = art.append_image_to_checkpoint_arrays(
            arrays, variant_ids=VARIANT_IDS, per_variant_confusion=record.per_variant_confusion, per_variant_scalars=record.per_variant_scalars,
        )
        completed_ids.append(canonical_id)
        complete_now = len(completed_ids) == e22.DEVELOPMENT_COUNT
        reopened = art.install_checkpoint_generation(
            output_root, identity_bindings_without_digest=identity_bindings_without_digest, variant_ids=VARIANT_IDS,
            arrays=arrays, completed_image_ids=completed_ids, generation=next_generation, complete=complete_now, final=False,
            final_result_binding=None, development_ids=development_ids, confirmation_ids=confirmation_ids,
        )
        next_generation += 1

    if len(completed_ids) != e22.DEVELOPMENT_COUNT:
        raise DfpPhase1bLightweightBoundaryExecuteError(f"internal error: completed {len(completed_ids)} images, expected {e22.DEVELOPMENT_COUNT}")

    return _finalize_only(output_root, reopened, providers=providers, development_ids=development_ids, confirmation_ids=confirmation_ids)


def _finalize_only(
    output_root: Path, checkpoint: Mapping[str, Any], *, providers: ExecutionProviders,
    development_ids: Sequence[str], confirmation_ids: Sequence[str],
) -> Mapping[str, Any]:
    """Complete-but-unfinalized recovery: build and install the
    development result from the ALREADY-COMPLETE checkpoint's arrays,
    then advance `final=true` on the checkpoint only after the result
    has been installed and post-install-reverified."""
    from src.dfp_phase1a_analysis import ConfusionStatistics, aggregate_confusion_statistics, compute_semantic_metrics

    agg = checkpoint["aggregate_confusion"]
    per_image = checkpoint["per_image_reconstruction_statistics"]

    def stats_from_matrix(cm: list) -> ConfusionStatistics:
        cm_arr = np.asarray(cm, dtype=np.int64)
        intersection = np.diag(cm_arr)
        pred = cm_arr.sum(axis=0)
        gt = cm_arr.sum(axis=1)
        union = pred + gt - intersection
        return ConfusionStatistics(intersection=intersection, union=union, predicted_pixels=pred, ground_truth_pixels=gt, confusion_matrix=cm_arr)

    control_results: dict[str, Any] = {}
    candidate_results: dict[str, Any] = {}
    for variant in VARIANT_IDS:
        stats = stats_from_matrix(agg[variant])
        metrics = compute_semantic_metrics(stats)
        evaluated = int(sum(per_image[variant]["evaluated_pixel_count"]))
        ignored = int(sum(per_image[variant]["ignored_pixel_count"]))
        if variant in e22.CANONICAL_CONTROLS:
            control_results[variant] = {
                "control_id": variant, "dataset_level_miou": float(metrics.mIoU), "per_class_iou": [float(x) for x in metrics.per_class_iou],
                "mean_class_accuracy": float(metrics.mAcc), "pixel_accuracy": float(metrics.aAcc),
                "evaluated_pixel_count": evaluated, "ignored_pixel_count": ignored,
            }
        else:
            s, r = e22.parse_candidate_id(variant)
            changed = int(sum(per_image[variant]["candidate_versus_t320_changed_pixel_count"]))
            preserved = int(sum(per_image[variant]["preservation_mask_pixel_count"]))
            boundary_p = int(sum(per_image[variant]["boundary_preservation_pixel_count"]))
            non_boundary_p = int(sum(per_image[variant]["non_boundary_preservation_pixel_count"]))
            t320_metrics = compute_semantic_metrics(stats_from_matrix(agg["T320"]))
            classwise = [float(a - b) for a, b in zip(metrics.per_class_iou, t320_metrics.per_class_iou)]
            candidate_results[variant] = {
                "candidate_id": variant, "stability_start": s, "radius": r,
                "dataset_level_miou": float(metrics.mIoU), "per_class_iou": [float(x) for x in metrics.per_class_iou],
                "mean_class_accuracy": float(metrics.mAcc), "pixel_accuracy": float(metrics.aAcc),
                "evaluated_pixel_count": evaluated, "ignored_pixel_count": ignored,
                "candidate_versus_t320_changed_pixel_count": changed, "preservation_mask_pixel_count": preserved,
                "preservation_mask_fraction": (preserved / evaluated) if evaluated else 0.0,
                "boundary_preservation_pixel_count": boundary_p, "non_boundary_preservation_pixel_count": non_boundary_p,
                "classwise_gain_loss_versus_t320": classwise,
            }

    import hashlib, json as _json
    digest_source = _json.dumps({"agg": agg}, sort_keys=True).encode("utf-8")
    scientific_output_digest = hashlib.sha256(digest_source).hexdigest()

    result_doc = art.build_development_result_document(
        identity_bindings=checkpoint["identity_bindings"], control_results=control_results, candidate_results=candidate_results,
        scientific_output_digest=scientific_output_digest, complete=True, final=True,
    )
    art.install_development_result(output_root, doc=result_doc, variant_ids=VARIANT_IDS)

    # The final=True checkpoint is a materially different document from
    # the complete-but-unfinalized generation that preceded it (it now
    # carries a final_result_binding), so it installs as its OWN new,
    # immutable generation -- never reusing the prior generation number,
    # which is already occupied by that complete-but-unfinalized document.
    finalize_generation = int(checkpoint["progress"]["generation"]) + 1
    final_checkpoint_metadata = art.build_checkpoint_metadata(
        identity_bindings=checkpoint["identity_bindings"], next_development_index=e22.DEVELOPMENT_COUNT,
        completed_image_ids=list(development_ids), generation=finalize_generation,
        complete=True, final=True, final_result_binding={
            "development_result_relative_path": "development_result/result.json",
            "development_result_sha256": _sha256_installed_result(output_root),
            "installed_at_generation": finalize_generation,
        },
    )
    reopened = art.install_checkpoint_generation(
        output_root, identity_bindings_without_digest={k: v for k, v in checkpoint["identity_bindings"].items() if k != "development_digest"},
        variant_ids=VARIANT_IDS, arrays={"aggregate_confusion": agg, "per_image_reconstruction_statistics": per_image},
        completed_image_ids=list(development_ids), generation=finalize_generation, complete=True, final=True,
        final_result_binding=final_checkpoint_metadata["final_result_binding"], development_ids=development_ids, confirmation_ids=confirmation_ids,
    )
    return result_doc


def _sha256_installed_result(output_root: Path) -> str:
    import hashlib
    path = Path(output_root) / "development_result" / "result.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Field names match E22's own `profiling_schema.per_variant_record` GPU
# memory field names exactly (`gpu_peak_allocated_bytes`,
# `gpu_peak_reserved_bytes`) even though a raw per-call
# `ProfilingRecorder` snapshot is not itself a `profiling_schema`
# document (B12: this task implements no paired T320/BALP benchmark and
# applies no operational gate -- see `gpu_snapshot` below). E22's
# profiling schema records no device-identifier/index field to
# cross-check against, so no such check is implemented here.
GPU_TELEMETRY_SNAPSHOT_KEYS = frozenset({"gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"})


def _validate_gpu_telemetry_snapshot(record: Any) -> Mapping[str, int]:
    """Strictly validates an injected GPU telemetry adapter's
    `snapshot()` return value before it is ever stored. Exact int only
    (bool and float -- including NaN/Infinity, which cannot pass an
    exact-int check -- are rejected), non-negative, exactly the closed
    two-key set (missing or extra keys rejected), and
    `gpu_peak_allocated_bytes` may never exceed `gpu_peak_reserved_bytes`.
    Raises `DfpPhase1bLightweightBoundaryExecuteError` -- never silently
    accepts malformed telemetry. Any exception raised by the adapter's
    own `snapshot()` call itself (fatal or otherwise) is never caught
    here and propagates unmodified."""
    if not isinstance(record, dict):
        raise DfpPhase1bLightweightBoundaryExecuteError(
            f"GPU telemetry snapshot must be a mapping, observed {type(record).__name__}"
        )
    actual = frozenset(record.keys())
    if actual != GPU_TELEMETRY_SNAPSHOT_KEYS:
        raise DfpPhase1bLightweightBoundaryExecuteError(
            f"GPU telemetry snapshot has the wrong key set -- "
            f"missing={sorted(GPU_TELEMETRY_SNAPSHOT_KEYS - actual)} unknown={sorted(actual - GPU_TELEMETRY_SNAPSHOT_KEYS)}"
        )
    for key in GPU_TELEMETRY_SNAPSHOT_KEYS:
        value = record[key]
        if isinstance(value, bool) or type(value) is not int:
            raise DfpPhase1bLightweightBoundaryExecuteError(
                f"GPU telemetry field {key!r} must be an exact int, observed {type(value).__name__}={value!r}"
            )
        if value < 0:
            raise DfpPhase1bLightweightBoundaryExecuteError(f"GPU telemetry field {key!r} must be non-negative, observed {value}")
    if record["gpu_peak_allocated_bytes"] > record["gpu_peak_reserved_bytes"]:
        raise DfpPhase1bLightweightBoundaryExecuteError(
            f"gpu_peak_allocated_bytes ({record['gpu_peak_allocated_bytes']}) must not exceed "
            f"gpu_peak_reserved_bytes ({record['gpu_peak_reserved_bytes']})"
        )
    return {key: record[key] for key in GPU_TELEMETRY_SNAPSHOT_KEYS}


@dataclass
class ProfilingRecorder:
    """Raw measurements only (B12): total wall time, per-image
    latency, host RSS, optional injectable GPU telemetry. Never
    affects scientific output/digests."""

    per_image_latency_ns: dict[str, int]
    gpu_telemetry: Any | None = None

    def __init__(self, *, gpu_telemetry: Any | None = None) -> None:
        self.per_image_latency_ns = {}
        self.gpu_telemetry = gpu_telemetry
        self._start_ns = time.perf_counter_ns()

    def record_image(self, canonical_id: str, elapsed_ns: int) -> None:
        self.per_image_latency_ns[canonical_id] = elapsed_ns

    def host_rss_bytes(self) -> int | None:
        try:
            text = Path("/proc/self/status").read_text()
        except OSError:
            return None
        for line in text.splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                return int(parts[1]) * 1024
        return None

    def total_wall_time_ns(self) -> int:
        return time.perf_counter_ns() - self._start_ns

    def gpu_snapshot(self) -> Mapping[str, int] | None:
        if self.gpu_telemetry is None:
            return None
        record = self.gpu_telemetry.snapshot()
        return _validate_gpu_telemetry_snapshot(record)
