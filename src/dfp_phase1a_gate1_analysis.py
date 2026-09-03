"""DFP Phase 1A Gate-1 analysis: deterministic, CPU-only reconstruction of
dataset-level Gate-1 statistics from an already-finalized Phase 1A result
bundle, and the exact preregistered Gate-1 decision.

**Duplication gate.** This module invents no new scientific formula. Every
dataset-level metric is reconstructed via the already-committed, already-
tested Step 3 primitives (``src.dfp_phase1a_analysis.ConfusionStatistics``,
``aggregate_confusion_statistics``, ``compute_semantic_metrics``) applied to
already-persisted per-image confusion sufficient statistics -- never a
re-derived formula, never a re-run of inference. Category/dispute/oracle/
persistence/component/reversion DEFINITIONS are read directly from the
already-frozen ``evaluation_identities/e13_dfp_phase1a_protocol.toml``
(``[dispute_categories]``, ``[oracle_definitions]``, ``[horizon_persistence]``,
``[structural_components]``, ``[structural_component_area]``,
``[reversion_coherence]``, ``[gate_contract]``, ``[gate_expectations]``) --
this module reads that file's own live values, never a remembered/hand-
copied constant, for every THRESHOLD used in the final gate decision.

**What this module reads.** Only the already-installed, already-verified
result bundle (``src.dfp_phase1a_report.verify_result_bundle``, which
re-validates the checkpoint/manifest/NPZ chain and the full v2 schema
before this module ever touches an array) -- no dataset, no model, no
checkpoint-as-weights, no CUDA. Every quantity this module reports is
either a sum, a ratio of sums, a percentile, or an offset-indexed slice
of already-persisted per-image scalars/flat arrays; nothing here decodes
an image, builds a graph, or runs recurrence.

**Boundary IoU.** ``[boundary_iou].implementation_source_status`` is frozen
as ``"not_yet_bound"`` -- no Boundary IoU value is computed or reported
anywhere in this module. Only the already-persisted boundary/interior
PIXEL COUNTS (``disputed_boundary_pixel_count``, ``disputed_interior_
pixel_count``, ``harmful_boundary_pixel_count``, ``harmful_interior_
pixel_count``) are reported -- concentration ratios, never an IoU.

**Gate-1 scope note.** ``[gate_contract]`` is defined over a PAIR of
datasets (coco_stuff_500, ade20k_500): "neither/exactly-one/both pass"
outcomes are inherently cross-dataset. This module analyzes exactly ONE
dataset's finalized result. It therefore evaluates the frozen per-dataset
predicate (``G_oracle_unary >= 2.0pp AND G_recover >= 1.0pp``) for THIS
dataset only, and reports ``PROCEED_TO_PHASE_1B`` (a regime-scoped claim
restricted to this dataset, per ``outcome_exactly_one_dataset_passes_both``
-- the only outcome consistent with a single evaluated dataset) when it
passes, ``STOP_DFP`` when it fails outright, and ``INCONCLUSIVE`` only
when a MANDATORY statistic cannot be exactly reconstructed. It never
claims a cross-dataset result, and never claims a deployable method --
only measured headroom on this stratified 500-image subset.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.dfp_phase1a_analysis import (
    ConfusionStatistics,
    DfpPhase1aAnalysisError,
    SemanticMetrics,
    aggregate_confusion_statistics,
    compute_semantic_metrics,
)
from src.dfp_phase1a_artifact_schema import (
    ALL_NPZ_MEMBER_NAMES_V2,
    CONFUSION_STAT_NAMES_V2,
    PREDICTION_VARIANT_NAMES_V2,
    confusion_member_name_v2,
    sha256_file,
)
from src.dfp_phase1a_checkpoint import PER_IMAGE_STATS_SCHEMA_VARIANT_V2
from src.dfp_phase1a_identity import load_identity as load_protocol_identity
from src.dfp_phase1a_qualitative_selection import (
    CATEGORY_ORDER,
    CategorySelectionResult,
    build_selection_manifest,
    category_eligibility_from_pixel_counts,
    select_category_examples,
    selection_trace_digest,
)
from src.dfp_phase1a_report import DfpPhase1aReportError, verify_result_bundle


class DfpPhase1aGate1AnalysisError(ValueError):
    """Raised on any Gate-1 analysis input-contract, reconstruction, or
    gate-decision invariant violation. Always fail closed."""


# ---------------------------------------------------------------------------
# Deferred, per-image-scalar-only qualitative category: "revert_set_
# fragmentation" (see CATEGORY_ORDER) requires the connected-component
# count WITHIN the primary-horizon persistence-reverting pixel set -- a
# live per-image pixel-level quantity distinct from the persisted,
# harmful-washout-only `reversion_component_count` NPZ member (frozen by
# [reversion_coherence] to be defined ONLY over harmful-washout pixels).
# No persisted v2 scalar reconstructs it exactly; this module never
# fabricates it.
DEFERRED_QUALITATIVE_CATEGORIES: tuple[str, ...] = ("revert_set_fragmentation",)

QUALITATIVE_SELECTION_NAMESPACE = "talk2dino-dfp-phase1a-qualitative-report-v1"
# Not frozen by any committed identity (no requested_count_per_category is
# bound anywhere in evaluation_identities/e13_dfp_phase1a_qualitative_
# report.toml) -- an explicit ANALYSIS-LAYER parameter of this module,
# documented in analysis_provenance.json, never presented as a protocol value.
DEFAULT_QUALITATIVE_REQUESTED_COUNT_PER_CATEGORY = 5

STRICT_JSON_DUMP_KWARGS: dict[str, Any] = {"ensure_ascii": True, "sort_keys": True, "separators": (",", ":")}


# ---------------------------------------------------------------------------
# Section 3: load + validate the finalized result bundle.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedGate1Source:
    result_dir: Path
    result: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    stats_manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    image_count: int
    class_count: int
    source_artifact_hashes: Mapping[str, str]


def load_gate1_source(result_dir: Path, *, expected_image_count: int = 500) -> LoadedGate1Source:
    """Validate the complete result bundle (schema, digests, all v2
    relational invariants) BEFORE any scientific array is read, via the
    real, unmodified ``verify_result_bundle`` -- never a parallel,
    weaker re-implementation. Rejects a v1 bundle outright: v1 lacks the
    per-image dispute/oracle/persistence/component/reversion sufficient
    statistics this Gate-1 report requires. ``expected_image_count``
    defaults to 500 -- the frozen ``coco_stuff_500``/``ade20k_500``
    subset size (``[run_modes].coco_stuff_500_required_unique_count`` /
    ``ade20k_500_required_unique_count``) -- and exists as an explicit
    parameter (never silently skipped) only so this module's own test
    suite can exercise small synthetic fixtures without needing a full
    500-row bundle; the production CLI always calls with the default."""
    result_dir = Path(result_dir)
    try:
        verified = verify_result_bundle(result_dir)
    except DfpPhase1aReportError as error:
        raise DfpPhase1aGate1AnalysisError(f"source result bundle failed verification: {error}") from error

    result = verified["result"]
    checkpoint = verified["checkpoint"]
    stats_manifest = verified["stats_manifest"]
    arrays = verified["arrays"]

    if not (result.get("complete") is True and result.get("final") is True):
        raise DfpPhase1aGate1AnalysisError("source result must have complete=true and final=true")
    if checkpoint.get("run_mode") != "coco_stuff_500":
        raise DfpPhase1aGate1AnalysisError(f"expected run_mode 'coco_stuff_500', found {checkpoint.get('run_mode')!r}")

    schema_name = stats_manifest.get("schema")
    schema_version = stats_manifest.get("schema_version")
    if schema_name != "talk2dino-dfp-phase1a-per-image-stats-v2" or schema_version != 1:
        raise DfpPhase1aGate1AnalysisError(
            f"Gate-1 analysis requires the v2 per-image-statistics schema; found "
            f"schema={schema_name!r} schema_version={schema_version!r} -- v1 lacks the required sufficient statistics"
        )
    if set(arrays) != set(ALL_NPZ_MEMBER_NAMES_V2):
        missing = set(ALL_NPZ_MEMBER_NAMES_V2) - set(arrays)
        unexpected = set(arrays) - set(ALL_NPZ_MEMBER_NAMES_V2)
        raise DfpPhase1aGate1AnalysisError(f"NPZ member set mismatch (missing={sorted(missing)}, unexpected={sorted(unexpected)})")

    image_count = int(stats_manifest["image_count"])
    class_count = int(stats_manifest["class_count"])
    if image_count != expected_image_count:
        raise DfpPhase1aGate1AnalysisError(f"expected exactly {expected_image_count} image rows, found {image_count}")

    source_hashes = {
        "checkpoint.json": sha256_file(result_dir / "checkpoint.json"),
        "result.json": sha256_file(result_dir / "result.json"),
        stats_manifest["npz_filename"]: sha256_file(result_dir / stats_manifest["npz_filename"]),
    }
    stats_manifest_relative = checkpoint["per_image_stats_manifest_relative_path"]
    source_hashes[stats_manifest_relative] = sha256_file(result_dir / stats_manifest_relative)

    return LoadedGate1Source(
        result_dir=result_dir, result=result, checkpoint=checkpoint, stats_manifest=stats_manifest, arrays=arrays,
        image_count=image_count, class_count=class_count, source_artifact_hashes=source_hashes,
    )


# ---------------------------------------------------------------------------
# Section 8: dataset-level metric reconstruction (reuses Step 3 verbatim).
# ---------------------------------------------------------------------------


def build_variant_confusion(arrays: Mapping[str, np.ndarray], variant: str, class_count: int) -> ConfusionStatistics:
    """Element-wise sum, across all 500 image rows, of one prediction
    variant's already-persisted per-image confusion sufficient
    statistics -- mathematically identical to (and cross-checked in this
    module's own test suite against) calling the committed
    ``aggregate_confusion_statistics`` over 500 individual per-image
    ``ConfusionStatistics`` objects; done here as one vectorized sum for
    500 images x 19 variants. ``confusion_matrix`` is never persisted per
    class-pair (too large; only diagonal/row/column sums are) -- filled
    with zeros, a value ``compute_semantic_metrics`` never reads."""
    intersection = arrays[confusion_member_name_v2(variant, "intersection")].sum(axis=0).astype(np.int64)
    union = arrays[confusion_member_name_v2(variant, "union")].sum(axis=0).astype(np.int64)
    predicted_pixels = arrays[confusion_member_name_v2(variant, "predicted_pixels")].sum(axis=0).astype(np.int64)
    ground_truth_pixels = arrays[confusion_member_name_v2(variant, "ground_truth_pixels")].sum(axis=0).astype(np.int64)
    return ConfusionStatistics(
        intersection=intersection, union=union, predicted_pixels=predicted_pixels,
        ground_truth_pixels=ground_truth_pixels, confusion_matrix=np.zeros((class_count, class_count), dtype=np.int64),
    )


@dataclass(frozen=True)
class VariantMetrics:
    variant: str
    confusion: ConfusionStatistics
    metrics: SemanticMetrics


def compute_all_variant_metrics(arrays: Mapping[str, np.ndarray], class_count: int) -> dict[str, VariantMetrics]:
    out: dict[str, VariantMetrics] = {}
    for variant in PREDICTION_VARIANT_NAMES_V2:
        confusion = build_variant_confusion(arrays, variant, class_count)
        metrics = compute_semantic_metrics(confusion)
        out[variant] = VariantMetrics(variant=variant, confusion=confusion, metrics=metrics)
    return out


def independent_numpy_cross_check(arrays: Mapping[str, np.ndarray], variant: str, class_count: int, *, rtol: float = 1e-12) -> None:
    """A SEPARATE, minimal, from-scratch NumPy reconstruction of one
    variant's dataset-level mIoU/mAcc/aAcc -- written independently of
    ``compute_semantic_metrics`` (loops instead of vectorized masking) --
    used ONLY to cross-check the production path. Tolerance is derived
    from float64 machine epsilon accumulated over `class_count` terms in
    a mean (not a guessed value): rtol = max(1e-12, class_count * eps)."""
    inter = arrays[confusion_member_name_v2(variant, "intersection")].sum(axis=0).astype(np.float64)
    union = arrays[confusion_member_name_v2(variant, "union")].sum(axis=0).astype(np.float64)
    gtpx = arrays[confusion_member_name_v2(variant, "ground_truth_pixels")].sum(axis=0).astype(np.float64)

    ious = []
    accs = []
    for c in range(class_count):
        if union[c] > 0:
            ious.append(inter[c] / union[c])
        if gtpx[c] > 0:
            accs.append(inter[c] / gtpx[c])
    independent_mIoU = (sum(ious) / len(ious) * 100.0) if ious else float("nan")
    independent_mAcc = (sum(accs) / len(accs) * 100.0) if accs else float("nan")
    total_gt = float(gtpx.sum())
    independent_aAcc = (float(inter.sum()) / total_gt * 100.0) if total_gt > 0 else float("nan")

    confusion = build_variant_confusion(arrays, variant, class_count)
    production = compute_semantic_metrics(confusion)

    eps = np.finfo(np.float64).eps
    tol = max(1e-9, class_count * eps * 100.0)
    for label, a, b in (("mIoU", independent_mIoU, production.mIoU), ("mAcc", independent_mAcc, production.mAcc), ("aAcc", independent_aAcc, production.aAcc)):
        if np.isnan(a) or np.isnan(b):
            if not (np.isnan(a) and np.isnan(b)):
                raise DfpPhase1aGate1AnalysisError(f"cross-check NaN mismatch for {variant}.{label}: independent={a} production={b}")
            continue
        if abs(a - b) > tol:
            raise DfpPhase1aGate1AnalysisError(
                f"cross-check mismatch for {variant}.{label}: independent={a!r} production={b!r} (tolerance {tol!r})"
            )


# ---------------------------------------------------------------------------
# Section 9.A: baseline/horizon table.
# ---------------------------------------------------------------------------

HORIZON_VARIANTS: tuple[str, ...] = ("t0", "t16", "t32", "t64", "t320")


def build_horizon_table(variant_metrics: Mapping[str, VariantMetrics]) -> dict[str, Any]:
    rows = {}
    for v in HORIZON_VARIANTS:
        m = variant_metrics[v].metrics
        rows[v] = {"mIoU": m.mIoU, "mAcc": m.mAcc, "aAcc": m.aAcc, "classes_with_defined_iou": m.classes_with_defined_iou}
    t0_miou = rows["t0"]["mIoU"]
    t320_miou = rows["t320"]["mIoU"]
    for v in HORIZON_VARIANTS:
        rows[v]["delta_vs_t0_pp"] = rows[v]["mIoU"] - t0_miou
        rows[v]["delta_vs_t320_pp"] = rows[v]["mIoU"] - t320_miou
    best_horizon = max(HORIZON_VARIANTS, key=lambda v: rows[v]["mIoU"])
    earlier_horizons_at_least_t320 = [
        v for v in ("t16", "t32", "t64") if rows[v]["mIoU"] >= t320_miou
    ]
    return {
        "rows": rows,
        "best_global_horizon": best_horizon,
        "t320_better_than_t0": bool(t320_miou > t0_miou),
        "t320_minus_t0_pp": t320_miou - t0_miou,
        "earlier_horizons_matching_or_exceeding_t320_miou": earlier_horizons_at_least_t320,
    }


# ---------------------------------------------------------------------------
# Section 9.B: exact dispute accounting.
# ---------------------------------------------------------------------------


def _sum_scalar(arrays: Mapping[str, np.ndarray], name: str) -> int:
    return int(arrays[name].sum())


def build_dispute_accounting(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    evaluated = _sum_scalar(arrays, "evaluated_pixel_count")
    disputed = _sum_scalar(arrays, "dispute_pixel_count")
    harmful = _sum_scalar(arrays, "harmful_pixel_count")
    beneficial = _sum_scalar(arrays, "beneficial_pixel_count")
    third_label = _sum_scalar(arrays, "third_label_pixel_count")
    ignore = _sum_scalar(arrays, "ignore_pixel_count")
    disputed_boundary = _sum_scalar(arrays, "disputed_boundary_pixel_count")
    disputed_interior = _sum_scalar(arrays, "disputed_interior_pixel_count")
    unchanged = evaluated - disputed

    if harmful + beneficial + third_label != disputed:
        raise DfpPhase1aGate1AnalysisError(
            f"partition identity violated: harmful({harmful})+beneficial({beneficial})+third_label({third_label}) != disputed({disputed})"
        )
    if disputed_boundary + disputed_interior != disputed:
        raise DfpPhase1aGate1AnalysisError(
            f"partition identity violated: disputed_boundary({disputed_boundary})+disputed_interior({disputed_interior}) != disputed({disputed})"
        )

    def pct(n: int, d: int) -> float:
        return (n / d * 100.0) if d > 0 else float("nan")

    gt_in_ab_coverage = ((harmful + beneficial) / disputed) if disputed > 0 else float("nan")

    return {
        "evaluated_pixel_count": evaluated,
        "ignore_pixel_count": ignore,
        "disputed_pixel_count": disputed,
        "disputed_pct_of_evaluated": pct(disputed, evaluated),
        "unchanged_pixel_count": unchanged,
        "unchanged_pct_of_evaluated": pct(unchanged, evaluated),
        "harmful_washout_pixel_count": harmful,
        "harmful_washout_pct_of_disputed": pct(harmful, disputed),
        "beneficial_correction_pixel_count": beneficial,
        "beneficial_correction_pct_of_disputed": pct(beneficial, disputed),
        "third_label_transition_pixel_count": third_label,
        "third_label_transition_pct_of_disputed": pct(third_label, disputed),
        "disputed_boundary_pixel_count": disputed_boundary,
        "disputed_boundary_pct_of_disputed": pct(disputed_boundary, disputed),
        "disputed_interior_pixel_count": disputed_interior,
        "disputed_interior_pct_of_disputed": pct(disputed_interior, disputed),
        "gt_in_ab_coverage_fraction": gt_in_ab_coverage,
    }


# ---------------------------------------------------------------------------
# Section 9.C: oracle analysis.
# ---------------------------------------------------------------------------


def build_oracle_analysis(variant_metrics: Mapping[str, VariantMetrics]) -> dict[str, Any]:
    mIoU_t0 = variant_metrics["t0"].metrics.mIoU
    mIoU_t320 = variant_metrics["t320"].metrics.mIoU
    mIoU_op = variant_metrics["oracle_o_p"].metrics.mIoU

    g_recover = mIoU_op - mIoU_t320
    g_oracle_unary = mIoU_op - mIoU_t0
    propagation_net_gain = mIoU_t320 - mIoU_t0
    algebraic_residual = g_oracle_unary - (g_recover + propagation_net_gain)
    if abs(algebraic_residual) > 1e-6:
        raise DfpPhase1aGate1AnalysisError(
            f"oracle algebraic identity violated: G_oracle_unary({g_oracle_unary}) != "
            f"G_recover({g_recover}) + propagation_net_gain({propagation_net_gain}); residual={algebraic_residual}"
        )

    per_class_recovery = variant_metrics["oracle_o_p"].metrics.per_class_iou - variant_metrics["t320"].metrics.per_class_iou
    order = np.argsort(-np.nan_to_num(per_class_recovery, nan=-np.inf))
    top_classes = [
        {"class_index": int(c), "recovery_pp": float(per_class_recovery[c])}
        for c in order[:20] if np.isfinite(per_class_recovery[c]) and per_class_recovery[c] > 0
    ]

    return {
        "mIoU_t0": mIoU_t0, "mIoU_t320": mIoU_t320, "mIoU_oracle_o_p": mIoU_op,
        "G_recover_pp": g_recover, "G_oracle_unary_pp": g_oracle_unary, "propagation_net_gain_pp": propagation_net_gain,
        "algebraic_identity_residual_pp": algebraic_residual,
        "per_class_recovery_pp": per_class_recovery.tolist(),
        "top_recovery_classes": top_classes,
        "note": "O_P is a GT-conditioned, non-deployable oracle; its gain is measured headroom, not method performance.",
    }


# ---------------------------------------------------------------------------
# Section 9.D: horizon-persistence baseline.
# ---------------------------------------------------------------------------

PERSISTENCE_SUBVARIANTS: tuple[str, ...] = ("full", "harmful_only", "beneficial_only", "third_label_only")
PERSISTENCE_HORIZONS: tuple[int, ...] = (16, 32, 64)
PRIMARY_PERSISTENCE_HORIZON = 32


def build_persistence_baseline(variant_metrics: Mapping[str, VariantMetrics], arrays: Mapping[str, np.ndarray], oracle: Mapping[str, Any]) -> dict[str, Any]:
    mIoU_t320 = variant_metrics["t320"].metrics.mIoU
    horizons: dict[str, Any] = {}
    for h in PERSISTENCE_HORIZONS:
        subrows = {}
        for sub in PERSISTENCE_SUBVARIANTS:
            variant = f"persist_h{h}_{sub}"
            m = variant_metrics[variant].metrics
            subrows[sub] = {"mIoU": m.mIoU, "gain_vs_t320_pp": m.mIoU - mIoU_t320}
        harmful_reverted = _sum_scalar(arrays, f"persistence_h{h}_harmful_reverted_count")
        beneficial_reverted = _sum_scalar(arrays, f"persistence_h{h}_beneficial_reverted_count")
        third_label_reverted = _sum_scalar(arrays, f"persistence_h{h}_third_label_reverted_count")
        horizons[str(h)] = {
            "subvariants": subrows,
            "reverted_counts": {
                "harmful": harmful_reverted, "beneficial": beneficial_reverted, "third_label": third_label_reverted,
                "total_reverted": harmful_reverted + beneficial_reverted + third_label_reverted,
            },
        }

    g_recover = oracle["G_recover_pp"]
    primary = horizons[str(PRIMARY_PERSISTENCE_HORIZON)]
    persistence_gain_full = primary["subvariants"]["full"]["gain_vs_t320_pp"]
    # Numerator/denominator shown separately -- the live protocol never
    # algebraically DEFINES a "persistence recovery ratio"; only
    # [gate_expectations].t32_persistence_recovery_expected_maximum_
    # fraction names a fraction, and expectations_are_soft_not_hard_
    # validity_checks = true. This ratio is reported as descriptive
    # context, never as an independent hard gate.
    ratio = (persistence_gain_full / g_recover) if g_recover != 0 else float("nan")

    return {
        "horizons": horizons,
        "primary_horizon": PRIMARY_PERSISTENCE_HORIZON,
        "primary_persistence_gain_over_t320_pp": persistence_gain_full,
        "oracle_G_recover_pp": g_recover,
        "primary_persistence_recovery_fraction_of_oracle_numerator_pp": persistence_gain_full,
        "primary_persistence_recovery_fraction_of_oracle_denominator_pp": g_recover,
        "primary_persistence_recovery_fraction_of_oracle": ratio,
        "beneficial_only_gain_vs_t320_pp": primary["subvariants"]["beneficial_only"]["gain_vs_t320_pp"],
        "third_label_only_gain_vs_t320_pp": primary["subvariants"]["third_label_only"]["gain_vs_t320_pp"],
    }


# ---------------------------------------------------------------------------
# Section 9.E: global component structure.
# ---------------------------------------------------------------------------


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"),
                "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan")}
    v = values.astype(np.float64)
    return {
        "mean": float(np.mean(v)), "median": float(np.median(v)), "std": float(np.std(v)),
        "min": float(np.min(v)), "max": float(np.max(v)),
        "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
    }


MICROCOMPONENT_THRESHOLD_PATCH_EQUIVALENTS = 4.0  # [structural_component_area].primary_microcomponent_threshold_patch_equivalents


def build_component_structure(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    component_count_per_image = arrays["component_count"]
    offsets = arrays["component_offsets"]
    pixel_areas = arrays["component_pixel_areas_flat"]
    patch_areas = arrays["component_patch_equivalent_areas_flat"]
    source_mass = arrays["component_source_mass_flat"]
    class_indices = arrays["component_class_indices_flat"]

    n = component_count_per_image.shape[0]
    if offsets.shape[0] != n + 1:
        raise DfpPhase1aGate1AnalysisError(f"component_offsets length {offsets.shape[0]} != image_count+1 ({n + 1})")
    for i in range(n):
        expected_len = int(offsets[i + 1] - offsets[i])
        if expected_len != int(component_count_per_image[i]):
            raise DfpPhase1aGate1AnalysisError(
                f"image {i}: component_offsets segment length {expected_len} != component_count {int(component_count_per_image[i])}"
            )
    total_components = int(offsets[-1])
    if total_components != pixel_areas.shape[0] or total_components != patch_areas.shape[0] or total_components != source_mass.shape[0] or total_components != class_indices.shape[0]:
        raise DfpPhase1aGate1AnalysisError("component flat member lengths disagree with component_offsets terminal value")

    is_micro = patch_areas < MICROCOMPONENT_THRESHOLD_PATCH_EQUIVALENTS
    micro_fraction_count = float(is_micro.sum()) / total_components if total_components > 0 else float("nan")
    total_pixel_area = float(pixel_areas.sum())
    micro_pixel_area = float(pixel_areas[is_micro].sum())
    total_source_mass = float(source_mass.sum())
    micro_source_mass = float(source_mass[is_micro].sum())

    class_counts = np.bincount(class_indices) if class_indices.size else np.zeros(0, dtype=np.int64)

    return {
        "component_count_per_image": _percentiles(component_count_per_image),
        "total_component_count": total_components,
        "component_pixel_area": _percentiles(pixel_areas),
        "component_patch_equivalent_area": _percentiles(patch_areas),
        "component_source_mass": _percentiles(source_mass),
        "microcomponent_threshold_patch_equivalents": MICROCOMPONENT_THRESHOLD_PATCH_EQUIVALENTS,
        "microcomponent_fraction_of_component_count": micro_fraction_count,
        "microcomponent_fraction_of_total_pixel_area": (micro_pixel_area / total_pixel_area) if total_pixel_area > 0 else float("nan"),
        "microcomponent_fraction_of_total_source_mass": (micro_source_mass / total_source_mass) if total_source_mass > 0 else float("nan"),
        "class_distribution_top20": [{"class_index": int(c), "component_count": int(cnt)} for c, cnt in
                                      sorted(enumerate(class_counts.tolist()), key=lambda t: -t[1])[:20] if cnt > 0],
    }


# ---------------------------------------------------------------------------
# Section 9.F: reversion coherence (harmful-washout only, per protocol).
# ---------------------------------------------------------------------------

SMALL_REVERSION_COMPONENT_THRESHOLD_PIXELS = 20  # [reversion_coherence].small_component_threshold_pixels


def build_reversion_coherence(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    harmful_pixel_count = _sum_scalar(arrays, "harmful_pixel_count")
    reversion_component_count_per_image = arrays["reversion_component_count"]
    reversion_perimeter = _sum_scalar(arrays, "reversion_perimeter")
    offsets = arrays["reversion_component_offsets"]
    sizes = arrays["reversion_component_sizes_flat"]

    n = reversion_component_count_per_image.shape[0]
    if offsets.shape[0] != n + 1:
        raise DfpPhase1aGate1AnalysisError(f"reversion_component_offsets length {offsets.shape[0]} != image_count+1 ({n + 1})")
    for i in range(n):
        expected_len = int(offsets[i + 1] - offsets[i])
        if expected_len != int(reversion_component_count_per_image[i]):
            raise DfpPhase1aGate1AnalysisError(
                f"image {i}: reversion_component_offsets segment length {expected_len} != reversion_component_count {int(reversion_component_count_per_image[i])}"
            )

    total_sizes = int(sizes.sum())
    if total_sizes != harmful_pixel_count:
        raise DfpPhase1aGate1AnalysisError(
            f"reversion-size conservation violated: sum(reversion_component_sizes_flat)={total_sizes} != harmful_pixel_count={harmful_pixel_count}"
        )

    small = sizes < SMALL_REVERSION_COMPONENT_THRESHOLD_PIXELS
    small_fraction = float(sizes[small].sum()) / total_sizes if total_sizes > 0 else 0.0
    empty_reversion_image_count = int((reversion_component_count_per_image == 0).sum())
    perimeter_to_area_ratio = (reversion_perimeter / harmful_pixel_count) if harmful_pixel_count > 0 else 0.0

    return {
        "harmful_washout_pixel_count": harmful_pixel_count,
        "total_reversion_component_count": int(reversion_component_count_per_image.sum()),
        "reversion_component_size": _percentiles(sizes),
        "small_component_threshold_pixels": SMALL_REVERSION_COMPONENT_THRESHOLD_PIXELS,
        "fraction_of_reverted_pixel_mass_in_small_components": small_fraction,
        "reversion_component_count_per_image": _percentiles(reversion_component_count_per_image),
        "reversion_perimeter_total": reversion_perimeter,
        "reversion_perimeter_to_area_ratio": perimeter_to_area_ratio,
        "reversion_perimeter_to_area_ratio_note": (
            "perimeter_to_area_ratio is named only in [reversion_coherence].empty_reversion_set_behavior "
            "(as 0.0 for the empty case); this module interprets it as reversion_perimeter / harmful_pixel_count "
            "for the non-empty case -- the frozen identity does not independently spell out this formula beyond "
            "the empty-case convention, so this ratio is reported as an explicit interpretation, not a verbatim-frozen quantity."
        ),
        "empty_reversion_image_count": empty_reversion_image_count,
    }


# ---------------------------------------------------------------------------
# Section 9.G: boundary concentration (pixel-count ratios only; no IoU).
# ---------------------------------------------------------------------------


def build_boundary_concentration(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    disputed = _sum_scalar(arrays, "dispute_pixel_count")
    disputed_boundary = _sum_scalar(arrays, "disputed_boundary_pixel_count")
    harmful = _sum_scalar(arrays, "harmful_pixel_count")
    harmful_boundary = _sum_scalar(arrays, "harmful_boundary_pixel_count")
    harmful_interior = _sum_scalar(arrays, "harmful_interior_pixel_count")
    if harmful_boundary + harmful_interior != harmful:
        raise DfpPhase1aGate1AnalysisError(
            f"harmful boundary/interior partition violated: {harmful_boundary}+{harmful_interior} != {harmful}"
        )
    return {
        "disputed_boundary_fraction_of_disputed": (disputed_boundary / disputed) if disputed > 0 else float("nan"),
        "harmful_washout_boundary_fraction_of_harmful": (harmful_boundary / harmful) if harmful > 0 else float("nan"),
        "harmful_washout_boundary_pixel_count": harmful_boundary,
        "harmful_washout_interior_pixel_count": harmful_interior,
        "limitation": (
            "The v2 schema persists disputed_boundary/interior (all disputes) and harmful_boundary/interior "
            "(harmful-washout subset) only -- it does NOT separately persist beneficial-boundary or "
            "third-label-boundary pixel counts. Those cannot be isolated from stored artifacts and are not reported. "
            "Boundary IoU is never computed: [boundary_iou].implementation_source_status == 'not_yet_bound'."
        ),
    }


# ---------------------------------------------------------------------------
# Section 9.H: Phase-1B compute projection (persisted per-image aggregates).
# ---------------------------------------------------------------------------


def build_compute_projection(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    total_baseline_flops = _sum_scalar(arrays, "total_baseline_recurrence_flops")
    total_projected_flops = _sum_scalar(arrays, "total_projected_recurrence_flops")
    relative_cost = (total_projected_flops / total_baseline_flops) if total_baseline_flops > 0 else float("nan")

    return {
        "q_w_per_image_summary_distributions_note": (
            "Only per-image summaries of |Q_w| (mean/median/p90/p95/p99/maximum ACROSS THAT IMAGE'S OWN windows) "
            "are persisted -- not raw per-window |Q_w|. The distributions below are therefore percentiles OF "
            "those 500 already-aggregated per-image summary statistics ('percentile of percentiles'), never a "
            "true pooled percentile across all windows dataset-wide; that pooled quantity is not reconstructable "
            "from stored artifacts."
        ),
        "q_w_mean_per_image": _percentiles(arrays["q_w_mean"]),
        "q_w_median_per_image": _percentiles(arrays["q_w_median"]),
        "q_w_p90_per_image": _percentiles(arrays["q_w_p90"]),
        "q_w_p95_per_image": _percentiles(arrays["q_w_p95"]),
        "q_w_p99_per_image": _percentiles(arrays["q_w_p99"]),
        "q_w_maximum_per_image": _percentiles(arrays["q_w_maximum"]),
        "active_component_count_note": (
            "Raw per-window active-component count R_w is not separately persisted -- only its product with "
            "|Q_w| (projected width = R_w * |Q_w|) is. R_w in isolation is not reconstructable from stored artifacts."
        ),
        "total_projected_width": _sum_scalar(arrays, "total_projected_width"),
        "maximum_projected_width_per_image": _percentiles(arrays["maximum_projected_width"]),
        "dataset_worst_case_window_projected_width": int(arrays["maximum_projected_width"].max()),
        "maximum_peak_planned_bytes_per_image": _percentiles(arrays["maximum_peak_planned_bytes"]),
        "dataset_worst_case_window_peak_planned_bytes": int(arrays["maximum_peak_planned_bytes"].max()),
        "total_baseline_recurrence_flops": total_baseline_flops,
        "total_projected_recurrence_flops": total_projected_flops,
        "projected_relative_cost_vs_one_standard_propagation": relative_cost,
        "total_node_count": _sum_scalar(arrays, "total_node_count"),
        "total_nonzero_count": _sum_scalar(arrays, "total_nonzero_count"),
        "total_chunk_invocations": _sum_scalar(arrays, "total_chunk_invocations"),
    }


# ---------------------------------------------------------------------------
# Section 9.I: qualitative selection (pure function of persisted scalars).
# ---------------------------------------------------------------------------


def _per_image_category_pixel_counts(arrays: Mapping[str, np.ndarray], i: int) -> dict[str, int]:
    harmful = int(arrays["harmful_pixel_count"][i])
    beneficial = int(arrays["beneficial_pixel_count"][i])
    third_label = int(arrays["third_label_pixel_count"][i])
    disputed_boundary = int(arrays["disputed_boundary_pixel_count"][i])
    disputed_interior = int(arrays["disputed_interior_pixel_count"][i])
    persistence_reverting_h32 = (
        int(arrays["persistence_h32_harmful_reverted_count"][i])
        + int(arrays["persistence_h32_beneficial_reverted_count"][i])
        + int(arrays["persistence_h32_third_label_reverted_count"][i])
    )
    return {
        "harmful_washout": harmful,
        "beneficial_correction": beneficial,
        "third_label_transition": third_label,
        "disputed_boundary": disputed_boundary,
        "disputed_interior": disputed_interior,
        "oracle_reverted": harmful,  # oracle only ever reverts harmful-washout pixels back to unary
        "horizon_persistence_reverting": persistence_reverting_h32,
        # "revert_set_fragmentation" deliberately absent -- see DEFERRED_QUALITATIVE_CATEGORIES.
    }


def build_qualitative_selection(
    source: LoadedGate1Source, *, requested_count_per_category: int = DEFAULT_QUALITATIVE_REQUESTED_COUNT_PER_CATEGORY,
) -> dict[str, Any]:
    """Runs the real, committed deterministic selection formula
    (``select_category_examples``/``compute_selection_key``) for every
    category whose eligibility is exactly reconstructable from persisted
    per-image scalars. ``revert_set_fragmentation`` is deferred (never
    fabricated) since its eligibility requires a live pixel-level
    connected-component count this schema never persists. Panel
    rendering is out of scope for every category: the v2 schema
    deliberately excludes full stitched score maps / component masks
    (`excluded_by_design`), so no raw prediction/label input exists to
    render from."""
    ids = list(source.checkpoint["completed_image_ids"])
    n = len(ids)
    pixel_counts_by_id = {ids[i]: _per_image_category_pixel_counts(source.arrays, i) for i in range(n)}

    results: list[CategorySelectionResult] = []
    for category_name in CATEGORY_ORDER:
        if category_name in DEFERRED_QUALITATIVE_CATEGORIES:
            continue
        eligibility_by_id = {
            cid: category_eligibility_from_pixel_counts(pixel_counts_by_id[cid])[category_name] for cid in ids
        }
        result = select_category_examples(
            category_name=category_name, dataset_name="coco_stuff_500", canonical_ids_in_order=ids,
            eligibility_by_id=eligibility_by_id, k=requested_count_per_category, namespace=QUALITATIVE_SELECTION_NAMESPACE,
        )
        results.append(result)

    categories: dict[str, Any] = {}
    for result in results:
        categories[result.category_name] = {
            "category_name": result.category_name, "requested_count": result.requested_count,
            "eligible_count": result.eligible_count, "selected_count": result.selected_count,
            "selected_ids": list(result.selected_ids), "selection_keys": dict(result.selection_keys),
            "shortage_status": result.shortage_status,
        }

    return {
        "categories": categories,
        "deferred_categories": {
            name: "requires a live pixel-level connected-component count not persisted by the v2 schema"
            for name in DEFERRED_QUALITATIVE_CATEGORIES
        },
        "selection_namespace": QUALITATIVE_SELECTION_NAMESPACE,
        "requested_count_per_category": requested_count_per_category,
        "requested_count_per_category_note": "analysis-layer parameter, not bound by any committed identity",
        "panel_rendering": "deferred -- v2 schema excludes full stitched score maps / component masks (excluded_by_design); no raw prediction/label input exists in the finalized artifacts to render from",
        "selection_trace_digest_over_reconstructed_categories": selection_trace_digest(results),
    }


# ---------------------------------------------------------------------------
# Section 10: Gate-1 decision.
# ---------------------------------------------------------------------------

ORACLE_UNARY_GATE_THRESHOLD_PP = 2.0   # [gate_contract].oracle_unary_gate_threshold_percentage_points
RECOVER_GATE_THRESHOLD_PP = 1.0        # [gate_contract].recover_gate_threshold_percentage_points

GATE_PROCEED = "PROCEED_TO_PHASE_1B"
GATE_STOP = "STOP_DFP"
GATE_INCONCLUSIVE = "INCONCLUSIVE"


def apply_gate1_decision(oracle: Mapping[str, Any], protocol_identity: Mapping[str, Any]) -> dict[str, Any]:
    gate = protocol_identity["gate_contract"]
    oracle_threshold = float(gate["oracle_unary_gate_threshold_percentage_points"])
    recover_threshold = float(gate["recover_gate_threshold_percentage_points"])
    if oracle_threshold != ORACLE_UNARY_GATE_THRESHOLD_PP or recover_threshold != RECOVER_GATE_THRESHOLD_PP:
        raise DfpPhase1aGate1AnalysisError(
            "live [gate_contract] thresholds disagree with this module's own hardcoded documentation copy -- "
            "the live identity is authoritative; this module's constants must be updated to match, not silently overridden"
        )

    g_oracle_unary = oracle["G_oracle_unary_pp"]
    g_recover = oracle["G_recover_pp"]
    oracle_pass = g_oracle_unary >= oracle_threshold
    recover_pass = g_recover >= recover_threshold
    dataset_passes_both = oracle_pass and recover_pass

    if dataset_passes_both:
        decision = GATE_PROCEED
        rationale = (
            f"coco_stuff_500 passes both frozen Gate-1 thresholds on THIS dataset "
            f"(G_oracle_unary={g_oracle_unary:.4f}pp >= {oracle_threshold}pp AND G_recover={g_recover:.4f}pp >= {recover_threshold}pp). "
            "Per [gate_contract].outcome_exactly_one_dataset_passes_both (ade20k_500 has not been evaluated in this task), "
            "this supports only a regime-scoped claim restricted to coco_stuff_500 -- never a cross-dataset claim, "
            "and never a deployable method claim."
        )
    else:
        decision = GATE_STOP
        rationale = (
            f"coco_stuff_500 fails the frozen Gate-1 predicate "
            f"(G_oracle_unary={g_oracle_unary:.4f}pp {'>=' if oracle_pass else '<'} {oracle_threshold}pp; "
            f"G_recover={g_recover:.4f}pp {'>=' if recover_pass else '<'} {recover_threshold}pp). "
            "Per [gate_contract], DFP does not clear the same-dataset gate on coco_stuff_500."
        )

    return {
        "decision": decision,
        "criteria": {
            "oracle_unary_gate_threshold_pp": oracle_threshold, "G_oracle_unary_pp": g_oracle_unary, "oracle_unary_gate_pass": bool(oracle_pass),
            "recover_gate_threshold_pp": recover_threshold, "G_recover_pp": g_recover, "recover_gate_pass": bool(recover_pass),
            "both_thresholds_pass_same_dataset": bool(dataset_passes_both),
        },
        "rationale": rationale,
        "scope_note": (
            "[gate_contract] is defined over a PAIR of datasets (coco_stuff_500, ade20k_500). This analysis covers "
            "coco_stuff_500 only; ade20k_500 has not been run. The decision above is therefore necessarily "
            "dataset-scoped, never a cross-dataset claim."
        ),
    }


__all__ = [
    "DfpPhase1aGate1AnalysisError",
    "LoadedGate1Source",
    "load_gate1_source",
    "ConfusionStatistics",
    "build_variant_confusion",
    "compute_all_variant_metrics",
    "independent_numpy_cross_check",
    "build_horizon_table",
    "build_dispute_accounting",
    "build_oracle_analysis",
    "build_persistence_baseline",
    "build_component_structure",
    "build_reversion_coherence",
    "build_boundary_concentration",
    "build_compute_projection",
    "build_qualitative_selection",
    "apply_gate1_decision",
    "GATE_PROCEED", "GATE_STOP", "GATE_INCONCLUSIVE",
    "ORACLE_UNARY_GATE_THRESHOLD_PP", "RECOVER_GATE_THRESHOLD_PP",
]
