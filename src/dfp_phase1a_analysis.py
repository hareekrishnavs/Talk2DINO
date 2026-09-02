"""DFP Phase 1A Step 3: pure-CPU analysis core over already-stitched score
arrays and label maps.

Dataset-independent, model-independent, CUDA-independent. This module
never decodes an image or mask, never resolves a dataset root, never
constructs a graph, never runs finite-step propagation, never
interpolates/stitches windows, and never extracts semantic components or
soft membership. It operates exclusively on NumPy arrays the caller
already produced elsewhere (a future execution driver, out of this
step's scope) and on primitive values the caller has already read from
the frozen `e13-dfp-phase1a-protocol` identity
(`src.dfp_phase1a_identity.load_identity`).

No operational default is hardcoded here for horizons, the primary
horizon, tie policy, metric units, ignore policy, class count, boundary-
width convention, persistence horizons, or coherence connectivity/
thresholds -- every one of those is an explicit parameter (or a field of
:class:`AnalysisContract`, itself built only from the live identity by
:func:`load_analysis_contract_from_protocol`). Ground truth is used only
to (a) validate/exclude ignore pixels, (b) classify the four dispute
categories, and (c) construct the GT-conditioned oracle diagnostic maps
explicitly named as such below -- never to construct a deployable
prediction.

Boundary IoU (``compute_boundary_statistics`` / ``aggregate_boundary_
statistics`` / ``compute_boundary_metrics``) is bound to the additive
``e13-dfp-phase1a-boundary-iou`` authority supplement
(``evaluation_identities/e13_dfp_phase1a_boundary_iou.toml``,
``src.dfp_phase1a_boundary_iou_identity``), which resolves -- without
ever rewriting -- the committed protocol identity's own deliberately
deferred ``[boundary_iou].implementation_source_status =
"not_yet_bound"``. The binary boundary primitive is a line-for-line
reproduction of the pinned official ``mask_to_boundary`` function from
Cheng et al.'s boundary-iou-api (CVPR 2021), exact commit and SHA-256
pinned in the supplement; see ``licenses/boundary_iou_api_LICENSE.txt``
for the required BSD notice. The classwise, ignore-aware, dataset-level
semantic aggregation built on top of that primitive is an explicit
repository-defined extension, never itself claimed as part of the
upstream API (see the supplement's own ``[semantic_extension]`` block).
None of these three functions is operational without an explicit,
already-verified :class:`BoundaryIouAuthority` (see
:func:`load_boundary_iou_authority`) -- there is no default and no
silent fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


class DfpPhase1aAnalysisError(ValueError):
    """Raised on any DFP Phase 1A Step 3 analysis invariant violation.
    Always fail closed. Never raised for KeyboardInterrupt, SystemExit,
    or MemoryError -- those always propagate."""


# ---------------------------------------------------------------------------
# Runtime contract -- built only from the live, already-validated protocol
# identity. Never a hardcoded default.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryIouAuthority:
    """Independently verified facts from the additive Boundary IoU
    authority supplement (`evaluation_identities/
    e13_dfp_phase1a_boundary_iou.toml`), never constructible except by
    `load_boundary_iou_authority` re-validating both identities from
    disk. Boundary IoU functions require one of these explicitly --
    there is no way to make them operational by accident."""

    protocol_identity_sha256: str
    official_source_commit: str
    official_source_sha256: str
    dilation_ratio_primary: float
    fixed_sensitivity_widths: tuple[int, ...]


def load_boundary_iou_authority(*, repo_root: Path) -> BoundaryIouAuthority:
    """Independently loads AND fully cross-validates the Boundary IoU
    authority supplement against the live protocol identity on disk --
    see `src.dfp_phase1a_boundary_iou_identity.validate_static_configuration`
    for the complete check (protocol path/hash match, the protocol's
    live `[boundary_iou].implementation_source_status` still equal to
    the deferred status this supplement exists to resolve, the local
    license notice file's existence). Never trusts an in-memory dict for
    any of this -- always re-reads from `repo_root`."""
    from src.dfp_phase1a_boundary_iou_identity import (
        DfpPhase1aBoundaryIouIdentityError,
        validate_static_configuration as _validate_boundary_supplement,
    )

    try:
        result = _validate_boundary_supplement(repo_root=repo_root)
    except DfpPhase1aBoundaryIouIdentityError as error:
        raise DfpPhase1aAnalysisError(f"Boundary IoU authority supplement failed validation: {error}") from error

    from src.dfp_phase1a_boundary_iou_identity import load_identity as _load_boundary_identity

    boundary_identity = _load_boundary_identity(repo_root=repo_root)
    return BoundaryIouAuthority(
        protocol_identity_sha256=result["protocol_identity_sha256"],
        official_source_commit=boundary_identity["official_source"]["repository_commit"],
        official_source_sha256=result["official_source_sha256"],
        dilation_ratio_primary=result["dilation_ratio_primary"],
        fixed_sensitivity_widths=result["fixed_sensitivity_widths"],
    )


@dataclass(frozen=True)
class AnalysisContract:
    horizons: tuple[int, ...]
    unary_horizon: int
    final_horizon: int
    primary_sensitivity_horizon: int
    secondary_sensitivity_horizons: tuple[int, ...]
    persistence_horizons: tuple[int, ...]
    primary_persistence_horizon: int
    class_count: int
    ignore_index: int
    metric_unit: str
    difference_unit: str
    tie_rule: str
    primary_connectivity: int
    sensitivity_connectivity: int
    reversion_connectivity: int
    reversion_small_component_threshold_pixels: int
    boundary_iou_authority: BoundaryIouAuthority | None = None


_SUPPORTED_TIE_RULE = "exact_argmax_ties_select_lowest_class_index"
_SUPPORTED_PERSISTENCE_RULE = (
    "for a T=320 dispute a->b: retain b if the stitched T=t winner is also b, otherwise use unary a; "
    "outside T=320 disputes, retain the T=320 prediction unchanged"
)
_DATASET_BLOCK_BY_KEY = {"coco_stuff": "coco_stuff_dataset", "ade20k": "ade20k_dataset"}


def load_analysis_contract_from_protocol(
    protocol_identity: Mapping[str, Any], *, dataset_key: str, bind_boundary_iou: bool = False, repo_root: Path | None = None,
) -> AnalysisContract:
    """Build an :class:`AnalysisContract` from an already-loaded, already-
    strictly-validated protocol identity (see
    `src.dfp_phase1a_identity.load_identity`). Never re-derives or
    re-guesses a value the identity does not state; rejects (rather than
    silently substituting) if the identity's own frozen algorithmic
    prose no longer matches the one behavior this analysis core actually
    implements.

    Boundary IoU becomes operational on the returned contract only when
    `bind_boundary_iou=True` AND `repo_root` is given: both the
    committed protocol identity and the additive Boundary IoU authority
    supplement are then independently re-verified from disk (see
    `load_boundary_iou_authority`) before `boundary_iou_authority` is
    populated. Omitting `bind_boundary_iou` leaves it `None`, and every
    Boundary IoU function then refuses to run -- there is no silent
    fallback."""
    if dataset_key not in _DATASET_BLOCK_BY_KEY:
        raise DfpPhase1aAnalysisError(f"unknown dataset_key {dataset_key!r}; expected one of {sorted(_DATASET_BLOCK_BY_KEY)}")
    dataset_block = protocol_identity[_DATASET_BLOCK_BY_KEY[dataset_key]]
    propagation = protocol_identity["propagation"]
    horizon_persistence = protocol_identity["horizon_persistence"]
    metric_contract = protocol_identity["metric_contract"]
    dispute_protocol = protocol_identity["dispute_protocol"]
    structural = protocol_identity["structural_components"]
    dispute_categories = protocol_identity["dispute_categories"]
    reversion_coherence = protocol_identity["reversion_coherence"]

    try:
        from src.dfp_phase1a_identity import DfpPhase1aIdentityError, validate_reversion_coherence_block

        validate_reversion_coherence_block(reversion_coherence, dispute_categories, structural)
    except DfpPhase1aIdentityError as exc:
        raise DfpPhase1aAnalysisError(str(exc)) from exc

    tie_rule = dispute_protocol["tie_rule"]
    if tie_rule != _SUPPORTED_TIE_RULE:
        raise DfpPhase1aAnalysisError(
            f"identity.dispute_protocol.tie_rule={tie_rule!r} is not the tie rule this analysis core implements "
            f"({_SUPPORTED_TIE_RULE!r}); refusing to silently proceed with a mismatched algorithm"
        )
    persistence_rule = horizon_persistence["rule"]
    if persistence_rule != _SUPPORTED_PERSISTENCE_RULE:
        raise DfpPhase1aAnalysisError(
            "identity.horizon_persistence.rule disagrees with the persistence algorithm this analysis core "
            "implements; refusing to silently proceed with a mismatched algorithm"
        )

    boundary_iou_authority = None
    if bind_boundary_iou:
        if repo_root is None:
            raise DfpPhase1aAnalysisError("repo_root is required when bind_boundary_iou=True")
        boundary_iou_authority = load_boundary_iou_authority(repo_root=repo_root)

    horizons = tuple(propagation["horizons"])
    return AnalysisContract(
        horizons=horizons,
        unary_horizon=horizons[0],
        final_horizon=horizons[-1],
        primary_sensitivity_horizon=propagation["primary_sensitivity_horizon"],
        secondary_sensitivity_horizons=tuple(propagation["secondary_sensitivity_horizons"]),
        persistence_horizons=tuple(horizon_persistence["applies_to_horizons"]),
        primary_persistence_horizon=horizon_persistence["primary_horizon"],
        class_count=dataset_block["class_count"],
        ignore_index=dataset_block["ignore_index"],
        metric_unit=metric_contract["reported_metric_unit"],
        difference_unit=metric_contract["difference_unit"],
        tie_rule=tie_rule,
        primary_connectivity=structural["primary_connectivity"],
        sensitivity_connectivity=structural["sensitivity_connectivity"],
        reversion_connectivity=reversion_coherence["connectivity"],
        reversion_small_component_threshold_pixels=reversion_coherence["small_component_threshold_pixels"],
        boundary_iou_authority=boundary_iou_authority,
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise DfpPhase1aAnalysisError(f"{label} must be an exact non-boolean integer")
    return value


def validate_stitched_inputs(
    scores: Mapping[int, np.ndarray], gt: np.ndarray, *, horizons: Sequence[int], class_count: int, ignore_index: int,
) -> None:
    """Fail-closed validation of one image's stitched score dict and
    ground-truth label map. Never mutates `scores` or `gt`. Score layout
    is frozen as `[class, height, width]` -- no alias, no dimension
    inference."""
    _require_exact_int(class_count, "class_count")
    if class_count <= 0:
        raise DfpPhase1aAnalysisError("class_count must be positive")
    _require_exact_int(ignore_index, "ignore_index")

    horizons = list(horizons)
    if len(set(horizons)) != len(horizons):
        raise DfpPhase1aAnalysisError(f"horizons contains a duplicate value: {horizons}")
    required = set(horizons)

    if not isinstance(scores, Mapping):
        raise DfpPhase1aAnalysisError("scores must be a mapping of horizon -> score array")
    observed_keys = set(scores)
    for key in observed_keys:
        if type(key) is not int:
            raise DfpPhase1aAnalysisError(f"scores dict key {key!r} must be an exact int (not bool/float/str/np.integer)")
    if observed_keys != required:
        missing = sorted(required - observed_keys)
        unknown = sorted(observed_keys - required)
        raise DfpPhase1aAnalysisError(f"scores horizon set mismatch: missing={missing}, unknown={unknown}")

    shape = None
    for horizon in horizons:
        arr = scores[horizon]
        label = f"scores[{horizon}]"
        if not isinstance(arr, np.ndarray):
            raise DfpPhase1aAnalysisError(f"{label} must be a numpy.ndarray")
        if arr.dtype != np.float32 and arr.dtype != np.float64:
            raise DfpPhase1aAnalysisError(f"{label} dtype must be exactly float32 or float64, observed {arr.dtype}")
        if arr.ndim != 3:
            raise DfpPhase1aAnalysisError(f"{label} must be exactly 3-D [class, height, width], observed {arr.ndim}-D")
        if arr.shape[0] != class_count:
            raise DfpPhase1aAnalysisError(f"{label} class dimension {arr.shape[0]} disagrees with class_count {class_count}")
        if arr.shape[1] <= 0 or arr.shape[2] <= 0:
            raise DfpPhase1aAnalysisError(f"{label} spatial dimensions must be positive, observed {arr.shape[1:]}")
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise DfpPhase1aAnalysisError(f"{label} shape {arr.shape} disagrees with horizon {horizons[0]}'s shape {shape}")
        if not np.all(np.isfinite(arr)):
            raise DfpPhase1aAnalysisError(f"{label} contains a non-finite value (NaN or Infinity)")

    if not isinstance(gt, np.ndarray):
        raise DfpPhase1aAnalysisError("gt must be a numpy.ndarray")
    if gt.dtype == np.bool_:
        raise DfpPhase1aAnalysisError("gt dtype must not be bool")
    if not np.issubdtype(gt.dtype, np.integer):
        raise DfpPhase1aAnalysisError(f"gt dtype must be a non-boolean integer dtype, observed {gt.dtype}")
    if gt.ndim != 2:
        raise DfpPhase1aAnalysisError(f"gt must be exactly 2-D [height, width], observed {gt.ndim}-D")
    if shape is not None and gt.shape != shape[1:]:
        raise DfpPhase1aAnalysisError(f"gt shape {gt.shape} disagrees with score spatial shape {shape[1:]}")

    legal = (gt == ignore_index) | ((gt >= 0) & (gt < class_count))
    if not np.all(legal):
        bad = np.unique(gt[~legal])
        raise DfpPhase1aAnalysisError(f"gt contains value(s) outside [0, class_count) and != ignore_index: {bad.tolist()[:10]}")


# ---------------------------------------------------------------------------
# Deterministic label maps (explicit lowest-index tie-break)
# ---------------------------------------------------------------------------


def _lowest_index_argmax(scores_chw: np.ndarray) -> np.ndarray:
    """label(x) = lowest class index c maximizing scores_chw[c, x].
    Implemented explicitly via a tie mask + argmin over class index --
    never via `np.argmax`'s incidental first-occurrence behavior."""
    class_count = scores_chw.shape[0]
    max_value = scores_chw.max(axis=0, keepdims=True)
    is_max = scores_chw == max_value
    class_index = np.arange(class_count, dtype=np.int64).reshape(class_count, 1, 1)
    sentinel = np.int64(class_count)
    masked = np.where(is_max, class_index, sentinel)
    return masked.min(axis=0).astype(np.int64)


def derive_horizon_labels(scores: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
    """For every horizon key present in `scores`, the lowest-index-argmax
    label map. Does not mutate `scores` or its arrays."""
    return {horizon: _lowest_index_argmax(arr) for horizon, arr in scores.items()}


# ---------------------------------------------------------------------------
# Four-way dispute accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DisputeResult:
    a: np.ndarray
    b: np.ndarray
    valid: np.ndarray
    dispute: np.ndarray
    harmful: np.ndarray
    beneficial: np.ndarray
    third_label: np.ndarray
    harmful_count: int
    beneficial_count: int
    third_label_count: int
    dispute_count: int


def classify_disputes(a: np.ndarray, b: np.ndarray, gt: np.ndarray, *, ignore_index: int) -> DisputeResult:
    """a = unary (U) label map, b = final (P) label map. `valid = GT !=
    ignore_index`; `D = valid & (a != b)`. harmful/beneficial/third_label
    are an exact, pairwise-disjoint partition of D (never containing an
    ignore pixel). Does not mutate any input array."""
    valid = gt != ignore_index
    dispute = valid & (a != b)
    harmful = dispute & (gt == a) & (gt != b)
    beneficial = dispute & (gt == b) & (gt != a)
    third_label = dispute & (gt != a) & (gt != b)
    return DisputeResult(
        a=a, b=b, valid=valid, dispute=dispute, harmful=harmful, beneficial=beneficial, third_label=third_label,
        harmful_count=int(harmful.sum()), beneficial_count=int(beneficial.sum()),
        third_label_count=int(third_label.sum()), dispute_count=int(dispute.sum()),
    )


# ---------------------------------------------------------------------------
# Oracle diagnostic maps -- GT-CONDITIONED, NEVER DEPLOYABLE PREDICTIONS.
# Ground truth is used here ONLY to construct these diagnostics.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleMaps:
    """GT-conditioned diagnostic label maps. None of these is a
    deployable prediction -- each uses ground truth directly and exists
    solely to measure a counterfactual metric effect."""

    o_p: np.ndarray  # harmful-only recovery oracle: P with harmful pixels replaced by a
    beneficial_undo_only: np.ndarray  # P with beneficial pixels replaced by a (undoes the correction)
    third_label_diagnostic: np.ndarray  # P with third-label pixels replaced by a


def construct_oracle_maps(dispute: DisputeResult) -> OracleMaps:
    """O_P: begin with P (=dispute.b); at harmful pixels replace with
    U/a; retain P everywhere else, including beneficial and third-label
    pixels. beneficial_undo_only and third_label_diagnostic are built
    identically but targeting their own category only -- one-sided
    diagnostics isolating a single counterfactual effect at a time.
    GT is used only through `dispute`'s already-computed category
    masks; no new GT access here."""
    p = dispute.b
    a = dispute.a
    o_p = np.where(dispute.harmful, a, p)
    beneficial_undo_only = np.where(dispute.beneficial, a, p)
    third_label_diagnostic = np.where(dispute.third_label, a, p)
    return OracleMaps(o_p=o_p, beneficial_undo_only=beneficial_undo_only, third_label_diagnostic=third_label_diagnostic)


# ---------------------------------------------------------------------------
# Horizon-persistence selectors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersistenceMaps:
    horizon: int
    full: np.ndarray
    harmful_recovery_only: np.ndarray
    beneficial_undo_only: np.ndarray
    third_label_fallback_only: np.ndarray
    changed_pixel_counts: dict[str, int]


def construct_horizon_persistence_maps(dispute: DisputeResult, label_t: np.ndarray, *, horizon: int) -> PersistenceMaps:
    """For a T=320 dispute a->b: retain b if the stitched T=t winner
    (`label_t`) is also b, otherwise use unary a -- even if label_t is a
    third class c not in {a,b}. Outside D, retain P (=b) unchanged.
    Since P == b everywhere (b IS the T=320 label map), this collapses
    to: revert to a exactly where D & (label_t != b), else keep b. The
    three one-sided variants apply the identical reversion rule
    restricted to one dispute category only, holding the other
    categories fixed at b -- mirroring the oracle maps' one-sided
    diagnostics so each category's effect can be measured
    independently."""
    b = dispute.b
    a = dispute.a
    reverts = dispute.dispute & (label_t != b)
    full = np.where(reverts, a, b)
    harmful_reverts = dispute.harmful & (label_t != b)
    beneficial_reverts = dispute.beneficial & (label_t != b)
    third_label_reverts = dispute.third_label & (label_t != b)
    harmful_recovery_only = np.where(harmful_reverts, a, b)
    beneficial_undo_only = np.where(beneficial_reverts, a, b)
    third_label_fallback_only = np.where(third_label_reverts, a, b)
    changed_pixel_counts = {
        "harmful": int(harmful_reverts.sum()),
        "beneficial": int(beneficial_reverts.sum()),
        "third_label": int(third_label_reverts.sum()),
    }
    return PersistenceMaps(
        horizon=horizon, full=full, harmful_recovery_only=harmful_recovery_only,
        beneficial_undo_only=beneficial_undo_only, third_label_fallback_only=third_label_fallback_only,
        changed_pixel_counts=changed_pixel_counts,
    )


# ---------------------------------------------------------------------------
# Dataset-level confusion statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfusionStatistics:
    intersection: np.ndarray
    union: np.ndarray
    predicted_pixels: np.ndarray
    ground_truth_pixels: np.ndarray
    confusion_matrix: np.ndarray  # [gt_class, predicted_class], int64


def compute_confusion_statistics(pred: np.ndarray, gt: np.ndarray, *, class_count: int, ignore_index: int) -> ConfusionStatistics:
    """Exact per-image confusion statistics, ignore pixels excluded.
    confusion_matrix[g, p] counts pixels with GT class g and predicted
    class p. intersection_c = confusion[c,c]; ground_truth_pixels_c =
    sum over predicted columns confusion[c,*]; predicted_pixels_c = sum
    over GT rows confusion[*,c]; union_c = ground_truth_pixels_c +
    predicted_pixels_c - intersection_c. All accumulation in int64 (no
    overflow risk at any realistic image/dataset scale). Does not
    mutate `pred` or `gt`."""
    valid = gt != ignore_index
    p = pred[valid].astype(np.int64)
    g = gt[valid].astype(np.int64)
    flat_index = g * class_count + p
    counts = np.bincount(flat_index, minlength=class_count * class_count).astype(np.int64)
    confusion = counts.reshape(class_count, class_count)
    intersection = np.diagonal(confusion).copy()
    ground_truth_pixels = confusion.sum(axis=1)
    predicted_pixels = confusion.sum(axis=0)
    union = ground_truth_pixels + predicted_pixels - intersection
    return ConfusionStatistics(
        intersection=intersection, union=union, predicted_pixels=predicted_pixels,
        ground_truth_pixels=ground_truth_pixels, confusion_matrix=confusion,
    )


def aggregate_confusion_statistics(stats: Sequence[ConfusionStatistics]) -> ConfusionStatistics:
    """Exact element-wise sum across images -- dataset metrics must be
    computed only after this summation, never averaged per image."""
    stats = list(stats)
    if not stats:
        raise DfpPhase1aAnalysisError("cannot aggregate an empty sequence of confusion statistics")
    intersection = np.sum([s.intersection for s in stats], axis=0, dtype=np.int64)
    union = np.sum([s.union for s in stats], axis=0, dtype=np.int64)
    predicted_pixels = np.sum([s.predicted_pixels for s in stats], axis=0, dtype=np.int64)
    ground_truth_pixels = np.sum([s.ground_truth_pixels for s in stats], axis=0, dtype=np.int64)
    confusion_matrix = np.sum([s.confusion_matrix for s in stats], axis=0, dtype=np.int64)
    return ConfusionStatistics(
        intersection=intersection, union=union, predicted_pixels=predicted_pixels,
        ground_truth_pixels=ground_truth_pixels, confusion_matrix=confusion_matrix,
    )


# ---------------------------------------------------------------------------
# Semantic metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticMetrics:
    aAcc: float
    per_class_iou: np.ndarray
    mIoU: float
    per_class_accuracy: np.ndarray
    mAcc: float
    classes_with_defined_iou: int
    classes_with_defined_accuracy: int


def compute_semantic_metrics(stats: ConfusionStatistics) -> SemanticMetrics:
    """All computation in float64; results reported as percent_0_100
    (never a fraction, never rounded prematurely). class IoU is
    undefined (NaN) when union == 0; class accuracy is undefined (NaN)
    when ground_truth_pixels == 0; mIoU/mAcc average only the defined
    classes; aAcc uses all valid pixels. Rejects an all-ignore
    aggregation (total ground-truth pixel count == 0) rather than
    returning a misleading metric."""
    total_gt = int(stats.ground_truth_pixels.sum())
    if total_gt == 0:
        raise DfpPhase1aAnalysisError(
            "cannot compute semantic metrics: zero valid (non-ignore) ground-truth pixels in the aggregated statistics"
        )
    intersection = stats.intersection.astype(np.float64)
    union = stats.union.astype(np.float64)
    ground_truth_pixels = stats.ground_truth_pixels.astype(np.float64)

    defined_iou = union > 0
    per_class_iou = np.full(intersection.shape, np.nan, dtype=np.float64)
    per_class_iou[defined_iou] = intersection[defined_iou] / union[defined_iou]

    defined_acc = ground_truth_pixels > 0
    per_class_accuracy = np.full(intersection.shape, np.nan, dtype=np.float64)
    per_class_accuracy[defined_acc] = intersection[defined_acc] / ground_truth_pixels[defined_acc]

    mIoU = float(np.mean(per_class_iou[defined_iou])) if np.any(defined_iou) else float("nan")
    mAcc = float(np.mean(per_class_accuracy[defined_acc])) if np.any(defined_acc) else float("nan")
    aAcc = float(intersection.sum() / total_gt)

    return SemanticMetrics(
        aAcc=aAcc * 100.0, per_class_iou=per_class_iou * 100.0, mIoU=mIoU * 100.0,
        per_class_accuracy=per_class_accuracy * 100.0, mAcc=mAcc * 100.0,
        classes_with_defined_iou=int(defined_iou.sum()), classes_with_defined_accuracy=int(defined_acc.sum()),
    )


# ---------------------------------------------------------------------------
# Gain definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GainSummary:
    propagation_gain: float
    g_recover: float
    g_oracle_unary: float
    relational_residual: float


def compute_gain_summary(*, metric_u: float, metric_p: float, metric_o_p: float) -> GainSummary:
    """propagation_gain = metric(P) - metric(U); G_recover =
    metric(O_P) - metric(P); G_oracle_unary = metric(O_P) - metric(U).
    `relational_residual` should be ~0 (bounded only by ordinary
    float64 subtraction reordering, never a scientific tolerance)."""
    propagation_gain = metric_p - metric_u
    g_recover = metric_o_p - metric_p
    g_oracle_unary = metric_o_p - metric_u
    relational_residual = g_oracle_unary - (g_recover + propagation_gain)
    return GainSummary(
        propagation_gain=propagation_gain, g_recover=g_recover, g_oracle_unary=g_oracle_unary,
        relational_residual=relational_residual,
    )


@dataclass(frozen=True)
class PersistenceGainSummary:
    horizon: int
    full_delta_vs_p: float
    harmful_recovery_gain: float
    beneficial_undo_loss: float
    third_label_fallback_delta: float


def compute_persistence_gain_summary(
    *, horizon: int, metric_p: float, metric_full: float, metric_harmful_only: float,
    metric_beneficial_only: float, metric_third_label_only: float,
) -> PersistenceGainSummary:
    """Each delta compares one independently-computed counterfactual
    map's metric against metric(P) -- never derived from another delta,
    since IoU-family metrics are nonlinear and these effects do not sum."""
    return PersistenceGainSummary(
        horizon=horizon,
        full_delta_vs_p=metric_full - metric_p,
        harmful_recovery_gain=metric_harmful_only - metric_p,
        beneficial_undo_loss=metric_beneficial_only - metric_p,
        third_label_fallback_delta=metric_third_label_only - metric_p,
    )


# ---------------------------------------------------------------------------
# Boundary IoU -- bound to the additive e13-dfp-phase1a-boundary-iou
# authority supplement (see src.dfp_phase1a_boundary_iou_identity and
# evaluation_identities/e13_dfp_phase1a_boundary_iou.toml). Every
# function below REQUIRES an explicit, already-verified
# BoundaryIouAuthority -- there is no default and no silent fallback;
# omitting one raises DfpPhase1aAnalysisError immediately.
#
# `_erode_and_subtract` / `_official_mask_to_boundary` reproduce, line
# for line, the pinned official `mask_to_boundary` function from
# boundary-iou-api (Cheng et al., CVPR 2021), commit
# 37d25586a677b043ed585f10e5c42d4e80176ea9, SHA-256
# a495cc15fdbded38681edef30b29c62014e81e864f0e68ca6062ce42db51fd9c --
# see licenses/boundary_iou_api_LICENSE.txt for the required BSD notice
# this reproduction is bound by. Behavior, not merely intent, is
# preserved exactly: identical `round()` convention, identical
# constant-zero one-pixel padding, identical 3x3 uint8 ones kernel,
# identical `cv2.erode` iteration count, identical crop, identical
# `mask - eroded_mask` boundary formula. `_boundary_fixed_width` reuses
# the identical erosion/crop/subtract core with an exact fixed iteration
# count instead of a ratio-derived one -- never reinterpreting a fixed
# pixel width as a ratio.
#
# Everything below "the semantic aggregation" line (per-class boundary
# masks, ignore handling, dataset-level accumulation, undefined/zero-
# union handling) is a REPOSITORY-DEFINED EXTENSION on top of that
# primitive -- the upstream boundary-iou-api package itself defines only
# the binary mask-to-boundary primitive, never a classwise semantic-
# segmentation aggregation. See the bound authority's own
# `[semantic_extension]` section (`is_official_semantic_definition =
# false`) for the frozen statement of this distinction.
# ---------------------------------------------------------------------------


def _erode_and_subtract(mask: np.ndarray, *, iterations: int) -> np.ndarray:
    h, w = mask.shape
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded_padded = cv2.erode(padded, kernel, iterations=iterations)
    eroded = eroded_padded[1 : h + 1, 1 : w + 1]
    return mask - eroded


def _official_mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    return _erode_and_subtract(mask, iterations=dilation)


def _boundary_fixed_width(mask: np.ndarray, width: int) -> np.ndarray:
    return _erode_and_subtract(mask, iterations=width)


def _require_boundary_width(boundary_width: Any, authority: BoundaryIouAuthority) -> None:
    if isinstance(boundary_width, bool):
        raise DfpPhase1aAnalysisError("boundary_width must not be a bool")
    if type(boundary_width) is float:
        if boundary_width != authority.dilation_ratio_primary:
            raise DfpPhase1aAnalysisError(
                f"boundary_width {boundary_width!r} disagrees with the authority's frozen primary ratio "
                f"{authority.dilation_ratio_primary!r}"
            )
        return
    if type(boundary_width) is int:
        if boundary_width not in authority.fixed_sensitivity_widths:
            raise DfpPhase1aAnalysisError(
                f"boundary_width {boundary_width!r} is not one of the authority's frozen fixed sensitivity widths "
                f"{authority.fixed_sensitivity_widths}"
            )
        return
    raise DfpPhase1aAnalysisError("boundary_width must be an exact float (the primary ratio) or an exact int (a fixed pixel width)")


@dataclass(frozen=True)
class BoundaryStatistics:
    boundary_intersection: np.ndarray
    boundary_union: np.ndarray


def compute_boundary_statistics(
    pred: np.ndarray, gt: np.ndarray, *, class_count: int, ignore_index: int, boundary_width: float | int,
    boundary_iou_authority: BoundaryIouAuthority,
) -> BoundaryStatistics:
    """Classwise semantic Boundary IoU sufficient statistics for ONE
    image (a repository-defined extension -- see module-level comment
    above), using the pinned Cheng et al. 2021 mask_to_boundary
    primitive. `boundary_width` selects either the primary normalized
    ratio (an exact float, must equal the authority's frozen
    `dilation_ratio_primary`) or a fixed pixel width (an exact int, must
    be one of the authority's frozen `fixed_sensitivity_widths`).
    `boundary_iou_authority` is REQUIRED -- there is no default. For
    every class c: `gt_c = (gt == c)` (already ignore-safe, since
    `ignore_index` is guaranteed outside `[0, class_count)`);
    `pred_c = (pred == c) & valid` (explicitly ignore-safe, since a
    prediction may legally take any class value at an ignored pixel).
    Boundary masks are additionally re-masked by `valid` after
    extraction as an explicit safety net. Does not mutate `pred` or
    `gt`."""
    if not isinstance(boundary_iou_authority, BoundaryIouAuthority):
        raise DfpPhase1aAnalysisError("boundary_iou_authority must be an already-verified BoundaryIouAuthority (see load_boundary_iou_authority)")
    _require_boundary_width(boundary_width, boundary_iou_authority)
    if not isinstance(pred, np.ndarray) or not isinstance(gt, np.ndarray):
        raise DfpPhase1aAnalysisError("pred and gt must be numpy.ndarray")
    if pred.shape != gt.shape:
        raise DfpPhase1aAnalysisError(f"pred shape {pred.shape} disagrees with gt shape {gt.shape}")
    if pred.ndim != 2:
        raise DfpPhase1aAnalysisError(f"pred/gt must be exactly 2-D, observed {pred.ndim}-D")
    _require_exact_int(class_count, "class_count")
    if class_count <= 0:
        raise DfpPhase1aAnalysisError("class_count must be positive")
    _require_exact_int(ignore_index, "ignore_index")

    valid = gt != ignore_index
    boundary_intersection = np.zeros(class_count, dtype=np.int64)
    boundary_union = np.zeros(class_count, dtype=np.int64)

    is_ratio = type(boundary_width) is float

    for c in range(class_count):
        gt_c = (gt == c).astype(np.uint8)
        pred_c = ((pred == c) & valid).astype(np.uint8)

        if is_ratio:
            gt_boundary = _official_mask_to_boundary(gt_c, dilation_ratio=boundary_width).astype(bool)
            pred_boundary = _official_mask_to_boundary(pred_c, dilation_ratio=boundary_width).astype(bool)
        else:
            gt_boundary = _boundary_fixed_width(gt_c, boundary_width).astype(bool)
            pred_boundary = _boundary_fixed_width(pred_c, boundary_width).astype(bool)

        gt_boundary = gt_boundary & valid
        pred_boundary = pred_boundary & valid

        boundary_intersection[c] = int(np.sum(gt_boundary & pred_boundary))
        boundary_union[c] = int(np.sum(gt_boundary | pred_boundary))

    return BoundaryStatistics(boundary_intersection=boundary_intersection, boundary_union=boundary_union)


def aggregate_boundary_statistics(stats: Sequence[BoundaryStatistics]) -> BoundaryStatistics:
    """Exact element-wise sum across images -- dataset-level boundary
    metrics must be computed only after this summation, never averaged
    per image."""
    stats = list(stats)
    if not stats:
        raise DfpPhase1aAnalysisError("cannot aggregate an empty sequence of boundary statistics")
    boundary_intersection = np.sum([s.boundary_intersection for s in stats], axis=0, dtype=np.int64)
    boundary_union = np.sum([s.boundary_union for s in stats], axis=0, dtype=np.int64)
    return BoundaryStatistics(boundary_intersection=boundary_intersection, boundary_union=boundary_union)


@dataclass(frozen=True)
class BoundaryMetrics:
    per_class_biou: np.ndarray
    mean_biou: float
    classes_with_defined_biou: int


def compute_boundary_metrics(stats: BoundaryStatistics) -> BoundaryMetrics:
    """float64, percent_0_100. class bIoU is undefined (NaN) where total
    boundary_union == 0; mean bIoU averages only classes with nonzero
    total boundary_union; rejects an aggregation where every class has
    zero total boundary_union rather than returning a misleading value."""
    if not isinstance(stats, BoundaryStatistics):
        raise DfpPhase1aAnalysisError("stats must be a BoundaryStatistics instance")
    total_union = int(stats.boundary_union.sum())
    if total_union == 0:
        raise DfpPhase1aAnalysisError(
            "cannot compute boundary metrics: every class has zero total boundary_union in the aggregated statistics"
        )
    intersection = stats.boundary_intersection.astype(np.float64)
    union = stats.boundary_union.astype(np.float64)
    defined = union > 0
    per_class_biou = np.full(intersection.shape, np.nan, dtype=np.float64)
    per_class_biou[defined] = intersection[defined] / union[defined]
    mean_biou = float(np.mean(per_class_biou[defined])) if np.any(defined) else float("nan")
    return BoundaryMetrics(
        per_class_biou=per_class_biou * 100.0, mean_biou=mean_biou * 100.0, classes_with_defined_biou=int(defined.sum()),
    )


# ---------------------------------------------------------------------------
# Reverted-set coherence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReversionCoherenceStatistics:
    total_reverted_pixels: int
    component_count: int
    component_sizes: tuple[int, ...]
    min_component_size: int
    max_component_size: int
    mean_component_size: float
    median_component_size: float
    small_component_fraction: float
    perimeter: int
    perimeter_to_area_ratio: float
    is_zero_reversion: bool


def _perimeter(mask: np.ndarray) -> int:
    """Count of foreground pixels with at least one non-foreground
    4-neighbor (image borders count as non-foreground). This is a
    self-contained convention independent of the labeling connectivity
    and of any external Boundary IoU authority."""
    if not mask.any():
        return 0
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = padded[1:-1, 1:-1]
    up = padded[0:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, 0:-2]
    right = padded[1:-1, 2:]
    boundary = interior & (~up | ~down | ~left | ~right)
    return int(boundary.sum())


def compute_reversion_coherence(
    reverted_mask: np.ndarray, *, connectivity: int, small_component_threshold: float,
) -> ReversionCoherenceStatistics:
    """Deterministic CPU connected-component statistics for an exact
    boolean reverted-pixel mask. `connectivity` must be exactly 4 or 8
    (sourced by the caller from the protocol's
    structural_components.sensitivity_connectivity /
    primary_connectivity). `small_component_threshold` is an explicit
    numeric threshold in PIXEL units -- any conversion from the
    protocol's patch-equivalent microcomponent threshold to pixels
    requires production stitching geometry that is out of this step's
    scope, so the caller must supply the already-converted value.
    Component sizes are returned sorted descending by size, ties broken
    by ascending raster-scan label id, for full determinism. Never
    smooths, closes, dilates, or otherwise modifies `reverted_mask`."""
    if not isinstance(reverted_mask, np.ndarray) or reverted_mask.dtype != np.bool_:
        raise DfpPhase1aAnalysisError("reverted_mask must be a numpy.ndarray with dtype exactly bool")
    if reverted_mask.ndim != 2:
        raise DfpPhase1aAnalysisError(f"reverted_mask must be exactly 2-D, observed {reverted_mask.ndim}-D")
    if connectivity not in (4, 8):
        raise DfpPhase1aAnalysisError(f"connectivity must be exactly 4 or 8, observed {connectivity!r}")
    if type(small_component_threshold) not in (int, float) or isinstance(small_component_threshold, bool):
        raise DfpPhase1aAnalysisError("small_component_threshold must be an exact int or float")
    if small_component_threshold < 0:
        raise DfpPhase1aAnalysisError("small_component_threshold must be non-negative")

    total_reverted = int(reverted_mask.sum())
    if total_reverted == 0:
        return ReversionCoherenceStatistics(
            total_reverted_pixels=0, component_count=0, component_sizes=(), min_component_size=0,
            max_component_size=0, mean_component_size=0.0, median_component_size=0.0,
            small_component_fraction=0.0, perimeter=0, perimeter_to_area_ratio=0.0, is_zero_reversion=True,
        )

    from scipy import ndimage

    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    labeled, num_components = ndimage.label(reverted_mask, structure=structure)
    raw_sizes = ndimage.sum_labels(reverted_mask, labeled, index=np.arange(1, num_components + 1)).astype(np.int64)
    order = np.lexsort((np.arange(num_components), -raw_sizes))
    sizes_sorted = tuple(int(s) for s in raw_sizes[order])

    small_pixels = sum(s for s in sizes_sorted if s < small_component_threshold)
    perimeter = _perimeter(reverted_mask)

    return ReversionCoherenceStatistics(
        total_reverted_pixels=total_reverted,
        component_count=int(num_components),
        component_sizes=sizes_sorted,
        min_component_size=int(min(sizes_sorted)),
        max_component_size=int(max(sizes_sorted)),
        mean_component_size=float(np.mean(sizes_sorted)),
        median_component_size=float(np.median(sizes_sorted)),
        small_component_fraction=float(small_pixels) / float(total_reverted),
        perimeter=perimeter,
        perimeter_to_area_ratio=float(perimeter) / float(total_reverted),
        is_zero_reversion=False,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase1aImageAnalysis:
    labels: dict[int, np.ndarray]
    dispute: DisputeResult
    oracle: OracleMaps
    persistence: dict[int, PersistenceMaps]
    confusion_p: ConfusionStatistics
    confusion_u: ConfusionStatistics
    confusion_o_p: ConfusionStatistics


def analyze_phase1a_image(
    scores: Mapping[int, np.ndarray], gt: np.ndarray, *, contract: AnalysisContract,
) -> Phase1aImageAnalysis:
    """Single-image orchestration: validate, derive horizon labels,
    classify disputes, construct the oracle diagnostic maps, construct
    every contract-specified persistence map, and compute this image's
    per-image confusion statistics for P, U, and O_P (ready for
    `aggregate_confusion_statistics` across a dataset). Does not compute
    metrics itself (metrics require dataset-level aggregation first --
    see `compute_semantic_metrics`), does not write anything, and does
    not touch Boundary IoU."""
    validate_stitched_inputs(
        scores, gt, horizons=contract.horizons, class_count=contract.class_count, ignore_index=contract.ignore_index,
    )
    labels = derive_horizon_labels(scores)
    a = labels[contract.unary_horizon]
    b = labels[contract.final_horizon]
    dispute = classify_disputes(a, b, gt, ignore_index=contract.ignore_index)
    oracle = construct_oracle_maps(dispute)

    persistence: dict[int, PersistenceMaps] = {}
    for horizon in contract.persistence_horizons:
        if horizon not in labels:
            raise DfpPhase1aAnalysisError(f"persistence horizon {horizon} has no corresponding entry in scores/labels")
        persistence[horizon] = construct_horizon_persistence_maps(dispute, labels[horizon], horizon=horizon)

    confusion_p = compute_confusion_statistics(b, gt, class_count=contract.class_count, ignore_index=contract.ignore_index)
    confusion_u = compute_confusion_statistics(a, gt, class_count=contract.class_count, ignore_index=contract.ignore_index)
    confusion_o_p = compute_confusion_statistics(oracle.o_p, gt, class_count=contract.class_count, ignore_index=contract.ignore_index)

    return Phase1aImageAnalysis(
        labels=labels, dispute=dispute, oracle=oracle, persistence=persistence,
        confusion_p=confusion_p, confusion_u=confusion_u, confusion_o_p=confusion_o_p,
    )


__all__ = [
    "DfpPhase1aAnalysisError",
    "BoundaryIouAuthority",
    "load_boundary_iou_authority",
    "AnalysisContract",
    "load_analysis_contract_from_protocol",
    "validate_stitched_inputs",
    "derive_horizon_labels",
    "DisputeResult",
    "classify_disputes",
    "OracleMaps",
    "construct_oracle_maps",
    "PersistenceMaps",
    "construct_horizon_persistence_maps",
    "ConfusionStatistics",
    "compute_confusion_statistics",
    "aggregate_confusion_statistics",
    "SemanticMetrics",
    "compute_semantic_metrics",
    "GainSummary",
    "compute_gain_summary",
    "PersistenceGainSummary",
    "compute_persistence_gain_summary",
    "BoundaryStatistics",
    "compute_boundary_statistics",
    "aggregate_boundary_statistics",
    "BoundaryMetrics",
    "compute_boundary_metrics",
    "ReversionCoherenceStatistics",
    "compute_reversion_coherence",
    "Phase1aImageAnalysis",
    "analyze_phase1a_image",
]
