"""Pure scientific/geometry logic for the 48-image semantic-evidence
discovery experiment: anchor selection, query construction, the four frozen
intervention views, Gaussian-blur intervention construction, score
extraction at the frozen pre-sigmoid boundary, raw-score validation, and
derived intervention features.

Frozen by (all values read from, and cross-checked against,
research_specs/semantic_evidence_discovery_execution_v1.toml's
[scientific_inheritance] table and its bound predecessor
research_specs/semantic_evidence_feasibility_v1.toml):

  [anchor_selection], [region], [interventions], [gaussian_blur],
  [scores.formulas], [scores.validity], [control] (ids 1-8).

This module never opens a file, never accesses GT/annotations, never
constructs a model, never loads a checkpoint, never initializes CUDA, never
selects images, never fits a statistical model, and never installs an
artifact. It operates purely on tensors/values already in memory, supplied
by the caller (the execution module in semantic_evidence_discovery_execute.py).

Torch is used only for the frozen Gaussian-blur primitive
(torchvision.transforms.functional.gaussian_blur, per [gaussian_blur].source)
and for tensor arithmetic identical in spirit to what score-domain formulas
already require elsewhere in this codebase (see semantic_evidence_t320.py);
no model, no CUDA, no checkpoint is touched by this module.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


class InterventionFeatureError(Exception):
    pass


def _reraise_if_fatal(exc: BaseException) -> None:
    if isinstance(exc, (KeyboardInterrupt, SystemExit, MemoryError)):
        raise exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InterventionFeatureError(message)


# ---------------------------------------------------------------------------
# Frozen scientific constants -- values only, each traceable to the exact
# TOML key named in the comment. The execution module cross-verifies these
# against the committed authority at runtime before calling into this
# module; this module itself hardcodes them as the pure-math layer, exactly
# as research_specs/semantic_evidence_discovery_execution_v1.toml's own
# [t0_t320_target_authority.frozen_constants] pattern does in
# src/semantic_evidence_t320.py.
# ---------------------------------------------------------------------------

PATCH_SIZE = 14  # [scientific_inheritance.e3_geometry].patch_size_pixels
GRID_SIDE = 32  # [scientific_inheritance.e3_geometry].patch_grid_side
ELIGIBLE_LO, ELIGIBLE_HI = 1, 30  # [scientific_inheritance.e3_geometry].eligible_grid_range

TARGET_ANCHORS_PER_WINDOW = 8  # [anchor_selection].target_anchors_per_window
STRATA = ("boundary_low", "boundary_high", "interior_low", "interior_high")  # [anchor_selection].strata
ANCHORS_PER_STRATUM = 2  # [anchor_selection].anchors_per_stratum
QUERIES_PER_ANCHOR = 2  # [anchor_selection].queries_per_anchor
ANCHOR_NAMESPACE = "talk2dino-semantic-evidence-anchor-selection-v1"  # [anchor_selection].anchor_namespace

CORE_PATCH_COUNT = 9  # [region].core_patch_count
CORE_PIXELS = 42  # [region].core_pixel_size (both dims equal)
CORE_HALF_PATCHES = 1  # 3x3 core => 1 patch on each side of the anchor

INTERVENTION_VIEWS = ("ORIGINAL", "CONTEXT_BLURRED", "REGION_BLURRED", "AREA_MATCHED_CONTROL")  # [interventions].views

BLUR_KERNEL_SIZE = (31, 31)  # [gaussian_blur].kernel_size
BLUR_SIGMA = (8.0, 8.0)  # [gaussian_blur].sigma
BLUR_PADDING_MODE = "reflect"  # [gaussian_blur].padding_mode -- torchvision's gaussian_blur always uses reflect

CONTROL_IDS = (1, 2, 3, 4, 5, 6, 7, 8)  # [control] ids

FEATURE_ORDER_MODEL_A = (1, 2, 3, 4, 5, 6, 7)  # [[statistics.model_fits]] id 1/3 feature_order
FEATURE_ORDER_MODEL_B_EXTRA = "intervention_reliability"  # appended for Model B, ids 2/4

INPUT_RANGE = (-1.0, 1.0)  # [scores.validity].input_range_required
OUTPUT_RANGE = (0.0, 1.0)  # [scores.validity].output_range_required


# ---------------------------------------------------------------------------
# Anchor key algorithm ([anchor_selection].anchor_key_algorithm)
# ---------------------------------------------------------------------------

def anchor_key(canonical_image_id: str, canonical_window_index: int, canonical_patch_index: int) -> str:
    """SHA256( UTF8(namespace) || 0x00 || ASCII(image_id) || 0x00 ||
    ASCII(window_index) || 0x00 || ASCII(patch_index) ).hexdigest()"""
    _require(isinstance(canonical_image_id, str) and canonical_image_id != "", "canonical_image_id must be a non-empty string")
    _require(isinstance(canonical_window_index, int) and not isinstance(canonical_window_index, bool), "canonical_window_index must be a real int")
    _require(isinstance(canonical_patch_index, int) and not isinstance(canonical_patch_index, bool), "canonical_patch_index must be a real int")
    parts = [
        ANCHOR_NAMESPACE.encode("utf-8"), b"\x00",
        canonical_image_id.encode("ascii"), b"\x00",
        str(canonical_window_index).encode("ascii"), b"\x00",
        str(canonical_patch_index).encode("ascii"),
    ]
    return hashlib.sha256(b"".join(parts)).hexdigest()


# ---------------------------------------------------------------------------
# Anchor selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatchInfo:
    row: int
    col: int
    patch_index: int  # row * GRID_SIDE + col, row-major
    unary_top1: int
    unary_top2: int
    unary_margin: float  # top1_score - top2_score


def _is_eligible(row: int, col: int) -> bool:
    return ELIGIBLE_LO <= row <= ELIGIBLE_HI and ELIGIBLE_LO <= col <= ELIGIBLE_HI


def _grid_neighbors_8(row: int, col: int) -> list:
    return [(row + dr, col + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]


def is_boundary_patch(patch: PatchInfo, patches_by_rc: dict) -> bool:
    """[anchor_selection].boundary_definition: ANY of the (always all 8, per
    eligible_patches_always_have_8_grid_neighbors) 8-connected grid
    neighbors has a different unary_top1 class."""
    for nr, nc in _grid_neighbors_8(patch.row, patch.col):
        neighbor = patches_by_rc.get((nr, nc))
        _require(neighbor is not None, f"eligible patch ({patch.row},{patch.col}) missing an expected 8-neighbor at ({nr},{nc})")
        if neighbor.unary_top1 != patch.unary_top1:
            return True
    return False


def select_anchors(canonical_image_id: str, canonical_window_index: int, eligible_patches: list,
                    full_grid_patches_by_rc: dict) -> list:
    """[anchor_selection] in full: stratify into boundary_low/boundary_high/
    interior_low/interior_high by median unary_margin split within each of
    boundary/interior; take the first 2 (by ascending (anchor_key,
    canonical_patch_index)) per stratum; fill any shortfall from remaining
    eligible patches in ascending global order until 8 total, no duplicate.
    Returns an ordered list of PatchInfo (length <= TARGET_ANCHORS_PER_WINDOW).

    `full_grid_patches_by_rc` must map every (row,col) in the full 32x32
    grid (not just the eligible 30x30 region) to its PatchInfo, since a
    boundary check on an eligible patch may need a non-eligible neighbor's
    own unary_top1 class (the neighbor need not itself be eligible)."""
    _require(len(eligible_patches) >= 1, "at least one eligible patch is required for anchor selection")
    _require(len(full_grid_patches_by_rc) == GRID_SIDE * GRID_SIDE, "full_grid_patches_by_rc must cover the entire 32x32 grid")

    boundary, interior = [], []
    for p in eligible_patches:
        (boundary if is_boundary_patch(p, full_grid_patches_by_rc) else interior).append(p)

    def median_split(group: list) -> tuple:
        if not group:
            return [], []
        margins = sorted(p.unary_margin for p in group)
        n = len(margins)
        median = margins[n // 2] if n % 2 == 1 else (margins[n // 2 - 1] + margins[n // 2]) / 2.0
        low = [p for p in group if p.unary_margin <= median]
        high = [p for p in group if p.unary_margin > median]
        return low, high

    boundary_low, boundary_high = median_split(boundary)
    interior_low, interior_high = median_split(interior)
    strata_members = {
        "boundary_low": boundary_low, "boundary_high": boundary_high,
        "interior_low": interior_low, "interior_high": interior_high,
    }

    def sort_key(p: PatchInfo) -> tuple:
        return (anchor_key(canonical_image_id, canonical_window_index, p.patch_index), p.patch_index)

    selected: list = []
    selected_indices: set = set()
    for stratum in STRATA:
        members = sorted(strata_members[stratum], key=sort_key)
        for p in members[:ANCHORS_PER_STRATUM]:
            if p.patch_index not in selected_indices:
                selected.append(p)
                selected_indices.add(p.patch_index)

    if len(selected) < TARGET_ANCHORS_PER_WINDOW:
        remaining = sorted(
            (p for p in eligible_patches if p.patch_index not in selected_indices),
            key=sort_key,
        )
        for p in remaining:
            if len(selected) >= TARGET_ANCHORS_PER_WINDOW:
                break
            selected.append(p)
            selected_indices.add(p.patch_index)

    return selected


@dataclass(frozen=True)
class QueryPair:
    q1: int  # anchor's own unary_top1 class
    q2: int  # anchor's own unary_top2 class


def construct_queries(patch: PatchInfo) -> QueryPair:
    """[anchor_selection].query_definition: q1=unary_top1, q2=unary_top2,
    fixed at anchor-selection time from the ORIGINAL view only, never
    reselected after an intervention view is constructed."""
    return QueryPair(q1=patch.unary_top1, q2=patch.unary_top2)


# ---------------------------------------------------------------------------
# Region / core-mask geometry ([region])
# ---------------------------------------------------------------------------

def core_pixel_bounds(anchor_row: int, anchor_col: int) -> tuple:
    """Returns (top, left, bottom, right) pixel bounds of the 3x3-patch core
    centered on (anchor_row, anchor_col), in the fixed 448x448 backbone
    canvas. Eligible anchors (row,col in [1,30]) always yield a fully
    in-bounds 42x42 core, per [region].core_patch_count_derivation."""
    _require(_is_eligible(anchor_row, anchor_col), "core_pixel_bounds requires an eligible (row,col)")
    top = (anchor_row - CORE_HALF_PATCHES) * PATCH_SIZE
    left = (anchor_col - CORE_HALF_PATCHES) * PATCH_SIZE
    bottom = top + CORE_PIXELS
    right = left + CORE_PIXELS
    return top, left, bottom, right


def core_patch_set(anchor_row: int, anchor_col: int) -> set:
    return {(anchor_row + dr, anchor_col + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def select_control_center(canonical_image_id: str, canonical_window_index: int,
                           anchor_row: int, anchor_col: int, eligible_patches: list) -> Optional[PatchInfo]:
    """[interventions.AREA_MATCHED_CONTROL].control_center_selection_rule:
    among eligible patches whose 3x3 core does not overlap the anchor's core
    (no shared patch, including diagonally), select by ascending
    (anchor_key, canonical_patch_index) using the anchor_key algorithm
    applied to the CONTROL candidate. Returns None (EXCLUDED_NO_VALID_CONTROL)
    if no such patch exists."""
    anchor_core = core_patch_set(anchor_row, anchor_col)
    candidates = []
    for p in eligible_patches:
        if (p.row, p.col) == (anchor_row, anchor_col):
            continue
        candidate_core = core_patch_set(p.row, p.col)
        if candidate_core & anchor_core:
            continue
        candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (anchor_key(canonical_image_id, canonical_window_index, p.patch_index), p.patch_index))
    return candidates[0]


# ---------------------------------------------------------------------------
# Gaussian blur ([gaussian_blur])
# ---------------------------------------------------------------------------

def deterministic_gaussian_blur(crop: torch.Tensor) -> torch.Tensor:
    """Applies the exact frozen Gaussian blur via
    torchvision.transforms.functional.gaussian_blur, kernel_size=(31,31),
    sigma=(8.0,8.0). torchvision's tensor-path implementation always uses
    reflect padding (per [gaussian_blur].padding_mode_verified_evidence);
    this function does not select a padding mode itself, since the frozen
    library call has no padding-mode parameter to override."""
    _require(isinstance(crop, torch.Tensor), "crop must be a torch.Tensor")
    return gaussian_blur(crop, kernel_size=list(BLUR_KERNEL_SIZE), sigma=list(BLUR_SIGMA))


def _region_mask(shape: tuple, bounds: tuple, device, dtype) -> torch.Tensor:
    """Boolean mask, True inside [top:bottom, left:right)."""
    top, left, bottom, right = bounds
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    mask[..., top:bottom, left:right] = True
    return mask


def build_intervention_view(view: str, original_crop: torch.Tensor, anchor_bounds: tuple,
                             control_bounds: Optional[tuple] = None) -> torch.Tensor:
    """Constructs one of the four frozen intervention views from the
    ORIGINAL crop. `original_crop` is [C, H, W] (or [1, C, H, W]), in the
    frozen [region].operates_on domain (before E3 normalization, before
    BGR->RGB swap). Returns a tensor of the same shape."""
    _require(view in INTERVENTION_VIEWS, f"unknown intervention view: {view!r}")
    if view == "ORIGINAL":
        return original_crop.clone()

    blurred = deterministic_gaussian_blur(original_crop)

    if view == "CONTEXT_BLURRED":
        # core unchanged bit-exactly; everything outside replaced by blur
        mask = _region_mask(original_crop.shape, anchor_bounds, original_crop.device, original_crop.dtype)
        out = torch.where(mask, original_crop, blurred)
        return out

    if view == "REGION_BLURRED":
        # everything outside unchanged bit-exactly; core replaced by blur
        mask = _region_mask(original_crop.shape, anchor_bounds, original_crop.device, original_crop.dtype)
        out = torch.where(mask, blurred, original_crop)
        return out

    if view == "AREA_MATCHED_CONTROL":
        _require(control_bounds is not None, "AREA_MATCHED_CONTROL requires control_bounds (no valid control excludes this anchor upstream)")
        control_mask = _region_mask(original_crop.shape, control_bounds, original_crop.device, original_crop.dtype)
        out = torch.where(control_mask, blurred, original_crop)
        return out

    raise InterventionFeatureError(f"unhandled view: {view!r}")


def verify_unchanged_domain(view: str, original_crop: torch.Tensor, view_crop: torch.Tensor,
                             anchor_bounds: tuple, control_bounds: Optional[tuple] = None) -> bool:
    """[gaussian_blur].unchanged_region_verification_requirement: for every
    generated view, pixels in the required-unchanged domain must be
    bit-exactly identical to ORIGINAL before that view's scores are used."""
    if view == "ORIGINAL":
        return torch.equal(original_crop, view_crop)
    if view == "CONTEXT_BLURRED":
        mask = _region_mask(original_crop.shape, anchor_bounds, original_crop.device, original_crop.dtype)
        return torch.equal(original_crop[mask], view_crop[mask])
    if view == "REGION_BLURRED":
        mask = _region_mask(original_crop.shape, anchor_bounds, original_crop.device, original_crop.dtype)
        inv = ~mask
        return torch.equal(original_crop[inv], view_crop[inv])
    if view == "AREA_MATCHED_CONTROL":
        anchor_mask = _region_mask(original_crop.shape, anchor_bounds, original_crop.device, original_crop.dtype)
        control_mask = _region_mask(original_crop.shape, control_bounds, original_crop.device, original_crop.dtype)
        unchanged_mask = ~(anchor_mask | control_mask) | anchor_mask
        return torch.equal(original_crop[unchanged_mask], view_crop[unchanged_mask])
    raise InterventionFeatureError(f"unhandled view: {view!r}")


# ---------------------------------------------------------------------------
# Score validity ([scores.validity]) and formulas ([scores.formulas])
# ---------------------------------------------------------------------------

def _finite_and_in_range(x: float) -> bool:
    return math.isfinite(x) and INPUT_RANGE[0] <= x <= INPUT_RANGE[1]


@dataclass(frozen=True)
class RawScores:
    s_original: float
    s_context: float
    s_region: float
    s_control: float


def validate_raw_scores(scores: RawScores) -> bool:
    """[scores.validity].input_check_order: every one of the four input
    fields is checked for finiteness AND membership in [-1,1] BEFORE any
    formula in [scores.formulas] is evaluated. Returns True iff all four
    pass; never clamps/imputes/replaces (input_failure_prohibited_actions)."""
    return all(_finite_and_in_range(v) for v in (scores.s_original, scores.s_context, scores.s_region, scores.s_control))


@dataclass(frozen=True)
class DerivedFeatures:
    context_change: float
    region_drop: float
    control_change: float
    context_stability: float
    specific_region_drop: float
    region_necessity: float
    intervention_reliability: float


def compute_derived_features(scores: RawScores) -> DerivedFeatures:
    """[scores.formulas], evaluated in the exact frozen order and argument
    order (constant listed first in min/max, matching the nan_laundering_
    hazard_note -- this function must only ever be called after
    validate_raw_scores() has returned True for this record)."""
    context_change = abs(scores.s_context - scores.s_original)
    region_drop = max(0, scores.s_original - scores.s_region)
    control_change = abs(scores.s_control - scores.s_original)
    context_stability = 1 - min(1, context_change / 2)
    specific_region_drop = max(0, region_drop - control_change)
    region_necessity = min(1, specific_region_drop / 2)
    intervention_reliability = context_stability * region_necessity
    return DerivedFeatures(
        context_change=context_change, region_drop=region_drop, control_change=control_change,
        context_stability=context_stability, specific_region_drop=specific_region_drop,
        region_necessity=region_necessity, intervention_reliability=intervention_reliability,
    )


def validate_derived_output(derived: DerivedFeatures) -> bool:
    """[scores.validity].output_fields_checked: a SEPARATE, second check
    after formula evaluation, on intervention_reliability only."""
    v = derived.intervention_reliability
    return math.isfinite(v) and OUTPUT_RANGE[0] <= v <= OUTPUT_RANGE[1]


# ---------------------------------------------------------------------------
# Control features ([control] ids 1-8; id 7 is NOT_AVAILABLE_IN_E3, unused)
# ---------------------------------------------------------------------------

def control_1_top1_unary_score(s_original_q: float) -> float:
    return s_original_q


def control_2_top1_minus_top2_margin(unary_margin: float) -> float:
    return unary_margin


def control_3_unary_entropy(unary_scores_171: torch.Tensor) -> float:
    """-sum(p_i * log(p_i)); numerically-stable softmax (subtract max logit
    before exponentiation)."""
    _require(unary_scores_171.numel() == 171, "control_3 requires the full 171-dim unary score vector")
    logits = unary_scores_171.double()
    p = torch.softmax(logits, dim=0)
    # avoid log(0): p is a softmax output, strictly positive for finite logits
    entropy = -(p * torch.log(p)).sum().item()
    return entropy


def control_4_dino_neighbor_cosine_similarity(anchor_dino_feature: torch.Tensor, neighbor_dino_features: torch.Tensor) -> float:
    """cosine_similarity(anchor, mean(8 neighbors)), raw unprojected DINO
    embeddings, ORIGINAL view only."""
    _require(neighbor_dino_features.shape[0] == 8, "control_4 requires exactly 8 neighbor features")
    mean_neighbor = neighbor_dino_features.mean(dim=0)
    cos = F.cosine_similarity(anchor_dino_feature.unsqueeze(0).double(), mean_neighbor.unsqueeze(0).double())
    return cos.item()


def control_5_normalized_distance_to_nearest_boundary(anchor_row: int, anchor_col: int, boundary_patches: list) -> Optional[float]:
    """[control] id 5. Returns None (missing) if there are zero
    boundary-flagged eligible patches in the window."""
    if not boundary_patches:
        return None
    normalization_constant = math.sqrt((ELIGIBLE_HI - ELIGIBLE_LO) ** 2 + (ELIGIBLE_HI - ELIGIBLE_LO) ** 2)
    raw_distance = min(
        math.sqrt((anchor_row - p.row) ** 2 + (anchor_col - p.col) ** 2)
        for p in boundary_patches
    )
    return min(1.0, raw_distance / normalization_constant)


def control_6_local_top1_agreement_fraction(patch: PatchInfo, patches_by_rc: dict) -> float:
    agree = 0
    for nr, nc in _grid_neighbors_8(patch.row, patch.col):
        neighbor = patches_by_rc.get((nr, nc))
        _require(neighbor is not None, "control_6 requires all 8 grid neighbors to be present")
        if neighbor.unary_top1 == patch.unary_top1:
            agree += 1
    return agree / 8.0


def control_8_area_matched_control_change(control_change_q: float) -> float:
    return control_change_q


# ---------------------------------------------------------------------------
# Feature row assembly (GT-free)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryCorrectnessRow:
    """One accepted (image_id, window_id, anchor_id, query_rank) row -- see
    [statistics.tasks.query_correctness]."""
    image_id: str
    window_ordinal: int
    anchor_id: str
    query_rank: str  # "q1" or "q2"
    query_class: int
    feature_a: tuple  # control 1,2,3,4,5,6,8 in that order (control_id order, index 1..7)
    intervention_reliability: float  # model-B-only 8th feature


@dataclass(frozen=True)
class PropagationOutcomeRow:
    """One eligible (image_id, window_id, anchor_id) row using q1 only --
    see [statistics.tasks.propagation_outcome]."""
    image_id: str
    window_ordinal: int
    anchor_id: str
    query_class: int  # always q1
    feature_a: tuple
    intervention_reliability: float


def assemble_query_correctness_row(image_id: str, window_ordinal: int, anchor_id: str,
                                    query_rank: str, query_class: int,
                                    control_1: float, control_2: float, control_3: float,
                                    control_4: float, control_5: Optional[float], control_6: float,
                                    control_8: float, intervention_reliability: float) -> QueryCorrectnessRow:
    """Assembles the exact GT-free feature tuple in
    [[statistics.tasks.query_correctness.feature_order]] index order
    (control_id 1,2,3,4,5,6,8 -> Model A's 7 features; index 8 =
    intervention_reliability, Model B only)."""
    _require(query_rank in ("q1", "q2"), "query_rank must be q1 or q2")
    feature_a = (control_1, control_2, control_3, control_4, control_5, control_6, control_8)
    return QueryCorrectnessRow(
        image_id=image_id, window_ordinal=window_ordinal, anchor_id=anchor_id,
        query_rank=query_rank, query_class=query_class, feature_a=feature_a,
        intervention_reliability=intervention_reliability,
    )


def assemble_propagation_outcome_row(image_id: str, window_ordinal: int, anchor_id: str, query_class: int,
                                      control_1: float, control_2: float, control_3: float,
                                      control_4: float, control_5: Optional[float], control_6: float,
                                      control_8: float, intervention_reliability: float) -> PropagationOutcomeRow:
    """Same feature order as query_correctness, fixed at q1 -- see
    [[statistics.tasks.propagation_outcome.feature_order]]."""
    feature_a = (control_1, control_2, control_3, control_4, control_5, control_6, control_8)
    return PropagationOutcomeRow(
        image_id=image_id, window_ordinal=window_ordinal, anchor_id=anchor_id,
        query_class=query_class, feature_a=feature_a, intervention_reliability=intervention_reliability,
    )


def row_is_complete(feature_a: tuple) -> bool:
    """[statistics.missing_value_policy]: a row with any missing/non-finite
    required control value is excluded from all fitting/evaluation. control_5
    may be legitimately None (missing); every other slot must be finite."""
    for i, v in enumerate(feature_a):
        if i == 4:  # control_5 slot, 0-indexed position of control_id 5
            if v is not None and not math.isfinite(v):
                return False
            continue
        if v is None or not math.isfinite(v):
            return False
    return True


# ---------------------------------------------------------------------------
# GT-free scientific digest
# ---------------------------------------------------------------------------

def gt_free_feature_digest(rows: list) -> str:
    """SHA256 over the canonical row order, ASCII-encoded, deterministic
    formatting of every feature field. Rows must already be in canonical
    (image, window, anchor, query) order before calling this."""
    parts = []
    for row in rows:
        fields = [
            row.image_id, str(row.window_ordinal), row.anchor_id,
            getattr(row, "query_rank", "q1"), str(row.query_class),
        ]
        fields.extend("" if v is None else repr(v) for v in row.feature_a)
        fields.append(repr(row.intervention_reliability))
        parts.append("\x1f".join(fields) + "\x1e")
    data = "".join(parts).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
