"""DFP Phase 1A Step 6: global unary spatial-semantic components.

Given one image's canonical stitched unary score field S0 (Step 5's
output, converted by the caller to the numpy ``[class, height, width]``
layout ``src.dfp_phase1a_analysis`` already uses), this module derives
the deterministic lowest-index unary label map and partitions the full
image domain into maximal 8-connected same-class components -- the
global component partition DFP Phase 1A's later soft-membership pullback
consumes.

Why S0, not individual windows
-------------------------------
The frozen protocol identity's ``[structural_components]`` pins
``component_source = "canonical_stitched_unary_(T=0)_field"`` and
``component_scope = "global_across_the_full_stitched_image_never_per_
sliding_window"`` -- components are a property of the final, already-
stitched image, never of any one sliding window. Using Step 5's stitched
S0 (never a raw per-window score) is therefore not a convenience, it is
the frozen contract.

Tie rule
--------
``[structural_components].tie_rule = "lowest_class_index"`` is
implemented by reusing ``src.dfp_phase1a_analysis.derive_horizon_labels``
-- the already-verified, explicit (never incidental-``argmax``) lowest-
index tie-break Step 3 already uses for T=0 and T=320 dispute labels.
This module adds no second tie-break implementation.

Connectivity and area
----------------------
``primary_connectivity = 8`` (the frozen sensitivity variant,
connectivity=4, is a *different*, later diagnostic -- never produced by
this module). Components are labeled per present class via
``scipy.ndimage.label`` with an 8-connected structuring element (the
same primitive ``src.dfp_phase1a_analysis.compute_reversion_coherence``
already uses for connectivity), then remapped from scipy's own
per-class-local numbering to canonical global IDs, ordered by ascending
minimum row-major flat pixel index -- scipy's internal label numbers are
never treated as an authority. Patch-equivalent area follows
``[structural_component_area].formula`` exactly:
``area_patch_equivalent = component_pixel_area / projected_patch_pitch_area``,
with the projected patch pitch reused, not re-derived, from
``src.dfp_phase1a_horizon_stitching``'s own identity-bound stitching
contract (``patch_size``), which is itself relationally bound to the
matched-k11/k12 parent identity's live geometry.

No ground truth
----------------
Neither ``[structural_components]`` nor ``[structural_component_area]``
references ground truth, an ignore index, or any annotation in any form.
This module accepts no such input anywhere in its API -- the component
partition is derived purely from the predicted stitched S0 field.

**No Phase 1A dataset diagnostic has been run by this step.** It only
adds and verifies the capacity to partition one already-stitched image's
unary field into global components -- soft membership, source mass,
component contributions, and the final DFP decision rule are all later,
unimplemented steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from src.dfp_phase1a_analysis import derive_horizon_labels
from src.dfp_phase1a_horizon_stitching import (
    DfpPhase1aHorizonStitchingError,
    dfp_stitching_contract_from_protocol,
)
from src.dfp_phase1a_identity import (
    DfpPhase1aIdentityError,
    validate_structural_component_area_block,
    validate_structural_components_block,
)


class DfpPhase1aGlobalUnaryComponentsError(ValueError):
    """Raised on any DFP Phase 1A global unary component invariant
    violation. Always fail closed: never silently sort, reshape, coerce,
    or repair a malformed input."""


def _require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise DfpPhase1aGlobalUnaryComponentsError(f"{label} must be an exact non-boolean integer")
    return value


def _require_positive_exact_int(value: Any, label: str) -> int:
    value = _require_exact_int(value, label)
    if value <= 0:
        raise DfpPhase1aGlobalUnaryComponentsError(f"{label} must be positive")
    return value


# ---------------------------------------------------------------------------
# Identity-bound component contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DfpComponentContract:
    """Independently validated component-construction parameters
    extracted from the live protocol identity -- never hardcoded module
    constants."""

    connectivity: int
    patch_pitch: tuple[int, int]
    microcomponent_threshold_patch_equivalents: float


def dfp_component_contract_from_protocol(
    protocol_identity: Mapping[str, Any], *, repo_root: Path,
) -> DfpComponentContract:
    """Strictly extract and validate the component-relevant fields of an
    already-loaded protocol identity. Delegates to the single existing
    canonical authorities -- ``src.dfp_phase1a_identity.
    validate_structural_components_block`` (connectivity, tie rule,
    component scope/source), ``validate_structural_component_area_block``
    (area formula, microcomponent threshold), and
    ``src.dfp_phase1a_horizon_stitching.dfp_stitching_contract_from_
    protocol`` (the projected patch pitch, reused rather than
    re-derived) -- rather than performing merely structural checks on a
    caller-provided mapping. Any failure from either delegate --
    including a ``DfpPhase1aHorizonStitchingError`` raised while
    deriving the patch pitch -- is always re-raised as this module's own
    ``DfpPhase1aGlobalUnaryComponentsError`` (chained via ``from exc``),
    so callers only ever need to catch one exception type from this
    function."""
    try:
        structural = protocol_identity["structural_components"]
        area = protocol_identity["structural_component_area"]
        validate_structural_components_block(structural)
        validate_structural_component_area_block(area)
    except DfpPhase1aIdentityError as exc:
        raise DfpPhase1aGlobalUnaryComponentsError(str(exc)) from exc
    except KeyError as exc:
        raise DfpPhase1aGlobalUnaryComponentsError(f"protocol_identity is missing a required section: {exc}") from exc

    connectivity = structural["primary_connectivity"]
    threshold = area["primary_microcomponent_threshold_patch_equivalents"]

    try:
        stitching_contract = dfp_stitching_contract_from_protocol(protocol_identity, repo_root=repo_root)
    except DfpPhase1aHorizonStitchingError as exc:
        raise DfpPhase1aGlobalUnaryComponentsError(str(exc)) from exc
    patch_pitch = stitching_contract.patch_size

    return DfpComponentContract(
        connectivity=connectivity, patch_pitch=patch_pitch, microcomponent_threshold_patch_equivalents=threshold,
    )


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


_RESULT_ARRAY_FIELDS = (
    "unary_labels", "component_ids", "component_class_indices", "component_pixel_areas",
    "component_patch_equivalent_areas", "component_first_flat_indices", "per_class_component_counts",
)


@dataclass(frozen=True)
class GlobalUnaryComponents:
    """One image's global unary component partition. All array fields
    are frozen read-only on construction (mutation-safe): reassigning an
    attribute is blocked by the frozen dataclass, and mutating an
    array's contents in place raises ``ValueError`` (numpy's own
    read-only-buffer contract)."""

    unary_labels: np.ndarray  # [H, W] int64
    component_ids: np.ndarray  # [H, W] int64, values 0..R-1
    component_class_indices: np.ndarray  # [R] int64
    component_pixel_areas: np.ndarray  # [R] int64
    component_patch_equivalent_areas: np.ndarray  # [R] float64
    component_first_flat_indices: np.ndarray  # [R] int64
    component_count: int
    per_class_component_counts: np.ndarray  # [C] int64

    def __post_init__(self) -> None:
        for name in _RESULT_ARRAY_FIELDS:
            getattr(self, name).flags.writeable = False


# ---------------------------------------------------------------------------
# Component construction
# ---------------------------------------------------------------------------


def compute_global_unary_components(
    s0: np.ndarray, *, contract: DfpComponentContract, class_count: int,
) -> GlobalUnaryComponents:
    """Derive the deterministic global unary component partition of one
    image's canonical stitched S0 field.

    Validation (exact rank/dtype/finiteness/class-count agreement, and
    ``contract.connectivity == 8``) happens strictly before any
    connected-component labeling. The lowest-index tie-break is Step 3's
    own ``derive_horizon_labels`` -- not reimplemented here. Components
    are labeled once per class present in the label map (never once per
    window, never per pixel pair), remapped to canonical global IDs
    ordered by ascending first row-major flat pixel index, and packaged
    into a read-only result. No ground truth is read anywhere in this
    function."""
    if not isinstance(contract, DfpComponentContract):
        raise DfpPhase1aGlobalUnaryComponentsError("contract must be a DfpComponentContract")
    if contract.connectivity != 8:
        raise DfpPhase1aGlobalUnaryComponentsError(
            f"contract.connectivity must be exactly 8 (the frozen primary connectivity), observed {contract.connectivity!r}"
        )
    class_count = _require_positive_exact_int(class_count, "class_count")

    if not isinstance(s0, np.ndarray):
        raise DfpPhase1aGlobalUnaryComponentsError("s0 must be a numpy.ndarray")
    if s0.dtype != np.float32 and s0.dtype != np.float64:
        raise DfpPhase1aGlobalUnaryComponentsError(f"s0 dtype must be exactly float32 or float64, observed {s0.dtype}")
    if s0.ndim != 3:
        raise DfpPhase1aGlobalUnaryComponentsError(f"s0 must be exactly 3-D [class, height, width], observed {s0.ndim}-D")
    if s0.shape[0] != class_count:
        raise DfpPhase1aGlobalUnaryComponentsError(f"s0 class dimension {s0.shape[0]} disagrees with class_count {class_count}")
    height, width = s0.shape[1], s0.shape[2]
    if height <= 0 or width <= 0:
        raise DfpPhase1aGlobalUnaryComponentsError(f"s0 spatial dimensions must be positive, observed {(height, width)}")
    if not np.all(np.isfinite(s0)):
        raise DfpPhase1aGlobalUnaryComponentsError("s0 contains a non-finite value (NaN or Infinity)")

    # -- lowest-index tie-break, reused unmodified from Step 3 --------
    unary_labels = derive_horizon_labels({0: s0})[0]
    assert unary_labels.dtype == np.int64 and unary_labels.shape == (height, width)

    # -- per-class 8-connected labeling, remapped to global raw ids --
    structure = ndimage.generate_binary_structure(2, 2)  # rank=2, connectivity=2 -> 8-connected
    raw_component_ids = np.zeros((height, width), dtype=np.int64)
    raw_id_to_class: list[int] = [-1]  # index 0 unused (0 means "unassigned", never present at return time)
    running_offset = 0
    for class_index in np.unique(unary_labels):
        class_index = int(class_index)
        mask = unary_labels == class_index
        local_labels, n_local = ndimage.label(mask, structure=structure)
        if n_local == 0:
            continue  # unreachable: mask is non-empty by construction of np.unique
        raw_component_ids[mask] = local_labels[mask].astype(np.int64) + running_offset
        raw_id_to_class.extend([class_index] * n_local)
        running_offset += n_local

    if np.any(raw_component_ids == 0):
        raise DfpPhase1aGlobalUnaryComponentsError("internal error: one or more pixels received no raw component id")

    # -- deterministic global ID assignment: ascending first flat index --
    flat = raw_component_ids.reshape(-1)
    unique_raw_ids, first_flat_indices, pixel_areas = np.unique(flat, return_index=True, return_counts=True)
    order = np.argsort(first_flat_indices, kind="stable")
    unique_raw_ids = unique_raw_ids[order]
    first_flat_indices = first_flat_indices[order].astype(np.int64)
    pixel_areas = pixel_areas[order].astype(np.int64)

    component_count = int(unique_raw_ids.shape[0])
    remap = np.zeros(running_offset + 1, dtype=np.int64)
    remap[unique_raw_ids] = np.arange(component_count, dtype=np.int64)
    component_ids = remap[raw_component_ids]

    component_class_indices = np.array(
        [raw_id_to_class[int(raw_id)] for raw_id in unique_raw_ids], dtype=np.int64,
    )

    patch_pitch_area = float(contract.patch_pitch[0]) * float(contract.patch_pitch[1])
    component_patch_equivalent_areas = pixel_areas.astype(np.float64) / patch_pitch_area

    per_class_component_counts = np.bincount(component_class_indices, minlength=class_count).astype(np.int64)

    result = GlobalUnaryComponents(
        unary_labels=unary_labels,
        component_ids=component_ids,
        component_class_indices=component_class_indices,
        component_pixel_areas=pixel_areas,
        component_patch_equivalent_areas=component_patch_equivalent_areas,
        component_first_flat_indices=first_flat_indices,
        component_count=component_count,
        per_class_component_counts=per_class_component_counts,
    )
    validate_global_unary_components_partition(result, image_shape=(height, width), class_count=class_count, contract=contract)
    return result


# ---------------------------------------------------------------------------
# Partition self-check (public validator, section 11)
# ---------------------------------------------------------------------------


def validate_global_unary_components_partition(
    result: GlobalUnaryComponents, *, image_shape: tuple[int, int], class_count: int,
    contract: "DfpComponentContract | None" = None,
) -> None:
    """Independently re-check every partition invariant against
    ``result``. Never mutates ``result``. Raises
    ``DfpPhase1aGlobalUnaryComponentsError`` on the first violation
    found."""
    if not isinstance(result, GlobalUnaryComponents):
        raise DfpPhase1aGlobalUnaryComponentsError("result must be a GlobalUnaryComponents")
    height, width = image_shape
    r = result.component_count
    if type(r) is not int or r <= 0:
        raise DfpPhase1aGlobalUnaryComponentsError("component_count must be a positive exact integer")

    for name, expected_shape, expected_dtype in (
        ("unary_labels", (height, width), np.int64),
        ("component_ids", (height, width), np.int64),
        ("component_class_indices", (r,), np.int64),
        ("component_pixel_areas", (r,), np.int64),
        ("component_patch_equivalent_areas", (r,), np.float64),
        ("component_first_flat_indices", (r,), np.int64),
        ("per_class_component_counts", (class_count,), np.int64),
    ):
        arr = getattr(result, name)
        if not isinstance(arr, np.ndarray):
            raise DfpPhase1aGlobalUnaryComponentsError(f"{name} must be a numpy.ndarray")
        if arr.dtype != expected_dtype:
            raise DfpPhase1aGlobalUnaryComponentsError(f"{name} dtype must be exactly {expected_dtype}, observed {arr.dtype}")
        if tuple(arr.shape) != expected_shape:
            raise DfpPhase1aGlobalUnaryComponentsError(f"{name} shape must be exactly {expected_shape}, observed {tuple(arr.shape)}")

    ids = result.component_ids
    if ids.min() != 0 or ids.max() != r - 1:
        raise DfpPhase1aGlobalUnaryComponentsError("component_ids must cover exactly the contiguous range 0..R-1")
    present = np.unique(ids)
    if present.shape[0] != r or not np.array_equal(present, np.arange(r)):
        raise DfpPhase1aGlobalUnaryComponentsError("component_ids does not contain every id in 0..R-1 exactly once as a set")

    # pixel areas reconcile with direct counts from component_ids
    direct_ids, direct_counts = np.unique(ids, return_counts=True)
    if not np.array_equal(direct_ids, np.arange(r)) or not np.array_equal(direct_counts.astype(np.int64), result.component_pixel_areas):
        raise DfpPhase1aGlobalUnaryComponentsError("component_pixel_areas disagrees with a direct recount of component_ids")

    if int(result.component_pixel_areas.sum()) != height * width:
        raise DfpPhase1aGlobalUnaryComponentsError("total component pixel area does not equal H*W")

    # class index for each component matches all corresponding unary pixels
    for component_id in range(r):
        pixel_mask = ids == component_id
        classes_here = np.unique(result.unary_labels[pixel_mask])
        if classes_here.shape[0] != 1 or int(classes_here[0]) != int(result.component_class_indices[component_id]):
            raise DfpPhase1aGlobalUnaryComponentsError(
                f"component {component_id} is not unary-class-homogeneous or disagrees with component_class_indices"
            )

    # first-index keys match direct reconstruction, and ordering is canonical (ascending)
    flat_labels = ids.reshape(-1)
    for component_id in range(r):
        direct_first = int(np.argmax(flat_labels == component_id))
        if direct_first != int(result.component_first_flat_indices[component_id]):
            raise DfpPhase1aGlobalUnaryComponentsError(f"component_first_flat_indices[{component_id}] disagrees with direct reconstruction")
    if not np.all(np.diff(result.component_first_flat_indices) > 0):
        raise DfpPhase1aGlobalUnaryComponentsError("component_first_flat_indices is not strictly ascending by global id (non-canonical ordering)")

    if int(result.per_class_component_counts.sum()) != r:
        raise DfpPhase1aGlobalUnaryComponentsError("per_class_component_counts does not sum to component_count")
    recomputed_per_class = np.bincount(result.component_class_indices, minlength=class_count)
    if not np.array_equal(recomputed_per_class.astype(np.int64), result.per_class_component_counts):
        raise DfpPhase1aGlobalUnaryComponentsError("per_class_component_counts disagrees with a direct recount of component_class_indices")

    # patch-equivalent areas reconcile exactly with pixel areas: the
    # implied pixel-area / patch-equivalent-area pitch must be the SAME
    # constant for every component (internal self-consistency, requires
    # no external knowledge of the pitch); when a contract is supplied,
    # additionally require that constant to equal the contract's own
    # authoritative patch pitch area exactly.
    nonzero = result.component_patch_equivalent_areas > 0
    if np.any(nonzero):
        implied_pitch_area = result.component_pixel_areas[nonzero].astype(np.float64) / result.component_patch_equivalent_areas[nonzero]
        if not np.allclose(implied_pitch_area, implied_pitch_area[0], rtol=1e-12, atol=1e-9):
            raise DfpPhase1aGlobalUnaryComponentsError(
                "component_patch_equivalent_areas does not reconcile with component_pixel_areas under one consistent patch pitch"
            )
        if contract is not None:
            expected_pitch_area = float(contract.patch_pitch[0]) * float(contract.patch_pitch[1])
            if not np.allclose(implied_pitch_area[0], expected_pitch_area, rtol=1e-12, atol=1e-9):
                raise DfpPhase1aGlobalUnaryComponentsError(
                    f"component_patch_equivalent_areas implies a patch pitch area of {implied_pitch_area[0]!r}, "
                    f"disagreeing with contract.patch_pitch area {expected_pitch_area!r}"
                )
    expected_total_patch_equivalent = (height * width) / (
        (float(contract.patch_pitch[0]) * float(contract.patch_pitch[1])) if contract is not None
        else (result.component_pixel_areas[nonzero][0] / result.component_patch_equivalent_areas[nonzero][0] if np.any(nonzero) else 1.0)
    )
    if not np.isclose(float(result.component_patch_equivalent_areas.sum()), expected_total_patch_equivalent, rtol=1e-9, atol=1e-9):
        raise DfpPhase1aGlobalUnaryComponentsError("total patch-equivalent area does not equal H*W / patch_pitch_area")


# ---------------------------------------------------------------------------
# Structural summaries (section 9) -- pure, no source mass / contributions
# ---------------------------------------------------------------------------


def component_summary(result: GlobalUnaryComponents) -> dict[str, Any]:
    """Pure summary reconstructable entirely from ``result``: total
    component count, per-class counts, pixel-area vector,
    patch-equivalent-area vector, and each component's fraction of the
    image's total pixels. Computes nothing about source mass,
    contributions, k-star, phi, or oracle gains."""
    total_pixels = int(result.component_pixel_areas.sum())
    fraction_of_image_pixels = result.component_pixel_areas.astype(np.float64) / float(total_pixels)
    return {
        "component_count": result.component_count,
        "per_class_component_counts": result.per_class_component_counts,
        "component_pixel_areas": result.component_pixel_areas,
        "component_patch_equivalent_areas": result.component_patch_equivalent_areas,
        "fraction_of_image_pixels": fraction_of_image_pixels,
    }


def microcomponent_summary(result: GlobalUnaryComponents, *, contract: DfpComponentContract) -> dict[str, Any]:
    """Count and pixel-mass share of components at or below the
    identity-bound primary microcomponent threshold
    (``contract.microcomponent_threshold_patch_equivalents``, frozen at
    4.0 patch-equivalents). The threshold is never invented here -- it
    is read from ``contract``, which itself came from the protocol
    identity."""
    if not isinstance(contract, DfpComponentContract):
        raise DfpPhase1aGlobalUnaryComponentsError("contract must be a DfpComponentContract")
    threshold = contract.microcomponent_threshold_patch_equivalents
    is_micro = result.component_patch_equivalent_areas <= threshold
    total_pixels = int(result.component_pixel_areas.sum())
    micro_pixels = int(result.component_pixel_areas[is_micro].sum())
    return {
        "threshold_patch_equivalents": threshold,
        "microcomponent_count": int(is_micro.sum()),
        "microcomponent_pixel_mass_share": float(micro_pixels) / float(total_pixels),
    }


__all__ = [
    "DfpPhase1aGlobalUnaryComponentsError",
    "DfpComponentContract",
    "GlobalUnaryComponents",
    "dfp_component_contract_from_protocol",
    "compute_global_unary_components",
    "validate_global_unary_components_partition",
    "component_summary",
    "microcomponent_summary",
]
