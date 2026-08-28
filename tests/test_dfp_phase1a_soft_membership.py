"""Tests for src/dfp_phase1a_soft_membership.py: the DFP Phase 1A
Step 8 thin wrapper over Step 7's already-verified soft-membership
formula. Also carries the specific hand-controlled fixture list and
invariants Step 8's own task text names explicitly (one-component,
two-vertical, two-horizontal, diagonal/irregular, ring-and-center,
disconnected-same-class, entirely-inside-a-patch-footprint,
crosses-one-boundary, crosses-several-components, overlapping windows,
flush-right/bottom/both, non-square, float32/float64) -- none of which
were committed as named tests by Step 7, even though Step 7's formula
implementation already satisfies them. For every fixture, an explicit
A_w is constructed in this file only, via Step 5's raw production
primitives -- never via Step 7's or Step 8's own pullback/membership
helpers -- to serve as the independent reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/open_vocabulary_segmentation"))

import src.dfp_phase1a_soft_membership as sm  # noqa: E402
import src.dfp_phase1a_stitching_pullback as pb  # noqa: E402
from segmentation.evaluation.sliding_window_geometry import SlidingWindowPlan, SpatialSize  # noqa: E402
from models.dinotext.cover_dr.stitching_control import interpolate_raw_scores, build_stitch_weight  # noqa: E402

GRID_HW = (2, 2)
NODE_COUNT = 4


# ---------------------------------------------------------------------------
# Independent explicit-A_w reference (Step 5 primitives only)
# ---------------------------------------------------------------------------


def _build_plan_and_denominator(image_size, crop, stride, dtype=torch.float64):
    plan = SlidingWindowPlan.build(image_size=SpatialSize(*image_size), crop_size=SpatialSize(*crop), stride=SpatialSize(*stride))
    h_img, w_img = image_size
    denominator = torch.zeros((1, 1, h_img, w_img), dtype=dtype)
    for window in plan.windows:
        crop_hw = window.extent.as_tuple()
        weight = build_stitch_weight(crop_hw[0], crop_hw[1], kind="uniform", dtype=dtype)
        rows, cols = window.accumulation_slice
        denominator[:, :, rows, cols] += weight
    return plan, denominator


def _independent_forward(u, *, plan, denominator, window_index, dtype):
    window = plan.windows[window_index]
    crop_hw = window.extent.as_tuple()
    patch_scores = u.reshape(1, -1, 1).to(dtype)
    crop = interpolate_raw_scores(patch_scores, GRID_HW, crop_hw).reshape(crop_hw)
    weight = build_stitch_weight(crop_hw[0], crop_hw[1], kind="uniform", dtype=dtype).reshape(crop_hw)
    rows, cols = window.accumulation_slice
    return crop * weight / denominator[0, 0, rows, cols]


def _independent_dense_A_w(plan, denominator, window_index, dtype=torch.float64):
    window = plan.windows[window_index]
    crop_h, crop_w = window.extent.as_tuple()
    dense = torch.zeros(crop_h * crop_w, NODE_COUNT, dtype=dtype)
    for j in range(NODE_COUNT):
        e_j = torch.zeros(NODE_COUNT, dtype=dtype)
        e_j[j] = 1.0
        dense[:, j] = _independent_forward(e_j, plan=plan, denominator=denominator, window_index=window_index, dtype=dtype).reshape(-1)
    return dense


def _independent_membership(plan, denominator, window_index, component_ids, n_components, dtype=torch.float64):
    """M_reference = (A_w.T @ indicator) / (A_w.T @ ones), built entirely
    from an explicit dense A_w and never via any production membership
    helper."""
    window = plan.windows[window_index]
    rows, cols = window.accumulation_slice
    dense = _independent_dense_A_w(plan, denominator, window_index, dtype=dtype)
    ones = torch.ones(dense.shape[0], dtype=dtype)
    q = dense.T @ ones
    ids_local = torch.from_numpy(component_ids)[rows, cols].reshape(-1)
    rows_out = []
    for r in range(n_components):
        indicator = (ids_local == r).to(dtype)
        numerator = dense.T @ indicator
        rows_out.append(numerator / q)
    return torch.stack(rows_out, dim=0), q  # [R, N], [N]


# ---------------------------------------------------------------------------
# Named hand-controlled fixtures (section 9)
# ---------------------------------------------------------------------------

IMAGE_SIZE = (14, 14)
CROP = (8, 8)
STRIDE = (4, 4)
CONTRACT = pb.DfpPullbackContract(crop=CROP, stride=STRIDE, patch_grid=GRID_HW, patch_size=(4, 4), align_corners=True, connectivity=8)


def _fixture_maps():
    fixtures = {}

    fixtures["one_component_whole_image"] = np.zeros(IMAGE_SIZE, dtype=np.int64)

    two_vert = np.zeros(IMAGE_SIZE, dtype=np.int64)
    two_vert[7:, :] = 1
    fixtures["two_vertical_components"] = two_vert

    two_horiz = np.zeros(IMAGE_SIZE, dtype=np.int64)
    two_horiz[:, 7:] = 1
    fixtures["two_horizontal_components"] = two_horiz

    irregular = np.zeros(IMAGE_SIZE, dtype=np.int64)
    for i in range(IMAGE_SIZE[0]):
        for j in range(IMAGE_SIZE[1]):
            irregular[i, j] = (i + j) % 3
    fixtures["diagonal_irregular_regions"] = irregular

    ring = np.zeros(IMAGE_SIZE, dtype=np.int64)
    ring[3:11, 3:11] = 1
    ring[6, 6] = 0
    fixtures["ring_and_center"] = ring

    disconnected = np.zeros(IMAGE_SIZE, dtype=np.int64)
    disconnected[0:2, 0:2] = 1
    disconnected[12:14, 12:14] = 1  # same unary class (id 1), two disconnected regions
    fixtures["disconnected_components_sharing_one_class"] = disconnected

    # a component that lies entirely inside one 4x4 patch footprint of window 0
    inside_footprint = np.zeros(IMAGE_SIZE, dtype=np.int64)
    inside_footprint[0:4, 0:4] = 1  # exactly window 0's first patch footprint (crop=8, patch=4x4 grid 2x2)
    fixtures["component_entirely_inside_a_patch_footprint"] = inside_footprint

    crosses_one_boundary = np.zeros(IMAGE_SIZE, dtype=np.int64)
    crosses_one_boundary[:, 3:5] = 1  # straddles the patch column boundary at col=4 within window 0
    fixtures["patch_footprint_crosses_one_boundary"] = crosses_one_boundary

    crosses_several = np.zeros(IMAGE_SIZE, dtype=np.int64)
    crosses_several[0:2, 0:2] = 1
    crosses_several[2:4, 2:4] = 2
    fixtures["patch_footprint_crosses_several_components"] = crosses_several

    return fixtures


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("fixture_name", list(_fixture_maps()))
def test_hand_controlled_fixture_matches_independent_reference(fixture_name, dtype):
    comp_map = _fixture_maps()[fixture_name]
    n_components = int(comp_map.max()) + 1
    ref_plan, ref_denom = _build_plan_and_denominator(IMAGE_SIZE, CROP, STRIDE, dtype=dtype)
    prod_plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=dtype)

    for window_index in range(len(ref_plan.windows)):
        M_ref, q_ref = _independent_membership(ref_plan, ref_denom, window_index, comp_map, n_components, dtype=dtype)

        records = list(sm.iter_soft_membership_records(
            comp_map, n_components, window_plan=prod_plan, component_chunk_size=n_components,
            expected_geometry_fingerprint=prod_plan.geometry_fingerprint,
        ))
        record = next(r for r in records if r.window_index == window_index)

        tol = 1e-9 if dtype == torch.float64 else 1e-3
        assert torch.allclose(record.membership, M_ref, atol=tol), f"[{fixture_name}/w{window_index}/{dtype}] mismatch"
        # q agreement (independent vs production plan's own q)
        assert torch.allclose(q_ref, prod_plan.per_window_q[window_index], atol=tol)


# ---------------------------------------------------------------------------
# Overlapping windows, flush-edge, non-square (still part of section 9)
# ---------------------------------------------------------------------------

GEOMETRIES = {
    "overlapping": ((14, 14), (8, 8), (4, 4)),
    "flush_right": ((8, 11), (8, 8), (4, 4)),
    "flush_bottom": ((11, 8), (8, 8), (4, 4)),
    "flush_both": ((11, 13), (8, 8), (4, 4)),
    "non_square": ((10, 16), (8, 8), (4, 4)),
}


@pytest.mark.parametrize("geom_name", list(GEOMETRIES))
def test_geometry_fixture_matches_independent_reference(geom_name):
    image_size, crop, stride = GEOMETRIES[geom_name]
    contract = pb.DfpPullbackContract(crop=crop, stride=stride, patch_grid=GRID_HW, patch_size=(4, 4), align_corners=True, connectivity=8)
    ref_plan, ref_denom = _build_plan_and_denominator(image_size, crop, stride)
    prod_plan = pb.build_stitching_pullback_plan(image_size, contract=contract, dtype=torch.float64)

    comp_map = np.zeros(image_size, dtype=np.int64)
    comp_map[:, image_size[1] // 2:] = 1
    n_components = 2

    for window_index in range(len(ref_plan.windows)):
        M_ref, _ = _independent_membership(ref_plan, ref_denom, window_index, comp_map, n_components)
        records = list(sm.iter_soft_membership_records(
            comp_map, n_components, window_plan=prod_plan, component_chunk_size=n_components,
        ))
        record = next(r for r in records if r.window_index == window_index)
        assert torch.allclose(record.membership, M_ref, atol=1e-9), f"[{geom_name}/w{window_index}]"


# ---------------------------------------------------------------------------
# Required invariants (section 10)
# ---------------------------------------------------------------------------


def test_component_rows_align_with_step6_ids_and_windows_align_with_step5_order():
    comp_map = _fixture_maps()["disconnected_components_sharing_one_class"]
    n_components = 2
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    records = list(sm.iter_soft_membership_records(comp_map, n_components, window_plan=plan, component_chunk_size=1))
    seen_windows = sorted({r.window_index for r in records})
    assert seen_windows == list(range(len(plan.plan.windows)))
    for r in records:
        assert 0 <= r.component_id_start < r.component_id_stop <= n_components


def test_single_component_membership_is_all_ones():
    comp_map = np.zeros(IMAGE_SIZE, dtype=np.int64)  # exactly one component
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    for record in sm.iter_soft_membership_records(comp_map, 1, window_plan=plan, component_chunk_size=1):
        assert torch.allclose(record.membership, torch.ones_like(record.membership), atol=1e-9)


def test_interior_supported_patch_has_membership_one():
    # Bilinear interpolation has COMPACT (not blocky) support: an interior
    # grid node's nonzero-weight region is a diamond confined to the
    # pixels between it and its immediate neighbors, not a naive
    # "nominal patch_size block". A 4x4 grid on an 8x8 crop makes this
    # empirically measurable: grid node (row=1,col=1) (flat index 5) has
    # nonzero weight ONLY within pixel rows 1-4, cols 1-4 of the window
    # (independently confirmed by direct inspection of interpolate_raw_
    # scores's output on a one-hot basis vector). A component covering
    # that exact support region must receive membership exactly 1 for
    # that node, since 100% of q[node]'s mass then lies inside it.
    fine_grid = (4, 4)
    fine_crop = (8, 8)
    fine_contract = pb.DfpPullbackContract(crop=fine_crop, stride=fine_crop, patch_grid=fine_grid, patch_size=(2, 2), align_corners=True, connectivity=8)
    plan = pb.build_stitching_pullback_plan(fine_crop, contract=fine_contract, dtype=torch.float64)

    comp_map = np.zeros(fine_crop, dtype=np.int64)
    comp_map[1:5, 1:5] = 1  # exactly node 5's empirically-measured nonzero support
    for record in sm.iter_soft_membership_records(comp_map, 2, window_plan=plan, component_chunk_size=2):
        assert abs(float(record.membership[1, 5]) - 1.0) < 1e-9
        assert abs(float(record.membership[0, 5]) - 0.0) < 1e-9


def test_boundary_crossing_patch_has_fractional_membership():
    comp_map = _fixture_maps()["patch_footprint_crosses_one_boundary"]
    n_components = 2
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    found_fractional = False
    for record in sm.iter_soft_membership_records(comp_map, n_components, window_plan=plan, component_chunk_size=n_components):
        if torch.any((record.membership > 1e-6) & (record.membership < 1 - 1e-6)):
            found_fractional = True
    assert found_fractional


def test_no_component_row_omitted():
    comp_map = _fixture_maps()["patch_footprint_crosses_several_components"]
    n_components = 3
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    seen_ids = set()
    for record in sm.iter_soft_membership_records(comp_map, n_components, window_plan=plan, component_chunk_size=1):
        seen_ids.update(range(record.component_id_start, record.component_id_stop))
    assert seen_ids == set(range(n_components))


def test_sum_over_components_equals_one_and_numerators_sum_to_q():
    comp_map = _fixture_maps()["diagonal_irregular_regions"]
    n_components = int(comp_map.max()) + 1
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    running = {w: None for w in range(len(plan.plan.windows))}
    for record in sm.iter_soft_membership_records(comp_map, n_components, window_plan=plan, component_chunk_size=2):
        contrib = record.membership.sum(dim=0)
        running[record.window_index] = contrib if running[record.window_index] is None else running[record.window_index] + contrib
    for w, total in running.items():
        assert torch.allclose(total, torch.ones_like(total), atol=1e-9)


def test_source_masses_sum_to_window_patch_copies():
    comp_map = _fixture_maps()["ring_and_center"]
    n_components = 2
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    from src.dfp_phase1a_stitching_pullback import iter_image_soft_memberships

    total = torch.zeros(n_components, dtype=torch.float64)
    for start, chunk in iter_image_soft_memberships(comp_map, n_components, window_plan=plan, component_chunk_size=n_components):
        total += sm.component_source_mass_float64(chunk)
    expected = len(plan.plan.windows) * NODE_COUNT
    assert abs(float(total.sum()) - expected) < 1e-6


def test_component_id_permutation_produces_corresponding_row_permutation():
    comp_map = _fixture_maps()["patch_footprint_crosses_several_components"]
    n_components = 3
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    original = {}
    for record in sm.iter_soft_membership_records(comp_map, n_components, window_plan=plan, component_chunk_size=n_components):
        original[record.window_index] = record.membership.clone()

    # relabel: swap component ids 1 and 2 (a valid re-permutation, since both are
    # still contiguous 0..n-1 after the swap)
    permuted_map = comp_map.copy()
    permuted_map[comp_map == 1] = 2
    permuted_map[comp_map == 2] = 1
    permuted = {}
    for record in sm.iter_soft_membership_records(permuted_map, n_components, window_plan=plan, component_chunk_size=n_components):
        permuted[record.window_index] = record.membership.clone()

    for w in original:
        expected = original[w][[0, 2, 1], :]  # rows 1 and 2 swapped
        assert torch.allclose(permuted[w], expected, atol=1e-9)


def test_component_class_metadata_change_without_changing_omega_does_not_change_M():
    # Step 6's component_class_indices (unary class per component) is not
    # part of this module's input at all -- only component_ids (the
    # spatial partition, Omega) is -- so this invariant is structural:
    # confirm the function signature never accepts a class-metadata
    # argument, and that calling it twice with the same component_ids
    # produces byte-identical results regardless of any external
    # class-label bookkeeping the caller might vary.
    import inspect

    params = list(inspect.signature(sm.iter_soft_membership_records).parameters)
    assert not any("class" in p.lower() for p in params)
    comp_map = _fixture_maps()["two_vertical_components"]
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    r1 = list(sm.iter_soft_membership_records(comp_map, 2, window_plan=plan, component_chunk_size=2))
    r2 = list(sm.iter_soft_membership_records(comp_map, 2, window_plan=plan, component_chunk_size=2))
    for a, b in zip(r1, r2):
        assert torch.equal(a.membership, b.membership)


# ---------------------------------------------------------------------------
# Chunking invariance (section 5, for the record-yielding wrapper)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 100])
def test_record_chunk_size_invariance(chunk_size):
    comp_map = np.random.RandomState(0).randint(0, 7, size=IMAGE_SIZE).astype(np.int64)
    for c in range(7):
        if not np.any(comp_map == c):
            comp_map[0, c] = c
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)

    reference = {}
    for record in sm.iter_soft_membership_records(comp_map, 7, window_plan=plan, component_chunk_size=1):
        reference.setdefault(record.window_index, []).append(record.membership)
    reference_full = {w: torch.cat(v, dim=0) for w, v in reference.items()}

    observed = {}
    for record in sm.iter_soft_membership_records(comp_map, 7, window_plan=plan, component_chunk_size=chunk_size):
        observed.setdefault(record.window_index, []).append(record.membership)
    observed_full = {w: torch.cat(v, dim=0) for w, v in observed.items()}

    for w in reference_full:
        assert torch.allclose(reference_full[w], observed_full[w], atol=1e-9)


def test_invalid_chunk_size_rejected():
    comp_map = np.zeros(IMAGE_SIZE, dtype=np.int64)
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    for bad in (0, -1, True, 1.5, "2"):
        with pytest.raises(sm.DfpPhase1aSoftMembershipError):
            list(sm.iter_soft_membership_records(comp_map, 1, window_plan=plan, component_chunk_size=bad))


# ---------------------------------------------------------------------------
# Geometry fingerprint staleness rejection (section 7)
# ---------------------------------------------------------------------------


def test_fingerprint_mismatch_rejected_before_first_chunk():
    comp_map = np.zeros(IMAGE_SIZE, dtype=np.int64)
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)

    adjoint_calls = {"n": 0}
    real_adjoint = pb._apply_window_bilinear_adjoint

    def _counting(*a, **kw):
        adjoint_calls["n"] += 1
        return real_adjoint(*a, **kw)

    pb._apply_window_bilinear_adjoint = _counting
    try:
        with pytest.raises(sm.DfpPhase1aSoftMembershipError, match="stale"):
            list(sm.iter_soft_membership_records(
                comp_map, 1, window_plan=plan, component_chunk_size=1, expected_geometry_fingerprint="not-the-real-fingerprint",
            ))
    finally:
        pb._apply_window_bilinear_adjoint = real_adjoint
    assert adjoint_calls["n"] == 0


def test_matching_fingerprint_accepted():
    comp_map = np.zeros(IMAGE_SIZE, dtype=np.int64)
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    records = list(sm.iter_soft_membership_records(
        comp_map, 1, window_plan=plan, component_chunk_size=1, expected_geometry_fingerprint=plan.geometry_fingerprint,
    ))
    assert len(records) == len(plan.plan.windows)
    for r in records:
        assert r.geometry_fingerprint == plan.geometry_fingerprint


def test_fingerprint_none_skips_check():
    comp_map = np.zeros(IMAGE_SIZE, dtype=np.int64)
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    records = list(sm.iter_soft_membership_records(comp_map, 1, window_plan=plan, component_chunk_size=1))
    assert len(records) == len(plan.plan.windows)


# ---------------------------------------------------------------------------
# Source mass distinctness (section 8)
# ---------------------------------------------------------------------------


def test_source_mass_accumulates_in_float64_regardless_of_plan_dtype():
    comp_map = np.zeros(IMAGE_SIZE, dtype=np.int64)
    plan32 = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float32)
    from src.dfp_phase1a_stitching_pullback import iter_image_soft_memberships

    for start, chunk in iter_image_soft_memberships(comp_map, 1, window_plan=plan32, component_chunk_size=1):
        m_r = sm.component_source_mass_float64(chunk)
        assert m_r.dtype == torch.float64


def test_source_mass_never_labeled_as_area():
    import inspect

    doc = sm.component_source_mass_float64.__doc__
    assert "pixel" not in doc.lower() or "never" in doc.lower()
    assert "patch-equivalent" not in doc.lower() or "never" in doc.lower()


def test_component_source_mass_rejects_malformed_input():
    with pytest.raises(sm.DfpPhase1aSoftMembershipError):
        sm.component_source_mass_float64({})
    with pytest.raises(sm.DfpPhase1aSoftMembershipError):
        sm.component_source_mass_float64("not a dict")
    with pytest.raises(sm.DfpPhase1aSoftMembershipError):
        sm.component_source_mass_float64({0: torch.zeros(4)})  # wrong rank (1-D)


# ---------------------------------------------------------------------------
# Strict validation (section 11) -- for the wrapper's own new surface
# ---------------------------------------------------------------------------


def test_wrong_container_for_window_plan_rejected():
    comp_map = np.zeros(IMAGE_SIZE, dtype=np.int64)
    with pytest.raises(sm.DfpPhase1aSoftMembershipError):
        list(sm.iter_soft_membership_records(comp_map, 1, window_plan={"not": "a plan"}, component_chunk_size=1))


def test_wrong_type_expected_fingerprint_rejected():
    comp_map = np.zeros(IMAGE_SIZE, dtype=np.int64)
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    with pytest.raises(sm.DfpPhase1aSoftMembershipError):
        list(sm.iter_soft_membership_records(comp_map, 1, window_plan=plan, component_chunk_size=1, expected_geometry_fingerprint=12345))


def test_dependency_errors_translated_with_chaining():
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    bad_ids = np.zeros((5, 5), dtype=np.int64)  # shape mismatch -> Step 7 rejects
    with pytest.raises(sm.DfpPhase1aSoftMembershipError) as excinfo:
        list(sm.iter_soft_membership_records(bad_ids, 1, window_plan=plan, component_chunk_size=1))
    assert excinfo.value.__cause__ is not None


@pytest.mark.parametrize("exc_cls", [KeyboardInterrupt, SystemExit, MemoryError])
def test_exception_propagates_uncaught(exc_cls, monkeypatch):
    comp_map = np.zeros(IMAGE_SIZE, dtype=np.int64)
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)

    def _boom(*a, **kw):
        raise exc_cls("poisoned")

    monkeypatch.setattr(pb, "interpolate_raw_scores", _boom)
    with pytest.raises(exc_cls):
        list(sm.iter_soft_membership_records(comp_map, 1, window_plan=plan, component_chunk_size=1))


# ---------------------------------------------------------------------------
# Immutability and streaming safety (section 12)
# ---------------------------------------------------------------------------


def test_component_ids_and_plan_unchanged_after_streaming():
    comp_map = _fixture_maps()["diagonal_irregular_regions"]
    n_components = int(comp_map.max()) + 1
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    ids_before = comp_map.copy()
    denom_before = plan.denominator.clone()
    q_before = [q.clone() for q in plan.per_window_q]
    for record in sm.iter_soft_membership_records(comp_map, n_components, window_plan=plan, component_chunk_size=2):
        pass
    assert np.array_equal(comp_map, ids_before)
    assert torch.equal(plan.denominator, denom_before)
    assert all(torch.equal(a, b) for a, b in zip(plan.per_window_q, q_before))


def test_records_do_not_alias_one_another():
    comp_map = _fixture_maps()["disconnected_components_sharing_one_class"]
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    records = list(sm.iter_soft_membership_records(comp_map, 2, window_plan=plan, component_chunk_size=1))
    ptrs = [r.membership.data_ptr() for r in records]
    assert len(ptrs) == len(set(ptrs))


def test_mutating_returned_membership_does_not_alter_plan():
    comp_map = np.zeros(IMAGE_SIZE, dtype=np.int64)
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    denom_before = plan.denominator.clone()
    record = next(iter(sm.iter_soft_membership_records(comp_map, 1, window_plan=plan, component_chunk_size=1)))
    record.membership.add_(1000.0)  # in-place tensor mutation; += would instead
    # try to reassign the frozen dataclass field and raise FrozenInstanceError
    assert torch.equal(plan.denominator, denom_before)


def test_no_files_written():
    import os

    before = set(os.listdir("/tmp"))
    comp_map = _fixture_maps()["ring_and_center"]
    plan = pb.build_stitching_pullback_plan(IMAGE_SIZE, contract=CONTRACT, dtype=torch.float64)
    for record in sm.iter_soft_membership_records(comp_map, 2, window_plan=plan, component_chunk_size=2):
        pass
    after = set(os.listdir("/tmp"))
    assert before == after


# ---------------------------------------------------------------------------
# No semantic leakage / no compute expansion (section 13)
# ---------------------------------------------------------------------------


def test_no_gt_parameter_anywhere():
    import inspect

    leak_tokens = ["gt", "ground_truth", "ignore", "annotation", "class_name", "text_embed", "horizon", "harmful", "beneficial", "oracle"]
    for fn in (sm.iter_soft_membership_records, sm.component_source_mass_float64):
        params = list(inspect.signature(fn).parameters)
        leaked = [p for p in params if any(tok in p.lower() for tok in leak_tokens)]
        assert not leaked, f"{fn.__name__} leaked: {leaked}"


def test_zero_forbidden_calls_in_source():
    import ast

    source = Path(sm.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc_node = node.body[0] if node.body else None
            if isinstance(doc_node, ast.Expr) and isinstance(doc_node.value, ast.Constant) and isinstance(doc_node.value.value, str):
                docstring_lines.update(range(doc_node.lineno, (doc_node.end_lineno or doc_node.lineno) + 1))
    code_text = "\n".join(l for i, l in enumerate(source.splitlines(), 1) if i not in docstring_lines and not l.strip().startswith("#"))
    for forbidden in ("finite_step_propagate", "build_directed_topk_graph", "generate_patch_snapshot",
                       "Dataset(", "DataLoader(", "boundary_iou", ".argmax(", "torch.sigmoid(", "checkpoint"):
        assert forbidden not in code_text, f"found forbidden reference {forbidden!r}"


def test_no_membership_math_duplicated_in_this_module():
    # confirm this module never reimplements interpolation/pullback math
    # of its own -- it only imports and calls Step 7's functions.
    source = Path(sm.__file__).read_text(encoding="utf-8")
    for forbidden in ("F.interpolate(", "torch.autograd.grad(", "def apply_window_stitching"):
        assert forbidden not in source


def test_cuda_never_initialized():
    assert torch.cuda.is_initialized() is False


# ---------------------------------------------------------------------------
# Determinism across PYTHONHASHSEED
# ---------------------------------------------------------------------------


def test_deterministic_across_hash_seeds(tmp_path):
    import os
    import subprocess

    script = tmp_path / "run_once.py"
    script.write_text(
        f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
sys.path.insert(0, {str(ROOT / "src/open_vocabulary_segmentation")!r})
import numpy as np
import torch
import src.dfp_phase1a_soft_membership as sm
import src.dfp_phase1a_stitching_pullback as pb

contract = pb.DfpPullbackContract(crop=(8,8), stride=(4,4), patch_grid=(2,2), patch_size=(4,4), align_corners=True, connectivity=8)
plan = pb.build_stitching_pullback_plan((14, 14), contract=contract, dtype=torch.float64)
rng = np.random.RandomState(0)
component_ids = rng.randint(0, 3, size=(14, 14)).astype(np.int64)
for c in range(3):
    if not np.any(component_ids == c):
        component_ids[0, c] = c

parts = [plan.geometry_fingerprint.encode()]
for record in sm.iter_soft_membership_records(component_ids, 3, window_plan=plan, component_chunk_size=2, expected_geometry_fingerprint=plan.geometry_fingerprint):
    parts.append(f"{{record.window_index}}:{{record.component_id_start}}:{{record.component_id_stop}}".encode())
    parts.append(record.membership.numpy().tobytes())
import hashlib
print(hashlib.sha256(b"|".join(parts)).hexdigest())
"""
    )
    outputs = []
    for seed in ("0", "1", "42"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env)
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout.strip())
    assert len(set(outputs)) == 1, f"non-deterministic across PYTHONHASHSEED: {outputs}"
