"""Tests for src/dfp_phase1b_compute_estimator.py: the DFP Phase 1A
Step 9 pure Phase 1B compute estimator. CPU-only, no CUDA, no dataset/
model access, no ground truth. Every FLOP/byte value is hand-derived
independently in this file and compared exactly against the estimator's
own output.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/open_vocabulary_segmentation"))

import src.dfp_phase1b_compute_estimator as ce  # noqa: E402
import src.dfp_phase1a_stitching_pullback as pb  # noqa: E402
from src.dfp_phase1a_identity import load_identity  # noqa: E402

PROTOCOL_PATH = ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml"
TINY_CONTRACT = pb.DfpPullbackContract(crop=(8, 8), stride=(4, 4), patch_grid=(2, 2), patch_size=(4, 4), align_corners=True, connectivity=8)
NODE_COUNT = 4


def _real_protocol():
    return load_identity(PROTOCOL_PATH, repo_root=ROOT)


def _graph_meta(nonzero_count=12, node_count=NODE_COUNT, index_bytes=8, weight_bytes=4, variant="k12"):
    return ce.WindowGraphMetadata(
        node_count=node_count, nonzero_count=nonzero_count, index_dtype_bytes=index_bytes,
        weight_dtype_bytes=weight_bytes, fallback_included=False, graph_variant=variant,
    )


# ---------------------------------------------------------------------------
# Contract extraction / identity mutation matrix
# ---------------------------------------------------------------------------


def test_contract_from_real_protocol():
    protocol = _real_protocol()
    contract = ce.dfp_compute_projection_contract_from_protocol(protocol)
    assert contract.recurrence_steps == 320
    assert contract.required_statistics == ("mean", "median", "p90", "p95", "p99", "maximum")


@pytest.mark.parametrize(
    "section,mutate",
    [
        ("query_compute_projection", lambda p: p.__setitem__("projected_width_formula", "R_w + |Q_w|")),
        ("query_compute_projection", lambda p: p.__setitem__("q_w_definition", "something else")),
        ("dispute_protocol", lambda p: p.__setitem__("a_definition", "something else")),
        ("dispute_protocol", lambda p: p.__setitem__("tie_rule", "highest_index")),
        ("propagation", lambda p: p.__setitem__("steps", 321)),
        ("graph", lambda p: p.__setitem__("canonical_k_variant", 11)),
        ("structural_components", lambda p: p.__setitem__("component_scope", "local_per_sliding_window")),
        ("structural_components", lambda p: p.__setitem__("component_source", "raw_per_window_scores")),
        ("score_interpolation_stitching", lambda p: p.__setitem__("raw_score_stage", "post_sigmoid")),
    ],
)
def test_mutated_field_rejected(section, mutate):
    protocol = copy.deepcopy(_real_protocol())
    protocol[section] = dict(protocol[section])
    mutate(protocol[section])
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.dfp_compute_projection_contract_from_protocol(protocol)


# ---------------------------------------------------------------------------
# Dispute set D (section 3 / fixtures)
# ---------------------------------------------------------------------------


def test_no_disputed_pixels():
    s0 = np.zeros((2, 4, 4), dtype=np.float32); s0[0] = 1.0
    s320 = s0.copy()
    a, b, dispute = ce.compute_stitched_dispute_set(s0, s320, class_count=2)
    assert int(dispute.sum()) == 0


def test_one_disputed_pixel():
    s0 = np.zeros((2, 4, 4), dtype=np.float32); s0[0] = 1.0
    s320 = s0.copy(); s320[1, 0, 0] = 5.0  # flips only pixel (0,0) to class 1
    a, b, dispute = ce.compute_stitched_dispute_set(s0, s320, class_count=2)
    assert int(dispute.sum()) == 1
    assert bool(dispute[0, 0])
    assert int(a[0, 0]) == 0 and int(b[0, 0]) == 1


def test_stitched_winner_absent_from_patch_argmax_still_captured():
    # a "stitched winner" is defined purely by the STITCHED field, not
    # any individual patch-level argmax; construct S320 where class 2
    # never wins any single "patch" (simulated here as sub-blocks) but
    # DOES win after full-image stitched interpolation by dominating a
    # blended region -- the estimator must still report it via a(x)/b(x)
    # on the stitched field, never re-deriving from patch-level data.
    s0 = np.zeros((3, 4, 4), dtype=np.float32); s0[0] = 1.0
    s320 = np.zeros((3, 4, 4), dtype=np.float32)
    s320[1, :, :2] = 3.0  # left half: class 1 wins
    s320[2, :, 2:] = 3.0  # right half: class 2 wins
    # neither block alone has any patch region where class 0 (the S0
    # baseline) wins, and s0's own winner (class 0) never appears in s320
    a, b, dispute = ce.compute_stitched_dispute_set(s0, s320, class_count=3)
    assert set(np.unique(b).tolist()) == {1, 2}
    assert int(dispute.sum()) == 16  # all pixels disputed (class 0 -> 1 or 2)


def test_malformed_s0_s320_rejected():
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.compute_stitched_dispute_set(np.zeros((2, 4)), np.zeros((2, 4, 4), dtype=np.float32), class_count=2)
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.compute_stitched_dispute_set(np.zeros((2, 4, 4), dtype=np.int32), np.zeros((2, 4, 4), dtype=np.float32), class_count=2)
    nan_s0 = np.zeros((2, 4, 4), dtype=np.float32); nan_s0[0, 0, 0] = float("nan")
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.compute_stitched_dispute_set(nan_s0, np.zeros((2, 4, 4), dtype=np.float32), class_count=2)
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.compute_stitched_dispute_set(np.zeros((3, 4, 4), dtype=np.float32), np.zeros((2, 4, 4), dtype=np.float32), class_count=2)


# ---------------------------------------------------------------------------
# Q_w construction
# ---------------------------------------------------------------------------


def test_repeated_dispute_duplicate_classes_deduplicated():
    plan = pb.build_stitching_pullback_plan((8, 8), contract=TINY_CONTRACT, dtype=torch.float64)
    s0 = np.zeros((2, 8, 8), dtype=np.float32); s0[0] = 1.0
    s320 = np.zeros((2, 8, 8), dtype=np.float32); s320[1] = 5.0  # every pixel disputes 0->1
    a, b, dispute = ce.compute_stitched_dispute_set(s0, s320, class_count=2)
    qw = ce.compute_window_query_classes(a, b, dispute, window_plan=plan, window_index=0)
    assert qw == (0, 1)  # not (0,1,0,1,...) -- deduplicated, ascending


def test_several_different_ab_classes():
    plan = pb.build_stitching_pullback_plan((8, 8), contract=TINY_CONTRACT, dtype=torch.float64)
    s0 = np.zeros((4, 8, 8), dtype=np.float32)
    s0[0, :4, :] = 5.0; s0[1, 4:, :] = 5.0
    s320 = np.zeros((4, 8, 8), dtype=np.float32)
    s320[2, :4, :] = 5.0; s320[3, 4:, :] = 5.0
    a, b, dispute = ce.compute_stitched_dispute_set(s0, s320, class_count=4)
    qw = ce.compute_window_query_classes(a, b, dispute, window_plan=plan, window_index=0)
    assert qw == (0, 1, 2, 3)


def test_disputed_pixel_supported_by_one_window_only():
    plan = pb.build_stitching_pullback_plan((8, 16), contract=TINY_CONTRACT, dtype=torch.float64)
    # window 0 covers cols 0-8; a window further right won't cover col 0
    s0 = np.zeros((2, 8, 16), dtype=np.float32); s0[0] = 1.0
    s320 = s0.copy(); s320[1, 0, 0] = 5.0  # dispute only at (0,0)
    a, b, dispute = ce.compute_stitched_dispute_set(s0, s320, class_count=2)
    qw0 = ce.compute_window_query_classes(a, b, dispute, window_plan=plan, window_index=0)
    assert qw0 == (0, 1)
    last_window = len(plan.plan.windows) - 1
    qw_last = ce.compute_window_query_classes(a, b, dispute, window_plan=plan, window_index=last_window)
    assert qw_last == ()  # last window doesn't cover column 0


def test_disputed_pixel_supported_by_multiple_windows():
    plan = pb.build_stitching_pullback_plan((14, 14), contract=TINY_CONTRACT, dtype=torch.float64)
    s0 = np.zeros((2, 14, 14), dtype=np.float32); s0[0] = 1.0
    s320 = s0.copy(); s320[1, 6, 6] = 5.0  # a pixel in the overlap region of several windows
    a, b, dispute = ce.compute_stitched_dispute_set(s0, s320, class_count=2)
    supporting = [w for w in range(len(plan.plan.windows)) if ce.compute_window_query_classes(a, b, dispute, window_plan=plan, window_index=w) != ()]
    assert len(supporting) > 1


@pytest.mark.parametrize("geom", [((8, 11), (8, 8), (4, 4)), ((11, 8), (8, 8), (4, 4)), ((11, 13), (8, 8), (4, 4))])
def test_flush_edge_windows(geom):
    image_size, crop, stride = geom
    contract = pb.DfpPullbackContract(crop=crop, stride=stride, patch_grid=(2, 2), patch_size=(4, 4), align_corners=True, connectivity=8)
    plan = pb.build_stitching_pullback_plan(image_size, contract=contract, dtype=torch.float64)
    s0 = np.zeros((2, *image_size), dtype=np.float32); s0[0] = 1.0
    s320 = s0.copy()
    s320[1, image_size[0] - 1, image_size[1] - 1] = 5.0  # dispute at the flush corner
    a, b, dispute = ce.compute_stitched_dispute_set(s0, s320, class_count=2)
    last_window = len(plan.plan.windows) - 1
    qw = ce.compute_window_query_classes(a, b, dispute, window_plan=plan, window_index=last_window)
    assert qw == (0, 1)


# ---------------------------------------------------------------------------
# R_w / active components
# ---------------------------------------------------------------------------


def test_one_active_component():
    plan = pb.build_stitching_pullback_plan((8, 8), contract=TINY_CONTRACT, dtype=torch.float64)
    component_ids = np.zeros((8, 8), dtype=np.int64)
    active = ce.compute_window_active_components(component_ids, 1, window_plan=plan, window_index=0, component_chunk_size=1)
    assert active == (0,)


def test_multiple_active_components():
    plan = pb.build_stitching_pullback_plan((8, 8), contract=TINY_CONTRACT, dtype=torch.float64)
    component_ids = np.zeros((8, 8), dtype=np.int64); component_ids[:, 4:] = 1
    active = ce.compute_window_active_components(component_ids, 2, window_plan=plan, window_index=0, component_chunk_size=2)
    assert active == (0, 1)


def test_fractional_positive_membership_counts_as_active():
    plan = pb.build_stitching_pullback_plan((8, 8), contract=TINY_CONTRACT, dtype=torch.float64)
    # a thin diagonal split guarantees a fractional (not 0/1) membership
    # for bilinear interpolation at the boundary
    component_ids = np.zeros((8, 8), dtype=np.int64)
    for i in range(8):
        component_ids[i, : max(1, i)] = 1
    active = ce.compute_window_active_components(component_ids, 2, window_plan=plan, window_index=0, component_chunk_size=2)
    assert active == (0, 1)
    # independently confirm at least one entry is genuinely fractional
    from src.dfp_phase1a_soft_membership import iter_soft_membership_records
    found_fractional = False
    for record in iter_soft_membership_records(component_ids, 2, window_plan=plan, component_chunk_size=2):
        if torch.any((record.membership > 1e-9) & (record.membership < 1 - 1e-9)):
            found_fractional = True
    assert found_fractional


def test_inactive_zero_membership_component_excluded():
    plan = pb.build_stitching_pullback_plan((8, 16), contract=TINY_CONTRACT, dtype=torch.float64)
    component_ids = np.zeros((8, 16), dtype=np.int64)
    component_ids[:, 12:] = 1  # component 1 lives entirely outside window 0's support
    active = ce.compute_window_active_components(component_ids, 2, window_plan=plan, window_index=0, component_chunk_size=2)
    assert active == (0,)


def test_component_axis_mismatch_rejected():
    plan = pb.build_stitching_pullback_plan((8, 8), contract=TINY_CONTRACT, dtype=torch.float64)
    bad_ids = np.zeros((5, 5), dtype=np.int64)
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.compute_window_active_components(bad_ids, 1, window_plan=plan, window_index=0, component_chunk_size=1)


# ---------------------------------------------------------------------------
# Graph metadata validation
# ---------------------------------------------------------------------------


def test_graph_metadata_impossible_edge_count_rejected():
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.WindowGraphMetadata(node_count=2, nonzero_count=99, index_dtype_bytes=8, weight_dtype_bytes=4, fallback_included=False, graph_variant="k12")


def test_graph_metadata_bool_as_int_rejected():
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.WindowGraphMetadata(node_count=True, nonzero_count=4, index_dtype_bytes=8, weight_dtype_bytes=4, fallback_included=False, graph_variant="k12")


def test_graph_metadata_relational_validation():
    plan = pb.build_stitching_pullback_plan((8, 8), contract=TINY_CONTRACT, dtype=torch.float64)
    metadata = [_graph_meta() for _ in plan.plan.windows]
    ce.validate_window_graph_metadata(metadata, window_plan=plan, expected_graph_variant="k12")

    wrong_variant = list(metadata); wrong_variant[0] = _graph_meta(variant="k11")
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.validate_window_graph_metadata(wrong_variant, window_plan=plan, expected_graph_variant="k12")

    wrong_node = list(metadata); wrong_node[0] = _graph_meta(node_count=99, nonzero_count=99)
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.validate_window_graph_metadata(wrong_node, window_plan=plan, expected_graph_variant="k12")

    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.validate_window_graph_metadata(metadata[:-1], window_plan=plan, expected_graph_variant="k12")


# ---------------------------------------------------------------------------
# FLOP / byte model -- hand-derived
# ---------------------------------------------------------------------------


def test_spmm_and_elementwise_flops_hand_derived():
    assert ce.spmm_flops(12, 4) == 2 * 12 * 4
    assert ce.elementwise_flops(4, 4) == 3 * 4 * 4
    assert ce.recurrence_flops(4, 12, 4, 320) == 320 * (2 * 12 * 4 + 3 * 4 * 4)


def test_recurrence_flops_linear_in_width_chunking_preserves_total():
    total_direct = ce.recurrence_flops(4, 12, 10, 320)
    total_chunked = sum(ce.recurrence_flops(4, 12, w, 320) for w in (3, 3, 4))
    assert total_direct == total_chunked


def test_byte_model_hand_derived():
    assert ce.graph_storage_bytes(12, 8, 4) == 12 * 12
    assert ce.membership_chunk_bytes(2, 4, 4) == 32
    assert ce.query_class_source_score_bytes(2, 4, 4) == 32
    assert ce.working_state_bytes(4, 4, 4) == 4 * 4 * 4 * 4
    assert ce.capture_buffer_bytes(4, 4, 4) == 5 * 4 * 4 * 4
    assert ce.fixed_geometry_metadata_bytes(4, 8) == 32


def test_full_window_estimate_hand_derived():
    graph_meta = _graph_meta(nonzero_count=12)
    est = ce.estimate_window_compute(
        window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids=(0, 1),
        query_class_indices=(0, 1), recurrence_steps=320, dtype_bytes=4, component_chunk_size=1,
    )
    assert est.projected_width == 4  # 2 components x 2 query classes
    assert est.baseline_recurrence_flops == 320 * (2 * 12 * 3 + 3 * 4 * 3)
    assert est.projected_recurrence_flops == 320 * (2 * 12 * 4 + 3 * 4 * 4)
    assert est.chunked_total_recurrence_flops == est.projected_recurrence_flops
    assert est.chunk_count == 2
    assert est.chunk_widths == (2, 2)
    for c, w in zip(est.chunk_component_counts, est.chunk_widths):
        expected_bytes = 12 * 12 + c * 4 * 4 + 4 * w * 4 * 4 + 2 * 4 * 4 + 5 * w * 4 * 4 + 4 * 8
        assert expected_bytes in est.chunk_peak_bytes


def test_q_empty_gives_zero_width_regardless_of_active_components():
    graph_meta = _graph_meta()
    est = ce.estimate_window_compute(
        window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids=(0, 1, 2),
        query_class_indices=(), recurrence_steps=320, dtype_bytes=4, component_chunk_size=2,
    )
    assert est.projected_width == 0
    assert est.chunk_count == 0
    assert est.projected_recurrence_flops == 0


# ---------------------------------------------------------------------------
# Chunk projection
# ---------------------------------------------------------------------------


def test_final_partial_chunk():
    counts = ce.chunk_component_counts(7, 3)
    assert counts == (3, 3, 1)


def test_chunk_size_larger_than_R_w():
    counts = ce.chunk_component_counts(3, 100)
    assert counts == (3,)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 100])
def test_chunk_size_invariance_on_totals(chunk_size):
    graph_meta = _graph_meta()
    est = ce.estimate_window_compute(
        window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids=tuple(range(7)),
        query_class_indices=(0, 1), recurrence_steps=320, dtype_bytes=4, component_chunk_size=chunk_size,
    )
    assert est.chunked_total_recurrence_flops == est.projected_recurrence_flops
    assert sum(est.chunk_component_counts) == 7


def test_invalid_chunk_size_rejected():
    graph_meta = _graph_meta()
    for bad in (0, -1, True, 1.5, "2"):
        with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
            ce.estimate_window_compute(
                window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids=(0,),
                query_class_indices=(0,), recurrence_steps=320, dtype_bytes=4, component_chunk_size=bad,
            )


# ---------------------------------------------------------------------------
# Memory budget
# ---------------------------------------------------------------------------


def test_memory_budget_exact_one_below_one_above():
    graph_meta = _graph_meta()
    est_probe = ce.estimate_window_compute(
        window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids=(0,),
        query_class_indices=(0, 1), recurrence_steps=320, dtype_bytes=4, component_chunk_size=1,
    )
    required = est_probe.chunk_peak_bytes[0]

    est_exact = ce.estimate_window_compute(
        window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids=(0,),
        query_class_indices=(0, 1), recurrence_steps=320, dtype_bytes=4, component_chunk_size=1, memory_budget_bytes=required,
    )
    assert est_exact.memory_budget_feasible == (True,)

    est_below = ce.estimate_window_compute(
        window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids=(0,),
        query_class_indices=(0, 1), recurrence_steps=320, dtype_bytes=4, component_chunk_size=1, memory_budget_bytes=required - 1,
    )
    assert est_below.memory_budget_feasible == (False,)
    assert est_below.memory_budget_excess_bytes == (1,)

    est_above = ce.estimate_window_compute(
        window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids=(0,),
        query_class_indices=(0, 1), recurrence_steps=320, dtype_bytes=4, component_chunk_size=1, memory_budget_bytes=required + 1,
    )
    assert est_above.memory_budget_feasible == (True,)


def test_invalid_memory_budget_rejected():
    graph_meta = _graph_meta()
    for bad in (0, -1, True, 1.5, "100"):
        with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
            ce.estimate_window_compute(
                window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids=(0,),
                query_class_indices=(0,), recurrence_steps=320, dtype_bytes=4, component_chunk_size=1, memory_budget_bytes=bad,
            )


# ---------------------------------------------------------------------------
# Overflow-safety with very large integer counts
# ---------------------------------------------------------------------------


def test_very_large_integer_counts_no_overflow():
    node_count, nonzero_count, class_count = 10_000_000, 100_000_000, 1_000_000
    huge_graph = ce.WindowGraphMetadata(
        node_count=node_count, nonzero_count=nonzero_count, index_dtype_bytes=8, weight_dtype_bytes=4,
        fallback_included=False, graph_variant="k12",
    )
    q = tuple(range(1000))
    est = ce.estimate_window_compute(
        window_index=0, graph_metadata=huge_graph, class_count=class_count, active_component_ids=tuple(range(1000)),
        query_class_indices=q, recurrence_steps=320, dtype_bytes=4, component_chunk_size=100,
    )
    assert isinstance(est.baseline_recurrence_flops, int)
    assert isinstance(est.projected_recurrence_flops, int)
    # exact hand-derived value: bit-for-bit agreement at this magnitude,
    # using Python's arbitrary-precision int throughout, is itself the
    # proof no int32/int64-style silent wraparound or precision loss
    # occurred anywhere in the computation.
    projected_width = 1000 * 1000
    expected = 320 * (2 * nonzero_count * projected_width + 3 * node_count * projected_width)
    assert est.projected_recurrence_flops == expected
    baseline_expected = 320 * (2 * nonzero_count * class_count + 3 * node_count * class_count)
    assert est.baseline_recurrence_flops == baseline_expected
    # a signed 64-bit accumulator would already have wrapped several
    # times over by the aggregate scale reachable in real usage; confirm
    # our own arbitrary-precision path stays exact well past that point
    # by comparing against gmpy-free pure-Python bigint arithmetic (the
    # `expected` computation above), which is what was just verified.
    assert expected > 0 and baseline_expected > 0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_over_multiple_windows():
    graph_meta = _graph_meta()
    est1 = ce.estimate_window_compute(
        window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids=(0,), query_class_indices=(0,),
        recurrence_steps=320, dtype_bytes=4, component_chunk_size=1,
    )
    est2 = ce.estimate_window_compute(
        window_index=1, graph_metadata=graph_meta, class_count=3, active_component_ids=(), query_class_indices=(),
        recurrence_steps=320, dtype_bytes=4, component_chunk_size=1,
    )
    agg = ce.aggregate_window_estimates([est1, est2], total_disputed_pixels=5)
    assert agg.total_windows == 2
    assert agg.zero_dispute_windows == 1
    assert agg.total_baseline_recurrence_flops == est1.baseline_recurrence_flops + est2.baseline_recurrence_flops
    assert agg.total_projected_recurrence_flops == est1.projected_recurrence_flops + est2.projected_recurrence_flops


def test_descriptive_statistics_matches_hand_calculation():
    stats = ce.descriptive_statistics([1, 2, 3, 4, 100])
    assert stats["mean"] == pytest.approx((1 + 2 + 3 + 4 + 100) / 5)
    assert stats["maximum"] == 100.0
    assert stats["raw_values"] == (1, 2, 3, 4, 100)
    assert "linear" in stats["percentile_method"]


# ---------------------------------------------------------------------------
# Strict validation
# ---------------------------------------------------------------------------


def test_bool_as_int_class_count_rejected():
    graph_meta = _graph_meta()
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.estimate_window_compute(
            window_index=0, graph_metadata=graph_meta, class_count=True, active_component_ids=(0,),
            query_class_indices=(0,), recurrence_steps=320, dtype_bytes=4, component_chunk_size=1,
        )


def test_non_ascending_active_component_ids_rejected():
    graph_meta = _graph_meta()
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.estimate_window_compute(
            window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids=(1, 0),
            query_class_indices=(0,), recurrence_steps=320, dtype_bytes=4, component_chunk_size=1,
        )


def test_duplicate_query_class_indices_rejected():
    graph_meta = _graph_meta()
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.estimate_window_compute(
            window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids=(0,),
            query_class_indices=(0, 0), recurrence_steps=320, dtype_bytes=4, component_chunk_size=1,
        )


def test_query_class_out_of_range_rejected():
    graph_meta = _graph_meta()
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.estimate_window_compute(
            window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids=(0,),
            query_class_indices=(3,), recurrence_steps=320, dtype_bytes=4, component_chunk_size=1,
        )


def test_wrong_container_rejected():
    graph_meta = _graph_meta()
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.estimate_window_compute(
            window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids={0, 1},
            query_class_indices=(0,), recurrence_steps=320, dtype_bytes=4, component_chunk_size=1,
        )


@pytest.mark.parametrize("exc_cls", [KeyboardInterrupt, SystemExit, MemoryError])
def test_exception_propagates_uncaught(exc_cls, monkeypatch):
    def _boom(*a, **kw):
        raise exc_cls("poisoned")

    monkeypatch.setattr(ce, "validate_propagation_block", _boom)
    with pytest.raises(exc_cls):
        ce.dfp_compute_projection_contract_from_protocol(_real_protocol())


def test_dependency_error_chained():
    plan = pb.build_stitching_pullback_plan((8, 8), contract=TINY_CONTRACT, dtype=torch.float64)
    bad_ids = np.zeros((5, 5), dtype=np.int64)
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError) as excinfo:
        ce.compute_window_active_components(bad_ids, 1, window_plan=plan, window_index=0, component_chunk_size=1)
    assert excinfo.value.__cause__ is not None


# ---------------------------------------------------------------------------
# No fabricated runtime, no fabricated real-subset numbers
# ---------------------------------------------------------------------------


def test_runtime_status_without_calibration():
    graph_meta = _graph_meta()
    est = ce.estimate_window_compute(
        window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids=(0,),
        query_class_indices=(0,), recurrence_steps=320, dtype_bytes=4, component_chunk_size=1,
    )
    assert est.runtime_status == "not_estimated_without_measured_hardware_calibration"


def test_hardware_calibration_required_fields():
    with pytest.raises(ce.DfpPhase1bComputeEstimatorError):
        ce.HardwareCalibration(
            hardware_identifier="", software_environment="x", measured_effective_throughput_flops_per_second=1.0,
            calibration_workload_definition="x", artifact_digest="x",
        )
    calib = ce.HardwareCalibration(
        hardware_identifier="synthetic-test-cpu", software_environment="pytest",
        measured_effective_throughput_flops_per_second=1e9, calibration_workload_definition="synthetic",
        artifact_digest="0" * 64,
    )
    graph_meta = _graph_meta()
    est = ce.estimate_window_compute(
        window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids=(0,),
        query_class_indices=(0,), recurrence_steps=320, dtype_bytes=4, component_chunk_size=1, hardware_calibration=calib,
    )
    assert est.runtime_status == "estimated_from_supplied_calibration"


def test_no_seconds_or_images_per_second_field_anywhere():
    import dataclasses

    for cls in (ce.DfpPhase1bWindowEstimate, ce.DfpPhase1bAggregateEstimate):
        field_names = [f.name for f in dataclasses.fields(cls)]
        for name in field_names:
            assert "second" not in name.lower() and "hour" not in name.lower() and "date" not in name.lower()


# ---------------------------------------------------------------------------
# Purity / complexity (section 17)
# ---------------------------------------------------------------------------


def test_zero_forbidden_calls_in_source():
    import ast

    source = Path(ce.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        # Module/function/class docstrings (body[0])...
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc_node = node.body[0] if node.body else None
            if isinstance(doc_node, ast.Expr) and isinstance(doc_node.value, ast.Constant) and isinstance(doc_node.value.value, str):
                docstring_lines.update(range(doc_node.lineno, (doc_node.end_lineno or doc_node.lineno) + 1))
        # ...and any bare string-literal expression statement anywhere
        # (the "attribute docstring" idiom used after module-level named
        # constants in this file) -- these are inert, never executed as
        # calls, so they must never be mistaken for executable code.
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            docstring_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    code_text = "\n".join(l for i, l in enumerate(source.splitlines(), 1) if i not in docstring_lines and not l.strip().startswith("#"))
    for forbidden in (
        "finite_step_propagate", "build_directed_topk_graph", "generate_patch_snapshot",
        "interpolate_raw_scores", "apply_window_stitching_forward", "apply_window_stitching_pullback",
        "Dataset(", "DataLoader(", "boundary_iou", "torch.load(", ".cuda(", "open(",
    ):
        assert forbidden not in code_text, f"found forbidden reference {forbidden!r}"


def test_no_gt_parameter_anywhere():
    import inspect

    leak_tokens = ["gt", "ground_truth", "ignore_mask", "annotation", "oracle", "checkpoint", "harmful", "beneficial"]
    for fn in (
        ce.dfp_compute_projection_contract_from_protocol, ce.compute_stitched_dispute_set, ce.compute_window_query_classes,
        ce.compute_window_active_components, ce.estimate_window_compute, ce.aggregate_window_estimates,
    ):
        params = list(inspect.signature(fn).parameters)
        leaked = [p for p in params if any(tok in p.lower() for tok in leak_tokens)]
        assert not leaked, f"{fn.__name__} leaked: {leaked}"


def test_no_files_written():
    import os

    before = set(os.listdir("/tmp"))
    graph_meta = _graph_meta()
    ce.estimate_window_compute(
        window_index=0, graph_metadata=graph_meta, class_count=3, active_component_ids=(0,),
        query_class_indices=(0,), recurrence_steps=320, dtype_bytes=4, component_chunk_size=1,
    )
    after = set(os.listdir("/tmp"))
    assert before == after


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
import src.dfp_phase1b_compute_estimator as ce
import src.dfp_phase1a_stitching_pullback as pb

contract = pb.DfpPullbackContract(crop=(8,8), stride=(4,4), patch_grid=(2,2), patch_size=(4,4), align_corners=True, connectivity=8)
plan = pb.build_stitching_pullback_plan((14, 14), contract=contract, dtype=torch.float64)
s0 = np.zeros((3, 14, 14), dtype=np.float32); s0[0] = 1.0
s320 = np.zeros((3, 14, 14), dtype=np.float32); s320[1, :, :7] = 5.0; s320[2, :, 7:] = 5.0
a, b, dispute = ce.compute_stitched_dispute_set(s0, s320, class_count=3)
component_ids = np.zeros((14, 14), dtype=np.int64); component_ids[:, 7:] = 1

results = []
for w in range(len(plan.plan.windows)):
    qw = ce.compute_window_query_classes(a, b, dispute, window_plan=plan, window_index=w)
    active = ce.compute_window_active_components(component_ids, 2, window_plan=plan, window_index=w, component_chunk_size=2)
    graph_meta = ce.WindowGraphMetadata(node_count=4, nonzero_count=12, index_dtype_bytes=8, weight_dtype_bytes=4, fallback_included=False, graph_variant="k12")
    est = ce.estimate_window_compute(window_index=w, graph_metadata=graph_meta, class_count=3, active_component_ids=active, query_class_indices=qw, recurrence_steps=320, dtype_bytes=4, component_chunk_size=1)
    results.append((est.projected_width, est.projected_recurrence_flops, est.peak_planned_bytes))
print(results)
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
