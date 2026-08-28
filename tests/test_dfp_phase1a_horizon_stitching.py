"""Tests for src/dfp_phase1a_horizon_stitching.py: the DFP Phase 1A Step 5
multi-horizon stitching module. CPU-only, no CUDA, no dataset/model
access. Independently reconstructs stitched fields in this file (never
importing the new module's own accumulation loop for that part) to prove
it adds no scientific behavior of its own beyond what the existing
geometry/interpolation/weighting primitives already provide.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/open_vocabulary_segmentation"))

import src.dfp_phase1a_horizon_stitching as hs  # noqa: E402
from src.dfp_phase1a_identity import load_identity  # noqa: E402
from models.dinotext.cover_dr.stitching_control import (  # noqa: E402
    StitchAccumulator,
    UNIFORM_SCORE,
    build_stitch_weight,
    interpolate_raw_scores,
)

SpatialSize = hs.SpatialSize
SlidingWindowPlan = hs.SlidingWindowPlan
WindowHorizonCaptures = hs.WindowHorizonCaptures

PROTOCOL_PATH = ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml"
HORIZONS = (0, 16, 32, 64, 320)


def _real_protocol():
    return load_identity(PROTOCOL_PATH, repo_root=ROOT)


def _tiny_contract(**overrides):
    base = dict(horizons=HORIZONS, crop=(8, 8), stride=(4, 4), patch_grid=(2, 2), patch_size=(4, 4), align_corners=True)
    base.update(overrides)
    return hs.DfpStitchingContract(**base)


def _random_captures(plan, *, class_count, node_count, dtype=torch.float32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    records = []
    for i in range(len(plan.windows)):
        caps = {t: torch.randn(node_count, class_count, generator=generator).to(dtype) for t in HORIZONS}
        records.append(WindowHorizonCaptures(window_index=i, captures=caps))
    return records


def _hand_stitch_single_horizon(plan, captures_for_horizon, *, grid_hw, class_count):
    """Independent reconstruction using ONLY the existing production
    primitives (interpolate_raw_scores, build_stitch_weight) plus a
    hand-written accumulation loop -- never importing
    stitch_dfp_phase1a_horizons."""
    h_img, w_img = plan.image_size.as_tuple()
    dtype = next(iter(captures_for_horizon.values())).dtype
    numerator = torch.zeros((1, class_count, h_img, w_img), dtype=dtype)
    denominator = torch.zeros((1, 1, h_img, w_img), dtype=dtype)
    for window in plan.windows:
        crop_hw = window.extent.as_tuple()
        patch_scores = captures_for_horizon[window.index].unsqueeze(0)
        crop = interpolate_raw_scores(patch_scores, grid_hw, crop_hw)
        weight = build_stitch_weight(crop_hw[0], crop_hw[1], kind="uniform", dtype=dtype)
        rows, cols = window.accumulation_slice
        numerator[:, :, rows, cols] += crop * weight
        denominator[:, :, rows, cols] += weight
    return numerator / denominator


# ---------------------------------------------------------------------------
# Contract extraction from the live protocol identity
# ---------------------------------------------------------------------------


def test_contract_from_real_protocol():
    protocol = _real_protocol()
    contract = hs.dfp_stitching_contract_from_protocol(protocol, repo_root=ROOT)
    assert contract.horizons == HORIZONS
    assert contract.crop == (448, 448)
    assert contract.stride == (224, 224)
    assert contract.patch_grid == (32, 32)
    assert contract.patch_size == (14, 14)
    assert contract.align_corners is True


def test_contract_does_not_hardcode_pinned_values():
    import ast

    source = Path(hs.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc_node = node.body[0] if node.body else None
            if isinstance(doc_node, ast.Expr) and isinstance(doc_node.value, ast.Constant) and isinstance(doc_node.value.value, str):
                docstring_lines.update(range(doc_node.lineno, (doc_node.end_lineno or doc_node.lineno) + 1))
    for lineno, line in enumerate(source.splitlines(), start=1):
        if lineno in docstring_lines:
            continue
        if line.strip().startswith("#"):
            continue
        assert "448" not in line, f"found a live crop literal at line {lineno}: {line!r}"
        assert "[0, 16, 32, 64, 320]" not in line and "(0, 16, 32, 64, 320)" not in line, f"found a live horizons literal at line {lineno}: {line!r}"


# ---------------------------------------------------------------------------
# Identity-binding adversarial matrix (section 6)
# ---------------------------------------------------------------------------


def _mutated_protocol(section, mutate):
    protocol = copy.deepcopy(_real_protocol())
    protocol[section] = dict(protocol[section])
    mutate(protocol[section])
    return protocol


@pytest.mark.parametrize(
    "section,mutate",
    [
        ("propagation", lambda p: p.__setitem__("horizons", [16, 32, 64, 320])),
        ("score_interpolation_stitching", lambda p: p.__setitem__("crop", [224, 224])),
        ("score_interpolation_stitching", lambda p: p.__setitem__("stride", [448, 448])),
        ("score_interpolation_stitching", lambda p: p.__setitem__("align_corners", False)),
        ("score_interpolation_stitching", lambda p: p.__setitem__("averaging", "hann")),
        ("score_interpolation_stitching", lambda p: p.__setitem__("hann", True)),
        ("score_interpolation_stitching", lambda p: p.__setitem__("window_enumeration", "row-major but different")),
        ("score_interpolation_stitching", lambda p: p.__setitem__("raw_score_stage", "post_sigmoid")),
        ("score_interpolation_stitching", lambda p: p.__setitem__("majority_vote", True)),
    ],
)
def test_altered_stitching_field_rejected(section, mutate):
    protocol = _mutated_protocol(section, mutate)
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.dfp_stitching_contract_from_protocol(protocol, repo_root=ROOT)


def test_altered_crop_stride_still_internally_consistent_but_disagrees_with_parent_rejected():
    # crop=224 with stride=112 is internally self-consistent (no local
    # structural check would catch it) but disagrees with the live
    # matched_k11_k12 parent identity's geometry.crop -- only the
    # relational cross-check catches this.
    protocol = _mutated_protocol("score_interpolation_stitching", lambda p: (p.__setitem__("crop", [224, 224]), p.__setitem__("stride", [112, 112])))
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.dfp_stitching_contract_from_protocol(protocol, repo_root=ROOT)


def test_exact_type_errors_without_coercion():
    protocol = _mutated_protocol("score_interpolation_stitching", lambda p: p.__setitem__("align_corners", 1))
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.dfp_stitching_contract_from_protocol(protocol, repo_root=ROOT)


def test_missing_key_rejected():
    protocol = copy.deepcopy(_real_protocol())
    protocol["score_interpolation_stitching"] = dict(protocol["score_interpolation_stitching"])
    del protocol["score_interpolation_stitching"]["crop"]
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.dfp_stitching_contract_from_protocol(protocol, repo_root=ROOT)


def test_real_contract_accepted():
    protocol = _real_protocol()
    contract = hs.dfp_stitching_contract_from_protocol(protocol, repo_root=ROOT)
    assert contract.horizons == HORIZONS


# ---------------------------------------------------------------------------
# Positive controls: one window, overlapping windows
# ---------------------------------------------------------------------------


def test_one_window_exact_crop_sized_image():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(8, 8), crop_size=SpatialSize(8, 8), stride=SpatialSize(8, 8))
    assert len(plan.windows) == 1
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=3, node_count=4)
    result = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=3)
    assert set(result) == set(HORIZONS)
    for t in HORIZONS:
        assert result[t].shape == (1, 3, 8, 8)
        assert torch.isfinite(result[t]).all()


def test_overlapping_windows():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(12, 12), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    assert len(plan.windows) > 1
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=2, node_count=4)
    result = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)
    for t in HORIZONS:
        assert torch.isfinite(result[t]).all()


def test_non_square_image():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(9, 15), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=2, node_count=4)
    result = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)
    for t in HORIZONS:
        assert result[t].shape == (1, 2, 9, 15)


# ---------------------------------------------------------------------------
# Flush-edge geometry (section 9)
# ---------------------------------------------------------------------------


def test_flush_right_edge():
    # width 11 with crop 8/stride 4 -> last horizontal window back-shifted
    plan = SlidingWindowPlan.build(image_size=SpatialSize(8, 11), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    last_col_window = plan.window_at(0, plan.grid_cols - 1)
    assert last_col_window.end.col == 11 and last_col_window.is_right
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=2, node_count=4)
    result = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)
    assert result[0].shape[-1] == 11


def test_flush_bottom_edge():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(11, 8), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    last_row_window = plan.window_at(plan.grid_rows - 1, 0)
    assert last_row_window.end.row == 11 and last_row_window.is_bottom
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=2, node_count=4)
    result = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)
    assert result[0].shape[-2] == 11


def test_flush_both_edges():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(11, 13), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    corner_window = plan.window_at(plan.grid_rows - 1, plan.grid_cols - 1)
    assert corner_window.is_right and corner_window.is_bottom
    assert corner_window.end.row == 11 and corner_window.end.col == 13
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=2, node_count=4)
    result = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)
    assert result[0].shape[-2:] == (11, 13)


def test_image_smaller_than_nominal_crop():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(5, 6), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    assert len(plan.windows) == 1
    window = plan.windows[0]
    assert window.extent.as_tuple() == (5, 6)  # smaller than nominal crop, per back-shifted clamp
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=2, node_count=4)
    result = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)
    assert result[0].shape[-2:] == (5, 6)


# ---------------------------------------------------------------------------
# Canonical window order
# ---------------------------------------------------------------------------


def test_canonical_window_order_is_row_major():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(12, 12), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    indices = [w.index for w in plan.windows]
    assert indices == sorted(indices)
    rows_cols = [(w.grid_row, w.grid_col) for w in plan.windows]
    assert rows_cols == sorted(rows_cols)  # row-major


# ---------------------------------------------------------------------------
# T0 / T320 endpoint equivalence against the existing production
# StitchAccumulator(UNIFORM_SCORE) class (section 8)
# ---------------------------------------------------------------------------


def _existing_uniform_score_sigmoided(plan, captures_for_horizon, *, grid_hw, class_count, dtype=torch.float32):
    h_img, w_img = plan.image_size.as_tuple()
    acc = StitchAccumulator(UNIFORM_SCORE, class_count=class_count, image_hw=(h_img, w_img), expected_window_count=len(plan.windows), device="cpu", dtype=dtype)
    for window in plan.windows:
        crop_hw = window.extent.as_tuple()
        patch_scores = captures_for_horizon[window.index].unsqueeze(0)
        crop = interpolate_raw_scores(patch_scores, grid_hw, crop_hw)
        weight = build_stitch_weight(crop_hw[0], crop_hw[1], kind="uniform", dtype=dtype)
        acc.add_window(window, crop, weight=weight)
    return acc.finalize()  # sigmoid applied, existing production behavior


@pytest.mark.parametrize("horizon", [0, 320])
def test_endpoint_equivalence_t0_t320(horizon):
    plan = SlidingWindowPlan.build(image_size=SpatialSize(12, 12), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=3, node_count=4, seed=42)
    mine = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=3)[horizon]

    captures_for_horizon = {r.window_index: r.captures[horizon] for r in captures}
    existing_sigmoided = _existing_uniform_score_sigmoided(plan, captures_for_horizon, grid_hw=contract.patch_grid, class_count=3)

    # my raw pre-sigmoid stitched field, sigmoided, must exactly reproduce
    # the EXISTING production UNIFORM_SCORE accumulator's finalized
    # (sigmoid-after-stitch) output: same interpolation, same uniform
    # weight, same accumulation order and arithmetic -- only the sigmoid
    # placement differs (never applied here, applied once there).
    assert torch.equal(torch.sigmoid(mine), existing_sigmoided)


# ---------------------------------------------------------------------------
# Independent T16/T32/T64 reconstruction (section 9), never importing
# stitch_dfp_phase1a_horizons for the reconstruction itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [16, 32, 64])
def test_independent_intermediate_horizon_reconstruction(horizon):
    plan = SlidingWindowPlan.build(image_size=SpatialSize(11, 13), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=4, node_count=4, seed=7)
    mine = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=4)[horizon]

    captures_for_horizon = {r.window_index: r.captures[horizon] for r in captures}
    reference = _hand_stitch_single_horizon(plan, captures_for_horizon, grid_hw=contract.patch_grid, class_count=4)
    assert torch.equal(mine, reference)


def test_float64_supported():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(9, 9), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=2, node_count=4, dtype=torch.float64, seed=3)
    result = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)
    assert result[0].dtype == torch.float64
    captures_for_horizon = {r.window_index: r.captures[64] for r in captures}
    reference = _hand_stitch_single_horizon(plan, captures_for_horizon, grid_hw=contract.patch_grid, class_count=2)
    assert torch.equal(result[64], reference)


# ---------------------------------------------------------------------------
# Denominator and geometry invariants (section 10)
# ---------------------------------------------------------------------------


def test_denominator_is_horizon_independent_and_shared():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(12, 12), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=2, node_count=4)

    denominator_calls = {"n": 0}
    import src.dfp_phase1a_horizon_stitching as _hs_module

    real_build_weight = _hs_module.build_stitch_weight

    def _counting_build_stitch_weight(*a, **kw):
        denominator_calls["n"] += 1
        return real_build_weight(*a, **kw)

    _hs_module.build_stitch_weight = _counting_build_stitch_weight
    try:
        hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)
    finally:
        _hs_module.build_stitch_weight = real_build_weight
    # build_stitch_weight is called once per window (denominator/weight
    # contribution), never once per (window, horizon) pair.
    assert denominator_calls["n"] == len(plan.windows)


def test_denominator_strictly_positive_everywhere():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(11, 13), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=2, node_count=4)
    result = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)
    # independently derive the overlap-count map and compare its zero
    # structure against coverage: every pixel must be covered.
    coverage = plan.build_coverage_map()
    for row in coverage:
        assert all(c > 0 for c in row)
    for t in HORIZONS:
        assert torch.isfinite(result[t]).all()


def test_uniform_weight_equals_overlap_count():
    # For uniform stitching, weight-per-pixel is always 1, so the
    # denominator must equal the plan's own independently-derived
    # coverage-count map exactly (an integer -- verified via
    # torch.equal after casting).
    plan = SlidingWindowPlan.build(image_size=SpatialSize(12, 12), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=1, node_count=4)
    result = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=1)
    coverage = torch.tensor(plan.build_coverage_map(), dtype=torch.float32)

    # Reconstruct the denominator independently (never importing the
    # module's internal denominator variable) via the same primitives.
    h_img, w_img = plan.image_size.as_tuple()
    denominator = torch.zeros((1, 1, h_img, w_img), dtype=torch.float32)
    for window in plan.windows:
        crop_hw = window.extent.as_tuple()
        weight = build_stitch_weight(crop_hw[0], crop_hw[1], kind="uniform", dtype=torch.float32)
        rows, cols = window.accumulation_slice
        denominator[:, :, rows, cols] += weight
    assert torch.equal(denominator[0, 0], coverage)


def test_changing_one_horizon_does_not_alter_another():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(8, 8), crop_size=SpatialSize(8, 8), stride=SpatialSize(8, 8))
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=2, node_count=4)
    result_a = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)

    mutated = [WindowHorizonCaptures(window_index=r.window_index, captures=dict(r.captures)) for r in captures]
    mutated[0].captures[16] = mutated[0].captures[16] + 1000.0
    result_b = hs.stitch_dfp_phase1a_horizons(plan, mutated, contract=contract, class_count=2)

    assert not torch.equal(result_a[16], result_b[16])
    for t in (0, 32, 64, 320):
        assert torch.equal(result_a[t], result_b[t])


def test_changing_one_window_affects_only_its_support():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(12, 12), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=2, node_count=4)
    result_a = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)

    mutated = [WindowHorizonCaptures(window_index=r.window_index, captures=dict(r.captures)) for r in captures]
    target_window = plan.windows[0]
    mutated[0].captures[0] = mutated[0].captures[0] + 500.0
    result_b = hs.stitch_dfp_phase1a_horizons(plan, mutated, contract=contract, class_count=2)

    rows, cols = target_window.accumulation_slice
    outside = torch.ones_like(result_a[0], dtype=torch.bool)
    outside[:, :, rows, cols] = False
    assert torch.equal(result_a[0][outside], result_b[0][outside])
    assert not torch.equal(result_a[0][:, :, rows, cols], result_b[0][:, :, rows, cols])


def test_original_image_dimensions_restored_exactly():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(17, 23), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=2, node_count=4)
    result = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)
    for t in HORIZONS:
        assert result[t].shape[-2:] == (17, 23)


# ---------------------------------------------------------------------------
# Strict input validation / rejection (section 7)
# ---------------------------------------------------------------------------


def _base_plan_and_captures():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(12, 12), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=2, node_count=4)
    return plan, contract, captures


def test_missing_window_rejected():
    plan, contract, captures = _base_plan_and_captures()
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.stitch_dfp_phase1a_horizons(plan, captures[:-1], contract=contract, class_count=2)


def test_duplicate_window_index_rejected():
    plan, contract, captures = _base_plan_and_captures()
    duplicated = list(captures)
    duplicated[-1] = WindowHorizonCaptures(window_index=duplicated[-2].window_index, captures=duplicated[-1].captures)
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.stitch_dfp_phase1a_horizons(plan, duplicated, contract=contract, class_count=2)


def test_reordered_window_rejected():
    plan, contract, captures = _base_plan_and_captures()
    reordered = list(captures)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.stitch_dfp_phase1a_horizons(plan, reordered, contract=contract, class_count=2)


def test_extra_window_rejected():
    plan, contract, captures = _base_plan_and_captures()
    extra = list(captures) + [WindowHorizonCaptures(window_index=len(captures), captures=dict(captures[0].captures))]
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.stitch_dfp_phase1a_horizons(plan, extra, contract=contract, class_count=2)


def test_missing_horizon_in_one_window_rejected():
    plan, contract, captures = _base_plan_and_captures()
    broken = list(captures)
    caps = dict(broken[0].captures)
    del caps[16]
    broken[0] = WindowHorizonCaptures(window_index=0, captures=caps)
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.stitch_dfp_phase1a_horizons(plan, broken, contract=contract, class_count=2)


def test_extra_horizon_in_one_window_rejected():
    plan, contract, captures = _base_plan_and_captures()
    broken = list(captures)
    caps = dict(broken[0].captures)
    caps[8] = caps[16]
    broken[0] = WindowHorizonCaptures(window_index=0, captures=caps)
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.stitch_dfp_phase1a_horizons(plan, broken, contract=contract, class_count=2)


def test_class_count_mismatch_rejected():
    plan, contract, captures = _base_plan_and_captures()
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=3)


def test_node_count_mismatch_rejected():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(12, 12), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=2, node_count=9)  # wrong node count (expected 2*2=4)
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)


def test_tensor_rank_mismatch_rejected():
    plan, contract, captures = _base_plan_and_captures()
    broken = list(captures)
    caps = dict(broken[0].captures)
    caps[0] = caps[0].unsqueeze(0)  # now 3-D
    broken[0] = WindowHorizonCaptures(window_index=0, captures=caps)
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.stitch_dfp_phase1a_horizons(plan, broken, contract=contract, class_count=2)


def test_mixed_dtype_rejected():
    plan, contract, captures = _base_plan_and_captures()
    broken = list(captures)
    caps = dict(broken[0].captures)
    caps[0] = caps[0].to(torch.float64)
    broken[0] = WindowHorizonCaptures(window_index=0, captures=caps)
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.stitch_dfp_phase1a_horizons(plan, broken, contract=contract, class_count=2)


def test_non_finite_value_rejected():
    plan, contract, captures = _base_plan_and_captures()
    broken = list(captures)
    caps = dict(broken[0].captures)
    tensor = caps[0].clone()
    tensor[0, 0] = float("nan")
    caps[0] = tensor
    broken[0] = WindowHorizonCaptures(window_index=0, captures=caps)
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.stitch_dfp_phase1a_horizons(plan, broken, contract=contract, class_count=2)


def test_integer_tensor_rejected():
    plan, contract, captures = _base_plan_and_captures()
    broken = list(captures)
    caps = dict(broken[0].captures)
    caps[0] = torch.randint(0, 5, caps[0].shape)
    broken[0] = WindowHorizonCaptures(window_index=0, captures=caps)
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.stitch_dfp_phase1a_horizons(plan, broken, contract=contract, class_count=2)


def test_wrong_container_for_window_captures_rejected():
    plan, contract, captures = _base_plan_and_captures()
    with pytest.raises(hs.DfpPhase1aHorizonStitchingError):
        hs.stitch_dfp_phase1a_horizons(plan, {i: r for i, r in enumerate(captures)}, contract=contract, class_count=2)


# ---------------------------------------------------------------------------
# Aliasing / mutation safety (section 11)
# ---------------------------------------------------------------------------


def test_inputs_not_mutated():
    plan, contract, captures = _base_plan_and_captures()
    before = [{t: v.clone() for t, v in r.captures.items()} for r in captures]
    hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)
    for record, snapshot in zip(captures, before):
        for t in HORIZONS:
            assert torch.equal(record.captures[t], snapshot[t])


def test_output_does_not_alias_input():
    plan, contract, captures = _base_plan_and_captures()
    result = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)
    for t in HORIZONS:
        for record in captures:
            assert result[t].data_ptr() != record.captures[t].data_ptr()


def test_output_horizons_do_not_alias_each_other():
    plan, contract, captures = _base_plan_and_captures()
    result = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)
    ptrs = [tensor.data_ptr() for tensor in result.values()]
    assert len(ptrs) == len(set(ptrs))


def test_mutating_s16_does_not_alter_others():
    plan, contract, captures = _base_plan_and_captures()
    result = hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)
    before = {t: v.clone() for t, v in result.items() if t != 16}
    result[16] += 12345.0
    for t, snapshot in before.items():
        assert torch.equal(result[t], snapshot)


def test_output_does_not_require_grad():
    plan, contract, captures = _base_plan_and_captures()
    grad_captures = list(captures)
    caps = {t: v.clone().requires_grad_(True) for t, v in grad_captures[0].captures.items()}
    grad_captures[0] = WindowHorizonCaptures(window_index=0, captures=caps)
    result = hs.stitch_dfp_phase1a_horizons(plan, grad_captures, contract=contract, class_count=2)
    for t in HORIZONS:
        assert not result[t].requires_grad


# ---------------------------------------------------------------------------
# Shared-execution instrumentation (sections 5 and 14)
# ---------------------------------------------------------------------------


def test_shared_execution_call_counts():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(12, 12), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    n_windows = len(plan.windows)
    contract = _tiny_contract()
    captures = _random_captures(plan, class_count=2, node_count=4)

    import src.dfp_phase1a_horizon_stitching as _hs_module

    interpolate_calls = {"n": 0}
    weight_calls = {"n": 0}
    real_interpolate = _hs_module.interpolate_raw_scores
    real_weight = _hs_module.build_stitch_weight

    def _counting_interpolate(*a, **kw):
        interpolate_calls["n"] += 1
        return real_interpolate(*a, **kw)

    def _counting_weight(*a, **kw):
        weight_calls["n"] += 1
        return real_weight(*a, **kw)

    _hs_module.interpolate_raw_scores = _counting_interpolate
    _hs_module.build_stitch_weight = _counting_weight
    try:
        hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)
    finally:
        _hs_module.interpolate_raw_scores = real_interpolate
        _hs_module.build_stitch_weight = real_weight

    assert weight_calls["n"] == n_windows
    assert interpolate_calls["n"] == len(HORIZONS) * n_windows


def test_zero_propagation_graph_model_calls():
    import ast

    source = Path(hs.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc_node = node.body[0] if node.body else None
            if isinstance(doc_node, ast.Expr) and isinstance(doc_node.value, ast.Constant) and isinstance(doc_node.value.value, str):
                docstring_lines.update(range(doc_node.lineno, (doc_node.end_lineno or doc_node.lineno) + 1))

    code_lines = [
        line for lineno, line in enumerate(source.splitlines(), start=1)
        if lineno not in docstring_lines and not line.strip().startswith("#")
    ]
    code_text = "\n".join(code_lines)
    for forbidden in (
        "finite_step_propagate", "build_directed_topk_graph", "generate_patch_snapshot",
        "masks_from_patch_scores", ".model.", ".argmax(", "torch.sigmoid(", "F.sigmoid(",
    ):
        assert forbidden not in code_text, f"found forbidden reference {forbidden!r} in {hs.__file__}"


# ---------------------------------------------------------------------------
# Failure/exception behavior (section 13)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc_cls", [KeyboardInterrupt, SystemExit, MemoryError])
def test_exception_propagates_uncaught(exc_cls, monkeypatch):
    plan, contract, captures = _base_plan_and_captures()

    def _boom(*a, **kw):
        raise exc_cls("poisoned")

    import src.dfp_phase1a_horizon_stitching as _hs_module

    monkeypatch.setattr(_hs_module, "interpolate_raw_scores", _boom)
    with pytest.raises(exc_cls):
        hs.stitch_dfp_phase1a_horizons(plan, captures, contract=contract, class_count=2)


def test_cuda_never_initialized():
    assert torch.cuda.is_initialized() is False
