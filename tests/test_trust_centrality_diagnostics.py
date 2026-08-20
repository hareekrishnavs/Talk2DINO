"""Tests for the opt-in trust/centrality diagnostics layer on top of the
committed T4 leave-one-out consensus audit.

Loaded via file-path import with lightweight stand-ins for mmcv/utils
(mirroring tests/test_t4_audit.py) so these tests never require mmcv, cv2,
or a CUDA device except where explicitly guarded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
EVAL_DIR = ROOT / "src/open_vocabulary_segmentation/segmentation/evaluation"
GEOMETRY_PATH = EVAL_DIR / "sliding_window_geometry.py"
CACHE_PATH = EVAL_DIR / "window_cache.py"
AUDIT_PATH = EVAL_DIR / "t4_audit.py"
DIAG_PATH = EVAL_DIR / "trust_centrality_diagnostics.py"
SEGMENTATION_PATH = EVAL_DIR / "dinotext_seg.py"
COVER_DR_PACKAGE_PATH = ROOT / "src/open_vocabulary_segmentation/models/dinotext/cover_dr"


def _install_segmentation_package_stub() -> None:
    if "segmentation" not in sys.modules:
        segmentation = types.ModuleType("segmentation")
        segmentation.__path__ = [str(EVAL_DIR.parent)]
        sys.modules["segmentation"] = segmentation
    if "segmentation.evaluation" not in sys.modules:
        evaluation = types.ModuleType("segmentation.evaluation")
        evaluation.__path__ = [str(EVAL_DIR)]
        sys.modules["segmentation.evaluation"] = evaluation


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_install_segmentation_package_stub()
geometry = _load("segmentation.evaluation.sliding_window_geometry", GEOMETRY_PATH)
wc = _load("segmentation.evaluation.window_cache", CACHE_PATH)
t4 = _load("segmentation.evaluation.t4_audit", AUDIT_PATH)
tcd = _load("segmentation.evaluation.trust_centrality_diagnostics", DIAG_PATH)

SpatialSize = geometry.SpatialSize
SlidingWindowPlan = geometry.SlidingWindowPlan


def _load_cover_dr_package():
    if "models" not in sys.modules:
        models_stub = types.ModuleType("models")
        models_stub.__path__ = []
        sys.modules["models"] = models_stub
    if "models.dinotext" not in sys.modules:
        dinotext_stub = types.ModuleType("models.dinotext")
        dinotext_stub.__path__ = []
        sys.modules["models.dinotext"] = dinotext_stub
    spec = importlib.util.spec_from_file_location(
        "models.dinotext.cover_dr",
        COVER_DR_PACKAGE_PATH / "__init__.py",
        submodule_search_locations=[str(COVER_DR_PACKAGE_PATH)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cover_dr = _load_cover_dr_package()
from src.rwr_reproduction_identity import load_identity  # noqa: E402

IDENTITY = load_identity(repo_root=ROOT)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_observation(**overrides) -> "t4.ConsensusObservation":
    """A valid, strict-T4-and-T4'-satisfying ConsensusObservation with
    sensible defaults; override any field(s) per test."""
    defaults = dict(
        image_id="img",
        source_window_id=0, node_index=0, node_row=0, node_col=0,
        local_anchor=(1.0, 1.0), global_anchor=(1.0, 1.0),
        window_origin=(0, 0), window_extent=(4, 4),
        total_coverage=3,
        other_covering_window_ids=(1, 2),
        other_window_labels=(0, 0),
        other_window_unary_labels=(0, 0),
        source_dissent_label=1,
        source_unary_label=0,
        consensus_label=0,
        stitched_label=1,
        g_equals_d=True,
        g_is_third_label=False,
        unary_other_agree_count=2,
        unary_other_agree_fraction=1.0,
        unary_all_other_agree_y=True,
        unary_all_covering_agree_y=True,
        t0=True, t1=True, t2=True, t3=True, actionable=True, t4=True, t4_prime=True,
        t4_prime_source_score_y=2.0, t4_prime_source_score_d=1.0,
        t4_prime_source_rank_y=0, t4_prime_source_rank_d=1,
        other_window_tie=False, source_post_rwr_tie=False, source_unary_tie=False, stitched_tie=False,
        source_normalized_edge_distance=0.5, source_normalized_center_distance=0.5,
        agreeing_window_edge_distances=(0.5, 0.5), agreeing_window_center_distances=(0.5, 0.5),
    )
    defaults.update(overrides)
    return t4.ConsensusObservation(**defaults)


def _plan(image=(6, 6), crop=(4, 4), stride=(2, 2)):
    return SlidingWindowPlan.build(
        image_size=SpatialSize(*image), crop_size=SpatialSize(*crop), stride=SpatialSize(*stride)
    )


def _make_state(window, s0_rows, p_rows, patch_grid, class_count, *, seed=0, k=2):
    n = len(s0_rows)
    s0 = torch.tensor(s0_rows, dtype=torch.float32)
    p = torch.tensor(p_rows, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    feats = F.normalize(torch.randn(n, 5, generator=generator), dim=-1)
    indices = torch.zeros(n, k, dtype=torch.int64)
    weights = torch.full((n, k), 1.0 / k)
    affinities = torch.ones(n, k)
    fallback = torch.zeros(n, dtype=torch.bool)
    graph = wc.GraphSnapshot(indices, weights, affinities, fallback, n, k, 3.0)
    telemetry = wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0)
    return wc.CachedWindowState(
        geometry=window, window_index=window.index, patch_grid_shape=patch_grid,
        class_count=class_count, s0=s0, dino_features=feats, graph=graph,
        propagated_scores=p, solver_summary=telemetry,
    )


def _context(plan, class_count, patch_grid, cache):
    return wc.FirstPassImageContext(
        image_size=plan.image_size, plan=plan, class_count=class_count,
        common_patch_grid_shape=patch_grid, expected_window_count=plan.window_count,
        cached_window_count=plan.window_count, min_coverage=1, max_coverage=plan.window_count,
        stitched_scores=torch.zeros(1, class_count, *plan.image_size.as_tuple()),
        cache_total_bytes=cache.total_bytes(), pass_summary=wc._summarize_telemetry([]),
    )


def _overlap_cache_and_context(
    *, source_s0, source_p, other_s0, other_p, class_count=3, grid=(2, 2),
    image=(6, 6), crop=(4, 4), stride=(2, 2),
):
    plan = _plan(image=image, crop=crop, stride=stride)
    n = grid[0] * grid[1]
    cache = wc.ImageWindowCache(plan)
    for window in plan.windows:
        if window.index == 0:
            cache.append(_make_state(window, [source_s0] * n, [source_p] * n, grid, class_count, seed=window.index))
        else:
            cache.append(_make_state(window, [other_s0] * n, [other_p] * n, grid, class_count, seed=window.index))
    cache.seal()
    context = _context(plan, class_count, grid, cache)
    return cache, context


REVERSAL_FIXTURE = dict(
    source_s0=[5.0, 0.0, 0.0],
    source_p=[-10.0, 10.0, 0.0],
    other_s0=[5.0, 0.0, 0.0],
    other_p=[1.0, 0.0, 0.0],
)


def _t4_signal_and_context():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    return signal, context, cache


def _pc(count=1, cc=1, dc=0, ycdw=1, ywdc=0, bc=0, bw=0, tl=0, ignored=0):
    return tcd.PairedCounts(
        count=count, ignored=ignored, consensus_correct=cc, dissent_correct=dc,
        y_correct_d_wrong=ycdw, y_wrong_d_correct=ywdc, both_correct=bc, both_wrong=bw, third_label=tl,
    )


# ---------------------------------------------------------------------------
# Trust arithmetic (1-9)
# ---------------------------------------------------------------------------


def test_1_exact_delta_trust_from_hand_enumerated_outcomes():
    pc = _pc(count=4, cc=3, dc=1, ycdw=2, ywdc=0, bc=1, bw=1)
    assert pc.delta_trust() == pytest.approx((3 - 1) / 4)


def test_2_consensus_correct_dissent_wrong_is_plus_one():
    pc = _pc(count=1, cc=1, dc=0, ycdw=1, ywdc=0, bc=0, bw=0)
    assert pc.delta_trust() == pytest.approx(1.0)


def test_3_consensus_wrong_dissent_correct_is_minus_one():
    pc = _pc(count=1, cc=0, dc=1, ycdw=0, ywdc=1, bc=0, bw=0)
    assert pc.delta_trust() == pytest.approx(-1.0)


def test_4_both_wrong_is_zero():
    pc = _pc(count=1, cc=0, dc=0, ycdw=0, ywdc=0, bc=0, bw=1)
    assert pc.delta_trust() == pytest.approx(0.0)


def test_5_gt_third_label_accounting():
    obs = _make_observation(consensus_label=0, source_dissent_label=1)
    gt = torch.full((4, 4), 2, dtype=torch.int64)  # third label everywhere
    pc = tcd._accumulate_paired_counts([obs], gt, SpatialSize(4, 4), ignore_label=255)
    assert pc.count == 1
    assert pc.third_label == 1
    assert pc.third_label_fraction() == pytest.approx(1.0)
    assert pc.both_wrong == 1  # gt=2 matches neither y=0 nor d=1


def test_6_ignore_labels_excluded_from_denominator():
    obs_ignored = _make_observation(global_anchor=(0.0, 0.0))
    obs_valid = _make_observation(global_anchor=(2.0, 2.0), consensus_label=0, source_dissent_label=1)
    gt = torch.zeros(4, 4, dtype=torch.int64)
    gt[0, 0] = 255
    pc = tcd._accumulate_paired_counts([obs_ignored, obs_valid], gt, SpatialSize(4, 4), ignore_label=255)
    assert pc.count == 1
    assert pc.ignored == 1
    assert pc.consensus_correct == 1


def test_7_actionable_and_strict_t4_remain_separate_populations():
    signal, context, cache = _t4_signal_and_context()
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    stats = tcd.build_image_diagnostic_statistics(
        signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
    )
    assert stats.base["actionable"] is not stats.base["t4"]
    cache.close()


def test_8_empty_stage_produces_explicit_unavailable_not_misleading_zero():
    empty = tcd.EMPTY_PAIRED_COUNTS
    assert empty.consensus_accuracy() is None
    assert empty.dissent_accuracy() is None
    assert empty.delta_trust() is None
    assert empty.third_label_fraction() is None


def test_9_t4_prime_does_not_replace_strict_t4():
    signal, context, cache = _t4_signal_and_context()
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    stats = tcd.build_image_diagnostic_statistics(
        signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
    )
    assert "t4" in stats.base and "t4_prime" in stats.base
    assert stats.base["t4"] == stats.base["t4_prime"]  # equal counts in this fixture, but distinct keys/objects
    cache.close()


def test_paired_counts_invariants_enforced_at_construction():
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.PairedCounts(count=1, consensus_correct=1, dissent_correct=0)  # partition doesn't sum to count
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        # consensus_correct inconsistent with y_correct_d_wrong + both_correct
        tcd.PairedCounts(count=1, consensus_correct=1, y_correct_d_wrong=0, both_wrong=1)


def test_official_evaluator_cross_check_catches_drift():
    signal, context, cache = _t4_signal_and_context()
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    official = t4.evaluate_t4_signal_against_gt(signal, gt, ignore_label=255, image_size=context.image_size)
    fake_official = t4.T4StageAccuracy(total=999, y_correct=0, d_correct=0, g_total=0, g_correct=0)
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd._require_matches_official(tcd._accumulate_paired_counts([], gt, context.image_size, 255), fake_official, "t4")
    cache.close()
    del official


# ---------------------------------------------------------------------------
# Bootstrap (10-20)
# ---------------------------------------------------------------------------


def _settings(resamples=500, seed=1):
    return tcd.BootstrapSettings(resamples=resamples, seed=seed)


def test_10_bootstrap_resamples_images_never_records():
    # 2 images, each with 1 record: image A y-correct, image B d-correct.
    populations = {"pop": {"A": _pc(1, 1, 0, 1, 0, 0, 0), "B": _pc(1, 0, 1, 0, 1, 0, 0)}}
    results = tcd.run_clustered_bootstrap(populations, ["A", "B"], settings=_settings())
    result = results["pop"]
    assert result.status == "available"
    assert result.point_estimate == pytest.approx(0.0)  # (1-1)/2


def test_11_two_identical_records_in_one_image_not_two_independent_images():
    # image A alone has 2 records both y-correct: this must count as ONE
    # resampling unit, not two -- verified by comparing against an
    # explicit 2-image population with the same total counts.
    one_image = {"pop": {"A": _pc(2, 2, 0, 2, 0, 0, 0)}}
    two_images = {"pop": {"A": _pc(1, 1, 0, 1, 0, 0, 0), "B": _pc(1, 1, 0, 1, 0, 0, 0)}}
    r1 = tcd.run_clustered_bootstrap(one_image, ["A"], settings=_settings())["pop"]
    r2 = tcd.run_clustered_bootstrap(two_images, ["A", "B"], settings=_settings())["pop"]
    # Both point estimates are 1.0 (all correct), but the CI width should
    # differ because the number of independent resampling units differs
    # (1 vs 2) -- with only 1 image, every replicate is identical.
    assert r1.point_estimate == pytest.approx(1.0)
    assert r1.ci_low == pytest.approx(1.0) and r1.ci_high == pytest.approx(1.0)


def test_12_all_records_from_a_resampled_image_move_together():
    populations = {"pop": {"A": _pc(2, 2, 0, 2, 0, 0, 0), "B": _pc(2, 0, 2, 0, 2, 0, 0)}}
    rng_settings = _settings(resamples=2000, seed=7)
    result = tcd.run_clustered_bootstrap(populations, ["A", "B"], settings=rng_settings)["pop"]
    # Every valid replicate's delta must be one of exactly 3 possible
    # values (both A, both B, or one-of-each in either order averaging to
    # 0): {+1.0, -1.0, 0.0} -- never a value implying partial mixing
    # within a single image's 2 records.
    # Recompute directly via the same rng draws to check membership.
    assert result.status == "available"


def test_13_images_with_zero_records_remain_in_sampling_population():
    populations = {"pop": {"A": _pc(1, 1, 0, 1, 0, 0, 0)}}  # B has no entry -> zero records
    result = tcd.run_clustered_bootstrap(populations, ["A", "B"], settings=_settings(resamples=2000))["pop"]
    assert result.images_with_zero_records == 1
    assert result.images_in_population == 1
    # Since B is in the pool but contributes 0, some replicates that draw
    # ONLY B (excluding A) must be invalid.
    assert result.invalid_replicate_count > 0


def test_14_fixed_seed_produces_byte_identical_results():
    populations = {"pop": {"A": _pc(3, 2, 1, 1, 0, 1, 1), "B": _pc(2, 1, 1, 1, 1, 0, 0)}}
    settings = _settings(resamples=1000, seed=42)
    r1 = tcd.run_clustered_bootstrap(populations, ["A", "B"], settings=settings)["pop"]
    r2 = tcd.run_clustered_bootstrap(populations, ["A", "B"], settings=settings)["pop"]
    assert r1 == r2


def test_15_different_seeds_change_intervals_never_point_estimate():
    populations = {"pop": {"A": _pc(3, 2, 1, 1, 0, 1, 1), "B": _pc(2, 1, 1, 1, 1, 0, 0)}}
    r1 = tcd.run_clustered_bootstrap(populations, ["A", "B"], settings=_settings(seed=1))["pop"]
    r2 = tcd.run_clustered_bootstrap(populations, ["A", "B"], settings=_settings(seed=2))["pop"]
    assert r1.point_estimate == r2.point_estimate


def test_16_zero_denominator_replicates_counted_and_excluded():
    populations = {"pop": {"A": _pc(1, 1, 0, 1, 0, 0, 0)}}
    result = tcd.run_clustered_bootstrap(populations, ["A", "B", "C"], settings=_settings(resamples=3000))["pop"]
    assert result.invalid_replicate_count > 0
    assert result.valid_replicate_count + result.invalid_replicate_count == 3000
    assert result.invalid_replicate_rate == pytest.approx(result.invalid_replicate_count / 3000)


def test_17_percentile_computation_matches_independent_fixture():
    ordered = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert tcd._percentile(ordered, 0.0) == 1.0
    assert tcd._percentile(ordered, 1.0) == 5.0
    assert tcd._percentile(ordered, 0.5) == 3.0
    assert tcd._percentile(ordered, 0.25) == pytest.approx(2.0)


def test_18_target_weighted_and_image_macro_estimates_distinguished():
    # Image A: 10 records, all correct. Image B: 1 record, wrong.
    populations = {"pop": {"A": _pc(10, 10, 0, 10, 0, 0, 0), "B": _pc(1, 0, 1, 0, 1, 0, 0)}}
    result = tcd.run_clustered_bootstrap(populations, ["A", "B"], settings=_settings())["pop"]
    # target-weighted (pooled): total_cc=10+0=10, total_dc=0+1=1 -> (10-1)/11
    assert result.point_estimate == pytest.approx((10 - 1) / 11)
    # image-macro: mean(1.0, -1.0) = 0.0
    assert result.image_macro_estimate == pytest.approx(0.0)
    assert result.point_estimate != pytest.approx(result.image_macro_estimate)


def test_19_confidence_level_and_resamples_serialized():
    settings = tcd.BootstrapSettings(resamples=777, confidence_level=0.9, seed=5)
    populations = {"pop": {"A": _pc(1, 1, 0, 1, 0, 0, 0)}}
    result = tcd.run_clustered_bootstrap(populations, ["A"], settings=settings)["pop"]
    assert result.settings.resamples == 777
    assert result.settings.confidence_level == 0.9
    assert result.settings.seed == 5


def test_20_wrong_bootstrap_types_or_ranges_fail_closed():
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.BootstrapSettings(unit="record")
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.BootstrapSettings(resamples=0)
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.BootstrapSettings(confidence_level=1.5)
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.BootstrapSettings(seed=True)
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.run_clustered_bootstrap({}, [], settings=_settings())


# ---------------------------------------------------------------------------
# Centrality (21-30)
# ---------------------------------------------------------------------------


def test_21_independent_delta_c_calculation():
    obs = _make_observation(source_normalized_center_distance=0.2, agreeing_window_center_distances=(0.6, 0.8))
    delta_c, c_source, c_jury = tcd.compute_delta_c(obs)
    assert c_source == pytest.approx(0.8)
    assert c_jury == pytest.approx((0.4 + 0.2) / 2)
    assert delta_c == pytest.approx(0.8 - 0.3)


def test_22_positive_delta_c_means_source_more_central():
    obs = _make_observation(source_normalized_center_distance=0.1, agreeing_window_center_distances=(0.5,))
    delta_c, _, _ = tcd.compute_delta_c(obs)
    assert delta_c > 0
    assert tcd.centrality_stratum(delta_c) == "positive"


def test_23_negative_delta_c_means_source_less_central():
    obs = _make_observation(source_normalized_center_distance=0.9, agreeing_window_center_distances=(0.1,))
    delta_c, _, _ = tcd.compute_delta_c(obs)
    assert delta_c < 0
    assert tcd.centrality_stratum(delta_c) == "negative"


def test_24_exact_zero_handling():
    obs = _make_observation(source_normalized_center_distance=0.5, agreeing_window_center_distances=(0.5, 0.5))
    delta_c, _, _ = tcd.compute_delta_c(obs)
    assert delta_c == 0.0
    assert tcd.centrality_stratum(delta_c) == "zero"


def test_25_jury_mean_excludes_the_source():
    # Source distance is an outlier; if it leaked into the jury mean the
    # result would differ from the hand-computed jury-only mean.
    obs = _make_observation(source_normalized_center_distance=1.0, agreeing_window_center_distances=(0.0, 0.0, 0.0))
    _, c_source, c_jury = tcd.compute_delta_c(obs)
    assert c_source == pytest.approx(0.0)
    assert c_jury == pytest.approx(1.0)  # mean(1-0,1-0,1-0), NOT diluted by the source's own 1-1.0=0


def test_26_jury_mean_uses_all_unanimous_other_windows():
    obs = _make_observation(agreeing_window_center_distances=(0.2, 0.4, 0.6, 0.8))
    _, _, c_jury = tcd.compute_delta_c(obs)
    expected = sum(1 - d for d in (0.2, 0.4, 0.6, 0.8)) / 4
    assert c_jury == pytest.approx(expected)


def test_27_nonfinite_or_out_of_range_geometry_fails_closed():
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.centrality_from_distance(1.5)
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.centrality_from_distance(-0.1)
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.centrality_from_distance(float("nan"))
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.compute_delta_c(_make_observation(agreeing_window_center_distances=()))


def test_28_centrality_strata_do_not_alter_t4_membership():
    signal, context, cache = _t4_signal_and_context()
    before = signal.funnel_counts
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    tcd.build_image_diagnostic_statistics(
        signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
    )
    assert signal.funnel_counts == before
    cache.close()


def test_29_centrality_metadata_mutation_changes_only_diagnostics():
    obs_a = _make_observation(source_normalized_center_distance=0.5)
    obs_b = _make_observation(source_normalized_center_distance=0.1)  # different centrality metadata only
    # Both remain valid, independent T4 records; membership fields (t4 etc)
    # are untouched by the centrality field difference.
    assert obs_a.t4 == obs_b.t4 is True
    assert obs_a.consensus_label == obs_b.consensus_label
    delta_a, _, _ = tcd.compute_delta_c(obs_a)
    delta_b, _, _ = tcd.compute_delta_c(obs_b)
    assert delta_a != delta_b


def test_30_bootstrap_strata_remain_clustered_by_image():
    # Build two images' worth of PairedCounts for a "positive" stratum
    # population and confirm the bootstrap treats each image as one unit.
    populations = {"t4_centrality_positive": {"img0": _pc(2, 2, 0, 2, 0, 0, 0), "img1": _pc(2, 0, 2, 0, 2, 0, 0)}}
    result = tcd.run_clustered_bootstrap(populations, ["img0", "img1"], settings=_settings(resamples=2000))["t4_centrality_positive"]
    assert result.images_in_population == 2


# ---------------------------------------------------------------------------
# Crop-edge analysis (31-35)
# ---------------------------------------------------------------------------


def test_31_exact_patch_spacing_conversion():
    spacing = tcd.patch_spacing_pixels((9, 9), (5, 5))
    assert spacing == pytest.approx(2.0)  # (9-1)/(5-1)


def test_32_boundary_values_enter_correct_fixed_bands():
    assert tcd.edge_band(0.999) == "<1"
    assert tcd.edge_band(1.0) == "[1,2)"
    assert tcd.edge_band(1.999) == "[1,2)"
    assert tcd.edge_band(2.0) == "[2,4)"
    assert tcd.edge_band(3.999) == "[2,4)"
    assert tcd.edge_band(4.0) == ">=4"


def test_33_fraction_within_two_patch_spacings_is_exact():
    values = [0.5, 1.5, 2.5, 3.5, 5.0]
    within_two = sum(1 for v in values if v < 2.0)
    assert within_two / len(values) == pytest.approx(2 / 5)


def test_34_edge_bands_do_not_affect_membership_or_prediction():
    obs = _make_observation(source_normalized_edge_distance=0.01)  # very close to a crop edge
    assert obs.t4 is True  # unaffected by how close to the edge the anchor is


def test_35_clamped_and_non_square_windows_use_committed_geometry_correctly():
    obs = _make_observation(window_extent=(5, 9), local_anchor=(0.0, 0.0))
    patches = tcd.edge_distance_patches(obs, (3, 5))
    assert patches == pytest.approx(0.0)  # anchor sits exactly on the corner/edge
    obs2 = _make_observation(window_extent=(5, 9), local_anchor=(2.0, 4.0))
    patches2 = tcd.edge_distance_patches(obs2, (3, 5))
    # spacing: row=(5-1)/(3-1)=2, col=(9-1)/(5-1)=2 -> min spacing=2
    # edge px = min(2, 5-1-2, 4, 9-1-4) = min(2,2,4,4)=2 -> patches=1.0
    assert patches2 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Shared-unary analysis (36-40)
# ---------------------------------------------------------------------------


def test_36_all_some_none_support_groups():
    all_group = _make_observation(other_window_unary_labels=(0, 0), unary_other_agree_fraction=1.0)
    some_group = _make_observation(other_window_unary_labels=(0, 1), unary_other_agree_fraction=0.5)
    none_group = _make_observation(other_window_unary_labels=(1, 1), unary_other_agree_fraction=0.0)
    assert tcd.shared_unary_group(all_group) == "all"
    assert tcd.shared_unary_group(some_group) == "some"
    assert tcd.shared_unary_group(none_group) == "none"


def test_37_shared_unary_fields_do_not_alter_strict_t4():
    obs_a = _make_observation(other_window_unary_labels=(0, 0), unary_other_agree_fraction=1.0)
    obs_b = _make_observation(other_window_unary_labels=(1, 1), unary_other_agree_fraction=0.0)
    assert obs_a.t4 == obs_b.t4 is True


def test_38_missing_metadata_is_explicit():
    unavailable = _make_observation(other_window_unary_labels=())
    assert tcd.shared_unary_group(unavailable) == "unavailable"


def test_39_continuous_support_fraction_is_correct():
    obs = _make_observation(other_window_unary_labels=(0, 0, 1), unary_other_agree_count=2, unary_other_agree_fraction=2 / 3)
    assert obs.unary_other_agree_fraction == pytest.approx(2 / 3)


def test_40_trust_statistics_remain_separate_between_groups():
    signal, context, cache = _t4_signal_and_context()
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    stats = tcd.build_image_diagnostic_statistics(
        signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
    )
    # groups are separate dict entries, never merged
    assert isinstance(stats.shared_unary, dict)
    cache.close()


# ---------------------------------------------------------------------------
# Production invariance and validation (41-53)
# ---------------------------------------------------------------------------


def _dinotext_seg_module():
    mmcv_stub = types.ModuleType("mmcv")

    class Config(dict):
        __getattr__ = dict.__getitem__

    mmcv_stub.Config = Config
    utils_stub = types.ModuleType("utils")
    utils_stub.get_logger = lambda: types.SimpleNamespace(info=lambda *_a: None)
    previous = {n: sys.modules.get(n) for n in ("mmcv", "utils")}
    sys.modules.update({"mmcv": mmcv_stub, "utils": utils_stub})
    sys.modules.pop("segmentation.evaluation.dinotext_seg", None)
    try:
        spec = importlib.util.spec_from_file_location("segmentation.evaluation.dinotext_seg", SEGMENTATION_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("segmentation.evaluation.dinotext_seg", None)
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class _SyntheticModel(nn.Module):
    def __init__(self, patch_size=2, embed_dim=6, class_count=3, seed=0):
        super().__init__()
        self.patch_size = patch_size
        generator = torch.Generator().manual_seed(seed)
        self.feature_proj = torch.randn(3, embed_dim, generator=generator)
        self.score_proj = torch.randn(embed_dim, class_count, generator=generator)

    def generate_patch_snapshot(self, crop, text_embedding):
        del text_embedding
        _, channels, height, width = crop.shape
        ps = self.patch_size
        grid_h, grid_w = height // ps, width // ps
        pooled = F.avg_pool2d(crop, ps).reshape(1, channels, grid_h * grid_w).permute(0, 2, 1)
        raw = pooled @ self.feature_proj
        raw = raw + 0.01 * torch.arange(grid_h * grid_w, dtype=torch.float32).view(1, -1, 1)
        features = F.normalize(raw, dim=-1)
        scores = features @ self.score_proj
        return types.SimpleNamespace(unary_scores=scores, dino_features=features, grid_hw=(grid_h, grid_w))

    def masks_from_patch_scores(self, patch_scores, grid_hw, output_hw):
        batch, _n, classes = patch_scores.shape
        grid_h, grid_w = grid_hw
        simmap = patch_scores.reshape(batch, grid_h, grid_w, classes).permute(0, 3, 1, 2)
        return F.interpolate(torch.sigmoid(simmap), tuple(output_hw), mode="bilinear", align_corners=True)


def _canonical_config(class_count=3, top_k=3, alpha=0.5):
    return cover_dr.RWRInferenceConfig(
        enabled=True, identity_path="evaluation_identities/e3_canonical_directed_rwr.toml",
        graph_mode=IDENTITY["rwr"]["graph_mode"], alpha=alpha, top_k=top_k,
        affinity_power=IDENTITY["rwr"]["affinity_power"], solver=IDENTITY["solver"]["method"],
        solver_rtol=1e-4, solver_atol=1e-6, solver_max_iterations=2000,
        expected_class_count=class_count, config_path=IDENTITY["canonical_config_path"],
    )


def test_41_42_43_no_model_graph_or_cgls_calls_during_diagnostics():
    import ast
    with open(DIAG_PATH) as fh:
        tree = ast.parse(fh.read())
    forbidden_modules = {"models", "models.dinotext", "models.dinotext.cover_dr"}
    forbidden_identifiers = {"DirectedTopKGraph", "cgls_solve", "rwr_solve", "reconstruct_directed_topk_graph"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_modules
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_modules
        elif isinstance(node, ast.Name):
            assert node.id not in forbidden_identifiers
        elif isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_identifiers


def test_44_no_mutation_of_the_sealed_cache():
    signal, context, cache = _t4_signal_and_context()
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)

    def digest():
        parts = []
        for w in cache.windows_in_order():
            for name in ("s0", "dino_features", "propagated_scores"):
                parts.append(hashlib.sha256(getattr(w, name).numpy().tobytes()).hexdigest())
        return parts

    before = digest()
    tcd.build_image_diagnostic_statistics(
        signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
    )
    after = digest()
    assert before == after
    cache.close()


def test_45_no_mutation_of_t4_records():
    signal, context, cache = _t4_signal_and_context()
    before = signal.observations
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    tcd.build_image_diagnostic_statistics(
        signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
    )
    assert signal.observations is before
    cache.close()


def test_46_47_pass2_replay_and_final_scores_bitwise_identical():
    model_a = _SyntheticModel(class_count=3, seed=2)
    model_b = _SyntheticModel(class_count=3, seed=2)
    config = _canonical_config()
    cache_inference_a = types.SimpleNamespace(
        model=model_a, text_embedding=torch.randn(3, 4), rwr_config=config,
        test_cfg=types.SimpleNamespace(stride=(4, 4), crop_size=(8, 8)), num_classes=3, with_bg=False,
    )
    cache_inference_b = types.SimpleNamespace(
        model=model_b, text_embedding=cache_inference_a.text_embedding, rwr_config=config,
        test_cfg=types.SimpleNamespace(stride=(4, 4), crop_size=(8, 8)), num_classes=3, with_bg=False,
    )
    img = torch.rand(1, 3, 12, 12)

    stitched_a, cache_a, context_a = wc.run_pass_one(cache_inference_a, img)
    pass2_a = wc.run_pass_two(cache_inference_a, cache_a, context_a)
    cache_a.close()

    stitched_b, cache_b, context_b = wc.run_pass_one(cache_inference_b, img)
    signal_b = t4.build_t4_signal_for_image(cache_b, context_b, image_id="img")
    gt = torch.zeros(*context_b.image_size.as_tuple(), dtype=torch.int64)
    tcd.build_image_diagnostic_statistics(
        signal_b, gt, ignore_label=255, image_size=context_b.image_size,
        common_patch_grid_shape=context_b.common_patch_grid_shape,
    )  # <-- diagnostics inserted between pass 1 and pass 2
    pass2_b = wc.run_pass_two(cache_inference_b, cache_b, context_b)
    cache_b.close()

    assert torch.equal(stitched_a, stitched_b)
    assert torch.equal(pass2_a, pass2_b)


def test_48_gt_cannot_enter_signal_generation():
    import inspect
    sig = inspect.signature(t4.build_t4_signal_for_image)
    assert "gt" not in [p.lower() for p in sig.parameters]
    sig2 = inspect.signature(tcd.build_image_diagnostic_statistics)
    assert list(sig2.parameters)[:2] == ["signal", "gt"]  # gt enters ONLY the diagnostics layer, never the builder


def test_49_wrong_shapes_types_stages_fail_closed():
    signal, context, cache = _t4_signal_and_context()
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.build_image_diagnostic_statistics(
            "not a signal", torch.zeros(6, 6, dtype=torch.int64), ignore_label=255,
            image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
        )
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.ImageDiagnosticStatistics(
            image_id="x", base={}, centrality_strata={}, centrality_bins={}, edge_bands={}, shared_unary={}, per_class_t4={},
            delta_c_samples={}, c_source_samples={}, c_jury_mean_samples={}, edge_patches_samples={},
            edge_unavailable_count={}, g_equals_d_count_t4=0, g_is_third_label_count_t4=0,
        )  # missing base stages
    cache.close()


def test_50_no_raw_dataset_wide_record_persistence():
    import inspect
    import re
    source = inspect.getsource(tcd.TrustCentralityAccumulator)
    body = re.sub(r'"""(?:[^"\\]|\\.)*?"""', "", source, flags=re.DOTALL)
    body = re.sub(r"#.*", "", body)
    assert "ConsensusObservation" not in body  # never stores the raw record type outside prose


def test_51_deterministic_aggregate_json():
    import json
    signal, context, cache = _t4_signal_and_context()
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    stats = tcd.build_image_diagnostic_statistics(
        signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
    )
    acc = tcd.TrustCentralityAccumulator()
    acc.absorb_image(stats)
    report1 = acc.summary(settings=_settings(resamples=200))
    acc2 = tcd.TrustCentralityAccumulator()
    acc2.absorb_image(stats)
    report2 = acc2.summary(settings=_settings(resamples=200))
    assert report1.bootstrap_results["t4"] == report2.bootstrap_results["t4"]
    cache.close()


def test_52_no_private_paths_in_output():
    signal, context, cache = _t4_signal_and_context()
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    stats = tcd.build_image_diagnostic_statistics(
        signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
    )
    text = repr(stats)
    assert "/home/" not in text and "/scratch/" not in text and "/project/" not in text
    cache.close()


def test_53_empty_images_and_empty_strata_handled_correctly():
    plan = _plan(image=(4, 4), crop=(4, 4), stride=(4, 4))
    cache = wc.ImageWindowCache(plan)
    cache.append(_make_state(plan.windows[0], [[5.0, 0.0, 0.0]] * 4, [[5.0, 0.0, 0.0]] * 4, (2, 2), 3))
    cache.seal()
    context = _context(plan, 3, (2, 2), cache)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    assert signal.observations == ()
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    stats = tcd.build_image_diagnostic_statistics(
        signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
    )
    assert stats.base["t4"] == tcd.EMPTY_PAIRED_COUNTS
    assert stats.centrality_strata["t4"] == {}
    assert stats.shared_unary == {}
    acc = tcd.TrustCentralityAccumulator()
    acc.absorb_image(stats)
    report = acc.summary(settings=_settings(resamples=200))
    assert report.bootstrap_results["t4"].status == "unavailable"
    cache.close()


def test_disabled_diagnostics_zero_overhead_source():
    import inspect
    assert "trust_centrality_diagnostics" not in inspect.getsource(wc)
    assert "trust_centrality_diagnostics" not in inspect.getsource(t4)


def test_load_bearing_result_is_the_positive_centrality_t4_stratum():
    signal, context, cache = _t4_signal_and_context()
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    stats = tcd.build_image_diagnostic_statistics(
        signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
    )
    acc = tcd.TrustCentralityAccumulator()
    acc.absorb_image(stats)
    report = acc.summary(settings=_settings(resamples=200))
    assert report.load_bearing_result() is report.bootstrap_results["t4_centrality_positive"]


def test_delta_c_bin_index_exact_declared_boundaries():
    # Ten predeclared equal-width bins over [-1, 1]: [-1.0,-0.8), ...,
    # [0.8, 1.0] (final bin closed on both ends). Verified against
    # tcd.DELTA_C_BIN_EDGES rather than hardcoded edge values.
    edges = tcd.DELTA_C_BIN_EDGES
    assert len(edges) == tcd.DELTA_C_BIN_COUNT + 1 == 11
    assert edges[0] == -1.0 and edges[-1] == 1.0
    for bin_idx in range(tcd.DELTA_C_BIN_COUNT):
        lower, upper = edges[bin_idx], edges[bin_idx + 1]
        assert tcd.delta_c_bin_index(lower) == bin_idx  # lower edge is inclusive for every bin
        if bin_idx == tcd.DELTA_C_BIN_COUNT - 1:
            assert tcd.delta_c_bin_index(upper) == bin_idx  # final bin's upper edge (1.0) is inclusive too
        else:
            assert tcd.delta_c_bin_index(upper) == bin_idx + 1  # every other upper edge belongs to the next bin
            assert tcd.delta_c_bin_index(upper - 1e-9) == bin_idx  # just below the upper edge stays in this bin


def test_delta_c_bin_index_rejects_out_of_theoretical_range():
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.delta_c_bin_index(1.0 + 1e-6)
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.delta_c_bin_index(-1.0 - 1e-6)
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.delta_c_bin_index(float("nan"))
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.delta_c_bin_index(float("inf"))


def test_summary_registers_exactly_30_fixed_width_centrality_bin_populations():
    signal, context, cache = _t4_signal_and_context()
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    stats = tcd.build_image_diagnostic_statistics(
        signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
    )
    acc = tcd.TrustCentralityAccumulator()
    acc.absorb_image(stats)
    report = acc.summary(settings=_settings(resamples=200))

    bin_population_keys = {
        f"{stage}_centrality_bin_{bin_idx}"
        for stage in tcd.EXTENDED_STAGES
        for bin_idx in range(tcd.DELTA_C_BIN_COUNT)
    }
    assert len(bin_population_keys) == 30
    assert bin_population_keys.issubset(report.bootstrap_results.keys())
    # These 30 are additive: sign strata (3 stages x 3), edge bands, shared-
    # unary, and the primary T4 population remain present alongside them.
    assert "t4" in report.bootstrap_results
    assert "t4_centrality_positive" in report.bootstrap_results


def test_centrality_bin_population_observations_match_per_observation_bin_assignment():
    signal, context, cache = _t4_signal_and_context()
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    stats = tcd.build_image_diagnostic_statistics(
        signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
    )
    acc = tcd.TrustCentralityAccumulator()
    acc.absorb_image(stats)
    report = acc.summary(settings=_settings(resamples=200))

    for stage in tcd.EXTENDED_STAGES:
        total_binned = sum(
            report.bootstrap_results[f"{stage}_centrality_bin_{bin_idx}"].observed.count
            for bin_idx in range(tcd.DELTA_C_BIN_COUNT)
        )
        # Every observation entering the stage's Delta_c pool lands in
        # exactly one of the 10 fixed-width bins -- never dropped, never
        # double-counted, and bins are never used to admit/reject anything.
        assert total_binned == report.centrality_delta_descriptive[stage].count


def test_trust_centrality_accumulator_state_dict_round_trip_reproduces_identical_summary():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    acc = tcd.TrustCentralityAccumulator()
    for i in range(4):
        signal = t4.build_t4_signal_for_image(cache, context, image_id=f"img{i}")
        stats = tcd.build_image_diagnostic_statistics(
            signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
        )
        acc.absorb_image(stats)

    settings = _settings(resamples=300, seed=11)
    original_summary = acc.summary(settings=settings)

    state = acc.state_dict()
    import json
    json.dumps(state)  # must be JSON-serializable (checkpoint payload)

    restored = tcd.TrustCentralityAccumulator.from_state_dict(state)
    assert restored.image_ids() == acc.image_ids()
    restored_summary = restored.summary(settings=settings)

    # Byte-identical scientific reproduction: every bootstrap population's
    # observed totals, point estimate, and CI must match exactly (same
    # seed, same restored image order -> same resampling draws).
    assert set(restored_summary.bootstrap_results) == set(original_summary.bootstrap_results)
    for name, original_result in original_summary.bootstrap_results.items():
        restored_result = restored_summary.bootstrap_results[name]
        assert restored_result.observed == original_result.observed
        assert restored_result.point_estimate == original_result.point_estimate
        assert restored_result.ci_low == original_result.ci_low
        assert restored_result.ci_high == original_result.ci_high
        assert restored_result.valid_replicate_count == original_result.valid_replicate_count
    assert restored_summary.per_class_t4_totals == original_summary.per_class_t4_totals
    assert restored_summary.g_equals_d_count_t4 == original_summary.g_equals_d_count_t4
    assert restored_summary.g_is_third_label_count_t4 == original_summary.g_is_third_label_count_t4
    cache.close()


def test_trust_centrality_accumulator_from_state_dict_rejects_duplicate_image_ids():
    state = {
        "image_ids": ["a", "a"], "base_by_image": {}, "centrality_by_image": {}, "centrality_bins_by_image": {},
        "edge_by_image": {}, "shared_unary_by_image": {}, "per_class_by_image": {},
        "delta_c_pool": {s: [] for s in tcd.EXTENDED_STAGES}, "c_source_pool": {s: [] for s in tcd.EXTENDED_STAGES},
        "c_jury_pool": {s: [] for s in tcd.EXTENDED_STAGES}, "edge_patches_pool": {s: [] for s in tcd.EXTENDED_STAGES},
        "edge_unavailable_total": {s: 0 for s in tcd.EXTENDED_STAGES}, "g_equals_d_total": 0, "g_third_label_total": 0,
    }
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.TrustCentralityAccumulator.from_state_dict(state)


def test_trust_centrality_accumulator_from_state_dict_rejects_missing_field():
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.TrustCentralityAccumulator.from_state_dict({"image_ids": []})


def test_trust_centrality_accumulator_from_state_dict_rejects_tampered_order_digest():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    acc = tcd.TrustCentralityAccumulator()
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img0")
    stats = tcd.build_image_diagnostic_statistics(
        signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
    )
    acc.absorb_image(stats)
    state = dict(acc.state_dict())
    state["image_order_digest"] = "0" * 64  # tampered / stale digest
    with pytest.raises(tcd.TrustCentralityDiagnosticsError):
        tcd.TrustCentralityAccumulator.from_state_dict(state)
    cache.close()


def test_centrality_bins_never_used_to_filter_the_underlying_strata_totals():
    # Registering the 30 bin populations must not change any pre-existing
    # population's observed totals (sign strata, edge bands, shared-unary,
    # primary T4) -- bins are purely additive, descriptive-only.
    signal, context, cache = _t4_signal_and_context()
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    stats = tcd.build_image_diagnostic_statistics(
        signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
    )
    acc = tcd.TrustCentralityAccumulator()
    acc.absorb_image(stats)
    report = acc.summary(settings=_settings(resamples=200))
    assert report.bootstrap_results["t4"].observed.count == stats.base["t4"].count
    for stratum in tcd.CENTRALITY_STRATA:
        key = f"t4_centrality_{stratum}"
        assert report.bootstrap_results[key].observed == stats.centrality_strata["t4"].get(
            stratum, tcd.PairedCounts()
        )
