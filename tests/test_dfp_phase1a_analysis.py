"""Tests for src/dfp_phase1a_analysis.py: hand-derived multi-image
fixtures, dispute classification, oracle/persistence maps, confusion
statistics, semantic metrics, gains, coherence, and the strict
input-validation contract. CPU-only, no CUDA, no dataset access.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import src.dfp_phase1a_analysis as m  # noqa: E402
from src.dfp_phase1a_identity import load_identity  # noqa: E402

PROTOCOL_PATH = ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml"


# ---------------------------------------------------------------------------
# Hand-derived anchor fixture (see conversation-level derivation): a 3x3
# image with 4 classes, ignore_index=255, exercising all four dispute
# categories plus two non-dispute correct pixels plus one ignore pixel.
# ---------------------------------------------------------------------------

A = np.array([[0, 1, 1], [0, 0, 2], [0, 1, 3]], dtype=np.int64)  # U = label_0
B = np.array([[0, 2, 2], [1, 1, 3], [1, 2, 3]], dtype=np.int64)  # P = label_320
GT = np.array([[0, 1, 2], [3, 255, 2], [1, 0, 3]], dtype=np.int64)
CLASS_COUNT = 4
IGNORE_INDEX = 255

EXPECTED_HARMFUL = np.array([[False, True, False], [False, False, True], [False, False, False]])
EXPECTED_BENEFICIAL = np.array([[False, False, True], [False, False, False], [True, False, False]])
EXPECTED_THIRD_LABEL = np.array([[False, False, False], [True, False, False], [False, True, False]])
EXPECTED_DISPUTE = EXPECTED_HARMFUL | EXPECTED_BENEFICIAL | EXPECTED_THIRD_LABEL
EXPECTED_VALID = GT != IGNORE_INDEX

EXPECTED_O_P = np.array([[0, 1, 2], [1, 1, 2], [1, 2, 3]], dtype=np.int64)
EXPECTED_BENEFICIAL_UNDO = np.array([[0, 2, 1], [1, 1, 3], [0, 2, 3]], dtype=np.int64)
EXPECTED_THIRD_LABEL_DIAG = np.array([[0, 2, 2], [0, 1, 3], [1, 1, 3]], dtype=np.int64)


def _dispute():
    return m.classify_disputes(A.copy(), B.copy(), GT.copy(), ignore_index=IGNORE_INDEX)


# ---------------------------------------------------------------------------
# Lowest-index tie-break (explicit, independent of np.argmax's incidental
# first-occurrence behavior)
# ---------------------------------------------------------------------------


def test_lowest_index_argmax_no_tie():
    scores = np.zeros((3, 1, 1), dtype=np.float32)
    scores[:, 0, 0] = [1.0, 5.0, 2.0]
    label = m._lowest_index_argmax(scores)
    assert label[0, 0] == 1


def test_lowest_index_argmax_exact_tie_picks_lowest():
    scores = np.zeros((4, 1, 1), dtype=np.float32)
    scores[:, 0, 0] = [3.0, 5.0, 5.0, 1.0]
    label = m._lowest_index_argmax(scores)
    assert label[0, 0] == 1  # not 2, even though class 2 also ties at 5.0


def test_lowest_index_argmax_all_tied():
    scores = np.full((5, 1, 1), 7.0, dtype=np.float32)
    label = m._lowest_index_argmax(scores)
    assert label[0, 0] == 0


def test_lowest_index_argmax_matches_numpy_argmax_on_no_tie_case():
    rng = np.random.RandomState(0)
    scores = rng.rand(6, 4, 5).astype(np.float32)
    # break ties by adding tiny distinct noise so np.argmax's own
    # first-occurrence behavior isn't exercised here -- this test only
    # confirms agreement in the NO-TIE regime.
    label = m._lowest_index_argmax(scores)
    assert np.array_equal(label, np.argmax(scores, axis=0))


def test_derive_horizon_labels_does_not_mutate_scores():
    scores = {0: np.random.RandomState(1).rand(3, 2, 2).astype(np.float32)}
    before = scores[0].copy()
    m.derive_horizon_labels(scores)
    assert np.array_equal(scores[0], before)


# ---------------------------------------------------------------------------
# Four-way dispute accounting
# ---------------------------------------------------------------------------


def test_classify_disputes_hand_derived_categories():
    d = _dispute()
    assert np.array_equal(d.harmful, EXPECTED_HARMFUL)
    assert np.array_equal(d.beneficial, EXPECTED_BENEFICIAL)
    assert np.array_equal(d.third_label, EXPECTED_THIRD_LABEL)
    assert np.array_equal(d.dispute, EXPECTED_DISPUTE)
    assert np.array_equal(d.valid, EXPECTED_VALID)
    assert d.harmful_count == 2
    assert d.beneficial_count == 2
    assert d.third_label_count == 2
    assert d.dispute_count == 6


def test_dispute_categories_pairwise_disjoint_and_union_equals_dispute():
    d = _dispute()
    assert not np.any(d.harmful & d.beneficial)
    assert not np.any(d.harmful & d.third_label)
    assert not np.any(d.beneficial & d.third_label)
    assert np.array_equal(d.harmful | d.beneficial | d.third_label, d.dispute)


def test_no_ignore_pixel_in_any_category():
    d = _dispute()
    ignore_mask = GT == IGNORE_INDEX
    assert not np.any(ignore_mask & (d.harmful | d.beneficial | d.third_label | d.dispute))


def test_category_masks_exact_bool_dtype_and_shape():
    d = _dispute()
    for mask in (d.valid, d.dispute, d.harmful, d.beneficial, d.third_label):
        assert mask.dtype == np.bool_
        assert mask.shape == GT.shape


def test_classify_disputes_does_not_mutate_inputs():
    a, b, gt = A.copy(), B.copy(), GT.copy()
    m.classify_disputes(a, b, gt, ignore_index=IGNORE_INDEX)
    assert np.array_equal(a, A)
    assert np.array_equal(b, B)
    assert np.array_equal(gt, GT)


def test_all_correct_fixture_zero_dispute():
    n = 5
    a = np.zeros((n, n), dtype=np.int64)
    b = np.zeros((n, n), dtype=np.int64)
    gt = np.zeros((n, n), dtype=np.int64)
    d = m.classify_disputes(a, b, gt, ignore_index=255)
    assert d.dispute_count == 0
    assert d.harmful_count == 0
    assert d.beneficial_count == 0
    assert d.third_label_count == 0


def test_all_harmful_fixture():
    n = 4
    a = np.zeros((n, n), dtype=np.int64)
    b = np.ones((n, n), dtype=np.int64)
    gt = np.zeros((n, n), dtype=np.int64)  # GT == a everywhere, != b
    d = m.classify_disputes(a, b, gt, ignore_index=255)
    assert d.harmful_count == n * n
    assert d.beneficial_count == 0
    assert d.third_label_count == 0


def test_all_beneficial_fixture():
    n = 4
    a = np.zeros((n, n), dtype=np.int64)
    b = np.ones((n, n), dtype=np.int64)
    gt = np.ones((n, n), dtype=np.int64)  # GT == b everywhere, != a
    d = m.classify_disputes(a, b, gt, ignore_index=255)
    assert d.beneficial_count == n * n
    assert d.harmful_count == 0
    assert d.third_label_count == 0


def test_all_third_label_fixture():
    n = 4
    a = np.zeros((n, n), dtype=np.int64)
    b = np.ones((n, n), dtype=np.int64)
    gt = np.full((n, n), 2, dtype=np.int64)  # GT differs from both a and b
    d = m.classify_disputes(a, b, gt, ignore_index=255)
    assert d.third_label_count == n * n
    assert d.harmful_count == 0
    assert d.beneficial_count == 0


def test_all_ignore_fixture_zero_dispute():
    n = 4
    a = np.zeros((n, n), dtype=np.int64)
    b = np.ones((n, n), dtype=np.int64)
    gt = np.full((n, n), 255, dtype=np.int64)
    d = m.classify_disputes(a, b, gt, ignore_index=255)
    assert d.dispute_count == 0
    assert not np.any(d.valid)


# ---------------------------------------------------------------------------
# Oracle diagnostic maps
# ---------------------------------------------------------------------------


def test_construct_oracle_maps_hand_derived():
    d = _dispute()
    oracle = m.construct_oracle_maps(d)
    assert np.array_equal(oracle.o_p, EXPECTED_O_P)
    assert np.array_equal(oracle.beneficial_undo_only, EXPECTED_BENEFICIAL_UNDO)
    assert np.array_equal(oracle.third_label_diagnostic, EXPECTED_THIRD_LABEL_DIAG)


def test_oracle_maps_retain_p_outside_their_own_category():
    d = _dispute()
    oracle = m.construct_oracle_maps(d)
    # O_P must equal P everywhere except harmful pixels
    outside_harmful = ~d.harmful
    assert np.array_equal(oracle.o_p[outside_harmful], d.b[outside_harmful])
    outside_beneficial = ~d.beneficial
    assert np.array_equal(oracle.beneficial_undo_only[outside_beneficial], d.b[outside_beneficial])
    outside_third = ~d.third_label
    assert np.array_equal(oracle.third_label_diagnostic[outside_third], d.b[outside_third])


def test_construct_oracle_maps_does_not_mutate_dispute_arrays():
    d = _dispute()
    a_before, b_before = d.a.copy(), d.b.copy()
    m.construct_oracle_maps(d)
    assert np.array_equal(d.a, a_before)
    assert np.array_equal(d.b, b_before)


# ---------------------------------------------------------------------------
# Horizon-persistence selectors
# ---------------------------------------------------------------------------


def test_persistence_intermediate_horizon_equals_b_no_reversion():
    d = _dispute()
    label_t = d.b.copy()  # T=t agrees with P everywhere -> nothing reverts
    persistence = m.construct_horizon_persistence_maps(d, label_t, horizon=32)
    assert np.array_equal(persistence.full, d.b)
    assert persistence.changed_pixel_counts == {"harmful": 0, "beneficial": 0, "third_label": 0}


def test_persistence_intermediate_horizon_equals_a_full_reversion_on_dispute():
    d = _dispute()
    label_t = d.a.copy()  # T=t agrees with U everywhere -> every dispute pixel reverts
    persistence = m.construct_horizon_persistence_maps(d, label_t, horizon=16)
    expected_full = np.where(d.dispute, d.a, d.b)
    assert np.array_equal(persistence.full, expected_full)
    assert persistence.changed_pixel_counts["harmful"] == d.harmful_count
    assert persistence.changed_pixel_counts["beneficial"] == d.beneficial_count
    assert persistence.changed_pixel_counts["third_label"] == d.third_label_count


def test_persistence_intermediate_horizon_equals_third_class_still_uses_a():
    d = _dispute()
    # label_t is a THIRD class (not a, not b) at the harmful pixel (0,1):
    # a=1, b=2 there; use label_t=3 (neither a nor b) at that pixel only,
    # matching everything else to b (no reversion elsewhere).
    label_t = d.b.copy()
    label_t[0, 1] = 3
    persistence = m.construct_horizon_persistence_maps(d, label_t, horizon=64)
    # must fall back to a=1 at (0,1), never to the third class c=3
    assert persistence.full[0, 1] == d.a[0, 1]
    assert persistence.full[0, 1] != 3


def test_persistence_full_outside_dispute_always_retains_p():
    d = _dispute()
    label_t = np.zeros_like(d.a)  # arbitrary, differs from b almost everywhere
    persistence = m.construct_horizon_persistence_maps(d, label_t, horizon=16)
    outside_dispute = ~d.dispute
    assert np.array_equal(persistence.full[outside_dispute], d.b[outside_dispute])


def test_persistence_one_sided_maps_isolate_single_category():
    d = _dispute()
    label_t = d.a.copy()  # full reversion case
    persistence = m.construct_horizon_persistence_maps(d, label_t, horizon=16)
    # harmful_recovery_only must equal b everywhere except harmful pixels
    outside_harmful = ~d.harmful
    assert np.array_equal(persistence.harmful_recovery_only[outside_harmful], d.b[outside_harmful])
    assert np.array_equal(persistence.harmful_recovery_only[d.harmful], d.a[d.harmful])
    outside_beneficial = ~d.beneficial
    assert np.array_equal(persistence.beneficial_undo_only[outside_beneficial], d.b[outside_beneficial])
    outside_third = ~d.third_label
    assert np.array_equal(persistence.third_label_fallback_only[outside_third], d.b[outside_third])


def test_persistence_does_not_mutate_inputs():
    d = _dispute()
    label_t = d.a.copy()
    label_t_before = label_t.copy()
    m.construct_horizon_persistence_maps(d, label_t, horizon=16)
    assert np.array_equal(label_t, label_t_before)


# ---------------------------------------------------------------------------
# Confusion statistics -- hand-derived on the anchor fixture
# ---------------------------------------------------------------------------

EXPECTED_CONFUSION_P = np.array(
    [[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.int64
)
EXPECTED_INTERSECTION_P = np.array([1, 1, 1, 1], dtype=np.int64)
EXPECTED_GT_PIXELS_P = np.array([2, 2, 2, 2], dtype=np.int64)
EXPECTED_PRED_PIXELS_P = np.array([1, 2, 3, 2], dtype=np.int64)
EXPECTED_UNION_P = np.array([2, 3, 4, 3], dtype=np.int64)


def test_compute_confusion_statistics_hand_derived():
    stats = m.compute_confusion_statistics(B, GT, class_count=CLASS_COUNT, ignore_index=IGNORE_INDEX)
    assert np.array_equal(stats.confusion_matrix, EXPECTED_CONFUSION_P)
    assert np.array_equal(stats.intersection, EXPECTED_INTERSECTION_P)
    assert np.array_equal(stats.ground_truth_pixels, EXPECTED_GT_PIXELS_P)
    assert np.array_equal(stats.predicted_pixels, EXPECTED_PRED_PIXELS_P)
    assert np.array_equal(stats.union, EXPECTED_UNION_P)
    assert stats.intersection.dtype == np.int64
    assert stats.union.dtype == np.int64


def test_confusion_statistics_ignore_pixel_excluded():
    stats = m.compute_confusion_statistics(B, GT, class_count=CLASS_COUNT, ignore_index=IGNORE_INDEX)
    assert int(stats.confusion_matrix.sum()) == 8  # 9 pixels - 1 ignore


def test_compute_confusion_statistics_does_not_mutate_inputs():
    pred, gt = B.copy(), GT.copy()
    m.compute_confusion_statistics(pred, gt, class_count=CLASS_COUNT, ignore_index=IGNORE_INDEX)
    assert np.array_equal(pred, B)
    assert np.array_equal(gt, GT)


def test_aggregate_confusion_statistics_sums_exactly():
    s1 = m.compute_confusion_statistics(B, GT, class_count=CLASS_COUNT, ignore_index=IGNORE_INDEX)
    s2 = m.compute_confusion_statistics(A, GT, class_count=CLASS_COUNT, ignore_index=IGNORE_INDEX)
    agg = m.aggregate_confusion_statistics([s1, s2])
    assert np.array_equal(agg.intersection, s1.intersection + s2.intersection)
    assert np.array_equal(agg.union, s1.union + s2.union)
    assert np.array_equal(agg.confusion_matrix, s1.confusion_matrix + s2.confusion_matrix)


def test_aggregate_confusion_statistics_rejects_empty_sequence():
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.aggregate_confusion_statistics([])


# ---------------------------------------------------------------------------
# Semantic metrics
# ---------------------------------------------------------------------------


def test_compute_semantic_metrics_hand_derived():
    stats = m.compute_confusion_statistics(B, GT, class_count=CLASS_COUNT, ignore_index=IGNORE_INDEX)
    metrics = m.compute_semantic_metrics(stats)
    assert metrics.mIoU == pytest.approx(35.416666666666664, abs=1e-9)
    assert metrics.mAcc == pytest.approx(50.0, abs=1e-9)
    assert metrics.aAcc == pytest.approx(50.0, abs=1e-9)
    assert metrics.classes_with_defined_iou == 4
    assert metrics.classes_with_defined_accuracy == 4


def test_semantic_metrics_zero_union_class_is_nan_and_excluded():
    # class 3 never appears in GT or prediction -> union==0 for it
    pred = np.array([[0, 1], [1, 0]], dtype=np.int64)
    gt = np.array([[0, 1], [1, 0]], dtype=np.int64)
    stats = m.compute_confusion_statistics(pred, gt, class_count=4, ignore_index=255)
    metrics = m.compute_semantic_metrics(stats)
    assert math_isnan(metrics.per_class_iou[3])
    assert math_isnan(metrics.per_class_accuracy[3])
    assert metrics.classes_with_defined_iou == 2
    assert metrics.mIoU == pytest.approx(100.0, abs=1e-9)


def test_semantic_metrics_missing_gt_class_only_false_positive():
    # class 2 predicted but never present in GT: union>0 (FP), but GT
    # pixel count is 0 -> accuracy undefined for class 2, IoU IS defined
    # (and equals 0 since intersection is 0)
    pred = np.array([[0, 2], [1, 1]], dtype=np.int64)
    gt = np.array([[0, 1], [1, 1]], dtype=np.int64)
    stats = m.compute_confusion_statistics(pred, gt, class_count=3, ignore_index=255)
    metrics = m.compute_semantic_metrics(stats)
    assert math_isnan(metrics.per_class_accuracy[2])
    assert not math_isnan(metrics.per_class_iou[2])
    assert metrics.per_class_iou[2] == pytest.approx(0.0, abs=1e-9)


def test_semantic_metrics_rejects_all_ignore_aggregation():
    gt = np.full((2, 2), 255, dtype=np.int64)
    pred = np.zeros((2, 2), dtype=np.int64)
    stats = m.compute_confusion_statistics(pred, gt, class_count=3, ignore_index=255)
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.compute_semantic_metrics(stats)


def test_semantic_metrics_percent_scale_not_fraction():
    pred = np.zeros((2, 2), dtype=np.int64)
    gt = np.zeros((2, 2), dtype=np.int64)
    stats = m.compute_confusion_statistics(pred, gt, class_count=1, ignore_index=255)
    metrics = m.compute_semantic_metrics(stats)
    assert metrics.aAcc == pytest.approx(100.0, abs=1e-9)  # not 1.0


def test_dataset_level_miou_differs_from_mean_of_per_image_miou():
    # image 1: class 0 perfect (2/2), class 1 absent (union=0, excluded)
    pred1 = np.array([[0, 0]], dtype=np.int64)
    gt1 = np.array([[0, 0]], dtype=np.int64)
    # image 2: class 0 has 1 FN + 1 FP against class 1 -- imperfect
    pred2 = np.array([[1, 0]], dtype=np.int64)
    gt2 = np.array([[0, 1]], dtype=np.int64)

    s1 = m.compute_confusion_statistics(pred1, gt1, class_count=2, ignore_index=255)
    s2 = m.compute_confusion_statistics(pred2, gt2, class_count=2, ignore_index=255)
    per_image_miou = [m.compute_semantic_metrics(s1).mIoU, m.compute_semantic_metrics(s2).mIoU]
    mean_per_image = sum(per_image_miou) / len(per_image_miou)

    dataset_stats = m.aggregate_confusion_statistics([s1, s2])
    dataset_miou = m.compute_semantic_metrics(dataset_stats).mIoU

    assert mean_per_image != pytest.approx(dataset_miou, abs=1e-9)
    # image1 mIoU = 100 (class0 perfect, class1 undefined/excluded);
    # image2 mIoU = 0 (both classes union>0, intersection=0 for both)
    assert per_image_miou[0] == pytest.approx(100.0, abs=1e-9)
    assert per_image_miou[1] == pytest.approx(0.0, abs=1e-9)
    assert mean_per_image == pytest.approx(50.0, abs=1e-9)
    # dataset-level: class0 confusion: inter=2(img1)+0(img2 pred1 gt0 miss)=2? let's just assert it's not 50
    assert dataset_miou != pytest.approx(50.0, abs=1e-9)


def math_isnan(x: float) -> bool:
    import math

    return math.isnan(x)


# ---------------------------------------------------------------------------
# Gains
# ---------------------------------------------------------------------------


def test_gain_summary_hand_values():
    gains = m.compute_gain_summary(metric_u=10.0, metric_p=25.0, metric_o_p=30.0)
    assert gains.propagation_gain == pytest.approx(15.0)
    assert gains.g_recover == pytest.approx(5.0)
    assert gains.g_oracle_unary == pytest.approx(20.0)
    assert gains.relational_residual == pytest.approx(0.0, abs=1e-12)


def test_gain_summary_relational_identity_tight_float_tolerance():
    rng = np.random.RandomState(2)
    for _ in range(200):
        u, p, o = rng.uniform(0, 100, size=3)
        gains = m.compute_gain_summary(metric_u=u, metric_p=p, metric_o_p=o)
        assert abs(gains.relational_residual) < 1e-9


def test_persistence_gain_summary_independent_deltas():
    gains = m.compute_persistence_gain_summary(
        horizon=32, metric_p=50.0, metric_full=55.0, metric_harmful_only=52.0,
        metric_beneficial_only=48.0, metric_third_label_only=50.0,
    )
    assert gains.full_delta_vs_p == pytest.approx(5.0)
    assert gains.harmful_recovery_gain == pytest.approx(2.0)
    assert gains.beneficial_undo_loss == pytest.approx(-2.0)
    assert gains.third_label_fallback_delta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Reversion coherence
# ---------------------------------------------------------------------------


def test_coherence_empty_mask_neutral_values():
    mask = np.zeros((5, 5), dtype=bool)
    stats = m.compute_reversion_coherence(mask, connectivity=8, small_component_threshold=4.0)
    assert stats.total_reverted_pixels == 0
    assert stats.component_count == 0
    assert stats.component_sizes == ()
    assert stats.is_zero_reversion is True
    assert stats.perimeter == 0
    assert stats.perimeter_to_area_ratio == 0.0
    assert not any(np.isnan(x) for x in (stats.mean_component_size, stats.median_component_size, stats.small_component_fraction, stats.perimeter_to_area_ratio))


def test_coherence_single_pixel():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    stats = m.compute_reversion_coherence(mask, connectivity=8, small_component_threshold=4.0)
    assert stats.total_reverted_pixels == 1
    assert stats.component_count == 1
    assert stats.component_sizes == (1,)
    assert stats.perimeter == 1
    assert stats.perimeter_to_area_ratio == pytest.approx(1.0)


def test_coherence_diagonal_contact_8_vs_4_connectivity():
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    mask[2, 2] = True  # diagonal chain
    stats_8 = m.compute_reversion_coherence(mask, connectivity=8, small_component_threshold=1.0)
    stats_4 = m.compute_reversion_coherence(mask, connectivity=4, small_component_threshold=1.0)
    assert stats_8.component_count == 1  # diagonal chain merges under 8-connectivity
    assert stats_4.component_count == 3  # each pixel isolated under 4-connectivity


def test_coherence_deterministic_component_ordering():
    mask = np.zeros((10, 10), dtype=bool)
    mask[9, 9] = True  # size 1, far corner (no diagonal contact with others)
    mask[5:8, 5:8] = True  # size 9
    mask[0:2, 0] = True  # size 2, far corner
    stats = m.compute_reversion_coherence(mask, connectivity=8, small_component_threshold=4.0)
    assert stats.component_sizes == (9, 2, 1)  # descending


def test_coherence_small_component_fraction():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True  # size 1, below threshold 4
    mask[5:8, 5:8] = True  # size 9, at/above threshold 4
    stats = m.compute_reversion_coherence(mask, connectivity=8, small_component_threshold=4.0)
    assert stats.total_reverted_pixels == 10
    assert stats.small_component_fraction == pytest.approx(1.0 / 10.0)


def test_coherence_border_touching_component():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, :] = True  # touches top border
    stats = m.compute_reversion_coherence(mask, connectivity=8, small_component_threshold=1.0)
    assert stats.component_count == 1
    assert stats.perimeter == 4  # all border pixels count (out-of-bounds = background)


def test_coherence_does_not_mutate_input():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    before = mask.copy()
    m.compute_reversion_coherence(mask, connectivity=8, small_component_threshold=1.0)
    assert np.array_equal(mask, before)


def test_coherence_rejects_non_bool_mask():
    mask = np.zeros((3, 3), dtype=np.int64)
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.compute_reversion_coherence(mask, connectivity=8, small_component_threshold=1.0)


def test_coherence_rejects_bad_connectivity():
    mask = np.zeros((3, 3), dtype=bool)
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.compute_reversion_coherence(mask, connectivity=6, small_component_threshold=1.0)


# ---------------------------------------------------------------------------
# Boundary IoU is now bound and implemented -- see
# tests/test_dfp_phase1a_boundary_metrics.py (semantic tests + official-
# primitive cross-check) and
# tests/test_dfp_phase1a_boundary_iou_identity.py (authority supplement).
# ---------------------------------------------------------------------------
# Contract loading from the live protocol identity
# ---------------------------------------------------------------------------


def test_load_analysis_contract_from_real_protocol():
    protocol = load_identity(PROTOCOL_PATH, repo_root=ROOT)
    contract = m.load_analysis_contract_from_protocol(protocol, dataset_key="coco_stuff")
    assert contract.horizons == (0, 16, 32, 64, 320)
    assert contract.unary_horizon == 0
    assert contract.final_horizon == 320
    assert contract.primary_sensitivity_horizon == 32
    assert contract.persistence_horizons == (16, 32, 64)
    assert contract.primary_persistence_horizon == 32
    assert contract.class_count == 171
    assert contract.ignore_index == 255
    assert contract.primary_connectivity == 8
    assert contract.sensitivity_connectivity == 4

    ade = m.load_analysis_contract_from_protocol(protocol, dataset_key="ade20k")
    assert ade.class_count == 150
    assert ade.ignore_index == 255


def test_load_analysis_contract_rejects_unknown_dataset_key():
    protocol = load_identity(PROTOCOL_PATH, repo_root=ROOT)
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.load_analysis_contract_from_protocol(protocol, dataset_key="voc2012")


def test_load_analysis_contract_rejects_mismatched_tie_rule():
    # src.dfp_phase1a_identity.load_identity() itself now exact-pins
    # tie_rule against its own canonical text (a stricter, independent
    # guarantee), so a mutated TOML round-trip can no longer reach this
    # function's own redundant check -- construct the mapping directly
    # to exercise load_analysis_contract_from_protocol's own defense.
    identity = load_identity(PROTOCOL_PATH, repo_root=ROOT)
    mutated = copy.deepcopy(identity)
    mutated["dispute_protocol"] = dict(mutated["dispute_protocol"])
    mutated["dispute_protocol"]["tie_rule"] = "highest_index_wins"
    with pytest.raises(m.DfpPhase1aAnalysisError, match="tie_rule"):
        m.load_analysis_contract_from_protocol(mutated, dataset_key="coco_stuff")


def test_load_analysis_contract_rejects_mismatched_persistence_rule():
    identity = load_identity(PROTOCOL_PATH, repo_root=ROOT)
    mutated = copy.deepcopy(identity)
    mutated["horizon_persistence"] = dict(mutated["horizon_persistence"])
    mutated["horizon_persistence"]["rule"] = "a completely different persistence rule"
    with pytest.raises(m.DfpPhase1aAnalysisError, match="horizon_persistence"):
        m.load_analysis_contract_from_protocol(mutated, dataset_key="coco_stuff")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _one_hot_scores(labels: dict[int, np.ndarray], class_count: int) -> dict[int, np.ndarray]:
    scores = {}
    for horizon, label in labels.items():
        h, w = label.shape
        arr = np.full((class_count, h, w), -1.0, dtype=np.float32)
        for c in range(class_count):
            arr[c][label == c] = 1.0
        scores[horizon] = arr
    return scores


def _small_contract(base_contract: m.AnalysisContract, *, class_count: int, ignore_index: int) -> m.AnalysisContract:
    return m.AnalysisContract(
        horizons=base_contract.horizons, unary_horizon=base_contract.unary_horizon,
        final_horizon=base_contract.final_horizon, primary_sensitivity_horizon=base_contract.primary_sensitivity_horizon,
        secondary_sensitivity_horizons=base_contract.secondary_sensitivity_horizons,
        persistence_horizons=base_contract.persistence_horizons,
        primary_persistence_horizon=base_contract.primary_persistence_horizon,
        class_count=class_count, ignore_index=ignore_index, metric_unit=base_contract.metric_unit,
        difference_unit=base_contract.difference_unit, tie_rule=base_contract.tie_rule,
        primary_connectivity=base_contract.primary_connectivity, sensitivity_connectivity=base_contract.sensitivity_connectivity,
        reversion_connectivity=base_contract.reversion_connectivity,
        reversion_small_component_threshold_pixels=base_contract.reversion_small_component_threshold_pixels,
    )


def test_analyze_phase1a_image_end_to_end():
    protocol = load_identity(PROTOCOL_PATH, repo_root=ROOT)
    base_contract = m.load_analysis_contract_from_protocol(protocol, dataset_key="coco_stuff")
    contract = _small_contract(base_contract, class_count=CLASS_COUNT, ignore_index=IGNORE_INDEX)

    labels = {h: A.copy() if h == 0 else B.copy() for h in contract.horizons}
    scores = _one_hot_scores(labels, CLASS_COUNT)
    result = m.analyze_phase1a_image(scores, GT.copy(), contract=contract)

    assert np.array_equal(result.dispute.harmful, EXPECTED_HARMFUL)
    assert np.array_equal(result.oracle.o_p, EXPECTED_O_P)
    assert set(result.persistence.keys()) == set(contract.persistence_horizons)
    assert np.array_equal(result.confusion_p.confusion_matrix, EXPECTED_CONFUSION_P)


def test_analyze_phase1a_image_does_not_mutate_inputs():
    protocol = load_identity(PROTOCOL_PATH, repo_root=ROOT)
    base_contract = m.load_analysis_contract_from_protocol(protocol, dataset_key="coco_stuff")
    contract = _small_contract(base_contract, class_count=CLASS_COUNT, ignore_index=IGNORE_INDEX)
    labels = {h: A.copy() if h == 0 else B.copy() for h in contract.horizons}
    scores = _one_hot_scores(labels, CLASS_COUNT)
    scores_before = {h: arr.copy() for h, arr in scores.items()}
    gt = GT.copy()
    gt_before = gt.copy()
    m.analyze_phase1a_image(scores, gt, contract=contract)
    for h in scores:
        assert np.array_equal(scores[h], scores_before[h])
    assert np.array_equal(gt, gt_before)


# ---------------------------------------------------------------------------
# validate_stitched_inputs: strict adversarial matrix
# ---------------------------------------------------------------------------


HORIZONS = (0, 16, 32, 64, 320)


def _valid_scores(class_count=4, h=3, w=3) -> dict[int, np.ndarray]:
    rng = np.random.RandomState(7)
    return {t: rng.rand(class_count, h, w).astype(np.float32) for t in HORIZONS}


def _valid_gt(class_count=4, h=3, w=3, ignore_index=255) -> np.ndarray:
    rng = np.random.RandomState(8)
    return rng.randint(0, class_count, size=(h, w)).astype(np.int32)


def test_validate_stitched_inputs_accepts_valid_case():
    scores, gt = _valid_scores(), _valid_gt()
    m.validate_stitched_inputs(scores, gt, horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_accepted_arrays_not_modified():
    scores, gt = _valid_scores(), _valid_gt()
    scores_before = {k: v.copy() for k, v in scores.items()}
    gt_before = gt.copy()
    m.validate_stitched_inputs(scores, gt, horizons=HORIZONS, class_count=4, ignore_index=255)
    for k in scores:
        assert np.array_equal(scores[k], scores_before[k])
    assert np.array_equal(gt, gt_before)


def test_validate_rejects_float16_scores():
    scores = _valid_scores()
    scores[0] = scores[0].astype(np.float16)
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, _valid_gt(), horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_integer_scores():
    scores = _valid_scores()
    scores[0] = (scores[0] * 10).astype(np.int32)
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, _valid_gt(), horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_bool_scores():
    scores = _valid_scores()
    scores[0] = (scores[0] > 0.5)
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, _valid_gt(), horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_nan_scores():
    scores = _valid_scores()
    scores[0][0, 0, 0] = np.nan
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, _valid_gt(), horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_infinity_scores():
    scores = _valid_scores()
    scores[0][0, 0, 0] = np.inf
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, _valid_gt(), horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_wrong_score_layout_transposed():
    scores = _valid_scores(class_count=4, h=3, w=5)
    scores[0] = np.transpose(scores[0], (1, 2, 0))  # [H,W,class] instead of [class,H,W]
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, _valid_gt(class_count=4, h=3, w=5), horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_wrong_channel_count():
    scores = _valid_scores(class_count=5)
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, _valid_gt(class_count=4), horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_missing_horizon():
    scores = _valid_scores()
    del scores[16]
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, _valid_gt(), horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_extra_horizon():
    scores = _valid_scores()
    scores[999] = scores[0].copy()
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, _valid_gt(), horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_duplicate_horizon_in_required_list():
    scores = _valid_scores()
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, _valid_gt(), horizons=(0, 16, 16, 32, 64, 320), class_count=4, ignore_index=255)


def test_validate_rejects_bool_key_aliasing_horizon_zero():
    scores = _valid_scores()
    del scores[0]
    scores[False] = _valid_scores()[0]  # type(False) is bool, not int, even though False == 0
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, _valid_gt(), horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_spatial_shape_mismatch_across_horizons():
    scores = _valid_scores()
    scores[16] = np.random.RandomState(9).rand(4, 4, 4).astype(np.float32)  # different H,W
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, _valid_gt(), horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_gt_shape_mismatch():
    scores = _valid_scores(h=3, w=3)
    gt = _valid_gt(h=4, w=4)
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, gt, horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_gt_not_2d():
    scores = _valid_scores()
    gt = _valid_gt().reshape(1, 3, 3)
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, gt, horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_gt_bool_dtype():
    scores = _valid_scores(class_count=2)
    gt = _valid_gt(class_count=2).astype(bool)
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, gt, horizons=HORIZONS, class_count=2, ignore_index=255)


def test_validate_rejects_gt_float_dtype():
    scores = _valid_scores()
    gt = _valid_gt().astype(np.float32)
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, gt, horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_negative_gt_label():
    scores = _valid_scores()
    gt = _valid_gt().copy()
    gt[0, 0] = -1
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, gt, horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_out_of_range_gt_label():
    scores = _valid_scores(class_count=4)
    gt = _valid_gt(class_count=4).copy()
    gt[0, 0] = 4  # class_count=4 means legal labels are 0..3
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, gt, horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_accepts_ignore_index_in_gt():
    scores = _valid_scores()
    gt = _valid_gt().copy()
    gt[0, 0] = 255
    m.validate_stitched_inputs(scores, gt, horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_bool_as_int_class_count():
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(_valid_scores(), _valid_gt(), horizons=HORIZONS, class_count=True, ignore_index=255)


def test_validate_rejects_int_as_float_class_count():
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(_valid_scores(), _valid_gt(), horizons=HORIZONS, class_count=4.0, ignore_index=255)


def test_validate_rejects_numeric_string_class_count():
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(_valid_scores(), _valid_gt(), horizons=HORIZONS, class_count="4", ignore_index=255)


def test_validate_rejects_bool_as_int_ignore_index():
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(_valid_scores(), _valid_gt(), horizons=HORIZONS, class_count=4, ignore_index=False)


def test_validate_rejects_int_as_float_ignore_index():
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(_valid_scores(), _valid_gt(), horizons=HORIZONS, class_count=4, ignore_index=255.0)


def test_validate_rejects_scores_not_a_mapping():
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs([("x", 1)], _valid_gt(), horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_rejects_score_value_not_ndarray():
    scores = _valid_scores()
    scores[0] = scores[0].tolist()
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.validate_stitched_inputs(scores, _valid_gt(), horizons=HORIZONS, class_count=4, ignore_index=255)


def test_validate_supports_noncontiguous_arrays_without_mutation():
    scores = _valid_scores(class_count=4, h=6, w=6)
    scores = {k: v[:, ::2, ::2] for k, v in scores.items()}  # noncontiguous view
    gt = _valid_gt(class_count=4, h=3, w=3)
    before = {k: v.copy() for k, v in scores.items()}
    m.validate_stitched_inputs(scores, gt, horizons=HORIZONS, class_count=4, ignore_index=255)
    for k in scores:
        assert np.array_equal(scores[k], before[k])


def test_memory_error_propagates_through_validate_stitched_inputs(monkeypatch):
    def _boom(*a, **kw):
        raise MemoryError("simulated allocation failure")

    monkeypatch.setattr(np, "isfinite", _boom)
    with pytest.raises(MemoryError):
        m.validate_stitched_inputs(_valid_scores(), _valid_gt(), horizons=HORIZONS, class_count=4, ignore_index=255)


def test_keyboard_interrupt_propagates_through_compute_confusion_statistics(monkeypatch):
    def _boom(*a, **kw):
        raise KeyboardInterrupt()

    monkeypatch.setattr(np, "bincount", _boom)
    with pytest.raises(KeyboardInterrupt):
        m.compute_confusion_statistics(B, GT, class_count=CLASS_COUNT, ignore_index=IGNORE_INDEX)


def test_validate_supports_readonly_arrays():
    scores = _valid_scores()
    for v in scores.values():
        v.setflags(write=False)
    gt = _valid_gt()
    gt.setflags(write=False)
    m.validate_stitched_inputs(scores, gt, horizons=HORIZONS, class_count=4, ignore_index=255)
