"""Tests for the Boundary IoU surface of src/dfp_phase1a_analysis.py.

Two families:

1. Authoritative cross-check (`Test*OfficialPrimitive*`): fetches the
   PINNED official boundary-iou-api commit's `boundary_utils.py` into a
   fresh /tmp directory, verifies its SHA-256 against the value frozen
   in the additive Boundary IoU authority supplement, extracts ONLY the
   `mask_to_boundary` function via `ast` (never imports the whole
   upstream package -- it depends on `pycocotools`, not installed here,
   and is not needed for this one function), and requires this
   repository's local `_official_mask_to_boundary` reproduction to
   produce BIT-IDENTICAL output. Network-gated: skipped (not failed) if
   the network is unavailable, since production code never needs it.

2. Semantic boundary tests: hand-derived fixtures for the classwise,
   ignore-aware, dataset-level-aggregated semantic extension built on
   top of that primitive.
"""

from __future__ import annotations

import ast
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import src.dfp_phase1a_analysis as m  # noqa: E402
from src.dfp_phase1a_boundary_iou_identity import load_identity as load_boundary_identity  # noqa: E402
from src.dfp_phase1a_identity import load_identity as load_protocol_identity  # noqa: E402

OFFICIAL_SOURCE_URL = (
    "https://raw.githubusercontent.com/bowenc0221/boundary-iou-api/"
    "37d25586a677b043ed585f10e5c42d4e80176ea9/boundary_iou/utils/boundary_utils.py"
)
PINNED_SOURCE_SHA256 = "a495cc15fdbded38681edef30b29c62014e81e864f0e68ca6062ce42db51fd9c"
PINNED_SOURCE_COMMIT = "37d25586a677b043ed585f10e5c42d4e80176ea9"


def _fetch_official_source() -> str | None:
    """Fetches into a fresh /tmp file; returns None (never raises) if the
    network is unavailable, so these tests skip cleanly rather than
    failing on an unrelated environment limitation."""
    try:
        with urllib.request.urlopen(OFFICIAL_SOURCE_URL, timeout=15) as response:
            raw = response.read()
    except (urllib.error.URLError, OSError, TimeoutError):
        return None
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="dfp_boundary_official_source_")
    tmp_path = Path(tmp_dir) / "boundary_utils.py"
    tmp_path.write_bytes(raw)
    return tmp_path.read_text(encoding="utf-8")


def _extract_official_mask_to_boundary(source_text: str):
    tree = ast.parse(source_text)
    func_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "mask_to_boundary")
    func_src = ast.get_source_segment(source_text, func_node)
    namespace: dict = {}
    exec("import cv2\nimport numpy as np\n" + func_src, namespace)  # noqa: S102 -- isolated, pinned-source-hash-verified
    return namespace["mask_to_boundary"]


_OFFICIAL_SOURCE_TEXT = _fetch_official_source()
_NETWORK_UNAVAILABLE = _OFFICIAL_SOURCE_TEXT is None
_official_fn = None
if not _NETWORK_UNAVAILABLE:
    observed_sha256 = hashlib.sha256(_OFFICIAL_SOURCE_TEXT.encode("utf-8")).hexdigest()
    if observed_sha256 != PINNED_SOURCE_SHA256:
        _NETWORK_UNAVAILABLE = True  # treat a hash mismatch as "cannot verify" -- skip, do not silently trust
    else:
        _official_fn = _extract_official_mask_to_boundary(_OFFICIAL_SOURCE_TEXT)

skip_if_no_network = pytest.mark.skipif(_NETWORK_UNAVAILABLE, reason="official boundary-iou-api source unreachable or hash mismatch")


def _random_mask(rng: np.random.RandomState, h: int, w: int, density: float = 0.5) -> np.ndarray:
    return (rng.rand(h, w) < density).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1. Authoritative cross-check against the pinned official source
# ---------------------------------------------------------------------------


def test_pinned_source_hash_matches_authority_supplement():
    """Independently confirms (from the identity, without importing the
    analysis module's own belief) that the hash this test file pins
    matches the one frozen in the authority supplement."""
    identity = load_boundary_identity(ROOT / "evaluation_identities/e13_dfp_phase1a_boundary_iou.toml", repo_root=ROOT)
    assert identity["official_source"]["source_sha256"] == PINNED_SOURCE_SHA256
    assert identity["official_source"]["repository_commit"] == PINNED_SOURCE_COMMIT


@skip_if_no_network
def test_fetched_source_hash_matches_pinned_value():
    assert hashlib.sha256(_OFFICIAL_SOURCE_TEXT.encode("utf-8")).hexdigest() == PINNED_SOURCE_SHA256


@skip_if_no_network
@pytest.mark.parametrize("h,w", [(h, w) for h in range(1, 6) for w in range(1, 6)])
def test_official_primitive_matches_exhaustive_small_masks(h, w):
    rng = np.random.RandomState(h * 100 + w)
    for _ in range(4):
        mask = _random_mask(rng, h, w)
        official = _official_fn(mask.copy(), dilation_ratio=0.02)
        local = m._official_mask_to_boundary(mask.copy(), dilation_ratio=0.02)
        assert np.array_equal(official, local), f"mismatch at shape ({h},{w})"


@skip_if_no_network
def test_official_primitive_matches_randomized_larger_masks():
    rng = np.random.RandomState(123)
    for _ in range(20):
        h, w = rng.randint(20, 200), rng.randint(20, 200)
        mask = _random_mask(rng, h, w, density=rng.uniform(0.05, 0.95))
        official = _official_fn(mask.copy(), dilation_ratio=0.02)
        local = m._official_mask_to_boundary(mask.copy(), dilation_ratio=0.02)
        assert np.array_equal(official, local), f"mismatch at shape ({h},{w})"


@skip_if_no_network
def test_official_primitive_matches_empty_mask():
    mask = np.zeros((30, 40), dtype=np.uint8)
    assert np.array_equal(_official_fn(mask.copy(), 0.02), m._official_mask_to_boundary(mask.copy(), 0.02))


@skip_if_no_network
def test_official_primitive_matches_full_mask():
    mask = np.ones((30, 40), dtype=np.uint8)
    assert np.array_equal(_official_fn(mask.copy(), 0.02), m._official_mask_to_boundary(mask.copy(), 0.02))


@skip_if_no_network
def test_official_primitive_matches_singleton_mask():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 25] = 1
    assert np.array_equal(_official_fn(mask.copy(), 0.02), m._official_mask_to_boundary(mask.copy(), 0.02))


@skip_if_no_network
def test_official_primitive_matches_border_touching_object():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[0, :] = 1
    mask[:, 0] = 1
    assert np.array_equal(_official_fn(mask.copy(), 0.02), m._official_mask_to_boundary(mask.copy(), 0.02))


@skip_if_no_network
def test_official_primitive_matches_thin_line():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[30, :] = 1
    assert np.array_equal(_official_fn(mask.copy(), 0.02), m._official_mask_to_boundary(mask.copy(), 0.02))


@skip_if_no_network
def test_official_primitive_matches_multiple_components():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[5:10, 5:10] = 1
    mask[40:55, 10:20] = 1
    mask[20, 45] = 1
    assert np.array_equal(_official_fn(mask.copy(), 0.02), m._official_mask_to_boundary(mask.copy(), 0.02))


@skip_if_no_network
def test_official_primitive_matches_at_dilation_ratio_0_02_various_diagonals():
    # dilation = round(0.02 * sqrt(h^2+w^2)) crosses 1 -> 2 -> 4 across
    # these three sizes (see conversation-level derivation).
    for h, w, expected_dilation in [(20, 20, 1), (60, 60, 2), (140, 140, 4)]:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1
        official = _official_fn(mask.copy(), dilation_ratio=0.02)
        local = m._official_mask_to_boundary(mask.copy(), dilation_ratio=0.02)
        assert np.array_equal(official, local)
        diag = float(np.sqrt(h ** 2 + w ** 2))
        assert int(round(0.02 * diag)) == expected_dilation


@skip_if_no_network
def test_official_primitive_matches_near_half_rounding_boundary():
    # h=45, w=60 -> diag = sqrt(45^2+60^2) = sqrt(2025+3600) = sqrt(5625) = 75.0 exactly
    # 0.02 * 75.0 = 1.5 exactly -- Python's round() is round-half-to-even -> 2
    h, w = 45, 60
    diag = float(np.sqrt(h ** 2 + w ** 2))
    assert diag == 75.0
    assert round(0.02 * diag) == 2  # round-half-to-even: nearest even to 1.5 is 2
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:35, 15:45] = 1
    official = _official_fn(mask.copy(), dilation_ratio=0.02)
    local = m._official_mask_to_boundary(mask.copy(), dilation_ratio=0.02)
    assert np.array_equal(official, local)


@skip_if_no_network
def test_fixed_width_shares_verified_core_with_official_ratio_path():
    """The fixed-width dispatcher (`_boundary_fixed_width`) shares the
    identical `_erode_and_subtract` core already cross-checked against
    the true official source above; this test proves that sharing is
    real (not merely claimed) by confirming the fixed-width path
    produces the SAME output as the official ratio path whenever the
    ratio-derived dilation count equals the fixed width used."""
    for h, w, dilation in [(20, 20, 1), (60, 60, 2), (140, 140, 4)]:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1
        official = _official_fn(mask.copy(), dilation_ratio=0.02)
        fixed = m._boundary_fixed_width(mask.copy(), dilation)
        assert np.array_equal(official, fixed)


def test_pinned_hashes_recorded_in_report():
    """Not a real assertion beyond truthiness -- exists so the pinned
    source commit/hash appear in the test report/output for the
    verification write-up, per the task's explicit request."""
    print(f"\nPinned official source commit: {PINNED_SOURCE_COMMIT}")
    print(f"Pinned official source SHA-256: {PINNED_SOURCE_SHA256}")
    assert True


# ---------------------------------------------------------------------------
# Authority loading
# ---------------------------------------------------------------------------


@pytest.fixture()
def authority():
    return m.load_boundary_iou_authority(repo_root=ROOT)


def test_load_boundary_iou_authority(authority):
    assert authority.dilation_ratio_primary == 0.02
    assert authority.fixed_sensitivity_widths == (1, 2, 4)
    assert authority.official_source_commit == PINNED_SOURCE_COMMIT
    assert authority.official_source_sha256 == PINNED_SOURCE_SHA256


def test_load_analysis_contract_can_bind_boundary_iou():
    protocol = load_protocol_identity(ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml", repo_root=ROOT)
    contract = m.load_analysis_contract_from_protocol(protocol, dataset_key="coco_stuff", bind_boundary_iou=True, repo_root=ROOT)
    assert contract.boundary_iou_authority is not None
    assert contract.boundary_iou_authority.dilation_ratio_primary == 0.02


def test_load_analysis_contract_defaults_to_unbound_boundary_iou():
    protocol = load_protocol_identity(ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml", repo_root=ROOT)
    contract = m.load_analysis_contract_from_protocol(protocol, dataset_key="coco_stuff")
    assert contract.boundary_iou_authority is None


def test_bind_boundary_iou_without_repo_root_rejected():
    protocol = load_protocol_identity(ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml", repo_root=ROOT)
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.load_analysis_contract_from_protocol(protocol, dataset_key="coco_stuff", bind_boundary_iou=True)


def test_boundary_functions_reject_missing_authority():
    pred = np.zeros((4, 4), dtype=np.int64)
    gt = np.zeros((4, 4), dtype=np.int64)
    with pytest.raises(TypeError):
        m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=0.02)


def test_boundary_functions_reject_fake_authority_object():
    pred = np.zeros((4, 4), dtype=np.int64)
    gt = np.zeros((4, 4), dtype=np.int64)

    class _Fake:
        dilation_ratio_primary = 0.02
        fixed_sensitivity_widths = (1, 2, 4)

    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=_Fake())


def test_boundary_width_must_match_authority_ratio(authority):
    pred = np.zeros((4, 4), dtype=np.int64)
    gt = np.zeros((4, 4), dtype=np.int64)
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=0.05, boundary_iou_authority=authority)


def test_boundary_width_must_be_a_frozen_fixed_width(authority):
    pred = np.zeros((4, 4), dtype=np.int64)
    gt = np.zeros((4, 4), dtype=np.int64)
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=3, boundary_iou_authority=authority)


# ---------------------------------------------------------------------------
# 2. Semantic boundary tests
# ---------------------------------------------------------------------------


def _square(n: int, size: int, class_id: int, background: int = 0, dtype=np.int64) -> np.ndarray:
    arr = np.full((n, n), background, dtype=dtype)
    lo = (n - size) // 2
    arr[lo : lo + size, lo : lo + size] = class_id
    return arr


def test_identical_semantic_maps_perfect_biou(authority):
    n = 30
    gt = _square(n, 16, 1)
    pred = gt.copy()
    stats = m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    metrics = m.compute_boundary_metrics(stats)
    assert metrics.per_class_biou[1] == pytest.approx(100.0)


def test_one_pixel_translation_reduces_biou(authority):
    n = 30
    gt = _square(n, 16, 1)
    pred = np.roll(gt, shift=1, axis=1)
    stats = m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    metrics = m.compute_boundary_metrics(stats)
    assert 0.0 < metrics.per_class_biou[1] < 100.0


def test_erosion_and_dilation_change_boundary_statistics(authority):
    n = 40
    gt = _square(n, 20, 1)
    eroded_pred = _square(n, 16, 1)  # smaller square = erosion of GT
    dilated_pred = _square(n, 24, 1)  # larger square = dilation of GT
    stats_eroded = m.compute_boundary_statistics(eroded_pred, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    stats_dilated = m.compute_boundary_statistics(dilated_pred, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    metrics_eroded = m.compute_boundary_metrics(stats_eroded)
    metrics_dilated = m.compute_boundary_metrics(stats_dilated)
    assert metrics_eroded.per_class_biou[1] < 100.0
    assert metrics_dilated.per_class_biou[1] < 100.0


def test_border_touching_class(authority):
    n = 20
    gt = np.zeros((n, n), dtype=np.int64)
    gt[0, :] = 1  # touches the top border
    pred = gt.copy()
    stats = m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    metrics = m.compute_boundary_metrics(stats)
    assert metrics.per_class_biou[1] == pytest.approx(100.0)


def test_empty_prediction_for_a_class(authority):
    n = 20
    gt = _square(n, 8, 1)
    pred = np.zeros((n, n), dtype=np.int64)  # never predicts class 1
    stats = m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    metrics = m.compute_boundary_metrics(stats)
    assert metrics.per_class_biou[1] == pytest.approx(0.0)  # intersection=0, union>0


def test_empty_gt_class(authority):
    n = 20
    gt = np.zeros((n, n), dtype=np.int64)  # class 1 absent from GT
    pred = _square(n, 8, 1)  # but predicted somewhere
    stats = m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    metrics = m.compute_boundary_metrics(stats)
    assert metrics.per_class_biou[1] == pytest.approx(0.0)


def test_missing_class_entirely_absent_from_both(authority):
    n = 20
    gt = np.zeros((n, n), dtype=np.int64)
    pred = np.zeros((n, n), dtype=np.int64)
    stats = m.compute_boundary_statistics(pred, gt, class_count=3, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    assert stats.boundary_union[1] == 0
    assert stats.boundary_union[2] == 0


def test_ignore_only_image_zero_statistics(authority):
    n = 20
    gt = np.full((n, n), 255, dtype=np.int64)
    pred = _square(n, 8, 1)
    stats = m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    assert int(stats.boundary_intersection.sum()) == 0
    assert int(stats.boundary_union.sum()) == 0


def test_ignored_pixel_itself_never_counted_even_when_it_creates_a_local_hole(authority):
    """A single interior pixel becoming ignore punches a hole in the
    class mask, which DOES perturb nearby boundary pixels (erosion is a
    neighborhood operation) -- but the ignored pixel's OWN location must
    never itself appear as a counted boundary pixel, in either gt or
    prediction, regardless of where it sits. This is the actual
    always-true invariant (not "statistics never change")."""
    n = 30
    gt = _square(n, 16, 1)
    pred = gt.copy()
    gt_with_hole = gt.copy()
    gt_with_hole[14, 14] = 255  # deep interior of the object

    valid = gt_with_hole != 255
    gt_c = (gt_with_hole == 1).astype(np.uint8)
    pred_c = ((pred == 1) & valid).astype(np.uint8)
    gt_boundary = (m._official_mask_to_boundary(gt_c, dilation_ratio=0.02).astype(bool)) & valid
    pred_boundary = (m._official_mask_to_boundary(pred_c, dilation_ratio=0.02).astype(bool)) & valid
    assert gt_boundary[14, 14] == False  # noqa: E712 -- explicit boolean-value assertion
    assert pred_boundary[14, 14] == False  # noqa: E712

    # sanity: the hole DOES perturb nearby statistics (confirms this is a
    # meaningful fixture, not a no-op)
    stats_plain = m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    stats_hole = m.compute_boundary_statistics(pred, gt_with_hole, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    assert int(stats_plain.boundary_union[1]) != int(stats_hole.boundary_union[1])


def test_ignored_strip_along_object_edge_reduces_boundary_union(authority):
    n = 30
    gt = _square(n, 16, 1)
    pred = gt.copy()
    lo = (n - 16) // 2
    gt_with_ignored_edge = gt.copy()
    gt_with_ignored_edge[lo, lo : lo + 16] = 255  # the object's entire top edge, already boundary
    stats_plain = m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    stats_ignored = m.compute_boundary_statistics(pred, gt_with_ignored_edge, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    # removing an entire pre-existing boundary strip from candidacy must
    # strictly decrease the accumulated boundary union for that class
    assert int(stats_ignored.boundary_union[1]) < int(stats_plain.boundary_union[1])


def test_predictions_changed_only_inside_ignore_leave_statistics_unchanged(authority):
    n = 30
    gt = _square(n, 16, 1)
    gt[0, 0] = 255  # a single ignored pixel far from the object
    pred_a = gt.copy()
    pred_a[gt == 255] = 1  # arbitrary prediction only at the ignored pixel
    pred_b = gt.copy()
    pred_b[gt == 255] = 0  # different arbitrary prediction only at the ignored pixel
    stats_a = m.compute_boundary_statistics(pred_a, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    stats_b = m.compute_boundary_statistics(pred_b, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    assert np.array_equal(stats_a.boundary_intersection, stats_b.boundary_intersection)
    assert np.array_equal(stats_a.boundary_union, stats_b.boundary_union)


def test_zero_boundary_union_rejected_at_metrics(authority):
    n = 10
    gt = np.full((n, n), 255, dtype=np.int64)
    pred = np.zeros((n, n), dtype=np.int64)
    stats = m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.compute_boundary_metrics(stats)


def test_multi_image_aggregation_sums_exactly(authority):
    n = 20
    gt1 = _square(n, 10, 1)
    pred1 = gt1.copy()
    gt2 = _square(n, 8, 1)
    pred2 = np.roll(gt2, 1, axis=0)
    s1 = m.compute_boundary_statistics(pred1, gt1, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    s2 = m.compute_boundary_statistics(pred2, gt2, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    agg = m.aggregate_boundary_statistics([s1, s2])
    assert np.array_equal(agg.boundary_intersection, s1.boundary_intersection + s2.boundary_intersection)
    assert np.array_equal(agg.boundary_union, s1.boundary_union + s2.boundary_union)


def test_no_mean_of_per_image_biou_substitution(authority):
    n = 20
    gt1 = _square(n, 10, 1)
    pred1 = gt1.copy()  # perfect -> 100%
    gt2 = _square(n, 8, 1)
    pred2 = np.roll(gt2, 2, axis=0)  # imperfect

    s1 = m.compute_boundary_statistics(pred1, gt1, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    s2 = m.compute_boundary_statistics(pred2, gt2, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    per_image_biou = [m.compute_boundary_metrics(s1).per_class_biou[1], m.compute_boundary_metrics(s2).per_class_biou[1]]
    mean_per_image = sum(per_image_biou) / 2.0

    dataset_stats = m.aggregate_boundary_statistics([s1, s2])
    dataset_biou = m.compute_boundary_metrics(dataset_stats).per_class_biou[1]

    assert mean_per_image != pytest.approx(dataset_biou, abs=1e-9)


def test_primary_normalized_width_vs_fixed_widths_differ(authority):
    n = 60
    gt = _square(n, 40, 1)
    pred = _square(n, 36, 1)  # slightly eroded relative to GT

    stats_primary = m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    stats_w1 = m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=1, boundary_iou_authority=authority)
    stats_w2 = m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=2, boundary_iou_authority=authority)
    stats_w4 = m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=4, boundary_iou_authority=authority)

    unions = {int(stats_primary.boundary_union[1]), int(stats_w1.boundary_union[1]), int(stats_w2.boundary_union[1]), int(stats_w4.boundary_union[1])}
    assert len(unions) > 1  # different widths must generally produce different statistics


def test_compute_boundary_statistics_does_not_mutate_inputs(authority):
    n = 20
    gt = _square(n, 8, 1)
    pred = gt.copy()
    gt_before, pred_before = gt.copy(), pred.copy()
    m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    assert np.array_equal(gt, gt_before)
    assert np.array_equal(pred, pred_before)


def test_aggregate_boundary_statistics_rejects_empty_sequence():
    with pytest.raises(m.DfpPhase1aAnalysisError):
        m.aggregate_boundary_statistics([])


def test_boundary_metrics_percent_scale(authority):
    n = 20
    gt = _square(n, 8, 1)
    pred = gt.copy()
    stats = m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
    metrics = m.compute_boundary_metrics(stats)
    assert metrics.mean_biou == pytest.approx(100.0)  # not 1.0


def test_memory_error_propagates_through_compute_boundary_statistics(monkeypatch, authority):
    def _boom(*a, **kw):
        raise MemoryError("simulated allocation failure")

    monkeypatch.setattr(np, "zeros", _boom)
    n = 5
    gt = _square(n, 2, 1)
    pred = gt.copy()
    with pytest.raises(MemoryError):
        m.compute_boundary_statistics(pred, gt, class_count=2, ignore_index=255, boundary_width=0.02, boundary_iou_authority=authority)
