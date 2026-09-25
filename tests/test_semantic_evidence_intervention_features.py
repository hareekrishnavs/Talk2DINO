"""CPU-only, independent-oracle tests for
src/semantic_evidence_intervention_features.py.

Never opens a file, never constructs a real model, never touches GT, never
initializes CUDA. Every check below either recomputes the expected value
from the committed TOML's own frozen formulas/constants using arithmetic
written independently in this file (never by calling back into the module
function under test to produce its own "expected" value), or exercises a
structural/adversarial property (determinism, exclusion behavior, range
enforcement, digest sensitivity) that does not depend on trusting the
implementation's own internal correctness.

NOTE: this repository runs on a shared HPC login node; torch's default
thread pool can severely oversubscribe cgroup-restricted CPUs, so
torch.set_num_threads(1) below is not optional -- see
tests/test_semantic_evidence_t320_real_smoke.py's own precedent.

Run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
        tests.test_semantic_evidence_intervention_features -v
"""
from __future__ import annotations

import hashlib
import math
import sys
import unittest
from pathlib import Path

import torch
torch.set_num_threads(1)
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import semantic_evidence_intervention_features as ifeat  # noqa: E402


def _make_patch(row, col, top1, top2, margin):
    return ifeat.PatchInfo(row=row, col=col, patch_index=row * ifeat.GRID_SIDE + col,
                            unary_top1=top1, unary_top2=top2, unary_margin=margin)


def _full_grid_uniform(top1_fn):
    """Builds a full 32x32 PatchInfo grid; top1_fn(row,col) -> class id."""
    grid = {}
    for r in range(ifeat.GRID_SIDE):
        for c in range(ifeat.GRID_SIDE):
            top1 = top1_fn(r, c)
            grid[(r, c)] = _make_patch(r, c, top1, (top1 + 1) % 171, float((r + c) % 5))
    return grid


class AnchorKeyTests(unittest.TestCase):
    def test_matches_independent_sha256(self):
        image_id, window_index, patch_index = "000000567197", 3, 517
        expected = hashlib.sha256(
            ifeat.ANCHOR_NAMESPACE.encode("utf-8") + b"\x00" +
            image_id.encode("ascii") + b"\x00" +
            str(window_index).encode("ascii") + b"\x00" +
            str(patch_index).encode("ascii")
        ).hexdigest()
        self.assertEqual(ifeat.anchor_key(image_id, window_index, patch_index), expected)

    def test_deterministic_and_sensitive_to_every_argument(self):
        base = ifeat.anchor_key("img", 0, 0)
        self.assertEqual(base, ifeat.anchor_key("img", 0, 0))
        self.assertNotEqual(base, ifeat.anchor_key("img2", 0, 0))
        self.assertNotEqual(base, ifeat.anchor_key("img", 1, 0))
        self.assertNotEqual(base, ifeat.anchor_key("img", 0, 1))

    def test_rejects_non_int_window_or_patch_index(self):
        with self.assertRaises(ifeat.InterventionFeatureError):
            ifeat.anchor_key("img", True, 0)  # bool must not pass as int
        with self.assertRaises(ifeat.InterventionFeatureError):
            ifeat.anchor_key("img", 0, "0")


class BoundaryDetectionTests(unittest.TestCase):
    def test_uniform_grid_has_no_boundary_patches(self):
        grid = _full_grid_uniform(lambda r, c: 5)
        for (r, c), p in grid.items():
            if not ifeat._is_eligible(r, c):
                continue
            self.assertFalse(ifeat.is_boundary_patch(p, grid), f"({r},{c}) should not be boundary in a uniform grid")

    def test_class_split_produces_boundary_only_at_seam(self):
        # left half class 1, right half class 2 -> boundary patches are
        # exactly those adjacent (8-connected) to the seam at col==16.
        grid = _full_grid_uniform(lambda r, c: 1 if c < 16 else 2)
        seam_adjacent = set()
        for r in range(ifeat.GRID_SIDE):
            for c in range(ifeat.GRID_SIDE):
                if abs(c - 16) <= 1 and abs(c - 15) <= 1:
                    seam_adjacent.add((r, c))
        for r in range(ifeat.ELIGIBLE_LO, ifeat.ELIGIBLE_HI + 1):
            for c in range(ifeat.ELIGIBLE_LO, ifeat.ELIGIBLE_HI + 1):
                p = grid[(r, c)]
                is_boundary = ifeat.is_boundary_patch(p, grid)
                near_seam = c in (15, 16)
                self.assertEqual(is_boundary, near_seam, f"mismatch at ({r},{c})")

    def test_missing_neighbor_raises(self):
        p = _make_patch(1, 1, 0, 1, 0.5)
        with self.assertRaises(ifeat.InterventionFeatureError):
            ifeat.is_boundary_patch(p, {})


class SelectAnchorsTests(unittest.TestCase):
    def test_requires_full_32x32_grid(self):
        grid = _full_grid_uniform(lambda r, c: 1)
        partial = {k: v for k, v in grid.items() if ifeat._is_eligible(*k)}
        eligible = list(partial.values())
        with self.assertRaises(ifeat.InterventionFeatureError):
            ifeat.select_anchors("img", 0, eligible, partial)

    def test_returns_at_most_target_and_no_duplicates(self):
        grid = _full_grid_uniform(lambda r, c: (r * 7 + c * 3) % 171)
        eligible = [p for (r, c), p in grid.items() if ifeat._is_eligible(r, c)]
        selected = ifeat.select_anchors("000000567197", 0, eligible, grid)
        self.assertLessEqual(len(selected), ifeat.TARGET_ANCHORS_PER_WINDOW)
        indices = [p.patch_index for p in selected]
        self.assertEqual(len(indices), len(set(indices)), "select_anchors must not return duplicate patches")

    def test_deterministic_for_same_inputs(self):
        grid = _full_grid_uniform(lambda r, c: (r * 11 + c * 5) % 171)
        eligible = [p for (r, c), p in grid.items() if ifeat._is_eligible(r, c)]
        a = ifeat.select_anchors("img", 2, eligible, grid)
        b = ifeat.select_anchors("img", 2, eligible, grid)
        self.assertEqual([p.patch_index for p in a], [p.patch_index for p in b])

    def test_uniform_grid_fill_rule_reaches_target_count(self):
        # A perfectly uniform grid has zero boundary patches, so all 8
        # anchors must come from the interior fill rule.
        grid = _full_grid_uniform(lambda r, c: 3)
        eligible = [p for (r, c), p in grid.items() if ifeat._is_eligible(r, c)]
        selected = ifeat.select_anchors("img", 0, eligible, grid)
        self.assertEqual(len(selected), ifeat.TARGET_ANCHORS_PER_WINDOW)

    def test_rejects_empty_eligible_list(self):
        grid = _full_grid_uniform(lambda r, c: 1)
        with self.assertRaises(ifeat.InterventionFeatureError):
            ifeat.select_anchors("img", 0, [], grid)


class QueryConstructionTests(unittest.TestCase):
    def test_q1_q2_come_from_top1_top2(self):
        p = _make_patch(5, 5, 42, 99, 0.3)
        q = ifeat.construct_queries(p)
        self.assertEqual(q.q1, 42)
        self.assertEqual(q.q2, 99)


class RegionGeometryTests(unittest.TestCase):
    def test_core_pixel_bounds_matches_independent_formula(self):
        for row, col in [(1, 1), (15, 20), (30, 30)]:
            top, left, bottom, right = ifeat.core_pixel_bounds(row, col)
            self.assertEqual(top, (row - 1) * ifeat.PATCH_SIZE)
            self.assertEqual(left, (col - 1) * ifeat.PATCH_SIZE)
            self.assertEqual(bottom - top, ifeat.CORE_PIXELS)
            self.assertEqual(right - left, ifeat.CORE_PIXELS)

    def test_core_pixel_bounds_rejects_ineligible(self):
        with self.assertRaises(ifeat.InterventionFeatureError):
            ifeat.core_pixel_bounds(0, 0)
        with self.assertRaises(ifeat.InterventionFeatureError):
            ifeat.core_pixel_bounds(31, 31)

    def test_core_patch_set_is_the_9_patch_neighborhood(self):
        s = ifeat.core_patch_set(10, 10)
        expected = {(r, c) for r in (9, 10, 11) for c in (9, 10, 11)}
        self.assertEqual(s, expected)
        self.assertEqual(len(s), ifeat.CORE_PATCH_COUNT)


class ControlCenterSelectionTests(unittest.TestCase):
    def test_excludes_overlapping_candidates_and_picks_by_anchor_key(self):
        grid = _full_grid_uniform(lambda r, c: 1)
        eligible = [p for (r, c), p in grid.items() if ifeat._is_eligible(r, c)]
        center = ifeat.select_control_center("img", 0, 15, 15, eligible)
        self.assertIsNotNone(center)
        anchor_core = ifeat.core_patch_set(15, 15)
        control_core = ifeat.core_patch_set(center.row, center.col)
        self.assertFalse(anchor_core & control_core, "control core must not overlap the anchor core")
        # independently verify it is the minimal-anchor_key candidate
        candidates = [
            p for p in eligible
            if (p.row, p.col) != (15, 15) and not (ifeat.core_patch_set(p.row, p.col) & anchor_core)
        ]
        expected = min(candidates, key=lambda p: (ifeat.anchor_key("img", 0, p.patch_index), p.patch_index))
        self.assertEqual((center.row, center.col), (expected.row, expected.col))

    def test_no_valid_control_returns_none(self):
        # A 1-patch eligible pool consisting only of the anchor itself (plus
        # patches that all overlap its core) must return None, never raise
        # and never silently pick an overlapping patch.
        grid = _full_grid_uniform(lambda r, c: 1)
        anchor = grid[(15, 15)]
        overlapping_only = [grid[(r, c)] for (r, c) in ifeat.core_patch_set(15, 15) if ifeat._is_eligible(r, c)]
        result = ifeat.select_control_center("img", 0, 15, 15, overlapping_only)
        self.assertIsNone(result)


class GaussianBlurTests(unittest.TestCase):
    def test_matches_direct_torchvision_call(self):
        crop = torch.rand(1, 3, 64, 64)
        expected = tv_gaussian_blur(crop, kernel_size=[31, 31], sigma=[8.0, 8.0])
        actual = ifeat.deterministic_gaussian_blur(crop)
        self.assertTrue(torch.equal(expected, actual))

    def test_rejects_non_tensor(self):
        with self.assertRaises(ifeat.InterventionFeatureError):
            ifeat.deterministic_gaussian_blur([[1, 2], [3, 4]])


class InterventionViewTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.crop = torch.rand(1, 3, 64, 64)
        self.anchor_bounds = (20, 20, 34, 34)
        self.control_bounds = (40, 40, 54, 54)

    def test_original_is_exact_clone(self):
        out = ifeat.build_intervention_view("ORIGINAL", self.crop, self.anchor_bounds)
        self.assertTrue(torch.equal(out, self.crop))
        self.assertTrue(ifeat.verify_unchanged_domain("ORIGINAL", self.crop, out, self.anchor_bounds))

    def test_context_blurred_keeps_core_bitwise_exact(self):
        out = ifeat.build_intervention_view("CONTEXT_BLURRED", self.crop, self.anchor_bounds)
        top, left, bottom, right = self.anchor_bounds
        self.assertTrue(torch.equal(out[:, :, top:bottom, left:right], self.crop[:, :, top:bottom, left:right]))
        self.assertTrue(ifeat.verify_unchanged_domain("CONTEXT_BLURRED", self.crop, out, self.anchor_bounds))
        # outside the core it must differ from ORIGINAL (blur is not identity for random noise)
        outside_changed = not torch.equal(out[:, :, :10, :10], self.crop[:, :, :10, :10])
        self.assertTrue(outside_changed)

    def test_region_blurred_keeps_outside_bitwise_exact(self):
        out = ifeat.build_intervention_view("REGION_BLURRED", self.crop, self.anchor_bounds)
        top, left, bottom, right = self.anchor_bounds
        mask = torch.ones(self.crop.shape, dtype=torch.bool)
        mask[..., top:bottom, left:right] = False
        self.assertTrue(torch.equal(out[mask], self.crop[mask]))
        self.assertTrue(ifeat.verify_unchanged_domain("REGION_BLURRED", self.crop, out, self.anchor_bounds))

    def test_area_matched_control_keeps_anchor_core_and_rest_exact(self):
        out = ifeat.build_intervention_view("AREA_MATCHED_CONTROL", self.crop, self.anchor_bounds, self.control_bounds)
        top, left, bottom, right = self.anchor_bounds
        self.assertTrue(torch.equal(out[:, :, top:bottom, left:right], self.crop[:, :, top:bottom, left:right]))
        ctop, cleft, cbottom, cright = self.control_bounds
        self.assertFalse(torch.equal(out[:, :, ctop:cbottom, cleft:cright], self.crop[:, :, ctop:cbottom, cleft:cright]))
        self.assertTrue(ifeat.verify_unchanged_domain("AREA_MATCHED_CONTROL", self.crop, out, self.anchor_bounds, self.control_bounds))

    def test_area_matched_control_requires_control_bounds(self):
        with self.assertRaises(ifeat.InterventionFeatureError):
            ifeat.build_intervention_view("AREA_MATCHED_CONTROL", self.crop, self.anchor_bounds)

    def test_unknown_view_rejected(self):
        with self.assertRaises(ifeat.InterventionFeatureError):
            ifeat.build_intervention_view("NOT_A_VIEW", self.crop, self.anchor_bounds)

    def test_tampered_view_fails_unchanged_domain_check(self):
        out = ifeat.build_intervention_view("REGION_BLURRED", self.crop, self.anchor_bounds)
        tampered = out.clone()
        tampered[:, :, 0, 0] += 1.0  # corrupt a pixel that must be unchanged
        self.assertFalse(ifeat.verify_unchanged_domain("REGION_BLURRED", self.crop, tampered, self.anchor_bounds))


class RawScoreValidityTests(unittest.TestCase):
    def test_accepts_boundary_values(self):
        scores = ifeat.RawScores(s_original=-1.0, s_context=1.0, s_region=0.0, s_control=0.5)
        self.assertTrue(ifeat.validate_raw_scores(scores))

    def test_rejects_out_of_range(self):
        scores = ifeat.RawScores(s_original=1.0001, s_context=0.0, s_region=0.0, s_control=0.0)
        self.assertFalse(ifeat.validate_raw_scores(scores))

    def test_rejects_nan_and_inf(self):
        for bad in (math.nan, math.inf, -math.inf):
            scores = ifeat.RawScores(s_original=bad, s_context=0.0, s_region=0.0, s_control=0.0)
            self.assertFalse(ifeat.validate_raw_scores(scores))


class DerivedFeatureFormulaTests(unittest.TestCase):
    def _independent_derived(self, s_original, s_context, s_region, s_control):
        context_change = abs(s_context - s_original)
        region_drop = max(0, s_original - s_region)
        control_change = abs(s_control - s_original)
        context_stability = 1 - min(1, context_change / 2)
        specific_region_drop = max(0, region_drop - control_change)
        region_necessity = min(1, specific_region_drop / 2)
        intervention_reliability = context_stability * region_necessity
        return (context_change, region_drop, control_change, context_stability,
                specific_region_drop, region_necessity, intervention_reliability)

    def test_matches_independent_formula_on_several_points(self):
        cases = [
            (0.5, 0.5, 0.5, 0.5),
            (1.0, -1.0, -1.0, 1.0),
            (0.2, 0.9, -0.3, 0.1),
            (-0.7, -0.7, -1.0, 0.3),
        ]
        for s_original, s_context, s_region, s_control in cases:
            raw = ifeat.RawScores(s_original=s_original, s_context=s_context, s_region=s_region, s_control=s_control)
            derived = ifeat.compute_derived_features(raw)
            expected = self._independent_derived(s_original, s_context, s_region, s_control)
            actual = (derived.context_change, derived.region_drop, derived.control_change,
                      derived.context_stability, derived.specific_region_drop,
                      derived.region_necessity, derived.intervention_reliability)
            for e, a in zip(expected, actual):
                self.assertAlmostEqual(e, a, places=12)

    def test_output_range_enforced(self):
        raw = ifeat.RawScores(s_original=1.0, s_context=1.0, s_region=1.0, s_control=1.0)
        derived = ifeat.compute_derived_features(raw)
        self.assertTrue(ifeat.validate_derived_output(derived))
        self.assertGreaterEqual(derived.intervention_reliability, 0.0)
        self.assertLessEqual(derived.intervention_reliability, 1.0)


class ControlFeatureTests(unittest.TestCase):
    def test_control_1_is_identity(self):
        self.assertEqual(ifeat.control_1_top1_unary_score(0.37), 0.37)

    def test_control_2_is_identity(self):
        self.assertEqual(ifeat.control_2_top1_minus_top2_margin(0.12), 0.12)

    def test_control_3_matches_independent_entropy(self):
        torch.manual_seed(1)
        scores = torch.rand(171) * 2 - 1
        expected = -(F.softmax(scores.double(), dim=0) * torch.log(F.softmax(scores.double(), dim=0))).sum().item()
        actual = ifeat.control_3_unary_entropy(scores)
        self.assertAlmostEqual(expected, actual, places=10)

    def test_control_3_rejects_wrong_dimension(self):
        with self.assertRaises(ifeat.InterventionFeatureError):
            ifeat.control_3_unary_entropy(torch.rand(10))

    def test_control_4_matches_independent_cosine(self):
        torch.manual_seed(2)
        anchor = torch.rand(16)
        neighbors = torch.rand(8, 16)
        expected = F.cosine_similarity(anchor.unsqueeze(0).double(), neighbors.mean(dim=0).unsqueeze(0).double()).item()
        actual = ifeat.control_4_dino_neighbor_cosine_similarity(anchor, neighbors)
        self.assertAlmostEqual(expected, actual, places=10)

    def test_control_4_requires_8_neighbors(self):
        with self.assertRaises(ifeat.InterventionFeatureError):
            ifeat.control_4_dino_neighbor_cosine_similarity(torch.rand(4), torch.rand(5, 4))

    def test_control_5_missing_when_no_boundary_patches(self):
        self.assertIsNone(ifeat.control_5_normalized_distance_to_nearest_boundary(10, 10, []))

    def test_control_5_matches_independent_formula(self):
        boundary = [_make_patch(1, 1, 0, 1, 0.0), _make_patch(30, 30, 0, 1, 0.0)]
        row, col = 15, 15
        norm_const = math.sqrt(29 ** 2 + 29 ** 2)
        raw_distance = min(math.hypot(row - 1, col - 1), math.hypot(row - 30, col - 30))
        expected = min(1.0, raw_distance / norm_const)
        actual = ifeat.control_5_normalized_distance_to_nearest_boundary(row, col, boundary)
        self.assertAlmostEqual(expected, actual, places=10)
        self.assertAlmostEqual(ifeat.control_5_normalized_distance_to_nearest_boundary(1, 1, boundary), 0.0, places=10)

    def test_control_5_normalization_constant_is_exact(self):
        self.assertAlmostEqual(math.sqrt(29 ** 2 + 29 ** 2), 41.012193308819754, places=12)

    def test_control_6_matches_independent_agreement_fraction(self):
        grid = _full_grid_uniform(lambda r, c: 1 if (r, c) != (10, 11) else 2)
        p = grid[(10, 10)]
        actual = ifeat.control_6_local_top1_agreement_fraction(p, grid)
        self.assertAlmostEqual(actual, 7 / 8, places=10)

    def test_control_8_is_identity(self):
        self.assertEqual(ifeat.control_8_area_matched_control_change(0.44), 0.44)


class RowAssemblyTests(unittest.TestCase):
    def test_query_correctness_feature_order_is_1_2_3_4_5_6_8(self):
        row = ifeat.assemble_query_correctness_row(
            "img", 0, "517", "q1", 42,
            control_1=0.1, control_2=0.2, control_3=0.3, control_4=0.4,
            control_5=0.5, control_6=0.6, control_8=0.8, intervention_reliability=0.9,
        )
        self.assertEqual(row.feature_a, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8))
        self.assertEqual(row.intervention_reliability, 0.9)

    def test_rejects_bad_query_rank(self):
        with self.assertRaises(ifeat.InterventionFeatureError):
            ifeat.assemble_query_correctness_row(
                "img", 0, "517", "q3", 42,
                control_1=0.1, control_2=0.2, control_3=0.3, control_4=0.4,
                control_5=0.5, control_6=0.6, control_8=0.8, intervention_reliability=0.9,
            )

    def test_row_is_complete_allows_only_control_5_missing(self):
        self.assertTrue(ifeat.row_is_complete((0.1, 0.2, 0.3, 0.4, None, 0.6, 0.8)))
        self.assertFalse(ifeat.row_is_complete((0.1, None, 0.3, 0.4, 0.5, 0.6, 0.8)))
        self.assertFalse(ifeat.row_is_complete((0.1, 0.2, math.nan, 0.4, 0.5, 0.6, 0.8)))
        self.assertFalse(ifeat.row_is_complete((0.1, 0.2, 0.3, 0.4, math.inf, 0.6, 0.8)))


class GtFreeDigestTests(unittest.TestCase):
    def _row(self, anchor_id, query_rank="q1", query_class=1):
        return ifeat.assemble_query_correctness_row(
            "img", 0, anchor_id, query_rank, query_class,
            control_1=0.1, control_2=0.2, control_3=0.3, control_4=0.4,
            control_5=0.5, control_6=0.6, control_8=0.8, intervention_reliability=0.9,
        )

    def test_deterministic(self):
        rows = [self._row("1"), self._row("2")]
        self.assertEqual(ifeat.gt_free_feature_digest(rows), ifeat.gt_free_feature_digest(rows))

    def test_sensitive_to_row_order(self):
        rows = [self._row("1"), self._row("2")]
        reversed_rows = list(reversed(rows))
        self.assertNotEqual(ifeat.gt_free_feature_digest(rows), ifeat.gt_free_feature_digest(reversed_rows))

    def test_sensitive_to_missing_control_5(self):
        with_c5 = [self._row("1")]
        without_c5 = [ifeat.assemble_query_correctness_row(
            "img", 0, "1", "q1", 1,
            control_1=0.1, control_2=0.2, control_3=0.3, control_4=0.4,
            control_5=None, control_6=0.6, control_8=0.8, intervention_reliability=0.9,
        )]
        self.assertNotEqual(ifeat.gt_free_feature_digest(with_c5), ifeat.gt_free_feature_digest(without_c5))

    def test_sensitive_to_content_change(self):
        base = [self._row("1")]
        changed = [self._row("1", query_class=2)]
        self.assertNotEqual(ifeat.gt_free_feature_digest(base), ifeat.gt_free_feature_digest(changed))


if __name__ == "__main__":
    unittest.main()
