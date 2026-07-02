import sys
import unittest
from pathlib import Path

import torch


MODELS = Path(__file__).resolve().parents[1] / "src" / "open_vocabulary_segmentation" / "models"
sys.path.insert(0, str(MODELS))

from visual_prototype_alignment import VisualPrototypeAlignment


def two_region_inputs():
    logits = torch.full((1, 4, 40), -8.0)
    logits[:, 0, :20] = torch.linspace(8.0, 6.0, 20)
    logits[:, 1, 20:] = torch.linspace(8.0, 6.0, 20)
    features = torch.zeros(1, 40, 4)
    features[:, :20, 0] = 1.0
    features[:, 20:, 1] = 1.0
    return logits, features


class VisualPrototypeAlignmentTest(unittest.TestCase):
    def test_lambda_zero_is_exact_noop_and_reports_invariant(self):
        vpa = VisualPrototypeAlignment(fusion_lambda=0.0)
        logits = torch.randn(2, 6, 16)
        features = torch.randn(2, 16, 8)

        final, stats = vpa(logits, features)

        self.assertIs(final, logits)
        self.assertTrue(torch.equal(final, logits))
        self.assertEqual(float(stats["vpa_lambda_zero_max_abs_diff"]), 0.0)
        self.assertEqual(float(stats["vpa_changed_fraction"]), 0.0)
        self.assertEqual(sum(p.numel() for p in vpa.parameters()), 0)

    def test_rank_seeds_produce_positive_gated_correction(self):
        vpa = VisualPrototypeAlignment(
            fusion_lambda=0.03,
            class_topk_patches=10,
            max_classes=2,
            patch_topk=1,
            use_min_seed_prob=False,
            min_seed_prob=1.0,
            seed_percentile=50,
            min_seed_pixels=8,
            max_seed_pixels=10,
            prototype_temperature=0.2,
        )
        logits, features = two_region_inputs()

        final, stats = vpa(logits, features)
        correction = final - logits
        top1_mask = torch.zeros_like(logits, dtype=torch.bool).scatter(
            1,
            logits.topk(1, dim=1).indices,
            True,
        )

        self.assertTrue(
            torch.equal(
                correction[~top1_mask],
                torch.zeros_like(correction[~top1_mask]),
            )
        )
        self.assertTrue(torch.equal(final[:, 2:], logits[:, 2:]))
        self.assertGreaterEqual(float(correction.min()), 0.0)
        self.assertLessEqual(float(correction.max()), 0.030001)
        self.assertGreater(float(stats["vpa_valid_classes_mean"]), 0.0)
        self.assertGreater(float(stats["vpa_changed_fraction"]), 0.0)
        self.assertEqual(float(stats["vpa_correction_negative_fraction"]), 0.0)
        self.assertEqual(float(stats["vpa_skip_low_prob_total"]), 0.0)
        self.assertGreaterEqual(float(stats["vpa_seed_pixels_min"]), 8.0)
        self.assertLessEqual(float(stats["vpa_seed_pixels_max"]), 10.0)

    def test_optional_probability_gate_is_only_applied_when_enabled(self):
        logits, features = two_region_inputs()
        vpa = VisualPrototypeAlignment(
            max_classes=2,
            patch_topk=1,
            use_min_seed_prob=True,
            min_seed_prob=1.0,
            seed_percentile=50,
            min_seed_pixels=8,
        )

        final, stats = vpa(logits, features)

        self.assertTrue(torch.equal(final, logits))
        self.assertEqual(float(stats["vpa_valid_classes_mean"]), 0.0)
        self.assertGreater(float(stats["vpa_skip_low_prob_total"]), 0.0)

    def test_too_few_ranked_seed_pixels_skips_class(self):
        vpa = VisualPrototypeAlignment(
            max_classes=4,
            patch_topk=1,
            seed_percentile=90,
            min_seed_pixels=8,
        )
        logits = torch.full((1, 4, 20), -8.0)
        for class_index in range(4):
            start = class_index * 5
            logits[:, class_index, start:start + 5] = 8.0
        features = torch.randn(1, 20, 6)

        final, stats = vpa(logits, features)

        self.assertTrue(torch.equal(final, logits))
        self.assertEqual(float(stats["vpa_valid_classes_mean"]), 0.0)
        self.assertEqual(float(stats["vpa_seed_pixels_mean"]), 0.0)
        self.assertEqual(float(stats["vpa_skip_too_few_pixels_total"]), 4.0)
        self.assertEqual(float(stats["vpa_no_valid_prototype_images"]), 1.0)

    def test_missing_dense_features_fails_clearly(self):
        vpa = VisualPrototypeAlignment()
        with self.assertRaisesRegex(RuntimeError, "in-memory dense DINO patch features"):
            vpa(torch.randn(1, 4, 9), None)

    def test_v3_rejects_unsupported_modes(self):
        with self.assertRaisesRegex(ValueError, "rank_percentile"):
            VisualPrototypeAlignment(seed_selection="absolute_probability")
        with self.assertRaisesRegex(ValueError, "positive_only=true"):
            VisualPrototypeAlignment(positive_only=False)
        with self.assertRaisesRegex(ValueError, "positive_quantile"):
            VisualPrototypeAlignment(correction_mode="zscore")


if __name__ == "__main__":
    unittest.main()
