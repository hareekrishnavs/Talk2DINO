import sys
import unittest
from pathlib import Path

import torch


MODELS = Path(__file__).resolve().parents[1] / "src" / "open_vocabulary_segmentation" / "models"
sys.path.insert(0, str(MODELS))

from cars_cpa import ClassAdaptiveResidualScaler
from class_prototype_alignment import apply_topk_prototype_residual


class ClassAdaptiveResidualScalerTest(unittest.TestCase):
    def test_zero_init_exactly_matches_fixed_cpa(self):
        torch.manual_seed(4)
        base = torch.randn(2, 5, 9)
        prototype = torch.randn(2, 5, 9)
        mapped = torch.randn(2, 5, 8)
        fixed, fixed_stats = apply_topk_prototype_residual(
            base,
            prototype,
            topk=3,
            residual_scale=0.25,
            residual_clip=0.5,
        )
        cars = ClassAdaptiveResidualScaler(dino_dim=8, base_scale=0.25)

        adaptive, alpha, stats = cars(
            mapped,
            base,
            prototype,
            fixed_stats["cpa_topk_mask"],
            residual_clip=0.5,
        )

        self.assertTrue(torch.equal(adaptive, fixed))
        self.assertTrue(torch.equal(alpha, torch.full_like(alpha, 0.25)))
        self.assertEqual(float(stats["cars_delta_abs_max"]), 0.0)
        self.assertEqual(float(stats["cars_changed_fraction"]), 0.0)

    def test_zero_init_matches_fixed_cpa_under_autocast(self):
        torch.manual_seed(9)
        base = torch.randn(2, 5, 9)
        prototype = torch.randn(2, 5, 9)
        mapped = torch.randn(2, 5, 8)
        cars = ClassAdaptiveResidualScaler(dino_dim=8, base_scale=0.25)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            fixed, fixed_stats = apply_topk_prototype_residual(
                base,
                prototype,
                topk=3,
                residual_scale=0.25,
                residual_clip=0.5,
            )
            adaptive, _, _ = cars(
                mapped,
                base,
                prototype,
                fixed_stats["cpa_topk_mask"],
                residual_clip=0.5,
            )

        self.assertTrue(torch.equal(adaptive, fixed))

    def test_unbatched_text_broadcasts_over_images(self):
        cars = ClassAdaptiveResidualScaler(dino_dim=6)
        mapped = torch.randn(4, 6)
        base = torch.randn(2, 4, 7)
        prototype = torch.randn(2, 4, 7)
        mask = torch.ones_like(base, dtype=torch.bool)

        final, alpha, _ = cars(mapped, base, prototype, mask)

        self.assertEqual(tuple(final.shape), tuple(base.shape))
        self.assertEqual(tuple(alpha.shape), (4, 1))

    def test_only_topk_residual_is_modified(self):
        cars = ClassAdaptiveResidualScaler(dino_dim=4, init_zero=False)
        with torch.no_grad():
            cars.scaler[-1].weight.fill_(0.1)
            cars.scaler[-1].bias.fill_(0.2)
        mapped = torch.randn(1, 3, 4)
        base = torch.randn(1, 3, 5)
        prototype = torch.randn(1, 3, 5)
        mask = torch.zeros_like(base, dtype=torch.bool)
        mask[:, 1, :2] = True

        final, alpha, stats = cars(mapped, base, prototype, mask)

        self.assertTrue(torch.equal(final[~mask], base[~mask]))
        self.assertGreater(float(stats["cars_changed_fraction"]), 0.0)
        self.assertTrue(torch.all(alpha >= cars.alpha_min))
        self.assertTrue(torch.all(alpha <= cars.alpha_max))

    def test_final_linear_is_zero_initialized(self):
        cars = ClassAdaptiveResidualScaler(dino_dim=8, hidden_dim=4)
        self.assertTrue(torch.equal(cars.scaler[-1].weight, torch.zeros_like(cars.scaler[-1].weight)))
        self.assertTrue(torch.equal(cars.scaler[-1].bias, torch.zeros_like(cars.scaler[-1].bias)))

    def test_fixed_alpha_025_exactly_matches_cpa_and_bypasses_mlp(self):
        base = torch.randn(1, 4, 7)
        prototype = torch.randn(1, 4, 7)
        mapped = torch.randn(4, 6)
        fixed, fixed_stats = apply_topk_prototype_residual(
            base,
            prototype,
            topk=2,
            residual_scale=0.25,
            residual_clip=0.5,
        )
        cars = ClassAdaptiveResidualScaler(
            dino_dim=6,
            force_fixed_alpha=True,
            fixed_alpha=0.25,
        )

        def fail_if_called(*args, **kwargs):
            raise AssertionError("CARS MLP must be bypassed in fixed-alpha mode")

        cars.scaler.forward = fail_if_called
        adaptive, alpha, _ = cars(
            mapped,
            base,
            prototype,
            fixed_stats["cpa_topk_mask"],
        )

        self.assertTrue(torch.equal(adaptive, fixed))
        self.assertTrue(torch.equal(alpha, torch.full_like(alpha, 0.25)))

    def test_fixed_alpha_stats_reflect_requested_value(self):
        cars = ClassAdaptiveResidualScaler(
            dino_dim=4,
            force_fixed_alpha=True,
            fixed_alpha=0.20,
        )
        mapped = torch.randn(3, 4)
        base = torch.randn(1, 3, 5)
        prototype = torch.randn(1, 3, 5)
        mask = torch.ones_like(base, dtype=torch.bool)

        _, alpha, stats = cars(mapped, base, prototype, mask)

        self.assertTrue(torch.equal(alpha, torch.full_like(alpha, 0.20)))
        self.assertAlmostEqual(float(stats["cars_alpha_mean"]), 0.20, places=6)
        self.assertAlmostEqual(float(stats["cars_delta_abs_mean"]), 0.05, places=6)


if __name__ == "__main__":
    unittest.main()
