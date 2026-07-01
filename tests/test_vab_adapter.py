import sys
import unittest
from pathlib import Path

import torch


MODELS = Path(__file__).resolve().parents[1] / "src" / "open_vocabulary_segmentation" / "models"
sys.path.insert(0, str(MODELS))

from vab_adapter import VocabularyAwareBridgeAdapter


class VocabularyAwareBridgeAdapterTest(unittest.TestCase):
    def make_adapter(self):
        return VocabularyAwareBridgeAdapter(
            dino_dim=32,
            hidden_dim=8,
            num_heads=4,
            gamma_init=0.01,
            gamma_max=0.03,
            delta_ratio_clip=0.05,
        )

    def test_zero_output_initialization_is_exact_noop(self):
        adapter = self.make_adapter()
        mapped = torch.randn(2, 5, 32)
        regions = torch.randn(2, 12, 32, requires_grad=True)
        output, stats = adapter(mapped, regions, return_stats=True)
        self.assertTrue(torch.equal(output, mapped))
        self.assertEqual(float(stats["vab_delta_base_ratio_max"]), 0.0)
        output.sum().backward()
        self.assertIsNone(regions.grad)
        self.assertGreater(float(adapter.mlp[-1].weight.grad.abs().sum()), 0.0)

    def test_shapes_ratio_cap_and_trainable_gradient(self):
        adapter = self.make_adapter()
        torch.nn.init.normal_(adapter.mlp[-1].weight, std=0.1)
        torch.nn.init.normal_(adapter.mlp[-1].bias, std=0.1)
        regions = torch.randn(3, 12, 32)

        aligned = torch.randn(3, 32)
        aligned_out, stats = adapter(aligned, regions, return_stats=True)
        self.assertEqual(aligned_out.shape, aligned.shape)
        self.assertLessEqual(float(stats["vab_delta_base_ratio_max"]), 0.050001)

        conditioned = torch.randn(3, 7, 32)
        conditioned_out = adapter(conditioned, regions)
        self.assertEqual(conditioned_out.shape, conditioned.shape)

        loss = conditioned_out.square().sum()
        loss.backward()
        self.assertIsNotNone(adapter.mlp[-1].weight.grad)
        self.assertGreater(float(adapter.mlp[-1].weight.grad.abs().sum()), 0.0)

    def test_shared_vocabulary_shape_for_one_image(self):
        adapter = self.make_adapter()
        mapped = torch.randn(171, 32)
        regions = torch.randn(1, 12, 32)
        output = adapter(mapped, regions)
        self.assertEqual(output.shape, mapped.shape)


if __name__ == "__main__":
    unittest.main()
