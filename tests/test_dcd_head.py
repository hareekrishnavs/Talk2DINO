import importlib.util
import pathlib
import unittest

import torch


MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "src/open_vocabulary_segmentation/models/dcd_head.py"
)
SPEC = importlib.util.spec_from_file_location("dcd_head", MODULE_PATH)
dcd_head = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dcd_head)


class DenseConsistencyDistillationHeadTest(unittest.TestCase):
    def test_zero_init_parity_and_attention_normalization(self):
        torch.manual_seed(7)
        head = dcd_head.DenseConsistencyDistillationHead(
            dino_dim=768,
            hidden_dim=32,
            scale=0.05,
            residual_clip=0.25,
            topk=1,
            topk_source="cpa",
        )
        final_linear = head.adapter[-1]
        self.assertTrue(torch.equal(final_linear.weight, torch.zeros_like(final_linear.weight)))
        self.assertTrue(torch.equal(final_linear.bias, torch.zeros_like(final_linear.bias)))

        semantic_region_features = torch.randn(2, 12, 768)
        mapped_text = torch.randn(2, 768)
        attention_lift = torch.rand(2, 12, 1024)
        dense_cpa_v1_logits = torch.randn(2, 2, 1024)

        final_logits, stats = head(
            semantic_region_features,
            mapped_text,
            attention_lift,
            dense_cpa_v1_logits,
        )

        self.assertEqual(tuple(final_logits.shape), (2, 2, 1024))
        self.assertTrue(torch.allclose(final_logits, dense_cpa_v1_logits, atol=0.0, rtol=0.0))
        self.assertEqual(float(stats["dcd_region_residual_abs_max"]), 0.0)
        self.assertEqual(float(stats["dcd_dense_residual_abs_max"]), 0.0)
        self.assertTrue(torch.allclose(
            head.normalize_attention_lift(attention_lift).sum(dim=1),
            torch.ones(2, 1024),
            atol=1e-6,
        ))

    def test_non_topk_logits_are_unchanged(self):
        head = dcd_head.DenseConsistencyDistillationHead(
            dino_dim=4,
            hidden_dim=4,
            scale=1.0,
            residual_clip=10.0,
            topk=1,
            topk_source="cpa",
        )
        final_linear = head.adapter[-1]
        torch.nn.init.eye_(final_linear.weight)
        torch.nn.init.ones_(final_linear.bias)

        semantic_region_features = torch.ones(1, 12, 4)
        mapped_text = torch.eye(4)[:3]
        attention_lift = torch.ones(1, 12, 1024)
        dense_cpa_v1_logits = torch.zeros(1, 3, 1024)
        dense_cpa_v1_logits[:, 1, :] = 5.0

        final_logits, _ = head(
            semantic_region_features,
            mapped_text,
            attention_lift,
            dense_cpa_v1_logits,
        )
        self.assertTrue(torch.equal(final_logits[:, 0, :], dense_cpa_v1_logits[:, 0, :]))
        self.assertFalse(torch.equal(final_logits[:, 1, :], dense_cpa_v1_logits[:, 1, :]))
        self.assertTrue(torch.equal(final_logits[:, 2, :], dense_cpa_v1_logits[:, 2, :]))

    def test_image_conditioned_mapped_text_shape(self):
        head = dcd_head.DenseConsistencyDistillationHead(dino_dim=8, hidden_dim=4)
        semantic_region_features = torch.randn(2, 12, 8)
        mapped_text = torch.randn(2, 3, 8)
        attention_lift = torch.rand(2, 12, 1024)
        dense_cpa_v1_logits = torch.randn(2, 3, 1024)

        final_logits, _ = head(
            semantic_region_features,
            mapped_text,
            attention_lift,
            dense_cpa_v1_logits,
        )
        self.assertEqual(tuple(final_logits.shape), (2, 3, 1024))


if __name__ == "__main__":
    unittest.main()
