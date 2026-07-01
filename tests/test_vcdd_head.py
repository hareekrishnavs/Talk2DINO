import importlib.util
import pathlib
import unittest

import torch


MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "src/open_vocabulary_segmentation/models/vcdd_head.py"
)
SPEC = importlib.util.spec_from_file_location("vcdd_head", MODULE_PATH)
vcdd_head = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vcdd_head)


class VocabularyConditionedDenseDistillationHeadTest(unittest.TestCase):
    def test_zero_init_parity(self):
        head = vcdd_head.VocabularyConditionedDenseDistillationHead(
            dino_dim=16,
            hidden_dim=8,
            topk=3,
        )
        self.assertTrue(torch.equal(head.adapter[-1].weight, torch.zeros_like(head.adapter[-1].weight)))
        self.assertTrue(torch.equal(head.adapter[-1].bias, torch.zeros_like(head.adapter[-1].bias)))
        semantic = torch.randn(2, 12, 16)
        text = torch.randn(2, 5, 16)
        attention = torch.rand(2, 12, 1024)
        cpa_logits = torch.randn(2, 5, 1024)
        final, stats = head(semantic, text, attention, cpa_logits)
        self.assertEqual(tuple(final.shape), (2, 5, 1024))
        self.assertTrue(torch.allclose(final, cpa_logits, atol=0.0, rtol=0.0))
        self.assertEqual(float(stats["vcdd_dense_residual_abs_max"]), 0.0)
        self.assertTrue(torch.allclose(
            head.normalize_attention_lift(attention).sum(dim=1),
            torch.ones(2, 1024),
            atol=1e-6,
        ))

    def test_non_topk_logits_unchanged(self):
        head = vcdd_head.VocabularyConditionedDenseDistillationHead(
            dino_dim=4,
            hidden_dim=4,
            scale=1.0,
            residual_clip=10.0,
            topk=1,
            topk_source="cpa",
        )
        torch.nn.init.eye_(head.adapter[-1].weight)
        torch.nn.init.ones_(head.adapter[-1].bias)
        semantic = torch.ones(1, 12, 4)
        text = torch.eye(4)[:3]
        attention = torch.ones(1, 12, 1024)
        cpa_logits = torch.zeros(1, 3, 1024)
        cpa_logits[:, 1, :] = 5.0
        final, _ = head(semantic, text, attention, cpa_logits)
        self.assertTrue(torch.equal(final[:, 0, :], cpa_logits[:, 0, :]))
        self.assertFalse(torch.equal(final[:, 1, :], cpa_logits[:, 1, :]))
        self.assertTrue(torch.equal(final[:, 2, :], cpa_logits[:, 2, :]))


if __name__ == "__main__":
    unittest.main()
