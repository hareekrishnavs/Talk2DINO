import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "src" / "open_vocabulary_segmentation" / "models"
OVS = ROOT / "src" / "open_vocabulary_segmentation"
sys.path.insert(0, str(MODELS))
sys.path.insert(0, str(OVS))

from opc_calibrator import ObjectPresenceCalibrator
from class_prototype_alignment import ClassPrototypeAlignmentHead
from train_xattn_clean import (
    CleanXAttnBridge,
    assert_opc_vocabulary_axis,
    build_opc_vocabulary_evidence,
    opc_training_losses,
)


class ObjectPresenceCalibratorTest(unittest.TestCase):
    def make_inputs(self, batch=2, classes=8, regions=12, patches=32, dim=16):
        region_base = torch.randn(batch, classes, regions)
        region_cpa = region_base + 0.1 * torch.randn(batch, classes, regions)
        mapped_text = torch.randn(batch, classes, dim)
        dense_cpa = torch.randn(batch, classes, patches)
        return region_base, region_cpa, mapped_text, dense_cpa

    def test_zero_init_is_exact_cpa_parity(self):
        opc = ObjectPresenceCalibrator(hidden_dim=8, init_zero=True)
        inputs = self.make_inputs()
        final, bias, stats = opc(*inputs, return_stats=True)
        self.assertTrue(torch.equal(final, inputs[-1]))
        self.assertEqual(float(bias.detach().abs().max()), 0.0)
        self.assertEqual(float(stats["opc_bias_abs_max"]), 0.0)
        self.assertEqual(opc.build_features(*inputs[:3]).shape, (2, 8, 15))

    def test_bias_shape_clip_and_shared_text(self):
        opc = ObjectPresenceCalibrator(
            hidden_dim=8,
            bias_scale=1.0,
            bias_clip=0.2,
            init_zero=False,
        )
        with torch.no_grad():
            opc.mlp[-1].weight.fill_(10.0)
            opc.mlp[-1].bias.fill_(10.0)
        region_base, region_cpa, mapped_text, dense_cpa = self.make_inputs()
        final, bias = opc(region_base, region_cpa, mapped_text[0], dense_cpa)
        self.assertEqual(bias.shape, (2, 8, 1))
        self.assertEqual(final.shape, dense_cpa.shape)
        self.assertLessEqual(float(bias.detach().abs().max()), 0.200001)

    def test_zero_init_ranking_loss_trains_final_layer(self):
        opc = ObjectPresenceCalibrator(hidden_dim=8, init_zero=True)
        region_base, region_cpa, mapped_text, dense_cpa = self.make_inputs()
        final, bias = opc(region_base, region_cpa, mapped_text, dense_cpa)
        cfg = OmegaConf.create({
            "margin": 0.1,
            "top_k_pos": 2,
            "top_k_neg": 3,
            "temperature": 1.0,
        })
        losses = opc_training_losses(dense_cpa, final, bias, region_cpa, cfg)
        losses["loss_opc_rank"].backward()
        self.assertGreater(float(opc.mlp[-1].weight.grad.abs().sum()), 0.0)
        self.assertEqual(sum(p.numel() for p in opc.parameters()), 167)

    def test_training_evidence_uses_vocabulary_class_axis(self):
        bridge = CleanXAttnBridge(
            clip_dim=16,
            dino_dim=32,
            d_model=8,
            num_heads=2,
        )
        cpa = ClassPrototypeAlignmentHead(
            dino_dim=32,
            num_prototypes=2,
            hidden_dim=8,
            topk=3,
        )
        class_clip = torch.randn(7, 2, 16)
        class_base = torch.randn(7, 2, 32)
        semantic = torch.randn(3, 4, 32)
        mapped, region_base, region_cpa = build_opc_vocabulary_evidence(
            bridge,
            cpa,
            class_clip,
            class_base,
            semantic,
            delta_scale=0.5,
        )
        self.assertEqual(mapped.shape, (3, 7, 32))
        self.assertEqual(region_base.shape, (3, 7, 4))
        self.assertEqual(region_cpa.shape, (3, 7, 4))

        valid = torch.randn(3, 171, 12)
        assert_opc_vocabulary_axis(valid, valid, vocab_size=171, batch_size=3)
        with self.assertRaisesRegex(RuntimeError, "eval-vocabulary evidence"):
            batch_caption_evidence = torch.randn(3, 3, 12)
            assert_opc_vocabulary_axis(
                batch_caption_evidence,
                batch_caption_evidence,
                vocab_size=171,
                batch_size=3,
            )


if __name__ == "__main__":
    unittest.main()
