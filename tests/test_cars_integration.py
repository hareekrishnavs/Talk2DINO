import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
OVS = ROOT / "src" / "open_vocabulary_segmentation"
sys.path.insert(0, str(OVS))
sys.path.insert(0, str(OVS / "models"))

from cars_cpa import ClassAdaptiveResidualScaler
from class_prototype_alignment import ClassPrototypeAlignmentHead
from eval_xattn_clean import load_cars_from_payload
from train_xattn_clean import cpa_auxiliary_losses, configure_trainable_modules


class CarsIntegrationTest(unittest.TestCase):
    def test_phase1_only_cars_is_trainable(self):
        frozen = nn.Linear(3, 3)
        bridge = nn.Linear(3, 3)
        cpa = ClassPrototypeAlignmentHead(dino_dim=8, hidden_dim=4)
        cars = ClassAdaptiveResidualScaler(dino_dim=8, hidden_dim=4)
        cfg = OmegaConf.create({
            "train_only_cars": True,
            "freeze_existing": True,
        })

        named = configure_trainable_modules(frozen, bridge, cpa, cars, cfg)

        self.assertTrue(named)
        self.assertTrue(all(name.startswith("cars.") for name, _ in named))
        self.assertEqual(sum(p.requires_grad for p in frozen.parameters()), 0)
        self.assertEqual(sum(p.requires_grad for p in bridge.parameters()), 0)
        self.assertEqual(sum(p.requires_grad for p in cpa.parameters()), 0)
        self.assertEqual(
            sum(p.numel() for _, p in named),
            sum(p.numel() for p in cars.parameters()),
        )

    def test_cars_scaled_region_infonce_reaches_cars_parameters(self):
        torch.manual_seed(3)
        cpa = ClassPrototypeAlignmentHead(
            dino_dim=8,
            num_prototypes=2,
            hidden_dim=4,
        )
        cpa.requires_grad_(False)
        cars = ClassAdaptiveResidualScaler(dino_dim=8, hidden_dim=4)
        mapped = torch.randn(3, 3, 8)
        visual = torch.randn(3, 5, 8)
        regions = torch.randn(3, 6, 8)
        base_logits = torch.randn(3, 3, 6)
        cpa_loss_cfg = OmegaConf.create({
            "confident_margin": 0.2,
            "temperature": 2.0,
            "diversity_margin": 0.9,
        })
        cars_loss_cfg = OmegaConf.create({
            "enabled": False,
            "alpha_center": 0.25,
            "preserve_temperature": 1.0,
        })

        losses = cpa_auxiliary_losses(
            cpa,
            mapped,
            visual,
            regions,
            base_logits,
            cpa_loss_cfg,
            0.07,
            cars=cars,
            cars_loss_cfg=cars_loss_cfg,
        )
        losses["loss_cpa_proto_infonce"].backward()

        grad = cars.scaler[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)
        self.assertEqual(float(losses["loss_cars_alpha_l2"]), 0.0)
        self.assertEqual(float(losses["loss_cars_alpha_center"]), 0.0)
        self.assertEqual(float(losses["loss_cars_preserve"]), 0.0)

    def test_eval_cars_requires_checkpoint_weights(self):
        cfg = OmegaConf.create({
            "bridge": {"dino_dim": 8},
            "cars": {
                "enabled": True,
                "hidden_dim": 4,
                "dropout": 0.0,
                "base_scale": 0.25,
                "delta_scale_max": 0.1,
                "alpha_min": 0.05,
                "alpha_max": 0.5,
                "init_zero": True,
                "allow_missing_init": False,
                "force_fixed_alpha": False,
                "fixed_alpha": 0.25,
            },
        })
        with self.assertRaisesRegex(RuntimeError, "does not contain CARS weights"):
            load_cars_from_payload(cfg, {}, "cpu", enabled=True)

    def test_eval_allow_missing_and_fixed_alpha_need_no_weights(self):
        base = {
            "enabled": True,
            "hidden_dim": 4,
            "dropout": 0.0,
            "base_scale": 0.25,
            "delta_scale_max": 0.1,
            "alpha_min": 0.05,
            "alpha_max": 0.5,
            "init_zero": True,
            "fixed_alpha": 0.25,
        }
        missing_cfg = OmegaConf.create({
            "bridge": {"dino_dim": 8},
            "cars": {
                **base,
                "allow_missing_init": True,
                "force_fixed_alpha": False,
            },
        })
        fixed_cfg = OmegaConf.create({
            "bridge": {"dino_dim": 8},
            "cars": {
                **base,
                "allow_missing_init": False,
                "force_fixed_alpha": True,
            },
        })

        missing = load_cars_from_payload(missing_cfg, {}, "cpu", enabled=True)
        fixed = load_cars_from_payload(fixed_cfg, {}, "cpu", enabled=True)

        self.assertEqual(missing.mode, "zero_init_missing")
        self.assertEqual(fixed.mode, "fixed_alpha")


if __name__ == "__main__":
    unittest.main()
