import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


MODELS = Path(__file__).resolve().parents[1] / "src/open_vocabulary_segmentation/models"
sys.path.insert(0, str(MODELS))

from class_prototype_alignment import apply_topk_prototype_residual
from rvs_cpa import RegionVisualSemanticCPA, extract_rvs_regions


class FakeCropEncoder:
    output_dim = 3
    device = torch.device("cpu")
    masked_crop = False

    def __init__(self):
        self.calls = 0

    def encode(self, image, boxes, masks=None):
        self.calls += 1
        feature = torch.tensor([1.0, 0.0, 0.0])
        return feature.repeat(len(boxes), 1)


def _config(**overrides):
    values = {
        "mode": "region_gate",
        "region_source": "tuned_cpa_argmax",
        "topk": 2,
        "residual_scale": 0.5,
        "residual_clip": 0.5,
        "min_region_area_ratio": 0.05,
        "max_region_area_ratio": 0.8,
        "max_regions_per_image": 8,
        "min_region_confidence": 0.0,
        "text_source": "clip_text",
        "candidate_text_source": "eval_class_names",
        "compare_scope": "candidate_topk",
        "margin_type": "top2",
        "gate_type": "margin_tanh",
        "gate_strength": 0.0,
        "margin_temperature": 0.05,
        "gate_min": 0.9,
        "gate_max": 1.1,
        "apply_to": "cpa_residual",
        "region_fill": "connected_component",
        "fallback_gate": 1.0,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def _cpa_inputs():
    base = torch.zeros(1, 3, 4, 4)
    base[:, 0, :, :2] = 2.0
    base[:, 1, :, 2:] = 2.0
    prototype = base.clone()
    prototype[:, 0] += 0.2
    prototype[:, 1] -= 0.1
    tuned, stats = apply_topk_prototype_residual(
        base,
        prototype,
        topk=2,
        residual_scale=0.5,
        residual_clip=0.5,
    )
    return base, tuned, stats


def test_gate_strength_zero_is_exact_tuned_cpa_parity():
    base, tuned, stats = _cpa_inputs()
    encoder = FakeCropEncoder()
    rvs = RegionVisualSemanticCPA(
        encoder,
        torch.eye(3),
        ["class0", "class1", "class2"],
        _config(gate_strength=0.0),
    )
    final, rvs_stats = rvs(base, tuned, stats, torch.zeros(3, 32, 32))
    assert torch.equal(final, tuned)
    assert rvs_stats["rvs_gate_strength_zero_max_abs_diff"] == 0.0
    assert rvs_stats["rvs_zero_strength_shortcut_used"] is True
    assert encoder.calls == 0


def test_no_valid_regions_returns_same_tensor():
    base, tuned, stats = _cpa_inputs()
    rvs = RegionVisualSemanticCPA(
        FakeCropEncoder(),
        torch.eye(3),
        ["class0", "class1", "class2"],
        _config(min_region_area_ratio=0.9, max_region_area_ratio=1.0),
    )
    final, rvs_stats = rvs(base, tuned, stats, torch.zeros(3, 32, 32))
    assert final is tuned
    assert rvs_stats["rvs_regions"] == 0


def test_whole_image_hwc_rgb_is_encoded_once():
    base, tuned, stats = _cpa_inputs()
    encoder = FakeCropEncoder()
    rvs = RegionVisualSemanticCPA(
        encoder,
        torch.eye(3),
        ["class0", "class1", "class2"],
        _config(gate_strength=0.1),
    )
    final, rvs_stats = rvs(
        base,
        tuned,
        stats,
        np.zeros((40, 60, 3), dtype=np.uint8),
    )
    assert final.shape == tuned.shape
    assert rvs_stats["rvs_regions"] > 0
    assert encoder.calls == 1


def test_region_extraction_splits_disconnected_components():
    logits = torch.zeros(1, 2, 4, 4)
    logits[:, 1] = 1.0
    logits[:, 0, 0, 0] = 2.0
    logits[:, 0, 3, 3] = 2.0
    regions, discovered = extract_rvs_regions(
        logits,
        topk_indices=None,
        min_area_ratio=0.0,
        max_area_ratio=1.0,
        max_regions=8,
    )
    class_zero = [region for region in regions if region["class_id"] == 0]
    assert discovered == 3
    assert len(class_zero) == 2


def test_invalid_fallback_gate_fails_clearly():
    with pytest.raises(ValueError, match="fallback_gate"):
        RegionVisualSemanticCPA(
            FakeCropEncoder(),
            torch.eye(3),
            ["class0", "class1", "class2"],
            _config(fallback_gate=0.9),
        )
