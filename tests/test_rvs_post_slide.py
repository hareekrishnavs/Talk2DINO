import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


pytest.importorskip("mmcv")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/open_vocabulary_segmentation"))

from segmentation.evaluation.dinotext_seg import DINOTextSegInference


class FakePostSlideModel(nn.Module):
    post_slide_component_cpa_enabled = True
    post_slide_component_cpa_requires_rgb = False

    def __init__(self):
        super().__init__()
        self.crop_calls = 0
        self.rvs_calls = 0
        self.full_base = None

    def generate_masks(self, image, *args, **kwargs):
        assert kwargs.get("return_cpa_components") is True
        self.crop_calls += 1
        value = image[0, 0, 0, 0]
        shape = (1, 2, *image.shape[-2:])
        base = torch.ones(shape, device=image.device) * value
        return {
            "base_logits": base,
            "prototype_logits": base + 0.25,
        }

    def apply_component_cpa_after_slide_aggregation(
        self,
        base,
        prototype,
        rgb_image,
    ):
        self.rvs_calls += 1
        self.full_base = base.clone()
        assert rgb_image is None
        return torch.sigmoid(base)


def test_slide_components_are_averaged_before_single_rvs_call():
    model = FakePostSlideModel()
    inference = DINOTextSegInference(
        model,
        torch.zeros(2, 1),
        ["class0", "class1"],
        with_bg=False,
        test_cfg={"mode": "slide", "stride": (2, 2), "crop_size": (4, 4)},
        pamr=False,
        sg_gate={"enabled": False},
    )
    image = torch.arange(6, dtype=torch.float32).view(1, 1, 1, 6)
    image = image.expand(1, 3, 4, 6).clone()
    metadata = [{
        "img_shape": (4, 6, 3),
        "ori_shape": (4, 6, 3),
        "flip": False,
    }]

    output = inference.slide_inference(image, metadata, rescale=False)

    assert model.crop_calls == 2
    assert model.rvs_calls == 1
    assert output.shape == (1, 2, 4, 6)
    assert torch.equal(
        model.full_base[0, 0, 0],
        torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]),
    )
