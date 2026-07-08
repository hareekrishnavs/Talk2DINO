import sys
from pathlib import Path

import torch


EVALUATION = (
    Path(__file__).resolve().parents[1]
    / "src/open_vocabulary_segmentation/segmentation/evaluation"
)
sys.path.insert(0, str(EVALUATION))

from multiscale_eval import aggregate_legacy_probability_maps


def test_single_scale_returns_exact_legacy_map():
    legacy = torch.rand(1, 3, 5, 7)
    output, per_scale = aggregate_legacy_probability_maps([legacy], 1)
    assert output is legacy
    assert per_scale == [legacy]


def test_scale_and_flip_averaging_order():
    outputs = [
        torch.full((1, 2, 2, 2), value)
        for value in (1.0, 3.0, 5.0, 7.0)
    ]
    final, per_scale = aggregate_legacy_probability_maps(
        outputs,
        num_scales=2,
        hflip=True,
    )
    assert torch.equal(per_scale[0], torch.full_like(outputs[0], 2.0))
    assert torch.equal(per_scale[1], torch.full_like(outputs[0], 6.0))
    assert torch.equal(final, torch.full_like(outputs[0], 4.0))
