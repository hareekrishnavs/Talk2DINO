import sys
from pathlib import Path

import pytest
import torch


MODELS = Path(__file__).resolve().parents[1] / "src/open_vocabulary_segmentation/models"
sys.path.insert(0, str(MODELS))

from class_prototype_alignment import apply_topk_prototype_residual
from rcc_cpa import RankCalibratedCPA, validate_rcc_compatibility


def test_uniform_rcc_exactly_matches_fixed_cpa():
    torch.manual_seed(7)
    base = torch.randn(2, 21, 4, 5)
    prototype = torch.randn_like(base)
    expected, _ = apply_topk_prototype_residual(
        base, prototype, topk=15, residual_scale=0.5, residual_clip=0.5
    )
    rcc = RankCalibratedCPA(
        topk=15,
        residual_clip=0.5,
        schedule="uniform",
        uniform_scale=0.5,
    )

    actual, stats = rcc(base, prototype)

    assert torch.equal(actual, expected)
    assert float(stats["rcc_non_topk_changed_fraction"]) == 0.0


def test_piecewise_scales_follow_candidate_rank():
    base = torch.tensor([[[4.0], [3.0], [2.0], [1.0]]])
    prototype = base + 1.0
    rcc = RankCalibratedCPA(
        topk=3,
        residual_clip=2.0,
        schedule="piecewise",
        rank_scales=[0.1, 0.2, 0.3],
    )

    actual, _ = rcc(base, prototype)

    assert torch.allclose(
        actual - base,
        torch.tensor([[[0.1], [0.2], [0.3], [0.0]]]),
    )


def test_linear_schedule_uses_requested_endpoints():
    rcc = RankCalibratedCPA(
        topk=5,
        schedule="linear",
        scale_start=0.4,
        scale_end=0.8,
    )
    scales = rcc.rank_scale_tensor(torch.device("cpu"), torch.float32)
    assert torch.allclose(scales, torch.tensor([0.4, 0.5, 0.6, 0.7, 0.8]))


def test_non_topk_logits_are_bitwise_unchanged():
    torch.manual_seed(11)
    base = torch.randn(1, 8, 9)
    prototype = torch.randn_like(base)
    rcc = RankCalibratedCPA(
        topk=3,
        schedule="linear",
        scale_start=0.4,
        scale_end=0.7,
    )

    actual, stats = rcc(base, prototype)
    outside = ~stats["rcc_topk_mask"]

    assert torch.equal(actual.masked_select(outside), base.masked_select(outside))
    assert float(stats["rcc_non_topk_changed_fraction"]) == 0.0


def test_piecewise_schedule_requires_one_scale_per_rank():
    with pytest.raises(ValueError, match="length must equal"):
        RankCalibratedCPA(
            topk=3,
            schedule="piecewise",
            rank_scales=[0.4, 0.5],
        )


def test_rcc_has_no_parameters_or_checkpoint_state():
    rcc = RankCalibratedCPA()
    assert list(rcc.parameters()) == []
    assert rcc.state_dict() == {}


@pytest.mark.parametrize(
    "cpa_enabled,extra_kwargs,match",
    [
        (False, {}, "requires CPA-v1"),
        (True, {"cars_enabled": True}, "CARS"),
        (True, {"vpa_enabled": True}, "VPA"),
        (True, {"vab_enabled": True}, "VAB"),
        (True, {"opc_enabled": True}, "OPC"),
        (True, {"cpa_router_enabled": True}, "CPA-Router"),
        (True, {"usrc_enabled": True}, "USRC"),
        (True, {"ccr_enabled": True}, "CCR"),
    ],
)
def test_rcc_rejects_incompatible_eval_methods(
    cpa_enabled, extra_kwargs, match
):
    with pytest.raises(ValueError, match=match):
        validate_rcc_compatibility(True, cpa_enabled, **extra_kwargs)


def test_only_base_topk_source_is_supported():
    with pytest.raises(ValueError, match="topk_source=base"):
        RankCalibratedCPA(topk_source="cpa")
