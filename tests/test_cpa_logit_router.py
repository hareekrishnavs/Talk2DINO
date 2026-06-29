import sys
from pathlib import Path

import torch


MODELS = Path(__file__).parents[1] / "src" / "open_vocabulary_segmentation" / "models"
sys.path.insert(0, str(MODELS))

from cpa_logit_router import (  # noqa: E402
    ConfidenceAwarePrototypeLogitRouter,
    assert_router_topk_safety,
)
from class_prototype_alignment import ClassPrototypeAlignmentHead  # noqa: E402


def make_router():
    torch.manual_seed(3)
    return ConfidenceAwarePrototypeLogitRouter(
        topk=5,
        hidden_dim=32,
        init_w_base=0.74,
        init_w_xattn=0.02,
        init_w_cpa=0.24,
    )


def check_router_shape(shape):
    router = make_router()
    base = torch.randn(*shape)
    xattn = torch.randn(*shape)
    prototype = torch.randn(*shape)
    final, stats = router(base, xattn, prototype)
    assert final.shape == base.shape
    weights = stats["router_weights"]
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))
    expected = torch.tensor([0.74, 0.02, 0.24])
    assert torch.allclose(weights, expected, atol=1e-6)
    outside = ~stats["router_topk_mask"]
    assert torch.equal(final[outside], base[outside])
    assert_router_topk_safety(base, final, stats)


def test_router_supports_dense_tokens():
    check_router_shape((2, 11, 17))


def test_router_supports_spatial_logits():
    check_router_shape((2, 11, 4, 5))


def test_router_is_mixed_precision_safe_and_differentiable():
    router = make_router()
    base = torch.randn(2, 9, 13, dtype=torch.float16)
    xattn = torch.randn(2, 9, 13, dtype=torch.float32, requires_grad=True)
    prototype = torch.randn(2, 9, 13, dtype=torch.float32, requires_grad=True)
    final, stats = router(base, xattn, prototype)
    final.float().sum().backward()
    assert final.dtype == torch.float16
    assert torch.isfinite(xattn.grad).all()
    assert torch.isfinite(prototype.grad).all()
    outside = ~stats["router_topk_mask"]
    assert torch.equal(final[outside], base[outside])


def test_router_initial_mixture_matches_safe_prior():
    router = make_router()
    base = torch.randn(2, 8, 7)
    xattn = torch.randn_like(base)
    prototype = torch.randn_like(base)
    final, stats = router(base, xattn, prototype)
    indices = stats["router_topk_indices"]
    expected = (
        0.74 * base.gather(1, indices)
        + 0.02 * xattn.gather(1, indices)
        + 0.24 * prototype.gather(1, indices)
    )
    assert torch.allclose(final.gather(1, indices), expected, atol=1e-6)


def test_cpa_v1_generator_is_preserved():
    torch.manual_seed(11)
    original = ClassPrototypeAlignmentHead(dino_dim=32, hidden_dim=16)
    explicit_v1 = ClassPrototypeAlignmentHead(dino_dim=32, hidden_dim=16, version="v1")
    explicit_v1.load_state_dict(original.state_dict())
    mapped = torch.randn(3, 32)
    assert torch.equal(original(mapped), explicit_v1(mapped))


def test_cpa_v2_and_v3_are_rejected():
    for version in ("v2", "v3_tangent_space"):
        try:
            ClassPrototypeAlignmentHead(dino_dim=32, hidden_dim=16, version=version)
        except ValueError as error:
            assert "Only CPA-v1" in str(error)
        else:
            raise AssertionError(f"CPA version {version} should not be active")
