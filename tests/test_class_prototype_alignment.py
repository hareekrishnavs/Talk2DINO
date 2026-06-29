import sys
from pathlib import Path

import torch
import torch.nn.functional as F


MODELS = Path(__file__).parents[1] / "src" / "open_vocabulary_segmentation" / "models"
sys.path.insert(0, str(MODELS))

from class_prototype_alignment import (  # noqa: E402
    ClassPrototypeAlignmentHead,
    apply_topk_prototype_residual,
    assert_topk_safety,
    compute_prototype_logits,
    prototype_geometry,
)


def build_head(dim=32, dropout=0.10):
    torch.manual_seed(7)
    return ClassPrototypeAlignmentHead(
        dino_dim=dim,
        num_prototypes=4,
        hidden_dim=16,
        prototype_radius_min=0.10,
        prototype_radius_max=0.35,
        prototype_radius_init=0.22,
        prototype_dropout=dropout,
    )


def assert_tangent_geometry(mapped, prototypes, details):
    mapped_unit = F.normalize(mapped.float(), dim=-1)
    tangent = details["tangent_directions"]
    assert torch.allclose(details["mapped_unit"], mapped_unit, atol=1e-6)
    assert torch.allclose(tangent.norm(dim=-1), torch.ones_like(details["radii"]), atol=1e-5)
    tangent_dot = (tangent * mapped_unit.unsqueeze(-2)).sum(dim=-1)
    assert torch.allclose(tangent_dot, torch.zeros_like(tangent_dot), atol=1e-5)
    assert torch.allclose(prototypes.norm(dim=-1), torch.ones_like(details["radii"]), atol=1e-5)
    assert float(details["radii"].min().detach()) >= 0.10
    assert float(details["radii"].max().detach()) <= 0.35
    assert torch.allclose(
        details["radii"],
        torch.full_like(details["radii"], 0.22),
        atol=1e-6,
    )


def test_tangent_prototypes_for_unconditioned_and_conditioned_text():
    head = build_head()
    assert head.shared_basis.shape == (4, 32)

    mapped = torch.randn(6, 32) * 7.1
    prototypes, details = head(mapped, return_details=True)
    assert prototypes.shape == (6, 4, 32)
    assert_tangent_geometry(mapped, prototypes, details)

    conditioned = torch.randn(2, 6, 32) * 7.1
    conditioned_prototypes, conditioned_details = head(
        conditioned, return_details=True
    )
    assert conditioned_prototypes.shape == (2, 6, 4, 32)
    assert_tangent_geometry(conditioned, conditioned_prototypes, conditioned_details)


def test_initial_prototypes_are_close_but_not_collapsed():
    head = build_head()
    prototypes, details = head(torch.randn(64, 32), return_details=True)
    geometry = prototype_geometry(prototypes, details["tangent_directions"])
    assert 0.80 < float(geometry["cpa_proto_pairwise_cos_mean"]) < 0.999
    assert float(geometry["cpa_proto_pairwise_cos_min"]) < 0.999
    base_cosine = (prototypes * details["mapped_unit"].unsqueeze(-2)).sum(-1)
    assert torch.all(base_cosine < 1.0)
    assert torch.all(base_cosine > 0.90)


def test_prototype_logits_support_global_dense_and_dropout():
    head = build_head(dropout=0.999)
    head.train()
    prototypes = head(torch.randn(5, 32))
    global_logits, global_active = compute_prototype_logits(
        torch.randn(3, 32),
        prototypes,
        prototype_dropout=head.prototype_dropout,
        training=True,
        return_active_fraction=True,
    )
    dense_logits, dense_active = compute_prototype_logits(
        torch.randn(3, 11, 32),
        prototypes,
        prototype_dropout=head.prototype_dropout,
        training=True,
        return_active_fraction=True,
    )
    assert global_logits.shape == (3, 5)
    assert dense_logits.shape == (3, 5, 11)
    assert torch.isfinite(global_logits).all()
    assert torch.isfinite(dense_logits).all()
    assert float(global_active) >= 0.25
    assert float(dense_active) >= 0.25

    conditioned = head(torch.randn(3, 5, 32))
    assert compute_prototype_logits(torch.randn(3, 32), conditioned).shape == (3, 5)
    assert compute_prototype_logits(torch.randn(3, 11, 32), conditioned).shape == (
        3,
        5,
        11,
    )


def test_topk_residual_leaves_all_other_logits_bit_exact():
    base = torch.randn(2, 9, 13)
    prototype = torch.randn_like(base)
    final, stats = apply_topk_prototype_residual(
        base, prototype, topk=5, residual_scale=0.25, residual_clip=0.5
    )
    outside = ~stats["cpa_topk_mask"]
    assert torch.equal(final[outside], base[outside])
    assert_topk_safety(base, final, stats)


def test_topk_residual_accepts_fp16_base_and_fp32_prototypes():
    base = torch.randn(2, 9, 13, dtype=torch.float16)
    prototype = torch.randn(2, 9, 13, dtype=torch.float32)
    final, stats = apply_topk_prototype_residual(base, prototype, topk=5)
    outside = ~stats["cpa_topk_mask"]
    assert final.dtype == torch.float16
    assert stats["cpa_residual"].dtype == torch.float32
    assert torch.equal(final[outside], base[outside])
    assert_topk_safety(base, final, stats)
