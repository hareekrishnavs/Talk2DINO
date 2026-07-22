import math

import pytest
import torch

from src.model import ProjectionLayer


def make_layer(strategy="all_pairs_lse", clip_dim=768, temperature=0.10):
    return ProjectionLayer(
        act=None,
        dino_embed_dim=768,
        clip_embed_dim=clip_dim,
        num_attn_head=12,
        alignment_strategy=strategy,
        region_temperature=temperature,
    )


def normalized_lse(values, temperature):
    return (
        temperature * torch.logsumexp(values / temperature, dim=0)
        - temperature * math.log(values.numel())
    )


def test_output_shape():
    layer = make_layer()
    text = torch.randn(4, 768)
    visual = torch.randn(4, 12, 768)

    scores = layer.compute_similarity(visual, text)

    assert scores.shape == (4, 4)


def test_matches_brute_force():
    torch.manual_seed(7)
    tau = 0.10
    layer = make_layer(temperature=tau)
    text = torch.randn(3, 768)
    visual = torch.randn(2, 5, 768)

    actual = layer.compute_similarity(visual, text)
    expected = torch.empty(3, 2)
    for text_index in range(text.shape[0]):
        for image_index in range(visual.shape[0]):
            values = torch.stack(
                [
                    torch.dot(text[text_index], visual[image_index, head_index])
                    for head_index in range(visual.shape[1])
                ]
            )
            expected[text_index, image_index] = normalized_lse(values, tau)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_equal_affinities_preserve_score():
    layer = ProjectionLayer(
        act=None,
        dino_embed_dim=2,
        clip_embed_dim=2,
        num_attn_head=4,
        alignment_strategy="all_pairs_lse",
        region_temperature=0.10,
    )
    text = torch.tensor([[1.0, 0.0]])
    visual = torch.tensor([[[2.5, 0.0]] * 4])

    scores = layer.compute_similarity(visual, text)

    torch.testing.assert_close(scores, torch.tensor([[2.5]]))


def test_single_head_equals_dot_product_and_preserves_autograd_metadata():
    layer = ProjectionLayer(
        act=None,
        dino_embed_dim=3,
        clip_embed_dim=3,
        alignment_strategy="all_pairs_lse",
        region_temperature=0.10,
    )
    text = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    visual = torch.randn(3, 1, 3, dtype=torch.float64, requires_grad=True)

    scores = layer.compute_similarity(visual, text)
    expected = torch.einsum("td,bhd->tb", text, visual)

    assert scores.shape == (2, 3)
    assert scores.dtype == text.dtype
    assert scores.device == text.device
    torch.testing.assert_close(scores, expected)

    scores.sum().backward()
    assert text.grad is not None and torch.isfinite(text.grad).all()
    assert visual.grad is not None and torch.isfinite(visual.grad).all()


def test_scores_are_between_mean_and_max_affinity():
    torch.manual_seed(11)
    layer = make_layer()
    text = torch.randn(3, 768)
    visual = torch.randn(2, 5, 768)
    affinities = torch.einsum("td,bhd->tbh", text, visual)

    scores = layer.compute_similarity(visual, text)

    assert torch.all(scores >= affinities.mean(dim=-1) - 1e-6)
    assert torch.all(scores <= affinities.max(dim=-1).values + 1e-6)


def test_pair_specific_scores():
    layer = ProjectionLayer(
        act=None,
        dino_embed_dim=2,
        clip_embed_dim=2,
        num_attn_head=3,
        alignment_strategy="all_pairs_lse",
        region_temperature=0.10,
    )
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    visual = torch.tensor([[[0.4, 0.1], [0.3, 0.1], [0.2, 0.8]]])
    affinities = torch.einsum("td,bhd->tbh", text, visual)

    scores = layer.compute_similarity(visual, text)
    expected = torch.stack(
        [normalized_lse(affinities[index, 0], 0.10) for index in range(2)]
    ).unsqueeze(1)

    torch.testing.assert_close(scores, expected)
    assert scores[0, 0] != scores[1, 0]


def test_gradient_flows_to_text_projection():
    torch.manual_seed(13)
    layer = make_layer(clip_dim=512)
    text = torch.randn(4, 512)
    visual = torch.randn(4, 12, 768)

    layer(visual, text).sum().backward()

    gradients = [parameter.grad for parameter in layer.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.norm() for gradient in gradients) > 0


def test_multiple_visual_heads_receive_gradient():
    layer = ProjectionLayer(
        act=None,
        dino_embed_dim=2,
        clip_embed_dim=2,
        num_attn_head=3,
        alignment_strategy="all_pairs_lse",
        region_temperature=0.10,
    )
    text = torch.tensor([[1.0, 0.0]])
    visual = torch.tensor(
        [[[0.10, 0.0], [0.20, 0.0], [0.30, 0.0]]], requires_grad=True
    )

    layer.compute_similarity(visual, text).sum().backward()

    head_gradient_norms = visual.grad.norm(dim=-1)
    assert torch.count_nonzero(head_gradient_norms) > 1


@pytest.mark.parametrize("temperature", [0.0, -0.1])
def test_temperature_must_be_positive(temperature):
    with pytest.raises(ValueError, match="strictly positive"):
        make_layer(temperature=temperature)


@pytest.mark.parametrize(
    ("visual", "text"),
    [
        (torch.randn(2, 3, 768), torch.randn(2, 4, 768)),
        (torch.randn(2, 768), torch.randn(2, 768)),
    ],
)
def test_invalid_shapes_are_rejected(visual, text):
    layer = make_layer()
    with pytest.raises(ValueError, match=r"visual=.*text="):
        layer.compute_similarity(visual, text)


def test_return_index_returns_diagonal_hard_max_indices():
    torch.manual_seed(17)
    layer = make_layer()
    text = torch.randn(4, 768)
    visual = torch.randn(4, 12, 768)

    _, indices = layer.compute_similarity(visual, text, return_index=True)
    affinities = torch.einsum("td,bhd->tbh", text, visual)

    assert indices.shape == (4,)
    torch.testing.assert_close(indices, affinities.argmax(dim=-1).diagonal())


def test_return_index_rejects_non_square_batch():
    layer = make_layer()
    with pytest.raises(ValueError, match="square"):
        layer.compute_similarity(
            torch.randn(2, 3, 768),
            torch.randn(4, 768),
            return_index=True,
        )


def test_from_config_reads_region_temperature():
    layer = ProjectionLayer.from_config(
        {"alignment_strategy": "all_pairs_lse", "region_temperature": 0.25}
    )
    assert layer.region_temperature == 0.25


def test_max_score_preserves_positive_selected_head_behavior():
    torch.manual_seed(19)
    layer = make_layer(strategy="max_score")
    text = torch.randn(4, 768)
    visual = torch.randn(4, 12, 768)

    actual, actual_indices = layer.compute_similarity(
        visual, text, return_index=True
    )
    positive_affinities = torch.einsum("ik,ijk->ij", text, visual)
    positive_probabilities = positive_affinities.softmax(dim=-1)
    expected_indices = positive_probabilities.argmax(dim=-1)
    gather_indices = expected_indices.view(-1, 1, 1).expand(-1, 1, 768)
    selected_visual = torch.gather(visual, 1, gather_indices).squeeze(1)
    expected = text @ selected_visual.transpose(1, 0)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_indices, expected_indices)
