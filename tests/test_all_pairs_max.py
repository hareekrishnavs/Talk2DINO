import pytest
import torch

from src.model import ProjectionLayer


def make_layer(strategy="all_pairs_max", clip_dim=768):
    return ProjectionLayer(
        act=None,
        dino_embed_dim=768,
        clip_embed_dim=clip_dim,
        num_attn_head=12,
        alignment_strategy=strategy,
    )


def test_output_shape():
    layer = make_layer()
    text = torch.randn(4, 768)
    visual = torch.randn(4, 12, 768)

    scores = layer.compute_similarity(visual, text)

    assert scores.shape == (4, 4)


def test_matches_brute_force():
    torch.manual_seed(7)
    layer = make_layer()
    text = torch.randn(3, 768)
    visual = torch.randn(2, 5, 768)

    actual = layer.compute_similarity(visual, text)
    expected = torch.empty(3, 2)
    for text_index in range(text.shape[0]):
        for image_index in range(visual.shape[0]):
            expected[text_index, image_index] = max(
                torch.dot(text[text_index], visual[image_index, head_index])
                for head_index in range(visual.shape[1])
            )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_pair_specific_head_selection():
    layer = ProjectionLayer(
        act=None,
        dino_embed_dim=2,
        clip_embed_dim=2,
        num_attn_head=2,
        alignment_strategy="all_pairs_max",
    )
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    visual = torch.tensor([[[3.0, 0.0], [0.0, 5.0]]])

    scores = layer.compute_similarity(visual, text)

    torch.testing.assert_close(scores, torch.tensor([[3.0], [5.0]]))


def test_gradient_flows_to_text_projection():
    torch.manual_seed(11)
    layer = make_layer(clip_dim=512)
    text = torch.randn(4, 512)
    visual = torch.randn(4, 12, 768)

    scores = layer(visual, text)
    scores.sum().backward()

    gradients = [parameter.grad for parameter in layer.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_return_index_returns_diagonal_pair_indices():
    torch.manual_seed(13)
    layer = make_layer()
    text = torch.randn(4, 768)
    visual = torch.randn(4, 12, 768)

    scores, indices = layer.compute_similarity(visual, text, return_index=True)
    affinities = torch.einsum("td,bhd->tbh", text, visual)
    expected_scores, pairwise_indices = affinities.max(dim=-1)

    assert indices.shape == (4,)
    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(indices, pairwise_indices.diagonal())


def test_return_index_rejects_non_square_batch():
    layer = make_layer()
    with pytest.raises(ValueError, match="square"):
        layer.compute_similarity(
            torch.randn(2, 3, 768),
            torch.randn(4, 768),
            return_index=True,
        )


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


def test_max_score_preserves_positive_selected_head_behavior():
    torch.manual_seed(17)
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
