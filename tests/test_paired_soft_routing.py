import pytest
import torch
import torch.nn.functional as F

import src.model as model_module
from src.model import ProjectionLayer


def make_layer(strategy="paired_soft_routing", temperature=0.10, dim=4):
    return ProjectionLayer(
        act=None,
        dino_embed_dim=dim,
        clip_embed_dim=dim,
        num_attn_head=3,
        alignment_strategy=strategy,
        routing_temperature=temperature,
    )


def reference_scores(text, visual, temperature):
    positive_affinities = torch.einsum("bd,bhd->bh", text, visual)
    routing_weights = torch.softmax(positive_affinities / temperature, dim=-1)
    routed_visual = torch.einsum("bh,bhd->bd", routing_weights, visual)
    routed_visual = F.normalize(routed_visual, p=2, dim=-1)
    return text @ routed_visual.transpose(0, 1)


def normalized_inputs(batch=3, heads=4, dim=5, dtype=torch.float32):
    text = F.normalize(torch.randn(batch, dim, dtype=dtype), dim=-1)
    visual = F.normalize(torch.randn(batch, heads, dim, dtype=dtype), dim=-1)
    return text, visual


def test_matches_direct_reference():
    torch.manual_seed(3)
    layer = make_layer(temperature=0.17, dim=5)
    text, visual = normalized_inputs(dim=5)

    actual = layer.compute_similarity(visual, text)
    expected = reference_scores(text, visual, 0.17)

    torch.testing.assert_close(actual, expected)


def test_output_shape():
    layer = make_layer(dim=6)
    text, visual = normalized_inputs(batch=4, heads=12, dim=6)
    assert layer.compute_similarity(visual, text).shape == (4, 4)


def test_routing_weights_use_only_positive_pairs():
    layer = make_layer(dim=2)
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    visual = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )

    scores = layer.compute_similarity(visual, text)
    expected = reference_scores(text, visual, 0.10)

    torch.testing.assert_close(scores, expected)


def test_negative_text_cannot_reroute_image():
    layer = make_layer(temperature=0.01, dim=2)
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    visual = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )

    scores = layer.compute_similarity(visual, text)

    assert scores[1, 0] < 1e-4
    assert scores[0, 0] > 0.9999


def test_weighted_sum_is_passed_to_normalization(monkeypatch):
    layer = make_layer(temperature=0.20, dim=2)
    text = torch.tensor([[1.0, 0.0]])
    visual = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    positive_affinities = torch.einsum("bd,bhd->bh", text, visual)
    weights = torch.softmax(positive_affinities / 0.20, dim=-1)
    expected_sum = torch.einsum("bh,bhd->bd", weights, visual)
    captured = {}
    real_normalize = F.normalize

    def capture_normalize(value, *args, **kwargs):
        captured["value"] = value.detach().clone()
        return real_normalize(value, *args, **kwargs)

    monkeypatch.setattr(model_module.F, "normalize", capture_normalize)
    layer.compute_similarity(visual, text)

    torch.testing.assert_close(captured["value"], expected_sum)
    assert not torch.allclose(captured["value"], expected_sum / visual.shape[1])


def test_routed_visual_is_l2_normalized(monkeypatch):
    layer = make_layer(dim=3)
    text, visual = normalized_inputs(batch=2, heads=3, dim=3)
    captured = {}
    real_normalize = F.normalize

    def capture_normalize(value, *args, **kwargs):
        result = real_normalize(value, *args, **kwargs)
        captured["result"] = result.detach().clone()
        return result

    monkeypatch.setattr(model_module.F, "normalize", capture_normalize)
    layer.compute_similarity(visual, text)

    torch.testing.assert_close(
        captured["result"].norm(dim=-1), torch.ones(text.shape[0])
    )


def test_single_head_is_ordinary_pairwise_similarity():
    layer = make_layer(dim=5)
    text, visual = normalized_inputs(batch=3, heads=1, dim=5)

    scores = layer.compute_similarity(visual, text)
    expected = text @ visual[:, 0].transpose(0, 1)

    torch.testing.assert_close(scores, expected)


def test_temperature_is_loaded_from_config():
    layer = ProjectionLayer.from_config(
        {
            "alignment_strategy": "paired_soft_routing",
            "routing_temperature": 0.25,
        }
    )
    assert layer.routing_temperature == 0.25


@pytest.mark.parametrize("temperature", [0.0, -0.1])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match=f"{temperature}"):
        make_layer(temperature=temperature)


def test_invalid_visual_dimensions_are_rejected():
    layer = make_layer()
    with pytest.raises(ValueError, match=r"visual=\(2, 4\).*text=\(2, 4\)"):
        layer.compute_similarity(torch.randn(2, 4), torch.randn(2, 4))


def test_invalid_text_dimensions_are_rejected():
    layer = make_layer()
    with pytest.raises(ValueError, match=r"visual=\(2, 3, 4\).*text=\(2, 1, 4\)"):
        layer.compute_similarity(torch.randn(2, 3, 4), torch.randn(2, 1, 4))


def test_non_square_batch_is_rejected():
    layer = make_layer()
    with pytest.raises(ValueError, match="equal text and image batch sizes"):
        layer.compute_similarity(torch.randn(2, 3, 4), torch.randn(3, 4))


def test_embedding_dimension_mismatch_is_rejected():
    layer = make_layer()
    with pytest.raises(ValueError, match="matching embedding dimensions"):
        layer.compute_similarity(torch.randn(2, 3, 5), torch.randn(2, 4))


def test_return_index_is_positive_pair_argmax():
    layer = make_layer(dim=3)
    text, visual = normalized_inputs(batch=4, heads=5, dim=3)

    scores, indices = layer.compute_similarity(visual, text, return_index=True)
    expected_indices = torch.einsum("bd,bhd->bh", text, visual).argmax(dim=-1)

    assert scores.shape == (4, 4)
    assert indices.shape == (4,)
    torch.testing.assert_close(indices, expected_indices)


def test_gradients_are_finite():
    torch.manual_seed(7)
    layer = make_layer(dim=4)
    text = torch.randn(3, 4, requires_grad=True)
    visual = torch.randn(3, 3, 4, requires_grad=True)

    layer.compute_similarity(visual, text).sum().backward()

    assert text.grad is not None and torch.isfinite(text.grad).all()
    assert visual.grad is not None and torch.isfinite(visual.grad).all()


def test_multiple_visual_heads_receive_gradient():
    layer = make_layer(temperature=0.50, dim=2)
    text = torch.tensor([[1.0, 0.0]])
    visual = torch.tensor(
        [[[0.8, 0.2], [0.6, 0.4], [0.4, 0.6]]], requires_grad=True
    )

    layer.compute_similarity(visual, text).sum().backward()

    assert torch.count_nonzero(visual.grad.norm(dim=-1)) > 1


def test_float64_dtype_is_preserved():
    layer = make_layer(dim=3)
    text, visual = normalized_inputs(batch=2, heads=2, dim=3, dtype=torch.float64)
    assert layer.compute_similarity(visual, text).dtype == torch.float64


def test_device_is_preserved():
    layer = make_layer(dim=3)
    text, visual = normalized_inputs(batch=2, heads=2, dim=3)
    assert layer.compute_similarity(visual, text).device == text.device


def test_batch_size_one_works():
    layer = make_layer(dim=3)
    text, visual = normalized_inputs(batch=1, heads=2, dim=3)
    scores, indices = layer.compute_similarity(visual, text, return_index=True)
    assert scores.shape == (1, 1)
    assert indices.shape == (1,)


def test_max_score_behavior_is_unchanged():
    torch.manual_seed(11)
    layer = make_layer(strategy="max_score", dim=4)
    text = torch.randn(3, 4)
    visual = torch.randn(3, 5, 4)

    actual, actual_indices = layer.compute_similarity(visual, text, return_index=True)
    positive_scores = torch.einsum("ik,ijk->ij", text, visual).softmax(dim=-1)
    expected_indices = positive_scores.argmax(dim=-1)
    gather_indices = expected_indices.view(-1, 1, 1).expand(-1, 1, 4)
    selected_visual = torch.gather(visual, 1, gather_indices).squeeze(1)
    expected = text @ selected_visual.transpose(1, 0)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_indices, expected_indices)


def test_weighted_avg_behavior_is_unchanged():
    torch.manual_seed(13)
    layer = make_layer(strategy="weighted_avg", dim=4)
    text = torch.randn(3, 4)
    visual = torch.randn(3, 5, 4)

    actual = layer.compute_similarity(visual, text)
    weights = torch.einsum("ik,ijk->ij", text, visual).softmax(dim=-1)
    expected_visual = (visual * weights.unsqueeze(-1)).mean(dim=1)
    expected = text @ expected_visual.transpose(1, 0)

    torch.testing.assert_close(actual, expected)
