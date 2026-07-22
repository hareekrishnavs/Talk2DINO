import io

import pytest
import torch
import torch.nn.functional as F

import src.model as model_module
from src.model import ProjectionLayer


def make_layer(strategy="multicaption_consensus_routing", dim=4, temperature=0.10):
    return ProjectionLayer(
        act=None,
        dino_embed_dim=dim,
        clip_embed_dim=dim,
        alignment_strategy=strategy,
        routing_temperature=temperature,
    )


def reference(text, visual, mask, temperature=0.10):
    affinities = torch.einsum("bkd,bhd->bkh", text, visual)
    consensus = (affinities * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    weights = torch.softmax(consensus / temperature, dim=-1)
    routed = F.normalize(torch.einsum("bh,bhd->bd", weights, visual), dim=-1)
    owner = torch.arange(text.shape[0], device=text.device).unsqueeze(1).expand_as(mask)[mask]
    return text[mask] @ routed.t(), owner, consensus.argmax(-1)


def test_direct_numerical_equivalence_and_owner_order():
    torch.manual_seed(5)
    layer = make_layer(dim=5, temperature=0.20)
    text = F.normalize(torch.randn(3, 4, 5), dim=-1)
    visual = F.normalize(torch.randn(3, 3, 5), dim=-1)
    mask = torch.tensor(
        [[True, True, False, False], [True, False, False, False], [True, True, True, False]]
    )

    scores, indices = layer.compute_similarity(visual, text, caption_mask=mask, return_index=True)
    expected, owner, expected_indices = reference(text, visual, mask, 0.20)

    assert scores.shape == (6, 3)
    torch.testing.assert_close(scores, expected)
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(owner, torch.tensor([0, 0, 1, 2, 2, 2]))


def test_padding_does_not_affect_routing_or_scores():
    layer = make_layer(dim=3)
    text = F.normalize(torch.randn(2, 3, 3), dim=-1)
    visual = F.normalize(torch.randn(2, 4, 3), dim=-1)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    changed = text.clone()
    changed[~mask] = 1000

    original = layer.compute_similarity(visual, text, caption_mask=mask)
    altered = layer.compute_similarity(visual, changed, caption_mask=mask)

    torch.testing.assert_close(original, altered)


def test_consensus_uses_all_valid_captions_and_is_order_invariant():
    layer = make_layer(dim=2, temperature=0.30)
    text = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    visual = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.tensor([[True, True, False]])

    scores = layer.compute_similarity(visual, text, caption_mask=mask)
    permuted = layer.compute_similarity(
        visual, text[:, [1, 0, 2]], caption_mask=mask
    )
    expected, _, _ = reference(text, visual, mask, 0.30)

    torch.testing.assert_close(scores, expected)
    torch.testing.assert_close(scores, permuted[[1, 0]])


def test_negative_caption_cannot_reroute_another_image():
    layer = make_layer(dim=2, temperature=0.01)
    text = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    visual = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]
    )
    mask = torch.ones(2, 1, dtype=torch.bool)

    scores = layer.compute_similarity(visual, text, caption_mask=mask)

    assert scores[1, 0] < 1e-4
    assert scores[0, 0] > 0.9999


def test_weighted_sum_and_post_routing_normalization(monkeypatch):
    layer = make_layer(dim=2, temperature=0.20)
    text = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    visual = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    affinities = torch.einsum("bkd,bhd->bkh", text, visual).mean(1)
    weights = torch.softmax(affinities / 0.20, dim=-1)
    expected_sum = torch.einsum("bh,bhd->bd", weights, visual)
    captured = {}
    real_normalize = F.normalize

    def capture(value, *args, **kwargs):
        captured["input"] = value.detach().clone()
        output = real_normalize(value, *args, **kwargs)
        captured["output"] = output.detach().clone()
        return output

    monkeypatch.setattr(model_module.F, "normalize", capture)
    layer.compute_similarity(visual, text, caption_mask=mask)

    torch.testing.assert_close(captured["input"], expected_sum)
    torch.testing.assert_close(captured["output"].norm(dim=-1), torch.ones(1))


def test_k_one_matches_paired_soft_routing():
    text, visual = torch.randn(3, 1, 4), torch.randn(3, 5, 4)
    text = F.normalize(text, dim=-1)
    visual = F.normalize(visual, dim=-1)
    mask = torch.ones(3, 1, dtype=torch.bool)
    consensus = make_layer(dim=4).compute_similarity(visual, text, caption_mask=mask)
    paired = make_layer(strategy="paired_soft_routing", dim=4).compute_similarity(
        visual, text[:, 0]
    )
    torch.testing.assert_close(consensus, paired)


def test_h_one_and_batch_one():
    layer = make_layer(dim=3)
    text = F.normalize(torch.randn(1, 2, 3), dim=-1)
    visual = F.normalize(torch.randn(1, 1, 3), dim=-1)
    mask = torch.ones(1, 2, dtype=torch.bool)
    scores, indices = layer.compute_similarity(visual, text, caption_mask=mask, return_index=True)
    torch.testing.assert_close(scores, text[mask] @ visual[:, 0].t())
    assert scores.shape == (2, 1)
    torch.testing.assert_close(indices, torch.zeros(1, dtype=torch.long))


@pytest.mark.parametrize("temperature", [0.0, -0.1])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="strictly positive"):
        make_layer(temperature=temperature)


@pytest.mark.parametrize(
    ("visual", "text", "mask", "message"),
    [
        (torch.randn(2, 4), torch.randn(2, 3, 4), torch.ones(2, 3, dtype=torch.bool), "visual"),
        (torch.randn(2, 3, 4), torch.randn(2, 4), torch.ones(2, 3, dtype=torch.bool), "text"),
        (torch.randn(2, 3, 4), torch.randn(3, 2, 4), torch.ones(3, 2, dtype=torch.bool), "batch"),
        (torch.randn(2, 3, 5), torch.randn(2, 2, 4), torch.ones(2, 2, dtype=torch.bool), "dimensions"),
    ],
)
def test_shape_errors(visual, text, mask, message):
    with pytest.raises(ValueError, match=message):
        make_layer().compute_similarity(visual, text, caption_mask=mask)


def test_invalid_and_empty_caption_masks():
    layer = make_layer()
    visual, text = torch.randn(2, 3, 4), torch.randn(2, 2, 4)
    with pytest.raises(ValueError, match="caption_mask"):
        layer.compute_similarity(visual, text, caption_mask=None)
    with pytest.raises(ValueError, match="at least one"):
        layer.compute_similarity(
            visual, text, caption_mask=torch.tensor([[True, False], [False, False]])
        )


def test_float64_device_and_finite_gradients_reach_captions_and_heads():
    layer = make_layer(dim=3, temperature=0.50)
    text = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
    visual = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(2, 2, dtype=torch.bool)

    scores = layer.compute_similarity(visual, text, caption_mask=mask)
    scores.sum().backward()

    assert scores.dtype == torch.float64 and scores.device == text.device
    assert torch.isfinite(scores).all()
    assert text.grad is not None and torch.isfinite(text.grad).all()
    assert visual.grad is not None and torch.isfinite(visual.grad).all()
    assert torch.count_nonzero(text.grad.norm(dim=-1)) > 1
    assert torch.count_nonzero(visual.grad.norm(dim=-1)) > 1


def test_checkpoint_state_dict_round_trip():
    layer = make_layer(dim=4)
    buffer = io.BytesIO()
    torch.save(layer.state_dict(), buffer)
    buffer.seek(0)
    restored = make_layer(dim=4)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    for actual, expected in zip(restored.parameters(), layer.parameters()):
        torch.testing.assert_close(actual, expected)
