import math

import pytest
import torch
import torch.nn.functional as F

from src.e9_sparse_region_alignment import (
    E9ValidationError,
    SparseRegionAlignmentAdapter,
    SparseRegionAlignmentConfig,
    compute_chunked_mil_scores,
    compute_e9_loss,
    symmetric_infonce,
)


def normalized(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(*shape, generator=generator), dim=-1)


def adapter(*, gamma_max=0.3):
    return SparseRegionAlignmentAdapter(
        SparseRegionAlignmentConfig(
            embedding_dim=8,
            bottleneck_dim=4,
            dropout=0,
            residual_max=0.25,
            gamma_max=gamma_max,
            gate_hidden_dim=4,
        )
    )


def inputs(requires_grad=False):
    query = normalized((3, 8), 1).requires_grad_(requires_grad)
    patches = normalized((3, 7, 8), 2).requires_grad_(requires_grad)
    priors = torch.softmax(torch.randn(3, 7, generator=torch.Generator().manual_seed(3)), -1).requires_grad_(requires_grad)
    return query, patches, priors


def test_adapter_shapes_normalization_bounds_and_no_class_parameters():
    model = adapter()
    query, patches, priors = inputs()
    output = model(query, patches, priors)
    assert output.base_score.shape == (3, 3, 7)
    assert output.adapted_score.shape == output.final_score.shape == output.gamma.shape
    assert output.query.shape == (3, 8) and output.patches.shape == (3, 7, 8)
    torch.testing.assert_close(output.query.norm(dim=-1), torch.ones(3))
    torch.testing.assert_close(output.patches.norm(dim=-1), torch.ones(3, 7))
    assert output.text_residual_gate.min() >= 0 and output.text_residual_gate.max() <= 0.25
    assert output.patch_residual_gate.min() >= 0 and output.patch_residual_gate.max() <= 0.25
    assert output.gamma.min() >= 0 and output.gamma.max() <= 0.30
    assert all(torch.isfinite(value).all() for value in vars(output).values())
    names = [name for name, _ in model.named_parameters()]
    assert not any("class" in name or "prototype" in name for name in names)


def test_query_image_and_patch_permutations_are_equivariant():
    model = adapter().eval()
    query, patches, priors = inputs()
    baseline = model(query, patches, priors).final_score
    image_order = torch.tensor([2, 0, 1])
    patch_order = torch.tensor([6, 2, 4, 0, 1, 5, 3])
    image_result = model(query, patches[image_order], priors[image_order]).final_score
    torch.testing.assert_close(image_result, baseline[:, image_order])
    patch_result = model(
        query, patches[:, patch_order], priors[:, patch_order]
    ).final_score
    torch.testing.assert_close(patch_result, baseline[:, :, patch_order])


def test_invalid_patches_do_not_affect_results():
    model = adapter().eval()
    query, patches, priors = inputs()
    valid = torch.ones(3, 7, dtype=torch.bool)
    valid[:, -1] = False
    first = model(query, patches, priors, valid).final_score
    changed = patches.clone()
    changed[:, -1] = normalized((3, 8), 9)
    second = model(query, changed, priors, valid).final_score
    torch.testing.assert_close(first[..., :-1], second[..., :-1])
    assert torch.equal(first[..., -1], torch.zeros_like(first[..., -1]))
    assert torch.equal(second[..., -1], torch.zeros_like(second[..., -1]))


def test_invalid_attention_prior_is_rejected():
    model = adapter().eval()
    query, patches, priors = inputs()
    priors = priors.clone()
    priors[0] *= 0.5
    with pytest.raises(E9ValidationError, match="sum to one"):
        model(query, patches, priors)


def test_gamma_zero_is_bitwise_exact_e3():
    model = adapter(gamma_max=0)
    query, patches, priors = inputs()
    output = model(query, patches, priors)
    assert torch.equal(output.gamma, torch.zeros_like(output.gamma))
    assert torch.equal(output.final_score, output.base_score)


def test_active_adapter_changes_a_nontrivial_case():
    model = adapter()
    with torch.no_grad():
        model.compatibility_gate[-1].bias.fill_(4)
        model.text_adapter.network[-1].weight.normal_(std=0.1)
        model.patch_adapter.network[-1].weight.normal_(std=0.1)
    query, patches, priors = inputs()
    output = model(query, patches, priors)
    assert not torch.equal(output.final_score, output.base_score)


def test_topk_lse_and_attention_selection_match_independent_reference():
    model = adapter().eval()
    query, patches, priors = inputs()
    full = model(query, patches, priors)
    output = compute_chunked_mil_scores(
        model, query, patches, priors,
        mil_top_k=3, mil_temperature=0.1,
        attention_selection_weight=0.05, pair_chunk_size=2,
        retain_pairwise_scores=True,
    )
    scaled = priors / priors.amax(-1, keepdim=True)
    selection = full.final_score + 0.05 * scaled[None]
    indices = torch.topk(selection, 3, dim=-1, sorted=True).indices
    selected = torch.gather(full.final_score, -1, indices)
    reference = 0.1 * torch.logsumexp(selected / 0.1, -1) - 0.1 * math.log(3)
    assert torch.equal(output.selected_indices, indices)
    torch.testing.assert_close(output.selected_scores, selected)
    torch.testing.assert_close(output.image_scores, reference)
    # Attention changes only selection; aggregation gathers the original score.
    assert torch.equal(output.selected_scores, torch.gather(output.final_score, -1, indices))


def test_dense_and_chunked_pairwise_agree():
    model = adapter().eval()
    query, patches, priors = inputs()
    dense = compute_chunked_mil_scores(
        model, query, patches, priors, mil_top_k=3,
        pair_chunk_size=3, retain_pairwise_scores=True,
    )
    chunked = compute_chunked_mil_scores(
        model, query, patches, priors, mil_top_k=3,
        pair_chunk_size=1, retain_pairwise_scores=True,
    )
    for name in (
        "image_scores", "selected_indices", "selected_scores", "gamma",
        "base_score", "adapted_score", "final_score",
    ):
        torch.testing.assert_close(getattr(dense, name), getattr(chunked, name))


def test_dense_and_chunked_losses_and_adapter_gradients_agree():
    dense_model = adapter()
    chunked_model = adapter()
    chunked_model.load_state_dict(dense_model.state_dict())
    query, patches, priors = inputs()
    dense = compute_chunked_mil_scores(
        dense_model, query, patches, priors, mil_top_k=3,
        pair_chunk_size=3,
    )
    chunked = compute_chunked_mil_scores(
        chunked_model, query, patches, priors, mil_top_k=3,
        pair_chunk_size=1,
    )
    dense_loss = compute_e9_loss(dense, query, patches)["loss"]
    chunked_loss = compute_e9_loss(chunked, query, patches)["loss"]
    torch.testing.assert_close(dense.image_scores, chunked.image_scores)
    torch.testing.assert_close(dense_loss, chunked_loss)
    dense_loss.backward()
    chunked_loss.backward()
    for (dense_name, dense_parameter), (chunked_name, chunked_parameter) in zip(
        dense_model.named_parameters(), chunked_model.named_parameters()
    ):
        assert dense_name == chunked_name
        torch.testing.assert_close(
            dense_parameter.grad, chunked_parameter.grad,
            atol=2e-6, rtol=2e-5,
        )
    assert dense.retained_autograd_activation_elements > dense.peak_activation_elements
    assert (
        chunked.retained_autograd_activation_elements
        > chunked.peak_activation_elements
    )
    assert (
        dense.retained_autograd_activation_elements
        == chunked.retained_autograd_activation_elements
    )


def test_batch_128_memory_accounting_includes_gate_hidden_activations():
    pair_chunk = 32
    batch = 128
    patches = 256
    top_k = 16
    hidden = 128
    pair_elements = pair_chunk * batch * patches
    expected_local = pair_elements * (hidden + 9) + pair_chunk * batch * top_k * 2
    shared = (
        batch * 768 * 4
        + batch * patches * 768 * 4
        + batch * 256 * 3
        + batch * patches * 256 * 3
        + batch
        + batch * patches
    )
    expected_retained = expected_local * (batch // pair_chunk) + shared
    assert expected_local == 143_785_984
    assert shared == 126_353_536
    assert expected_retained == 701_497_472
    assert expected_local * 4 / 2**20 == pytest.approx(548.5)
    assert expected_retained * 4 / 2**30 == pytest.approx(2.6133, rel=1e-3)


def test_symmetric_infonce_and_diagonal_orientation_match_reference():
    scores = torch.tensor([[2.0, 0.0, -1.0], [0.0, 3.0, 1.0], [-1.0, 1.0, 4.0]])
    labels = torch.arange(3)
    reference = 0.5 * (
        F.cross_entropy(scores / 0.07, labels)
        + F.cross_entropy(scores.T / 0.07, labels)
    )
    torch.testing.assert_close(symmetric_infonce(scores), reference)
    assert symmetric_infonce(scores) < symmetric_infonce(scores.roll(1, dims=1))


def test_losses_and_gradients_are_finite_and_only_adapter_trains():
    model = adapter()
    query, patches, priors = inputs(requires_grad=True)
    output = compute_chunked_mil_scores(
        model, query, patches, priors, mil_top_k=3, pair_chunk_size=2
    )
    losses = compute_e9_loss(output, query, patches)
    expected_text = (1 - F.cosine_similarity(
        output.query, F.normalize(query.detach(), dim=-1), dim=-1
    )).mean()
    expected_patch = (1 - F.cosine_similarity(
        output.patches, F.normalize(patches.detach(), dim=-1), dim=-1
    )).mean()
    torch.testing.assert_close(losses["anchor_text_loss"], expected_text)
    torch.testing.assert_close(losses["anchor_patch_loss"], expected_patch)
    diagonal = torch.arange(output.image_scores.shape[0])
    positive_scores = output.selected_scores[diagonal, diagonal]
    positive_attention = output.selected_attention[diagonal, diagonal]
    responsibility = torch.softmax(positive_scores / 0.10, dim=-1)
    expected_support = (
        1 - (responsibility * positive_attention).sum(dim=-1)
    ).mean()
    torch.testing.assert_close(losses["attention_support_loss"], expected_support)
    torch.testing.assert_close(losses["gate_loss"], output.gate_mean)
    expected_total = (
        losses["nce_loss"]
        + 0.10 * 0.5 * (expected_text + expected_patch)
        + 0.02 * expected_support
        + 0.001 * output.gate_mean
    )
    torch.testing.assert_close(losses["loss"], expected_total)
    losses["loss"].backward()
    assert query.grad is None and patches.grad is None and priors.grad is None
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)
    assert any(torch.count_nonzero(value) for value in gradients)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -1.0))
def test_invalid_hyperparameters_are_rejected(value):
    with pytest.raises(E9ValidationError):
        SparseRegionAlignmentConfig(
            embedding_dim=8, bottleneck_dim=4, dropout=0,
            residual_max=0.25, gamma_max=value, gate_hidden_dim=4,
        )


@pytest.mark.parametrize("value", (0.0, float("nan"), float("inf")))
def test_direct_loss_helper_rejects_invalid_mil_temperature(value):
    model = adapter()
    query, patches, priors = inputs()
    output = compute_chunked_mil_scores(
        model, query, patches, priors, mil_top_k=3, pair_chunk_size=2
    )
    with pytest.raises(E9ValidationError, match="mil_temperature"):
        compute_e9_loss(output, query, patches, mil_temperature=value)
