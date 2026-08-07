import copy
import inspect
import math
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import src.e8_balanced_retrieval_adapter as e8_module
import train_e8_balanced_retrieval_adapter as training_module
from src.e8_balanced_retrieval_adapter import (
    E8_ADAPTER_CHECKPOINT_FORMAT,
    E8_DIAGNOSTIC_METRIC_KEYS,
    E8ScoreOutput,
    BalancedAdapterOutput,
    BalancedRetrievalPrototypeAdapter,
    BalancedRetrievalPrototypeAdapterConfig,
    build_e8_run_identity,
    compute_alpha_calibration_loss,
    compute_balanced_mode_losses,
    compute_beta_calibration_loss,
    compute_corrupt_abstention_loss,
    compute_e8_loss,
    compute_e8_scores,
    compute_responsibility_reliability,
    corrupt_retrieval_by_rotation,
    load_e8_adapter_checkpoint,
    validate_e8_adapter_checkpoint,
    validate_e8_training_config,
)
from src.e7_training_bank import E7TrainingBankValidationError


def normalized(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(*shape, generator=generator), dim=-1)


def small_config(**overrides):
    values = {
        "embedding_dim": 12,
        "num_prototypes": 3,
        "num_attention_heads": 3,
        "num_cross_attention_layers": 2,
        "ffn_dim": 24,
        "dropout": 0.0,
        "alpha_max": 0.35,
        "beta_max": 0.30,
        "retrieval_count": 5,
        "assignment_temperature": 0.07,
        "retrieval_weight_temperature": 0.07,
        "responsibility_temperature": 0.10,
        "separation_margin": 0.50,
        "alpha_advantage_scale": 0.10,
        "beta_advantage_scale": 0.10,
    }
    values.update(overrides)
    return BalancedRetrievalPrototypeAdapterConfig(**values)


def adapter_inputs(batch=4, memory=5, dimension=12):
    query = normalized((batch, dimension), 1)
    vectors = normalized((batch * memory, dimension), 2).reshape(
        batch, memory, dimension
    )
    scores = torch.linspace(0.2, 0.9, batch * memory).reshape(batch, memory)
    valid = torch.ones(batch, memory, dtype=torch.bool)
    return query, vectors, scores, valid


def test_adapter_shapes_normalization_bounds_and_balanced_probabilities():
    adapter = BalancedRetrievalPrototypeAdapter(small_config()).eval()
    query, vectors, scores, valid = adapter_inputs()
    valid[1, -2:] = False
    output = adapter(query, vectors, scores, valid)

    assert output.mode_vectors.shape == (4, 3, 12)
    assert output.prototypes.shape == (4, 3, 12)
    assert output.alpha.shape == (4, 3)
    assert output.beta.shape == (4,)
    assert output.candidate_assignments.shape == (4, 5, 3)
    assert output.candidate_weights.shape == (4, 5)
    assert output.slot_mass.shape == (4, 3)
    assert output.retrieval_count.tolist() == [5, 3, 5, 5]
    for tensor in (
        output.mode_vectors,
        output.prototypes,
        output.alpha,
        output.beta,
        output.candidate_assignments,
        output.candidate_weights,
        output.slot_mass,
    ):
        assert torch.isfinite(tensor).all()
    torch.testing.assert_close(
        output.mode_vectors.norm(dim=-1), torch.ones(4, 3), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        output.prototypes.norm(dim=-1), torch.ones(4, 3), atol=1e-6, rtol=1e-6
    )
    assert torch.all((0 <= output.alpha) & (output.alpha <= 0.35))
    assert torch.all((0 <= output.beta) & (output.beta <= 0.30))
    torch.testing.assert_close(
        output.candidate_assignments[valid].sum(dim=-1),
        torch.ones(int(valid.sum())),
    )
    torch.testing.assert_close(output.candidate_weights.sum(dim=-1), torch.ones(4))
    torch.testing.assert_close(output.slot_mass.sum(dim=-1), torch.ones(4))
    assert torch.count_nonzero(output.candidate_assignments[~valid]) == 0
    assert torch.count_nonzero(output.candidate_weights[~valid]) == 0


def test_retrieval_permutation_invariance_and_invalid_padding_independence():
    adapter = BalancedRetrievalPrototypeAdapter(small_config()).eval()
    query, vectors, scores, valid = adapter_inputs()
    valid[:, -1] = False
    expected = adapter(query, vectors, scores, valid)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    actual = adapter(
        query,
        vectors[:, permutation],
        scores[:, permutation],
        valid[:, permutation],
    )
    for name in ("mode_vectors", "prototypes", "alpha", "beta", "slot_mass"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
    torch.testing.assert_close(
        actual.candidate_assignments,
        expected.candidate_assignments[:, permutation],
    )
    torch.testing.assert_close(
        actual.candidate_weights,
        expected.candidate_weights[:, permutation],
    )

    changed_vectors = vectors.clone()
    changed_scores = scores.clone()
    changed_vectors[~valid] = 1e6
    changed_scores[~valid] = -1e6
    changed = adapter(query, changed_vectors, changed_scores, valid)
    for name in BalancedAdapterOutput.__dataclass_fields__:
        first = getattr(expected, name)
        second = getattr(changed, name)
        if first.dtype == torch.int64:
            assert torch.equal(first, second)
        else:
            torch.testing.assert_close(first, second)


def test_empty_retrieval_is_exact_e3_and_all_assignment_outputs_are_zero():
    adapter = BalancedRetrievalPrototypeAdapter(small_config()).eval()
    query, vectors, scores, valid = adapter_inputs()
    valid.zero_()
    output = adapter(query, vectors, scores, valid)
    expected = F.normalize(query, dim=-1)[:, None].expand_as(output.prototypes)
    assert torch.equal(output.mode_vectors, expected)
    assert torch.equal(output.prototypes, expected)
    for value in (
        output.alpha,
        output.beta,
        output.candidate_assignments,
        output.candidate_weights,
        output.slot_mass,
        output.retrieval_count,
    ):
        assert torch.count_nonzero(value) == 0


def test_slot_mass_and_specialization_losses_match_independent_references():
    adapter = BalancedRetrievalPrototypeAdapter(small_config()).eval()
    query, vectors, scores, valid = adapter_inputs(batch=3)
    valid[2, 2:] = False
    output = adapter(query, vectors, scores, valid)
    losses = compute_balanced_mode_losses(output, vectors, valid)

    expected_slot_mass = torch.zeros_like(output.slot_mass)
    for batch in range(3):
        for candidate in range(5):
            for slot in range(3):
                expected_slot_mass[batch, slot] += (
                    output.candidate_weights[batch, candidate]
                    * output.candidate_assignments[batch, candidate, slot]
                )
    torch.testing.assert_close(output.slot_mass, expected_slot_mass)

    modes = F.normalize(output.mode_vectors, dim=-1)
    candidates = F.normalize(vectors, dim=-1)
    coverage_rows = []
    sharpness_rows = []
    balance_rows = []
    separation_values = []
    for batch in range(3):
        coverage = 0.0
        sharpness = 0.0
        for candidate in range(5):
            if not valid[batch, candidate]:
                continue
            quality = 0.0
            entropy = 0.0
            for slot in range(3):
                assignment = output.candidate_assignments[batch, candidate, slot]
                quality += assignment * torch.dot(
                    candidates[batch, candidate], modes[batch, slot]
                )
                entropy -= assignment * assignment.clamp_min(1e-30).log()
            weight = output.candidate_weights[batch, candidate]
            coverage += weight * (1 - quality)
            sharpness += weight * entropy / math.log(3)
        coverage_rows.append(coverage)
        sharpness_rows.append(sharpness)
        if int(valid[batch].sum()) >= 3:
            balance_rows.append(
                3 * ((output.slot_mass[batch] - 1 / 3) ** 2).mean()
            )
        for first in range(3):
            for second in range(first + 1, 3):
                separation_values.append(
                    F.relu(torch.dot(modes[batch, first], modes[batch, second]) - 0.5)
                )
    torch.testing.assert_close(losses["coverage_loss"], torch.stack(coverage_rows).mean())
    torch.testing.assert_close(losses["balance_loss"], torch.stack(balance_rows).mean())
    torch.testing.assert_close(
        losses["assignment_sharpness_loss"], torch.stack(sharpness_rows).mean()
    )
    torch.testing.assert_close(
        losses["mode_separation_loss"], torch.stack(separation_values).mean()
    )


def manual_output(modes, assignments, weights):
    batch, memory, slots = assignments.shape
    return BalancedAdapterOutput(
        mode_vectors=modes,
        prototypes=modes,
        alpha=modes.new_zeros((batch, slots)),
        beta=modes.new_zeros(batch),
        candidate_assignments=assignments,
        candidate_weights=weights,
        slot_mass=torch.einsum("bm,bmk->bk", weights, assignments),
        retrieval_count=torch.full((batch,), memory, dtype=torch.int64),
    )


def test_collapsed_modes_have_larger_separation_and_sharpness_penalties():
    candidates = torch.eye(3).unsqueeze(0)
    valid = torch.ones(1, 3, dtype=torch.bool)
    weights = torch.full((1, 3), 1 / 3)
    collapsed_modes = torch.tensor([[[1.0, 0, 0]]]).expand(1, 3, 3).clone()
    collapsed_assignments = torch.full((1, 3, 3), 1 / 3)
    collapsed = compute_balanced_mode_losses(
        manual_output(collapsed_modes, collapsed_assignments, weights),
        candidates,
        valid,
    )
    distinct_modes = torch.eye(3).unsqueeze(0)
    distinct_assignments = torch.softmax(torch.eye(3).unsqueeze(0) / 0.07, dim=-1)
    distinct = compute_balanced_mode_losses(
        manual_output(distinct_modes, distinct_assignments, weights),
        candidates,
        valid,
    )
    assert collapsed["mode_separation_loss"] > distinct["mode_separation_loss"]
    assert collapsed["assignment_sharpness_loss"] > distinct["assignment_sharpness_loss"]


def test_specialization_losses_give_finite_nonidentical_mode_token_gradients():
    adapter = BalancedRetrievalPrototypeAdapter(small_config()).train()
    query, vectors, scores, valid = adapter_inputs()
    output = adapter(query, vectors, scores, valid)
    losses = compute_balanced_mode_losses(output, vectors, valid)
    sum(losses.values()).backward()
    gradient = adapter.mode_tokens.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.all(gradient.norm(dim=-1) > 0)
    assert not torch.equal(gradient[0], gradient[1])


def test_alpha_calibration_matches_detached_support_reference():
    adapter = BalancedRetrievalPrototypeAdapter(small_config()).eval()
    query, vectors, scores, valid = adapter_inputs()
    output = adapter(query, vectors, scores, valid)
    loss, target = compute_alpha_calibration_loss(
        output,
        query,
        vectors,
        valid,
        alpha_max=0.35,
        alpha_advantage_scale=0.10,
    )
    support = (output.candidate_weights[..., None] * output.candidate_assignments).detach()
    mass = support.sum(dim=1)
    mode_quality = (
        support
        * torch.einsum(
            "bkd,bmd->bmk", F.normalize(output.mode_vectors, dim=-1), F.normalize(vectors, dim=-1)
        )
    ).sum(dim=1) / mass
    base_quality = (
        support * torch.einsum("bd,bmd->bm", F.normalize(query, dim=-1), F.normalize(vectors, dim=-1))[..., None]
    ).sum(dim=1) / mass
    expected_target = ((mode_quality - base_quality) / 0.10).clamp(0, 1).detach()
    expected_loss = F.smooth_l1_loss(output.alpha / 0.35, expected_target)
    torch.testing.assert_close(target, expected_target)
    torch.testing.assert_close(loss, expected_loss)
    zero_loss, _ = compute_alpha_calibration_loss(
        output, query, vectors, valid, alpha_max=0.0
    )
    assert torch.equal(zero_loss, torch.zeros_like(zero_loss))


def test_masked_routing_and_alpha_target_match_independent_reconstruction():
    adapter = BalancedRetrievalPrototypeAdapter(small_config()).eval()
    query, vectors, scores, valid = adapter_inputs(batch=3)
    valid[0, -1] = False
    valid[1, -2:] = False
    output = adapter(query, vectors, scores, valid)

    expected_weights = torch.zeros_like(output.candidate_weights)
    expected_assignments = torch.zeros_like(output.candidate_assignments)
    normalized_vectors = F.normalize(vectors, dim=-1)
    for batch in range(len(query)):
        indices = torch.nonzero(valid[batch], as_tuple=False).flatten()
        weight_logits = (
            scores[batch, indices] / adapter.config.retrieval_weight_temperature
        )
        weight_exp = torch.exp(weight_logits - weight_logits.max())
        expected_weights[batch, indices] = weight_exp / weight_exp.sum()
        for memory in indices.tolist():
            assignment_logits = torch.stack(
                [
                    torch.dot(
                        normalized_vectors[batch, memory],
                        output.mode_vectors[batch, slot],
                    )
                    / adapter.config.assignment_temperature
                    for slot in range(adapter.config.num_prototypes)
                ]
            )
            assignment_exp = torch.exp(
                assignment_logits - assignment_logits.max()
            )
            expected_assignments[batch, memory] = (
                assignment_exp / assignment_exp.sum()
            )
    expected_slot_mass = torch.zeros_like(output.slot_mass)
    for batch in range(len(query)):
        for memory in range(vectors.shape[1]):
            for slot in range(adapter.config.num_prototypes):
                expected_slot_mass[batch, slot] += (
                    expected_weights[batch, memory]
                    * expected_assignments[batch, memory, slot]
                )

    torch.testing.assert_close(output.candidate_weights, expected_weights)
    torch.testing.assert_close(output.candidate_assignments, expected_assignments)
    torch.testing.assert_close(output.slot_mass, expected_slot_mass)

    routing_losses = compute_balanced_mode_losses(output, vectors, valid)
    coverage_rows = []
    balance_rows = []
    sharpness_rows = []
    for batch in range(len(query)):
        coverage = query.new_zeros(())
        sharpness = query.new_zeros(())
        for memory in range(vectors.shape[1]):
            if not valid[batch, memory]:
                continue
            quality = sum(
                expected_assignments[batch, memory, slot]
                * torch.dot(
                    output.mode_vectors[batch, slot],
                    normalized_vectors[batch, memory],
                )
                for slot in range(adapter.config.num_prototypes)
            )
            weight = expected_weights[batch, memory]
            coverage = coverage + weight * (1 - quality)
            probabilities = expected_assignments[batch, memory]
            entropy = -sum(
                probability * probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
                for probability in probabilities
            ) / math.log(adapter.config.num_prototypes)
            sharpness = sharpness + weight * entropy
        coverage_rows.append(coverage)
        sharpness_rows.append(sharpness)
        if int(valid[batch].sum()) >= adapter.config.num_prototypes:
            balance_rows.append(
                adapter.config.num_prototypes
                * (
                    expected_slot_mass[batch]
                    - 1 / adapter.config.num_prototypes
                ).square().mean()
            )
    torch.testing.assert_close(
        routing_losses["coverage_loss"],
        torch.stack(coverage_rows).mean(),
    )
    torch.testing.assert_close(
        routing_losses["balance_loss"],
        torch.stack(balance_rows).mean(),
    )
    torch.testing.assert_close(
        routing_losses["assignment_sharpness_loss"],
        torch.stack(sharpness_rows).mean(),
    )

    _, actual_target = compute_alpha_calibration_loss(
        output,
        query,
        vectors,
        valid,
        alpha_max=adapter.config.alpha_max,
        alpha_advantage_scale=adapter.config.alpha_advantage_scale,
    )
    expected_target = torch.zeros_like(actual_target)
    normalized_query = F.normalize(query, dim=-1)
    for batch in range(len(query)):
        for slot in range(adapter.config.num_prototypes):
            support = expected_weights[batch] * expected_assignments[batch, :, slot]
            mass = support.sum()
            if mass > 0:
                mode_quality = sum(
                    support[memory]
                    * torch.dot(
                        output.mode_vectors[batch, slot],
                        normalized_vectors[batch, memory],
                    )
                    for memory in range(vectors.shape[1])
                ) / mass
                base_quality = sum(
                    support[memory]
                    * torch.dot(
                        normalized_query[batch],
                        normalized_vectors[batch, memory],
                    )
                    for memory in range(vectors.shape[1])
                ) / mass
                expected_target[batch, slot] = (
                    (mode_quality - base_quality)
                    / adapter.config.alpha_advantage_scale
                ).clamp(0, 1)
    torch.testing.assert_close(actual_target, expected_target)


def test_unsupported_alpha_slot_is_zero_and_alpha_head_gets_gradient():
    adapter = BalancedRetrievalPrototypeAdapter(small_config()).train()
    query, vectors, scores, valid = adapter_inputs()
    output = adapter(query, vectors, scores, valid)

    assignments = output.candidate_assignments.detach().clone()
    assignments[..., -1] = 0
    assignments[..., :-1] /= assignments[..., :-1].sum(
        dim=-1,
        keepdim=True,
    )
    unsupported = replace(
        output,
        candidate_assignments=assignments,
        slot_mass=torch.einsum(
            "bm,bmk->bk",
            output.candidate_weights.detach(),
            assignments,
        ),
    )
    _, unsupported_target = compute_alpha_calibration_loss(
        unsupported,
        query,
        vectors,
        valid,
        alpha_max=adapter.config.alpha_max,
        alpha_advantage_scale=adapter.config.alpha_advantage_scale,
    )
    assert torch.equal(
        unsupported_target[:, -1],
        torch.zeros_like(unsupported_target[:, -1]),
    )

    adapter.zero_grad(set_to_none=True)
    alpha_loss, _ = compute_alpha_calibration_loss(
        output,
        query,
        vectors,
        valid,
        alpha_max=adapter.config.alpha_max,
        alpha_advantage_scale=adapter.config.alpha_advantage_scale,
    )
    alpha_loss.backward()
    gradients = [
        parameter.grad
        for parameter in adapter.alpha_head.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_responsibility_scores_beta_targets_and_symmetric_nce_match_loops():
    query = normalized((4, 9), 20)
    target = normalized((4, 9), 21)
    prototypes = normalized((12, 9), 22).reshape(4, 3, 9)
    beta = torch.tensor([0.0, 0.1, 0.2, 0.3])
    actual = compute_e8_scores(query, target, prototypes, beta)

    base = torch.empty(4, 4)
    modes = torch.empty(4, 4, 3)
    grounded = torch.empty(4, 4)
    reliability = torch.empty(4, 4)
    final = torch.empty(4, 4)
    for row in range(4):
        for column in range(4):
            base[row, column] = torch.dot(query[row], target[column])
            for slot in range(3):
                modes[row, column, slot] = torch.dot(
                    prototypes[row, slot], target[column]
                )
            grounded[row, column] = 0.10 * (
                torch.logsumexp(modes[row, column] / 0.10, dim=0) - math.log(3)
            )
            responsibility = torch.softmax(modes[row, column] / 0.10, dim=0)
            entropy = -(responsibility * responsibility.log()).sum()
            reliability[row, column] = (1 - entropy / math.log(3)).clamp(0, 1)
            effective = beta[row] * reliability[row, column]
            final[row, column] = (1 - effective) * base[row, column] + effective * grounded[row, column]
    torch.testing.assert_close(actual.base_score, base)
    torch.testing.assert_close(actual.prototype_score, modes)
    torch.testing.assert_close(actual.grounded_score, grounded)
    torch.testing.assert_close(actual.reliability, reliability)
    torch.testing.assert_close(actual.final_score, final)

    beta_loss, beta_target = compute_beta_calibration_loss(
        actual, beta, beta_max=0.30, beta_advantage_scale=0.10
    )
    labels = torch.arange(4)
    expected_target = (
        (
            F.cross_entropy(base / 0.07, labels, reduction="none")
            - F.cross_entropy(grounded / 0.07, labels, reduction="none")
        )
        / 0.10
    ).clamp(0, 1)
    torch.testing.assert_close(beta_target, expected_target)
    torch.testing.assert_close(
        beta_loss, F.smooth_l1_loss(beta / 0.30, expected_target)
    )


def test_identical_scores_zero_reliability_dominant_mode_high_and_k1_is_one():
    identical = torch.full((2, 5, 3), 0.25)
    responsibilities, reliability = compute_responsibility_reliability(identical)
    torch.testing.assert_close(responsibilities, torch.full_like(responsibilities, 1 / 3))
    assert torch.equal(reliability, torch.zeros_like(reliability))
    dominant = identical.clone()
    dominant[..., 0] = 1.0
    _, dominant_reliability = compute_responsibility_reliability(dominant, 0.01)
    assert torch.all(dominant_reliability > 0.99)
    _, one_reliability = compute_responsibility_reliability(torch.randn(2, 5, 1))
    assert torch.equal(one_reliability, torch.ones_like(one_reliability))
    _, empty_reliability = compute_responsibility_reliability(
        identical, prototype_valid_mask=torch.zeros(2, 3, dtype=torch.bool)
    )
    assert torch.count_nonzero(empty_reliability) == 0


def test_beta_zero_and_no_retrieval_scores_are_bitwise_e3():
    query = normalized((3, 8), 30)
    target = normalized((3, 8), 31)
    prototypes = normalized((9, 8), 32).reshape(3, 3, 8)
    zero = compute_e8_scores(query, target, prototypes, torch.zeros(3))
    assert torch.equal(zero.final_score, zero.base_score)
    empty = compute_e8_scores(
        query,
        target,
        prototypes,
        torch.full((3,), 0.3),
        has_retrieval=torch.zeros(3, dtype=torch.bool),
    )
    assert torch.equal(empty.final_score, empty.base_score)


def test_beta_calibration_uses_the_complete_batch_reference():
    base = torch.zeros(2, 2)
    prototype = torch.zeros(2, 2, 3)
    scores = E8ScoreOutput(
        base_score=base,
        prototype_score=prototype,
        grounded_score=base.clone(),
        responsibility=torch.full_like(prototype, 1 / 3),
        reliability=base.clone(),
        effective_beta=base.clone(),
        final_score=base.clone(),
        valid_query_mask=torch.tensor([True, False]),
    )
    beta = torch.tensor([0.30, 0.00])
    actual, target = compute_beta_calibration_loss(
        scores,
        beta,
        beta_max=0.30,
    )
    expected_target = torch.zeros_like(beta)
    expected = F.smooth_l1_loss(
        beta / 0.30,
        expected_target,
        reduction="mean",
    )
    assert torch.equal(target, expected_target)
    assert torch.equal(actual, expected)
    assert actual.item() == 0.25

    zero, _ = compute_beta_calibration_loss(scores, beta * 0, beta_max=0.0)
    assert torch.equal(zero, torch.zeros_like(zero))


def test_corrupt_abstention_is_complete_batch_mean_of_squares():
    beta = torch.tensor([0.30, 0.00])
    valid = torch.tensor([True, False])
    actual = compute_corrupt_abstention_loss(
        beta,
        valid,
        beta_max=0.30,
    )
    expected = ((beta / 0.30) ** 2).mean()
    assert torch.equal(actual, expected)
    assert actual.item() == 0.5

    zero = compute_corrupt_abstention_loss(beta * 0, valid, beta_max=0.0)
    assert torch.equal(zero, torch.zeros_like(zero))


def test_corruption_rotation_is_deterministic_and_removes_receiver_images():
    vectors = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    scores = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    valid = torch.ones(2, 3, dtype=torch.bool)
    images = torch.tensor([[20, 11, 21], [10, 30, 31]], dtype=torch.int64)
    annotations = torch.arange(100, 106, dtype=torch.int64).reshape(2, 3)
    query_images = torch.tensor([10, 20], dtype=torch.int64)
    first = corrupt_retrieval_by_rotation(
        vectors, scores, valid, images, query_images, annotations
    )
    second = corrupt_retrieval_by_rotation(
        vectors, scores, valid, images, query_images, annotations
    )
    for name in first.__dataclass_fields__:
        assert torch.equal(getattr(first, name), getattr(second, name))
    assert first.invalidated_same_image_count == 2
    assert first.exclusion_violations == 0
    assert not torch.any(first.valid_mask & (first.image_ids == query_images[:, None]))
    assert torch.count_nonzero(first.routed_vectors[~first.valid_mask]) == 0
    assert torch.count_nonzero(first.scores[~first.valid_mask]) == 0
    assert torch.all(first.annotation_ids[~first.valid_mask] == -1)


def test_full_loss_orientation_corruption_is_auxiliary_and_frozen_inputs_get_no_grad():
    adapter = BalancedRetrievalPrototypeAdapter(small_config()).train()
    query, vectors, retrieval_scores, valid = adapter_inputs()
    query.requires_grad_()
    targets = normalized((4, 12), 40).requires_grad_()
    vectors.requires_grad_()
    retrieval_scores.requires_grad_()
    output = adapter(query, vectors, retrieval_scores, valid)
    scores = compute_e8_scores(query, targets, output.prototypes, output.beta)
    corrupt_beta = torch.full((4,), 0.1, requires_grad=True)
    losses = compute_e8_loss(
        scores,
        output,
        query,
        vectors,
        valid,
        corrupted_beta=corrupt_beta,
        corrupted_valid_rows=torch.ones(4, dtype=torch.bool),
    )
    labels = torch.arange(4)
    expected_nce = 0.5 * (
        F.cross_entropy(scores.final_score / 0.07, labels)
        + F.cross_entropy(scores.final_score.T / 0.07, labels)
    )
    torch.testing.assert_close(losses["nce_loss"], expected_nce)
    losses["loss"].backward()
    gradients = [parameter.grad for parameter in adapter.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert any(torch.count_nonzero(value) for value in gradients)
    assert query.grad is None
    assert targets.grad is None
    assert vectors.grad is None
    assert retrieval_scores.grad is None
    assert corrupt_beta.grad is not None

    second = compute_e8_loss(
        scores,
        output,
        query,
        vectors,
        valid,
        corrupted_beta=torch.full((4,), 0.2),
        corrupted_valid_rows=torch.ones(4, dtype=torch.bool),
    )
    torch.testing.assert_close(second["nce_loss"], losses["nce_loss"])


def bank_identity(split):
    return {
        "format_version": "talk2dino-e7-training-bank-v1",
        "split_name": split,
        "source_feature_sha256": ("a" if split == "train" else "b") * 64,
        "annotation_id_fingerprint": ("c" if split == "train" else "d") * 64,
        "selected_annotation_count": 10,
        "e3_config_sha256": "e" * 64,
        "e3_checkpoint_sha256": "f" * 64,
        "routing_temperature": 0.10,
        "source_git_commit": "1" * 40,
    }


def complete_training_config(config):
    return {
        "seed": 42,
        "num_epochs": 2,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "cosine",
        "warmup_ratio": 0.05,
        "amp_dtype": "bfloat16",
        "save_best_model": True,
        "early_stopping_patience": 2,
        "adapter": asdict(config),
        "retrieval": {
            "queue_size": 16,
            "candidate_pool": 5,
            "retrieval_count": config.retrieval_count,
            "retrieval_min_similarity": 0.18,
        },
        "loss": {
            "logit_temperature": 0.07,
            "prototype_temperature": 0.10,
            "anchor_weight": 0.10,
            "coverage_weight": 0.10,
            "balance_weight": 0.05,
            "sharpness_weight": 0.02,
            "separation_weight": 0.05,
            "alpha_calibration_weight": 0.05,
            "beta_calibration_weight": 0.10,
            "corrupt_abstention_weight": 0.10,
            "beta_usage_weight": 0.001,
        },
    }


def complete_diagnostic_summary(value=0.0):
    metrics = {name: value for name in E8_DIAGNOSTIC_METRIC_KEYS}
    return {"train": dict(metrics), "validation": dict(metrics)}


def checkpoint_payload(
    adapter,
    config,
    *,
    train_is_pilot=False,
    validation_is_pilot=False,
    train_complete=True,
    validation_complete=True,
    max_train_batches=None,
    max_validation_batches=None,
    source_git_dirty=False,
):
    provenance = {
        "source_git_commit": "2" * 40,
        "source_git_dirty": source_git_dirty,
        "source_git_diff_sha256": "9" * 64 if source_git_dirty else None,
    }
    return {
        "format_version": E8_ADAPTER_CHECKPOINT_FORMAT,
        "adapter_state_dict": adapter.state_dict(),
        "architecture_config": asdict(config),
        "training_config": complete_training_config(config),
        "e3_identity": {
            "config_sha256": "e" * 64,
            "checkpoint_sha256": "f" * 64,
        },
        "train_bank_identity": bank_identity("train"),
        "validation_bank_identity": bank_identity("val"),
        "run_identity": build_e8_run_identity(
            {"is_pilot": train_is_pilot, "complete": train_complete},
            {
                "is_pilot": validation_is_pilot,
                "complete": validation_complete,
            },
            max_train_batches=max_train_batches,
            max_validation_batches=max_validation_batches,
            source_git_dirty=source_git_dirty,
        ),
        "source_git_provenance": provenance,
        "epoch": 3,
        "best_validation_metric": 1.0,
        "final_diagnostic_summary": complete_diagnostic_summary(1.0),
    }


def test_checkpoint_closed_schema_round_trip_and_preconstruction_rejection(tmp_path, monkeypatch):
    config = small_config()
    adapter = BalancedRetrievalPrototypeAdapter(config).eval()
    payload = checkpoint_payload(adapter, config)
    validate_e8_adapter_checkpoint(payload)
    path = tmp_path / "adapter.pth"
    torch.save(payload, path)
    loaded, _ = load_e8_adapter_checkpoint(
        path,
        expected_e3_identity=payload["e3_identity"],
        expected_train_bank_identity=payload["train_bank_identity"],
    )
    query, vectors, scores, valid = adapter_inputs()
    torch.testing.assert_close(
        loaded(query, vectors, scores, valid).prototypes,
        adapter(query, vectors, scores, valid).prototypes,
    )
    unknown = copy.deepcopy(payload)
    unknown["retrieval_bank"] = torch.randn(1)
    with pytest.raises(ValueError, match="closed schema"):
        validate_e8_adapter_checkpoint(unknown)

    incompatible = copy.deepcopy(payload)
    incompatible["e3_identity"]["config_sha256"] = "0" * 64
    bad_path = tmp_path / "bad.pth"
    torch.save(incompatible, bad_path)
    constructed = []
    monkeypatch.setattr(
        e8_module,
        "BalancedRetrievalPrototypeAdapter",
        lambda *args, **kwargs: constructed.append(True),
    )
    with pytest.raises(ValueError, match="does not match"):
        load_e8_adapter_checkpoint(bad_path)
    assert constructed == []


def test_v2_run_identity_production_pilot_bounded_and_dirty_rules():
    adapter = BalancedRetrievalPrototypeAdapter(small_config()).eval()

    production = checkpoint_payload(adapter, adapter.config)
    validate_e8_adapter_checkpoint(production)
    assert production["run_identity"] == {
        "train_bank_is_pilot": False,
        "validation_bank_is_pilot": False,
        "train_bank_complete": True,
        "validation_bank_complete": True,
        "max_train_batches": None,
        "max_validation_batches": None,
        "bounded_training": False,
        "pilot_training": False,
        "production_eligible": True,
    }

    pilot = checkpoint_payload(
        adapter,
        adapter.config,
        train_is_pilot=True,
        validation_is_pilot=True,
    )
    with pytest.raises(ValueError, match="allow_pilot_checkpoint"):
        validate_e8_adapter_checkpoint(pilot)
    validate_e8_adapter_checkpoint(pilot, allow_pilot_checkpoint=True)
    assert pilot["run_identity"]["pilot_training"] is True
    assert pilot["run_identity"]["production_eligible"] is False

    for limit_name in ("max_train_batches", "max_validation_batches"):
        bounded = checkpoint_payload(
            adapter,
            adapter.config,
            **{limit_name: 1},
        )
        with pytest.raises(ValueError, match="allow_pilot_checkpoint"):
            validate_e8_adapter_checkpoint(bounded)
        validate_e8_adapter_checkpoint(
            bounded,
            allow_pilot_checkpoint=True,
        )
        assert bounded["run_identity"]["bounded_training"] is True
        assert bounded["run_identity"]["production_eligible"] is False

    dirty = checkpoint_payload(
        adapter,
        adapter.config,
        train_is_pilot=True,
        validation_is_pilot=True,
        source_git_dirty=True,
    )
    with pytest.raises(ValueError, match="dirty-source"):
        validate_e8_adapter_checkpoint(dirty, allow_pilot_checkpoint=True)
    validate_e8_adapter_checkpoint(
        dirty,
        allow_pilot_checkpoint=True,
        allow_dirty_source=True,
    )
    assert dirty["run_identity"]["production_eligible"] is False


@pytest.mark.parametrize("train_is_pilot,validation_is_pilot", ((False, True), (True, False)))
def test_mixed_pilot_bank_pair_is_rejected(train_is_pilot, validation_is_pilot):
    with pytest.raises(ValueError, match="identical is_pilot"):
        build_e8_run_identity(
            {"is_pilot": train_is_pilot, "complete": True},
            {"is_pilot": validation_is_pilot, "complete": True},
            max_train_batches=None,
            max_validation_batches=None,
            source_git_dirty=False,
        )

    def metadata(is_pilot):
        return {
            "is_pilot": is_pilot,
            "complete": True,
            "source_git_dirty": False,
            "source_git_commit": "1" * 40,
            "e3_config_sha256": "2" * 64,
            "e3_checkpoint_sha256": "3" * 64,
            "format_version": "talk2dino-e7-training-bank-v1",
            "dimensions": {
                "caption_embeddings": 512,
                "mapped_query_embeddings": 768,
                "routed_target_embeddings": 768,
            },
        }

    with pytest.raises(E7TrainingBankValidationError, match="identical is_pilot"):
        training_module._validate_e8_training_bank_pair(
            {"metadata": metadata(train_is_pilot)},
            {"metadata": metadata(validation_is_pilot)},
        )


def test_forged_production_flag_and_legacy_v1_are_rejected():
    adapter = BalancedRetrievalPrototypeAdapter(small_config()).eval()
    bounded = checkpoint_payload(
        adapter,
        adapter.config,
        max_train_batches=1,
    )
    bounded["run_identity"]["production_eligible"] = True
    with pytest.raises(ValueError, match="production_eligible.*inconsistent"):
        validate_e8_adapter_checkpoint(
            bounded,
            allow_pilot_checkpoint=True,
        )

    legacy = checkpoint_payload(adapter, adapter.config)
    legacy["format_version"] = "talk2dino-e8-balanced-adapter-v1"
    legacy.pop("run_identity")
    with pytest.raises(ValueError, match="legacy E8 v1"):
        validate_e8_adapter_checkpoint(legacy)


@pytest.mark.parametrize(
    "mutation,match",
    (
        (lambda payload: payload["training_config"].update({"unknown": 1}), "closed schema"),
        (lambda payload: payload["training_config"].pop("scheduler"), "closed schema"),
        (lambda payload: payload["training_config"]["adapter"].pop("alpha_max"), "closed schema"),
        (lambda payload: payload["training_config"]["adapter"].update({"alpha_max": True}), "alpha_max"),
        (lambda payload: payload["training_config"]["adapter"].update({"beta_max": 0.2}), "exactly equal"),
        (lambda payload: payload["final_diagnostic_summary"]["train"].update({"unknown": 1.0}), "closed schema"),
        (lambda payload: payload["final_diagnostic_summary"]["train"].update({"loss": None}), "must be finite"),
        (lambda payload: payload["final_diagnostic_summary"]["train"].update({"loss": float("inf")}), "must be finite"),
    ),
)
def test_checkpoint_nested_schemas_are_deeply_closed(mutation, match):
    adapter = BalancedRetrievalPrototypeAdapter(small_config()).eval()
    payload = checkpoint_payload(adapter, adapter.config)
    mutation(payload)
    with pytest.raises(ValueError, match=match):
        validate_e8_adapter_checkpoint(payload)


@pytest.mark.parametrize(
    "loader_kwargs,match",
    (
        ({"expected_prototype_temperature": 0.11}, "prototype temperatures differ"),
        ({"expected_responsibility_temperature": 0.11}, "responsibility temperatures differ"),
        ({"expected_retrieval_count": 4}, "retrieval counts differ"),
        ({"expected_embedding_dim": 768}, "embedding dimension"),
    ),
)
def test_inference_identity_mismatch_fails_before_adapter_construction(
    tmp_path,
    monkeypatch,
    loader_kwargs,
    match,
):
    config = small_config()
    adapter = BalancedRetrievalPrototypeAdapter(config).eval()
    path = tmp_path / "adapter.pth"
    torch.save(checkpoint_payload(adapter, config), path)
    constructed = []

    def forbidden_constructor(*args, **kwargs):
        constructed.append((args, kwargs))
        raise AssertionError("adapter constructor must not run")

    monkeypatch.setattr(
        e8_module,
        "BalancedRetrievalPrototypeAdapter",
        forbidden_constructor,
    )
    with pytest.raises(ValueError, match=match):
        load_e8_adapter_checkpoint(path, **loader_kwargs)
    assert constructed == []


@pytest.mark.parametrize(
    "payload_kwargs,match",
    (
        ({"train_is_pilot": True, "validation_is_pilot": True}, "allow_pilot_checkpoint"),
        ({"max_validation_batches": 1}, "allow_pilot_checkpoint"),
        ({"train_complete": False}, "allow_pilot_checkpoint"),
        (
            {
                "train_is_pilot": True,
                "validation_is_pilot": True,
                "source_git_dirty": True,
            },
            "dirty-source",
        ),
    ),
)
def test_nonproduction_checkpoint_fails_before_adapter_construction(
    tmp_path,
    monkeypatch,
    payload_kwargs,
    match,
):
    config = small_config()
    adapter = BalancedRetrievalPrototypeAdapter(config).eval()
    payload = checkpoint_payload(adapter, config, **payload_kwargs)
    path = tmp_path / "bounded.pth"
    torch.save(payload, path)
    constructed = []
    monkeypatch.setattr(
        e8_module,
        "BalancedRetrievalPrototypeAdapter",
        lambda *args, **kwargs: constructed.append(True),
    )
    with pytest.raises(ValueError, match=match):
        load_e8_adapter_checkpoint(path)
    assert constructed == []


def test_training_config_validator_is_shared_and_rejects_bool_numeric():
    config = small_config()
    training_config = complete_training_config(config)
    assert validate_e8_training_config(training_config) == training_config
    invalid = copy.deepcopy(training_config)
    invalid["loss"]["prototype_temperature"] = True
    with pytest.raises(ValueError, match="prototype_temperature"):
        validate_e8_training_config(invalid)


def test_generic_loader_requires_explicit_pilot_override(tmp_path):
    config = small_config()
    adapter = BalancedRetrievalPrototypeAdapter(config).eval()
    path = tmp_path / "pilot.pth"
    torch.save(
        checkpoint_payload(
            adapter,
            config,
            train_is_pilot=True,
            validation_is_pilot=True,
        ),
        path,
    )
    with pytest.raises(ValueError, match="allow_pilot_checkpoint"):
        load_e8_adapter_checkpoint(path)
    loaded, checkpoint = load_e8_adapter_checkpoint(
        path,
        allow_pilot_checkpoint=True,
    )
    assert isinstance(loaded, BalancedRetrievalPrototypeAdapter)
    assert checkpoint["run_identity"]["production_eligible"] is False


def initialize_git_repository(path):
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    tracked = path / "implementation.py"
    tracked.write_text("VERSION = 1\n")
    subprocess.run(["git", "add", "implementation.py"], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=E8 Test",
            "-c",
            "user.email=e8-test@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=path,
        check=True,
    )
    return tracked


@pytest.mark.parametrize("dirty_kind", ("tracked", "staged", "untracked"))
def test_source_mutation_blocks_publication_and_preserves_checkpoint(tmp_path, dirty_kind):
    repository = tmp_path / dirty_kind
    tracked = initialize_git_repository(repository)
    initial = training_module.source_git_provenance(repository)
    output = tmp_path / f"{dirty_kind}.pth"
    output.write_bytes(b"existing-checkpoint")
    if dirty_kind == "tracked":
        tracked.write_text("VERSION = 2\n")
    elif dirty_kind == "staged":
        tracked.write_text("VERSION = 2\n")
        subprocess.run(["git", "add", "implementation.py"], cwd=repository, check=True)
    else:
        (repository / "new_e8_source.py").write_text("E8 = True\n")
    with pytest.raises(E7TrainingBankValidationError, match="provenance changed"):
        training_module._publish_verified_checkpoint(
            {"new": "checkpoint"},
            output,
            repository_root=repository,
            initial_provenance=initial,
            allow_dirty_source=False,
        )
    assert output.read_bytes() == b"existing-checkpoint"


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_temperatures_and_scales_are_rejected(value):
    for field in (
        "assignment_temperature",
        "retrieval_weight_temperature",
        "responsibility_temperature",
        "alpha_advantage_scale",
        "beta_advantage_scale",
    ):
        with pytest.raises(ValueError, match=field):
            small_config(**{field: value})


def test_training_order_validation_memory_and_scientific_isolation_are_explicit():
    training_source = inspect.getsource(training_module._run_epoch)
    assert training_source.index("corrupted_output =") < training_source.index("queue.update(")
    assert training_source.index("score_output =") < training_source.index("queue.update(")
    assert "corrupted_output" not in training_source[
        training_source.index("compute_e8_scores(") : training_source.index("losses =")
    ]
    train_source = inspect.getsource(training_module.train_adapter)
    assert "validation_queue.initialize_from_bank(train_bank" in train_source
    assert "validation_queue.update" not in train_source
    assert "adapter.parameters()" in train_source

    implementation = Path("src/e8_balanced_retrieval_adapter.py").read_text().lower()
    training = Path("train_e8_balanced_retrieval_adapter.py").read_text().lower()
    for forbidden in (
        "patch_tokens",
        "self_attn_maps",
        "coco class",
        "class_id",
        "segmentation m",
        "generated_text",
        "faiss",
        "k-means",
        "kmeans",
        "mmr",
    ):
        assert forbidden not in implementation
        assert forbidden not in training
