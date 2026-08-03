import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

import src.e7_prototype_adapter as adapter_module
import train_e7_prototype_adapter as training_module
from src.e7_prototype_adapter import (
    ADAPTER_CHECKPOINT_FORMAT,
    LearnedRetrievalSettings,
    QueueRetrieval,
    RetrievalPrototypeAdapter,
    RetrievalPrototypeAdapterConfig,
    UniqueImageBatchSampler,
    UniqueImageRetrievalQueue,
    _exact_chunked_cosine_topk,
    compute_e7_loss,
    compute_e7_scores,
    load_e7_adapter_checkpoint,
    load_learned_retrieval_prototypes,
    validate_e7_adapter_checkpoint,
)
from src.e7_training_bank import E7TrainingBankValidationError, FORMAT_VERSION
from train_e7_prototype_adapter import (
    _gather,
    _run_epoch,
    _validate_e7_training_bank_pair,
)


def normalized(rows, dimensions, seed):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(rows, dimensions, generator=generator), dim=-1)


def queue_bank(images=12, captions_per_image=2):
    image_ids = torch.arange(images).repeat_interleave(captions_per_image)
    rows = len(image_ids)
    return {
        "caption_embeddings": normalized(rows, 512, 1).half(),
        "mapped_query_embeddings": normalized(rows, 768, 2).half(),
        "routed_target_embeddings": normalized(rows, 768, 3).half(),
        "image_ids": image_ids.to(torch.int64),
        "annotation_ids": torch.arange(100, 100 + rows, dtype=torch.int64),
    }


def pair_bank(*, split, commit="e" * 40, dirty=False):
    return {
        "metadata": {
            "format_version": FORMAT_VERSION,
            "split_name": split,
            "source_git_commit": commit,
            "source_git_dirty": dirty,
            "e3_config_sha256": "c" * 64,
            "e3_checkpoint_sha256": "d" * 64,
            "dimensions": {
                "caption_embeddings": 512,
                "mapped_query_embeddings": 768,
                "routed_target_embeddings": 768,
            },
        }
    }


def scaled_half(rows, dimensions, seed):
    scales = torch.linspace(0.6, 1.4, rows).unsqueeze(-1)
    return (normalized(rows, dimensions, seed) * scales).half()


def small_config(**overrides):
    values = {
        "embedding_dim": 16,
        "num_prototypes": 3,
        "num_attention_heads": 4,
        "num_cross_attention_layers": 2,
        "ffn_dim": 32,
        "dropout": 0.0,
        "alpha_max": 0.35,
        "beta_max": 0.30,
        "retrieval_count": 5,
    }
    values.update(overrides)
    return RetrievalPrototypeAdapterConfig(**values)


def adapter_inputs(batch=4, memory=5, dimension=16):
    query = normalized(batch, dimension, 10)
    vectors = normalized(batch * memory, dimension, 11).reshape(
        batch, memory, dimension
    )
    scores = torch.linspace(0.2, 0.9, batch * memory).reshape(batch, memory)
    valid = torch.ones(batch, memory, dtype=torch.bool)
    return query, vectors, scores, valid


def test_default_adapter_shapes_bounds_normalization_and_finiteness():
    adapter = RetrievalPrototypeAdapter().eval()
    query, vectors, scores, valid = adapter_inputs(
        batch=2, memory=4, dimension=768
    )
    output = adapter(query, vectors, scores, valid)
    assert output.prototypes.shape == (2, 3, 768)
    assert output.alpha.shape == (2, 3)
    assert output.beta.shape == (2,)
    torch.testing.assert_close(
        output.prototypes.norm(dim=-1), torch.ones(2, 3), atol=1e-5, rtol=1e-5
    )
    assert torch.isfinite(output.prototypes).all()
    assert torch.all((0 <= output.alpha) & (output.alpha <= 0.35))
    assert torch.all((0 <= output.beta) & (output.beta <= 0.30))
    assert output.beta.max() < 0.03  # conservative -3 gate initialization


def test_adapter_is_permutation_invariant_in_eval_mode():
    adapter = RetrievalPrototypeAdapter(small_config()).eval()
    query, vectors, scores, valid = adapter_inputs()
    expected = adapter(query, vectors, scores, valid)
    permutation = torch.tensor([3, 1, 4, 0, 2])
    actual = adapter(
        query,
        vectors[:, permutation],
        scores[:, permutation],
        valid[:, permutation],
    )
    torch.testing.assert_close(actual.prototypes, expected.prototypes)
    torch.testing.assert_close(actual.alpha, expected.alpha)
    torch.testing.assert_close(actual.beta, expected.beta)


def test_no_retrieval_falls_back_exactly_to_query_and_zero_beta():
    adapter = RetrievalPrototypeAdapter(small_config()).eval()
    query, vectors, scores, valid = adapter_inputs()
    valid.zero_()
    output = adapter(query, vectors, scores, valid)
    expected = F.normalize(query, dim=-1)[:, None].expand_as(output.prototypes)
    assert torch.equal(output.prototypes, expected)
    assert torch.count_nonzero(output.alpha) == 0
    assert torch.count_nonzero(output.beta) == 0
    assert torch.count_nonzero(output.retrieval_count) == 0


def test_unique_image_sampler_rotates_captions_deterministically():
    image_ids = torch.arange(37).repeat_interleave(5)
    first = UniqueImageBatchSampler(image_ids, batch_size=8, seed=42)
    replay = UniqueImageBatchSampler(image_ids, batch_size=8, seed=42)
    first_batches = list(first)
    assert first_batches == list(replay)
    covered_images = []
    for batch in first_batches:
        batch_images = image_ids[batch].tolist()
        assert len(batch_images) == len(set(batch_images))
        covered_images.extend(batch_images)
    assert sorted(covered_images) == list(range(37))
    first.set_epoch(1)
    second_indices = [row for batch in first for row in batch]
    first_indices = [row for batch in first_batches for row in batch]
    assert first_indices != second_indices
    assert all(image_ids[a] == image_ids[b] for a, b in zip(
        sorted(first_indices, key=lambda row: int(image_ids[row])),
        sorted(second_indices, key=lambda row: int(image_ids[row])),
    ))


def test_queue_has_one_entry_per_image_and_replaces_updates():
    bank = queue_bank(images=9, captions_per_image=3)
    queue = UniqueImageRetrievalQueue(
        queue_size=7,
        candidate_pool=7,
        retrieval_count=4,
        retrieval_min_similarity=-1,
    )
    queue.initialize_from_bank(bank, seed=9)
    assert queue.active_count == 7
    assert len(set(queue.image_ids.tolist())) == 7
    image = queue.image_ids[:1].clone()
    queue.update(
        normalized(1, 512, 40),
        normalized(1, 768, 41),
        image,
        torch.tensor([999], dtype=torch.int64),
    )
    assert queue.active_count == 7
    assert len(set(queue.image_ids.tolist())) == 7
    assert 999 in queue.annotation_ids.tolist()


def test_gather_restores_unit_float32_geometry_for_train_and_validation():
    bank = queue_bank(images=6, captions_per_image=1)
    bank["caption_embeddings"] = scaled_half(6, 512, 70)
    bank["mapped_query_embeddings"] = scaled_half(6, 768, 71)
    bank["routed_target_embeddings"] = scaled_half(6, 768, 72)
    training = _gather(bank, [0, 2, 4], torch.device("cpu"))
    validation = _gather(bank, [0, 2, 4], torch.device("cpu"))
    for key in ("caption", "mapped", "target"):
        assert training[key].dtype == torch.float32
        assert not training[key].requires_grad
        torch.testing.assert_close(
            training[key].norm(dim=-1),
            torch.ones(3),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(validation[key], training[key])


def test_queue_caches_unit_float32_embeddings_after_initialize_and_update():
    bank = queue_bank(images=5, captions_per_image=1)
    bank["caption_embeddings"] = scaled_half(5, 512, 73).requires_grad_()
    bank["routed_target_embeddings"] = scaled_half(
        5, 768, 74
    ).requires_grad_()
    queue = UniqueImageRetrievalQueue(
        queue_size=5,
        candidate_pool=5,
        retrieval_count=3,
        retrieval_min_similarity=-1,
    )
    queue.initialize_from_bank(bank)
    for value in (queue.caption_embeddings, queue.routed_embeddings):
        assert value.dtype == torch.float32
        assert not value.requires_grad
        torch.testing.assert_close(
            value.norm(dim=-1),
            torch.ones(len(value)),
            atol=1e-6,
            rtol=1e-6,
        )
    new_captions = (normalized(2, 512, 75) * 1.7).requires_grad_()
    new_routed = (normalized(2, 768, 76) * 0.4).requires_grad_()
    queue.update(
        new_captions,
        new_routed,
        torch.tensor([100, 101], dtype=torch.int64),
        torch.tensor([1000, 1001], dtype=torch.int64),
    )
    torch.testing.assert_close(
        queue.caption_embeddings.norm(dim=-1),
        torch.ones(len(queue.caption_embeddings)),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        queue.routed_embeddings.norm(dim=-1),
        torch.ones(len(queue.routed_embeddings)),
        atol=1e-6,
        rtol=1e-6,
    )


def test_chunked_e7_retrieval_uses_explicit_cosine_geometry():
    query = normalized(3, 11, 77) * torch.tensor([[0.7], [1.3], [2.0]])
    bank = scaled_half(17, 11, 78)
    scores, indices = _exact_chunked_cosine_topk(query, bank, 6, 5)
    reference = F.normalize(query.float(), dim=-1) @ F.normalize(
        bank.float(), dim=-1
    ).T
    expected_scores, expected_indices = torch.topk(reference, 6, dim=-1)
    torch.testing.assert_close(scores, expected_scores)
    assert torch.equal(indices, expected_indices)


def test_same_image_and_target_exclusion_are_exact():
    bank = queue_bank(images=10, captions_per_image=1)
    queue = UniqueImageRetrievalQueue(
        queue_size=10,
        candidate_pool=10,
        retrieval_count=6,
        retrieval_min_similarity=-1,
    )
    queue.initialize_from_bank(bank, seed=0)
    query_images = queue.image_ids[:4].clone()
    query_captions = queue.caption_embeddings[:4].float().clone()
    result = queue.retrieve(query_captions, query_images)
    assert result.exclusion_violations.item() == 0
    assert not torch.any(
        result.valid_mask & (result.image_ids == query_images[:, None])
    )


def test_validation_retrieval_uses_fixed_training_memory_only():
    train = queue_bank(images=8, captions_per_image=2)
    validation = queue_bank(images=3, captions_per_image=1)
    validation["image_ids"] += 1000
    validation["annotation_ids"] += 1000
    queue = UniqueImageRetrievalQueue(
        queue_size=8,
        candidate_pool=8,
        retrieval_count=4,
        retrieval_min_similarity=-1,
    )
    queue.initialize_from_bank(train, seed=42)
    memory_before = queue.annotation_ids.clone()
    result = queue.retrieve(
        validation["caption_embeddings"].float(), validation["image_ids"]
    )
    assert set(result.annotation_ids[result.valid_mask].tolist()).issubset(
        set(train["annotation_ids"].tolist())
    )
    assert not set(result.annotation_ids[result.valid_mask].tolist()).intersection(
        set(validation["annotation_ids"].tolist())
    )
    assert torch.equal(queue.annotation_ids, memory_before)


def test_vectorized_scores_match_independent_loop():
    query = normalized(4, 9, 20)
    target = normalized(4, 9, 21)
    prototypes = normalized(4 * 3, 9, 22).reshape(4, 3, 9)
    beta = torch.tensor([0.0, 0.1, 0.2, 0.3])
    actual = compute_e7_scores(query, target, prototypes, beta, 0.10)
    expected_base = torch.empty(4, 4)
    expected_modes = torch.empty(4, 4, 3)
    expected_grounded = torch.empty(4, 4)
    expected_final = torch.empty(4, 4)
    for i in range(4):
        for j in range(4):
            expected_base[i, j] = torch.dot(query[i], target[j])
            for k in range(3):
                expected_modes[i, j, k] = torch.dot(prototypes[i, k], target[j])
            expected_grounded[i, j] = 0.10 * (
                torch.logsumexp(expected_modes[i, j] / 0.10, dim=0)
                - math.log(3)
            )
            expected_final[i, j] = (
                (1 - beta[i]) * expected_base[i, j]
                + beta[i] * expected_grounded[i, j]
            )
    torch.testing.assert_close(actual.base_score, expected_base)
    torch.testing.assert_close(actual.prototype_score, expected_modes)
    torch.testing.assert_close(actual.grounded_score, expected_grounded)
    torch.testing.assert_close(actual.final_score, expected_final)


def test_scores_restore_cosine_geometry_for_base_and_prototypes():
    query = normalized(4, 9, 79) * torch.tensor([[0.5], [0.8], [1.4], [2.0]])
    target = normalized(4, 9, 80) * torch.tensor([[1.8], [0.6], [1.1], [2.3]])
    prototypes = normalized(12, 9, 81).reshape(4, 3, 9)
    prototypes = prototypes * torch.tensor([0.4, 1.2, 2.1])[None, :, None]
    actual = compute_e7_scores(query, target, prototypes, torch.zeros(4))
    query_reference = F.normalize(query.float(), dim=-1)
    target_reference = F.normalize(target.float(), dim=-1)
    prototype_reference = F.normalize(prototypes.float(), dim=-1)
    expected_base = query_reference @ target_reference.T
    expected_prototypes = torch.einsum(
        "ikd,jd->ijk", prototype_reference, target_reference
    )
    torch.testing.assert_close(actual.base_score, expected_base)
    torch.testing.assert_close(actual.prototype_score, expected_prototypes)


def test_symmetric_infonce_and_auxiliary_losses_match_reference():
    query = normalized(3, 8, 30)
    target = normalized(3, 8, 31)
    prototypes = normalized(9, 8, 32).reshape(3, 3, 8)
    beta = torch.tensor([0.05, 0.10, 0.15])
    scores = compute_e7_scores(query, target, prototypes, beta)
    actual = compute_e7_loss(scores, prototypes, query, beta)
    labels = torch.arange(3)
    nce = 0.5 * (
        F.cross_entropy(scores.final_score / 0.07, labels)
        + F.cross_entropy(scores.final_score.T / 0.07, labels)
    )
    anchor = (1 - F.cosine_similarity(prototypes, query[:, None], dim=-1)).mean()
    pairwise = prototypes @ prototypes.transpose(1, 2)
    pair_mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    diversity = F.relu(pairwise[:, pair_mask] - 0.80).mean()
    total = nce + 0.10 * anchor + 0.05 * diversity + 0.001 * beta.mean()
    torch.testing.assert_close(actual["nce_loss"], nce)
    torch.testing.assert_close(actual["anchor_loss"], anchor)
    torch.testing.assert_close(actual["diversity_loss"], diversity)
    torch.testing.assert_close(actual["loss"], total)


def test_production_training_step_detaches_all_frozen_inputs(monkeypatch):
    train_bank = queue_bank(images=4, captions_per_image=1)
    candidate_bank = queue_bank(images=6, captions_per_image=1)
    for bank in (train_bank, candidate_bank):
        bank["caption_embeddings"] = scaled_half(
            len(bank["image_ids"]), 512, 82
        ).requires_grad_()
        bank["mapped_query_embeddings"] = scaled_half(
            len(bank["image_ids"]), 768, 83
        ).requires_grad_()
        bank["routed_target_embeddings"] = scaled_half(
            len(bank["image_ids"]), 768, 84
        ).requires_grad_()

    adapter = RetrievalPrototypeAdapter(
        RetrievalPrototypeAdapterConfig(
            embedding_dim=768,
            num_prototypes=2,
            num_attention_heads=8,
            num_cross_attention_layers=1,
            ffn_dim=32,
            dropout=0.0,
            retrieval_count=2,
        )
    )
    queue = UniqueImageRetrievalQueue(
        queue_size=6,
        candidate_pool=6,
        retrieval_count=2,
        retrieval_min_similarity=-1,
    )
    queue.initialize_from_bank(candidate_bank)
    sampler = UniqueImageBatchSampler(train_bank["image_ids"], batch_size=4)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
    captured = {}
    original_gather = training_module._gather

    def capture_gather(*args, **kwargs):
        result = original_gather(*args, **kwargs)
        captured["batch"] = result
        return result

    monkeypatch.setattr(training_module, "_gather", capture_gather)
    original_retrieve = queue.retrieve

    def retrieval_with_gradient_leaves(*args, **kwargs):
        result = original_retrieve(*args, **kwargs)
        candidate_vectors = result.routed_vectors.detach().requires_grad_()
        retrieval_scores = result.scores.detach().requires_grad_()
        captured["candidate_vectors"] = candidate_vectors
        captured["retrieval_scores"] = retrieval_scores
        return QueueRetrieval(
            routed_vectors=candidate_vectors,
            scores=retrieval_scores,
            valid_mask=result.valid_mask,
            image_ids=result.image_ids,
            annotation_ids=result.annotation_ids,
            exclusion_violations=result.exclusion_violations,
        )

    queue.retrieve = retrieval_with_gradient_leaves
    metrics = _run_epoch(
        adapter=adapter,
        bank=train_bank,
        sampler=sampler,
        queue=queue,
        loss_config={
            "logit_temperature": 0.07,
            "prototype_temperature": 0.10,
            "anchor_weight": 0.10,
            "diversity_weight": 0.05,
            "diversity_margin": 0.80,
            "gate_weight": 0.001,
        },
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=None,
        max_batches=1,
        amp_enabled=False,
    )
    assert math.isfinite(metrics["loss"])
    gradients = [
        parameter.grad
        for parameter in adapter.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    for key in ("caption", "mapped", "target"):
        assert not captured["batch"][key].requires_grad
        assert captured["batch"][key].grad is None
    assert captured["candidate_vectors"].grad is None
    assert captured["retrieval_scores"].grad is None
    for bank in (train_bank, candidate_bank):
        for key in (
            "caption_embeddings",
            "mapped_query_embeddings",
            "routed_target_embeddings",
        ):
            assert bank[key].grad is None
    for value in (queue.caption_embeddings, queue.routed_embeddings):
        assert not value.requires_grad
        assert value.grad is None


def test_beta_zero_final_scores_are_bitwise_e3_scores():
    query = normalized(4, 8, 60)
    target = normalized(4, 8, 61)
    prototypes = normalized(12, 8, 62).reshape(4, 3, 8)
    scores = compute_e7_scores(
        query, target, prototypes, torch.zeros(4)
    )
    assert torch.equal(scores.final_score, scores.base_score)


def test_matching_clean_training_bank_pair_is_accepted():
    _validate_e7_training_bank_pair(
        pair_bank(split="train"),
        pair_bank(split="val"),
    )


def test_training_bank_pair_rejects_different_source_commits():
    with pytest.raises(
        E7TrainingBankValidationError,
        match="different E7 implementations",
    ):
        _validate_e7_training_bank_pair(
            pair_bank(split="train", commit="a" * 40),
            pair_bank(split="val", commit="b" * 40),
        )


@pytest.mark.parametrize("dirty_split", ("train", "val"))
def test_training_bank_pair_rejects_dirty_bank(dirty_split):
    train = pair_bank(split="train", dirty=dirty_split == "train")
    validation = pair_bank(split="val", dirty=dirty_split == "val")
    bank_name = "validation" if dirty_split == "val" else "train"
    with pytest.raises(
        E7TrainingBankValidationError,
        match=f"clean {bank_name} bank",
    ):
        _validate_e7_training_bank_pair(train, validation)


def test_bank_pair_rejection_precedes_adapter_and_training_setup(
    tmp_path,
    monkeypatch,
):
    train = pair_bank(split="train", commit="a" * 40)
    validation = pair_bank(split="val", commit="b" * 40)
    monkeypatch.setattr(training_module, "_load_config", lambda unused: {})
    monkeypatch.setattr(
        training_module,
        "source_git_provenance",
        lambda *args, **kwargs: {
            "source_git_commit": "c" * 40,
            "source_git_dirty": False,
            "source_git_diff_sha256": None,
        },
    )
    monkeypatch.setattr(
        training_module,
        "load_e7_training_bank",
        lambda path, **kwargs: train
        if kwargs["expected_split"] == "train"
        else validation,
    )
    constructed = []

    def unexpected_adapter_construction(*args, **kwargs):
        constructed.append(True)
        raise AssertionError("adapter construction must not be reached")

    monkeypatch.setattr(
        training_module,
        "RetrievalPrototypeAdapter",
        unexpected_adapter_construction,
    )
    with pytest.raises(
        E7TrainingBankValidationError,
        match="different E7 implementations",
    ):
        training_module.train_adapter(
            config_path=tmp_path / "config.yaml",
            train_bank_path=tmp_path / "train.pth",
            validation_bank_path=tmp_path / "val.pth",
            output_path=tmp_path / "adapter.pth",
            device="cpu",
        )
    assert constructed == []


def checkpoint_payload(adapter, architecture):
    train_identity = {
        "format_version": "talk2dino-e7-training-bank-v1",
        "split_name": "train",
        "source_feature_sha256": "a" * 64,
        "annotation_id_fingerprint": "b" * 64,
        "selected_annotation_count": 10,
        "e3_config_sha256": "c" * 64,
        "e3_checkpoint_sha256": "d" * 64,
        "routing_temperature": 0.10,
        "source_git_commit": "e" * 40,
    }
    return {
        "format_version": ADAPTER_CHECKPOINT_FORMAT,
        "adapter_state_dict": adapter.state_dict(),
        "architecture_config": asdict(architecture),
        "training_config": {"seed": 42},
        "e3_identity": {
            "config_sha256": "c" * 64,
            "checkpoint_sha256": "d" * 64,
        },
        "train_bank_identity": train_identity,
        "validation_bank_identity": {**train_identity, "split_name": "val"},
        "source_git_provenance": {
            "source_git_commit": "e" * 40,
            "source_git_dirty": False,
            "source_git_diff_sha256": None,
        },
        "epoch": 2,
        "best_validation_metric": 1.25,
    }


def test_production_loader_rejects_non_768_checkpoint_before_construction(
    tmp_path,
    monkeypatch,
):
    architecture = small_config()
    adapter = RetrievalPrototypeAdapter(architecture)
    checkpoint = checkpoint_payload(adapter, architecture)
    checkpoint_path = tmp_path / "adapter.pth"
    torch.save(checkpoint, checkpoint_path)
    identity = checkpoint["train_bank_identity"]
    bank = queue_bank(images=10, captions_per_image=1)
    bank["metadata"] = {
        **identity,
        "dimensions": {
            "caption_embeddings": 512,
            "mapped_query_embeddings": 768,
            "routed_target_embeddings": 768,
        },
    }
    monkeypatch.setattr(
        adapter_module,
        "load_e7_training_bank",
        lambda *args, **kwargs: bank,
    )
    constructed = []

    def unexpected_adapter_construction(*args, **kwargs):
        constructed.append(True)
        raise AssertionError("adapter construction must not be reached")

    monkeypatch.setattr(
        adapter_module,
        "RetrievalPrototypeAdapter",
        unexpected_adapter_construction,
    )
    with pytest.raises(ValueError, match="embedding dimension is incompatible"):
        load_learned_retrieval_prototypes(
            tmp_path / "bank.pth",
            checkpoint_path,
            LearnedRetrievalSettings(prototype_retrieval_count=5),
            e3_config_sha256="c" * 64,
            e3_checkpoint_sha256="d" * 64,
            device="cpu",
        )
    assert constructed == []


def test_checkpoint_round_trip_and_compatibility_rejection(tmp_path):
    architecture = small_config()
    adapter = RetrievalPrototypeAdapter(architecture).eval()
    payload = checkpoint_payload(adapter, architecture)
    validate_e7_adapter_checkpoint(payload)
    path = tmp_path / "adapter.pth"
    torch.save(payload, path)
    loaded, checkpoint = load_e7_adapter_checkpoint(
        path,
        expected_e3_identity=payload["e3_identity"],
        expected_train_bank_identity=payload["train_bank_identity"],
    )
    query, vectors, scores, valid = adapter_inputs()
    torch.testing.assert_close(
        loaded(query, vectors, scores, valid).prototypes,
        adapter(query, vectors, scores, valid).prototypes,
    )
    with pytest.raises(ValueError, match="E3 identity"):
        load_e7_adapter_checkpoint(
            path,
            expected_e3_identity={
                "config_sha256": "f" * 64,
                "checkpoint_sha256": "d" * 64,
            },
        )
    unknown = dict(checkpoint)
    unknown["bank_tensor"] = torch.randn(2)
    with pytest.raises(ValueError, match="closed schema"):
        validate_e7_adapter_checkpoint(unknown)
    dirty = checkpoint_payload(adapter, architecture)
    dirty["source_git_provenance"] = {
        "source_git_commit": "e" * 40,
        "source_git_dirty": True,
        "source_git_diff_sha256": "f" * 64,
    }
    with pytest.raises(ValueError, match="not evaluable"):
        validate_e7_adapter_checkpoint(dirty)
    validate_e7_adapter_checkpoint(dirty, allow_dirty_source=True)
    assert not any(
        token in key.lower()
        for key in payload["adapter_state_dict"]
        for token in ("bank", "queue", "clip", "dino", "projection")
    )


def test_e7_configuration_is_isolated_and_e6_unchanged():
    root = Path("src/open_vocabulary_segmentation/configs/stuff")
    e3_name = "dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_tau010.yml"
    e6 = yaml.safe_load(
        (root / "dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_rgtp.yml").read_text()
    )
    e7_path = root / "dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_learned_rpa.yml"
    e7 = yaml.safe_load(e7_path.read_text())
    assert e7["_base_"] == e3_name
    assert set(e7["model"]) == {"learned_retrieval_prototypes"}
    settings = dict(e7["model"]["learned_retrieval_prototypes"])
    assert settings.pop("bank_path") == "${oc.env:TALK2DINO_PROTOTYPE_BANK}"
    assert settings.pop("adapter_path") == "${oc.env:TALK2DINO_E7_ADAPTER}"
    assert LearnedRetrievalSettings.from_mapping(settings) == LearnedRetrievalSettings()
    assert set(e6["model"]) == {"retrieval_grounded_prototypes"}
    serialized = e7_path.read_text()
    for forbidden in (
        "all_pairs_max",
        "all_pairs_lse",
        "multi_positive",
        "dense_consistency",
        "patch_tokens",
        "mmr_lambda",
        "kmeans",
    ):
        assert forbidden not in serialized
