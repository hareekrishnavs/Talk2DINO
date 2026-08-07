"""Focused synthetic tests for the production E8 training and publication paths."""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import train_e8_balanced_retrieval_adapter as training_module
from src.e7_prototype_adapter import QueueRetrieval, UniqueImageBatchSampler
from src.e7_training_bank import E7TrainingBankValidationError
from src.e8_balanced_retrieval_adapter import (
    BalancedRetrievalPrototypeAdapter,
    BalancedRetrievalPrototypeAdapterConfig,
    corrupt_retrieval_by_rotation,
)


def _normalized(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(*shape, generator=generator), dim=-1)


def _small_adapter_config() -> BalancedRetrievalPrototypeAdapterConfig:
    return BalancedRetrievalPrototypeAdapterConfig(
        embedding_dim=12,
        num_prototypes=3,
        num_attention_heads=3,
        num_cross_attention_layers=1,
        ffn_dim=24,
        dropout=0.0,
        alpha_max=0.35,
        beta_max=0.30,
        retrieval_count=3,
        assignment_temperature=0.07,
        retrieval_weight_temperature=0.07,
        responsibility_temperature=0.10,
        separation_margin=0.50,
        alpha_advantage_scale=0.10,
        beta_advantage_scale=0.10,
    )


def _loss_config() -> dict[str, float]:
    return {
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
    }


def _bank(
    image_ids: list[int],
    *,
    seed: int,
    dimension: int = 12,
) -> dict[str, object]:
    rows = len(image_ids)
    # These leaves deliberately require gradients. The real gather path must
    # detach every frozen bank tensor before the adapter sees it.
    captions = _normalized((rows, 512), seed).requires_grad_()
    mapped = _normalized((rows, dimension), seed + 1).requires_grad_()
    targets = _normalized((rows, dimension), seed + 2).requires_grad_()
    return {
        "caption_embeddings": captions,
        "mapped_query_embeddings": mapped,
        "routed_target_embeddings": targets,
        "image_ids": torch.tensor(image_ids, dtype=torch.int64),
        "annotation_ids": torch.arange(
            seed * 100,
            seed * 100 + rows,
            dtype=torch.int64,
        ),
        "metadata": {
            "format_version": "talk2dino-e7-training-bank-v1",
            "complete": True,
            "is_pilot": False,
            "source_git_commit": "1" * 40,
            "source_git_dirty": False,
            "source_git_diff_sha256": None,
            "e3_config_sha256": "2" * 64,
            "e3_checkpoint_sha256": "3" * 64,
            "dimensions": {
                "caption_embeddings": 512,
                "mapped_query_embeddings": 768,
                "routed_target_embeddings": 768,
            },
        },
    }


class _InstrumentedQueue:
    """Tiny differentiable memory whose outputs must be detached by _run_epoch."""

    def __init__(
        self,
        events: list[str],
        *,
        dimension: int = 12,
        memory: int = 3,
        clean_violations: int = 0,
    ) -> None:
        self.events = events
        self.clean_violations = clean_violations
        self.update_calls = 0
        self.retrieve_calls = 0
        self.queries: list[torch.Tensor] = []
        self.caption_memory = _normalized((memory, 512), 71).requires_grad_()
        self.routed_memory = _normalized((memory, dimension), 72).requires_grad_()

    def retrieve(
        self,
        caption_queries: torch.Tensor,
        query_image_ids: torch.Tensor,
    ) -> QueueRetrieval:
        self.events.append("retrieve")
        self.retrieve_calls += 1
        self.queries.append(caption_queries.detach().clone())
        batch = len(caption_queries)
        memory = len(self.routed_memory)
        scores = caption_queries @ F.normalize(self.caption_memory, dim=-1).T
        vectors = self.routed_memory.unsqueeze(0).expand(batch, -1, -1)
        valid = torch.ones((batch, memory), dtype=torch.bool)

        # Row r is clean for its own query, but after the deterministic +1 row
        # rotation its first candidate equals the receiving query's image. The
        # production corruption helper must invalidate all such candidates.
        image_ids = torch.full((batch, memory), 900, dtype=torch.int64)
        image_ids[:, 0] = query_image_ids.roll(-1)
        annotation_ids = torch.arange(
            1000,
            1000 + batch * memory,
            dtype=torch.int64,
        ).reshape(batch, memory)
        return QueueRetrieval(
            routed_vectors=vectors,
            scores=scores,
            valid_mask=valid,
            image_ids=image_ids,
            annotation_ids=annotation_ids,
            exclusion_violations=torch.tensor(
                self.clean_violations,
                dtype=torch.int64,
            ),
        )

    def update(self, *unused: torch.Tensor) -> None:
        self.events.append("update")
        self.update_calls += 1


class _EventAdapter(BalancedRetrievalPrototypeAdapter):
    def __init__(
        self,
        config: BalancedRetrievalPrototypeAdapterConfig,
        events: list[str],
    ) -> None:
        super().__init__(config)
        self.events = events
        self.forward_calls = 0

    def forward(self, *args, **kwargs):
        label = "clean_forward" if self.forward_calls % 2 == 0 else "corrupt_forward"
        self.events.append(label)
        self.forward_calls += 1
        return super().forward(*args, **kwargs)


class _TrackingAdamW(torch.optim.AdamW):
    def __init__(self, parameters, **kwargs) -> None:
        super().__init__(parameters, **kwargs)
        self.zero_calls = 0
        self.step_calls = 0

    def zero_grad(self, *args, **kwargs) -> None:
        self.zero_calls += 1
        return super().zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


def _run_real_epoch(
    adapter: BalancedRetrievalPrototypeAdapter,
    bank: dict[str, object],
    sampler,
    queue: _InstrumentedQueue,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    return training_module._run_epoch(
        adapter=adapter,
        bank=bank,
        sampler=sampler,
        queue=queue,
        loss_config=_loss_config(),
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=None,
        max_batches=None,
        amp_enabled=False,
    )


def test_real_training_epoch_orders_counterfactual_work_and_keeps_inputs_frozen(
    monkeypatch,
):
    events: list[str] = []
    bank = _bank([10, 20, 30, 40], seed=1)
    sampler = UniqueImageBatchSampler(bank["image_ids"], batch_size=4, seed=9)
    adapter = _EventAdapter(_small_adapter_config(), events)
    optimizer = _TrackingAdamW(adapter.parameters(), lr=1e-3)
    queue = _InstrumentedQueue(events)
    observed_corruption = []

    def recording_corruption(*args, **kwargs):
        events.append("corrupt_retrieval")
        result = corrupt_retrieval_by_rotation(*args, **kwargs)
        observed_corruption.append(result)
        return result

    monkeypatch.setattr(
        training_module,
        "corrupt_retrieval_by_rotation",
        recording_corruption,
    )
    metrics = _run_real_epoch(adapter, bank, sampler, queue, optimizer)

    assert events == [
        "retrieve",
        "corrupt_retrieval",
        "clean_forward",
        "corrupt_forward",
        "update",
    ]
    assert optimizer.zero_calls == optimizer.step_calls == 1
    assert queue.update_calls == 1
    assert observed_corruption[0].invalidated_same_image_count == 4
    assert observed_corruption[0].exclusion_violations == 0
    assert metrics["retrieval_exclusion_violations"] == 0
    assert metrics["corrupted_retrieval_exclusion_violations"] == 0

    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_parameter_ids == {id(parameter) for parameter in adapter.parameters()}
    gradients = [parameter.grad for parameter in adapter.parameters()]
    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and torch.count_nonzero(gradient) > 0
        for gradient in gradients
    )
    for name in (
        "caption_embeddings",
        "mapped_query_embeddings",
        "routed_target_embeddings",
    ):
        assert bank[name].grad is None
    assert queue.caption_memory.grad is None
    assert queue.routed_memory.grad is None


def test_duplicate_batch_is_rejected_before_retrieval_optimizer_or_update():
    events: list[str] = []
    bank = _bank([10, 10], seed=2)
    adapter = _EventAdapter(_small_adapter_config(), events)
    optimizer = _TrackingAdamW(adapter.parameters(), lr=1e-3)
    queue = _InstrumentedQueue(events)

    with pytest.raises(RuntimeError, match="duplicate image IDs"):
        _run_real_epoch(adapter, bank, [[0, 1]], queue, optimizer)

    assert events == []
    assert optimizer.zero_calls == optimizer.step_calls == 0
    assert queue.retrieve_calls == queue.update_calls == 0


@pytest.mark.parametrize("violation_kind", ("clean", "corrupt"))
def test_exclusion_violation_fails_before_optimizer_or_queue_update(
    monkeypatch,
    violation_kind,
):
    events: list[str] = []
    bank = _bank([10, 20], seed=3)
    adapter = _EventAdapter(_small_adapter_config(), events)
    optimizer = _TrackingAdamW(adapter.parameters(), lr=1e-3)
    queue = _InstrumentedQueue(
        events,
        clean_violations=1 if violation_kind == "clean" else 0,
    )
    if violation_kind == "corrupt":

        def forced_corrupt_violation(*args, **kwargs):
            result = corrupt_retrieval_by_rotation(*args, **kwargs)
            return dataclasses.replace(
                result,
                exclusion_violations=torch.ones_like(result.exclusion_violations),
            )

        monkeypatch.setattr(
            training_module,
            "corrupt_retrieval_by_rotation",
            forced_corrupt_violation,
        )

    with pytest.raises(RuntimeError, match="held-out target image"):
        _run_real_epoch(adapter, bank, [[0, 1]], queue, optimizer)

    assert optimizer.zero_calls == optimizer.step_calls == 0
    assert queue.update_calls == 0
    assert adapter.forward_calls == 0


def test_real_validation_epoch_uses_validation_queries_without_mutating_memory():
    events: list[str] = []
    validation_bank = _bank([51, 52, 53], seed=4)
    adapter = _EventAdapter(_small_adapter_config(), events)
    queue = _InstrumentedQueue(events)
    sampler = [[0, 1, 2]]

    metrics = _run_real_epoch(adapter, validation_bank, sampler, queue, optimizer=None)

    expected_queries = F.normalize(
        validation_bank["caption_embeddings"].detach().float(),
        dim=-1,
    )
    assert torch.equal(queue.queries[0], expected_queries)
    assert queue.update_calls == 0
    assert metrics["retrieval_exclusion_violations"] == 0
    assert metrics["corrupted_retrieval_exclusion_violations"] == 0
    assert all(parameter.grad is None for parameter in adapter.parameters())
    assert validation_bank["caption_embeddings"].grad is None
    assert validation_bank["mapped_query_embeddings"].grad is None
    assert validation_bank["routed_target_embeddings"].grad is None
    assert queue.caption_memory.grad is None
    assert queue.routed_memory.grad is None


def _mask_queue_rows(queue: _InstrumentedQueue, eligible_rows: torch.Tensor) -> None:
    original_retrieve = queue.retrieve

    def retrieve(caption_queries, query_image_ids):
        result = original_retrieve(caption_queries, query_image_ids)
        valid = result.valid_mask & eligible_rows[:, None]
        return dataclasses.replace(
            result,
            routed_vectors=torch.where(
                valid[..., None],
                result.routed_vectors,
                torch.zeros_like(result.routed_vectors),
            ),
            scores=torch.where(valid, result.scores, torch.zeros_like(result.scores)),
            valid_mask=valid,
            image_ids=torch.where(
                valid,
                result.image_ids,
                torch.full_like(result.image_ids, -1),
            ),
            annotation_ids=torch.where(
                valid,
                result.annotation_ids,
                torch.full_like(result.annotation_ids, -1),
            ),
        )

    queue.retrieve = retrieve


def test_training_diagnostics_exclude_no_retrieval_rows():
    events: list[str] = []
    bank = _bank([61, 62], seed=8)
    adapter = _EventAdapter(_small_adapter_config(), events).eval()
    queue = _InstrumentedQueue(events)
    eligible = torch.tensor([True, False])
    _mask_queue_rows(queue, eligible)

    batch = training_module._gather(bank, [0, 1], torch.device("cpu"))
    retrieved = queue.retrieve(batch["caption"], batch["image_ids"])
    vectors = training_module.normalize_frozen_retrieval_candidates(
        retrieved.routed_vectors,
        retrieved.valid_mask,
        "diagnostic reference",
    )
    with torch.no_grad():
        expected_output = adapter(
            batch["mapped"],
            vectors,
            retrieved.scores.detach().float(),
            retrieved.valid_mask,
        )
    modes = F.normalize(expected_output.mode_vectors[eligible].float(), dim=-1)
    prototypes = F.normalize(
        expected_output.prototypes[eligible].float(),
        dim=-1,
    )
    mode_pairs = []
    prototype_pairs = []
    for first in range(adapter.config.num_prototypes):
        for second in range(first + 1, adapter.config.num_prototypes):
            mode_pairs.append(torch.dot(modes[0, first], modes[0, second]))
            prototype_pairs.append(
                torch.dot(prototypes[0, first], prototypes[0, second])
            )
    slot_mass = expected_output.slot_mass[eligible].flatten()
    normalized_mass = slot_mass / slot_mass.sum()
    effective_slots = torch.exp(
        -(normalized_mass * normalized_mass.clamp_min(1e-12).log()).sum()
    )

    queue = _InstrumentedQueue([])
    _mask_queue_rows(queue, eligible)
    metrics = _run_real_epoch(adapter, bank, [[0, 1]], queue, optimizer=None)
    for prefix, values in (
        ("mode_pairwise_cosine", torch.stack(mode_pairs)),
        ("prototype_pairwise_cosine", torch.stack(prototype_pairs)),
        ("slot_mass", slot_mass),
    ):
        assert metrics[f"{prefix}_min"] == pytest.approx(float(values.min()))
        assert metrics[f"{prefix}_mean"] == pytest.approx(float(values.mean()))
        assert metrics[f"{prefix}_max"] == pytest.approx(float(values.max()))
    assert metrics["effective_slot_count_mean"] == pytest.approx(
        float(effective_slots)
    )


def test_training_diagnostics_use_finite_zero_sentinel_when_all_rows_empty():
    events: list[str] = []
    bank = _bank([71, 72], seed=9)
    adapter = _EventAdapter(_small_adapter_config(), events).eval()
    queue = _InstrumentedQueue(events)
    _mask_queue_rows(queue, torch.tensor([False, False]))

    metrics = _run_real_epoch(adapter, bank, [[0, 1]], queue, optimizer=None)
    for name in (
        "mode_pairwise_cosine_min",
        "mode_pairwise_cosine_mean",
        "mode_pairwise_cosine_max",
        "prototype_pairwise_cosine_min",
        "prototype_pairwise_cosine_mean",
        "prototype_pairwise_cosine_max",
        "slot_mass_min",
        "slot_mass_mean",
        "slot_mass_max",
        "effective_slot_count_mean",
    ):
        assert metrics[name] == 0.0
        assert torch.isfinite(torch.tensor(metrics[name]))


def _training_config() -> dict[str, object]:
    adapter = asdict(_small_adapter_config())
    return {
        "seed": 42,
        "num_epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "cosine",
        "warmup_ratio": 0.05,
        "amp_dtype": "bfloat16",
        "save_best_model": True,
        "early_stopping_patience": 1,
        "adapter": adapter,
        "retrieval": {
            "queue_size": 4,
            "candidate_pool": 3,
            "retrieval_count": 3,
            "retrieval_min_similarity": -1.0,
        },
        "loss": _loss_config(),
    }


def test_train_adapter_initializes_both_memories_from_train_and_routes_val_queries(
    tmp_path,
    monkeypatch,
):
    train_bank = _bank([10, 20], seed=5)
    validation_bank = _bank([30, 40], seed=6)
    queues = []
    epoch_calls = []

    class InitializationSpyQueue:
        def __init__(self, **unused):
            self.initialized_bank = None
            queues.append(self)

        def initialize_from_bank(self, bank, seed):
            self.initialized_bank = bank

    def fake_load(unused_path, *, expected_split, **unused):
        return train_bank if expected_split == "train" else validation_bank

    def fake_epoch(*, bank, queue, optimizer, **unused):
        epoch_calls.append((bank, queue, optimizer is not None))
        return {"loss": 1.0}

    monkeypatch.setattr(training_module, "_load_config", lambda unused: _training_config())
    monkeypatch.setattr(training_module, "load_e7_training_bank", fake_load)
    monkeypatch.setattr(training_module, "_validate_e8_training_bank_pair", lambda *args: None)
    monkeypatch.setattr(
        training_module,
        "source_git_provenance",
        lambda *args, **kwargs: {
            "source_git_commit": "4" * 40,
            "source_git_dirty": False,
            "source_git_diff_sha256": None,
        },
    )
    monkeypatch.setattr(training_module, "UniqueImageRetrievalQueue", InitializationSpyQueue)
    monkeypatch.setattr(training_module, "_run_epoch", fake_epoch)
    monkeypatch.setattr(training_module, "_checkpoint", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        training_module,
        "_publish_verified_checkpoint",
        lambda *args, **kwargs: None,
    )

    training_module.train_adapter(
        config_path=tmp_path / "config.yaml",
        train_bank_path=tmp_path / "train.pth",
        validation_bank_path=tmp_path / "val.pth",
        output_path=tmp_path / "best.pth",
        device="cpu",
    )

    assert len(queues) == 2
    assert queues[0].initialized_bank is train_bank
    assert queues[1].initialized_bank is train_bank
    assert epoch_calls == [
        (train_bank, queues[0], True),
        (validation_bank, queues[1], False),
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _initialize_git_repository(path: Path) -> Path:
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


@pytest.fixture
def clean_git_provenance(tmp_path):
    repository = tmp_path / "repository"
    tracked = _initialize_git_repository(repository)
    initial = training_module.source_git_provenance(repository)
    return repository, tracked, initial


def _provenance_kwargs(clean_git_provenance):
    repository, _, initial = clean_git_provenance
    return {
        "initial_provenance": initial,
        "repository_root": repository,
        "allow_dirty_source": False,
    }


def _mutate_repository(
    dirty_kind: str,
    repository: Path,
    tracked: Path,
) -> None:
    if dirty_kind == "tracked":
        tracked.write_text("VERSION = 2\n")
    elif dirty_kind == "staged":
        tracked.write_text("VERSION = 2\n")
        subprocess.run(
            ["git", "add", "implementation.py"],
            cwd=repository,
            check=True,
        )
    elif dirty_kind == "untracked":
        (repository / "new_e8_source.py").write_text("E8 = True\n")
    else:  # pragma: no cover - guarded by the parametrization
        raise AssertionError(f"unsupported dirty kind: {dirty_kind}")


def test_atomic_save_rechecks_unchanged_provenance_after_serialization_and_publishes(
    tmp_path,
    clean_git_provenance,
):
    output = tmp_path / "checkpoint.pth"
    training_module._atomic_save(
        {"created": True},
        output,
        overwrite=False,
        **_provenance_kwargs(clean_git_provenance),
    )

    assert torch.load(output, map_location="cpu", weights_only=False) == {
        "created": True
    }
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_atomic_create_if_absent_rejects_existing_output_and_cleans_temp(
    tmp_path,
    clean_git_provenance,
):
    output = tmp_path / "checkpoint.pth"
    output.write_bytes(b"concurrently-created")
    before = _sha256(output)

    with pytest.raises(FileExistsError):
        training_module._atomic_save(
            {"replacement": True},
            output,
            overwrite=False,
            **_provenance_kwargs(clean_git_provenance),
        )

    assert _sha256(output) == before
    assert output.read_bytes() == b"concurrently-created"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_atomic_create_if_absent_preserves_target_created_at_publish_race(
    tmp_path,
    monkeypatch,
    clean_git_provenance,
):
    output = tmp_path / "checkpoint.pth"
    original_link = training_module.os.link

    def racing_link(source, destination):
        Path(destination).write_bytes(b"won-by-concurrent-writer")
        return original_link(source, destination)

    monkeypatch.setattr(training_module.os, "link", racing_link)
    with pytest.raises(FileExistsError, match="concurrent overwrite"):
        training_module._atomic_save(
            {"replacement": True},
            output,
            overwrite=False,
            **_provenance_kwargs(clean_git_provenance),
        )

    assert output.read_bytes() == b"won-by-concurrent-writer"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_atomic_explicit_overwrite_replaces_existing_output_and_cleans_temp(
    tmp_path,
    clean_git_provenance,
):
    output = tmp_path / "checkpoint.pth"
    output.write_bytes(b"old-checkpoint")
    before = _sha256(output)

    training_module._atomic_save(
        {"replacement": True},
        output,
        overwrite=True,
        **_provenance_kwargs(clean_git_provenance),
    )

    assert _sha256(output) != before
    assert torch.load(output, map_location="cpu", weights_only=False) == {
        "replacement": True
    }
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


@pytest.mark.parametrize("dirty_kind", ("tracked", "staged", "untracked"))
def test_mutation_during_serialization_refuses_publication_and_cleans_temp(
    tmp_path,
    monkeypatch,
    clean_git_provenance,
    dirty_kind,
):
    repository, tracked, _ = clean_git_provenance
    output = tmp_path / "checkpoint.pth"
    original_save = training_module.torch.save
    publication_calls = []

    def save_then_mutate(payload, path):
        original_save(payload, path)
        _mutate_repository(dirty_kind, repository, tracked)

    def record_link(*args, **kwargs):
        publication_calls.append("link")
        raise AssertionError("os.link must not run after provenance failure")

    def record_replace(*args, **kwargs):
        publication_calls.append("replace")
        raise AssertionError("os.replace must not run after provenance failure")

    monkeypatch.setattr(training_module.torch, "save", save_then_mutate)
    # Publication must never reach either function; do not call through and
    # accidentally recurse into the monkeypatched attributes.
    monkeypatch.setattr(training_module.os, "link", record_link)
    monkeypatch.setattr(training_module.os, "replace", record_replace)
    with pytest.raises(
        E7TrainingBankValidationError,
        match=(
            "Git source provenance changed during E8 checkpoint "
            "serialization/publication"
        ),
    ):
        training_module._atomic_save(
            {"checkpoint": True},
            output,
            overwrite=False,
            **_provenance_kwargs(clean_git_provenance),
        )

    assert publication_calls == []
    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_overwrite_mutation_during_serialization_preserves_existing_output(
    tmp_path,
    monkeypatch,
    clean_git_provenance,
):
    repository, tracked, _ = clean_git_provenance
    output = tmp_path / "checkpoint.pth"
    output.write_bytes(b"existing-checkpoint")
    before_bytes = output.read_bytes()
    before_sha256 = _sha256(output)
    original_save = training_module.torch.save
    replace_calls = []

    def save_then_mutate(payload, path):
        original_save(payload, path)
        _mutate_repository("tracked", repository, tracked)

    def record_replace(*args, **kwargs):
        replace_calls.append((args, kwargs))

    monkeypatch.setattr(training_module.torch, "save", save_then_mutate)
    monkeypatch.setattr(training_module.os, "replace", record_replace)
    with pytest.raises(
        E7TrainingBankValidationError,
        match=(
            "Git source provenance changed during E8 checkpoint "
            "serialization/publication"
        ),
    ):
        training_module._atomic_save(
            {"replacement": True},
            output,
            overwrite=True,
            **_provenance_kwargs(clean_git_provenance),
        )

    assert replace_calls == []
    assert output.read_bytes() == before_bytes
    assert _sha256(output) == before_sha256
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_atomic_publication_orders_final_provenance_check_before_publish(
    tmp_path,
    monkeypatch,
    clean_git_provenance,
):
    output = tmp_path / "checkpoint.pth"
    events = []
    original_save = training_module.torch.save
    original_fsync = training_module.os.fsync
    original_check = training_module._require_unchanged_git_provenance
    original_link = training_module.os.link

    def record_save(payload, path):
        result = original_save(payload, path)
        events.append("serialize")
        return result

    def record_fsync(descriptor):
        result = original_fsync(descriptor)
        events.append("fsync-temporary")
        return result

    def record_check(*args, **kwargs):
        result = original_check(*args, **kwargs)
        events.append("final-provenance-check")
        return result

    def record_link(source, destination):
        result = original_link(source, destination)
        events.append("publish")
        return result

    def record_directory_fsync(unused_directory):
        events.append("fsync-directory")

    monkeypatch.setattr(training_module.torch, "save", record_save)
    monkeypatch.setattr(training_module.os, "fsync", record_fsync)
    monkeypatch.setattr(
        training_module,
        "_require_unchanged_git_provenance",
        record_check,
    )
    monkeypatch.setattr(training_module.os, "link", record_link)
    monkeypatch.setattr(training_module, "_fsync_directory", record_directory_fsync)

    training_module._atomic_save(
        {"checkpoint": True},
        output,
        overwrite=False,
        **_provenance_kwargs(clean_git_provenance),
    )

    assert events[:5] == [
        "serialize",
        "fsync-temporary",
        "final-provenance-check",
        "publish",
        "fsync-directory",
    ]
    assert output.exists()
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_publish_wrapper_forwards_provenance_to_protected_atomic_save(
    tmp_path,
    monkeypatch,
):
    atomic_calls = []
    initial = {"source_git_dirty": False}
    output = tmp_path / "checkpoint.pth"

    monkeypatch.setattr(
        training_module,
        "_atomic_save",
        lambda *args, **kwargs: atomic_calls.append((args, kwargs)),
    )
    training_module._publish_verified_checkpoint(
        {"checkpoint": True},
        output,
        repository_root=tmp_path,
        initial_provenance=initial,
        allow_dirty_source=False,
        overwrite=True,
    )

    assert atomic_calls == [
        (
            ({"checkpoint": True}, output),
            {
                "overwrite": True,
                "initial_provenance": initial,
                "repository_root": tmp_path,
                "allow_dirty_source": False,
            },
        )
    ]


@pytest.mark.parametrize("save_best_model", (True, False))
def test_both_primary_checkpoint_routes_use_protected_atomic_save(
    tmp_path,
    monkeypatch,
    save_best_model,
):
    config = _training_config()
    config["save_best_model"] = save_best_model
    train_bank = _bank([10, 20], seed=10)
    validation_bank = _bank([30, 40], seed=11)
    provenance = {
        "source_git_commit": "4" * 40,
        "source_git_dirty": False,
        "source_git_diff_sha256": None,
    }
    atomic_calls = []

    class InitializationOnlyQueue:
        def __init__(self, **unused):
            pass

        def initialize_from_bank(self, bank, seed):
            pass

    def fake_load(unused_path, *, expected_split, **unused):
        return train_bank if expected_split == "train" else validation_bank

    def record_atomic(payload, output_path, **kwargs):
        atomic_calls.append((payload, output_path, kwargs))

    monkeypatch.setattr(training_module, "_load_config", lambda unused: config)
    monkeypatch.setattr(training_module, "load_e7_training_bank", fake_load)
    monkeypatch.setattr(
        training_module,
        "_validate_e8_training_bank_pair",
        lambda *args: None,
    )
    monkeypatch.setattr(
        training_module,
        "source_git_provenance",
        lambda *args, **kwargs: provenance,
    )
    monkeypatch.setattr(
        training_module,
        "UniqueImageRetrievalQueue",
        InitializationOnlyQueue,
    )
    monkeypatch.setattr(training_module, "_run_epoch", lambda **unused: {"loss": 1.0})
    monkeypatch.setattr(
        training_module,
        "_checkpoint",
        lambda *args, **kwargs: {"route": "protected"},
    )
    monkeypatch.setattr(training_module, "_atomic_save", record_atomic)

    output = tmp_path / "checkpoint.pth"
    training_module.train_adapter(
        config_path=tmp_path / "config.yaml",
        train_bank_path=tmp_path / "train.pth",
        validation_bank_path=tmp_path / "val.pth",
        output_path=output,
        device="cpu",
    )

    assert atomic_calls == [
        (
            {"route": "protected"},
            output,
            {
                "overwrite": False,
                "initial_provenance": provenance,
                "repository_root": Path(training_module.__file__).resolve().parent,
                "allow_dirty_source": False,
            },
        )
    ]
