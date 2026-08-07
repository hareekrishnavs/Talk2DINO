#!/usr/bin/env python3
"""Train only the E8 balanced retrieval prototype adapter."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import yaml

from build_e7_training_bank import (
    _require_unchanged_git_provenance,
    source_git_provenance,
)
from src.e7_prototype_adapter import (
    UniqueImageBatchSampler,
    UniqueImageRetrievalQueue,
    normalize_frozen_embeddings,
    normalize_frozen_retrieval_candidates,
)
from src.e7_training_bank import (
    E7BankIdentity,
    E7TrainingBankValidationError,
    FORMAT_VERSION,
    load_e7_training_bank,
)
from src.e8_balanced_retrieval_adapter import (
    E8_ADAPTER_CHECKPOINT_FORMAT,
    BalancedRetrievalPrototypeAdapter,
    build_e8_run_identity,
    compute_e8_loss,
    compute_e8_scores,
    corrupt_retrieval_by_rotation,
    validate_e8_adapter_checkpoint,
    validate_e8_training_config,
)


_LOSS_KEYS = {
    "logit_temperature",
    "prototype_temperature",
    "anchor_weight",
    "coverage_weight",
    "balance_weight",
    "sharpness_weight",
    "separation_weight",
    "alpha_calibration_weight",
    "beta_calibration_weight",
    "corrupt_abstention_weight",
    "beta_usage_weight",
}
_LOSS_COMPONENTS = {
    "loss",
    "nce_loss",
    "anchor_loss",
    "coverage_loss",
    "balance_loss",
    "assignment_sharpness_loss",
    "mode_separation_loss",
    "alpha_calibration_loss",
    "beta_calibration_loss",
    "corrupt_abstention_loss",
    "beta_usage_loss",
}


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return validate_e8_training_config(config)


def _gather(
    bank: Mapping[str, Any], indices: list[int], device: torch.device
) -> dict[str, torch.Tensor]:
    index = torch.tensor(indices, dtype=torch.int64)
    return {
        "caption": normalize_frozen_embeddings(
            bank["caption_embeddings"][index].to(device),
            "E8 training caption embeddings",
        ),
        "mapped": normalize_frozen_embeddings(
            bank["mapped_query_embeddings"][index].to(device),
            "E8 training mapped queries",
        ),
        "target": normalize_frozen_embeddings(
            bank["routed_target_embeddings"][index].to(device),
            "E8 training routed targets",
        ),
        "image_ids": bank["image_ids"][index].detach().to(device),
        "annotation_ids": bank["annotation_ids"][index].detach().to(device),
    }


def _validate_e8_training_bank_pair(
    train_bank: Mapping[str, Any],
    validation_bank: Mapping[str, Any],
) -> None:
    train_metadata = train_bank["metadata"]
    validation_metadata = validation_bank["metadata"]
    for label, metadata in (
        ("train", train_metadata),
        ("validation", validation_metadata),
    ):
        for key in ("is_pilot", "complete"):
            if type(metadata.get(key)) is not bool:
                raise E7TrainingBankValidationError(
                    f"{label} bank metadata.{key} must be boolean"
                )
    if train_metadata["is_pilot"] != validation_metadata["is_pilot"]:
        raise E7TrainingBankValidationError(
            "train and validation banks must have identical is_pilot status"
        )
    if train_metadata["source_git_dirty"]:
        raise E7TrainingBankValidationError("E8 training requires a clean train bank")
    if validation_metadata["source_git_dirty"]:
        raise E7TrainingBankValidationError(
            "E8 training requires a clean validation bank"
        )
    if train_metadata["source_git_commit"] != validation_metadata["source_git_commit"]:
        raise E7TrainingBankValidationError(
            "train and validation banks were produced by different E7 implementations"
        )
    for key, label in (
        ("e3_config_sha256", "E3 configuration SHA256"),
        ("e3_checkpoint_sha256", "E3 checkpoint SHA256"),
    ):
        if train_metadata[key] != validation_metadata[key]:
            raise E7TrainingBankValidationError(
                f"train and validation bank {label} values differ"
            )
    if (
        train_metadata["format_version"] != FORMAT_VERSION
        or validation_metadata["format_version"] != FORMAT_VERSION
    ):
        raise E7TrainingBankValidationError("E8 requires finalized E7 bank formats")
    expected_dimensions = {
        "caption_embeddings": 512,
        "mapped_query_embeddings": 768,
        "routed_target_embeddings": 768,
    }
    if (
        train_metadata["dimensions"] != expected_dimensions
        or validation_metadata["dimensions"] != expected_dimensions
    ):
        raise E7TrainingBankValidationError(
            "E8 requires the finalized 512/768/768 E7 bank dimensions"
        )


class _TensorStats:
    """Memory-bounded population statistics for diagnostics."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value: torch.Tensor) -> None:
        flat = value.detach().float().reshape(-1)
        if flat.numel() == 0:
            return
        if not torch.isfinite(flat).all():
            raise RuntimeError("E8 diagnostics contain a non-finite value")
        self.count += flat.numel()
        self.total += float(flat.sum().cpu())
        self.total_square += float((flat * flat).sum().cpu())
        self.minimum = min(self.minimum, float(flat.min().cpu()))
        self.maximum = max(self.maximum, float(flat.max().cpu()))

    def summary(self, prefix: str, *, include_std: bool = False) -> dict[str, float]:
        if self.count == 0:
            result = {f"{prefix}_min": 0.0, f"{prefix}_mean": 0.0, f"{prefix}_max": 0.0}
            if include_std:
                result[f"{prefix}_std"] = 0.0
            return result
        mean = self.total / self.count
        result = {
            f"{prefix}_min": self.minimum,
            f"{prefix}_mean": mean,
            f"{prefix}_max": self.maximum,
        }
        if include_std:
            variance = max(self.total_square / self.count - mean * mean, 0.0)
            result[f"{prefix}_std"] = math.sqrt(variance)
        return result


def _pairwise_upper(value: torch.Tensor) -> torch.Tensor:
    count = value.shape[1]
    if count < 2:
        return value.new_empty(0)
    normalized = F.normalize(value.float(), dim=-1)
    pairwise = torch.einsum("bkd,bld->bkl", normalized, normalized)
    mask = torch.triu(
        torch.ones((count, count), dtype=torch.bool, device=value.device),
        diagonal=1,
    )
    return pairwise[:, mask]


def _effective_slot_count(slot_mass: torch.Tensor) -> torch.Tensor:
    total = slot_mass.sum(dim=-1, keepdim=True)
    normalized = torch.where(total > 0, slot_mass / total.clamp_min(1e-12), slot_mass)
    entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum(dim=-1)
    return torch.where(total.squeeze(-1) > 0, entropy.exp(), torch.zeros_like(entropy))


def _loss_kwargs(loss_config: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(loss_config[name])
        for name in _LOSS_KEYS
        if name != "prototype_temperature"
    }


def _run_epoch(
    *,
    adapter: BalancedRetrievalPrototypeAdapter,
    bank: Mapping[str, Any],
    sampler: UniqueImageBatchSampler,
    queue: UniqueImageRetrievalQueue,
    loss_config: Mapping[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    max_batches: int | None,
    amp_enabled: bool,
) -> dict[str, float]:
    training = optimizer is not None
    adapter.train(training)
    loss_totals: defaultdict[str, float] = defaultdict(float)
    stats = {
        "alpha": _TensorStats(),
        "beta": _TensorStats(),
        "corrupted_beta": _TensorStats(),
        "mode_pairwise_cosine": _TensorStats(),
        "prototype_pairwise_cosine": _TensorStats(),
        "slot_mass": _TensorStats(),
        "effective_slot_count": _TensorStats(),
        "positive_reliability": _TensorStats(),
        "negative_reliability": _TensorStats(),
        "effective_beta": _TensorStats(),
        "positive_base_score": _TensorStats(),
        "positive_grounded_score": _TensorStats(),
        "positive_final_score": _TensorStats(),
        "retrieval_count": _TensorStats(),
    }
    example_count = 0
    batch_count = 0
    retrieval_exclusion_violations = 0
    corrupted_exclusion_violations = 0

    for batch_number, indices in enumerate(sampler):
        if max_batches is not None and batch_number >= max_batches:
            break
        batch = _gather(bank, indices, device)
        if torch.unique(batch["image_ids"]).numel() != len(indices):
            raise RuntimeError("E8 contrastive batch contains duplicate image IDs")
        retrieval = queue.retrieve(batch["caption"], batch["image_ids"])
        clean_violations = int(retrieval.exclusion_violations.detach().cpu())
        if clean_violations:
            raise RuntimeError("E8 retrieval contains the held-out target image")
        retrieved_vectors = normalize_frozen_retrieval_candidates(
            retrieval.routed_vectors,
            retrieval.valid_mask,
            "E8 training retrieval candidates",
        )
        retrieval_scores = retrieval.scores.detach().to(dtype=torch.float32)
        retrieval_valid_mask = retrieval.valid_mask.detach()
        retrieval_image_ids = retrieval.image_ids.detach()
        if not torch.isfinite(retrieval_scores).all():
            raise ValueError("E8 retrieval scores contain non-finite values")

        corrupted = None
        if len(indices) >= 2:
            corrupted = corrupt_retrieval_by_rotation(
                retrieved_vectors,
                retrieval_scores,
                retrieval_valid_mask,
                retrieval_image_ids,
                batch["image_ids"],
                retrieval.annotation_ids.detach(),
            )
            corrupt_violations = int(corrupted.exclusion_violations.detach().cpu())
            if corrupt_violations:
                raise RuntimeError(
                    "E8 corrupted retrieval contains the held-out target image"
                )
            corrupted_vectors = normalize_frozen_retrieval_candidates(
                corrupted.routed_vectors,
                corrupted.valid_mask,
                "E8 corrupted retrieval candidates",
            )
            corrupted_scores = corrupted.scores.detach().to(dtype=torch.float32)
            corrupted_mask = corrupted.valid_mask.detach()
        else:
            corrupt_violations = 0
            corrupted_vectors = None
            corrupted_scores = None
            corrupted_mask = None

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                output = adapter(
                    batch["mapped"],
                    retrieved_vectors,
                    retrieval_scores,
                    retrieval_valid_mask,
                )
                corrupted_output = (
                    adapter(
                        batch["mapped"],
                        corrupted_vectors,
                        corrupted_scores,
                        corrupted_mask,
                    )
                    if corrupted is not None
                    else None
                )
                score_output = compute_e8_scores(
                    batch["mapped"],
                    batch["target"],
                    output.prototypes,
                    output.beta,
                    prototype_temperature=loss_config["prototype_temperature"],
                    responsibility_temperature=adapter.config.responsibility_temperature,
                    has_retrieval=output.retrieval_count > 0,
                )
                losses = compute_e8_loss(
                    score_output,
                    output,
                    batch["mapped"],
                    retrieved_vectors,
                    retrieval_valid_mask,
                    corrupted_beta=(
                        corrupted_output.beta if corrupted_output is not None else None
                    ),
                    corrupted_valid_rows=(
                        corrupted_mask.any(dim=-1) if corrupted_mask is not None else None
                    ),
                    separation_margin=adapter.config.separation_margin,
                    alpha_advantage_scale=adapter.config.alpha_advantage_scale,
                    beta_advantage_scale=adapter.config.beta_advantage_scale,
                    alpha_max=adapter.config.alpha_max,
                    beta_max=adapter.config.beta_max,
                    **_loss_kwargs(loss_config),
                )
            if training:
                losses["loss"].backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        # Both clean and counterfactual forwards are complete before mutation.
        if training:
            queue.update(
                batch["caption"],
                batch["target"],
                batch["image_ids"],
                batch["annotation_ids"],
            )

        if set(losses) != _LOSS_COMPONENTS:
            raise RuntimeError(
                "E8 loss returned unexpected components: "
                f"{sorted(set(losses).symmetric_difference(_LOSS_COMPONENTS))}"
            )
        batch_size = len(indices)
        for name, value in losses.items():
            scalar = float(value.detach().float().cpu())
            if not math.isfinite(scalar):
                raise RuntimeError(f"E8 {name} is non-finite")
            loss_totals[name] += scalar * batch_size

        corrupted_beta = (
            corrupted_output.beta
            if corrupted_output is not None
            else output.beta.new_zeros(output.beta.shape)
        )
        stats["alpha"].add(output.alpha)
        stats["beta"].add(output.beta)
        stats["corrupted_beta"].add(corrupted_beta)
        # Collapse/utilization diagnostics describe learned retrieval behavior,
        # so exact E3 fallback rows must not contribute repeated query vectors
        # or synthetic zero slot mass.  ``retrieval_count > 0`` is the shared
        # training/inference eligibility definition.
        diagnostic_rows = output.retrieval_count > 0
        stats["mode_pairwise_cosine"].add(
            _pairwise_upper(output.mode_vectors[diagnostic_rows])
        )
        stats["prototype_pairwise_cosine"].add(
            _pairwise_upper(output.prototypes[diagnostic_rows])
        )
        eligible_slot_mass = output.slot_mass[diagnostic_rows]
        stats["slot_mass"].add(eligible_slot_mass)
        stats["effective_slot_count"].add(
            _effective_slot_count(eligible_slot_mass)
        )
        diagonal = torch.arange(batch_size, device=device)
        stats["positive_reliability"].add(score_output.reliability[diagonal, diagonal])
        if batch_size > 1:
            off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            stats["negative_reliability"].add(score_output.reliability[off_diagonal])
        stats["effective_beta"].add(score_output.effective_beta)
        stats["positive_base_score"].add(score_output.base_score[diagonal, diagonal])
        stats["positive_grounded_score"].add(
            score_output.grounded_score[diagonal, diagonal]
        )
        stats["positive_final_score"].add(score_output.final_score[diagonal, diagonal])
        stats["retrieval_count"].add(output.retrieval_count)
        retrieval_exclusion_violations += clean_violations
        corrupted_exclusion_violations += corrupt_violations
        example_count += batch_size
        batch_count += 1

    if batch_count == 0:
        raise RuntimeError("E8 epoch produced no batches")
    metrics = {name: total / example_count for name, total in loss_totals.items()}
    for name in ("alpha", "beta", "corrupted_beta"):
        metrics.update(stats[name].summary(name, include_std=True))
    for name in (
        "mode_pairwise_cosine",
        "prototype_pairwise_cosine",
        "slot_mass",
        "retrieval_count",
    ):
        metrics.update(stats[name].summary(name))
    for name in (
        "effective_slot_count",
        "positive_reliability",
        "negative_reliability",
        "effective_beta",
        "positive_base_score",
        "positive_grounded_score",
        "positive_final_score",
    ):
        metrics[f"{name}_mean"] = stats[name].summary(name)[f"{name}_mean"]
    metrics["retrieval_exclusion_violations"] = float(
        retrieval_exclusion_violations
    )
    metrics["corrupted_retrieval_exclusion_violations"] = float(
        corrupted_exclusion_violations
    )
    return metrics


def _fsync_directory(directory: Path) -> None:
    """Best-effort durability barrier for a completed directory update."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some filesystems do not support directory fsync.  The checkpoint file
        # itself has already been fsynced, so this is a portability fallback.
        pass
    finally:
        os.close(descriptor)


def _atomic_save(
    payload: Mapping[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
    initial_provenance: Mapping[str, Any],
    repository_root: Path,
    allow_dirty_source: bool,
) -> None:
    """Publish only after a post-serialization Git-provenance check."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(payload), temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        try:
            _require_unchanged_git_provenance(
                repository_root,
                initial_provenance,
                allow_dirty_source=allow_dirty_source,
            )
        except E7TrainingBankValidationError as error:
            raise E7TrainingBankValidationError(
                "Git source provenance changed during E8 checkpoint "
                "serialization/publication"
            ) from error
        if overwrite:
            os.replace(temporary, output_path)
        else:
            try:
                # link(2) is an atomic create-if-absent operation.  Unlike
                # os.replace, it cannot clobber a target created concurrently.
                os.link(temporary, output_path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing concurrent overwrite of E8 checkpoint: {output_path}"
                ) from error
        _fsync_directory(output_path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(output_path.parent)


def _checkpoint(
    adapter: BalancedRetrievalPrototypeAdapter,
    config: Mapping[str, Any],
    train_bank: Mapping[str, Any],
    validation_bank: Mapping[str, Any],
    provenance: Mapping[str, Any],
    epoch: int,
    metric: float,
    diagnostic_summary: Mapping[str, Any],
    *,
    max_train_batches: int | None,
    max_validation_batches: int | None,
) -> dict[str, Any]:
    payload = {
        "format_version": E8_ADAPTER_CHECKPOINT_FORMAT,
        "adapter_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in adapter.state_dict().items()
        },
        "architecture_config": dict(config["adapter"]),
        "training_config": dict(config),
        "e3_identity": {
            "config_sha256": train_bank["metadata"]["e3_config_sha256"],
            "checkpoint_sha256": train_bank["metadata"]["e3_checkpoint_sha256"],
        },
        "train_bank_identity": E7BankIdentity.from_metadata(
            train_bank["metadata"]
        ).as_dict(),
        "validation_bank_identity": E7BankIdentity.from_metadata(
            validation_bank["metadata"]
        ).as_dict(),
        "run_identity": build_e8_run_identity(
            train_bank["metadata"],
            validation_bank["metadata"],
            max_train_batches=max_train_batches,
            max_validation_batches=max_validation_batches,
            source_git_dirty=provenance["source_git_dirty"],
        ),
        "source_git_provenance": dict(provenance),
        "epoch": epoch,
        "best_validation_metric": float(metric),
        "final_diagnostic_summary": dict(diagnostic_summary),
    }
    validate_e8_adapter_checkpoint(
        payload,
        allow_pilot_checkpoint=True,
        allow_dirty_source=True,
    )
    return payload


def _publish_verified_checkpoint(
    payload: Mapping[str, Any],
    output_path: Path,
    *,
    repository_root: Path,
    initial_provenance: Mapping[str, Any],
    allow_dirty_source: bool,
    overwrite: bool = False,
) -> None:
    _atomic_save(
        payload,
        output_path,
        overwrite=overwrite,
        initial_provenance=initial_provenance,
        repository_root=repository_root,
        allow_dirty_source=allow_dirty_source,
    )


def train_adapter(
    *,
    config_path: Path,
    train_bank_path: Path,
    validation_bank_path: Path,
    output_path: Path,
    last_output_path: Path | None = None,
    device: str = "cuda",
    allow_pilot_banks: bool = False,
    allow_dirty_source: bool = False,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    for name, value in (
        ("max_train_batches", max_train_batches),
        ("max_validation_batches", max_validation_batches),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite E8 checkpoint: {output_path}")
    config = _load_config(config_path)
    if last_output_path is not None:
        if last_output_path.resolve() == output_path.resolve():
            raise ValueError("best and last E8 checkpoint paths must differ")
        if not config["save_best_model"]:
            raise ValueError(
                "last_output_path requires save_best_model: true so output_path "
                "remains the best checkpoint"
            )
        if last_output_path.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite last E8 checkpoint: {last_output_path}"
            )

    repository_root = Path(__file__).resolve().parent
    provenance = source_git_provenance(
        repository_root,
        allow_dirty_source=allow_dirty_source,
    )
    if provenance["source_git_dirty"] and not (
        allow_pilot_banks
        and max_train_batches is not None
        and max_validation_batches is not None
    ):
        raise E7TrainingBankValidationError(
            "dirty-source E8 training is restricted to explicit bounded pilots"
        )
    train_bank = load_e7_training_bank(
        train_bank_path,
        allow_pilot=allow_pilot_banks,
        allow_dirty_source=allow_pilot_banks,
        expected_split="train",
    )
    validation_bank = load_e7_training_bank(
        validation_bank_path,
        allow_pilot=allow_pilot_banks,
        allow_dirty_source=allow_pilot_banks,
        expected_split="val",
    )
    _validate_e8_training_bank_pair(train_bank, validation_bank)

    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(config["seed"])
    if device_value.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    adapter = BalancedRetrievalPrototypeAdapter(config["adapter"]).to(device_value)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    train_sampler = UniqueImageBatchSampler(
        train_bank["image_ids"], config["batch_size"], config["seed"]
    )
    validation_sampler = UniqueImageBatchSampler(
        validation_bank["image_ids"], config["batch_size"], config["seed"]
    )
    batches_per_epoch = (
        min(len(train_sampler), max_train_batches)
        if max_train_batches is not None
        else len(train_sampler)
    )
    total_steps = config["num_epochs"] * batches_per_epoch
    warmup_steps = max(1, round(total_steps * config["warmup_ratio"]))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    train_queue = UniqueImageRetrievalQueue(
        device=device_value, **config["retrieval"]
    )
    train_queue.initialize_from_bank(train_bank, seed=config["seed"])
    validation_queue = UniqueImageRetrievalQueue(
        device=device_value, **config["retrieval"]
    )
    # Fixed validation retrieval memory consists only of deterministic train rows.
    validation_queue.initialize_from_bank(train_bank, seed=config["seed"])

    amp_enabled = device_value.type == "cuda"
    best_metric = math.inf
    best_epoch = -1
    best_payload: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(config["num_epochs"]):
        train_sampler.set_epoch(epoch)
        validation_sampler.set_epoch(0)
        train_metrics = _run_epoch(
            adapter=adapter,
            bank=train_bank,
            sampler=train_sampler,
            queue=train_queue,
            loss_config=config["loss"],
            device=device_value,
            optimizer=optimizer,
            scheduler=scheduler,
            max_batches=max_train_batches,
            amp_enabled=amp_enabled,
        )
        validation_metrics = _run_epoch(
            adapter=adapter,
            bank=validation_bank,
            sampler=validation_sampler,
            queue=validation_queue,
            loss_config=config["loss"],
            device=device_value,
            optimizer=None,
            scheduler=None,
            max_batches=max_validation_batches,
            amp_enabled=amp_enabled,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        metric = validation_metrics["loss"]
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            stale_epochs = 0
            if config["save_best_model"]:
                # Snapshot CPU clones now, then publish the single winning
                # checkpoint only after training.  This preserves strict
                # create-if-absent publication for non-overwrite runs.
                best_payload = _checkpoint(
                    adapter,
                    config,
                    train_bank,
                    validation_bank,
                    provenance,
                    epoch,
                    metric,
                    {"train": train_metrics, "validation": validation_metrics},
                    max_train_batches=max_train_batches,
                    max_validation_batches=max_validation_batches,
                )
        else:
            stale_epochs += 1
            if stale_epochs >= config["early_stopping_patience"]:
                break

    final_record = history[-1]
    if config["save_best_model"]:
        if best_payload is None:
            raise RuntimeError("E8 training did not produce a best checkpoint")
        _publish_verified_checkpoint(
            best_payload,
            output_path,
            repository_root=repository_root,
            initial_provenance=provenance,
            allow_dirty_source=allow_dirty_source,
            overwrite=overwrite,
        )
    else:
        _publish_verified_checkpoint(
            _checkpoint(
                adapter,
                config,
                train_bank,
                validation_bank,
                provenance,
                final_record["epoch"],
                final_record["validation"]["loss"],
                {
                    "train": final_record["train"],
                    "validation": final_record["validation"],
                },
                max_train_batches=max_train_batches,
                max_validation_batches=max_validation_batches,
            ),
            output_path,
            repository_root=repository_root,
            initial_provenance=provenance,
            allow_dirty_source=allow_dirty_source,
            overwrite=overwrite,
        )
    if config["save_best_model"] and last_output_path is not None:
        _publish_verified_checkpoint(
            _checkpoint(
                adapter,
                config,
                train_bank,
                validation_bank,
                provenance,
                final_record["epoch"],
                best_metric,
                {
                    "train": final_record["train"],
                    "validation": final_record["validation"],
                },
                max_train_batches=max_train_batches,
                max_validation_batches=max_validation_batches,
            ),
            last_output_path,
            repository_root=repository_root,
            initial_provenance=provenance,
            allow_dirty_source=allow_dirty_source,
            overwrite=overwrite,
        )
    return {
        "best_epoch": best_epoch,
        "best_validation_loss": best_metric,
        "epochs_completed": len(history),
        "checkpoint": str(output_path),
        "last_checkpoint": (
            str(last_output_path) if last_output_path is not None else None
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train_bank", required=True, type=Path)
    parser.add_argument("--validation_bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--last_output",
        type=Path,
        help="optional final-epoch checkpoint; --output remains the best checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow_pilot_banks", action="store_true")
    parser.add_argument("--allow_dirty_source", action="store_true")
    parser.add_argument("--max_train_batches", type=int)
    parser.add_argument("--max_validation_batches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = train_adapter(
        config_path=args.config,
        train_bank_path=args.train_bank,
        validation_bank_path=args.validation_bank,
        output_path=args.output,
        last_output_path=args.last_output,
        device=args.device,
        allow_pilot_banks=args.allow_pilot_banks,
        allow_dirty_source=args.allow_dirty_source,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
