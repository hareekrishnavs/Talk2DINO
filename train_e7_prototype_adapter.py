#!/usr/bin/env python3
"""Train only the E7 retrieval-conditioned prototype adapter."""

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
import yaml

from build_e7_training_bank import (
    _require_unchanged_git_provenance,
    source_git_provenance,
)
from src.e7_prototype_adapter import (
    ADAPTER_CHECKPOINT_FORMAT,
    RetrievalPrototypeAdapter,
    RetrievalPrototypeAdapterConfig,
    UniqueImageBatchSampler,
    UniqueImageRetrievalQueue,
    compute_e7_loss,
    compute_e7_scores,
    normalize_frozen_embeddings,
    normalize_frozen_retrieval_candidates,
)
from src.e7_training_bank import (
    E7BankIdentity,
    E7TrainingBankValidationError,
    FORMAT_VERSION,
    load_e7_training_bank,
)


_CONFIG_KEYS = {
    "seed",
    "num_epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "optimizer",
    "scheduler",
    "warmup_ratio",
    "amp_dtype",
    "save_best_model",
    "early_stopping_patience",
    "adapter",
    "retrieval",
    "loss",
}


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping) or set(config) != _CONFIG_KEYS:
        raise ValueError("E7 training configuration has an invalid closed schema")
    config = dict(config)
    for name in ("seed", "num_epochs", "batch_size", "early_stopping_patience"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if config["num_epochs"] == 0 or config["batch_size"] == 0:
        raise ValueError("num_epochs and batch_size must be positive")
    for name in ("learning_rate", "weight_decay", "warmup_ratio"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if not 0 <= config["warmup_ratio"] < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if config["optimizer"] != "AdamW" or config["scheduler"] != "cosine":
        raise ValueError("E7 requires AdamW with cosine scheduling")
    if config["amp_dtype"] != "bfloat16":
        raise ValueError("E7 amp_dtype must be bfloat16")
    if type(config["save_best_model"]) is not bool:
        raise ValueError("save_best_model must be boolean")
    RetrievalPrototypeAdapterConfig.from_mapping(config["adapter"])
    retrieval = config["retrieval"]
    if not isinstance(retrieval, Mapping) or set(retrieval) != {
        "queue_size", "candidate_pool", "retrieval_count", "retrieval_min_similarity"
    }:
        raise ValueError("invalid retrieval configuration")
    if retrieval["retrieval_count"] != config["adapter"]["retrieval_count"]:
        raise ValueError("adapter and queue retrieval_count must match")
    loss = config["loss"]
    if not isinstance(loss, Mapping) or set(loss) != {
        "logit_temperature",
        "prototype_temperature",
        "anchor_weight",
        "diversity_weight",
        "diversity_margin",
        "gate_weight",
    }:
        raise ValueError("invalid loss configuration")
    return config


def _gather(
    bank: Mapping[str, Any], indices: list[int], device: torch.device
) -> dict[str, torch.Tensor]:
    index = torch.tensor(indices, dtype=torch.int64)
    return {
        "caption": normalize_frozen_embeddings(
            bank["caption_embeddings"][index].to(device),
            "training caption embeddings",
        ),
        "mapped": normalize_frozen_embeddings(
            bank["mapped_query_embeddings"][index].to(device),
            "training mapped queries",
        ),
        "target": normalize_frozen_embeddings(
            bank["routed_target_embeddings"][index].to(device),
            "training routed targets",
        ),
        "image_ids": bank["image_ids"][index].to(device),
        "annotation_ids": bank["annotation_ids"][index].to(device),
    }


def _validate_e7_training_bank_pair(
    train_bank: Mapping[str, Any],
    validation_bank: Mapping[str, Any],
) -> None:
    train_metadata = train_bank["metadata"]
    validation_metadata = validation_bank["metadata"]
    if train_metadata["source_git_dirty"]:
        raise E7TrainingBankValidationError(
            "E7 adapter training requires a clean train bank"
        )
    if validation_metadata["source_git_dirty"]:
        raise E7TrainingBankValidationError(
            "E7 adapter training requires a clean validation bank"
        )
    if train_metadata["source_git_commit"] != validation_metadata["source_git_commit"]:
        raise E7TrainingBankValidationError(
            "train and validation banks were produced by different E7 "
            "implementations"
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
        or train_metadata["format_version"] != validation_metadata["format_version"]
    ):
        raise E7TrainingBankValidationError(
            "train and validation E7 bank formats are incompatible"
        )
    expected_dimensions = {
        "caption_embeddings": 512,
        "mapped_query_embeddings": 768,
        "routed_target_embeddings": 768,
    }
    if (
        train_metadata["dimensions"] != expected_dimensions
        or validation_metadata["dimensions"] != expected_dimensions
        or train_metadata["dimensions"] != validation_metadata["dimensions"]
    ):
        raise E7TrainingBankValidationError(
            "train and validation E7 bank embedding dimensions are incompatible"
        )


def _mean_metrics(totals: Mapping[str, float], batches: int) -> dict[str, float]:
    extrema = {
        "alpha_min",
        "alpha_max",
        "beta_min",
        "beta_max",
        "retrieval_count_min",
        "retrieval_count_max",
        "retrieval_exclusion_violations",
    }
    return {
        key: value if key in extrema else value / max(batches, 1)
        for key, value in totals.items()
    }


def _run_epoch(
    *,
    adapter: RetrievalPrototypeAdapter,
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
    totals: defaultdict[str, float] = defaultdict(float)
    batch_count = 0
    for batch_number, indices in enumerate(sampler):
        if max_batches is not None and batch_number >= max_batches:
            break
        batch = _gather(bank, indices, device)
        unique_images = torch.unique(batch["image_ids"])
        if len(unique_images) != len(batch["image_ids"]):
            raise RuntimeError("contrastive batch contains duplicate image IDs")
        retrieval = queue.retrieve(batch["caption"], batch["image_ids"])
        if int(retrieval.exclusion_violations.detach().cpu()) != 0:
            raise RuntimeError("retrieval contains the held-out target image")
        retrieved_vectors = normalize_frozen_retrieval_candidates(
            retrieval.routed_vectors,
            retrieval.valid_mask,
            "training retrieval candidates",
        )
        retrieval_scores = retrieval.scores.detach().to(dtype=torch.float32)
        if not torch.isfinite(retrieval_scores).all():
            raise ValueError("training retrieval scores contain non-finite values")
        retrieval_valid_mask = retrieval.valid_mask.detach()
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
                score_output = compute_e7_scores(
                    batch["mapped"],
                    batch["target"],
                    output.prototypes,
                    output.beta,
                    prototype_temperature=loss_config["prototype_temperature"],
                )
                losses = compute_e7_loss(
                    score_output,
                    output.prototypes,
                    batch["mapped"],
                    output.beta,
                    logit_temperature=loss_config["logit_temperature"],
                    anchor_weight=loss_config["anchor_weight"],
                    diversity_weight=loss_config["diversity_weight"],
                    diversity_margin=loss_config["diversity_margin"],
                    gate_weight=loss_config["gate_weight"],
                )
            if training:
                losses["loss"].backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        # The current image becomes memory only after its retrieval is complete.
        if training:
            queue.update(
                batch["caption"],
                batch["target"],
                batch["image_ids"],
                batch["annotation_ids"],
            )

        for name, value in losses.items():
            totals[name] += float(value.detach().float().cpu())
        diagnostic_values = (
            ("alpha_min", output.alpha.min()),
            ("alpha_mean", output.alpha.mean()),
            ("alpha_max", output.alpha.max()),
            ("beta_min", output.beta.min()),
            ("beta_mean", output.beta.mean()),
            ("beta_max", output.beta.max()),
            ("retrieval_count_min", output.retrieval_count.min()),
            ("retrieval_count_mean", output.retrieval_count.float().mean()),
            ("retrieval_count_max", output.retrieval_count.max()),
            ("positive_base_score", score_output.base_score.diag().mean()),
            (
                "positive_prototype_score",
                score_output.prototype_score[
                    torch.arange(len(indices), device=device),
                    torch.arange(len(indices), device=device),
                ].max(dim=-1).values.mean(),
            ),
            ("positive_final_score", score_output.final_score.diag().mean()),
        )
        for name, value in diagnostic_values:
            scalar = float(value.detach().float().cpu())
            if name.endswith("_min"):
                totals[name] = (
                    scalar if batch_count == 0 else min(totals[name], scalar)
                )
            elif name.endswith("_max"):
                totals[name] = (
                    scalar if batch_count == 0 else max(totals[name], scalar)
                )
            else:
                totals[name] += scalar
        totals["retrieval_exclusion_violations"] += float(
            retrieval.exclusion_violations.detach().cpu()
        )
        batch_count += 1
    if batch_count == 0:
        raise RuntimeError("E7 epoch produced no batches")
    return _mean_metrics(totals, batch_count)


def _atomic_save(payload: Mapping[str, Any], output_path: Path) -> None:
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
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint(
    adapter: RetrievalPrototypeAdapter,
    config: Mapping[str, Any],
    train_bank: Mapping[str, Any],
    validation_bank: Mapping[str, Any],
    provenance: Mapping[str, Any],
    epoch: int,
    metric: float,
) -> dict[str, Any]:
    return {
        "format_version": ADAPTER_CHECKPOINT_FORMAT,
        "adapter_state_dict": {
            key: value.detach().cpu() for key, value in adapter.state_dict().items()
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
        "source_git_provenance": dict(provenance),
        "epoch": epoch,
        "best_validation_metric": float(metric),
    }


def _publish_verified_checkpoint(
    payload: Mapping[str, Any],
    output_path: Path,
    *,
    repository_root: Path,
    initial_provenance: Mapping[str, Any],
    allow_dirty_source: bool,
) -> None:
    """Publish only while adapter-training source provenance is unchanged."""

    try:
        _require_unchanged_git_provenance(
            repository_root,
            initial_provenance,
            allow_dirty_source=allow_dirty_source,
        )
    except E7TrainingBankValidationError as error:
        raise E7TrainingBankValidationError(
            "Git source provenance changed during E7 adapter training"
        ) from error
    _atomic_save(payload, output_path)


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
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite adapter checkpoint: {output_path}")
    config = _load_config(config_path)
    if last_output_path is not None:
        if last_output_path.resolve() == output_path.resolve():
            raise ValueError("best and last adapter checkpoint paths must differ")
        if not config["save_best_model"]:
            raise ValueError(
                "last_output_path requires save_best_model: true so output_path "
                "remains the best checkpoint"
            )
        if last_output_path.exists() and not overwrite:
            raise FileExistsError(
                "refusing to overwrite last adapter checkpoint: "
                f"{last_output_path}"
            )
    repository_root = Path(__file__).resolve().parent
    provenance = source_git_provenance(
        repository_root,
        allow_dirty_source=allow_dirty_source,
    )
    if provenance["source_git_dirty"] and not (
        allow_pilot_banks and max_train_batches is not None
    ):
        raise E7TrainingBankValidationError(
            "dirty-source adapter training is restricted to explicit pilots"
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
    _validate_e7_training_bank_pair(train_bank, validation_bank)
    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(config["seed"])
    if device_value.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    adapter = RetrievalPrototypeAdapter(config["adapter"]).to(device_value)
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
    retrieval = config["retrieval"]
    train_queue = UniqueImageRetrievalQueue(device=device_value, **retrieval)
    train_queue.initialize_from_bank(train_bank, seed=config["seed"])
    validation_queue = UniqueImageRetrievalQueue(device=device_value, **retrieval)
    # This queue is fixed and contains training-bank rows only.
    validation_queue.initialize_from_bank(train_bank, seed=config["seed"])
    amp_enabled = device_value.type == "cuda"
    best_metric = math.inf
    best_epoch = -1
    stale_epochs = 0
    history = []
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
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        metric = validation_metrics["loss"]
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            stale_epochs = 0
            if config["save_best_model"]:
                _publish_verified_checkpoint(
                    _checkpoint(
                        adapter,
                        config,
                        train_bank,
                        validation_bank,
                        provenance,
                        epoch,
                        metric,
                    ),
                    output_path,
                    repository_root=repository_root,
                    initial_provenance=provenance,
                    allow_dirty_source=allow_dirty_source,
                )
        else:
            stale_epochs += 1
            if stale_epochs >= config["early_stopping_patience"]:
                break
    if not config["save_best_model"]:
        _publish_verified_checkpoint(
            _checkpoint(
                adapter,
                config,
                train_bank,
                validation_bank,
                provenance,
                len(history) - 1,
                history[-1]["validation"]["loss"],
            ),
            output_path,
            repository_root=repository_root,
            initial_provenance=provenance,
            allow_dirty_source=allow_dirty_source,
        )
    elif last_output_path is not None:
        _publish_verified_checkpoint(
            _checkpoint(
                adapter,
                config,
                train_bank,
                validation_bank,
                provenance,
                len(history) - 1,
                history[-1]["validation"]["loss"],
            ),
            last_output_path,
            repository_root=repository_root,
            initial_provenance=provenance,
            allow_dirty_source=allow_dirty_source,
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
