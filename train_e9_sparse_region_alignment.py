#!/usr/bin/env python3
"""Train only the E9 sparse region-alignment adapter."""

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
from src.e6_prototype_bank import sha256_file
from src.e7_training_bank import load_e7_training_bank
from src.e9_sparse_region_alignment import (
    CANONICAL_E9_TRAINING_CONFIG,
    E9_ADAPTER_CHECKPOINT_FORMAT,
    SparseRegionAlignmentAdapter,
    SparseRegionAlignmentConfig,
    compute_chunked_mil_scores,
    compute_e9_loss,
    validate_e9_training_config,
    validate_e9_checkpoint,
)
from src.e9_spatial_bank import (
    E9SpatialBank,
    SpatialUniqueImageBatchSampler,
    dataset_identity_from_rows,
    reject_train_validation_overlap,
    validate_query_spatial_join,
)


def _query_identity(
    query_bank: Mapping[str, Any], *, bank_sha256: str
) -> dict[str, Any]:
    metadata = query_bank["metadata"]
    dataset = dataset_identity_from_rows(
        split=metadata["split_name"],
        image_ids=query_bank["image_ids"].tolist(),
        annotation_ids=query_bank["annotation_ids"].tolist(),
        source_image_count=metadata["source_image_count"],
        source_annotation_count=metadata["source_annotation_count"],
    )
    is_pilot = (
        dataset["selected_image_count"] < dataset["source_image_count"]
        or dataset["selected_annotation_count"]
        < dataset["source_annotation_count"]
    )
    return {
        "format_version": metadata["format_version"],
        "split_name": metadata["split_name"],
        "bank_sha256": bank_sha256,
        "source_feature_sha256": metadata["source_feature_sha256"],
        **{key: dataset[key] for key in (
            "source_image_count", "source_annotation_count",
            "selected_image_count", "selected_annotation_count",
            "image_id_fingerprint", "annotation_id_fingerprint",
            "annotation_to_image_fingerprint",
        )},
        "e3_config_sha256": metadata["e3_config_sha256"],
        "e3_checkpoint_sha256": metadata["e3_checkpoint_sha256"],
        "routing_temperature": float(metadata["routing_temperature"]),
        "source_git_commit": metadata["source_git_commit"],
        "source_git_dirty": metadata["source_git_dirty"],
        "source_git_diff_sha256": metadata["source_git_diff_sha256"],
        "complete": bool(metadata["complete"]),
        "is_pilot": is_pilot,
    }


def _spatial_identity(bank: E9SpatialBank) -> dict[str, Any]:
    manifest = bank.manifest
    return {
        "format_version": manifest["format_version"],
        "split": manifest["split"],
        "manifest_sha256": bank.manifest_sha256,
        "source_feature_sha256": list(manifest["source_feature_sha256"]),
        "dataset_identity": dict(bank.validation["dataset_identity"]),
        "complete": bank.validation["complete"],
        "is_pilot": bank.validation["is_pilot"],
        "production_eligible": bank.validation["production_eligible"],
        "source_git_commit": manifest["source_git_commit"],
        "source_git_dirty": manifest["source_git_dirty"],
        "source_git_diff_sha256": manifest["source_git_diff_sha256"],
    }


def _require_input_identity_relationships(
    query_identities: Mapping[str, Any],
    spatial_identities: Mapping[str, Any],
    source_identities: Mapping[str, Any],
    e3_identity: Mapping[str, Any],
) -> None:
    """Fail before adapter construction if independently loaded inputs disagree."""

    for split_key in ("train", "validation"):
        query = query_identities[split_key]
        spatial = spatial_identities[split_key]
        split_name = "train" if split_key == "train" else "val"
        query_dataset = {
            "split": split_name,
            **{key: query[key] for key in (
                "source_image_count", "source_annotation_count",
                "selected_image_count", "selected_annotation_count",
                "image_id_fingerprint", "annotation_id_fingerprint",
                "annotation_to_image_fingerprint",
            )},
        }
        if query_dataset != spatial["dataset_identity"]:
            raise ValueError(f"{split_key} query/spatial identity mismatch")
        if source_identities[split_key] != {
            "query_bank_sha256": query["bank_sha256"],
            "query_source_feature_sha256": query["source_feature_sha256"],
            "spatial_manifest_sha256": spatial["manifest_sha256"],
            "spatial_source_artifact_sha256": spatial["source_feature_sha256"],
        }:
            raise ValueError(f"{split_key} repeated source identity mismatch")
        if query["split_name"] != split_name or spatial["split"] != split_name:
            raise ValueError(f"{split_key} split identity mismatch")
        if (
            query["e3_config_sha256"] != e3_identity["config_sha256"]
            or query["e3_checkpoint_sha256"] != e3_identity["checkpoint_sha256"]
        ):
            raise ValueError(f"{split_key} E3 identity mismatch")


def load_e9_training_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    value = validate_e9_training_config(value)
    if not value["save_best_model"]:
        raise ValueError("E9 training requires save_best_model=true")
    if (value["pooled_grid_height"], value["pooled_grid_width"], value["embedding_dim"]) != (16, 16, 768):
        raise ValueError("production E9 geometry must be 16x16x768")
    SparseRegionAlignmentConfig.from_mapping({
        key: value[key] for key in (
            "embedding_dim", "bottleneck_dim", "dropout", "residual_max",
            "gamma_max", "gate_hidden_dim",
        )
    })
    return value


def select_query_rows_for_spatial(
    query_bank: Mapping[str, Any],
    spatial_bank: E9SpatialBank,
    *,
    allow_subset: bool,
) -> Mapping[str, Any]:
    """Return the immutable mapped-query-only view covered by the spatial bank.

    Production retains every row and a strict all-query join.  A pilot may
    select only rows whose image is present in its explicitly pilot-marked
    spatial bank.  Caption and routed-target tensors are deliberately dropped
    after E7 contract validation; the original full bank identity remains in
    the checkpoint.
    """

    image_ids = query_bank["image_ids"]
    spatial_image_ids = spatial_bank.image_ids
    covered = torch.tensor(
        [int(image_id) in spatial_image_ids for image_id in image_ids.tolist()],
        dtype=torch.bool,
    )
    if bool(covered.all()):
        return {
            "mapped_query_embeddings": query_bank["mapped_query_embeddings"],
            "image_ids": image_ids,
            "annotation_ids": query_bank["annotation_ids"],
            "metadata": query_bank["metadata"],
        }
    if not allow_subset:
        missing = image_ids[~covered]
        raise ValueError(f"query bank contains missing spatial image ID {int(missing[0])}")
    if not bool(covered.any()):
        raise ValueError("pilot query/spatial join contains no rows")
    rows = torch.nonzero(covered, as_tuple=False).flatten()
    return {
        "mapped_query_embeddings": query_bank["mapped_query_embeddings"].index_select(0, rows),
        "image_ids": image_ids.index_select(0, rows),
        "annotation_ids": query_bank["annotation_ids"].index_select(0, rows),
        "metadata": query_bank["metadata"],
    }


def _batch(
    query_bank: Mapping[str, Any],
    spatial_bank: E9SpatialBank,
    indices: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.tensor(indices, dtype=torch.long)
    queries = F.normalize(
        query_bank["mapped_query_embeddings"][index].float().to(device).detach(),
        dim=-1,
    )
    image_ids = query_bank["image_ids"][index].clone()
    annotation_ids = query_bank["annotation_ids"][index].clone()
    if torch.unique(image_ids).numel() != len(indices):
        raise RuntimeError("E9 contrastive batch contains duplicate image IDs")
    patches, priors = zip(*(spatial_bank.get(int(image_id)) for image_id in image_ids))
    patches = F.normalize(torch.stack(patches).float().to(device).detach(), dim=-1)
    return (
        queries,
        patches,
        torch.stack(priors).to(device).detach(),
        image_ids,
        annotation_ids,
    )


def _run_epoch(
    adapter: SparseRegionAlignmentAdapter,
    query_bank: Mapping[str, Any],
    spatial_bank: E9SpatialBank,
    sampler: SpatialUniqueImageBatchSampler,
    config: Mapping[str, Any],
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    adapter.train(training)
    totals: defaultdict[str, float] = defaultdict(float)
    batches = 0
    duplicate_violations = 0
    for batch_number, indices in enumerate(sampler):
        if max_batches is not None and batch_number >= max_batches:
            break
        queries, patches, priors, image_ids, _ = _batch(
            query_bank, spatial_bank, indices, device
        )
        duplicate_violations += int(torch.unique(image_ids).numel() != len(indices))
        if training:
            optimizer.zero_grad(set_to_none=True)
        context = torch.enable_grad() if training else torch.no_grad()
        with context, torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = compute_chunked_mil_scores(
                adapter, queries, patches, priors,
                mil_top_k=config["mil_top_k"],
                mil_temperature=config["mil_temperature"],
                attention_selection_weight=config["attention_selection_weight"],
                pair_chunk_size=config["pair_chunk_size"],
                inputs_normalized=True,
            )
            losses = compute_e9_loss(
                output, queries, patches,
                mil_temperature=config["mil_temperature"],
                infonce_temperature=config["infonce_temperature"],
                anchor_weight=config["anchor_weight"],
                attention_support_weight=config["attention_support_weight"],
                gate_weight=config["gate_weight"],
            )
            if training:
                losses["loss"].backward()
                if not all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in adapter.parameters()
                ):
                    raise RuntimeError("E9 adapter gradient is non-finite")
                optimizer.step()
        diagonal = torch.arange(len(indices), device=device)
        positives = output.image_scores[diagonal, diagonal]
        negative_mask = ~torch.eye(len(indices), dtype=torch.bool, device=device)
        negatives = output.image_scores[negative_mask]
        selected_positive = output.selected_scores[diagonal, diagonal]
        selected_indices = output.selected_indices[diagonal, diagonal]
        selected_attention = output.selected_attention[diagonal, diagonal]
        responsibilities = torch.softmax(
            selected_positive / config["mil_temperature"], dim=-1
        )
        entropy = -(
            responsibilities * responsibilities.clamp_min(
                torch.finfo(responsibilities.dtype).tiny
            ).log()
        ).sum(dim=-1)
        diagnostics = {
            **losses,
            "gamma_min": output.gamma_min,
            "gamma_mean": output.gate_mean,
            "gamma_max": output.gamma_max,
            "text_residual_gate_min": output.text_residual_gate.min(),
            "text_residual_gate_mean": output.text_residual_gate.mean(),
            "text_residual_gate_max": output.text_residual_gate.max(),
            "patch_residual_gate_min": output.patch_residual_gate.min(),
            "patch_residual_gate_mean": output.patch_residual_gate.mean(),
            "patch_residual_gate_max": output.patch_residual_gate.max(),
            "positive_image_score": positives.mean(),
            "negative_image_score": negatives.mean() if negatives.numel() else positives.new_zeros(()),
            "positive_minus_negative_margin": positives.mean() - (negatives.mean() if negatives.numel() else 0),
            "positive_base_patch_score": output.positive_base_patch_score.mean(),
            "positive_adapted_patch_score": output.positive_adapted_patch_score.mean(),
            "positive_final_patch_score": output.positive_final_patch_score.mean(),
            "selected_region_attention_mass": selected_attention.sum(dim=-1).mean(),
            "selected_region_responsibility_entropy": entropy.mean(),
            "unique_selected_patch_count": torch.tensor([len(torch.unique(row)) for row in selected_indices], device=device).float().mean(),
            "top_k_index_diversity": torch.unique(selected_indices).numel() / selected_indices.numel(),
            "query_residual_angle": torch.acos(torch.einsum("bd,bd->b", output.query, output.base_query).clamp(-1, 1)).mean(),
            "patch_residual_angle": torch.acos(torch.einsum("bpd,bpd->bp", output.patches, output.base_patches).clamp(-1, 1)).mean(),
            "peak_activation_elements": float(output.peak_activation_elements),
            "retained_autograd_activation_elements": float(
                output.retained_autograd_activation_elements
            ),
        }
        for name, value in diagnostics.items():
            totals[name] += float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        batches += 1
    if batches == 0:
        raise RuntimeError("E9 epoch produced no batches")
    result = {name: value / batches for name, value in totals.items()}
    result["duplicate_image_batch_violations"] = float(duplicate_violations)
    result["nonfinite_counts"] = 0.0
    result["train_validation_image_overlap_violations"] = 0.0
    return result


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def atomic_publish_checkpoint(
    payload: Mapping[str, Any], output: Path, *, overwrite: bool,
    provenance: Mapping[str, Any], repository_root: Path,
    input_hashes: Mapping[str, str], allow_dirty_source: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        for path, digest in input_hashes.items():
            if sha256_file(path) != digest:
                raise RuntimeError(f"E9 input changed during checkpoint serialization: {path}")
        # This is deliberately the final validation operation.  Publication
        # follows immediately, with no callback, hash, or metadata work in
        # between that could mutate the source identity.
        try:
            _require_unchanged_git_provenance(
                repository_root, provenance,
                allow_dirty_source=allow_dirty_source,
            )
        except ValueError as error:
            raise RuntimeError(
                "E9 Git source provenance changed during checkpoint serialization"
            ) from error
        if overwrite:
            os.replace(temporary, output)
        else:
            try:
                os.link(temporary, output)
            except FileExistsError as error:
                raise FileExistsError(f"refusing to overwrite E9 checkpoint: {output}") from error
        _fsync_directory(output.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(output.parent)


def train_e9(
    *, config_path: Path, train_query_path: Path, validation_query_path: Path,
    train_spatial_path: Path, validation_spatial_path: Path, output_path: Path,
    final_output_path: Path | None = None,
    device: str, max_train_batches: int | None = None,
    max_validation_batches: int | None = None, allow_dirty_source: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    initial_config_sha256 = sha256_file(config_path)
    config = load_e9_training_config(config_path)
    if sha256_file(config_path) != initial_config_sha256:
        raise RuntimeError("E9 configuration changed while it was loaded")
    for name, value in (
        ("max_train_batches", max_train_batches),
        ("max_validation_batches", max_validation_batches),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    repository_root = Path(__file__).parent.resolve()
    output_path = output_path.resolve()
    final_output_path = (
        final_output_path.resolve() if final_output_path is not None else None
    )
    if final_output_path == output_path:
        raise ValueError("best and final E9 checkpoint paths must differ")
    for candidate in (output_path, final_output_path):
        if candidate is None:
            continue
        try:
            candidate.relative_to(repository_root)
        except ValueError:
            pass
        else:
            raise ValueError(
                "E9 checkpoints must be published outside the source repository"
            )
        if candidate.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite E9 checkpoint: {candidate}")
    torch.manual_seed(config["seed"])
    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    pilot = max_train_batches is not None or max_validation_batches is not None
    if not pilot and config != CANONICAL_E9_TRAINING_CONFIG:
        raise ValueError("full E9 training requires the canonical E9 configuration")
    try:
        provenance = source_git_provenance(
            repository_root, allow_dirty_source=allow_dirty_source
        )
    except ValueError as error:
        raise ValueError(
            "E9 training requires a clean Git worktree; commit E9 first, or "
            "use --allow_dirty_source only for a bounded pilot"
        ) from error
    if provenance["source_git_dirty"] and not pilot:
        raise ValueError("dirty-source E9 training is allowed only for pilots")
    train_query_sha256 = sha256_file(train_query_path)
    validation_query_sha256 = sha256_file(validation_query_path)
    train_spatial = E9SpatialBank(train_spatial_path, require_production=not pilot)
    validation_spatial = E9SpatialBank(validation_spatial_path, require_production=not pilot)
    reject_train_validation_overlap(train_spatial, validation_spatial)
    train_query = load_e7_training_bank(
        train_query_path, expected_split="train", allow_pilot=pilot
    )
    train_meta = train_query["metadata"]
    train_query = select_query_rows_for_spatial(
        train_query, train_spatial,
        allow_subset=pilot and train_spatial.manifest["is_pilot"],
    )
    validation_query = load_e7_training_bank(
        validation_query_path, expected_split="val", allow_pilot=pilot
    )
    val_meta = validation_query["metadata"]
    if (
        sha256_file(train_query_path) != train_query_sha256
        or sha256_file(validation_query_path) != validation_query_sha256
    ):
        raise RuntimeError("E9 query bank changed while it was loaded")
    validation_query = select_query_rows_for_spatial(
        validation_query, validation_spatial,
        allow_subset=pilot and validation_spatial.manifest["is_pilot"],
    )
    validate_query_spatial_join(train_query, train_spatial, expected_split="train")
    validate_query_spatial_join(validation_query, validation_spatial, expected_split="val")
    for key in ("e3_config_sha256", "e3_checkpoint_sha256"):
        if train_meta[key] != val_meta[key]:
            raise ValueError(f"train/validation {key} mismatch")
    query_identities = {
        "train": _query_identity(
            train_query, bank_sha256=train_query_sha256
        ),
        "validation": _query_identity(
            validation_query, bank_sha256=validation_query_sha256
        ),
    }
    spatial_identities = {
        "train": _spatial_identity(train_spatial),
        "validation": _spatial_identity(validation_spatial),
    }
    source_identities = {
        split: {
            "query_bank_sha256": query_identities[split]["bank_sha256"],
            "query_source_feature_sha256": query_identities[split]["source_feature_sha256"],
            "spatial_manifest_sha256": spatial_identities[split]["manifest_sha256"],
            "spatial_source_artifact_sha256": spatial_identities[split]["source_feature_sha256"],
        }
        for split in ("train", "validation")
    }
    e3_identity = {
        "config_sha256": train_meta["e3_config_sha256"],
        "checkpoint_sha256": train_meta["e3_checkpoint_sha256"],
    }
    _require_input_identity_relationships(
        query_identities, spatial_identities, source_identities, e3_identity
    )
    architecture = SparseRegionAlignmentConfig.from_mapping({
        key: config[key] for key in (
            "embedding_dim", "bottleneck_dim", "dropout", "residual_max",
            "gamma_max", "gate_hidden_dim",
        )
    })
    adapter = SparseRegionAlignmentAdapter(architecture).to(device_value)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    warmup_epochs = math.ceil(config["num_epochs"] * config["warmup_ratio"])

    def learning_rate_multiplier(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(
            1, config["num_epochs"] - warmup_epochs
        )
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, learning_rate_multiplier
    )
    train_sampler = SpatialUniqueImageBatchSampler(
        train_query["image_ids"], train_spatial,
        config["batch_size"], config["seed"],
    )
    validation_sampler = SpatialUniqueImageBatchSampler(
        validation_query["image_ids"], validation_spatial,
        config["batch_size"], config["seed"],
    )
    input_paths = {
        str(config_path): initial_config_sha256,
        str(train_query_path): train_query_sha256,
        str(validation_query_path): validation_query_sha256,
        str(train_spatial_path / "manifest.json"): train_spatial.manifest_sha256,
        str(validation_spatial_path / "manifest.json"):
            validation_spatial.manifest_sha256,
    }
    run_identity = {
        "pilot_training": pilot,
        "bounded_training": max_train_batches is not None,
        "bounded_validation": max_validation_batches is not None,
        "train_query_complete": query_identities["train"]["complete"],
        "train_query_is_pilot": query_identities["train"]["is_pilot"],
        "validation_query_complete": query_identities["validation"]["complete"],
        "validation_query_is_pilot": query_identities["validation"]["is_pilot"],
        "train_spatial_complete": spatial_identities["train"]["complete"],
        "train_spatial_is_pilot": spatial_identities["train"]["is_pilot"],
        "validation_spatial_complete": spatial_identities["validation"]["complete"],
        "validation_spatial_is_pilot": spatial_identities["validation"]["is_pilot"],
        "source_git_dirty": provenance["source_git_dirty"],
        "train_validation_overlap": False,
        "production_eligible": False,
    }
    run_identity["production_eligible"] = not any((
        run_identity["pilot_training"], run_identity["bounded_training"],
        run_identity["bounded_validation"], not run_identity["train_query_complete"],
        run_identity["train_query_is_pilot"], not run_identity["validation_query_complete"],
        run_identity["validation_query_is_pilot"], not run_identity["train_spatial_complete"],
        run_identity["train_spatial_is_pilot"], not run_identity["validation_spatial_complete"],
        run_identity["validation_spatial_is_pilot"], run_identity["source_git_dirty"],
        run_identity["train_validation_overlap"],
    ))
    best = float("inf")
    patience = 0
    history = {}
    best_published_by_this_run = False

    def checkpoint_payload(epoch_number: int) -> dict[str, Any]:
        return {
            "format_version": E9_ADAPTER_CHECKPOINT_FORMAT,
            "adapter_state_dict": {
                key: value.detach().cpu()
                for key, value in adapter.state_dict().items()
            },
            "architecture_config": dict(architecture.__dict__),
            "training_config": dict(config),
            "epoch": epoch_number,
            "best_validation_metric": best,
            "query_bank_identities": query_identities,
            "spatial_bank_identities": spatial_identities,
            "e3_identity": e3_identity,
            "source_feature_identities": source_identities,
            "run_identity": run_identity,
            "source_git_provenance": provenance,
            "diagnostic_summary": history,
        }

    completed_epoch = 0
    for epoch in range(config["num_epochs"]):
        train_sampler.set_epoch(epoch)
        validation_sampler.set_epoch(epoch)
        train_metrics = _run_epoch(
            adapter, train_query, train_spatial, train_sampler, config, device_value,
            optimizer=optimizer, max_batches=max_train_batches,
        )
        validation_metrics = _run_epoch(
            adapter, validation_query, validation_spatial, validation_sampler,
            config, device_value, optimizer=None, max_batches=max_validation_batches,
        )
        scheduler.step()
        history = {"train": train_metrics, "validation": validation_metrics}
        completed_epoch = epoch + 1
        metric = validation_metrics["loss"]
        print(json.dumps({"epoch": epoch + 1, **history}, sort_keys=True))
        if metric < best:
            best = metric
            patience = 0
            payload = checkpoint_payload(epoch + 1)
            validate_e9_checkpoint(payload, require_production=False)
            atomic_publish_checkpoint(
                payload, output_path,
                overwrite=overwrite or best_published_by_this_run,
                provenance=provenance, repository_root=repository_root,
                input_hashes=input_paths, allow_dirty_source=allow_dirty_source,
            )
            best_published_by_this_run = True
        else:
            patience += 1
            if patience >= config["early_stopping_patience"]:
                break
    if final_output_path is not None:
        final_payload = checkpoint_payload(completed_epoch)
        validate_e9_checkpoint(final_payload, require_production=False)
        atomic_publish_checkpoint(
            final_payload, final_output_path, overwrite=overwrite,
            provenance=provenance, repository_root=repository_root,
            input_hashes=input_paths, allow_dirty_source=allow_dirty_source,
        )
    return {"best_validation_metric": best, "diagnostics": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train_query_bank", required=True, type=Path)
    parser.add_argument("--validation_query_bank", required=True, type=Path)
    parser.add_argument("--train_spatial_bank", required=True, type=Path)
    parser.add_argument("--validation_spatial_bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--final_output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_train_batches", type=int)
    parser.add_argument("--max_validation_batches", type=int)
    parser.add_argument("--allow_dirty_source", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = train_e9(
        config_path=args.config, train_query_path=args.train_query_bank,
        validation_query_path=args.validation_query_bank,
        train_spatial_path=args.train_spatial_bank,
        validation_spatial_path=args.validation_spatial_bank,
        output_path=args.output, final_output_path=args.final_output,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        allow_dirty_source=args.allow_dirty_source, overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
