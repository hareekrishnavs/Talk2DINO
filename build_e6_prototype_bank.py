#!/usr/bin/env python3
"""Build one compact E6 retrieval-grounded prototype bank."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import subprocess
import tempfile
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from src.e6_prototype_bank import (
    CAPTION_EMBED_DIM,
    EXPECTED_ATTENTION_HEADS,
    EXPECTED_E3_CHECKPOINT_NAME,
    EXPECTED_E3_CONFIG_NAME,
    EXPECTED_ROUTING_TEMPERATURE,
    FORMAT_VERSION,
    ROUTED_DINO_EMBED_DIM,
    PrototypeBankValidationError,
    annotation_id_set_fingerprint,
    route_annotation_batch,
    sha256_file,
    validate_prototype_bank,
)
from src.model import ProjectionLayer


def _integer_id(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise PrototypeBankValidationError(
            f"{field_name} must be an integer ID, got {value!r}"
        )
    result = int(value)
    if not -(2**63) <= result < 2**63:
        raise PrototypeBankValidationError(
            f"{field_name} is not a valid int64 ID: {value!r}"
        )
    return result


def _require_tensor(
    record: Mapping[str, Any],
    field_name: str,
    expected_shape: tuple[int, ...],
    record_description: str,
) -> torch.Tensor:
    if field_name not in record:
        raise PrototypeBankValidationError(
            f"{record_description} is missing {field_name}"
        )
    value = record[field_name]
    if not torch.is_tensor(value):
        raise PrototypeBankValidationError(
            f"{record_description}.{field_name} must be a tensor"
        )
    if tuple(value.shape) != expected_shape:
        raise PrototypeBankValidationError(
            f"{record_description}.{field_name} must have shape "
            f"{list(expected_shape)}, got {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise PrototypeBankValidationError(
            f"{record_description}.{field_name} must be floating point"
        )
    if not torch.isfinite(value).all():
        raise PrototypeBankValidationError(
            f"{record_description}.{field_name} contains non-finite values"
        )
    return value


def _run_git(
    repository_root: Path,
    arguments: list[str],
    *,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise RuntimeError(
            f"could not inspect Git provenance in {repository_root}"
        ) from error
    if result.returncode not in allowed_returncodes:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Git provenance command failed in {repository_root}: "
            f"git {' '.join(arguments)}: {stderr}"
        )
    return result


def _complete_git_diff_sha256(
    repository_root: Path,
    untracked_paths: list[str],
) -> str:
    """Hash tracked/staged changes plus deterministic patches for untracked files."""

    digest = hashlib.sha256()
    tracked_diff = _run_git(
        repository_root,
        ["diff", "--binary", "HEAD", "--"],
    ).stdout
    digest.update(b"tracked-and-staged\0")
    digest.update(tracked_diff)
    for relative_path in sorted(untracked_paths):
        untracked_diff = _run_git(
            repository_root,
            [
                "diff",
                "--binary",
                "--no-index",
                "--",
                "/dev/null",
                relative_path,
            ],
            allowed_returncodes=(0, 1),
        ).stdout
        digest.update(b"untracked\0")
        digest.update(relative_path.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(untracked_diff)
    return digest.hexdigest()


def source_git_provenance(
    repository_root: Path,
    *,
    allow_dirty_source: bool = False,
) -> dict[str, Any]:
    """Return reproducible source provenance, rejecting dirty trees by default."""

    repository_root = Path(repository_root).resolve()
    commit = _run_git(repository_root, ["rev-parse", "HEAD"]).stdout.decode(
        "ascii"
    ).strip()
    tracked_or_staged_dirty = (
        _run_git(
            repository_root,
            ["diff", "--quiet", "HEAD", "--"],
            allowed_returncodes=(0, 1),
        ).returncode
        == 1
    )
    untracked_output = _run_git(
        repository_root,
        ["ls-files", "--others", "--exclude-standard", "-z"],
    ).stdout
    untracked_paths = [
        path.decode("utf-8", errors="surrogateescape")
        for path in untracked_output.split(b"\0")
        if path
    ]
    dirty = tracked_or_staged_dirty or bool(untracked_paths)
    if dirty and not allow_dirty_source:
        status = _run_git(
            repository_root,
            ["status", "--short", "--untracked-files=all"],
        ).stdout.decode("utf-8", errors="replace").strip()
        status_preview = "\n".join(status.splitlines()[:20])
        raise PrototypeBankValidationError(
            "E6 bank construction requires a clean Git worktree. Commit the "
            "E6 implementation first. For development pilots only, pass "
            "--allow_dirty_source together with --max_annotations. "
            f"Dirty paths:\n{status_preview}"
        )
    return {
        "source_git_commit": commit,
        "source_git_dirty": dirty,
        "source_git_diff_sha256": (
            _complete_git_diff_sha256(repository_root, untracked_paths)
            if dirty
            else None
        ),
    }


def _repository_root() -> Path:
    return Path(__file__).resolve().parent


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise PrototypeBankValidationError(
            "E3 checkpoint must contain a projection state_dict"
        )
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            checkpoint = candidate
            break
    if not isinstance(checkpoint, Mapping) or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in checkpoint.items()
    ):
        raise PrototypeBankValidationError(
            "E3 checkpoint does not contain a valid projection state_dict"
        )
    return dict(checkpoint)


def _load_e3_projection(
    model_config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ProjectionLayer, float]:
    if model_config_path.name != EXPECTED_E3_CONFIG_NAME:
        raise PrototypeBankValidationError(
            "E6 bank construction requires the E3 configuration "
            f"{EXPECTED_E3_CONFIG_NAME}, got {model_config_path.name}"
        )
    if checkpoint_path.name != EXPECTED_E3_CHECKPOINT_NAME:
        raise PrototypeBankValidationError(
            "E6 bank construction requires the E3 checkpoint "
            f"{EXPECTED_E3_CHECKPOINT_NAME}, got {checkpoint_path.name}"
        )
    with open(model_config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping) or not isinstance(config.get("model"), Mapping):
        raise PrototypeBankValidationError(
            f"{model_config_path} does not contain a model mapping"
        )
    model_config = dict(config["model"])
    if model_config.get("alignment_strategy") != "paired_soft_routing":
        raise PrototypeBankValidationError(
            "E6 bank construction requires the E3 paired_soft_routing strategy"
        )
    try:
        routing_temperature = float(model_config["routing_temperature"])
    except (KeyError, TypeError, ValueError) as error:
        raise PrototypeBankValidationError(
            "E3 configuration must define a numeric routing_temperature"
        ) from error
    if not math.isfinite(routing_temperature) or routing_temperature <= 0:
        raise PrototypeBankValidationError(
            "E3 routing_temperature must be finite and strictly positive"
        )
    if not math.isclose(
        routing_temperature,
        EXPECTED_ROUTING_TEMPERATURE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PrototypeBankValidationError(
            "E6 requires the E3 routing_temperature of exactly 0.10, got "
            f"{routing_temperature}"
        )
    if model_config.get("act") != "tanh":
        raise PrototypeBankValidationError(
            "E6 requires the E3 tanh projection activation"
        )
    if model_config.get("hidden_layer") is not True:
        raise PrototypeBankValidationError(
            "E6 requires the E3 hidden_layer=True projection"
        )
    if int(model_config.get("clip_embed_dim", CAPTION_EMBED_DIM)) != CAPTION_EMBED_DIM:
        raise PrototypeBankValidationError("E6 requires clip_embed_dim=512")
    if (
        int(model_config.get("dino_embed_dim", 1024))
        != ROUTED_DINO_EMBED_DIM
    ):
        raise PrototypeBankValidationError("E6 requires dino_embed_dim=768")

    projection = ProjectionLayer.from_config(model_config)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    projection.load_state_dict(_checkpoint_state_dict(checkpoint), strict=True)
    projection.requires_grad_(False)
    projection.eval()
    projection.to(device)
    return projection, routing_temperature


def _load_source_archive(source_features_path: Path) -> tuple[list[Any], list[Any]]:
    source = torch.load(
        source_features_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(source, Mapping):
        raise PrototypeBankValidationError(
            "source feature archive must be a mapping"
        )
    images = source.get("images")
    annotations = source.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise PrototypeBankValidationError(
            "source feature archive must contain images and annotations lists"
        )
    return images, annotations


def _index_source(
    images: list[Any],
    annotations: list[Any],
) -> tuple[dict[int, Mapping[str, Any]], list[int], list[int]]:
    images_by_id: dict[int, Mapping[str, Any]] = {}
    for index, image in enumerate(images):
        if not isinstance(image, Mapping):
            raise PrototypeBankValidationError(
                f"source image at index {index} must be a mapping"
            )
        if "id" not in image:
            raise PrototypeBankValidationError(
                f"source image at index {index} is missing id"
            )
        image_id = _integer_id(image["id"], f"images[{index}].id")
        if image_id in images_by_id:
            raise PrototypeBankValidationError(
                f"duplicate source image ID: {image_id}"
            )
        images_by_id[image_id] = image

    annotation_ids: list[int] = []
    annotation_image_ids: list[int] = []
    seen_annotation_ids: set[int] = set()
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            raise PrototypeBankValidationError(
                f"source annotation at index {index} must be a mapping"
            )
        if "id" not in annotation or "image_id" not in annotation:
            raise PrototypeBankValidationError(
                f"source annotation at index {index} is missing id or image_id"
            )
        annotation_id = _integer_id(
            annotation["id"], f"annotations[{index}].id"
        )
        image_id = _integer_id(
            annotation["image_id"], f"annotations[{index}].image_id"
        )
        if annotation_id in seen_annotation_ids:
            raise PrototypeBankValidationError(
                f"duplicate source annotation ID: {annotation_id}"
            )
        if image_id not in images_by_id:
            raise PrototypeBankValidationError(
                f"annotation {annotation_id} refers to missing image {image_id}"
            )
        seen_annotation_ids.add(annotation_id)
        annotation_ids.append(annotation_id)
        annotation_image_ids.append(image_id)
    return images_by_id, annotation_ids, annotation_image_ids


def _publish_atomic(
    bank: Mapping[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing prototype bank: {output_path}; "
            "pass --overwrite explicitly"
        )
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(bank), temporary_path)
        with open(temporary_path, "rb") as handle:
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, output_path)
        else:
            try:
                os.link(temporary_path, output_path)
            except FileExistsError as error:
                raise FileExistsError(
                    "refusing to overwrite prototype bank created "
                    f"concurrently: {output_path}"
                ) from error
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_prototype_bank(
    *,
    source_features_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    model_config_path: os.PathLike[str] | str,
    checkpoint_path: os.PathLike[str] | str,
    batch_size: int = 1024,
    max_annotations: int | None = None,
    overwrite: bool = False,
    allow_dirty_source: bool = False,
    device: os.PathLike[str] | str | torch.device = "cpu",
) -> dict[str, Any]:
    """Construct, validate, and atomically publish one E6 bank."""

    source_features_path = Path(source_features_path).expanduser().resolve()
    output_path = Path(output_path).expanduser()
    model_config_path = Path(model_config_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not source_features_path.is_file():
        raise FileNotFoundError(f"source feature archive not found: {source_features_path}")
    if not model_config_path.is_file():
        raise FileNotFoundError(f"E3 model configuration not found: {model_config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"E3 checkpoint not found: {checkpoint_path}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if max_annotations is not None and (
        isinstance(max_annotations, bool)
        or not isinstance(max_annotations, int)
        or max_annotations <= 0
    ):
        raise ValueError("max_annotations must be a positive integer")
    provenance = source_git_provenance(
        _repository_root(),
        allow_dirty_source=allow_dirty_source,
    )
    if provenance["source_git_dirty"] and max_annotations is None:
        raise PrototypeBankValidationError(
            "--allow_dirty_source is restricted to development pilots; "
            "provide --max_annotations or commit the implementation before "
            "building a full bank"
        )
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing prototype bank: {output_path}; "
            "pass --overwrite explicitly"
        )

    source_feature_sha256 = sha256_file(source_features_path)
    images, annotations = _load_source_archive(source_features_path)
    images_by_id, source_annotation_ids, source_annotation_image_ids = _index_source(
        images,
        annotations,
    )
    source_image_count = len(images)
    source_annotation_count = len(annotations)
    is_pilot = max_annotations is not None
    selected_annotation_count = (
        min(max_annotations, source_annotation_count)
        if max_annotations is not None
        else source_annotation_count
    )
    selected_annotation_ids = source_annotation_ids[:selected_annotation_count]
    selected_image_ids = source_annotation_image_ids[:selected_annotation_count]

    config_sha256 = sha256_file(model_config_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    projection, routing_temperature = _load_e3_projection(
        model_config_path,
        checkpoint_path,
        device,
    )
    caption_embeddings = torch.empty(
        (selected_annotation_count, CAPTION_EMBED_DIM),
        dtype=torch.float16,
    )
    routed_dino_embeddings = torch.empty(
        (selected_annotation_count, ROUTED_DINO_EMBED_DIM),
        dtype=torch.float16,
    )
    image_ids = torch.tensor(selected_image_ids, dtype=torch.int64)
    annotation_ids = torch.tensor(selected_annotation_ids, dtype=torch.int64)
    validated_image_ids: set[int] = set()

    with torch.inference_mode():
        for start in range(0, selected_annotation_count, batch_size):
            stop = min(start + batch_size, selected_annotation_count)
            annotation_batch = annotations[start:stop]
            annotation_features = []
            image_head_features = []
            for offset, annotation in enumerate(annotation_batch, start=start):
                annotation_id = selected_annotation_ids[offset]
                image_id = selected_image_ids[offset]
                annotation_features.append(
                    _require_tensor(
                        annotation,
                        "ann_feats",
                        (CAPTION_EMBED_DIM,),
                        f"annotation {annotation_id}",
                    )
                )
                if image_id not in validated_image_ids:
                    image_heads = _require_tensor(
                        images_by_id[image_id],
                        "disentangled_self_attn",
                        (EXPECTED_ATTENTION_HEADS, ROUTED_DINO_EMBED_DIM),
                        f"image {image_id}",
                    )
                    validated_image_ids.add(image_id)
                else:
                    image_heads = images_by_id[image_id]["disentangled_self_attn"]
                image_head_features.append(image_heads)
            ann_batch_tensor = torch.stack(annotation_features).to(device)
            head_batch_tensor = torch.stack(image_head_features).to(device)
            normalized_captions, routed_dino = route_annotation_batch(
                projection,
                ann_batch_tensor,
                head_batch_tensor,
                routing_temperature,
            )
            caption_embeddings[start:stop].copy_(
                normalized_captions.to(device="cpu", dtype=torch.float16)
            )
            routed_dino_embeddings[start:stop].copy_(
                routed_dino.to(device="cpu", dtype=torch.float16)
            )

    expected_ids = set(selected_annotation_ids)
    completed_ids = set(annotation_ids.tolist())
    complete = (
        len(completed_ids) == selected_annotation_count
        and completed_ids == expected_ids
    )
    if not is_pilot:
        complete = complete and (
            selected_annotation_count == source_annotation_count
            and completed_ids == set(source_annotation_ids)
        )
    if not complete:
        missing_ids = sorted(expected_ids.difference(completed_ids))
        extra_ids = sorted(completed_ids.difference(expected_ids))
        raise RuntimeError(
            "prototype bank annotation coverage is incomplete: "
            f"missing={missing_ids[:20]}, extra={extra_ids[:20]}"
        )

    if sha256_file(source_features_path) != source_feature_sha256:
        raise RuntimeError(
            "source feature archive changed during bank construction: "
            f"{source_features_path}"
        )
    if sha256_file(model_config_path) != config_sha256:
        raise RuntimeError(
            f"E3 configuration changed during bank construction: {model_config_path}"
        )
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError(
            f"E3 checkpoint changed during bank construction: {checkpoint_path}"
        )
    metadata = {
        "format_version": FORMAT_VERSION,
        "complete": complete,
        "is_pilot": is_pilot,
        "source_feature_path": str(source_features_path),
        "source_feature_sha256": source_feature_sha256,
        "source_image_count": source_image_count,
        "source_annotation_count": source_annotation_count,
        "selected_annotation_count": selected_annotation_count,
        "e3_config_name": model_config_path.name,
        "e3_config_sha256": config_sha256,
        "e3_checkpoint_name": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "routing_temperature": routing_temperature,
        **provenance,
        "dimensions": {
            "caption_embeddings": CAPTION_EMBED_DIM,
            "routed_dino_embeddings": ROUTED_DINO_EMBED_DIM,
        },
        "dtypes": {
            "caption_embeddings": "float16",
            "routed_dino_embeddings": "float16",
            "image_ids": "int64",
            "annotation_ids": "int64",
        },
        "annotation_id_fingerprint": annotation_id_set_fingerprint(annotation_ids),
    }
    bank = {
        "caption_embeddings": caption_embeddings,
        "routed_dino_embeddings": routed_dino_embeddings,
        "image_ids": image_ids,
        "annotation_ids": annotation_ids,
        "metadata": metadata,
    }
    summary = validate_prototype_bank(
        bank,
        allow_pilot=is_pilot,
        allow_dirty_source=provenance["source_git_dirty"],
        require_complete=True,
        expected_config_path=model_config_path,
        expected_checkpoint_path=checkpoint_path,
        expected_source_features_path=source_features_path,
    )
    _publish_atomic(bank, output_path, overwrite=overwrite)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one compact E6 RGTP prototype bank."
    )
    parser.add_argument(
        "--source_features",
        "--source-features",
        required=True,
        type=Path,
        help="Monolithic Talk2DINO train.pth feature archive.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Atomic output .pth bank path.",
    )
    parser.add_argument(
        "--model_config",
        "--model-config",
        required=True,
        type=Path,
        help="E3 paired-soft-routing model YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Frozen E3 projection checkpoint.",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=1024,
        help="Bounded annotation routing batch size (default: 1024).",
    )
    parser.add_argument(
        "--max_annotations",
        "--max-annotations",
        type=int,
        default=None,
        help="Build an explicitly marked pilot from the first N annotations.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Projection compute device, e.g. cpu or cuda (default: cpu).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing bank atomically.",
    )
    parser.add_argument(
        "--allow_dirty_source",
        "--allow-dirty-source",
        action="store_true",
        help=(
            "Permit dirty Git provenance for development pilots only; requires "
            "--max_annotations and produces a bank rejected by evaluation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_prototype_bank(
        source_features_path=args.source_features,
        output_path=args.output,
        model_config_path=args.model_config,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        max_annotations=args.max_annotations,
        overwrite=args.overwrite,
        allow_dirty_source=args.allow_dirty_source,
        device=args.device,
    )
    print("E6 prototype bank written successfully")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
