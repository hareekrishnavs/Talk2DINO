#!/usr/bin/env python3
"""Build one compact E7 leave-one-image-out query-target bank."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from build_e6_prototype_bank import (
    _index_source,
    _load_e3_projection,
    _load_source_archive,
    _require_tensor,
)
from src.e6_prototype_bank import (
    CAPTION_EMBED_DIM,
    EXPECTED_ATTENTION_HEADS,
    ROUTED_DINO_EMBED_DIM,
    annotation_id_set_fingerprint,
    sha256_file,
)
from src.e7_training_bank import (
    FORMAT_VERSION,
    E7TrainingBankValidationError,
    route_e7_annotation_batch,
    validate_e7_training_bank,
)


def _run_git(root: Path, arguments: list[str], returncodes=(0,)):
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
    )
    if result.returncode not in returncodes:
        raise RuntimeError(
            f"Git provenance command failed: git {' '.join(arguments)}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result


def source_git_provenance(
    repository_root: Path,
    *,
    allow_dirty_source: bool = False,
) -> dict[str, Any]:
    """Record tracked, staged, and untracked source identity."""

    root = repository_root.resolve()
    commit = _run_git(root, ["rev-parse", "HEAD"]).stdout.decode("ascii").strip()
    tracked_dirty = _run_git(
        root, ["diff", "--quiet", "HEAD", "--"], returncodes=(0, 1)
    ).returncode == 1
    untracked = [
        item.decode("utf-8", errors="surrogateescape")
        for item in _run_git(
            root, ["ls-files", "--others", "--exclude-standard", "-z"]
        ).stdout.split(b"\0")
        if item
    ]
    dirty = tracked_dirty or bool(untracked)
    if dirty and not allow_dirty_source:
        status = _run_git(
            root, ["status", "--short", "--untracked-files=all"]
        ).stdout.decode(errors="replace")
        raise E7TrainingBankValidationError(
            "E7 bank construction requires a clean Git worktree. Commit the "
            "E7 implementation first. Use --allow_dirty_source only with a "
            f"development pilot. Dirty paths:\n{status.strip()}"
        )
    diff_sha = None
    if dirty:
        digest = hashlib.sha256()
        digest.update(b"tracked-and-staged\0")
        digest.update(_run_git(root, ["diff", "--binary", "HEAD", "--"]).stdout)
        for relative_path in sorted(untracked):
            digest.update(b"untracked\0")
            digest.update(relative_path.encode(errors="surrogateescape"))
            digest.update(b"\0")
            digest.update(
                _run_git(
                    root,
                    [
                        "diff",
                        "--binary",
                        "--no-index",
                        "--",
                        "/dev/null",
                        relative_path,
                    ],
                    returncodes=(0, 1),
                ).stdout
            )
        diff_sha = digest.hexdigest()
    return {
        "source_git_commit": commit,
        "source_git_dirty": dirty,
        "source_git_diff_sha256": diff_sha,
    }


def _require_unchanged_git_provenance(
    repository_root: Path,
    initial_provenance: Mapping[str, Any],
    *,
    allow_dirty_source: bool,
) -> dict[str, Any]:
    """Reject any tracked, staged, or untracked source mutation during a build."""

    try:
        current = source_git_provenance(
            repository_root,
            allow_dirty_source=allow_dirty_source,
        )
    except E7TrainingBankValidationError as error:
        raise E7TrainingBankValidationError(
            "Git source provenance changed during E7 bank construction"
        ) from error
    if dict(current) != dict(initial_provenance):
        raise E7TrainingBankValidationError(
            "Git source provenance changed during E7 bank construction: "
            f"initial={dict(initial_provenance)}, current={dict(current)}"
        )
    return current


def _require_unchanged_inputs(
    identities: tuple[tuple[Path, str, str], ...],
) -> None:
    for path, expected_sha256, label in identities:
        if sha256_file(path) != expected_sha256:
            raise RuntimeError(f"{label} changed during E7 bank construction")


def _publish_atomic(
    bank: Mapping[str, Any], output_path: Path, *, overwrite: bool
) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite E7 training bank: {output_path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(bank), temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, output_path)
        else:
            try:
                os.link(temporary, output_path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing concurrent overwrite of {output_path}"
                ) from error
    finally:
        temporary.unlink(missing_ok=True)


def build_e7_training_bank(
    *,
    source_features_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    split_name: str,
    model_config_path: os.PathLike[str] | str,
    checkpoint_path: os.PathLike[str] | str,
    batch_size: int = 1024,
    max_annotations: int | None = None,
    overwrite: bool = False,
    allow_dirty_source: bool = False,
    device: os.PathLike[str] | str | torch.device = "cpu",
    repository_root: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Construct, verify, and atomically publish a train or validation bank."""

    source_path = Path(source_features_path).expanduser().resolve()
    output_path = Path(output_path).expanduser()
    config_path = Path(model_config_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parent
    )
    if split_name not in {"train", "val"}:
        raise ValueError("split_name must be train or val")
    for path, description in (
        (source_path, "source feature archive"),
        (config_path, "E3 configuration"),
        (checkpoint_path, "E3 checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if max_annotations is not None and (
        isinstance(max_annotations, bool)
        or not isinstance(max_annotations, int)
        or max_annotations <= 0
    ):
        raise ValueError("max_annotations must be a positive integer")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output_path}")
    provenance = source_git_provenance(
        root, allow_dirty_source=allow_dirty_source
    )
    if provenance["source_git_dirty"] and max_annotations is None:
        raise E7TrainingBankValidationError(
            "dirty-source E7 construction is restricted to pilots; provide "
            "--max_annotations or commit the implementation"
        )
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    source_sha = sha256_file(source_path)
    config_sha = sha256_file(config_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    images, annotations = _load_source_archive(source_path)
    images_by_id, source_annotation_ids, source_image_ids = _index_source(
        images, annotations
    )
    selected_count = (
        min(max_annotations, len(annotations))
        if max_annotations is not None
        else len(annotations)
    )
    selected_annotation_ids = source_annotation_ids[:selected_count]
    selected_image_ids = source_image_ids[:selected_count]
    projection, routing_temperature = _load_e3_projection(
        config_path, checkpoint_path, device
    )

    captions = torch.empty((selected_count, 512), dtype=torch.float16)
    mapped_queries = torch.empty((selected_count, 768), dtype=torch.float16)
    routed_targets = torch.empty((selected_count, 768), dtype=torch.float16)
    image_ids = torch.tensor(selected_image_ids, dtype=torch.int64)
    annotation_ids = torch.tensor(selected_annotation_ids, dtype=torch.int64)
    validated_images: set[int] = set()
    with torch.inference_mode():
        for start in range(0, selected_count, batch_size):
            stop = min(start + batch_size, selected_count)
            ann_features = []
            image_heads = []
            for row in range(start, stop):
                annotation_id = selected_annotation_ids[row]
                image_id = selected_image_ids[row]
                ann_features.append(
                    _require_tensor(
                        annotations[row],
                        "ann_feats",
                        (CAPTION_EMBED_DIM,),
                        f"annotation {annotation_id}",
                    )
                )
                if image_id not in validated_images:
                    heads = _require_tensor(
                        images_by_id[image_id],
                        "disentangled_self_attn",
                        (EXPECTED_ATTENTION_HEADS, ROUTED_DINO_EMBED_DIM),
                        f"image {image_id}",
                    )
                    validated_images.add(image_id)
                else:
                    heads = images_by_id[image_id]["disentangled_self_attn"]
                image_heads.append(heads)
            raw, mapped, routed = route_e7_annotation_batch(
                projection,
                torch.stack(ann_features).to(device),
                torch.stack(image_heads).to(device),
                routing_temperature,
            )
            captions[start:stop].copy_(raw.cpu().half())
            mapped_queries[start:stop].copy_(mapped.cpu().half())
            routed_targets[start:stop].copy_(routed.cpu().half())

    expected_ids = set(selected_annotation_ids)
    completed_ids = set(annotation_ids.tolist())
    complete = completed_ids == expected_ids and len(completed_ids) == selected_count
    if not complete:
        raise RuntimeError("E7 bank annotation coverage is incomplete")
    input_identities = (
        (source_path, source_sha, "source archive"),
        (config_path, config_sha, "E3 configuration"),
        (checkpoint_path, checkpoint_sha, "E3 checkpoint"),
    )
    _require_unchanged_inputs(input_identities)

    metadata = {
        "format_version": FORMAT_VERSION,
        "complete": complete,
        "is_pilot": max_annotations is not None,
        "split_name": split_name,
        "source_feature_path": str(source_path),
        "source_feature_sha256": source_sha,
        "source_image_count": len(images),
        "source_annotation_count": len(annotations),
        "selected_annotation_count": selected_count,
        "annotation_id_fingerprint": annotation_id_set_fingerprint(annotation_ids),
        "e3_config_name": config_path.name,
        "e3_config_sha256": config_sha,
        "e3_checkpoint_name": checkpoint_path.name,
        "e3_checkpoint_sha256": checkpoint_sha,
        "routing_temperature": routing_temperature,
        **provenance,
        "dimensions": {
            "caption_embeddings": 512,
            "mapped_query_embeddings": 768,
            "routed_target_embeddings": 768,
        },
        "dtypes": {
            "caption_embeddings": "float16",
            "mapped_query_embeddings": "float16",
            "routed_target_embeddings": "float16",
            "image_ids": "int64",
            "annotation_ids": "int64",
        },
    }
    bank = {
        "caption_embeddings": captions,
        "mapped_query_embeddings": mapped_queries,
        "routed_target_embeddings": routed_targets,
        "image_ids": image_ids,
        "annotation_ids": annotation_ids,
        "metadata": metadata,
    }
    _require_unchanged_git_provenance(
        root,
        provenance,
        allow_dirty_source=allow_dirty_source,
    )
    summary = validate_e7_training_bank(
        bank,
        allow_pilot=metadata["is_pilot"],
        allow_dirty_source=metadata["source_git_dirty"],
        expected_split=split_name,
        expected_config_path=config_path,
        expected_checkpoint_path=checkpoint_path,
        expected_source_features_path=source_path,
    )
    _require_unchanged_inputs(input_identities)
    _require_unchanged_git_provenance(
        root,
        provenance,
        allow_dirty_source=allow_dirty_source,
    )
    _publish_atomic(bank, output_path, overwrite=overwrite)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--model_config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_annotations", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_dirty_source", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_e7_training_bank(
        source_features_path=args.source_features,
        output_path=args.output,
        split_name=args.split,
        model_config_path=args.model_config,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        max_annotations=args.max_annotations,
        overwrite=args.overwrite,
        allow_dirty_source=args.allow_dirty_source,
        device=args.device,
    )
    print("E7 training bank written successfully")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
