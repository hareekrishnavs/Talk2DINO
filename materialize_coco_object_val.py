#!/usr/bin/env python3
"""Materialize the 5000 COCO-Object val2017 '_instanceTrainIds.png' masks
that the repository's existing COCOObjectDataset/coco.py protocol already
expects but that do not yet exist on disk.

Data preparation only: no CUDA, no model, no scientific evaluation. Reuses
the already-committed canonical converter's own clsID_to_trID mapping
table verbatim (hash-checked against the identity before use) -- never a
redesigned class mapping, background threshold, or annotation-conversion
algorithm. Val split only: this CLI has no flag capable of producing
train2017 masks.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import subprocess
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent

from src.coco_object_val_materialization import (
    CLASS_COUNT,
    CocoObjectValMaterializationError,
    aggregate_decoded_digest,
    aggregate_encoded_digest,
    aggregate_records,
    build_lookup_table,
    canonical_image_ids,
    convert_one_image,
    image_order_digest as compute_image_order_digest,
    load_canonical_mapping,
    output_mask_path,
    raw_mask_path,
    sha256_file,
    source_masks_digest as compute_source_masks_digest,
    validate_output_root_isolation,
    write_json_atomically,
)
from src.coco_object_val_materialization_checkpoint import (
    resume_next_index,
    validate_checkpoint_against_canonical_order,
    validate_checkpoint_structure,
)
from src.coco_object_val_materialization_identity import (
    CocoObjectValMaterializationIdentityError,
    load_identity,
    validate_static_configuration,
)
from src.native_edge_support_checkpoint import parse_strict_json_document

MIN_FREE_BYTES = 5 * 1024 * 1024 * 1024  # conservative: ~5000 small PNGs comfortably fit well under this


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize the 5000 COCO-Object val2017 _instanceTrainIds.png masks from the "
        "already-present raw COCO-Stuff per-pixel category masks, using the repository's own "
        "canonical convert_coco_object.py mapping table verbatim."
    )
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--identity", type=Path, default=None)
    parser.add_argument("--source-masks", type=Path, required=True, help="directory of raw (non-suffixed) COCO-Stuff category-ID .png masks, e.g. .../coco_stuff164k/annotations/val2017")
    parser.add_argument("--source-images", type=Path, required=True, help="directory of canonical .jpg validation images, e.g. .../coco_stuff164k/images/val2017")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-incomplete", action="store_true", help="allow starting fresh against a nonempty but unproven output directory")
    parser.add_argument(
        "--test-limit", type=int, default=None,
        help="bounded, explicitly noncanonical test mode: convert only the first N images. Requires --allow-noncanonical-test-mode.",
    )
    parser.add_argument("--allow-noncanonical-test-mode", action="store_true")
    return parser


def _validate_split_directory(path: Path, *, expected_split: str, label: str) -> None:
    if path.name != expected_split:
        raise CocoObjectValMaterializationError(
            f"{label} {path} does not resolve to the canonical split directory name {expected_split!r} "
            f"(observed final path component: {path.name!r}); refusing an arbitrary/train split"
        )
    if not path.is_dir():
        raise CocoObjectValMaterializationError(f"{label} does not exist or is not a directory: {path}")


def _check_free_space(path: Path) -> None:
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            raise CocoObjectValMaterializationError(f"cannot resolve any existing ancestor of {path} to check free space")
        probe = probe.parent
    free_bytes = shutil.disk_usage(probe).free
    if free_bytes < MIN_FREE_BYTES:
        raise CocoObjectValMaterializationError(
            f"insufficient free space at {probe}: {free_bytes} bytes free, require at least {MIN_FREE_BYTES} bytes"
        )


def _setup_output_structure(output_root: Path, source_images: Path) -> None:
    (output_root / "annotations" / "val2017").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
    (output_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    images_link = output_root / "images" / "val2017"
    images_link.parent.mkdir(parents=True, exist_ok=True)
    if images_link.exists() or images_link.is_symlink():
        if not (images_link.is_symlink() and images_link.resolve() == source_images.resolve()):
            raise CocoObjectValMaterializationError(
                f"{images_link} already exists and is not a symlink to the canonical source images directory"
            )
    else:
        images_link.symlink_to(source_images.resolve())


def _verify_existing_mask(path: Path, record: dict) -> None:
    if not path.is_file():
        raise CocoObjectValMaterializationError(f"checkpoint claims {record['image_id']} is complete but {path} does not exist")
    encoded = sha256_file(path)
    if encoded != record["encoded_png_sha256"]:
        raise CocoObjectValMaterializationError(f"installed mask {path} encoded-PNG hash disagrees with checkpoint record for {record['image_id']}")
    decoded = np.array(Image.open(path))
    if hashlib.sha256(decoded.tobytes()).hexdigest() != record["decoded_pixel_sha256"]:
        raise CocoObjectValMaterializationError(f"installed mask {path} decoded-pixel hash disagrees with checkpoint record for {record['image_id']}")


def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _run(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    identity = load_identity(args.identity, repo_root=root)
    validate_static_configuration(repo_root=root, identity_path=args.identity, check_git=True)
    identity_path = args.identity if args.identity is not None else root / "evaluation_identities/e12_coco_object_val_materialization.toml"
    identity_sha256 = sha256_file(identity_path)

    split = identity["protocol"]["split"]
    _validate_split_directory(args.source_masks, expected_split=split, label="--source-masks")
    _validate_split_directory(args.source_images, expected_split=split, label="--source-images")

    validate_output_root_isolation(
        output_root=args.output_root, source_masks_dir=args.source_masks,
        source_annotation_root_marker=identity["policy"]["source_annotation_root_marker"],
    )
    resolved_output = str(args.output_root.resolve())
    if identity["policy"]["source_annotation_root_marker"] in resolved_output:
        raise CocoObjectValMaterializationError(f"--output-root {resolved_output} resolves inside the source annotation root")

    if args.test_limit is not None and not args.allow_noncanonical_test_mode:
        raise CocoObjectValMaterializationError("--test-limit requires --allow-noncanonical-test-mode")
    if args.test_limit is not None and args.test_limit >= identity["protocol"]["expected_image_count"]:
        raise CocoObjectValMaterializationError("--test-limit must be smaller than the canonical expected image count")

    _check_free_space(args.output_root if args.output_root.exists() else args.output_root.parent)

    mapping = load_canonical_mapping(root, identity)
    lut = build_lookup_table(mapping)

    ids = canonical_image_ids(args.source_images, image_suffix=identity["source"]["image_suffix"])
    expected_count = args.test_limit if args.test_limit is not None else identity["protocol"]["expected_image_count"]
    if args.test_limit is None and len(ids) != identity["protocol"]["expected_image_count"]:
        raise CocoObjectValMaterializationError(
            f"--source-images contains {len(ids)} images, expected exactly {identity['protocol']['expected_image_count']}"
        )
    if len(set(ids)) != len(ids):
        raise CocoObjectValMaterializationError("--source-images contains a duplicate image ID")
    ids = ids[:expected_count]

    for image_id in ids:
        if not raw_mask_path(args.source_masks, image_id).is_file():
            raise CocoObjectValMaterializationError(f"missing source raw mask for canonical image {image_id} under {args.source_masks}")

    digest = compute_image_order_digest(ids)

    _setup_output_structure(args.output_root, args.source_images)
    output_annotations_dir = args.output_root / "annotations" / split
    output_mask_suffix = identity["converter"]["output_mask_suffix"]
    checkpoint_schema_name = identity["checkpoint"]["schema_name"]

    per_image_records: list[dict] = []
    if args.checkpoint.exists():
        if not args.resume:
            raise CocoObjectValMaterializationError(f"--checkpoint {args.checkpoint} already exists; pass --resume to continue it")
        checkpoint = parse_strict_json_document(args.checkpoint, label="materialization checkpoint")
        validate_checkpoint_structure(
            checkpoint, identity=identity, identity_sha256=identity_sha256, checkpoint_schema_name=checkpoint_schema_name,
            expected_image_count=expected_count,
        )
        validate_checkpoint_against_canonical_order(checkpoint, ids, image_order_digest=digest)
        next_index = resume_next_index(checkpoint)
        for record in checkpoint["per_image_records"]:
            _verify_existing_mask(output_mask_path(output_annotations_dir, record["image_id"], suffix=output_mask_suffix), record)
        per_image_records = list(checkpoint["per_image_records"])
    else:
        if not args.resume:
            existing_masks = list(output_annotations_dir.glob(f"*{output_mask_suffix}"))
            if existing_masks and not args.overwrite_incomplete:
                raise CocoObjectValMaterializationError(
                    f"{output_annotations_dir} already contains {len(existing_masks)} mask file(s) with no checkpoint "
                    "provenance; pass --overwrite-incomplete to start fresh against unproven output, or --resume "
                    "if a checkpoint exists"
                )
        next_index = 0

    created_at_utc = _now_utc()
    for index in range(next_index, len(ids)):
        image_id = ids[index]
        record = convert_one_image(
            image_id, source_masks_dir=args.source_masks, output_annotations_dir=output_annotations_dir,
            lut=lut, output_mask_suffix=output_mask_suffix,
        )
        per_image_records.append(record)
        aggregate = aggregate_records(per_image_records)
        checkpoint_document = {
            "schema": checkpoint_schema_name,
            "identity": identity["identity"]["name"],
            "identity_sha256": identity_sha256,
            "git_commit": _git_commit(root),
            "split": split,
            "expected_image_count": identity["protocol"]["expected_image_count"] if args.test_limit is None else len(ids),
            "image_order_digest": digest,
            "next_index": index + 1,
            "completed_image_ids": [r["image_id"] for r in per_image_records],
            "per_image_records": per_image_records,
            "aggregate_label_histogram": aggregate["aggregate_label_histogram"],
            "total_pixels": aggregate["total_pixels"],
            "masks_with_foreground": aggregate["masks_with_foreground"],
            "all_background_masks": aggregate["all_background_masks"],
            "complete": (index + 1) == len(ids),
            "created_at_utc": created_at_utc,
            "updated_at_utc": _now_utc(),
        }
        write_json_atomically(args.checkpoint, checkpoint_document)

    if len(per_image_records) != len(ids):
        raise CocoObjectValMaterializationError("internal inconsistency: per_image_records count disagrees with the canonical image list")

    if args.test_limit is not None:
        print(
            f"COCO-OBJECT VAL MATERIALIZATION PASS (noncanonical test mode, {len(ids)} images) "
            f"checkpoint={args.checkpoint} (no manifest written in test mode)"
        )
        return 0

    aggregate = aggregate_records(per_image_records)
    manifest_document = {
        "schema": identity["manifest"]["schema_name"],
        "identity": identity["identity"]["name"],
        "identity_sha256": identity_sha256,
        "git_commit": _git_commit(root),
        "split": split,
        "complete": True,
        "final": True,
        "image_count": len(ids),
        "image_order_digest": digest,
        "completed_image_ids_digest": compute_image_order_digest([r["image_id"] for r in per_image_records]),
        "aggregate_decoded_mask_digest": aggregate_decoded_digest(per_image_records),
        "aggregate_encoded_file_digest": aggregate_encoded_digest(per_image_records),
        "source_masks_digest": compute_source_masks_digest(per_image_records),
        "converter_sha256": identity["converter"]["canonical_converter_sha256"],
        "dataset_class_sha256": identity["converter"]["dataset_class_sha256"],
        "dataset_config_sha256": identity["converter"]["dataset_config_sha256"],
        "materialization_identity_sha256": identity_sha256,
        "class_count": CLASS_COUNT,
        "background_class_index": identity["class_contract"]["background_class_index"],
        "output_mask_suffix": output_mask_suffix,
        "dtype": identity["output"]["dtype"],
        "aggregate_label_histogram": aggregate["aggregate_label_histogram"],
        "total_pixels": aggregate["total_pixels"],
        "masks_with_foreground": aggregate["masks_with_foreground"],
        "all_background_masks": aggregate["all_background_masks"],
        "crowd_overlap_policy": identity["label_mapping"]["crowd_or_overlap_handling"],
        "no_source_mutation": True,
        "no_train_masks_generated": True,
        "software_versions": _software_versions(),
        "created_at_utc": _now_utc(),
    }
    from src.coco_object_val_materialization_report import verify_record
    verify_record(manifest_document, identity, identity_sha256=identity_sha256)
    write_json_atomically(args.manifest, manifest_document)

    print(
        f"COCO-OBJECT VAL MATERIALIZATION PASS split={split} images={len(ids)} "
        f"total_pixels={aggregate['total_pixels']} masks_with_foreground={aggregate['masks_with_foreground']} "
        f"all_background_masks={aggregate['all_background_masks']} -> {args.manifest}"
    )
    return 0


def _git_commit(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise CocoObjectValMaterializationError(f"cannot resolve git HEAD: {result.stderr.strip()}")
    return result.stdout.strip()


def _software_versions() -> dict[str, str]:
    import PIL
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pillow": PIL.__version__}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (
        CocoObjectValMaterializationError,
        CocoObjectValMaterializationIdentityError,
        OSError,
        ValueError,
    ) as error:
        print(f"COCO-OBJECT VAL MATERIALIZATION FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
