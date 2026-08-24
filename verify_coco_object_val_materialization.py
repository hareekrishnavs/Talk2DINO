#!/usr/bin/env python3
"""Independent verifier for the COCO-Object val2017 mask-materialization
stage. ``verify-output`` never trusts the manifest's own claims -- every
count, hash, and digest it reports is independently recomputed from the
installed files on disk, and a fixed-selection spot-check re-applies the
hash-verified canonical mapping table via code that does not call the
production ``apply_canonical_mapping``/``convert_one_image`` wrappers."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent

from src.coco_object_val_materialization import (
    CLASS_COUNT,
    CocoObjectValMaterializationError,
    canonical_image_ids,
    image_order_digest as compute_image_order_digest,
    load_canonical_mapping,
    raw_mask_path,
    sha256_file,
)
from src.coco_object_val_materialization_checkpoint import validate_checkpoint_structure
from src.coco_object_val_materialization_identity import (
    CocoObjectValMaterializationIdentityError,
    load_identity,
    validate_static_configuration,
)
from src.coco_object_val_materialization_report import verify_record as verify_manifest_record
from src.native_edge_support_checkpoint import parse_strict_json_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the COCO-Object val2017 mask-materialization identity/checkpoint/output")
    parser.add_argument("--identity", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="validate identity/configuration without conversion")
    preflight.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    preflight.add_argument("--source-masks", type=Path, default=None)
    preflight.add_argument("--source-images", type=Path, default=None)
    preflight.add_argument("--output-root", type=Path, default=None)

    verify_ckpt = sub.add_parser("verify-checkpoint", help="validate an incomplete resumable checkpoint")
    verify_ckpt.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    verify_ckpt.add_argument("--checkpoint", type=Path, required=True)
    verify_ckpt.add_argument("--source-images", type=Path, required=True)

    verify_output = sub.add_parser("verify-output", help="independently scan and verify all final masks")
    verify_output.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    verify_output.add_argument("--manifest", type=Path, required=True)
    verify_output.add_argument("--output-root", type=Path, required=True)
    verify_output.add_argument("--source-masks", type=Path, required=True)
    verify_output.add_argument("--source-images", type=Path, required=True)

    return parser


def _cmd_preflight(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    identity = load_identity(args.identity, repo_root=root)
    validate_static_configuration(repo_root=root, identity_path=args.identity, check_git=True)
    load_canonical_mapping(root, identity)  # hash-checks + imports the converter; raises on any provenance mismatch

    split = identity["protocol"]["split"]
    checks = ["configuration", "converter_provenance"]
    if args.source_masks is not None:
        if args.source_masks.name != split:
            raise CocoObjectValMaterializationError(f"--source-masks does not resolve to split {split!r}")
        raw_files = [p for p in args.source_masks.glob("*.png") if "TrainIds" not in p.name]
        if len(raw_files) != identity["protocol"]["expected_image_count"]:
            raise CocoObjectValMaterializationError(
                f"--source-masks contains {len(raw_files)} raw masks, expected exactly {identity['protocol']['expected_image_count']}"
            )
        checks.append("source_masks_count")
    if args.source_images is not None:
        if args.source_images.name != split:
            raise CocoObjectValMaterializationError(f"--source-images does not resolve to split {split!r}")
        ids = canonical_image_ids(args.source_images, image_suffix=identity["source"]["image_suffix"])
        if len(ids) != identity["protocol"]["expected_image_count"]:
            raise CocoObjectValMaterializationError(
                f"--source-images contains {len(ids)} images, expected exactly {identity['protocol']['expected_image_count']}"
            )
        checks.append("source_images_count")
    if args.output_root is not None:
        from src.coco_object_val_materialization import validate_output_root_isolation
        validate_output_root_isolation(
            output_root=args.output_root, source_masks_dir=args.source_masks or (root / "does-not-exist"),
            source_annotation_root_marker=identity["policy"]["source_annotation_root_marker"],
        )
        checks.append("output_root_isolation")

    print(f"COCO-OBJECT VAL MATERIALIZATION PREFLIGHT PASS identity={identity['identity']['name']} split={split} checks={','.join(checks)}")
    return 0


def _cmd_verify_checkpoint(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    identity = load_identity(args.identity, repo_root=root)
    identity_path = args.identity if args.identity is not None else root / "evaluation_identities/e12_coco_object_val_materialization.toml"
    identity_sha256 = sha256_file(identity_path)

    checkpoint = parse_strict_json_document(args.checkpoint, label="materialization checkpoint")
    validate_checkpoint_structure(
        checkpoint, identity=identity, identity_sha256=identity_sha256,
        checkpoint_schema_name=identity["checkpoint"]["schema_name"],
    )

    ids = canonical_image_ids(args.source_images, image_suffix=identity["source"]["image_suffix"])
    digest = compute_image_order_digest(ids[: checkpoint["expected_image_count"]])
    from src.coco_object_val_materialization_checkpoint import validate_checkpoint_against_canonical_order
    validate_checkpoint_against_canonical_order(checkpoint, ids, image_order_digest=digest)

    print(
        f"COCO-OBJECT VAL MATERIALIZATION CHECKPOINT PASS next_index={checkpoint['next_index']} "
        f"completed={len(checkpoint['completed_image_ids'])} complete={checkpoint['complete']}"
    )
    return 0


_SPOT_CHECK_MIN_COUNT = 20


def _independent_remap(raw_mask: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Deliberately NOT calling apply_canonical_mapping/build_lookup_table:
    a plain per-value boolean-mask loop, independent of the production
    vectorized-LUT code path, sharing only the already hash-verified
    clsID_to_trID table itself."""
    out = np.empty_like(raw_mask, dtype=np.uint8)
    covered = np.zeros_like(raw_mask, dtype=bool)
    for raw_value, mapped_value in mapping.items():
        hit = raw_mask == raw_value
        out[hit] = mapped_value
        covered |= hit
    if not covered.all():
        bad = sorted(set(raw_mask[~covered].tolist()))
        raise CocoObjectValMaterializationError(f"independent spot-check found raw value(s) not covered by clsID_to_trID: {bad}")
    return out


def _select_spot_check_ids(ids: list[str], fresh_histograms: dict[str, list[int]]) -> list[str]:
    selected: list[str] = [ids[0], ids[-1]]
    all_background = next((i for i in ids if sum(fresh_histograms[i][1:]) == 0), None)
    if all_background is not None:
        selected.append(all_background)
    most_classes = max(ids, key=lambda i: sum(1 for c in fresh_histograms[i][1:] if c > 0))
    selected.append(most_classes)
    nonzero_min = None
    nonzero_min_count = None
    for image_id in ids:
        for count in fresh_histograms[image_id][1:]:
            if count > 0 and (nonzero_min_count is None or count < nonzero_min_count):
                nonzero_min_count = count
                nonzero_min = image_id
    if nonzero_min is not None:
        selected.append(nonzero_min)
    step = max(len(ids) // (_SPOT_CHECK_MIN_COUNT * 2), 1)
    index = 0
    while len(set(selected)) < _SPOT_CHECK_MIN_COUNT and index < len(ids):
        selected.append(ids[index])
        index += step
    deduped = list(dict.fromkeys(selected))
    return deduped[:max(_SPOT_CHECK_MIN_COUNT, len(deduped))]


def _cmd_verify_output(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    identity = load_identity(args.identity, repo_root=root)
    identity_path = args.identity if args.identity is not None else root / "evaluation_identities/e12_coco_object_val_materialization.toml"
    identity_sha256 = sha256_file(identity_path)

    manifest = parse_strict_json_document(args.manifest, label="materialization manifest")
    verify_manifest_record(manifest, identity, identity_sha256=identity_sha256)

    split = identity["protocol"]["split"]
    output_mask_suffix = identity["converter"]["output_mask_suffix"]
    output_annotations_dir = args.output_root / "annotations" / split

    ids = canonical_image_ids(args.source_images, image_suffix=identity["source"]["image_suffix"])
    if len(ids) != identity["protocol"]["expected_image_count"]:
        raise CocoObjectValMaterializationError(f"--source-images contains {len(ids)} images, expected {identity['protocol']['expected_image_count']}")
    fresh_digest = compute_image_order_digest(ids)
    if fresh_digest != manifest["image_order_digest"]:
        raise CocoObjectValMaterializationError("manifest.image_order_digest disagrees with a freshly-recomputed canonical order")

    # Enumerate every file in the directory, not just files matching the
    # expected suffix: a suffix-filtered glob would be blind to a stray
    # file with any other name (e.g. a leftover *_labelTrainIds.png), and
    # such a file must not be silently tolerated.
    on_disk = sorted(p.name for p in output_annotations_dir.iterdir() if p.is_file())
    expected_names = sorted(f"{image_id}{output_mask_suffix}" for image_id in ids)
    if on_disk != expected_names:
        missing = set(expected_names) - set(on_disk)
        extra = set(on_disk) - set(expected_names)
        raise CocoObjectValMaterializationError(f"output mask filename set mismatch: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")

    leftover_tmp = list(output_annotations_dir.glob("*.tmp*"))
    if leftover_tmp:
        raise CocoObjectValMaterializationError(f"temporary files remain in output: {[p.name for p in leftover_tmp][:5]}")
    # No separate "train-like filename" substring check: the output suffix
    # itself ("_instanceTrainIds.png") legitimately contains "train", so a
    # naive substring match would flag every correct file. The exact
    # filename-set equality check above (on_disk == expected_names) is
    # the actual train/extra-file detector: any id absent from the
    # canonical val image-order list (including a real train2017 id)
    # would already have failed that check.

    per_image_records = []
    aggregate_histogram = [0] * CLASS_COUNT
    fresh_histograms: dict[str, list[int]] = {}
    for image_id in ids:
        mask_path = output_annotations_dir / f"{image_id}{output_mask_suffix}"
        arr = np.array(Image.open(mask_path))
        if arr.dtype != np.uint8:
            raise CocoObjectValMaterializationError(f"{mask_path} has dtype {arr.dtype}, expected uint8")
        img_path = args.source_images / f"{image_id}{identity['source']['image_suffix']}"
        with Image.open(img_path) as source_image:
            expected_wh = source_image.size  # (width, height)
        if (arr.shape[1], arr.shape[0]) != expected_wh:
            raise CocoObjectValMaterializationError(f"{mask_path} dimensions {arr.shape[::-1]} disagree with source image size {expected_wh}")
        observed_labels = set(np.unique(arr).tolist())
        if not observed_labels <= set(range(CLASS_COUNT)):
            raise CocoObjectValMaterializationError(f"{mask_path} contains out-of-contract label(s): {observed_labels - set(range(CLASS_COUNT))}")

        histogram = np.bincount(arr.reshape(-1), minlength=CLASS_COUNT).astype(np.int64).tolist()
        fresh_histograms[image_id] = histogram
        for index in range(CLASS_COUNT):
            aggregate_histogram[index] += histogram[index]

        decoded_sha256 = hashlib.sha256(arr.tobytes()).hexdigest()
        encoded_sha256 = sha256_file(mask_path)
        raw_sha256 = sha256_file(raw_mask_path(args.source_masks, image_id))
        per_image_records.append(
            {"image_id": image_id, "decoded_pixel_sha256": decoded_sha256, "encoded_png_sha256": encoded_sha256, "raw_mask_sha256": raw_sha256}
        )

    if aggregate_histogram != manifest["aggregate_label_histogram"]:
        raise CocoObjectValMaterializationError("freshly recomputed aggregate_label_histogram disagrees with the manifest")
    total_pixels = sum(aggregate_histogram)
    if total_pixels != manifest["total_pixels"]:
        raise CocoObjectValMaterializationError("freshly recomputed total_pixels disagrees with the manifest")
    masks_with_foreground = sum(1 for h in fresh_histograms.values() if sum(h[1:]) > 0)
    all_background_masks = len(ids) - masks_with_foreground
    if masks_with_foreground != manifest["masks_with_foreground"] or all_background_masks != manifest["all_background_masks"]:
        raise CocoObjectValMaterializationError("freshly recomputed foreground/background mask counts disagree with the manifest")

    import json
    encoded_digest = hashlib.sha256(
        json.dumps([[r["image_id"], r["encoded_png_sha256"]] for r in per_image_records], ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    decoded_digest = hashlib.sha256(
        json.dumps([[r["image_id"], r["decoded_pixel_sha256"]] for r in per_image_records], ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if encoded_digest != manifest["aggregate_encoded_file_digest"]:
        raise CocoObjectValMaterializationError("freshly recomputed aggregate_encoded_file_digest disagrees with the manifest")
    if decoded_digest != manifest["aggregate_decoded_mask_digest"]:
        raise CocoObjectValMaterializationError("freshly recomputed aggregate_decoded_mask_digest disagrees with the manifest")
    source_digest = hashlib.sha256(
        json.dumps({r["image_id"]: r["raw_mask_sha256"] for r in per_image_records}, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if source_digest != manifest["source_masks_digest"]:
        raise CocoObjectValMaterializationError("freshly recomputed source_masks_digest disagrees with the manifest -- source masks may have changed since materialization")

    mapping = load_canonical_mapping(root, identity)
    spot_check_ids = _select_spot_check_ids(ids, fresh_histograms)
    for image_id in spot_check_ids:
        raw = np.array(Image.open(raw_mask_path(args.source_masks, image_id)))
        independent = _independent_remap(raw, mapping)
        installed = np.array(Image.open(output_annotations_dir / f"{image_id}{output_mask_suffix}"))
        if not np.array_equal(independent, installed):
            raise CocoObjectValMaterializationError(f"independent spot-check mismatch for {image_id}")

    print(
        f"COCO-OBJECT VAL MATERIALIZATION VERIFY-OUTPUT PASS images={len(ids)} total_pixels={total_pixels} "
        f"masks_with_foreground={masks_with_foreground} all_background_masks={all_background_masks} "
        f"spot_checked={len(spot_check_ids)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            return _cmd_preflight(args)
        if args.command == "verify-checkpoint":
            return _cmd_verify_checkpoint(args)
        if args.command == "verify-output":
            return _cmd_verify_output(args)
        parser.error(f"unknown command {args.command!r}")
        return 2
    except (
        CocoObjectValMaterializationError,
        CocoObjectValMaterializationIdentityError,
        OSError,
        ValueError,
    ) as error:
        print(f"COCO-OBJECT VAL MATERIALIZATION VERIFY FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
