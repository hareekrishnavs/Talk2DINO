#!/usr/bin/env python3
"""Verifier CLI for the ADE20K (ADEChallengeData2016) dataset-source
identity.

``preflight`` validates the identity/configuration and the live
dataset's basic shape (root resolution, split integrity) without a full
per-file scan. ``generate-manifest`` performs the full exhaustive scan
(every validation image and mask, decoded and hashed) and writes a
deterministic, path-independent manifest. ``verify-manifest`` re-scans
the live dataset and requires it to match an existing manifest exactly.

Never initializes CUDA, never loads a model, never imports torch, and
never extracts DINO/CLIP features. Never writes into the dataset root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ade20k_dataset_identity import (
    Ade20kDatasetIdentityError,
    load_identity,
    validate_class_contract_against_installed_mmseg,
    validate_static_configuration,
)
from src.ade20k_dataset_manifest import (
    MANIFEST_TOP_KEYS,
    build_manifest,
    canonical_validation_ids,
    check_train_val_disjointness,
    image_order_digest,
    resolve_dataset_root,
    scan_validation_split,
    sha256_file,
    verify_manifest_against_identity,
)
from src.native_edge_support_checkpoint import parse_strict_json_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the ADE20K (ADEChallengeData2016) dataset-source identity/manifest")
    parser.add_argument("--identity", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="validate identity/configuration and basic real-data shape, no full scan")
    preflight.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    preflight.add_argument("--data-root", type=Path, required=True)

    generate = sub.add_parser("generate-manifest", help="exhaustively scan the live dataset and write a deterministic manifest")
    generate.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    generate.add_argument("--data-root", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--overwrite", action="store_true")

    verify = sub.add_parser("verify-manifest", help="re-scan the live dataset and require it to match an existing manifest exactly")
    verify.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    verify.add_argument("--data-root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)

    return parser


def _identity_path_for(args: argparse.Namespace, root: Path) -> Path:
    return args.identity if args.identity is not None else root / "evaluation_identities/e12_ade20k_dataset_source.toml"


def _cmd_preflight(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    identity = load_identity(args.identity, repo_root=root)
    validate_static_configuration(repo_root=root, identity_path=args.identity, check_git=True)
    validate_class_contract_against_installed_mmseg(identity)

    ade_root = resolve_dataset_root(args.data_root, identity)
    ids = canonical_validation_ids(ade_root, identity)
    check_train_val_disjointness(ade_root, identity, ids)

    print(
        f"ADE20K DATASET SOURCE PREFLIGHT PASS identity={identity['identity']['name']} "
        f"split={identity['protocol']['split']} image_count={len(ids)} "
        f"checks=configuration,loader_provenance,class_contract,root_resolution,split_integrity,train_val_disjointness"
    )
    return 0


def _cmd_generate_manifest(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    identity = load_identity(args.identity, repo_root=root)
    identity_sha256 = sha256_file(_identity_path_for(args, root))
    validate_static_configuration(repo_root=root, identity_path=args.identity, check_git=True)
    validate_class_contract_against_installed_mmseg(identity)

    if args.output.exists() and not args.overwrite:
        raise Ade20kDatasetIdentityError(f"refusing to overwrite existing manifest at {args.output} without --overwrite")

    ade_root = resolve_dataset_root(args.data_root, identity)
    ids = canonical_validation_ids(ade_root, identity)
    training_image_count = check_train_val_disjointness(ade_root, identity, ids)
    scan = scan_validation_split(ade_root, identity, ids)
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    manifest = build_manifest(
        identity=identity, identity_sha256=identity_sha256, image_ids=ids, scan=scan,
        training_image_count=training_image_count, generated_at_utc=generated_at_utc,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
    try:
        text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
        temp_path.write_text(text, encoding="utf-8")
        reloaded = parse_strict_json_document(temp_path, label="freshly-written ade20k manifest")
        verify_manifest_against_identity(reloaded, identity, identity_sha256=identity_sha256)
        os.replace(temp_path, args.output)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    print(
        f"ADE20K DATASET SOURCE MANIFEST GENERATED images={manifest['image_count']} "
        f"observed_labels={manifest['observed_label_set']} -> {args.output}"
    )
    return 0


def _cmd_verify_manifest(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    identity = load_identity(args.identity, repo_root=root)
    identity_sha256 = sha256_file(_identity_path_for(args, root))
    validate_static_configuration(repo_root=root, identity_path=args.identity, check_git=True)

    manifest = parse_strict_json_document(args.manifest, label="ade20k dataset manifest")
    verify_manifest_against_identity(manifest, identity, identity_sha256=identity_sha256)

    ade_root = resolve_dataset_root(args.data_root, identity)
    ids = canonical_validation_ids(ade_root, identity)
    if image_order_digest(ids) != manifest["image_order_digest"]:
        raise Ade20kDatasetIdentityError("freshly-resolved image_order_digest disagrees with the manifest")

    training_image_count = check_train_val_disjointness(ade_root, identity, ids)
    scan = scan_validation_split(ade_root, identity, ids)
    fresh = build_manifest(
        identity=identity, identity_sha256=identity_sha256, image_ids=ids, scan=scan,
        training_image_count=training_image_count, generated_at_utc=manifest["generated_at_utc"],
    )
    if fresh != manifest:
        diffs = [key for key in MANIFEST_TOP_KEYS if fresh.get(key) != manifest.get(key)]
        raise Ade20kDatasetIdentityError(f"freshly-recomputed manifest disagrees with the supplied manifest at field(s): {diffs}")

    print(
        f"ADE20K DATASET SOURCE MANIFEST VERIFY PASS images={manifest['image_count']} "
        f"observed_labels={manifest['observed_label_set']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            return _cmd_preflight(args)
        if args.command == "generate-manifest":
            return _cmd_generate_manifest(args)
        if args.command == "verify-manifest":
            return _cmd_verify_manifest(args)
        parser.error(f"unknown command {args.command!r}")
        return 2
    except (Ade20kDatasetIdentityError, OSError, ValueError) as error:
        print(f"ADE20K DATASET SOURCE VERIFY FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
