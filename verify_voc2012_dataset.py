#!/usr/bin/env python3
"""Verifier CLI for the shared PASCAL VOC 2012 dataset-source identity
underlying Talk2DINO's V20 (no-background) and V21 (with-background)
protocols.

``preflight`` validates the identity/configuration and the live
dataset's basic shape (root resolution, split integrity) without a full
per-file scan. ``generate-manifest`` performs the full exhaustive scan
(every validation image and mask, decoded and hashed) and writes a
deterministic manifest. ``verify-manifest`` re-scans the live dataset
and requires it to match an existing manifest exactly.

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

from src.native_edge_support_checkpoint import parse_strict_json_document
from src.voc2012_dataset_identity import (
    Voc2012DatasetIdentityError,
    load_identity,
    validate_static_configuration,
    validate_v20_class_contract_against_source,
    validate_v21_class_contract_against_installed_mmseg,
)
from src.voc2012_dataset_manifest import (
    MANIFEST_TOP_KEYS,
    build_manifest,
    canonical_validation_ids,
    image_order_digest,
    resolve_dataset_root,
    scan_validation_split,
    sha256_file,
    verify_manifest_against_identity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the shared PASCAL VOC 2012 (V20/V21) dataset-source identity/manifest")
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
    return args.identity if args.identity is not None else root / "evaluation_identities/e12_voc2012_dataset_source.toml"


def _cmd_preflight(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    identity = load_identity(args.identity, repo_root=root)
    validate_static_configuration(repo_root=root, identity_path=args.identity, check_git=True)
    validate_v20_class_contract_against_source(root, identity)
    validate_v21_class_contract_against_installed_mmseg(identity)

    voc_root = resolve_dataset_root(args.data_root, identity)
    ids = canonical_validation_ids(voc_root, identity)

    print(
        f"VOC2012 DATASET SOURCE PREFLIGHT PASS identity={identity['identity']['name']} "
        f"split={identity['protocol']['split']} image_count={len(ids)} "
        f"checks=configuration,loader_provenance,v20_class_contract,v21_class_contract,root_resolution,split_integrity"
    )
    return 0


def _cmd_generate_manifest(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    identity = load_identity(args.identity, repo_root=root)
    identity_sha256 = sha256_file(_identity_path_for(args, root))
    validate_static_configuration(repo_root=root, identity_path=args.identity, check_git=True)
    validate_v20_class_contract_against_source(root, identity)
    validate_v21_class_contract_against_installed_mmseg(identity)

    if args.output.exists() and not args.overwrite:
        raise Voc2012DatasetIdentityError(f"refusing to overwrite existing manifest at {args.output} without --overwrite")

    voc_root = resolve_dataset_root(args.data_root, identity)
    ids = canonical_validation_ids(voc_root, identity)
    scan = scan_validation_split(voc_root, identity, ids)
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    manifest = build_manifest(
        identity=identity, identity_sha256=identity_sha256, voc_root=voc_root, image_ids=ids, scan=scan,
        generated_at_utc=generated_at_utc,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
    text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, args.output)

    print(
        f"VOC2012 DATASET SOURCE MANIFEST GENERATED images={manifest['image_count']} "
        f"observed_labels={manifest['observed_label_set']} -> {args.output}"
    )
    return 0


def _cmd_verify_manifest(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    identity = load_identity(args.identity, repo_root=root)
    identity_sha256 = sha256_file(_identity_path_for(args, root))
    validate_static_configuration(repo_root=root, identity_path=args.identity, check_git=True)

    manifest = parse_strict_json_document(args.manifest, label="voc2012 dataset manifest")
    verify_manifest_against_identity(manifest, identity, identity_sha256=identity_sha256)

    voc_root = resolve_dataset_root(args.data_root, identity)
    ids = canonical_validation_ids(voc_root, identity)
    if image_order_digest(ids) != manifest["image_order_digest"]:
        raise Voc2012DatasetIdentityError("freshly-resolved image_order_digest disagrees with the manifest")

    scan = scan_validation_split(voc_root, identity, ids)
    fresh = build_manifest(
        identity=identity, identity_sha256=identity_sha256, voc_root=voc_root, image_ids=ids, scan=scan,
        generated_at_utc=manifest["generated_at_utc"],
    )
    if fresh != manifest:
        diffs = [key for key in MANIFEST_TOP_KEYS if fresh.get(key) != manifest.get(key)]
        raise Voc2012DatasetIdentityError(f"freshly-recomputed manifest disagrees with the supplied manifest at field(s): {diffs}")

    print(
        f"VOC2012 DATASET SOURCE MANIFEST VERIFY PASS images={manifest['image_count']} "
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
    except (Voc2012DatasetIdentityError, OSError, ValueError) as error:
        print(f"VOC2012 DATASET SOURCE VERIFY FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
