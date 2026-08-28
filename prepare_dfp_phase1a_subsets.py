#!/usr/bin/env python3
"""DFP Phase 1A Step 2 CLI: generate/verify the two frozen 500-image
deterministic evaluation-subset manifests (COCO-Stuff, ADE20K).

CPU/data-read-only. Never initializes CUDA, never loads a model or
checkpoint into a model, never runs inference or propagation, never
modifies a dataset file. Reads real dataset images/masks only to (a)
independently verify them against their own frozen source identity and
(b) build binary image-level class-presence vectors for deterministic
stratification -- ground truth is never used for prediction
construction.

Subcommands
-----------
generate               resolve the dataset root, scan+verify the
                        canonical validation split, build class-presence
                        vectors, run the deterministic stratification
                        algorithm, build a manifest, verify it
                        independently through the production verifier,
                        and atomically install it.
verify                 strict schema/relational verification of an
                        already-generated manifest against the live
                        protocol/source identities (no dataset access).
verify-source-binding   like `verify`, plus independent recomputation of
                        the selection from the real dataset (requires the
                        dataset root).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dfp_phase1a_identity import (
    DfpPhase1aIdentityError,
    IDENTITY_RELATIVE_PATH as PROTOCOL_IDENTITY_RELATIVE_PATH,
    validate_static_configuration,
)
from src.dfp_phase1a_subset_manifest import (
    ADE20K_CLASS_COUNT,
    DATASET_CONTRACTS,
    DfpPhase1aSubsetManifestError,
    SELECTED_COUNT,
    ade20k_class_presence,
    build_manifest,
    coco_stuff_canonical_validation_ids,
    coco_stuff_class_presence,
    deterministic_iterative_stratify,
    parse_strict_json_document,
    require_env_root,
    resolve_coco_stuff_root,
    sha256_file,
    verify_manifest_document,
    verify_manifest_reconciles_with_selection,
    write_json_atomically,
)

ENV_VAR_BY_DATASET = {"coco_stuff": "COCO_STUFF_REAL_DATA_ROOT", "ade20k": "ADE20K_REAL_DATA_ROOT"}
DEFAULT_MANIFEST_PATH = {
    "coco_stuff": Path("evaluation_subsets/dfp_phase1a_coco_stuff_500.json"),
    "ade20k": Path("evaluation_subsets/dfp_phase1a_ade20k_500.json"),
}


class _AllErrors(Exception):
    """Not raised; exists only so mypy/readers see the caught union in one place."""


def _protocol_and_source_identity_sha256s(repo_root: Path, dataset_key: str, *, check_git: bool) -> tuple[str, str, str]:
    """Returns (protocol_identity_sha256, source_identity_sha256,
    class_names_digest). Validates the protocol identity via its own
    production loader (never re-implementing that check), and the
    dataset-specific parent identity likewise."""
    validate_static_configuration(repo_root=repo_root, check_git=check_git)
    protocol_sha256 = sha256_file(repo_root / PROTOCOL_IDENTITY_RELATIVE_PATH)

    contract = DATASET_CONTRACTS[dataset_key]
    source_path = repo_root / contract["source_identity_relative_path"]
    if dataset_key == "coco_stuff":
        from src.matched_k11_k12_identity import load_identity as load_matched
        identity = load_matched(source_path, repo_root=repo_root)
        class_names_digest = identity["dataset"].get("class_names_digest")
        if class_names_digest is None:
            # e12_matched_k11_k12_t320.toml's own [dataset] section does not
            # itself carry a class_names_digest field -- the DFP protocol
            # identity's coco_stuff_dataset.class_names_digest is the
            # authoritative binding for this fact, already validated by
            # validate_static_configuration above.
            from src.dfp_phase1a_identity import load_identity as load_protocol
            protocol = load_protocol(repo_root / PROTOCOL_IDENTITY_RELATIVE_PATH, repo_root=repo_root)
            class_names_digest = protocol["coco_stuff_dataset"]["class_names_digest"]
    else:
        from src.ade20k_dataset_identity import load_identity as load_ade_source
        identity = load_ade_source(source_path, repo_root=repo_root)
        class_names_digest = identity["class_contract"]["class_names_digest"]

    source_sha256 = sha256_file(source_path)
    return protocol_sha256, source_sha256, class_names_digest


def _build_manifest_for_dataset(
    *, repo_root: Path, dataset_key: str, data_root_override: str | None, ade20k_source_manifest: Path | None,
    check_git: bool,
) -> dict:
    protocol_sha256, source_sha256, class_names_digest = _protocol_and_source_identity_sha256s(
        repo_root, dataset_key, check_git=check_git,
    )

    root = require_env_root(ENV_VAR_BY_DATASET[dataset_key], override=data_root_override)

    source_manifest_sha256 = None
    if dataset_key == "coco_stuff":
        contract = DATASET_CONTRACTS["coco_stuff"]
        dataset_root = resolve_coco_stuff_root(root)
        canonical_ids = coco_stuff_canonical_validation_ids(dataset_root, expected_count=contract["source_validation_count"])
        presence = coco_stuff_class_presence(dataset_root, canonical_ids)
        class_count = contract["class_count"]
    else:
        from src.ade20k_dataset_identity import load_identity as load_ade_source
        from src.ade20k_dataset_manifest import (
            canonical_validation_ids as ade_canonical_ids,
            resolve_dataset_root as ade_resolve_root,
            verify_manifest_against_identity,
        )

        ade_identity = load_ade_source(repo_root / DATASET_CONTRACTS["ade20k"]["source_identity_relative_path"], repo_root=repo_root)
        dataset_root = ade_resolve_root(root, ade_identity)
        canonical_ids = ade_canonical_ids(dataset_root, ade_identity)

        if ade20k_source_manifest is not None:
            source_manifest_sha256 = sha256_file(ade20k_source_manifest)
            source_manifest_doc = parse_strict_json_document(ade20k_source_manifest, label="ADE20K source manifest")
            verify_manifest_against_identity(source_manifest_doc, ade_identity, identity_sha256=sha256_file(repo_root / DATASET_CONTRACTS["ade20k"]["source_identity_relative_path"]))
            if source_manifest_doc["image_order_digest"] != _ade_image_order_digest(canonical_ids):
                raise DfpPhase1aSubsetManifestError(
                    "the provided ADE20K source manifest's image_order_digest disagrees with the freshly "
                    "rescanned canonical validation IDs"
                )

        presence = ade20k_class_presence(dataset_root, ade_identity, canonical_ids)
        class_count = ADE20K_CLASS_COUNT

    result = deterministic_iterative_stratify(
        canonical_ids=canonical_ids, presence=presence, class_count=class_count,
        selected_count=SELECTED_COUNT, dataset_key=dataset_key,
    )
    manifest = build_manifest(
        dataset_key=dataset_key, protocol_identity_sha256=protocol_sha256, source_identity_sha256=source_sha256,
        source_manifest_sha256=source_manifest_sha256, class_names_digest=class_names_digest,
        canonical_ids=canonical_ids, presence=presence, stratify_result=result,
    )
    return manifest


def _ade_image_order_digest(image_ids: list[str]) -> str:
    from src.ade20k_dataset_manifest import image_order_digest
    return image_order_digest(image_ids)


def _cmd_generate(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve() if args.output else (repo_root / DEFAULT_MANIFEST_PATH[args.dataset]).resolve()

    if output_path.exists() and not args.overwrite:
        print(f"DFP PHASE 1A SUBSET GENERATE FAIL: {output_path} already exists; pass --overwrite to replace it", file=sys.stderr)
        return 2

    try:
        manifest = _build_manifest_for_dataset(
            repo_root=repo_root, dataset_key=args.dataset, data_root_override=args.data_root,
            ade20k_source_manifest=Path(args.ade20k_source_manifest) if args.ade20k_source_manifest else None,
            check_git=not args.no_git_check,
        )
    except (DfpPhase1aIdentityError, DfpPhase1aSubsetManifestError, OSError, ValueError) as error:
        print(f"DFP PHASE 1A SUBSET GENERATE FAIL: {error}", file=sys.stderr)
        return 2

    contract = DATASET_CONTRACTS[args.dataset]
    protocol_sha256, source_sha256, class_names_digest = _protocol_and_source_identity_sha256s(repo_root, args.dataset, check_git=not args.no_git_check)
    try:
        verify_manifest_document(
            manifest, dataset_key_expected=args.dataset, protocol_identity_sha256=protocol_sha256,
            source_identity_sha256=source_sha256, class_names_digest=class_names_digest,
        )
    except DfpPhase1aSubsetManifestError as error:
        print(f"DFP PHASE 1A SUBSET GENERATE FAIL: freshly built manifest failed independent verification: {error}", file=sys.stderr)
        return 2

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomically(output_path, manifest)
    except (DfpPhase1aIdentityError, DfpPhase1aSubsetManifestError, OSError, ValueError) as error:
        print(f"DFP PHASE 1A SUBSET GENERATE FAIL: cannot write {output_path}: {error}", file=sys.stderr)
        return 2

    print(f"PASS: DFP Phase 1A subset manifest generated: {output_path}")
    print(
        f"# dataset={args.dataset} selected={manifest['selected_count']} source={manifest['source_validation_count']} "
        f"selected_ids_digest={manifest['selected_ids_digest']}",
        file=sys.stderr,
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    try:
        manifest = parse_strict_json_document(manifest_path, label="subset manifest")
        protocol_sha256, source_sha256, class_names_digest = _protocol_and_source_identity_sha256s(repo_root, args.dataset, check_git=not args.no_git_check)
        verify_manifest_document(
            manifest, dataset_key_expected=args.dataset, protocol_identity_sha256=protocol_sha256,
            source_identity_sha256=source_sha256, class_names_digest=class_names_digest,
        )
    except (DfpPhase1aIdentityError, DfpPhase1aSubsetManifestError, OSError, ValueError) as error:
        print(f"DFP PHASE 1A SUBSET VERIFY FAIL: {error}", file=sys.stderr)
        return 2
    print("PASS: DFP Phase 1A subset manifest verified")
    print(f"# dataset={args.dataset} manifest={manifest_path}", file=sys.stderr)
    return 0


def _cmd_verify_source_binding(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    try:
        manifest = parse_strict_json_document(manifest_path, label="subset manifest")
        protocol_sha256, source_sha256, class_names_digest = _protocol_and_source_identity_sha256s(repo_root, args.dataset, check_git=not args.no_git_check)
        verify_manifest_document(
            manifest, dataset_key_expected=args.dataset, protocol_identity_sha256=protocol_sha256,
            source_identity_sha256=source_sha256, class_names_digest=class_names_digest,
        )

        root = require_env_root(ENV_VAR_BY_DATASET[args.dataset], override=args.data_root)
        if args.dataset == "coco_stuff":
            contract = DATASET_CONTRACTS["coco_stuff"]
            dataset_root = resolve_coco_stuff_root(root)
            canonical_ids = coco_stuff_canonical_validation_ids(dataset_root, expected_count=contract["source_validation_count"])
            presence = coco_stuff_class_presence(dataset_root, canonical_ids)
        else:
            from src.ade20k_dataset_identity import load_identity as load_ade_source
            from src.ade20k_dataset_manifest import canonical_validation_ids as ade_canonical_ids, resolve_dataset_root as ade_resolve_root

            ade_identity = load_ade_source(repo_root / DATASET_CONTRACTS["ade20k"]["source_identity_relative_path"], repo_root=repo_root)
            dataset_root = ade_resolve_root(root, ade_identity)
            canonical_ids = ade_canonical_ids(dataset_root, ade_identity)
            presence = ade20k_class_presence(dataset_root, ade_identity, canonical_ids)

        verify_manifest_reconciles_with_selection(manifest, canonical_ids=canonical_ids, presence=presence)
    except (DfpPhase1aIdentityError, DfpPhase1aSubsetManifestError, OSError, ValueError) as error:
        print(f"DFP PHASE 1A SUBSET VERIFY-SOURCE-BINDING FAIL: {error}", file=sys.stderr)
        return 2
    print("PASS: DFP Phase 1A subset manifest verified against the real dataset source binding")
    print(f"# dataset={args.dataset} manifest={manifest_path}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate/verify DFP Phase 1A deterministic evaluation-subset manifests")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    common_dataset = dict(choices=("coco_stuff", "ade20k"), required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--dataset", **common_dataset)
    gen.add_argument("--output", type=Path, default=None)
    gen.add_argument("--data-root", type=str, default=None, help="override for the *_REAL_DATA_ROOT environment variable")
    gen.add_argument("--ade20k-source-manifest", type=str, default=None, help="path to an externally-verified ADE20K source manifest JSON")
    gen.add_argument("--overwrite", action="store_true")
    gen.add_argument("--no-git-check", action="store_true")
    gen.set_defaults(func=_cmd_generate)

    ver = sub.add_parser("verify")
    ver.add_argument("--dataset", **common_dataset)
    ver.add_argument("--manifest", type=Path, required=True)
    ver.add_argument("--no-git-check", action="store_true")
    ver.set_defaults(func=_cmd_verify)

    vsb = sub.add_parser("verify-source-binding")
    vsb.add_argument("--dataset", **common_dataset)
    vsb.add_argument("--manifest", type=Path, required=True)
    vsb.add_argument("--data-root", type=str, default=None)
    vsb.add_argument("--no-git-check", action="store_true")
    vsb.set_defaults(func=_cmd_verify_source_binding)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
