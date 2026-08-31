#!/usr/bin/env python3
"""CPU-only standalone CLI for DFP Phase 1A Step 11 deterministic
qualitative reporting.

Every subcommand takes explicit verified input artifact paths or a
synthetic fixture-bundle directory. Never resolves or scans a real
dataset root, never accesses real COCO-Stuff/ADE20K data, never
initializes CUDA, never builds a model, never loads model weights into
torch, and never runs graph/recurrence code.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dfp_phase1a_artifact_schema import DfpPhase1aArtifactSchemaError, parse_strict_json_document, sha256_file, write_json_atomically
from src.dfp_phase1a_qualitative_input import build_example_analysis, load_fixture_bundle, validate_input_record_sequence
from src.dfp_phase1a_qualitative_render import (
    DfpPhase1aQualitativeReportError as RenderError,
    PanelRecord,
    build_bundle_manifest,
    generate_class_palette,
    install_qualitative_bundle,
    palette_digest,
    render_category_overlay_panel,
    render_label_panel,
    render_rgb_panel,
    verify_qualitative_bundle,
    write_panel_png,
)
from src.dfp_phase1a_qualitative_selection import (
    CATEGORY_ORDER,
    DfpPhase1aQualitativeReportError,
    build_selection_manifest,
    select_category_examples,
    validate_selection_manifest_structure,
    validate_static_configuration,
)

_KNOWN_ERRORS = (DfpPhase1aQualitativeReportError, RenderError, DfpPhase1aArtifactSchemaError, OSError, ValueError)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-only DFP Phase 1A Step 11 qualitative-reporting CLI")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    select_cmd = sub.add_parser("select", help="deterministically select qualitative examples from a synthetic fixture bundle")
    select_cmd.add_argument("--fixture-bundle", type=Path, required=True)
    select_cmd.add_argument("--dataset", choices=["coco_stuff", "ade20k"], required=True)
    select_cmd.add_argument("--output", type=Path, required=True)
    select_cmd.add_argument("--source-result-sha256", type=str, default="0" * 64)

    render_cmd = sub.add_parser("render", help="render and atomically install a qualitative bundle from a selection manifest + fixture bundle")
    render_cmd.add_argument("--fixture-bundle", type=Path, required=True)
    render_cmd.add_argument("--selection-manifest", type=Path, required=True)
    render_cmd.add_argument("--dataset", choices=["coco_stuff", "ade20k"], required=True)
    render_cmd.add_argument("--output-dir", type=Path, required=True)

    verify_selection_cmd = sub.add_parser("verify-selection", help="verify a selection manifest's structure and relational hashes")
    verify_selection_cmd.add_argument("--selection-manifest", type=Path, required=True)

    verify_report_cmd = sub.add_parser("verify-report", help="verify an installed qualitative bundle")
    verify_report_cmd.add_argument("--bundle", type=Path, required=True)

    return parser


def _build_contract(repo_root: Path, dataset: str):
    from src.dfp_phase1a_analysis import load_analysis_contract_from_protocol
    from src.dfp_phase1a_identity import load_identity as load_protocol_identity

    protocol = load_protocol_identity(repo_root / "evaluation_identities/e13_dfp_phase1a_protocol.toml", repo_root=repo_root)
    return load_analysis_contract_from_protocol(protocol, dataset_key=dataset, bind_boundary_iou=True, repo_root=repo_root)


def _cmd_select(args: argparse.Namespace) -> int:
    schema_summary = validate_static_configuration(repo_root=args.repo_root)
    identity_sha256 = sha256_file(args.repo_root / "evaluation_identities/e13_dfp_phase1a_qualitative_report.toml")
    contract = _build_contract(args.repo_root, args.dataset)

    records = load_fixture_bundle(args.fixture_bundle)
    image_ids = [r.canonical_image_id for r in records]
    validate_input_record_sequence(records, expected_image_ids=image_ids)

    analyses = {r.canonical_image_id: build_example_analysis(r, contract=contract, boundary_iou_authority=contract.boundary_iou_authority) for r in records}
    dataset_name = "coco_stuff_500" if args.dataset == "coco_stuff" else "ade20k_500"

    category_results = []
    pixel_counts_by_id: dict[str, dict[str, int]] = {}
    for category in CATEGORY_ORDER:
        eligibility = {image_id: analyses[image_id].category_eligibility[category] for image_id in image_ids}
        result = select_category_examples(
            category_name=category, dataset_name=dataset_name, canonical_ids_in_order=image_ids,
            eligibility_by_id=eligibility, k=3, namespace="talk2dino-dfp-phase1a-qualitative-report-v1", separator="\x00",
        )
        category_results.append(result)
        for image_id in result.selected_ids:
            pixel_counts_by_id.setdefault(image_id, {})
            pixel_counts_by_id[image_id][category] = analyses[image_id].category_pixel_counts[category]

    for image_id, counts in pixel_counts_by_id.items():
        for category in CATEGORY_ORDER:
            counts.setdefault(category, analyses[image_id].category_pixel_counts[category])

    class_names_digest_key = "coco_stuff_class_names_digest" if args.dataset == "coco_stuff" else "ade20k_class_names_digest"
    import src.dfp_phase1a_artifact_schema as artifact_schema

    artifact_identity = artifact_schema.load_identity(repo_root=args.repo_root)

    manifest = build_selection_manifest(
        reporting_identity_sha256=identity_sha256, protocol_identity_sha256=schema_summary["protocol_identity_sha256"],
        artifact_schema_identity_sha256=schema_summary["artifact_schema_identity_sha256"],
        subset_manifest_sha256=schema_summary[f"{args.dataset}_manifest_sha256"], selected_ids_digest="0" * 64,
        run_mode=dataset_name, image_count_expected=len(image_ids), class_count=contract.class_count,
        class_names_digest=artifact_identity["dataset_binding"][class_names_digest_key],
        requested_count_per_category=3, category_results=category_results, category_pixel_counts_by_id=pixel_counts_by_id,
        source_commit="0" * 40, source_result_sha256=args.source_result_sha256,
    )
    validate_selection_manifest_structure(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomically(args.output, manifest, label="selection manifest")
    print(f"PASS: DFP Phase 1A qualitative selection written to {args.output}")
    print(f"# trace_digest={manifest['selection_trace_digest']}", file=sys.stderr)
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    from src.dfp_phase1a_identity import load_identity as load_protocol_identity
    import src.dfp_phase1a_qualitative_selection as qsel

    manifest = parse_strict_json_document(args.selection_manifest, label="selection manifest")
    validate_selection_manifest_structure(manifest)
    contract = _build_contract(args.repo_root, args.dataset)
    records = {r.canonical_image_id: r for r in load_fixture_bundle(args.fixture_bundle)}

    palette = generate_class_palette(contract.class_count)
    reporting_identity_sha256 = sha256_file(args.repo_root / "evaluation_identities/e13_dfp_phase1a_qualitative_report.toml")

    panels: list[PanelRecord] = []
    png_bytes: dict[str, bytes] = {}
    staging = args.output_dir.parent / f".render_scratch_{args.output_dir.name}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for category, record_block in manifest["categories"].items():
            for image_id in record_block["selected_ids"]:
                if image_id not in records:
                    raise DfpPhase1aQualitativeReportError(f"selection references image_id {image_id!r} not present in the fixture bundle")
                input_record = records[image_id]
                example = build_example_analysis(input_record, contract=contract, boundary_iou_authority=contract.boundary_iou_authority)

                rgb_panel = render_rgb_panel(input_record.rgb, output_height=256, output_width=256)
                png_path = staging / f"{image_id}_{category}_rgb.png"
                encoded_sha, decoded_sha = write_panel_png(rgb_panel, png_path)
                panels.append(PanelRecord(image_id=image_id, category_name=category, panel_name="rgb", png_filename=png_path.name, png_sha256=encoded_sha, decoded_pixel_sha256=decoded_sha, width=rgb_panel.shape[1], height=rgb_panel.shape[0]))
                png_bytes[png_path.name] = png_path.read_bytes()

                gt_panel = render_label_panel(input_record.gt, palette=palette, ignore_index=contract.ignore_index, ignore_color=(0, 0, 0), output_height=256, output_width=256)
                png_path = staging / f"{image_id}_{category}_gt.png"
                encoded_sha, decoded_sha = write_panel_png(gt_panel, png_path)
                panels.append(PanelRecord(image_id=image_id, category_name=category, panel_name="gt", png_filename=png_path.name, png_sha256=encoded_sha, decoded_pixel_sha256=decoded_sha, width=gt_panel.shape[1], height=gt_panel.shape[0]))
                png_bytes[png_path.name] = png_path.read_bytes()
    finally:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)

    scratch_selection_path = args.output_dir.parent / f".scratch_selection_{args.output_dir.name}.json"
    write_json_atomically(scratch_selection_path, manifest, label="selection manifest")
    selection_manifest_sha256 = sha256_file(scratch_selection_path)
    scratch_selection_path.unlink(missing_ok=True)

    bundle_manifest = build_bundle_manifest(
        reporting_identity_sha256=reporting_identity_sha256, selection_manifest_sha256=selection_manifest_sha256,
        source_result_sha256=manifest["source_result_sha256"], run_mode=manifest["run_mode"], panel_order=["rgb", "gt"],
        palette_digest_value=palette_digest(palette), panels=panels,
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    installed = install_qualitative_bundle(destination_dir=args.output_dir, selection_manifest=manifest, bundle_manifest=bundle_manifest, panel_png_bytes=png_bytes)
    print(f"PASS: DFP Phase 1A qualitative bundle installed at {installed}")
    print(f"# panels={len(panels)}", file=sys.stderr)
    return 0


def _cmd_verify_selection(args: argparse.Namespace) -> int:
    manifest = parse_strict_json_document(args.selection_manifest, label="selection manifest")
    validate_selection_manifest_structure(manifest)
    print("PASS: DFP Phase 1A qualitative selection manifest verified")
    print(f"# run_mode={manifest['run_mode']} trace_digest={manifest['selection_trace_digest']}", file=sys.stderr)
    return 0


def _cmd_verify_report(args: argparse.Namespace) -> int:
    result = verify_qualitative_bundle(args.bundle)
    print("PASS: DFP Phase 1A qualitative bundle verified")
    print(f"# run_mode={result['selection_manifest']['run_mode']} panels={len(result['bundle_manifest']['panels'])}", file=sys.stderr)
    return 0


_DISPATCH = {
    "select": _cmd_select,
    "render": _cmd_render,
    "verify-selection": _cmd_verify_selection,
    "verify-report": _cmd_verify_report,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _DISPATCH[args.command](args)
    except _KNOWN_ERRORS as error:
        print(f"DFP PHASE 1A QUALITATIVE REPORT FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
