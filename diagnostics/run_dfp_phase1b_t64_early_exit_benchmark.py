#!/usr/bin/env python3
"""CLI for the E26-authorized T64-versus-T320 early-exit wall-clock
benchmark, whose scientific/operational contract is frozen, unmodified,
by E25.

Subcommands
-----------
verify-protocol   Verify E25 + E25-artifacts + E26 (closed schema and
                   relational SHA-256 hashes). No torch.
preflight          Full repository-readiness check: identity + manifest
                   + git ancestry/clean-worktree. No torch, no CUDA, no
                   dataset access. Prints ``PREFLIGHT PASS -- no
                   benchmark execution has occurred``.
run-trial          Run exactly one fresh-process trial (a warm-up trial
                   with ``--warmup``, or one of the six scheduled
                   measured trials). Requires ``--device cuda``. The
                   only subcommand that imports torch/model/dataset
                   code, and only after preflight has already passed.
verify-trial       Verify one trial artifact's closed schema and
                   relational bindings. No torch.
merge              Merge exactly the 2 warm-up + 6 measured trial
                   artifacts and install the merged result plus its
                   decision as one pair-atomic bundle directory
                   (``merged_result.json`` + ``decision.json``, one
                   atomic directory rename -- never a result without
                   its decision or vice versa). No torch -- pure CPU
                   aggregation/verification.
verify-result      Independently re-verify an installed merged result
                   by recomputing the entire merge from its own
                   embedded raw trials. No torch.
verify-decision    Independently re-verify a decision artifact against
                   its bound merged result. No torch.

Every subcommand except ``run-trial`` performs no torch import, no CUDA
initialization, and no real dataset/model access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OVS_ROOT = _REPO_ROOT / "src/open_vocabulary_segmentation"
for _path in (_REPO_ROOT, _OVS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.dfp_phase1b_t64_early_exit_benchmark import (
    COUNTERBALANCED_ORDER,
    RESULT_BUNDLE_DECISION_FILENAME,
    RESULT_BUNDLE_RESULT_FILENAME,
    DfpPhase1bT64EarlyExitBenchmarkError,
    MemoryProvider,
    TrialProviders,
    WindowInputs,
    build_decision_document,
    install_artifact_atomically,
    install_result_decision_bundle_atomically,
    merge_trials,
    perform_repository_readiness_check,
    run_trial,
    verify_decision_document,
    verify_protocol,
    verify_result_document,
    verify_trial_document,
)
from src.dfp_phase1b_t64_early_exit_benchmark_identity import DfpPhase1bT64EarlyExitBenchmarkIdentityError
from src.dfp_phase1b_t64_early_exit_benchmark_implementation_identity import DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError

_KNOWN_ERRORS = (
    DfpPhase1bT64EarlyExitBenchmarkError, DfpPhase1bT64EarlyExitBenchmarkIdentityError,
    DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError, OSError, ValueError, KeyError,
)


def _print_banner(command: str) -> None:
    print(f"DFP Phase 1B T64 early-exit benchmark [{command}] -- OPERATIONAL WALL-CLOCK MEASUREMENT ONLY.")
    print("  - Never accesses GT. Never runs confirmation or the full 5,000-image evaluation.")
    print("  - A PROCEED outcome only authorizes freezing a separate T64 confirmation protocol.")


def _reject_lexical_path_traversal(path: Path) -> None:
    for part in Path(path).parts:
        if part == "..":
            raise DfpPhase1bT64EarlyExitBenchmarkError(f"path {path} contains a lexical '..' component -- refused")


def _cmd_verify_protocol(args: argparse.Namespace) -> int:
    _print_banner("verify-protocol")
    verify_protocol(Path(args.repo_root))
    print("E25 + E25-artifacts + E26 identities verified: closed schema + relational hashes all pass.")
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    _print_banner("preflight")
    perform_repository_readiness_check(
        repo_root=Path(args.repo_root), output_path=Path(args.output) if args.output else None,
    )
    print("E25/E26 identities verified.")
    print("Benchmark manifest verified.")
    print("Repository ancestry and clean-worktree verified.")
    print("PREFLIGHT PASS -- no benchmark execution has occurred.")
    return 0


def _real_window_inputs_builder_factory(resources):
    """Build the REAL per-window input provider: slices the manifest
    record's own top/left/bottom/right crop out of the real dataset's
    prepared full-canvas image tensor, rescaling final output to the
    real image's own ``ori_shape``. Runs no model, no recurrence --
    only dataset decode/crop, exactly like every other production
    window preparer in this repository."""
    from diagnostics.run_k11_k12_stability import _extract_prepared_image
    from src.dfp_phase1a_dataset_view import verify_position_identity

    positions_by_id = {p.canonical_id: p for p in resources.execution_view.positions}
    dataset = resources.dataset
    _prepared_cache: dict[str, object] = {}

    def _prepared_for(image_id: str):
        if image_id in _prepared_cache:
            return _prepared_cache[image_id]
        position = positions_by_id.get(image_id)
        if position is None:
            raise DfpPhase1bT64EarlyExitBenchmarkError(f"image_id {image_id!r} has no resolved native-dataset position")
        verify_position_identity(dataset, manifest_position=position.manifest_position, canonical_id=position.canonical_id, native_index=position.native_index)
        raw = _extract_prepared_image(dataset, position.native_index, canonical_image_id=position.relative_path)
        _prepared_cache[image_id] = raw
        return raw

    def _build(record) -> WindowInputs:
        raw = _prepared_for(record["image_id"])
        image_tensor = raw.image_tensor.unsqueeze(0)
        top, left, bottom, right = record["top"], record["left"], record["bottom"], record["right"]
        crop = image_tensor[:, :, top:bottom, left:right]
        ori_shape = tuple(int(v) for v in raw.img_metas["ori_shape"][:2])
        return WindowInputs(crop_tensor=crop, crop_height=int(bottom - top), crop_width=int(right - left), ori_shape=ori_shape)

    return _build


def _build_real_providers(*, log_dir: Path) -> TrialProviders:
    from src.dfp_phase1b_calibration_production import build_production_dataset_resources, build_production_inference

    def dataset_resources_builder(repo_root: Path, selected_ids):
        return build_production_dataset_resources(repo_root=repo_root, selected_ids=selected_ids)

    def inference_builder(resources):
        return build_production_inference(resources, device="cuda", log_dir=log_dir)

    return TrialProviders(
        dataset_resources_builder=dataset_resources_builder,
        inference_builder=inference_builder,
        window_inputs_builder_factory=_real_window_inputs_builder_factory,
        stitching_geometry_factory=lambda resources: resources.contracts.stitching,
        final_resolution_contract_factory=lambda resources: resources.contracts.final_resolution,
        class_count_of=lambda resources: resources.class_count,
        affinity_power_of=lambda resources: resources.affinity_power,
        alpha_of=lambda resources: resources.contracts.recurrence.alpha,
        k_of=lambda resources: 12,
    )


def _cmd_run_trial(args: argparse.Namespace) -> int:
    _print_banner("run-trial")
    if args.device != "cuda":
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"--device {args.device!r} is not authorized -- run-trial always requires --device cuda")
    if args.warmup:
        if args.variant is None:
            raise DfpPhase1bT64EarlyExitBenchmarkError("--warmup requires --variant")
        if args.repeat != 0:
            raise DfpPhase1bT64EarlyExitBenchmarkError("--warmup requires --repeat 0")
    else:
        entry = {"trial_index": args.trial_index, "variant": args.variant, "repeat": args.repeat}
        if entry not in COUNTERBALANCED_ORDER:
            raise DfpPhase1bT64EarlyExitBenchmarkError(f"(--trial-index {args.trial_index}, --variant {args.variant}, --repeat {args.repeat}) is not in E25's frozen counterbalanced schedule")

    repo_root = Path(args.repo_root)
    output_path = Path(args.output)
    readiness = perform_repository_readiness_check(repo_root=repo_root, output_path=output_path)
    print("Preflight PASS -- beginning real trial execution.")

    import torch

    if torch.cuda.device_count() != 1:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"run-trial requires exactly one visible CUDA device, observed {torch.cuda.device_count()}")
    observed_gpu_name = torch.cuda.get_device_name(0)
    if observed_gpu_name != "NVIDIA H100 80GB HBM3":
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"run-trial requires GPU model 'NVIDIA H100 80GB HBM3', observed {observed_gpu_name!r}")

    manifest = readiness.manifest
    benchmark_records = manifest["benchmark_records"]
    warmup_record = manifest["warmup_record"]
    selected_ids = sorted({warmup_record["image_id"]} | {r["image_id"] for r in benchmark_records})

    providers = _build_real_providers(log_dir=output_path.parent)
    document = run_trial(
        variant_id=args.variant, trial_index=args.trial_index if not args.warmup else 0, repeat_index=args.repeat,
        is_warmup=bool(args.warmup), manifest=manifest, readiness=readiness, selected_ids=selected_ids, providers=providers,
    )

    def _verify(doc):
        verify_trial_document(doc, artifacts_doc=readiness.artifacts_doc, expected_identity_sha256=readiness.identity_sha256, expected_manifest_sha256=readiness.manifest_sha256)

    install_artifact_atomically(document, output_path, verify_fn=_verify)
    print(f"Trial artifact written to {output_path}")
    print(f"measured_window_loop_wall_ns={document['timing']['measured_window_loop_wall_ns']}")
    print(f"scientific_output_collection_digest={document['scientific_output']['scientific_output_collection_digest']}")
    return 0


def _cmd_verify_trial(args: argparse.Namespace) -> int:
    _print_banner("verify-trial")
    with open(args.trial_file, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    _, _, artifacts_doc = verify_protocol(Path(args.repo_root))
    verify_trial_document(document, artifacts_doc=artifacts_doc)
    print(f"Trial artifact {args.trial_file} PASSES independent verification.")
    print(f"(variant_id={document['variant_id']}, trial_index={document['trial_index']}, repeat_index={document['repeat_index']}, is_warmup={document['is_warmup']})")
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    """Merges the trial artifacts and installs the merged result and its
    decision as one pair-atomic bundle directory (``--output-dir``,
    containing exactly ``merged_result.json`` and ``decision.json``):
    after a single atomic directory rename, the destination is either
    completely absent or a complete, cross-verified pair -- never a
    result without its decision or vice versa. No overwrite, no force
    mode: a pre-existing bundle directory is always rejected."""
    _print_banner("merge")
    bundle_dir = Path(args.output_dir)
    _reject_lexical_path_traversal(bundle_dir)
    if bundle_dir.exists():
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"refusing to overwrite existing result/decision bundle {bundle_dir} (no overwrite, no force mode authorized)")

    _, _, artifacts_doc = verify_protocol(Path(args.repo_root))
    measured_trials = []
    for path in args.trial:
        with open(path, "r", encoding="utf-8") as handle:
            measured_trials.append(json.load(handle))
    warmup_trials = []
    for path in args.warmup_trial:
        with open(path, "r", encoding="utf-8") as handle:
            warmup_trials.append(json.load(handle))

    merged = merge_trials(measured_trials=measured_trials, warmup_trials=warmup_trials, artifacts_doc=artifacts_doc)

    def _build_decision(merged_result_sha256: str):
        return build_decision_document(artifacts_doc=artifacts_doc, merged_result=merged, merged_result_sha256=merged_result_sha256)

    install_result_decision_bundle_atomically(merged_result=merged, build_decision=_build_decision, bundle_dir=bundle_dir, artifacts_doc=artifacts_doc)
    print(f"Result/decision bundle written to {bundle_dir}")
    print(f"  {RESULT_BUNDLE_RESULT_FILENAME}")
    print(f"  {RESULT_BUNDLE_DECISION_FILENAME}")
    print(f"outcome={merged['outcome']}")
    print(f"t64_speedup_over_t320={merged['speedup']['t64_speedup_over_t320']}")
    return 0


def _cmd_verify_result(args: argparse.Namespace) -> int:
    _print_banner("verify-result")
    with open(args.result, "r", encoding="utf-8") as handle:
        merged = json.load(handle)
    _, _, artifacts_doc = verify_protocol(Path(args.repo_root))
    verify_result_document(merged, artifacts_doc=artifacts_doc)
    print(f"Merged result {args.result} PASSES independent verification.")
    print(f"outcome={merged['outcome']}")
    return 0


def _cmd_verify_decision(args: argparse.Namespace) -> int:
    _print_banner("verify-decision")
    with open(args.decision, "r", encoding="utf-8") as handle:
        decision = json.load(handle)
    _, _, artifacts_doc = verify_protocol(Path(args.repo_root))
    merged_result = None
    merged_result_sha256 = None
    if args.result:
        with open(args.result, "r", encoding="utf-8") as handle:
            merged_result = json.load(handle)
        verify_result_document(merged_result, artifacts_doc=artifacts_doc)
        merged_result_sha256 = hashlib.sha256(Path(args.result).read_bytes()).hexdigest()
    verify_decision_document(decision, artifacts_doc=artifacts_doc, merged_result=merged_result, merged_result_sha256=merged_result_sha256)
    print(f"Decision artifact {args.decision} PASSES independent verification.")
    print(f"outcome={decision['outcome']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DFP Phase 1B T64-versus-T320 early-exit benchmark CLI (operational wall-clock measurement only)")
    parser.add_argument("--repo-root", type=str, default=str(_REPO_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("verify-protocol", help="verify E25 + E25-artifacts + E26 identities")

    p = sub.add_parser("preflight", help="full repository-readiness check; no GPU/dataset/model access")
    p.add_argument("--output", type=str, default=None, help="optional -- also validate this future run-trial/merge output destination is absent")

    p = sub.add_parser("run-trial", help="run exactly one fresh-process trial (requires --device cuda)")
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--variant", choices=("T64_early_exit", "T320_reference"), required=True)
    p.add_argument("--trial-index", type=int, default=0)
    p.add_argument("--repeat", type=int, required=True)
    p.add_argument("--warmup", action="store_true", help="run a standalone warm-up trial instead of a scheduled measured trial")
    p.add_argument("--output", type=str, required=True)

    p = sub.add_parser("verify-trial", help="verify one trial artifact's closed schema and relational bindings")
    p.add_argument("--trial-file", type=str, required=True)

    p = sub.add_parser(
        "merge",
        help="merge the 2 warm-up + 6 measured trial artifacts and install the merged result + decision as one pair-atomic bundle directory",
    )
    p.add_argument("--trial", type=str, action="append", required=True, help="repeat once per measured trial file (exactly six required)")
    p.add_argument("--warmup-trial", type=str, action="append", required=True, help="repeat once per warm-up trial file (exactly two required)")
    p.add_argument("--output-dir", type=str, required=True, help="fresh (must not yet exist) bundle directory that will contain merged_result.json and decision.json")

    p = sub.add_parser("verify-result", help="independently re-verify an installed merged result")
    p.add_argument("--result", type=str, required=True)

    p = sub.add_parser("verify-decision", help="independently re-verify a decision artifact")
    p.add_argument("--decision", type=str, required=True)
    p.add_argument("--result", type=str, default=None, help="optional -- also cross-verify against the bound merged result")

    return parser


_DISPATCH = {
    "verify-protocol": _cmd_verify_protocol,
    "preflight": _cmd_preflight,
    "run-trial": _cmd_run_trial,
    "verify-trial": _cmd_verify_trial,
    "merge": _cmd_merge,
    "verify-result": _cmd_verify_result,
    "verify-decision": _cmd_verify_decision,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _DISPATCH[args.command](args)
    except _KNOWN_ERRORS as error:
        print(f"DFP PHASE 1B T64 EARLY-EXIT BENCHMARK FAIL [{args.command}]: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
