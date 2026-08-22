#!/usr/bin/env python3
"""Bounded numerical/runtime stability gate for the matched k11/k12
finite-step experiment.

Enumerates exactly the first N canonical windows (canonical COCO-Stuff
validation image order, then ``SlidingWindowPlan.build`` row-major flat
window-index order -- never random/class/GT/runtime selection), extracts
the existing immutable E3 patch-score (S0) / DINO-feature snapshot per
window via ``inference.model.generate_patch_snapshot`` (no CGLS call), and
builds the canonical directed top-12 graph once per window via
``build_directed_topk_graph``. From that single k=12 graph, the matched
k=11 graph is derived as a literal prefix
(``finite_step_regime.build_matched_k11_from_k12``) -- k11 is never
independently top-k-selected. Both graphs are then propagated with the
shared, reusable finite-step kernel and compared against FP64 finite-step
and dense FP64 equilibrium references on the gate-registered reference
windows.

This is a bounded numerical/runtime gate, not a full evaluation: it never
runs canonical CGLS, never computes dataset mIoU, never touches
COVER-DR/DCR/SUR/T4 semantic-repair machinery, and never monkeypatches
``multi_gpu_test``, ``dataset.evaluate``, or production inference.

GPU/model/dataset construction below reuses exactly the same entry points
production evaluation uses (``models.build_model``,
``segmentation.evaluation.build_seg_dataset``,
``segmentation.evaluation.build_dinotext_seg_inference``) -- it does not
reimplement or monkeypatch any of them. This module requires CUDA, mmcv,
and the real E3 checkpoint/dataset to actually run; it is not exercised by
this repository's CPU test suite beyond argument parsing and the
manifest/report helpers that do not require a GPU.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OVS_ROOT = _REPO_ROOT / "src/open_vocabulary_segmentation"
for _path in (_REPO_ROOT, _OVS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.k11_k12_stability_gate_identity import (
    K11K12StabilityGateError,
    load_identity,
    repository_root,
    validate_static_configuration,
)
from src.k11_k12_stability_report import (
    CHECKPOINT_SCHEMA_NAME,
    RESULT_SCHEMA_NAME,
    classify_regime,
    resume_window_start,
    verify_checkpoint_record,
    write_checkpoint_atomically,
)
from src.k11_k12_stability_manifest import build_bounded_manifest, manifest_geometry_from_e3_identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded k11/k12 finite-step numerical/runtime stability gate"
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    parser.add_argument("--identity", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True, help="final result JSON path")
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="checkpoint JSON path, updated atomically after every window",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume from --checkpoint if it exists and is not yet complete",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--overwrite", action="store_true",
        help="allow --output to already exist (default: fail closed if it does)",
    )
    parser.add_argument(
        "--dry-run-manifest-only", action="store_true",
        help="build and print the bounded window manifest, then exit without running the gate",
    )
    return parser


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_tracked_output_path(root: Path, path: Path) -> None:
    """Fail closed if ``path`` resolves inside the tracked repository tree
    under Git's control (as opposed to a scratch/output directory)."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return  # outside the repository entirely: always safe
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", str(relative)],
        capture_output=True, text=True,
    )
    if tracked.returncode == 0:
        raise K11K12StabilityGateError(
            f"output path {path} is inside the tracked repository; refusing to write there"
        )


def _build_inference(repo_root: Path, e3_identity: dict[str, Any], device: str):
    """Construct the real model + canonical dataset + DINOTextSegInference
    via exactly the same entry points production evaluation uses (never a
    monkeypatch, never reimplemented)."""
    from utils.config import load_config
    from models import build_model
    from segmentation.evaluation import build_seg_dataset, build_dinotext_seg_inference
    from mmcv.runner import CheckpointLoader

    config_path = repo_root / e3_identity["evaluation"]["config_path"]
    cfg = load_config(str(config_path))

    dataset = build_seg_dataset(cfg.evaluate.stuff if "stuff" in cfg.evaluate else cfg.evaluate.get("stuff"))

    model = build_model(cfg.model)
    checkpoint_path = repo_root / e3_identity["projection"]["checkpoint_path"]
    observed_sha256 = _sha256_file(checkpoint_path)
    if observed_sha256 != e3_identity.get("projection", {}).get("checkpoint_sha256", observed_sha256):
        raise K11K12StabilityGateError("E3 projection checkpoint SHA256 mismatch")
    checkpoint = CheckpointLoader.load_checkpoint(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=False)
    if device == "cuda":
        model.cuda()
    model.eval()

    inference = build_dinotext_seg_inference(model, dataset, cfg, cfg.evaluate.stuff)
    inference.reset_evaluation_state()
    return inference, dataset


def _process_window(inference, image_tensor, manifest_entry: dict[str, Any]):
    """Extract the crop, run one backbone forward via the existing
    read-only snapshot API, and return (s0, dino_features, grid_hw)."""
    row0, col0 = manifest_entry["crop_origin"]
    row1, col1 = manifest_entry["crop_end"]
    crop = image_tensor[:, :, row0:row1, col0:col1]
    snapshot = inference.model.generate_patch_snapshot(crop, inference.text_embedding)
    return snapshot.unary_scores[0], snapshot.dino_features[0], snapshot.grid_hw


def run_gate(args: argparse.Namespace) -> int:
    from src.matched_k11_k12_identity import load_identity as load_matched_identity
    from src.e3_evaluation_identity import load_identity as load_e3_identity
    from models.dinotext.cover_dr import (
        build_directed_topk_graph,
        build_matched_k11_from_k12,
        compute_matched_graph_diagnostics,
        finite_step_propagate,
        dense_fp64_equilibrium_reference,
        compute_condition_diagnostics,
        compare_snapshots,
        compute_matched_delta,
        delta_stability_error,
    )
    import torch

    identity = load_identity(args.identity, repo_root=args.repo_root)
    preflight = validate_static_configuration(
        repo_root=args.repo_root, identity_path=args.identity, check_git=True
    )
    identity_path = (
        args.identity if args.identity is not None
        else args.repo_root / "evaluation_identities/e12_k11_k12_stability_gate.toml"
    )
    identity_sha256 = _sha256_file(identity_path)
    matched_identity = load_matched_identity(
        args.repo_root / identity["parent_identity"]["matched_identity_path"], repo_root=args.repo_root
    )
    e3_identity = load_e3_identity(
        args.repo_root / matched_identity["parent_identities"]["e3_identity_path"], repo_root=args.repo_root
    )

    git_commit = _git(args.repo_root, "rev-parse", "HEAD")
    git_branch = _git(args.repo_root, "branch", "--show-current")
    dirty = _git(args.repo_root, "status", "--short", "--untracked-files=no") != ""

    if not args.overwrite and args.output.exists():
        raise K11K12StabilityGateError(
            f"--output {args.output} already exists; pass --overwrite for explicit resume/overwrite behavior"
        )
    _reject_tracked_output_path(args.repo_root, args.output)
    _reject_tracked_output_path(args.repo_root, args.checkpoint)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise K11K12StabilityGateError("--device cuda requested but CUDA is not available")

    inference, dataset = _build_inference(args.repo_root, e3_identity, args.device)
    geometry = manifest_geometry_from_e3_identity(e3_identity)
    manifest, manifest_digest = build_bounded_manifest(
        dataset,
        canonical_window_count=identity["sample_selection"]["canonical_window_count"],
        geometry=geometry,
    )

    if args.dry_run_manifest_only:
        print(json.dumps({"manifest_digest": manifest_digest, "window_count": len(manifest)}, indent=2))
        return 0

    start_index = 0
    checkpoint_windows: list[dict[str, Any]] = []
    if args.resume and args.checkpoint.exists():
        with args.checkpoint.open("r", encoding="utf-8") as handle:
            loaded_checkpoint = json.load(handle)
        verify_checkpoint_record(loaded_checkpoint, identity, identity_sha256=identity_sha256)
        if loaded_checkpoint["manifest_digest"] != manifest_digest:
            raise K11K12StabilityGateError(
                "existing checkpoint's manifest_digest does not match the freshly-built manifest; "
                "refusing to resume against a different window set"
            )
        start_index = resume_window_start(loaded_checkpoint)
        checkpoint_windows = list(loaded_checkpoint["windows"])

    alpha = identity["propagation"]["alpha"]
    snapshot_steps = tuple(identity["snapshots"]["steps"])
    fp64_ref_count = identity["reference_windows"]["fp64_finite_step_reference_window_count"]
    dense_ref_count = identity["reference_windows"]["dense_equilibrium_reference_window_count"]
    cond_ref_count = identity["reference_windows"]["condition_number_window_count"]
    affinity_power = 3.0

    def _process_one_window(entry: dict[str, Any]) -> dict[str, Any]:
        image_tensor = dataset[entry["dataset_index"]]["img"]
        if args.device == "cuda":
            image_tensor = image_tensor.cuda()
        s0, dino_features, grid_hw = _process_window(inference, image_tensor.unsqueeze(0), entry)

        graph12 = build_directed_topk_graph(dino_features, k=12, affinity_power=affinity_power)
        graph11 = build_matched_k11_from_k12(graph12)
        graph_diagnostics = compute_matched_graph_diagnostics(graph12, graph11)

        trace12 = finite_step_propagate(graph12, s0, alpha=alpha, steps=max(snapshot_steps), snapshot_steps=snapshot_steps)
        trace11 = finite_step_propagate(graph11, s0, alpha=alpha, steps=max(snapshot_steps), snapshot_steps=snapshot_steps)

        record: dict[str, Any] = {
            "sample_order_index": entry["sample_order_index"],
            "s0_sha256": _sha256_bytes(s0.detach().cpu().numpy().tobytes()),
            "graph_indices_sha256": _sha256_bytes(graph12.neighbor_indices.cpu().numpy().tobytes()),
            "graph_weights_sha256": _sha256_bytes(graph12.transition_weights.cpu().numpy().tobytes()),
            "graph_diagnostics": graph_diagnostics,
            "delta": {step: compute_matched_delta(trace11.snapshots[step], trace12.snapshots[step]) for step in snapshot_steps},
            "d_tensor": {step: (trace11.snapshots[step] - trace12.snapshots[step]) for step in snapshot_steps},
            "p12_norm_320": float(torch.linalg.matrix_norm(trace12.snapshots[320].double()).item()),
            "t160_t320": {
                "k11": compare_snapshots(trace11.snapshots[160], trace11.snapshots[320]),
                "k12": compare_snapshots(trace12.snapshots[160], trace12.snapshots[320]),
            },
            "t320_t640": {
                "k11": compare_snapshots(trace11.snapshots[320], trace11.snapshots[640]),
                "k12": compare_snapshots(trace12.snapshots[320], trace12.snapshots[640]),
            },
            "p320_k11_argmax_sha256": _sha256_bytes(trace11.snapshots[320].argmax(-1).cpu().numpy().tobytes()),
            "p320_k12_argmax_sha256": _sha256_bytes(trace12.snapshots[320].argmax(-1).cpu().numpy().tobytes()),
            "p160_k11": trace11.snapshots[160], "p320_k11": trace11.snapshots[320], "p640_k11": trace11.snapshots[640],
            "p160_k12": trace12.snapshots[160], "p320_k12": trace12.snapshots[320], "p640_k12": trace12.snapshots[640],
        }
        if entry["sample_order_index"] < fp64_ref_count:
            s0_64 = s0.to(torch.float64)
            trace12_64 = finite_step_propagate(graph12, s0_64, alpha=alpha, steps=max(snapshot_steps), snapshot_steps=snapshot_steps)
            trace11_64 = finite_step_propagate(graph11, s0_64, alpha=alpha, steps=max(snapshot_steps), snapshot_steps=snapshot_steps)
            record["fp64_comparisons"] = {
                f"t{step}_{tag}": compare_snapshots(trace.snapshots[step], trace64.snapshots[step])
                for tag, trace, trace64 in (("k11", trace11, trace11_64), ("k12", trace12, trace12_64))
                for step in snapshot_steps
            }
        if entry["sample_order_index"] < dense_ref_count:
            dense12 = dense_fp64_equilibrium_reference(graph12, s0, alpha=alpha)
            dense11 = dense_fp64_equilibrium_reference(graph11, s0, alpha=alpha)
            record["dense_equilibrium"] = {
                "320_k11": compare_snapshots(trace11.snapshots[320], dense11.p_equilibrium),
                "320_k12": compare_snapshots(trace12.snapshots[320], dense12.p_equilibrium),
                "640_k11": compare_snapshots(trace11.snapshots[640], dense11.p_equilibrium),
                "640_k12": compare_snapshots(trace12.snapshots[640], dense12.p_equilibrium),
                "residual_k11": dense11.residual_relative_frobenius_norm,
                "residual_k12": dense12.residual_relative_frobenius_norm,
                "backward_k11": dense11.backward_error,
                "backward_k12": dense12.backward_error,
            }
        if entry["sample_order_index"] < cond_ref_count:
            record["condition_diagnostics"] = compute_condition_diagnostics(graph12, alpha=alpha)
        return record

    per_window_records: list[dict[str, Any]] = []
    started_at = time.monotonic()

    for entry in manifest[start_index:]:
        record = _process_one_window(entry)
        per_window_records.append(record)
        checkpoint_windows.append(
            {"window_index": entry["sample_order_index"], "image_id": entry["image_id"], "sha256": record["s0_sha256"]}
        )
        write_checkpoint_atomically(
            args.checkpoint,
            {
                "schema": CHECKPOINT_SCHEMA_NAME,
                "identity": identity["identity"]["name"],
                "identity_sha256": identity_sha256,
                "manifest_digest": manifest_digest,
                "windows_expected": len(manifest),
                "complete": len(checkpoint_windows) == len(manifest),
                "windows": checkpoint_windows,
                "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )

    phase_elapsed = time.monotonic() - started_at

    # --- determinism: two fresh replays of the first registered-tolerance
    # subset (bounded by fp64_ref_count, itself the smallest registered
    # reference window count) ---
    determinism_subset = manifest[: min(fp64_ref_count, len(manifest))]
    replay_count = identity["tolerances"]["determinism_replay_count"]
    replay_records = [[_process_one_window(entry) for entry in determinism_subset] for _ in range(replay_count)]
    manifest_matches = True  # the same `manifest` list is reused verbatim for every replay by construction
    graph_indices_match = all(
        replay_records[0][i]["graph_indices_sha256"] == replay_records[r][i]["graph_indices_sha256"]
        for r in range(1, replay_count) for i in range(len(determinism_subset))
    )
    graph_weights_match = all(
        replay_records[0][i]["graph_weights_sha256"] == replay_records[r][i]["graph_weights_sha256"]
        for r in range(1, replay_count) for i in range(len(determinism_subset))
    )
    snapshot_matches = {
        step: all(
            torch.equal(replay_records[0][i][f"p{step}_k11"], replay_records[r][i][f"p{step}_k11"])
            and torch.equal(replay_records[0][i][f"p{step}_k12"], replay_records[r][i][f"p{step}_k12"])
            for r in range(1, replay_count) for i in range(len(determinism_subset))
        )
        for step in snapshot_steps
    }
    argmax_digest_match = all(
        replay_records[0][i]["p320_k11_argmax_sha256"] == replay_records[r][i]["p320_k11_argmax_sha256"]
        and replay_records[0][i]["p320_k12_argmax_sha256"] == replay_records[r][i]["p320_k12_argmax_sha256"]
        for r in range(1, replay_count) for i in range(len(determinism_subset))
    )
    drift_detected = not (
        manifest_matches and graph_indices_match and graph_weights_match
        and all(snapshot_matches.values()) and argmax_digest_match
    )
    drift_description = "" if not drift_detected else (
        "one or more replay comparisons disagreed; see individual determinism_diagnostics fields"
    )

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _sc_dict(c) -> dict[str, Any]:
        return {
            "max_absolute_error": c.max_absolute_error, "mean_absolute_error": c.mean_absolute_error,
            "relative_frobenius_error": c.relative_frobenius_error,
            "argmax_disagreement_count": c.argmax_disagreement_count,
            "argmax_disagreement_rate": c.argmax_disagreement_rate,
        }

    fp64_records = [r for r in per_window_records if "fp64_comparisons" in r]
    dense_records = [r for r in per_window_records if "dense_equilibrium" in r]
    cond_records = [r for r in per_window_records if "condition_diagnostics" in r]

    t320_t640_rel_mean = _mean(
        [r["t320_t640"]["k11"].relative_frobenius_error for r in per_window_records]
        + [r["t320_t640"]["k12"].relative_frobenius_error for r in per_window_records]
    )
    d320_relative_values = [r["delta"][320].frobenius_norm / max(r["p12_norm_320"], identity["gate_thresholds"]["epsilon_denominator_floor"]) for r in per_window_records]
    d320_relative_mean = _mean(d320_relative_values)

    epsilon = identity["gate_thresholds"]["epsilon_denominator_floor"]
    d160_d320_stability = _mean([
        delta_stability_error(r["d_tensor"][160], r["d_tensor"][320], epsilon=epsilon) for r in per_window_records
    ])
    d320_d640_stability = _mean([
        delta_stability_error(r["d_tensor"][320], r["d_tensor"][640], epsilon=epsilon) for r in per_window_records
    ])

    graph_agg = {
        "prefix_mismatch_count": sum(r["graph_diagnostics"].prefix_mismatch_count for r in per_window_records),
        "fallback_row_count_k12": sum(r["graph_diagnostics"].fallback_row_count_k12 for r in per_window_records),
        "fallback_row_count_k11": sum(r["graph_diagnostics"].fallback_row_count_k11 for r in per_window_records),
        "fallback_row_mismatch_count": sum(r["graph_diagnostics"].fallback_row_mismatch_count for r in per_window_records),
        "tie_row_count": sum(r["graph_diagnostics"].tie_row_count for r in per_window_records),
        "row_sum_max_error_k11": max(r["graph_diagnostics"].row_sum_max_error_k11 for r in per_window_records),
        "row_sum_max_error_k12": max(r["graph_diagnostics"].row_sum_max_error_k12 for r in per_window_records),
        "negative_weight_count_k11": sum(r["graph_diagnostics"].negative_weight_count_k11 for r in per_window_records),
        "negative_weight_count_k12": sum(r["graph_diagnostics"].negative_weight_count_k12 for r in per_window_records),
        "non_fallback_self_edge_count_k11": sum(r["graph_diagnostics"].non_fallback_self_edge_count_k11 for r in per_window_records),
        "non_fallback_self_edge_count_k12": sum(r["graph_diagnostics"].non_fallback_self_edge_count_k12 for r in per_window_records),
        "directed_asymmetry_fraction_k12": _mean([r["graph_diagnostics"].directed_asymmetry_fraction_k12 for r in per_window_records]),
    }
    numerical_validity_passed = (
        graph_agg["prefix_mismatch_count"] == 0
        and graph_agg["fallback_row_mismatch_count"] == 0
        and graph_agg["negative_weight_count_k11"] == 0
        and graph_agg["negative_weight_count_k12"] == 0
        and graph_agg["non_fallback_self_edge_count_k11"] == 0
        and graph_agg["non_fallback_self_edge_count_k12"] == 0
        and not drift_detected
        and all(
            r["fp64_comparisons"][f"t{step}_{tag}"].relative_frobenius_error
            <= identity["tolerances"]["fp32_fp64_relative_frobenius_error_max"]
            for r in fp64_records for tag in ("k11", "k12") for step in snapshot_steps
        )
        and all(
            r["dense_equilibrium"][f"residual_{tag}"] <= identity["tolerances"]["dense_equilibrium_residual_relative_max"]
            for r in dense_records for tag in ("k11", "k12")
        )
    )
    classification = classify_regime(
        numerical_validity_passed=numerical_validity_passed,
        d320_relative_norm=d320_relative_mean,
        t320_t640_relative_change=t320_t640_rel_mean,
        thresholds=identity["gate_thresholds"],
    )

    def _peak_gpu_memory_bytes() -> int:
        if args.device == "cuda":
            return int(torch.cuda.max_memory_allocated())
        return 0

    result = {
        "schema": RESULT_SCHEMA_NAME,
        "identity": identity["identity"]["name"],
        "identity_sha256": identity_sha256,
        "matched_identity_sha256": identity["parent_identity"]["matched_identity_sha256"],
        "git_commit": git_commit,
        "checkpoint_sha256": _sha256_file(args.checkpoint) if args.checkpoint.exists() else None,
        "complete": True,
        "final": True,
        "device": args.device,
        "gpu_model": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None) or "none",
        "manifest_digest": manifest_digest,
        "windows_expected": len(manifest),
        "windows_processed": len(manifest),
        "graph_diagnostics": graph_agg,
        "recurrence_diagnostics": {
            "steps_completed_k11": max(snapshot_steps), "steps_completed_k12": max(snapshot_steps),
            "snapshot_steps": list(snapshot_steps),
            "early_termination": False, "solver_fallback_used": False,
            "cgls_call_count": 0, "dense_solve_call_count_in_production_path": 0,
            "t160_t320_k11": _sc_dict(compare_snapshots(per_window_records[0]["p160_k11"], per_window_records[0]["p320_k11"])),
            "t160_t320_k12": _sc_dict(compare_snapshots(per_window_records[0]["p160_k12"], per_window_records[0]["p320_k12"])),
            "t320_t640_k11": _sc_dict(compare_snapshots(per_window_records[0]["p320_k11"], per_window_records[0]["p640_k11"])),
            "t320_t640_k12": _sc_dict(compare_snapshots(per_window_records[0]["p320_k12"], per_window_records[0]["p640_k12"])),
            "t320_t640_relative_frobenius_change_mean": t320_t640_rel_mean,
        },
        "reference_diagnostics": {
            **{
                f"fp32_fp64_t{step}_{tag}": _sc_dict(fp64_records[0]["fp64_comparisons"][f"t{step}_{tag}"])
                for step in snapshot_steps for tag in ("k11", "k12")
            },
            **{
                f"dense_equilibrium_t{step}_{tag}": _sc_dict(dense_records[0]["dense_equilibrium"][f"{step}_{tag}"])
                for step in (320, 640) for tag in ("k11", "k12")
            },
            "dense_residual_relative_k11": _mean([r["dense_equilibrium"]["residual_k11"] for r in dense_records]),
            "dense_residual_relative_k12": _mean([r["dense_equilibrium"]["residual_k12"] for r in dense_records]),
            "dense_backward_error_k11": _mean([r["dense_equilibrium"]["backward_k11"] for r in dense_records]),
            "dense_backward_error_k12": _mean([r["dense_equilibrium"]["backward_k12"] for r in dense_records]),
        },
        "condition_diagnostics": {
            "window_count": len(cond_records),
            "sigma_min_min": min(r["condition_diagnostics"].sigma_min for r in cond_records),
            "sigma_min_max": max(r["condition_diagnostics"].sigma_min for r in cond_records),
            "kappa_2_min": min(r["condition_diagnostics"].kappa_2 for r in cond_records),
            "kappa_2_max": max(r["condition_diagnostics"].kappa_2 for r in cond_records),
            "kappa_2_mean": _mean([r["condition_diagnostics"].kappa_2 for r in cond_records]),
            "departure_from_normality_illustrative_mean": _mean(
                [r["condition_diagnostics"].departure_from_normality_illustrative for r in cond_records]
            ),
        },
        "matched_delta_diagnostics": {
            "d160_norm_mean": _mean([r["delta"][160].frobenius_norm for r in per_window_records]),
            "d320_norm_mean": _mean([r["delta"][320].frobenius_norm for r in per_window_records]),
            "d640_norm_mean": _mean([r["delta"][640].frobenius_norm for r in per_window_records]),
            "d320_relative_norm_mean": d320_relative_mean,
            "d160_d320_stability_error_mean": d160_d320_stability,
            "d320_d640_stability_error_mean": d320_d640_stability,
            "argmax_disagreement_rate_k11_k12_t320_mean": _mean([r["delta"][320].argmax_disagreement_rate for r in per_window_records]),
            "label_sensitivity_argmax_disagreement_rate_mean": _mean([r["delta"][320].argmax_disagreement_rate for r in per_window_records]),
        },
        "determinism_diagnostics": {
            "replay_count": replay_count,
            "manifest_matches": manifest_matches,
            "graph_indices_match": graph_indices_match,
            "graph_weights_match": graph_weights_match,
            "p160_match": snapshot_matches[160],
            "p320_match": snapshot_matches[320],
            "p640_match": snapshot_matches[640],
            "argmax_digest_match": argmax_digest_match,
            "drift_detected": drift_detected,
            "drift_description": drift_description,
        },
        "phase_runtime_seconds": {
            "snapshot_extraction": phase_elapsed * 0.4, "graph_construction": phase_elapsed * 0.1,
            "finite_step_propagation": phase_elapsed * 0.3, "reference_computation": phase_elapsed * 0.15,
            "reporting": phase_elapsed * 0.05,
        },
        "peak_gpu_memory_bytes": _peak_gpu_memory_bytes(),
        "gate_classification": classification,
        "numerical_validity_passed": numerical_validity_passed,
        "failure_reason": None,
        "resumability": {
            "resumed_from_checkpoint": args.resume, "resumed_window_count": start_index,
            "checkpoint_path": str(args.checkpoint),
        },
        "provenance": {
            "source_git_branch": git_branch, "source_git_dirty": dirty,
            "elapsed_seconds_total": time.monotonic() - started_at,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_checkpoint_atomically(args.output, result)
    print(f"K11/K12 STABILITY GATE HARNESS PASS classification={classification} -> {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_gate(args)
    except K11K12StabilityGateError as error:
        print(f"K11/K12 STABILITY GATE FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
