#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import torch


REPOSITORY = Path("/project/6114407/haree/Talk2DINO")
OPEN_VOCABULARY_ROOT = REPOSITORY / "src/open_vocabulary_segmentation"
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(OPEN_VOCABULARY_ROOT))

from models.dinotext.cover_dr import inference as inference_module
from models.dinotext.cover_dr.graph import build_directed_topk_graph
from models.dinotext.cover_dr.rwr import RWRSolverError
from segmentation.evaluation import dinotext_seg as segmentation_module


CAPTURE_DIRECTORY = Path(os.environ["TALK2DINO_RWR_CAPTURE_DIR"])
HISTORICAL_MANIFEST = Path(
    "/scratch/haree/talk2dino_e3_affinity_oracle/cache/full/manifest.json"
)
_captured = False
_production_apply = inference_module.apply_rwr_to_e3_snapshot


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().cpu().numpy().tobytes(order="C")


def _tensor_record(value: torch.Tensor) -> dict[str, object]:
    payload = _tensor_bytes(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "sha256": _sha256_bytes(payload),
        "minimum": float(value.min().item()),
        "maximum": float(value.max().item()),
        "finite": bool(torch.isfinite(value).all().item()),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failure_record(error: RWRSolverError) -> dict[str, object]:
    fields = (
        "method",
        "iteration",
        "abs_primal_residual_inf",
        "max_scaled_primal_residual",
        "rtol",
        "atol",
        "reason",
        "failing_rhs",
        "active_rhs",
        "stage",
        "tensor",
        "detail",
        "breakdown_value",
        "total_restart_count",
        "max_restarts_per_rhs",
        "total_residual_replacement_count",
        "restarts_per_rhs",
        "residual_replacements_per_rhs",
        "restart_reason_counts",
        "latest_restart_reason",
    )
    result = {"type": type(error).__name__, "message": str(error)}
    for field in fields:
        value = getattr(error, field, None)
        if isinstance(value, tuple):
            value = list(value)
        result[field] = value
    return result


def capture_first_failure(snapshot, config):
    global _captured
    if _captured:
        return _production_apply(snapshot, config)
    _captured = True
    try:
        return _production_apply(snapshot, config)
    except RWRSolverError as error:
        if CAPTURE_DIRECTORY.exists():
            raise FileExistsError(
                f"refusing to overwrite diagnostic capture {CAPTURE_DIRECTORY}"
            ) from error
        CAPTURE_DIRECTORY.mkdir(parents=True)

        scores = snapshot.unary_scores.detach()
        features = snapshot.dino_features.detach()
        graph = build_directed_topk_graph(
            features[0],
            k=config.top_k,
            affinity_power=config.affinity_power,
        )
        rhs = (1 - config.alpha) * scores[0].to(torch.float32)
        feature_norms = features[0].to(torch.float32).norm(dim=-1)
        historical = json.loads(HISTORICAL_MANIFEST.read_text())
        image = historical["images"][0]
        class_names = historical["class_names"]
        failure = _failure_record(error)
        failing_rhs = failure.get("failing_rhs") or []

        fixture_path = CAPTURE_DIRECTORY / "fixture.pt"
        torch.save(
            {
                "neighbor_indices": graph.neighbor_indices.cpu(),
                "transition_weights": graph.transition_weights.cpu(),
                "edge_affinities": graph.edge_affinities.cpu(),
                "self_loop_fallback": graph.self_loop_fallback.cpu(),
                "unary_scores": scores[0].cpu(),
            },
            fixture_path,
            _use_new_zipfile_serialization=False,
        )

        identity_path = REPOSITORY / "evaluation_identities/e3_canonical_directed_rwr.toml"
        config_path = REPOSITORY / config.config_path
        checkpoint_path = (
            REPOSITORY / "weights/vitb_mlp_infonce_paired_soft_routing_tau010.pth"
        )
        metadata = {
            "format_version": "talk2dino-rwr-first-crop-debug-v1",
            "dataset_index": 0,
            "image_id": image["image_id"],
            "filename": image["filename"],
            "crop_origin_yx": [0, 0],
            "crop_extent_yxyx": [0, 0, 448, 448],
            "window_grid_index": [0, 0],
            "patch_grid": list(snapshot.grid_hw),
            "alpha": config.alpha,
            "top_k": config.top_k,
            "affinity_power": config.affinity_power,
            "rtol": config.solver_rtol,
            "atol": config.solver_atol,
            "max_completed_iterations": config.solver_max_iterations,
            "identity_path": str(identity_path),
            "identity_sha256": _file_sha256(identity_path),
            "config_path": str(config_path),
            "config_sha256": _file_sha256(config_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "dino_checkpoint_sha256": historical["dino_checkpoint_sha256"],
            "clip_checkpoint_sha256": historical["clip_checkpoint_sha256"],
            "unary_scores": _tensor_record(scores[0]),
            "dino_features": {
                **_tensor_record(features[0]),
                "norm_minimum": float(feature_norms.min().item()),
                "norm_maximum": float(feature_norms.max().item()),
                "norm_mean": float(feature_norms.mean().item()),
            },
            "neighbor_indices": _tensor_record(graph.neighbor_indices),
            "transition_weights": _tensor_record(graph.transition_weights),
            "edge_affinities": _tensor_record(graph.edge_affinities),
            "rhs": _tensor_record(rhs),
            "fallback_row_count": int(graph.self_loop_fallback.sum().item()),
            "failing_rhs": failing_rhs,
            "failing_class_names": [class_names[index] for index in failing_rhs],
            "solver_failure": failure,
        }
        metadata_path = CAPTURE_DIRECTORY / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        checksums = {
            "fixture.pt": _file_sha256(fixture_path),
            "metadata.json": _file_sha256(metadata_path),
        }
        checksum_path = CAPTURE_DIRECTORY / "SHA256SUMS.json"
        checksum_path.write_text(
            json.dumps(checksums, indent=2, sort_keys=True) + "\n"
        )
        print(
            "RWR_FIRST_CROP_CAPTURE "
            + json.dumps(
                {
                    "directory": str(CAPTURE_DIRECTORY),
                    "fixture_sha256": checksums["fixture.pt"],
                    "metadata_sha256": checksums["metadata.json"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise


inference_module.apply_rwr_to_e3_snapshot = capture_first_failure
segmentation_module.apply_rwr_to_e3_snapshot = capture_first_failure

runpy.run_path(
    str(OPEN_VOCABULARY_ROOT / "main.py"),
    run_name="__main__",
)
