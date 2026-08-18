#!/usr/bin/env python3
"""Stop after comparing the first production graph with cached E3 bytes."""

from __future__ import annotations

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
from segmentation.evaluation import dinotext_seg as segmentation_module


OUTPUT = Path(os.environ["TALK2DINO_RWR_GRAPH_INSPECTION"])
HISTORICAL_SHARD = Path(
    "/scratch/haree/talk2dino_e3_affinity_oracle/cache/full/shards/windows-000000.pth"
)


def inspect_first(snapshot, config):
    features = snapshot.dino_features[0].detach().to(torch.float32)
    scores = snapshot.unary_scores[0].detach().to(torch.float32)
    graph = build_directed_topk_graph(
        features, k=config.top_k, affinity_power=config.affinity_power
    )
    historical = torch.load(HISTORICAL_SHARD, map_location="cpu", weights_only=True)
    old_indices = historical["knn_indices"][0].to(features.device, torch.int64)
    old_weights = historical["knn_weights"][0].to(features.device, torch.float32)
    old_scores = (
        historical["raw_scores"][0]
        .reshape(scores.shape[1], -1)
        .transpose(0, 1)
        .contiguous()
        .to(features.device, torch.float32)
    )
    affinity = (features @ features.T).clamp_min(0).pow(config.affinity_power)
    affinity.fill_diagonal_(-torch.inf)

    semantic = []
    for row in range(graph.num_nodes):
        current_set = set(graph.neighbor_indices[row].tolist())
        old_set = set(old_indices[row].tolist())
        if current_set == old_set:
            continue
        current_only = sorted(current_set - old_set)
        old_only = sorted(old_set - current_set)
        current_only_values = [float(affinity[row, item].item()) for item in current_only]
        old_only_values = [float(affinity[row, item].item()) for item in old_only]
        semantic.append({
            "row": row,
            "current_only": current_only,
            "historical_only": old_only,
            "current_only_affinity": current_only_values,
            "historical_only_affinity_under_current_features": old_only_values,
            "maximum_boundary_gap": max(
                [abs(left - right) for left in current_only_values for right in old_only_values],
                default=0.0,
            ),
        })
    gaps = [entry["maximum_boundary_gap"] for entry in semantic]
    score_difference = (scores - old_scores).abs()
    report = {
        "image_index": 0,
        "crop": [0, 0, 448, 448],
        "semantic_mismatch_rows": len(semantic),
        "maximum_boundary_affinity_gap": max(gaps, default=0.0),
        "mean_boundary_affinity_gap": sum(gaps) / max(len(gaps), 1),
        "near_tie_rows_gap_le_1e_5": sum(gap <= 1e-5 for gap in gaps),
        "near_tie_rows_gap_le_1e_4": sum(gap <= 1e-4 for gap in gaps),
        "first_ten": semantic[:10],
        "score_maximum_absolute_difference": float(score_difference.max().item()),
        "score_mean_absolute_difference": float(score_difference.mean().item()),
        "historical_graph_row_sum_range": [
            float(old_weights.sum(1).min().item()),
            float(old_weights.sum(1).max().item()),
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("RWR_FIRST_GRAPH_INSPECTION " + json.dumps(report, sort_keys=True), flush=True)
    raise RuntimeError("diagnostic stop after first production graph inspection")


inference_module.apply_rwr_to_e3_snapshot = inspect_first
segmentation_module.apply_rwr_to_e3_snapshot = inspect_first
runpy.run_path(str(OPEN_VOCABULARY_ROOT / "main.py"), run_name="__main__")
