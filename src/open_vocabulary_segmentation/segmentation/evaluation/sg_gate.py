import time

import numpy as np
import torch
import torch.nn.functional as F


COCO_STUFF_THING_CLASSES = 80


def robust_cluster_norm(values, eps=1e-6):
    values = values.float()

    if values.numel() == 0:
        return values

    if values.numel() <= 4:
        lo = values.min()
        hi = values.max()
    else:
        lo = torch.quantile(values, 0.05)
        hi = torch.quantile(values, 0.95)

    if (hi - lo).abs() < eps:
        return torch.zeros_like(values)

    return ((values - lo) / (hi - lo + eps)).clamp(0, 1)


def _local_affinity_graph(raw_patch_features, grid_shape, tau_edge):
    from scipy.sparse import coo_matrix

    hp, wp = grid_shape
    features = F.normalize(raw_patch_features.float(), dim=-1).cpu()
    rows = []
    cols = []
    weights = []
    offsets = ((0, 1), (1, -1), (1, 0), (1, 1))

    for dy, dx in offsets:
        y0 = max(0, -dy)
        y1 = min(hp, hp - dy)
        x0 = max(0, -dx)
        x1 = min(wp, wp - dx)
        if y0 >= y1 or x0 >= x1:
            continue

        yy, xx = torch.meshgrid(
            torch.arange(y0, y1),
            torch.arange(x0, x1),
            indexing="ij",
        )
        src = (yy * wp + xx).reshape(-1)
        dst = ((yy + dy) * wp + (xx + dx)).reshape(-1)
        similarity = (features[src] * features[dst]).sum(dim=-1)
        keep = similarity >= tau_edge
        if not keep.any():
            continue

        src = src[keep].numpy()
        dst = dst[keep].numpy()
        similarity = similarity[keep].numpy()
        rows.extend((src, dst))
        cols.extend((dst, src))
        weights.extend((similarity, similarity))

    n = hp * wp
    return coo_matrix(
        (np.concatenate(weights) if weights else np.empty(0, dtype=np.float32),
         (np.concatenate(rows) if rows else np.empty(0, dtype=np.int64),
          np.concatenate(cols) if cols else np.empty(0, dtype=np.int64))),
        shape=(n, n),
        dtype=np.float32,
    ).tocsr()


def _connected_component_clusters(graph, max_clusters):
    from scipy.sparse.csgraph import connected_components

    _, labels = connected_components(graph, directed=False, return_labels=True)
    components = [
        np.flatnonzero(labels == label)
        for label in np.unique(labels)
    ]
    components.sort(key=len, reverse=True)
    if len(components) > max_clusters:
        components = components[:max_clusters - 1] + [
            np.concatenate(components[max_clusters - 1:])
        ]
    return components


def _components_to_cluster_map(components, grid_shape):
    hp, wp = grid_shape
    num_patches = hp * wp
    cluster_map = torch.empty(num_patches, dtype=torch.long)
    cluster_records = []
    for cluster_id, patch_indices in enumerate(components):
        patch_indices = np.asarray(patch_indices, dtype=np.int64)
        if patch_indices.size == 0:
            continue
        indices_tensor = torch.from_numpy(patch_indices)
        cluster_map[indices_tensor] = cluster_id
        cluster_records.append(
            {
                "cluster_id": cluster_id,
                "patch_indices": indices_tensor,
                "area": int(indices_tensor.numel()),
            }
        )

    if not cluster_records:
        cluster_map.zero_()
        cluster_records = [{
            "cluster_id": 0,
            "patch_indices": torch.arange(num_patches),
            "area": num_patches,
        }]

    return cluster_map.reshape(hp, wp), cluster_records


def build_connected_component_clusters(raw_patch_features, grid_shape, tau_edge):
    from scipy.sparse.csgraph import connected_components

    hp, wp = grid_shape
    num_patches = hp * wp
    if raw_patch_features.ndim != 2 or raw_patch_features.shape[0] != num_patches:
        raise ValueError(
            f"Expected raw patch features [{num_patches}, D], got "
            f"{tuple(raw_patch_features.shape)}"
        )

    try:
        graph = _local_affinity_graph(raw_patch_features, grid_shape, tau_edge)
        _, labels = connected_components(graph, directed=False, return_labels=True)
        components = [
            np.flatnonzero(label == labels)
            for label in np.unique(labels)
        ]
        components.sort(key=len, reverse=True)
    except Exception:
        components = [np.arange(num_patches)]

    if not components:
        components = [np.arange(num_patches)]
    return _components_to_cluster_map(components, grid_shape)


def _spectral_split(graph, indices, min_cluster_area):
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh

    if len(indices) < 2 * min_cluster_area:
        return None

    subgraph = graph[indices][:, indices]
    if subgraph.nnz == 0 or np.asarray(subgraph.sum(axis=1)).max() <= 0:
        return None

    normalized_laplacian = laplacian(subgraph, normed=True)
    _, vectors = eigsh(
        normalized_laplacian,
        k=2,
        which="SM",
        maxiter=100,
    )
    fiedler = vectors[:, 1]
    left_mask = fiedler <= np.median(fiedler)
    left = indices[left_mask]
    right = indices[~left_mask]
    smaller_ratio = min(len(left), len(right)) / float(len(indices))
    if (
        len(left) < min_cluster_area
        or len(right) < min_cluster_area
        or smaller_ratio < 0.1
    ):
        return None
    return left, right


def build_structural_clusters(
    raw_patch_features,
    grid_shape,
    tau_edge,
    max_clusters,
    min_cluster_area,
    cluster_method="connected_components",
    max_seconds=2.0,
):
    hp, wp = grid_shape
    num_patches = hp * wp
    if raw_patch_features.ndim != 2 or raw_patch_features.shape[0] != num_patches:
        raise ValueError(
            f"Expected raw patch features [{num_patches}, D], got "
            f"{tuple(raw_patch_features.shape)}"
        )

    if cluster_method == "connected_components":
        return build_connected_component_clusters(
            raw_patch_features,
            grid_shape,
            tau_edge,
        )
    if cluster_method != "spectral":
        raise ValueError(
            "sg_gate.cluster_method must be 'connected_components' or "
            f"'spectral', got {cluster_method!r}"
        )

    try:
        graph = _local_affinity_graph(raw_patch_features, grid_shape, tau_edge)
        clusters = [np.arange(num_patches)]
        deadline = time.monotonic() + max_seconds
        split_succeeded = False

        while len(clusters) < max_clusters and time.monotonic() < deadline:
            candidates = sorted(
                range(len(clusters)),
                key=lambda index: len(clusters[index]),
                reverse=True,
            )
            split = None
            split_index = None
            for index in candidates:
                try:
                    split = _spectral_split(
                        graph,
                        clusters[index],
                        min_cluster_area,
                    )
                except Exception:
                    split = None
                if split is not None:
                    split_index = index
                    break
            if split is None:
                break

            clusters.pop(split_index)
            clusters.extend(split)
            split_succeeded = True

        if not split_succeeded:
            clusters = _connected_component_clusters(graph, max_clusters)
    except Exception:
        clusters = [np.arange(num_patches)]

    if not clusters:
        clusters = [np.arange(num_patches)]
    return _components_to_cluster_map(clusters[:max_clusters], grid_shape)


def _top_percent_seed(score_map, top_percent):
    flat_score = score_map.reshape(-1)
    num_seeds = max(1, int(np.ceil(flat_score.numel() * top_percent / 100.0)))
    seed_indices = torch.topk(flat_score, k=min(num_seeds, flat_score.numel())).indices
    seed_mask = torch.zeros_like(flat_score, dtype=torch.bool)
    seed_mask[seed_indices] = True
    return seed_mask.reshape_as(score_map)


def _normalized_seed(score_map, top_percent):
    score_min = score_map.min()
    score_range = score_map.max() - score_min
    if score_range.abs() < 1e-6:
        score_norm = torch.zeros_like(score_map)
        seed_mask = torch.zeros_like(score_norm, dtype=torch.bool)
        seed_mask.reshape(-1)[score_map.argmax()] = True
    else:
        score_norm = (score_map - score_min) / (score_range + 1e-6)
        seed_mask = _top_percent_seed(score_norm, float(top_percent))
    if not seed_mask.any():
        seed_mask.reshape(-1)[score_map.argmax()] = True
    return score_norm, seed_mask


def _get_config(config, key, default=None):
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _fallback_config(config):
    fallback = _get_config(config, "raw_fallback", None)
    if fallback is None:
        return {
            "enabled": False,
            "min_area_ratio": 0.60,
            "max_area_ratio": 1.60,
            "min_agreement_iou": 0.35,
            "min_raw_area_pixels": 16,
            "allow_sg_when_raw_empty": False,
            "fallback_if_sg_empty": True,
            "fallback_if_raw_empty_and_no_anchor": True,
        }
    return {
        "enabled": bool(_get_config(fallback, "enabled", True)),
        "min_area_ratio": float(_get_config(fallback, "min_area_ratio", 0.60)),
        "max_area_ratio": float(_get_config(fallback, "max_area_ratio", 1.60)),
        "min_agreement_iou": float(_get_config(fallback, "min_agreement_iou", 0.35)),
        "min_raw_area_pixels": int(_get_config(fallback, "min_raw_area_pixels", 16)),
        "allow_sg_when_raw_empty": bool(_get_config(fallback, "allow_sg_when_raw_empty", False)),
        "fallback_if_sg_empty": bool(_get_config(fallback, "fallback_if_sg_empty", True)),
        "fallback_if_raw_empty_and_no_anchor": bool(
            _get_config(fallback, "fallback_if_raw_empty_and_no_anchor", True)
        ),
    }


def _alignment_audit_enabled(config):
    audit = _get_config(config, "alignment_audit", None)
    return bool(_get_config(audit, "enabled", False)) if audit is not None else False


def _dense_raw_fallback_enabled(config):
    dense = _get_config(config, "dense_raw_fallback", None)
    return bool(_get_config(dense, "enabled", False)) if dense is not None else False


def _soft_reweight_config(config):
    soft = _get_config(config, "soft_reweight", None)
    candidate_source = _get_config(soft, "candidate_source", None)
    if candidate_source is None:
        only_trusted = bool(_get_config(soft, "only_trusted_clusters", True))
        candidate_source = "trusted_clusters" if only_trusted else "anchored_clusters"
    return {
        "enabled": bool(_get_config(soft, "enabled", False)),
        "candidate_source": str(candidate_source),
        "min_soft_gate_score": float(_get_config(soft, "min_soft_gate_score", 0.35)),
        "require_semantic_anchor": bool(_get_config(soft, "require_semantic_anchor", True)),
        "raw_support_topk": int(_get_config(soft, "raw_support_topk", 5)),
        "min_cluster_raw_topk_support": float(
            _get_config(soft, "min_cluster_raw_topk_support", 0.10)
        ),
    }


def _bool_iou(mask_a, mask_b):
    union = int((mask_a | mask_b).sum().item())
    if union == 0:
        return 1.0
    intersection = int((mask_a & mask_b).sum().item())
    return intersection / union


def _downsample_bool_mask(mask, grid_shape):
    pooled = F.interpolate(
        mask[None, None].float(),
        size=grid_shape,
        mode="area",
    )[0, 0]
    return pooled > 0.5


def _empty_gate_stats(num_classes, num_patches, num_clusters):
    return {
        "total_class_decisions": num_classes,
        "class_decisions": num_classes,
        "fallback_used_count": 0,
        "fallback_used": 0,
        "sg_too_small": 0,
        "sg_too_large": 0,
        "sg_empty": 0,
        "low_agreement_iou": 0,
        "low_agreement_iou_count": 0,
        "raw_too_small_block_sg": 0,
        "raw_too_small_block_sg_count": 0,
        "raw_empty_no_anchor": 0,
        "use_sg_raw_empty_with_anchor": 0,
        "fallback_sg_too_small": 0,
        "fallback_sg_too_large": 0,
        "fallback_sg_empty": 0,
        "fallback_low_agreement_iou": 0,
        "fallback_raw_too_small_block_sg": 0,
        "fallback_raw_empty_no_anchor": 0,
        "use_sg": 0,
        "raw_area_sum": 0.0,
        "sg_area_sum": 0.0,
        "sg_to_raw_ratio_sum": 0.0,
        "sg_to_raw_ratio_count": 0,
        "raw_sg_agreement_iou_sum": 0.0,
        "raw_sg_agreement_iou_count": 0,
        "num_patches": num_patches,
        "num_classes": num_classes,
        "num_clusters": num_clusters,
        "ignore_pixels_before_strict": 0,
        "alignment_seed_inside_sg_patch_rate_sum": 0.0,
        "alignment_seed_inside_sg_patch_rate_count": 0,
        "alignment_seed_hit_sg_count": 0,
        "alignment_raw_patch_seed_iou_values": [],
        "alignment_raw_patch_seed_hit_count": 0,
        "alignment_raw_patch_seed_count": 0,
        "alignment_sg_roundtrip_iou_values": [],
        "alignment_num_image_class_mask_decisions": 0,
        "alignment_raw_sg_iou_pre_fallback_values": [],
        "alignment_raw_final_iou_post_fallback_values": [],
        "count_score_map_interpolated_to_dino_grid": 0,
        "count_score_map_already_matching_dino_grid": 0,
        "count_shape_mismatch_after_interpolation": 0,
        "count_raw_mask_size_mismatch": 0,
        "count_sg_mask_size_mismatch": 0,
        "trusted_sg_pixel_overwrite_sum": 0,
        "soft_candidate_classes": 0,
        "soft_candidate_clusters": 0,
        "soft_total_clusters": num_classes * num_clusters,
        "raw_supported_candidate_classes": 0,
        "raw_supported_candidate_clusters": 0,
        "raw_supported_topk_support_sum": 0.0,
        "raw_supported_topk_support_count": 0,
        "seed_only_candidate_classes": 0,
        "seed_overlap_candidate_classes": 0,
        "seed_overlap_candidate_clusters": 0,
        "seed_overlap_total_clusters": num_classes * num_clusters,
    }


def build_seed_only_prior(score_maps, config, return_stats=False):
    if score_maps.ndim != 3:
        raise ValueError(f"Expected score maps [C, H, W], got {tuple(score_maps.shape)}")

    num_classes, hp, wp = score_maps.shape
    prior = score_maps.new_zeros((num_classes, hp, wp))
    ignore = torch.zeros((hp, wp), dtype=torch.bool, device=score_maps.device)
    stats = _empty_gate_stats(num_classes, hp * wp, 0)

    for class_index in range(num_classes):
        score_norm, seed_mask = _normalized_seed(
            score_maps[class_index],
            float(config.top_percent),
        )
        prior[class_index, seed_mask] = score_norm[seed_mask].clamp(0, 1)
        if seed_mask.any():
            stats["seed_only_candidate_classes"] += 1
            stats["soft_candidate_classes"] += 1

    if return_stats:
        return prior, ignore, stats
    return prior, ignore


def build_seed_overlap_cluster_prior(score_maps, cluster_map, config, return_stats=False):
    if score_maps.ndim != 3:
        raise ValueError(f"Expected score maps [C, H, W], got {tuple(score_maps.shape)}")

    hp, wp = cluster_map.shape
    if score_maps.shape[-2:] != (hp, wp):
        score_maps = F.interpolate(
            score_maps.unsqueeze(0),
            size=(hp, wp),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    num_classes = score_maps.shape[0]
    cluster_ids = torch.unique(cluster_map, sorted=True)
    cluster_masks = []
    for cluster_id in cluster_ids:
        cluster_mask = cluster_map == cluster_id
        if cluster_mask.any():
            cluster_masks.append(cluster_mask)

    prior = score_maps.new_zeros((num_classes, hp, wp))
    ignore = torch.zeros((hp, wp), dtype=torch.bool, device=score_maps.device)
    stats = _empty_gate_stats(num_classes, hp * wp, len(cluster_masks))
    min_seed_overlap = float(
        _get_config(
            _get_config(config, "seed_overlap_clusters", None),
            "min_seed_overlap",
            0.01,
        )
    )

    for class_index in range(num_classes):
        _, seed_mask = _normalized_seed(
            score_maps[class_index],
            float(config.top_percent),
        )
        class_has_candidate = False
        for cluster_mask in cluster_masks:
            seed_overlap = float(seed_mask[cluster_mask].float().mean().item())
            if seed_overlap < min_seed_overlap:
                continue
            prior[class_index, cluster_mask] = torch.maximum(
                prior[class_index, cluster_mask],
                score_maps.new_full(
                    prior[class_index, cluster_mask].shape,
                    min(seed_overlap / max(min_seed_overlap, 1e-6), 1.0),
                ),
            )
            class_has_candidate = True
            stats["soft_candidate_clusters"] += 1
            stats["seed_overlap_candidate_clusters"] += 1
        if class_has_candidate:
            stats["soft_candidate_classes"] += 1
            stats["seed_overlap_candidate_classes"] += 1

    if return_stats:
        return prior, ignore, stats
    return prior, ignore


def apply_structural_gate(
    score_maps,
    raw_patch_features,
    cluster_map,
    config,
    return_stats=False,
    raw_label_img=None,
    image_size=None,
):
    if score_maps.ndim != 3:
        raise ValueError(f"Expected score maps [C, H, W], got {tuple(score_maps.shape)}")

    hp, wp = cluster_map.shape
    if score_maps.shape[-2:] != (hp, wp):
        score_maps = F.interpolate(
            score_maps.unsqueeze(0),
            size=(hp, wp),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    assert score_maps.shape[-2:] == cluster_map.shape

    num_classes = score_maps.shape[0]
    num_patches = hp * wp
    raw_patch_features = raw_patch_features.float()
    normalized_features = F.normalize(raw_patch_features, dim=-1)
    cluster_ids = torch.unique(cluster_map, sorted=True)
    cluster_masks = []
    flat_cluster_masks = []
    cluster_centroids = []
    area_ratio_values = []
    cluster_coherence_values = []
    for cluster_id in cluster_ids:
        cluster_mask = cluster_map == cluster_id
        flat_cluster_mask = cluster_mask.reshape(-1)
        area = int(flat_cluster_mask.sum().item())
        if area == 0:
            continue
        cluster_features = raw_patch_features[flat_cluster_mask]
        cluster_centroid = F.normalize(cluster_features.mean(dim=0), dim=0)
        cluster_masks.append(cluster_mask)
        flat_cluster_masks.append(flat_cluster_mask)
        cluster_centroids.append(cluster_centroid)
        area_ratio_values.append(score_maps.new_tensor(area / float(num_patches)))
        cluster_coherence_values.append(
            (
                normalized_features[flat_cluster_mask]
                * cluster_centroid
            ).sum(dim=-1).mean()
        )

    if not cluster_masks:
        positive_scores = score_maps.new_zeros((num_classes, hp, wp))
        ignore = torch.zeros((hp, wp), dtype=torch.bool, device=score_maps.device)
        stats = _empty_gate_stats(num_classes, num_patches, 0)
        if return_stats:
            return positive_scores, ignore, stats
        return positive_scores, ignore

    cluster_centroids = torch.stack(cluster_centroids)
    area_ratio_values = torch.stack(area_ratio_values)
    cluster_coherence_values = torch.stack(cluster_coherence_values)
    thing_area_ratio_norm = robust_cluster_norm(area_ratio_values)
    thing_cluster_coherence_norm = robust_cluster_norm(cluster_coherence_values)

    ignore_mask = torch.zeros((hp, wp), dtype=torch.bool, device=score_maps.device)
    sg_masks = torch.zeros((num_classes, hp, wp), dtype=torch.bool, device=score_maps.device)
    sg_gate_scores = score_maps.new_full((num_classes, hp, wp), -torch.inf)
    sg_mean_scores = score_maps.new_full((num_classes, hp, wp), -torch.inf)
    semantic_anchor_exists = torch.zeros(num_classes, dtype=torch.bool, device=score_maps.device)
    fallback = _fallback_config(config)
    audit_enabled = _alignment_audit_enabled(config)
    dense_raw_fallback = _dense_raw_fallback_enabled(config)
    soft_config = _soft_reweight_config(config)
    raw_label = score_maps.argmax(dim=0)
    stats = _empty_gate_stats(num_classes, num_patches, len(cluster_masks))
    seed_masks = []
    trusted_masks = torch.zeros_like(sg_masks)
    soft_candidate_masks = torch.zeros_like(sg_masks)
    soft_candidate_scores = score_maps.new_full((num_classes, hp, wp), -torch.inf)
    soft_candidate_class_has_mask = torch.zeros(num_classes, dtype=torch.bool, device=score_maps.device)
    raw_supported_class_has_mask = torch.zeros(num_classes, dtype=torch.bool, device=score_maps.device)
    sg_masks_img = None
    raw_label_for_trust = None
    raw_topk_support_mask = None
    if soft_config["enabled"] and soft_config["candidate_source"] == "raw_supported_clusters":
        raw_support_topk = max(1, min(int(soft_config["raw_support_topk"]), num_classes))
        raw_topk_indices = score_maps.topk(k=raw_support_topk, dim=0).indices
        raw_topk_support_mask = torch.zeros_like(score_maps, dtype=torch.bool)
        raw_topk_support_mask.scatter_(0, raw_topk_indices, True)

    for class_index in range(num_classes):
        is_thing = class_index < COCO_STUFF_THING_CLASSES
        raw_score = score_maps[class_index]
        score_min = raw_score.min()
        score_range = raw_score.max() - score_min
        if score_range.abs() < 1e-6:
            score_norm = torch.zeros_like(raw_score)
            seed_mask = torch.zeros_like(score_norm, dtype=torch.bool)
            seed_mask.reshape(-1)[raw_score.argmax()] = True
        else:
            score_norm = (raw_score - score_min) / (score_range + 1e-6)
            seed_mask = _top_percent_seed(score_norm, float(config.top_percent))
        if not seed_mask.any():
            seed_mask.reshape(-1)[raw_score.argmax()] = True
        seed_masks.append(seed_mask)
        seed_features = raw_patch_features[seed_mask.reshape(-1)]
        seed_centroid = F.normalize(seed_features.mean(dim=0), dim=0)

        seed_overlap = []
        mean_talk_score = []
        for cluster_mask in cluster_masks:
            seed_overlap.append(seed_mask[cluster_mask].float().mean())
            mean_talk_score.append(score_norm[cluster_mask].mean())

        seed_overlap = torch.stack(seed_overlap)
        mean_talk_score = torch.stack(mean_talk_score)
        affinity_to_seed = cluster_centroids @ seed_centroid
        seed_overlap_norm = robust_cluster_norm(seed_overlap)
        mean_talk_score_norm = robust_cluster_norm(mean_talk_score)
        affinity_to_seed_norm = robust_cluster_norm(affinity_to_seed)

        gate_score = (
            2.0 * seed_overlap_norm
            + mean_talk_score_norm
            + affinity_to_seed_norm
        )
        if is_thing:
            gate_score = (
                gate_score
                + 0.2 * thing_cluster_coherence_norm
                - 0.1 * thing_area_ratio_norm
            )
            tau_pos = float(config.tau_pos_thing)
            tau_ignore = float(config.tau_ignore_thing)
        else:
            tau_pos = float(config.tau_pos_stuff)
            tau_ignore = float(config.tau_ignore_stuff)

        semantic_anchor = (
            (seed_overlap > 0)
            | (mean_talk_score >= float(config.score_anchor_threshold))
        )
        positive = semantic_anchor & (gate_score >= tau_pos)
        ignored = semantic_anchor & ~positive & (gate_score >= tau_ignore)
        semantic_anchor_exists[class_index] = (
            semantic_anchor[positive].any()
            or (score_norm.max() >= float(config.score_anchor_threshold))
        )

        for cluster_index, cluster_mask in enumerate(cluster_masks):
            soft_candidate = (
                bool(semantic_anchor[cluster_index].item())
                and bool((gate_score[cluster_index] >= soft_config["min_soft_gate_score"]).item())
            )
            if soft_config["require_semantic_anchor"]:
                soft_candidate = soft_candidate and bool(semantic_anchor_exists[class_index].item())
            if soft_config["enabled"] and soft_config["candidate_source"] == "anchored_clusters" and soft_candidate:
                soft_candidate_masks[class_index, cluster_mask] = True
                soft_candidate_scores[class_index, cluster_mask] = torch.maximum(
                    soft_candidate_scores[class_index, cluster_mask],
                    gate_score[cluster_index].expand_as(soft_candidate_scores[class_index, cluster_mask]),
                )
                soft_candidate_class_has_mask[class_index] = True
                stats["soft_candidate_clusters"] += 1
            if (
                soft_config["enabled"]
                and soft_config["candidate_source"] == "raw_supported_clusters"
                and soft_candidate
                and bool(positive[cluster_index].item())
                and raw_topk_support_mask is not None
            ):
                cluster_raw_topk_support = float(
                    raw_topk_support_mask[class_index, cluster_mask].float().mean().item()
                )
                stats["raw_supported_topk_support_sum"] += cluster_raw_topk_support
                stats["raw_supported_topk_support_count"] += 1
                if cluster_raw_topk_support >= soft_config["min_cluster_raw_topk_support"]:
                    soft_candidate_masks[class_index, cluster_mask] = True
                    soft_candidate_scores[class_index, cluster_mask] = torch.maximum(
                        soft_candidate_scores[class_index, cluster_mask],
                        gate_score[cluster_index].expand_as(
                            soft_candidate_scores[class_index, cluster_mask]
                        ),
                    )
                    soft_candidate_class_has_mask[class_index] = True
                    raw_supported_class_has_mask[class_index] = True
                    stats["soft_candidate_clusters"] += 1
                    stats["raw_supported_candidate_clusters"] += 1
            if positive[cluster_index]:
                sg_masks[class_index, cluster_mask] = True
                sg_gate_scores[class_index, cluster_mask] = gate_score[cluster_index]
                sg_mean_scores[class_index, cluster_mask] = mean_talk_score[cluster_index]
            elif ignored[cluster_index]:
                ignore_mask |= cluster_mask

    if dense_raw_fallback and raw_label_img is not None and image_size is not None:
        if tuple(raw_label_img.shape[-2:]) == tuple(image_size):
            raw_label_for_trust = raw_label_img
            sg_masks_img = F.interpolate(
                sg_masks.float().unsqueeze(0),
                size=image_size,
                mode="nearest",
            )[0] > 0

    final_label = raw_label.clone()
    for class_index in range(num_classes):
        raw_mask = raw_label == class_index
        sg_mask = sg_masks[class_index]
        if raw_label_for_trust is not None and sg_masks_img is not None:
            raw_mask_for_decision = raw_label_for_trust == class_index
            sg_mask_for_decision = sg_masks_img[class_index]
        else:
            raw_mask_for_decision = raw_mask
            sg_mask_for_decision = sg_mask
        raw_area = int(raw_mask_for_decision.sum().item())
        sg_area = int(sg_mask_for_decision.sum().item())
        union = int((raw_mask_for_decision | sg_mask_for_decision).sum().item())
        intersection = int((raw_mask_for_decision & sg_mask_for_decision).sum().item())
        agreement_iou = 1.0 if union == 0 else intersection / union
        stats["raw_area_sum"] += raw_area
        stats["sg_area_sum"] += sg_area
        if union > 0:
            stats["raw_sg_agreement_iou_sum"] += agreement_iou
            stats["raw_sg_agreement_iou_count"] += 1

        if not fallback["enabled"]:
            reason = "use_sg"
        elif raw_area < fallback["min_raw_area_pixels"]:
            reason = "raw_too_small_block_sg"
        else:
            ratio = sg_area / (raw_area + 1e-6)
            stats["sg_to_raw_ratio_sum"] += ratio
            stats["sg_to_raw_ratio_count"] += 1
            if sg_area == 0 and fallback["fallback_if_sg_empty"]:
                reason = "sg_empty"
            elif ratio < fallback["min_area_ratio"]:
                reason = "sg_too_small"
            elif ratio > fallback["max_area_ratio"]:
                reason = "sg_too_large"
            elif agreement_iou < fallback["min_agreement_iou"]:
                reason = "low_agreement_iou"
            else:
                reason = "use_sg"

        if reason in {"use_sg", "use_sg_raw_empty_with_anchor"}:
            stats["use_sg"] += 1
            if reason == "use_sg_raw_empty_with_anchor":
                stats["use_sg_raw_empty_with_anchor"] += 1
            trusted_masks[class_index] = sg_mask
            if soft_config["enabled"] and soft_config["candidate_source"] == "trusted_clusters":
                soft_candidate_masks[class_index] = sg_mask
                soft_candidate_scores[class_index] = torch.maximum(
                    soft_candidate_scores[class_index],
                    sg_gate_scores[class_index],
                )
                soft_candidate_class_has_mask[class_index] = sg_mask.any()
            final_label[sg_mask] = class_index
        else:
            stats["fallback_used"] += 1
            stats["fallback_used_count"] += 1
            stats[reason] += 1
            stats[f"fallback_{reason}"] += 1
            if reason == "low_agreement_iou":
                stats["low_agreement_iou_count"] += 1
            elif reason == "raw_too_small_block_sg":
                stats["raw_too_small_block_sg_count"] += 1

    stats["ignore_pixels_before_strict"] = int(ignore_mask.sum().item())
    stats["trusted_sg_pixel_overwrite_sum"] = int(trusted_masks.any(dim=0).sum().item())
    stats["soft_candidate_classes"] = int(soft_candidate_class_has_mask.sum().item())
    stats["raw_supported_candidate_classes"] = int(raw_supported_class_has_mask.sum().item())
    final_masks = torch.zeros_like(sg_masks)
    for class_index in range(num_classes):
        final_masks[class_index] = final_label == class_index

    if audit_enabled:
        min_raw_area_pixels = fallback["min_raw_area_pixels"]
        for class_index in range(num_classes):
            seed_mask = seed_masks[class_index]
            sg_mask = sg_masks[class_index]
            intersection = int((seed_mask & sg_mask).sum().item())
            seed_area = int(seed_mask.sum().item())
            stats["alignment_seed_inside_sg_patch_rate_sum"] += (
                intersection / (seed_area + 1e-6)
            )
            stats["alignment_seed_inside_sg_patch_rate_count"] += 1
            stats["alignment_seed_hit_sg_count"] += 1 if intersection > 0 else 0

        if raw_label_img is not None and image_size is not None:
            if tuple(raw_label_img.shape[-2:]) != tuple(image_size):
                stats["count_raw_mask_size_mismatch"] += 1
            else:
                audit_sg_masks_img = F.interpolate(
                    sg_masks.float().unsqueeze(0),
                    size=image_size,
                    mode="nearest",
                )[0] > 0
                final_masks_img = F.interpolate(
                    final_masks.float().unsqueeze(0),
                    size=image_size,
                    mode="nearest",
                )[0] > 0
                sg_roundtrip = F.interpolate(
                    audit_sg_masks_img.float().unsqueeze(0),
                    size=(hp, wp),
                    mode="area",
                )[0] > 0.5
                if audit_sg_masks_img.shape[-2:] != tuple(image_size):
                    stats["count_sg_mask_size_mismatch"] += 1

                for class_index in range(num_classes):
                    raw_mask_img = raw_label_img == class_index
                    seed_mask = seed_masks[class_index]
                    raw_mask_patch = _downsample_bool_mask(raw_mask_img, (hp, wp))
                    if (
                        int(raw_mask_img.sum().item()) > min_raw_area_pixels
                        or int(seed_mask.sum().item()) > 0
                    ):
                        stats["alignment_raw_patch_seed_iou_values"].append(
                            _bool_iou(raw_mask_patch, seed_mask)
                        )
                        hit = bool((raw_mask_patch & seed_mask).any().item())
                        stats["alignment_raw_patch_seed_hit_count"] += 1 if hit else 0
                        stats["alignment_raw_patch_seed_count"] += 1

                    stats["alignment_sg_roundtrip_iou_values"].append(
                        _bool_iou(sg_masks[class_index], sg_roundtrip[class_index])
                    )

                    raw_area_img = int(raw_mask_img.sum().item())
                    sg_area_img = int(audit_sg_masks_img[class_index].sum().item())
                    if (
                        raw_area_img > min_raw_area_pixels
                        or sg_area_img > min_raw_area_pixels
                    ):
                        stats["alignment_num_image_class_mask_decisions"] += 1
                        stats["alignment_raw_sg_iou_pre_fallback_values"].append(
                            _bool_iou(raw_mask_img, audit_sg_masks_img[class_index])
                        )
                        stats["alignment_raw_final_iou_post_fallback_values"].append(
                            _bool_iou(raw_mask_img, final_masks_img[class_index])
                        )

    best_raw_score = score_maps.new_full((hp, wp), -torch.inf)
    best_gate_score = score_maps.new_full((hp, wp), -torch.inf)
    best_class = torch.full((hp, wp), -1, dtype=torch.long, device=score_maps.device)
    prior_masks = soft_candidate_masks if soft_config["enabled"] else trusted_masks
    prior_scores = soft_candidate_scores if soft_config["enabled"] else sg_gate_scores
    for class_index in range(num_classes):
        class_mask = prior_masks[class_index]
        if not class_mask.any():
            continue
        raw_score = score_maps[class_index]
        candidate_raw = raw_score[class_mask]
        candidate_gate = prior_scores[class_index, class_mask]
        current_raw = best_raw_score[class_mask]
        current_gate = best_gate_score[class_mask]
        raw_tie = candidate_raw == current_raw
        gate_tie = candidate_gate == current_gate
        better = (
            (candidate_raw > current_raw)
            | (raw_tie & (candidate_gate > current_gate))
            | (raw_tie & gate_tie & (class_index < best_class[class_mask]))
        )
        class_positions = class_mask.nonzero(as_tuple=True)
        selected_positions = tuple(positions[better] for positions in class_positions)
        best_raw_score[selected_positions] = candidate_raw[better]
        best_gate_score[selected_positions] = candidate_gate[better]
        best_class[selected_positions] = class_index

    positive_scores = score_maps.new_zeros((num_classes, hp, wp))
    has_trusted = best_class >= 0
    if has_trusted.any():
        positions = has_trusted.nonzero(as_tuple=True)
        trusted_prior = torch.clamp(best_gate_score[has_trusted], min=0.0, max=1.0)
        positive_scores[
            best_class[has_trusted],
            positions[0],
            positions[1],
        ] = trusted_prior
    if return_stats:
        return positive_scores, ignore_mask, stats
    return positive_scores, ignore_mask
