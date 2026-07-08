import mmcv
import math
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from omegaconf import OmegaConf
from utils import get_logger
from .sg_gate import (
    apply_structural_gate,
    build_seed_only_prior,
    build_seed_overlap_cluster_prior,
    build_structural_clusters,
)
from .multiscale_eval import aggregate_legacy_probability_maps

try:
    from mmcv.parallel import DataContainer
except Exception:
    DataContainer = None


def unwrap_datacontainer(value):
    if DataContainer is not None and isinstance(value, DataContainer):
        value = value.data

    while (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]

    return value


def to_mmcv_config(value):
    if isinstance(value, mmcv.Config):
        return value
    if value is None:
        value = {"enabled": False}
    elif OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return mmcv.Config(value)


class DINOTextSegInference(nn.Module):
    def __init__(
            self,
            model,
            text_embedding,
            classnames,
            with_bg,
            test_cfg=dict(),
            pamr=False,
            bg_thresh=0.5,
            bg_strategy="base",
            sg_gate=None,
            msa=None,
            # kp_w=0.3,
            **kwargs,
    ):
        super().__init__()

        if not isinstance(test_cfg, mmcv.Config):
            test_cfg = mmcv.Config(test_cfg)
        self.test_cfg = test_cfg
        self.pamr = pamr
        self.bg_thresh = bg_thresh
        self.bg_strategy = bg_strategy
        self.sg_gate = to_mmcv_config(sg_gate)
        self.sg_gate_enabled = bool(self.sg_gate.get("enabled", False))
        self.debug_force_raw_output = bool(self.sg_gate.get("debug_force_raw_output", False))
        self.debug_force_raw_apply_pamr = bool(
            self.sg_gate.get("debug_force_raw_apply_pamr", False)
        )
        self.ablation_mode = str(self.sg_gate.get("ablation_mode", "full_e1_e2_e3"))
        self.true_pamr_dual_path_debug = (
            self.ablation_mode == "true_pamr_dual_path_debug"
        )
        self.sg_prediction_enabled = self.sg_gate_enabled and not self.debug_force_raw_output
        self.msa = to_mmcv_config(msa)
        self.msa_enabled = bool(self.msa.get("enabled", False))
        if self.msa_enabled and self.sg_gate_enabled:
            raise ValueError("MSA is incompatible with SG-Gate")
        self.msa_scales = [float(scale) for scale in self.msa.get("scales", [1.0])]
        self.msa_hflip = bool(self.msa.get("hflip", False))
        self.msa_log_per_scale_stats = bool(
            self.msa.get("log_per_scale_stats", True)
        )
        self.msa_scale_weights = self.msa.get("weights", None)
        if self.msa_scale_weights is not None:
            self.msa_scale_weights = [float(value) for value in self.msa_scale_weights]
        self._msa_images = 0
        self._msa_forward_passes = 0
        self._msa_time_total = 0.0
        self._msa_single_scale_max_abs_diff = 0.0
        self._msa_per_scale_mean = [0.0 for _ in self.msa_scales]
        self._msa_per_scale_std = [0.0 for _ in self.msa_scales]
        # self.kp_w = kp_w

        self.model = model
        self.register_buffer("text_embedding", text_embedding)
        self.classnames = classnames
        self.with_bg = with_bg
        if self.with_bg:
            self.num_classes = len(text_embedding) + 1
        else:
            self.num_classes = len(text_embedding)

        self.align_corners = False
        self.out_channels = self.num_classes
        self.fp16_enabled = False
        logger = get_logger()
        logger.info(
            f"Building DINOTextSegInference with {self.num_classes} classes, test_cfg={test_cfg}, with_bg={with_bg}"
            f", pamr={pamr}, bg_thresh={bg_thresh}, sg_gate={self.sg_gate_enabled}, debug_force_raw={self.debug_force_raw_output}"
        )
        if self.sg_prediction_enabled and self.sg_gate.get("use_scratch_cache", False):
            logger.info("SG-Gate runs online; no scratch tensor cache will be written")
        self._sg_timing = None
        self._sg_shuffle_index = 0

    def msa_summary(self):
        if not self.msa_enabled:
            return None
        images = max(1, self._msa_images)
        return {
            "msa_enabled": True,
            "msa_output_representation": (
                "softmax_probability_of_slide_averaged_sigmoid_scores"
            ),
            "msa_scales": list(self.msa_scales),
            "msa_hflip": self.msa_hflip,
            "msa_num_forward_passes_per_image": (
                self._msa_forward_passes / images
            ),
            "msa_aggregation": str(self.msa.get("aggregation", "mean")),
            "msa_scale_weighting": str(
                self.msa.get("scale_weighting", "uniform")
            ),
            "msa_resize_mode": str(self.msa.get("resize_mode", "bilinear")),
            "msa_align_corners": bool(self.msa.get("align_corners", False)),
            "msa_single_scale_max_abs_diff": self._msa_single_scale_max_abs_diff,
            "msa_per_scale_output_mean": [
                value / images for value in self._msa_per_scale_mean
            ],
            "msa_per_scale_output_std": [
                value / images for value in self._msa_per_scale_std
            ],
            "msa_time_per_image": self._msa_time_total / images,
            "msa_time_total": self._msa_time_total,
        }

    def reset_sg_timing(self):
        self._sg_timing = {
            "talk2dino_forward_time": 0.0,
            "e2_clustering_time": 0.0,
            "e3_gate_time": 0.0,
        }
        self._sg_stats = {
            "total_class_decisions": 0,
            "class_decisions": 0,
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
            "num_patches": 0,
            "num_classes": 0,
            "num_clusters": 0,
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
            "uncertain_pixels": 0,
            "apply_pamr_after_soft": False,
            "pamr_source": "not_available",
            "shuffle_prior_enabled": False,
            "shuffle_prior_mode": "not_available",
            "shuffle_prior_seed": "not_available",
            "max_abs_diff_before_pamr": 0.0,
            "random_prior_density_pixels": 0,
            "raw_topk_prior_density_pixels": 0,
            "pamr_dual_path_debug_enabled": False,
            "raw_scores_shape": "not_available",
            "path_a_pamr_input_shape": "not_available",
            "path_b_pamr_input_shape": "not_available",
            "raw_scores_dtype": "not_available",
            "path_a_input_dtype": "not_available",
            "path_b_input_dtype": "not_available",
            "max_abs_diff_raw_vs_path_a_input": 0.0,
            "max_abs_diff_raw_vs_path_b_input": 0.0,
            "max_abs_diff_path_a_input_vs_path_b_input": 0.0,
            "max_abs_diff_path_a_pamr_output_vs_path_b_pamr_output": 0.0,
            "mean_abs_diff_path_a_pamr_output_vs_path_b_pamr_output_sum": 0.0,
            "mean_abs_diff_path_a_pamr_output_vs_path_b_pamr_output_count": 0,
            "final_label_disagreement_pixels": 0,
            "final_label_disagreement_total": 0,
            "path_a_pamr_num_iter": "not_available",
            "path_b_pamr_num_iter": "not_available",
            "path_a_pamr_dilations": "not_available",
            "path_b_pamr_dilations": "not_available",
            "pamr_kernel_or_neighborhood": "not_available",
            "path_a_image_normalization_input": "not_available",
            "path_b_image_normalization_input": "not_available",
            "pamr_uses_image_rgb_or_feature": "image_rgb",
            "path_a_scores_are": "raw_scores",
            "path_b_scores_are": "raw_scores",
            "path_a_softmax_before_pamr": False,
            "path_b_softmax_before_pamr": False,
            "path_a_resize_before_or_after_pamr": "not_available",
            "path_b_resize_before_or_after_pamr": "not_available",
            "path_a_final_argmax_resolution": "not_available",
            "path_b_final_argmax_resolution": "not_available",
            "evaluator_input_resolution": "not_available",
        }

    def consume_sg_timing(self):
        timing = self._sg_timing or {}
        self._sg_timing = None
        return timing

    def consume_sg_stats(self):
        stats = getattr(self, "_sg_stats", None) or {}
        self._sg_stats = None
        return stats

    def add_sg_stats(self, stats):
        for key, value in stats.items():
            if isinstance(value, list):
                self._sg_stats.setdefault(key, [])
                self._sg_stats[key].extend(value)
            elif key in {
                "max_abs_diff_before_pamr",
                "max_abs_diff_raw_vs_path_a_input",
                "max_abs_diff_raw_vs_path_b_input",
                "max_abs_diff_path_a_input_vs_path_b_input",
                "max_abs_diff_path_a_pamr_output_vs_path_b_pamr_output",
            }:
                self._sg_stats[key] = max(float(self._sg_stats.get(key, 0.0)), float(value))
            elif isinstance(value, (str, bool)):
                self._sg_stats.setdefault(key, value)
            else:
                self._sg_stats[key] = self._sg_stats.get(key, 0) + value

    def _apply_pamr_in_chunks(self, pamr_image, scores):
        scores = scores.clone()
        for class_start in range(0, scores.shape[1], 30):
            scores[:, class_start:class_start + 30] = self.model.apply_pamr(
                pamr_image,
                scores[:, class_start:class_start + 30],
            )
        return scores

    def _pamr_debug_config(self, image_shape, score_shape):
        pamr_module = getattr(self.model, "pamr", None)
        dilations = "not_available"
        if pamr_module is not None and hasattr(pamr_module, "aff_x"):
            dilations = str(getattr(pamr_module.aff_x, "dilations", "not_available"))
        num_iter = (
            getattr(pamr_module, "num_iter", "not_available")
            if pamr_module is not None
            else "not_available"
        )
        return {
            "path_a_pamr_num_iter": num_iter,
            "path_b_pamr_num_iter": num_iter,
            "path_a_pamr_dilations": dilations,
            "path_b_pamr_dilations": dilations,
            "pamr_kernel_or_neighborhood": dilations,
            "path_a_image_normalization_input": f"BGR->RGB tensor shape={tuple(image_shape)}",
            "path_b_image_normalization_input": f"BGR->RGB tensor shape={tuple(image_shape)}",
            "path_a_resize_before_or_after_pamr": (
                "same raw_scores tensor after slide/resize aggregation"
            ),
            "path_b_resize_before_or_after_pamr": (
                "same raw_scores tensor after slide/resize aggregation"
            ),
            "path_a_final_argmax_resolution": str(tuple(score_shape[-2:])),
            "path_b_final_argmax_resolution": str(tuple(score_shape[-2:])),
            "evaluator_input_resolution": str(tuple(score_shape[-2:])),
        }

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.
        """
        assert img.shape[0] == 1, "batch size must be 1"

        # masks [B, N, H, W]
        # simmap [B, N, H//4, W//4]
        # soft mask (logit-like) is required
        if self.sg_prediction_enabled:
            talk_start = time.perf_counter()
            masks, simmap, raw_patch_features = self.model.generate_masks(
                img,
                img_metas,
                self.text_embedding,
                self.classnames,
                apply_pamr=False,
                return_sg_inputs=True,
            )
            self._sg_timing["talk2dino_forward_time"] += (
                time.perf_counter() - talk_start
            )
        else:
            masks, simmap = self.model.generate_masks(
                img,
                img_metas,
                self.text_embedding,
                self.classnames,
                apply_pamr=(
                    self.debug_force_raw_apply_pamr
                    if self.debug_force_raw_output
                    else self.pamr
                ),
                # kp_w=self.kp_w,
            )

        raw_label_img = None
        if self.sg_prediction_enabled:
            B, N = masks.shape[:2]
            H, W = img.shape[-2:]
            if B == 1 and N == self.num_classes:
                raw_label_img = masks[0].argmax(dim=0)
        else:
            B, N, H, W = masks.shape

        if self.with_bg:

            masks = masks.cpu()

            background = torch.full(
                [B, 1, H, W], self.bg_thresh, dtype=torch.float, device=masks.device
            )
            masks = torch.cat([background, masks], dim=1)
            masks = masks.to(img.device)

        if not self.sg_prediction_enabled:
            return masks
        raw_scores = masks
        if raw_scores.shape[-2:] != (H, W):
            raw_scores = F.interpolate(
                raw_scores,
                size=(H, W),
                mode="bilinear",
                align_corners=self.align_corners,
            )
        if B == 1 and raw_scores.shape[1] == self.num_classes:
            raw_label_img = raw_scores[0].argmax(dim=0)

        score_maps = simmap[0]
        soft_cfg = self.sg_gate.get("soft_reweight", {})
        candidate_source = str(soft_cfg.get("candidate_source", "trusted_clusters"))
        cluster_map = None
        shape_stats = {
            "count_score_map_interpolated_to_dino_grid": 0,
            "count_score_map_already_matching_dino_grid": 0,
            "count_shape_mismatch_after_interpolation": 0,
        }

        if candidate_source == "seed_only":
            gate_start = time.perf_counter()
            positive_scores, ignore_mask, gate_stats = build_seed_only_prior(
                score_maps,
                self.sg_gate,
                return_stats=True,
            )
        else:
            num_patches = raw_patch_features.shape[1]
            grid_size = math.isqrt(num_patches)
            if grid_size * grid_size != num_patches:
                raise ValueError(f"SG-Gate requires a square DINO patch grid, got {num_patches} tokens")

            cluster_start = time.perf_counter()
            cluster_map, _ = build_structural_clusters(
                raw_patch_features[0],
                (grid_size, grid_size),
                tau_edge=float(self.sg_gate.tau_edge),
                max_clusters=int(self.sg_gate.max_clusters),
                min_cluster_area=int(self.sg_gate.min_cluster_area),
                cluster_method=self.sg_gate.get("cluster_method", "connected_components"),
            )
            self._sg_timing["e2_clustering_time"] += (
                time.perf_counter() - cluster_start
            )
            cluster_map = cluster_map.to(simmap.device)
            if score_maps.shape[-2:] != cluster_map.shape:
                shape_stats["count_score_map_interpolated_to_dino_grid"] += 1
                score_maps = F.interpolate(
                    score_maps.unsqueeze(0),
                    size=cluster_map.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            else:
                shape_stats["count_score_map_already_matching_dino_grid"] += 1
            if score_maps.shape != (self.num_classes, *cluster_map.shape):
                shape_stats["count_shape_mismatch_after_interpolation"] += 1
                raise ValueError(
                    "Score map shape could not be aligned to DINO grid: "
                    f"score_maps={tuple(score_maps.shape)}, cluster_map={tuple(cluster_map.shape)}"
                )

            gate_start = time.perf_counter()
            if candidate_source == "seed_overlap_clusters":
                positive_scores, ignore_mask, gate_stats = build_seed_overlap_cluster_prior(
                    score_maps,
                    cluster_map,
                    self.sg_gate,
                    return_stats=True,
                )
            else:
                positive_scores, ignore_mask, gate_stats = apply_structural_gate(
                    score_maps,
                    raw_patch_features[0],
                    cluster_map,
                    self.sg_gate,
                    return_stats=True,
                    raw_label_img=raw_label_img,
                    image_size=(H, W),
                )
        for key, value in shape_stats.items():
            gate_stats[key] = gate_stats.get(key, 0) + value
        self.add_sg_stats(gate_stats)
        self._sg_timing["e3_gate_time"] += time.perf_counter() - gate_start
        positive_scores = F.interpolate(
            positive_scores.unsqueeze(0),
            size=(H, W),
            mode="nearest",
        )
        ignore_score = F.interpolate(
            ignore_mask[None, None].float(),
            size=(H, W),
            mode="nearest",
        )
        return {
            "positive": positive_scores,
            "ignore": ignore_score,
            "raw": raw_scores,
        }

    def slide_inference(self, img, img_meta, rescale):
        img_meta = unwrap_datacontainer(img_meta)
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, self.out_channels, h_img, w_img))
        raw_preds = img.new_zeros((batch_size, self.out_channels, h_img, w_img)) \
            if self.sg_prediction_enabled else None
        ignore_preds = img.new_zeros((batch_size, 1, h_img, w_img)) \
            if self.sg_prediction_enabled else None
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        log_first_crop = not getattr(self, "_logged_first_slide_debug", False)
        if log_first_crop:
            get_logger().info(
                "Slide inference debug: "
                f"image={tuple(img.shape)}, grids={h_grids}x{w_grids}, "
                f"crop={self.test_cfg.crop_size}, stride={self.test_cfg.stride}"
            )

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_start = time.perf_counter()
                if log_first_crop and h_idx == 0 and w_idx == 0:
                    get_logger().info(
                        "Slide inference debug: starting first crop "
                        f"shape={tuple(crop_img.shape)}"
                    )
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                if log_first_crop and h_idx == 0 and w_idx == 0:
                    get_logger().info(
                        "Slide inference debug: first crop finished in "
                        f"{time.perf_counter() - crop_start:.2f}s"
                    )
                    self._logged_first_slide_debug = True
                crop_preds = (
                    crop_seg_logit["positive"]
                    if self.sg_prediction_enabled
                    else crop_seg_logit
                )
                preds += F.pad(
                    crop_preds,
                    (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)),
                )
                if self.sg_prediction_enabled:
                    raw_preds += F.pad(
                        crop_seg_logit["raw"],
                        (
                            int(x1),
                            int(raw_preds.shape[3] - x2),
                            int(y1),
                            int(raw_preds.shape[2] - y2),
                        ),
                    )
                    ignore_preds += F.pad(
                        crop_seg_logit["ignore"],
                        (
                            int(x1),
                            int(ignore_preds.shape[3] - x2),
                            int(y1),
                            int(ignore_preds.shape[2] - y2),
                        ),
                    )
                count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        if self.sg_prediction_enabled:
            raw_preds = raw_preds / count_mat
            ignore_preds = ignore_preds / count_mat
        if rescale:
            resize_shape = img_meta[0]['img_shape'][:2]
            preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
            preds = F.interpolate(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='nearest' if self.sg_prediction_enabled else 'bilinear',
                align_corners=None if self.sg_prediction_enabled else self.align_corners,
            )
            if self.sg_prediction_enabled:
                raw_preds = raw_preds[:, :, :resize_shape[0], :resize_shape[1]]
                raw_preds = F.interpolate(
                    raw_preds,
                    size=img_meta[0]['ori_shape'][:2],
                    mode='bilinear',
                    align_corners=self.align_corners,
                )
                ignore_preds = ignore_preds[:, :, :resize_shape[0], :resize_shape[1]]
                ignore_preds = F.interpolate(
                    ignore_preds,
                    size=img_meta[0]['ori_shape'][:2],
                    mode='nearest',
                )
        if self.sg_prediction_enabled:
            return {"positive": preds, "ignore": ignore_preds, "raw": raw_preds}
        return preds

    def whole_inference(self, img, img_meta, rescale):
        img_meta = unwrap_datacontainer(img_meta)
        seg_logit = self.encode_decode(img, img_meta)
        if self.sg_prediction_enabled:
            positive = seg_logit["positive"]
            ignore = seg_logit["ignore"]
            raw = seg_logit["raw"]
            if rescale:
                resize_shape = img_meta[0]['img_shape'][:2]
                positive = positive[:, :, :resize_shape[0], :resize_shape[1]]
                ignore = ignore[:, :, :resize_shape[0], :resize_shape[1]]
                raw = raw[:, :, :resize_shape[0], :resize_shape[1]]
                positive = F.interpolate(
                    positive,
                    size=img_meta[0]['ori_shape'][:2],
                    mode='nearest',
                )
                ignore = F.interpolate(
                    ignore,
                    size=img_meta[0]['ori_shape'][:2],
                    mode='nearest',
                )
                raw = F.interpolate(
                    raw,
                    size=img_meta[0]['ori_shape'][:2],
                    mode='bilinear',
                    align_corners=self.align_corners,
                )
            return {"positive": positive, "ignore": ignore, "raw": raw}
        if rescale:
            resize_shape = img_meta[0]['img_shape'][:2]
            seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
            seg_logit = F.interpolate(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
            )
        return seg_logit

    def inference(self, img, img_meta, rescale):
        assert self.test_cfg.mode in ['slide', 'whole']
        img_meta = unwrap_datacontainer(img_meta)
        img = img.to(self.text_embedding.device, non_blocking=True)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)

        if self.sg_prediction_enabled:
            output = seg_logit
            output["pamr_image"] = img[:, [2, 1, 0], :, :]
        else:
            output = F.softmax(seg_logit, dim=1)
        if img_meta[0]['flip']:
            flip_direction = img_meta[0]['flip_direction']
            if flip_direction == 'horizontal':
                if self.sg_prediction_enabled:
                    output = {key: value.flip(dims=(3,)) for key, value in output.items()}
                else:
                    output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                if self.sg_prediction_enabled:
                    output = {key: value.flip(dims=(2,)) for key, value in output.items()}
                else:
                    output = output.flip(dims=(2,))
            else:
                raise ValueError(f"Unsupported flip direction: {flip_direction}")
        return output

    def compile_sg_prediction(self, output):
        positive_strength, positive_class = output["positive"].max(dim=1)
        sg_prior = output["positive"]
        has_positive = positive_strength > 0
        soft_cfg = self.sg_gate.get("soft_reweight", {})
        soft_enabled = bool(soft_cfg.get("enabled", False))
        disable_hard_overwrite = bool(soft_cfg.get("disable_hard_overwrite", False))
        if (
            self.sg_gate.get("dense_raw_fallback", {}).get("enabled", False)
            and "raw" in output
        ):
            raw_scores = output["raw"]
            raw_label = raw_scores.argmax(dim=1)
            if soft_enabled:
                candidate_source = str(soft_cfg.get("candidate_source", "trusted_clusters"))
                sg_prior = sg_prior.clamp(0.0, 1.0) \
                    if bool(soft_cfg.get("normalize_prior", True)) else sg_prior
                shuffle_cfg = self.sg_gate.get("shuffle_prior", {})
                shuffle_prior_enabled = bool(shuffle_cfg.get("enabled", False))
                shuffle_prior_mode = str(shuffle_cfg.get("mode", "class_permutation"))
                shuffle_prior_seed = int(shuffle_cfg.get("seed", 42))
                if shuffle_prior_enabled:
                    if shuffle_prior_mode != "class_permutation":
                        raise ValueError(
                            "sg_gate.shuffle_prior.mode must be 'class_permutation', "
                            f"got {shuffle_prior_mode!r}"
                        )
                    shuffled_prior = sg_prior.clone()
                    start_channel = 1 if self.with_bg and sg_prior.shape[1] > 1 else 0
                    shuffle_channels = sg_prior.shape[1] - start_channel
                    if shuffle_channels > 1:
                        for batch_index in range(sg_prior.shape[0]):
                            generator = torch.Generator(device="cpu")
                            generator.manual_seed(
                                shuffle_prior_seed
                                + self._sg_shuffle_index
                                + batch_index
                            )
                            perm = torch.randperm(
                                shuffle_channels,
                                generator=generator,
                            ).to(sg_prior.device) + start_channel
                            shuffled_prior[batch_index, start_channel:] = (
                                sg_prior[batch_index, perm]
                            )
                    self._sg_shuffle_index += sg_prior.shape[0]
                    sg_prior = shuffled_prior
                lambda_boost = float(soft_cfg.get("lambda_boost", 0.10))
                topk_guard = int(soft_cfg.get("topk_guard", 5))
                margin_guard = float(soft_cfg.get("margin_guard", 0.05))
                guard_mode = str(soft_cfg.get("guard_mode", "current_or"))
                uncertainty_margin = float(soft_cfg.get("uncertainty_margin", 0.03))
                num_classes = raw_scores.shape[1]
                k = max(1, min(topk_guard, num_classes))
                random_prior_density_pixels = 0
                raw_topk_prior_density_pixels = 0
                if candidate_source == "random_prior":
                    random_cfg = self.sg_gate.get("random_prior", {})
                    random_seed = int(random_cfg.get("seed", 42))
                    density = float(
                        random_cfg.get(
                            "density",
                            float(self.sg_gate.get("top_percent", 3.0)) / 100.0,
                        )
                    )
                    density = max(0.0, min(1.0, density))
                    sg_prior = raw_scores.new_zeros(raw_scores.shape)
                    for batch_index in range(raw_scores.shape[0]):
                        generator = torch.Generator(device="cpu")
                        generator.manual_seed(
                            random_seed + self._sg_shuffle_index + batch_index
                        )
                        random_mask = torch.rand(
                            raw_scores.shape[1:],
                            generator=generator,
                            device="cpu",
                        ).to(raw_scores.device) < density
                        random_values = torch.rand(
                            raw_scores.shape[1:],
                            generator=generator,
                            device="cpu",
                        ).to(raw_scores.device)
                        sg_prior[batch_index] = random_mask.float() * random_values
                    self._sg_shuffle_index += raw_scores.shape[0]
                    random_prior_density_pixels = int((sg_prior > 0).any(dim=1).sum().item())
                elif candidate_source == "raw_topk":
                    raw_topk_indices = raw_scores.topk(k=k, dim=1).indices
                    sg_prior = torch.zeros_like(raw_scores)
                    sg_prior.scatter_(1, raw_topk_indices, 1.0)
                    raw_topk_prior_density_pixels = int((sg_prior > 0).any(dim=1).sum().item())
                topk_indices = raw_scores.topk(k=k, dim=1).indices
                topk_mask = torch.zeros_like(raw_scores, dtype=torch.bool)
                topk_mask.scatter_(1, topk_indices, True)
                top2 = raw_scores.topk(k=min(2, num_classes), dim=1).values
                raw_top1 = top2[:, :1]
                if num_classes > 1:
                    uncertainty = raw_top1 - top2[:, 1:2]
                else:
                    uncertainty = torch.zeros_like(raw_top1)
                uncertain_mask = uncertainty <= uncertainty_margin
                if guard_mode == "topk_and_uncertain":
                    boost_mask = (sg_prior > 0) & topk_mask & uncertain_mask
                else:
                    guard_mode = "current_or"
                    margin_mask = raw_scores >= (raw_top1 - margin_guard)
                    boost_mask = (sg_prior > 0) & (topk_mask | margin_mask)
                score_scale_type = str(soft_cfg.get("score_scale", "std"))
                if score_scale_type == "std":
                    score_scale = raw_scores.std(dim=1, keepdim=True, unbiased=False) + 1e-6
                else:
                    score_scale_type = "constant"
                    score_scale = 1.0
                adjusted_scores = raw_scores.clone()
                adjusted_scores = adjusted_scores + (
                    lambda_boost * sg_prior * score_scale * boost_mask.float()
                )
                max_abs_diff_before_pamr = float(
                    (adjusted_scores - raw_scores).abs().max().item()
                )
                apply_pamr_after_soft = bool(soft_cfg.get("apply_pamr_after_soft", False))
                debug_path_a = None
                debug_path_b = None
                if apply_pamr_after_soft:
                    pamr_image = output.get("pamr_image", None)
                    if pamr_image is None:
                        raise ValueError(
                            "apply_pamr_after_soft=True requires pamr_image in SG output"
                        )
                    if self.true_pamr_dual_path_debug:
                        path_a_input = raw_scores.clone()
                        path_b_input = adjusted_scores.clone()
                        path_a_pamr_scores = self._apply_pamr_in_chunks(
                            pamr_image,
                            path_a_input,
                        )
                        path_b_pamr_scores = self._apply_pamr_in_chunks(
                            pamr_image,
                            path_b_input,
                        )
                        debug_path_a = path_a_pamr_scores.argmax(dim=1)
                        debug_path_b = path_b_pamr_scores.argmax(dim=1)
                        output_diff = (path_a_pamr_scores - path_b_pamr_scores).abs()
                        label_disagreement = debug_path_a != debug_path_b
                        self._sg_stats["pamr_dual_path_debug_enabled"] = True
                        self._sg_stats["raw_scores_shape"] = str(tuple(raw_scores.shape))
                        self._sg_stats["path_a_pamr_input_shape"] = str(tuple(path_a_input.shape))
                        self._sg_stats["path_b_pamr_input_shape"] = str(tuple(path_b_input.shape))
                        self._sg_stats["raw_scores_dtype"] = str(raw_scores.dtype)
                        self._sg_stats["path_a_input_dtype"] = str(path_a_input.dtype)
                        self._sg_stats["path_b_input_dtype"] = str(path_b_input.dtype)
                        self._sg_stats["max_abs_diff_raw_vs_path_a_input"] = float(
                            (raw_scores - path_a_input).abs().max().item()
                        )
                        self._sg_stats["max_abs_diff_raw_vs_path_b_input"] = float(
                            (raw_scores - path_b_input).abs().max().item()
                        )
                        self._sg_stats["max_abs_diff_path_a_input_vs_path_b_input"] = float(
                            (path_a_input - path_b_input).abs().max().item()
                        )
                        self._sg_stats[
                            "max_abs_diff_path_a_pamr_output_vs_path_b_pamr_output"
                        ] = float(output_diff.max().item())
                        self._sg_stats[
                            "mean_abs_diff_path_a_pamr_output_vs_path_b_pamr_output_sum"
                        ] = float(output_diff.mean().item())
                        self._sg_stats[
                            "mean_abs_diff_path_a_pamr_output_vs_path_b_pamr_output_count"
                        ] = 1
                        self._sg_stats["final_label_disagreement_pixels"] = int(
                            label_disagreement.sum().item()
                        )
                        self._sg_stats["final_label_disagreement_total"] = int(
                            label_disagreement.numel()
                        )
                        self._sg_stats.update(
                            self._pamr_debug_config(
                                pamr_image.shape,
                                raw_scores.shape,
                            )
                        )
                        adjusted_scores = path_b_pamr_scores
                    else:
                        adjusted_scores = self._apply_pamr_in_chunks(
                            pamr_image,
                            adjusted_scores,
                        )
                strict = adjusted_scores.argmax(dim=1)
                if debug_path_b is None and self.true_pamr_dual_path_debug:
                    debug_path_b = strict.clone()
                has_positive = boost_mask.any(dim=1)
                prior_pixels = (sg_prior > 0).any(dim=1)
                changed_pixels = strict != raw_label
                self._sg_stats["soft_prior_pixels"] = int(prior_pixels.sum().item())
                self._sg_stats["boosted_pixels"] = int(has_positive.sum().item())
                self._sg_stats["changed_pixels_vs_raw"] = int(changed_pixels.sum().item())
                self._sg_stats["raw_preserved_pixels"] = int((~changed_pixels).sum().item())
                self._sg_stats["dense_total_pixels"] = int(raw_label.numel())
                self._sg_stats["sg_overwrite_pixels"] = 0
                self._sg_stats["trusted_sg_pixel_overwrite_sum"] = int(prior_pixels.sum().item())
                self._sg_stats["score_scale_type"] = score_scale_type
                self._sg_stats["guard_mode"] = guard_mode
                self._sg_stats["uncertainty_margin"] = uncertainty_margin
                self._sg_stats["uncertain_pixels"] = int(uncertain_mask[:, 0].sum().item())
                self._sg_stats["apply_pamr_after_soft"] = apply_pamr_after_soft
                self._sg_stats["pamr_source"] = (
                    "true_dual_path_debug"
                    if self.true_pamr_dual_path_debug
                    else "post_soft_pamr_branch"
                    if apply_pamr_after_soft
                    else "not_available"
                )
                self._sg_stats["shuffle_prior_enabled"] = shuffle_prior_enabled
                self._sg_stats["shuffle_prior_mode"] = (
                    shuffle_prior_mode if shuffle_prior_enabled else "not_available"
                )
                self._sg_stats["shuffle_prior_seed"] = (
                    shuffle_prior_seed if shuffle_prior_enabled else "not_available"
                )
                self._sg_stats["max_abs_diff_before_pamr"] = max_abs_diff_before_pamr
                self._sg_stats["random_prior_density_pixels"] = random_prior_density_pixels
                self._sg_stats["raw_topk_prior_density_pixels"] = raw_topk_prior_density_pixels
            else:
                strict = raw_label.clone()
                if not disable_hard_overwrite:
                    strict[has_positive] = positive_class[has_positive]
            ignored = torch.zeros_like(has_positive, dtype=torch.bool)
        else:
            strict = torch.zeros_like(positive_class)
            strict[has_positive] = positive_class[has_positive]
            ignored = ~has_positive & (output["ignore"][:, 0] > 0)

        diagnostic = strict.clone()
        diagnostic[ignored] = 255
        debug_predictions = {}
        if "debug_path_a" in locals() and debug_path_a is not None:
            debug_predictions["pamr_path_a"] = debug_path_a
        if "debug_path_b" in locals() and debug_path_b is not None:
            debug_predictions["pamr_path_b"] = debug_path_b
        return strict, diagnostic, ignored, has_positive, debug_predictions

    def build_sg_result(
        self,
        strict,
        diagnostic,
        ignored,
        has_positive,
        index,
        debug_predictions=None,
    ):
        stats = self.consume_sg_stats()
        positive_pixels = int(has_positive[index].sum().item())
        total_pixels = int(ignored[index].numel())
        ignore_before = int((diagnostic[index] == 255).sum().item())
        stats.setdefault("sg_overwrite_pixels", positive_pixels)
        stats.setdefault("trusted_sg_pixel_overwrite_sum", positive_pixels)
        stats.setdefault("raw_preserved_pixels", total_pixels - positive_pixels)
        stats.setdefault("dense_total_pixels", total_pixels)
        result = {
            "strict": strict[index].cpu().numpy(),
            "diagnostic": diagnostic[index].cpu().numpy(),
            "ignore_pixels": int(ignored[index].sum().item()),
            "total_pixels": total_pixels,
            "positive_pixels": positive_pixels,
            "ignore_pixels_after_compile": ignore_before,
            "timing": self.consume_sg_timing(),
            "stats": stats,
        }
        for key, value in (debug_predictions or {}).items():
            result[key] = value[index].cpu().numpy()
        return result

    def simple_test(self, img, img_meta, rescale=True):
        if self.sg_gate_enabled:
            self.reset_sg_timing()
        output = self.inference(img, img_meta, rescale)
        if not self.sg_gate_enabled:
            seg_pred = output.argmax(dim=1)
            return list(seg_pred.cpu().numpy())
        if self.debug_force_raw_output:
            seg_pred = output.argmax(dim=1)
            return list(seg_pred.cpu().numpy())

        strict, diagnostic, ignored, has_positive, debug_predictions = (
            self.compile_sg_prediction(output)
        )
        return [
            self.build_sg_result(
                strict,
                diagnostic,
                ignored,
                has_positive,
                index,
                debug_predictions,
            )
            for index in range(strict.shape[0])
        ]

    def msa_test(self, imgs, img_metas, rescale=True):
        if not rescale:
            raise ValueError("MSA requires rescale=True")
        started = time.perf_counter()
        variants_per_scale = 2 if self.msa_hflip else 1
        expected = len(self.msa_scales) * variants_per_scale
        if len(imgs) != expected:
            raise ValueError(f"MSA expected {expected} inputs, got {len(imgs)}")
        if self.msa_scale_weights is None:
            weights = [1.0 / len(self.msa_scales) for _ in self.msa_scales]
        else:
            weight_sum = sum(self.msa_scale_weights)
            weights = [value / weight_sum for value in self.msa_scale_weights]

        final = None
        cursor = 0
        for index, weight in enumerate(weights):
            variants = []
            for _ in range(variants_per_scale):
                variants.append(
                    self.inference(
                        imgs[cursor],
                        img_metas[cursor],
                        rescale=True,
                    )
                )
                cursor += 1
            scale_output, _ = aggregate_legacy_probability_maps(
                variants,
                num_scales=1,
                hflip=self.msa_hflip,
            )
            if len(self.msa_scales) == 1:
                final = scale_output
            elif final is None:
                final = weight * scale_output
            else:
                final.add_(scale_output, alpha=weight)

            if self.msa_scales == [1.0] and not self.msa_hflip:
                difference = float(
                    (final - variants[0]).abs().max().detach().cpu()
                )
                self._msa_single_scale_max_abs_diff = max(
                    self._msa_single_scale_max_abs_diff,
                    difference,
                )
                if difference > 1e-7:
                    raise AssertionError(
                        "MSA scales=[1.0] changed the legacy output map: "
                        f"max_abs_diff={difference:.9g}"
                    )
            if self.msa_log_per_scale_stats:
                values = scale_output.float()
                self._msa_per_scale_mean[index] += float(
                    values.mean().detach().cpu()
                )
                self._msa_per_scale_std[index] += float(
                    values.std(unbiased=False).detach().cpu()
                )
        self._msa_images += int(final.shape[0])
        self._msa_forward_passes += expected
        self._msa_time_total += time.perf_counter() - started
        prediction = final.argmax(dim=1)
        return list(prediction.cpu().numpy())

    def aug_test(self, imgs, img_metas, rescale=True):
        assert rescale
        if self.sg_gate_enabled:
            self.reset_sg_timing()
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            next_logit = self.inference(imgs[i], img_metas[i], rescale)
            if self.sg_prediction_enabled:
                for key in seg_logit:
                    seg_logit[key] += next_logit[key]
            else:
                seg_logit += next_logit
        if not self.sg_gate_enabled:
            seg_pred = (seg_logit / len(imgs)).argmax(dim=1)
            return list(seg_pred.cpu().numpy())
        if self.debug_force_raw_output:
            seg_pred = (seg_logit / len(imgs)).argmax(dim=1)
            return list(seg_pred.cpu().numpy())

        for key in seg_logit:
            seg_logit[key] /= len(imgs)
        strict, diagnostic, ignored, has_positive, debug_predictions = (
            self.compile_sg_prediction(seg_logit)
        )
        return [
            self.build_sg_result(
                strict,
                diagnostic,
                ignored,
                has_positive,
                index,
                debug_predictions,
            )
            for index in range(strict.shape[0])
        ]

    def forward_test(self, imgs, img_metas, **kwargs):
        img_metas = unwrap_datacontainer(img_metas)
        if isinstance(img_metas, (list, tuple)):
            img_metas = [unwrap_datacontainer(meta) for meta in img_metas]

        if len(imgs) != len(img_metas):
            raise ValueError(
                f'num of augmentations ({len(imgs)}) != num of image meta ({len(img_metas)})'
            )
        if self.msa_enabled:
            return self.msa_test(imgs, img_metas, **kwargs)
        if len(imgs) == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        return self.aug_test(imgs, img_metas, **kwargs)

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            raise NotImplementedError("DINOTextSegInference is evaluation-only")
        return self.forward_test(img, img_metas, **kwargs)
