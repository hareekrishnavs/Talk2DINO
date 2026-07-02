import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualPrototypeAlignment(nn.Module):
    def __init__(
        self,
        fusion_lambda=0.03,
        seed_temperature=1.0,
        class_topk_patches=20,
        max_classes=12,
        patch_topk=3,
        seed_selection="rank_percentile",
        use_min_seed_prob=False,
        min_seed_prob=0.0,
        seed_percentile=90,
        min_seed_pixels=8,
        max_seed_pixels=64,
        prototype_temperature=0.2,
        sim_clip=1.0,
        positive_only=True,
        correction_mode="positive_quantile",
        require_patch_topk=True,
        min_margin=0.0,
        use_margin_filter=False,
        debug_assert_changes=False,
        eps=1.0e-6,
    ):
        super().__init__()
        self.fusion_lambda = float(fusion_lambda)
        self.seed_temperature = float(seed_temperature)
        self.class_topk_patches = int(class_topk_patches)
        self.max_classes = int(max_classes)
        self.patch_topk = int(patch_topk)
        self.seed_selection = str(seed_selection)
        self.use_min_seed_prob = bool(use_min_seed_prob)
        self.min_seed_prob = float(min_seed_prob)
        self.seed_percentile = float(seed_percentile)
        self.min_seed_pixels = int(min_seed_pixels)
        self.max_seed_pixels = int(max_seed_pixels)
        self.prototype_temperature = float(prototype_temperature)
        self.sim_clip = float(sim_clip)
        self.positive_only = bool(positive_only)
        self.correction_mode = str(correction_mode)
        self.require_patch_topk = bool(require_patch_topk)
        self.min_margin = float(min_margin)
        self.use_margin_filter = bool(use_margin_filter)
        self.debug_assert_changes = bool(debug_assert_changes)
        self.eps = float(eps)

        if self.fusion_lambda < 0:
            raise ValueError("vpa.fusion_lambda must be non-negative")
        if self.seed_temperature <= 0:
            raise ValueError("vpa.seed_temperature must be positive")
        if self.prototype_temperature <= 0:
            raise ValueError("vpa.prototype_temperature must be positive")
        if not 0 <= self.min_seed_prob <= 1:
            raise ValueError("vpa.min_seed_prob must be in [0, 1]")
        if not 0 <= self.seed_percentile <= 100:
            raise ValueError("vpa.seed_percentile must be in [0, 100]")
        if min(
            self.class_topk_patches,
            self.max_classes,
            self.patch_topk,
            self.min_seed_pixels,
            self.max_seed_pixels,
        ) < 1:
            raise ValueError("VPA top-k and seed-count settings must be positive")
        if self.max_seed_pixels < self.min_seed_pixels:
            raise ValueError("vpa.max_seed_pixels must be >= vpa.min_seed_pixels")
        if self.sim_clip <= 0:
            raise ValueError("vpa.sim_clip must be positive")
        if self.seed_selection != "rank_percentile":
            raise ValueError(
                "VPA-v3 only supports vpa.seed_selection=rank_percentile"
            )
        if not self.positive_only:
            raise ValueError("VPA-v3 requires vpa.positive_only=true")
        if self.correction_mode != "positive_quantile":
            raise ValueError(
                "VPA-v3 only supports vpa.correction_mode=positive_quantile"
            )

    @staticmethod
    def _empty_stats(logits, batch_size):
        zero = logits.new_tensor(0.0)
        return {
            "vpa_called_images": logits.new_tensor(float(batch_size)),
            "vpa_valid_classes_mean": zero,
            "vpa_valid_classes_min": zero,
            "vpa_valid_classes_max": zero,
            "vpa_seed_pixels_mean": zero,
            "vpa_seed_pixels_min": zero,
            "vpa_seed_pixels_max": zero,
            "vpa_class_score_mean": zero,
            "vpa_class_score_max": zero,
            "vpa_seed_score_mean": zero,
            "vpa_seed_score_min": zero,
            "vpa_seed_score_max": zero,
            "vpa_correction_abs_mean": zero,
            "vpa_correction_abs_max": zero,
            "vpa_correction_negative_fraction": zero,
            "vpa_changed_fraction": zero,
            "vpa_no_valid_prototype_images": zero,
            "vpa_lambda_zero_max_abs_diff": zero,
            "vpa_skip_low_prob_total": zero,
            "vpa_skip_not_topk_total": zero,
            "vpa_skip_too_few_pixels_total": zero,
            "vpa_skip_bad_prototype_total": zero,
        }

    def forward(
        self,
        input_logits,
        dense_patch_features,
        class_names=None,
    ):
        if dense_patch_features is None:
            raise RuntimeError(
                "VPA requires in-memory dense DINO patch features [B,P,D] from "
                "image_feat.flatten(2).transpose(1,2) in official evaluation"
            )
        if input_logits.dim() != 3:
            raise ValueError("VPA input_logits must be [B,C,P]")
        if dense_patch_features.dim() != 3:
            raise ValueError("VPA dense_patch_features must be [B,P,D]")
        batch_size, num_classes, num_patches = input_logits.shape
        if dense_patch_features.shape[:2] != (batch_size, num_patches):
            raise ValueError(
                "VPA logits/features disagree on [B,P]: "
                f"{tuple(input_logits.shape)} vs {tuple(dense_patch_features.shape)}"
            )
        if class_names is not None and len(class_names) != num_classes:
            raise ValueError("VPA class_names length must match the class dimension")

        if self.fusion_lambda == 0.0:
            final_logits = input_logits
            max_abs_diff = float(
                (final_logits.float() - input_logits.float()).abs().max().detach().cpu()
            )
            if max_abs_diff > 1e-7:
                raise RuntimeError(
                    "VPA lambda-zero invariant failed: "
                    f"max_abs_diff={max_abs_diff:.10f}"
                )
            return final_logits, self._empty_stats(input_logits, batch_size)

        logits = input_logits.float()
        features = F.normalize(dense_patch_features.float(), dim=-1, eps=self.eps)
        class_topk = min(self.class_topk_patches, num_patches)
        class_score = logits.topk(class_topk, dim=-1).values.mean(dim=-1)
        candidate_count = min(self.max_classes, num_classes)
        candidate_classes = class_score.topk(candidate_count, dim=1).indices
        candidate_scores = class_score.gather(1, candidate_classes)

        patch_topk = min(self.patch_topk, num_classes)
        patch_topk_indices = logits.topk(patch_topk, dim=1).indices
        patch_topk_mask = torch.zeros_like(logits, dtype=torch.bool).scatter(
            1,
            patch_topk_indices,
            True,
        )
        probs = None
        if self.use_min_seed_prob:
            probs = F.softmax(logits / self.seed_temperature, dim=1)
        margin = None
        if self.use_margin_filter:
            if num_classes < 2:
                margin = torch.full_like(logits[:, 0], float("inf"))
            else:
                top2 = logits.topk(2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]

        visual_correction = torch.zeros_like(logits)
        valid_counts = []
        seed_counts = []
        seed_score_tensors = []
        no_valid_images = 0
        skip_low_prob = 0
        skip_not_topk = 0
        skip_too_few_pixels = 0
        skip_bad_prototype = 0
        percentile = self.seed_percentile / 100.0
        for batch_index in range(batch_size):
            valid_for_image = 0
            for class_index in candidate_classes[batch_index].tolist():
                scores = logits[batch_index, class_index]
                allowed_patches = patch_topk_mask[batch_index, class_index]
                if not allowed_patches.any():
                    skip_not_topk += 1
                    continue

                threshold = torch.quantile(scores, percentile)
                seed_mask = scores >= threshold
                if self.require_patch_topk:
                    seed_mask = seed_mask & allowed_patches
                    if not seed_mask.any():
                        skip_not_topk += 1
                        continue
                if self.use_min_seed_prob:
                    seed_mask = seed_mask & (
                        probs[batch_index, class_index] >= self.min_seed_prob
                    )
                    if not seed_mask.any():
                        skip_low_prob += 1
                        continue
                if self.use_margin_filter:
                    seed_mask = seed_mask & (
                        margin[batch_index] >= self.min_margin
                    )

                seed_indices = seed_mask.nonzero(as_tuple=False).flatten()
                if seed_indices.numel() > self.max_seed_pixels:
                    selected = scores[seed_indices].topk(
                        self.max_seed_pixels
                    ).indices
                    seed_indices = seed_indices[selected]
                if seed_indices.numel() < self.min_seed_pixels:
                    skip_too_few_pixels += 1
                    continue

                seed_scores = scores[seed_indices]
                weights = F.softmax(
                    seed_scores / self.prototype_temperature,
                    dim=0,
                )
                if not torch.isfinite(weights).all():
                    skip_bad_prototype += 1
                    continue
                prototype = (
                    features[batch_index, seed_indices] * weights.unsqueeze(-1)
                ).sum(dim=0)
                prototype_norm = prototype.norm()
                if not torch.isfinite(prototype).all() or not torch.isfinite(
                    prototype_norm
                ):
                    skip_bad_prototype += 1
                    continue
                if float(prototype_norm) <= self.eps:
                    skip_bad_prototype += 1
                    continue

                prototype = F.normalize(prototype, dim=0, eps=self.eps)
                similarity = features[batch_index] @ prototype
                q50 = torch.quantile(similarity, 0.5)
                q90 = torch.quantile(similarity, 0.9)
                positive_similarity = (similarity - q50) / (
                    q90 - q50 + self.eps
                )
                positive_similarity = positive_similarity.clamp(
                    0.0,
                    self.sim_clip,
                ) / self.sim_clip
                if not torch.isfinite(positive_similarity).all():
                    skip_bad_prototype += 1
                    continue

                visual_correction[batch_index, class_index, allowed_patches] = (
                    positive_similarity[allowed_patches]
                )
                valid_for_image += 1
                seed_counts.append(float(seed_indices.numel()))
                seed_score_tensors.append(seed_scores)
            valid_counts.append(float(valid_for_image))
            if valid_for_image == 0:
                no_valid_images += 1

        scaled_correction = self.fusion_lambda * visual_correction
        final_logits = input_logits + scaled_correction.to(input_logits.dtype)
        valid_tensor = logits.new_tensor(valid_counts)
        seed_count_tensor = logits.new_tensor(seed_counts)
        seed_scores = (
            torch.cat(seed_score_tensors)
            if seed_score_tensors
            else logits.new_zeros(1)
        )
        negative_fraction = (scaled_correction < 0).float().mean()
        if float(negative_fraction.detach().cpu()) != 0.0:
            raise RuntimeError(
                "VPA-v3 positive-only invariant failed: correction contains "
                "negative values"
            )
        stats = {
            "vpa_called_images": logits.new_tensor(float(batch_size)),
            "vpa_valid_classes_mean": valid_tensor.mean(),
            "vpa_valid_classes_min": valid_tensor.min(),
            "vpa_valid_classes_max": valid_tensor.max(),
            "vpa_seed_pixels_mean": (
                seed_count_tensor.mean() if seed_counts else logits.new_tensor(0.0)
            ),
            "vpa_seed_pixels_min": (
                seed_count_tensor.min() if seed_counts else logits.new_tensor(0.0)
            ),
            "vpa_seed_pixels_max": (
                seed_count_tensor.max() if seed_counts else logits.new_tensor(0.0)
            ),
            "vpa_class_score_mean": candidate_scores.mean(),
            "vpa_class_score_max": candidate_scores.max(),
            "vpa_seed_score_mean": (
                seed_scores.mean() if seed_score_tensors else logits.new_tensor(0.0)
            ),
            "vpa_seed_score_min": (
                seed_scores.min() if seed_score_tensors else logits.new_tensor(0.0)
            ),
            "vpa_seed_score_max": (
                seed_scores.max() if seed_score_tensors else logits.new_tensor(0.0)
            ),
            "vpa_correction_abs_mean": scaled_correction.abs().mean(),
            "vpa_correction_abs_max": scaled_correction.abs().max(),
            "vpa_correction_negative_fraction": negative_fraction,
            "vpa_changed_fraction": (scaled_correction != 0).float().mean(),
            "vpa_no_valid_prototype_images": logits.new_tensor(
                float(no_valid_images)
            ),
            "vpa_lambda_zero_max_abs_diff": logits.new_tensor(0.0),
            "vpa_skip_low_prob_total": logits.new_tensor(float(skip_low_prob)),
            "vpa_skip_not_topk_total": logits.new_tensor(float(skip_not_topk)),
            "vpa_skip_too_few_pixels_total": logits.new_tensor(
                float(skip_too_few_pixels)
            ),
            "vpa_skip_bad_prototype_total": logits.new_tensor(
                float(skip_bad_prototype)
            ),
        }
        return final_logits, stats
