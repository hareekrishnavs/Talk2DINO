import math
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy import ndimage as scipy_ndimage
except ImportError:
    scipy_ndimage = None


@torch.no_grad()
def build_rvs_clip_text_features(
    class_names,
    templates,
    text_encoder,
    tokenizer,
    device,
    batch_size=32,
):
    class_names = [str(name) for name in class_names]
    templates = list(templates)
    if not templates or not all(isinstance(template, str) for template in templates):
        raise ValueError("rvs.text_templates must be a non-empty list of strings")
    prompts = [template.format(name) for name in class_names for template in templates]
    encoded = []
    for start in range(0, len(prompts), int(batch_size)):
        tokens = tokenizer(prompts[start:start + int(batch_size)]).to(device)
        encoded.append(text_encoder(tokens).float())
    features = torch.cat(encoded, dim=0).reshape(len(class_names), len(templates), -1)
    features = F.normalize(features, dim=-1).mean(dim=1)
    return F.normalize(features, dim=-1)


def _connected_components(mask):
    mask = np.asarray(mask, dtype=bool)
    if scipy_ndimage is not None:
        labels, count = scipy_ndimage.label(
            mask,
            structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
        )
        areas = np.bincount(labels.reshape(-1), minlength=count + 1)
        for index in range(1, count + 1):
            if areas[index]:
                yield labels == index
        return

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        ys = []
        xs = []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and mask[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))
        component = np.zeros_like(mask, dtype=bool)
        component[ys, xs] = True
        yield component


def extract_rvs_regions(
    tuned_logits,
    topk_indices,
    min_area_ratio=0.005,
    max_area_ratio=0.80,
    max_regions=20,
    min_confidence=0.0,
    candidate_topk=15,
):
    if tuned_logits.dim() != 4 or tuned_logits.shape[0] != 1:
        raise ValueError(
            f"RVS expects tuned logits [1,C,H,W], got {tuple(tuned_logits.shape)}"
        )
    min_area_ratio = float(min_area_ratio)
    max_area_ratio = float(max_area_ratio)
    if not 0.0 <= min_area_ratio <= max_area_ratio <= 1.0:
        raise ValueError("RVS region area ratios must satisfy 0 <= min <= max <= 1")
    if int(max_regions) < 1:
        raise ValueError("rvs.max_regions_per_image must be at least 1")

    logits = tuned_logits.float()
    max_logits, prediction = logits.max(dim=1)
    confidence_map = torch.exp(max_logits - torch.logsumexp(logits, dim=1))[0]
    confidence_map = confidence_map.detach().cpu()
    prediction = prediction[0].detach().cpu()
    topk_cpu = None
    if topk_indices is not None:
        topk_cpu = topk_indices[0].detach().to(torch.int16).cpu().numpy()
    num_classes = tuned_logits.shape[1]
    total_pixels = prediction.numel()
    discovered = 0
    regions = []
    for class_id in prediction.unique(sorted=True).tolist():
        class_mask = prediction.eq(int(class_id)).numpy()
        for component in _connected_components(class_mask):
            discovered += 1
            area_pixels = int(component.sum())
            area_ratio = area_pixels / max(1, total_pixels)
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            component_tensor = torch.from_numpy(component)
            confidence = float(confidence_map[component_tensor].mean())
            if confidence < float(min_confidence):
                continue
            ys, xs = np.nonzero(component)
            candidates = list(range(num_classes))
            if topk_cpu is not None:
                values = topk_cpu[:, component].reshape(-1).astype(np.int64)
                counts = np.bincount(values, minlength=num_classes)
                ranked = np.argsort(-counts, kind="stable")
                candidates = [
                    int(index)
                    for index in ranked
                    if int(counts[index]) > 0
                ][:max(1, int(candidate_topk))]
                if int(class_id) not in candidates:
                    candidates = [int(class_id)] + candidates
                    candidates = candidates[:max(1, int(candidate_topk))]
            regions.append({
                "class_id": int(class_id),
                "mask": component_tensor,
                "bbox_grid": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                ],
                "area_pixels": area_pixels,
                "area_ratio": area_ratio,
                "confidence": confidence,
                "candidate_ids": candidates,
            })
    regions.sort(
        key=lambda region: region["area_pixels"] * region["confidence"],
        reverse=True,
    )
    return regions[:int(max_regions)], discovered


def _summary(values, fallback):
    if not values:
        return float(fallback), float(fallback), float(fallback), 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return (
        float(tensor.mean()),
        float(tensor.min()),
        float(tensor.max()),
        float(tensor.std(unbiased=False)),
    )


class RegionVisualSemanticCPA:
    def __init__(self, crop_encoder, clip_text_features, class_names, config):
        self.crop_encoder = crop_encoder
        self.clip_text_features = F.normalize(clip_text_features.float(), dim=-1)
        self.class_names = list(class_names)
        self.cfg = config
        self._validate()

    def _validate(self):
        required = {
            "mode": "region_gate",
            "region_source": "tuned_cpa_argmax",
            "text_source": "clip_text",
            "candidate_text_source": "eval_class_names",
            "gate_type": "margin_tanh",
            "apply_to": "cpa_residual",
            "region_fill": "connected_component",
        }
        for key, expected in required.items():
            actual = str(self.cfg.get(key, expected))
            if actual != expected:
                raise ValueError(f"rvs.{key} must be `{expected}`, got `{actual}`")
        if str(self.cfg.compare_scope) not in {"candidate_topk", "all_classes"}:
            raise ValueError("rvs.compare_scope must be candidate_topk or all_classes")
        if str(self.cfg.margin_type) not in {"top2", "mean"}:
            raise ValueError("rvs.margin_type must be top2 or mean")
        if float(self.cfg.margin_temperature) <= 0.0:
            raise ValueError("rvs.margin_temperature must be positive")
        if not float(self.cfg.gate_min) <= 1.0 <= float(self.cfg.gate_max):
            raise ValueError("RVS gate range must contain 1.0")
        if float(self.cfg.fallback_gate) != 1.0:
            raise ValueError("rvs.fallback_gate must be 1.0 for baseline parity")
        if len(self.class_names) != self.clip_text_features.shape[0]:
            raise ValueError("RVS class names and CLIP text features do not match")
        if self.crop_encoder.output_dim != self.clip_text_features.shape[1]:
            raise ValueError(
                "CLIP crop/text dimensions do not match: "
                f"{self.crop_encoder.output_dim} vs {self.clip_text_features.shape[1]}"
            )

    def _synchronize(self):
        if self.crop_encoder.device.type == "cuda":
            torch.cuda.synchronize(self.crop_encoder.device)

    @torch.no_grad()
    def __call__(self, base_logits, tuned_logits, cpa_stats, rgb_image):
        total_started = time.perf_counter()
        cpa_residual = cpa_stats["cpa_residual"]
        reference = base_logits + cpa_residual
        reference_diff = float((reference - tuned_logits).abs().max().detach().cpu())
        if reference_diff > 1e-7:
            raise AssertionError(
                f"RVS input does not match tuned CPA logits: {reference_diff:.9g}"
            )
        if float(self.cfg.gate_strength) == 0.0:
            elapsed = time.perf_counter() - total_started
            return tuned_logits, {
                "rvs_regions": 0,
                "rvs_discovered_regions": 0,
                "rvs_valid_region_fraction": 0.0,
                "rvs_gate_values": [],
                "rvs_margin_values": [],
                "rvs_region_areas": [],
                "rvs_clip_crop_feature_norm_mean": 0.0,
                "rvs_clip_text_feature_norm_mean": float(
                    self.clip_text_features.norm(dim=-1).mean().cpu()
                ),
                "rvs_changed_pixel_fraction": 0.0,
                "rvs_changed_residual_fraction": 0.0,
                "rvs_gate_strength_zero_max_abs_diff": 0.0,
                "rvs_time_crop_encode": 0.0,
                "rvs_time_total": elapsed,
                "rvs_zero_strength_shortcut_used": True,
            }
        regions, discovered = extract_rvs_regions(
            tuned_logits,
            cpa_stats.get("cpa_topk_indices"),
            min_area_ratio=float(self.cfg.min_region_area_ratio),
            max_area_ratio=float(self.cfg.max_region_area_ratio),
            max_regions=int(self.cfg.max_regions_per_image),
            min_confidence=float(self.cfg.min_region_confidence),
            candidate_topk=int(self.cfg.topk),
        )
        if not regions:
            elapsed = time.perf_counter() - total_started
            return tuned_logits, {
                "rvs_regions": 0,
                "rvs_discovered_regions": discovered,
                "rvs_valid_region_fraction": 0.0,
                "rvs_gate_values": [],
                "rvs_margin_values": [],
                "rvs_region_areas": [],
                "rvs_clip_crop_feature_norm_mean": 0.0,
                "rvs_clip_text_feature_norm_mean": float(
                    self.clip_text_features.norm(dim=-1).mean().cpu()
                ),
                "rvs_changed_pixel_fraction": 0.0,
                "rvs_changed_residual_fraction": 0.0,
                "rvs_gate_strength_zero_max_abs_diff": 0.0,
                "rvs_time_crop_encode": 0.0,
                "rvs_time_total": elapsed,
                "rvs_zero_strength_shortcut_used": False,
            }

        image_shape = tuple(rgb_image.shape)
        if len(image_shape) != 3:
            raise ValueError(f"RVS RGB image must be 3D, got shape {image_shape}")
        if image_shape[-1] in (1, 3, 4):
            image_height, image_width = image_shape[:2]
        elif image_shape[0] in (1, 3, 4):
            image_height, image_width = image_shape[-2:]
        else:
            raise ValueError(f"RVS cannot infer RGB layout from shape {image_shape}")
        grid_height, grid_width = tuned_logits.shape[-2:]
        boxes = []
        image_masks = []
        for region in regions:
            x1, y1, x2, y2 = region["bbox_grid"]
            boxes.append([
                x1 * image_width / grid_width,
                y1 * image_height / grid_height,
                x2 * image_width / grid_width,
                y2 * image_height / grid_height,
            ])
            if self.crop_encoder.masked_crop:
                image_masks.append(
                    F.interpolate(
                        region["mask"].float()[None, None],
                        size=(image_height, image_width),
                        mode="nearest",
                    )[0, 0].bool()
                )

        self._synchronize()
        crop_started = time.perf_counter()
        crop_features = self.crop_encoder.encode(
            rgb_image,
            boxes,
            masks=(torch.stack(image_masks) if image_masks else None),
        )
        self._synchronize()
        crop_seconds = time.perf_counter() - crop_started
        if crop_features.shape[0] != len(regions):
            raise RuntimeError("RVS crop encoder changed the region count")
        if crop_features.shape[1] != self.clip_text_features.shape[1]:
            raise RuntimeError("RVS CLIP image/text feature dimensions differ")

        similarities = F.normalize(crop_features.float(), dim=-1) @ self.clip_text_features.T
        gate_map = torch.ones(
            (1, 1, grid_height, grid_width),
            device=base_logits.device,
            dtype=base_logits.dtype,
        )
        gates = []
        margins = []
        for index, region in enumerate(regions):
            class_id = region["class_id"]
            if str(self.cfg.compare_scope) == "candidate_topk":
                candidate_ids = list(region["candidate_ids"])
            else:
                candidate_ids = list(range(len(self.class_names)))
            competitor_ids = [value for value in candidate_ids if value != class_id]
            class_score = similarities[index, class_id]
            if not competitor_ids:
                margin = class_score.new_tensor(0.0)
            else:
                competitor_scores = similarities[index, competitor_ids]
                if str(self.cfg.margin_type) == "top2":
                    margin = class_score - competitor_scores.max()
                else:
                    margin = class_score - competitor_scores.mean()
            gate = 1.0 + float(self.cfg.gate_strength) * torch.tanh(
                margin / float(self.cfg.margin_temperature)
            )
            gate = gate.clamp(float(self.cfg.gate_min), float(self.cfg.gate_max))
            mask = region["mask"].to(device=base_logits.device)
            gate_map[0, 0][mask] = gate.to(base_logits.dtype)
            gates.append(float(gate.detach().cpu()))
            margins.append(float(margin.detach().cpu()))

        gated_residual = gate_map * cpa_residual
        final_logits = base_logits + gated_residual
        total_seconds = time.perf_counter() - total_started
        return final_logits, {
            "rvs_regions": len(regions),
            "rvs_discovered_regions": discovered,
            "rvs_valid_region_fraction": len(regions) / max(1, discovered),
            "rvs_gate_values": gates,
            "rvs_margin_values": margins,
            "rvs_region_areas": [region["area_ratio"] for region in regions],
            "rvs_clip_crop_feature_norm_mean": float(
                crop_features.float().norm(dim=-1).mean().detach().cpu()
            ),
            "rvs_clip_text_feature_norm_mean": float(
                self.clip_text_features.norm(dim=-1).mean().detach().cpu()
            ),
            "rvs_changed_pixel_fraction": float(
                gate_map.ne(1.0).float().mean().detach().cpu()
            ),
            "rvs_changed_residual_fraction": float(
                gated_residual.ne(cpa_residual).float().mean().detach().cpu()
            ),
            "rvs_gate_strength_zero_max_abs_diff": 0.0,
            "rvs_time_crop_encode": crop_seconds,
            "rvs_time_total": total_seconds,
            "rvs_zero_strength_shortcut_used": False,
        }


def summarize_rvs_values(values, fallback=0.0):
    return _summary(values, fallback)
