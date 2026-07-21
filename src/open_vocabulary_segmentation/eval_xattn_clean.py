import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
sys.path.insert(0, os.path.dirname(__file__))
from xattn_bridge_clean import (
    METHOD_NAME,
    build_eval_loader,
    build_coco_stuff_eval_dataset,
    build_frozen_talk2dino,
    load_bridge_from_checkpoint,
    load_clean_config,
)
from ic_cpa import AttentionSlotDynamicPrototypeHead


def parse_args():
    parser = argparse.ArgumentParser(METHOD_NAME + " evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--features",
        default=None,
        help="Cached eval feature directory. Required only with --cached_fast_eval.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opts", nargs="+", default=None)
    parser.add_argument(
        "--cached_fast_eval",
        action="store_true",
        help=(
            "Use the fast cached evaluator instead of official slide-inference parity. "
            "Fast cached numbers are useful for development but are not directly comparable "
            "to official Talk2DINO E0."
        ),
    )
    return parser.parse_args()


def load_all_eval_samples(feature_dir):
    feature_dir = Path(feature_dir)
    with open(feature_dir / "manifest.json", "r") as f:
        manifest = json.load(f)
    samples = []
    cache = {}
    for item in manifest["index"]:
        shard_name = item["shard"]
        if shard_name not in cache:
            cache[shard_name] = torch.load(
                feature_dir / shard_name,
                map_location="cpu",
                weights_only=False,
            )
        samples.append(cache[shard_name]["samples"][item["offset"]])
    return manifest, samples


@torch.no_grad()
def build_class_embeddings(model, cfg, classnames, device, keep_templates=False):
    text_tokens = model.build_dataset_class_tokens(cfg.evaluate.template, classnames)
    text_tokens = text_tokens.to(device)
    num_classes, num_templates = text_tokens.shape[:2]
    text = text_tokens.reshape(num_classes * num_templates, -1)
    chunks = []
    for i in range(0, text.shape[0], 32):
        chunks.append(model.encode_text(text[i:i + 32]).float())
    clip_templates = torch.cat(chunks, dim=0).reshape(num_classes, num_templates, -1)
    if keep_templates:
        base_templates = model._frozen_base_text_to_dino(
            clip_templates.reshape(num_classes * num_templates, -1)
        ).reshape(num_classes, num_templates, -1)
        return clip_templates.float(), base_templates.float()
    clip_emb = clip_templates.mean(dim=1)
    base_emb = model._frozen_base_text_to_dino(clip_emb)
    return clip_emb.float(), base_emb.float()


def bridge_class_embeddings(bridge, class_clip, patch_tokens, class_base, delta_scale=0.5):
    delta_scale = float(delta_scale)
    if class_clip.dim() == 3:
        num_classes, num_templates, clip_dim = class_clip.shape
        flat_clip = class_clip.reshape(num_classes * num_templates, clip_dim)
        flat_base = class_base.reshape(num_classes * num_templates, class_base.shape[-1])
        mapped, stats = bridge(flat_clip, patch_tokens, flat_base, return_stats=True)
        mapped = F.normalize(flat_base + delta_scale * stats["delta"].float(), dim=-1)
        if mapped.dim() == 3:
            if mapped.shape[0] != 1:
                raise ValueError(
                    "Template-aware eval expects one image/crop at a time; "
                    f"got mapped shape {tuple(mapped.shape)}"
                )
            mapped = mapped[0]
        mapped = mapped.reshape(num_classes, num_templates, -1).mean(dim=1)
        return F.normalize(mapped, dim=-1)
    mapped, stats = bridge(class_clip, patch_tokens, class_base, return_stats=True)
    if class_base.dim() == 3:
        mapped = F.normalize(class_base + delta_scale * stats["delta"].float(), dim=-1)
    else:
        mapped = F.normalize(class_base + delta_scale * stats["delta"].float(), dim=-1)
    if mapped.dim() == 3 and mapped.shape[0] == 1:
        mapped = mapped[0]
    return F.normalize(mapped, dim=-1)


def mean_template_embeddings(class_base):
    if class_base.dim() == 3:
        return F.normalize(class_base.mean(dim=1), dim=-1)
    return F.normalize(class_base, dim=-1)


def reject_msa_eval_config(cfg):
    evaluate_enabled = bool(cfg.evaluate.get("msa_enabled", False))
    msa_enabled = bool(cfg.get("msa", {}).get("enabled", False))
    if evaluate_enabled or msa_enabled:
        raise ValueError(
            "MSA/scale-aware inference has been removed from this method. "
            "Run the normal single-scale eval path without evaluate.msa_enabled "
            "or msa.enabled."
        )


def get_ic_cpa_cfg(cfg):
    defaults = OmegaConf.create({
        "enabled": False,
        "dino_dim": 768,
        "num_prototypes": 4,
        "beta": 1.0,
        "gamma": 0.010,
        "temperature": 0.07,
        "topk": 15,
        "residual_scale": 0.05,
        "residual_clip": 0.20,
        "return_aux_during_train": False,
        "out_proj_init": "identity",
        "out_proj_init_scale": 0.01,
    })
    return OmegaConf.merge(defaults, cfg.get("ic_cpa", {}))


def ic_cpa_enabled(cfg):
    return bool(cfg.evaluate.get("ic_cpa_enabled", False)) or bool(
        cfg.get("ic_cpa", {}).get("enabled", False)
    )


def checkpoint_ic_cpa_config(payload):
    if isinstance(payload.get("ic_cpa_config"), dict):
        return payload["ic_cpa_config"]
    config = payload.get("config", {})
    if isinstance(config, dict):
        return config.get("ic_cpa", {})
    return {}


def validate_ic_cpa_state_dict(ic_state):
    required_keys = {
        "prototype_slots",
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "out_proj.weight",
    }
    if ic_state is None:
        return False, sorted(required_keys)
    state_keys = set(ic_state.keys())
    missing = sorted(required_keys - state_keys)
    return len(missing) == 0, missing


def build_ic_cpa_from_payload(cfg, payload, device):
    enabled = ic_cpa_enabled(cfg)
    print(f"IC-CPA enabled: {enabled}", flush=True)
    print("MSA enabled: false", flush=True)
    checkpoint_key_terms = (
        "ic_cpa",
        "attention_slot",
        "prototype_slots",
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "dynamic_prototype",
        "slot",
    )
    payload_ic_keys = [
        str(key) for key in payload.keys()
        if any(term in str(key) for term in checkpoint_key_terms)
    ]
    ic_state = payload.get("ic_cpa")
    ic_state_keys = sorted(str(key) for key in ic_state.keys()) if ic_state else []
    print(
        "Checkpoint key scan: "
        f"total_top_level_keys={len(payload.keys())} "
        f"top_level_ic_cpa_matches={len(payload_ic_keys)} "
        f"nested_ic_cpa_keys={len(ic_state_keys)} "
        f"first_ic_cpa_keys={ic_state_keys[:10]}",
        flush=True,
    )
    if not enabled:
        if ic_state is not None:
            print(
                "Checkpoint contains IC-CPA weights, but ic_cpa.enabled=false; ignoring them.",
                flush=True,
            )
        print(
            "IC-CPA checkpoint weights loaded: false "
            f"(checkpoint_ic_cpa_keys={len(payload_ic_keys)})",
            flush=True,
        )
        return None
    ic_cfg = get_ic_cpa_cfg(cfg)
    ckpt_ic_cfg = checkpoint_ic_cpa_config(payload)
    if ckpt_ic_cfg:
        train_gamma = ckpt_ic_cfg.get("gamma", None)
        print(
            "IC-CPA checkpoint config: "
            f"enabled={ckpt_ic_cfg.get('enabled', None)} "
            f"K={ckpt_ic_cfg.get('num_prototypes', None)} "
            f"beta={ckpt_ic_cfg.get('beta', None)} "
            f"gamma={train_gamma} "
            f"temperature={ckpt_ic_cfg.get('temperature', None)} "
            f"topk={ckpt_ic_cfg.get('topk', None)} "
            f"residual_scale={ckpt_ic_cfg.get('residual_scale', None)} "
            f"residual_clip={ckpt_ic_cfg.get('residual_clip', None)}",
            flush=True,
        )
        if train_gamma is not None and abs(float(train_gamma) - float(ic_cfg.gamma)) > 1e-12:
            print(
                "WARNING: IC-CPA gamma mismatch between checkpoint config and eval config: "
                f"checkpoint_gamma={float(train_gamma):.6f} eval_gamma={float(ic_cfg.gamma):.6f}",
                flush=True,
            )
    kwargs = OmegaConf.to_container(ic_cfg, resolve=True)
    kwargs.pop("enabled", None)
    head = AttentionSlotDynamicPrototypeHead(**kwargs).to(device)
    print(
        "IC-CPA config: "
        f"enabled=true "
        f"K={int(ic_cfg.num_prototypes)}, "
        f"beta={float(ic_cfg.beta):.3f}, "
        f"gamma={float(ic_cfg.gamma):.3f}, "
        f"temperature={float(ic_cfg.temperature):.3f}, "
        f"topK={int(ic_cfg.topk)}, "
        f"residual_scale={float(ic_cfg.residual_scale):.3f}, "
        f"residual_clip={float(ic_cfg.residual_clip):.3f}",
        flush=True,
    )
    print(
        "IC-CPA initialization: small near-zero out_proj for base-parity; "
        "random prototype slot init for attention-slot specialization",
        flush=True,
    )
    valid_ic_state, missing_required = validate_ic_cpa_state_dict(ic_state)
    if not valid_ic_state:
        if ic_state is None:
            raise ValueError(
                "IC-CPA eval was requested, but the checkpoint has no top-level "
                "'ic_cpa' weights. Run with ic_cpa.enabled=false/evaluate.ic_cpa_enabled=false "
                "for baseline eval, or evaluate a checkpoint saved after IC-CPA training."
            )
        raise ValueError(
            "Checkpoint contains config-shaped 'ic_cpa' instead of IC-CPA "
            "weights. This checkpoint was saved before the checkpoint "
            "overwrite fix. Retrain or use a fixed checkpoint. Missing "
            f"required IC-CPA weight keys: {missing_required}. "
            f"Found keys: {ic_state_keys[:20]}"
        )
    missing, unexpected = head.load_state_dict(ic_state, strict=False)
    print(
        "IC-CPA checkpoint weights loaded: true "
        f"(num_ic_cpa_keys={len(ic_state_keys)}, "
        f"missing={list(missing)}, unexpected={list(unexpected)})",
        flush=True,
    )
    print(
        "IC-CPA checkpoint key sample: "
        f"{ic_state_keys[:8]}",
        flush=True,
    )
    norms = head.parameter_norms()
    print(
        "IC-CPA parameter norms: "
        f"slot_mean={norms['prototype_slots_mean']:.6f} "
        f"slot_std={norms['prototype_slots_std']:.6f} "
        f"slot_norm={norms['prototype_slots_norm']:.6f} "
        f"q={norms['q_proj_weight_norm']:.6f} "
        f"k={norms['k_proj_weight_norm']:.6f} "
        f"v={norms['v_proj_weight_norm']:.6f} "
        f"out={norms['out_proj_weight_norm']:.6f}",
        flush=True,
    )
    head.eval()
    return head


def fuse_xattn_logits(
    base_logits,
    xattn_logits,
    alpha=0.5,
    uncertainty_gate=False,
    margin_threshold=None,
):
    alpha = float(alpha)
    if uncertainty_gate:
        probs = F.softmax(base_logits, dim=1)
        top2 = probs.topk(k=min(2, probs.shape[1]), dim=1).values
        if top2.shape[1] < 2:
            gate = torch.ones_like(base_logits[:, :1])
        else:
            softmax_margin = top2[:, :1] - top2[:, 1:2]
            gate = (1.0 - softmax_margin).clamp(0.0, 1.0)
    elif margin_threshold is None:
        gate = 1.0
    else:
        top2 = base_logits.topk(k=min(2, base_logits.shape[1]), dim=1).values
        if top2.shape[1] < 2:
            margin = torch.zeros_like(base_logits[:, :1])
        else:
            margin = top2[:, :1] - top2[:, 1:2]
        gate = (margin <= float(margin_threshold)).to(base_logits.dtype)
    return base_logits + gate * alpha * (xattn_logits - base_logits)


def intersect_and_union(pred, gt, num_classes, ignore_index):
    mask = gt != ignore_index
    pred = pred[mask]
    gt = gt[mask]
    intersect = pred[pred == gt]
    area_intersect = torch.histc(intersect.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_pred = torch.histc(pred.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_gt = torch.histc(gt.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_union = area_pred + area_gt - area_intersect
    return area_intersect, area_union


def init_transition_diag():
    return {
        "pixels": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
        "correct_to_wrong": 0,
        "wrong_to_correct": 0,
        "wrong_to_wrong_changed": 0,
        "abs_diff_sum": 0.0,
        "abs_diff_count": 0,
        "abs_diff_max": 0.0,
        "modified_sum": 0.0,
        "images": 0,
    }


def update_transition_diag(diag, base_logits, final_logits, gt, ignore_index):
    base_logits = base_logits.detach().float().cpu()
    final_logits = final_logits.detach().float().cpu()
    gt = gt.cpu()
    base_pred = base_logits.argmax(dim=1)[0]
    final_pred = final_logits.argmax(dim=1)[0]
    valid = gt != int(ignore_index)
    if not bool(valid.any()):
        return
    base_valid = base_pred[valid]
    final_valid = final_pred[valid]
    gt_valid = gt[valid]
    base_correct = base_valid == gt_valid
    final_correct = final_valid == gt_valid
    unchanged = final_valid == base_valid
    diag["pixels"] += int(valid.sum())
    diag["unchanged_correct"] += int((base_correct & final_correct).sum())
    diag["unchanged_wrong"] += int((~base_correct & unchanged).sum())
    diag["correct_to_wrong"] += int((base_correct & ~final_correct).sum())
    diag["wrong_to_correct"] += int((~base_correct & final_correct).sum())
    diag["wrong_to_wrong_changed"] += int((~base_correct & ~final_correct & ~unchanged).sum())
    diff = (final_logits - base_logits).abs()
    diag["abs_diff_sum"] += float(diff.sum())
    diag["abs_diff_count"] += int(diff.numel())
    diag["abs_diff_max"] = max(diag["abs_diff_max"], float(diff.max()))
    diag["modified_sum"] += float((final_pred != base_pred).float().mean())
    diag["images"] += 1


def summarize_transition_diag(diag):
    pixels = max(1, int(diag["pixels"]))
    abs_count = max(1, int(diag["abs_diff_count"]))
    images = max(1, int(diag["images"]))
    return {
        "pixels": int(diag["pixels"]),
        "unchanged_correct_pct": 100.0 * diag["unchanged_correct"] / pixels,
        "unchanged_wrong_pct": 100.0 * diag["unchanged_wrong"] / pixels,
        "correct_to_wrong_pct": 100.0 * diag["correct_to_wrong"] / pixels,
        "wrong_to_correct_pct": 100.0 * diag["wrong_to_correct"] / pixels,
        "wrong_to_wrong_changed_pct": 100.0 * diag["wrong_to_wrong_changed"] / pixels,
        "mean_abs_final_minus_base": diag["abs_diff_sum"] / abs_count,
        "max_abs_final_minus_base": diag["abs_diff_max"],
        "modified_fraction": diag["modified_sum"] / images,
        "images": int(diag["images"]),
    }


class CleanOfficialEvalModel(nn.Module):
    def __init__(
        self,
        frozen,
        bridge,
        class_clip,
        class_base,
        ic_cpa=None,
        xattn_delta_scale=0.5,
        xattn_logit_alpha=0.5,
        xattn_uncertainty_gate_enabled=True,
        xattn_margin_gate_enabled=False,
        xattn_margin_threshold=0.05,
        ic_cpa_parity_check=False,
    ):
        super().__init__()
        self.frozen = frozen
        self.bridge = bridge
        self.ic_cpa = ic_cpa
        self.register_buffer("class_clip", class_clip.float())
        self.register_buffer("class_base", class_base.float())
        self.xattn_delta_scale = float(xattn_delta_scale)
        self.xattn_logit_alpha = float(xattn_logit_alpha)
        self.xattn_uncertainty_gate_enabled = bool(xattn_uncertainty_gate_enabled)
        self.xattn_margin_gate_enabled = bool(xattn_margin_gate_enabled)
        self.xattn_margin_threshold = float(xattn_margin_threshold)
        self.ic_cpa_parity_check = bool(ic_cpa_parity_check)
        self._logged_xattn_eval_path = False
        self._logged_ic_cpa_path = False
        self._ic_cpa_parity_checked = False
        self._ic_cpa_sum = {
            "residual_abs_mean": 0.0,
            "modified_fraction": 0.0,
            "prototype_base_abs_mean": 0.0,
            "final_base_abs_mean": 0.0,
            "prototype_vs_text_cos_mean": 0.0,
            "prototype_slot_pairwise_cos_mean": 0.0,
            "attention_pairwise_cos_mean": 0.0,
        }
        self._ic_cpa_residual_abs_max = 0.0
        self._ic_cpa_prototype_base_abs_max = 0.0
        self._ic_cpa_final_base_abs_max = 0.0
        self._ic_cpa_count = 0

    def ic_cpa_summary(self):
        if self.ic_cpa is None:
            return None
        count = max(1, self._ic_cpa_count)
        return {
            "enabled": True,
            "num_prototypes": int(self.ic_cpa.num_prototypes),
            "beta": float(self.ic_cpa.beta),
            "gamma": float(self.ic_cpa.gamma),
            "temperature": float(self.ic_cpa.temperature),
            "topk": int(self.ic_cpa.topk),
            "residual_scale": float(self.ic_cpa.residual_scale),
            "residual_clip": float(self.ic_cpa.residual_clip),
            "residual_abs_mean": self._ic_cpa_sum["residual_abs_mean"] / count,
            "residual_abs_max": self._ic_cpa_residual_abs_max,
            "modified_fraction": self._ic_cpa_sum["modified_fraction"] / count,
            "prototype_base_abs_mean": self._ic_cpa_sum["prototype_base_abs_mean"] / count,
            "prototype_base_abs_max": self._ic_cpa_prototype_base_abs_max,
            "final_base_abs_mean": self._ic_cpa_sum["final_base_abs_mean"] / count,
            "final_base_abs_max": self._ic_cpa_final_base_abs_max,
            "prototype_vs_text_cos_mean": self._ic_cpa_sum["prototype_vs_text_cos_mean"] / count,
            "prototype_slot_pairwise_cos_mean": self._ic_cpa_sum["prototype_slot_pairwise_cos_mean"] / count,
            "attention_pairwise_cos_mean": self._ic_cpa_sum["attention_pairwise_cos_mean"] / count,
        }

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.frozen, name)

    @torch.no_grad()
    def generate_masks(
        self,
        image,
        img_metas,
        text_emb,
        classnames,
        text_is_token=False,
        apply_pamr=False,
        background_func="weighted_average_sigmoid",
        lambda_bg=0.2,
        return_sg_inputs=False,
    ):
        H, W = image.shape[2:]
        pH, pW = image.shape[2:]
        image = image[:, [2, 1, 0], :, :]
        ori_image = image.clone()
        img_preprocessed = self.frozen.image_transforms(image).to(next(self.frozen.parameters()).device)
        if "dinov2" in self.frozen.model_name:
            image_feat = self.frozen.model.forward_features(img_preprocessed)["x_norm_patchtokens"]
        elif "dinov3" in self.frozen.model_name:
            image_feat = self.frozen.model.forward_features(img_preprocessed)[:, 5:, :]
        else:
            image_feat = self.frozen.model.forward_features(img_preprocessed)[:, 1:, :]

        batch_size, num_tokens, embed_dim = image_feat.shape
        self_attn, self_attn_maps = self.frozen.process_self_attention(
            self.frozen.feats["self_attn"],
            batch_size,
            num_tokens + self.frozen.num_global_tokens,
            self.frozen.num_attn_heads,
            embed_dim,
            self.frozen.scale,
            self.frozen.num_global_tokens,
            ret_self_attn_maps=True,
        )
        semantic_features = (
            image_feat.unsqueeze(1) * self_attn_maps.softmax(dim=-1).unsqueeze(-1)
        ).mean(dim=2)
        mapped_text = bridge_class_embeddings(
            self.bridge,
            self.class_clip.to(image_feat.device),
            semantic_features,
            self.class_base.to(image_feat.device),
            delta_scale=self.xattn_delta_scale,
        )
        if not self._logged_xattn_eval_path:
            from utils import get_logger

            get_logger().info(
                "Official eval XAttn path active: "
                f"class_clip={tuple(self.class_clip.shape)}, "
                f"semantic_xattn_kv={tuple(semantic_features.shape)}, "
                "raw_patch_tokens_as_xattn_kv=false, "
                f"class_base={tuple(self.class_base.shape)}, "
                f"mapped_text={tuple(mapped_text.shape)}, "
                "base_guided_attention="
                f"{bool(getattr(self.bridge, 'base_guided_attention', False))}, "
                f"beta={float(getattr(self.bridge, 'base_guidance_beta', 0.0)):.3f}, "
                f"delta_scale={self.xattn_delta_scale:.3f}, "
                f"logit_alpha={self.xattn_logit_alpha:.3f}, "
                f"uncertainty_gate={self.xattn_uncertainty_gate_enabled}, "
                f"margin_gate={self.xattn_margin_gate_enabled}, "
                f"margin_threshold={self.xattn_margin_threshold:.3f}"
            )
            self._logged_xattn_eval_path = True
        b, npatches, channels = image_feat.shape
        grid = int(npatches ** 0.5)
        image_feat = image_feat.reshape(b, grid, grid, channels).permute(0, 3, 1, 2)
        base_text = mean_template_embeddings(self.class_base.to(image_feat.device))
        _, base_simmap = self.frozen.masker.forward_seg(image_feat, base_text, hard=False)
        _, xattn_simmap = self.frozen.masker.forward_seg(image_feat, mapped_text, hard=False)
        margin_threshold = (
            self.xattn_margin_threshold
            if self.xattn_margin_gate_enabled
            else None
        )
        fused_base_simmap = fuse_xattn_logits(
            base_simmap,
            xattn_simmap,
            alpha=self.xattn_logit_alpha,
            uncertainty_gate=self.xattn_uncertainty_gate_enabled,
            margin_threshold=margin_threshold,
        )
        if self.ic_cpa is not None:
            spatial_features = image_feat.flatten(2).transpose(1, 2)
            simmap, ic_stats = self.ic_cpa(
                mapped_text,
                spatial_features,
                fused_base_simmap,
                return_stats=True,
            )
            if self.ic_cpa_parity_check and not self._ic_cpa_parity_checked:
                original_scale = self.ic_cpa.residual_scale
                self.ic_cpa.residual_scale = 0.0
                try:
                    zero_simmap = self.ic_cpa(
                        mapped_text,
                        spatial_features,
                        fused_base_simmap,
                    )
                finally:
                    self.ic_cpa.residual_scale = original_scale
                diff = (zero_simmap.float() - fused_base_simmap.float()).abs()
                mean_diff = float(diff.mean().detach().cpu())
                max_diff = float(diff.max().detach().cpu())
                print(
                    "IC-CPA zero-scale parity check: "
                    f"mean_abs_diff={mean_diff:.8e} max_abs_diff={max_diff:.8e}",
                    flush=True,
                )
                if mean_diff >= 1e-6 or max_diff >= 1e-5:
                    raise AssertionError(
                        "IC-CPA zero-scale parity failed against fused baseline: "
                        f"mean_abs_diff={mean_diff:.8e}, max_abs_diff={max_diff:.8e}"
                    )
                self._ic_cpa_parity_checked = True
            if not self._logged_ic_cpa_path:
                from utils import get_logger

                get_logger().info(
                    "IC-CPA active: "
                    f"text_feats={ic_stats['text_shape']} "
                    f"visual_feats={ic_stats['visual_shape']} "
                    f"prototypes={ic_stats['prototypes_shape']} "
                    f"base_logits={ic_stats['base_logits_shape']} "
                    f"final_logits={ic_stats['final_logits_shape']} "
                    "final_logits_used_for_argmax=true "
                    f"final_base_mean={float(ic_stats['final_base_abs_mean'].detach().cpu()):.6f} "
                    f"final_base_max={float(ic_stats['final_base_abs_max'].detach().cpu()):.6f} "
                    f"topk={int(self.ic_cpa.topk)} "
                    f"residual_scale={float(self.ic_cpa.residual_scale):.3f} "
                    f"residual_clip={float(self.ic_cpa.residual_clip):.3f}"
                )
                self._logged_ic_cpa_path = True
            self._ic_cpa_sum["residual_abs_mean"] += float(
                ic_stats["residual_abs_mean"].detach().cpu()
            )
            self._ic_cpa_sum["modified_fraction"] += float(
                ic_stats["modified_fraction"].detach().cpu()
            )
            self._ic_cpa_sum["prototype_base_abs_mean"] += float(
                ic_stats["prototype_base_abs_mean"].detach().cpu()
            )
            self._ic_cpa_sum["final_base_abs_mean"] += float(
                ic_stats["final_base_abs_mean"].detach().cpu()
            )
            self._ic_cpa_sum["prototype_vs_text_cos_mean"] += float(
                ic_stats["prototype_vs_text_cos_mean"].detach().cpu()
            )
            self._ic_cpa_sum["prototype_slot_pairwise_cos_mean"] += float(
                ic_stats["prototype_slot_pairwise_cos_mean"].detach().cpu()
            )
            self._ic_cpa_sum["attention_pairwise_cos_mean"] += float(
                ic_stats["attention_pairwise_cos_mean"].detach().cpu()
            )
            self._ic_cpa_residual_abs_max = max(
                self._ic_cpa_residual_abs_max,
                float(ic_stats["residual_abs_max"].detach().cpu()),
            )
            self._ic_cpa_prototype_base_abs_max = max(
                self._ic_cpa_prototype_base_abs_max,
                float(ic_stats["prototype_base_abs_max"].detach().cpu()),
            )
            self._ic_cpa_final_base_abs_max = max(
                self._ic_cpa_final_base_abs_max,
                float(ic_stats["final_base_abs_max"].detach().cpu()),
            )
            self._ic_cpa_count += 1
        else:
            simmap = fused_base_simmap
        mask = torch.sigmoid(simmap)
        if getattr(self.frozen, "with_bg_clean", False):
            mask = self.frozen.similarity_assignment_weighted(
                mask,
                image_feat,
                self_attn_maps,
                mapped_text,
                lambda_bg,
            )
        mask = F.interpolate(mask, (pH, pW), mode="bilinear", align_corners=True)
        if apply_pamr:
            for c in range(0, mask.shape[1], 30):
                mask[:, c:c + 30] = self.frozen.apply_pamr(ori_image, mask[:, c:c + 30])
        assert mask.shape[2] == H and mask.shape[3] == W
        if return_sg_inputs:
            return mask, simmap, image_feat
        return mask, simmap


@torch.no_grad()
def official_parity_eval(args, cfg, device):
    from segmentation.evaluation.dinotext_seg import DINOTextSegInference
    import mmcv
    import us

    reject_msa_eval_config(cfg)
    frozen = build_frozen_talk2dino(cfg, device)
    dataset = build_coco_stuff_eval_dataset(cfg)
    loader = build_eval_loader(dataset, cfg)
    classnames = dataset.CLASSES
    with_bg = classnames[0] == "background"
    eval_classnames = classnames[1:] if with_bg else classnames
    class_clip, class_base = build_class_embeddings(
        frozen,
        cfg,
        eval_classnames,
        device,
        keep_templates=True,
    )
    print(f"Evaluated checkpoint path: {args.checkpoint}", flush=True)
    bridge, payload = load_bridge_from_checkpoint(args.checkpoint, cfg, device)
    if payload.get("cpa") is not None:
        print(
            "Ignoring legacy CPA weights in checkpoint; old MLP CPA is inactive.",
            flush=True,
        )
    ic_cpa = build_ic_cpa_from_payload(cfg, payload, device)
    print("Running single-scale inference; MSA disabled.", flush=True)
    wrapped = CleanOfficialEvalModel(
        frozen,
        bridge,
        class_clip,
        class_base,
        ic_cpa=ic_cpa,
        xattn_delta_scale=float(cfg.evaluate.get("xattn_delta_scale", 0.5)),
        xattn_logit_alpha=float(cfg.evaluate.get("xattn_logit_alpha", 0.5)),
        xattn_uncertainty_gate_enabled=bool(
            cfg.evaluate.get("xattn_uncertainty_gate_enabled", True)
        ),
        xattn_margin_gate_enabled=bool(
            cfg.evaluate.get("xattn_margin_gate_enabled", False)
        ),
        xattn_margin_threshold=float(
            cfg.evaluate.get("xattn_margin_threshold", 0.05)
        ),
        ic_cpa_parity_check=bool(cfg.evaluate.get("ic_cpa_parity_check", False)),
    ).to(device)
    if bool(cfg.evaluate.get("ic_cpa_transition_diag", False)):
        print(
            "IC-CPA transition diagnostic is not available in official slide eval; "
            "use --cached_fast_eval for transition diagnostics.",
            flush=True,
        )
    seg_text_embedding = class_base.mean(dim=1) if class_base.dim() == 3 else class_base
    dset_cfg = mmcv.Config.fromfile(cfg.evaluate.coco_stuff)
    seg_model = DINOTextSegInference(
        wrapped,
        seg_text_embedding,
        eval_classnames,
        with_bg=with_bg,
        test_cfg=dset_cfg.test_cfg,
        pamr=bool(cfg.evaluate.pamr),
        bg_thresh=float(cfg.evaluate.get("bg_thresh", 0.4)),
        sg_gate={"enabled": False},
    ).to(device)
    seg_model.eval()
    results, _, _, _ = us.multi_gpu_test(
        model=seg_model,
        data_loader=loader,
        tmpdir=None,
        gpu_collect=device == "cuda",
        efficient_test=False,
        pre_eval=True,
        format_only=False,
        show_progress=bool(cfg.evaluate.get("show_progress", True)),
        progress_log_interval=int(cfg.evaluate.get("progress_log_interval", 0)),
        diagnostic_ignore_eval=True,
    )
    metric = dataset.evaluate(results, logger=None)
    miou = float(metric["mIoU"] * 100)
    return miou, payload, wrapped.ic_cpa_summary()


def init_eval_logger(cfg, out):
    from utils import get_logger

    logger_cfg = OmegaConf.create({
        "model_name": str(cfg.get("method_name", METHOD_NAME)),
        "output": str(out),
    })
    return get_logger(logger_cfg)


def main():
    args = parse_args()
    cfg = load_clean_config(args.config, args.opts)
    reject_msa_eval_config(cfg)
    if bool(cfg.evaluate.get("ccr_enabled", False)) or bool(cfg.get("ccr", {}).get("enabled", False)):
        raise ValueError("CCR is disabled for clean XAttn evaluation")
    if args.cached_fast_eval and not args.features:
        raise ValueError("--features is required when --cached_fast_eval is used")
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    init_eval_logger(cfg, out)

    if not args.cached_fast_eval:
        print("Eval mode: official Talk2DINO slide-inference parity", flush=True)
        print("Cached eval features are not used for prediction in this mode.", flush=True)
        miou, payload, ic_cpa_summary = official_parity_eval(
            args,
            cfg,
            device,
        )
        print("=" * 64, flush=True)
        print("EVALUATION RESULTS", flush=True)
        print("=" * 64, flush=True)
        mode_label = (
            f"{METHOD_NAME} + IC-CPA official slide eval"
            if ic_cpa_summary is not None
            else f"{METHOD_NAME} official parity"
        )
        print(f"Mode                 : {mode_label}", flush=True)
        print(f"coco_stuff mIoU      : {miou:.2f}%", flush=True)
        print(f"PAMR enabled         : {bool(cfg.evaluate.pamr)}", flush=True)
        print("CCR enabled          : false", flush=True)
        print(
            "XAttnBridge_Clean=true "
            f"CPA-v1=false IC-CPA={ic_cpa_summary is not None} MSA=false "
            "PAMR=false RVS=false CARS=false RCC=false VPA=false VAB=false "
            "OPC=false Router=false USRC=false CCR=false",
            flush=True,
        )
        if ic_cpa_summary is not None:
            print(
                "IC-CPA eval stats     : "
                f"num_prototypes={ic_cpa_summary['num_prototypes']} "
                f"beta={ic_cpa_summary['beta']:.3f} "
                f"gamma={ic_cpa_summary['gamma']:.3f} "
                f"temperature={ic_cpa_summary['temperature']:.3f} "
                f"topk={ic_cpa_summary['topk']} "
                f"residual_scale={ic_cpa_summary['residual_scale']:.3f} "
                f"residual_clip={ic_cpa_summary['residual_clip']:.3f} "
                f"mean_residual={ic_cpa_summary['residual_abs_mean']:.6f} "
                f"max_residual={ic_cpa_summary['residual_abs_max']:.6f} "
                f"modified_fraction={ic_cpa_summary['modified_fraction']:.4f} "
                f"proto_base_mean={ic_cpa_summary['prototype_base_abs_mean']:.6f} "
                f"proto_base_max={ic_cpa_summary['prototype_base_abs_max']:.6f} "
                f"final_base_mean={ic_cpa_summary['final_base_abs_mean']:.6f} "
                f"final_base_max={ic_cpa_summary['final_base_abs_max']:.6f} "
                f"proto_text_cos={ic_cpa_summary['prototype_vs_text_cos_mean']:.6f} "
                f"slot_cos={ic_cpa_summary['prototype_slot_pairwise_cos_mean']:.6f} "
                f"attn_cos={ic_cpa_summary['attention_pairwise_cos_mean']:.6f}",
                flush=True,
            )
        with open(out / "summary.json", "w") as f:
            json.dump({
                "method_name": METHOD_NAME,
                "eval_mode": "official_talk2dino_slide_parity",
                "official_comparable_to_talk2dino_e0": True,
                "cached_features_used_for_prediction": False,
                "pamr": bool(cfg.evaluate.pamr),
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": payload.get("epoch"),
                "coco_stuff_miou": miou,
                "cpa_enabled": False,
                "ic_cpa_enabled": ic_cpa_summary is not None,
                "ic_cpa_stats": ic_cpa_summary,
                "msa_enabled": False,
            }, f, indent=2)
        return

    frozen = build_frozen_talk2dino(cfg, device)
    dataset = build_coco_stuff_eval_dataset(cfg)
    print("Eval mode: cached fast eval, not official parity", flush=True)
    classnames = dataset.CLASSES
    clip_cls, base_cls = build_class_embeddings(
        frozen,
        cfg,
        classnames,
        device,
        keep_templates=True,
    )
    print(f"Evaluated checkpoint path: {args.checkpoint}", flush=True)
    bridge, payload = load_bridge_from_checkpoint(args.checkpoint, cfg, device)
    if payload.get("cpa") is not None:
        print(
            "Ignoring legacy CPA weights in checkpoint; old MLP CPA is inactive.",
            flush=True,
        )
    ic_cpa = build_ic_cpa_from_payload(cfg, payload, device)
    manifest, samples = load_all_eval_samples(args.features)
    base_text = mean_template_embeddings(base_cls.to(device))

    num_classes = len(classnames)
    total_inter = torch.zeros(num_classes)
    total_union = torch.zeros(num_classes)
    xattn_logit_alpha = float(cfg.evaluate.get("xattn_logit_alpha", 0.5))
    xattn_delta_scale = float(cfg.evaluate.get("xattn_delta_scale", 0.5))
    xattn_uncertainty_gate_enabled = bool(
        cfg.evaluate.get("xattn_uncertainty_gate_enabled", True)
    )
    xattn_margin_threshold = (
        float(cfg.evaluate.get("xattn_margin_threshold", 0.05))
        if bool(cfg.evaluate.get("xattn_margin_gate_enabled", False))
        else None
    )
    transition_diag_enabled = bool(cfg.evaluate.get("ic_cpa_transition_diag", False))
    transition_diag = init_transition_diag() if transition_diag_enabled else None
    parity_check_enabled = bool(cfg.evaluate.get("ic_cpa_parity_check", False))
    parity_checked = False
    for idx, sample in enumerate(samples):
        patches = sample["patch_tokens"].unsqueeze(0).to(device).float()
        mapped = bridge_class_embeddings(
            bridge,
            clip_cls,
            patches,
            base_cls,
            delta_scale=xattn_delta_scale,
        )
        patch_norm = F.normalize(patches, dim=-1)
        xattn_logits = torch.einsum("bnd,cd->bcn", patch_norm, mapped)
        base_logits = torch.einsum("bnd,cd->bcn", patch_norm, base_text)
        fused_base_logits = fuse_xattn_logits(
            base_logits,
            xattn_logits,
            alpha=xattn_logit_alpha,
            uncertainty_gate=xattn_uncertainty_gate_enabled,
            margin_threshold=xattn_margin_threshold,
        )
        if ic_cpa is not None:
            logits, ic_stats = ic_cpa(
                mapped,
                patches,
                fused_base_logits,
                return_stats=True,
            )
            if parity_check_enabled and not parity_checked:
                original_scale = ic_cpa.residual_scale
                ic_cpa.residual_scale = 0.0
                try:
                    zero_logits = ic_cpa(mapped, patches, fused_base_logits)
                finally:
                    ic_cpa.residual_scale = original_scale
                diff = (zero_logits.float() - fused_base_logits.float()).abs()
                mean_diff = float(diff.mean().detach().cpu())
                max_diff = float(diff.max().detach().cpu())
                print(
                    "IC-CPA zero-scale parity check: "
                    f"mean_abs_diff={mean_diff:.8e} max_abs_diff={max_diff:.8e}",
                    flush=True,
                )
                if mean_diff >= 1e-6 or max_diff >= 1e-5:
                    raise AssertionError(
                        "IC-CPA zero-scale parity failed against fused baseline: "
                        f"mean_abs_diff={mean_diff:.8e}, max_abs_diff={max_diff:.8e}"
                    )
                parity_checked = True
            if idx == 0:
                print(
                    "IC-CPA active: "
                    f"text_feats={ic_stats['text_shape']} "
                    f"visual_feats={ic_stats['visual_shape']} "
                    f"prototypes={ic_stats['prototypes_shape']} "
                    f"base_logits={ic_stats['base_logits_shape']} "
                    f"final_logits={ic_stats['final_logits_shape']} "
                    "final_logits_used_for_argmax=true "
                    f"final_base_mean={float(ic_stats['final_base_abs_mean'].detach().cpu()):.6f} "
                    f"final_base_max={float(ic_stats['final_base_abs_max'].detach().cpu()):.6f}",
                    flush=True,
                )
        else:
            logits = fused_base_logits
        n = logits.shape[-1]
        h = w = int(n ** 0.5)
        logits = logits[:, :, : h * w].reshape(1, num_classes, h, w)
        base_for_diag = fused_base_logits[:, :, : h * w].reshape(1, num_classes, h, w)
        gt = sample["gt"].long()
        logits = F.interpolate(logits, size=tuple(gt.shape), mode="bilinear", align_corners=False)
        if transition_diag is not None and ic_cpa is not None:
            base_for_diag = F.interpolate(
                base_for_diag,
                size=tuple(gt.shape),
                mode="bilinear",
                align_corners=False,
            )
            update_transition_diag(
                transition_diag,
                base_for_diag,
                logits,
                gt,
                int(dataset.ignore_index),
            )
        pred = logits.argmax(dim=1)[0].cpu()
        inter, union = intersect_and_union(pred, gt, num_classes, int(dataset.ignore_index))
        total_inter += inter
        total_union += union
        if idx % 50 == 0:
            print(f"Eval {idx}/{len(samples)}", flush=True)

    iou = total_inter / total_union.clamp_min(1)
    miou = float(torch.nanmean(iou) * 100.0)
    print("=" * 64, flush=True)
    print("EVALUATION RESULTS", flush=True)
    print("=" * 64, flush=True)
    print(f"Mode                 : {METHOD_NAME}", flush=True)
    print(f"coco_stuff mIoU      : {miou:.2f}%", flush=True)
    transition_summary = summarize_transition_diag(transition_diag) if transition_diag is not None else None
    if transition_summary is not None:
        print(
            "IC-CPA transition diag: "
            f"images={transition_summary['images']} "
            f"pixels={transition_summary['pixels']} "
            f"unchanged_correct={transition_summary['unchanged_correct_pct']:.2f}% "
            f"unchanged_wrong={transition_summary['unchanged_wrong_pct']:.2f}% "
            f"correct_to_wrong={transition_summary['correct_to_wrong_pct']:.2f}% "
            f"wrong_to_correct={transition_summary['wrong_to_correct_pct']:.2f}% "
            f"wrong_to_wrong_changed={transition_summary['wrong_to_wrong_changed_pct']:.2f}% "
            f"mean_abs_delta={transition_summary['mean_abs_final_minus_base']:.6f} "
            f"max_abs_delta={transition_summary['max_abs_final_minus_base']:.6f} "
            f"modified_fraction={transition_summary['modified_fraction']:.6f}",
            flush=True,
        )
    with open(out / "summary.json", "w") as f:
        json.dump({
            "method_name": METHOD_NAME,
            "eval_mode": "cached_fast_eval_not_official_parity",
            "official_comparable_to_talk2dino_e0": False,
            "cached_features_used_for_prediction": True,
            "pamr": bool(cfg.evaluate.pamr),
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": payload.get("epoch"),
            "coco_stuff_miou": miou,
            "ic_cpa_enabled": ic_cpa is not None,
            "ic_cpa_transition_diag": transition_summary,
        }, f, indent=2)


if __name__ == "__main__":
    main()
