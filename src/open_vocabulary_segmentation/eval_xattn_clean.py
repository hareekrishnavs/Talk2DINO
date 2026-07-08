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
from class_prototype_alignment import (
    ClassPrototypeAlignmentHead,
    apply_topk_prototype_residual,
    compute_prototype_logits,
)


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


def load_cpa_from_payload(cfg, payload, device, enabled):
    has_weights = payload.get("cpa") is not None
    print(
        f"CPA enabled={bool(enabled)} checkpoint contains CPA weights "
        f"{'yes' if has_weights else 'no'}",
        flush=True,
    )
    if not enabled:
        return None
    if not has_weights:
        raise RuntimeError("CPA enabled but checkpoint does not contain CPA weights.")
    kwargs = OmegaConf.to_container(cfg.cpa, resolve=True)
    kwargs.pop("enabled", None)
    cpa = ClassPrototypeAlignmentHead(**kwargs).to(device)
    cpa.load_state_dict(payload["cpa"])
    cpa.eval()
    return cpa


def _refinement_enabled(cfg, name):
    return bool(cfg.get(name, {}).get("enabled", False)) or bool(
        cfg.evaluate.get(f"{name}_enabled", False)
    )


def reject_msa_eval_config(cfg):
    evaluate_enabled = bool(cfg.evaluate.get("msa_enabled", False))
    msa_enabled = bool(cfg.get("msa", {}).get("enabled", False))
    if evaluate_enabled or msa_enabled:
        raise ValueError(
            "MSA/scale-aware inference has been removed from this method. "
            "Run the normal single-scale eval path without evaluate.msa_enabled "
            "or msa.enabled."
        )


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


class CleanOfficialEvalModel(nn.Module):
    def __init__(
        self,
        frozen,
        bridge,
        class_clip,
        class_base,
        cpa=None,
        xattn_delta_scale=0.5,
        xattn_logit_alpha=0.5,
        xattn_uncertainty_gate_enabled=True,
        xattn_margin_gate_enabled=False,
        xattn_margin_threshold=0.05,
    ):
        super().__init__()
        self.frozen = frozen
        self.bridge = bridge
        self.cpa = cpa
        self.register_buffer("class_clip", class_clip.float())
        self.register_buffer("class_base", class_base.float())
        self.xattn_delta_scale = float(xattn_delta_scale)
        self.xattn_logit_alpha = float(xattn_logit_alpha)
        self.xattn_uncertainty_gate_enabled = bool(xattn_uncertainty_gate_enabled)
        self.xattn_margin_gate_enabled = bool(xattn_margin_gate_enabled)
        self.xattn_margin_threshold = float(xattn_margin_threshold)
        self._logged_xattn_eval_path = False
        self._cpa_sum = {
            "cpa_residual_abs_mean": 0.0,
            "cpa_modified_fraction": 0.0,
        }
        self._cpa_residual_abs_max = 0.0
        self._cpa_count = 0

    def cpa_summary(self):
        count = max(1, self._cpa_count)
        return {
            "cpa_residual_abs_mean": self._cpa_sum["cpa_residual_abs_mean"] / count,
            "cpa_residual_abs_max": self._cpa_residual_abs_max,
            "cpa_modified_fraction": self._cpa_sum["cpa_modified_fraction"] / count,
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
        if self.cpa is not None:
            prototypes = self.cpa(mapped_text)
            spatial_features = image_feat.flatten(2).transpose(1, 2)
            prototype_logits = compute_prototype_logits(
                spatial_features,
                prototypes,
                temperature=self.cpa.prototype_temperature,
                aggregation=self.cpa.prototype_aggregation,
            ).reshape_as(base_simmap)
            simmap, cpa_stats = apply_topk_prototype_residual(
                base_simmap,
                prototype_logits,
                topk=self.cpa.topk,
                residual_scale=self.cpa.residual_scale,
                residual_clip=self.cpa.residual_clip,
            )
            self._cpa_sum["cpa_residual_abs_mean"] += float(
                cpa_stats["cpa_residual_abs_mean"].detach().cpu()
            )
            self._cpa_sum["cpa_modified_fraction"] += float(
                cpa_stats["cpa_modified_fraction"].detach().cpu()
            )
            self._cpa_residual_abs_max = max(
                self._cpa_residual_abs_max,
                float(cpa_stats["cpa_residual_abs_max"].detach().cpu()),
            )
            self._cpa_count += 1
        else:
            simmap = fuse_xattn_logits(
                base_simmap,
                xattn_simmap,
                alpha=self.xattn_logit_alpha,
                uncertainty_gate=self.xattn_uncertainty_gate_enabled,
                margin_threshold=margin_threshold,
            )
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
    bridge, payload = load_bridge_from_checkpoint(args.checkpoint, cfg, device)
    cpa_enabled = bool(cfg.evaluate.get("cpa_enabled", cfg.cpa.get("enabled", False)))
    if cpa_enabled and bool(cfg.evaluate.pamr):
        raise ValueError("CPA evaluation requires evaluate.pamr=false")
    cpa = load_cpa_from_payload(cfg, payload, device, cpa_enabled)
    print("Running single-scale inference; MSA disabled.", flush=True)
    wrapped = CleanOfficialEvalModel(
        frozen,
        bridge,
        class_clip,
        class_base,
        cpa=cpa,
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
    ).to(device)
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
    cpa_summary = wrapped.cpa_summary() if cpa is not None else None
    return miou, payload, cpa_summary


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
        raise ValueError("CCR is disabled for CPA evaluation")
    if bool(cfg.evaluate.get("cpa_enabled", cfg.cpa.get("enabled", False))) and args.cached_fast_eval:
        raise ValueError("CPA official semantic evaluation does not use cached patch-token inputs")
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
        miou, payload, cpa_summary = official_parity_eval(
            args,
            cfg,
            device,
        )
        print("=" * 64, flush=True)
        print("EVALUATION RESULTS", flush=True)
        print("=" * 64, flush=True)
        print(f"Mode                 : {METHOD_NAME} official parity", flush=True)
        print(f"coco_stuff mIoU      : {miou:.2f}%", flush=True)
        print(f"PAMR enabled         : {bool(cfg.evaluate.pamr)}", flush=True)
        print("CCR enabled          : false", flush=True)
        print(
            "XAttnBridge_Clean=true "
            f"CPA-v1={cpa_summary is not None} MSA=false "
            "PAMR=false RVS=false CARS=false RCC=false VPA=false VAB=false "
            "OPC=false Router=false USRC=false CCR=false",
            flush=True,
        )
        if cpa_summary is not None:
            print(
                "CPA eval stats        : "
                f"num_prototypes={int(cfg.cpa.num_prototypes)} "
                f"aggregation={cfg.cpa.prototype_aggregation} "
                f"topk={int(cfg.cpa.topk)} "
                f"residual_scale={float(cfg.cpa.residual_scale):.3f} "
                f"residual_clip={float(cfg.cpa.residual_clip):.3f} "
                f"mean_residual={cpa_summary['cpa_residual_abs_mean']:.6f} "
                f"max_residual={cpa_summary['cpa_residual_abs_max']:.6f} "
                f"modified_fraction={cpa_summary['cpa_modified_fraction']:.4f}",
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
                "cpa_enabled": cpa_summary is not None,
                "cpa_stats": cpa_summary,
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
    bridge, payload = load_bridge_from_checkpoint(args.checkpoint, cfg, device)
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
        logits = fuse_xattn_logits(
            base_logits,
            xattn_logits,
            alpha=xattn_logit_alpha,
            uncertainty_gate=xattn_uncertainty_gate_enabled,
            margin_threshold=xattn_margin_threshold,
        )
        n = logits.shape[-1]
        h = w = int(n ** 0.5)
        logits = logits[:, :, : h * w].reshape(1, num_classes, h, w)
        gt = sample["gt"].long()
        logits = F.interpolate(logits, size=tuple(gt.shape), mode="bilinear", align_corners=False)
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
        }, f, indent=2)


if __name__ == "__main__":
    main()
