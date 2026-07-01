import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
from xattn_bridge_clean import (
    METHOD_NAME,
    CleanBaselinePthFeatureDataset,
    CleanFeatureDataset,
    CleanXAttnBridge,
    build_coco_stuff_eval_dataset,
    build_frozen_talk2dino,
    collate_train_features,
    contrastive_loss,
    format_seconds,
    load_clean_config,
    pairwise_scores,
    save_checkpoint_clean,
)
from class_prototype_alignment import (
    ClassPrototypeAlignmentHead,
    apply_topk_prototype_residual,
    compute_prototype_logits,
    prototype_diversity_loss,
)
from vcdd_head import VocabularyConditionedDenseDistillationHead
from eval_xattn_clean import (
    bridge_class_embeddings,
    build_class_embeddings,
    fuse_xattn_logits,
    intersect_and_union,
    load_all_eval_samples,
    mean_template_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser(METHOD_NAME + " training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_features", required=True)
    parser.add_argument("--eval_features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opts", nargs="+", default=None)
    return parser.parse_args()


def trainable_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def clockstamp():
    return time.strftime("%H:%M:%S")


def tensor_shape(value):
    if torch.is_tensor(value):
        return "x".join(str(dim) for dim in value.shape)
    return type(value).__name__


def file_size_mb(path):
    path = Path(path)
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 ** 2)


def get_safety_cfg(cfg):
    defaults = OmegaConf.create({
        "enabled": True,
        "delta_l2_weight": 0.001,
        "ratio_target": 0.07,
        "ratio_penalty_weight": 0.02,
        "min_base_mapped_cosine": 0.98,
        "cosine_penalty_weight": 0.01,
        "attn_prior_weight": 0.01,
        "attn_prior_temperature": 0.07,
        "patch_preserve_weight": 0.005,
        "patch_preserve_temperature": 0.5,
        "stop_if_ratio_above": 0.30,
        "warn_if_ratio_above": 0.20,
    })
    return OmegaConf.merge(defaults, cfg.get("xattn_safety", {}))


def get_cpa_cfg(cfg):
    defaults = OmegaConf.create({
        "enabled": True,
        "version": "v1",
        "num_prototypes": 4,
        "hidden_dim": 256,
        "prototype_scale": 0.10,
        "prototype_aggregation": "logsumexp",
        "prototype_temperature": 0.07,
        "normalize": True,
        "topk": 5,
        "residual_scale": 0.25,
        "residual_clip": 0.5,
    })
    return OmegaConf.merge(defaults, cfg.get("cpa", {}))


def get_vcdd_cfg(cfg):
    defaults = OmegaConf.create({
        "enabled": False,
        "compact_attention": True,
        "train_attention_path": "/scratch/haree/talk2dino_features/coco_stuff2017_vitb/vcdd_attention_lift_train_fp16.pt",
        "val_attention_path": "/scratch/haree/talk2dino_features/coco_stuff2017_vitb/vcdd_attention_lift_val_fp16.pt",
        "hidden_dim": 256,
        "dropout": 0.0,
        "scale": 0.05,
        "residual_clip": 0.25,
        "topk": 5,
        "topk_source": "cpa",
        "detach_attention": True,
    })
    return OmegaConf.merge(defaults, cfg.get("vcdd", {}))


def get_vcdd_loss_cfg(cfg):
    defaults = OmegaConf.create({
        "enabled": False,
        "ce_weight": 0.05,
        "kl_weight": 0.02,
        "preserve_weight": 0.01,
        "residual_l1_weight": 0.001,
        "teacher_temperature": 0.07,
        "kl_temperature": 0.07,
        "preserve_temperature": 0.07,
        "min_prob": 0.30,
        "min_margin": 0.10,
        "max_entropy": 0.70,
        "min_reliable_fraction": 0.05,
        "entropy_sharpen_weight": 0.0,
    })
    return OmegaConf.merge(defaults, cfg.get("vcdd_loss", {}))


def get_cpa_loss_cfg(cfg):
    defaults = OmegaConf.create({
        "enabled": True,
        "prototype_infonce_weight": 0.20,
        "preserve_weight": 0.01,
        "temperature": 2.0,
        "confident_margin": 0.20,
        "diversity_weight": 0.001,
        "diversity_margin": 0.90,
        "residual_l1_weight": 0.001,
    })
    return OmegaConf.merge(defaults, cfg.get("cpa_loss", {}))


def validate_cpa_semantic_setup(cfg, feature_path, sample):
    if not bool(cfg.get("cpa", {}).get("enabled", False)):
        return
    expected = "disentangled_self_attn"
    names = {
        "features_name": str(cfg.data.get("features_name", expected)),
        "visual_features_name": str(cfg.data.get("visual_features_name", expected)),
        "patch_features_name": str(cfg.data.get("patch_features_name", expected)),
    }
    if any(value != expected for value in names.values()):
        raise ValueError(
            f"CPA requires semantic disentangled_self_attn features; got {names}"
        )
    for key in ("patch_tokens_dir", "train_patch_tokens_dir", "val_patch_tokens_dir"):
        if cfg.data.get(key, None) not in {None, "", "null", "None"}:
            raise ValueError(f"CPA does not support raw patch-token option data.{key}")
    if not feature_path.is_file():
        raise ValueError("CPA requires the baseline-style semantic feature .pth")
    visual = sample.get("visual_embed")
    semantic = sample.get("patch_tokens")
    if visual is None or semantic is None:
        raise RuntimeError("CPA sample is missing semantic features")
    if visual.shape != semantic.shape or visual.data_ptr() != semantic.data_ptr():
        raise RuntimeError(
            "CPA XAttn K/V must be the same disentangled_self_attn tensor as visual_embed"
        )


def kl_rows(logits_log_probs, target_probs):
    return F.kl_div(
        logits_log_probs.reshape(-1, logits_log_probs.shape[-1]),
        target_probs.reshape(-1, target_probs.shape[-1]),
        reduction="batchmean",
    )


def bridge_safety_losses(stats, safety_cfg):
    delta = stats["delta"].float()
    ratio = stats["delta_base_ratio"].float()
    cosine = stats["cosine_base_mapped"].float()
    zero = delta.new_tensor(0.0)
    if not bool(safety_cfg.enabled):
        return {
            "loss_delta_l2": zero,
            "loss_ratio": zero,
            "loss_cos": zero,
            "loss_attn_prior": zero,
            "loss_patch_preserve": zero,
            "attn_prior_kl": zero,
            "patch_preserve_kl": zero,
        }
    loss_delta_l2 = delta.norm(dim=-1).pow(2).mean()
    loss_ratio = F.relu(
        ratio - float(safety_cfg.ratio_target)
    ).pow(2).mean()
    loss_cos = F.relu(
        float(safety_cfg.min_base_mapped_cosine) - cosine
    ).pow(2).mean()
    loss_attn_prior = zero
    if "attention_probs" in stats and "base_sim" in stats:
        temp = max(float(safety_cfg.attn_prior_temperature), 1e-6)
        base_prior = F.softmax(stats["base_sim"].detach().float() / temp, dim=-1)
        xattn_attn = stats["attention_probs"].float().mean(dim=1).clamp_min(1e-8)
        loss_attn_prior = kl_rows(torch.log(xattn_attn), base_prior)
    loss_patch_preserve = zero
    if "xattn_patch_logits" in stats and "base_patch_logits" in stats:
        temp = max(float(safety_cfg.patch_preserve_temperature), 1e-6)
        p_base = F.softmax(stats["base_patch_logits"].detach().float() / temp, dim=-1)
        log_p_xattn = F.log_softmax(
            stats["xattn_patch_logits"].float() / temp,
            dim=-1,
        )
        loss_patch_preserve = kl_rows(log_p_xattn, p_base)
    return {
        "loss_delta_l2": loss_delta_l2,
        "loss_ratio": loss_ratio,
        "loss_cos": loss_cos,
        "loss_attn_prior": loss_attn_prior,
        "loss_patch_preserve": loss_patch_preserve,
        "attn_prior_kl": loss_attn_prior.detach(),
        "patch_preserve_kl": loss_patch_preserve.detach(),
    }


def zero_cpa_losses(reference):
    zero = reference.float().new_tensor(0.0)
    return {
        "loss_cpa_proto_infonce": zero,
        "loss_cpa_preserve": zero,
        "loss_cpa_diversity": zero,
        "loss_cpa_residual_l1": zero,
        "cpa_residual_abs_mean": zero,
        "cpa_residual_abs_max": zero,
        "cpa_modified_fraction": zero,
        "cpa_confident_patch_frac": zero,
        "cpa_proto_pairwise_cos_mean": zero,
        "cpa_proto_pairwise_cos_max": zero,
        "cpa_topk": zero,
    }


def cpa_auxiliary_losses(
    cpa,
    mapped_text,
    visual_embed,
    semantic_features,
    base_patch_logits,
    loss_cfg,
    contrastive_temperature,
):
    prototypes = cpa(mapped_text)
    prototype_dense_scores = compute_prototype_logits(
        visual_embed,
        prototypes,
        temperature=cpa.prototype_temperature,
        aggregation=cpa.prototype_aggregation,
    )
    prototype_scores = (
        prototype_dense_scores.max(dim=-1).values
        if prototype_dense_scores.dim() == 3
        else prototype_dense_scores
    )
    loss_proto_infonce = contrastive_loss(
        prototype_scores / max(float(contrastive_temperature), 1e-6)
    )
    prototype_patch_logits = compute_prototype_logits(
        semantic_features,
        prototypes,
        temperature=cpa.prototype_temperature,
        aggregation=cpa.prototype_aggregation,
    )
    final_patch_logits, cpa_stats = apply_topk_prototype_residual(
        base_patch_logits,
        prototype_patch_logits,
        topk=cpa.topk,
        residual_scale=cpa.residual_scale,
        residual_clip=cpa.residual_clip,
    )

    base = base_patch_logits.float()
    final = final_patch_logits.float()
    top2 = base.topk(min(2, base.shape[1]), dim=1).values
    margin = (
        top2[:, 0] - top2[:, 1]
        if top2.shape[1] > 1
        else torch.zeros_like(top2[:, 0])
    )
    confident_mask = margin > float(loss_cfg.confident_margin)
    loss_preserve = base.new_tensor(0.0)
    if confident_mask.any():
        temperature = max(float(loss_cfg.temperature), 1e-6)
        p_base = F.softmax((base / temperature).detach(), dim=1)
        log_p_final = F.log_softmax(final / temperature, dim=1)
        loss_preserve = F.kl_div(
            log_p_final.permute(0, 2, 1)[confident_mask],
            p_base.permute(0, 2, 1)[confident_mask],
            reduction="batchmean",
        )
    loss_diversity, pair_cos_mean, pair_cos_max = prototype_diversity_loss(
        prototypes,
        margin=float(loss_cfg.diversity_margin),
    )
    selected_residual = cpa_stats["cpa_residual"].masked_select(
        cpa_stats["cpa_topk_mask"]
    )
    loss_residual_l1 = selected_residual.abs().mean()
    return {
        "loss_cpa_proto_infonce": loss_proto_infonce,
        "loss_cpa_preserve": loss_preserve,
        "loss_cpa_diversity": loss_diversity,
        "loss_cpa_residual_l1": loss_residual_l1,
        "cpa_residual_abs_mean": cpa_stats["cpa_residual_abs_mean"],
        "cpa_residual_abs_max": cpa_stats["cpa_residual_abs_max"],
        "cpa_modified_fraction": cpa_stats["cpa_modified_fraction"],
        "cpa_confident_patch_frac": confident_mask.float().mean().detach(),
        "cpa_proto_pairwise_cos_mean": pair_cos_mean,
        "cpa_proto_pairwise_cos_max": pair_cos_max,
        "cpa_topk": cpa_stats["cpa_topk"],
    }


def load_vcdd_attention_index(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"VCDD attention file not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "image_ids" not in payload or "attention" not in payload:
        raise KeyError("VCDD attention file must contain `image_ids` and `attention`")
    image_ids = payload["image_ids"].long()
    attention = payload["attention"]
    if attention.ndim != 3 or attention.shape[1:] != (12, 1024):
        raise ValueError(f"VCDD attention must be [N,12,1024], got {tuple(attention.shape)}")
    if image_ids.numel() != attention.shape[0]:
        raise ValueError("VCDD attention image_ids length does not match rows")
    if not torch.isfinite(attention.float()).all():
        raise ValueError("VCDD attention contains NaN/Inf")
    index = {}
    for row, image_id in enumerate(image_ids.tolist()):
        if int(image_id) in index:
            raise ValueError(f"Duplicate VCDD attention image_id={image_id}")
        index[int(image_id)] = row
    sums = attention.float().sum(dim=1)
    print(
        "VCDD attention loaded     : "
        f"path={path} shape={tuple(attention.shape)} dtype={attention.dtype} "
        f"sum_mean={float(sums.mean()):.6f} sum_min={float(sums.min()):.6f} "
        f"sum_max={float(sums.max()):.6f}",
        flush=True,
    )
    return {"path": path, "attention": attention, "index": index, "meta": payload.get("meta", {})}


def gather_vcdd_attention(attention_store, image_ids, device):
    rows = []
    missing = []
    for image_id in image_ids.detach().cpu().tolist():
        row = attention_store["index"].get(int(image_id))
        if row is None:
            missing.append(int(image_id))
        else:
            rows.append(row)
    if missing:
        raise KeyError(f"VCDD attention missing image_ids: {missing[:8]}")
    return attention_store["attention"][rows].to(device, non_blocking=True).float()


def lift_region_logits(region_logits, attention_lift):
    attention = attention_lift.float()
    attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.einsum("brp,bcr->bcp", attention, region_logits.float())


@torch.no_grad()
def build_eval_vocabulary(frozen, cfg, device):
    dataset = build_coco_stuff_eval_dataset(cfg)
    class_names = list(dataset.CLASSES)
    if class_names and class_names[0] == "background":
        class_names = class_names[1:]
    class_clip, class_base = build_class_embeddings(
        frozen,
        cfg,
        class_names,
        device,
        keep_templates=True,
    )
    return class_names, class_clip.float(), class_base.float()


def bridge_vocabulary_embeddings(bridge, class_clip, semantic_features, class_base, delta_scale=0.5):
    num_classes, num_templates, clip_dim = class_clip.shape
    flat_clip = class_clip.reshape(num_classes * num_templates, clip_dim)
    flat_base = class_base.reshape(num_classes * num_templates, class_base.shape[-1])
    mapped, stats = bridge(flat_clip, semantic_features, flat_base, return_stats=True)
    mapped = F.normalize(flat_base.unsqueeze(0) + float(delta_scale) * stats["delta"].float(), dim=-1)
    return F.normalize(mapped.reshape(semantic_features.shape[0], num_classes, num_templates, -1).mean(dim=2), dim=-1)


def zero_vcdd_losses(reference, vocab_size=0):
    zero = reference.float().new_tensor(0.0)
    return {
        "loss_vcdd_ce": zero,
        "loss_vcdd_kl": zero,
        "loss_vcdd_preserve": zero,
        "loss_vcdd_residual_l1": zero,
        "vcdd_reliable_patch_fraction": zero,
        "vcdd_fallback_used_fraction": zero,
        "vcdd_teacher_top1_prob_mean": zero,
        "vcdd_teacher_margin_mean": zero,
        "vcdd_teacher_entropy_mean": zero,
        "vcdd_region_residual_abs_mean": zero,
        "vcdd_region_residual_abs_max": zero,
        "vcdd_dense_residual_abs_mean": zero,
        "vcdd_dense_residual_abs_max": zero,
        "vcdd_modified_fraction": zero,
        "vcdd_attention_sum_mean": zero,
        "vcdd_vocab_size": zero.new_tensor(float(vocab_size)),
    }


def vcdd_dense_losses(
    vcdd,
    semantic_features,
    mapped_vocab_text,
    attention_lift,
    dense_base_logits,
    dense_cpa_v1_logits,
    loss_cfg,
):
    final_dense_logits, vcdd_stats = vcdd(
        semantic_features,
        mapped_vocab_text,
        attention_lift,
        dense_cpa_v1_logits,
        dense_base_logits=dense_base_logits,
    )
    teacher = dense_cpa_v1_logits.detach().float()
    final = final_dense_logits.float()
    teacher_temp = max(float(loss_cfg.teacher_temperature), 1e-6)
    teacher_probs = F.softmax(teacher / teacher_temp, dim=1)
    top2 = teacher_probs.topk(k=min(2, teacher_probs.shape[1]), dim=1).values
    top1_prob = top2[:, 0]
    margin = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else torch.zeros_like(top2[:, 0])
    entropy = -(teacher_probs * teacher_probs.clamp_min(1e-8).log()).sum(dim=1)
    entropy_norm = entropy / math.log(max(teacher_probs.shape[1], 2))
    reliable = (
        (top1_prob >= float(loss_cfg.min_prob))
        & (margin >= float(loss_cfg.min_margin))
        & (entropy_norm <= float(loss_cfg.max_entropy))
    )
    fallback_used = final.new_tensor(0.0)
    min_count = int(math.ceil(float(loss_cfg.min_reliable_fraction) * reliable.numel()))
    if reliable.sum().item() < min_count and min_count > 0:
        flat_conf = top1_prob.reshape(-1)
        top_idx = flat_conf.topk(min(min_count, flat_conf.numel())).indices
        reliable = torch.zeros_like(flat_conf, dtype=torch.bool).scatter(0, top_idx, True).reshape_as(reliable)
        fallback_used = final.new_tensor(1.0)
    unreliable = ~reliable
    labels = teacher.argmax(dim=1)
    zero = final.new_tensor(0.0)
    loss_ce = zero
    loss_kl = zero
    if reliable.any():
        final_rel = final.permute(0, 2, 1)[reliable]
        labels_rel = labels[reliable]
        loss_ce = F.cross_entropy(final_rel, labels_rel)
        kl_temp = max(float(loss_cfg.kl_temperature), 1e-6)
        p_teacher = F.softmax((teacher / kl_temp).permute(0, 2, 1)[reliable], dim=-1)
        log_p_student = F.log_softmax((final / kl_temp).permute(0, 2, 1)[reliable], dim=-1)
        loss_kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (kl_temp ** 2)
    loss_preserve = zero
    if unreliable.any():
        preserve_temp = max(float(loss_cfg.preserve_temperature), 1e-6)
        p_cpa = F.softmax((teacher / preserve_temp).permute(0, 2, 1)[unreliable], dim=-1)
        log_p_final = F.log_softmax((final / preserve_temp).permute(0, 2, 1)[unreliable], dim=-1)
        loss_preserve = F.kl_div(log_p_final, p_cpa, reduction="batchmean") * (preserve_temp ** 2)
    loss_residual_l1 = (final - teacher).abs().mean()
    out = {
        "loss_vcdd_ce": loss_ce,
        "loss_vcdd_kl": loss_kl,
        "loss_vcdd_preserve": loss_preserve,
        "loss_vcdd_residual_l1": loss_residual_l1,
        "vcdd_reliable_patch_fraction": reliable.float().mean().detach(),
        "vcdd_fallback_used_fraction": fallback_used.detach(),
        "vcdd_teacher_top1_prob_mean": top1_prob.mean().detach(),
        "vcdd_teacher_margin_mean": margin.mean().detach(),
        "vcdd_teacher_entropy_mean": entropy_norm.mean().detach(),
        "vcdd_vocab_size": final.new_tensor(float(final.shape[1])).detach(),
    }
    out.update(vcdd_stats)
    return out


def init_epoch_accumulators():
    return {
        "loss_total": 0.0,
        "loss_infonce": 0.0,
        "loss_delta_l2": 0.0,
        "loss_ratio": 0.0,
        "loss_cos": 0.0,
        "loss_attn_prior": 0.0,
        "loss_patch_preserve": 0.0,
        "attn_prior_kl": 0.0,
        "patch_preserve_kl": 0.0,
        "loss_cpa_proto_infonce": 0.0,
        "loss_cpa_preserve": 0.0,
        "loss_cpa_diversity": 0.0,
        "loss_cpa_residual_l1": 0.0,
        "cpa_residual_abs_mean": 0.0,
        "cpa_residual_abs_max": 0.0,
        "cpa_modified_fraction": 0.0,
        "cpa_confident_patch_frac": 0.0,
        "cpa_proto_pairwise_cos_mean": 0.0,
        "cpa_proto_pairwise_cos_max": 0.0,
        "cpa_topk": 0.0,
        "loss_vcdd_ce": 0.0,
        "loss_vcdd_kl": 0.0,
        "loss_vcdd_preserve": 0.0,
        "loss_vcdd_residual_l1": 0.0,
        "vcdd_reliable_patch_fraction": 0.0,
        "vcdd_fallback_used_fraction": 0.0,
        "vcdd_teacher_top1_prob_mean": 0.0,
        "vcdd_teacher_margin_mean": 0.0,
        "vcdd_teacher_entropy_mean": 0.0,
        "vcdd_region_residual_abs_mean": 0.0,
        "vcdd_region_residual_abs_max": 0.0,
        "vcdd_dense_residual_abs_mean": 0.0,
        "vcdd_dense_residual_abs_max": 0.0,
        "vcdd_modified_fraction": 0.0,
        "vcdd_attention_sum_mean": 0.0,
        "vcdd_vocab_size": 0.0,
        "base_norm_mean": 0.0,
        "delta_norm_mean": 0.0,
        "delta_base_ratio_mean": 0.0,
        "delta_base_ratio_max": 0.0,
        "cosine_base_mapped_mean": 0.0,
        "cosine_base_mapped_min": 1.0,
        "gamma": 0.0,
        "base_guidance_beta": 0.0,
        "data_time": 0.0,
        "compute_time": 0.0,
    }


def update_epoch_accumulators(acc, loss_total, loss_infonce, losses, stats, cpa_losses, vcdd_losses):
    acc["loss_total"] += float(loss_total.detach().cpu())
    acc["loss_infonce"] += float(loss_infonce.detach().cpu())
    acc["loss_delta_l2"] += float(losses["loss_delta_l2"].detach().cpu())
    acc["loss_ratio"] += float(losses["loss_ratio"].detach().cpu())
    acc["loss_cos"] += float(losses["loss_cos"].detach().cpu())
    acc["loss_attn_prior"] += float(losses["loss_attn_prior"].detach().cpu())
    acc["loss_patch_preserve"] += float(losses["loss_patch_preserve"].detach().cpu())
    acc["attn_prior_kl"] += float(losses["attn_prior_kl"].detach().cpu())
    acc["patch_preserve_kl"] += float(losses["patch_preserve_kl"].detach().cpu())
    for key in (
        "loss_cpa_proto_infonce",
        "loss_cpa_preserve",
        "loss_cpa_diversity",
        "loss_cpa_residual_l1",
        "cpa_residual_abs_mean",
        "cpa_modified_fraction",
        "cpa_confident_patch_frac",
        "cpa_proto_pairwise_cos_mean",
        "cpa_topk",
    ):
        acc[key] += float(cpa_losses[key].detach().cpu())
    acc["cpa_residual_abs_max"] = max(
        acc["cpa_residual_abs_max"],
        float(cpa_losses["cpa_residual_abs_max"].detach().cpu()),
    )
    acc["cpa_proto_pairwise_cos_max"] = max(
        acc["cpa_proto_pairwise_cos_max"],
        float(cpa_losses["cpa_proto_pairwise_cos_max"].detach().cpu()),
    )
    for key in (
        "loss_vcdd_ce",
        "loss_vcdd_kl",
        "loss_vcdd_preserve",
        "loss_vcdd_residual_l1",
        "vcdd_reliable_patch_fraction",
        "vcdd_fallback_used_fraction",
        "vcdd_teacher_top1_prob_mean",
        "vcdd_teacher_margin_mean",
        "vcdd_teacher_entropy_mean",
        "vcdd_region_residual_abs_mean",
        "vcdd_dense_residual_abs_mean",
        "vcdd_modified_fraction",
        "vcdd_attention_sum_mean",
        "vcdd_vocab_size",
    ):
        acc[key] += float(vcdd_losses[key].detach().cpu())
    acc["vcdd_region_residual_abs_max"] = max(
        acc["vcdd_region_residual_abs_max"],
        float(vcdd_losses["vcdd_region_residual_abs_max"].detach().cpu()),
    )
    acc["vcdd_dense_residual_abs_max"] = max(
        acc["vcdd_dense_residual_abs_max"],
        float(vcdd_losses["vcdd_dense_residual_abs_max"].detach().cpu()),
    )
    acc["base_norm_mean"] += float(stats["base_norm"].detach().float().mean().cpu())
    acc["delta_norm_mean"] += float(stats["delta_norm"].detach().float().mean().cpu())
    ratio = stats["delta_base_ratio"].detach().float()
    cosine = stats["cosine_base_mapped"].detach().float()
    acc["delta_base_ratio_mean"] += float(ratio.mean().cpu())
    acc["delta_base_ratio_max"] = max(acc["delta_base_ratio_max"], float(ratio.max().cpu()))
    acc["cosine_base_mapped_mean"] += float(cosine.mean().cpu())
    acc["cosine_base_mapped_min"] = min(acc["cosine_base_mapped_min"], float(cosine.min().cpu()))
    if "gamma" in stats:
        acc["gamma"] += float(stats["gamma"].detach().float().mean().cpu())
    if "base_guidance_beta" in stats:
        acc["base_guidance_beta"] += float(
            stats["base_guidance_beta"].detach().float().mean().cpu()
        )


def finalize_epoch_accumulators(acc, count):
    count = max(1, count)
    mean_keys = [
        "loss_total",
        "loss_infonce",
        "loss_delta_l2",
        "loss_ratio",
        "loss_cos",
        "loss_attn_prior",
        "loss_patch_preserve",
        "attn_prior_kl",
        "patch_preserve_kl",
        "loss_cpa_proto_infonce",
        "loss_cpa_preserve",
        "loss_cpa_diversity",
        "loss_cpa_residual_l1",
        "cpa_residual_abs_mean",
        "cpa_modified_fraction",
        "cpa_confident_patch_frac",
        "cpa_proto_pairwise_cos_mean",
        "cpa_topk",
        "loss_vcdd_ce",
        "loss_vcdd_kl",
        "loss_vcdd_preserve",
        "loss_vcdd_residual_l1",
        "vcdd_reliable_patch_fraction",
        "vcdd_fallback_used_fraction",
        "vcdd_teacher_top1_prob_mean",
        "vcdd_teacher_margin_mean",
        "vcdd_teacher_entropy_mean",
        "vcdd_region_residual_abs_mean",
        "vcdd_dense_residual_abs_mean",
        "vcdd_modified_fraction",
        "vcdd_attention_sum_mean",
        "vcdd_vocab_size",
        "base_norm_mean",
        "delta_norm_mean",
        "delta_base_ratio_mean",
        "cosine_base_mapped_mean",
        "gamma",
        "base_guidance_beta",
        "data_time",
        "compute_time",
    ]
    return {
        key: (value / count if key in mean_keys else value)
        for key, value in acc.items()
    }


def print_epoch_progress(epoch, epochs, step, total_steps, acc, start_time, lr):
    total_steps = max(1, total_steps)
    step = min(step, total_steps)
    percent = 100.0 * step / total_steps
    filled = int(24 * step / total_steps)
    bar = "=" * filled + "." * (24 - filled)
    elapsed = time.time() - start_time
    speed = step / max(elapsed, 1e-6)
    eta = (total_steps - step) / max(speed, 1e-6)
    metrics = finalize_epoch_accumulators(acc, step)
    print(
        "\r"
        f"{clockstamp()} ep{epoch} "
        f"[{bar}] {step}/{total_steps} {percent:4.0f}% "
        f"L={metrics['loss_total']:.4f} "
        f"inf={metrics['loss_infonce']:.4f} "
        f"attn={metrics['attn_prior_kl']:.4f} "
        f"patch={metrics['patch_preserve_kl']:.4f} "
        f"cpa={metrics['loss_cpa_proto_infonce']:.4f} "
        f"cpa|r|={metrics['cpa_residual_abs_mean']:.4f} "
        f"vcdd={metrics['loss_vcdd_ce']:.4f} "
        f"vrel={metrics['vcdd_reliable_patch_fraction']:.3f} "
        f"d/b={metrics['delta_base_ratio_mean']:.3f} "
        f"dmax={metrics['delta_base_ratio_max']:.4f} "
        f"cos={metrics['cosine_base_mapped_mean']:.4f} "
        f"g={metrics['gamma']:.3f} "
        f"bg={metrics['base_guidance_beta']:.2f} "
        f"dt={metrics['data_time']:.3f}s "
        f"xa={metrics['compute_time']:.3f}s "
        f"lr={lr:.2e} "
        f"{speed:.2f} it/s "
        f"el={format_seconds(elapsed)} eta={format_seconds(eta)}",
        end="",
        flush=True,
    )
    if step >= total_steps:
        print("", flush=True)


def projection_trainable_count(model):
    if not hasattr(model, "proj"):
        return 0
    return sum(p.numel() for p in model.proj.parameters() if p.requires_grad)


def maybe_auto_resume(out, cfg, bridge, cpa, vcdd, optimizer, scheduler, scaler, device):
    auto_resume = bool(cfg.train.get("auto_resume", True))
    resume_path = cfg.train.get("resume", None)
    if resume_path in {"", "null", "None"}:
        resume_path = None
    if resume_path is None and auto_resume:
        candidate = out / "checkpoint_last.pth"
        if candidate.exists():
            resume_path = candidate
    if resume_path is None:
        return 1, -float("inf"), 0

    resume_path = Path(resume_path)
    if not resume_path.exists():
        raise FileNotFoundError(f"Requested resume checkpoint does not exist: {resume_path}")
    payload = torch.load(resume_path, map_location=device, weights_only=False)
    state = payload.get("bridge", payload.get("model"))
    bridge.load_state_dict(state)
    if cpa is not None:
        if payload.get("cpa") is None:
            raise RuntimeError(
                "CPA enabled but checkpoint does not contain CPA weights. "
                "Start from scratch or use a CPA checkpoint."
            )
        cpa.load_state_dict(payload["cpa"])
    if vcdd is not None:
        if payload.get("vcdd") is None:
            print(
                "VCDD enabled but resume checkpoint has no VCDD weights; keeping zero-init VCDD head.",
                flush=True,
            )
        else:
            vcdd.load_state_dict(payload["vcdd"])
    bridge._skip_zero_init_parity_check = True
    bridge._parity_checked = True
    if payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if payload.get("scheduler") is not None and scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if payload.get("scaler") is not None and scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    last_epoch = int(payload.get("epoch", 0))
    best_miou = float(payload.get("best_miou", -float("inf")))
    best_epoch = int(payload.get("best_epoch", 0))
    if best_epoch == 0 and math.isinf(best_miou):
        fallback_metric = payload.get("val_loss", None)
        if fallback_metric is None:
            fallback_metric = payload.get("loss_total", None)
        if fallback_metric is not None:
            best_miou = float(fallback_metric)
            best_epoch = last_epoch
    print(
        f"[{timestamp()}] Auto-continue loaded checkpoint: {resume_path} "
        f"(last_epoch={last_epoch}, next_epoch={last_epoch + 1}, "
        f"best_metric={best_miou:.4f}, best_epoch={best_epoch})",
        flush=True,
    )
    return last_epoch + 1, best_miou, best_epoch


@torch.no_grad()
def evaluate_baseline_val_loss(bridge, val_loader, frozen, device):
    bridge.eval()
    losses = []
    for batch in val_loader:
        text_clip = batch["text_clip"].to(device, non_blocking=True).float()
        with torch.no_grad():
            text_base = frozen._frozen_base_text_to_dino(text_clip).float()
        patches = batch["patch_tokens"].to(device, non_blocking=True).float()
        visual = batch["visual_embed"].to(device, non_blocking=True).float()
        scores = pairwise_scores(bridge, text_clip, text_base, patches, visual)
        loss = contrastive_loss(scores)
        losses.append(float(loss.detach().cpu()))
    bridge.train()
    if not losses:
        return float("inf")
    return float(torch.tensor(losses).mean())


@torch.no_grad()
def evaluate_cached_miou(
    bridge,
    eval_samples,
    class_clip,
    class_base,
    num_classes,
    ignore_index,
    device,
    xattn_delta_scale=0.5,
    xattn_logit_alpha=0.5,
    xattn_uncertainty_gate_enabled=True,
    xattn_margin_threshold=None,
):
    bridge.eval()
    total_inter = torch.zeros(num_classes)
    total_union = torch.zeros(num_classes)
    base_text = mean_template_embeddings(class_base.to(device))
    for sample in eval_samples:
        patches = sample["patch_tokens"].unsqueeze(0).to(device).float()
        mapped = bridge_class_embeddings(
            bridge,
            class_clip,
            patches,
            class_base,
            delta_scale=xattn_delta_scale,
        )
        patch_norm = torch.nn.functional.normalize(patches, dim=-1)
        xattn_logits = torch.einsum(
            "bnd,cd->bcn",
            patch_norm,
            mapped,
        )
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
        logits = torch.nn.functional.interpolate(
            logits,
            size=tuple(gt.shape),
            mode="bilinear",
            align_corners=False,
        )
        pred = logits.argmax(dim=1)[0].cpu()
        inter, union = intersect_and_union(pred, gt, num_classes, ignore_index)
        total_inter += inter
        total_union += union
    bridge.train()
    return float(torch.nanmean(total_inter / total_union.clamp_min(1)) * 100.0)


def main():
    args = parse_args()
    cfg = load_clean_config(args.config, args.opts)
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    train_feature_path = Path(args.train_features)
    if train_feature_path.is_file():
        train_set = CleanBaselinePthFeatureDataset(
            train_feature_path,
            features_name=str(cfg.data.get("features_name", "disentangled_self_attn")),
            text_features=str(cfg.data.get("text_features", "ann_feats")),
            mmap=bool(cfg.data.get("mmap_features", True)),
        )
        train_feature_source = "baseline_pth_in_memory"
    else:
        train_set = CleanFeatureDataset(
            args.train_features,
            shard_cache_size=int(cfg.data.get("shard_cache_size", 2)),
            preload=bool(cfg.data.get("preload_cache", False)),
        )
        train_feature_source = "xattn_sharded_cache"
    loader_kwargs = {
        "num_workers": int(cfg.data.num_workers),
        "pin_memory": bool(cfg.data.pin_memory),
        "persistent_workers": (
            int(cfg.data.num_workers) > 0
            and bool(cfg.data.get("persistent_workers", False))
        ),
        "collate_fn": collate_train_features,
    }
    if int(cfg.data.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.data.get("prefetch_factor", 1))
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg.train.batch_size),
        shuffle=bool(cfg.train.get("shuffle", True)),
        drop_last=True,
        **loader_kwargs,
    )
    first_train_sample = train_set[0]
    validate_cpa_semantic_setup(cfg, train_feature_path, first_train_sample)
    frozen = build_frozen_talk2dino(cfg, device)
    eval_feature_path = Path(args.eval_features)
    if eval_feature_path.is_file():
        val_set = CleanBaselinePthFeatureDataset(
            eval_feature_path,
            features_name=str(cfg.data.get("features_name", "disentangled_self_attn")),
            text_features=str(cfg.data.get("text_features", "ann_feats")),
            mmap=bool(cfg.data.get("mmap_features", True)),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=int(cfg.train.batch_size),
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        eval_mode = "baseline_pth_val_loss"
        eval_samples = None
        class_clip = class_base = None
        num_classes = ignore_index = None
    else:
        eval_manifest, eval_samples = load_all_eval_samples(args.eval_features)
        if "classes" in eval_manifest and "ignore_index" in eval_manifest:
            eval_classes = eval_manifest["classes"]
            ignore_index = int(eval_manifest["ignore_index"])
        else:
            eval_dataset = build_coco_stuff_eval_dataset(cfg)
            eval_classes = list(eval_dataset.CLASSES)
            ignore_index = int(eval_dataset.ignore_index)
        class_clip, class_base = build_class_embeddings(
            frozen,
            cfg,
            eval_classes,
            device,
            keep_templates=True,
        )
        num_classes = len(eval_classes)
        val_loader = None
        eval_mode = "cached_seg_miou"

    safety_cfg = get_safety_cfg(cfg)
    cpa_cfg = get_cpa_cfg(cfg)
    cpa_loss_cfg = get_cpa_loss_cfg(cfg)
    vcdd_cfg = get_vcdd_cfg(cfg)
    vcdd_loss_cfg = get_vcdd_loss_cfg(cfg)
    bridge = CleanXAttnBridge(**OmegaConf.to_container(cfg.bridge, resolve=True)).to(device)
    cpa_kwargs = OmegaConf.to_container(cpa_cfg, resolve=True)
    cpa_kwargs.pop("enabled", None)
    cpa_version = cpa_kwargs.pop("version", "v1")
    if str(cpa_version) != "v1":
        raise ValueError("Only CPA-v1 is supported in the VCDD path")
    cpa = ClassPrototypeAlignmentHead(**cpa_kwargs).to(device) if bool(cpa_cfg.enabled) else None
    if bool(vcdd_cfg.enabled) and cpa is None:
        raise ValueError("VCDD requires CPA-v1 to be enabled")
    vcdd_attention_store = (
        load_vcdd_attention_index(vcdd_cfg.train_attention_path)
        if bool(vcdd_cfg.enabled)
        else None
    )
    vcdd_class_names = []
    vcdd_class_clip = None
    vcdd_class_base = None
    if bool(vcdd_cfg.enabled):
        vcdd_class_names, vcdd_class_clip, vcdd_class_base = build_eval_vocabulary(
            frozen,
            cfg,
            device,
        )
    vcdd = None
    if bool(vcdd_cfg.enabled):
        vcdd = VocabularyConditionedDenseDistillationHead(
            dino_dim=int(cfg.bridge.get("dino_dim", 768)),
            hidden_dim=int(vcdd_cfg.hidden_dim),
            dropout=float(vcdd_cfg.dropout),
            scale=float(vcdd_cfg.scale),
            residual_clip=float(vcdd_cfg.residual_clip),
            topk=int(vcdd_cfg.topk),
            topk_source=str(vcdd_cfg.topk_source),
            detach_attention=bool(vcdd_cfg.detach_attention),
        ).to(device)
    trainable_parameters = list(bridge.parameters())
    if cpa is not None:
        trainable_parameters.extend(cpa.parameters())
    if vcdd is not None:
        trainable_parameters.extend(vcdd.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(cfg.train.epochs) * max(1, len(train_loader))),
        eta_min=float(cfg.train.min_lr),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.fp16))
    train_log_path = out / "train_log.jsonl"

    print("=" * 78, flush=True)
    method_label = (
        f"{METHOD_NAME} + CPA-v1 + VCDD"
        if vcdd is not None
        else (f"{METHOD_NAME} + Class Prototype Alignment Head" if cpa is not None else METHOD_NAME)
    )
    print(f"[{timestamp()}] Starting {method_label} training", flush=True)
    print("=" * 78, flush=True)
    print(f"Output directory          : {out}", flush=True)
    print(f"Train features            : {args.train_features}", flush=True)
    print(f"Eval features             : {args.eval_features}", flush=True)
    print(f"Device                    : {device}", flush=True)
    print(f"Train samples             : {len(train_set)}", flush=True)
    print(f"Train batches/epoch       : {len(train_loader)}", flush=True)
    print(f"Train epochs              : {cfg.train.epochs}", flush=True)
    print(f"contrastive_temperature   : {float(cfg.train.get('contrastive_temperature', 0.07)):.4f}", flush=True)
    print(f"Debug max_batches cap     : {cfg.train.get('max_batches', None)}", flush=True)
    print(f"Checkpoint policy         : last every epoch, epoch snapshot every {cfg.train.save_every}, best on eval improvement", flush=True)
    print("-" * 78, flush=True)
    print("Method wiring", flush=True)
    print(f"  Text/Q source           : CLIP annotation feature `{cfg.data.get('text_features', 'ann_feats')}`", flush=True)
    print(f"  DINO K/V source         : image feature `{cfg.data.get('features_name', 'disentangled_self_attn')}`", flush=True)
    print(f"  Visual target           : same DINO region-aware feature `{cfg.data.get('features_name', 'disentangled_self_attn')}`", flush=True)
    print("  Base text projection    : frozen original Talk2DINO projection", flush=True)
    print("  Objective               : pairwise BxB InfoNCE over score(text_i, image_j)", flush=True)
    print(
        "  Trainable modules       : "
        f"XAttnBridge_Clean, CPA-v1{', VCDD' if vcdd is not None else ''}",
        flush=True,
    )
    print(
        "  Bridge architecture     : "
        f"clip_dim={cfg.bridge.clip_dim}, dino_dim={cfg.bridge.dino_dim}, "
        f"d_model={cfg.bridge.d_model}, heads={cfg.bridge.num_heads}, "
        f"layers={cfg.bridge.num_layers}, dropout={cfg.bridge.dropout}, "
        f"gamma_init={cfg.bridge.get('residual_gamma_init', 0.01)}, "
        f"gamma_max={cfg.bridge.get('residual_gamma_max', 0.05)}, "
        f"delta/base_cap={cfg.bridge.get('clamp_delta_base_ratio', 0.10)}",
        flush=True,
    )
    print(
        "  Base-guided attention   : "
        f"enabled={bool(cfg.bridge.get('base_guided_attention', True))}, "
        f"beta={float(cfg.bridge.get('base_guidance_beta', 1.0)):.3f}, "
        f"stopgrad={bool(cfg.bridge.get('base_guidance_stopgrad', True))}, "
        f"normalize={bool(cfg.bridge.get('base_guidance_normalize', True))}, "
        f"temperature={float(cfg.bridge.get('base_guidance_temperature', 1.0)):.3f}",
        flush=True,
    )
    print(
        "  Q/K/V                   : "
        "Q=Linear(CLIP text -> d_model), "
        "K=Linear(DINO tokens -> d_model), "
        "V=Linear(DINO tokens -> d_model)",
        flush=True,
    )
    print("-" * 78, flush=True)
    print(f"Sample text_clip shape    : {tensor_shape(first_train_sample['text_clip'])}", flush=True)
    print(f"Sample patch_tokens shape : {tensor_shape(first_train_sample['patch_tokens'])}", flush=True)
    print(f"Sample visual_embed shape : {tensor_shape(first_train_sample['visual_embed'])}", flush=True)
    if "text_base" in first_train_sample:
        print(f"Sample text_base shape    : {tensor_shape(first_train_sample['text_base'])}", flush=True)
    print("-" * 78, flush=True)
    print("Frozen CLIP confirmation  : features are cached; no CLIP model is trainable", flush=True)
    print("Frozen DINO confirmation  : features are cached; no DINO model is trainable", flush=True)
    print(f"CLIP trainable params     : {trainable_count(frozen.clip_model)}", flush=True)
    print(f"DINO trainable params     : {trainable_count(frozen.model)}", flush=True)
    print(f"Talk2DINO proj trainable  : {projection_trainable_count(frozen)}", flush=True)
    print(
        "WARNING: cached eval mIoU is a fast clean-pipeline metric, not official "
        "Talk2DINO slide-inference parity.",
        flush=True,
    )
    print(f"Trainable parameter count : {trainable_count(bridge)}", flush=True)
    print(f"CPA trainable params      : {trainable_count(cpa) if cpa is not None else 0}", flush=True)
    print(f"VCDD trainable params     : {trainable_count(vcdd) if vcdd is not None else 0}", flush=True)
    print("CLIP/DINO remain frozen; no raw patch tokens, dense_patch_features, precomputed teacher logits, or GT masks are used.", flush=True)
    print(f"train_feature_source      : {train_feature_source}", flush=True)
    print(f"train_eval_mode           : {eval_mode}", flush=True)
    if train_feature_source == "baseline_pth_in_memory":
        print(f"data.features_name        : {cfg.data.get('features_name', 'disentangled_self_attn')}", flush=True)
        print(f"data.text_features        : {cfg.data.get('text_features', 'ann_feats')}", flush=True)
        print(f"data.mmap_features        : {cfg.data.get('mmap_features', True)}", flush=True)
    print(f"data.num_workers          : {cfg.data.num_workers}", flush=True)
    print(f"data.prefetch_factor      : {cfg.data.get('prefetch_factor', 1)}", flush=True)
    print(f"data.shard_cache_size     : {cfg.data.get('shard_cache_size', 2)}", flush=True)
    print(f"data.preload_cache        : {cfg.data.get('preload_cache', False)}", flush=True)
    print(f"data.persistent_workers   : {cfg.data.get('persistent_workers', False)}", flush=True)
    print(f"train.shuffle             : {cfg.train.get('shuffle', True)}", flush=True)
    print(
        "XAttn safety: "
        f"enabled={bool(safety_cfg.enabled)} "
        f"delta_l2_weight={float(safety_cfg.delta_l2_weight):.4g} "
        f"ratio_target={float(safety_cfg.ratio_target):.3f} "
        f"ratio_penalty_weight={float(safety_cfg.ratio_penalty_weight):.4g} "
        f"min_base_mapped_cosine={float(safety_cfg.min_base_mapped_cosine):.3f} "
        f"cosine_penalty_weight={float(safety_cfg.cosine_penalty_weight):.4g} "
        f"attn_prior_weight={float(safety_cfg.attn_prior_weight):.4g} "
        f"attn_prior_temperature={float(safety_cfg.attn_prior_temperature):.4g} "
        f"patch_preserve_weight={float(safety_cfg.patch_preserve_weight):.4g} "
        f"patch_preserve_temperature={float(safety_cfg.patch_preserve_temperature):.4g}",
        flush=True,
    )
    print(
        "Feature source: "
        f"visual_features_name={cfg.data.get('visual_features_name', 'disentangled_self_attn')} "
        f"patch_features_name={cfg.data.get('patch_features_name', 'disentangled_self_attn')} "
        f"allow_patch_visual_same_fallback={cfg.data.get('allow_patch_visual_same_fallback', True)} "
        "raw_patch_tokens=false patch_token_memmap=false",
        flush=True,
    )
    print(
        "CPA: "
        f"enabled={bool(cpa_cfg.enabled)} num_prototypes={int(cpa_cfg.num_prototypes)} "
        f"hidden_dim={int(cpa_cfg.hidden_dim)} prototype_scale={float(cpa_cfg.prototype_scale):.3f} "
        f"aggregation={cpa_cfg.prototype_aggregation} "
        f"prototype_temperature={float(cpa_cfg.prototype_temperature):.3f} "
        f"topk={int(cpa_cfg.topk)} residual_scale={float(cpa_cfg.residual_scale):.3f} "
        f"residual_clip={float(cpa_cfg.residual_clip):.3f} normalize={bool(cpa_cfg.normalize)}",
        flush=True,
    )
    print(
        "CPA loss: "
        f"enabled={bool(cpa_loss_cfg.enabled)} "
        f"prototype_infonce_weight={float(cpa_loss_cfg.prototype_infonce_weight):.3f} "
        f"preserve_weight={float(cpa_loss_cfg.preserve_weight):.4g} "
        f"diversity_weight={float(cpa_loss_cfg.diversity_weight):.4g} "
        f"residual_l1_weight={float(cpa_loss_cfg.residual_l1_weight):.4g} "
        f"confident_margin={float(cpa_loss_cfg.confident_margin):.3f} "
        f"temperature={float(cpa_loss_cfg.temperature):.3f}",
        flush=True,
    )
    print(
        "VCDD: "
        f"enabled={bool(vcdd_cfg.enabled)} attention_path={vcdd_cfg.train_attention_path} "
        f"hidden_dim={int(vcdd_cfg.hidden_dim)} scale={float(vcdd_cfg.scale):.3f} "
        f"residual_clip={float(vcdd_cfg.residual_clip):.3f} topk={int(vcdd_cfg.topk)} "
        f"topk_source={vcdd_cfg.topk_source} detach_attention={bool(vcdd_cfg.detach_attention)} "
        f"vocab_size={len(vcdd_class_names)}",
        flush=True,
    )
    if vcdd_class_names:
        print(f"VCDD first classes        : {vcdd_class_names[:10]}", flush=True)
    print(
        "VCDD loss: "
        f"enabled={bool(vcdd_loss_cfg.enabled)} ce={float(vcdd_loss_cfg.ce_weight):.4g} "
        f"kl={float(vcdd_loss_cfg.kl_weight):.4g} preserve={float(vcdd_loss_cfg.preserve_weight):.4g} "
        f"residual_l1={float(vcdd_loss_cfg.residual_l1_weight):.4g} "
        f"teacher_temperature={float(vcdd_loss_cfg.teacher_temperature):.3f} "
        f"kl_temperature={float(vcdd_loss_cfg.kl_temperature):.3f} "
        f"preserve_temperature={float(vcdd_loss_cfg.preserve_temperature):.3f}",
        flush=True,
    )
    print("CCR: enabled=false", flush=True)
    print("CPA-Router=false USRC=false CPA-v2/v3=false PAMR=false max_train_hours=false", flush=True)
    print("Main InfoNCE: mapped_text vs visual_embed; CPA affects main InfoNCE=false", flush=True)

    start_epoch, best_miou, best_epoch = maybe_auto_resume(
        out, cfg, bridge, cpa, vcdd, optimizer, scheduler, scaler, device
    )
    if eval_mode == "baseline_pth_val_loss" and best_miou == -float("inf"):
        best_miou = float("inf")
    start = time.time()
    epochs = int(cfg.train.epochs)
    max_batches = cfg.train.get("max_batches", None)
    max_batches = None if max_batches is None else int(max_batches)
    if max_batches is None:
        print(f"[{timestamp()}] Full training mode: no train.max_batches cap is active.", flush=True)
    else:
        print(f"[{timestamp()}] DEBUG training cap active: train.max_batches={max_batches}", flush=True)
    if start_epoch > epochs:
        print(
            f"Training already complete for train.epochs={epochs}; "
            f"checkpoint_last.pth is at epoch {start_epoch - 1}.",
            flush=True,
        )
        return
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        bridge.train()
        if cpa is not None:
            cpa.train()
        if vcdd is not None:
            vcdd.train()
        acc = init_epoch_accumulators()
        count = 0
        stop_training = False
        total_steps = len(train_loader)
        if max_batches is not None:
            total_steps = min(total_steps, max_batches)
        progress_interval = max(1, int(cfg.train.get("progress_interval", 10)))
        print(
            f"[{timestamp()}] Starting epoch {epoch:03d}/{epochs:03d}: waiting for first cached batch "
            f"({total_steps} batches, batch_size={cfg.train.batch_size})...",
            flush=True,
        )
        last_step_end = time.time()
        for batch in train_loader:
            data_time = time.time() - last_step_end
            compute_start = time.time()
            text_clip = batch["text_clip"].to(device, non_blocking=True).float()
            if "text_base" in batch:
                text_base = batch["text_base"].to(device, non_blocking=True).float()
            else:
                with torch.no_grad():
                    text_base = frozen._frozen_base_text_to_dino(text_clip).float()
            patches = batch["patch_tokens"].to(device, non_blocking=True).float()
            visual = batch["visual_embed"].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(cfg.train.fp16)):
                scores, safety_stats = pairwise_scores(
                    bridge,
                    text_clip,
                    text_base,
                    patches,
                    visual,
                    return_stats=True,
                )
                contrastive_temperature = max(
                    float(cfg.train.get("contrastive_temperature", 0.07)),
                    1e-6,
                )
                loss_infonce = contrastive_loss(scores / contrastive_temperature)
                safety_losses = bridge_safety_losses(safety_stats, safety_cfg)
                if cpa is not None and bool(cpa_loss_cfg.enabled):
                    if "mapped_text" not in safety_stats or "base_patch_logits" not in safety_stats:
                        raise RuntimeError(
                            "CPA requires mapped_text and semantic base_patch_logits from XAttn"
                        )
                    cpa_losses = cpa_auxiliary_losses(
                        cpa,
                        safety_stats["mapped_text"],
                        visual,
                        patches,
                        safety_stats["base_patch_logits"],
                        cpa_loss_cfg,
                        contrastive_temperature,
                    )
                else:
                    cpa_losses = zero_cpa_losses(safety_stats["delta"])
                if vcdd is not None and bool(vcdd_loss_cfg.enabled):
                    attention_lift = gather_vcdd_attention(
                        vcdd_attention_store,
                        batch["image_id"],
                        device,
                    )
                    mapped_vocab_text = bridge_vocabulary_embeddings(
                        bridge,
                        vcdd_class_clip.to(device),
                        patches,
                        vcdd_class_base.to(device),
                        delta_scale=float(cfg.evaluate.get("xattn_delta_scale", 0.5)),
                    )
                    base_vocab_text = mean_template_embeddings(vcdd_class_base.to(device))
                    dense_base_region_logits = torch.einsum(
                        "brd,cd->bcr",
                        F.normalize(patches.float(), dim=-1),
                        F.normalize(base_vocab_text.float(), dim=-1),
                    )
                    vocab_prototypes = cpa(mapped_vocab_text)
                    region_prototype_logits = compute_prototype_logits(
                        patches,
                        vocab_prototypes,
                        temperature=cpa.prototype_temperature,
                        aggregation=cpa.prototype_aggregation,
                    )
                    dense_base_logits = lift_region_logits(
                        dense_base_region_logits,
                        attention_lift,
                    )
                    dense_prototype_logits = lift_region_logits(
                        region_prototype_logits,
                        attention_lift,
                    )
                    dense_cpa_v1_logits, _ = apply_topk_prototype_residual(
                        dense_base_logits,
                        dense_prototype_logits,
                        topk=cpa.topk,
                        residual_scale=cpa.residual_scale,
                        residual_clip=cpa.residual_clip,
                    )
                    vcdd_losses = vcdd_dense_losses(
                        vcdd,
                        patches,
                        mapped_vocab_text,
                        attention_lift,
                        dense_base_logits,
                        dense_cpa_v1_logits,
                        vcdd_loss_cfg,
                    )
                else:
                    vcdd_losses = zero_vcdd_losses(
                        safety_stats["delta"],
                        len(vcdd_class_names),
                    )
                loss = loss_infonce
                if bool(safety_cfg.enabled):
                    loss = (
                        loss
                        + float(safety_cfg.delta_l2_weight) * safety_losses["loss_delta_l2"]
                        + float(safety_cfg.ratio_penalty_weight) * safety_losses["loss_ratio"]
                        + float(safety_cfg.cosine_penalty_weight) * safety_losses["loss_cos"]
                        + float(safety_cfg.attn_prior_weight) * safety_losses["loss_attn_prior"]
                        + float(safety_cfg.patch_preserve_weight) * safety_losses["loss_patch_preserve"]
                    )
                if cpa is not None and bool(cpa_loss_cfg.enabled):
                    loss = (
                        loss
                        + float(cpa_loss_cfg.prototype_infonce_weight) * cpa_losses["loss_cpa_proto_infonce"]
                        + float(cpa_loss_cfg.preserve_weight) * cpa_losses["loss_cpa_preserve"]
                        + float(cpa_loss_cfg.diversity_weight) * cpa_losses["loss_cpa_diversity"]
                        + float(cpa_loss_cfg.residual_l1_weight) * cpa_losses["loss_cpa_residual_l1"]
                    )
                if vcdd is not None and bool(vcdd_loss_cfg.enabled):
                    loss = (
                        loss
                        + float(vcdd_loss_cfg.ce_weight) * vcdd_losses["loss_vcdd_ce"]
                        + float(vcdd_loss_cfg.kl_weight) * vcdd_losses["loss_vcdd_kl"]
                        + float(vcdd_loss_cfg.preserve_weight) * vcdd_losses["loss_vcdd_preserve"]
                        + float(vcdd_loss_cfg.residual_l1_weight) * vcdd_losses["loss_vcdd_residual_l1"]
                    )
            if not torch.isfinite(loss):
                print("WARNING: loss became NaN/Inf; saving checkpoint_last.pth and stopping cleanly.", flush=True)
                stop_training = True
                break
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_epoch_accumulators(
                acc,
                loss,
                loss_infonce,
                safety_losses,
                safety_stats,
                cpa_losses,
                vcdd_losses,
            )
            acc["data_time"] += data_time
            acc["compute_time"] += time.time() - compute_start
            count += 1
            if count == 1 or count % progress_interval == 0 or count >= total_steps:
                print_epoch_progress(
                    epoch,
                    epochs,
                    count,
                    total_steps,
                    acc,
                    epoch_start,
                    optimizer.param_groups[0]["lr"],
                )
            if max_batches is not None and count >= max_batches:
                break
            last_step_end = time.time()

        elapsed = time.time() - start
        epoch_time = time.time() - epoch_start
        epochs_completed_this_run = max(1, epoch - start_epoch + 1)
        avg_epoch = elapsed / epochs_completed_this_run
        eta = avg_epoch * (epochs - epoch)
        lr = optimizer.param_groups[0]["lr"]
        epoch_metrics = finalize_epoch_accumulators(acc, count)
        epoch_it_s = count / max(epoch_time, 1e-6)
        print(
            f"[{timestamp()}] Epoch {epoch:03d}/{epochs:03d} summary "
            f"loss_total={epoch_metrics['loss_total']:.6f} "
            f"loss_infonce={epoch_metrics['loss_infonce']:.6f} "
            f"loss_delta_l2={epoch_metrics['loss_delta_l2']:.6f} "
            f"loss_ratio={epoch_metrics['loss_ratio']:.6f} "
            f"loss_cos={epoch_metrics['loss_cos']:.6f} "
            f"loss_attn_prior={epoch_metrics['loss_attn_prior']:.6f} "
            f"loss_patch_preserve={epoch_metrics['loss_patch_preserve']:.6f} "
            f"attn_prior_kl={epoch_metrics['attn_prior_kl']:.6f} "
            f"patch_preserve_kl={epoch_metrics['patch_preserve_kl']:.6f} "
            f"loss_cpa_proto_infonce={epoch_metrics['loss_cpa_proto_infonce']:.6f} "
            f"loss_cpa_preserve={epoch_metrics['loss_cpa_preserve']:.6f} "
            f"loss_cpa_diversity={epoch_metrics['loss_cpa_diversity']:.6f} "
            f"loss_cpa_residual_l1={epoch_metrics['loss_cpa_residual_l1']:.6f} "
            f"cpa_residual_abs_mean={epoch_metrics['cpa_residual_abs_mean']:.6f} "
            f"cpa_residual_abs_max={epoch_metrics['cpa_residual_abs_max']:.6f} "
            f"cpa_modified_fraction={epoch_metrics['cpa_modified_fraction']:.4f} "
            f"cpa_confident_patch_frac={epoch_metrics['cpa_confident_patch_frac']:.4f} "
            f"cpa_proto_pairwise_cos_mean={epoch_metrics['cpa_proto_pairwise_cos_mean']:.4f} "
            f"cpa_proto_pairwise_cos_max={epoch_metrics['cpa_proto_pairwise_cos_max']:.4f} "
            f"cpa_topk={epoch_metrics['cpa_topk']:.0f} "
            f"loss_vcdd_ce={epoch_metrics['loss_vcdd_ce']:.6f} "
            f"loss_vcdd_kl={epoch_metrics['loss_vcdd_kl']:.6f} "
            f"loss_vcdd_preserve={epoch_metrics['loss_vcdd_preserve']:.6f} "
            f"loss_vcdd_residual_l1={epoch_metrics['loss_vcdd_residual_l1']:.6f} "
            f"vcdd_reliable_patch_fraction={epoch_metrics['vcdd_reliable_patch_fraction']:.4f} "
            f"vcdd_fallback_used_fraction={epoch_metrics['vcdd_fallback_used_fraction']:.4f} "
            f"vcdd_teacher_top1_prob_mean={epoch_metrics['vcdd_teacher_top1_prob_mean']:.4f} "
            f"vcdd_teacher_margin_mean={epoch_metrics['vcdd_teacher_margin_mean']:.4f} "
            f"vcdd_teacher_entropy_mean={epoch_metrics['vcdd_teacher_entropy_mean']:.4f} "
            f"vcdd_region_residual_abs_mean={epoch_metrics['vcdd_region_residual_abs_mean']:.6f} "
            f"vcdd_dense_residual_abs_mean={epoch_metrics['vcdd_dense_residual_abs_mean']:.6f} "
            f"vcdd_dense_residual_abs_max={epoch_metrics['vcdd_dense_residual_abs_max']:.6f} "
            f"vcdd_modified_fraction={epoch_metrics['vcdd_modified_fraction']:.4f} "
            f"vcdd_attention_sum_mean={epoch_metrics['vcdd_attention_sum_mean']:.6f} "
            f"vcdd_vocab_size={epoch_metrics['vcdd_vocab_size']:.0f} "
            f"base_norm={epoch_metrics['base_norm_mean']:.4f} "
            f"delta_norm={epoch_metrics['delta_norm_mean']:.4f} "
            f"delta/base_mean={epoch_metrics['delta_base_ratio_mean']:.4f} "
            f"delta/base_max={epoch_metrics['delta_base_ratio_max']:.4f} "
        f"cos_base_mapped_mean={epoch_metrics['cosine_base_mapped_mean']:.4f} "
        f"cos_min={epoch_metrics['cosine_base_mapped_min']:.4f} "
        f"gamma={epoch_metrics['gamma']:.4f} "
        f"base_guidance_beta={epoch_metrics['base_guidance_beta']:.4f} "
        f"data/batch={epoch_metrics['data_time']:.3f}s "
            f"xattn-step={epoch_metrics['compute_time']:.3f}s "
            f"lr={lr:.3e} epoch_time={format_seconds(epoch_time)} "
            f"it/s={epoch_it_s:.3f} elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
            flush=True,
        )
        if epoch_metrics["delta_base_ratio_mean"] > float(safety_cfg.warn_if_ratio_above):
            print(
                "WARNING: XAttn correction is becoming large; absent-class risk may increase.",
                flush=True,
            )
        if epoch_metrics["cosine_base_mapped_mean"] < 0.90:
            print(
                "WARNING: mapped text is drifting far from Talk2DINO base space.",
                flush=True,
            )
        if epoch_metrics["delta_base_ratio_mean"] > float(safety_cfg.stop_if_ratio_above):
            print(
                f"[{timestamp()}] SAFETY STOP: delta/base mean "
                f"{epoch_metrics['delta_base_ratio_mean']:.4f} exceeded "
                f"xattn_safety.stop_if_ratio_above={float(safety_cfg.stop_if_ratio_above):.4f}. "
                "Saving checkpoint_last.pth and stopping cleanly.",
                flush=True,
            )
            stop_training = True
        miou = None
        val_loss = None
        is_eval_epoch = epoch % int(cfg.train.save_every) == 0 or epoch == epochs
        if is_eval_epoch:
            if eval_mode == "baseline_pth_val_loss":
                val_loss = evaluate_baseline_val_loss(bridge, val_loader, frozen, device)
                print(
                    f"[{timestamp()}] Epoch {epoch:03d}/{epochs:03d} baseline val contrastive loss={val_loss:.6f}",
                    flush=True,
                )
            else:
                miou = evaluate_cached_miou(
                    bridge,
                    eval_samples,
                    class_clip,
                    class_base,
                    num_classes,
                    ignore_index,
                    device,
                    xattn_delta_scale=float(
                        cfg.evaluate.get("xattn_delta_scale", 0.5)
                    ),
                    xattn_logit_alpha=float(
                        cfg.evaluate.get("xattn_logit_alpha", 0.5)
                    ),
                    xattn_uncertainty_gate_enabled=bool(
                        cfg.evaluate.get("xattn_uncertainty_gate_enabled", True)
                    ),
                    xattn_margin_threshold=(
                        float(cfg.evaluate.get("xattn_margin_threshold", 0.05))
                        if bool(cfg.evaluate.get("xattn_margin_gate_enabled", False))
                        else None
                    ),
                )
                print(f"[{timestamp()}] Epoch {epoch:03d}/{epochs:03d} cached eval coco_stuff mIoU={miou:.2f}%", flush=True)
        if eval_mode == "baseline_pth_val_loss":
            current_best_metric = val_loss if val_loss is not None else epoch_metrics["loss_total"]
            current_best_metric_name = "val_loss" if val_loss is not None else "train_loss_initial"
            is_best = best_epoch == 0 or current_best_metric < best_miou
        else:
            current_best_metric = miou if miou is not None else -epoch_metrics["loss_total"]
            current_best_metric_name = "miou" if miou is not None else "negative_train_loss_initial"
            is_best = best_epoch == 0 or current_best_metric > best_miou
        if is_best:
            best_miou = float(current_best_metric)
            best_epoch = epoch
        checkpoint_metrics = {
            "best_epoch": best_epoch,
            "best_metric_name": current_best_metric_name,
            "loss_total": epoch_metrics["loss_total"],
            "loss_infonce": epoch_metrics["loss_infonce"],
            "loss_delta_l2": epoch_metrics["loss_delta_l2"],
            "loss_ratio": epoch_metrics["loss_ratio"],
            "loss_cos": epoch_metrics["loss_cos"],
            "loss_attn_prior": epoch_metrics["loss_attn_prior"],
            "loss_patch_preserve": epoch_metrics["loss_patch_preserve"],
            "attn_prior_kl": epoch_metrics["attn_prior_kl"],
            "patch_preserve_kl": epoch_metrics["patch_preserve_kl"],
            "loss_cpa_proto_infonce": epoch_metrics["loss_cpa_proto_infonce"],
            "loss_cpa_preserve": epoch_metrics["loss_cpa_preserve"],
            "loss_cpa_diversity": epoch_metrics["loss_cpa_diversity"],
            "loss_cpa_residual_l1": epoch_metrics["loss_cpa_residual_l1"],
            "cpa_residual_abs_mean": epoch_metrics["cpa_residual_abs_mean"],
            "cpa_residual_abs_max": epoch_metrics["cpa_residual_abs_max"],
            "cpa_modified_fraction": epoch_metrics["cpa_modified_fraction"],
            "cpa_confident_patch_frac": epoch_metrics["cpa_confident_patch_frac"],
            "cpa_proto_pairwise_cos_mean": epoch_metrics["cpa_proto_pairwise_cos_mean"],
            "cpa_proto_pairwise_cos_max": epoch_metrics["cpa_proto_pairwise_cos_max"],
            "cpa_topk": epoch_metrics["cpa_topk"],
            "loss_vcdd_ce": epoch_metrics["loss_vcdd_ce"],
            "loss_vcdd_kl": epoch_metrics["loss_vcdd_kl"],
            "loss_vcdd_preserve": epoch_metrics["loss_vcdd_preserve"],
            "loss_vcdd_residual_l1": epoch_metrics["loss_vcdd_residual_l1"],
            "vcdd_reliable_patch_fraction": epoch_metrics["vcdd_reliable_patch_fraction"],
            "vcdd_fallback_used_fraction": epoch_metrics["vcdd_fallback_used_fraction"],
            "vcdd_teacher_top1_prob_mean": epoch_metrics["vcdd_teacher_top1_prob_mean"],
            "vcdd_teacher_margin_mean": epoch_metrics["vcdd_teacher_margin_mean"],
            "vcdd_teacher_entropy_mean": epoch_metrics["vcdd_teacher_entropy_mean"],
            "vcdd_region_residual_abs_mean": epoch_metrics["vcdd_region_residual_abs_mean"],
            "vcdd_region_residual_abs_max": epoch_metrics["vcdd_region_residual_abs_max"],
            "vcdd_dense_residual_abs_mean": epoch_metrics["vcdd_dense_residual_abs_mean"],
            "vcdd_dense_residual_abs_max": epoch_metrics["vcdd_dense_residual_abs_max"],
            "vcdd_modified_fraction": epoch_metrics["vcdd_modified_fraction"],
            "vcdd_attention_sum_mean": epoch_metrics["vcdd_attention_sum_mean"],
            "vcdd_vocab_size": epoch_metrics["vcdd_vocab_size"],
            "miou": miou,
            "val_loss": val_loss,
            "eval_mode": eval_mode,
            "lr": lr,
            "contrastive_temperature": float(cfg.train.get("contrastive_temperature", 0.07)),
            "xattn_safety": OmegaConf.to_container(safety_cfg, resolve=True),
            "cpa_config": OmegaConf.to_container(cpa_cfg, resolve=True),
            "cpa_loss_config": OmegaConf.to_container(cpa_loss_cfg, resolve=True),
            "vcdd_config": OmegaConf.to_container(vcdd_cfg, resolve=True),
            "vcdd_loss_config": OmegaConf.to_container(vcdd_loss_cfg, resolve=True),
            "bridge_base_guidance": {
                "base_guided_attention": bool(cfg.bridge.get("base_guided_attention", True)),
                "base_guidance_beta": float(cfg.bridge.get("base_guidance_beta", 1.0)),
                "base_guidance_stopgrad": bool(cfg.bridge.get("base_guidance_stopgrad", True)),
                "base_guidance_normalize": bool(cfg.bridge.get("base_guidance_normalize", True)),
                "base_guidance_temperature": float(cfg.bridge.get("base_guidance_temperature", 1.0)),
            },
            "delta_base_ratio_mean": epoch_metrics["delta_base_ratio_mean"],
            "delta_base_ratio_max": epoch_metrics["delta_base_ratio_max"],
            "cosine_base_mapped_mean": epoch_metrics["cosine_base_mapped_mean"],
            "cosine_base_mapped_min": epoch_metrics["cosine_base_mapped_min"],
            "gamma": epoch_metrics["gamma"],
            "base_guidance_beta": epoch_metrics["base_guidance_beta"],
            "base_norm_mean": epoch_metrics["base_norm_mean"],
            "delta_norm_mean": epoch_metrics["delta_norm_mean"],
            "data_time": epoch_metrics["data_time"],
            "compute_time": epoch_metrics["compute_time"],
            "avg_dataload_time_sec": epoch_metrics["data_time"],
            "avg_xattn_step_time_sec": epoch_metrics["compute_time"],
            "epoch_it_s": epoch_it_s,
        }
        save_checkpoint_clean(
            out / "checkpoint_last.pth",
            epoch,
            bridge,
            optimizer,
            scheduler,
            best_miou,
            cfg,
            checkpoint_metrics,
            scaler,
            cpa=cpa,
            vcdd=vcdd,
        )
        last_checkpoint = out / "checkpoint_last.pth"
        best_checkpoint = out / "checkpoint_best.pth"
        epoch_checkpoint = out / f"checkpoint_epoch_{epoch:03d}.pth"
        saved_epoch_checkpoint = False
        print(
            f"[{timestamp()}] Saved last checkpoint: {last_checkpoint} "
            f"({file_size_mb(last_checkpoint):.1f} MB, epoch={epoch})",
            flush=True,
        )
        log_row = {
            "timestamp": timestamp(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_metric": best_miou,
            "best_metric_name": current_best_metric_name,
            "loss_total": epoch_metrics["loss_total"],
            "loss_infonce": epoch_metrics["loss_infonce"],
            "loss_delta_l2": epoch_metrics["loss_delta_l2"],
            "loss_ratio": epoch_metrics["loss_ratio"],
            "loss_cos": epoch_metrics["loss_cos"],
            "loss_attn_prior": epoch_metrics["loss_attn_prior"],
            "loss_patch_preserve": epoch_metrics["loss_patch_preserve"],
            "attn_prior_kl": epoch_metrics["attn_prior_kl"],
            "patch_preserve_kl": epoch_metrics["patch_preserve_kl"],
            "loss_cpa_proto_infonce": epoch_metrics["loss_cpa_proto_infonce"],
            "loss_cpa_preserve": epoch_metrics["loss_cpa_preserve"],
            "loss_cpa_diversity": epoch_metrics["loss_cpa_diversity"],
            "loss_cpa_residual_l1": epoch_metrics["loss_cpa_residual_l1"],
            "cpa_residual_abs_mean": epoch_metrics["cpa_residual_abs_mean"],
            "cpa_residual_abs_max": epoch_metrics["cpa_residual_abs_max"],
            "cpa_modified_fraction": epoch_metrics["cpa_modified_fraction"],
            "cpa_confident_patch_frac": epoch_metrics["cpa_confident_patch_frac"],
            "cpa_proto_pairwise_cos_mean": epoch_metrics["cpa_proto_pairwise_cos_mean"],
            "cpa_proto_pairwise_cos_max": epoch_metrics["cpa_proto_pairwise_cos_max"],
            "cpa_topk": epoch_metrics["cpa_topk"],
            "loss_vcdd_ce": epoch_metrics["loss_vcdd_ce"],
            "loss_vcdd_kl": epoch_metrics["loss_vcdd_kl"],
            "loss_vcdd_preserve": epoch_metrics["loss_vcdd_preserve"],
            "loss_vcdd_residual_l1": epoch_metrics["loss_vcdd_residual_l1"],
            "vcdd_reliable_patch_fraction": epoch_metrics["vcdd_reliable_patch_fraction"],
            "vcdd_fallback_used_fraction": epoch_metrics["vcdd_fallback_used_fraction"],
            "vcdd_teacher_top1_prob_mean": epoch_metrics["vcdd_teacher_top1_prob_mean"],
            "vcdd_teacher_margin_mean": epoch_metrics["vcdd_teacher_margin_mean"],
            "vcdd_teacher_entropy_mean": epoch_metrics["vcdd_teacher_entropy_mean"],
            "vcdd_region_residual_abs_mean": epoch_metrics["vcdd_region_residual_abs_mean"],
            "vcdd_region_residual_abs_max": epoch_metrics["vcdd_region_residual_abs_max"],
            "vcdd_dense_residual_abs_mean": epoch_metrics["vcdd_dense_residual_abs_mean"],
            "vcdd_dense_residual_abs_max": epoch_metrics["vcdd_dense_residual_abs_max"],
            "vcdd_modified_fraction": epoch_metrics["vcdd_modified_fraction"],
            "vcdd_attention_sum_mean": epoch_metrics["vcdd_attention_sum_mean"],
            "vcdd_vocab_size": epoch_metrics["vcdd_vocab_size"],
            "base_norm_mean": epoch_metrics["base_norm_mean"],
            "delta_norm_mean": epoch_metrics["delta_norm_mean"],
            "delta_base_ratio_mean": epoch_metrics["delta_base_ratio_mean"],
            "delta_base_ratio_max": epoch_metrics["delta_base_ratio_max"],
            "cosine_base_mapped_mean": epoch_metrics["cosine_base_mapped_mean"],
            "cosine_base_mapped_min": epoch_metrics["cosine_base_mapped_min"],
            "gamma": epoch_metrics["gamma"],
            "base_guidance_beta": epoch_metrics["base_guidance_beta"],
            "miou": miou,
            "val_loss": val_loss,
            "eval_mode": eval_mode,
            "lr": lr,
            "epoch_time_sec": epoch_time,
            "eta_sec": eta,
            "data_time": epoch_metrics["data_time"],
            "compute_time": epoch_metrics["compute_time"],
            "avg_dataload_time_sec": epoch_metrics["data_time"],
            "avg_xattn_step_time_sec": epoch_metrics["compute_time"],
            "epoch_it_s": epoch_it_s,
        }
        with open(train_log_path, "a") as f:
            f.write(json.dumps(log_row) + "\n")
        if is_best:
            save_checkpoint_clean(
                best_checkpoint,
                epoch,
                bridge,
                optimizer,
                scheduler,
                best_miou,
                cfg,
                checkpoint_metrics,
                scaler,
                cpa=cpa,
                vcdd=vcdd,
            )
            print(
                f"[{timestamp()}] Saved best checkpoint: {best_checkpoint} "
                f"({file_size_mb(best_checkpoint):.1f} MB, "
                f"best_epoch={best_epoch}, best_metric={best_miou:.6f})",
                flush=True,
            )
        if epoch % int(cfg.train.save_every) == 0:
            save_checkpoint_clean(
                epoch_checkpoint,
                epoch,
                bridge,
                optimizer,
                scheduler,
                best_miou,
                cfg,
                checkpoint_metrics,
                scaler,
                cpa=cpa,
                vcdd=vcdd,
            )
            saved_epoch_checkpoint = True
            print(
                f"[{timestamp()}] Saved epoch checkpoint: {epoch_checkpoint} "
                f"({file_size_mb(epoch_checkpoint):.1f} MB)",
                flush=True,
            )
        checkpoint_status = {
            "timestamp": timestamp(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_metric": best_miou,
            "best_metric_name": current_best_metric_name,
            "save_every": int(cfg.train.save_every),
            "last_checkpoint": str(last_checkpoint),
            "last_checkpoint_exists": last_checkpoint.exists(),
            "last_checkpoint_size_mb": file_size_mb(last_checkpoint),
            "best_checkpoint": str(best_checkpoint),
            "best_checkpoint_exists": best_checkpoint.exists(),
            "best_checkpoint_size_mb": file_size_mb(best_checkpoint),
            "epoch_checkpoint": str(epoch_checkpoint) if saved_epoch_checkpoint else None,
            "epoch_checkpoint_exists": epoch_checkpoint.exists() if saved_epoch_checkpoint else False,
            "epoch_checkpoint_size_mb": file_size_mb(epoch_checkpoint) if saved_epoch_checkpoint else 0.0,
        }
        with open(out / "checkpoint_status.json", "w") as f:
            json.dump(checkpoint_status, f, indent=2)
        if stop_training:
            break


if __name__ == "__main__":
    main()
