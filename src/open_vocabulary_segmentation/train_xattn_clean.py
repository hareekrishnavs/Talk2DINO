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


def update_epoch_accumulators(acc, loss_total, loss_infonce, losses, stats):
    acc["loss_total"] += float(loss_total.detach().cpu())
    acc["loss_infonce"] += float(loss_infonce.detach().cpu())
    acc["loss_delta_l2"] += float(losses["loss_delta_l2"].detach().cpu())
    acc["loss_ratio"] += float(losses["loss_ratio"].detach().cpu())
    acc["loss_cos"] += float(losses["loss_cos"].detach().cpu())
    acc["loss_attn_prior"] += float(losses["loss_attn_prior"].detach().cpu())
    acc["loss_patch_preserve"] += float(losses["loss_patch_preserve"].detach().cpu())
    acc["attn_prior_kl"] += float(losses["attn_prior_kl"].detach().cpu())
    acc["patch_preserve_kl"] += float(losses["patch_preserve_kl"].detach().cpu())
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


def maybe_auto_resume(out, cfg, bridge, optimizer, scheduler, scaler, device):
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

    bridge = CleanXAttnBridge(**OmegaConf.to_container(cfg.bridge, resolve=True)).to(device)
    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(cfg.train.epochs) * max(1, len(train_loader))),
        eta_min=float(cfg.train.min_lr),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.fp16))
    safety_cfg = get_safety_cfg(cfg)
    train_log_path = out / "train_log.jsonl"

    print("=" * 78, flush=True)
    print(f"[{timestamp()}] Starting {METHOD_NAME} training", flush=True)
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
    print("  Trainable module        : XAttnBridge_Clean only", flush=True)
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
    print("Only XAttnBridge_Clean parameters are optimized in this script.", flush=True)
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

    start_epoch, best_miou, best_epoch = maybe_auto_resume(out, cfg, bridge, optimizer, scheduler, scaler, device)
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
            if not torch.isfinite(loss):
                print("WARNING: loss became NaN/Inf; saving checkpoint_last.pth and stopping cleanly.", flush=True)
                stop_training = True
                break
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_epoch_accumulators(acc, loss, loss_infonce, safety_losses, safety_stats)
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
        projected_hours = avg_epoch * epochs / 3600.0
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
            f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
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
        if projected_hours > float(cfg.train.max_train_hours):
            print(
                f"WARNING projected {epochs} epochs = {projected_hours:.2f}h "
                f"> train.max_train_hours={cfg.train.max_train_hours}",
                flush=True,
            )

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
            "miou": miou,
            "val_loss": val_loss,
            "eval_mode": eval_mode,
            "lr": lr,
            "contrastive_temperature": float(cfg.train.get("contrastive_temperature", 0.07)),
            "xattn_safety": OmegaConf.to_container(safety_cfg, resolve=True),
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
            "contrastive_temperature": float(cfg.train.get("contrastive_temperature", 0.07)),
            "xattn_safety": OmegaConf.to_container(safety_cfg, resolve=True),
            "bridge_base_guidance": {
                "base_guided_attention": bool(cfg.bridge.get("base_guided_attention", True)),
                "base_guidance_beta": float(cfg.bridge.get("base_guidance_beta", 1.0)),
                "base_guidance_stopgrad": bool(cfg.bridge.get("base_guidance_stopgrad", True)),
                "base_guidance_normalize": bool(cfg.bridge.get("base_guidance_normalize", True)),
                "base_guidance_temperature": float(cfg.bridge.get("base_guidance_temperature", 1.0)),
            },
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
