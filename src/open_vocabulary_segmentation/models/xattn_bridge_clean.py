import argparse
import csv
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset


METHOD_NAME = "Talk2DINO_XAttnBridge_Clean"


def register_float_image_pipeline():
    from mmseg.datasets import PIPELINES

    if "FloatImage" in PIPELINES.module_dict:
        return

    @PIPELINES.register_module()
    class FloatImage:
        def __call__(self, results):
            results["img"] = results["img"].astype("float32")
            return results


def setup_paths():
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "src" / "open_vocabulary_segmentation"))
    return repo_root


def load_clean_config(path, opts=None):
    path = Path(path)
    cfg = OmegaConf.load(path)
    base = cfg.get("_base_", None)
    if base is not None:
        base_path = path.parent / base
        base_cfg = load_clean_config(base_path)
        cfg = OmegaConf.merge(base_cfg, cfg)
        if "_base_" in cfg:
            del cfg["_base_"]
    if opts:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(opts))
    return cfg


def format_seconds(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def image_transform_448():
    return T.Compose([
        T.Resize(448, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(448),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def build_frozen_talk2dino(cfg, device):
    setup_paths()
    from models import build_model

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    if "xattn_bridge" in model_cfg:
        model_cfg.xattn_bridge.enabled = False
    model = build_model(model_cfg).to(device)
    model.eval()
    model.requires_grad_(False)
    if hasattr(model, "clip_model"):
        model.clip_model.eval()
        model.clip_model.requires_grad_(False)
    if hasattr(model, "model"):
        model.model.eval()
        model.model.requires_grad_(False)
    if hasattr(model, "proj"):
        model.proj.eval()
        model.proj.requires_grad_(False)
    return model


def frozen_status(model):
    clip_trainable = sum(p.numel() for p in model.clip_model.parameters() if p.requires_grad)
    dino_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    return clip_trainable, dino_trainable


@torch.no_grad()
def encode_train_batch(model, images, texts, device):
    images = images.to(device, non_blocking=True)
    texts = texts.to(device, non_blocking=True)
    text_clip = model.encode_text(texts).float()
    text_base = model._frozen_base_text_to_dino(text_clip).float()
    visual_embed, _, patch_tokens = model.encode_image_with_patch_tokens(images)
    if hasattr(model.proj, "project_visual") and model.proj.__class__.__name__ == "DoubleMLP":
        visual_embed = model.proj.project_visual(visual_embed.float())
    visual_embed = visual_embed.float()
    patch_tokens = patch_tokens.float()
    return {
        "text_clip": text_clip.cpu().half(),
        "text_base": text_base.cpu().half(),
        "visual_embed": visual_embed.cpu().half(),
        "patch_tokens": patch_tokens.cpu().half(),
    }


@torch.no_grad()
def encode_eval_image(model, img, device):
    img = img.to(device, non_blocking=True)
    rgb = img[:, [2, 1, 0], :, :]
    img_preprocessed = model.image_transforms(rgb).to(device)
    if "dinov2" in model.model_name:
        patch_tokens = model.model.forward_features(img_preprocessed)["x_norm_patchtokens"]
    elif "dinov3" in model.model_name:
        patch_tokens = model.model.forward_features(img_preprocessed)[:, 5:, :]
    else:
        patch_tokens = model.model.forward_features(img_preprocessed)[:, 1:, :]
    return patch_tokens.float().cpu().half()


class CleanXAttnBridge(nn.Module):
    def __init__(
        self,
        clip_dim=512,
        dino_dim=768,
        d_model=256,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        residual_gamma_init=0.01,
        residual_gamma_max=0.05,
        clamp_delta_base_ratio=0.10,
        base_guided_attention=True,
        base_guidance_beta=1.0,
        base_guidance_stopgrad=True,
        base_guidance_normalize=True,
        base_guidance_temperature=1.0,
    ):
        super().__init__()
        self.residual_gamma_max = float(residual_gamma_max)
        self.clamp_delta_base_ratio = float(clamp_delta_base_ratio)
        self.base_guided_attention = bool(base_guided_attention)
        self.base_guidance_beta = float(base_guidance_beta)
        self.base_guidance_stopgrad = bool(base_guidance_stopgrad)
        self.base_guidance_normalize = bool(base_guidance_normalize)
        self.base_guidance_temperature = float(base_guidance_temperature)
        self.text_proj = nn.Linear(clip_dim, d_model)
        self.patch_k_proj = nn.Linear(dino_dim, d_model)
        self.patch_v_proj = nn.Linear(dino_dim, d_model)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.out_proj = nn.Linear(d_model, dino_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.residual_gamma = nn.Parameter(torch.tensor(float(residual_gamma_init)))
        self._parity_checked = False
        self._skip_zero_init_parity_check = False

    def _base_attention_prior(self, base_text, dino_patches):
        if not self.base_guided_attention or self.base_guidance_beta == 0.0:
            return None
        if self.base_guidance_normalize:
            base_for_sim = F.normalize(base_text, dim=-1)
            patch_for_sim = F.normalize(dino_patches, dim=-1)
        else:
            base_for_sim = base_text
            patch_for_sim = dino_patches
        base_sim = torch.einsum("btd,bnd->btn", base_for_sim, patch_for_sim)
        temp = max(float(self.base_guidance_temperature), 1e-6)
        base_sim = base_sim / temp
        if self.base_guidance_stopgrad:
            base_sim = base_sim.detach()
        return base_sim

    def _manual_attn(self, attn, q, k, v, base_sim):
        embed_dim = q.shape[-1]
        num_heads = attn.num_heads
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )

        q_weight, k_weight, v_weight = attn.in_proj_weight.chunk(3, dim=0)
        if attn.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = attn.in_proj_bias.chunk(3, dim=0)
        q_proj = F.linear(q, q_weight, q_bias)
        k_proj = F.linear(k, k_weight, k_bias)
        v_proj = F.linear(v, v_weight, v_bias)

        bsz, num_text, _ = q_proj.shape
        num_patches = k_proj.shape[1]
        q_heads = q_proj.reshape(bsz, num_text, num_heads, head_dim).transpose(1, 2)
        k_heads = k_proj.reshape(bsz, num_patches, num_heads, head_dim).transpose(1, 2)
        v_heads = v_proj.reshape(bsz, num_patches, num_heads, head_dim).transpose(1, 2)

        logits = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(head_dim)
        if base_sim is not None:
            logits = logits + float(self.base_guidance_beta) * base_sim.unsqueeze(1)
        weights = torch.softmax(logits, dim=-1)
        weights = F.dropout(weights, p=attn.dropout, training=self.training)
        context = torch.matmul(weights, v_heads)
        context = context.transpose(1, 2).reshape(bsz, num_text, embed_dim)
        return attn.out_proj(context), weights

    def forward(self, text_feat, dino_patches, base_text, return_stats=False):
        squeeze_text = text_feat.dim() == 2
        aligned_text_batch = (
            squeeze_text
            and dino_patches.dim() == 3
            and text_feat.shape[0] == dino_patches.shape[0]
        )
        if squeeze_text:
            text_feat = text_feat.unsqueeze(1) if aligned_text_batch else text_feat.unsqueeze(0)
        if base_text.dim() == 2:
            base_text = base_text.unsqueeze(1) if aligned_text_batch else base_text.unsqueeze(0)
        if text_feat.shape[0] == 1 and dino_patches.shape[0] > 1:
            text_feat = text_feat.expand(dino_patches.shape[0], -1, -1)
            base_text = base_text.expand(dino_patches.shape[0], -1, -1)

        text_feat = text_feat.float()
        dino_patches = dino_patches.float()
        base_text = base_text.float()

        q = self.text_proj(F.normalize(text_feat, dim=-1))
        k = self.patch_k_proj(F.normalize(dino_patches, dim=-1))
        v = self.patch_v_proj(F.normalize(dino_patches, dim=-1))
        base_sim = self._base_attention_prior(base_text, dino_patches)
        last_attn = None
        for attn, norm in zip(self.attn_layers, self.norm_layers):
            attn_out, last_attn = self._manual_attn(attn, q, k, v, base_sim)
            q = norm(q + attn_out)
        gamma = self.residual_gamma.clamp(0.0, self.residual_gamma_max)
        delta = gamma * self.out_proj(q)
        if self.clamp_delta_base_ratio > 0:
            base_norm_for_clamp = base_text.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            delta_norm_for_clamp = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            max_delta_norm = float(self.clamp_delta_base_ratio) * base_norm_for_clamp
            delta_scale = torch.clamp(max_delta_norm / delta_norm_for_clamp, max=1.0)
            delta = delta * delta_scale
        mapped_pre_norm = base_text + delta

        if not self._parity_checked and not self._skip_zero_init_parity_check:
            with torch.no_grad():
                diff = (mapped_pre_norm - base_text).abs().max().item()
                if diff >= 1e-5:
                    raise AssertionError(
                        f"{METHOD_NAME} zero-init parity failed: max_abs_diff={diff}"
                    )
                print(
                    f"{METHOD_NAME} zero-init parity passed: max_abs_diff={diff:.8f}",
                    flush=True,
                )
            self._parity_checked = True
        mapped = F.normalize(mapped_pre_norm, dim=-1)
        if return_stats:
            base_norm = base_text.norm(dim=-1).clamp_min(1e-6)
            delta_norm = delta.norm(dim=-1)
            ratio = delta_norm / base_norm
            cosine = F.cosine_similarity(base_text, mapped_pre_norm, dim=-1)
            stats = {
                "base_norm": base_norm,
                "delta_norm": delta_norm,
                "delta_base_ratio": ratio,
                "cosine_base_mapped": cosine,
                "delta": delta,
                "gamma": gamma.detach().reshape(1),
                "base_guidance_beta": delta.new_tensor(float(self.base_guidance_beta)),
                "mapped_text": mapped,
            }
            if base_sim is not None:
                stats["base_sim"] = base_sim
                patch_norm = F.normalize(dino_patches.float(), dim=-1)
                base_norm_for_patch = F.normalize(base_text.float(), dim=-1)
                stats["base_patch_logits"] = torch.einsum(
                    "btd,bnd->btn",
                    base_norm_for_patch,
                    patch_norm,
                )
                stats["xattn_patch_logits"] = torch.einsum(
                    "btd,bnd->btn",
                    mapped.float(),
                    patch_norm,
                )
            if last_attn is not None:
                stats["attention_probs"] = last_attn
                stats["xattn_attn_head_mean"] = last_attn.detach().mean(dim=1)
            if squeeze_text:
                mapped = mapped.squeeze(1) if aligned_text_batch else mapped.squeeze(0)
                stats = {
                    key: (
                        value.squeeze(1)
                        if aligned_text_batch and torch.is_tensor(value) and value.dim() > 1
                        else (
                            value.squeeze(0)
                            if torch.is_tensor(value) and value.dim() > 0
                            else value
                        )
                    )
                    for key, value in stats.items()
                }
            return mapped, stats
        if squeeze_text:
            return mapped.squeeze(1) if aligned_text_batch else mapped.squeeze(0)
        return mapped


def pairwise_scores(bridge, text_clip, text_base, patch_tokens, visual_embed, return_stats=False):
    bsz = text_clip.shape[0]
    text_for_images = text_clip.unsqueeze(0).expand(bsz, -1, -1)
    base_for_images = text_base.unsqueeze(0).expand(bsz, -1, -1)
    if return_stats:
        mapped, stats = bridge(text_for_images, patch_tokens, base_for_images, return_stats=True)
    else:
        mapped = bridge(text_for_images, patch_tokens, base_for_images)
        stats = None
    mapped = F.normalize(mapped.float(), dim=-1)
    visual = F.normalize(visual_embed.float(), dim=-1)
    if visual.dim() == 2:
        scores = torch.einsum("jtd,jd->jt", mapped, visual)
        return (scores, stats) if return_stats else scores
    if visual.dim() == 3:
        scores = torch.einsum("jtd,jrd->jtr", mapped, visual).max(dim=-1).values
        return (scores, stats) if return_stats else scores
    raise ValueError(f"Unsupported visual_embed shape: {tuple(visual.shape)}")


def contrastive_loss(scores):
    target = torch.arange(scores.shape[0], device=scores.device)
    return 0.5 * (
        F.cross_entropy(scores, target)
        + F.cross_entropy(scores.t(), target)
    )


class CleanFeatureDataset(Dataset):
    def __init__(self, feature_dir, shard_cache_size=2, preload=False):
        self.feature_dir = Path(feature_dir)
        manifest_path = self.feature_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        self.format = self.manifest.get("format", "flat_v1")
        self.index = self.manifest.get("index", self.manifest.get("text_index", []))
        self.image_index = {
            str(item["image_id"]): item
            for item in self.manifest.get("image_index", [])
        }
        self.shard_cache_size = max(0, int(shard_cache_size))
        self._cache = OrderedDict()
        if preload:
            self._preload_shards()

    def __len__(self):
        return len(self.index)

    def _preload_shards(self):
        shard_names = {
            item["shard"]
            for item in list(self.index) + list(self.image_index.values())
            if "shard" in item
        }
        print(
            f"Preloading {len(shard_names)} cached feature shards from {self.feature_dir}...",
            flush=True,
        )
        for idx, name in enumerate(sorted(shard_names), 1):
            self._cache[name] = torch.load(
                self.feature_dir / name,
                map_location="cpu",
                weights_only=False,
            )
            if idx == 1 or idx % 25 == 0 or idx == len(shard_names):
                print(f"Preloaded shards: {idx}/{len(shard_names)}", flush=True)
        self.shard_cache_size = max(self.shard_cache_size, len(self._cache))

    def _load_shard(self, name):
        if name not in self._cache:
            self._cache[name] = torch.load(
                self.feature_dir / name,
                map_location="cpu",
                weights_only=False,
            )
            while self.shard_cache_size > 0 and len(self._cache) > self.shard_cache_size:
                self._cache.popitem(last=False)
            if self.shard_cache_size == 0:
                shard = self._cache.pop(name)
                return shard
        else:
            self._cache.move_to_end(name)
        return self._cache[name]

    def __getitem__(self, idx):
        item = self.index[idx]
        shard = self._load_shard(item["shard"])
        sample = shard["samples"][item["offset"]]
        if self.format != "dedup_train_v2":
            return sample
        image_ref = self.image_index[str(sample["image_id"])]
        image_shard = self._load_shard(image_ref["shard"])
        image_sample = image_shard["samples"][image_ref["offset"]]
        return {
            "text_clip": sample["text_clip"],
            "text_base": sample["text_base"],
            "visual_embed": image_sample["visual_embed"],
            "patch_tokens": image_sample["patch_tokens"],
            "image_id": sample["image_id"],
            "annotation_id": sample.get("annotation_id", -1),
        }


def collate_train_features(batch):
    output = {
        "text_clip": torch.stack([x["text_clip"] for x in batch]),
        "visual_embed": torch.stack([x["visual_embed"] for x in batch]),
        "patch_tokens": torch.stack([x["patch_tokens"] for x in batch]),
    }
    if "text_base" in batch[0]:
        output["text_base"] = torch.stack([x["text_base"] for x in batch])
    return output


class CleanBaselinePthFeatureDataset(Dataset):
    def __init__(
        self,
        features_file,
        features_name="disentangled_self_attn",
        text_features="ann_feats",
        mmap=False,
    ):
        self.features_file = Path(features_file)
        self.features_name = features_name
        self.text_features = text_features
        file_size_gb = self.features_file.stat().st_size / (1024 ** 3)
        load_start = time.time()
        print(
            f"Loading baseline-style feature file: {self.features_file} "
            f"({file_size_gb:.2f} GB)",
            flush=True,
        )
        load_kwargs = {
            "map_location": "cpu",
            "weights_only": False,
        }
        if mmap:
            load_kwargs["mmap"] = True
        data = torch.load(self.features_file, **load_kwargs)
        print(
            f"Baseline-style feature file loaded in {format_seconds(time.time() - load_start)}.",
            flush=True,
        )
        build_start = time.time()
        images = {int(img["id"]): img for img in data["images"]}
        self.data = []
        missing = 0
        for ann in data["annotations"]:
            image_id = int(ann["image_id"])
            image = images.get(image_id)
            if image is None or features_name not in image or text_features not in ann:
                missing += 1
                continue
            self.data.append({
                "text_clip": ann[text_features],
                "visual_embed": image[features_name],
                "patch_tokens": image[features_name],
                "image_id": image_id,
                "annotation_id": int(ann.get("id", len(self.data))),
            })
        if missing:
            print(f"WARNING: skipped {missing} annotations with missing features.", flush=True)
        print(
            f"Baseline-style train samples: {len(self.data)} "
            f"(features_name={features_name}, text_features={text_features}, "
            f"index_build={format_seconds(time.time() - build_start)})",
            flush=True,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def save_checkpoint_clean(
    path,
    epoch,
    bridge,
    optimizer,
    scheduler,
    best_miou,
    cfg,
    extra_metrics=None,
    scaler=None,
    cpa=None,
    vab=None,
):
    extra_metrics = extra_metrics or {}
    payload = {
        "epoch": epoch,
        "method_name": METHOD_NAME,
        "model": bridge.state_dict(),
        "bridge": bridge.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_miou": best_miou,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if cpa is not None:
        payload["cpa"] = cpa.state_dict()
    if vab is not None:
        payload["vab"] = vab.state_dict()
    payload.update(extra_metrics)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_bridge_from_checkpoint(checkpoint, cfg, device):
    bridge = CleanXAttnBridge(**OmegaConf.to_container(cfg.bridge, resolve=True)).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("bridge", payload.get("model"))
    bridge.load_state_dict(state)
    bridge._skip_zero_init_parity_check = True
    bridge._parity_checked = True
    bridge.eval()
    return bridge, payload


def build_coco_stuff_eval_dataset(cfg):
    import mmcv
    from mmseg.datasets import build_dataset

    register_float_image_pipeline()
    dset_cfg = mmcv.Config.fromfile(cfg.evaluate.coco_stuff)
    dataset = build_dataset(dset_cfg.data.test)
    return dataset


def build_eval_loader(dataset, cfg):
    from mmseg.datasets import build_dataloader

    return build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=int(cfg.data.num_workers),
        dist=True,
        shuffle=False,
        persistent_workers=int(cfg.data.num_workers) > 0,
        pin_memory=bool(cfg.data.pin_memory),
    )
