import os
from math import sqrt

import timm
import torch
import torch.nn as nn
from timm.layers.pos_embed import resample_abs_pos_embed


DEFAULT_WEIGHT_DIR = os.environ.get("TALK2DINO_WEIGHT_DIR", "/scratch/haree/weights")

CLIP_WEIGHT_PATHS = {
    "ViT-B/16": "ViT-B-16.pt",
    "ViT-B/32": "ViT-B-32.pt",
}

DINO_MODEL_INFO = {
    "dinov2_vitb14_reg": ("vit_base_patch14_reg4_dinov2", "dinov2_vitb14_reg4_pretrain.pth", 5),
    "dinov2_vitl14_reg": ("vit_large_patch14_reg4_dinov2", "dinov2_vitl14_reg4_pretrain.pth", 5),
    "vit_base_patch16_dinov3.lvd1689m": (
        "vit_base_patch16_dinov3.lvd1689m",
        "vit_base_patch16_dinov3.lvd1689m.pth",
        5,
    ),
    "vit_large_patch16_dinov3.lvd1689m": (
        "vit_large_patch16_dinov3.lvd1689m",
        "vit_large_patch16_dinov3.lvd1689m.pth",
        5,
    ),
}


def resolve_weight_path(path_or_name, weight_dir=DEFAULT_WEIGHT_DIR):
    if path_or_name is None:
        return None
    if os.path.isabs(path_or_name):
        return path_or_name
    return os.path.join(weight_dir, path_or_name)


def require_file(path, description):
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing local {description}: {path}. "
            "Auto-download is disabled; put the file there or pass an explicit local path."
        )
    return path


def _load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "teacher", "student"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if isinstance(checkpoint, dict):
        checkpoint = {
            key.removeprefix("module.").removeprefix("backbone."): value
            for key, value in checkpoint.items()
        }
    return checkpoint


def load_state_dict_from_local_file(model, weights_path, description, strict=False):
    weights_path = require_file(weights_path, description)
    state_dict = _load_checkpoint(weights_path)
    if isinstance(state_dict, dict):
        if "register_tokens" in state_dict and "reg_token" not in state_dict:
            state_dict["reg_token"] = state_dict.pop("register_tokens")
        if not hasattr(model, "mask_token"):
            state_dict.pop("mask_token", None)

        if "pos_embed" in state_dict and hasattr(model, "pos_embed"):
            source = state_dict["pos_embed"]
            target = model.pos_embed
            if source.shape != target.shape:
                has_cls_pos = source.shape[1] == int(sqrt(source.shape[1] - 1)) ** 2 + 1
                if has_cls_pos and target.shape[1] != source.shape[1]:
                    source = source[:, 1:, :]

                source_grid = int(sqrt(source.shape[1]))
                target_grid = int(sqrt(target.shape[1]))
                if source_grid * source_grid == source.shape[1] and target_grid * target_grid == target.shape[1]:
                    source = resample_abs_pos_embed(
                        source,
                        new_size=[target_grid, target_grid],
                        old_size=[source_grid, source_grid],
                        num_prefix_tokens=0,
                    )

                if source.shape == target.shape:
                    state_dict["pos_embed"] = source
                else:
                    state_dict.pop("pos_embed")

    incompatible = model.load_state_dict(state_dict, strict=strict)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Could not load {description} strictly from {weights_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model


class DinoTimmWrapper(nn.Module):
    def __init__(self, backbone, num_global_tokens):
        super().__init__()
        self.backbone = backbone
        self.num_global_tokens = num_global_tokens
        self.blocks = backbone.blocks
        self.num_heads = backbone.blocks[-1].attn.num_heads

    def forward(self, x, is_training=False):
        tokens = self.backbone.forward_features(x)
        return {
            "x_norm_clstoken": tokens[:, 0, :],
            "x_norm_patchtokens": tokens[:, self.num_global_tokens:, :],
        }

    def forward_features(self, x):
        tokens = self.backbone.forward_features(x)
        return {
            "x_norm_clstoken": tokens[:, 0, :],
            "x_norm_patchtokens": tokens[:, self.num_global_tokens:, :],
        }


def load_local_clip(model_name, device="cpu", model_path=None, weight_dir=DEFAULT_WEIGHT_DIR):
    import clip

    model_path = model_path or CLIP_WEIGHT_PATHS.get(model_name)
    model_path = resolve_weight_path(model_path, weight_dir)
    require_file(model_path, f"CLIP weights for {model_name}")
    return clip.load(model_path, device=device, download_root=weight_dir)


def load_local_vision_backbone(model_name, img_size, weights_path=None, weight_dir=DEFAULT_WEIGHT_DIR):
    if model_name not in DINO_MODEL_INFO:
        raise ValueError(
            f"No local loading rule for backbone '{model_name}'. "
            "Add it to src/local_weights.py with a local weight filename."
        )

    timm_name, default_weights, num_global_tokens = DINO_MODEL_INFO[model_name]
    weights_path = resolve_weight_path(weights_path or default_weights, weight_dir)
    require_file(weights_path, f"backbone weights for {model_name}")

    model = timm.create_model(
        timm_name,
        pretrained=False,
        num_classes=0,
        img_size=img_size,
    )
    load_state_dict_from_local_file(model, weights_path, f"backbone weights for {model_name}", strict=False)

    if "dinov2" in model_name:
        return DinoTimmWrapper(model, num_global_tokens)
    return model
