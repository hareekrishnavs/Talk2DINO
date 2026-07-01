import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VocabularyAwareBridgeAdapter(nn.Module):
    def __init__(
        self,
        dino_dim=768,
        hidden_dim=128,
        num_heads=4,
        dropout=0.0,
        gamma_init=0.0,
        gamma_max=0.03,
        delta_ratio_clip=0.05,
        normalize_output=True,
        detach_visual=True,
    ):
        super().__init__()
        self.dino_dim = int(dino_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.gamma_max = float(gamma_max)
        self.delta_ratio_clip = float(delta_ratio_clip)
        self.normalize_output = bool(normalize_output)
        self.detach_visual = bool(detach_visual)
        if self.dino_dim % self.num_heads != 0:
            raise ValueError("vab.dino_dim must be divisible by vab.num_heads")
        if not 0.0 <= float(gamma_init) <= self.gamma_max:
            raise ValueError("vab.gamma_init must be in [0, vab.gamma_max]")

        self.query_norm = nn.LayerNorm(self.dino_dim)
        self.visual_norm = nn.LayerNorm(self.dino_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.dino_dim,
            self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.dino_dim),
            nn.Linear(self.dino_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.dino_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _prepare_text(self, mapped_text, visual_batch):
        if mapped_text.dim() == 3:
            if mapped_text.shape[0] not in {1, visual_batch}:
                raise ValueError(
                    "VAB text batch must be one or match semantic features; got "
                    f"{tuple(mapped_text.shape)} and batch={visual_batch}"
                )
            text = mapped_text
            restore = ("three_dim", mapped_text.shape[0])
        elif mapped_text.dim() == 2:
            aligned = mapped_text.shape[0] == visual_batch and visual_batch > 1
            text = mapped_text.unsqueeze(1) if aligned else mapped_text.unsqueeze(0)
            restore = ("aligned" if aligned else "shared", mapped_text.shape[0])
        else:
            raise ValueError(
                "VAB mapped_text must be [T,D], [B,D], or [B,T,D], got "
                f"{tuple(mapped_text.shape)}"
            )
        if text.shape[0] == 1 and visual_batch > 1:
            text = text.expand(visual_batch, -1, -1)
        return text, restore

    @staticmethod
    def _restore_text_shape(value, restore, visual_batch):
        kind, original_batch = restore
        if kind == "aligned":
            return value.squeeze(1)
        if kind == "shared" and visual_batch == 1:
            return value.squeeze(0)
        if kind == "three_dim" and original_batch == 1 and visual_batch == 1:
            return value
        return value

    def forward(
        self,
        mapped_text,
        semantic_region_features,
        original_text_features=None,
        return_stats=False,
    ):
        if semantic_region_features.dim() != 3:
            raise ValueError(
                "VAB semantic_region_features must be [B,R,D], got "
                f"{tuple(semantic_region_features.shape)}"
            )
        if mapped_text.shape[-1] != self.dino_dim:
            raise ValueError(
                f"VAB expected text dim {self.dino_dim}, got {mapped_text.shape[-1]}"
            )
        if semantic_region_features.shape[-1] != self.dino_dim:
            raise ValueError(
                "VAB text and semantic feature dimensions must match; got "
                f"{self.dino_dim} and {semantic_region_features.shape[-1]}"
            )

        visual = semantic_region_features.detach() if self.detach_visual else semantic_region_features
        text, restore = self._prepare_text(mapped_text.float(), visual.shape[0])
        visual = visual.float()
        query = self.query_norm(F.normalize(text, dim=-1))
        key_value = self.visual_norm(F.normalize(visual, dim=-1))
        attn_out, attn_weights = self.cross_attention(
            query,
            key_value,
            key_value,
            need_weights=True,
            average_attn_weights=False,
        )
        raw_delta = self.mlp(attn_out)
        base_norm = text.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        raw_delta_norm = raw_delta.norm(dim=-1, keepdim=True)
        if self.delta_ratio_clip > 0:
            max_norm = self.delta_ratio_clip * base_norm
            scale = torch.clamp(max_norm / raw_delta_norm.clamp_min(1e-6), max=1.0)
            delta = raw_delta * scale
        else:
            delta = raw_delta
        gamma = self.gamma.clamp(0.0, self.gamma_max)
        mapped_pre_norm = text + gamma * delta
        if self.normalize_output:
            mapped_norm = mapped_pre_norm.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            mapped_vab = mapped_pre_norm * (base_norm / mapped_norm)
        else:
            mapped_vab = mapped_pre_norm

        mapped_vab = self._restore_text_shape(mapped_vab, restore, visual.shape[0])
        if not return_stats:
            return mapped_vab

        delta_norm = delta.norm(dim=-1)
        ratio = delta_norm / base_norm.squeeze(-1)
        cosine = F.cosine_similarity(text, mapped_pre_norm, dim=-1)
        probs = attn_weights.float().clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)
        if probs.shape[-1] > 1:
            entropy = entropy / math.log(probs.shape[-1])
        stats = {
            "vab_gamma": gamma,
            "vab_delta": delta,
            "vab_delta_norm_mean": delta_norm.mean().detach(),
            "vab_delta_base_ratio": ratio,
            "vab_delta_base_ratio_mean": ratio.mean().detach(),
            "vab_delta_base_ratio_max": ratio.max().detach(),
            "vab_cos_base_vab": cosine,
            "vab_cos_base_vab_mean": cosine.mean().detach(),
            "vab_cos_base_vab_min": cosine.min().detach(),
            "vab_attn_entropy_mean": entropy.mean().detach(),
            "vab_enabled": delta.new_tensor(1.0).detach(),
        }
        return mapped_vab, stats
