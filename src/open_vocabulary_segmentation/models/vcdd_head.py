import torch
import torch.nn as nn
import torch.nn.functional as F


class VocabularyConditionedDenseDistillationHead(nn.Module):
    def __init__(
        self,
        dino_dim=768,
        hidden_dim=256,
        dropout=0.0,
        scale=0.05,
        residual_clip=0.25,
        topk=5,
        topk_source="cpa",
        detach_attention=True,
    ):
        super().__init__()
        self.dino_dim = int(dino_dim)
        self.scale = float(scale)
        self.residual_clip = float(residual_clip)
        self.topk = int(topk)
        self.topk_source = str(topk_source)
        self.detach_attention = bool(detach_attention)
        if self.topk < 1:
            raise ValueError("vcdd.topk must be at least 1")
        if self.topk_source not in {"cpa", "base"}:
            raise ValueError("vcdd.topk_source must be `cpa` or `base`")

        self.adapter = nn.Sequential(
            nn.LayerNorm(self.dino_dim),
            nn.Linear(self.dino_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.dino_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def normalize_attention_lift(self, attention_lift):
        if attention_lift.dim() != 3:
            raise ValueError(
                f"attention_lift must be [B,R,P], got {tuple(attention_lift.shape)}"
            )
        attention = attention_lift.float()
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
        if self.detach_attention:
            attention = attention.detach()
        return attention

    def _topk_mask(self, dense_cpa_v1_logits, dense_base_logits):
        if self.topk_source == "cpa":
            source_logits = dense_cpa_v1_logits
        elif self.topk_source == "base":
            if dense_base_logits is None:
                raise ValueError("dense_base_logits is required when vcdd.topk_source=`base`")
            source_logits = dense_base_logits
        else:
            raise ValueError(f"Unsupported vcdd.topk_source: {self.topk_source}")
        topk = min(max(1, int(self.topk)), source_logits.shape[1])
        indices = source_logits.topk(topk, dim=1).indices
        mask = torch.zeros_like(source_logits, dtype=torch.bool).scatter(1, indices, True)
        return mask, topk

    def forward(
        self,
        semantic_region_features,
        mapped_vocab_text,
        attention_lift,
        dense_cpa_v1_logits,
        dense_base_logits=None,
    ):
        if semantic_region_features.dim() != 3:
            raise ValueError(
                "semantic_region_features must be [B,R,D], got "
                f"{tuple(semantic_region_features.shape)}"
            )
        if mapped_vocab_text.dim() not in {2, 3}:
            raise ValueError(
                "mapped_vocab_text must be [C,D] or [B,C,D], got "
                f"{tuple(mapped_vocab_text.shape)}"
            )
        if dense_cpa_v1_logits.dim() != 3:
            raise ValueError(
                "dense_cpa_v1_logits must be [B,C,P], got "
                f"{tuple(dense_cpa_v1_logits.shape)}"
            )
        batch, regions, dim = semantic_region_features.shape
        classes = mapped_vocab_text.shape[-2]
        patches = attention_lift.shape[-1]
        if dim != self.dino_dim or mapped_vocab_text.shape[-1] != self.dino_dim:
            raise ValueError(
                f"VCDD expected dino_dim={self.dino_dim}, got "
                f"semantic={dim}, text={mapped_vocab_text.shape[-1]}"
            )
        if mapped_vocab_text.dim() == 3 and mapped_vocab_text.shape[0] != batch:
            raise ValueError("mapped_vocab_text batch must match semantic_region_features")
        if attention_lift.shape[:2] != (batch, regions):
            raise ValueError(
                f"attention_lift {tuple(attention_lift.shape)} does not match "
                f"semantic features {tuple(semantic_region_features.shape)}"
            )
        if dense_cpa_v1_logits.shape != (batch, classes, patches):
            raise ValueError(
                "dense_cpa_v1_logits must match [B,C,P], got "
                f"{tuple(dense_cpa_v1_logits.shape)} expected {(batch, classes, patches)}"
            )
        if dense_base_logits is not None and dense_base_logits.shape != dense_cpa_v1_logits.shape:
            raise ValueError("dense_base_logits must match dense_cpa_v1_logits")

        delta_region_feature = self.adapter(semantic_region_features.float())
        text = F.normalize(mapped_vocab_text.float(), dim=-1)
        if text.dim() == 2:
            region_residual_logits = torch.einsum("brd,cd->bcr", delta_region_feature, text)
        else:
            region_residual_logits = torch.einsum("brd,bcd->bcr", delta_region_feature, text)
        attention = self.normalize_attention_lift(attention_lift)
        dense_residual_logits = torch.einsum("brp,bcr->bcp", attention, region_residual_logits)
        dense_residual_logits = dense_residual_logits.clamp(
            -self.residual_clip,
            self.residual_clip,
        )
        topk_mask, topk = self._topk_mask(dense_cpa_v1_logits, dense_base_logits)
        scaled_residual = (
            topk_mask.to(dense_cpa_v1_logits.dtype)
            * self.scale
            * dense_residual_logits.to(dense_cpa_v1_logits.dtype)
        )
        final_dense_logits = dense_cpa_v1_logits + scaled_residual
        selected = dense_residual_logits.masked_select(topk_mask)
        if selected.numel() == 0:
            selected = dense_residual_logits.new_zeros(1)
        attention_sum = attention.sum(dim=1)
        return final_dense_logits, {
            "vcdd_region_residual_abs_mean": region_residual_logits.abs().mean().detach(),
            "vcdd_region_residual_abs_max": region_residual_logits.abs().max().detach(),
            "vcdd_dense_residual_abs_mean": selected.abs().mean().detach(),
            "vcdd_dense_residual_abs_max": selected.abs().max().detach(),
            "vcdd_scaled_residual_abs_mean": scaled_residual.abs().mean().detach(),
            "vcdd_modified_fraction": topk_mask.float().mean().detach(),
            "vcdd_scale": dense_residual_logits.new_tensor(self.scale).detach(),
            "vcdd_topk": dense_residual_logits.new_tensor(float(topk)).detach(),
            "vcdd_attention_sum_mean": attention_sum.mean().detach(),
        }
