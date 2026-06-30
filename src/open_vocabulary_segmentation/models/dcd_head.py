import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseConsistencyDistillationHead(nn.Module):
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
        self.hidden_dim = int(hidden_dim)
        self.scale = float(scale)
        self.residual_clip = float(residual_clip)
        self.topk = int(topk)
        self.topk_source = str(topk_source)
        self.detach_attention = bool(detach_attention)
        if self.topk < 1:
            raise ValueError("dcd.topk must be at least 1")
        if self.topk_source not in {"cpa", "base"}:
            raise ValueError("dcd.topk_source must be `cpa` or `base`")

        self.adapter = nn.Sequential(
            nn.LayerNorm(self.dino_dim),
            nn.Linear(self.dino_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.dino_dim),
        )
        final_linear = self.adapter[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def normalize_attention_lift(self, attention_lift):
        if attention_lift.dim() != 3:
            raise ValueError(
                "attention_lift must be [B,R,P], got "
                f"{tuple(attention_lift.shape)}"
            )
        attention = attention_lift.float()
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
        if self.detach_attention:
            attention = attention.detach()
        return attention

    def _topk_mask(self, dense_cpa_v1_logits, dense_base_logits):
        source = self.topk_source
        if source == "cpa":
            logits = dense_cpa_v1_logits
        elif source == "base":
            if dense_base_logits is None:
                raise ValueError(
                    "dense_base_logits is required when dcd.topk_source=`base`"
                )
            logits = dense_base_logits
        else:
            raise ValueError(f"Unsupported dcd.topk_source: {source}")

        topk = min(max(1, self.topk), logits.shape[1])
        indices = logits.topk(topk, dim=1).indices
        return torch.zeros_like(logits, dtype=torch.bool).scatter(1, indices, True), topk

    def forward(
        self,
        semantic_region_features,
        mapped_text,
        attention_lift,
        dense_cpa_v1_logits,
        dense_base_logits=None,
    ):
        if semantic_region_features.dim() != 3:
            raise ValueError(
                "semantic_region_features must be [B,R,D], got "
                f"{tuple(semantic_region_features.shape)}"
            )
        if mapped_text.dim() not in {2, 3}:
            raise ValueError(
                f"mapped_text must be [T,D] or [B,T,D], got {tuple(mapped_text.shape)}"
            )
        if dense_cpa_v1_logits.dim() != 3:
            raise ValueError(
                "dense_cpa_v1_logits must be [B,T,P], got "
                f"{tuple(dense_cpa_v1_logits.shape)}"
            )
        batch, regions, dim = semantic_region_features.shape
        texts = mapped_text.shape[-2]
        patches = attention_lift.shape[-1]
        expected = (batch, texts, patches)
        if dim != self.dino_dim or mapped_text.shape[-1] != self.dino_dim:
            raise ValueError(
                f"DCD expected dino_dim={self.dino_dim}, got "
                f"semantic={dim}, text={mapped_text.shape[-1]}"
            )
        if mapped_text.dim() == 3 and mapped_text.shape[0] != batch:
            raise ValueError(
                "Image-conditioned mapped_text batch must match semantic features, got "
                f"{tuple(mapped_text.shape)} and {tuple(semantic_region_features.shape)}"
            )
        if dense_cpa_v1_logits.shape != expected:
            raise ValueError(
                "dense_cpa_v1_logits shape must match [B,T,P], got "
                f"{tuple(dense_cpa_v1_logits.shape)} expected {expected}"
            )
        if attention_lift.shape[:2] != (batch, regions):
            raise ValueError(
                "attention_lift shape must match semantic regions, got "
                f"{tuple(attention_lift.shape)} and {tuple(semantic_region_features.shape)}"
            )
        if dense_base_logits is not None and dense_base_logits.shape != dense_cpa_v1_logits.shape:
            raise ValueError(
                "dense_base_logits must match dense_cpa_v1_logits, got "
                f"{tuple(dense_base_logits.shape)} and {tuple(dense_cpa_v1_logits.shape)}"
            )

        delta_region_feature = self.adapter(semantic_region_features.float())
        mapped_text_unit = F.normalize(mapped_text.float(), dim=-1)
        if mapped_text_unit.dim() == 2:
            region_residual_logits = torch.einsum(
                "brd,td->btr",
                delta_region_feature,
                mapped_text_unit,
            )
        else:
            region_residual_logits = torch.einsum(
                "brd,btd->btr",
                delta_region_feature,
                mapped_text_unit,
            )

        attention = self.normalize_attention_lift(attention_lift)
        dense_residual_logits = torch.einsum(
            "brp,btr->btp",
            attention,
            region_residual_logits,
        )
        dense_residual_logits = dense_residual_logits.clamp(
            -self.residual_clip,
            self.residual_clip,
        )
        topk_mask, topk = self._topk_mask(dense_cpa_v1_logits, dense_base_logits)
        final_dense_logits = dense_cpa_v1_logits + (
            topk_mask.to(dense_cpa_v1_logits.dtype)
            * self.scale
            * dense_residual_logits.to(dense_cpa_v1_logits.dtype)
        )

        selected = dense_residual_logits.masked_select(topk_mask)
        if selected.numel() == 0:
            selected = dense_residual_logits.new_zeros(1)
        attention_sum = attention.sum(dim=1)
        stats = {
            "dcd_region_residual_abs_mean": region_residual_logits.abs().mean().detach(),
            "dcd_region_residual_abs_max": region_residual_logits.abs().max().detach(),
            "dcd_dense_residual_abs_mean": selected.abs().mean().detach(),
            "dcd_dense_residual_abs_max": selected.abs().max().detach(),
            "dcd_modified_fraction": topk_mask.float().mean().detach(),
            "dcd_scale": dense_residual_logits.new_tensor(self.scale).detach(),
            "dcd_topk": dense_residual_logits.new_tensor(float(topk)).detach(),
            "dcd_attention_sum_mean": attention_sum.mean().detach(),
            "dcd_attention_sum_min": attention_sum.min().detach(),
            "dcd_attention_sum_max": attention_sum.max().detach(),
        }
        return final_dense_logits, stats
