import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectPresenceCalibrator(nn.Module):
    feature_dim = 15

    def __init__(
        self,
        hidden_dim=64,
        dropout=0.0,
        region_temperature=0.07,
        bias_scale=0.20,
        bias_clip=0.35,
        init_zero=True,
        topk_stats=3,
    ):
        super().__init__()
        self.region_temperature = float(region_temperature)
        self.bias_scale = float(bias_scale)
        self.bias_clip = float(bias_clip)
        self.topk_stats = int(topk_stats)
        if self.region_temperature <= 0:
            raise ValueError("opc.region_temperature must be positive")
        if self.bias_scale < 0 or self.bias_clip < 0:
            raise ValueError("opc.bias_scale and opc.bias_clip must be non-negative")
        if self.topk_stats < 1:
            raise ValueError("opc.topk_stats must be at least 1")

        self.mlp = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        if bool(init_zero):
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def _topk_mean(self, values):
        k = min(self.topk_stats, values.shape[-1])
        return values.topk(k, dim=-1).values.mean(dim=-1)

    @staticmethod
    def _expand_mapped_text(mapped_text, batch_size, num_classes):
        if mapped_text.dim() == 2:
            if mapped_text.shape[0] != num_classes:
                raise ValueError("OPC mapped_text class count does not match region logits")
            return mapped_text.unsqueeze(0).expand(batch_size, -1, -1)
        if mapped_text.dim() == 3:
            if mapped_text.shape[:2] != (batch_size, num_classes):
                raise ValueError("OPC mapped_text must match [B,C] region-logit dimensions")
            return mapped_text
        raise ValueError(
            f"OPC mapped_text must be [C,D] or [B,C,D], got {tuple(mapped_text.shape)}"
        )

    def build_features(self, region_base_logits, region_cpa_logits, mapped_text):
        if region_base_logits.shape != region_cpa_logits.shape:
            raise ValueError("OPC base and CPA region logits must have identical shapes")
        if region_base_logits.dim() != 3:
            raise ValueError("OPC region logits must be [B,C,R]")
        batch_size, num_classes, _ = region_base_logits.shape
        base = region_base_logits.float()
        cpa = region_cpa_logits.float()
        residual = cpa - base

        base_stats = (
            base.max(dim=-1).values,
            self._topk_mean(base),
            base.mean(dim=-1),
            base.std(dim=-1, unbiased=False),
        )
        cpa_stats = (
            cpa.max(dim=-1).values,
            self._topk_mean(cpa),
            cpa.mean(dim=-1),
            cpa.std(dim=-1, unbiased=False),
        )
        residual_stats = (
            residual.max(dim=-1).values,
            self._topk_mean(residual),
            residual.mean(dim=-1),
        )

        class_prob = F.softmax(cpa / self.region_temperature, dim=1)
        prob_stats = (
            class_prob.max(dim=-1).values,
            self._topk_mean(class_prob),
        )
        top2 = cpa.topk(min(2, num_classes), dim=1)
        winner = top2.indices[:, 0]
        if num_classes > 1:
            region_margin = top2.values[:, 0] - top2.values[:, 1]
        else:
            region_margin = torch.zeros_like(top2.values[:, 0])
        winner_mask = F.one_hot(winner, num_classes=num_classes).permute(0, 2, 1).bool()
        class_margin_max = torch.where(
            winner_mask,
            region_margin.unsqueeze(1),
            torch.zeros_like(cpa),
        ).max(dim=-1).values

        text = self._expand_mapped_text(mapped_text.float(), batch_size, num_classes)
        mapped_text_norm = text.norm(dim=-1)
        return torch.stack(
            (*base_stats, *cpa_stats, *residual_stats, *prob_stats, class_margin_max, mapped_text_norm),
            dim=-1,
        )

    def forward(
        self,
        region_base_logits,
        region_cpa_logits,
        mapped_text,
        dense_cpa_logits,
        semantic_region_features=None,
        return_stats=False,
    ):
        if dense_cpa_logits.dim() != 3:
            raise ValueError("OPC dense_cpa_logits must be [B,C,P]")
        if dense_cpa_logits.shape[:2] != region_base_logits.shape[:2]:
            raise ValueError("OPC dense and region logits must share [B,C]")
        features = self.build_features(region_base_logits, region_cpa_logits, mapped_text)
        raw_bias = self.mlp(features)
        presence_bias = (self.bias_scale * raw_bias).clamp(
            -self.bias_clip,
            self.bias_clip,
        )
        final_dense_logits = dense_cpa_logits + presence_bias
        if not return_stats:
            return final_dense_logits, presence_bias

        bias = presence_bias.detach().float()
        feature_values = features.detach().float()
        stats = {
            "opc_bias_mean": bias.mean(),
            "opc_bias_abs_mean": bias.abs().mean(),
            "opc_bias_abs_max": bias.abs().max(),
            "opc_bias_positive_frac": (bias > 0).float().mean(),
            "opc_bias_negative_frac": (bias < 0).float().mean(),
            "opc_feature_mean": feature_values.mean(),
            "opc_feature_std": feature_values.std(unbiased=False),
            "opc_enabled": bias.new_tensor(1.0),
        }
        return final_dense_logits, presence_bias, stats
