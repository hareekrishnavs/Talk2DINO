import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceAwarePrototypeLogitRouter(nn.Module):
    def __init__(
        self,
        topk=5,
        hidden_dim=32,
        init_w_base=0.74,
        init_w_xattn=0.02,
        init_w_cpa=0.24,
        use_margin=True,
        use_entropy=True,
    ):
        super().__init__()
        self.topk = int(topk)
        self.hidden_dim = int(hidden_dim)
        self.use_margin = bool(use_margin)
        self.use_entropy = bool(use_entropy)
        initial = torch.tensor(
            [float(init_w_base), float(init_w_xattn), float(init_w_cpa)]
        )
        if self.topk < 1:
            raise ValueError("cpa_router.topk must be at least 1")
        if self.hidden_dim < 1:
            raise ValueError("cpa_router.hidden_dim must be at least 1")
        if bool((initial <= 0).any()):
            raise ValueError("CPA-Router initial weights must be positive")
        initial = initial / initial.sum()
        self.register_buffer("initial_weights", initial)
        self.mlp = nn.Sequential(
            nn.Linear(9, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 3),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        with torch.no_grad():
            self.mlp[-1].bias.copy_(initial.log())

    @staticmethod
    def _flatten(logits):
        if logits.dim() == 3:
            return logits, None
        if logits.dim() == 4:
            return logits.flatten(2), logits.shape[-2:]
        raise ValueError(
            "CPA-Router logits must be [B,C,N] or [B,C,H,W], got "
            f"{tuple(logits.shape)}"
        )

    def forward(self, base_logits, xattn_logits, prototype_logits):
        if (
            base_logits.shape != xattn_logits.shape
            or base_logits.shape != prototype_logits.shape
        ):
            raise ValueError(
                "CPA-Router base, XAttn, and prototype logits must have equal shapes"
            )
        base, spatial_shape = self._flatten(base_logits)
        xattn, _ = self._flatten(xattn_logits)
        prototype, _ = self._flatten(prototype_logits)
        batch, classes, patches = base.shape
        topk = min(self.topk, classes)

        base_float = base.float()
        xattn_float = xattn.float()
        prototype_float = prototype.float()
        topk_base, topk_indices = base_float.topk(topk, dim=1)
        topk_xattn = xattn_float.gather(1, topk_indices)
        topk_prototype = prototype_float.gather(1, topk_indices)
        probabilities = F.softmax(base_float, dim=1)
        topk_probabilities = probabilities.gather(1, topk_indices)

        top2 = base_float.topk(min(2, classes), dim=1).values
        margin = (
            top2[:, :1] - top2[:, 1:2]
            if top2.shape[1] > 1
            else torch.zeros_like(top2[:, :1])
        )
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
            dim=1, keepdim=True
        )
        margin = margin if self.use_margin else torch.zeros_like(margin)
        entropy = entropy if self.use_entropy else torch.zeros_like(entropy)
        rank = torch.linspace(0.0, 1.0, topk, device=base.device, dtype=torch.float32)
        rank = rank.view(1, topk, 1).expand(batch, topk, patches)

        features = torch.stack(
            (
                topk_base,
                topk_xattn,
                topk_prototype,
                topk_xattn - topk_base,
                topk_prototype - topk_base,
                topk_probabilities,
                margin.expand(-1, topk, -1),
                entropy.expand(-1, topk, -1),
                rank,
            ),
            dim=-1,
        ).permute(0, 2, 1, 3)
        weights = F.softmax(self.mlp(features), dim=-1)
        experts = torch.stack(
            (topk_base, topk_xattn, topk_prototype), dim=-1
        ).permute(0, 2, 1, 3)
        routed_topk = (weights * experts).sum(dim=-1)
        final = base.scatter(
            1,
            topk_indices,
            routed_topk.permute(0, 2, 1).to(dtype=base.dtype),
        )
        topk_mask = torch.zeros_like(base, dtype=torch.bool).scatter(
            1, topk_indices, True
        )
        router_entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)

        stats = {
            "router_weights": weights,
            "router_topk_indices": topk_indices,
            "router_topk_mask": topk_mask,
            "router_w_base_mean": weights[..., 0].mean().detach(),
            "router_w_xattn_mean": weights[..., 1].mean().detach(),
            "router_w_cpa_mean": weights[..., 2].mean().detach(),
            "router_w_base_min": weights[..., 0].min().detach(),
            "router_w_xattn_min": weights[..., 1].min().detach(),
            "router_w_cpa_min": weights[..., 2].min().detach(),
            "router_w_base_max": weights[..., 0].max().detach(),
            "router_w_xattn_max": weights[..., 1].max().detach(),
            "router_w_cpa_max": weights[..., 2].max().detach(),
            "router_entropy_mean": router_entropy.mean().detach(),
            "router_modified_fraction": topk_mask.float().mean().detach(),
            "router_topk": base.new_tensor(float(topk)).detach(),
        }
        if spatial_shape is not None:
            final = final.reshape(batch, classes, *spatial_shape)
            stats["router_topk_mask"] = topk_mask.reshape(batch, classes, *spatial_shape)
        return final, stats


def assert_router_topk_safety(base_logits, final_logits, router_stats):
    outside = ~router_stats["router_topk_mask"]
    difference = (final_logits - base_logits).masked_select(outside)
    max_abs = difference.abs().max() if difference.numel() else base_logits.new_tensor(0.0)
    if float(max_abs.detach().cpu()) != 0.0:
        raise AssertionError(
            f"CPA-Router changed a non-topK logit: max_abs_diff={float(max_abs):.8f}"
        )
    return max_abs
