import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionSlotDynamicPrototypeHead(nn.Module):
    def __init__(
        self,
        dino_dim=768,
        num_prototypes=4,
        beta=1.0,
        gamma=0.010,
        temperature=0.07,
        topk=15,
        residual_scale=0.05,
        residual_clip=0.20,
        return_aux_during_train=True,
        out_proj_init="identity",
        out_proj_init_scale=0.01,
    ):
        super().__init__()
        self.dino_dim = int(dino_dim)
        self.num_prototypes = int(num_prototypes)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.temperature = float(temperature)
        self.topk = int(topk)
        self.residual_scale = float(residual_scale)
        self.residual_clip = float(residual_clip)
        self.return_aux_during_train = bool(return_aux_during_train)
        self.out_proj_init = str(out_proj_init)
        self.out_proj_init_scale = float(out_proj_init_scale)
        if self.num_prototypes < 1:
            raise ValueError("ic_cpa.num_prototypes must be at least 1")
        if self.topk < 1:
            raise ValueError("ic_cpa.topk must be at least 1")

        self.prototype_slots = nn.Parameter(
            torch.empty(self.num_prototypes, self.dino_dim)
        )
        self.q_proj = nn.Linear(self.dino_dim, self.dino_dim, bias=False)
        self.k_proj = nn.Linear(self.dino_dim, self.dino_dim, bias=False)
        self.v_proj = nn.Linear(self.dino_dim, self.dino_dim, bias=False)
        self.out_proj = nn.Linear(self.dino_dim, self.dino_dim, bias=False)
        # out_proj uses small near-zero initialization to preserve base parity
        # while allowing gradients to flow into attention-slot prototypes.
        nn.init.normal_(self.prototype_slots, mean=0.0, std=0.02)
        self._init_out_proj()
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _init_out_proj(self):
        if self.out_proj_init in {"zero", "zeros"}:
            nn.init.zeros_(self.out_proj.weight)
            return
        if self.out_proj_init in {"identity", "eye"}:
            nn.init.zeros_(self.out_proj.weight)
            with torch.no_grad():
                self.out_proj.weight.copy_(
                    torch.eye(self.dino_dim) * self.out_proj_init_scale
                )
            return
        if self.out_proj_init in {"normal", "small_normal"}:
            nn.init.normal_(
                self.out_proj.weight,
                mean=0.0,
                std=self.out_proj_init_scale,
            )
            return
        raise ValueError(
            "ic_cpa.out_proj_init must be one of zero, identity, or normal; "
            f"got {self.out_proj_init}"
        )

    @staticmethod
    def _tensor_norm(value):
        return float(value.detach().float().norm().cpu())

    def parameter_norms(self):
        norms = {
            "prototype_slots_mean": float(
                self.prototype_slots.detach().float().mean().cpu()
            ),
            "prototype_slots_std": float(
                self.prototype_slots.detach().float().std(unbiased=False).cpu()
            ),
            "prototype_slots_norm": self._tensor_norm(self.prototype_slots),
            "q_proj_weight_norm": self._tensor_norm(self.q_proj.weight),
            "k_proj_weight_norm": self._tensor_norm(self.k_proj.weight),
            "v_proj_weight_norm": self._tensor_norm(self.v_proj.weight),
            "out_proj_weight_norm": self._tensor_norm(self.out_proj.weight),
        }
        if self.out_proj.bias is not None:
            norms["out_proj_bias_norm"] = self._tensor_norm(self.out_proj.bias)
        return norms

    def gradient_norms(self):
        def grad_norm(param):
            if param.grad is None:
                return 0.0
            return float(param.grad.detach().float().norm().cpu())

        norms = {
            "prototype_slots_grad_norm": grad_norm(self.prototype_slots),
            "q_proj_grad_norm": grad_norm(self.q_proj.weight),
            "k_proj_grad_norm": grad_norm(self.k_proj.weight),
            "v_proj_grad_norm": grad_norm(self.v_proj.weight),
            "out_proj_grad_norm": grad_norm(self.out_proj.weight),
        }
        if self.out_proj.bias is not None:
            norms["out_proj_bias_grad_norm"] = grad_norm(self.out_proj.bias)
        return norms

    @staticmethod
    def _upper_pair_mean(sim):
        num_items = sim.shape[-1]
        if num_items < 2:
            return sim.new_tensor(0.0)
        mask = torch.triu(
            torch.ones(num_items, num_items, device=sim.device, dtype=torch.bool),
            diagonal=1,
        )
        return sim[..., mask].mean()

    def _prepare_inputs(self, text_feats, visual_feats, base_logits):
        base_shape = tuple(base_logits.shape)
        if base_logits.dim() == 2:
            base_flat = base_logits.unsqueeze(0)
            spatial_shape = None
        elif base_logits.dim() == 3:
            base_flat = base_logits
            spatial_shape = None
        elif base_logits.dim() == 4:
            spatial_shape = base_logits.shape[-2:]
            base_flat = base_logits.flatten(2)
        else:
            raise ValueError(
                "IC-CPA base_logits must be [C,R], [B,C,R], or [B,C,H,W], "
                f"got {base_shape}"
            )

        if text_feats.dim() == 2:
            text = text_feats.unsqueeze(0)
        elif text_feats.dim() == 3:
            text = text_feats
        else:
            raise ValueError(
                f"IC-CPA text_feats must be [C,D] or [B,C,D], got {tuple(text_feats.shape)}"
            )

        if visual_feats.dim() == 2:
            visual = visual_feats.unsqueeze(0)
        elif visual_feats.dim() == 3:
            visual = visual_feats
        else:
            raise ValueError(
                "IC-CPA visual_feats must be [R,D] or [B,R,D], "
                f"got {tuple(visual_feats.shape)}"
            )

        batch = base_flat.shape[0]
        if text.shape[0] == 1 and batch > 1:
            text = text.expand(batch, -1, -1)
        if visual.shape[0] == 1 and batch > 1:
            visual = visual.expand(batch, -1, -1)
        if text.shape[0] != batch or visual.shape[0] != batch:
            raise ValueError(
                "IC-CPA batch mismatch: "
                f"text={tuple(text.shape)}, visual={tuple(visual.shape)}, "
                f"base_logits={tuple(base_flat.shape)}"
            )
        if text.shape[-1] != self.dino_dim or visual.shape[-1] != self.dino_dim:
            raise ValueError(
                "IC-CPA expected feature dim "
                f"{self.dino_dim}, got text={text.shape[-1]} visual={visual.shape[-1]}"
            )
        if text.shape[1] != base_flat.shape[1]:
            raise ValueError(
                "IC-CPA class mismatch: "
                f"text classes={text.shape[1]} base classes={base_flat.shape[1]}"
            )
        if visual.shape[1] != base_flat.shape[2]:
            raise ValueError(
                "IC-CPA region mismatch: "
                f"visual regions={visual.shape[1]} base regions={base_flat.shape[2]}"
            )
        return text, visual, base_flat, base_shape, spatial_shape

    def _restore_shape(self, final_flat, base_shape, spatial_shape):
        if len(base_shape) == 2:
            return final_flat[0]
        if spatial_shape is None:
            return final_flat
        return final_flat.reshape(final_flat.shape[0], final_flat.shape[1], *spatial_shape)

    def forward(
        self,
        text_feats,
        visual_feats,
        base_logits,
        return_stats=False,
        return_aux=False,
    ):
        text, visual, base_flat, base_shape, spatial_shape = self._prepare_inputs(
            text_feats,
            visual_feats,
            base_logits,
        )
        dtype = base_logits.dtype
        text_norm = F.normalize(text.float(), dim=-1)
        visual_norm = F.normalize(visual.float(), dim=-1)

        slot_queries = text_norm.unsqueeze(2) + self.prototype_slots.float().view(
            1,
            1,
            self.num_prototypes,
            self.dino_dim,
        )
        q = self.q_proj(slot_queries)
        k_img = self.k_proj(visual_norm)
        v_img = self.v_proj(visual_norm)
        attn_logits = torch.einsum("bckd,brd->bckr", q, k_img) / math.sqrt(
            self.dino_dim
        )
        base_guidance = torch.einsum("bcd,brd->bcr", text_norm, visual_norm)
        attn_logits = attn_logits + self.beta * base_guidance.unsqueeze(2)
        attention = F.softmax(attn_logits, dim=-1)
        context = torch.einsum("bckr,brd->bckd", attention, v_img)
        prototypes = F.normalize(
            text_norm.unsqueeze(2) + self.gamma * self.out_proj(context),
            dim=-1,
        )
        prototype_scores = torch.einsum("bckd,brd->bckr", prototypes, visual_norm)
        temperature = max(self.temperature, 1e-6)
        prototype_logits = temperature * torch.logsumexp(
            prototype_scores / temperature,
            dim=2,
        )
        prototype_logits = prototype_logits - temperature * math.log(
            self.num_prototypes
        )

        topk = min(max(1, self.topk), base_flat.shape[1])
        topk_indices = base_flat.float().topk(topk, dim=1).indices
        topk_mask = torch.zeros_like(base_flat, dtype=torch.bool).scatter(
            1,
            topk_indices,
            True,
        )
        residual_unclipped = prototype_logits - base_flat.float()
        residual = residual_unclipped.clamp(
            -self.residual_clip,
            self.residual_clip,
        )
        final_flat = base_flat.float() + torch.where(
            topk_mask,
            self.residual_scale * residual,
            torch.zeros_like(residual),
        )
        final = self._restore_shape(final_flat, base_shape, spatial_shape).to(dtype)
        if not return_stats and not return_aux:
            return final
        raw_changed = topk_mask & (residual.abs() > 1e-6)
        selected_residual = residual.masked_select(topk_mask)
        if selected_residual.numel() == 0:
            residual_abs_mean = residual.new_tensor(0.0)
            residual_abs_max = residual.new_tensor(0.0)
        else:
            residual_abs_mean = selected_residual.abs().mean()
            residual_abs_max = selected_residual.abs().max()
        prototype_vs_text_cos_mean = F.cosine_similarity(
            prototypes.float(),
            text_norm.unsqueeze(2).expand_as(prototypes),
            dim=-1,
        ).mean()
        prototype_pair_sim = torch.einsum(
            "bckd,bcld->bckl",
            F.normalize(prototypes.float(), dim=-1, eps=1e-6),
            F.normalize(prototypes.float(), dim=-1, eps=1e-6),
        )
        attention_pair_sim = torch.einsum(
            "bckr,bclr->bckl",
            F.normalize(attention.float(), dim=-1, eps=1e-6),
            F.normalize(attention.float(), dim=-1, eps=1e-6),
        )
        prototype_base_diff = (prototype_logits - base_flat.float()).abs()
        final_base_diff = (final_flat - base_flat.float()).abs()
        applied_changed = topk_mask & (final_base_diff > 1e-6)
        stats = {
            "text_shape": tuple(text.shape),
            "visual_shape": tuple(visual.shape),
            "prototypes_shape": tuple(prototypes.shape),
            "base_logits_shape": tuple(base_flat.shape),
            "final_logits_shape": tuple(final_flat.shape),
            "prototype_base_abs_mean": prototype_base_diff.mean().detach(),
            "prototype_base_abs_max": prototype_base_diff.max().detach(),
            "final_base_abs_mean": final_base_diff.mean().detach(),
            "final_base_abs_max": final_base_diff.max().detach(),
            "residual_abs_mean": residual_abs_mean.detach(),
            "residual_abs_max": residual_abs_max.detach(),
            "raw_modified_fraction": raw_changed.float().mean().detach(),
            "applied_modified_fraction": applied_changed.float().mean().detach(),
            "modified_fraction": applied_changed.float().mean().detach(),
            "prototype_vs_text_cos_mean": prototype_vs_text_cos_mean.detach(),
            "prototype_slot_pairwise_cos_mean": self._upper_pair_mean(
                prototype_pair_sim
            ).detach(),
            "attention_pairwise_cos_mean": self._upper_pair_mean(
                attention_pair_sim
            ).detach(),
        }
        if return_aux:
            aux = {
                **stats,
                "prototypes": prototypes,
                "prototype_scores": prototype_scores,
                "prototype_logits": prototype_logits,
                "final_logits": final_flat.float(),
                "base_logits": base_flat.float(),
                "residual": residual_unclipped,
                "clipped_residual": residual,
                "attention_weights": attention,
                "topk_mask": topk_mask,
                "text_feats": text,
                "visual_feats": visual,
            }
            return final, aux
        return final, stats
