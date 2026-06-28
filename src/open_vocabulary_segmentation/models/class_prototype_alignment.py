import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassPrototypeAlignmentHead(nn.Module):
    def __init__(
        self,
        dino_dim=768,
        num_prototypes=4,
        hidden_dim=256,
        prototype_scale=0.10,
        prototype_aggregation="logsumexp",
        prototype_temperature=0.07,
        normalize=True,
        topk=5,
        residual_scale=0.25,
        residual_clip=0.5,
    ):
        super().__init__()
        self.dino_dim = int(dino_dim)
        self.num_prototypes = int(num_prototypes)
        self.prototype_scale = float(prototype_scale)
        self.prototype_aggregation = str(prototype_aggregation)
        self.prototype_temperature = float(prototype_temperature)
        self.normalize = bool(normalize)
        self.topk = int(topk)
        self.residual_scale = float(residual_scale)
        self.residual_clip = float(residual_clip)
        if self.num_prototypes < 1:
            raise ValueError("cpa.num_prototypes must be at least 1")
        if self.topk < 1:
            raise ValueError("cpa.topk must be at least 1")
        if self.prototype_aggregation not in {"logsumexp", "max"}:
            raise ValueError(
                "cpa.prototype_aggregation must be `logsumexp` or `max`"
            )

        self.input_proj = nn.Linear(self.dino_dim, int(hidden_dim))
        self.output_proj = nn.Linear(
            int(hidden_dim),
            self.num_prototypes * self.dino_dim,
        )
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, mapped_text, base_text=None):
        if mapped_text.shape[-1] != self.dino_dim:
            raise ValueError(
                f"CPA expected embedding dim {self.dino_dim}, got {mapped_text.shape[-1]}"
            )
        source = F.normalize(mapped_text.float(), dim=-1) if self.normalize else mapped_text.float()
        hidden = F.gelu(self.input_proj(source))
        offsets = self.output_proj(hidden).reshape(
            *mapped_text.shape[:-1],
            self.num_prototypes,
            self.dino_dim,
        )
        prototypes = source.unsqueeze(-2) + self.prototype_scale * offsets
        if self.normalize:
            prototypes = F.normalize(prototypes, dim=-1)
        return prototypes


def _aggregate_prototype_similarity(similarity, aggregation, temperature):
    if aggregation == "max":
        return similarity.max(dim=-1).values
    if aggregation != "logsumexp":
        raise ValueError(f"Unsupported prototype aggregation: {aggregation}")
    temperature = max(float(temperature), 1e-6)
    return temperature * torch.logsumexp(similarity / temperature, dim=-1)


def compute_prototype_logits(
    image_features,
    prototype_text,
    temperature=0.07,
    aggregation="logsumexp",
):
    image = F.normalize(image_features.float(), dim=-1)
    prototypes = F.normalize(prototype_text.float(), dim=-1)
    conditioned = prototypes.dim() == 4

    if image.dim() == 2:
        if conditioned:
            if image.shape[0] != prototypes.shape[0]:
                raise ValueError("Conditioned CPA prototypes must match image batch")
            similarity = torch.einsum("bd,btmd->btm", image, prototypes)
        elif prototypes.dim() == 3:
            similarity = torch.einsum("bd,tmd->btm", image, prototypes)
        else:
            raise ValueError(f"Unsupported prototype shape: {tuple(prototypes.shape)}")
        return _aggregate_prototype_similarity(similarity, aggregation, temperature)

    if image.dim() == 3:
        if conditioned:
            if image.shape[0] != prototypes.shape[0]:
                raise ValueError("Conditioned CPA prototypes must match image batch")
            similarity = torch.einsum("bnd,btmd->btnm", image, prototypes)
        elif prototypes.dim() == 3:
            similarity = torch.einsum("bnd,tmd->btnm", image, prototypes)
        else:
            raise ValueError(f"Unsupported prototype shape: {tuple(prototypes.shape)}")
        return _aggregate_prototype_similarity(
            similarity,
            aggregation,
            temperature,
        )

    raise ValueError(
        f"CPA image features must be [B,D] or [B,N,D], got {tuple(image.shape)}"
    )


def prototype_diversity_loss(prototype_text, margin=0.90):
    prototypes = F.normalize(prototype_text.float(), dim=-1)
    num_prototypes = prototypes.shape[-2]
    zero = prototypes.new_tensor(0.0)
    if num_prototypes < 2:
        return zero, zero, zero
    pairwise = torch.matmul(prototypes, prototypes.transpose(-2, -1))
    mask = ~torch.eye(
        num_prototypes,
        dtype=torch.bool,
        device=prototypes.device,
    )
    values = pairwise.masked_select(mask.expand_as(pairwise))
    loss = F.relu(values - float(margin)).pow(2).mean()
    return loss, values.mean().detach(), values.max().detach()


def apply_topk_prototype_residual(
    base_logits,
    prototype_logits,
    topk=5,
    residual_scale=0.25,
    residual_clip=0.5,
):
    if base_logits.shape != prototype_logits.shape:
        raise ValueError(
            "base_logits and prototype_logits must match, got "
            f"{tuple(base_logits.shape)} and {tuple(prototype_logits.shape)}"
        )
    if base_logits.dim() not in {3, 4}:
        raise ValueError(
            f"CPA dense logits must be [B,C,N] or [B,C,H,W], got {tuple(base_logits.shape)}"
        )
    num_classes = base_logits.shape[1]
    topk = min(max(1, int(topk)), num_classes)
    topk_indices = base_logits.topk(topk, dim=1).indices
    topk_mask = torch.zeros_like(base_logits, dtype=torch.bool).scatter(
        1,
        topk_indices,
        True,
    )
    raw_residual = (prototype_logits - base_logits).clamp(
        -float(residual_clip),
        float(residual_clip),
    )
    cpa_residual = torch.where(
        topk_mask,
        float(residual_scale) * raw_residual,
        torch.zeros_like(raw_residual),
    )
    final_logits = base_logits + cpa_residual
    selected = cpa_residual.masked_select(topk_mask)
    stats = {
        "cpa_residual": cpa_residual,
        "cpa_topk_mask": topk_mask,
        "cpa_topk_indices": topk_indices,
        "cpa_residual_abs_mean": selected.abs().mean().detach(),
        "cpa_residual_abs_max": selected.abs().max().detach(),
        "cpa_modified_fraction": topk_mask.float().mean().detach(),
        "cpa_topk": base_logits.new_tensor(float(topk)).detach(),
    }
    return final_logits, stats


def assert_topk_safety(base_logits, final_logits, cpa_stats):
    outside_topk = ~cpa_stats["cpa_topk_mask"]
    difference = (final_logits - base_logits).masked_select(outside_topk)
    max_abs = difference.abs().max() if difference.numel() else base_logits.new_tensor(0.0)
    if float(max_abs.detach().cpu()) != 0.0:
        raise AssertionError(
            f"CPA changed a non-topK logit: max_abs_diff={float(max_abs):.8f}"
        )
    return max_abs
