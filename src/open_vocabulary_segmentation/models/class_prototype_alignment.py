import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _off_diagonal_values(vectors):
    vectors = F.normalize(vectors.float(), dim=-1)
    count = vectors.shape[-2]
    if count < 2:
        return vectors.new_zeros(0)
    pairwise = torch.matmul(vectors, vectors.transpose(-2, -1))
    mask = ~torch.eye(count, dtype=torch.bool, device=vectors.device)
    return pairwise.masked_select(mask.expand_as(pairwise))


class ClassPrototypeAlignmentHead(nn.Module):
    def __init__(
        self,
        dino_dim=768,
        num_prototypes=4,
        hidden_dim=256,
        prototype_scale=None,
        prototype_aggregation="logsumexp",
        prototype_temperature=0.07,
        normalize=True,
        topk=5,
        residual_scale=0.25,
        residual_clip=0.5,
        version="v3_tangent_space",
        use_shared_tangent_basis=True,
        conditional_basis_weights=False,
        prototype_radius_min=0.10,
        prototype_radius_max=0.35,
        prototype_radius_init=0.22,
        prototype_dropout=0.10,
    ):
        super().__init__()
        self.dino_dim = int(dino_dim)
        self.num_prototypes = int(num_prototypes)
        self.hidden_dim = int(hidden_dim)
        self.prototype_aggregation = str(prototype_aggregation)
        self.prototype_temperature = float(prototype_temperature)
        self.normalize = bool(normalize)
        self.topk = int(topk)
        self.residual_scale = float(residual_scale)
        self.residual_clip = float(residual_clip)
        self.version = str(version)
        self.use_shared_tangent_basis = bool(use_shared_tangent_basis)
        self.conditional_basis_weights = bool(conditional_basis_weights)
        self.prototype_radius_min = float(prototype_radius_min)
        self.prototype_radius_max = float(prototype_radius_max)
        self.prototype_radius_init = float(prototype_radius_init)
        self.prototype_dropout = float(prototype_dropout)
        # Accepted only so CPA-v1 configs can still instantiate this head.
        self.prototype_scale = None if prototype_scale is None else float(prototype_scale)

        if self.version != "v3_tangent_space":
            raise ValueError(f"Unsupported CPA version: {self.version}")
        if not self.use_shared_tangent_basis:
            raise ValueError("CPA-v3 requires cpa.use_shared_tangent_basis=true")
        if self.conditional_basis_weights:
            raise ValueError("CPA-v3 conditional basis weights are disabled")
        if self.num_prototypes < 1:
            raise ValueError("cpa.num_prototypes must be at least 1")
        if self.topk < 1:
            raise ValueError("cpa.topk must be at least 1")
        if self.prototype_aggregation not in {"logsumexp", "max"}:
            raise ValueError("cpa.prototype_aggregation must be `logsumexp` or `max`")
        if not 0.0 <= self.prototype_dropout < 1.0:
            raise ValueError("cpa.prototype_dropout must be in [0, 1)")
        if not (
            0.0 < self.prototype_radius_min
            < self.prototype_radius_max
            and self.prototype_radius_min
            <= self.prototype_radius_init
            <= self.prototype_radius_max
        ):
            raise ValueError(
                "CPA radii must satisfy 0 < radius_min <= radius_init <= radius_max"
            )

        basis = torch.randn(self.dino_dim, self.num_prototypes)
        basis = torch.linalg.qr(basis, mode="reduced").Q.transpose(0, 1)
        self.shared_basis = nn.Parameter(F.normalize(basis, dim=-1))
        self.radius_head = nn.Sequential(
            nn.Linear(self.dino_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_prototypes),
        )
        nn.init.zeros_(self.radius_head[-1].weight)
        radius_fraction = (
            (self.prototype_radius_init - self.prototype_radius_min)
            / (self.prototype_radius_max - self.prototype_radius_min)
        )
        radius_fraction = min(max(radius_fraction, 1e-6), 1.0 - 1e-6)
        nn.init.constant_(
            self.radius_head[-1].bias,
            math.log(radius_fraction / (1.0 - radius_fraction)),
        )

    def forward(self, mapped_text, base_text=None, return_details=False):
        if mapped_text.shape[-1] != self.dino_dim:
            raise ValueError(
                f"CPA expected embedding dim {self.dino_dim}, got {mapped_text.shape[-1]}"
            )
        mapped_unit = F.normalize(mapped_text.float(), dim=-1)
        basis = F.normalize(self.shared_basis.float(), dim=-1)
        view_shape = (1,) * (mapped_unit.dim() - 1) + basis.shape
        directions = basis.view(view_shape).expand(*mapped_unit.shape[:-1], *basis.shape)
        parallel = (directions * mapped_unit.unsqueeze(-2)).sum(dim=-1, keepdim=True)
        tangent_directions = F.normalize(
            directions - parallel * mapped_unit.unsqueeze(-2),
            dim=-1,
        )
        radius_logits = self.radius_head(mapped_unit)
        radii = self.prototype_radius_min + (
            self.prototype_radius_max - self.prototype_radius_min
        ) * torch.sigmoid(radius_logits)
        prototypes = F.normalize(
            mapped_unit.unsqueeze(-2) + radii.unsqueeze(-1) * tangent_directions,
            dim=-1,
        )
        if not return_details:
            return prototypes
        return prototypes, {
            "mapped_unit": mapped_unit,
            "tangent_directions": tangent_directions,
            "radii": radii,
        }


def _aggregate_prototype_similarity(
    similarity,
    aggregation,
    temperature,
    prototype_dropout=0.0,
    training=False,
    return_active_fraction=False,
):
    active_fraction = similarity.new_tensor(1.0)
    if training and float(prototype_dropout) > 0.0 and similarity.shape[-1] > 1:
        mask_shape = list(similarity.shape)
        if similarity.dim() >= 4:
            mask_shape[-2] = 1
        random_values = torch.rand(mask_shape, device=similarity.device)
        active = random_values >= float(prototype_dropout)
        all_dropped = ~active.any(dim=-1, keepdim=True)
        fallback = torch.zeros_like(active).scatter_(-1, random_values.argmax(-1, keepdim=True), True)
        active = active | (all_dropped & fallback)
        similarity = similarity.masked_fill(~active, torch.finfo(similarity.dtype).min)
        active_fraction = active.float().mean().detach()
    if aggregation == "max":
        logits = similarity.max(dim=-1).values
    elif aggregation == "logsumexp":
        temperature = max(float(temperature), 1e-6)
        logits = temperature * torch.logsumexp(similarity / temperature, dim=-1)
    else:
        raise ValueError(f"Unsupported prototype aggregation: {aggregation}")
    if return_active_fraction:
        return logits, active_fraction
    return logits


def compute_prototype_logits(
    image_features,
    prototype_text,
    temperature=0.07,
    aggregation="logsumexp",
    prototype_dropout=0.0,
    training=False,
    return_active_fraction=False,
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
    elif image.dim() == 3:
        if conditioned:
            if image.shape[0] != prototypes.shape[0]:
                raise ValueError("Conditioned CPA prototypes must match image batch")
            similarity = torch.einsum("bnd,btmd->btnm", image, prototypes)
        elif prototypes.dim() == 3:
            similarity = torch.einsum("bnd,tmd->btnm", image, prototypes)
        else:
            raise ValueError(f"Unsupported prototype shape: {tuple(prototypes.shape)}")
    else:
        raise ValueError(
            f"CPA image features must be [B,D] or [B,N,D], got {tuple(image.shape)}"
        )
    return _aggregate_prototype_similarity(
        similarity,
        aggregation,
        temperature,
        prototype_dropout=prototype_dropout,
        training=training,
        return_active_fraction=return_active_fraction,
    )


def prototype_geometry(prototype_text, tangent_directions=None):
    prototype_values = _off_diagonal_values(prototype_text)
    zero = prototype_text.new_tensor(0.0)
    stats = {
        "cpa_proto_pairwise_cos_mean": prototype_values.mean().detach()
        if prototype_values.numel() else zero,
        "cpa_proto_pairwise_cos_max": prototype_values.max().detach()
        if prototype_values.numel() else zero,
        "cpa_proto_pairwise_cos_min": prototype_values.min().detach()
        if prototype_values.numel() else zero,
    }
    tangent_values = (
        _off_diagonal_values(tangent_directions)
        if tangent_directions is not None else prototype_values.new_zeros(0)
    )
    stats.update({
        "cpa_tangent_dir_pairwise_cos_mean": tangent_values.mean().detach()
        if tangent_values.numel() else zero,
        "cpa_tangent_dir_pairwise_cos_abs_mean": tangent_values.abs().mean().detach()
        if tangent_values.numel() else zero,
    })
    return stats


def tangent_orthogonality_loss(tangent_directions):
    values = _off_diagonal_values(tangent_directions)
    if not values.numel():
        return tangent_directions.new_tensor(0.0)
    return values.pow(2).mean()


def prototype_diversity_loss(prototype_text, margin=0.90):
    values = _off_diagonal_values(prototype_text)
    zero = prototype_text.new_tensor(0.0)
    if not values.numel():
        return zero, zero, zero
    loss = F.relu(values - float(margin)).pow(2).mean()
    return loss, values.mean().detach(), values.max().detach()


def load_cpa_state_dict_compatible(cpa, state_dict, context="checkpoint"):
    current = cpa.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }
    shape_mismatches = [
        key
        for key, value in state_dict.items()
        if key in current and current[key].shape != value.shape
    ]
    incompatible = cpa.load_state_dict(compatible, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = [key for key in state_dict if key not in current]
    print(
        f"CPA weights from {context}: missing_keys={missing} "
        f"unexpected_keys={unexpected} shape_mismatches={shape_mismatches}",
        flush=True,
    )
    return missing, unexpected + shape_mismatches


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
        1, topk_indices, True
    )
    raw_residual = (prototype_logits - base_logits).clamp(
        -float(residual_clip), float(residual_clip)
    )
    cpa_residual = torch.where(
        topk_mask,
        float(residual_scale) * raw_residual,
        torch.zeros_like(raw_residual),
    )
    final_logits = base_logits.clone()
    residual_for_logits = cpa_residual.to(dtype=base_logits.dtype)
    final_logits[topk_mask] = (
        base_logits[topk_mask] + residual_for_logits[topk_mask]
    )
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
