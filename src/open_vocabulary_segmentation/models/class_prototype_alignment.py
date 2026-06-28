import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassPrototypeAlignmentHead(nn.Module):
    def __init__(
        self,
        dino_dim=768,
        num_prototypes=4,
        hidden_dim=256,
        prototype_scale=0.15,
        prototype_aggregation="logsumexp",
        prototype_temperature=0.07,
        normalize=True,
        topk=5,
        residual_scale=0.25,
        residual_clip=0.5,
        version="v2_conditional_orthogonal",
        use_shared_orthogonal_basis=True,
        conditional_basis_weights=True,
        weight_activation="sigmoid",
        weight_init_value=0.075,
        prototype_dropout=0.25,
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
        self.version = str(version)
        self.use_shared_orthogonal_basis = bool(use_shared_orthogonal_basis)
        self.conditional_basis_weights = bool(conditional_basis_weights)
        self.weight_activation = str(weight_activation)
        self.weight_init_value = float(weight_init_value)
        self.prototype_dropout = float(prototype_dropout)
        if self.num_prototypes < 1:
            raise ValueError("cpa.num_prototypes must be at least 1")
        if self.topk < 1:
            raise ValueError("cpa.topk must be at least 1")
        if self.prototype_aggregation not in {"logsumexp", "max"}:
            raise ValueError(
                "cpa.prototype_aggregation must be `logsumexp` or `max`"
            )
        if self.weight_activation not in {"sigmoid", "softplus"}:
            raise ValueError("cpa.weight_activation must be `sigmoid` or `softplus`")
        if not 0.0 <= self.prototype_dropout < 1.0:
            raise ValueError("cpa.prototype_dropout must be in [0,1)")

        basis = torch.randn(self.dino_dim, self.num_prototypes)
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        self.shared_basis = nn.Parameter(basis.transpose(0, 1).contiguous())
        self.input_proj = nn.Linear(self.dino_dim, int(hidden_dim))
        self.weight_output = nn.Linear(int(hidden_dim), self.num_prototypes)
        nn.init.normal_(self.weight_output.weight, mean=0.0, std=1e-3)
        if self.weight_activation == "sigmoid":
            value = min(max(self.weight_init_value, 1e-5), 1.0 - 1e-5)
            initial_bias = torch.logit(torch.tensor(value)).item()
        else:
            value = max(self.weight_init_value, 1e-5)
            initial_bias = torch.log(torch.expm1(torch.tensor(value))).item()
        nn.init.constant_(self.weight_output.bias, initial_bias)

    def _activate_weights(self, logits):
        if self.weight_activation == "sigmoid":
            return torch.sigmoid(logits)
        return F.softplus(logits)

    def forward(self, mapped_text, base_text=None, return_stats=False):
        if mapped_text.shape[-1] != self.dino_dim:
            raise ValueError(
                f"CPA expected embedding dim {self.dino_dim}, got {mapped_text.shape[-1]}"
            )
        source = F.normalize(mapped_text.float(), dim=-1) if self.normalize else mapped_text.float()
        hidden = F.gelu(self.input_proj(source))
        if self.conditional_basis_weights:
            conditional_weights = self._activate_weights(self.weight_output(hidden))
        else:
            conditional_weights = source.new_full(
                (*source.shape[:-1], self.num_prototypes),
                self.weight_init_value,
            )
        basis = (
            F.normalize(self.shared_basis.float(), dim=-1)
            if self.use_shared_orthogonal_basis
            else self.shared_basis.float()
        )
        offsets = conditional_weights.unsqueeze(-1) * basis
        prototypes = source.unsqueeze(-2) + self.prototype_scale * offsets
        if self.normalize:
            prototypes = F.normalize(prototypes, dim=-1)
        if not return_stats:
            return prototypes
        stats = {
            "cpa_conditional_weights": conditional_weights,
            "cpa_conditional_weight_mean": conditional_weights.detach().mean(),
            "cpa_conditional_weight_min": conditional_weights.detach().min(),
            "cpa_conditional_weight_max": conditional_weights.detach().max(),
            "cpa_basis_norm_mean": basis.detach().norm(dim=-1).mean(),
        }
        return prototypes, stats


def load_cpa_state_compat(cpa, state_dict, log_fn=print):
    current = cpa.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }
    unexpected = sorted(key for key in state_dict if key not in compatible)
    result = cpa.load_state_dict(compatible, strict=False)
    missing = sorted(result.missing_keys)
    if missing or unexpected:
        log_fn(
            "CPA checkpoint compatibility load: "
            f"loaded={len(compatible)} missing={missing} unexpected={unexpected}"
        )
    return missing, unexpected


def _prototype_dropout_mask(similarity, dropout, training):
    if not training or float(dropout) <= 0.0 or similarity.shape[-1] == 1:
        return torch.ones_like(similarity, dtype=torch.bool)
    mask_shape = list(similarity.shape)
    if similarity.dim() == 4:
        mask_shape[-2] = 1
    keep = torch.rand(mask_shape, device=similarity.device) >= float(dropout)
    all_dropped = ~keep.any(dim=-1, keepdim=True)
    keep[..., :1] = keep[..., :1] | all_dropped
    return keep.expand_as(similarity)


def _aggregate_prototype_similarity(
    similarity,
    aggregation,
    temperature,
    prototype_dropout=0.0,
    training=False,
):
    active_mask = _prototype_dropout_mask(
        similarity,
        prototype_dropout,
        training,
    )
    masked_similarity = similarity.masked_fill(~active_mask, -1e4)
    if aggregation == "max":
        return masked_similarity.max(dim=-1).values, active_mask
    if aggregation != "logsumexp":
        raise ValueError(f"Unsupported prototype aggregation: {aggregation}")
    temperature = max(float(temperature), 1e-6)
    return (
        temperature * torch.logsumexp(masked_similarity / temperature, dim=-1),
        active_mask,
    )


def compute_prototype_logits(
    image_features,
    prototype_text,
    temperature=0.07,
    aggregation="logsumexp",
    prototype_dropout=0.0,
    training=False,
    return_stats=False,
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
        logits, active_mask = _aggregate_prototype_similarity(
            similarity,
            aggregation,
            temperature,
            prototype_dropout,
            training,
        )
        if return_stats:
            return logits, {"cpa_prototype_active_fraction": active_mask.float().mean().detach()}
        return logits

    if image.dim() == 3:
        if conditioned:
            if image.shape[0] != prototypes.shape[0]:
                raise ValueError("Conditioned CPA prototypes must match image batch")
            similarity = torch.einsum("bnd,btmd->btnm", image, prototypes)
        elif prototypes.dim() == 3:
            similarity = torch.einsum("bnd,tmd->btnm", image, prototypes)
        else:
            raise ValueError(f"Unsupported prototype shape: {tuple(prototypes.shape)}")
        logits, active_mask = _aggregate_prototype_similarity(
            similarity,
            aggregation,
            temperature,
            prototype_dropout,
            training,
        )
        if return_stats:
            return logits, {"cpa_prototype_active_fraction": active_mask.float().mean().detach()}
        return logits

    raise ValueError(
        f"CPA image features must be [B,D] or [B,N,D], got {tuple(image.shape)}"
    )


def prototype_spread_loss(prototype_text, target_cosine=0.90):
    prototypes = F.normalize(prototype_text.float(), dim=-1)
    num_prototypes = prototypes.shape[-2]
    zero = prototypes.new_tensor(0.0)
    if num_prototypes < 2:
        return zero, zero, zero, zero
    pairwise = torch.matmul(prototypes, prototypes.transpose(-2, -1))
    mask = ~torch.eye(
        num_prototypes,
        dtype=torch.bool,
        device=prototypes.device,
    )
    values = pairwise.masked_select(mask.expand_as(pairwise))
    loss = (values - float(target_cosine)).pow(2).mean()
    return (
        loss,
        values.mean().detach(),
        values.max().detach(),
        values.min().detach(),
    )


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
