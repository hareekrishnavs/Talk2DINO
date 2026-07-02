import torch
import torch.nn as nn


class ClassAdaptiveResidualScaler(nn.Module):
    def __init__(
        self,
        dino_dim=768,
        hidden_dim=64,
        dropout=0.0,
        base_scale=0.25,
        delta_scale_max=0.10,
        alpha_min=0.05,
        alpha_max=0.50,
        init_zero=True,
        allow_missing_init=False,
        force_fixed_alpha=False,
        fixed_alpha=0.25,
    ):
        super().__init__()
        self.dino_dim = int(dino_dim)
        self.base_scale = float(base_scale)
        self.delta_scale_max = float(delta_scale_max)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.init_zero = bool(init_zero)
        self.allow_missing_init = bool(allow_missing_init)
        self.force_fixed_alpha = bool(force_fixed_alpha)
        self.fixed_alpha = float(fixed_alpha)
        if self.alpha_min > self.alpha_max:
            raise ValueError("cars.alpha_min must be <= cars.alpha_max")
        if not self.alpha_min <= self.base_scale <= self.alpha_max:
            raise ValueError("cars.base_scale must lie within [alpha_min, alpha_max]")
        if self.delta_scale_max < 0:
            raise ValueError("cars.delta_scale_max must be non-negative")
        if self.force_fixed_alpha and not (
            self.alpha_min <= self.fixed_alpha <= self.alpha_max
        ):
            raise ValueError("cars.fixed_alpha must lie within [alpha_min, alpha_max]")

        self.scaler = nn.Sequential(
            nn.LayerNorm(self.dino_dim),
            nn.Linear(self.dino_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        if self.init_zero:
            nn.init.zeros_(self.scaler[-1].weight)
            nn.init.zeros_(self.scaler[-1].bias)

    def forward(
        self,
        mapped_text,
        base_logits,
        prototype_logits,
        topk_mask,
        residual_clip=0.5,
    ):
        if mapped_text.dim() not in {2, 3}:
            raise ValueError("CARS mapped_text must be [C,D] or [B,C,D]")
        if mapped_text.shape[-1] != self.dino_dim:
            raise ValueError(
                f"CARS expected embedding dim {self.dino_dim}, "
                f"got {mapped_text.shape[-1]}"
            )
        if base_logits.dim() != 3:
            raise ValueError("CARS base_logits must be [B,C,R] or [B,C,P]")
        if prototype_logits.shape != base_logits.shape:
            raise ValueError("CARS prototype_logits must match base_logits")
        if mapped_text.shape[-2] != base_logits.shape[1]:
            raise ValueError("CARS mapped_text class dimension must match logits")
        if mapped_text.dim() == 3 and mapped_text.shape[0] != base_logits.shape[0]:
            raise ValueError("Batched CARS mapped_text must match the logits batch")
        try:
            torch.broadcast_shapes(tuple(topk_mask.shape), tuple(base_logits.shape))
        except RuntimeError as exc:
            raise ValueError("CARS topk_mask must be broadcastable to logits") from exc

        if self.force_fixed_alpha:
            alpha = mapped_text.new_full(
                (*mapped_text.shape[:-1], 1),
                self.fixed_alpha,
                dtype=torch.float32,
            )
        else:
            raw_delta = self.scaler(mapped_text.float())
            delta = self.delta_scale_max * torch.tanh(raw_delta)
            alpha = (self.base_scale + delta).clamp(
                self.alpha_min,
                self.alpha_max,
            )
        residual = (prototype_logits - base_logits).clamp(
            -float(residual_clip),
            float(residual_clip),
        )
        scaled_residual = torch.where(
            topk_mask,
            alpha.to(residual.dtype) * residual,
            torch.zeros_like(residual),
        )
        final_logits = base_logits + scaled_residual

        alpha_float = alpha.float()
        delta_float = alpha_float - self.base_scale
        stats = {
            "cars_alpha_mean": alpha_float.mean().detach(),
            "cars_alpha_min": alpha_float.min().detach(),
            "cars_alpha_max": alpha_float.max().detach(),
            "cars_alpha_std": alpha_float.std(unbiased=False).detach(),
            "cars_delta_abs_mean": delta_float.abs().mean().detach(),
            "cars_delta_abs_max": delta_float.abs().max().detach(),
            "cars_changed_fraction": (delta_float.abs() > 1e-8).float().mean().detach(),
            "cars_enabled": alpha_float.new_tensor(1.0),
            "cars_scaled_residual": scaled_residual,
        }
        return final_logits, alpha, stats
