import math

import torch
import torch.nn as nn


def validate_rcc_compatibility(
    rcc_enabled,
    cpa_enabled,
    cars_enabled=False,
    vpa_enabled=False,
    vab_enabled=False,
    opc_enabled=False,
    cpa_router_enabled=False,
    usrc_enabled=False,
    ccr_enabled=False,
    dcd_enabled=False,
    vcdd_enabled=False,
):
    if rcc_enabled and not cpa_enabled:
        raise ValueError("RCC requires CPA-v1; enable evaluate.cpa_enabled and cpa.enabled")
    incompatible = {
        "CARS": cars_enabled,
        "VPA": vpa_enabled,
        "VAB": vab_enabled,
        "OPC": opc_enabled,
        "CPA-Router": cpa_router_enabled,
        "USRC": usrc_enabled,
        "CCR": ccr_enabled,
        "DCD": dcd_enabled,
        "VCDD": vcdd_enabled,
    }
    active = [name for name, enabled in incompatible.items() if enabled]
    if rcc_enabled and active:
        raise ValueError(
            "RCC cannot be combined with other refinement modules: "
            + ", ".join(active)
        )


class RankCalibratedCPA(nn.Module):
    """Apply the existing CPA residual with a scale determined by candidate rank."""

    def __init__(
        self,
        topk=15,
        residual_clip=0.5,
        topk_source="base",
        schedule="uniform",
        uniform_scale=0.5,
        rank_scales=None,
        scale_start=0.5,
        scale_end=0.5,
    ):
        super().__init__()
        self.topk = int(topk)
        self.residual_clip = float(residual_clip)
        self.topk_source = str(topk_source).lower()
        self.schedule = str(schedule).lower()
        self.uniform_scale = float(uniform_scale)
        self.rank_scales = (
            None if rank_scales is None else [float(scale) for scale in rank_scales]
        )
        self.scale_start = float(scale_start)
        self.scale_end = float(scale_end)
        self._validate_config()

    def _validate_config(self):
        if self.topk < 1:
            raise ValueError("rcc.topk must be at least 1")
        if self.residual_clip < 0.0:
            raise ValueError("rcc.residual_clip must be non-negative")
        if self.topk_source != "base":
            raise ValueError(
                "Only rcc.topk_source=base is supported; candidate ranks must "
                "come from the CPA-v1 base logits"
            )
        if self.schedule not in {"uniform", "piecewise", "linear"}:
            raise ValueError(
                "rcc.schedule must be `uniform`, `piecewise`, or `linear`"
            )
        if self.schedule == "piecewise":
            if self.rank_scales is None:
                raise ValueError(
                    "rcc.rank_scales is required when rcc.schedule=piecewise"
                )
            if len(self.rank_scales) != self.topk:
                raise ValueError(
                    "rcc.rank_scales length must equal rcc.topk, got "
                    f"{len(self.rank_scales)} and {self.topk}"
                )
        values = [self.uniform_scale, self.scale_start, self.scale_end]
        if self.rank_scales is not None:
            values.extend(self.rank_scales)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("RCC scales must be finite")

    def rank_scale_tensor(self, device, dtype, effective_topk=None):
        effective_topk = self.topk if effective_topk is None else int(effective_topk)
        if effective_topk < 1 or effective_topk > self.topk:
            raise ValueError(
                f"effective_topk must be in [1, {self.topk}], got {effective_topk}"
            )
        if self.schedule == "uniform":
            return torch.full(
                (effective_topk,),
                self.uniform_scale,
                device=device,
                dtype=dtype,
            )
        if self.schedule == "piecewise":
            return torch.tensor(
                self.rank_scales[:effective_topk],
                device=device,
                dtype=dtype,
            )
        return torch.linspace(
            self.scale_start,
            self.scale_end,
            self.topk,
            device=device,
            dtype=dtype,
        )[:effective_topk]

    def forward(self, base_logits, prototype_logits):
        if base_logits.shape != prototype_logits.shape:
            raise ValueError(
                "base_logits and prototype_logits must match, got "
                f"{tuple(base_logits.shape)} and {tuple(prototype_logits.shape)}"
            )
        if base_logits.dim() not in {3, 4}:
            raise ValueError(
                "RCC dense logits must be [B,C,N] or [B,C,H,W], got "
                f"{tuple(base_logits.shape)}"
            )

        effective_topk = min(self.topk, base_logits.shape[1])
        topk_indices = base_logits.topk(effective_topk, dim=1).indices
        topk_mask = torch.zeros_like(base_logits, dtype=torch.bool).scatter(
            1, topk_indices, True
        )
        rank_scales = self.rank_scale_tensor(
            base_logits.device,
            base_logits.dtype,
            effective_topk=effective_topk,
        )
        scale_shape = [1, effective_topk] + [1] * (base_logits.dim() - 2)
        scattered_scales = torch.zeros_like(base_logits).scatter(
            1,
            topk_indices,
            rank_scales.reshape(scale_shape).expand_as(topk_indices),
        )
        raw_residual = (prototype_logits - base_logits).clamp(
            -self.residual_clip,
            self.residual_clip,
        )
        rcc_residual = torch.where(
            topk_mask,
            scattered_scales * raw_residual,
            torch.zeros_like(raw_residual),
        )
        final_logits = base_logits + rcc_residual

        selected = rcc_residual.masked_select(topk_mask)
        outside_difference = (final_logits - base_logits).masked_select(~topk_mask)
        zero = base_logits.new_tensor(0.0)
        stats = {
            "rcc_residual": rcc_residual,
            "rcc_topk_mask": topk_mask,
            "rcc_topk_indices": topk_indices,
            "rcc_topk": base_logits.new_tensor(float(effective_topk)).detach(),
            "rcc_scale_min": rank_scales.min().detach(),
            "rcc_scale_max": rank_scales.max().detach(),
            "rcc_scale_mean": rank_scales.mean().detach(),
            "rcc_changed_fraction": (rcc_residual != 0).float().mean().detach(),
            "rcc_residual_abs_mean": (
                selected.abs().mean().detach() if selected.numel() else zero
            ),
            "rcc_residual_abs_max": (
                selected.abs().max().detach() if selected.numel() else zero
            ),
            "rcc_non_topk_changed_fraction": (
                (outside_difference != 0).float().mean().detach()
                if outside_difference.numel()
                else zero
            ),
        }
        return final_logits, stats
