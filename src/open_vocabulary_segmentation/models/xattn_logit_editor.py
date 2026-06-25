import torch
import torch.nn.functional as F


def topk_xattn_logit_editor(
    base_logits,
    xattn_logits,
    topk=5,
    min_base_prob=0.02,
    alpha=0.5,
    delta_scale=0.5,
    margin_threshold=0.10,
    margin_temperature=0.05,
    use_uncertainty_gate=True,
):
    if base_logits.shape != xattn_logits.shape:
        raise ValueError(
            f"base_logits and xattn_logits must match, got "
            f"{tuple(base_logits.shape)} vs {tuple(xattn_logits.shape)}"
        )
    if base_logits.dim() not in {3, 4}:
        raise ValueError(f"Unsupported logits shape: {tuple(base_logits.shape)}")

    class_dim = 1
    num_classes = base_logits.shape[class_dim]
    topk = max(1, min(int(topk), num_classes))
    probs = F.softmax(base_logits.float(), dim=class_dim)
    topk_idx = base_logits.topk(topk, dim=class_dim).indices
    topk_mask = torch.zeros_like(base_logits, dtype=torch.bool)
    topk_mask.scatter_(class_dim, topk_idx, True)
    prob_mask = probs > float(min_base_prob)
    allowed_mask = topk_mask & prob_mask

    top2 = base_logits.topk(min(2, num_classes), dim=class_dim).values
    if top2.shape[class_dim] < 2:
        margin = torch.zeros_like(base_logits[:, :1])
    else:
        margin = top2[:, :1] - top2[:, 1:2]
    if use_uncertainty_gate:
        temp = max(float(margin_temperature), 1e-6)
        gate = torch.sigmoid((float(margin_threshold) - margin) / temp)
    else:
        gate = torch.ones_like(margin)

    edited_logits = base_logits.clone()
    correction = float(alpha) * float(delta_scale) * gate * (xattn_logits - base_logits)
    edited_logits = torch.where(allowed_mask, base_logits + correction, edited_logits)

    stats = {
        "edited_fraction": allowed_mask.float().mean().detach(),
        "gate_mean": gate.float().mean().detach(),
        "gate_max": gate.float().max().detach(),
        "margin_mean": margin.float().mean().detach(),
        "margin_min": margin.float().min().detach(),
        "topk": torch.as_tensor(float(topk), device=base_logits.device),
        "min_base_prob": torch.as_tensor(float(min_base_prob), device=base_logits.device),
    }
    return edited_logits, stats
