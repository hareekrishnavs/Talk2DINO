import torch


def aggregate_legacy_probability_maps(
    outputs,
    num_scales,
    hflip=False,
    scale_weights=None,
):
    """Average original-size probability maps without changing legacy CPA."""
    num_scales = int(num_scales)
    variants_per_scale = 2 if bool(hflip) else 1
    expected = num_scales * variants_per_scale
    if num_scales < 1 or len(outputs) != expected:
        raise ValueError(
            f"MSA expected {expected} outputs, got {len(outputs)}"
        )
    reference_shape = outputs[0].shape
    if any(output.shape != reference_shape for output in outputs):
        raise ValueError("MSA outputs must share class order and spatial shape")

    per_scale = []
    for index in range(num_scales):
        start = index * variants_per_scale
        variants = outputs[start:start + variants_per_scale]
        per_scale.append(
            variants[0]
            if len(variants) == 1
            else torch.stack(variants, dim=0).mean(dim=0)
        )

    if scale_weights is None:
        weights = outputs[0].new_ones(num_scales)
    else:
        weights = outputs[0].new_tensor(scale_weights)
        if weights.numel() != num_scales:
            raise ValueError("MSA scale weights must match msa.scales")
        if not torch.isfinite(weights).all() or bool((weights <= 0).any()):
            raise ValueError("MSA scale weights must be finite and positive")
    weights = weights / weights.sum()

    if num_scales == 1:
        final = per_scale[0]
    else:
        final = sum(
            weight * output
            for weight, output in zip(weights, per_scale)
        )
    return final, per_scale
