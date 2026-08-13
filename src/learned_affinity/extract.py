"""D2: on-the-fly DINO feature + S_0 (text-patch score) extraction for a
single 448x448 training crop. Reuses model.generate_masks (the SAME
function the sliding-window evaluation calls once per window) directly --
one crop is already exactly one window, so no sliding-window wrapper is
needed. Zero reimplementation of resize/normalise/similarity math.

model.generate_masks is decorated @torch.no_grad() -- everything it
computes (patch features, S_0) is inherently non-differentiable, which is
correct: DINOv2 and CLIP are frozen (T1), only LearnedMetric's own forward
pass (applied OUTSIDE this function, on the detached features this returns)
needs gradients."""
from __future__ import annotations

import torch


class _PatchFeatureTap:
    """Mirrors the observer-hook interface Part E's WindowFeatureCapture and
    the online E3 cache both use (`model.masker.affinity_oracle_observer`,
    called as `observer(normalized_features, raw_scores)`). Only the
    features are captured here -- generate_masks already RETURNS simmap
    (S_0) directly as its second output, so no separate tap is needed for
    that half."""

    def __init__(self):
        self.features: torch.Tensor | None = None

    def __call__(self, normalized_features: torch.Tensor, raw_scores: torch.Tensor) -> None:
        self.features = normalized_features.detach()


def extract_training_sample(
    model, crop_bgr: torch.Tensor, text_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """crop_bgr: [3,448,448] or [1,3,448,448], BGR, raw [0,255] values (see
    crop_dataset.load_random_crop_bgr). text_embedding: [C,dim] L2-normalised,
    the per-step gathered vocabulary subset (D4).

    Returns:
      features [1024,768] float32, L2-normalised DINOv2 patch features (the
        input to LearnedMetric).
      raw_scores [C,1024] float32, S_0 -- pre-sigmoid text-patch similarity
        (SCORE_STAGE = "pre_sigmoid_pre_upsample_pre_stitch", matching the
        E3 oracle's own raw_scores convention exactly).
    """
    if crop_bgr.ndim == 3:
        crop_bgr = crop_bgr.unsqueeze(0)
    device = next(model.parameters()).device
    crop_bgr = crop_bgr.to(device=device, dtype=torch.float32)
    text_embedding = text_embedding.to(device=device, dtype=torch.float32)

    tap = _PatchFeatureTap()
    previous = getattr(model.masker, "affinity_oracle_observer", None)
    model.masker.affinity_oracle_observer = tap
    try:
        _mask, simmap = model.generate_masks(crop_bgr, text_embedding)
    finally:
        model.masker.affinity_oracle_observer = previous

    if tap.features is None:
        raise RuntimeError(
            "affinity_oracle_observer was never called -- masker.forward_seg's "
            "observer hook point may have changed; do not silently proceed"
        )

    channels, h, w = tap.features.shape[1:]
    features = tap.features[0].reshape(channels, h * w).transpose(0, 1).contiguous()  # [1024,768]

    c = simmap.shape[1]
    raw_scores = simmap[0].reshape(c, h * w).contiguous().float()  # [C,1024]

    return features.float(), raw_scores
