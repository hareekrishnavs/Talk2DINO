"""D1: one 448x448 crop per image per step (random crop + horizontal flip).

Produces a BGR-ordered [3,448,448] float tensor with RAW pixel values in
[0,255] (not yet /255'd or normalised) -- this matches exactly what
DINOTextInference.generate_masks expects as input: it does its own
`image[:, [2,1,0]]` BGR->RGB flip followed by `self.image_transforms`
(resize/÷255/ImageNet-normalise) internally. Loading via PIL (RGB) and
flipping to BGR here, rather than using cv2 directly, avoids the
AVX-512/cv2 issue documented in capture_dino_features.py -- this dataset
never touches cv2 at all."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CROP_SIZE = 448


def load_random_crop_bgr(
    image_path: Path, *, crop_size: int = CROP_SIZE, rng: random.Random,
) -> torch.Tensor:
    """PIL-load, random 448x448 crop (padding up first if the image is
    smaller than the crop in either dimension), random horizontal flip.
    Returns [3,crop_size,crop_size] float32, BGR order, values in [0,255]."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w < crop_size or h < crop_size:
        pad_w, pad_h = max(0, crop_size - w), max(0, crop_size - h)
        padded = Image.new("RGB", (w + pad_w, h + pad_h))
        padded.paste(img, (0, 0))
        img = padded
        w, h = img.size
    x0 = rng.randint(0, w - crop_size)
    y0 = rng.randint(0, h - crop_size)
    crop = img.crop((x0, y0, x0 + crop_size, y0 + crop_size))
    if rng.random() < 0.5:
        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
    array = np.array(crop, dtype=np.float32)  # [H,W,3] RGB, 0..255
    tensor = torch.from_numpy(array).permute(2, 0, 1)  # [3,H,W] RGB
    return tensor[[2, 1, 0], :, :].contiguous()  # -> BGR, for generate_masks' internal re-flip


class CocoCaptionCropDataset(Dataset):
    """Yields (crop_bgr [3,448,448], image_id, present_nouns: list[str]).
    Images with no present nouns after vocabulary filtering are skipped at
    construction time (a step needs at least one present noun for L1's
    target and L2's positive set)."""

    def __init__(
        self, images_dir: Path, image_ids: list[int], file_names: dict[int, str],
        per_image_nouns: dict[int, list[str]], *, seed: int = 0,
    ):
        self.images_dir = Path(images_dir)
        self.file_names = file_names
        self.per_image_nouns = per_image_nouns
        self.image_ids = [i for i in image_ids if per_image_nouns.get(i)]
        self.seed = seed
        if not self.image_ids:
            raise ValueError("no images with at least one vocabulary noun in this split")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = self.image_ids[index]
        # A genuinely fresh RNG per call (not index-derived) -- crop/flip
        # augmentation must vary across epochs, not be a fixed, memorised
        # choice per image. `self.seed` only seeds dataset CONSTRUCTION
        # (which images are included), not per-item augmentation.
        rng = random.Random()
        path = self.images_dir / self.file_names[image_id]
        crop = load_random_crop_bgr(path, rng=rng)
        return {"crop_bgr": crop, "image_id": image_id, "present_nouns": self.per_image_nouns[image_id]}
