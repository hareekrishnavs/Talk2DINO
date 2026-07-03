import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.local_weights import load_local_clip


def image_to_rgb_array(image):
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    if torch.is_tensor(image):
        value = image.detach().cpu()
        if value.dim() == 4 and value.shape[0] == 1:
            value = value[0]
        if value.dim() == 3 and value.shape[0] in {1, 3, 4}:
            value = value.permute(1, 2, 0)
        image = value.numpy()
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError(
            f"CLIP crop image must be HxW or HxWxC, got shape {array.shape}"
        )
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] >= 3:
        array = array[:, :, :3]
    else:
        raise ValueError(f"Unsupported image channel count: {array.shape[2]}")
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if array.size and float(array.max()) <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def sanitize_crop_boxes(
    boxes,
    image_width,
    image_height,
    padding_ratio=0.10,
    min_area_ratio=0.001,
    max_crops=None,
):
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.numel() == 0:
        return torch.empty((0, 4), dtype=torch.int64)
    if boxes.dim() != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Crop boxes must have shape [N,4], got {tuple(boxes.shape)}")
    if image_width < 1 or image_height < 1:
        raise ValueError("CLIP crop image must have positive width and height")
    padding_ratio = float(padding_ratio)
    min_area_ratio = float(min_area_ratio)
    if padding_ratio < 0.0:
        raise ValueError("clip_image.crop_padding_ratio must be non-negative")
    if not 0.0 <= min_area_ratio <= 1.0:
        raise ValueError("clip_image.min_crop_area_ratio must be in [0,1]")
    if max_crops is not None:
        max_crops = int(max_crops)
        if max_crops < 1:
            raise ValueError("clip_image.max_crops_per_image must be at least 1")
        boxes = boxes[:max_crops]

    resolved = []
    min_side = math.sqrt(min_area_ratio * image_width * image_height)
    for raw_box in boxes:
        x_a, y_a, x_b, y_b = [float(value) for value in raw_box]
        x1, x2 = sorted((x_a, x_b))
        y1, y2 = sorted((y_a, y_b))
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        x1 -= width * padding_ratio
        x2 += width * padding_ratio
        y1 -= height * padding_ratio
        y2 += height * padding_ratio

        center_x = min(max((x1 + x2) * 0.5, 0.0), float(image_width))
        center_y = min(max((y1 + y2) * 0.5, 0.0), float(image_height))
        width = max(x2 - x1, min_side, 1.0)
        height = max(y2 - y1, min_side, 1.0)
        width = min(width, float(image_width))
        height = min(height, float(image_height))
        x1 = min(max(center_x - width * 0.5, 0.0), image_width - width)
        y1 = min(max(center_y - height * 0.5, 0.0), image_height - height)
        x2 = x1 + width
        y2 = y1 + height

        ix1 = max(0, min(image_width - 1, int(math.floor(x1))))
        iy1 = max(0, min(image_height - 1, int(math.floor(y1))))
        ix2 = max(ix1 + 1, min(image_width, int(math.ceil(x2))))
        iy2 = max(iy1 + 1, min(image_height, int(math.ceil(y2))))
        resolved.append([ix1, iy1, ix2, iy2])
    return torch.tensor(resolved, dtype=torch.int64)


def _background_color(image, mode):
    mode = str(mode).lower()
    if mode == "mean":
        return image.reshape(-1, 3).mean(axis=0)
    if mode in {"gray", "grey"}:
        return np.array([127.5, 127.5, 127.5], dtype=np.float32)
    if mode == "black":
        return np.zeros(3, dtype=np.float32)
    if mode == "white":
        return np.full(3, 255.0, dtype=np.float32)
    raise ValueError("clip_image.background must be mean, gray, black, or white")


class ClipCropEncoder:
    def __init__(
        self,
        model_name,
        model_path,
        device="cuda",
        batch_size=32,
        normalize=True,
        cache_features=False,
        cache_dir=None,
        crop_size=224,
        crop_padding_ratio=0.10,
        masked_crop=False,
        background="mean",
        min_crop_area_ratio=0.001,
        max_crops_per_image=32,
        model=None,
        preprocess=None,
    ):
        if not model_name:
            raise ValueError("clip_image.model_name must be set")
        if model is None and not model_path:
            raise ValueError("clip_image.model_path must be set")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("clip_image.device is CUDA but CUDA is unavailable")
        self.model_name = str(model_name)
        self.model_path = None if model_path is None else str(model_path)
        self.batch_size = int(batch_size)
        self.normalize = bool(normalize)
        self.crop_size = int(crop_size)
        self.crop_padding_ratio = float(crop_padding_ratio)
        self.masked_crop = bool(masked_crop)
        self.background = str(background)
        self.min_crop_area_ratio = float(min_crop_area_ratio)
        self.max_crops_per_image = int(max_crops_per_image)
        if self.batch_size < 1:
            raise ValueError("clip_image.batch_size must be at least 1")
        if cache_features:
            raise NotImplementedError(
                "CLIP crop disk caching is intentionally disabled for the smoke stage"
            )
        if cache_dir not in {None, "", "null", "None"}:
            raise ValueError(
                "clip_image.cache_dir requires clip_image.cache_features=true"
            )

        if model is None:
            self.model, self.preprocess = load_local_clip(
                self.model_name,
                device=str(self.device),
                model_path=self.model_path,
            )
        else:
            self.model = model.to(self.device)
            if preprocess is None:
                from clip.clip import _transform

                native_size = int(
                    getattr(self.model.visual, "input_resolution", self.crop_size)
                )
                preprocess = _transform(native_size)
            self.preprocess = preprocess
        self.model.eval()
        self.model.requires_grad_(False)
        native_size = int(getattr(self.model.visual, "input_resolution", self.crop_size))
        if self.crop_size != native_size:
            raise ValueError(
                f"clip_image.crop_size={self.crop_size} does not match "
                f"{self.model_name} native resolution {native_size}"
            )
        self.output_dim = int(
            getattr(self.model.visual, "output_dim", self.model.text_projection.shape[1])
        )

    def _prepare_crops(self, image, boxes, masks=None):
        image = image_to_rgb_array(image)
        height, width = image.shape[:2]
        boxes = sanitize_crop_boxes(
            boxes,
            width,
            height,
            padding_ratio=self.crop_padding_ratio,
            min_area_ratio=self.min_crop_area_ratio,
            max_crops=self.max_crops_per_image,
        )
        if boxes.shape[0] == 0:
            return [], boxes

        mask_array = None
        if self.masked_crop:
            if masks is None:
                raise ValueError("masked_crop=true requires one mask per crop box")
            if torch.is_tensor(masks):
                masks = masks.detach().cpu().numpy()
            mask_array = np.asarray(masks)
            if mask_array.ndim == 2:
                mask_array = mask_array[None]
            if mask_array.ndim != 3 or mask_array.shape[1:] != (height, width):
                raise ValueError(
                    "Crop masks must have shape [N,H,W] matching the RGB image"
                )
            if mask_array.shape[0] < boxes.shape[0]:
                raise ValueError("masked_crop=true requires one mask per retained box")
            mask_array = mask_array[:boxes.shape[0]].astype(bool)
        background = (
            _background_color(image, self.background)
            if mask_array is not None
            else None
        )

        crops = []
        for index, box in enumerate(boxes.tolist()):
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2].copy()
            if mask_array is not None:
                crop_mask = mask_array[index, y1:y2, x1:x2]
                crop[~crop_mask] = background.astype(np.uint8)
            crops.append(Image.fromarray(crop, mode="RGB"))
        return crops, boxes

    @torch.inference_mode()
    def encode(self, image, boxes, masks=None):
        crops, _ = self._prepare_crops(image, boxes, masks=masks)
        if not crops:
            return torch.empty((0, self.output_dim), device=self.device)
        features = []
        for start in range(0, len(crops), self.batch_size):
            batch = torch.stack([
                self.preprocess(crop)
                for crop in crops[start:start + self.batch_size]
            ]).to(self.device)
            encoded = self.model.encode_image(batch).float()
            if self.normalize:
                encoded = F.normalize(encoded, dim=-1)
            features.append(encoded)
        return torch.cat(features, dim=0)
