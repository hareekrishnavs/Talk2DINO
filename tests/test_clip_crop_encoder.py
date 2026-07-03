import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


MODELS = Path(__file__).resolve().parents[1] / "src/open_vocabulary_segmentation/models"
sys.path.insert(0, str(MODELS))

from clip_crop_encoder import ClipCropEncoder, image_to_rgb_array, sanitize_crop_boxes


def test_image_conversion_handles_grayscale_and_chw_float():
    gray = Image.fromarray(np.full((5, 7), 128, dtype=np.uint8))
    assert image_to_rgb_array(gray).shape == (5, 7, 3)
    chw = torch.ones(3, 4, 6) * 0.5
    converted = image_to_rgb_array(chw)
    assert converted.shape == (4, 6, 3)
    assert converted.dtype == np.uint8


def test_boxes_are_clamped_padded_and_nonempty():
    boxes = sanitize_crop_boxes(
        [[-20, -10, 2, 2], [8, 8, 8, 8], [9, 9, 3, 3]],
        image_width=10,
        image_height=10,
        padding_ratio=0.1,
        min_area_ratio=0.1,
    )
    assert boxes.shape == (3, 4)
    assert torch.all(boxes[:, 0] >= 0) and torch.all(boxes[:, 1] >= 0)
    assert torch.all(boxes[:, 2] <= 10) and torch.all(boxes[:, 3] <= 10)
    assert torch.all(boxes[:, 2] > boxes[:, 0])
    assert torch.all(boxes[:, 3] > boxes[:, 1])


def test_empty_boxes_and_max_crop_cap():
    assert sanitize_crop_boxes([], 10, 10).shape == (0, 4)
    boxes = sanitize_crop_boxes([[0, 0, 5, 5]] * 5, 10, 10, max_crops=2)
    assert boxes.shape[0] == 2


def test_invalid_box_shape_fails_clearly():
    with pytest.raises(ValueError, match="shape"):
        sanitize_crop_boxes([[0, 1, 2]], 10, 10)


def test_masked_crop_replaces_background_without_loading_clip():
    encoder = ClipCropEncoder.__new__(ClipCropEncoder)
    encoder.crop_padding_ratio = 0.0
    encoder.min_crop_area_ratio = 0.0
    encoder.max_crops_per_image = 4
    encoder.masked_crop = True
    encoder.background = "black"
    image = np.full((4, 4, 3), 255, dtype=np.uint8)
    mask = np.zeros((1, 4, 4), dtype=bool)
    mask[:, 1:3, 1:3] = True
    crops, boxes = encoder._prepare_crops(image, [[0, 0, 4, 4]], masks=mask)
    crop = np.asarray(crops[0])
    assert boxes.shape == (1, 4)
    assert np.all(crop[0, 0] == 0)
    assert np.all(crop[1, 1] == 255)
