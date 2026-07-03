import argparse
import random
import runpy
import sys
import time
from pathlib import Path

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "open_vocabulary_segmentation" / "models"))

from clip_crop_encoder import ClipCropEncoder
from xattn_bridge_clean import load_clean_config


def parse_args():
    parser = argparse.ArgumentParser("Smoke-test frozen CLIP crop encoding")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--clip_model_path", default=None)
    parser.add_argument("--clip_model_name", default=None)
    parser.add_argument("--num_images", type=int, default=2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_image_dir(cfg, args):
    if args.image_dir:
        return Path(args.image_dir)
    dataset_config = Path(args.dataset_config or cfg.evaluate.coco_stuff)
    if not dataset_config.is_absolute():
        dataset_config = REPO_ROOT / dataset_config
    payload = runpy.run_path(str(dataset_config))
    test_cfg = payload["data"]["test"]
    root = Path(test_cfg["data_root"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root / test_cfg["img_dir"]


def build_boxes(width, height, seed):
    rng = random.Random(seed)
    random_width = max(1, int(width * 0.45))
    random_height = max(1, int(height * 0.45))
    random_x = rng.randint(0, max(0, width - random_width))
    random_y = rng.randint(0, max(0, height - random_height))
    return torch.tensor([
        [0, 0, width, height],
        [width * 0.25, height * 0.25, width * 0.75, height * 0.75],
        [
            random_x,
            random_y,
            random_x + random_width,
            random_y + random_height,
        ],
    ], dtype=torch.float32)


def main():
    args = parse_args()
    cfg = load_clean_config(args.config)
    clip_cfg = cfg.clip_image
    model_name = (
        args.clip_model_name
        or clip_cfg.get("model_name", None)
        or cfg.model.clip_model_name
    )
    model_path = (
        args.clip_model_path
        or clip_cfg.get("model_path", None)
        or cfg.model.clip_model_path
    )
    device = args.device or clip_cfg.device
    encoder = ClipCropEncoder(
        model_name=model_name,
        model_path=model_path,
        device=device,
        batch_size=int(clip_cfg.batch_size),
        normalize=bool(clip_cfg.normalize),
        cache_features=bool(clip_cfg.cache_features),
        cache_dir=clip_cfg.cache_dir,
        crop_size=int(clip_cfg.crop_size),
        crop_padding_ratio=float(clip_cfg.crop_padding_ratio),
        masked_crop=bool(clip_cfg.masked_crop),
        background=str(clip_cfg.background),
        min_crop_area_ratio=float(clip_cfg.min_crop_area_ratio),
        max_crops_per_image=int(clip_cfg.max_crops_per_image),
    )

    image_dir = resolve_image_dir(cfg, args)
    image_paths = sorted(
        path
        for suffix in ("*.jpg", "*.jpeg", "*.png")
        for path in image_dir.glob(suffix)
    )[:max(1, args.num_images)]
    if not image_paths:
        raise FileNotFoundError(f"No RGB images found under {image_dir}")

    all_features = []
    total_crops = 0
    encode_seconds = 0.0
    for index, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        boxes = build_boxes(*image.size, seed=index)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        features = encoder.encode(image, boxes)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        encode_seconds += time.perf_counter() - started
        total_crops += features.shape[0]
        all_features.append(features.cpu())
        print(
            f"image={image_path.name} size={image.size} "
            f"boxes={len(boxes)} features={tuple(features.shape)}",
            flush=True,
        )

    features = torch.cat(all_features, dim=0)
    norms = features.norm(dim=-1)
    print(f"model_name={model_name}", flush=True)
    print(f"model_path={model_path}", flush=True)
    print(f"image_dir={image_dir}", flush=True)
    print(f"num_images={len(image_paths)} num_crops={total_crops}", flush=True)
    print(f"feature_shape={tuple(features.shape)}", flush=True)
    print(
        f"feature_norm_mean={float(norms.mean()):.6f} "
        f"feature_norm_std={float(norms.std(unbiased=False)):.6f}",
        flush=True,
    )
    print(
        f"feature_min={float(features.min()):.6f} "
        f"feature_max={float(features.max()):.6f}",
        flush=True,
    )
    print(
        f"encode_seconds={encode_seconds:.4f} "
        f"seconds_per_crop={encode_seconds / max(1, total_crops):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

