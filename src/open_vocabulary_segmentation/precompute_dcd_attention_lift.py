import argparse
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
from xattn_bridge_clean import (  # noqa: E402
    build_frozen_talk2dino,
    format_seconds,
    image_transform_448,
    load_clean_config,
    setup_paths,
)


def parse_args():
    parser = argparse.ArgumentParser("Precompute compact DCD attention lift")
    parser.add_argument("--config", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["fp16"], default="fp16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def file_size_mb(path):
    return Path(path).stat().st_size / (1024 ** 2)


def load_feature_images(features_path):
    features_path = Path(features_path)
    if not features_path.is_file():
        raise FileNotFoundError(f"--features must be a baseline-style .pth file: {features_path}")
    data = torch.load(features_path, map_location="cpu", weights_only=False)
    if "images" not in data:
        raise KeyError(f"{features_path} does not contain an `images` list")
    images = list(data["images"])
    if not images:
        raise RuntimeError(f"{features_path} contains no images")
    return images


def candidate_roots(cfg, split):
    roots = []
    split_key = f"{split}_image_dir"
    for key in (split_key, "image_dir", "train_image_dir"):
        value = cfg.data.get(key, None)
        if value not in {None, "", "null", "None"}:
            roots.append(Path(str(value)))
    train_dir = cfg.data.get("train_image_dir", None)
    if split == "val" and train_dir not in {None, "", "null", "None"}:
        train_dir = Path(str(train_dir))
        roots.append(Path(str(train_dir).replace("train2017", "val2017")))
        if train_dir.name == "train2017":
            roots.append(train_dir.parent / "val2017")
    roots.extend([
        Path(f"/scratch/haree/coco_stuff164k/images/{split}2017"),
        Path(f"/scratch/haree/coco/images/{split}2017"),
    ])
    unique = []
    seen = set()
    for root in roots:
        if root not in seen:
            unique.append(root)
            seen.add(root)
    return unique


def resolve_image_path(image_info, roots):
    file_name = image_info.get("file_name", None)
    if file_name is None:
        raise KeyError(f"Image entry {image_info.get('id', '<unknown>')} has no file_name")
    path = Path(str(file_name))
    if path.is_absolute() and path.exists():
        return path
    for root in roots:
        candidate = root / path
        if candidate.exists():
            return candidate
        candidate = root / path.name
        if candidate.exists():
            return candidate
    searched = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(
        f"Could not resolve image_id={image_info.get('id')} file_name={file_name}. "
        f"Searched roots: {searched}"
    )


class ImageIdDataset(Dataset):
    def __init__(self, images, roots):
        self.items = []
        for image_info in images:
            if "id" not in image_info:
                raise KeyError(f"Image entry missing id: {image_info}")
            self.items.append({
                "image_id": int(image_info["id"]),
                "path": resolve_image_path(image_info, roots),
            })
        self.transform = image_transform_448()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["path"])
        if image.mode == "L":
            image = image.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        return {
            "image_id": item["image_id"],
            "image": self.transform(image),
        }


def normalize_attention(self_attn_maps):
    if self_attn_maps is None:
        raise RuntimeError(
            "model.encode_image_with_patch_tokens did not return self_attn_maps. "
            "DCD attention lift requires disentangled self-attention maps."
        )
    attention = self_attn_maps.float()
    if attention.ndim != 3:
        raise ValueError(f"Expected attention [B,12,1024], got {tuple(attention.shape)}")
    attention = attention.clamp_min(0)
    attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return attention


def validate_attention(attention):
    if attention.ndim != 3:
        raise ValueError(f"attention.ndim must be 3, got {attention.ndim}")
    if attention.shape[1] != 12:
        raise ValueError(f"attention.shape[1] must be 12, got {attention.shape[1]}")
    if attention.shape[2] != 1024:
        raise ValueError(f"attention.shape[2] must be 1024, got {attention.shape[2]}")
    if not torch.isfinite(attention).all():
        raise ValueError("attention contains NaN or Inf")
    sums = attention.float().sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=2e-3, rtol=2e-3):
        raise ValueError(
            "attention.sum(dim=1) is not close to 1: "
            f"mean={float(sums.mean()):.6f} min={float(sums.min()):.6f} "
            f"max={float(sums.max()):.6f}"
        )
    return sums


@torch.no_grad()
def main():
    args = parse_args()
    setup_paths()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_clean_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    roots = candidate_roots(cfg, args.split)
    images = load_feature_images(args.features)
    dataset = ImageIdDataset(images, roots)
    batch_size = args.batch_size or int(cfg.extract.get("batch_size", 64))
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": int(cfg.data.get("num_workers", 0)),
        "pin_memory": bool(cfg.data.get("pin_memory", True)),
    }
    if int(loader_kwargs["num_workers"]) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(cfg.data.get("prefetch_factor", 2))
    loader = DataLoader(dataset, **loader_kwargs)

    model = build_frozen_talk2dino(cfg, device)
    count = len(dataset)
    image_ids = torch.empty(count, dtype=torch.long)
    attention = torch.empty((count, 12, 1024), dtype=torch.float16)

    print(f"split                    : {args.split}", flush=True)
    print(f"features                 : {args.features}", flush=True)
    print(f"number of images         : {count}", flush=True)
    print(f"device                   : {device}", flush=True)
    print(f"batch_size               : {batch_size}", flush=True)
    print(f"output                   : {output}", flush=True)
    print(f"image roots              : {', '.join(str(root) for root in roots)}", flush=True)

    offset = 0
    started = time.time()
    for batch_idx, batch in enumerate(loader):
        images_gpu = batch["image"].to(device, non_blocking=True)
        _, self_attn_maps, _ = model.encode_image_with_patch_tokens(images_gpu)
        batch_attention = normalize_attention(self_attn_maps).cpu()
        bsz = batch_attention.shape[0]
        if batch_attention.shape[1:] != (12, 1024):
            raise ValueError(
                "Expected batch attention shape [B,12,1024], got "
                f"{tuple(batch_attention.shape)}"
            )
        image_ids[offset:offset + bsz] = batch["image_id"].long()
        attention[offset:offset + bsz] = batch_attention.half()
        offset += bsz
        if batch_idx == 0 or offset % max(batch_size * 10, 1) == 0 or offset == count:
            elapsed = time.time() - started
            rate = offset / max(elapsed, 1e-6)
            eta = (count - offset) / max(rate, 1e-6)
            print(
                f"processed {offset}/{count} elapsed={format_seconds(elapsed)} "
                f"eta={format_seconds(eta)}",
                flush=True,
            )

    if offset != count:
        raise RuntimeError(f"Internal error: wrote {offset} rows for {count} images")
    sums = validate_attention(attention)
    payload = {
        "image_ids": image_ids,
        "attention": attention,
        "meta": {
            "format": "dcd_attention_lift_fp16_v1",
            "split": args.split,
            "features": str(args.features),
            "config": str(args.config),
            "dtype": args.dtype,
            "shape": list(attention.shape),
            "normalization": "clamp_min_0_then_sum_over_regions",
            "attention_source": "model.encode_image_with_patch_tokens self_attn_maps",
        },
    }
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, output)

    print(f"attention tensor shape   : {tuple(attention.shape)}", flush=True)
    print(f"dtype                    : {attention.dtype}", flush=True)
    print(f"file size                : {file_size_mb(output):.2f} MB", flush=True)
    print(
        f"attention min/max        : {float(attention.float().min()):.8f} / "
        f"{float(attention.float().max()):.8f}",
        flush=True,
    )
    print(
        f"sum over regions         : mean={float(sums.mean()):.6f} "
        f"min={float(sums.min()):.6f} max={float(sums.max()):.6f}",
        flush=True,
    )
    print(f"first image_ids          : {image_ids[:8].tolist()}", flush=True)
    print(f"saved                    : {output}", flush=True)


if __name__ == "__main__":
    main()
