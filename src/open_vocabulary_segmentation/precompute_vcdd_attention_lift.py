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
    parser = argparse.ArgumentParser("Precompute compact VCDD attention lift")
    parser.add_argument("--config", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["fp16"], default="fp16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def candidate_roots(cfg, split):
    roots = []
    for key in (f"{split}_image_dir", "image_dir", "train_image_dir"):
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
    out = []
    seen = set()
    for root in roots:
        if root not in seen:
            out.append(root)
            seen.add(root)
    return out


def resolve_image_path(image_info, roots):
    file_name = image_info.get("file_name", None)
    if file_name is None:
        raise KeyError(f"Image entry {image_info.get('id', '<unknown>')} has no file_name")
    rel = Path(str(file_name))
    if rel.is_absolute() and rel.exists():
        return rel
    for root in roots:
        for candidate in (root / rel, root / rel.name):
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"Could not resolve image_id={image_info.get('id')} file_name={file_name}; "
        f"searched roots={', '.join(str(root) for root in roots)}"
    )


class ImageDataset(Dataset):
    def __init__(self, images, roots):
        self.transform = image_transform_448()
        self.items = [
            {
                "image_id": int(image["id"]),
                "path": resolve_image_path(image, roots),
            }
            for image in images
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["path"]).convert("RGB")
        return {
            "image_id": item["image_id"],
            "image": self.transform(image),
        }


def normalize_attention(self_attn_maps):
    if self_attn_maps is None:
        raise RuntimeError("encode_image_with_patch_tokens did not return self_attn_maps")
    attention = self_attn_maps.float()
    if attention.ndim != 3:
        raise ValueError(f"Expected attention [B,12,1024], got {tuple(attention.shape)}")
    attention = attention.clamp_min(0)
    attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return attention


def validate_attention(attention):
    if attention.ndim != 3 or attention.shape[1:] != (12, 1024):
        raise ValueError(f"Expected [N,12,1024] attention, got {tuple(attention.shape)}")
    if not torch.isfinite(attention.float()).all():
        raise ValueError("attention contains NaN/Inf")
    sums = attention.float().sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=2e-3, rtol=2e-3):
        raise ValueError(
            f"attention region sums not close to 1: mean={float(sums.mean()):.6f} "
            f"min={float(sums.min()):.6f} max={float(sums.max()):.6f}"
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
    raw = torch.load(args.features, map_location="cpu", weights_only=False)
    images = list(raw["images"])
    roots = candidate_roots(cfg, args.split)
    dataset = ImageDataset(images, roots)
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
    image_ids = torch.empty(len(dataset), dtype=torch.long)
    attention = torch.empty((len(dataset), 12, 1024), dtype=torch.float16)
    print(f"split                    : {args.split}", flush=True)
    print(f"number of images         : {len(dataset)}", flush=True)
    print(f"output                   : {output}", flush=True)
    print(f"image roots              : {', '.join(str(root) for root in roots)}", flush=True)
    start = time.time()
    offset = 0
    for batch_idx, batch in enumerate(loader):
        images_gpu = batch["image"].to(device, non_blocking=True)
        _, self_attn_maps, _ = model.encode_image_with_patch_tokens(images_gpu)
        batch_attention = normalize_attention(self_attn_maps).cpu()
        bsz = batch_attention.shape[0]
        if batch_attention.shape[1:] != (12, 1024):
            raise ValueError(f"Expected [B,12,1024], got {tuple(batch_attention.shape)}")
        image_ids[offset:offset + bsz] = batch["image_id"].long()
        attention[offset:offset + bsz] = batch_attention.half()
        offset += bsz
        if batch_idx == 0 or offset % max(1, batch_size * 10) == 0 or offset == len(dataset):
            elapsed = time.time() - start
            rate = offset / max(elapsed, 1e-6)
            eta = (len(dataset) - offset) / max(rate, 1e-6)
            print(f"processed {offset}/{len(dataset)} elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}", flush=True)
    sums = validate_attention(attention)
    payload = {
        "image_ids": image_ids,
        "attention": attention,
        "meta": {
            "format": "vcdd_attention_lift_fp16_v1",
            "split": args.split,
            "features": str(args.features),
            "config": str(args.config),
            "shape": list(attention.shape),
            "normalization": "clamp_min_0_then_sum_over_regions",
        },
    }
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, output)
    size_mb = output.stat().st_size / (1024 ** 2)
    print(f"attention tensor shape   : {tuple(attention.shape)}", flush=True)
    print(f"dtype                    : {attention.dtype}", flush=True)
    print(f"file size                : {size_mb:.2f} MB", flush=True)
    print(f"attention min/max        : {float(attention.float().min()):.8f} / {float(attention.float().max()):.8f}", flush=True)
    print(f"sum over regions         : mean={float(sums.mean()):.6f} min={float(sums.min()):.6f} max={float(sums.max()):.6f}", flush=True)
    print(f"first image_ids          : {image_ids[:8].tolist()}", flush=True)


if __name__ == "__main__":
    main()
