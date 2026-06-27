import argparse
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
from xattn_bridge_clean import build_frozen_talk2dino, format_seconds, load_clean_config


def parse_args():
    parser = argparse.ArgumentParser("Add raw DINO patch tokens to baseline-style .pth features")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--opts", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def file_size_gb(path):
    path = Path(path)
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 ** 3)


def patch_dtype_from_cfg(cfg):
    dtype_name = str(cfg.get("patch_tokens", {}).get("dtype", "fp16")).lower()
    if dtype_name in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype_name in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported patch_tokens.dtype={dtype_name}")


def patch_key_from_cfg(cfg):
    return str(cfg.get("patch_tokens", {}).get("key", "patch_tokens"))


def split_image_dir_candidates(cfg, split):
    candidates = []
    split_key = f"{split}_image_dir"
    if cfg.get("data", {}).get(split_key, None):
        candidates.append(Path(str(cfg.data.get(split_key))))
    if split == "train" and cfg.get("data", {}).get("train_image_dir", None):
        candidates.append(Path(str(cfg.data.train_image_dir)))
    if split == "val":
        if cfg.get("data", {}).get("val_image_dir", None):
            candidates.append(Path(str(cfg.data.val_image_dir)))
        train_dir = cfg.get("data", {}).get("train_image_dir", None)
        if train_dir:
            train_dir = str(train_dir)
            candidates.append(Path(train_dir.replace("train2017", "val2017")))
    candidates.extend([
        Path(f"/scratch/haree/coco_stuff164k/images/{split}2017"),
        Path(f"data/coco_stuff164k/images/{split}2017"),
    ])
    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def resolve_image_path(image_info, image_dirs):
    for key in ("path", "image_path", "filename", "file_name"):
        value = image_info.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.is_absolute() and path.exists():
            return path
        for image_dir in image_dirs:
            candidate = image_dir / path
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"Could not resolve image path for image id={image_info.get('id')} "
        f"file_name={image_info.get('file_name')}. Tried dirs: "
        f"{[str(path) for path in image_dirs]}"
    )


class ImageRecordDataset(Dataset):
    def __init__(self, image_records, image_dirs, transform):
        self.image_records = image_records
        self.image_dirs = image_dirs
        self.transform = transform

    def __len__(self):
        return len(self.image_records)

    def __getitem__(self, idx):
        image_info = self.image_records[idx]
        path = resolve_image_path(image_info, self.image_dirs)
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return {
            "idx": idx,
            "image_id": int(image_info["id"]),
            "path": str(path),
            "image": tensor,
        }


def collate_image_records(batch):
    return {
        "idx": [item["idx"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "path": [item["path"] for item in batch],
        "image": torch.stack([item["image"] for item in batch]),
    }


@torch.no_grad()
def extract_patch_tokens(frozen, images):
    if "dinov2" in frozen.model_name:
        return frozen.model.forward_features(images)["x_norm_patchtokens"]
    if "dinov3" in frozen.model_name:
        return frozen.model.forward_features(images)[:, 5:, :]
    return frozen.model.forward_features(images)[:, 1:, :]


def validate_output(input_data, output_path, patch_key, expected_dtype):
    print(f"[{timestamp()}] Validating output: {output_path}", flush=True)
    output_data = torch.load(output_path, map_location="cpu", weights_only=False)
    if len(output_data.get("images", [])) != len(input_data.get("images", [])):
        raise AssertionError("Output image count differs from input image count")
    if len(output_data.get("annotations", [])) != len(input_data.get("annotations", [])):
        raise AssertionError("Output annotation count differs from input annotation count")
    for before, after in zip(input_data["images"], output_data["images"]):
        missing = set(before.keys()) - set(after.keys())
        if missing:
            raise AssertionError(f"Output image id={before.get('id')} lost keys: {sorted(missing)}")
        if patch_key not in after:
            raise AssertionError(f"Output image id={before.get('id')} missing `{patch_key}`")
    sample = output_data["images"][0][patch_key]
    if sample.dtype != expected_dtype:
        raise AssertionError(f"Expected {patch_key} dtype {expected_dtype}, got {sample.dtype}")
    if sample.shape[-1] != 768:
        raise AssertionError(f"Expected {patch_key} last dim 768, got shape {tuple(sample.shape)}")
    if torch.isnan(sample.float()).any():
        raise AssertionError(f"Sample {patch_key} contains NaN")
    print(
        f"[{timestamp()}] Validation passed: sample {patch_key} "
        f"shape={tuple(sample.shape)} dtype={sample.dtype}",
        flush=True,
    )


def main():
    args = parse_args()
    cfg = load_clean_config(args.config, args.opts)
    input_path = Path(args.input)
    output_path = Path(args.output)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    patch_key = patch_key_from_cfg(cfg)
    patch_dtype = patch_dtype_from_cfg(cfg)

    if output_path.exists() and not args.overwrite:
        print(
            f"[{timestamp()}] Output already exists; validating and exiting: {output_path}",
            flush=True,
        )
        input_data = torch.load(input_path, map_location="cpu", weights_only=False)
        validate_output(input_data, output_path, patch_key, patch_dtype)
        return

    print(f"[{timestamp()}] Input file : {input_path}", flush=True)
    print(f"[{timestamp()}] Output file: {output_path}", flush=True)
    print(f"[{timestamp()}] Split      : {args.split}", flush=True)
    print(f"[{timestamp()}] Patch key  : {patch_key}", flush=True)
    print(f"[{timestamp()}] Patch dtype: {patch_dtype}", flush=True)
    load_start = time.time()
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    print(
        f"[{timestamp()}] Loaded input in {format_seconds(time.time() - load_start)} "
        f"({file_size_gb(input_path):.2f} GB)",
        flush=True,
    )
    if "images" not in data:
        raise KeyError(f"{input_path} does not contain `images`")
    image_records = data["images"]
    print(f"[{timestamp()}] Number of image entries: {len(image_records)}", flush=True)
    if image_records:
        print(
            f"[{timestamp()}] Available image keys: {sorted(image_records[0].keys())}",
            flush=True,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and os.environ.get("LOCAL_RANK") is not None:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    frozen = build_frozen_talk2dino(cfg, device)
    frozen.eval()
    frozen.requires_grad_(False)
    if hasattr(frozen, "model"):
        frozen.model.eval()
        frozen.model.requires_grad_(False)

    image_dirs = split_image_dir_candidates(cfg, args.split)
    print(f"[{timestamp()}] Image dir candidates: {[str(path) for path in image_dirs]}", flush=True)
    dataset = ImageRecordDataset(image_records, image_dirs, frozen.image_transforms)
    loader_kwargs = {
        "num_workers": int(cfg.data.get("num_workers", 0)),
        "pin_memory": bool(cfg.data.get("pin_memory", True)),
        "collate_fn": collate_image_records,
    }
    if int(cfg.data.get("num_workers", 0)) > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.data.get("prefetch_factor", 2))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("patch_tokens", {}).get("batch_size", cfg.extract.get("batch_size", 64))),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    updated_images = [dict(item) for item in image_records]
    start = time.time()
    processed = 0
    first_shape = None
    first_dtype = None
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        tokens = extract_patch_tokens(frozen, images).detach().cpu().to(dtype=patch_dtype)
        for row, image_idx in enumerate(batch["idx"]):
            updated_images[int(image_idx)][patch_key] = tokens[row].contiguous()
        processed += len(batch["idx"])
        if first_shape is None:
            first_shape = tuple(tokens[0].shape)
            first_dtype = tokens.dtype
        elapsed = time.time() - start
        speed = processed / max(elapsed, 1e-6)
        eta = (len(dataset) - processed) / max(speed, 1e-6)
        if processed == len(batch["idx"]) or processed % 100 == 0 or processed == len(dataset):
            print(
                f"[{timestamp()}] {processed}/{len(dataset)} images "
                f"last_id={batch['image_id'][-1]} last_path={batch['path'][-1]} "
                f"patch_shape={tuple(tokens[-1].shape)} dtype={tokens.dtype} "
                f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
                flush=True,
            )

    output_data = dict(data)
    output_data["images"] = updated_images
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[{timestamp()}] Writing temporary output: {tmp_path} "
        f"(first patch shape={first_shape}, dtype={first_dtype})",
        flush=True,
    )
    torch.save(output_data, tmp_path)
    os.replace(tmp_path, output_path)
    print(
        f"[{timestamp()}] Wrote output: {output_path} ({file_size_gb(output_path):.2f} GB)",
        flush=True,
    )
    validate_output(data, output_path, patch_key, patch_dtype)


if __name__ == "__main__":
    main()
