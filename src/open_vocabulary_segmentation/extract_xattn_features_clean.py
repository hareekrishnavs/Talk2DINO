import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import clip
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
from xattn_bridge_clean import (
    METHOD_NAME,
    build_coco_stuff_eval_dataset,
    build_eval_loader,
    build_frozen_talk2dino,
    encode_eval_image,
    encode_train_batch,
    frozen_status,
    image_transform_448,
    load_clean_config,
    setup_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(METHOD_NAME + " feature extraction")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opts", nargs="+", default=None)
    return parser.parse_args()


def save_shard(output, shard_id, samples):
    name = f"shard_{shard_id:05d}.pt"
    torch.save({"samples": samples}, Path(output) / name)
    return name


def extract_train(cfg, output, model, device):
    setup_paths()
    raw = torch.load(cfg.data.train_ann_path, map_location="cpu", weights_only=False)
    images = {int(img["id"]): img for img in raw["images"]}
    annotations = list(raw["annotations"])
    max_items = cfg.extract.get("max_items", None)
    if max_items is not None:
        annotations = annotations[: int(max_items)]
    used_image_ids = sorted({int(ann["image_id"]) for ann in annotations})
    transform = image_transform_448()

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, ids):
            self.ids = ids

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            image_id = self.ids[idx]
            image_info = images[image_id]
            image = Image.open(Path(cfg.data.train_image_dir) / image_info["file_name"])
            if image.mode == "L":
                image = image.convert("RGB")
            return {
                "image_id": image_id,
                "image": transform(image),
            }

    image_loader_kwargs = {
        "num_workers": int(cfg.data.num_workers),
        "pin_memory": bool(cfg.data.pin_memory),
        "persistent_workers": int(cfg.data.num_workers) > 0,
    }
    if int(cfg.data.num_workers) > 0:
        image_loader_kwargs["prefetch_factor"] = 2
    image_loader = DataLoader(
        ImageDataset(used_image_ids),
        batch_size=int(cfg.extract.batch_size),
        shuffle=False,
        drop_last=False,
        **image_loader_kwargs,
    )
    manifest = {
        "method_name": METHOD_NAME,
        "split": "train",
        "format": "dedup_train_v2",
        "image_index": [],
        "text_index": [],
    }
    image_shard, image_shard_id, image_count = [], 0, 0
    for batch_idx, batch in enumerate(image_loader):
        with torch.no_grad():
            images_gpu = batch["image"].to(device, non_blocking=True)
            visual_embed, _, patch_tokens = model.encode_image_with_patch_tokens(images_gpu)
            if hasattr(model.proj, "project_visual") and model.proj.__class__.__name__ == "DoubleMLP":
                visual_embed = model.proj.project_visual(visual_embed.float())
        bsz = patch_tokens.shape[0]
        for i in range(bsz):
            image_shard.append({
                "image_id": int(batch["image_id"][i]),
                "visual_embed": visual_embed[i].float().cpu().half(),
                "patch_tokens": patch_tokens[i].float().cpu().half(),
            })
            if len(image_shard) >= int(cfg.extract.shard_size):
                shard_name = save_shard(output, image_shard_id, image_shard)
                for offset, sample in enumerate(image_shard):
                    manifest["image_index"].append({
                        "image_id": int(sample["image_id"]),
                        "shard": shard_name,
                        "offset": offset,
                    })
                image_shard, image_shard_id = [], image_shard_id + 1
        image_count += bsz
        if batch_idx % 10 == 0:
            print(f"Extract train image features: {image_count}/{len(used_image_ids)}", flush=True)
    if image_shard:
        shard_name = save_shard(output, image_shard_id, image_shard)
        for offset, sample in enumerate(image_shard):
            manifest["image_index"].append({
                "image_id": int(sample["image_id"]),
                "shard": shard_name,
                "offset": offset,
            })

    text_shard, text_shard_id, text_count = [], 100000, 0
    for start in range(0, len(annotations), int(cfg.extract.batch_size)):
        batch_anns = annotations[start:start + int(cfg.extract.batch_size)]
        tokens = clip.tokenize([ann["caption"] for ann in batch_anns]).to(device)
        with torch.no_grad():
            text_clip = model.encode_text(tokens).float()
            text_base = model._frozen_base_text_to_dino(text_clip).float()
        for i, ann in enumerate(batch_anns):
            text_shard.append({
                "annotation_id": int(ann.get("id", start + i)),
                "image_id": int(ann["image_id"]),
                "text_clip": text_clip[i].cpu().half(),
                "text_base": text_base[i].cpu().half(),
            })
            if len(text_shard) >= int(cfg.extract.shard_size):
                shard_name = save_shard(output, text_shard_id, text_shard)
                for offset in range(len(text_shard)):
                    manifest["text_index"].append({"shard": shard_name, "offset": offset})
                text_shard, text_shard_id = [], text_shard_id + 1
        text_count += len(batch_anns)
        if text_count % 5000 == 0:
            print(f"Extract train text features: {text_count}/{len(annotations)}", flush=True)
    if text_shard:
        shard_name = save_shard(output, text_shard_id, text_shard)
        for offset in range(len(text_shard)):
            manifest["text_index"].append({"shard": shard_name, "offset": offset})
    manifest["index"] = manifest["text_index"]
    manifest["num_unique_images"] = len(manifest["image_index"])
    manifest["num_captions"] = len(manifest["text_index"])
    with open(Path(output) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def extract_val(cfg, output, model, device):
    dataset = build_coco_stuff_eval_dataset(cfg)
    loader = build_eval_loader(dataset, cfg)
    manifest = {"method_name": METHOD_NAME, "split": "val", "index": []}
    shard, shard_id, global_idx = [], 0, 0
    max_items = cfg.extract.get("max_items", None)
    for batch_indices, data in zip(loader.batch_sampler, loader):
        if max_items is not None and global_idx >= int(max_items):
            break
        img = data["img"].data[0] if hasattr(data["img"], "data") else data["img"]
        if isinstance(img, (list, tuple)):
            img = img[0]
        if img.dim() == 3:
            img = img.unsqueeze(0)
        patch_tokens = encode_eval_image(model, img, device)[0]
        idx = int(batch_indices[0])
        gt = dataset.get_gt_seg_map_by_idx(idx)
        shard.append({
            "patch_tokens": patch_tokens,
            "gt": torch.as_tensor(gt.copy()).short(),
            "ori_shape": tuple(gt.shape),
            "index": idx,
        })
        if len(shard) >= int(cfg.extract.shard_size):
            shard_name = save_shard(output, shard_id, shard)
            for offset in range(len(shard)):
                manifest["index"].append({"shard": shard_name, "offset": offset})
            shard, shard_id = [], shard_id + 1
        global_idx += 1
        if global_idx % 50 == 0:
            print(f"Extract val: {global_idx}/{len(dataset)}", flush=True)
    if shard:
        shard_name = save_shard(output, shard_id, shard)
        for offset in range(len(shard)):
            manifest["index"].append({"shard": shard_name, "offset": offset})
    manifest["classes"] = list(dataset.CLASSES)
    manifest["ignore_index"] = int(dataset.ignore_index)
    with open(Path(output) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    args = parse_args()
    cfg = load_clean_config(args.config, args.opts)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_frozen_talk2dino(cfg, device)
    clip_trainable, dino_trainable = frozen_status(model)
    print(f"CLIP trainable params: {clip_trainable}", flush=True)
    print(f"DINO trainable params: {dino_trainable}", flush=True)
    print(f"data.num_workers: {cfg.data.num_workers}", flush=True)
    with torch.no_grad():
        if args.split == "train":
            extract_train(cfg, output, model, device)
        else:
            extract_val(cfg, output, model, device)
    print(f"Saved {args.split} features to {output}", flush=True)


if __name__ == "__main__":
    main()
