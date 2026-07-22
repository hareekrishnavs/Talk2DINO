import argparse
import json
import math
import os
import requests
import subprocess
import timm
import time
import torch
import torchvision.transforms as T

from io import BytesIO
from src.hooks import get_self_attention, process_self_attention, get_second_last_out, get_vit_out, feats
from src.webdatasets_util import cc2coco_format, create_webdataset_tar, read_coco_format_wds
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AddedToken
from src.local_weights import (
    DEFAULT_WEIGHT_DIR,
    load_local_clip,
    load_local_vision_backbone,
    load_state_dict_from_local_file,
    resolve_weight_path,
    save_torch_artifact,
)
from src.dense_features import DenseFeatureShardWriter

# Initialize global variables
# feats = {}
# num_global_tokens = 1
# num_patch_tokens = 518 // 14 * 518 // 14
# num_tokens = num_global_tokens + num_patch_tokens
# embed_dim = 1024
# num_attn_heads = 16
# scale = 0.125
# batch_size_ = 1
def generate_caption(model, processor, images, prompt="a photography of"):
    image_token = AddedToken("<image>", normalized=False, special=True)
    processor.tokenizer.add_tokens([image_token], special_tokens=True)

    model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64) # pad for efficient computation
    model.config.image_token_index = len(processor.tokenizer) - 1
    inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt").to(next(model.parameters()).device)
    inputs['pixel_values'] = inputs['pixel_values'].float()

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [x.strip() for x in generated_text]    


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _annotations_by_image(data):
    grouped = {}
    for annotation in data["annotations"]:
        missing = {"id", "image_id", "caption", "ann_feats"}.difference(annotation)
        if missing:
            raise ValueError(
                f"annotation is missing required fields: {sorted(missing)}"
            )
        grouped.setdefault(annotation["image_id"], []).append(annotation)
    return grouped


def _open_source_image(image, data_dir):
    if "http" in image["file_name"]:
        return Image.open(BytesIO(requests.get(image["file_name"]).content))
    if "train" in image["file_name"]:
        return Image.open(os.path.join(data_dir, f"train2014/{image['file_name']}"))
    if "val" in image["file_name"]:
        return Image.open(os.path.join(data_dir, f"val2014/{image['file_name']}"))
    if "test" in image["file_name"]:
        return Image.open(os.path.join(data_dir, f"test2014/{image['file_name']}"))
    if "train" in image["coco_url"]:
        return Image.open(os.path.join(data_dir, f"train2017/{image['file_name']}"))
    if "val" in image["coco_url"]:
        return Image.open(os.path.join(data_dir, f"val2017/{image['file_name']}"))
    return None

def run_dinov2_extraction(model_name, data_dir, ann_path, batch_size, resize_dim=518, crop_dim=518, out_path=None, 
                          write_as_wds=False, num_shards=25, n_in_splits=4, in_batch_offset=0, out_offset=0,
                          extract_cls=False, extract_avg_self_attn=False, extract_second_last_out=False,
                          extract_patch_tokens=False, extract_self_attn_maps=False, extract_disentangled_self_attn=False,
                          blip_model_name=None, weight_dir=DEFAULT_WEIGHT_DIR, backbone_weights=None, clip_weights=None,
                          streaming_shards=False, images_per_shard=128, max_images=None, split_name=None,
                          overwrite=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # global num_global_tokens, num_patch_tokens, num_tokens, embed_dim, num_attn_heads, scale, batch_size_
    
    num_global_tokens = 1 if "reg" not in model_name else 5
    num_patch_tokens = crop_dim // 14 * crop_dim // 14
    num_tokens = num_global_tokens + num_patch_tokens
    if 'vitl' in model_name or 'vit_large' in model_name or 'ViT-L' in model_name:
        embed_dim = 1024
    elif 'vitb' in model_name or '_base' in model_name or 'ViT-B' in model_name:
        embed_dim = 768
    elif 'vits' in model_name or 'vit_small' in model_name:
        embed_dim = 384
    else:
        raise Exception("Unknown ViT model")
    
    
    scale = 0.125
    batch_size_ = batch_size
    
    # loading the model
    if 'dinov2' in model_name:
        model = load_local_vision_backbone(
            model_name,
            crop_dim,
            backbone_weights,
            weight_dir,
        )
        image_transforms = T.Compose([
            T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(crop_dim),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        num_attn_heads = model.num_heads
        
    elif 'mae' in model_name or 'sam' in model_name or 'clip' in model_name or 'dino' in model_name or 'beit':
        if backbone_weights is None:
            raise FileNotFoundError(
                f"Missing local backbone_weights for {model_name}. "
                "Auto-download is disabled."
            )
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,  # remove classifier nn.Linear
            img_size=crop_dim
        )
        load_state_dict_from_local_file(
            model,
            resolve_weight_path(backbone_weights, weight_dir),
            f"backbone weights for {model_name}",
            strict=False,
        )
        # the resize dimension will be the one native of the model
        data_config = timm.data.resolve_model_data_config(model)
        image_transforms = timm.data.create_transform(**data_config, is_training=False)
        
        # adjusting the dimensions
        if 'dinov3' in model_name:
            data_config['input_size'] = (3, crop_dim, crop_dim)
            image_transforms = timm.data.create_transform(**data_config, is_training=False)
            num_patch_tokens = crop_dim // 16 * crop_dim // 16
            num_global_tokens = 5
            num_tokens = num_global_tokens + num_patch_tokens
        elif 'mae' in model_name or 'dino' in model_name or 'beit':
            num_patch_tokens = crop_dim // 16 * crop_dim // 16
            num_tokens = 1 + num_patch_tokens
        elif 'sam' in model_name:
            num_patch_tokens = crop_dim // 16 * crop_dim // 16
            num_tokens = num_patch_tokens
            num_global_tokens = 0
            model.blocks[-1].register_forward_hook(get_vit_out)
        elif 'clip' in model_name:
            crop_dim = resize_dim = 224
            num_patch_tokens = crop_dim // 16 * crop_dim // 16 if 'vit_base' in model_name else crop_dim // 14 * crop_dim // 14
            num_tokens = 1 + num_patch_tokens  
        num_attn_heads = model.blocks[-1].attn.num_heads
    elif 'ViT' in model_name:
        # CLIP extraction using clip library
        # use it only for CLS token
        model, image_transforms = load_local_clip(
            model_name,
            device,
            clip_weights,
            weight_dir,
        )
        num_attn_heads = model.num_heads
    else:
        raise Exception("Unknown ViT model")
    
     
    model.eval()
    model.to(device)
    
    
    if blip_model_name is not None:
        blip_processor = Blip2Processor.from_pretrained(
            blip_model_name,
            local_files_only=True,
        )
        blip_model = Blip2ForConditionalGeneration.from_pretrained(
            blip_model_name,
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to(device)
        blip_processor.num_query_tokens = blip_model.config.num_query_tokens
    
    if os.path.isdir(ann_path):
        # if we have a dir as path we assume that the path refere to gcc3m webdataset
        data = cc2coco_format(ann_path, n_in_splits, in_batch_offset)
    elif '.tar' in ann_path:
        # if we have a webdataset template, we read the input dataset as webdatset assuming that it is in COCO format
        data = read_coco_format_wds(ann_path)
    else:
        # otherwise we treat the dataset as a COCO dataset
        if ann_path.endswith('.json'):
            print("Loading the annotations JSON")
            with open(ann_path, 'r') as f:
                data = json.load(f)
        else:
            print("Loading the annotations PTH")
            data = torch.load(ann_path, weights_only=False)
        
    if extract_second_last_out:
        model.blocks[-2].register_forward_hook(get_second_last_out)
    if extract_avg_self_attn or extract_self_attn_maps or extract_disentangled_self_attn:
        model.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)
        if 'beit' in model_name:
            model.blocks[-1].attn.qkv_bias_separate = True

    writer = None
    annotations_by_image = None
    selected_count = min(
        len(data["images"]),
        max_images if max_images is not None else len(data["images"]),
    )
    selected_indices = list(range(selected_count))
    if streaming_shards:
        if out_path is None:
            raise ValueError("--out_path is required with --streaming_shards")
        if write_as_wds:
            raise ValueError("--write_as_wds and --streaming_shards are mutually exclusive")
        if blip_model_name is not None:
            raise ValueError("recaptioning is not supported by the E5 streaming format")
        required_flags = (
            extract_patch_tokens,
            extract_self_attn_maps,
            extract_disentangled_self_attn,
        )
        if not all(required_flags):
            raise ValueError(
                "E5 streaming extraction requires --extract_patch_tokens, "
                "--extract_self_attn_maps, and --extract_disentangled_self_attn"
            )
        if (num_patch_tokens, embed_dim, num_attn_heads) != (1024, 768, 12):
            raise ValueError(
                "E5 Phase 1 requires ViT-B at 32x32 resolution: "
                f"got P={num_patch_tokens}, D={embed_dim}, H={num_attn_heads}"
            )
        if max_images is not None and max_images <= 0:
            raise ValueError("--max_images must be greater than zero")
        if split_name is None:
            raise ValueError("--split_name is required with --streaming_shards")
        annotations_by_image = _annotations_by_image(data)
        selected_image_ids = [data["images"][index]["id"] for index in selected_indices]
        if len(set(selected_image_ids)) != len(selected_image_ids):
            raise ValueError("source contains duplicate image IDs in the selected range")
        selected_annotation_ids = [
            annotation["id"]
            for image_id in selected_image_ids
            for annotation in annotations_by_image.get(image_id, [])
        ]
        if len(set(selected_annotation_ids)) != len(selected_annotation_ids):
            raise ValueError("source contains duplicate selected annotation IDs")
        extraction_config = {
            "annotation_path": os.path.abspath(ann_path),
            "data_dir": os.path.abspath(data_dir),
            "model": model_name,
            "resize_dim": resize_dim,
            "crop_dim": crop_dim,
            "patch_count": num_patch_tokens,
            "embedding_dim": embed_dim,
            "attention_heads": num_attn_heads,
            "attention_map_format": "probabilities",
            "patch_tokens_dtype": "float16",
            "self_attn_maps_dtype": "float16",
            "disentangled_self_attn_dtype": "float32",
        }
        writer = DenseFeatureShardWriter(
            out_path,
            split=split_name,
            extraction_config=extraction_config,
            source_commit=_git_commit(),
            source_images=len(data["images"]),
            source_annotations=len(data["annotations"]),
            selected_image_ids=selected_image_ids,
            selected_annotation_ids=selected_annotation_ids,
            max_images=max_images,
            images_per_shard=images_per_shard,
            overwrite=overwrite,
        )

    print("Starting the features extraction...")
    if writer is not None:
        selected_indices = [
            index
            for index in selected_indices
            if data["images"][index]["id"] not in writer.completed_image_ids
        ]
    n_imgs = len(selected_indices)
    n_batch = math.ceil(n_imgs / batch_size)
    n_errors = 0
    extraction_started = time.monotonic()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in tqdm(range(n_batch)):
        start = i * batch_size
        end = start + batch_size if i < n_batch - 1 else n_imgs
        batch_indices = selected_indices[start:end]
        batch_size_ = len(batch_indices)
        raw_imgs = []
        failed_ids = []
        for j in batch_indices:
            if 'jpg' in data['images'][j]:
                # CC3M case
                pil_img = data['images'][j]['jpg']
                # saving space by eliminating the jpg
                del data['images'][j]['jpg']
            else:
                try:
                    pil_img = _open_source_image(data["images"][j], data_dir)
                except Exception:
                    if writer is None and "http" not in data["images"][j]["file_name"]:
                        raise
                    pil_img = None
                if pil_img is None:
                    pil_img = Image.new("RGB", (224, 224))
                    failed_ids.append(j)
                    n_errors += 1
                    if writer is not None:
                        writer.record_failure(data["images"][j]["id"])

            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            raw_imgs.append(pil_img)

        batch_imgs = torch.stack([image_transforms(img) for img in raw_imgs]).to(device)
                
        with torch.no_grad():
            if 'dinov2' in model_name:
                outs = model(batch_imgs, is_training=True)
            elif 'dinov3' in model_name:
                output = model.forward_features(batch_imgs)
                # reporting output in DINOv2 format
                outs = {
                    'x_norm_clstoken': output[:, 0, :],
                    'x_norm_patchtokens': output[:, 5:, :],
                }
            elif 'mae' in model_name or 'clip' in model_name or 'dino' in model_name or 'beit' in model_name:
                output = model.forward_features(batch_imgs)
                # reporting output in DINOv2 format
                outs = {
                    'x_norm_clstoken': output[:, 0, :],
                    'x_norm_patchtokens': output[:, 1:, :],
                }
            elif 'sam' in model_name:
                sam_output = model.forward_features(batch_imgs)
                if extract_cls:
                    cls = model.forward_head(sam_output, pre_logits=True)
                else:
                    cls = None
                outs = {
                    'x_norm_clstoken': cls,
                    'x_norm_patchtokens': feats['vit_out'].reshape(batch_size_, num_patch_tokens, embed_dim)
                }
            elif 'ViT' in model_name:
                outs = {
                    'x_norm_clstoken': model.encode_image(batch_imgs)
                }
            cls_token = outs['x_norm_clstoken']
            if extract_avg_self_attn or extract_self_attn_maps or extract_disentangled_self_attn:
                self_attn, self_attn_maps = process_self_attention(feats['self_attn'], batch_size_, num_tokens, num_attn_heads, embed_dim, scale, num_global_tokens, ret_self_attn_maps=True)
            if extract_avg_self_attn:
                avg_self_attn_token = (self_attn.unsqueeze(-1) * outs['x_norm_patchtokens']).mean(dim=1)
            if extract_disentangled_self_attn:
                self_attn_maps = self_attn_maps.softmax(dim=-1)
                disentangled_self_attn = (outs['x_norm_patchtokens'].unsqueeze(1) * self_attn_maps.unsqueeze(-1)).mean(dim=2)
            if extract_second_last_out:
                second_last_cls = feats['second_last_out'][:, 0, :] # keeping only the CLS token
            if blip_model_name is not None:
                new_capts = generate_caption(blip_model, blip_processor, raw_imgs)
        
        # writing the outputs in the original data
        for local_index, j in enumerate(batch_indices):
            if j in failed_ids:
                continue

            if writer is not None:
                image = data["images"][j]
                image_annotations = annotations_by_image.get(image["id"], [])
                record = {
                    "image_id": image["id"],
                    "file_name": image["file_name"],
                    "disentangled_self_attn": disentangled_self_attn[local_index]
                    .detach()
                    .cpu()
                    .to(torch.float32),
                    "patch_tokens": outs["x_norm_patchtokens"][local_index]
                    .detach()
                    .cpu()
                    .to(torch.float16),
                    "self_attn_maps": self_attn_maps[local_index]
                    .detach()
                    .cpu()
                    .to(torch.float16),
                    "captions": [annotation["caption"] for annotation in image_annotations],
                    "ann_feats": [annotation["ann_feats"] for annotation in image_annotations],
                    "annotation_ids": [annotation["id"] for annotation in image_annotations],
                }
                writer.add(record)
            else:
                if extract_cls or (not extract_avg_self_attn and not extract_second_last_out):
                    data['images'][j]['dino_features'] = cls_token[local_index].to('cpu')
                if extract_avg_self_attn:
                    data['images'][j]['avg_self_attn_out'] = avg_self_attn_token[local_index].to('cpu')
                if extract_second_last_out:
                    data['images'][j]['second_last_out'] = second_last_cls[local_index].to('cpu')
                if extract_patch_tokens:
                    data['images'][j]['patch_tokens'] = outs['x_norm_patchtokens'][local_index].to('cpu')
                if extract_self_attn_maps:
                    data['images'][j]['self_attn_maps'] = self_attn_maps[local_index].to('cpu')
                if extract_disentangled_self_attn:
                    data['images'][j]['disentangled_self_attn'] = disentangled_self_attn[local_index].to('cpu')
                if blip_model_name is not None:
                    data['annotations'][j]['caption'] = new_capts[local_index]
                
    print("Feature extraction done!")
    print(f"Failed to extract {n_errors} of {n_imgs}")

    if writer is not None:
        writer.finish()
        elapsed = time.monotonic() - extraction_started
        processed = n_imgs - n_errors
        metrics = {
            "elapsed_seconds": elapsed,
            "extracted_images": processed,
            "images_per_second": processed / elapsed if elapsed else None,
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated() if device == "cuda" else None
            ),
        }
        print("E5 extraction metrics: " + json.dumps(metrics, sort_keys=True))
        print(f"Streaming features saved at {out_path}")
        return
    
    
    if write_as_wds:
        os.makedirs(out_path, exist_ok=True)
        create_webdataset_tar(data, out_path, num_shards, out_offset)
    else:
        if out_path is None:
            # we use as output path the ann_path but with the extension pth
            out_path = os.path.splitext(ann_path)[0] + '.pth' 
        save_torch_artifact(data, out_path)
        print(f"Features saved at {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', type=str, default="coco/test1k.json", help="Directory of the annotation file") 
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--data_dir', type=str, default="../coco/", help="Directory of the images") 
    parser.add_argument('--blip_model', type=str, default=None, help="BLIP model to recaption with. If None we will use standard captions. For CC3M we use Salesforce/blip2-opt-6.7b-coco") 
    parser.add_argument('--model', type=str, default="dinov2_vitl14_reg", help="Model configuration to extract features from")
    parser.add_argument('--weight_dir', type=str, default=DEFAULT_WEIGHT_DIR, help="Directory containing local model weights")
    parser.add_argument('--backbone_weights', type=str, default=None, help="Local DINO/timm checkpoint path or filename")
    parser.add_argument('--clip_weights', type=str, default=None, help="Local CLIP checkpoint path or filename")
    parser.add_argument('--resize_dim', type=int, default=518, help="Resize dimension")
    parser.add_argument('--crop_dim', type=int, default=518, help="Crop dimension")
    parser.add_argument('--extract_cls', default=False, action="store_true", help="If setted, adds the CLS token to the output")
    parser.add_argument('--extract_avg_self_attn', default=False, action="store_true", help="If setted, adds the token obtained by weighting using the self attention to the output")
    parser.add_argument('--extract_second_last_out', default=False, action="store_true", help="If setted, adds the second last CLS to the output") 
    parser.add_argument('--extract_patch_tokens', default=False, action="store_true", help="If setted, we extract all the patch tokens") 
    parser.add_argument('--extract_self_attn_maps', default=False, action="store_true", help="If setted, we extract all the self-attention maps") 
    parser.add_argument('--extract_disentangled_self_attn', default=False, action="store_true", help="If setted, adds the token obtained by weighting using the self attention to the output, without averaging the attention heads") 
    parser.add_argument('--out_path', type=str, default=None, help="Pth of the output file, if setted to None. out_pat = ann_path") 
    parser.add_argument('--write_as_wds', action="store_true", default=False, help="If setted, the output will be written as a webdataset") 
    parser.add_argument('--n_shards', type=int, default=25, help="Number of shards in which the webdataset is splitted. Only relevant if --write_as_wds is setted.")
    parser.add_argument('--n_in_split', type=int, default=1, help="Number of splits in which we want to divide the tar files. For example, with 4 n_split we elaborate 332 // 4 = 83 tar files.")
    parser.add_argument('--in_batch_offset', type=int, default=0, help="Of the n_splits in which we have divided tars, we decide which of them elaborate")
    parser.add_argument('--out_offset', type=int, default=0, help="Index of the first shard to save")
    parser.add_argument('--streaming_shards', action="store_true", default=False, help="Write E5 image-level dense-feature shards continuously")
    parser.add_argument('--images_per_shard', type=int, default=128, help="Images per finalized E5 shard")
    parser.add_argument('--max_images', type=int, default=None, help="Limit extraction to the first N unique images")
    parser.add_argument('--split_name', type=str, default=None, help="Shard filename prefix, for example train or val")
    parser.add_argument('--overwrite', action="store_true", default=False, help="Explicitly replace an existing E5 extraction")
    args = parser.parse_args()
    
    run_dinov2_extraction(args.model, args.data_dir, args.ann_path, args.batch_size, args.resize_dim, args.crop_dim, args.out_path,
                          args.write_as_wds, args.n_shards, args.n_in_split, args.in_batch_offset, args.out_offset,
                          args.extract_cls, args.extract_avg_self_attn, args.extract_second_last_out, args.extract_patch_tokens, args.extract_self_attn_maps,
                          args.extract_disentangled_self_attn, args.blip_model, args.weight_dir, args.backbone_weights, args.clip_weights,
                          args.streaming_shards, args.images_per_shard, args.max_images, args.split_name, args.overwrite)
if __name__ == '__main__':
    main()
