import webdataset as wds
import os
import torch

from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset

from src.dense_features import (
    DenseFeatureStreamingDataset,
    iter_dense_shard_records,
)

class DinoClipDataset(Dataset):
    def __init__(self, features_file, features_name='dino_features', text_features='ann_feats', load_attn_maps=False, is_wds=False):
        if is_wds:
            self.__load_wds_dataset(features_file, features_name, text_features, load_attn_maps)
        else:
            self.__load_pth_dataset(features_file, features_name, text_features, load_attn_maps)
            
            
    def __getitem__(self, idx):
        annotation = self.data[idx]['annotation']
        image = self.data[idx]['image']
        metadata = {
            'annotation_id': self.data[idx]['annotation_id'],
            'image_id': self.data[idx]['image_id']
        }
        
        to_ret = {
            'annotation': annotation,
            'image': image,
            'metadata': metadata,
        }
        
        if 'self_attn_maps' in self.data[idx]:
            to_ret['self_attn_maps'] = self.data[idx]['self_attn_maps']
            to_ret['dino_features'] = self.data[idx]['dino_features']
            
        if 'text_input_mask' in self.data[idx]:
            to_ret['text_input_mask'] = self.data[idx]['text_input_mask']
            
        if 'text_argmax' in self.data[idx]:
            to_ret['text_argmax'] = self.data[idx]['text_argmax']
        
        return to_ret
    
    def __len__(self):
        return len(self.data)
    
    def __load_pth_dataset(self, features_file, features_name='dino_features', text_features='ann_feats', load_attn_maps=False):
        print("Loading dataset...")
        data = torch.load(features_file, map_location='cpu', weights_only=False)
        print("Dataset loaded!")
        images = {imm['id']: imm for imm in data['images']}
        del data['images']
        self.data = {}
        
        for idx, ann in enumerate(data['annotations']):
            ann_id = ann['id']
            imm_id = ann['image_id']
            self.data[idx] = {}
            if text_features != 'clip_txt_out_tokens_avg':
                self.data[idx]['annotation'] = ann[text_features] 
            else:
                mask = ann['text_input_mask']
                mask[mask.sum() - 1] = False # excluding end of sequence
                mask[0] = False # excluding CLS token
                self.data[idx]['annotation'] = ann['clip_txt_out_tokens'][mask].mean(dim=0)
            if text_features == 'clip_second_last_out':
                self.data[idx]['text_argmax'] = ann['text_argmax']
            self.data[idx]['image'] = images[imm_id][features_name]
            if load_attn_maps:
                self.data[idx]['self_attn_maps'] = images[imm_id]['self_attn_maps']
                self.data[idx]['dino_features'] = images[imm_id]['dino_features']
            if text_features == 'clip_txt_out_tokens':
                self.data[idx]['text_input_mask'] = ann['text_input_mask']
            self.data[idx]['image_id'] = imm_id
            self.data[idx]['annotation_id'] = ann_id
            
    def __load_wds_dataset(self, features_file, features_name='dino_features', text_features='ann_feats', load_attn_maps=False):
        print("Loading dataset...")
        def my_decoder(key, value):
            if not key.endswith(".pth"):
                return None
            return torch.load(BytesIO(value))
        dataset = wds.WebDataset(features_file).decode(my_decoder)
        
        self.data = {}
        for idx, obj in enumerate(dataset):
            self.data[idx] = {}
            if text_features != 'clip_txt_out_tokens_avg':
                self.data[idx]['annotation'] = obj['pth'][text_features]
            else:
                mask = obj['pth']['text_input_mask']
                mask[mask.sum() - 1] = False # excluding end of sequence
                mask[0] = False # excluding CLS token
                self.data[idx]['annotation'] = obj['pth']['clip_txt_out_tokens'][mask].mean(dim=0)
            self.data[idx]['image'] = obj['pth'][features_name]
            if load_attn_maps:
                self.data[idx]['self_attn_maps'] = obj['pth']['self_attn_maps']
                self.data[idx]['dino_features'] = obj['pth']['dino_features']
            if text_features == 'clip_txt_out_tokens':
                self.data[idx]['text_input_mask'] = obj['pth']['text_input_mask']
            self.data[idx]['image_id'] = obj['pth']['image_id']
            self.data[idx]['annotation_id'] = obj['pth']['id']
        print("Dataset loaded!")


class DenseConsistencyDataset(DenseFeatureStreamingDataset):
    """Strict annotation-level E5 view over complete dense-feature shards."""

    def __init__(self, root, dino_embed_dim=768, **kwargs):
        self.dino_embed_dim = int(dino_embed_dim)
        super().__init__(root, **kwargs)

    def _expanded_samples(self, shards):
        required = {
            "image_id",
            "disentangled_self_attn",
            "patch_tokens",
            "self_attn_maps",
            "captions",
            "ann_feats",
            "annotation_ids",
        }
        for shard in shards:
            for record in iter_dense_shard_records(shard):
                missing = required.difference(record)
                if missing:
                    raise ValueError(
                        "dense consistency record is missing required fields: "
                        f"{sorted(missing)}"
                    )
                heads = record["disentangled_self_attn"]
                patches = record["patch_tokens"]
                attention_maps = record["self_attn_maps"]
                if not all(
                    torch.is_tensor(tensor)
                    for tensor in (heads, patches, attention_maps)
                ):
                    raise ValueError("dense consistency features must be tensors")
                if heads.ndim != 2 or patches.ndim != 2 or attention_maps.ndim != 2:
                    raise ValueError(
                        "dense consistency shapes must be heads [H,D], patches "
                        "[P,D], and attention maps [H,P]"
                    )
                if heads.shape[0] != attention_maps.shape[0]:
                    raise ValueError("head count H does not match attention maps")
                if patches.shape[0] != attention_maps.shape[1]:
                    raise ValueError("patch count P does not match attention maps")
                if (
                    heads.shape[1] != self.dino_embed_dim
                    or patches.shape[1] != self.dino_embed_dim
                ):
                    raise ValueError(
                        "dense feature dimension D does not match dino_embed_dim="
                        f"{self.dino_embed_dim}"
                    )
                for name, tensor in (
                    ("disentangled_self_attn", heads),
                    ("patch_tokens", patches),
                    ("self_attn_maps", attention_maps),
                ):
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f"{name} contains non-finite values")
                captions = record["captions"]
                ann_feats = record["ann_feats"]
                annotation_ids = record["annotation_ids"]
                if not (
                    len(captions) == len(ann_feats) == len(annotation_ids)
                ):
                    raise ValueError(
                        "captions, ann_feats, and annotation_ids are not aligned"
                    )
                for caption, ann_feat, annotation_id in zip(
                    captions,
                    ann_feats,
                    annotation_ids,
                ):
                    if not torch.is_tensor(ann_feat) or not torch.isfinite(
                        ann_feat
                    ).all():
                        raise ValueError("ann_feats must be finite tensors")
                    yield {
                        "annotation": ann_feat,
                        "image": heads,
                        "metadata": {
                            "image_id": record["image_id"],
                            "annotation_id": annotation_id,
                        },
                        "caption": caption,
                        "patch_tokens": patches,
                        "self_attn_maps": attention_maps,
                    }


class COCOCaptions(Dataset):
    def __init__(self, ann_path, data_dir, split="train", image_transform=None, text_transform=None, device="cuda"):
        self.data = torch.load(ann_path, weights_only=False)
        self.data_dir = data_dir
        self.split = split
        images = {imm['id']: imm for imm in self.data['images']}
        self.samples = []
        for ann in self.data['annotations']:
            if split not in images[ann['image_id']]['file_name']:
                continue
            self.samples.append({
                'annotation': ann['caption'],
                'image_path': images[ann['image_id']]['file_name']
            })
        if len(self.samples) == 0:
            for ann in self.data['annotations']:
                self.samples.append({
                    'annotation': ann['caption'],
                    'image_path': images[ann['image_id']]['file_name']
                })
        self.n_imgs = len(self.samples)
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.device = device
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        annotation = self.samples[idx]['annotation']
        image_path = self.samples[idx]['image_path']
        image = Image.open(os.path.join(self.data_dir, image_path))
        if image.mode == 'L':
            image = image.convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        if self.text_transform:
            annotation = self.text_transform(annotation)[0]
        
        return {"image": image, "annotation": annotation}
    
    def __len__(self):
        return self.n_imgs
    
