import webdataset as wds
import os
import random
import torch

from io import BytesIO
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset, default_collate

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


E5_HEAD_SHAPE = (12, 768)
E5_PATCH_SHAPE = (1024, 768)
E5_MAP_SHAPE = (12, 1024)
E5_ANN_SHAPE = (512,)
E5_ATTENTION_SUM_TOLERANCE = 2e-3


def validate_e5_dense_record(record):
    """Validate one E5 image record without inspecting any other shard member."""
    required = {
        "image_id",
        "disentangled_self_attn",
        "patch_tokens",
        "self_attn_maps",
        "captions",
        "ann_feats",
        "annotation_ids",
    }
    if not isinstance(record, dict):
        raise ValueError("dense consistency record must be a dictionary")
    missing = required.difference(record)
    if missing:
        raise ValueError(
            "dense consistency record is missing required fields: "
            f"{sorted(missing)}"
        )

    tensor_contract = (
        ("disentangled_self_attn", E5_HEAD_SHAPE, torch.float32),
        ("patch_tokens", E5_PATCH_SHAPE, torch.float16),
        ("self_attn_maps", E5_MAP_SHAPE, torch.float16),
    )
    for name, expected_shape, expected_dtype in tensor_contract:
        tensor = record[name]
        if not torch.is_tensor(tensor):
            raise ValueError(f"{name} must be a tensor")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {list(expected_shape)}, got "
                f"{list(tensor.shape)}"
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"{name} must have dtype {expected_dtype}, got {tensor.dtype}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")

    attention_maps = record["self_attn_maps"]
    if (attention_maps < 0).any():
        raise ValueError("self_attn_maps contains negative probabilities")
    row_sums = attention_maps.float().sum(dim=-1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=E5_ATTENTION_SUM_TOLERANCE,
        rtol=E5_ATTENTION_SUM_TOLERANCE,
    ):
        raise ValueError(
            "self_attn_maps rows must sum to one within extraction tolerance"
        )

    captions = record["captions"]
    ann_feats = record["ann_feats"]
    annotation_ids = record["annotation_ids"]
    if not all(isinstance(values, list) for values in (captions, ann_feats, annotation_ids)):
        raise ValueError("captions, ann_feats, and annotation_ids must be lists")
    if not captions or not (
        len(captions) == len(ann_feats) == len(annotation_ids)
    ):
        raise ValueError(
            "captions, ann_feats, and annotation_ids are not aligned"
        )
    if any(not isinstance(caption, str) for caption in captions):
        raise ValueError("captions must contain only strings")
    if len(set(annotation_ids)) != len(annotation_ids):
        raise ValueError("annotation_ids must be unique within each image record")
    for ann_feat in ann_feats:
        if not torch.is_tensor(ann_feat):
            raise ValueError("ann_feats must contain tensors")
        if tuple(ann_feat.shape) != E5_ANN_SHAPE:
            raise ValueError(
                f"ann_feats must have shape {list(E5_ANN_SHAPE)}, got "
                f"{list(ann_feat.shape)}"
            )
        if not torch.isfinite(ann_feat).all():
            raise ValueError("ann_feats must contain finite tensors")
    return record


def _annotation_sample(record, annotation_index):
    return {
        "annotation": record["ann_feats"][annotation_index],
        "image": record["disentangled_self_attn"],
        "metadata": {
            "image_id": record["image_id"],
            "annotation_id": record["annotation_ids"][annotation_index],
        },
        "caption": record["captions"][annotation_index],
        "patch_tokens": record["patch_tokens"],
        "self_attn_maps": record["self_attn_maps"],
    }


def balanced_record_pool_sizes(image_count, record_pool_size):
    """Split a known image count into approximately equal bounded pools."""
    image_count = int(image_count)
    record_pool_size = int(record_pool_size)
    if image_count < 0:
        raise ValueError("image_count cannot be negative")
    if record_pool_size <= 0:
        raise ValueError("record_pool_size must be positive")
    if image_count == 0:
        return []

    pool_count = (image_count + record_pool_size - 1) // record_pool_size
    smaller_pool_size, larger_pool_count = divmod(image_count, pool_count)
    return [
        smaller_pool_size + (pool_index < larger_pool_count)
        for pool_index in range(pool_count)
    ]


def iter_image_aware_batches(
    records,
    *,
    image_count,
    batch_size,
    record_pool_size,
    seed,
):
    """Collate every annotation once while keeping image IDs unique per batch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if record_pool_size < batch_size:
        raise ValueError("record_pool_size must be at least batch_size")

    rng = random.Random(int(seed))
    record_iterator = iter(records)
    partial_assignments = []
    pool_sizes = balanced_record_pool_sizes(image_count, record_pool_size)

    def collate_assignments(assignments):
        image_ids = [record["image_id"] for record, _ in assignments]
        if len(image_ids) != len(set(image_ids)):
            raise RuntimeError(
                "image-aware batch contains duplicate image IDs"
            )
        return default_collate(
            [
                _annotation_sample(record, annotation_index)
                for record, annotation_index in assignments
            ]
        )

    for pool_size in pool_sizes:
        record_pool = []
        for _ in range(pool_size):
            try:
                record_pool.append(next(record_iterator))
            except StopIteration:
                raise RuntimeError(
                    "image record stream ended before manifest selected_images "
                    f"count {image_count}"
                ) from None

        total_annotations = sum(
            len(record["annotation_ids"]) for record in record_pool
        )
        local_batch_count = (
            total_annotations + batch_size - 1
        ) // batch_size
        final_local_size = total_annotations - (
            local_batch_count - 1
        ) * batch_size
        capacities = [batch_size] * (local_batch_count - 1) + [
            final_local_size
        ]
        local_assignments = [[] for _ in capacities]

        states = []
        for record in record_pool:
            annotation_order = list(range(len(record["annotation_ids"])))
            rng.shuffle(annotation_order)
            states.append((record, annotation_order))
        rng.shuffle(states)
        states.sort(key=lambda item: len(item[1]), reverse=True)

        for record, annotation_order in states:
            candidate_batches = [
                index for index, capacity in enumerate(capacities) if capacity > 0
            ]
            rng.shuffle(candidate_batches)
            candidate_batches.sort(
                key=lambda index: capacities[index],
                reverse=True,
            )
            if len(candidate_batches) < len(annotation_order):
                raise RuntimeError(
                    "an image has too many captions for the configured "
                    "record_pool_size; increase record_pool_size"
                )
            for annotation_index, local_batch_index in zip(
                annotation_order,
                candidate_batches,
            ):
                local_assignments[local_batch_index].append(
                    (record, annotation_index)
                )
                capacities[local_batch_index] -= 1

        if any(capacities):
            raise RuntimeError("unable to construct complete unique-image batches")
        rng.shuffle(local_assignments)
        full_assignments = [
            assignments
            for assignments in local_assignments
            if len(assignments) == batch_size
        ]
        local_partial = [
            assignments
            for assignments in local_assignments
            if len(assignments) < batch_size
        ]
        if len(local_partial) > 1:
            raise RuntimeError("record pool produced multiple partial batches")

        for assignments in full_assignments:
            rng.shuffle(assignments)
            yield collate_assignments(assignments)

        if local_partial:
            partial_assignments.extend(local_partial[0])
            while len(partial_assignments) >= batch_size:
                assignments = partial_assignments[:batch_size]
                del partial_assignments[:batch_size]
                rng.shuffle(assignments)
                yield collate_assignments(assignments)

        # Release this pool before reading the next one. The carried partial
        # keeps at most batch_size - 1 records alive across pool boundaries.
        record_pool = None
        states = None
        local_assignments = None
        full_assignments = None
        local_partial = None
        record = None
        assignments = None

    try:
        next(record_iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError(
            "image record stream contains more records than manifest "
            f"selected_images count {image_count}"
        )

    if partial_assignments:
        rng.shuffle(partial_assignments)
        yield collate_assignments(partial_assignments)


class ImageAwareBatchLoader:
    """Main-process image-aware batching over worker-partitioned record streams."""

    def __init__(
        self,
        record_dataset,
        *,
        image_count,
        annotation_count,
        batch_size,
        record_pool_size,
        seed,
        num_workers,
    ):
        self.record_dataset = record_dataset
        self.image_count = int(image_count)
        self.annotation_count = int(annotation_count)
        self.batch_size = int(batch_size)
        self.record_pool_size = int(record_pool_size)
        self.seed = int(seed)
        self.num_workers = int(num_workers)

    def __len__(self):
        return (self.annotation_count + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        loader_kwargs = {
            "batch_size": None,
            "num_workers": self.num_workers,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 1
        records = DataLoader(self.record_dataset, **loader_kwargs)
        epoch = getattr(self.record_dataset, "epoch", 0)
        yield from iter_image_aware_batches(
            records,
            image_count=self.image_count,
            batch_size=self.batch_size,
            record_pool_size=self.record_pool_size,
            seed=self.seed + 1_000_003 * epoch,
        )


class _DenseConsistencyRecordDataset(IterableDataset):
    def __init__(self, source):
        super().__init__()
        self.source = source

    @property
    def epoch(self):
        return self.source.epoch

    def __iter__(self):
        yield from self.source._validated_records(self.source._shards_for_worker())


class DenseConsistencyDataset(DenseFeatureStreamingDataset):
    """Strict annotation-level E5 view over complete dense-feature shards."""

    def __init__(self, root, record_pool_size=None, **kwargs):
        self.record_pool_size = record_pool_size
        super().__init__(root, **kwargs)

    def _validated_records(self, shards):
        for shard in shards:
            for record in iter_dense_shard_records(shard):
                yield validate_e5_dense_record(record)

    def _expanded_samples(self, shards):
        for record in self._validated_records(shards):
            for annotation_index in range(len(record["annotation_ids"])):
                yield _annotation_sample(record, annotation_index)

    def make_batch_loader(self, batch_size, seed, num_workers=2):
        self.set_seed(seed)
        record_pool_size = self.record_pool_size or 2 * batch_size
        return ImageAwareBatchLoader(
            _DenseConsistencyRecordDataset(self),
            image_count=self.manifest["selected_images"],
            annotation_count=len(self),
            batch_size=batch_size,
            record_pool_size=record_pool_size,
            seed=seed,
            num_workers=num_workers,
        )


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
    
