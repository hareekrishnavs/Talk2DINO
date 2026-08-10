# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
import mmcv
import torch
from mmseg.datasets import build_dataloader, build_dataset
from datasets import get_template

def build_dataset_class_tokens(text_transform, template_set, classnames):
    tokens = []
    templates = get_template(template_set)
    for classname in classnames:
        tokens.append(
            torch.stack([text_transform(template.format(classname)) for template in templates])
        )
    # [N, T, L], N: number of instance, T: number of captions (including ensembled), L: sequence length
    tokens = torch.stack(tokens)

    return tokens


def build_seg_dataset(config):
    """Build a dataset from config."""
    cfg = mmcv.Config.fromfile(config)
    dataset = build_dataset(cfg.data.test)
    return dataset


def build_seg_dataloader(dataset, *, affinity_oracle_enabled=False):
    # batch size is set to 1 to handle varying image size (due to different aspect ratio)
    # Oracle capture holds CUDA model state while iterating.  Loading samples in the
    # rank process avoids forking a persistent worker after CUDA initialization.
    workers_per_gpu = 0 if affinity_oracle_enabled else 1
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
        dist=True,
        shuffle=False,
        persistent_workers=workers_per_gpu > 0,
        pin_memory=False,
    )
    return data_loader
