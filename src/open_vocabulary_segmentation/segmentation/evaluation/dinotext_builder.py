# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
import mmcv
import shlex
import sys
import torch
import torch.distributed as dist

from .dinotext_seg import DINOTextSegInference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_dinotext_seg_inference(
    model,
    dataset,
    config,
    seg_config,
):
    dset_cfg = mmcv.Config.fromfile(seg_config)  # dataset config
    with_bg = dataset.dataset.CLASSES[0] == "background"
    if with_bg:
        classnames = dataset.dataset.CLASSES[1:]
    else:
        classnames = dataset.dataset.CLASSES
    text_tokens = model.build_dataset_class_tokens(config.evaluate.template, classnames)
    text_embedding = model.build_text_embedding(text_tokens)
    kwargs = dict(with_bg=with_bg)
    if hasattr(dset_cfg, "test_cfg"):
        kwargs["test_cfg"] = dset_cfg.test_cfg

    oracle_cfg = config.evaluate.get("affinity_oracle_cache")
    if oracle_cfg is not None and oracle_cfg.get("enabled", False):
        if (
            dist.is_available() and dist.is_initialized()
            and dist.get_world_size() != 1
        ):
            raise RuntimeError(
                "affinity oracle cache v1 requires exactly one process/GPU"
            )
        from pathlib import Path
        from src.e3_affinity_oracle import (
            AffinityOracleError,
            OnlineAffinityOracleCapture,
            OracleProtocol,
            tensor_sha256,
        )

        concrete_dataset = dataset.dataset
        test_cfg = kwargs["test_cfg"]
        protocol = OracleProtocol(
            class_count=len(text_embedding),
            patch_grid=(32, 32),
            embedding_dimension=768,
            crop_size=tuple(test_cfg.crop_size),
            stride=tuple(test_cfg.stride),
            knn_k=int(oracle_cfg.get("knn_k", 12)),
            affinity_power=float(oracle_cfg.get("affinity_power", 3.0)),
            propagation_steps=int(oracle_cfg.get("propagation_steps", 10)),
            score_dtype=str(oracle_cfg.get("score_dtype", "float16")),
            with_background=with_bg,
            background_threshold=float(config.evaluate.bg_thresh),
            pamr=bool(config.evaluate.pamr),
            flip=False,
            augmentation_scales=1,
        )
        protocol.validate()
        output_dir = oracle_cfg.get("output_dir")
        if not output_dir:
            raise AffinityOracleError("affinity oracle output_dir is required")
        model_cfg = config.model
        weight_dir = Path(model_cfg.get("weight_dir", "weights"))
        backbone = model_cfg.get("backbone_weights")
        dino_path = (
            Path(backbone) if backbone
            else weight_dir / "dinov2_vitb14_reg4_pretrain.pth"
        )
        clip_checkpoint = model_cfg.get("clip_model_path")
        clip_path = (
            Path(clip_checkpoint) if clip_checkpoint
            else weight_dir / "ViT-B-16.pt"
        )
        identity_paths = {
            "e3_config": oracle_cfg.get("e3_config_path"),
            "projection_config": str(
                Path("configs") / f"{model_cfg.proj_class}.yaml"
            ),
            "e3_checkpoint": str(
                Path("weights") / f"{model_cfg.proj_name}.pth"
            ),
            "dino_checkpoint": str(dino_path),
            "clip_checkpoint": str(clip_path),
            "dataset_config": seg_config,
        }
        if any(value is None for value in identity_paths.values()):
            raise AffinityOracleError("oracle identity paths are incomplete")
        kwargs["affinity_oracle_capture"] = OnlineAffinityOracleCapture(
            output_dir=Path(output_dir),
            protocol=protocol,
            class_names=list(classnames),
            source_image_count=len(concrete_dataset),
            max_images=oracle_cfg.get("max_images"),
            windows_per_shard=int(oracle_cfg.get("windows_per_shard", 256)),
            identity_paths=identity_paths,
            dino_identity=model_cfg.model_name,
            text_embedding_sha256=tensor_sha256(text_embedding),
            allow_dirty_source=bool(
                oracle_cfg.get("allow_dirty_source", False)
            ),
            overwrite=bool(oracle_cfg.get("overwrite", False)),
            commands=[" ".join(shlex.quote(argument) for argument in sys.argv)],
            fp16_control_manifest=(
                Path(oracle_cfg.get("fp16_control_manifest"))
                if oracle_cfg.get("fp16_control_manifest") else None
            ),
        )
        kwargs["oracle_dataset"] = concrete_dataset

    model_type = config.model.type
    if model_type == "DINOText":
        seg_model = DINOTextSegInference(model, text_embedding, classnames, **kwargs, **config.evaluate)
    else:
        raise ValueError(model_type)

    seg_model.CLASSES = dataset.dataset.CLASSES
    seg_model.PALETTE = dataset.dataset.PALETTE

    return seg_model
