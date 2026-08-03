# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
import mmcv
import torch

from .dinotext_seg import DINOTextSegInference
from utils import get_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _diagnostic_range(values):
    values = values.detach().to(device="cpu", dtype=torch.float32)
    if values.numel() == 0:
        return "n/a"
    return (
        f"{values.min().item():.3g}/"
        f"{values.mean().item():.3g}/"
        f"{values.max().item():.3g}"
    )


def _log_rgtp_summary(generated, settings, classnames):
    selected = generated.selected_retrieval_count.detach().cpu()
    underfilled = selected < settings.prototype_retrieval_count
    underfilled_indices = torch.nonzero(
        underfilled,
        as_tuple=False,
    ).flatten().tolist()
    underfilled_names = [
        str(classnames[index])
        for index in underfilled_indices
        if index < len(classnames)
    ]
    valid_prototypes = generated.valid_mask.sum(dim=-1)
    effective_beta = (
        generated.confidence
        * settings.prototype_fusion_weight
        * generated.valid_mask.any(dim=-1).to(generated.confidence.dtype)
    )
    names = (
        f"; underfilled_classes={','.join(underfilled_names)}"
        if underfilled_names
        else ""
    )
    get_logger().info(
        "E6 RGTP summary (min/mean/max): "
        f"candidate_pool={_diagnostic_range(generated.candidate_pool_used)}; "
        "valid_unique="
        f"{_diagnostic_range(generated.threshold_valid_count)}; "
        f"selected={_diagnostic_range(selected)}; "
        f"underfilled={len(underfilled_indices)}/{len(classnames)}; "
        f"confidence={_diagnostic_range(generated.confidence)}; "
        f"valid_prototypes={_diagnostic_range(valid_prototypes)}; "
        f"effective_beta={_diagnostic_range(effective_beta)}"
        f"{names}"
    )


def _log_e7_summary(generated, settings, classnames):
    del classnames
    get_logger().info(
        "E7 learned RPA summary (min/mean/max): "
        f"retrieval_count={_diagnostic_range(generated.retrieval_count)}; "
        f"alpha={_diagnostic_range(generated.alpha)}; "
        f"beta={_diagnostic_range(generated.beta)}; "
        f"temperature={settings.prototype_temperature}"
    )


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
    has_e6 = getattr(model, "rgtp", None) is not None
    has_e7 = getattr(model, "learned_rpa", None) is not None
    if not has_e6 and not has_e7:
        text_embedding = model.build_text_embedding(text_tokens)
    else:
        raw_text_embedding, text_embedding = model.build_text_embedding(
            text_tokens,
            return_raw=True,
        )
        if has_e7:
            generated = model.build_learned_retrieval_prototypes(
                raw_text_embedding,
                text_embedding,
            )
            _log_e7_summary(
                generated, model.learned_rpa.settings, classnames
            )
        else:
            generated = model.build_retrieval_grounded_prototypes(
                raw_text_embedding,
                text_embedding,
            )
            _log_rgtp_summary(generated, model.rgtp.settings, classnames)
    kwargs = dict(with_bg=with_bg)
    if hasattr(dset_cfg, "test_cfg"):
        kwargs["test_cfg"] = dset_cfg.test_cfg

    model_type = config.model.type
    if model_type == "DINOText":
        seg_model = DINOTextSegInference(model, text_embedding, classnames, **kwargs, **config.evaluate)
    else:
        raise ValueError(model_type)

    seg_model.CLASSES = dataset.dataset.CLASSES
    seg_model.PALETTE = dataset.dataset.PALETTE

    return seg_model
