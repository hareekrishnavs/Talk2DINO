# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
import mmcv
import torch

from .dinotext_seg import DINOTextSegInference
from src.e8_balanced_retrieval_adapter import (
    E8InferenceAblationSettings,
    select_e8_scoring_vectors,
)
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


def _e8_diagnostic_range(values):
    """Format an E8 range, using JSON-style ``null`` for no eligible rows."""

    if values.numel() == 0:
        return "null"
    return _diagnostic_range(values)


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


def _pairwise_cosines(values, valid_mask=None):
    values = torch.nn.functional.normalize(values.detach().float(), dim=-1)
    count = values.shape[1]
    if count < 2:
        return values.new_empty(0)
    pairs = torch.triu(
        torch.ones(
            (count, count),
            dtype=torch.bool,
            device=values.device,
        ),
        diagonal=1,
    ).unsqueeze(0)
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=values.device, dtype=torch.bool)
        pairs = pairs & valid_mask[:, :, None] & valid_mask[:, None, :]
    return torch.einsum("ckd,cld->ckl", values, values)[pairs.expand(
        values.shape[0], -1, -1
    )]


def _e8_diagnostic_values(generated, ablation_settings=None):
    """Return collapse diagnostics from retrieval-valid rows only.

    Retrieval eligibility is defined identically to training: a strictly
    positive retrieval count.  Empty eligible sets remain empty so the logger
    can report the documented ``null`` sentinel rather than fabricated zeros.
    """

    if ablation_settings is None:
        ablation_settings = E8InferenceAblationSettings()
    scoring_vectors = select_e8_scoring_vectors(
        generated,
        ablation_settings,
    )
    retrieval_count = generated.retrieval_count.detach()
    eligible = retrieval_count > 0
    slot_mass = generated.slot_mass.detach().float()[eligible]
    mass_sum = slot_mass.sum(dim=-1, keepdim=True)
    normalized_mass = torch.where(
        mass_sum > 0,
        slot_mass / mass_sum.clamp_min(torch.finfo(slot_mass.dtype).tiny),
        torch.zeros_like(slot_mass),
    )
    entropy_terms = torch.where(
        normalized_mass > 0,
        normalized_mass
        * normalized_mass.clamp_min(torch.finfo(normalized_mass.dtype).tiny).log(),
        torch.zeros_like(normalized_mass),
    )
    effective_slots = (-entropy_terms.sum(dim=-1)).exp()
    prototypes = generated.prototypes.detach()[eligible]
    prototype_valid_mask = generated.valid_mask.detach()[eligible]
    return {
        "eligible": eligible,
        "alpha": generated.alpha.detach()[eligible],
        "beta": generated.beta.detach()[eligible],
        "slot_mass": slot_mass,
        "effective_slots": effective_slots,
        "mode_pairwise_cosine": _pairwise_cosines(
            generated.mode_vectors.detach()[eligible]
        ),
        "prototype_pairwise_cosine": _pairwise_cosines(
            prototypes,
            prototype_valid_mask,
        ),
        "scoring_vector_pairwise_cosine": _pairwise_cosines(
            scoring_vectors.detach()[eligible],
            prototype_valid_mask,
        ),
    }


def _log_e8_summary(
    generated,
    settings,
    classnames,
    ablation_settings=None,
):
    if ablation_settings is None:
        ablation_settings = E8InferenceAblationSettings()
    retrieval_count = generated.retrieval_count.detach()
    diagnostics = _e8_diagnostic_values(generated, ablation_settings)
    valid_rows = diagnostics["eligible"]
    fallback = (~valid_rows) | (generated.beta.detach() == 0)
    fallback_indices = torch.nonzero(
        fallback.to(device="cpu"),
        as_tuple=False,
    ).flatten().tolist()
    fallback_names = [
        str(classnames[index])
        for index in fallback_indices
        if index < len(classnames)
    ]

    names = (
        f"; fallback_classes={','.join(fallback_names)}"
        if fallback_names
        else ""
    )
    get_logger().info(
        "E8 balanced retrieval ablation: "
        f"prototype_source={ablation_settings.prototype_source}; "
        f"reliability_mode={ablation_settings.reliability_mode}; "
        "summary (min/mean/max): "
        f"retrieval_count={_diagnostic_range(retrieval_count)}; "
        f"alpha={_e8_diagnostic_range(diagnostics['alpha'])}; "
        f"beta={_e8_diagnostic_range(diagnostics['beta'])}; "
        f"slot_mass={_e8_diagnostic_range(diagnostics['slot_mass'])}; "
        "effective_slots="
        f"{_e8_diagnostic_range(diagnostics['effective_slots'])}; "
        "mode_pairwise_cosine="
        f"{_e8_diagnostic_range(diagnostics['mode_pairwise_cosine'])}; "
        "prototype_pairwise_cosine="
        f"{_e8_diagnostic_range(diagnostics['prototype_pairwise_cosine'])}; "
        "scoring_vector_pairwise_cosine="
        f"{_e8_diagnostic_range(diagnostics['scoring_vector_pairwise_cosine'])}; "
        f"fallback={len(fallback_indices)}/{len(classnames)}; "
        f"prototype_temperature={settings.prototype_temperature}; "
        "responsibility_temperature="
        f"{settings.responsibility_temperature}"
        f"{names}"
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
    has_e8 = getattr(model, "balanced_rpa", None) is not None
    if sum((has_e6, has_e7, has_e8)) > 1:
        raise ValueError(
            "E6 RGTP, E7 learned RPA, and E8 balanced retrieval are "
            "mutually exclusive"
        )
    if not has_e6 and not has_e7 and not has_e8:
        text_embedding = model.build_text_embedding(text_tokens)
    else:
        raw_text_embedding, text_embedding = model.build_text_embedding(
            text_tokens,
            return_raw=True,
        )
        if has_e8:
            generated = model.build_balanced_retrieval_prototypes(
                raw_text_embedding,
                text_embedding,
            )
            _log_e8_summary(
                generated,
                model.balanced_rpa.settings,
                classnames,
                model.balanced_retrieval_ablation,
            )
        elif has_e7:
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
