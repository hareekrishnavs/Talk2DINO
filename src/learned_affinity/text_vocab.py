"""D3: encode the corpus noun vocabulary ONCE with the frozen CLIP text
encoder, using the SAME prompt template the evaluation uses
(cfg.evaluate.template -> "subset" -> sub_imagenet_template, confirmed by
replaying the recorded cache-build command in
ablationAll/e10_adaptive_diffusion/scripts/build_text_embedding.py). This
is frozen encoding of open-vocabulary caption nouns -- there is no
trainable text-side parameter anywhere in this module."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def build_merged_config():
    """Identical pattern to capture_dino_features.py / build_text_embedding.py
    -- same eval_cfg/eval_base_cfg files, same merge order, same opts (only
    the two model-affecting ones: backbone/CLIP weight paths)."""
    from omegaconf import OmegaConf
    from utils.config import load_config

    default_cfg = load_config(
        "src/open_vocabulary_segmentation/configs/stuff/"
        "dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_tau010.yml"
    )
    org_cfg = OmegaConf.create()
    eval_cfg = OmegaConf.load("src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml")
    cfg = OmegaConf.merge(default_cfg, org_cfg, eval_cfg)
    opts = [
        "model.backbone_weights=/scratch/haree/weights/dinov2_vitb14_reg4_pretrain.pth",
        "model.clip_model_path=/scratch/haree/weights/ViT-B-16.pt",
    ]
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(opts))


def encode_noun_vocabulary(model, vocabulary: list[str], template: str) -> torch.Tensor:
    """model: the frozen DINOText model (as built by build_merged_config +
    build_model). Returns [len(vocabulary), embed_dim] L2-normalised text
    embeddings -- the SAME build_dataset_class_tokens/build_text_embedding
    calls the real cache-build command used, applied to caption nouns
    instead of the 171 COCO-Stuff class names."""
    tokens = model.build_dataset_class_tokens(template, vocabulary)
    return model.build_text_embedding(tokens)


def load_or_build_vocabulary_embeddings(
    cache_path: Path, model, vocabulary: list[str], template: str, *, overwrite: bool = False,
) -> dict[str, Any]:
    """Caches {vocabulary, template, embeddings [N,dim]} to disk (D3: "Cache
    the vocabulary embeddings"). Re-encoding a large vocabulary costs one
    forward pass through the frozen CLIP text tower per noun -- worth
    caching once, reused for every training step afterwards via a plain
    index-select (D4's per-step class list is just a lookup into this
    tensor, no re-encoding)."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not overwrite:
        payload = torch.load(cache_path, weights_only=False)
        if payload["vocabulary"] != vocabulary or payload["template"] != template:
            raise ValueError(
                f"cached vocabulary/template at {cache_path} does not match the "
                f"requested ones -- pass overwrite=True to rebuild"
            )
        return payload
    embeddings = encode_noun_vocabulary(model, vocabulary, template)
    payload = {"vocabulary": vocabulary, "template": template, "embeddings": embeddings.detach().cpu()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


class VocabularyEmbeddings:
    """Thin wrapper: noun -> row index -> embedding, for D4's per-step
    gather. No trainable parameters; embeddings are a frozen buffer."""

    def __init__(self, vocabulary: list[str], embeddings: torch.Tensor):
        if embeddings.shape[0] != len(vocabulary):
            raise ValueError("embeddings row count must match vocabulary length")
        self.vocabulary = vocabulary
        self.index = {noun: i for i, noun in enumerate(vocabulary)}
        self.embeddings = embeddings

    def gather(self, class_list: list[str], *, device=None) -> torch.Tensor:
        missing = [w for w in class_list if w not in self.index]
        if missing:
            raise KeyError(f"nouns not in the cached vocabulary: {missing[:5]}...")
        indices = torch.tensor([self.index[w] for w in class_list], dtype=torch.int64)
        embeddings = self.embeddings[indices]
        return embeddings.to(device) if device is not None else embeddings
