"""Reproduce the exact E3 text embedding (171 COCO-Stuff classes) and verify
it against the recorded manifest hash. Read-only: touches no files under
src/open_vocabulary_segmentation/{models,configs}, no checkpoints, no cache.

This replays the EXACT merge sequence recorded in
manifest["commands"][0] (main.py --eval --eval_cfg ... --eval_base_cfg ...
--opts ...), not a guessed one -- an Explore-agent summary claimed the
template resolves to "simple"; replaying the real command chain shows it is
actually "sub_imagenet_template". Trust the reproduction, not the summary.

Note: `import utils` transitively pulls in mmcv -> cv2 even though nothing
on this text-only path ever calls an actual cv2 function. The only `cv2`
module available on this cluster is built against AVX-512
(`module avail opencv` lists nothing else), and this GPU node's CPU has no
AVX-512 (`grep avx512f /proc/cpuinfo` is empty here) -- importing it raises
SIGILL. Stubbing `sys.modules["cv2"]` with a MagicMock (so attribute access
like `cv2.COLOR_BGR2RGB` succeeds without executing real OpenCV code) avoids
the crash without touching anything under src/open_vocabulary_segmentation.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("cv2", MagicMock())

REPO_ROOT = Path("/project/6114407/haree/Talk2DINO")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "open_vocabulary_segmentation"))

import torch
from omegaconf import OmegaConf

from utils.config import load_config
from models import build_model

from src.e3_affinity_oracle import tensor_sha256, load_cache_manifest

CACHE = Path("/scratch/haree/talk2dino_e3_affinity_oracle/cache/full")


def build_merged_config():
    default_cfg = load_config(
        "src/open_vocabulary_segmentation/configs/stuff/"
        "dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_tau010.yml"
    )
    org_cfg = OmegaConf.create()
    eval_cfg = OmegaConf.load(
        "src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml"
    )
    cfg = OmegaConf.merge(default_cfg, org_cfg, eval_cfg)
    opts = [
        "model.backbone_weights=/scratch/haree/weights/dinov2_vitb14_reg4_pretrain.pth",
        "model.clip_model_path=/scratch/haree/weights/ViT-B-16.pt",
        "evaluate.affinity_oracle_cache.enabled=true",
    ]
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(opts))
    return cfg


def main():
    import os
    os.chdir(REPO_ROOT)
    manifest = load_cache_manifest(CACHE, verify_shards=False)
    class_names = manifest["class_names"]
    assert len(class_names) == 171

    cfg = build_merged_config()
    print("template:", cfg.evaluate.template)
    print("model cfg:", OmegaConf.to_yaml(cfg.model))

    model = build_model(cfg.model)
    # Match main.py's train()/build path exactly: build_model() alone does
    # not move every submodule to CUDA (e.g. logit_scale is constructed
    # before dinotext.py's internal .to(device) calls settle), main.py
    # follows up with an explicit model.cuda() when device == "cuda".
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    tokens = model.build_dataset_class_tokens(cfg.evaluate.template, class_names)
    text_embedding = model.build_text_embedding(tokens)
    print("shape:", tuple(text_embedding.shape), "dtype:", text_embedding.dtype)

    actual_hash = tensor_sha256(text_embedding)
    expected_hash = manifest["text_embedding_sha256"]
    print("actual_hash:  ", actual_hash)
    print("expected_hash:", expected_hash)
    matched = actual_hash == expected_hash
    print("MATCH" if matched else "MISMATCH", "-- reproduction", "==" if matched else "!=", "manifest.text_embedding_sha256")

    out_path = Path(
        "/project/6114407/haree/Talk2DINO/ablationAll/e10_adaptive_diffusion/"
        "results/text_embedding.pt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "text_embedding": text_embedding.detach().cpu(),
            "class_names": class_names,
            "sha256": actual_hash,
            "expected_sha256": expected_hash,
            "hash_matched": matched,
            "device_used": str(next(model.parameters()).device),
            "template": cfg.evaluate.template,
            "source_command": manifest["commands"][0],
        },
        out_path,
    )
    print("saved:", out_path)
    if not matched:
        raise SystemExit("MISMATCH -- reproduction does not match the recorded cache identity")


if __name__ == "__main__":
    main()
