#!/usr/bin/env python3
"""Capture read-only, L2-normalised DINOv2 patch features per sliding
window, for offline whole-image-vs-per-window kNN experiments (Part E).

This is infrastructure, not an experiment: it must reproduce the EXACT
per-window feature tensor the existing E3 affinity-oracle cache was built
from, so that a kNN graph rebuilt from these captured features agrees with
the cache's stored knn_indices/knn_weights (verified separately by the
`verify-feature-capture` subcommand in run_e3_affinity_oracle.py).

Design constraint (E2): do not reimplement resize/crop/normalise. This
script drives the REAL, UNMODIFIED sliding-window loop in
segmentation/evaluation/dinotext_seg.py::DINOTextSegInference by attaching
a capture object through the SAME extension point the E3 oracle cache
itself uses (DINOTextMasker.affinity_oracle_observer, wired up via the
`affinity_oracle_capture`/`oracle_dataset` constructor arguments) --
begin_image/set_window/observe/end_image mirror
src.e3_affinity_oracle.OnlineAffinityOracleCapture's interface exactly, so
DINOTextSegInference.forward()/slide_inference() (unmodified) drive it with
zero reimplementation of the window grid, crop, resize or normalisation
math. No file under src/open_vocabulary_segmentation/{models,configs}/ is
read for writing or edited; the model is used strictly read-only (forward
passes only, no optimizer, no .backward()).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

# Unlike build_text_embedding.py (which never decodes an image and can get
# away with stubbing cv2 out), this script loads real images through
# mmseg's LoadImageFromFile -- it needs a REAL, working cv2, not a stub.
# Some GPU node CPUs on this cluster lack AVX-512 and SIGILL on the
# wheelhouse's AVX-512-only opencv build; this node has avx512f, so
# `module load gcc opencv` (BEFORE activating the venv and BEFORE this
# interpreter starts) provides a real, working cv2 -- verified interactively.
# If cv2 is missing, fail loudly rather than silently mis-decoding images.
import cv2  # noqa: F401  -- import-only; confirms the environment module is loaded

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "open_vocabulary_segmentation"))

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
import mmcv
from torch.utils.data import Subset

from utils.config import load_config
from utils.logger import get_logger
from models import build_model
from segmentation.evaluation.builder import build_seg_dataset, build_seg_dataloader

# main.py registers the "FloatImage" mmseg pipeline transform as an import
# side effect (@PIPELINES.register_module(), main.py:54-58); every dataset
# config's test pipeline references it by name (e.g. stuff.py:22), so
# build_seg_dataset() fails with "FloatImage is not in the pipeline
# registry" unless something has imported main.py first. main() itself only
# runs under main.py's own __main__ guard, so this import is side-effect-free
# beyond that one registration.
import main  # noqa: F401

from src.e3_affinity_oracle import (
    CAPTURE_FORMAT,
    AffinityOracleError,
    atomic_json,
    git_provenance,
    ordered_fingerprint,
    sha256_bytes,
    sha256_file,
)
from src.e3_affinity_oracle import _atomic_bytes  # same private write-helper OnlineAffinityOracleCapture uses

CACHE = Path("/scratch/haree/talk2dino_e3_affinity_oracle/cache/full")
SEED = 42


class WindowFeatureCapture:
    """Mirrors OnlineAffinityOracleCapture's begin_image/set_window/observe/
    end_image interface (src/e3_affinity_oracle.py:1165-1198) so the
    existing, unmodified DINOTextSegInference sliding-window loop drives it.
    Stores raw L2-normalised patch features instead of building a kNN graph."""

    def __init__(self, *, output_dir: Path, shard_size: int, overwrite: bool, dtype: torch.dtype = torch.float16):
        if shard_size <= 0:
            raise AffinityOracleError("shard_size must be positive")
        if dtype not in (torch.float16, torch.float32):
            raise AffinityOracleError("dtype must be float16 or float32")
        self.output_dir = Path(output_dir)
        if self.output_dir.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_dir = self.output_dir / "shards"
        self.shard_dir.mkdir(exist_ok=True)
        self.shard_size = shard_size
        self.overwrite = overwrite
        self.dtype = dtype

        self._rows: list[torch.Tensor] = []
        self.shards: list[dict] = []
        self.window_count = 0
        self.images: list[dict] = []

        self._active = False
        self._context: dict | None = None
        self._image_windows: list[dict] = []

    def begin_image(self, metadata, *, dataset_index: int, annotation_path: str) -> None:
        self._active = True
        self._image_windows = []
        self._context = {
            "dataset_index": dataset_index,
            "metadata": dict(metadata),
            "window_start": self.window_count,
        }

    def set_window(self, *, coordinates, grid_indices) -> None:
        if self._active:
            self._context["coordinates"] = list(coordinates)
            self._context["grid_indices"] = list(grid_indices)

    def observe(self, normalized_features: torch.Tensor, raw_scores) -> None:
        # normalized_features: [1,768,32,32] BCHW, already L2-normalised at
        # masker.py:222 (us.normalize(image_feat, dim=1)) -- confirmed in E1b.
        if not self._active:
            return
        context = self._context
        if context is None or "coordinates" not in context:
            raise AffinityOracleError("capture observer lacks sliding-window context")
        features = normalized_features.detach()[0]            # [768,32,32]
        channels, h, w = features.shape
        features = features.reshape(channels, h * w).transpose(0, 1)  # [1024,768]
        features_stored = features.to(device="cpu", dtype=self.dtype).contiguous()
        if tuple(features_stored.shape) != (1024, 768):
            raise AffinityOracleError(
                f"unexpected captured feature shape {tuple(features_stored.shape)}"
            )
        window_index = len(self._image_windows)
        self._rows.append(features_stored)
        self._image_windows.append({
            "window_index": window_index,
            "coordinates": [int(v) for v in context["coordinates"]],
            "grid_indices": [int(v) for v in context["grid_indices"]],
            "global_window_index": self.window_count,
        })
        self.window_count += 1
        if len(self._rows) >= self.shard_size:
            self.flush()

    def end_image(self, *, resized_input_shape, reference_prediction=None, augmentation_index: int = 0) -> None:
        if not self._active:
            return
        context = self._context
        metadata = context["metadata"]
        image_id = metadata.get("ori_filename", metadata.get("filename", context["dataset_index"]))
        self.images.append({
            "dataset_index": int(context["dataset_index"]),
            "image_id": str(image_id),
            "resized_input_shape": [int(v) for v in resized_input_shape],
            "window_start": context["window_start"],
            "window_end": self.window_count,
            "windows": self._image_windows,
        })
        self._active = False
        self._context = None
        self._image_windows = []

    def flush(self) -> None:
        if not self._rows:
            return
        shard = torch.stack(self._rows)  # [n,1024,768] float16
        number = len(self.shards)
        name = f"windows-{number:06d}.pt"
        buffer = io.BytesIO()
        torch.save(shard, buffer)
        payload = buffer.getvalue()
        path = self.shard_dir / name
        _atomic_bytes(path, payload, overwrite=self.overwrite)
        start = self.window_count - len(self._rows)
        self.shards.append({
            "name": f"shards/{name}", "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "window_start": start, "window_end": self.window_count,
        })
        self._rows.clear()

    def finalize(self, *, split: str, limit: int | None, commands: list[str]) -> dict:
        self.flush()
        provenance = git_provenance(REPO_ROOT, allow_dirty=True)
        manifest = {
            "format_version": CAPTURE_FORMAT,
            "split": split,
            "limit": limit,
            "seed": SEED,
            **provenance,
            "existing_cache_manifest_sha256": sha256_file(CACHE / "manifest.json"),
            "selected_image_count": len(self.images),
            "selected_window_count": self.window_count,
            "shards": self.shards,
            "total_bytes": sum(row["bytes"] for row in self.shards),
            "images": self.images,
            "commands": commands,
        }
        atomic_json(self.output_dir / "manifest.json", manifest, overwrite=self.overwrite)
        return manifest


def build_merged_config():
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


def init_single_process_dist(device: str) -> None:
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29511")
        backend = "nccl" if device == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", world_size=1, rank=0)
    if device == "cuda":
        torch.cuda.set_device(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--indices-file", type=Path, default=None,
        help="JSON list of {\"dataset_index\": int} to select exact dataset "
             "indices instead of a --limit prefix range (diagnostic A/B use; "
             "mutually exclusive with --limit).",
    )
    args = parser.parse_args()
    if args.indices_file is not None and args.limit is not None:
        raise AffinityOracleError("--indices-file and --limit are mutually exclusive")

    os.chdir(REPO_ROOT)
    init_single_process_dist(args.device)

    cfg = build_merged_config()
    model = build_model(cfg.model)
    if args.device == "cuda":
        model.cuda()
    model.eval()

    seg_config_path = "src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/stuff.py"
    dataset = build_seg_dataset(seg_config_path)
    concrete_dataset = dataset
    if args.indices_file is not None:
        entries = json.loads(args.indices_file.read_text())
        indices = sorted({int(e["dataset_index"]) for e in entries})
        dataset = Subset(dataset, indices)
    elif args.limit is not None:
        if args.limit <= 0:
            raise AffinityOracleError("--limit must be positive")
        dataset = Subset(dataset, range(0, min(args.limit, len(dataset))))
    loader = build_seg_dataloader(dataset, affinity_oracle_enabled=True)

    text_bundle = torch.load(REPO_ROOT / "ablationAll/e10_adaptive_diffusion/results/text_embedding.pt", weights_only=False)
    text_embedding = text_bundle["text_embedding"].to(next(model.parameters()).device)
    classnames = text_bundle["class_names"]
    with_bg = concrete_dataset.CLASSES[0] == "background"

    dset_cfg = mmcv.Config.fromfile(seg_config_path)

    capture = WindowFeatureCapture(
        output_dir=args.output_dir, shard_size=args.shard_size, overwrite=args.overwrite,
        dtype=getattr(torch, args.dtype),
    )

    # DINOTextSegInference calls the module-global utils.logger.get_logger()
    # with no arguments, which requires get_logger(cfg) to have set its
    # logger_name global at least once first (utils/logger.py:14-20) --
    # normally done by main.py's own train() before it builds anything. Only
    # .model_name/.output are read; the output dir already exists at this
    # point (WindowFeatureCapture.__init__ created it, after its own
    # overwrite/FileExistsError guard ran), so this doesn't affect that
    # check's semantics.
    get_logger(OmegaConf.create({
        "model_name": "e10_feature_capture", "output": str(args.output_dir),
    }))

    from segmentation.evaluation.dinotext_seg import DINOTextSegInference
    seg_model = DINOTextSegInference(
        model, text_embedding, classnames, with_bg,
        test_cfg=dset_cfg.test_cfg,
        affinity_oracle_capture=capture,
        oracle_dataset=concrete_dataset,
    )
    seg_model.eval()

    model_device = next(model.parameters()).device
    started = time.monotonic()
    n_images = 0
    with torch.no_grad():
        for data in loader:
            data["img"] = [image.to(model_device, non_blocking=True) for image in data["img"]]
            data["img_metas"] = [e.data[0] for e in data["img_metas"]]
            seg_model(return_loss=False, rescale=True, **data)
            n_images += 1
            if n_images % 100 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"capture progress: {n_images}/{len(dataset)} images, "
                    f"{capture.window_count} windows, {elapsed:.1f}s elapsed, "
                    f"{elapsed / n_images:.3f}s/image"
                )

    elapsed = time.monotonic() - started
    invocation = " ".join(sys.argv)
    manifest = capture.finalize(split=args.split, limit=args.limit, commands=[invocation])
    print(
        f"DONE: {n_images} images, {capture.window_count} windows, "
        f"{elapsed:.1f}s total, {elapsed / max(n_images,1):.3f}s/image, "
        f"{manifest['total_bytes'] / 1e9:.3f} GB written to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
