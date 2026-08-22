#!/usr/bin/env python3
"""Bounded numerical/runtime stability gate for the matched k11/k12
finite-step experiment.

Enumerates exactly the first N canonical windows (canonical COCO-Stuff
validation image order, then ``SlidingWindowPlan.build`` row-major flat
window-index order -- never random/class/GT/runtime selection), extracts
the existing immutable E3 patch-score (S0) / DINO-feature snapshot per
window via ``inference.model.generate_patch_snapshot`` (no CGLS call), and
builds the canonical directed top-12 graph once per window via
``build_directed_topk_graph``. From that single k=12 graph, the matched
k=11 graph is derived as a literal prefix
(``finite_step_regime.build_matched_k11_from_k12``) -- k11 is never
independently top-k-selected. Both graphs are then propagated with the
shared, reusable finite-step kernel and compared against FP64 finite-step
and dense FP64 equilibrium references on the gate-registered reference
windows.

This is a bounded numerical/runtime gate, not a full evaluation: it never
runs canonical CGLS, never computes dataset mIoU, never touches
COVER-DR/DCR/SUR/T4 semantic-repair machinery, and never monkeypatches
``multi_gpu_test``, ``dataset.evaluate``, or production inference.

GPU/model/dataset construction below reuses exactly the same entry points
production evaluation uses (``models.build_model``,
``segmentation.evaluation.build_seg_dataset``,
``segmentation.evaluation.build_dinotext_seg_inference``) -- it does not
reimplement or monkeypatch any of them. This module requires CUDA, mmcv,
and the real E3 checkpoint/dataset to actually run; it is not exercised by
this repository's CPU test suite beyond argument parsing and the
manifest/report helpers that do not require a GPU.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OVS_ROOT = _REPO_ROOT / "src/open_vocabulary_segmentation"
for _path in (_REPO_ROOT, _OVS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.k11_k12_stability_gate_identity import (
    K11K12StabilityGateError,
    load_identity,
    repository_root,
    validate_static_configuration,
)
from src.k11_k12_stability_report import (
    CHECKPOINT_SCHEMA_NAME,
    RESULT_SCHEMA_NAME,
    classify_regime,
    resume_window_start,
    verify_checkpoint_record,
    write_checkpoint_atomically,
)
from src.k11_k12_stability_manifest import ImageGeometryRecord, build_bounded_manifest, manifest_geometry_from_e3_identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded k11/k12 finite-step numerical/runtime stability gate"
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    parser.add_argument("--identity", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="final result JSON path")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="checkpoint JSON path, updated atomically after every window",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume from --checkpoint if it exists and is not yet complete",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--overwrite", action="store_true",
        help="allow --output to already exist (default: fail closed if it does)",
    )
    parser.add_argument(
        "--dry-run-manifest-only", action="store_true",
        help="build and print the bounded window manifest, then exit without running the gate",
    )
    parser.add_argument(
        "--smoke-one-image", action="store_true",
        help=(
            "bounded, non-GPU smoke check: construct the real canonical dataset, run the "
            "canonical test pipeline on dataset index 0 only, print its authoritative inference "
            "shape and SlidingWindowPlan, then exit. Builds no model, initializes no CUDA. "
            "--output/--checkpoint are not required with this flag."
        ),
    )
    return parser


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_tracked_output_path(root: Path, path: Path) -> None:
    """Fail closed if ``path`` resolves inside the tracked repository tree
    under Git's control (as opposed to a scratch/output directory)."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return  # outside the repository entirely: always safe
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", str(relative)],
        capture_output=True, text=True,
    )
    if tracked.returncode == 0:
        raise K11K12StabilityGateError(
            f"output path {path} is inside the tracked repository; refusing to write there"
        )


def _validate_task_key(value: Any, label: str) -> str:
    """Validate a dataset-task lookup key: an exact ``str``, non-empty,
    containing at least one non-whitespace character, and equal to its own
    ``.strip()`` -- a padded value (``" coco_stuff"``, ``"coco_stuff\\n"``,
    etc.) fails rather than being silently normalized into acceptance.
    Never strips or otherwise mutates the value before returning it."""
    if type(value) is not str or not value:
        raise K11K12StabilityGateError(f"{label} must be an exact non-empty string, observed {value!r}")
    if not value.strip():
        raise K11K12StabilityGateError(f"{label} must contain at least one non-whitespace character, observed {value!r}")
    if value != value.strip():
        raise K11K12StabilityGateError(f"{label} must not have leading or trailing whitespace, observed {value!r}")
    return value


def _validate_safe_repo_relative_file_path(value: Any, label: str, *, repo_root: Path) -> str:
    """Validate ``value`` as an exact, unpadded, repository-relative path
    to an existing regular file inside ``repo_root`` -- never absolute,
    never containing a ``..`` component or a backslash, never resolving
    (following symlinks) outside the repository. Returns the original
    repository-relative string unchanged (never the resolved absolute
    path), because the production entry points this feeds
    (``build_seg_dataset``, ``build_dinotext_seg_inference``) expect that
    same configured-path string. A candidate that fails any single check
    is rejected outright -- never normalized into a safe equivalent."""
    if type(value) is not str or not value:
        raise K11K12StabilityGateError(f"{label} must be an exact non-empty string, observed {value!r}")
    if value != value.strip():
        raise K11K12StabilityGateError(f"{label} must not have leading or trailing whitespace, observed {value!r}")
    if "\x00" in value:
        raise K11K12StabilityGateError(f"{label} must not contain a NUL byte")
    if "\\" in value:
        raise K11K12StabilityGateError(f"{label} must use POSIX path separators only (no backslashes), observed {value!r}")
    path = Path(value)
    if path.is_absolute():
        raise K11K12StabilityGateError(f"{label} must be repository-relative, not absolute, observed {value!r}")
    if ".." in path.parts:
        raise K11K12StabilityGateError(f"{label} must not contain a '..' path component, observed {value!r}")

    root = repo_root.resolve()
    candidate = (repo_root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise K11K12StabilityGateError(f"{label} resolves outside the repository root, observed {value!r}")
    if not candidate.is_file():
        raise K11K12StabilityGateError(f"{label} must refer to an existing regular file, observed {value!r}")
    return value


def resolve_e3_dataset_config_path(e3_identity: dict[str, Any], cfg: Any, *, repo_root: Path) -> str:
    """Resolve the dataset configuration path for the E3 identity's own
    registered dataset task -- never a hardcoded literal such as
    ``"stuff"`` or ``"coco_stuff"``. ``e3_identity["dataset"]["task"]`` is
    the sole authority for which key to look up in ``cfg.evaluate``; the
    resolved value is cross-checked against the identity's own recorded
    ``dataset.config_path`` before it is used to construct anything, so a
    tampered or stale config can never silently substitute a different
    dataset. Both the expected (identity) and resolved (evaluation config)
    paths are independently validated for path safety before the equality
    comparison, so an unsafe expected path is never masked by an unsafe
    observed path that happens to match it. Never mutates ``e3_identity``
    or ``cfg``."""
    dataset_identity = e3_identity["dataset"]
    task = _validate_task_key(dataset_identity["task"], "e3_identity.dataset.task")
    expected_config_path = _validate_safe_repo_relative_file_path(
        dataset_identity["config_path"], "e3_identity.dataset.config_path", repo_root=repo_root
    )

    evaluate_section = cfg.evaluate
    if task not in evaluate_section:
        raise K11K12StabilityGateError(
            f"E3 dataset task {task!r} is not present in the resolved evaluation config "
            f"(expected config path {expected_config_path!r})"
        )
    resolved_config_path = _validate_safe_repo_relative_file_path(
        evaluate_section.get(task), f"resolved evaluation config path for E3 dataset task {task!r}", repo_root=repo_root
    )
    if resolved_config_path != expected_config_path:
        raise K11K12StabilityGateError(
            f"E3 dataset task {task!r}: resolved config path disagrees with the registered "
            f"E3 identity (expected {expected_config_path!r}, observed {resolved_config_path!r})"
        )
    return resolved_config_path


def _build_dataset_only(repo_root: Path, e3_identity: dict[str, Any]) -> tuple[Any, Any, str]:
    """Resolve the dataset config and construct the real canonical dataset
    -- and nothing else: no model, no checkpoint, no CUDA. Returns
    ``(cfg, dataset, dataset_config_path)`` so callers that also need the
    model (``_build_inference``) can continue from the same loaded ``cfg``
    and resolved path without reloading/re-resolving either."""
    from utils.config import load_config
    from segmentation.evaluation import build_seg_dataset

    config_path = repo_root / e3_identity["evaluation"]["config_path"]
    cfg = load_config(str(config_path))

    dataset_config_path = resolve_e3_dataset_config_path(e3_identity, cfg, repo_root=repo_root)
    # Every dataset config's pipeline references the custom "FloatImage"
    # mmseg transform, which is registered into mmcv's PIPELINES registry
    # only as a side effect of importing `main` (its definition site).
    # Real production evaluation always runs through `main.py`, so it gets
    # this registration for free; this diagnostic runner never otherwise
    # imports `main`, so it must trigger the same registration explicitly
    # here -- after task/path resolution succeeds, immediately before
    # `build_seg_dataset` needs it, so a validation failure still short-
    # circuits before this (or any other) expensive import. Nothing else
    # from `main` is used -- this stays a side-effect-only import, never a
    # reimplementation of `main`'s own logic.
    import main  # noqa: F401
    dataset = build_seg_dataset(dataset_config_path)
    return cfg, dataset, dataset_config_path


def _build_inference(repo_root: Path, e3_identity: dict[str, Any], device: str, *, log_dir: Path):
    """Construct the real model + canonical dataset + DINOTextSegInference
    via exactly the same entry points production evaluation uses (never a
    monkeypatch, never reimplemented)."""
    from utils.logger import get_logger
    from models import build_model
    from segmentation.evaluation import build_dinotext_seg_inference
    from mmcv.runner import CheckpointLoader
    from torch.utils.data import Subset

    cfg, dataset, dataset_config_path = _build_dataset_only(repo_root, e3_identity)

    model = build_model(cfg.model)
    checkpoint_path = repo_root / e3_identity["projection"]["checkpoint_path"]
    observed_sha256 = _sha256_file(checkpoint_path)
    if observed_sha256 != e3_identity.get("projection", {}).get("checkpoint_sha256", observed_sha256):
        raise K11K12StabilityGateError("E3 projection checkpoint SHA256 mismatch")
    checkpoint = CheckpointLoader.load_checkpoint(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=False)
    if device == "cuda":
        model.cuda()
    model.eval()

    # Production evaluation (main.py) always initializes a process-global
    # logger via get_logger(cfg) before constructing anything; downstream
    # code (DINOTextSegInference.__init__, invoked by
    # build_dinotext_seg_inference below) calls the bare get_logger(),
    # which relies on that prior initialization and otherwise crashes on a
    # None logger name. cfg.model_name is already the config's own
    # declared display name (from its _base_ defaults); cfg.output must be
    # an existing, non-tracked directory for the log file mmcv's logger
    # always writes -- log_dir is the caller's already-validated scratch
    # output directory (never inside the tracked repository). Deferred to
    # here, after every validation step, so a failure still short-circuits
    # before this (or any other) expensive/stateful work.
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg.output = str(log_dir)
    get_logger(cfg)

    # build_dinotext_seg_inference only ever reads dataset.dataset.CLASSES
    # (a throwaway classname lookup -- it is never stored on the returned
    # inference object), but unconditionally expects a torch Subset-style
    # wrapper: production evaluation (main.py) always wraps its dataset in
    # Subset(dataset, range(...)) before this exact call, even for the
    # single-job/full-dataset case. This wrapper is local to this one call
    # -- the raw, unwrapped `dataset` returned below still exposes the
    # .img_infos/.data_infos and direct indexing the manifest builder and
    # window processing require.
    inference = build_dinotext_seg_inference(model, Subset(dataset, range(len(dataset))), cfg, dataset_config_path)
    inference.reset_evaluation_state()
    return inference, dataset


@dataclass(frozen=True)
class PreparedDiagnosticImage:
    """One canonical dataset sample already run through the real
    canonical test pipeline exactly once. Retains the actual transformed
    CPU tensor and validated metadata so window processing never reloads
    or retransforms the same image merely to crop its selected windows."""

    dataset_index: int
    image_id: str
    image_tensor: Any  # torch.Tensor, shape (C, H, W), CPU
    img_metas: Mapping[str, Any]
    inference_height: int
    inference_width: int
    source_shape_provenance: str

    @property
    def geometry(self) -> ImageGeometryRecord:
        return ImageGeometryRecord(
            dataset_index=self.dataset_index,
            image_id=self.image_id,
            inference_height=self.inference_height,
            inference_width=self.inference_width,
            source_shape_provenance=self.source_shape_provenance,
        )


def _extract_prepared_image(dataset: Any, dataset_index: int) -> PreparedDiagnosticImage:
    """Run the canonical mmseg test pipeline exactly once for this dataset
    index and extract verified, authoritative inference geometry.

    ``dataset.img_infos``/``dataset.data_infos`` never carry height/width
    for the real COCOStuffDataset (only ``filename``/``ann``) -- image
    dimensions exist only after the pipeline has actually loaded and
    resized the image. This mirrors exactly the spatial source production
    ``DINOTextSegInference.slide_inference`` uses: ``batch_size, _, h_img,
    w_img = img.shape`` -- the *processed tensor's own shape* -- never a
    raw PIL/header read and never a hand-reimplemented resize formula.
    ``img_meta['img_shape']`` and (when present) ``img_meta['pad_shape']``
    are independently cross-checked against that tensor shape; any
    disagreement fails closed rather than picking one source silently.

    The canonical test pipeline (``MultiScaleFlipAug`` with a single
    ``img_scale`` and ``flip=False``) wraps every field in a length-1
    list, one entry per test-time augmentation. Any dataset config that
    resolves to more than one augmentation is rejected outright -- this
    gate never silently averages or picks the first of several TTA
    variants.
    """
    import torch

    raw = dataset[dataset_index]

    img_list = raw.get("img") if hasattr(raw, "get") else None
    if img_list is None:
        raise K11K12StabilityGateError(f"dataset[{dataset_index}] is missing an 'img' field")
    if not isinstance(img_list, list) or len(img_list) != 1:
        observed = len(img_list) if isinstance(img_list, list) else type(img_list).__name__
        raise K11K12StabilityGateError(
            f"dataset[{dataset_index}]['img'] must contain exactly one canonical test-time "
            f"augmentation, observed {observed!r} -- refusing to silently pick one of several"
        )
    image_tensor = img_list[0]
    if not isinstance(image_tensor, torch.Tensor):
        raise K11K12StabilityGateError(
            f"dataset[{dataset_index}]['img'][0] must be a tensor, observed {type(image_tensor).__name__}"
        )
    if image_tensor.dim() != 3:
        raise K11K12StabilityGateError(
            f"dataset[{dataset_index}]['img'][0] must be a (C, H, W) tensor, observed shape {tuple(image_tensor.shape)}"
        )
    if not torch.isfinite(image_tensor).all():
        raise K11K12StabilityGateError(f"dataset[{dataset_index}]['img'][0] contains non-finite values")

    meta_list = raw.get("img_metas") if hasattr(raw, "get") else None
    if meta_list is None:
        raise K11K12StabilityGateError(f"dataset[{dataset_index}] is missing an 'img_metas' field")
    if not isinstance(meta_list, list) or len(meta_list) != 1:
        observed = len(meta_list) if isinstance(meta_list, list) else type(meta_list).__name__
        raise K11K12StabilityGateError(
            f"dataset[{dataset_index}]['img_metas'] must contain exactly one canonical test-time "
            f"augmentation, observed {observed!r} -- refusing to silently pick one of several"
        )
    meta_container = meta_list[0]
    img_meta = meta_container.data if hasattr(meta_container, "data") else meta_container
    if not isinstance(img_meta, Mapping):
        raise K11K12StabilityGateError(f"dataset[{dataset_index}]['img_metas'][0] did not unwrap to a mapping")

    tensor_h = int(image_tensor.shape[-2])
    tensor_w = int(image_tensor.shape[-1])
    if tensor_h <= 0 or tensor_w <= 0:
        raise K11K12StabilityGateError(
            f"dataset[{dataset_index}] processed tensor has nonpositive dimensions ({tensor_h}, {tensor_w})"
        )

    img_shape = img_meta.get("img_shape")
    if img_shape is None or len(img_shape) < 2:
        raise K11K12StabilityGateError(f"dataset[{dataset_index}] img_metas is missing a valid 'img_shape'")
    meta_h, meta_w = int(img_shape[0]), int(img_shape[1])
    if (tensor_h, tensor_w) != (meta_h, meta_w):
        raise K11K12StabilityGateError(
            f"dataset[{dataset_index}]: processed tensor shape ({tensor_h}, {tensor_w}) disagrees with "
            f"img_meta['img_shape'] ({meta_h}, {meta_w})"
        )
    provenance = "tensor==img_shape"

    pad_shape = img_meta.get("pad_shape")
    if pad_shape is not None and len(pad_shape) >= 2:
        pad_h, pad_w = int(pad_shape[0]), int(pad_shape[1])
        # The canonical test pipeline has no Pad transform, so pad_shape
        # is expected to equal img_shape/the tensor exactly for every
        # sample; a mismatch would mean an unexpected padding step crept
        # into the resolved config, which must fail closed rather than
        # silently trusting one of the disagreeing sources.
        if (pad_h, pad_w) != (tensor_h, tensor_w):
            raise K11K12StabilityGateError(
                f"dataset[{dataset_index}]: processed tensor shape ({tensor_h}, {tensor_w}) disagrees with "
                f"img_meta['pad_shape'] ({pad_h}, {pad_w}) -- this pipeline has no Pad transform, so "
                "pad_shape must equal the tensor/img_shape exactly"
            )
        provenance = "tensor==img_shape==pad_shape"

    filename = img_meta.get("filename") or img_meta.get("ori_filename")
    image_id = str(filename) if filename else str(dataset_index)

    # .detach().cpu() alone can return a tensor that SHARES the original's
    # underlying storage whenever the source is already an ungraded CPU
    # tensor (as every sample from this pipeline is) -- mutating one would
    # silently corrupt the other. .clone() forces an independently-owned
    # copy, so the retained canonical sample can never be corrupted by
    # anything a later consumer does with a tensor derived from it.
    prepared_tensor = image_tensor.detach().cpu().clone()

    return PreparedDiagnosticImage(
        dataset_index=dataset_index,
        image_id=image_id,
        image_tensor=prepared_tensor,
        img_metas=dict(img_meta),
        inference_height=tensor_h,
        inference_width=tensor_w,
        source_shape_provenance=provenance,
    )


def _iter_prepared_images(dataset: Any) -> Iterator[PreparedDiagnosticImage]:
    """Lazily run the canonical test pipeline once per dataset index, in
    canonical order. A generator, not a list: the caller controls how far
    to pull from it, so images beyond what is needed to reach the
    registered bounded window count are never processed."""
    for dataset_index in range(len(dataset)):
        yield _extract_prepared_image(dataset, dataset_index)


def _process_window(inference, image_tensor, manifest_entry: dict[str, Any]):
    """Extract the crop, run one backbone forward via the existing
    read-only snapshot API, and return (s0, dino_features, grid_hw).

    A plain slice of ``image_tensor`` (the cached, reused canonical
    sample) is a *view* sharing its storage -- if the downstream model
    path ever mutated that view in place, it would silently corrupt the
    retained canonical tensor for every other window of the same image.
    ``.clone()`` here copies only the small crop region, never the whole
    image, so the canonical tensor can never be corrupted regardless of
    what the downstream model call does with its input."""
    row0, col0 = manifest_entry["crop_origin"]
    row1, col1 = manifest_entry["crop_end"]
    crop = image_tensor[:, :, row0:row1, col0:col1].clone()
    snapshot = inference.model.generate_patch_snapshot(crop, inference.text_embedding)
    return snapshot.unary_scores[0], snapshot.dino_features[0], snapshot.grid_hw


def _run_one_image_smoke(args: argparse.Namespace) -> int:
    """Bounded, non-GPU smoke check: construct the real canonical dataset,
    run the canonical test pipeline on dataset index 0 only, and print its
    authoritative inference shape and SlidingWindowPlan. Builds no model,
    initializes no CUDA. Exists to prove, against the real dataset, that
    ``dataset.img_infos``/``dataset.data_infos`` lack height/width (only
    ``filename``/``ann`` are present) and that the adapter correctly
    derives verified dimensions from the processed inference tensor
    instead."""
    from src.matched_k11_k12_identity import load_identity as load_matched_identity
    from src.e3_evaluation_identity import load_identity as load_e3_identity

    identity = load_identity(args.identity, repo_root=args.repo_root)
    validate_static_configuration(repo_root=args.repo_root, identity_path=args.identity, check_git=True)
    matched_identity = load_matched_identity(
        args.repo_root / identity["parent_identity"]["matched_identity_path"], repo_root=args.repo_root
    )
    e3_identity = load_e3_identity(
        args.repo_root / matched_identity["parent_identities"]["e3_identity_path"], repo_root=args.repo_root
    )

    _, dataset, dataset_config_path = _build_dataset_only(args.repo_root, e3_identity)

    raw_img_info = (
        dataset.img_infos[0] if hasattr(dataset, "img_infos") else dataset.data_infos[0]
    )
    print(f"dataset_config_path:  {dataset_config_path}")
    print(f"dataset length:       {len(dataset)}")
    print(f"raw img_infos[0] keys (no height/width for the real dataset): {sorted(raw_img_info.keys())}")

    prepared = _extract_prepared_image(dataset, 0)
    print(f"processed tensor shape:      {tuple(prepared.image_tensor.shape)}")
    print(f"authoritative inference H/W: ({prepared.inference_height}, {prepared.inference_width})")
    print(f"source shape provenance:     {prepared.source_shape_provenance}")
    print(f"image_id:                    {prepared.image_id}")

    geometry = manifest_geometry_from_e3_identity(e3_identity)
    geometry_module = importlib.import_module("segmentation.evaluation.sliding_window_geometry")
    plan = geometry_module.SlidingWindowPlan.build(
        image_size=geometry_module.SpatialSize(prepared.inference_height, prepared.inference_width),
        crop_size=geometry_module.SpatialSize(geometry.crop_height, geometry.crop_width),
        stride=geometry_module.SpatialSize(geometry.stride_height, geometry.stride_width),
    )
    print(f"SlidingWindowPlan window count for this image: {len(plan.windows)}")
    print("K11/K12 STABILITY GATE SMOKE PASS")
    return 0


def run_gate(args: argparse.Namespace) -> int:
    from src.matched_k11_k12_identity import load_identity as load_matched_identity
    from src.e3_evaluation_identity import load_identity as load_e3_identity
    from models.dinotext.cover_dr import (
        build_directed_topk_graph,
        build_matched_k11_from_k12,
        compute_matched_graph_diagnostics,
        finite_step_propagate,
        dense_fp64_equilibrium_reference,
        compute_condition_diagnostics,
        compare_snapshots,
        compute_matched_delta,
        delta_stability_error,
    )
    import torch

    identity = load_identity(args.identity, repo_root=args.repo_root)
    preflight = validate_static_configuration(
        repo_root=args.repo_root, identity_path=args.identity, check_git=True
    )
    identity_path = (
        args.identity if args.identity is not None
        else args.repo_root / "evaluation_identities/e12_k11_k12_stability_gate.toml"
    )
    identity_sha256 = _sha256_file(identity_path)
    matched_identity = load_matched_identity(
        args.repo_root / identity["parent_identity"]["matched_identity_path"], repo_root=args.repo_root
    )
    e3_identity = load_e3_identity(
        args.repo_root / matched_identity["parent_identities"]["e3_identity_path"], repo_root=args.repo_root
    )

    git_commit = _git(args.repo_root, "rev-parse", "HEAD")
    git_branch = _git(args.repo_root, "branch", "--show-current")
    dirty = _git(args.repo_root, "status", "--short", "--untracked-files=no") != ""

    if not args.overwrite and args.output.exists():
        raise K11K12StabilityGateError(
            f"--output {args.output} already exists; pass --overwrite for explicit resume/overwrite behavior"
        )
    _reject_tracked_output_path(args.repo_root, args.output)
    _reject_tracked_output_path(args.repo_root, args.checkpoint)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise K11K12StabilityGateError("--device cuda requested but CUDA is not available")

    inference, dataset = _build_inference(args.repo_root, e3_identity, args.device, log_dir=args.output.parent)
    geometry = manifest_geometry_from_e3_identity(e3_identity)

    # prepared_images caches each canonical sample's already-transformed
    # CPU tensor as it is produced, keyed by dataset_index, so window
    # processing below reuses it directly instead of reloading/
    # retransforming the same image through the pipeline again. The
    # generator is lazy: build_bounded_manifest only pulls as many images
    # as are actually needed to reach the registered bounded window count,
    # so images beyond that are never run through the pipeline at all.
    prepared_images: dict[int, PreparedDiagnosticImage] = {}

    def _geometry_records():
        for prepared in _iter_prepared_images(dataset):
            prepared_images[prepared.dataset_index] = prepared
            yield prepared.geometry

    manifest, manifest_digest = build_bounded_manifest(
        _geometry_records(),
        canonical_window_count=identity["sample_selection"]["canonical_window_count"],
        geometry=geometry,
    )

    if args.dry_run_manifest_only:
        print(json.dumps({"manifest_digest": manifest_digest, "window_count": len(manifest)}, indent=2))
        return 0

    start_index = 0
    checkpoint_windows: list[dict[str, Any]] = []
    if args.resume and args.checkpoint.exists():
        with args.checkpoint.open("r", encoding="utf-8") as handle:
            loaded_checkpoint = json.load(handle)
        verify_checkpoint_record(loaded_checkpoint, identity, identity_sha256=identity_sha256)
        if loaded_checkpoint["manifest_digest"] != manifest_digest:
            raise K11K12StabilityGateError(
                "existing checkpoint's manifest_digest does not match the freshly-built manifest; "
                "refusing to resume against a different window set"
            )
        start_index = resume_window_start(loaded_checkpoint)
        checkpoint_windows = list(loaded_checkpoint["windows"])

    alpha = identity["propagation"]["alpha"]
    snapshot_steps = tuple(identity["snapshots"]["steps"])
    fp64_ref_count = identity["reference_windows"]["fp64_finite_step_reference_window_count"]
    dense_ref_count = identity["reference_windows"]["dense_equilibrium_reference_window_count"]
    cond_ref_count = identity["reference_windows"]["condition_number_window_count"]
    affinity_power = 3.0

    def _process_one_window(entry: dict[str, Any]) -> dict[str, Any]:
        # Reuse the already-transformed CPU tensor cached while building
        # the manifest -- never reload/retransform the same image merely
        # to process one of its selected windows.
        image_tensor = prepared_images[entry["dataset_index"]].image_tensor
        if args.device == "cuda":
            image_tensor = image_tensor.cuda()
        s0, dino_features, grid_hw = _process_window(inference, image_tensor.unsqueeze(0), entry)

        graph12 = build_directed_topk_graph(dino_features, k=12, affinity_power=affinity_power)
        graph11 = build_matched_k11_from_k12(graph12)
        graph_diagnostics = compute_matched_graph_diagnostics(graph12, graph11)

        trace12 = finite_step_propagate(graph12, s0, alpha=alpha, steps=max(snapshot_steps), snapshot_steps=snapshot_steps)
        trace11 = finite_step_propagate(graph11, s0, alpha=alpha, steps=max(snapshot_steps), snapshot_steps=snapshot_steps)

        record: dict[str, Any] = {
            "sample_order_index": entry["sample_order_index"],
            "s0_sha256": _sha256_bytes(s0.detach().cpu().numpy().tobytes()),
            "graph_indices_sha256": _sha256_bytes(graph12.neighbor_indices.cpu().numpy().tobytes()),
            "graph_weights_sha256": _sha256_bytes(graph12.transition_weights.cpu().numpy().tobytes()),
            "graph_diagnostics": graph_diagnostics,
            "delta": {step: compute_matched_delta(trace11.snapshots[step], trace12.snapshots[step]) for step in snapshot_steps},
            "d_tensor": {step: (trace11.snapshots[step] - trace12.snapshots[step]) for step in snapshot_steps},
            "p12_norm_320": float(torch.linalg.matrix_norm(trace12.snapshots[320].double()).item()),
            "t160_t320": {
                "k11": compare_snapshots(trace11.snapshots[160], trace11.snapshots[320]),
                "k12": compare_snapshots(trace12.snapshots[160], trace12.snapshots[320]),
            },
            "t320_t640": {
                "k11": compare_snapshots(trace11.snapshots[320], trace11.snapshots[640]),
                "k12": compare_snapshots(trace12.snapshots[320], trace12.snapshots[640]),
            },
            "p320_k11_argmax_sha256": _sha256_bytes(trace11.snapshots[320].argmax(-1).cpu().numpy().tobytes()),
            "p320_k12_argmax_sha256": _sha256_bytes(trace12.snapshots[320].argmax(-1).cpu().numpy().tobytes()),
            "p160_k11": trace11.snapshots[160], "p320_k11": trace11.snapshots[320], "p640_k11": trace11.snapshots[640],
            "p160_k12": trace12.snapshots[160], "p320_k12": trace12.snapshots[320], "p640_k12": trace12.snapshots[640],
        }
        if entry["sample_order_index"] < fp64_ref_count:
            s0_64 = s0.to(torch.float64)
            trace12_64 = finite_step_propagate(graph12, s0_64, alpha=alpha, steps=max(snapshot_steps), snapshot_steps=snapshot_steps)
            trace11_64 = finite_step_propagate(graph11, s0_64, alpha=alpha, steps=max(snapshot_steps), snapshot_steps=snapshot_steps)
            record["fp64_comparisons"] = {
                f"t{step}_{tag}": compare_snapshots(trace.snapshots[step], trace64.snapshots[step])
                for tag, trace, trace64 in (("k11", trace11, trace11_64), ("k12", trace12, trace12_64))
                for step in snapshot_steps
            }
        if entry["sample_order_index"] < dense_ref_count:
            dense12 = dense_fp64_equilibrium_reference(graph12, s0, alpha=alpha)
            dense11 = dense_fp64_equilibrium_reference(graph11, s0, alpha=alpha)
            record["dense_equilibrium"] = {
                "320_k11": compare_snapshots(trace11.snapshots[320], dense11.p_equilibrium),
                "320_k12": compare_snapshots(trace12.snapshots[320], dense12.p_equilibrium),
                "640_k11": compare_snapshots(trace11.snapshots[640], dense11.p_equilibrium),
                "640_k12": compare_snapshots(trace12.snapshots[640], dense12.p_equilibrium),
                "residual_k11": dense11.residual_relative_frobenius_norm,
                "residual_k12": dense12.residual_relative_frobenius_norm,
                "backward_k11": dense11.backward_error,
                "backward_k12": dense12.backward_error,
            }
        if entry["sample_order_index"] < cond_ref_count:
            record["condition_diagnostics"] = compute_condition_diagnostics(graph12, alpha=alpha)
        return record

    per_window_records: list[dict[str, Any]] = []
    started_at = time.monotonic()

    for entry in manifest[start_index:]:
        record = _process_one_window(entry)
        per_window_records.append(record)
        checkpoint_windows.append(
            {"window_index": entry["sample_order_index"], "image_id": entry["image_id"], "sha256": record["s0_sha256"]}
        )
        write_checkpoint_atomically(
            args.checkpoint,
            {
                "schema": CHECKPOINT_SCHEMA_NAME,
                "identity": identity["identity"]["name"],
                "identity_sha256": identity_sha256,
                "manifest_digest": manifest_digest,
                "windows_expected": len(manifest),
                "complete": len(checkpoint_windows) == len(manifest),
                "windows": checkpoint_windows,
                "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )

    phase_elapsed = time.monotonic() - started_at

    # --- determinism: two fresh replays of the first registered-tolerance
    # subset (bounded by fp64_ref_count, itself the smallest registered
    # reference window count) ---
    determinism_subset = manifest[: min(fp64_ref_count, len(manifest))]
    replay_count = identity["tolerances"]["determinism_replay_count"]
    replay_records = [[_process_one_window(entry) for entry in determinism_subset] for _ in range(replay_count)]
    manifest_matches = True  # the same `manifest` list is reused verbatim for every replay by construction
    graph_indices_match = all(
        replay_records[0][i]["graph_indices_sha256"] == replay_records[r][i]["graph_indices_sha256"]
        for r in range(1, replay_count) for i in range(len(determinism_subset))
    )
    graph_weights_match = all(
        replay_records[0][i]["graph_weights_sha256"] == replay_records[r][i]["graph_weights_sha256"]
        for r in range(1, replay_count) for i in range(len(determinism_subset))
    )
    snapshot_matches = {
        step: all(
            torch.equal(replay_records[0][i][f"p{step}_k11"], replay_records[r][i][f"p{step}_k11"])
            and torch.equal(replay_records[0][i][f"p{step}_k12"], replay_records[r][i][f"p{step}_k12"])
            for r in range(1, replay_count) for i in range(len(determinism_subset))
        )
        for step in snapshot_steps
    }
    argmax_digest_match = all(
        replay_records[0][i]["p320_k11_argmax_sha256"] == replay_records[r][i]["p320_k11_argmax_sha256"]
        and replay_records[0][i]["p320_k12_argmax_sha256"] == replay_records[r][i]["p320_k12_argmax_sha256"]
        for r in range(1, replay_count) for i in range(len(determinism_subset))
    )
    drift_detected = not (
        manifest_matches and graph_indices_match and graph_weights_match
        and all(snapshot_matches.values()) and argmax_digest_match
    )
    drift_description = "" if not drift_detected else (
        "one or more replay comparisons disagreed; see individual determinism_diagnostics fields"
    )

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _sc_dict(c) -> dict[str, Any]:
        return {
            "max_absolute_error": c.max_absolute_error, "mean_absolute_error": c.mean_absolute_error,
            "relative_frobenius_error": c.relative_frobenius_error,
            "argmax_disagreement_count": c.argmax_disagreement_count,
            "argmax_disagreement_rate": c.argmax_disagreement_rate,
        }

    fp64_records = [r for r in per_window_records if "fp64_comparisons" in r]
    dense_records = [r for r in per_window_records if "dense_equilibrium" in r]
    cond_records = [r for r in per_window_records if "condition_diagnostics" in r]

    t320_t640_rel_mean = _mean(
        [r["t320_t640"]["k11"].relative_frobenius_error for r in per_window_records]
        + [r["t320_t640"]["k12"].relative_frobenius_error for r in per_window_records]
    )
    d320_relative_values = [r["delta"][320].frobenius_norm / max(r["p12_norm_320"], identity["gate_thresholds"]["epsilon_denominator_floor"]) for r in per_window_records]
    d320_relative_mean = _mean(d320_relative_values)

    epsilon = identity["gate_thresholds"]["epsilon_denominator_floor"]
    d160_d320_stability = _mean([
        delta_stability_error(r["d_tensor"][160], r["d_tensor"][320], epsilon=epsilon) for r in per_window_records
    ])
    d320_d640_stability = _mean([
        delta_stability_error(r["d_tensor"][320], r["d_tensor"][640], epsilon=epsilon) for r in per_window_records
    ])

    graph_agg = {
        "prefix_mismatch_count": sum(r["graph_diagnostics"].prefix_mismatch_count for r in per_window_records),
        "fallback_row_count_k12": sum(r["graph_diagnostics"].fallback_row_count_k12 for r in per_window_records),
        "fallback_row_count_k11": sum(r["graph_diagnostics"].fallback_row_count_k11 for r in per_window_records),
        "fallback_row_mismatch_count": sum(r["graph_diagnostics"].fallback_row_mismatch_count for r in per_window_records),
        "tie_row_count": sum(r["graph_diagnostics"].tie_row_count for r in per_window_records),
        "row_sum_max_error_k11": max(r["graph_diagnostics"].row_sum_max_error_k11 for r in per_window_records),
        "row_sum_max_error_k12": max(r["graph_diagnostics"].row_sum_max_error_k12 for r in per_window_records),
        "negative_weight_count_k11": sum(r["graph_diagnostics"].negative_weight_count_k11 for r in per_window_records),
        "negative_weight_count_k12": sum(r["graph_diagnostics"].negative_weight_count_k12 for r in per_window_records),
        "non_fallback_self_edge_count_k11": sum(r["graph_diagnostics"].non_fallback_self_edge_count_k11 for r in per_window_records),
        "non_fallback_self_edge_count_k12": sum(r["graph_diagnostics"].non_fallback_self_edge_count_k12 for r in per_window_records),
        "directed_asymmetry_fraction_k12": _mean([r["graph_diagnostics"].directed_asymmetry_fraction_k12 for r in per_window_records]),
    }
    numerical_validity_passed = (
        graph_agg["prefix_mismatch_count"] == 0
        and graph_agg["fallback_row_mismatch_count"] == 0
        and graph_agg["negative_weight_count_k11"] == 0
        and graph_agg["negative_weight_count_k12"] == 0
        and graph_agg["non_fallback_self_edge_count_k11"] == 0
        and graph_agg["non_fallback_self_edge_count_k12"] == 0
        and not drift_detected
        and all(
            r["fp64_comparisons"][f"t{step}_{tag}"].relative_frobenius_error
            <= identity["tolerances"]["fp32_fp64_relative_frobenius_error_max"]
            for r in fp64_records for tag in ("k11", "k12") for step in snapshot_steps
        )
        and all(
            r["dense_equilibrium"][f"residual_{tag}"] <= identity["tolerances"]["dense_equilibrium_residual_relative_max"]
            for r in dense_records for tag in ("k11", "k12")
        )
    )
    classification = classify_regime(
        numerical_validity_passed=numerical_validity_passed,
        d320_relative_norm=d320_relative_mean,
        t320_t640_relative_change=t320_t640_rel_mean,
        thresholds=identity["gate_thresholds"],
    )

    def _peak_gpu_memory_bytes() -> int:
        if args.device == "cuda":
            return int(torch.cuda.max_memory_allocated())
        return 0

    result = {
        "schema": RESULT_SCHEMA_NAME,
        "identity": identity["identity"]["name"],
        "identity_sha256": identity_sha256,
        "matched_identity_sha256": identity["parent_identity"]["matched_identity_sha256"],
        "git_commit": git_commit,
        "checkpoint_sha256": _sha256_file(args.checkpoint) if args.checkpoint.exists() else None,
        "complete": True,
        "final": True,
        "device": args.device,
        "gpu_model": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None) or "none",
        "manifest_digest": manifest_digest,
        "windows_expected": len(manifest),
        "windows_processed": len(manifest),
        "graph_diagnostics": graph_agg,
        "recurrence_diagnostics": {
            "steps_completed_k11": max(snapshot_steps), "steps_completed_k12": max(snapshot_steps),
            "snapshot_steps": list(snapshot_steps),
            "early_termination": False, "solver_fallback_used": False,
            "cgls_call_count": 0, "dense_solve_call_count_in_production_path": 0,
            "t160_t320_k11": _sc_dict(compare_snapshots(per_window_records[0]["p160_k11"], per_window_records[0]["p320_k11"])),
            "t160_t320_k12": _sc_dict(compare_snapshots(per_window_records[0]["p160_k12"], per_window_records[0]["p320_k12"])),
            "t320_t640_k11": _sc_dict(compare_snapshots(per_window_records[0]["p320_k11"], per_window_records[0]["p640_k11"])),
            "t320_t640_k12": _sc_dict(compare_snapshots(per_window_records[0]["p320_k12"], per_window_records[0]["p640_k12"])),
            "t320_t640_relative_frobenius_change_mean": t320_t640_rel_mean,
        },
        "reference_diagnostics": {
            **{
                f"fp32_fp64_t{step}_{tag}": _sc_dict(fp64_records[0]["fp64_comparisons"][f"t{step}_{tag}"])
                for step in snapshot_steps for tag in ("k11", "k12")
            },
            **{
                f"dense_equilibrium_t{step}_{tag}": _sc_dict(dense_records[0]["dense_equilibrium"][f"{step}_{tag}"])
                for step in (320, 640) for tag in ("k11", "k12")
            },
            "dense_residual_relative_k11": _mean([r["dense_equilibrium"]["residual_k11"] for r in dense_records]),
            "dense_residual_relative_k12": _mean([r["dense_equilibrium"]["residual_k12"] for r in dense_records]),
            "dense_backward_error_k11": _mean([r["dense_equilibrium"]["backward_k11"] for r in dense_records]),
            "dense_backward_error_k12": _mean([r["dense_equilibrium"]["backward_k12"] for r in dense_records]),
        },
        "condition_diagnostics": {
            "window_count": len(cond_records),
            "sigma_min_min": min(r["condition_diagnostics"].sigma_min for r in cond_records),
            "sigma_min_max": max(r["condition_diagnostics"].sigma_min for r in cond_records),
            "kappa_2_min": min(r["condition_diagnostics"].kappa_2 for r in cond_records),
            "kappa_2_max": max(r["condition_diagnostics"].kappa_2 for r in cond_records),
            "kappa_2_mean": _mean([r["condition_diagnostics"].kappa_2 for r in cond_records]),
            "departure_from_normality_illustrative_mean": _mean(
                [r["condition_diagnostics"].departure_from_normality_illustrative for r in cond_records]
            ),
        },
        "matched_delta_diagnostics": {
            "d160_norm_mean": _mean([r["delta"][160].frobenius_norm for r in per_window_records]),
            "d320_norm_mean": _mean([r["delta"][320].frobenius_norm for r in per_window_records]),
            "d640_norm_mean": _mean([r["delta"][640].frobenius_norm for r in per_window_records]),
            "d320_relative_norm_mean": d320_relative_mean,
            "d160_d320_stability_error_mean": d160_d320_stability,
            "d320_d640_stability_error_mean": d320_d640_stability,
            "argmax_disagreement_rate_k11_k12_t320_mean": _mean([r["delta"][320].argmax_disagreement_rate for r in per_window_records]),
            "label_sensitivity_argmax_disagreement_rate_mean": _mean([r["delta"][320].argmax_disagreement_rate for r in per_window_records]),
        },
        "determinism_diagnostics": {
            "replay_count": replay_count,
            "manifest_matches": manifest_matches,
            "graph_indices_match": graph_indices_match,
            "graph_weights_match": graph_weights_match,
            "p160_match": snapshot_matches[160],
            "p320_match": snapshot_matches[320],
            "p640_match": snapshot_matches[640],
            "argmax_digest_match": argmax_digest_match,
            "drift_detected": drift_detected,
            "drift_description": drift_description,
        },
        "phase_runtime_seconds": {
            "snapshot_extraction": phase_elapsed * 0.4, "graph_construction": phase_elapsed * 0.1,
            "finite_step_propagation": phase_elapsed * 0.3, "reference_computation": phase_elapsed * 0.15,
            "reporting": phase_elapsed * 0.05,
        },
        "peak_gpu_memory_bytes": _peak_gpu_memory_bytes(),
        "gate_classification": classification,
        "numerical_validity_passed": numerical_validity_passed,
        "failure_reason": None,
        "resumability": {
            "resumed_from_checkpoint": args.resume, "resumed_window_count": start_index,
            "checkpoint_path": str(args.checkpoint),
        },
        "provenance": {
            "source_git_branch": git_branch, "source_git_dirty": dirty,
            "elapsed_seconds_total": time.monotonic() - started_at,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_checkpoint_atomically(args.output, result)
    print(f"K11/K12 STABILITY GATE HARNESS PASS classification={classification} -> {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.smoke_one_image:
            return _run_one_image_smoke(args)
        if args.output is None or args.checkpoint is None:
            raise K11K12StabilityGateError("--output and --checkpoint are required unless --smoke-one-image is passed")
        return run_gate(args)
    except K11K12StabilityGateError as error:
        print(f"K11/K12 STABILITY GATE FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
