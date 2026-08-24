"""End-to-end synthetic CPU tests for the real native-edge-support-audit
evaluator CLI (``diagnostics/run_native_edge_support_audit.py``). Only
``_build_inference`` (needs CUDA/checkpoint weights) and the identity's
registered ``class_count``/``mechanics20_image_count`` (to keep this file's
noncanonical synthetic run small and fast on a shared login node) are
monkeypatched; everything else -- window enumeration, the shared
per-window flow, real 32x32-grid directed top-12 graph construction, real
T=320 finite-step propagation (unmodified, per the identity's own
registered alpha/steps), cross-view support computation, GT reachability
diagnostics, checkpoint/result writing, and the real checkpoint/report
validators -- runs for real, on CPU.

Windows must be genuinely 448x448 pixels (32x32 patches at patch_size=14
-- the identity's own registered, non-overridable grid) for
``compute_window_support`` to accept them, so the synthetic images here
are 449x449 (just over one crop width/height), producing a real 2x2 = 4
window grid per image -- large enough to exercise real cross-view
support, still small enough for a login-node-safe CPU run."""

from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/open_vocabulary_segmentation"))

import diagnostics.run_native_edge_support_audit as cli  # noqa: E402
from src.native_edge_support_identity import NativeEdgeSupportAuditIdentityError  # noqa: E402
from src.native_edge_support_report import verify_result  # noqa: E402


@pytest.fixture(autouse=True)
def _repair_dinotext_attribute_after_each_test():
    """Pre-existing structural quirk in models/dinotext/__init__.py
    (unrelated to this suite, never modified here) -- see
    test_stitching_control_api.py for the full explanation. This file's
    CLI module imports models.dinotext lazily (inside _run_evaluation, at
    call time, not at collection time), so the repair must run AFTER each
    test."""
    yield
    if "models" in sys.modules and "models.dinotext" in sys.modules:
        sys.modules["models"].dinotext = sys.modules["models.dinotext"]


REAL_STABILITY_RESULT = Path("/scratch/haree/e12_k11_k12_stability/result-20300858.json")
pytestmark = pytest.mark.skipif(not REAL_STABILITY_RESULT.exists(), reason="real GPU stability-gate result not present on this machine")

NUM_CLASSES = 4  # deliberately noncanonical
# 448 (crop) + 14 (one patch) = 462: the second window's clamped origin
# lands exactly 14 pixels (one patch) from the first window's origin, so
# the pair is natively aligned despite being clamped -- a plain "+1
# pixel" oversize (e.g. 449) would clamp to a 1-pixel, patch-unaligned
# shift and produce zero aligned-observer coverage in this 2x2 grid.
IMG_H, IMG_W = 462, 462  # a real 2x2 window grid, with a natively aligned clamped pair
NUM_IMAGES = 2  # overrides the registered mechanics20 image count of 20, for a login-node-safe CPU run
IGNORE_INDEX = 255


class _TinyModel(nn.Module):
    def __init__(self, patch_size=14, embed_dim=6, class_count=NUM_CLASSES, seed=0):
        super().__init__()
        self.patch_size = patch_size
        g = torch.Generator().manual_seed(seed)
        self.feature_proj = torch.randn(3, embed_dim, generator=g)
        self.score_proj = torch.randn(embed_dim, class_count, generator=g)

    def generate_patch_snapshot(self, crop, text_embedding):
        del text_embedding
        _, channels, height, width = crop.shape
        ps = self.patch_size
        grid_h, grid_w = height // ps, width // ps
        pooled = F.avg_pool2d(crop, ps)
        flat = pooled.reshape(1, channels, grid_h * grid_w).permute(0, 2, 1)
        raw_feat = flat @ self.feature_proj
        bias = 0.01 * torch.arange(grid_h * grid_w, dtype=torch.float32).view(1, -1, 1)
        features = F.normalize(raw_feat + bias, dim=-1)
        scores = features @ self.score_proj
        return types.SimpleNamespace(unary_scores=scores, dino_features=features, grid_hw=(grid_h, grid_w))

    def masks_from_patch_scores(self, patch_scores, grid_hw, output_hw):
        batch, _n, classes = patch_scores.shape
        grid_h, grid_w = grid_hw
        simmap = patch_scores.reshape(batch, grid_h, grid_w, classes).permute(0, 3, 1, 2)
        mask = torch.sigmoid(simmap)
        return F.interpolate(mask, tuple(output_hw), mode="bilinear", align_corners=True)


def _make_fake_dataset(seed=1234):
    generator = torch.Generator().manual_seed(seed)
    images = []
    for i in range(NUM_IMAGES):
        tensor = torch.rand(3, IMG_H, IMG_W, generator=generator)
        gt = torch.randint(0, NUM_CLASSES, (IMG_H, IMG_W), generator=generator).numpy()
        # Sampling happens at native patch CENTRES (spaced ~14px apart),
        # not raw pixel (0,0), so the ignored region must be large enough
        # to contain at least one full patch's centre pixel.
        gt[0:20, 0:20] = IGNORE_INDEX
        images.append({"filename": f"nesa_{i:03d}.jpg", "tensor": tensor, "gt": gt})

    class FakeDataset:
        def __init__(self):
            self.images = images
            self.img_infos = [{"filename": im["filename"], "ann": {"seg_map": "x"}} for im in images]
            self.CLASSES = [f"c{i}" for i in range(NUM_CLASSES)]
            self.ignore_index = IGNORE_INDEX
            self.getitem_calls: list[int] = []

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            self.getitem_calls.append(index)
            img = self.images[index]
            h, w = img["tensor"].shape[-2:]
            meta = {
                "filename": img["filename"], "ori_filename": img["filename"],
                "img_shape": (h, w, 3), "pad_shape": (h, w, 3), "ori_shape": (h, w, 3),
                "scale_factor": 1.0, "flip": False, "flip_direction": "horizontal", "img_norm_cfg": {},
            }
            return {"img": [img["tensor"]], "img_metas": [meta]}

        def get_gt_seg_map_by_idx(self, index):
            return self.images[index]["gt"]

    return FakeDataset()


def _make_fake_inference(model):
    return types.SimpleNamespace(model=model, text_embedding=None, num_classes=NUM_CLASSES, align_corners=False)


def run_synthetic_evaluation(*, checkpoint_path, result_path, stats_path, resume=False, seed=1234):
    dataset = _make_fake_dataset(seed=seed)
    model = _TinyModel()
    inference = _make_fake_inference(model)

    real_build_inference = cli._build_inference

    def fake_build_inference(root, e3_identity, device, *, log_dir):
        return inference, dataset

    cli._build_inference = fake_build_inference

    import src.native_edge_support_identity as neai_module

    real_load_identity = neai_module.load_identity

    def fake_load_identity(*a, **k):
        identity = copy.deepcopy(real_load_identity(*a, **k))
        identity["dataset"]["classes"] = NUM_CLASSES
        identity["run_modes"]["mechanics20_image_count"] = NUM_IMAGES
        return identity

    cli.load_identity = fake_load_identity

    argv = [
        "--run-mode", "mechanics20", "--checkpoint", str(checkpoint_path), "--result", str(result_path),
        "--per-image-stats", str(stats_path), "--stability-result", str(REAL_STABILITY_RESULT),
        "--device", "cpu", "--overwrite",
    ] + (["--resume"] if resume else [])

    try:
        exit_code = cli.main(argv)
    finally:
        cli._build_inference = real_build_inference
        cli.load_identity = real_load_identity

    return exit_code, dataset, model


@pytest.fixture
def scratch(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# Baseline end-to-end run
# ---------------------------------------------------------------------------


def test_full_synthetic_run_multi_window(scratch):
    exit_code, dataset, model = run_synthetic_evaluation(
        checkpoint_path=scratch / "checkpoint.json", result_path=scratch / "result.json", stats_path=scratch / "stats.json",
    )
    assert exit_code == 0
    assert dataset.getitem_calls == list(range(NUM_IMAGES))

    result = json.loads((scratch / "result.json").read_text())
    assert result["complete"] is True
    assert result["final"] is False
    assert result["image_count_processed"] == NUM_IMAGES
    assert result["class_count"] == NUM_CLASSES
    assert result["funnel"]["images"] == NUM_IMAGES
    assert result["funnel"]["windows"] == NUM_IMAGES * 4  # a real 2x2 window grid per image
    assert result["funnel"]["graph_rows"] == NUM_IMAGES * 4 * 1024
    assert result["funnel"]["directed_edges"] == NUM_IMAGES * 4 * 1024 * 12
    # every row has at least 3 other windows in-image; the canonical
    # aligned pair (offset exactly one stride/patch-size apart) must
    # produce at least some aligned-observer coverage.
    assert result["funnel"]["rows_with_aligned_observer"] > 0
    assert result["ignored_gt_count"] > 0  # the sprinkled ignore-index pixels were sampled at least once

    checkpoint = json.loads((scratch / "checkpoint.json").read_text())
    assert checkpoint["complete"] is True
    assert checkpoint["funnel"] == result["funnel"]

    # independently re-verify the result via the real report validator
    from src.native_edge_support_identity import load_identity
    import hashlib

    identity = copy.deepcopy(load_identity(repo_root=ROOT))
    identity["dataset"]["classes"] = NUM_CLASSES
    identity["run_modes"]["mechanics20_image_count"] = NUM_IMAGES
    identity_sha256 = hashlib.sha256((ROOT / "evaluation_identities/e12_native_edge_support_audit.toml").read_bytes()).hexdigest()
    from src.native_edge_support_report import verify_record

    verify_record(result, identity, identity_sha256=identity_sha256)

    telemetry = result["operation_telemetry"]
    assert telemetry["backbone_snapshot_calls"] == result["windows_processed_total"]
    assert telemetry["graph_builds"] == result["windows_processed_total"]
    assert telemetry["propagation_calls"] == result["windows_processed_total"]


def test_per_image_stats_reconciliation(scratch):
    exit_code, _, _ = run_synthetic_evaluation(
        checkpoint_path=scratch / "checkpoint.json", result_path=scratch / "result.json", stats_path=scratch / "stats.json",
    )
    assert exit_code == 0
    checkpoint = json.loads((scratch / "checkpoint.json").read_text())
    stats = json.loads((scratch / "stats.json").read_text())
    summed = {key: 0 for key in checkpoint["funnel"]}
    for row in stats["per_image_funnel"]:
        for key, value in row.items():
            summed[key] += value
    assert summed == checkpoint["funnel"]


# ---------------------------------------------------------------------------
# Checkpoint / failure safety
# ---------------------------------------------------------------------------


def test_interrupted_and_resumed_run_scientifically_identical(scratch):
    checkpoint_path, result_path, stats_path = scratch / "checkpoint.json", scratch / "result.json", scratch / "stats.json"

    real_write = cli.write_checkpoint_atomically
    count = {"n": 0}

    def interrupting_write(path, payload):
        real_write(path, payload)
        if path == checkpoint_path:
            count["n"] += 1
            if count["n"] == 1:
                raise RuntimeError("simulated crash after the first image")

    cli.write_checkpoint_atomically = interrupting_write
    try:
        with pytest.raises(RuntimeError):
            run_synthetic_evaluation(checkpoint_path=checkpoint_path, result_path=result_path, stats_path=stats_path)
    finally:
        cli.write_checkpoint_atomically = real_write

    partial_checkpoint = json.loads(checkpoint_path.read_text())
    assert partial_checkpoint["complete"] is False
    assert partial_checkpoint["images_completed_count"] == 1

    exit_code, _, _ = run_synthetic_evaluation(
        checkpoint_path=checkpoint_path, result_path=result_path, stats_path=stats_path, resume=True,
    )
    assert exit_code == 0
    resumed_result = json.loads(result_path.read_text())

    fresh_checkpoint, fresh_result, fresh_stats = scratch / "fresh_checkpoint.json", scratch / "fresh_result.json", scratch / "fresh_stats.json"
    exit_code2, _, _ = run_synthetic_evaluation(checkpoint_path=fresh_checkpoint, result_path=fresh_result, stats_path=fresh_stats)
    assert exit_code2 == 0
    fresh_result_data = json.loads(fresh_result.read_text())

    assert resumed_result["funnel"] == fresh_result_data["funnel"]
    assert resumed_result["undefined_reason_counts"] == fresh_result_data["undefined_reason_counts"]
    assert resumed_result["support_count_histogram"] == fresh_result_data["support_count_histogram"]
    assert resumed_result["correctness_cross_tabs"] == fresh_result_data["correctness_cross_tabs"]


def test_malformed_per_image_stats_json_rejected(scratch):
    checkpoint_path, result_path, stats_path = scratch / "checkpoint.json", scratch / "result.json", scratch / "stats.json"
    exit_code, _, _ = run_synthetic_evaluation(checkpoint_path=checkpoint_path, result_path=result_path, stats_path=stats_path)
    assert exit_code == 0

    stats_path.write_text('{"schema": "x", duplicate_key_test: 1, "duplicate_key_test": 2}')
    exit_code2, _, _ = run_synthetic_evaluation(
        checkpoint_path=checkpoint_path, result_path=result_path, stats_path=stats_path, resume=True,
    )
    assert exit_code2 == 2


def test_existing_result_preserved_without_overwrite(scratch):
    checkpoint_path, result_path, stats_path = scratch / "checkpoint.json", scratch / "result.json", scratch / "stats.json"
    exit_code, _, _ = run_synthetic_evaluation(checkpoint_path=checkpoint_path, result_path=result_path, stats_path=stats_path)
    assert exit_code == 0
    original = result_path.read_text()

    argv = [
        "--run-mode", "mechanics20", "--checkpoint", str(checkpoint_path), "--result", str(result_path),
        "--per-image-stats", str(stats_path), "--stability-result", str(REAL_STABILITY_RESULT), "--device", "cpu",
    ]  # deliberately no --overwrite
    dataset = _make_fake_dataset()
    model = _TinyModel()
    inference = _make_fake_inference(model)
    real_build_inference = cli._build_inference
    cli._build_inference = lambda root, e3_identity, device, *, log_dir: (inference, dataset)
    try:
        exit_code2 = cli.main(argv)
    finally:
        cli._build_inference = real_build_inference
    assert exit_code2 == 2
    assert result_path.read_text() == original


def test_keyboard_interrupt_not_swallowed(scratch):
    checkpoint_path, result_path, stats_path = scratch / "checkpoint.json", scratch / "result.json", scratch / "stats.json"
    dataset = _make_fake_dataset()
    model = _TinyModel()
    inference = _make_fake_inference(model)

    real_process = cli._process_one_image
    real_build_inference = cli._build_inference
    real_load_identity = cli.load_identity

    def raising_process(*args, **kwargs):
        raise KeyboardInterrupt()

    def fake_load_identity(*a, **k):
        identity = copy.deepcopy(real_load_identity(*a, **k))
        identity["dataset"]["classes"] = NUM_CLASSES
        identity["run_modes"]["mechanics20_image_count"] = NUM_IMAGES
        return identity

    cli._build_inference = lambda root, e3_identity, device, *, log_dir: (inference, dataset)
    cli._process_one_image = raising_process
    cli.load_identity = fake_load_identity
    try:
        argv = [
            "--run-mode", "mechanics20", "--checkpoint", str(checkpoint_path), "--result", str(result_path),
            "--per-image-stats", str(stats_path), "--stability-result", str(REAL_STABILITY_RESULT),
            "--device", "cpu", "--overwrite",
        ]
        with pytest.raises(KeyboardInterrupt):
            cli.main(argv)
    finally:
        cli._process_one_image = real_process
        cli._build_inference = real_build_inference
        cli.load_identity = real_load_identity
