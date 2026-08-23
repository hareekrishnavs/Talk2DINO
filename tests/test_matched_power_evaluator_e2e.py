"""End-to-end synthetic tests for the real evaluator CLI
(``diagnostics/run_matched_k11_k12_evaluation.py``).

Only ``_build_inference`` (which needs CUDA/checkpoint weights) and
``load_identity``'s registered ``class_count`` (to match this file's
noncanonical synthetic model) are monkeypatched; every other function --
``_extract_prepared_image``, ``stitch_one_image``, ``process_one_window``,
``finalize_prediction``, ``finite_step_propagate``,
``build_directed_topk_graph``, ``build_matched_k11_from_k12``,
``compute_full_precision_metrics``, ``write_checkpoint_atomically``, the
real identity/stability-binding validators, and the real checkpoint
validator -- runs for real, on CPU, against tiny (8x8) synthetic images.
This is deliberately lightweight for a shared login node: torch threading
is pinned to 1, images are tiny, and each test only runs the exact number
of images it needs.

Covers the exact independently-confirmed regression (a checkpoint with
``next_dataset_index`` inconsistent with its own completed-image count
must never cause an image to be silently skipped) and Finding 3's
noncanonical-class-count derivation.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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
    pass  # already set by a prior test module in this process

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/open_vocabulary_segmentation"))

import diagnostics.run_matched_k11_k12_evaluation as cli  # noqa: E402

REAL_STABILITY_RESULT = Path("/scratch/haree/e12_k11_k12_stability/result-20300858.json")
pytestmark = pytest.mark.skipif(
    not REAL_STABILITY_RESULT.exists(), reason="real GPU stability-gate result not present on this machine"
)

NUM_CLASSES = 5  # deliberately noncanonical -- proves Finding 3's class-count derivation
IMG_H, IMG_W = 8, 8
NUM_IMAGES = 20  # pilot20's registered image count


def _independent_intersect_and_union(pred, label, num_classes, ignore_index):
    pred = np.asarray(pred).ravel()
    label = np.asarray(label).ravel()
    mask = label != ignore_index
    pred = pred[mask]
    label = label[mask]
    intersect = pred[pred == label]
    area_intersect = np.histogram(intersect, bins=num_classes, range=(0, num_classes))[0].astype(np.float64)
    area_pred = np.histogram(pred, bins=num_classes, range=(0, num_classes))[0].astype(np.float64)
    area_label = np.histogram(label, bins=num_classes, range=(0, num_classes))[0].astype(np.float64)
    area_union = area_pred + area_label - area_intersect
    return area_intersect, area_union, area_pred, area_label


class _CountingModel(nn.Module):
    def __init__(self, patch_size=2, embed_dim=6, class_count=NUM_CLASSES, seed=0):
        super().__init__()
        self.patch_size = patch_size
        generator = torch.Generator().manual_seed(seed)
        self.feature_proj = torch.randn(3, embed_dim, generator=generator)
        self.score_proj = torch.randn(embed_dim, class_count, generator=generator)
        self.snapshot_call_count = 0
        self.downstream_call_count = 0

    def generate_patch_snapshot(self, crop, text_embedding):
        del text_embedding
        self.snapshot_call_count += 1
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
        self.downstream_call_count += 1
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
        images.append({"filename": f"synthetic_{i:03d}.jpg", "tensor": tensor, "gt": gt})

    class FakeDataset:
        def __init__(self):
            self.images = images
            self.img_infos = [{"filename": img["filename"], "ann": {"seg_map": "x"}} for img in images]
            self.CLASSES = [f"c{i}" for i in range(NUM_CLASSES)]
            self.ignore_index = 255
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

        def pre_eval(self, preds, indices):
            if not isinstance(indices, list):
                indices = [indices]
            if not isinstance(preds, list):
                preds = [preds]
            results = []
            for pred, index in zip(preds, indices):
                gt = self.images[index]["gt"]
                stats = _independent_intersect_and_union(pred, gt, NUM_CLASSES, self.ignore_index)
                results.append(tuple(torch.from_numpy(x) for x in stats))
            return results

    return FakeDataset()


def _make_fake_inference(model):
    return types.SimpleNamespace(model=model, text_embedding=None, num_classes=NUM_CLASSES, align_corners=False)


def run_synthetic_evaluation(*, checkpoint_path, result_path, stats_path, resume=False, seed=1234, interrupt_after=None):
    """Run the real evaluator CLI end to end with only _build_inference and
    the identity's class_count monkeypatched."""
    dataset = _make_fake_dataset(seed=seed)
    model = _CountingModel()
    inference = _make_fake_inference(model)

    real_build_inference = cli._build_inference
    real_load_identity = cli.load_identity
    real_write_checkpoint = cli.write_checkpoint_atomically

    def fake_build_inference(root, e3_identity, device, *, log_dir):
        return inference, dataset

    def fake_load_identity(*a, **k):
        identity = copy.deepcopy(real_load_identity(*a, **k))
        identity["metrics"]["class_count"] = NUM_CLASSES
        return identity

    images_done = {"n": 0}

    def counting_write_checkpoint(path, record):
        real_write_checkpoint(path, record)
        if path == checkpoint_path and interrupt_after is not None:
            images_done["n"] = record.get("images_completed_count", images_done["n"])
            if images_done["n"] >= interrupt_after:
                raise RuntimeError("SIMULATED_CRASH_AFTER_CHECKPOINT")

    cli._build_inference = fake_build_inference
    cli.load_identity = fake_load_identity
    cli.write_checkpoint_atomically = counting_write_checkpoint

    argv = [
        "--run-mode", "pilot20", "--checkpoint", str(checkpoint_path), "--result", str(result_path),
        "--per-image-stats", str(stats_path), "--stability-result", str(REAL_STABILITY_RESULT),
        "--device", "cpu", "--overwrite",
    ] + (["--resume"] if resume else [])

    try:
        exit_code = cli.main(argv)
    finally:
        cli._build_inference = real_build_inference
        cli.load_identity = real_load_identity
        cli.write_checkpoint_atomically = real_write_checkpoint

    return exit_code, dataset, model


@pytest.fixture
def scratch(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# Baseline: uninterrupted pilot20 run, noncanonical class count
# ---------------------------------------------------------------------------


def test_noncanonical_class_count_end_to_end_run(scratch):
    exit_code, dataset, model = run_synthetic_evaluation(
        checkpoint_path=scratch / "checkpoint.json", result_path=scratch / "result.json", stats_path=scratch / "stats.json",
    )
    assert exit_code == 0
    assert dataset.getitem_calls == list(range(NUM_IMAGES))

    result = json.loads((scratch / "result.json").read_text())
    assert result["complete"] is True
    assert result["image_count_processed"] == NUM_IMAGES
    # the crux of Finding 3: class_count follows the live inference value,
    # never the old hardcoded 171
    assert result["class_count"] == NUM_CLASSES
    assert result["class_count"] != 171

    with np.load((scratch / "stats.npz"), allow_pickle=False) as arrays:
        for key in ("label", "intersect_k11", "union_k11", "pred_k11", "intersect_k12", "union_k12", "pred_k12"):
            assert arrays[key].shape == (NUM_IMAGES, NUM_CLASSES)

    manifest = json.loads((scratch / "stats.json").read_text())
    assert manifest["class_count"] == NUM_CLASSES

    checkpoint = json.loads((scratch / "checkpoint.json").read_text())
    assert checkpoint["class_count"] == NUM_CLASSES
    assert checkpoint["complete"] is True


def test_live_identity_class_count_mismatch_fails(scratch):
    """inference.num_classes disagreeing with the identity's registered
    class count must fail closed, before the per-image loop ever starts."""
    dataset = _make_fake_dataset()
    model = _CountingModel()
    inference = _make_fake_inference(model)  # num_classes=5, but identity is NOT patched this time

    real_build_inference = cli._build_inference
    cli._build_inference = lambda root, e3_identity, device, *, log_dir: (inference, dataset)
    try:
        exit_code = cli.main([
            "--run-mode", "pilot20", "--checkpoint", str(scratch / "c.json"), "--result", str(scratch / "r.json"),
            "--per-image-stats", str(scratch / "s.json"), "--stability-result", str(REAL_STABILITY_RESULT),
            "--device", "cpu", "--overwrite",
        ])
    finally:
        cli._build_inference = real_build_inference
    assert exit_code == 2
    assert dataset.getitem_calls == []  # never even started the per-image loop
    assert not (scratch / "r.json").exists()


# ---------------------------------------------------------------------------
# The exact independently-confirmed regression
# ---------------------------------------------------------------------------


def test_exact_skipped_image_regression_now_fails(scratch):
    """Reproduces exactly: expected=20, next_dataset_index=11, only 10
    completed images. Must fail before dataset index 10 can be skipped."""
    checkpoint_path = scratch / "checkpoint.json"
    result_path = scratch / "result.json"
    stats_path = scratch / "stats.json"

    with pytest.raises(RuntimeError, match="SIMULATED_CRASH_AFTER_CHECKPOINT"):
        run_synthetic_evaluation(
            checkpoint_path=checkpoint_path, result_path=result_path, stats_path=stats_path, interrupt_after=10,
        )
    checkpoint = json.loads(checkpoint_path.read_text())
    assert len(checkpoint["completed_image_ids"]) == 10

    # corrupt exactly as the independent verification found
    checkpoint["next_dataset_index"] = 11
    checkpoint_path.write_text(json.dumps(checkpoint))

    exit_code, dataset, model = run_synthetic_evaluation(
        checkpoint_path=checkpoint_path, result_path=result_path, stats_path=stats_path, resume=True,
    )
    assert exit_code != 0
    assert 10 not in dataset.getitem_calls
    assert not result_path.exists()

    final_checkpoint = json.loads(checkpoint_path.read_text())
    assert final_checkpoint["next_dataset_index"] == 11  # never silently repaired


def test_evaluator_never_prints_pass_on_incomplete_output(scratch, capsys):
    checkpoint_path = scratch / "checkpoint.json"
    result_path = scratch / "result.json"
    stats_path = scratch / "stats.json"

    with pytest.raises(RuntimeError):
        run_synthetic_evaluation(
            checkpoint_path=checkpoint_path, result_path=result_path, stats_path=stats_path, interrupt_after=10,
        )
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["next_dataset_index"] = 11
    checkpoint_path.write_text(json.dumps(checkpoint))

    capsys.readouterr()
    run_synthetic_evaluation(
        checkpoint_path=checkpoint_path, result_path=result_path, stats_path=stats_path, resume=True,
    )
    captured = capsys.readouterr()
    assert "PASS" not in captured.out


# ---------------------------------------------------------------------------
# Interrupted + resumed run is scientifically identical to uninterrupted
# ---------------------------------------------------------------------------


def test_uninterrupted_and_resumed_runs_scientifically_identical(scratch):
    baseline_dir = scratch / "baseline"
    baseline_dir.mkdir()
    exit_code, _, _ = run_synthetic_evaluation(
        checkpoint_path=baseline_dir / "checkpoint.json", result_path=baseline_dir / "result.json",
        stats_path=baseline_dir / "stats.json", seed=999,
    )
    assert exit_code == 0
    baseline_result = json.loads((baseline_dir / "result.json").read_text())

    resumed_dir = scratch / "resumed"
    resumed_dir.mkdir()
    with pytest.raises(RuntimeError):
        run_synthetic_evaluation(
            checkpoint_path=resumed_dir / "checkpoint.json", result_path=resumed_dir / "result.json",
            stats_path=resumed_dir / "stats.json", seed=999, interrupt_after=8,
        )
    exit_code, dataset, _ = run_synthetic_evaluation(
        checkpoint_path=resumed_dir / "checkpoint.json", result_path=resumed_dir / "result.json",
        stats_path=resumed_dir / "stats.json", seed=999, resume=True,
    )
    assert exit_code == 0
    assert dataset.getitem_calls == list(range(8, NUM_IMAGES))  # only the remaining images were reprocessed
    resumed_result = json.loads((resumed_dir / "result.json").read_text())

    for field in ("metrics_k11", "metrics_k12", "delta_mIoU_percentage_points", "windows_processed_total",
                  "operation_telemetry", "image_order_digest", "image_count_processed", "class_count"):
        assert resumed_result[field] == baseline_result[field], field

    with np.load(baseline_dir / "stats.npz", allow_pickle=False) as base_arr, \
         np.load(resumed_dir / "stats.npz", allow_pickle=False) as res_arr:
        for key in base_arr.files:
            assert np.array_equal(base_arr[key], res_arr[key]), key
