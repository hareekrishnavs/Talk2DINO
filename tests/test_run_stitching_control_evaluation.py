"""End-to-end synthetic CPU tests for the real stitching-control evaluator
CLI (``diagnostics/run_stitching_control_evaluation.py``). Only
``_build_inference`` (needs CUDA/checkpoint weights) and the identity's
registered ``class_count`` (to match this file's noncanonical synthetic
model) are monkeypatched; everything else -- window enumeration, the
shared per-window flow, all four stitching variants, checkpoint/result
writing, and the real checkpoint/report validators -- runs for real, on
CPU, against tiny (8x8) synthetic images. Deliberately lightweight for a
shared login node: torch threading pinned to 1, tiny images, minimal
replicate counts."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import copy
import json
import subprocess
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

import diagnostics.run_stitching_control_evaluation as cli  # noqa: E402
from src.stitching_control_identity import CANONICAL_VARIANT_NAMES, StitchingControlIdentityError  # noqa: E402
from src.k11_k12_power_evaluation_identity import K11K12PowerEvaluationError  # noqa: E402

@pytest.fixture(autouse=True)
def _repair_dinotext_attribute_after_each_test():
    """Pre-existing structural quirk in models/dinotext/__init__.py
    (unrelated to this suite, never modified here) -- see
    test_stitching_control_api.py for the full explanation. This file's
    CLI module imports models.dinotext lazily (inside _run_evaluation, at
    call time, not at collection time), so the repair must run AFTER each
    test, not just once at collection -- an autouse fixture teardown
    guarantees that regardless of which test first triggers the real
    import, sys.modules is left healthy for whatever test file runs next
    in the same session."""
    yield
    if "models" in sys.modules and "models.dinotext" in sys.modules:
        sys.modules["models"].dinotext = sys.modules["models.dinotext"]

REAL_STABILITY_RESULT = Path("/scratch/haree/e12_k11_k12_stability/result-20300858.json")
pytestmark = pytest.mark.skipif(not REAL_STABILITY_RESULT.exists(), reason="real GPU stability-gate result not present on this machine")

NUM_CLASSES = 4  # deliberately noncanonical
IMG_H, IMG_W = 8, 8
NUM_IMAGES = 20  # pilot20's registered image count


def _independent_intersect_and_union(pred, label, num_classes, ignore_index):
    pred = np.asarray(pred).ravel()
    label = np.asarray(label).ravel()
    mask = label != ignore_index
    pred, label = pred[mask], label[mask]
    intersect = pred[pred == label]
    area_i = np.histogram(intersect, bins=num_classes, range=(0, num_classes))[0].astype(np.float64)
    area_p = np.histogram(pred, bins=num_classes, range=(0, num_classes))[0].astype(np.float64)
    area_l = np.histogram(label, bins=num_classes, range=(0, num_classes))[0].astype(np.float64)
    area_u = area_p + area_l - area_i
    return area_i, area_u, area_p, area_l


class _CountingModel(nn.Module):
    def __init__(self, patch_size=2, embed_dim=6, class_count=NUM_CLASSES, seed=0):
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
        images.append({"filename": f"stitchctl_{i:03d}.jpg", "tensor": tensor, "gt": gt})

    class FakeDataset:
        def __init__(self):
            self.images = images
            self.img_infos = [{"filename": im["filename"], "ann": {"seg_map": "x"}} for im in images]
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


def _make_fake_dataset_with_resize(seed=5678, ori_h=6, ori_w=5):
    """Regression fixture for the real-dataset failure mode where the
    processed/inference tensor shape (``img_shape``) differs from the
    true original image shape (``ori_shape``) -- e.g. a resize transform
    in the real mmseg pipeline. ``_make_fake_dataset`` above always uses
    img_shape == ori_shape, which structurally cannot exercise this path;
    a real pilot20 GPU run hit exactly this divergence and crashed inside
    ``dataset.pre_eval`` because the evaluator fabricated ``img_meta``
    from the processed shape for both fields instead of using the
    dataset's real, independent ``ori_shape``."""
    generator = torch.Generator().manual_seed(seed)
    images = []
    for i in range(NUM_IMAGES):
        tensor = torch.rand(3, IMG_H, IMG_W, generator=generator)
        gt = torch.randint(0, NUM_CLASSES, (ori_h, ori_w), generator=generator).numpy()
        images.append({"filename": f"stitchctl_resize_{i:03d}.jpg", "tensor": tensor, "gt": gt})

    class FakeResizedDataset:
        def __init__(self):
            self.images = images
            self.img_infos = [{"filename": im["filename"], "ann": {"seg_map": "x"}} for im in images]
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
                "img_shape": (h, w, 3), "pad_shape": (h, w, 3), "ori_shape": (ori_h, ori_w, 3),
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
                assert pred.shape[-2:] == (ori_h, ori_w), (
                    f"prediction shape {pred.shape[-2:]} must be rescaled to ori_shape ({ori_h}, {ori_w}), "
                    "not left at the processed img_shape"
                )
                gt = self.images[index]["gt"]
                stats = _independent_intersect_and_union(pred, gt, NUM_CLASSES, self.ignore_index)
                results.append(tuple(torch.from_numpy(x) for x in stats))
            return results

    return FakeResizedDataset()


def _make_fake_inference(model):
    return types.SimpleNamespace(model=model, text_embedding=None, num_classes=NUM_CLASSES, align_corners=False)


def run_synthetic_evaluation(*, checkpoint_path, result_path, stats_path, resume=False, seed=1234, dataset_factory=_make_fake_dataset):
    dataset = dataset_factory(seed=seed)
    model = _CountingModel()
    inference = _make_fake_inference(model)

    real_build_inference = cli._build_inference
    real_load_identity_power = None

    def fake_build_inference(root, e3_identity, device, *, log_dir):
        return inference, dataset

    cli._build_inference = fake_build_inference

    import src.stitching_control_identity as sci_module

    real_load_identity = sci_module.load_identity

    def fake_load_identity(*a, **k):
        identity = copy.deepcopy(real_load_identity(*a, **k))
        identity["metrics"]["class_count"] = NUM_CLASSES
        return identity

    cli.load_identity = fake_load_identity

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

    return exit_code, dataset, model


@pytest.fixture
def scratch(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# Baseline end-to-end run
# ---------------------------------------------------------------------------


def test_full_synthetic_run_all_four_variants(scratch):
    exit_code, dataset, model = run_synthetic_evaluation(
        checkpoint_path=scratch / "checkpoint.json", result_path=scratch / "result.json", stats_path=scratch / "stats.json",
    )
    assert exit_code == 0
    assert dataset.getitem_calls == list(range(NUM_IMAGES))

    result = json.loads((scratch / "result.json").read_text())
    assert result["complete"] is True
    assert result["image_count_processed"] == NUM_IMAGES
    assert result["class_count"] == NUM_CLASSES
    assert result["variant_names"] == list(CANONICAL_VARIANT_NAMES)
    for variant in CANONICAL_VARIANT_NAMES:
        assert variant in result["metrics"]
        assert 0.0 <= result["metrics"][variant]["mIoU"] <= 100.0
    for variant in ("hann_probability", "uniform_score", "hann_score"):
        assert variant in result["delta_mIoU_percentage_points_vs_uniform_probability"]

    with np.load(scratch / "stats.npz", allow_pickle=False) as arrays:
        for variant in CANONICAL_VARIANT_NAMES:
            for suffix in ("intersect", "union", "pred"):
                assert arrays[f"{suffix}_{variant}"].shape == (NUM_IMAGES, NUM_CLASSES)
        assert arrays["label"].shape == (NUM_IMAGES, NUM_CLASSES)

    manifest = json.loads((scratch / "stats.json").read_text())
    assert manifest["variant_names"] == list(CANONICAL_VARIANT_NAMES)
    assert manifest["class_count"] == NUM_CLASSES

    checkpoint = json.loads((scratch / "checkpoint.json").read_text())
    assert checkpoint["variant_names"] == list(CANONICAL_VARIANT_NAMES)
    assert checkpoint["complete"] is True

    # model calls == windows, not windows*variants: total snapshot calls
    # must equal windows_processed_total exactly.
    assert result["operation_telemetry"]["backbone_snapshot_calls"] == result["windows_processed_total"]
    assert result["operation_telemetry"]["graph_builds"] == result["windows_processed_total"]
    assert result["operation_telemetry"]["propagation_calls"] == result["windows_processed_total"]
    assert result["operation_telemetry"]["probability_interpolation_calls"] <= result["windows_processed_total"]
    assert result["operation_telemetry"]["score_interpolation_calls"] <= result["windows_processed_total"]
    assert result["operation_telemetry"]["accumulator_finalizations"] == NUM_IMAGES * len(CANONICAL_VARIANT_NAMES)


def test_ori_shape_differs_from_processed_shape_regression(scratch):
    """Regression test for a real pilot20 GPU failure: finalize_prediction
    must rescale to the dataset's real, independent ori_shape rather than
    a fabricated img_meta built from the processed inference-canvas shape.
    The fixture's pre_eval asserts the prediction is at (ori_h, ori_w)
    before doing anything else, so this fails loudly if that regresses."""
    exit_code, dataset, _ = run_synthetic_evaluation(
        checkpoint_path=scratch / "checkpoint.json", result_path=scratch / "result.json", stats_path=scratch / "stats.json",
        dataset_factory=_make_fake_dataset_with_resize,
    )
    assert exit_code == 0
    result = json.loads((scratch / "result.json").read_text())
    assert result["complete"] is True
    assert result["image_count_processed"] == NUM_IMAGES


def test_shared_GT_array_used_for_all_variants(scratch):
    exit_code, _, _ = run_synthetic_evaluation(
        checkpoint_path=scratch / "checkpoint.json", result_path=scratch / "result.json", stats_path=scratch / "stats.json",
    )
    assert exit_code == 0
    with np.load(scratch / "stats.npz", allow_pickle=False) as arrays:
        assert "label" in arrays
        for variant in CANONICAL_VARIANT_NAMES:
            assert f"label_{variant}" not in arrays.files


def test_no_private_paths_in_image_ids(scratch):
    exit_code, _, _ = run_synthetic_evaluation(
        checkpoint_path=scratch / "checkpoint.json", result_path=scratch / "result.json", stats_path=scratch / "stats.json",
    )
    assert exit_code == 0
    checkpoint = json.loads((scratch / "checkpoint.json").read_text())
    for image_id in checkpoint["completed_image_ids"]:
        assert "/" not in image_id
        assert str(scratch) not in image_id


# ---------------------------------------------------------------------------
# Checkpoint / failure safety
# ---------------------------------------------------------------------------


def test_variant_set_mismatch_checkpoint_rejected(scratch):
    exit_code, _, _ = run_synthetic_evaluation(
        checkpoint_path=scratch / "checkpoint.json", result_path=scratch / "result.json", stats_path=scratch / "stats.json",
    )
    assert exit_code == 0
    checkpoint = json.loads((scratch / "checkpoint.json").read_text())
    checkpoint["variant_names"] = ["uniform_probability", "hann_probability", "uniform_score"]  # dropped hann_score
    (scratch / "checkpoint.json").write_text(json.dumps(checkpoint))

    exit_code2, _, _ = run_synthetic_evaluation(
        checkpoint_path=scratch / "checkpoint.json", result_path=scratch / "result.json", stats_path=scratch / "stats.json", resume=True,
    )
    assert exit_code2 == 2


def test_corrupted_prefix_next_index_mismatch_rejected(scratch):
    checkpoint_path, result_path, stats_path = scratch / "checkpoint.json", scratch / "result.json", scratch / "stats.json"
    # interrupt after 10 images via a monkeypatched write_checkpoint_atomically
    real_write = cli.write_checkpoint_atomically
    count = {"n": 0}

    def counting_write(path, record):
        real_write(path, record)
        if path == checkpoint_path:
            count["n"] = record.get("images_completed_count", count["n"])
            if count["n"] >= 10:
                raise RuntimeError("SIMULATED_CRASH")

    cli.write_checkpoint_atomically = counting_write
    try:
        with pytest.raises(RuntimeError, match="SIMULATED_CRASH"):
            run_synthetic_evaluation(checkpoint_path=checkpoint_path, result_path=result_path, stats_path=stats_path)
    finally:
        cli.write_checkpoint_atomically = real_write

    checkpoint = json.loads(checkpoint_path.read_text())
    assert len(checkpoint["completed_image_ids"]) == 10
    checkpoint["next_dataset_index"] = 11  # corrupt: skip index 10
    checkpoint_path.write_text(json.dumps(checkpoint))

    exit_code, dataset, _ = run_synthetic_evaluation(checkpoint_path=checkpoint_path, result_path=result_path, stats_path=stats_path, resume=True)
    assert exit_code != 0
    assert 10 not in dataset.getitem_calls
    assert not result_path.exists()


def test_interrupted_and_resumed_run_scientifically_identical(scratch):
    baseline_dir = scratch / "baseline"
    baseline_dir.mkdir()
    exit_code, _, _ = run_synthetic_evaluation(
        checkpoint_path=baseline_dir / "checkpoint.json", result_path=baseline_dir / "result.json", stats_path=baseline_dir / "stats.json", seed=999,
    )
    assert exit_code == 0
    baseline_result = json.loads((baseline_dir / "result.json").read_text())

    resumed_dir = scratch / "resumed"
    resumed_dir.mkdir()
    real_write = cli.write_checkpoint_atomically
    count = {"n": 0}

    def counting_write(path, record):
        real_write(path, record)
        if path == resumed_dir / "checkpoint.json":
            count["n"] = record.get("images_completed_count", count["n"])
            if count["n"] >= 8:
                raise RuntimeError("SIMULATED_CRASH")

    cli.write_checkpoint_atomically = counting_write
    try:
        with pytest.raises(RuntimeError):
            run_synthetic_evaluation(
                checkpoint_path=resumed_dir / "checkpoint.json", result_path=resumed_dir / "result.json", stats_path=resumed_dir / "stats.json", seed=999,
            )
    finally:
        cli.write_checkpoint_atomically = real_write

    exit_code, dataset, _ = run_synthetic_evaluation(
        checkpoint_path=resumed_dir / "checkpoint.json", result_path=resumed_dir / "result.json", stats_path=resumed_dir / "stats.json", seed=999, resume=True,
    )
    assert exit_code == 0
    assert dataset.getitem_calls == list(range(8, NUM_IMAGES))
    resumed_result = json.loads((resumed_dir / "result.json").read_text())

    for field in ("metrics", "delta_mIoU_percentage_points_vs_uniform_probability", "windows_processed_total", "image_order_digest", "image_count_processed", "class_count"):
        assert resumed_result[field] == baseline_result[field], field


def test_class_count_mismatch_fails_before_dataset_loop(scratch):
    dataset = _make_fake_dataset()
    model = _CountingModel()
    inference = _make_fake_inference(model)  # num_classes=NUM_CLASSES, but identity is NOT patched this time

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
    assert dataset.getitem_calls == []
    assert not (scratch / "r.json").exists()


def test_malformed_checkpoint_json_rejected(scratch):
    checkpoint_path = scratch / "checkpoint.json"
    checkpoint_path.write_text('{"a": 1,}')
    exit_code, _, _ = run_synthetic_evaluation(checkpoint_path=checkpoint_path, result_path=scratch / "r.json", stats_path=scratch / "s.json", resume=True)
    assert exit_code == 2


def test_existing_result_preserved_without_overwrite(scratch):
    result_path = scratch / "result.json"
    result_path.write_text('{"sentinel": true}')
    original_bytes = result_path.read_bytes()

    dataset = _make_fake_dataset()
    model = _CountingModel()
    inference = _make_fake_inference(model)
    real_build_inference = cli._build_inference
    cli._build_inference = lambda root, e3_identity, device, *, log_dir: (inference, dataset)
    try:
        argv = [
            "--run-mode", "pilot20", "--checkpoint", str(scratch / "c.json"), "--result", str(result_path),
            "--per-image-stats", str(scratch / "s.json"), "--stability-result", str(REAL_STABILITY_RESULT), "--device", "cpu",
        ]  # no --overwrite
        exit_code = cli.main(argv)
    finally:
        cli._build_inference = real_build_inference
    assert exit_code == 2
    assert result_path.read_bytes() == original_bytes


def test_keyboardinterrupt_not_swallowed(scratch):
    real_run = cli._run_evaluation

    def raising(args):
        raise KeyboardInterrupt()

    cli._run_evaluation = raising
    try:
        with pytest.raises(KeyboardInterrupt):
            cli.main([
                "--run-mode", "pilot20", "--checkpoint", str(scratch / "c.json"), "--result", str(scratch / "r.json"),
                "--per-image-stats", str(scratch / "s.json"), "--stability-result", str(REAL_STABILITY_RESULT), "--device", "cpu",
            ])
    finally:
        cli._run_evaluation = real_run


def test_systemexit_not_swallowed(scratch):
    real_run = cli._run_evaluation

    def raising(args):
        raise SystemExit(3)

    cli._run_evaluation = raising
    try:
        with pytest.raises(SystemExit):
            cli.main([
                "--run-mode", "pilot20", "--checkpoint", str(scratch / "c.json"), "--result", str(scratch / "r.json"),
                "--per-image-stats", str(scratch / "s.json"), "--stability-result", str(REAL_STABILITY_RESULT), "--device", "cpu",
            ])
    finally:
        cli._run_evaluation = real_run
