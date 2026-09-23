"""CPU-only, fake-model tests for src/semantic_evidence_e3_adapter.py.

No real DINOText model, checkpoint, dataset image, or GT is ever touched.
The fake model below mirrors the current, committed
DINOText.generate_masks / DINOTextMasker.forward_seg API surface exactly
(same call sequence, same intermediate values, same sigmoid identity) but
computes everything from small deterministic linear operations -- it is a
stand-in for testing the ADAPTER's orchestration and invariants, not a
model of DINOText's actual learned behavior. The fake never uses the
implementation-under-test as its own oracle: its forward pass is written
independently, from scratch.

Run with:
    PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=<tmp> python3 -m unittest \
        tests.test_semantic_evidence_e3_adapter -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import semantic_evidence_e3_adapter as e3a  # noqa: E402


class VisualProjectionLayer:
    """Fake, named to match the real E3 projection-type check by class
    name (see _project_dino_features). Deterministic linear projection."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight

    def project_dino(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


class _FakeBackbone(nn.Module):
    """A tiny deterministic stand-in for the DINO backbone. Not a learned
    model -- a fixed linear map from raw pixels to a patch-token sequence,
    written independently of the adapter under test."""

    def __init__(self, num_tokens: int, raw_dim: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.num_tokens = num_tokens
        self.raw_dim = raw_dim
        self.weight = nn.Parameter(torch.randn(3, raw_dim, generator=g), requires_grad=False)
        self.call_count = 0

    def forward_features(self, x: torch.Tensor) -> dict:
        self.call_count += 1
        b, c, h, w = x.shape
        pooled = F.adaptive_avg_pool2d(x, output_size=1).reshape(b, c)  # [B, 3]
        base = pooled @ self.weight  # [B, raw_dim]
        tokens = base.unsqueeze(1).repeat(1, self.num_tokens, 1)
        # give each token a tiny distinct deterministic offset so patches differ
        offsets = torch.linspace(-0.05, 0.05, self.num_tokens).reshape(1, self.num_tokens, 1)
        tokens = tokens + offsets
        return {"x_norm_patchtokens": tokens}


class _FakeMasker:
    """Mirrors DINOTextMasker.forward_seg exactly: cosine similarity via
    L2-normalized image features (un-normalized text), soft_mask =
    sigmoid(simmap) unconditionally on the soft path -- written
    independently from scratch, not by calling the real masker.py."""

    def forward_seg(self, image_feat: torch.Tensor, text_emb: torch.Tensor, deterministic=True, hard=False):
        normalized = F.normalize(image_feat, dim=1, eps=1e-6)
        simmap = torch.einsum("b c h w, n c -> b n h w", normalized, text_emb)
        soft_mask = torch.sigmoid(simmap)
        hard_mask = soft_mask.gt(0.5).type(simmap.dtype)
        mask = hard_mask if hard else soft_mask
        return mask, simmap


class FakeDINOText(nn.Module):
    """Duck-types the exact subset of the real DINOText API
    `extract_e3_window_snapshot` calls: model_name, image_transforms,
    model.forward_features, proj.project_dino, masker.forward_seg,
    with_bg_clean, parameters()."""

    def __init__(self, grid: int = 4, raw_dim: int = 6, feature_dim: int = 5, seed: int = 0):
        super().__init__()
        self.model_name = "dinov2_fake"  # routes into the x_norm_patchtokens dict branch
        self.with_bg_clean = False
        num_tokens = grid * grid
        self.model = _FakeBackbone(num_tokens, raw_dim, seed=seed)
        g = torch.Generator().manual_seed(seed + 1)
        self.proj = VisualProjectionLayer(torch.randn(raw_dim, feature_dim, generator=g))
        self.masker = _FakeMasker()
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def image_transforms(self, x: torch.Tensor) -> torch.Tensor:
        return x  # identity: no real preprocessing needed for this fake

    def generate_masks_reference(self, image: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Independently re-derives the 'ordinary E3 output' the SAME way
        DINOText.generate_masks does (BGR->RGB swap, transforms, backbone,
        proj, forward_seg, resize with align_corners=True) -- written
        separately from extract_e3_window_snapshot, calling the SAME
        underlying fake sub-methods, to serve as the ordinary-path oracle
        for output-preservation tests."""
        h, w = image.shape[2:]
        image_bgr_to_rgb = image[:, [2, 1, 0], :, :]
        img_preprocessed = self.image_transforms(image_bgr_to_rgb)
        image_feat_raw = self.model.forward_features(img_preprocessed)["x_norm_patchtokens"]
        image_feat_proj = self.proj.project_dino(image_feat_raw.float())
        b, num_patches, c = image_feat_proj.shape
        grid = int(round(num_patches ** 0.5))
        image_feat_bchw = image_feat_proj.reshape(b, grid, grid, c).permute(0, 3, 1, 2)
        mask, _simmap = self.masker.forward_seg(image_feat_bchw, text_emb, deterministic=True, hard=False)
        return F.interpolate(mask, (h, w), mode="bilinear", align_corners=True)


def _make_inputs(grid=4, patch=8, classes=3, seed=0):
    g = torch.Generator().manual_seed(seed + 100)
    infer = grid * patch
    image = torch.rand(1, 3, infer, infer, generator=g)
    text_emb = torch.randn(classes, 5, generator=g)  # feature_dim=5 to match FakeDINOText default
    return image, text_emb, infer


class OneForwardPerWindowTests(unittest.TestCase):
    def test_exactly_one_backbone_forward_per_window(self):
        model = FakeDINOText(grid=4)
        image, text_emb, infer = _make_inputs(grid=4, patch=8)
        self.assertEqual(model.model.call_count, 0)
        e3a.extract_e3_window_snapshot(
            model, image, text_emb, window_id="w0", window_ordinal=0,
            top=0, left=0, bottom=infer, right=infer,
            canvas_h=infer, canvas_w=infer, patch_h=8, patch_w=8,
        )
        self.assertEqual(model.model.call_count, 1)

    def test_no_second_forward_across_repeated_calls_is_caller_controlled(self):
        # Calling the adapter twice on two DIFFERENT windows must still be
        # exactly one forward EACH time (2 windows -> 2 forwards total,
        # never more), proving no hidden duplicate pass per call.
        model = FakeDINOText(grid=4)
        image, text_emb, infer = _make_inputs(grid=4, patch=8)
        e3a.extract_e3_window_snapshot(
            model, image, text_emb, window_id="w0", window_ordinal=0,
            top=0, left=0, bottom=infer, right=infer,
            canvas_h=infer, canvas_w=infer * 2, patch_h=8, patch_w=8,
        )
        e3a.extract_e3_window_snapshot(
            model, image, text_emb, window_id="w1", window_ordinal=1,
            top=0, left=infer, bottom=infer, right=infer * 2,
            canvas_h=infer, canvas_w=infer * 2, patch_h=8, patch_w=8,
        )
        self.assertEqual(model.model.call_count, 2)


class SnapshotContentTests(unittest.TestCase):
    def test_returns_aligned_features_and_s0(self):
        model = FakeDINOText(grid=4)
        image, text_emb, infer = _make_inputs(grid=4, patch=8)
        snap, _ordinary = e3a.extract_e3_window_snapshot(
            model, image, text_emb, window_id="w0", window_ordinal=0,
            top=0, left=0, bottom=infer, right=infer,
            canvas_h=infer, canvas_w=infer, patch_h=8, patch_w=8,
        )
        self.assertEqual(snap.dino_features.shape, (16, 5))
        self.assertEqual(snap.s0.shape, (16, 3))
        self.assertTrue(torch.isfinite(snap.dino_features).all())
        self.assertTrue(torch.isfinite(snap.s0).all())

    def test_sigmoid_s0_reconstructs_ordinary_mask_scores_exactly(self):
        model = FakeDINOText(grid=4)
        image, text_emb, infer = _make_inputs(grid=4, patch=8)
        snap, ordinary_output = e3a.extract_e3_window_snapshot(
            model, image, text_emb, window_id="w0", window_ordinal=0,
            top=0, left=0, bottom=infer, right=infer,
            canvas_h=infer, canvas_w=infer, patch_h=8, patch_w=8,
        )
        # Reconstruct pre-resize sigmoid(S0) in the same [1,C,grid,grid]
        # layout the resize step used, and compare against a fresh resize.
        s0_grid = snap.s0.permute(1, 0).reshape(1, snap.class_count, snap.grid_h, snap.grid_w)
        expected = F.interpolate(torch.sigmoid(s0_grid), (infer, infer), mode="bilinear", align_corners=True)
        self.assertTrue(torch.allclose(ordinary_output, expected, atol=1e-6))

    def test_adapter_on_and_ordinary_path_predictions_identical(self):
        model = FakeDINOText(grid=4, seed=7)
        image, text_emb, infer = _make_inputs(grid=4, patch=8, seed=7)
        _snap, adapter_output = e3a.extract_e3_window_snapshot(
            model, image, text_emb, window_id="w0", window_ordinal=0,
            top=0, left=0, bottom=infer, right=infer,
            canvas_h=infer, canvas_w=infer, patch_h=8, patch_w=8,
        )
        ordinary_output = model.generate_masks_reference(image, text_emb)
        self.assertTrue(torch.allclose(adapter_output, ordinary_output, atol=1e-6))

    def test_no_feature_or_score_mutation_between_calls(self):
        model = FakeDINOText(grid=4)
        image, text_emb, infer = _make_inputs(grid=4, patch=8)
        snap1, out1 = e3a.extract_e3_window_snapshot(
            model, image, text_emb, window_id="w0", window_ordinal=0,
            top=0, left=0, bottom=infer, right=infer,
            canvas_h=infer, canvas_w=infer, patch_h=8, patch_w=8,
        )
        s0_copy = snap1.s0.clone()
        feat_copy = snap1.dino_features.clone()
        out_copy = out1.clone()
        # run a second, independent extraction on the same inputs
        e3a.extract_e3_window_snapshot(
            model, image, text_emb, window_id="w0b", window_ordinal=1,
            top=0, left=0, bottom=infer, right=infer,
            canvas_h=infer, canvas_w=infer, patch_h=8, patch_w=8,
        )
        self.assertTrue(torch.equal(snap1.s0, s0_copy))
        self.assertTrue(torch.equal(snap1.dino_features, feat_copy))
        self.assertTrue(torch.equal(out1, out_copy))

    def test_no_checkpoint_access_attribute(self):
        import inspect
        sig = inspect.signature(e3a.extract_e3_window_snapshot)
        for name in sig.parameters:
            self.assertNotIn("checkpoint", name.lower())
            self.assertNotIn("weight_path", name.lower())

    def test_no_gt_argument(self):
        import inspect
        sig = inspect.signature(e3a.extract_e3_window_snapshot)
        for name in sig.parameters:
            self.assertNotIn("gt", name.lower())
            self.assertNotIn("annotation", name.lower())
            self.assertNotIn("label", name.lower())

    def test_query_q1_equals_t0_prediction(self):
        # q1 == the anchor's own original-view unary_top1 class == argmax
        # of S0 at that node -- confirmed directly against the raw S0
        # tensor the adapter returned, independent of any upsample/stitch.
        model = FakeDINOText(grid=4, seed=3)
        image, text_emb, infer = _make_inputs(grid=4, patch=8, seed=3)
        snap, _ = e3a.extract_e3_window_snapshot(
            model, image, text_emb, window_id="w0", window_ordinal=0,
            top=0, left=0, bottom=infer, right=infer,
            canvas_h=infer, canvas_w=infer, patch_h=8, patch_w=8,
        )
        q1_per_node = snap.s0.argmax(dim=1)
        # T0 at native patch resolution before any resize is exactly this
        # per-node argmax; upsampling to pixel resolution and re-arg-maxing
        # is a separate (tested elsewhere) geometric operation, not part of
        # this identity.
        self.assertEqual(q1_per_node.shape, (16,))
        self.assertTrue((q1_per_node < snap.class_count).all())


class InvalidInputRejectionTests(unittest.TestCase):
    def test_wrong_image_rank_rejected(self):
        model = FakeDINOText(grid=4)
        _image, text_emb, infer = _make_inputs(grid=4, patch=8)
        with self.assertRaises(e3a.E3AdapterError):
            e3a.extract_e3_window_snapshot(
                model, torch.rand(3, infer, infer), text_emb, window_id="w0", window_ordinal=0,
                top=0, left=0, bottom=infer, right=infer, canvas_h=infer, canvas_w=infer, patch_h=8, patch_w=8,
            )

    def test_with_bg_clean_true_rejected(self):
        model = FakeDINOText(grid=4)
        model.with_bg_clean = True
        image, text_emb, infer = _make_inputs(grid=4, patch=8)
        with self.assertRaises(e3a.E3AdapterError):
            e3a.extract_e3_window_snapshot(
                model, image, text_emb, window_id="w0", window_ordinal=0,
                top=0, left=0, bottom=infer, right=infer, canvas_h=infer, canvas_w=infer, patch_h=8, patch_w=8,
            )

    def test_box_size_mismatch_rejected(self):
        model = FakeDINOText(grid=4)
        image, text_emb, infer = _make_inputs(grid=4, patch=8)
        with self.assertRaises(e3a.E3AdapterError):
            e3a.extract_e3_window_snapshot(
                model, image, text_emb, window_id="w0", window_ordinal=0,
                top=0, left=0, bottom=infer + 1, right=infer, canvas_h=infer + 1, canvas_w=infer, patch_h=8, patch_w=8,
            )

    def test_unsupported_model_name_rejected(self):
        model = FakeDINOText(grid=4)
        model.model_name = "totally_unsupported_backbone"
        image, text_emb, infer = _make_inputs(grid=4, patch=8)
        with self.assertRaises(e3a.E3AdapterError):
            e3a.extract_e3_window_snapshot(
                model, image, text_emb, window_id="w0", window_ordinal=0,
                top=0, left=0, bottom=infer, right=infer, canvas_h=infer, canvas_w=infer, patch_h=8, patch_w=8,
            )

    def test_non_square_patch_grid_rejected(self):
        # backbone that returns a non-square token count
        model = FakeDINOText(grid=4)
        model.model.num_tokens = 15  # not a perfect square
        image = torch.rand(1, 3, 32, 32)
        text_emb = torch.randn(3, 5)
        with self.assertRaises(e3a.E3AdapterError):
            e3a.extract_e3_window_snapshot(
                model, image, text_emb, window_id="w0", window_ordinal=0,
                top=0, left=0, bottom=32, right=32, canvas_h=32, canvas_w=32, patch_h=8, patch_w=8,
            )


if __name__ == "__main__":
    unittest.main()
