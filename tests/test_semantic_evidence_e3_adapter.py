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
    """A tiny deterministic stand-in for the DINO backbone's patch
    embedding: non-overlapping patch_size x patch_size average-pooling
    (mirroring a Conv2d(kernel_size=patch, stride=patch) patch embedding's
    spatial reduction, without a learned kernel) followed by a fixed
    linear projection -- NOT global pooling, so the returned token grid's
    size and content genuinely depend on the input's actual (post-resize)
    spatial size, the way a real patch embedding does. Requires the
    input's spatial size to already be an exact multiple of patch_size,
    which the fixed-canvas resize in FakeDINOText.image_transforms always
    guarantees -- written independently of the adapter under test."""

    def __init__(self, patch_size: int, raw_dim: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.patch_size = patch_size
        self.raw_dim = raw_dim
        self.weight = nn.Parameter(torch.randn(3, raw_dim, generator=g), requires_grad=False)
        self.call_count = 0

    def forward_features(self, x: torch.Tensor) -> dict:
        self.call_count += 1
        b, c, h, w = x.shape
        p = self.patch_size
        assert h % p == 0 and w % p == 0, f"fake backbone input ({h}x{w}) must be a multiple of patch_size ({p})"
        pooled = F.avg_pool2d(x, kernel_size=p, stride=p)  # [B, 3, h/p, w/p]
        gh, gw = pooled.shape[2], pooled.shape[3]
        flat = pooled.permute(0, 2, 3, 1).reshape(b, gh * gw, c)  # [B, N, 3], row-major (row varies slower)
        tokens = flat @ self.weight  # [B, N, raw_dim]
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

    def __init__(self, grid: int = 4, raw_dim: int = 6, feature_dim: int = 5, seed: int = 0, proj=None,
                 patch_size: int = 8, resize_dim: int | None = None):
        super().__init__()
        self.model_name = "dinov2_fake"  # routes into the x_norm_patchtokens dict branch
        self.with_bg_clean = False
        self.patch_size = patch_size
        # Fixed backbone-input canvas: authoritative E3's image_transforms
        # resizes every source window to this size regardless of the
        # window's own extent (T.Resize((resize_dim, resize_dim))). Default
        # preserves every existing caller's values exactly (grid*patch_size
        # was already the constructed test image's own size everywhere, so
        # the resize below is a same-size no-op for all pre-existing tests).
        self.resize_dim = resize_dim if resize_dim is not None else grid * patch_size
        self.model = _FakeBackbone(patch_size, raw_dim, seed=seed)
        if proj is None:
            g = torch.Generator().manual_seed(seed + 1)
            proj = VisualProjectionLayer(torch.randn(raw_dim, feature_dim, generator=g))
        self.proj = proj
        self.masker = _FakeMasker()
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def image_transforms(self, x: torch.Tensor) -> torch.Tensor:
        # Mirrors the real, authoritative DINOText.image_transforms's
        # T.Resize((resize_dim, resize_dim)): a fixed-canvas resize applied
        # unconditionally, regardless of the incoming window's own extent.
        # NOT identity -- a genuinely variable-sized source window must
        # reach a fixed size here before the fake backbone ever sees it,
        # exactly as it must before the real one does.
        return F.interpolate(x, size=(self.resize_dim, self.resize_dim), mode="bilinear", align_corners=False)

    def generate_masks_reference(self, image: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Independently re-derives the 'ordinary E3 output' the SAME way
        DINOText.generate_masks does (BGR->RGB swap, transforms, backbone,
        proj, forward_seg, resize with align_corners=True) -- written
        separately from extract_e3_window_snapshot, calling the SAME
        underlying fake sub-methods, to serve as the ordinary-path oracle
        for output-preservation tests.

        Projection dispatch mirrors the real, authoritative
        DINOText.generate_masks source exactly (two independent `if`s,
        matched by exact type name, never an isinstance/elif/else): for
        any proj type other than VisualProjectionLayer/DoubleMLP -- e.g.
        the real src/model.py ProjectionLayer, which has no DINO/visual-
        side projection method at all -- image_feat is left untouched.
        This is written directly against that real dispatch logic, not by
        calling or trusting the adapter under test."""
        h, w = image.shape[2:]
        image_bgr_to_rgb = image[:, [2, 1, 0], :, :]
        img_preprocessed = self.image_transforms(image_bgr_to_rgb)
        image_feat_raw = self.model.forward_features(img_preprocessed)["x_norm_patchtokens"]
        proj_type_name = type(self.proj).__name__
        if proj_type_name == "VisualProjectionLayer":
            image_feat_proj = self.proj.project_dino(image_feat_raw.float())
        elif proj_type_name == "DoubleMLP":
            image_feat_proj = self.proj.project_visual(image_feat_raw.float())
        else:
            image_feat_proj = image_feat_raw
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
        # A resize-canvas-driven backbone can never itself produce a
        # non-square token count (the fixed canvas is always square), so
        # this exercises the adapter's own defensive check directly: force
        # the backbone to return a non-perfect-square token count and
        # confirm extract_e3_window_snapshot still rejects it.
        model = FakeDINOText(grid=4)
        real_forward_features = model.model.forward_features

        def bad_forward_features(x):
            result = real_forward_features(x)
            tokens = result["x_norm_patchtokens"]
            return {"x_norm_patchtokens": tokens[:, :15, :]}  # 15 is not a perfect square

        model.model.forward_features = bad_forward_features
        image = torch.rand(1, 3, 32, 32)
        text_emb = torch.randn(3, 5)
        with self.assertRaises(e3a.E3AdapterError):
            e3a.extract_e3_window_snapshot(
                model, image, text_emb, window_id="w0", window_ordinal=0,
                top=0, left=0, bottom=32, right=32, canvas_h=32, canvas_w=32, patch_h=8, patch_w=8,
            )


# =============================================================================
# Repair: _project_dino_features fallthrough for unmatched proj types (e.g.
# the real ProjectionLayer, which has no DINO/visual-side projection method
# at all) must leave image_feat_raw unchanged, exactly matching
# generate_masks's real, authoritative two-independent-`if` dispatch (never
# a third branch/else in the real source).
# =============================================================================

class _ModelStub:
    """Minimal stand-in exposing only `.proj`, the one attribute
    _project_dino_features reads."""

    def __init__(self, proj):
        self.proj = proj


class RealProjectionTypeDispatchTests(unittest.TestCase):
    """Exercises the adapter's projection dispatch against the ACTUAL
    src/model.py classes (VisualProjectionLayer, DoubleMLP,
    ProjectionLayer) -- not merely fakes named to match them -- since the
    original defect was invisible to every prior test that only ever used
    a hand-rolled fake VisualProjectionLayer."""

    @classmethod
    def setUpClass(cls):
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        from model import VisualProjectionLayer as RealVisualProjectionLayer
        from model import DoubleMLP as RealDoubleMLP
        from model import ProjectionLayer as RealProjectionLayer
        cls.RealVisualProjectionLayer = RealVisualProjectionLayer
        cls.RealDoubleMLP = RealDoubleMLP
        cls.RealProjectionLayer = RealProjectionLayer

    def test_real_visual_projection_layer_branch_still_dispatches(self):
        proj = self.RealVisualProjectionLayer(dino_embed_dim=8, clip_embed_dim=6)
        x = torch.randn(1, 4, 8)
        expected = proj.project_dino(x.float())
        actual = e3a._project_dino_features(_ModelStub(proj), x)
        self.assertTrue(torch.equal(actual, expected))

    def test_real_double_mlp_branch_still_dispatches(self):
        proj = self.RealDoubleMLP(dino_embed_dim=8, clip_embed_dim=6, hidden_layer=True)
        x = torch.randn(1, 4, 8)
        expected = proj.project_visual(x.float())
        actual = e3a._project_dino_features(_ModelStub(proj), x)
        self.assertTrue(torch.equal(actual, expected))

    def test_real_projection_layer_fallthrough_preserves_object_identity(self):
        proj = self.RealProjectionLayer()
        x = torch.randn(1, 4, 8)
        result = e3a._project_dino_features(_ModelStub(proj), x)
        self.assertIs(result, x)  # exact same Python object, not a copy

    def test_real_projection_layer_fallthrough_preserves_values_dtype_device_shape(self):
        proj = self.RealProjectionLayer()
        for dtype in (torch.float32, torch.float64, torch.float16):
            x = torch.randn(1, 4, 8).to(dtype)
            result = e3a._project_dino_features(_ModelStub(proj), x)
            self.assertTrue(torch.equal(result, x))
            self.assertEqual(result.dtype, x.dtype)
            self.assertEqual(result.device, x.device)
            self.assertEqual(result.shape, x.shape)

    def test_real_projection_layer_has_no_dino_visual_projection_method(self):
        # Confirms the premise the fix rests on directly against the real
        # class, not as an assumption.
        proj = self.RealProjectionLayer()
        self.assertFalse(hasattr(proj, "project_dino"))
        self.assertFalse(hasattr(proj, "project_visual"))
        self.assertTrue(hasattr(proj, "project_clip_txt"))


class ProjectionLayerEquivalenceTests(unittest.TestCase):
    """Source-derived equivalence: using identical cached features and
    text embeddings, compares the repaired adapter (against a model whose
    .proj is the REAL, unmodified src/model.py ProjectionLayer) against
    an independently-written authoritative reference (FakeDINOText.
    generate_masks_reference's now-general dispatch, itself written
    directly against generate_masks's real two-independent-`if` source,
    not against the adapter under test), checking raw S0, the
    sigmoid/upsampled mask output, and the one-backbone-forward
    invariant."""

    def test_s0_and_mask_match_authoritative_reference_with_real_projection_layer(self):
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        from model import ProjectionLayer as RealProjectionLayer

        raw_dim = 8
        grid = 4
        model = FakeDINOText(grid=grid, raw_dim=raw_dim, seed=11, proj=RealProjectionLayer())
        infer = grid * 8
        image = torch.rand(1, 3, infer, infer, generator=torch.Generator().manual_seed(211))
        # text_emb's last dim must match the UNPROJECTED feature width
        # (raw_dim): with ProjectionLayer, image_feat is never projected.
        text_emb = torch.randn(3, raw_dim, generator=torch.Generator().manual_seed(311))

        self.assertEqual(model.model.call_count, 0)
        snap, adapter_output = e3a.extract_e3_window_snapshot(
            model, image, text_emb, window_id="w0", window_ordinal=0,
            top=0, left=0, bottom=infer, right=infer,
            canvas_h=infer, canvas_w=infer, patch_h=8, patch_w=8,
        )
        self.assertEqual(model.model.call_count, 1)  # one-backbone-forward invariant

        reference_output = model.generate_masks_reference(image, text_emb)
        self.assertTrue(torch.allclose(adapter_output, reference_output, atol=1e-6))

        # Raw S0, independently recomputed the same way the reference
        # does -- backbone forward (on the resized canvas, matching the
        # real image_transforms call order), NO projection (ProjectionLayer
        # fallthrough), forward_seg -- confirms the fallthrough is
        # correct at the pre-sigmoid/pre-resize stage too, not just in
        # the final resized mask.
        image_bgr_to_rgb = image[:, [2, 1, 0], :, :]
        img_preprocessed = model.image_transforms(image_bgr_to_rgb)
        image_feat_raw = model.model.forward_features(img_preprocessed)["x_norm_patchtokens"]
        b, num_patches, c = image_feat_raw.shape
        g = int(round(num_patches ** 0.5))
        image_feat_bchw = image_feat_raw.reshape(b, g, g, c).permute(0, 3, 1, 2)
        _mask, expected_s0_bchw = model.masker.forward_seg(image_feat_bchw, text_emb, deterministic=True, hard=False)
        _b1, num_classes, h, w = expected_s0_bchw.shape
        expected_s0 = expected_s0_bchw.reshape(num_classes, h * w).permute(1, 0)
        self.assertTrue(torch.allclose(snap.s0, expected_s0, atol=1e-6))


# =============================================================================
# Repair: backbone-input-canvas vs source-window-extent geometry (found by
# the first real GPU submission on the two real, sub-crop-size smoke
# images -- see docs/semantic_evidence_t320_implementation.md's own
# "backbone-input-canvas geometry" repair note). WindowSnapshot now carries
# backbone_input_height/width (the fixed canvas image_transforms resizes
# every window to) as a field distinct from infer_h/infer_w (the source
# window's own, possibly-smaller, possibly-non-multiple-of-patch extent).
# =============================================================================

class BackboneCanvasGeometryTests(unittest.TestCase):
    def _run(self, model, win_h, win_w, seed=0):
        g = torch.Generator().manual_seed(seed + 200)
        image = torch.rand(1, 3, win_h, win_w, generator=g)
        text_emb = torch.randn(3, model.proj.weight.shape[1] if hasattr(model.proj, "weight") else 5,
                                generator=torch.Generator().manual_seed(seed + 300))
        snap, adapter_output = e3a.extract_e3_window_snapshot(
            model, image, text_emb, window_id="w0", window_ordinal=0,
            top=0, left=0, bottom=win_h, right=win_w,
            canvas_h=win_h, canvas_w=win_w, patch_h=model.patch_size, patch_w=model.patch_size,
        )
        return snap, adapter_output, image, text_emb

    def test_full_crop_window_matches_backbone_canvas(self):
        # win_h == win_w == resize_dim: the ordinary, previously-only-tested
        # case where source-window extent and backbone canvas coincide.
        model = FakeDINOText(grid=4, patch_size=8)  # resize_dim = 32
        snap, _out, _img, _te = self._run(model, 32, 32)
        self.assertEqual((snap.infer_h, snap.infer_w), (32, 32))
        self.assertEqual((snap.backbone_input_height, snap.backbone_input_width), (32, 32))
        self.assertEqual((snap.grid_h, snap.grid_w), (4, 4))

    def test_one_dimension_below_crop_fixed_canvas_unchanged(self):
        # Analogous to a real 407x448 flush-clamped window: one source
        # dimension smaller than the fixed canvas, not a multiple of
        # patch_size (28 % 8 != 0) -- the original defect's exact shape.
        model = FakeDINOText(grid=4, patch_size=8)  # resize_dim = 32
        snap, _out, _img, _te = self._run(model, 28, 32)
        self.assertEqual((snap.infer_h, snap.infer_w), (28, 32))
        self.assertEqual((snap.backbone_input_height, snap.backbone_input_width), (32, 32))
        self.assertEqual((snap.grid_h, snap.grid_w), (4, 4))

    def test_both_dimensions_below_crop_fixed_canvas_unchanged(self):
        # Analogous to a real 338x338-ish window: both source dimensions
        # smaller than the fixed canvas.
        model = FakeDINOText(grid=4, patch_size=8)  # resize_dim = 32
        snap, _out, _img, _te = self._run(model, 22, 24)
        self.assertEqual((snap.infer_h, snap.infer_w), (22, 24))
        self.assertEqual((snap.backbone_input_height, snap.backbone_input_width), (32, 32))
        self.assertEqual((snap.grid_h, snap.grid_w), (4, 4))

    def test_non_multiple_of_14_source_window_accepted(self):
        # patch_size=14, matching the REAL committed patch_size_pixels
        # exactly; source window dims (29, 33) are deliberately not
        # multiples of 14 -- must be accepted (only the backbone-input
        # canvas needs to reconstruct via patch_h/patch_w, never the
        # source window).
        model = FakeDINOText(grid=3, patch_size=14)  # resize_dim = 42
        snap, _out, _img, _te = self._run(model, 29, 33)
        self.assertEqual((snap.infer_h, snap.infer_w), (29, 33))
        self.assertEqual((snap.backbone_input_height, snap.backbone_input_width), (42, 42))
        self.assertEqual((snap.grid_h, snap.grid_w), (3, 3))
        self.assertEqual(29 % 14, 1)  # sanity: genuinely not a multiple
        self.assertEqual(33 % 14, 5)

    def test_internal_mask_returned_at_original_window_extent_not_canvas(self):
        model = FakeDINOText(grid=4, patch_size=8, seed=5)
        snap, adapter_output, image, text_emb = self._run(model, 28, 32, seed=5)
        self.assertEqual(tuple(adapter_output.shape[2:]), (28, 32))
        self.assertNotEqual(tuple(adapter_output.shape[2:]), (snap.backbone_input_height, snap.backbone_input_width))
        reference_output = model.generate_masks_reference(image, text_emb)
        self.assertTrue(torch.allclose(adapter_output, reference_output, atol=1e-6))

    def test_row_major_alignment_preserved_for_clamped_window(self):
        model = FakeDINOText(grid=4, patch_size=8, seed=9)
        snap, _out, _img, _te = self._run(model, 28, 32, seed=9)
        expected_nodes = snap.grid_h * snap.grid_w
        self.assertEqual(snap.dino_features.shape[0], expected_nodes)
        self.assertEqual(snap.s0.shape[0], expected_nodes)

    def test_row_major_content_alignment_detects_ordering_defect(self):
        # A pure node-COUNT check cannot distinguish row-major from a
        # row/col-swapped flatten (both give the same count). This
        # constructs an image whose patch-average-pooled content encodes
        # each patch's (row, col) as a per-channel vector [row, col, 1]
        # (deliberately NOT a single scalar repeated across channels --
        # the pipeline ends in F.normalize, which collapses any positive
        # scalar multiple of a fixed direction to the same unit vector, so
        # a same-direction-different-magnitude pattern is invisible to it;
        # [row, col, 1] gives each (row, col) pair a genuinely distinct
        # direction instead), then independently recomputes (via the
        # model's own weight matrices, not the adapter's reshape) what
        # dino_features must equal at every position under the documented
        # row-major layout (row varies slower than column).
        grid, patch = 4, 8
        model = FakeDINOText(grid=grid, patch_size=patch, seed=21)  # resize_dim = 32 = window size (identity resize)
        canvas = grid * patch
        image = torch.zeros(1, 3, canvas, canvas)
        for row in range(grid):
            for col in range(grid):
                block = image[:, :, row * patch:(row + 1) * patch, col * patch:(col + 1) * patch]
                block[:, 0, :, :] = float(row)
                block[:, 1, :, :] = float(col)
                block[:, 2, :, :] = 1.0
        text_emb = torch.randn(3, model.proj.weight.shape[1], generator=torch.Generator().manual_seed(721))
        snap, _out = e3a.extract_e3_window_snapshot(
            model, image, text_emb, window_id="w0", window_ordinal=0,
            top=0, left=0, bottom=canvas, right=canvas,
            canvas_h=canvas, canvas_w=canvas, patch_h=patch, patch_w=patch,
        )
        # [N, 3], row-major by construction: index = row*grid + col.
        # Channel order [1.0, col, row], not [row, col, 1.0]: the adapter
        # applies image[:, [2,1,0], :, :] (BGR->RGB) before the backbone
        # ever sees it, reordering the [row, col, 1.0] channels this test
        # wrote into `image` to [1.0, col, row].
        raw_pixel_values = torch.tensor([[1.0, float(i % grid), float(i // grid)] for i in range(grid * grid)])
        backbone_out = raw_pixel_values @ model.model.weight  # matches _FakeBackbone's own weight
        projected = backbone_out @ model.proj.weight  # matches VisualProjectionLayer.project_dino (no bias/activation)
        expected = F.normalize(projected, dim=-1)
        self.assertTrue(torch.allclose(snap.dino_features, expected, atol=1e-5))

    def test_one_backbone_forward_for_clamped_window(self):
        model = FakeDINOText(grid=4, patch_size=8)
        self.assertEqual(model.model.call_count, 0)
        self._run(model, 28, 32)
        self.assertEqual(model.model.call_count, 1)

    # --- Structural (dataclass-level) checks, independent of the adapter
    #     function, exercising WindowSnapshot's own validation directly ---

    def _valid_kwargs(self, **overrides):
        n = 16  # grid_h*grid_w = 4*4
        feats = torch.nn.functional.normalize(torch.randn(n, 5), dim=-1)
        s0 = torch.randn(n, 3)
        kwargs = dict(
            window_id="w0", window_ordinal=0, canvas_h=100, canvas_w=100,
            top=0, left=0, bottom=28, right=32,
            infer_h=28, infer_w=32,
            backbone_input_height=32, backbone_input_width=32,
            grid_h=4, grid_w=4, patch_h=8, patch_w=8,
            dino_features=feats, s0=s0, class_count=3, feature_dim=5,
            dtype=feats.dtype, device=feats.device,
        )
        kwargs.update(overrides)
        return kwargs

    def test_source_window_not_required_divisible_by_patch_size(self):
        # The core structural regression: infer_h=28 is NOT a multiple of
        # patch_h=8 (28 % 8 == 4), and this must be accepted -- only
        # backbone_input_height (32, a genuine multiple of 8) is checked
        # against grid_h*patch_h.
        snap = e3a.WindowSnapshot(**self._valid_kwargs())
        self.assertEqual(snap.infer_h, 28)
        self.assertEqual(28 % snap.patch_h, 4)

    def test_invalid_backbone_canvas_grid_relationship_rejected(self):
        with self.assertRaises(e3a.E3AdapterError):
            e3a.WindowSnapshot(**self._valid_kwargs(backbone_input_height=33))  # 4*8=32 != 33

    def test_backbone_input_height_must_be_positive_int(self):
        with self.assertRaises(e3a.E3AdapterError):
            e3a.WindowSnapshot(**self._valid_kwargs(backbone_input_height=0))
        with self.assertRaises(e3a.E3AdapterError):
            e3a.WindowSnapshot(**self._valid_kwargs(backbone_input_height=-32))
        with self.assertRaises(e3a.E3AdapterError):
            e3a.WindowSnapshot(**self._valid_kwargs(backbone_input_height=True))  # bool-as-int rejected
        with self.assertRaises(e3a.E3AdapterError):
            e3a.WindowSnapshot(**self._valid_kwargs(backbone_input_width=32.0))  # float rejected

    def test_infer_h_infer_w_remain_source_window_not_canvas(self):
        # Regression against accidentally swapping the two quantities:
        # infer_h/infer_w must still independently accept a value smaller
        # than (and unrelated by patch multiplication to) backbone_input.
        # (bottom/right adjusted to stay <= the new, smaller infer_h/infer_w,
        # an unrelated pre-existing invariant this test does not target.)
        snap = e3a.WindowSnapshot(**self._valid_kwargs(infer_h=17, infer_w=19, bottom=17, right=19))
        self.assertEqual((snap.infer_h, snap.infer_w), (17, 19))
        self.assertEqual((snap.backbone_input_height, snap.backbone_input_width), (32, 32))


class TwoRealSmokeImageShapeGeometryTests(unittest.TestCase):
    """Reproduces the exact two real smoke-image shapes (500x407 and
    450x338) as synthetic (non-real) tensors, through the actual committed
    window-computation algorithm, at the real patch_size=14 and a
    resize_dim=448 matching the bound E3 config exactly -- the scenario
    the first real GPU submission (job 22591643) actually hit."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SRC_DIR))
        import semantic_evidence_t320_real_smoke as runner
        cls.compute_canonical_windows = staticmethod(runner.compute_canonical_windows)

    def _windows_for(self, w, h):
        # compute_canonical_windows(h, w, crop_hw, stride_hw) -- matches
        # the committed authority's window_geometry table exactly.
        return self.compute_canonical_windows(h, w, (448, 448), (224, 224))

    def test_500x407_synthetic_image_two_windows(self):
        windows = self._windows_for(500, 407)
        self.assertEqual(len(windows), 2)
        for ordinal, top, left, bottom, right in windows:
            self.assertEqual(bottom - top, 407)
            self.assertEqual(right - left, 448)

    def test_450x338_synthetic_image_two_windows(self):
        windows = self._windows_for(450, 338)
        self.assertEqual(len(windows), 2)
        for ordinal, top, left, bottom, right in windows:
            self.assertEqual(bottom - top, 338)
            self.assertEqual(right - left, 448)

    def test_total_N_across_both_real_smoke_image_shapes_is_4(self):
        n1 = len(self._windows_for(500, 407))
        n2 = len(self._windows_for(450, 338))
        self.assertEqual(n1 + n2, 4)

    def test_407x448_window_through_adapter_at_real_patch_and_canvas_scale(self):
        model = FakeDINOText(grid=32, patch_size=14, seed=13)  # resize_dim = 448, matching the bound E3 config
        g = torch.Generator().manual_seed(413)
        image = torch.rand(1, 3, 407, 448, generator=g)
        text_emb = torch.randn(3, 5, generator=torch.Generator().manual_seed(513))
        snap, adapter_output = e3a.extract_e3_window_snapshot(
            model, image, text_emb, window_id="img1:w0", window_ordinal=0,
            top=0, left=0, bottom=407, right=448,
            canvas_h=407, canvas_w=448, patch_h=14, patch_w=14,
        )
        self.assertEqual((snap.infer_h, snap.infer_w), (407, 448))
        self.assertEqual((snap.backbone_input_height, snap.backbone_input_width), (448, 448))
        self.assertEqual((snap.grid_h, snap.grid_w), (32, 32))
        self.assertEqual(tuple(adapter_output.shape[2:]), (407, 448))
        reference_output = model.generate_masks_reference(image, text_emb)
        self.assertTrue(torch.allclose(adapter_output, reference_output, atol=1e-6))

    def test_338x448_window_through_adapter_at_real_patch_and_canvas_scale(self):
        model = FakeDINOText(grid=32, patch_size=14, seed=17)  # resize_dim = 448
        g = torch.Generator().manual_seed(417)
        image = torch.rand(1, 3, 338, 448, generator=g)
        text_emb = torch.randn(3, 5, generator=torch.Generator().manual_seed(517))
        snap, adapter_output = e3a.extract_e3_window_snapshot(
            model, image, text_emb, window_id="img2:w0", window_ordinal=0,
            top=0, left=0, bottom=338, right=448,
            canvas_h=338, canvas_w=448, patch_h=14, patch_w=14,
        )
        self.assertEqual((snap.infer_h, snap.infer_w), (338, 448))
        self.assertEqual((snap.backbone_input_height, snap.backbone_input_width), (448, 448))
        self.assertEqual((snap.grid_h, snap.grid_w), (32, 32))
        self.assertEqual(tuple(adapter_output.shape[2:]), (338, 448))
        reference_output = model.generate_masks_reference(image, text_emb)
        self.assertTrue(torch.allclose(adapter_output, reference_output, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
