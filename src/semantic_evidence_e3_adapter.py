"""Output-preserving, read-only E3 single-window extraction adapter.

Exposes the exact aligned DINO patch features and pre-sigmoid unary cosine
scores (S0) that the committed E3 model already computes internally during
one ordinary forward pass, without modifying any E3 scientific file and
without running the backbone a second time.

Source basis (current, committed branch -- NOT the historical e12 branch):
  src/open_vocabulary_segmentation/models/dinotext/dinotext.py
    DINOText.generate_masks (lines ~297-354 at the reviewed E3 base,
    commit 9f314dc973657719abb15cc33473dafc185444c4)
  src/open_vocabulary_segmentation/models/dinotext/masker.py
    DINOTextMasker.forward_seg, DINOTextSim2Mask.forward

`extract_e3_window_snapshot` composes exactly the same existing, committed
methods `generate_masks` itself calls internally -- `self.model.forward_features`,
`self.proj.project_dino`/`project_visual`, `self.masker.forward_seg` -- in
the same order, over the same single backbone call. `generate_masks`
computes these same intermediate tensors but does not return them; this
module orchestrates the identical sequence of existing calls to make them
visible, and additionally reconstructs the ordinary E3 mask output from the
exact same intermediate values, as an output-preservation check. Nothing in
`dinotext.py`/`masker.py` is imported for its source code or reimplemented
-- only its existing public methods are called, unchanged. The one
supporting tensor op not obtained via an existing method call is the L2
normalization `forward_seg` applies internally before its own cosine
similarity; this module recomputes it via `torch.nn.functional.normalize(t,
dim=1, eps=1e-6)`, the exact call the repository's own `us.normalize`
utility (src/open_vocabulary_segmentation/us/misc.py) wraps -- called
directly here only to avoid that package's unrelated, heavy `mmcv`/`cv2`
import chain, not as a reimplementation of any E3 formula.

Scope boundary: `model.with_bg_clean=True` post-processing
(`DINOText.similarity_assignment_weighted`) is out of scope for T0/T320
generation and is not replicated here; `extract_e3_window_snapshot` raises
`E3AdapterError` if the supplied model has `with_bg_clean=True`, rather than
silently producing an ordinary-output value that would not match
`generate_masks`. This module never accesses ground truth, annotations, or
any correctness label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


class E3AdapterError(Exception):
    """Raised for any structural, shape, dtype, device, or scope violation
    in the E3 extraction interface."""


_FATAL = (KeyboardInterrupt, SystemExit, MemoryError)


def _reraise_if_fatal(exc: BaseException) -> None:
    if isinstance(exc, _FATAL):
        raise exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise E3AdapterError(message)


def _require_finite(tensor: torch.Tensor, label: str) -> None:
    if not torch.isfinite(tensor).all():
        raise E3AdapterError(f"{label} contains non-finite values")


@dataclass(frozen=True)
class WindowSnapshot:
    """Everything T0/T320 generation needs from one E3 inference window,
    and nothing else -- no ground truth, annotations, correctness labels,
    or intervention output ever appears here.

    `dino_features` and `s0` share the same canonical row-major node order
    (row index varies slower than column index, matching the patch grid's
    natural [grid_h, grid_w] -> flatten(grid_h * grid_w) layout), so
    `dino_features[i]` and `s0[i]` describe the same patch for every i.
    """

    window_id: str
    window_ordinal: int
    canvas_h: int
    canvas_w: int
    top: int
    left: int
    bottom: int
    right: int
    infer_h: int
    infer_w: int
    grid_h: int
    grid_w: int
    patch_h: int
    patch_w: int
    dino_features: torch.Tensor  # [N, D], L2-normalized, float
    s0: torch.Tensor  # [N, C], pre-sigmoid cosine scores, float
    class_count: int
    feature_dim: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self) -> None:
        _require(isinstance(self.window_id, str) and self.window_id != "", "window_id must be a non-empty string")
        _require(isinstance(self.window_ordinal, int) and not isinstance(self.window_ordinal, bool) and self.window_ordinal >= 0,
                  "window_ordinal must be a non-negative int")
        for name in ("canvas_h", "canvas_w", "infer_h", "infer_w", "grid_h", "grid_w", "patch_h", "patch_w",
                     "class_count", "feature_dim"):
            value = getattr(self, name)
            _require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"{name} must be a positive int")
        for name in ("top", "left", "bottom", "right"):
            value = getattr(self, name)
            _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"{name} must be a non-negative int")
        _require(self.top < self.bottom, "top must be < bottom")
        _require(self.left < self.right, "left must be < right")
        _require(self.bottom <= self.canvas_h, "bottom must be <= canvas_h")
        _require(self.right <= self.canvas_w, "right must be <= canvas_w")
        _require((self.bottom - self.top) <= self.infer_h, "window height exceeds infer_h")
        _require((self.right - self.left) <= self.infer_w, "window width exceeds infer_w")
        _require(self.grid_h * self.patch_h == self.infer_h, "grid_h * patch_h must equal infer_h")
        _require(self.grid_w * self.patch_w == self.infer_w, "grid_w * patch_w must equal infer_w")

        _require(isinstance(self.dino_features, torch.Tensor), "dino_features must be a torch.Tensor")
        _require(isinstance(self.s0, torch.Tensor), "s0 must be a torch.Tensor")
        _require(self.dino_features.ndim == 2, f"dino_features must be rank 2, got {self.dino_features.ndim}")
        _require(self.s0.ndim == 2, f"s0 must be rank 2, got {self.s0.ndim}")

        expected_nodes = self.grid_h * self.grid_w
        _require(self.dino_features.shape[0] == expected_nodes,
                  f"dino_features node count {self.dino_features.shape[0]} != grid_h*grid_w {expected_nodes}")
        _require(self.s0.shape[0] == expected_nodes,
                  f"s0 node count {self.s0.shape[0]} != grid_h*grid_w {expected_nodes}")
        _require(self.dino_features.shape[0] == self.s0.shape[0],
                  "dino_features and s0 node order must match (same node count)")
        _require(self.dino_features.shape[1] == self.feature_dim,
                  f"dino_features feature dim {self.dino_features.shape[1]} != feature_dim {self.feature_dim}")
        _require(self.s0.shape[1] == self.class_count,
                  f"s0 class dim {self.s0.shape[1]} != class_count {self.class_count}")

        _require(self.dino_features.is_floating_point(), "dino_features must be floating point")
        _require(self.s0.is_floating_point(), "s0 must be floating point")
        _require(self.dino_features.dtype == self.dtype, "dino_features.dtype must match declared dtype")
        _require(self.s0.dtype == self.dtype, "s0.dtype must match declared dtype")
        _require(self.dino_features.device == self.device, "dino_features.device must match declared device")
        _require(self.s0.device == self.device, "s0.device must match declared device")

        _require_finite(self.dino_features, "dino_features")
        _require_finite(self.s0, "s0")


def _project_dino_features(model: Any, image_feat_raw: torch.Tensor) -> torch.Tensor:
    """Apply E3's existing projection, exactly as `generate_masks` does
    (lines ~331-334 at the reviewed E3 base):

        if type(self.proj) == VisualProjectionLayer:
            image_feat = self.proj.project_dino(image_feat.float())
        if type(self.proj) == DoubleMLP:
            image_feat = self.proj.project_visual(image_feat.float())

    These are two independent `if`s, not `if`/`elif`/`else` -- for any
    other `self.proj` type (e.g. `ProjectionLayer`, which projects the
    CLIP text side via `project_clip_txt` and has no DINO/visual-side
    projection method at all -- src/model.py's `ProjectionLayer`) neither
    branch matches and `image_feat` is left completely untouched by the
    real, authoritative source. This function mirrors that exactly: the
    fall-through returns `image_feat_raw` unchanged -- no clone, no
    dtype/device cast, no normalization -- never `.float()`, which the
    real source only applies inside the two matched branches, immediately
    before the call each one already makes."""
    proj_type_name = type(model.proj).__name__
    if proj_type_name == "VisualProjectionLayer":
        return model.proj.project_dino(image_feat_raw.float())
    if proj_type_name == "DoubleMLP":
        return model.proj.project_visual(image_feat_raw.float())
    return image_feat_raw


def extract_e3_window_snapshot(
    model: Any,
    image: torch.Tensor,
    text_emb: torch.Tensor,
    *,
    window_id: str,
    window_ordinal: int,
    top: int,
    left: int,
    bottom: int,
    right: int,
    canvas_h: int,
    canvas_w: int,
    patch_h: int,
    patch_w: int,
) -> tuple[WindowSnapshot, torch.Tensor]:
    """Run exactly one E3 backbone forward for this window and return
    (WindowSnapshot, ordinary_mask_output).

    `ordinary_mask_output` is the same tensor `DINOText.generate_masks`
    would return as `mask` for this exact `(image, text_emb)` pair, in the
    `with_bg_clean=False`, `apply_pamr=False` configuration (the configuration
    the committed Step 0 baseline uses: with_bg=False, pamr=False) --
    reconstructed from the same single-pass intermediate values, not by
    calling `generate_masks` a second time.

    Exactly one call to `model.model.forward_features` occurs. No checkpoint
    is loaded or reloaded here; `model` must already be constructed. No GT,
    annotation, or correctness-label argument exists on this function.

    Raises E3AdapterError if `model.with_bg_clean` is True (out of scope;
    see module docstring) or if any shape/dtype/finiteness invariant fails.
    """
    _require(image.ndim == 4 and image.shape[0] == 1, f"image must be [1, 3, H, W], got {tuple(image.shape)}")
    _require(getattr(model, "with_bg_clean", False) is False,
              "extract_e3_window_snapshot does not support with_bg_clean=True (out of scope for T0/T320 generation)")

    win_h, win_w = image.shape[2:]
    _require(win_h == (bottom - top) and win_w == (right - left),
              "image spatial size must equal (bottom-top, right-left)")

    # Exactly the same preprocessing generate_masks performs (BGR->RGB,
    # image_transforms), before the single backbone forward.
    image_bgr_to_rgb = image[:, [2, 1, 0], :, :]
    device = next(model.parameters()).device
    img_preprocessed = model.image_transforms(image_bgr_to_rgb).to(device)

    # Exactly one backbone forward, dispatched the same way generate_masks
    # dispatches on model_name -- no reimplementation of the backbone itself.
    model_name = model.model_name
    if "dinov2" in model_name:
        image_feat_raw = model.model.forward_features(img_preprocessed)["x_norm_patchtokens"]
    elif "dinov3" in model_name:
        image_feat_raw = model.model.forward_features(img_preprocessed)[:, 5:, :]
    elif "mae" in model_name or "clip" in model_name or "dino" in model_name:
        image_feat_raw = model.model.forward_features(img_preprocessed)[:, 1:, :]
    else:
        raise E3AdapterError(f"unsupported model_name {model_name!r} for single-forward extraction")

    batch_size, num_tokens, embed_dim = image_feat_raw.shape
    _require(batch_size == 1, "extract_e3_window_snapshot processes exactly one window at a time")

    image_feat_proj = _project_dino_features(model, image_feat_raw)
    b, num_patches, channels = image_feat_proj.shape
    grid_h = grid_w = int(round(num_patches ** 0.5))
    _require(grid_h * grid_w == num_patches, f"E3 patch boundary requires a square patch grid, got {num_patches} tokens")
    _require(grid_h * patch_h == win_h and grid_w * patch_w == win_w,
              "patch_h/patch_w * grid do not reconstruct the window size")

    image_feat_bchw = image_feat_proj.reshape(b, grid_h, grid_w, channels).permute(0, 3, 1, 2)

    # Existing E3 method, called unmodified: gives (mask, simmap) from ONE
    # call, where simmap is the pre-sigmoid S0 and mask == sigmoid(simmap)
    # (DINOTextSim2Mask.forward's soft_mask branch is plain torch.sigmoid,
    # unconditionally -- no gumbel noise on the soft path).
    mask_pre_resize, simmap = model.masker.forward_seg(image_feat_bchw, text_emb, deterministic=True, hard=False)

    # The L2-normalized features forward_seg used internally, recomputed via
    # F.normalize(t, dim=1, eps=1e-6) -- a cheap, deterministic, side-effect-
    # free tensor op, not a second inference pass. This is mathematically
    # and behaviorally identical to the repository's own `us.normalize`
    # utility (src/open_vocabulary_segmentation/us/misc.py:46-48), which is
    # a one-line wrapper: `return F.normalize(t, dim=dim, eps=eps)` with the
    # same default eps=1e-6. Called directly here, rather than importing the
    # `us` package, only to avoid that package's unrelated, heavy transitive
    # `mmcv`/`cv2` dependency (its `__init__.py` eagerly imports mmcv for
    # utilities this adapter never uses) -- not a reimplementation of any
    # E3 scientific formula, since both call the identical torch primitive
    # with identical arguments.
    dino_features_normalized_bchw = F.normalize(image_feat_bchw, dim=1, eps=1e-6)

    n_classes = simmap.shape[1]
    class_count = int(n_classes)
    feature_dim = int(channels)

    # Row-major node flatten: row (grid_h) varies slower than column (grid_w).
    dino_features = dino_features_normalized_bchw[0].permute(1, 2, 0).reshape(grid_h * grid_w, feature_dim).contiguous()
    s0 = simmap[0].permute(1, 2, 0).reshape(grid_h * grid_w, class_count).contiguous()

    dtype = dino_features.dtype
    snapshot = WindowSnapshot(
        window_id=window_id,
        window_ordinal=window_ordinal,
        canvas_h=canvas_h,
        canvas_w=canvas_w,
        top=top,
        left=left,
        bottom=bottom,
        right=right,
        infer_h=win_h,
        infer_w=win_w,
        grid_h=grid_h,
        grid_w=grid_w,
        patch_h=patch_h,
        patch_w=patch_w,
        dino_features=dino_features,
        s0=s0,
        class_count=class_count,
        feature_dim=feature_dim,
        dtype=dtype,
        device=dino_features.device,
    )

    # Ordinary E3 output: the exact resize step generate_masks applies
    # (mode='bilinear', align_corners=True), from the same mask_pre_resize.
    ordinary_mask_output = F.interpolate(mask_pre_resize, (win_h, win_w), mode="bilinear", align_corners=True)

    return snapshot, ordinary_mask_output
