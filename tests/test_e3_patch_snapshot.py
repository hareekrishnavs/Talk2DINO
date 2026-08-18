import dataclasses
import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).parents[1]
MODULE_PATH = (
    ROOT
    / "src/open_vocabulary_segmentation/models/dinotext/dinotext.py"
)


def _load_dinotext_module():
    class _Registry:
        @staticmethod
        def register_module():
            return lambda value: value

    models = types.ModuleType("models")
    models.__path__ = []
    dinotext_package = types.ModuleType("models.dinotext")
    dinotext_package.__path__ = []
    builder = types.ModuleType("models.builder")
    builder.MODELS = _Registry()
    pamr = types.ModuleType("models.dinotext.pamr")
    pamr.PAMR = nn.Identity
    masker = types.ModuleType("models.dinotext.masker")
    masker.DINOTextMasker = nn.Identity
    us = types.ModuleType("us")
    us.normalize = lambda value, dim, eps=1e-6: F.normalize(
        value, dim=dim, eps=eps
    )
    datasets = types.ModuleType("datasets")
    datasets.get_template = lambda _name: []

    stubs = {
        "models": models,
        "models.builder": builder,
        "models.dinotext": dinotext_package,
        "models.dinotext.pamr": pamr,
        "models.dinotext.masker": masker,
        "us": us,
        "datasets": datasets,
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "e3_patch_snapshot_dinotext", MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


dinotext_module = _load_dinotext_module()
DINOText = dinotext_module.DINOText
E3PatchSnapshot = dinotext_module.E3PatchSnapshot
ProjectionLayer = dinotext_module.ProjectionLayer
VisualProjectionLayer = dinotext_module.VisualProjectionLayer
_build_e3_patch_snapshot = dinotext_module._build_e3_patch_snapshot


class _FakeBackbone(nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.register_buffer("tokens", tokens.clone())
        self.forward_count = 0

    def forward_features(self, _images):
        self.forward_count += 1
        return {"x_norm_patchtokens": self.tokens}


class _FakeMasker(nn.Module):
    @torch.no_grad()
    def raw_similarity(self, image_feat, text_emb):
        normalized = F.normalize(image_feat, dim=1, eps=1e-6)
        return torch.einsum("bchw,nc->bnhw", normalized, text_emb)

    @torch.no_grad()
    def mask_from_similarity(self, simmap, deterministic=True, hard=False):
        del deterministic, hard
        return torch.sigmoid(simmap)

    @torch.no_grad()
    def forward_seg(self, image_feat, text_emb, deterministic=True, hard=False):
        simmap = self.raw_similarity(image_feat, text_emb)
        return self.mask_from_similarity(simmap, deterministic, hard), simmap


def _fake_attention(
    self,
    output,
    batch_size,
    num_tokens,
    num_attn_heads,
    embed_dim,
    scale,
    num_global_tokens,
    ret_self_attn_maps=False,
):
    del self, output, num_tokens, embed_dim, scale, num_global_tokens
    attention = torch.zeros(batch_size, 4)
    maps = torch.zeros(batch_size, num_attn_heads, 4)
    return (attention, maps) if ret_self_attn_maps else attention


def _make_model(tokens, visual_projection=False):
    model = DINOText.__new__(DINOText)
    nn.Module.__init__(model)
    model.model_name = "dinov2_vitb14_reg"
    model.model = _FakeBackbone(tokens)
    model.image_transforms = lambda image: image
    if visual_projection:
        model.proj = VisualProjectionLayer(
            act=None,
            hidden_layer=False,
            dino_embed_dim=tokens.shape[-1],
            clip_embed_dim=2,
        )
    else:
        model.proj = ProjectionLayer(
            act=None,
            hidden_layer=True,
            dino_embed_dim=tokens.shape[-1],
            clip_embed_dim=tokens.shape[-1],
        )
    model.masker = _FakeMasker()
    model.feats = {"self_attn": torch.empty(0)}
    model.num_global_tokens = 5
    model.num_attn_heads = 2
    model.scale = 0.125
    model.with_bg_clean = False
    model.pamr = None
    model.process_self_attention = types.MethodType(_fake_attention, model)
    return model


def _inputs(dtype=torch.float32):
    tokens = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [2.0, -1.0, 0.5],
                [-1.0, 3.0, 2.0],
                [0.5, 0.25, -2.0],
            ],
            [
                [3.0, 1.0, -1.0],
                [1.0, 4.0, 2.0],
                [2.0, -3.0, 1.0],
                [-2.0, 0.5, 3.0],
            ],
        ],
        dtype=dtype,
    )
    image = torch.arange(2 * 3 * 4 * 4, dtype=dtype).reshape(2, 3, 4, 4)
    text = F.normalize(
        torch.tensor([[1.0, -2.0, 0.5], [-1.0, 1.0, 2.0]], dtype=dtype),
        dim=-1,
    )
    return tokens, image, text


def test_existing_generate_masks_signature_and_return_arity_are_unchanged():
    signature = inspect.signature(DINOText.generate_masks)
    assert str(signature) == (
        "(self, image, text_emb, apply_pamr=False, lambda_bg=0.2)"
    )
    tokens, image, text = _inputs()
    result = _make_model(tokens).generate_masks(image, text)
    assert type(result) is tuple
    assert len(result) == 2


def test_default_path_does_not_invoke_snapshot_builder(monkeypatch):
    tokens, image, text = _inputs()
    model = _make_model(tokens)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("default E3 path invoked snapshot construction")

    monkeypatch.setattr(dinotext_module, "_build_e3_patch_snapshot", forbidden)
    model.generate_masks(image, text)
    assert model.model.forward_count == 1


def test_opt_in_uses_one_forward_and_is_bitwise_identical_to_default():
    tokens, image, text = _inputs()
    model = _make_model(tokens)
    default_masks, default_simmap = model.generate_masks(image, text)
    masks, simmap, snapshot = model.generate_masks_with_patch_snapshot(image, text)

    assert model.model.forward_count == 2
    assert torch.equal(masks, default_masks)
    assert torch.equal(simmap, default_simmap)
    assert isinstance(snapshot, E3PatchSnapshot)


def test_snapshot_only_path_stops_before_sigmoid_interpolation_and_mask(monkeypatch):
    tokens, image, text = _inputs()
    model = _make_model(tokens)
    monkeypatch.setattr(
        model.masker,
        "mask_from_similarity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("snapshot-only path constructed a mask")
        ),
    )
    monkeypatch.setattr(
        dinotext_module.F,
        "interpolate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("snapshot-only path interpolated")
        ),
    )
    snapshot = model.generate_patch_snapshot(image, text)
    assert isinstance(snapshot, E3PatchSnapshot)
    assert snapshot.grid_hw == (2, 2)
    assert model.model.forward_count == 1


def test_shared_downstream_matches_existing_e3_transform_exactly():
    tokens, image, text = _inputs()
    model = _make_model(tokens)
    masks, simmap = model.generate_masks(image, text)
    scores = simmap.permute(0, 2, 3, 1).reshape(2, 4, 2)
    shared = model.masks_from_patch_scores(scores, (2, 2), (4, 4))
    assert torch.equal(shared, masks)


def test_snapshot_layout_values_and_processing_stage_are_exact():
    tokens, image, text = _inputs()
    masks, simmap, snapshot = _make_model(tokens).generate_masks_with_patch_snapshot(
        image, text
    )
    expected = simmap.permute(0, 2, 3, 1).reshape(2, 4, 2)

    assert snapshot.unary_scores.shape == (2, 4, 2)
    assert snapshot.grid_hw == (2, 2)
    assert torch.equal(snapshot.unary_scores, expected)
    assert torch.equal(
        masks,
        F.interpolate(
            torch.sigmoid(simmap),
            (4, 4),
            mode="bilinear",
            align_corners=True,
        ),
    )
    assert simmap.shape[-2:] == (2, 2)
    assert masks.shape[-2:] == (4, 4)
    assert not torch.equal(snapshot.unary_scores, torch.sigmoid(expected))


def test_features_are_normalized_raw_backbone_tokens_and_reconstruct_e3_scores():
    tokens, image, text = _inputs()
    _masks, _simmap, snapshot = _make_model(
        tokens
    ).generate_masks_with_patch_snapshot(image, text)
    expected_features = F.normalize(tokens, dim=-1, eps=1e-6)
    reconstructed = torch.einsum("bnd,cd->bnc", snapshot.dino_features, text)

    assert snapshot.dino_features.shape == (2, 4, 3)
    torch.testing.assert_close(snapshot.dino_features, expected_features)
    torch.testing.assert_close(snapshot.dino_features.norm(dim=-1), torch.ones(2, 4))
    torch.testing.assert_close(reconstructed, snapshot.unary_scores)


def test_features_are_captured_before_visual_projection():
    tokens, image, _text = _inputs()
    model = _make_model(tokens, visual_projection=True)
    text = F.normalize(torch.tensor([[1.0, 2.0], [-2.0, 1.0]]), dim=-1)
    with torch.no_grad():
        model.proj.linear_layer.weight.copy_(
            torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, -3.0]])
        )
        model.proj.linear_layer.bias.copy_(torch.tensor([0.5, -1.0]))

    _masks, _simmap, snapshot = model.generate_masks_with_patch_snapshot(image, text)

    assert snapshot.dino_features.shape[-1] == 3
    torch.testing.assert_close(
        snapshot.dino_features, F.normalize(tokens, dim=-1, eps=1e-6)
    )


def test_snapshot_container_is_frozen_but_tensor_contents_remain_mutable():
    tokens, image, text = _inputs()
    snapshot = _make_model(tokens).generate_masks_with_patch_snapshot(image, text)[2]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.grid_hw = (1, 4)
    snapshot.unary_scores[0, 0, 0] = 123.0
    assert snapshot.unary_scores[0, 0, 0] == 123.0


def test_snapshot_storage_is_independent_in_both_directions():
    tokens, image, text = _inputs()
    model = _make_model(tokens)
    masks, simmap, first = model.generate_masks_with_patch_snapshot(image, text)
    _masks, _simmap, second = model.generate_masks_with_patch_snapshot(image, text)
    saved_masks = masks.clone()
    saved_simmap = simmap.clone()
    saved_second_unary = second.unary_scores.clone()
    saved_first_features = first.dino_features.clone()

    assert (
        first.unary_scores.untyped_storage().data_ptr()
        != simmap.untyped_storage().data_ptr()
    )
    assert (
        first.dino_features.untyped_storage().data_ptr()
        != model.model.tokens.untyped_storage().data_ptr()
    )
    assert (
        first.unary_scores.untyped_storage().data_ptr()
        != second.unary_scores.untyped_storage().data_ptr()
    )
    assert (
        first.dino_features.untyped_storage().data_ptr()
        != second.dino_features.untyped_storage().data_ptr()
    )

    model.model.tokens.add_(1000)
    assert torch.equal(first.dino_features, saved_first_features)
    first.unary_scores.zero_()
    first.dino_features.zero_()
    assert torch.equal(masks, saved_masks)
    assert torch.equal(simmap, saved_simmap)
    assert torch.equal(second.unary_scores, saved_second_unary)


def test_snapshot_tensors_are_detached_contiguous_and_preserve_dtype_device():
    tokens, image, text = _inputs(dtype=torch.float64)
    snapshot = _make_model(tokens).generate_masks_with_patch_snapshot(image, text)[2]
    for tensor in (snapshot.unary_scores, snapshot.dino_features):
        assert tensor.dtype == torch.float64
        assert tensor.device.type == "cpu"
        assert tensor.is_contiguous()
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None


def test_snapshot_helper_detaches_inputs_with_gradient_history():
    features = torch.randn(1, 4, 3, requires_grad=True) * 2
    scores = torch.randn(1, 2, 2, 2, requires_grad=True) * 3
    snapshot = _build_e3_patch_snapshot(scores, features, (2, 2))
    assert snapshot.unary_scores.requires_grad is False
    assert snapshot.unary_scores.grad_fn is None
    assert snapshot.dino_features.requires_grad is False
    assert snapshot.dino_features.grad_fn is None


@pytest.mark.parametrize(
    ("scores", "features", "grid", "message"),
    [
        (torch.zeros(1, 2, 4), torch.zeros(1, 4, 3), (2, 2), "scores"),
        (torch.zeros(1, 2, 2, 2), torch.zeros(1, 12), (2, 2), "features"),
        (torch.zeros(2, 2, 2, 2), torch.zeros(1, 4, 3), (2, 2), "batch-size"),
        (torch.zeros(1, 2, 2, 2), torch.zeros(1, 3, 3), (2, 2), "patch-count"),
        (torch.zeros(1, 2, 2, 2), torch.zeros(1, 4, 3), (1, 4), "grid metadata"),
        (torch.zeros(1, 2, 0, 2), torch.zeros(1, 0, 3), (0, 2), "non-empty"),
    ],
)
def test_invalid_snapshot_shapes_fail(scores, features, grid, message):
    with pytest.raises(ValueError, match=message):
        _build_e3_patch_snapshot(scores, features, grid)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["scores", "features"])
def test_nonfinite_snapshot_inputs_fail(value, field):
    scores = torch.zeros(1, 2, 2, 2)
    features = torch.ones(1, 4, 3)
    if field == "scores":
        scores[0, 0, 0, 0] = value
    else:
        features[0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        _build_e3_patch_snapshot(scores, features, (2, 2))


def test_zero_norm_features_fail_clearly():
    with pytest.raises(ValueError, match="non-zero L2 norm"):
        _build_e3_patch_snapshot(
            torch.zeros(1, 2, 2, 2),
            torch.zeros(1, 4, 3),
            (2, 2),
        )


def test_non_square_patch_count_fails_clearly_on_opt_in_path():
    tokens, image, text = _inputs()
    model = _make_model(tokens[:, :3])
    with pytest.raises(ValueError, match="square patch grid"):
        model.generate_masks_with_patch_snapshot(image, text)


def test_default_and_snapshot_paths_do_not_change_model_state():
    tokens, image, text = _inputs()
    model = _make_model(tokens)
    original_keys = tuple(model.state_dict().keys())
    original_parameters = tuple(name for name, _value in model.named_parameters())

    model.generate_masks(image, text)
    model.generate_masks_with_patch_snapshot(image, text)

    assert tuple(model.state_dict().keys()) == original_keys
    assert tuple(name for name, _value in model.named_parameters()) == original_parameters
