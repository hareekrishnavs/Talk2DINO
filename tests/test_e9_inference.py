import hashlib
import inspect
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from src.e9_sparse_region_alignment import (
    SparseRegionAlignmentAdapter,
    SparseRegionAlignmentConfig,
)


def module():
    root = Path(__file__).parents[1] / "src/open_vocabulary_segmentation"
    sys.path.insert(0, str(root))
    try:
        from models.dinotext import masker
    finally:
        sys.path.pop(0)
    return masker


def adapter(gamma):
    return SparseRegionAlignmentAdapter(SparseRegionAlignmentConfig(
        embedding_dim=5, bottleneck_dim=3, dropout=0,
        residual_max=0.25, gamma_max=gamma, gate_hidden_dim=4,
    )).eval()


def values():
    generator = torch.Generator().manual_seed(12)
    image = torch.randn(2, 5, 3, 4, generator=generator)
    text = F.normalize(torch.randn(4, 5, generator=generator), dim=-1)
    prior = torch.softmax(torch.randn(2, 12, generator=generator), -1)
    return image, text, prior


def test_gamma_zero_survives_reshape_and_mask_as_bitwise_e3():
    masker = module().DINOTextMasker()
    image, text, prior = values()
    baseline_mask, baseline_score = masker.forward_seg(image, text)
    e9_mask, e9_score = masker.forward_seg_with_sparse_region_alignment(
        image, text, prior, adapter(0)
    )
    assert torch.equal(e9_score, baseline_score)
    assert torch.equal(e9_mask, baseline_mask)


def test_active_inference_has_one_normalization_base_and_sigmoid(monkeypatch):
    imported = module()
    masker = imported.DINOTextMasker()
    image, text, prior = values()
    original_normalize = imported.us.normalize
    original_functional_normalize = F.normalize
    original_einsum = torch.einsum
    original_sigmoid = torch.sigmoid
    counts = {
        "project_patch_normalize": 0,
        "functional_normalize": 0,
        "base": 0,
        "sigmoid": 0,
    }

    def normalize(value, *args, **kwargs):
        if value is image:
            counts["project_patch_normalize"] += 1
        return original_normalize(value, *args, **kwargs)

    def functional_normalize(value, *args, **kwargs):
        if value is image:
            pass
        else:
            counts["functional_normalize"] += 1
        return original_functional_normalize(value, *args, **kwargs)

    def einsum(equation, *args, **kwargs):
        if equation == "b c h w, n c -> b n h w":
            counts["base"] += 1
        return original_einsum(equation, *args, **kwargs)

    def sigmoid(value):
        counts["sigmoid"] += 1
        return original_sigmoid(value)

    monkeypatch.setattr(imported.us, "normalize", normalize)
    monkeypatch.setattr(F, "normalize", functional_normalize)
    monkeypatch.setattr(torch, "einsum", einsum)
    monkeypatch.setattr(torch, "sigmoid", sigmoid)
    mask, score = masker.forward_seg_with_sparse_region_alignment(
        image, text, prior, adapter(0.3)
    )
    assert mask.shape == score.shape == (2, 4, 3, 4)
    # Three explicit gates use sigmoid, plus exactly one final mask sigmoid.
    assert counts == {
        "project_patch_normalize": 1,
        "functional_normalize": 2,
        "base": 1,
        "sigmoid": 4,
    }
    source = inspect.getsource(masker.forward_seg_with_sparse_region_alignment)
    assert "training spatial" not in source.lower()
    assert "retriev" not in source.lower()


def test_e9_configuration_is_isolated_and_uses_external_artifact_binding():
    path = Path(
        "src/open_vocabulary_segmentation/configs/stuff/"
        "dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_"
        "sparse_region_alignment.yml"
    )
    config = yaml.safe_load(path.read_text())
    assert config["_base_"] == (
        "dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_tau010.yml"
    )
    assert config["model"] == {
        "sparse_region_alignment": {
            "adapter_path": "${oc.env:TALK2DINO_E9_ADAPTER}",
            "adapter_sha256": "${oc.env:TALK2DINO_E9_ADAPTER_SHA256}",
            "source_git_commit": "${oc.env:TALK2DINO_E9_SOURCE_GIT_COMMIT}",
            "dino_source_commit": "${oc.env:TALK2DINO_E9_DINO_SOURCE_COMMIT}",
            "dino_checkpoint_sha256": (
                "${oc.env:TALK2DINO_E9_DINO_CHECKPOINT_SHA256}"
            ),
        }
    }
    serialized = path.read_text().lower()
    for forbidden in ("retrieval", "prototype", "segmentation label", "mask"):
        assert forbidden not in serialized


def test_inference_grid_rejects_incompatible_patch_count():
    module()
    from models.dinotext.dinotext import _square_grid_side

    assert _square_grid_side(1024) == 32
    with pytest.raises(ValueError, match="does not form a square"):
        _square_grid_side(1023)


@pytest.mark.parametrize(
    "sparse,extra,match",
    (
        ({}, {}, "must contain exactly"),
        ({"adapter_path": "x", "unknown": 1}, {}, "must contain exactly"),
        (
            {
                "adapter_path": "x", "adapter_sha256": "a" * 64,
                "source_git_commit": "b" * 40,
                "dino_source_commit": "c" * 40,
                "dino_checkpoint_sha256": "d" * 64,
            },
            {"balanced_retrieval_prototypes": {}},
            "mutually exclusive",
        ),
    ),
)
def test_invalid_e9_configuration_fails_before_model_construction(
    sparse, extra, match
):
    module()
    from models.dinotext.dinotext import DINOText
    with pytest.raises(ValueError, match=match):
        DINOText(
            model_name="construction-must-not-run", resize_dim=1,
            clip_model_name="construction-must-not-run",
            proj_class="construction-must-not-run",
            proj_name="construction-must-not-run",
            proj_model="construction-must-not-run",
            sparse_region_alignment=sparse,
            **extra,
        )


@pytest.mark.parametrize(
    "model_name,expected_digest",
    (
        ("dinov2_vitb14_reg", "0" * 64),
        ("dinov2_vitb14_reg", "not-a-digest"),
        ("dinov2_vitl14_reg", "0" * 64),
    ),
)
def test_runtime_dino_identity_fails_before_adapter_or_backbone_construction(
    tmp_path, monkeypatch, model_name, expected_digest
):
    module()
    from models.dinotext import dinotext

    weights = tmp_path / "dino.pth"
    weights.write_bytes(b"tiny synthetic DINO identity probe")
    actual = hashlib.sha256(weights.read_bytes()).hexdigest()
    if expected_digest == "0" * 64 and model_name == "dinov2_vitb14_reg":
        assert actual != expected_digest
    counts = {"adapter": 0, "backbone": 0}

    def adapter_loader(*args, **kwargs):
        counts["adapter"] += 1
        raise AssertionError("adapter construction must not run")

    def backbone_loader(*args, **kwargs):
        counts["backbone"] += 1
        raise AssertionError("DINO construction must not run")

    monkeypatch.setattr(dinotext, "load_e9_adapter", adapter_loader)
    monkeypatch.setattr(dinotext, "load_local_vision_backbone", backbone_loader)
    sparse = {
        "adapter_path": "unused",
        "adapter_sha256": "a" * 64,
        "source_git_commit": "b" * 40,
        "dino_source_commit": "c" * 40,
        "dino_checkpoint_sha256": expected_digest,
    }
    with pytest.raises((ValueError, RuntimeError), match="DINO|SHA256|digest|model"):
        dinotext.DINOText(
            model_name=model_name,
            resize_dim=448,
            clip_model_name="construction-must-not-run",
            proj_class="construction-must-not-run",
            proj_name="construction-must-not-run",
            proj_model="construction-must-not-run",
            backbone_weights=str(weights),
            sparse_region_alignment=sparse,
        )
    assert counts == {"adapter": 0, "backbone": 0}
