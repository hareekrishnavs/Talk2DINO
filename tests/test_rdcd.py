import io
import json
import math
import tarfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.dataset import DenseConsistencyDataset, DinoClipDataset
from src.loss import (
    ContrastiveLoss,
    compute_rdcd_components,
    prepare_attention_probabilities,
)
from src.model import ProjectionLayer
from src.train_util import train, validate
from src.training_features import validation_feature_name


def make_dense_record(batch_id=7, heads=3, patches=5, dim=4, captions=2):
    maps = torch.rand(heads, patches)
    maps = maps / maps.sum(dim=-1, keepdim=True)
    return {
        "image_id": batch_id,
        "disentangled_self_attn": torch.randn(heads, dim),
        "patch_tokens": torch.randn(patches, dim),
        "self_attn_maps": maps,
        "captions": [f"caption-{index}" for index in range(captions)],
        "ann_feats": [torch.randn(dim) for _ in range(captions)],
        "annotation_ids": [batch_id * 10 + index for index in range(captions)],
    }


def write_dense_fixture(root, record):
    root.mkdir()
    payload = io.BytesIO()
    torch.save(record, payload)
    shard = root / "train-000000.tar"
    with tarfile.open(shard, "w") as archive:
        member = tarfile.TarInfo("00000000.pth")
        member.size = len(payload.getvalue())
        archive.addfile(member, io.BytesIO(payload.getvalue()))
    manifest = {
        "format_version": 1,
        "split": "train",
        "images_per_shard": 1,
        "source_images": 1,
        "source_annotations": len(record.get("annotation_ids", [])),
        "selected_images": 1,
        "selected_annotations": len(record.get("annotation_ids", [])),
        "complete": True,
        "failed_image_ids": [],
        "is_pilot": False,
        "images": 1,
        "annotations": len(record.get("annotation_ids", [])),
        "shards": [{"name": shard.name, "images": 1}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def normalized_inputs(batch=3, heads=4, patches=6, dim=5, dtype=torch.float32, device="cpu"):
    text = F.normalize(torch.randn(batch, dim, dtype=dtype, device=device), dim=-1)
    visual = F.normalize(
        torch.randn(batch, heads, dim, dtype=dtype, device=device), dim=-1
    )
    patch_tokens = torch.randn(batch, patches, dim, dtype=dtype, device=device)
    maps = torch.rand(batch, heads, patches, dtype=dtype, device=device)
    maps = maps / maps.sum(dim=-1, keepdim=True)
    return text, visual, patch_tokens, maps


def rdcd(text, heads, patches, maps, routing_temperature=0.1, dense_temperature=0.1):
    return compute_rdcd_components(
        text,
        heads,
        patches,
        maps,
        routing_temperature=routing_temperature,
        dense_temperature=dense_temperature,
        attention_map_format="probabilities",
    )


def identity_layer(strategy, dim=4, dense_loss_weight=1.0):
    layer = ProjectionLayer(
        act=None,
        hidden_layer=False,
        dino_embed_dim=dim,
        clip_embed_dim=dim,
        num_attn_head=3,
        alignment_strategy=strategy,
        routing_temperature=0.1,
        dense_temperature=0.1,
        dense_loss_weight=dense_loss_weight,
        attention_map_format="probabilities",
    )
    with torch.no_grad():
        layer.linear_layer.weight.copy_(torch.eye(dim))
        layer.linear_layer.bias.zero_()
    return layer


def test_e5_dataset_returns_annotation_level_dense_samples(tmp_path):
    record = make_dense_record()
    dataset = DenseConsistencyDataset(
        write_dense_fixture(tmp_path / "dense", record), dino_embed_dim=4
    )
    samples = list(dataset)
    assert len(dataset) == len(samples) == 2
    assert samples[0]["image"].shape == (3, 4)
    assert samples[0]["patch_tokens"].shape == (5, 4)
    assert samples[0]["self_attn_maps"].shape == (3, 5)
    assert samples[0]["metadata"] == {"image_id": 7, "annotation_id": 70}


def test_default_dinoclip_dataset_is_unchanged(tmp_path):
    path = tmp_path / "ordinary.pth"
    torch.save(
        {
            "images": [{"id": 1, "dino_features": torch.randn(4)}],
            "annotations": [
                {"id": 2, "image_id": 1, "ann_feats": torch.randn(4)}
            ],
        },
        path,
    )
    sample = DinoClipDataset(path)[0]
    assert set(sample) == {"image", "annotation", "metadata"}


def test_dense_dataset_missing_auxiliary_field_fails(tmp_path):
    record = make_dense_record()
    record.pop("patch_tokens")
    dataset = DenseConsistencyDataset(
        write_dense_fixture(tmp_path / "missing", record), dino_embed_dim=4
    )
    with pytest.raises(ValueError, match="patch_tokens"):
        list(dataset)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("self_attn_maps", torch.ones(2, 5), "head count H"),
        ("self_attn_maps", torch.ones(3, 4), "patch count P"),
        ("patch_tokens", torch.ones(5, 6), "dimension D"),
    ],
)
def test_dense_dataset_shape_mismatches_fail(
    tmp_path, field, replacement, message
):
    record = make_dense_record()
    record[field] = replacement
    dataset = DenseConsistencyDataset(
        write_dense_fixture(tmp_path / field, record), dino_embed_dim=4
    )
    with pytest.raises(ValueError, match=message):
        list(dataset)


@pytest.mark.parametrize("field", ["disentangled_self_attn", "patch_tokens", "ann_feats"])
def test_dense_dataset_nonfinite_fields_fail(tmp_path, field):
    record = make_dense_record()
    if field == "ann_feats":
        record[field][0][0] = float("nan")
    else:
        record[field][0, 0] = float("nan")
    dataset = DenseConsistencyDataset(
        write_dense_fixture(tmp_path / field, record), dino_embed_dim=4
    )
    with pytest.raises(ValueError, match="finite"):
        list(dataset)


def test_probability_map_normalization_without_second_softmax():
    maps = torch.tensor([[[0.8, 0.2], [2.0, 1.0]]])
    actual = prepare_attention_probabilities(maps, "probabilities")
    expected = torch.tensor([[[0.8, 0.2], [2 / 3, 1 / 3]]])
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual[0, 0], torch.softmax(maps[0, 0], dim=-1))


def test_logit_map_softmax():
    logits = torch.tensor([[[2.0, -1.0, 0.5]]], dtype=torch.float64)
    actual = prepare_attention_probabilities(logits, "logits")
    torch.testing.assert_close(actual, torch.softmax(logits, dim=-1))


@pytest.mark.parametrize(
    ("maps", "message"),
    [
        (torch.zeros(1, 1, 2), "zero-sum"),
        (torch.tensor([[[0.5, float("nan")]]]), "non-finite"),
        (torch.tensor([[[1.1, -0.1]]]), "negative"),
    ],
)
def test_invalid_probability_maps_fail(maps, message):
    with pytest.raises(ValueError, match=message):
        prepare_attention_probabilities(maps, "probabilities")


def test_rdcd_pcrr_scores_are_identical_to_e3():
    torch.manual_seed(4)
    text, heads, patches, maps = normalized_inputs()
    e3 = ProjectionLayer(
        act=None,
        dino_embed_dim=5,
        clip_embed_dim=5,
        alignment_strategy="paired_soft_routing",
        routing_temperature=0.1,
    )
    expected = e3.compute_similarity(heads, text)
    actual = rdcd(text, heads, patches, maps)["pcrr_scores"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_negative_queries_cannot_reroute_images():
    text = torch.eye(2)
    heads = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]
    )
    patches = torch.randn(2, 3, 2)
    maps = torch.full((2, 2, 3), 1 / 3)
    scores = rdcd(text, heads, patches, maps, routing_temperature=0.01)[
        "pcrr_scores"
    ]
    assert scores[1, 0] < 1e-4
    assert scores[0, 0] > 0.9999


def test_teacher_matches_reference_and_rows_sum_to_one():
    text, heads, patches, maps = normalized_inputs()
    result = rdcd(text, heads, patches, maps, routing_temperature=0.2)
    affinities = torch.einsum("bd,bhd->bh", text, heads)
    weights = torch.softmax(affinities / 0.2, dim=-1)
    expected = torch.einsum("bh,bhp->bp", weights.detach(), maps)
    expected = expected.clamp_min(1e-8)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(result["teacher_saliency"], expected)
    torch.testing.assert_close(
        result["teacher_saliency"].sum(-1), torch.ones(text.shape[0])
    )


def test_dense_teacher_does_not_backpropagate_into_routing_or_maps():
    text, heads, patches, maps = normalized_inputs(batch=2)
    text.requires_grad_()
    heads.requires_grad_()
    maps.requires_grad_()
    result = rdcd(text, heads, patches, maps)
    result["routing_weights"].retain_grad()
    result["dense_loss"].backward()
    assert text.grad is not None and torch.isfinite(text.grad).all()
    assert heads.grad is None
    assert maps.grad is None
    assert result["routing_weights"].grad is None
    assert result["teacher_saliency"].requires_grad is False


def test_student_distribution_matches_reference_and_has_no_cross_image_mixing():
    text, heads, patches, maps = normalized_inputs(batch=2)
    result = rdcd(text, heads, patches, maps, dense_temperature=0.17)
    normalized_patches = F.normalize(patches, dim=-1)
    logits = torch.einsum("bd,bpd->bp", text, normalized_patches)
    expected = torch.log_softmax(logits / 0.17, dim=-1)
    torch.testing.assert_close(result["student_log_prob"], expected)

    changed = patches.clone()
    changed[1] = torch.randn_like(changed[1]) * 100
    changed_result = rdcd(text, heads, changed, maps, dense_temperature=0.17)
    torch.testing.assert_close(
        result["student_log_prob"][0], changed_result["student_log_prob"][0]
    )


def test_kl_zero_for_matching_distribution_and_increases_when_mass_moves():
    text = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    heads = text.unsqueeze(1)
    patches = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    logits = torch.einsum("bd,bpd->bp", text, F.normalize(patches, dim=-1))
    teacher = torch.softmax(logits / 0.1, dim=-1).unsqueeze(1)
    matching = rdcd(text, heads, patches, teacher)["dense_kl"]
    moved = rdcd(text, heads, patches.flip(1), teacher)["dense_kl"]
    assert matching.abs() < 1e-12
    assert moved > matching + 1.0


def test_lambda_zero_is_exactly_e3_loss_and_gradients():
    torch.manual_seed(9)
    e3_model = identity_layer("paired_soft_routing")
    e5_model = identity_layer(
        "paired_soft_routing_rdcd", dense_loss_weight=0.0
    )
    e5_model.load_state_dict(e3_model.state_dict())
    heads = torch.randn(3, 2, 4)
    text = torch.randn(3, 4)
    patches = torch.randn(3, 5, 4)
    maps = torch.rand(3, 2, 5)
    maps = maps / maps.sum(-1, keepdim=True)

    e3_loss = ContrastiveLoss(e3_model, ltype="infonce")(heads, text)
    e5_loss = ContrastiveLoss(e5_model, ltype="infonce_rdcd")(
        heads, text, patch_tokens=patches, self_attn_maps=maps
    )
    torch.testing.assert_close(e5_loss, e3_loss, rtol=0, atol=0)
    e3_loss.backward()
    e5_loss.backward()
    for e3_parameter, e5_parameter in zip(
        e3_model.parameters(), e5_model.parameters()
    ):
        torch.testing.assert_close(e5_parameter.grad, e3_parameter.grad, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("batch", "heads_count", "patch_count"),
    [(2, 1, 4), (2, 3, 1), (1, 3, 4)],
)
def test_degenerate_dimensions(batch, heads_count, patch_count):
    text, heads, patches, maps = normalized_inputs(
        batch=batch, heads=heads_count, patches=patch_count
    )
    result = rdcd(text, heads, patches, maps)
    assert result["pcrr_scores"].shape == (batch, batch)
    assert result["teacher_saliency"].shape == (batch, patch_count)
    assert torch.isfinite(result["dense_loss"])


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_float64_device_preservation_and_finite_backward(device):
    text, heads, patches, maps = normalized_inputs(
        batch=2, dtype=torch.float64, device=device
    )
    text.requires_grad_()
    result = rdcd(text, heads, patches, maps)
    assert result["teacher_saliency"].dtype == torch.float64
    assert result["student_log_prob"].dtype == torch.float64
    assert result["dense_loss"].device.type == device
    result["dense_loss"].backward()
    assert torch.isfinite(text.grad).all()


def test_patch_gradients_are_opt_in():
    text, heads, patches, maps = normalized_inputs(batch=2)
    text.requires_grad_()
    result = rdcd(text, heads, patches, maps)
    result["dense_loss"].backward()
    assert patches.grad is None

    text2, heads2, patches2, maps2 = normalized_inputs(batch=2)
    text2.requires_grad_()
    patches2.requires_grad_()
    rdcd(text2, heads2, patches2, maps2)["dense_loss"].backward()
    assert patches2.grad is not None and torch.isfinite(patches2.grad).all()


def test_rdcd_loss_reaches_text_projection():
    model = identity_layer("paired_soft_routing_rdcd")
    criterion = ContrastiveLoss(model, ltype="infonce_rdcd")
    heads = torch.randn(2, 3, 4)
    text = torch.randn(2, 4)
    patches = torch.randn(2, 5, 4)
    maps = torch.rand(2, 3, 5)
    maps = maps / maps.sum(-1, keepdim=True)
    loss = criterion(
        heads,
        text,
        patch_tokens=patches,
        self_attn_maps=maps,
    )
    loss.backward()
    assert model.linear_layer.weight.grad is not None
    assert torch.isfinite(model.linear_layer.weight.grad).all()


def test_training_and_validation_loops_forward_dense_inputs():
    model = identity_layer("paired_soft_routing_rdcd")
    criterion = ContrastiveLoss(model, ltype="infonce_rdcd")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    maps = torch.rand(2, 3, 5)
    maps = maps / maps.sum(-1, keepdim=True)
    samples = [
        {
            "image": torch.randn(3, 4),
            "annotation": torch.randn(4),
            "patch_tokens": torch.randn(5, 4),
            "self_attn_maps": maps[index],
        }
        for index in range(2)
    ]
    loader = DataLoader(samples, batch_size=2)
    train_loss = train(model, loader, criterion, optimizer)
    val_loss = validate(model, loader, criterion)
    assert torch.isfinite(torch.tensor([train_loss, val_loss])).all()


def test_standard_infonce_and_triplet_numerics_unchanged():
    scores = torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float64)
    infonce = ContrastiveLoss(None, ltype="infonce")
    scale = torch.tensor(math.log(1 / 0.07), dtype=torch.float32).to(
        torch.float64
    ).exp()
    logits = scale * scores
    labels = torch.arange(2)
    expected_infonce = (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    ) / 2 / 2**2
    torch.testing.assert_close(
        infonce.compute_contrastive_loss(scores), expected_infonce
    )

    triplet = ContrastiveLoss(None, margin=0.2, max_violation=False, ltype="triplet")
    diagonal = scores.diag().view(2, 1)
    cost_s = (0.2 + scores - diagonal.expand_as(scores)).clamp(min=0)
    cost_im = (0.2 + scores - diagonal.t().expand_as(scores)).clamp(min=0)
    mask = torch.eye(2, dtype=torch.bool)
    expected_triplet = (
        cost_s.masked_fill(mask, 0).sum() + cost_im.masked_fill(mask, 0).sum()
    ) / 2**2
    torch.testing.assert_close(
        triplet.compute_contrastive_loss(scores), expected_triplet
    )


def test_fixed_logit_scale_is_a_buffer():
    criterion = ContrastiveLoss(None, ltype="infonce")
    assert "logit_scale" in dict(criterion.named_buffers())
    assert "logit_scale" not in dict(criterion.named_parameters())
    assert criterion.logit_scale.requires_grad is False
    scores = torch.eye(2, requires_grad=True)
    criterion.compute_contrastive_loss(scores).backward()
    assert criterion.logit_scale.grad is None


def test_state_dict_round_trip_and_evaluation_configuration(tmp_path):
    model = identity_layer("paired_soft_routing_rdcd")
    checkpoint = tmp_path / "vitb_mlp_infonce_rdcd_tau010.pth"
    torch.save(model.state_dict(), checkpoint)
    restored = identity_layer("paired_soft_routing_rdcd")
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected)

    config = yaml.safe_load(
        Path(
            "src/open_vocabulary_segmentation/configs/stuff/"
            "dinotext_stuff_vitb_mlp_infonce_rdcd_tau010.yml"
        ).read_text()
    )
    assert config["model"]["proj_class"] == "vitb_mlp_infonce_rdcd_tau010"
    assert config["model"]["proj_model"] == "ProjectionLayer"


def test_training_config_and_validation_feature_selection():
    config = yaml.safe_load(
        Path("configs/vitb_mlp_infonce_rdcd_tau010.yaml").read_text()
    )
    assert config["model"]["alignment_strategy"] == "paired_soft_routing_rdcd"
    assert config["model"]["attention_map_format"] == "probabilities"
    assert config["train"]["ltype"] == "infonce_rdcd"
    assert config["train"]["dense_consistency"] is True
    assert (
        validation_feature_name(
            "disentangled_self_attn", "paired_soft_routing_rdcd"
        )
        == "disentangled_self_attn"
    )
