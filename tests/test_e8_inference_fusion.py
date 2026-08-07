import inspect
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml

from src.e8_balanced_retrieval_adapter import (
    BalancedRetrievalPrototypes,
    BalancedRetrievalSettings,
    compute_e8_scores,
)


def normalized(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(*shape, generator=generator), dim=-1)


def masker_module():
    segmentation_root = (
        Path(__file__).parents[1] / "src" / "open_vocabulary_segmentation"
    )
    sys.path.insert(0, str(segmentation_root))
    try:
        from models.dinotext import masker
    finally:
        sys.path.pop(0)
    return masker


def dinotext_builder_module():
    segmentation_root = (
        Path(__file__).parents[1] / "src" / "open_vocabulary_segmentation"
    )
    sys.path.insert(0, str(segmentation_root))
    try:
        from segmentation.evaluation import dinotext_builder
    finally:
        sys.path.pop(0)
    return dinotext_builder


def independent_spatial_reference(
    image,
    text,
    prototypes,
    beta,
    *,
    prototype_temperature,
    responsibility_temperature,
):
    image = F.normalize(image.float(), dim=1)
    text = F.normalize(text.float(), dim=-1)
    prototypes = F.normalize(prototypes.float(), dim=-1)
    batch, _, height, width = image.shape
    classes, prototype_count, _ = prototypes.shape
    final = torch.empty(batch, classes, height, width)
    reliability = torch.empty_like(final)
    for batch_index in range(batch):
        for class_index in range(classes):
            for row in range(height):
                for column in range(width):
                    target = image[batch_index, :, row, column]
                    base = torch.dot(text[class_index], target)
                    prototype_scores = torch.stack(
                        [
                            torch.dot(prototypes[class_index, slot], target)
                            for slot in range(prototype_count)
                        ]
                    )
                    grounded = prototype_temperature * (
                        torch.logsumexp(
                            prototype_scores / prototype_temperature,
                            dim=0,
                        )
                        - math.log(prototype_count)
                    )
                    responsibilities = torch.softmax(
                        prototype_scores / responsibility_temperature,
                        dim=0,
                    )
                    if prototype_count == 1:
                        local_reliability = torch.ones_like(base)
                    else:
                        entropy = -torch.sum(
                            responsibilities
                            * responsibilities.clamp_min(
                                torch.finfo(responsibilities.dtype).tiny
                            ).log()
                        )
                        local_reliability = torch.clamp(
                            1 - entropy / math.log(prototype_count),
                            0,
                            1,
                        )
                    effective_beta = beta[class_index] * local_reliability
                    final[batch_index, class_index, row, column] = (
                        (1 - effective_beta) * base
                        + effective_beta * grounded
                    )
                    reliability[
                        batch_index, class_index, row, column
                    ] = local_reliability
    return final, reliability


def test_spatial_target_conditioned_fusion_matches_loop_and_has_one_sigmoid(
    monkeypatch,
):
    module = masker_module()
    masker = module.DINOTextMasker()
    image = normalized((2 * 2 * 3, 6), 1).reshape(2, 2, 3, 6).permute(
        0, 3, 1, 2
    )
    text = normalized((4, 6), 2)
    prototypes = normalized((4 * 3, 6), 3).reshape(4, 3, 6)
    beta = torch.tensor([0.05, 0.10, 0.20, 0.30])
    valid = torch.ones(4, 3, dtype=torch.bool)

    original_sigmoid = torch.sigmoid
    sigmoid_calls = 0

    def counting_sigmoid(value):
        nonlocal sigmoid_calls
        sigmoid_calls += 1
        return original_sigmoid(value)

    monkeypatch.setattr(torch, "sigmoid", counting_sigmoid)
    mask, scores = masker.forward_seg_with_balanced_prototypes(
        image,
        text,
        prototypes,
        valid,
        beta,
        prototype_temperature=0.10,
        responsibility_temperature=0.10,
    )
    expected, expected_reliability = independent_spatial_reference(
        image,
        text,
        prototypes,
        beta,
        prototype_temperature=0.10,
        responsibility_temperature=0.10,
    )

    assert sigmoid_calls == 1
    assert scores.shape == (2, 4, 2, 3)
    assert expected_reliability.shape == scores.shape
    torch.testing.assert_close(scores, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(mask, original_sigmoid(expected))


def test_local_reliability_is_spatial_and_deactivates_identical_modes():
    text = F.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=-1)
    targets = F.normalize(
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        dim=-1,
    )
    distinct = torch.eye(3).unsqueeze(0)
    beta = torch.tensor([0.3])
    valid = torch.ones(1, 3, dtype=torch.bool)
    scores = compute_e8_scores(
        text,
        targets,
        distinct,
        beta,
        responsibility_temperature=0.10,
        prototype_valid_mask=valid,
    )
    assert scores.reliability.shape == (1, 2)
    assert scores.reliability[0, 0] > scores.reliability[0, 1]
    assert scores.reliability[0, 0] > 0.95

    identical = text[:, None, :].expand(-1, 3, -1).clone()
    identical_scores = compute_e8_scores(
        text,
        targets,
        identical,
        beta,
        responsibility_temperature=0.10,
        prototype_valid_mask=valid,
    )
    assert torch.equal(
        identical_scores.reliability,
        torch.zeros_like(identical_scores.reliability),
    )
    assert torch.equal(
        identical_scores.final_score,
        identical_scores.base_score,
    )


def test_beta_zero_and_no_retrieval_are_bitwise_e3(monkeypatch):
    module = masker_module()
    masker = module.DINOTextMasker()
    image = normalized((2 * 2 * 2, 5), 5).reshape(2, 2, 2, 5).permute(
        0, 3, 1, 2
    )
    text = normalized((3, 5), 6)
    prototypes = normalized((3 * 3, 5), 7).reshape(3, 3, 5)
    valid = torch.ones(3, 3, dtype=torch.bool)

    baseline_mask, baseline_score = masker.forward_seg(image, text)

    def scoring_must_not_run(*args, **kwargs):
        raise AssertionError("E8 scoring ran during an exact E3 control")

    monkeypatch.setattr(module, "compute_e8_scores", scoring_must_not_run)
    zero_mask, zero_score = masker.forward_seg_with_balanced_prototypes(
        image,
        text,
        prototypes,
        valid,
        torch.zeros(3),
    )
    empty_mask, empty_score = masker.forward_seg_with_balanced_prototypes(
        image,
        text,
        prototypes,
        torch.zeros_like(valid),
        torch.full((3,), 0.3),
    )
    assert torch.equal(zero_score, baseline_score)
    assert torch.equal(zero_mask, baseline_mask)
    assert torch.equal(empty_score, baseline_score)
    assert torch.equal(empty_mask, baseline_mask)


def test_mixed_active_and_fallback_classes_are_bitwise_e3(monkeypatch):
    module = masker_module()
    masker = module.DINOTextMasker()
    generator = torch.Generator().manual_seed(14)
    image = torch.randn(2, 5, 2, 3, generator=generator)
    text = normalized((4, 5), 15)
    prototypes = normalized((4 * 3, 5), 16).reshape(4, 3, 5)
    prototypes[3] = prototypes[3, :1].expand(3, -1)
    valid = torch.tensor(
        [
            [True, True, True],
            [True, True, True],
            [False, False, False],
            [True, True, True],
        ]
    )
    beta = torch.tensor([0.30, 0.00, 0.30, 0.30])

    baseline_mask, baseline_score = masker.forward_seg(image, text)
    original_normalize = module.us.normalize
    original_sigmoid = torch.sigmoid
    image_normalizations = 0
    sigmoid_calls = 0

    def counting_normalize(value, *args, **kwargs):
        nonlocal image_normalizations
        if value is image:
            image_normalizations += 1
        return original_normalize(value, *args, **kwargs)

    def counting_sigmoid(value):
        nonlocal sigmoid_calls
        sigmoid_calls += 1
        return original_sigmoid(value)

    monkeypatch.setattr(module.us, "normalize", counting_normalize)
    monkeypatch.setattr(torch, "sigmoid", counting_sigmoid)
    e8_mask, e8_score = masker.forward_seg_with_balanced_prototypes(
        image,
        text,
        prototypes,
        valid,
        beta,
        prototype_temperature=0.10,
        responsibility_temperature=0.10,
    )

    assert image_normalizations == 1
    assert sigmoid_calls == 1
    # beta=0, no retrieval, and exactly-zero responsibility reliability.
    for inactive_class in (1, 2, 3):
        assert torch.equal(
            e8_score[:, inactive_class],
            baseline_score[:, inactive_class],
        )
        assert torch.equal(
            e8_mask[:, inactive_class],
            baseline_mask[:, inactive_class],
        )
    assert not torch.equal(e8_score[:, 0], baseline_score[:, 0])
    assert not torch.equal(e8_mask[:, 0], baseline_mask[:, 0])


def test_invalid_prototypes_do_not_affect_spatial_fusion():
    module = masker_module()
    masker = module.DINOTextMasker()
    image = normalized((1 * 2 * 2, 4), 8).reshape(1, 2, 2, 4).permute(
        0, 3, 1, 2
    )
    text = normalized((2, 4), 9)
    prototypes = normalized((2 * 3, 4), 10).reshape(2, 3, 4)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    beta = torch.tensor([0.2, 0.3])
    first = masker.forward_seg_with_balanced_prototypes(
        image,
        text,
        prototypes,
        valid,
        beta,
    )[1]
    changed = prototypes.clone()
    changed[~valid] = normalized((int((~valid).sum()), 4), 11)
    second = masker.forward_seg_with_balanced_prototypes(
        image,
        text,
        changed,
        valid,
        beta,
    )[1]
    assert torch.equal(first, second)


@pytest.mark.parametrize(
    "name,value",
    (
        ("prototype_temperature", float("nan")),
        ("prototype_temperature", float("inf")),
        ("prototype_temperature", 0.0),
        ("responsibility_temperature", float("nan")),
        ("responsibility_temperature", float("inf")),
        ("responsibility_temperature", 0.0),
    ),
)
def test_inference_rejects_invalid_temperatures(name, value):
    module = masker_module()
    kwargs = {name: value}
    with pytest.raises(ValueError, match=name):
        module.DINOTextMasker().forward_seg_with_balanced_prototypes(
            torch.ones(1, 3, 1, 1),
            normalized((1, 3), 12),
            normalized((3, 3), 13).reshape(1, 3, 3),
            torch.ones(1, 3, dtype=torch.bool),
            torch.zeros(1),
            **kwargs,
        )


def diagnostic_batch(*, all_empty=False):
    retrieval_count = (
        torch.zeros(3, dtype=torch.long)
        if all_empty
        else torch.tensor([2, 0, 0])
    )
    valid_mask = torch.tensor(
        [
            [not all_empty, not all_empty],
            [False, False],
            [False, False],
        ]
    )
    return SimpleNamespace(
        retrieval_count=retrieval_count,
        valid_mask=valid_mask,
        alpha=torch.tensor([0.20, 0.99, 0.98]),
        beta=torch.tensor([0.30, 0.97, 0.96]),
        slot_mass=torch.tensor(
            [[0.50, 0.50], [100.0, 0.0], [200.0, 0.0]]
        ),
        mode_vectors=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]
        ),
        prototypes=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]
        ),
    )


def test_e8_inference_diagnostics_exclude_no_retrieval_rows():
    builder = dinotext_builder_module()
    generated = diagnostic_batch()
    diagnostics = builder._e8_diagnostic_values(generated)

    assert torch.equal(
        diagnostics["eligible"],
        generated.retrieval_count > 0,
    )
    assert torch.equal(diagnostics["alpha"], torch.tensor([0.20]))
    assert torch.equal(diagnostics["beta"], torch.tensor([0.30]))
    assert torch.equal(
        diagnostics["slot_mass"],
        torch.tensor([[0.50, 0.50]]),
    )
    assert torch.equal(diagnostics["effective_slots"], torch.tensor([2.0]))
    assert torch.equal(
        diagnostics["mode_pairwise_cosine"],
        torch.tensor([0.0]),
    )
    assert torch.equal(
        diagnostics["prototype_pairwise_cosine"],
        torch.tensor([0.0]),
    )


def test_e8_inference_diagnostics_use_null_for_all_empty(monkeypatch):
    builder = dinotext_builder_module()
    generated = diagnostic_batch(all_empty=True)
    diagnostics = builder._e8_diagnostic_values(generated)
    for name in (
        "alpha",
        "beta",
        "slot_mass",
        "effective_slots",
        "mode_pairwise_cosine",
        "prototype_pairwise_cosine",
    ):
        assert diagnostics[name].numel() == 0
        assert builder._e8_diagnostic_range(diagnostics[name]) == "null"

    messages = []
    monkeypatch.setattr(
        builder,
        "get_logger",
        lambda: SimpleNamespace(info=messages.append),
    )
    builder._log_e8_summary(
        generated,
        SimpleNamespace(
            prototype_temperature=0.10,
            responsibility_temperature=0.10,
        ),
        ["one", "two", "three"],
    )
    assert len(messages) == 1
    assert "alpha=null" in messages[0]
    assert "mode_pairwise_cosine=null" in messages[0]
    assert "prototype_pairwise_cosine=null" in messages[0]


def test_e8_generation_cache_identity_binds_every_inference_input():
    raw = normalized((2, 4), 40)
    mapped = normalized((2, 5), 41)
    settings = BalancedRetrievalSettings()
    generator = BalancedRetrievalPrototypes(
        {},
        object(),
        settings,
        bank_identity={"identity": "bank-a"},
        checkpoint_sha256="a" * 64,
    )
    baseline = generator._cache_key(raw, mapped)
    assert baseline == generator._cache_key(raw.clone(), mapped.clone())

    changed_raw = raw.clone()
    changed_raw[0, 0] += 1e-3
    changed_mapped = mapped.clone()
    changed_mapped[0, 0] += 1e-3
    assert generator._cache_key(changed_raw, mapped) != baseline
    assert generator._cache_key(raw, changed_mapped) != baseline

    changed_settings = BalancedRetrievalPrototypes(
        {},
        object(),
        BalancedRetrievalSettings(prototype_temperature=0.20),
        bank_identity={"identity": "bank-a"},
        checkpoint_sha256="a" * 64,
    )
    changed_bank = BalancedRetrievalPrototypes(
        {},
        object(),
        settings,
        bank_identity={"identity": "bank-b"},
        checkpoint_sha256="a" * 64,
    )
    changed_checkpoint = BalancedRetrievalPrototypes(
        {},
        object(),
        settings,
        bank_identity={"identity": "bank-a"},
        checkpoint_sha256="b" * 64,
    )
    assert changed_settings._cache_key(raw, mapped) != baseline
    assert changed_bank._cache_key(raw, mapped) != baseline
    assert changed_checkpoint._cache_key(raw, mapped) != baseline


@pytest.mark.parametrize(
    "enabled",
    (
        {
            "retrieval_grounded_prototypes": {},
            "learned_retrieval_prototypes": {},
        },
        {
            "retrieval_grounded_prototypes": {},
            "balanced_retrieval_prototypes": {},
        },
        {
            "learned_retrieval_prototypes": {},
            "balanced_retrieval_prototypes": {},
        },
    ),
)
def test_e6_e7_e8_are_rejected_before_model_construction(enabled):
    masker_module()
    from models.dinotext.dinotext import DINOText

    with pytest.raises(ValueError, match="mutually exclusive"):
        DINOText(
            model_name="construction-must-not-run",
            resize_dim=1,
            clip_model_name="construction-must-not-run",
            proj_class="construction-must-not-run",
            proj_name="construction-must-not-run",
            proj_model="construction-must-not-run",
            **enabled,
        )


def test_e8_configuration_is_separate_and_closed():
    root = Path("src/open_vocabulary_segmentation/configs/stuff")
    path = root / (
        "dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_"
        "balanced_retrieval_fusion.yml"
    )
    config = yaml.safe_load(path.read_text())
    assert config["_base_"] == (
        "dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_tau010.yml"
    )
    assert set(config) == {"_base_", "model"}
    assert set(config["model"]) == {"balanced_retrieval_prototypes"}
    settings = dict(config["model"]["balanced_retrieval_prototypes"])
    assert settings.pop("bank_path") == "${oc.env:TALK2DINO_PROTOTYPE_BANK}"
    assert settings.pop("adapter_path") == "${oc.env:TALK2DINO_E8_ADAPTER}"
    assert BalancedRetrievalSettings.from_mapping(settings) == (
        BalancedRetrievalSettings()
    )

    source = inspect.getsource(
        masker_module().DINOTextMasker.forward_seg_with_balanced_prototypes
    )
    assert ".item(" not in source
    assert "for class" not in source
    serialized = path.read_text().lower()
    for forbidden in (
        "retrieval_grounded_prototypes",
        "learned_retrieval_prototypes",
        "mmr",
        "kmeans",
        "patch_tokens",
        "mask",
    ):
        assert forbidden not in serialized
