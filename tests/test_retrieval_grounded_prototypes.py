import copy
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

import build_e6_prototype_bank as bank_builder
from build_e6_prototype_bank import build_prototype_bank, source_git_provenance
from src.e6_prototype_bank import (
    FORMAT_VERSION,
    PrototypeBankValidationError,
    annotation_id_set_fingerprint,
    load_prototype_bank,
    route_annotation_batch,
    sha256_file,
    validate_prototype_bank,
)
from src.model import ProjectionLayer
from src.retrieval_grounded_prototypes import (
    PrototypeBank,
    RGTPSettings,
    RetrievalGroundedPrototypes,
    anchor_prototypes,
    deduplicate_by_image,
    deterministic_mmr,
    deterministic_spherical_kmeans,
    exact_chunked_topk,
    fuse_prototype_scores,
    normalized_logsumexp,
    retrieval_confidence,
)


def normalized(rows, dimensions, seed):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(rows, dimensions, generator=generator), dim=-1)


def initialize_git_repository(path):
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    tracked = path / "implementation.py"
    tracked.write_text("VERSION = 1\n")
    subprocess.run(["git", "add", "implementation.py"], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=E6 Test",
            "-c",
            "user.email=e6-test@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=path,
        check=True,
    )
    return tracked


def clean_provenance():
    return {
        "source_git_commit": "b" * 40,
        "source_git_dirty": False,
        "source_git_diff_sha256": None,
    }


def write_builder_inputs(tmp_path):
    config_path = (
        tmp_path / "vitb_mlp_infonce_paired_soft_routing_tau010.yaml"
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "act": "tanh",
                    "hidden_layer": True,
                    "dino_embed_dim": 768,
                    "alignment_strategy": "paired_soft_routing",
                    "routing_temperature": 0.10,
                }
            }
        )
    )
    projection = ProjectionLayer.from_config(
        yaml.safe_load(config_path.read_text())["model"]
    )
    checkpoint_path = (
        tmp_path / "vitb_mlp_infonce_paired_soft_routing_tau010.pth"
    )
    torch.save(projection.state_dict(), checkpoint_path)
    source_path = tmp_path / "train.pth"
    images = [
        {"id": index, "disentangled_self_attn": torch.randn(12, 768)}
        for index in range(2)
    ]
    annotations = [
        {
            "id": 10 + index,
            "image_id": index % 2,
            "ann_feats": torch.randn(512),
        }
        for index in range(3)
    ]
    torch.save({"images": images, "annotations": annotations}, source_path)
    return config_path, checkpoint_path, source_path


def bank_payload(
    entries=8,
    *,
    pilot=False,
    complete=True,
    repeated_images=False,
):
    annotation_ids = torch.arange(100, 100 + entries, dtype=torch.int64)
    if repeated_images:
        image_ids = torch.arange(entries, dtype=torch.int64) // 2
    else:
        image_ids = torch.arange(entries, dtype=torch.int64)
    source_annotations = entries + 3 if pilot else entries
    payload = {
        "caption_embeddings": normalized(entries, 512, 1).half(),
        "routed_dino_embeddings": normalized(entries, 768, 2).half(),
        "image_ids": image_ids,
        "annotation_ids": annotation_ids,
        "metadata": {
            "format_version": FORMAT_VERSION,
            "complete": complete,
            "is_pilot": pilot,
            "source_feature_path": "/synthetic/train.pth",
            "source_image_count": entries,
            "source_annotation_count": source_annotations,
            "selected_annotation_count": entries,
            "e3_config_name": (
                "vitb_mlp_infonce_paired_soft_routing_tau010.yaml"
            ),
            "e3_config_sha256": "c" * 64,
            "e3_checkpoint_name": (
                "vitb_mlp_infonce_paired_soft_routing_tau010.pth"
            ),
            "checkpoint_sha256": "a" * 64,
            "routing_temperature": 0.10,
            "source_git_commit": "b" * 40,
            "source_git_dirty": False,
            "source_git_diff_sha256": None,
            "dimensions": {
                "caption_embeddings": 512,
                "routed_dino_embeddings": 768,
            },
            "dtypes": {
                "caption_embeddings": "float16",
                "routed_dino_embeddings": "float16",
                "image_ids": "int64",
                "annotation_ids": "int64",
            },
            "annotation_id_fingerprint": annotation_id_set_fingerprint(
                annotation_ids
            ),
        },
    }
    return payload


def runtime_bank(entries=8, repeated_images=False):
    payload = bank_payload(entries, repeated_images=repeated_images)
    return PrototypeBank(
        payload["caption_embeddings"],
        payload["routed_dino_embeddings"],
        payload["image_ids"],
        payload["annotation_ids"],
        payload["metadata"],
    )


def test_clean_git_provenance_is_accepted(tmp_path):
    repository = tmp_path / "clean"
    initialize_git_repository(repository)
    provenance = source_git_provenance(repository)
    assert provenance["source_git_dirty"] is False
    assert provenance["source_git_diff_sha256"] is None
    assert len(provenance["source_git_commit"]) == 40


@pytest.mark.parametrize("dirty_kind", ["tracked", "staged", "untracked"])
def test_dirty_git_provenance_is_rejected_by_default(tmp_path, dirty_kind):
    repository = tmp_path / dirty_kind
    tracked = initialize_git_repository(repository)
    if dirty_kind == "tracked":
        tracked.write_text("VERSION = 2\n")
    elif dirty_kind == "staged":
        tracked.write_text("VERSION = 2\n")
        subprocess.run(
            ["git", "add", "implementation.py"],
            cwd=repository,
            check=True,
        )
    else:
        (repository / "new_e6_source.py").write_text("E6 = True\n")

    with pytest.raises(
        PrototypeBankValidationError,
        match="Commit the E6 implementation first",
    ):
        source_git_provenance(repository)


def test_explicit_dirty_provenance_records_commit_flag_and_diff_hash(tmp_path):
    repository = tmp_path / "dirty"
    tracked = initialize_git_repository(repository)
    tracked.write_text("VERSION = 2\n")
    (repository / "new_e6_source.py").write_text("E6 = True\n")

    first = source_git_provenance(repository, allow_dirty_source=True)
    second = source_git_provenance(repository, allow_dirty_source=True)
    assert first == second
    assert first["source_git_dirty"] is True
    assert len(first["source_git_commit"]) == 40
    assert len(first["source_git_diff_sha256"]) == 64


def test_bank_routing_matches_independent_loop_reference():
    torch.manual_seed(3)
    projection = ProjectionLayer(
        act=None,
        hidden_layer=False,
        dino_embed_dim=768,
        clip_embed_dim=512,
        alignment_strategy="paired_soft_routing",
        routing_temperature=0.10,
    )
    annotations = torch.randn(3, 512)
    heads = torch.randn(3, 12, 768)
    captions, routed = route_annotation_batch(
        projection,
        annotations,
        heads,
        0.10,
    )

    expected_captions = []
    expected_routed = []
    for annotation, image_heads in zip(annotations, heads):
        expected_captions.append(F.normalize(annotation, dim=0))
        projected = F.normalize(
            projection.project_clip_txt(annotation.unsqueeze(0))[0],
            dim=0,
        )
        normalized_heads = F.normalize(image_heads, dim=-1)
        weights = torch.softmax(
            torch.stack(
                [torch.dot(projected, head) for head in normalized_heads]
            )
            / 0.10,
            dim=0,
        )
        expected_routed.append(
            F.normalize(
                sum(
                    weight * head
                    for weight, head in zip(weights, normalized_heads)
                ),
                dim=0,
            )
        )
    torch.testing.assert_close(captions, torch.stack(expected_captions))
    torch.testing.assert_close(routed, torch.stack(expected_routed))


def test_bank_schema_shapes_dtypes_norms_and_finiteness():
    payload = bank_payload()
    summary = validate_prototype_bank(payload)
    assert payload["caption_embeddings"].shape == (8, 512)
    assert payload["routed_dino_embeddings"].shape == (8, 768)
    assert payload["caption_embeddings"].dtype == torch.float16
    assert payload["routed_dino_embeddings"].dtype == torch.float16
    assert summary["entries"] == 8
    assert torch.isfinite(payload["caption_embeddings"]).all()
    assert torch.isfinite(payload["routed_dino_embeddings"]).all()


def test_bank_validator_enforces_e3_identity_tau_and_int64_id_tensors():
    wrong_tau = bank_payload()
    wrong_tau["metadata"]["routing_temperature"] = 0.20
    with pytest.raises(PrototypeBankValidationError, match="0.10"):
        validate_prototype_bank(wrong_tau)

    wrong_config = bank_payload()
    wrong_config["metadata"]["e3_config_name"] = "different.yaml"
    with pytest.raises(PrototypeBankValidationError, match="configuration"):
        validate_prototype_bank(wrong_config)

    list_ids = bank_payload()
    list_ids["annotation_ids"] = list_ids["annotation_ids"].tolist()
    with pytest.raises(PrototypeBankValidationError, match="torch.int64"):
        validate_prototype_bank(list_ids)


def test_pilot_and_incomplete_banks_are_rejected_by_default(tmp_path):
    pilot_path = tmp_path / "pilot.pth"
    torch.save(bank_payload(pilot=True), pilot_path)
    with pytest.raises(PrototypeBankValidationError, match="pilot"):
        load_prototype_bank(pilot_path)
    assert load_prototype_bank(pilot_path, allow_pilot=True)["metadata"][
        "is_pilot"
    ]

    incomplete_path = tmp_path / "incomplete.pth"
    torch.save(bank_payload(complete=False), incomplete_path)
    with pytest.raises(PrototypeBankValidationError, match="incomplete"):
        load_prototype_bank(incomplete_path)
    assert not load_prototype_bank(
        incomplete_path,
        require_complete=False,
    )["metadata"]["complete"]

    full_path = tmp_path / "full.pth"
    torch.save(bank_payload(), full_path)
    with pytest.raises(ValueError, match="different E3 checkpoint"):
        PrototypeBank.load(
            full_path,
            expected_checkpoint_sha256="c" * 64,
        )


def test_dirty_bank_is_rejected_by_evaluator_default(tmp_path):
    payload = bank_payload(pilot=True)
    payload["metadata"]["source_git_dirty"] = True
    payload["metadata"]["source_git_diff_sha256"] = "d" * 64
    path = tmp_path / "dirty-pilot.pth"
    torch.save(payload, path)

    with pytest.raises(PrototypeBankValidationError, match="dirty Git worktree"):
        load_prototype_bank(path)
    accepted = load_prototype_bank(
        path,
        allow_pilot=True,
        allow_dirty_source=True,
    )
    assert accepted["metadata"]["source_git_dirty"] is True


def test_modified_same_named_e3_configuration_is_rejected(tmp_path):
    config_path = (
        tmp_path / "vitb_mlp_infonce_paired_soft_routing_tau010.yaml"
    )
    config_path.write_bytes(b"model:\n  routing_temperature: 0.10\n")
    payload = bank_payload()
    payload["metadata"]["e3_config_sha256"] = sha256_file(config_path)
    bank_path = tmp_path / "bank.pth"
    torch.save(payload, bank_path)

    config_path.write_bytes(b"model:\n  routing_temperature: 0.20\n")
    with pytest.raises(
        PrototypeBankValidationError,
        match="configuration SHA256",
    ):
        load_prototype_bank(
            bank_path,
            expected_config_path=config_path,
        )
    with pytest.raises(ValueError, match="different E3 configuration"):
        PrototypeBank.load(
            bank_path,
            expected_config_sha256=sha256_file(config_path),
        )


def test_unknown_metadata_key_is_rejected():
    payload = bank_payload()
    payload["metadata"]["ambiguous_legacy_field"] = "unexpected"
    with pytest.raises(PrototypeBankValidationError, match="unknown keys"):
        validate_prototype_bank(payload)


def test_clean_committed_bank_is_accepted(tmp_path):
    path = tmp_path / "clean-bank.pth"
    torch.save(bank_payload(), path)
    loaded = load_prototype_bank(path)
    assert loaded["metadata"]["source_git_dirty"] is False
    assert loaded["metadata"]["source_git_diff_sha256"] is None


def test_ambiguous_legacy_bank_is_rejected():
    payload = bank_payload()
    payload["metadata"]["format_version"] = "talk2dino-e6-rgtp-v1"
    payload["metadata"].pop("e3_config_sha256")
    payload["metadata"].pop("source_git_dirty")
    payload["metadata"].pop("source_git_diff_sha256")
    with pytest.raises(PrototypeBankValidationError, match="missing keys"):
        validate_prototype_bank(payload)


def test_builder_writes_atomic_compact_pilot_and_refuses_overwrite(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        bank_builder,
        "source_git_provenance",
        lambda *args, **kwargs: clean_provenance(),
    )
    config_path, checkpoint_path, source_path = write_builder_inputs(tmp_path)
    output_path = tmp_path / "bank.pth"

    summary = build_prototype_bank(
        source_features_path=source_path,
        output_path=output_path,
        model_config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch_size=1,
        max_annotations=2,
    )
    assert summary["entries"] == 2
    assert summary["complete"] is True
    assert summary["is_pilot"] is True
    assert not list(tmp_path.glob(".bank.pth.*.tmp"))
    with pytest.raises(FileExistsError):
        build_prototype_bank(
            source_features_path=source_path,
            output_path=output_path,
            model_config_path=config_path,
            checkpoint_path=checkpoint_path,
            max_annotations=2,
        )


def test_explicit_dirty_pilot_bank_records_provenance(tmp_path, monkeypatch):
    dirty_provenance = {
        "source_git_commit": "e" * 40,
        "source_git_dirty": True,
        "source_git_diff_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        bank_builder,
        "source_git_provenance",
        lambda *args, **kwargs: dirty_provenance,
    )
    config_path, checkpoint_path, source_path = write_builder_inputs(tmp_path)
    output_path = tmp_path / "dirty-pilot.pth"

    summary = build_prototype_bank(
        source_features_path=source_path,
        output_path=output_path,
        model_config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch_size=1,
        max_annotations=2,
        allow_dirty_source=True,
    )
    payload = torch.load(output_path, map_location="cpu", weights_only=False)
    assert summary["is_pilot"] is True
    assert payload["metadata"]["source_git_commit"] == "e" * 40
    assert payload["metadata"]["source_git_dirty"] is True
    assert payload["metadata"]["source_git_diff_sha256"] == "f" * 64
    with pytest.raises(PrototypeBankValidationError, match="dirty Git worktree"):
        load_prototype_bank(output_path, allow_pilot=True)


def test_dirty_source_cannot_build_a_full_production_bank(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bank_builder,
        "source_git_provenance",
        lambda *args, **kwargs: {
            "source_git_commit": "e" * 40,
            "source_git_dirty": True,
            "source_git_diff_sha256": "f" * 64,
        },
    )
    config_path, checkpoint_path, source_path = write_builder_inputs(tmp_path)
    with pytest.raises(
        PrototypeBankValidationError,
        match="development pilots",
    ):
        build_prototype_bank(
            source_features_path=source_path,
            output_path=tmp_path / "full.pth",
            model_config_path=config_path,
            checkpoint_path=checkpoint_path,
            allow_dirty_source=True,
        )


def test_chunked_topk_exactly_matches_brute_force():
    queries = normalized(4, 11, 11)
    bank = normalized(37, 11, 12)
    actual_scores, actual_indices = exact_chunked_topk(
        queries,
        bank,
        top_k=9,
        chunk_size=7,
    )
    expected_scores, expected_indices = torch.topk(
        queries @ bank.T,
        k=9,
        dim=-1,
    )
    torch.testing.assert_close(actual_scores, expected_scores)
    torch.testing.assert_close(actual_indices, expected_indices)


def test_chunked_topk_handles_float16_bank_and_stable_ties():
    queries = torch.tensor([[1.0, 0.0]])
    bank = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.5, 0.5], [1.0, 0.0]],
        dtype=torch.float16,
    )
    scores, indices = exact_chunked_topk(
        queries,
        bank,
        top_k=3,
        chunk_size=2,
    )
    torch.testing.assert_close(scores, torch.ones(1, 3))
    torch.testing.assert_close(indices, torch.tensor([[0, 1, 3]]))


def test_repeated_image_ids_keep_highest_scoring_caption():
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    indices = torch.tensor([1, 0, 2, 3])
    image_ids = torch.tensor([5, 5, 6, 7])
    kept_scores, kept_indices = deduplicate_by_image(
        scores,
        indices,
        image_ids,
    )
    torch.testing.assert_close(kept_scores, torch.tensor([0.9, 0.7, 0.6]))
    torch.testing.assert_close(kept_indices, torch.tensor([1, 2, 3]))


def test_mmr_is_deterministic_and_selects_diverse_candidates():
    similarities = torch.tensor([0.90, 0.89, 0.80])
    routed = F.normalize(
        torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]),
        dim=-1,
    )
    first = deterministic_mmr(similarities, routed, 2, 0.5)
    second = deterministic_mmr(similarities, routed, 2, 0.5)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, torch.tensor([0, 2]))


def test_spherical_kmeans_is_deterministic_and_empty_cluster_safe():
    embeddings = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    )
    relevance = torch.tensor([0.9, 0.8, 0.7])
    first_labels, first_centers = deterministic_spherical_kmeans(
        embeddings,
        relevance,
        3,
    )
    second_labels, second_centers = deterministic_spherical_kmeans(
        embeddings,
        relevance,
        3,
    )
    torch.testing.assert_close(first_labels, second_labels)
    torch.testing.assert_close(first_centers, second_centers)
    assert torch.isfinite(first_centers).all()
    torch.testing.assert_close(
        first_centers.norm(dim=-1),
        torch.ones(3),
    )

    antipodal = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    _, antipodal_center = deterministic_spherical_kmeans(
        antipodal,
        torch.tensor([0.5, 0.5]),
        1,
    )
    torch.testing.assert_close(
        antipodal_center.norm(dim=-1),
        torch.ones(1),
    )


def test_confidence_is_zero_without_candidates_and_monotonic():
    assert retrieval_confidence(torch.empty(0)).item() == 0
    low = retrieval_confidence(torch.tensor([0.15, 0.20]))
    high = retrieval_confidence(torch.tensor([0.35, 0.40]))
    assert torch.isfinite(torch.stack((low, high))).all()
    assert high > low


def test_anchoring_produces_normalized_class_mode_prototypes():
    base = normalized(2, 8, 20)
    modes = normalized(6, 8, 21).reshape(2, 3, 8)
    confidence = torch.tensor([0.2, 0.9])
    prototypes = anchor_prototypes(base, modes, confidence, 0.30)
    assert prototypes.shape == (2, 3, 8)
    torch.testing.assert_close(
        prototypes.norm(dim=-1),
        torch.ones(2, 3),
    )


def test_normalized_logsumexp_removes_prototype_count_bias():
    single = normalized_logsumexp(torch.tensor([[0.4]]), 0.10)
    repeated = normalized_logsumexp(torch.full((1, 5), 0.4), 0.10)
    torch.testing.assert_close(single, torch.tensor([0.4]))
    torch.testing.assert_close(repeated, torch.tensor([0.4]))
    tiny_temperature = normalized_logsumexp(
        torch.tensor([[1.0, -1.0]]),
        torch.finfo(torch.float32).tiny,
    )
    assert torch.isfinite(tiny_temperature).all()


def test_zero_fusion_and_no_valid_retrieval_exactly_equal_e3_scores_and_masks():
    base = torch.randn(2, 4)
    prototype_scores = torch.randn(2, 4, 3)
    confidence = torch.rand(4)
    valid = torch.ones(4, 3, dtype=torch.bool)
    zero_weight = fuse_prototype_scores(
        base,
        prototype_scores,
        valid,
        confidence,
        prototype_fusion_weight=0,
    )
    assert torch.equal(zero_weight, base)
    assert torch.equal(torch.sigmoid(zero_weight), torch.sigmoid(base))

    no_valid = fuse_prototype_scores(
        base,
        prototype_scores,
        torch.zeros_like(valid),
        confidence,
    )
    assert torch.equal(no_valid, base)
    assert torch.equal(torch.sigmoid(no_valid), torch.sigmoid(base))


def test_one_and_fewer_than_k_valid_prototypes_are_finite():
    base = torch.randn(2, 2)
    prototype_scores = torch.randn(2, 2, 3)
    valid = torch.tensor([[True, False, False], [True, True, False]])
    result = fuse_prototype_scores(
        base,
        prototype_scores,
        valid,
        torch.tensor([0.8, 0.6]),
    )
    assert result.shape == base.shape
    assert torch.isfinite(result).all()
    assert torch.isfinite(torch.sigmoid(result)).all()


def test_generation_cache_and_projection_state_isolation():
    bank = runtime_bank(12, repeated_images=True)
    settings = RGTPSettings(
        prototype_candidate_pool=10,
        prototype_retrieval_count=4,
        retrieval_chunk_size=3,
        retrieval_min_similarity=-1,
        prototype_count=3,
    )
    rgtp = RetrievalGroundedPrototypes(bank, settings)
    raw = normalized(2, 512, 31)
    mapped = normalized(2, 768, 32)
    first = rgtp.generate(raw, mapped)
    second = rgtp.generate(raw.clone(), mapped.clone())
    assert first is second
    assert rgtp.cache_misses == 1
    assert rgtp.cache_hits == 1
    changed_raw = raw.clone()
    changed_raw[0, 0] += 0.01
    rgtp.generate(changed_raw, mapped)
    assert rgtp.cache_misses == 2
    assert first.prototypes.shape == (2, 3, 768)
    assert first.valid_mask.shape == (2, 3)
    assert first.valid_mask.sum(dim=-1).le(3).all()
    assert torch.isfinite(first.prototypes).all()
    assert torch.isfinite(first.confidence).all()

    projection = torch.nn.Linear(512, 768)
    before = set(projection.state_dict())
    projection.rgtp = rgtp
    assert set(projection.state_dict()) == before


def test_generation_no_candidate_and_single_candidate_paths_are_finite():
    payload = bank_payload(entries=3)
    payload["caption_embeddings"] = torch.stack(
        (
            F.normalize(torch.ones(512), dim=0),
            F.normalize(-torch.ones(512), dim=0),
            F.normalize(
                torch.cat((torch.ones(256), -torch.ones(256))),
                dim=0,
            ),
        )
    ).half()
    bank = PrototypeBank(
        payload["caption_embeddings"],
        payload["routed_dino_embeddings"],
        payload["image_ids"],
        payload["annotation_ids"],
        payload["metadata"],
    )
    raw = F.normalize(torch.ones(1, 512), dim=-1)
    mapped = normalized(1, 768, 77)

    one = RetrievalGroundedPrototypes(
        bank,
        RGTPSettings(
            prototype_candidate_pool=3,
            prototype_retrieval_count=3,
            retrieval_chunk_size=2,
            retrieval_min_similarity=0.99,
        ),
    ).generate(raw, mapped)
    assert one.valid_mask.sum().item() == 1
    assert one.confidence.item() > 0
    assert torch.isfinite(one.prototypes).all()

    unrelated_raw = torch.zeros(1, 512)
    unrelated_raw[0, 0] = 1
    none = RetrievalGroundedPrototypes(
        bank,
        RGTPSettings(
            prototype_candidate_pool=3,
            prototype_retrieval_count=3,
            retrieval_chunk_size=2,
            retrieval_min_similarity=1.0,
        ),
    ).generate(unrelated_raw, mapped)
    assert not none.valid_mask.any()
    assert none.confidence.item() == 0
    assert torch.isfinite(none.prototypes).all()


def test_masker_explicit_prototype_path_preserves_exact_fallbacks():
    segmentation_root = (
        Path(__file__).parents[1] / "src" / "open_vocabulary_segmentation"
    )
    sys.path.insert(0, str(segmentation_root))
    try:
        from models.dinotext.masker import DINOTextMasker
    finally:
        sys.path.pop(0)

    masker = DINOTextMasker()
    image = torch.randn(2, 6, 3, 4)
    text = normalized(4, 6, 40)
    prototypes = normalized(12, 6, 41).reshape(4, 3, 6)
    valid = torch.ones(4, 3, dtype=torch.bool)
    confidence = torch.rand(4)
    baseline_mask, baseline_scores = masker.forward_seg(image, text)
    zero_mask, zero_scores = masker.forward_seg_with_prototypes(
        image,
        text,
        prototypes,
        valid,
        confidence,
        prototype_fusion_weight=0,
    )
    assert torch.equal(zero_scores, baseline_scores)
    assert torch.equal(zero_mask, baseline_mask)
    empty_mask, empty_scores = masker.forward_seg_with_prototypes(
        image,
        text,
        prototypes,
        torch.zeros_like(valid),
        confidence,
    )
    assert torch.equal(empty_scores, baseline_scores)
    assert torch.equal(empty_mask, baseline_mask)

    actual_mask, actual_scores = masker.forward_seg_with_prototypes(
        image,
        text,
        prototypes,
        valid,
        confidence,
        prototype_temperature=0.10,
        prototype_fusion_weight=0.25,
    )
    normalized_image = F.normalize(image, dim=1)
    expected_base = torch.einsum(
        "b d h w, n d -> b n h w",
        normalized_image,
        text,
    )
    expected_modes = torch.einsum(
        "b d h w, n k d -> b n k h w",
        normalized_image,
        prototypes,
    )
    expected_grounded = 0.10 * (
        torch.logsumexp(expected_modes / 0.10, dim=2)
        - torch.log(torch.tensor(3.0))
    )
    beta = 0.25 * confidence[None, :, None, None]
    expected_scores = (1 - beta) * expected_base + beta * expected_grounded
    torch.testing.assert_close(actual_scores, expected_scores)
    torch.testing.assert_close(actual_mask, torch.sigmoid(expected_scores))


def test_e6_config_inherits_e3_without_experiment_contamination():
    e3_path = Path(
        "src/open_vocabulary_segmentation/configs/stuff/"
        "dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_tau010.yml"
    )
    e6_path = Path(
        "src/open_vocabulary_segmentation/configs/stuff/"
        "dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_rgtp.yml"
    )
    e3 = yaml.safe_load(e3_path.read_text())
    e6 = yaml.safe_load(e6_path.read_text())
    assert e6["_base_"] == e3_path.name
    assert set(e6) == {"_base_", "model"}
    assert set(e6["model"]) == {"retrieval_grounded_prototypes"}
    rgtp_config = dict(e6["model"]["retrieval_grounded_prototypes"])
    assert rgtp_config.pop("bank_path") == (
        "${oc.env:TALK2DINO_PROTOTYPE_BANK}"
    )
    assert RGTPSettings.from_mapping(rgtp_config) == RGTPSettings()
    assert e3["model"]["proj_name"] == (
        "vitb_mlp_infonce_paired_soft_routing_tau010"
    )
    serialized = e6_path.read_text()
    for contamination in (
        "all_pairs_max",
        "all_pairs_lse",
        "multi_positive",
        "dense_consistency",
        "paired_soft_routing_rdcd",
    ):
        assert contamination not in serialized
