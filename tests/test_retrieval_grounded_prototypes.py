import copy
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

import build_e6_prototype_bank as bank_builder
import src.retrieval_grounded_prototypes as rgtp_module
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
    select_adaptive_unique_image_candidates,
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
            "source_feature_sha256": "9" * 64,
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


def test_source_feature_sha256_validation_and_identity(tmp_path):
    source_path = tmp_path / "train.pth"
    source_path.write_bytes(b"synthetic source archive")
    payload = bank_payload()
    payload["metadata"]["source_feature_sha256"] = sha256_file(source_path)
    summary = validate_prototype_bank(
        payload,
        expected_source_features_path=source_path,
    )
    assert summary["source_feature_sha256"] == sha256_file(source_path)

    first = PrototypeBank(
        payload["caption_embeddings"],
        payload["routed_dino_embeddings"],
        payload["image_ids"],
        payload["annotation_ids"],
        payload["metadata"],
    )
    changed = copy.deepcopy(payload["metadata"])
    changed["source_feature_sha256"] = "8" * 64
    second = PrototypeBank(
        payload["caption_embeddings"],
        payload["routed_dino_embeddings"],
        payload["image_ids"],
        payload["annotation_ids"],
        changed,
    )
    assert first.identity != second.identity


def test_malformed_and_mismatched_source_feature_sha256_are_rejected(tmp_path):
    malformed = bank_payload()
    malformed["metadata"]["source_feature_sha256"] = "not-a-sha256"
    with pytest.raises(PrototypeBankValidationError, match="source_feature"):
        validate_prototype_bank(malformed)

    source_path = tmp_path / "train.pth"
    source_path.write_bytes(b"different archive")
    with pytest.raises(PrototypeBankValidationError, match="source feature SHA256"):
        validate_prototype_bank(
            bank_payload(),
            expected_source_features_path=source_path,
        )


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


def test_v2_bank_is_explicitly_rejected():
    payload = bank_payload()
    payload["metadata"]["format_version"] = "talk2dino-e6-rgtp-v2"
    payload["metadata"].pop("source_feature_sha256")
    with pytest.raises(PrototypeBankValidationError, match="v1/v2 banks"):
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
    assert summary["source_feature_sha256"] == sha256_file(source_path)
    assert not list(tmp_path.glob(".bank.pth.*.tmp"))
    with pytest.raises(FileExistsError):
        build_prototype_bank(
            source_features_path=source_path,
            output_path=output_path,
            model_config_path=config_path,
            checkpoint_path=checkpoint_path,
            max_annotations=2,
        )


def test_source_archive_mutation_before_publication_fails(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        bank_builder,
        "source_git_provenance",
        lambda *args, **kwargs: clean_provenance(),
    )
    config_path, checkpoint_path, source_path = write_builder_inputs(tmp_path)
    output_path = tmp_path / "mutated-source-bank.pth"
    real_sha256_file = bank_builder.sha256_file
    source_hash_calls = 0

    def changing_source_digest(path):
        nonlocal source_hash_calls
        if Path(path).resolve() == source_path.resolve():
            source_hash_calls += 1
            if source_hash_calls > 1:
                return "0" * 64
        return real_sha256_file(path)

    monkeypatch.setattr(
        bank_builder,
        "sha256_file",
        changing_source_digest,
    )
    with pytest.raises(RuntimeError, match="source feature archive changed"):
        build_prototype_bank(
            source_features_path=source_path,
            output_path=output_path,
            model_config_path=config_path,
            checkpoint_path=checkpoint_path,
            batch_size=1,
            max_annotations=2,
        )
    assert not output_path.exists()


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


def test_adaptive_retrieval_matches_fixed_256_prefix_when_sufficient():
    bank = runtime_bank(300)
    raw = normalized(1, 512, 101)
    mapped = normalized(1, 768, 102)
    common = {
        "prototype_candidate_pool": 256,
        "prototype_retrieval_count": 64,
        "retrieval_chunk_size": 71,
        "retrieval_min_similarity": -1,
    }
    fixed = RetrievalGroundedPrototypes(
        bank,
        RGTPSettings(prototype_candidate_pool_max=256, **common),
    ).generate(raw, mapped)
    adaptive = RetrievalGroundedPrototypes(
        bank,
        RGTPSettings(prototype_candidate_pool_max=300, **common),
    ).generate(raw, mapped)

    torch.testing.assert_close(
        adaptive.retrieval_indices,
        fixed.retrieval_indices,
    )
    torch.testing.assert_close(adaptive.retrieval_scores, fixed.retrieval_scores)
    torch.testing.assert_close(adaptive.prototypes, fixed.prototypes)
    torch.testing.assert_close(adaptive.confidence, fixed.confidence)
    torch.testing.assert_close(
        adaptive.candidate_pool_used,
        torch.tensor([256]),
    )


def test_repeated_images_force_adaptive_expansion_from_256_to_512():
    scores = torch.linspace(1.0, 0.5, 512)
    indices = torch.arange(512, dtype=torch.int64)
    image_ids = torch.arange(512, dtype=torch.int64) // 8
    result = select_adaptive_unique_image_candidates(
        scores,
        indices,
        image_ids,
        initial_pool=256,
        maximum_pool=512,
        retrieval_count=64,
        minimum_similarity=0.0,
    )
    selected_scores, selected_indices, pool_used, deduplicated, valid = result
    assert pool_used == 512
    assert deduplicated == 64
    assert valid == 64
    assert len(selected_scores) == len(selected_indices) == 64


def test_adaptive_expansion_uses_deterministic_doubling_prefixes(
    monkeypatch,
):
    observed_prefixes = []
    original = rgtp_module.deduplicate_by_image

    def recording_deduplication(scores, indices, image_ids):
        observed_prefixes.append(len(scores))
        return original(scores, indices, image_ids)

    monkeypatch.setattr(
        rgtp_module,
        "deduplicate_by_image",
        recording_deduplication,
    )
    scores = torch.linspace(1.0, 0.5, 32)
    indices = torch.arange(32, dtype=torch.int64)
    select_adaptive_unique_image_candidates(
        scores,
        indices,
        indices.clone(),
        initial_pool=4,
        maximum_pool=32,
        retrieval_count=17,
        minimum_similarity=0.0,
    )
    assert observed_prefixes == [4, 8, 16, 32]


def test_adaptive_expansion_stops_when_prefix_tail_is_below_threshold():
    scores = torch.cat((torch.full((50,), 0.8), torch.full((462,), 0.1)))
    indices = torch.arange(512, dtype=torch.int64)
    result = select_adaptive_unique_image_candidates(
        scores,
        indices,
        indices.clone(),
        initial_pool=256,
        maximum_pool=512,
        retrieval_count=64,
        minimum_similarity=0.5,
    )
    selected_scores, _, pool_used, deduplicated, valid = result
    assert pool_used == 256
    assert deduplicated == 256
    assert valid == len(selected_scores) == 50


def test_adaptive_retrieval_stops_at_maximum_with_finite_fallback():
    scores = torch.linspace(1.0, 0.5, 300)
    indices = torch.arange(300, dtype=torch.int64)
    image_ids = torch.arange(300, dtype=torch.int64) % 10
    arguments = {
        "initial_pool": 128,
        "maximum_pool": 256,
        "retrieval_count": 64,
        "minimum_similarity": 0.0,
    }
    first = select_adaptive_unique_image_candidates(
        scores,
        indices,
        image_ids,
        **arguments,
    )
    second = select_adaptive_unique_image_candidates(
        scores,
        indices,
        image_ids,
        **arguments,
    )
    assert first[2:] == (256, 10, 10)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.isfinite(first[0]).all()


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


def test_vectorized_fusion_matches_independent_mixed_validity_reference():
    torch.manual_seed(61)
    base = torch.randn(2, 3, 4)
    prototype_scores = torch.randn(2, 3, 4, 3)
    prototype_scores[..., 0, :] = 1e4
    valid = torch.tensor(
        [
            [False, False, False],
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    )
    confidence = torch.tensor([0.9, 0.8, 0.7, 0.6])
    actual = fuse_prototype_scores(
        base,
        prototype_scores,
        valid,
        confidence,
        prototype_fusion_weight=0.25,
        prototype_temperature=0.10,
    )

    expected = base.clone()
    for class_index in range(4):
        class_scores = prototype_scores[..., class_index, :][
            ..., valid[class_index]
        ]
        if class_scores.shape[-1] == 0:
            continue
        grounded = 0.10 * (
            torch.logsumexp(class_scores / 0.10, dim=-1)
            - math.log(class_scores.shape[-1])
        )
        beta = 0.25 * confidence[class_index]
        expected[..., class_index] = (
            (1 - beta) * base[..., class_index] + beta * grounded
        )

    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual[..., 0], base[..., 0])
    assert actual.shape == base.shape
    assert actual.dtype == base.dtype
    assert actual.device == base.device
    assert torch.isfinite(actual).all()


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
    for diagnostic in (
        first.candidate_pool_used,
        first.deduplicated_candidate_count,
        first.threshold_valid_count,
        first.selected_retrieval_count,
    ):
        assert diagnostic.shape == (2,)
        assert diagnostic.dtype == torch.int64
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


def test_masker_explicit_prototype_path_preserves_exact_fallbacks(
    monkeypatch,
):
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
    valid = torch.tensor(
        [
            [False, False, False],
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    )
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

    original_sigmoid = torch.sigmoid
    sigmoid_calls = 0

    def counting_sigmoid(value):
        nonlocal sigmoid_calls
        sigmoid_calls += 1
        return original_sigmoid(value)

    monkeypatch.setattr(torch, "sigmoid", counting_sigmoid)
    actual_mask, actual_scores = masker.forward_seg_with_prototypes(
        image,
        text,
        prototypes,
        valid,
        confidence,
        prototype_temperature=0.10,
        prototype_fusion_weight=0.25,
    )
    assert sigmoid_calls == 1
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
    expected_scores = expected_base.clone()
    for class_index in range(4):
        class_modes = expected_modes[:, class_index, valid[class_index]]
        if class_modes.shape[1] == 0:
            continue
        expected_grounded = 0.10 * (
            torch.logsumexp(class_modes / 0.10, dim=1)
            - math.log(class_modes.shape[1])
        )
        beta = 0.25 * confidence[class_index]
        expected_scores[:, class_index] = (
            (1 - beta) * expected_base[:, class_index]
            + beta * expected_grounded
        )
    torch.testing.assert_close(actual_scores, expected_scores)
    assert torch.equal(actual_scores[:, 0], expected_base[:, 0])
    torch.testing.assert_close(actual_mask, original_sigmoid(expected_scores))
    assert torch.isfinite(actual_scores).all()
    assert torch.isfinite(actual_mask).all()


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
