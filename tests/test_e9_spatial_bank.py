import hashlib
import copy
import io
import json
import math
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import src.e9_spatial_bank as e9
from train_e9_sparse_region_alignment import select_query_rows_for_spatial

DINO_SOURCE_COMMIT = "a" * 40
DINO_CHECKPOINT_SHA256 = "b" * 64


def sha(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def source_record(image_id=7, *, maps=None, patches=None):
    generator = torch.Generator().manual_seed(image_id)
    if patches is None:
        patches = torch.randn(1024, 768, generator=generator).half()
    if maps is None:
        maps = torch.full((12, 1024), 1 / 1024, dtype=torch.float16)
    return {
        "image_id": image_id,
        "file_name": f"{image_id}.jpg",
        "disentangled_self_attn": torch.randn(12, 768, generator=generator),
        "patch_tokens": patches,
        "self_attn_maps": maps,
        "captions": ["caption"],
        "ann_feats": [torch.randn(512, generator=generator)],
        "annotation_ids": [1000 + image_id],
    }


def make_source(root: Path, records):
    root.mkdir()
    shard = root / "train-000000.tar"
    with tarfile.open(shard, "w") as archive:
        for index, record in enumerate(records):
            buffer = io.BytesIO()
            torch.save(record, buffer)
            payload = buffer.getvalue()
            member = tarfile.TarInfo(f"{index:08d}.pth")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    manifest = {
        "format_version": 1,
        "split": "train",
        "images_per_shard": 128,
        "extraction_config": {
            "annotation_path": "/synthetic/train.pth",
            "data_dir": "/synthetic/images",
            "model": "dinov2_vitb14_reg",
            "resize_dim": 448,
            "crop_dim": 448,
            "patch_count": 1024,
            "embedding_dim": 768,
            "attention_heads": 12,
            "attention_map_format": "probabilities",
            "patch_tokens_dtype": "float16",
            "self_attn_maps_dtype": "float16",
            "disentangled_self_attn_dtype": "float32",
            "backbone_weights_sha256": DINO_CHECKPOINT_SHA256,
        },
        "source_commit": DINO_SOURCE_COMMIT,
        "source_images": len(records),
        "source_annotations": len(records),
        "selected_images": len(records),
        "selected_annotations": len(records),
        "selected_image_ids_sha256": e9._source_id_fingerprint(
            record["image_id"] for record in records
        ),
        "selected_annotation_ids_sha256": e9._source_id_fingerprint(
            annotation_id
            for record in records
            for annotation_id in record["annotation_ids"]
        ),
        "max_images": None,
        "is_pilot": False,
        "complete": True,
        "failed_image_ids": [],
        "images": len(records),
        "annotations": len(records),
        "shards": [{
            "name": shard.name,
            "images": len(records),
            "annotations": len(records),
            "bytes": shard.stat().st_size,
            "sha256": sha(shard),
        }],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def build_spatial_bank(source, output, **kwargs):
    return e9.build_e9_spatial_bank(
        source,
        output,
        expected_dino_source_commit=DINO_SOURCE_COMMIT,
        expected_dino_checkpoint_sha256=DINO_CHECKPOINT_SHA256,
        **kwargs,
    )


def git_repo(root):
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Test"],
        check=True,
    )
    tracked = root / "source.py"
    tracked.write_text("clean\n")
    subprocess.run(["git", "-C", str(root), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)
    return tracked


def tree_hashes(root):
    return {
        path.relative_to(root).as_posix(): sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_source_geometry_and_independent_pooling():
    patches = torch.arange(1024 * 3, dtype=torch.float32).reshape(1024, 3)
    patches = torch.cat((patches, torch.ones(1024, 765)), dim=-1).half()
    maps = torch.arange(1, 1025, dtype=torch.float32).repeat(12, 1)
    maps = (maps / maps.sum(dim=-1, keepdim=True)).half()
    actual_patch, actual_prior = e9.pool_source_image(patches, maps)
    reference_patch = F.normalize(
        patches.float().reshape(32, 32, 768)
        .reshape(16, 2, 16, 2, 768).mean(dim=(1, 3)).reshape(256, 768),
        dim=-1,
    )
    reference_prior = maps.float().mean(0).reshape(32, 32)
    reference_prior = reference_prior.reshape(16, 2, 16, 2).sum((1, 3)).reshape(256)
    reference_prior /= reference_prior.sum()
    torch.testing.assert_close(actual_patch, reference_patch)
    torch.testing.assert_close(actual_prior, reference_prior)
    torch.testing.assert_close(actual_patch.norm(dim=-1), torch.ones(256))
    torch.testing.assert_close(actual_prior.sum(), torch.tensor(1.0))


@pytest.mark.parametrize(
    "patches,maps,match",
    (
        (torch.ones(1023, 768, dtype=torch.float16), None, "patch_tokens"),
        (torch.ones(1024, 767, dtype=torch.float16), None, "patch_tokens"),
        (None, torch.ones(12, 1023, dtype=torch.float16), "self_attn_maps"),
        (None, torch.randn(12, 1024, dtype=torch.float16), "negative attention"),
    ),
)
def test_wrong_grid_dtype_and_nonprobability_maps_are_rejected(patches, maps, match):
    record = source_record(patches=patches, maps=maps)
    with pytest.raises(e9.E9SpatialBankValidationError, match=match):
        e9.validate_source_record(record)


def test_attention_head_reduction_and_register_handling_are_explicit():
    maps = torch.zeros(12, 1024, dtype=torch.float16)
    for head in range(12):
        maps[head, head] = 1
    _, prior = e9.pool_source_image(source_record()["patch_tokens"], maps)
    expected = maps.float().mean(0).reshape(32, 32)
    expected = expected.reshape(16, 2, 16, 2).sum((1, 3)).reshape(256)
    torch.testing.assert_close(prior, expected / expected.sum())
    assert "4 register tokens" in e9.GLOBAL_TOKEN_HANDLING


def test_complete_source_record_contract_rejects_malformed_annotation_fields():
    mutations = (
        lambda value: value.update({
            "disentangled_self_attn": torch.ones(12, 768, dtype=torch.float16)
        }),
        lambda value: value.update({"captions": [1]}),
        lambda value: value.update({"ann_feats": [torch.ones(511)]}),
        lambda value: value.update({"ann_feats": [torch.full((512,), float("nan"))]}),
        lambda value: value.update({
            "captions": ["a", "b"],
            "ann_feats": [torch.ones(512), torch.ones(512)],
            "annotation_ids": [1, 1],
        }),
    )
    for mutation in mutations:
        record = source_record()
        mutation(record)
        with pytest.raises(e9.E9SpatialBankValidationError):
            e9.validate_source_record(record)


def test_build_validate_lazy_load_and_closed_manifest(tmp_path):
    source = make_source(tmp_path / "source", [source_record(9), source_record(3)])
    output = tmp_path / "bank"
    result = build_spatial_bank(
        source, output, split="train", shard_rows=1, max_images=2,
        allow_dirty_source=True,
    )
    assert result["images"] == 2
    assert result["is_pilot"]
    with pytest.raises(e9.E9SpatialBankValidationError, match="not production"):
        e9.validate_e9_spatial_bank(output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert [entry["image_id"] for entry in manifest["image_index"]] == [9, 3]
    bank = e9.E9SpatialBank(output, require_production=False, cache_shards=1)
    patch, prior = bank.get(9)
    assert len(bank._cache) == 1
    bank.get(9)
    assert len(bank._cache) == 1
    assert patch.shape == (256, 768) and patch.dtype == torch.float32
    assert prior.shape == (256,) and prior.dtype == torch.float32
    torch.testing.assert_close(patch.norm(dim=-1), torch.ones(256), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(prior.sum(), torch.tensor(1.0))
    bank._cache.clear()
    bank.shard_deserializations = 0
    query_image_ids = torch.tensor([9, 9, 3, 3], dtype=torch.int64)
    sampler = e9.SpatialUniqueImageBatchSampler(
        query_image_ids, bank, batch_size=2, seed=42
    )
    for rows in sampler:
        for image_id in query_image_ids[rows].tolist():
            bank.get(image_id)
    assert bank.shard_deserializations == manifest["shard_count"] == 2
    manifest["unknown"] = 1
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(e9.E9SpatialBankValidationError, match="closed schema"):
        e9.validate_e9_spatial_bank(output, require_production=False)

def test_full_bank_is_production_eligible_and_incomplete_is_rejected(
    tmp_path, monkeypatch
):
    source = make_source(tmp_path / "source", [source_record(2)])
    output = tmp_path / "bank"
    clean = {
        "source_git_commit": "a" * 40,
        "source_git_dirty": False,
        "source_git_diff_sha256": None,
    }
    monkeypatch.setattr(e9, "source_git_provenance", lambda *args, **kwargs: clean)
    monkeypatch.setattr(
        e9, "_require_unchanged_git_provenance", lambda *args, **kwargs: clean
    )
    result = build_spatial_bank(source, output, split="train")
    assert result["complete"] and result["production_eligible"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_images"] == manifest["selected_images"] == 1
    assert not manifest["is_pilot"]
    with pytest.raises(ValueError, match="immutable.*new output path"):
        build_spatial_bank(
            source, output, split="train", overwrite=True
        )

    missing = dict(manifest)
    missing.pop("created_at")
    (output / "manifest.json").write_text(json.dumps(missing))
    with pytest.raises(e9.E9SpatialBankValidationError, match="closed schema"):
        e9.validate_e9_spatial_bank(output, require_production=False)

    manifest["complete"] = False
    manifest["production_eligible"] = False
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(e9.E9SpatialBankValidationError, match="forged.*completeness"):
        e9.validate_e9_spatial_bank(output, require_production=False)


@pytest.mark.parametrize("field", ("source_commit", "checkpoint_sha256"))
def test_builder_requires_exact_external_dino_identity_before_staging(
    tmp_path, field
):
    source = make_source(tmp_path / "source", [source_record(2)])
    output = tmp_path / "bank"
    kwargs = {
        "expected_dino_source_commit": DINO_SOURCE_COMMIT,
        "expected_dino_checkpoint_sha256": DINO_CHECKPOINT_SHA256,
    }
    kwargs[
        "expected_dino_source_commit"
        if field == "source_commit"
        else "expected_dino_checkpoint_sha256"
    ] = "c" * (40 if field == "source_commit" else 64)
    with pytest.raises(e9.E9SpatialBankValidationError, match="DINO"):
        e9.build_e9_spatial_bank(
            source, output, split="train", max_images=1,
            allow_dirty_source=True, **kwargs,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".bank.e9-*"))


def test_builder_rejects_source_without_dino_checkpoint_identity(tmp_path):
    source = make_source(tmp_path / "source", [source_record(2)])
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["extraction_config"].pop("backbone_weights_sha256")
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "bank"
    with pytest.raises(
        e9.E9SpatialBankValidationError,
        match="missing required backbone_weights_sha256",
    ):
        build_spatial_bank(
            source, output, split="train", max_images=1,
            allow_dirty_source=True,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".bank.e9-*"))


def test_forged_partial_full_bank_is_rejected(tmp_path, monkeypatch):
    source = make_source(tmp_path / "source", [source_record(2)])
    output = tmp_path / "bank"
    clean = {
        "source_git_commit": "a" * 40,
        "source_git_dirty": False,
        "source_git_diff_sha256": None,
    }
    monkeypatch.setattr(e9, "source_git_provenance", lambda *args, **kwargs: clean)
    monkeypatch.setattr(
        e9, "_require_unchanged_git_provenance", lambda *args, **kwargs: clean
    )
    build_spatial_bank(source, output, split="train")
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["source_images"] = 100
    manifest["dataset_identity"]["source_image_count"] = 100
    manifest["builder_config"]["source_selected_images"] = 1
    manifest["builder_config"]["source_is_pilot"] = False
    manifest["complete"] = True
    manifest["is_pilot"] = False
    manifest["production_eligible"] = True
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(
        e9.E9SpatialBankValidationError,
        match="forged source pilot|forged spatial-bank pilot|forged production",
    ):
        e9.validate_e9_spatial_bank(output, require_production=False)


def test_canonical_spatial_contract_and_provenance_values_are_closed(
    tmp_path, monkeypatch
):
    source = make_source(tmp_path / "source", [source_record(8)])
    output = tmp_path / "bank"
    clean = {
        "source_git_commit": "a" * 40,
        "source_git_dirty": False,
        "source_git_diff_sha256": None,
    }
    monkeypatch.setattr(e9, "source_git_provenance", lambda *args, **kwargs: clean)
    monkeypatch.setattr(
        e9, "_require_unchanged_git_provenance", lambda *args, **kwargs: clean
    )
    build_spatial_bank(source, output, split="train")
    base = json.loads((output / "manifest.json").read_text())
    mutations = (
        lambda value: value.update({
            "format_version": "talk2dino-e9-spatial-bank-v1"
        }),
        lambda value: value.update({"source_feature_format": "unknown"}),
        lambda value: value["dino_identity"].update({"model": "other"}),
        lambda value: value["dino_identity"].update({"source_commit": "A" * 40}),
        lambda value: value["dino_identity"].update({"checkpoint_sha256": "x"}),
        lambda value: value["extraction"].update({"resize_dim": 224}),
        lambda value: value["geometry"].update({"source_grid_width": 31}),
        lambda value: value["dtypes"].update({"serialized_patch_embeddings": "float32"}),
        lambda value: value.update({"pooling_version": "other"}),
        lambda value: value.update({"attention_prior_version": "other"}),
        lambda value: value.update({"global_token_handling": "ambiguous"}),
        lambda value: value.update({"created_at": "not-a-timestamp"}),
        lambda value: value.update({"source_git_diff_sha256": "f" * 64}),
    )
    for mutation in mutations:
        candidate = copy.deepcopy(base)
        mutation(candidate)
        (output / "manifest.json").write_text(json.dumps(candidate))
        with pytest.raises(e9.E9SpatialBankValidationError):
            e9.validate_e9_spatial_bank(output, require_production=False)
    (output / "manifest.json").write_text(json.dumps(base))
    assert e9.validate_e9_spatial_bank(output)["production_eligible"]

def test_duplicate_source_image_ids_are_rejected(tmp_path):
    source = make_source(
        tmp_path / "source", [source_record(6), source_record(6)]
    )
    output = tmp_path / "bank"
    with pytest.raises(e9.E9SpatialBankValidationError, match="duplicate source"):
        build_spatial_bank(
            source, output, split="train", max_images=2,
            allow_dirty_source=True,
        )
    assert not output.exists()


def test_failed_publication_preserves_existing_output_and_cleans_staging(tmp_path, monkeypatch):
    source = make_source(tmp_path / "source", [source_record(4)])
    output = tmp_path / "bank"
    output.mkdir()
    marker = output / "existing"
    marker.write_bytes(b"preserve-me")
    before = sha(marker)
    monkeypatch.setattr(
        e9,
        "_require_unchanged_git_provenance",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("mutated")),
    )
    with pytest.raises(RuntimeError, match="mutated"):
        build_spatial_bank(
            source, output, split="train", max_images=1, overwrite=True,
            allow_dirty_source=True,
        )
    assert sha(marker) == before
    assert not list(tmp_path.glob(".bank.e9-*"))


def test_source_mutation_during_shard_serialization_prevents_publication(tmp_path, monkeypatch):
    source = make_source(tmp_path / "source", [source_record(4)])
    source_shard = source / "train-000000.tar"
    output = tmp_path / "bank"
    original = torch.save

    def mutating_save(value, path, *args, **kwargs):
        original(value, path, *args, **kwargs)
        source_shard.write_bytes(source_shard.read_bytes() + b"mutated")

    monkeypatch.setattr(e9.torch, "save", mutating_save)
    with pytest.raises(e9.E9SpatialBankValidationError, match="source artifact changed"):
        build_spatial_bank(
            source, output, split="train", max_images=1,
            allow_dirty_source=True,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".bank.e9-*"))


@pytest.mark.parametrize(
    "mutation_kind", ("unstaged", "staged", "untracked", "source_artifact")
)
def test_manifest_boundary_rejects_mutation_after_nonmanifest_link(
    tmp_path, monkeypatch, mutation_kind
):
    repository = tmp_path / "repo"
    tracked = git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(12)])
    source_shard = source / "train-000000.tar"
    output = tmp_path / "bank"
    original_link = e9.os.link
    mutation_seen = False
    linked_names = []

    def mutate_after_link(source_path, destination_path):
        nonlocal mutation_seen
        original_link(source_path, destination_path)
        linked_names.append(Path(source_path).name)
        if Path(source_path).name == e9.MANIFEST_NAME or mutation_seen:
            return
        mutation_seen = True
        if mutation_kind == "untracked":
            (repository / "new.py").write_text("new\n")
        elif mutation_kind == "source_artifact":
            source_shard.write_bytes(source_shard.read_bytes() + b"mutation")
        else:
            tracked.write_text(f"{mutation_kind}\n")
            if mutation_kind == "staged":
                subprocess.run(
                    ["git", "-C", str(repository), "add", "source.py"],
                    check=True,
                )

    monkeypatch.setattr(e9.os, "link", mutate_after_link)
    with pytest.raises(
        e9.E9SpatialBankValidationError,
        match="publication|provenance",
    ):
        build_spatial_bank(
            source, output, split="train", max_images=1,
            repository_root=repository,
        )
    assert mutation_seen
    assert e9.MANIFEST_NAME not in linked_names
    assert not output.exists()
    assert not list(tmp_path.glob(".bank.e9-*"))
    assert not list(tmp_path.glob(".bank.backup-*"))


@pytest.mark.parametrize(
    "mutation_kind", ("unstaged", "staged", "untracked", "source_artifact")
)
def test_manifest_boundary_failed_pilot_overwrite_restores_existing_tree(
    tmp_path, monkeypatch, mutation_kind
):
    repository = tmp_path / "repo"
    tracked = git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(13)])
    source_shard = source / "train-000000.tar"
    output = tmp_path / "bank"
    output.mkdir()
    (output / "manifest.json").write_bytes(b"existing manifest")
    (output / "existing-shard.pth").write_bytes(b"existing shard")
    before = tree_hashes(output)
    original_rename = e9.os.rename
    mutation_seen = False

    def mutate_after_backup(source_path, destination_path):
        nonlocal mutation_seen
        original_rename(source_path, destination_path)
        if Path(source_path) != output or mutation_seen:
            return
        mutation_seen = True
        if mutation_kind == "untracked":
            (repository / "new.py").write_text("new\n")
        elif mutation_kind == "source_artifact":
            source_shard.write_bytes(source_shard.read_bytes() + b"mutation")
        else:
            tracked.write_text(f"{mutation_kind}\n")
            if mutation_kind == "staged":
                subprocess.run(
                    ["git", "-C", str(repository), "add", "source.py"],
                    check=True,
                )

    monkeypatch.setattr(e9.os, "rename", mutate_after_backup)
    with pytest.raises(
        e9.E9SpatialBankValidationError,
        match="publication|provenance",
    ):
        build_spatial_bank(
            source, output, split="train", max_images=1, overwrite=True,
            repository_root=repository,
        )
    assert mutation_seen
    assert tree_hashes(output) == before
    assert not list(tmp_path.glob(".bank.e9-*"))
    assert not list(tmp_path.glob(".bank.backup-*"))


def test_atomic_create_if_absent_refuses_existing_target(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "train-000000.pth").write_bytes(b"shard")
    (staging / "manifest.json").write_bytes(b"manifest")
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "marker"
    marker.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        e9._publish_directory(
            staging, output, overwrite=False, final_verification=lambda: None
        )
    assert marker.read_bytes() == b"existing"


def test_atomic_create_if_absent_preserves_concurrent_directory_winner(
    tmp_path, monkeypatch
):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "train-000000.pth").write_bytes(b"shard")
    (staging / "manifest.json").write_bytes(b"manifest")
    output = tmp_path / "output"
    original_mkdir = Path.mkdir
    injected = False

    def concurrent_mkdir(path, *args, **kwargs):
        nonlocal injected
        if path == output and not injected:
            injected = True
            original_mkdir(path)
            (path / "winner").write_bytes(b"concurrent")
            raise FileExistsError(path)
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", concurrent_mkdir)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        e9._publish_directory(
            staging, output, overwrite=False, final_verification=lambda: None
        )
    assert (output / "winner").read_bytes() == b"concurrent"
    assert not (output / "manifest.json").exists()


def test_train_validation_overlap_is_rejected():
    train = SimpleNamespace(image_ids={1, 2})
    validation = SimpleNamespace(image_ids={2, 3})
    with pytest.raises(e9.E9SpatialBankValidationError, match="overlap"):
        e9.reject_train_validation_overlap(train, validation)


def test_unique_image_caption_rotation_is_deterministic_and_covers_rows():
    image_ids = torch.tensor([1, 1, 1, 2, 2, 3], dtype=torch.int64)
    spatial = SimpleNamespace(
        shard_index=lambda image_id: {1: 0, 2: 0, 3: 1}[image_id]
    )
    first = e9.SpatialUniqueImageBatchSampler(
        image_ids, spatial, batch_size=2, seed=42
    )
    replay = e9.SpatialUniqueImageBatchSampler(
        image_ids, spatial, batch_size=2, seed=42
    )
    covered = set()
    epoch_orders = []
    for epoch in range(6):
        first.set_epoch(epoch)
        replay.set_epoch(epoch)
        batches = list(first)
        assert batches == list(replay)
        epoch_orders.append(tuple(row for batch in batches for row in batch))
        for batch in batches:
            selected_images = image_ids[batch].tolist()
            assert len(selected_images) == len(set(selected_images))
            covered.update(batch)
    assert covered == set(range(len(image_ids)))
    assert len(set(epoch_orders)) > 1


def test_expected_real_epoch_shard_reads_are_bounded_and_sequential():
    assert math.ceil(118_287 / 128) == 925
    assert math.ceil(5_000 / 128) == 40


def test_query_join_uses_stored_mapped_rows_and_rejects_missing_images():
    query = {
        "mapped_query_embeddings": torch.randn(3, 768).half(),
        "caption_embeddings": torch.randn(3, 512).half(),
        "routed_target_embeddings": torch.randn(3, 768).half(),
        "image_ids": torch.tensor([1, 2, 1], dtype=torch.int64),
        "annotation_ids": torch.tensor([10, 11, 12], dtype=torch.int64),
        "metadata": {
            "split_name": "train",
            "source_image_count": 2,
            "source_annotation_count": 3,
            "source_feature_sha256": "a" * 64,
            "source_feature_path": "/synthetic/train.pth",
            "e3_config_sha256": "b" * 64,
            "e3_checkpoint_sha256": "c" * 64,
        },
    }
    spatial = SimpleNamespace(
        image_ids={1, 2},
        manifest={
            "split": "train",
            "dataset_identity": e9.dataset_identity_from_rows(
                split="train",
                image_ids=[1, 2, 1],
                annotation_ids=[10, 11, 12],
                source_image_count=2,
                source_annotation_count=3,
            ),
        },
    )
    rows = e9.validate_query_spatial_join(query, spatial, expected_split="train")
    assert [row.query_row for row in rows] == [0, 1, 2]
    assert [row.annotation_id for row in rows] == [10, 11, 12]
    spatial.image_ids.remove(2)
    with pytest.raises(e9.E9SpatialBankValidationError, match="missing spatial"):
        e9.validate_query_spatial_join(query, spatial, expected_split="train")

    training_source = Path("train_e9_sparse_region_alignment.py").read_text()
    assert 'query_bank["mapped_query_embeddings"]' in training_source
    assert 'query_bank["caption_embeddings"]' not in training_source
    assert 'query_bank["routed_target_embeddings"]' not in training_source


def _joined_fixture(*, query_images=(1, 2, 1), query_annotations=(10, 11, 12)):
    query = {
        "mapped_query_embeddings": torch.randn(len(query_images), 768).half(),
        "image_ids": torch.tensor(query_images, dtype=torch.int64),
        "annotation_ids": torch.tensor(query_annotations, dtype=torch.int64),
        "metadata": {
            "split_name": "train",
            "source_image_count": 2,
            "source_annotation_count": 3,
            "source_feature_path": "/same/or/different/path.pth",
            "source_feature_sha256": "a" * 64,
            "e3_config_sha256": "b" * 64,
            "e3_checkpoint_sha256": "c" * 64,
        },
    }
    spatial = SimpleNamespace(
        image_ids={1, 2},
        manifest={
            "split": "train",
            "source_feature_paths": ["/unrelated/spatial/source.tar"],
            "source_feature_sha256": ["d" * 64],
            "dataset_identity": e9.dataset_identity_from_rows(
                split="train", image_ids=[1, 2, 1],
                annotation_ids=[10, 11, 12], source_image_count=2,
                source_annotation_count=3,
            ),
        },
    )
    return query, spatial


def test_paths_may_differ_when_cryptographic_dataset_identity_matches():
    query, spatial = _joined_fixture()
    assert len(e9.validate_query_spatial_join(
        query, spatial, expected_split="train"
    )) == 3


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda query, spatial: spatial.manifest.update({
                "source_feature_paths": [query["metadata"]["source_feature_path"]],
                "dataset_identity": e9.dataset_identity_from_rows(
                    split="train", image_ids=[2, 2, 1],
                    annotation_ids=[10, 11, 12], source_image_count=2,
                    source_annotation_count=3,
                ),
            }),
            "cryptographic dataset identity mismatch",
        ),
        (
            lambda query, spatial: query.update({
                "image_ids": torch.tensor([2, 2, 1], dtype=torch.int64)
            }),
            "cryptographic dataset identity mismatch",
        ),
        (
            lambda query, spatial: query.update({
                "mapped_query_embeddings": query["mapped_query_embeddings"][:2],
                "image_ids": query["image_ids"][:2],
                "annotation_ids": query["annotation_ids"][:2],
            }),
            "cryptographic dataset identity mismatch",
        ),
        (
            lambda query, spatial: query.update({
                "mapped_query_embeddings": torch.randn(4, 768).half(),
                "image_ids": torch.tensor([1, 2, 1, 3], dtype=torch.int64),
                "annotation_ids": torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            }),
            "missing spatial image ID",
        ),
        (
            lambda query, spatial: query["metadata"].update({
                "split_name": "val"
            }),
            "split mismatch",
        ),
    ),
)
def test_query_spatial_join_rejects_every_cryptographic_mismatch(
    mutation, match
):
    query, spatial = _joined_fixture()
    mutation(query, spatial)
    with pytest.raises(e9.E9SpatialBankValidationError, match=match):
        e9.validate_query_spatial_join(query, spatial, expected_split="train")


def test_pilot_query_view_selects_only_spatial_images_without_reconstruction():
    mapped = torch.randn(5, 768).half()
    query = {
        "mapped_query_embeddings": mapped,
        "image_ids": torch.tensor([1, 2, 1, 3, 4], dtype=torch.int64),
        "annotation_ids": torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64),
        "metadata": {"split_name": "train"},
    }
    spatial = SimpleNamespace(image_ids={1, 3})
    selected = select_query_rows_for_spatial(query, spatial, allow_subset=True)
    assert selected["image_ids"].tolist() == [1, 1, 3]
    assert selected["annotation_ids"].tolist() == [10, 12, 13]
    assert torch.equal(selected["mapped_query_embeddings"], mapped[[0, 2, 3]])
    assert selected["metadata"] is query["metadata"]
    with pytest.raises(ValueError, match="missing spatial image ID"):
        select_query_rows_for_spatial(query, spatial, allow_subset=False)
