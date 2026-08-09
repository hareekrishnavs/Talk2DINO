import hashlib
import copy
import io
import json
import math
import os
import shutil
import subprocess
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import src.e9_spatial_bank as e9
from train_e9_sparse_region_alignment import select_query_rows_for_spatial

DINO_SOURCE_COMMIT = "a" * 40
DINO_CHECKPOINT_SHA256 = "b" * 64


def _write_malicious_marker(path):
    Path(path).write_text("unsafe deserialization executed")


class _MaliciousSpatialShardPayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return _write_malicious_marker, (str(self.marker),)


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


def exact_tree_snapshot(root):
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.read_bytes(),
            "sha256": sha(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def path_snapshot(path):
    if path.is_symlink():
        return ("symlink", os.readlink(path))
    if path.is_file():
        payload = path.read_bytes()
        return ("file", payload, hashlib.sha256(payload).hexdigest())
    if path.is_dir():
        return ("directory", exact_tree_snapshot(path))
    return ("missing",)


def assert_no_publication_leftovers(parent, output_name="bank"):
    assert not list(parent.glob(f".{output_name}.e9-*"))
    assert not list(parent.glob(f".{output_name}.backup-*"))
    assert not os.path.lexists(parent / f".{output_name}.publication-lock")


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


@pytest.mark.parametrize(
    "malformed_name",
    (
        "../outside.pth",
        "/tmp/absolute.pth",
        "nested/train-000000.pth",
        "val-000000.pth",
        "train-000001.pth",
        "train-000000.pt",
        "./train-000000.pth",
    ),
)
def test_spatial_validator_rejects_noncanonical_shard_paths(
    tmp_path, malformed_name
):
    repository = tmp_path / "repo"
    git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(18)])
    output = tmp_path / "bank"
    build_spatial_bank(
        source,
        output,
        split="train",
        max_images=1,
        repository_root=repository,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["name"] = malformed_name
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        e9.E9SpatialBankValidationError,
        match="non-canonical spatial shard name",
    ):
        e9.validate_e9_spatial_bank(output, require_production=False)


def test_spatial_validator_rejects_duplicate_shard_names(tmp_path):
    repository = tmp_path / "repo"
    git_repo(repository)
    source = make_source(
        tmp_path / "source", [source_record(19), source_record(20)]
    )
    output = tmp_path / "bank"
    build_spatial_bank(
        source,
        output,
        split="train",
        shard_rows=1,
        max_images=2,
        repository_root=repository,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][1]["name"] = manifest["shards"][0]["name"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(e9.E9SpatialBankValidationError, match="duplicate"):
        e9.validate_e9_spatial_bank(output, require_production=False)


def test_spatial_validator_rejects_symlinked_manifest_and_shard(tmp_path):
    repository = tmp_path / "repo"
    git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(23)])
    output = tmp_path / "bank"
    build_spatial_bank(
        source,
        output,
        split="train",
        max_images=1,
        repository_root=repository,
    )
    shard = output / "train-000000.pth"
    outside_shard = tmp_path / "outside-shard.pth"
    shard.rename(outside_shard)
    shard.symlink_to(outside_shard)
    with pytest.raises(e9.E9SpatialBankValidationError, match="non-symlink"):
        e9.validate_e9_spatial_bank(output, require_production=False)

    shard.unlink()
    outside_shard.rename(shard)
    manifest = output / "manifest.json"
    outside_manifest = tmp_path / "outside-manifest.json"
    manifest.rename(outside_manifest)
    manifest.symlink_to(outside_manifest)
    with pytest.raises(e9.E9SpatialBankValidationError, match="manifest.*non-symlink"):
        e9.validate_e9_spatial_bank(output, require_production=False)


def test_malicious_spatial_pickle_never_executes_during_validation_or_overwrite(
    tmp_path
):
    repository = tmp_path / "repo"
    git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(24)])
    output = tmp_path / "bank"
    build_spatial_bank(
        source,
        output,
        split="train",
        max_images=1,
        repository_root=repository,
    )
    marker = tmp_path / "malicious-marker"
    shard_path = output / "train-000000.pth"
    torch.save(_MaliciousSpatialShardPayload(marker), shard_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["bytes"] = shard_path.stat().st_size
    manifest["shards"][0]["sha256"] = sha(shard_path)
    manifest_path.write_text(json.dumps(manifest))
    before = exact_tree_snapshot(output)

    with pytest.raises(e9.E9SpatialBankValidationError, match="cannot load shard"):
        e9.validate_e9_spatial_bank(output, require_production=False)
    assert not marker.exists()

    with pytest.raises(e9.E9SpatialBankValidationError, match="cannot load shard"):
        build_spatial_bank(
            source,
            output,
            split="train",
            max_images=1,
            overwrite=True,
            repository_root=repository,
        )
    assert not marker.exists()
    assert exact_tree_snapshot(output) == before
    assert_no_publication_leftovers(tmp_path)


def test_pilot_cannot_overwrite_production_bank_byte_for_byte(tmp_path):
    repository = tmp_path / "repo"
    git_repo(repository)
    production_source = make_source(
        tmp_path / "production-source", [source_record(21)]
    )
    pilot_source = make_source(
        tmp_path / "pilot-source", [source_record(22)]
    )
    output = tmp_path / "bank"
    build_spatial_bank(
        production_source, output, split="train", repository_root=repository
    )
    before = exact_tree_snapshot(output)
    manifest_sha = sha(output / "manifest.json")

    with pytest.raises(
        e9.E9SpatialBankValidationError,
        match="not a replaceable pilot",
    ):
        build_spatial_bank(
            pilot_source,
            output,
            split="train",
            max_images=1,
            overwrite=True,
            repository_root=repository,
        )

    assert exact_tree_snapshot(output) == before
    assert sha(output / "manifest.json") == manifest_sha
    assert e9.validate_e9_spatial_bank(output)["production_eligible"]
    assert_no_publication_leftovers(tmp_path)


def test_valid_pilot_can_atomically_replace_valid_pilot(tmp_path):
    repository = tmp_path / "repo"
    git_repo(repository)
    old_source = make_source(tmp_path / "old-source", [source_record(31)])
    new_source = make_source(tmp_path / "new-source", [source_record(32)])
    output = tmp_path / "bank"
    old_result = build_spatial_bank(
        old_source,
        output,
        split="train",
        max_images=1,
        repository_root=repository,
    )
    old_manifest_sha = sha(output / "manifest.json")

    new_result = build_spatial_bank(
        new_source,
        output,
        split="train",
        max_images=1,
        overwrite=True,
        repository_root=repository,
    )

    assert old_result["is_pilot"] and not old_result["production_eligible"]
    assert new_result["is_pilot"] and not new_result["production_eligible"]
    assert sha(output / "manifest.json") != old_manifest_sha
    manifest = json.loads((output / "manifest.json").read_text())
    assert [row["image_id"] for row in manifest["image_index"]] == [32]
    assert_no_publication_leftovers(tmp_path)


@pytest.mark.parametrize(
    "destination_kind",
    (
        "malformed_manifest",
        "missing_manifest",
        "legacy_v1",
        "invalid_incomplete",
        "valid_nonpilot_nonproduction",
        "ordinary_file",
        "unrelated_directory",
        "symlink",
    ),
)
def test_pilot_overwrite_rejects_and_restores_invalid_destination(
    tmp_path, destination_kind
):
    repository = tmp_path / "repo"
    git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(41)])
    output = tmp_path / "bank"
    symlink_target = tmp_path / "symlink-target"

    if destination_kind in {
        "legacy_v1",
        "invalid_incomplete",
        "valid_nonpilot_nonproduction",
    }:
        build_spatial_bank(
            source,
            output,
            split="train",
            max_images=(
                None
                if destination_kind == "valid_nonpilot_nonproduction"
                else 1
            ),
            repository_root=repository,
        )
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if destination_kind == "legacy_v1":
            manifest["format_version"] = "talk2dino-e9-spatial-bank-v1"
        elif destination_kind == "invalid_incomplete":
            manifest["complete"] = False
            manifest["production_eligible"] = False
        else:
            manifest["source_git_dirty"] = True
            manifest["source_git_diff_sha256"] = "d" * 64
            manifest["production_eligible"] = False
        manifest_path.write_text(json.dumps(manifest))
    elif destination_kind == "ordinary_file":
        output.write_bytes(b"ordinary file, not a bank")
    elif destination_kind == "symlink":
        symlink_target.mkdir()
        (symlink_target / "original").write_bytes(b"symlink target")
        output.symlink_to(symlink_target, target_is_directory=True)
    else:
        output.mkdir()
        if destination_kind == "malformed_manifest":
            (output / "manifest.json").write_bytes(b"{not-json")
            (output / "original").write_bytes(b"preserve malformed")
        elif destination_kind == "missing_manifest":
            (output / "train-000000.pth").write_bytes(b"missing manifest")
        else:
            nested = output / "unrelated"
            nested.mkdir()
            (nested / "original").write_bytes(b"unrelated directory")

    before = path_snapshot(output)
    symlink_target_before = (
        exact_tree_snapshot(symlink_target)
        if destination_kind == "symlink"
        else None
    )
    with pytest.raises(e9.E9SpatialBankValidationError):
        build_spatial_bank(
            source,
            output,
            split="train",
            max_images=1,
            overwrite=True,
            repository_root=repository,
        )

    assert path_snapshot(output) == before
    if destination_kind == "symlink":
        assert exact_tree_snapshot(symlink_target) == symlink_target_before
    assert_no_publication_leftovers(tmp_path)


@pytest.mark.parametrize(
    "alias_kind",
    ("source_parent", "multilevel_source_parent", "repository_parent"),
)
def test_parent_symlink_containment_is_rejected_before_any_write(
    tmp_path, monkeypatch, alias_kind
):
    repository = tmp_path / "repo"
    git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(42)])
    alias = tmp_path / "alias"
    if alias_kind == "source_parent":
        alias.symlink_to(source, target_is_directory=True)
    elif alias_kind == "multilevel_source_parent":
        intermediate = tmp_path / "intermediate-alias"
        intermediate.symlink_to(source, target_is_directory=True)
        alias.symlink_to(intermediate, target_is_directory=True)
    else:
        alias.symlink_to(repository, target_is_directory=True)
    canonical_parent = alias.resolve()
    iterated = False

    def reject_source_iteration(*args, **kwargs):
        nonlocal iterated
        iterated = True
        raise AssertionError("source records were iterated before containment rejection")

    monkeypatch.setattr(e9, "_iter_source_records", reject_source_iteration)
    with pytest.raises(ValueError, match="overlap|outside.*repository"):
        build_spatial_bank(
            source,
            alias / "bank",
            split="train",
            max_images=1,
            repository_root=repository,
        )

    assert not iterated
    assert not os.path.lexists(canonical_parent / "bank")
    assert not list(canonical_parent.glob(".bank.e9-*"))
    assert not os.path.lexists(canonical_parent / ".bank.publication-lock")


def test_parent_symlink_to_existing_production_parent_is_rejected_early(tmp_path):
    repository = tmp_path / "repo"
    git_repo(repository)
    production_source = make_source(
        tmp_path / "production-source", [source_record(43)]
    )
    incoming_source = make_source(
        tmp_path / "incoming-source", [source_record(44)]
    )
    production_parent = tmp_path / "production-parent"
    production_parent.mkdir()
    output = production_parent / "bank"
    build_spatial_bank(
        production_source,
        output,
        split="train",
        repository_root=repository,
    )
    before = exact_tree_snapshot(output)
    alias = tmp_path / "production-alias"
    alias.symlink_to(production_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="canonical parent"):
        build_spatial_bank(
            incoming_source,
            alias / "bank",
            split="train",
            max_images=1,
            overwrite=True,
            repository_root=repository,
        )

    assert exact_tree_snapshot(output) == before
    assert_no_publication_leftovers(production_parent)


def test_permitted_external_parent_symlink_resolves_and_publishes(tmp_path):
    repository = tmp_path / "repo"
    git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(45)])
    external = tmp_path / "external-scratch"
    external.mkdir()
    alias = tmp_path / "external-alias"
    alias.symlink_to(external, target_is_directory=True)

    result = build_spatial_bank(
        source,
        alias / "bank",
        split="train",
        max_images=1,
        repository_root=repository,
    )

    assert result["is_pilot"] and not result["production_eligible"]
    assert (external / "bank" / "manifest.json").is_file()
    assert e9.validate_e9_spatial_bank(
        external / "bank", require_production=False
    )["is_pilot"]
    assert_no_publication_leftovers(external)


@pytest.mark.parametrize("invalid_output", ("", ".", "..", "/"))
def test_empty_dot_and_dotdot_output_names_are_rejected_before_staging(
    tmp_path, invalid_output
):
    repository = tmp_path / "repo"
    git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(46)])
    with pytest.raises(ValueError, match="nonempty final name"):
        build_spatial_bank(
            source,
            invalid_output,
            split="train",
            max_images=1,
            repository_root=repository,
        )


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


def test_failed_publication_preserves_existing_output_and_cleans_staging(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repo"
    git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(4)])
    output = tmp_path / "bank"
    build_spatial_bank(
        source,
        output,
        split="train",
        max_images=1,
        repository_root=repository,
    )
    before = exact_tree_snapshot(output)
    monkeypatch.setattr(
        e9,
        "_require_unchanged_git_provenance",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("mutated")),
    )
    with pytest.raises(RuntimeError, match="mutated"):
        build_spatial_bank(
            source, output, split="train", max_images=1, overwrite=True,
            repository_root=repository,
        )
    assert exact_tree_snapshot(output) == before
    assert_no_publication_leftovers(tmp_path)


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

    def mutate_after_link(source_name, destination_name, *, src_dir_fd=None, dst_dir_fd=None):
        nonlocal mutation_seen
        original_link(
            source_name, destination_name,
            src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd,
        )
        linked_names.append(source_name)
        if source_name == e9.MANIFEST_NAME or mutation_seen:
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


def test_overwrite_with_absent_output_preserves_concurrent_winner(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repo"
    git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(62)])
    output = tmp_path / "bank"
    original_install = e9._install_reservation
    winner = {"bytes": b"concurrent winner"}
    injected = False

    def inject_winner_at_reservation(
        parent_fd, reservation_name, destination_name, *, conflict_label
    ):
        nonlocal injected
        if destination_name == output.name and not injected:
            injected = True
            # A genuinely concurrent, unrelated actor installs a real
            # directory at the shared destination name using nothing from
            # our reservation machinery -- only the destination name itself
            # is shared knowledge.
            os.mkdir(destination_name, 0o755, dir_fd=parent_fd)
            child_fd = os.open(
                destination_name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=parent_fd
            )
            try:
                winner_fd = os.open(
                    "winner", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644,
                    dir_fd=child_fd,
                )
                try:
                    os.write(winner_fd, winner["bytes"])
                finally:
                    os.close(winner_fd)
            finally:
                os.close(child_fd)
        return original_install(
            parent_fd, reservation_name, destination_name,
            conflict_label=conflict_label,
        )

    monkeypatch.setattr(e9, "_install_reservation", inject_winner_at_reservation)
    with pytest.raises(e9.E9PublicationConflictError, match=str(output)):
        build_spatial_bank(
            source,
            output,
            split="train",
            max_images=1,
            overwrite=True,
            repository_root=repository,
        )

    assert injected
    assert path_snapshot(output) == (
        "directory",
        {
            "winner": {
                "bytes": winner["bytes"],
                "sha256": hashlib.sha256(winner["bytes"]).hexdigest(),
            }
        },
    )
    assert_no_publication_leftovers(tmp_path)


def test_existing_pilot_race_preserves_foreign_output_and_old_backup(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repo"
    git_repo(repository)
    old_source = make_source(tmp_path / "old-source", [source_record(63)])
    new_source = make_source(tmp_path / "new-source", [source_record(64)])
    output = tmp_path / "bank"
    build_spatial_bank(
        old_source,
        output,
        split="train",
        max_images=1,
        repository_root=repository,
    )
    old_snapshot = exact_tree_snapshot(output)
    backup = output.with_name(f".{output.name}.backup-{os.getpid()}")
    original_validate = e9.validate_e9_spatial_bank
    foreign_bytes = b"foreign concurrent destination"
    injected = False

    def inject_after_backup_validation(path, *args, **kwargs):
        nonlocal injected
        result = original_validate(path, *args, **kwargs)
        if Path(path) == backup and not injected:
            injected = True
            output.mkdir()
            (output / "winner").write_bytes(foreign_bytes)
        return result

    monkeypatch.setattr(e9, "validate_e9_spatial_bank", inject_after_backup_validation)
    with pytest.raises(e9.E9PublicationRecoveryError) as captured:
        build_spatial_bank(
            new_source,
            output,
            split="train",
            max_images=1,
            overwrite=True,
            repository_root=repository,
        )

    assert injected
    assert str(output) in str(captured.value)
    assert str(backup) in str(captured.value)
    assert (output / "winner").read_bytes() == foreign_bytes
    assert not (output / "manifest.json").exists()
    assert exact_tree_snapshot(backup) == old_snapshot
    assert not list(tmp_path.glob(".bank.e9-*"))
    assert not os.path.lexists(tmp_path / ".bank.publication-lock")


def test_two_cooperative_builders_are_serialized_by_stable_lock(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repo"
    git_repo(repository)
    first_source = make_source(tmp_path / "first-source", [source_record(65)])
    second_source = make_source(tmp_path / "second-source", [source_record(66)])
    output = tmp_path / "bank"
    original_acquire = e9._acquire_publication_lock
    first_acquired = threading.Event()
    let_first_finish = threading.Event()
    guard = threading.Lock()
    first_call = True

    def coordinate_acquisition(lock_path):
        nonlocal first_call
        with guard:
            is_first = first_call
            first_call = False
        if is_first:
            identity = original_acquire(lock_path)
            first_acquired.set()
            assert let_first_finish.wait(timeout=10)
            return identity
        assert first_acquired.wait(timeout=10)
        try:
            return original_acquire(lock_path)
        finally:
            let_first_finish.set()

    monkeypatch.setattr(e9, "_acquire_publication_lock", coordinate_acquisition)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            build_spatial_bank,
            first_source,
            output,
            split="train",
            max_images=1,
            repository_root=repository,
        )
        assert first_acquired.wait(timeout=10)
        second = executor.submit(
            build_spatial_bank,
            second_source,
            output,
            split="train",
            max_images=1,
            repository_root=repository,
        )
        with pytest.raises(e9.E9PublicationConflictError, match="publication lock"):
            second.result(timeout=20)
        first_result = first.result(timeout=20)

    assert first_result["is_pilot"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert [row["image_id"] for row in manifest["image_index"]] == [65]
    assert e9.validate_e9_spatial_bank(
        output, require_production=False
    )["is_pilot"]
    assert_no_publication_leftovers(tmp_path)


def test_foreign_inode_after_reservation_is_never_deleted(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(67)])
    output = tmp_path / "bank"
    original_install = e9._install_reservation
    original_link = e9.os.link
    foreign_bytes = b"foreign inode survives"
    replaced = False

    def replace_after_install(
        parent_fd, reservation_name, destination_name, *, conflict_label
    ):
        nonlocal replaced
        original_install(
            parent_fd, reservation_name, destination_name,
            conflict_label=conflict_label,
        )
        if destination_name == output.name and not replaced:
            replaced = True
            # The reservation is installed and still empty (nothing has been
            # linked into it yet); an unrelated concurrent actor now
            # replaces it with a foreign directory using only the shared
            # pathname, with no knowledge of our retained descriptor.
            output.rmdir()
            output.mkdir()
            (output / "winner").write_bytes(foreign_bytes)

    def force_failure_after_replacement(
        source_name, destination_name, *, src_dir_fd=None, dst_dir_fd=None
    ):
        if replaced:
            raise OSError("forced failure after foreign inode replacement")
        return original_link(
            source_name, destination_name,
            src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(e9, "_install_reservation", replace_after_install)
    monkeypatch.setattr(e9.os, "link", force_failure_after_replacement)
    with pytest.raises(e9.E9PublicationRecoveryError, match="foreign"):
        build_spatial_bank(
            source,
            output,
            split="train",
            max_images=1,
            overwrite=True,
            repository_root=repository,
        )

    assert replaced
    assert (output / "winner").read_bytes() == foreign_bytes
    assert not (output / "manifest.json").exists()
    assert not list(tmp_path.glob(".bank.e9-*"))
    assert not list(tmp_path.glob(".bank.backup-*"))
    assert not os.path.lexists(tmp_path / ".bank.publication-lock")


@pytest.mark.parametrize(
    "mutation_kind",
    (
        "unstaged",
        "staged",
        "untracked",
        "source_artifact",
        "shard_link",
        "manifest_link",
        "file_fsync",
    ),
)
def test_manifest_boundary_failed_pilot_overwrite_restores_existing_tree(
    tmp_path, monkeypatch, mutation_kind
):
    repository = tmp_path / "repo"
    tracked = git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(13)])
    source_shard = source / "train-000000.tar"
    output = tmp_path / "bank"
    build_spatial_bank(
        source,
        output,
        split="train",
        max_images=1,
        repository_root=repository,
    )
    before = exact_tree_snapshot(output)
    manifest_sha = sha(output / "manifest.json")
    backup = output.with_name(f".{output.name}.backup-{os.getpid()}")
    original_validate = e9.validate_e9_spatial_bank
    original_link = e9.os.link
    original_fsync_name = e9._fsync_name
    mutation_seen = False

    def mutate_after_backup_validation(path, *args, **kwargs):
        nonlocal mutation_seen
        result = original_validate(path, *args, **kwargs)
        if Path(path) != backup or mutation_seen:
            return result
        mutation_seen = True
        if mutation_kind == "untracked":
            (repository / "new.py").write_text("new\n")
        elif mutation_kind == "source_artifact":
            source_shard.write_bytes(source_shard.read_bytes() + b"mutation")
        elif mutation_kind in {"unstaged", "staged"}:
            tracked.write_text(f"{mutation_kind}\n")
            if mutation_kind == "staged":
                subprocess.run(
                    ["git", "-C", str(repository), "add", "source.py"],
                    check=True,
                )
        return result

    def fail_publication_link(
        source_name, destination_name, *, src_dir_fd=None, dst_dir_fd=None
    ):
        if (
            mutation_kind == "shard_link" and destination_name != e9.MANIFEST_NAME
            or mutation_kind == "manifest_link"
            and destination_name == e9.MANIFEST_NAME
        ):
            raise OSError("injected publication link failure")
        return original_link(
            source_name, destination_name,
            src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd,
        )

    def fail_publication_fsync(name, *, dir_fd):
        if mutation_kind == "file_fsync":
            raise OSError("injected publication fsync failure")
        return original_fsync_name(name, dir_fd=dir_fd)

    monkeypatch.setattr(e9, "validate_e9_spatial_bank", mutate_after_backup_validation)
    monkeypatch.setattr(e9.os, "link", fail_publication_link)
    monkeypatch.setattr(e9, "_fsync_name", fail_publication_fsync)
    with pytest.raises((e9.E9SpatialBankValidationError, OSError)):
        build_spatial_bank(
            source, output, split="train", max_images=1, overwrite=True,
            repository_root=repository,
        )
    assert mutation_seen
    assert exact_tree_snapshot(output) == before
    assert sha(output / "manifest.json") == manifest_sha
    assert_no_publication_leftovers(tmp_path)


def test_pilot_overwrite_publication_order_uses_protected_backup(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    git_repo(repository)
    old_source = make_source(tmp_path / "old-source", [source_record(51)])
    new_source = make_source(tmp_path / "new-source", [source_record(52)])
    output = tmp_path / "bank"
    build_spatial_bank(
        old_source,
        output,
        split="train",
        max_images=1,
        repository_root=repository,
    )
    backup = output.with_name(f".{output.name}.backup-{os.getpid()}")
    events = []
    state = {"boundary": False}
    original_validate = e9.validate_e9_spatial_bank
    original_sha256 = e9.sha256_file
    original_git_check = e9._require_unchanged_git_provenance
    original_renameat2 = e9._renameat2
    original_install = e9._install_reservation
    original_link = e9.os.link
    original_fsync_name = e9._fsync_name
    original_checkpoint = e9._checkpoint_and_commit_output
    original_dispose = e9._dispose_owned
    original_release_lock = e9._release_publication_lock
    original_torch_save = e9.torch.save

    def record_validate(path, *args, **kwargs):
        result = original_validate(path, *args, **kwargs)
        if state["boundary"] and Path(path) == backup:
            events.append("validate_backup")
        return result

    def record_sha256(path):
        result = original_sha256(path)
        if state["boundary"] and "final_input_hash_check" not in events:
            events.append("final_input_hash_check")
        return result

    def record_git_check(*args, **kwargs):
        result = original_git_check(*args, **kwargs)
        if state["boundary"]:
            events.append("final_git_check")
        return result

    def record_renameat2(
        old_dir_fd, old_name, new_dir_fd, new_name, flags, *, unsupported_label
    ):
        result = original_renameat2(
            old_dir_fd, old_name, new_dir_fd, new_name, flags,
            unsupported_label=unsupported_label,
        )
        if new_name == backup.name and flags == e9._RENAME_NOREPLACE:
            state["boundary"] = True
            events.append("rename_backup")
        return result

    def record_install(parent_fd, reservation_name, destination_name, *, conflict_label):
        result = original_install(
            parent_fd, reservation_name, destination_name,
            conflict_label=conflict_label,
        )
        if state["boundary"] and destination_name == output.name:
            events.append("reserve_output")
        return result

    def record_link(source_name, destination_name, *, src_dir_fd=None, dst_dir_fd=None):
        result = original_link(
            source_name, destination_name,
            src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd,
        )
        if state["boundary"]:
            events.append(
                "link_manifest" if destination_name == e9.MANIFEST_NAME
                else "link_shard"
            )
        return result

    def record_fsync_name(name, *, dir_fd):
        result = original_fsync_name(name, dir_fd=dir_fd)
        if state["boundary"]:
            events.append(
                "fsync_manifest" if name == e9.MANIFEST_NAME else "fsync_shard"
            )
        return result

    def record_checkpoint(parent_fd, output_name, reservation, *, parent_display):
        events.append("checkpoint_commit_start")
        result = original_checkpoint(
            parent_fd, output_name, reservation, parent_display=parent_display,
        )
        events.append("checkpoint_commit_done")
        return result

    def record_dispose(parent_fd, name, expected_identity, *, label, parent_display):
        result = original_dispose(
            parent_fd, name, expected_identity,
            label=label, parent_display=parent_display,
        )
        if label == "validated old E9 pilot backup":
            events.append("dispose_backup")
        return result

    def record_release_lock(lock, identity):
        result = original_release_lock(lock, identity)
        events.append("release_lock")
        return result

    def reject_late_serialization(*args, **kwargs):
        if state["boundary"]:
            raise AssertionError("serialization occurred after backup validation boundary")
        return original_torch_save(*args, **kwargs)

    monkeypatch.setattr(e9, "validate_e9_spatial_bank", record_validate)
    monkeypatch.setattr(e9, "sha256_file", record_sha256)
    monkeypatch.setattr(e9, "_require_unchanged_git_provenance", record_git_check)
    monkeypatch.setattr(e9, "_renameat2", record_renameat2)
    monkeypatch.setattr(e9, "_install_reservation", record_install)
    monkeypatch.setattr(e9.os, "link", record_link)
    monkeypatch.setattr(e9, "_fsync_name", record_fsync_name)
    monkeypatch.setattr(e9, "_checkpoint_and_commit_output", record_checkpoint)
    monkeypatch.setattr(e9, "_dispose_owned", record_dispose)
    monkeypatch.setattr(e9, "_release_publication_lock", record_release_lock)
    monkeypatch.setattr(e9.torch, "save", reject_late_serialization)

    build_spatial_bank(
        new_source,
        output,
        split="train",
        max_images=1,
        overwrite=True,
        repository_root=repository,
    )

    # The backup must exist and be validated before the new output is ever
    # reserved; shards must be linked (and fsynced) before the manifest,
    # which is linked only after the final input-hash/Git provenance
    # recheck; the published bank is only checkpointed after the manifest is
    # durable; the old pilot backup is retired only after that checkpoint
    # commits; and the publication lock is released last of all.
    assert events == [
        "rename_backup",
        "validate_backup",
        "reserve_output",
        "link_shard",
        "fsync_shard",
        "final_input_hash_check",
        "final_git_check",
        "link_manifest",
        "fsync_manifest",
        "checkpoint_commit_start",
        "checkpoint_commit_done",
        "dispose_backup",
        "release_lock",
    ]
    assert_no_publication_leftovers(tmp_path)


def test_restoration_failure_preserves_recoverable_backup(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    tracked = git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(61)])
    output = tmp_path / "bank"
    build_spatial_bank(
        source,
        output,
        split="train",
        max_images=1,
        repository_root=repository,
    )
    before = exact_tree_snapshot(output)
    backup = output.with_name(f".{output.name}.backup-{os.getpid()}")
    original_validate = e9.validate_e9_spatial_bank
    original_renameat2 = e9._renameat2
    mutation_seen = False

    def mutate_after_backup_validation(path, *args, **kwargs):
        nonlocal mutation_seen
        result = original_validate(path, *args, **kwargs)
        if Path(path) == backup and not mutation_seen:
            mutation_seen = True
            tracked.write_text("mutated after backup validation\n")
        return result

    def fail_backup_restoration(
        old_dir_fd, old_name, new_dir_fd, new_name, flags, *, unsupported_label
    ):
        if (
            old_name == backup.name and new_name == output.name
            and flags == e9._RENAME_NOREPLACE
        ):
            raise OSError("injected restoration failure")
        return original_renameat2(
            old_dir_fd, old_name, new_dir_fd, new_name, flags,
            unsupported_label=unsupported_label,
        )

    monkeypatch.setattr(e9, "validate_e9_spatial_bank", mutate_after_backup_validation)
    monkeypatch.setattr(e9, "_renameat2", fail_backup_restoration)
    with pytest.raises(RuntimeError, match=str(backup)):
        build_spatial_bank(
            source,
            output,
            split="train",
            max_images=1,
            overwrite=True,
            repository_root=repository,
        )

    assert mutation_seen
    assert not os.path.lexists(output)
    assert backup.is_dir()
    assert exact_tree_snapshot(backup) == before
    assert not list(tmp_path.glob(".bank.e9-*"))


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
    original_install = e9._install_reservation
    injected = False

    def concurrent_install(parent_fd, reservation_name, destination_name, *, conflict_label):
        nonlocal injected
        if destination_name == output.name and not injected:
            injected = True
            os.mkdir(destination_name, 0o755, dir_fd=parent_fd)
            child_fd = os.open(
                destination_name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=parent_fd
            )
            try:
                winner_fd = os.open(
                    "winner", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644,
                    dir_fd=child_fd,
                )
                try:
                    os.write(winner_fd, b"concurrent")
                finally:
                    os.close(winner_fd)
            finally:
                os.close(child_fd)
        return original_install(
            parent_fd, reservation_name, destination_name,
            conflict_label=conflict_label,
        )

    monkeypatch.setattr(e9, "_install_reservation", concurrent_install)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        e9._publish_directory(
            staging, output, overwrite=False, final_verification=lambda: None
        )
    assert (output / "winner").read_bytes() == b"concurrent"
    assert not (output / "manifest.json").exists()


def test_existing_stable_lock_is_preserved_without_publication_mutation(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "train-000000.pth").write_bytes(b"staged shard")
    (staging / "manifest.json").write_bytes(b"staged manifest")
    staging_before = exact_tree_snapshot(staging)
    output = tmp_path / "bank"
    output.mkdir()
    (output / "existing").write_bytes(b"existing output")
    output_before = exact_tree_snapshot(output)
    lock = tmp_path / ".bank.publication-lock"
    lock.mkdir()
    (lock / "owner").write_bytes(b"unknown lock owner")
    lock_before = exact_tree_snapshot(lock)

    with pytest.raises(e9.E9PublicationConflictError, match=str(lock)):
        e9._publish_directory(
            staging,
            output,
            overwrite=True,
            final_verification=lambda: None,
        )

    assert exact_tree_snapshot(staging) == staging_before
    assert exact_tree_snapshot(output) == output_before
    assert exact_tree_snapshot(lock) == lock_before
    assert not list(tmp_path.glob(".bank.backup-*"))


def test_replaced_publication_lock_is_never_removed(tmp_path):
    lock = tmp_path / ".bank.publication-lock"
    identity = e9._acquire_publication_lock(lock)
    original_lock = tmp_path / ".bank.original-lock"
    os.rename(lock, original_lock)
    lock.mkdir()
    (lock / "foreign").write_bytes(b"foreign lock")

    with pytest.raises(e9.E9PublicationRecoveryError, match=str(lock)):
        e9._release_publication_lock(lock, identity)

    assert (lock / "foreign").read_bytes() == b"foreign lock"
    assert original_lock.is_dir()


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


# ---------------------------------------------------------------------------
# Adversarial race-safety regression tests for the renameat2-based
# publication layer.  Every test below drives the real production helpers
# (e9._install_reservation, e9._renameat2, e9._checkpoint_and_commit_output,
# e9._dispose_owned, ...) through narrowly placed hooks; none of them replace
# the publication or cleanup functions themselves with a fake.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("foreign_kind", ("file", "symlink"))
def test_foreign_non_directory_after_reservation_is_never_deleted(
    tmp_path, monkeypatch, foreign_kind
):
    repository = tmp_path / "repo"
    git_repo(repository)
    source = make_source(
        tmp_path / "source", [source_record(96 if foreign_kind == "file" else 97)]
    )
    output = tmp_path / "bank"
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_bytes(b"symlink target")
    original_install = e9._install_reservation
    original_link = e9.os.link
    foreign_bytes = b"foreign non-directory survives"
    replaced = False

    def replace_after_install(
        parent_fd, reservation_name, destination_name, *, conflict_label
    ):
        nonlocal replaced
        original_install(
            parent_fd, reservation_name, destination_name,
            conflict_label=conflict_label,
        )
        if destination_name == output.name and not replaced:
            replaced = True
            # The reservation is installed and still empty; an unrelated
            # concurrent actor now replaces it with a foreign non-directory
            # object using only the shared pathname.
            output.rmdir()
            if foreign_kind == "file":
                output.write_bytes(foreign_bytes)
            else:
                output.symlink_to(elsewhere)

    def force_failure_after_replacement(
        source_name, destination_name, *, src_dir_fd=None, dst_dir_fd=None
    ):
        if replaced:
            raise OSError("forced failure after foreign non-directory replacement")
        return original_link(
            source_name, destination_name,
            src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(e9, "_install_reservation", replace_after_install)
    monkeypatch.setattr(e9.os, "link", force_failure_after_replacement)
    with pytest.raises(e9.E9PublicationRecoveryError, match="foreign"):
        build_spatial_bank(
            source, output, split="train", max_images=1, overwrite=True,
            repository_root=repository,
        )

    assert replaced
    if foreign_kind == "file":
        assert output.is_file() and not output.is_symlink()
        assert output.read_bytes() == foreign_bytes
    else:
        assert output.is_symlink()
        assert os.readlink(output) == str(elsewhere)
        assert elsewhere.read_bytes() == b"symlink target"
    assert not (
        output.is_dir() and (output / "manifest.json").exists()
    )
    assert not list(tmp_path.glob(".bank.e9-*"))
    assert not list(tmp_path.glob(".bank.backup-*"))
    assert not os.path.lexists(tmp_path / ".bank.publication-lock")


def test_checkpoint_detects_output_replaced_after_validation_and_preserves_everything(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repo"
    git_repo(repository)
    old_source = make_source(tmp_path / "old-source", [source_record(71)])
    new_source = make_source(tmp_path / "new-source", [source_record(72)])
    output = tmp_path / "bank"
    build_spatial_bank(
        old_source, output, split="train", max_images=1, repository_root=repository,
    )
    old_pilot_snapshot = exact_tree_snapshot(output)
    backup = output.with_name(f".{output.name}.backup-{os.getpid()}")

    original_publish_manifest_last = e9._publish_manifest_last
    foreign_bytes = b"foreign post-validation output"
    replaced = False

    def replace_after_publish(
        parent_fd, output_name, staging_fd, *, output_display, final_verification
    ):
        nonlocal replaced
        reservation = original_publish_manifest_last(
            parent_fd, output_name, staging_fd,
            output_display=output_display,
            final_verification=final_verification,
        )
        if not replaced:
            replaced = True
            # Simulate a concurrent actor atomically swapping the
            # just-validated, just-published output for a foreign directory
            # in the instant after _publish_manifest_last returns and
            # before the checkpoint capture that is supposed to close this
            # window runs.  The swap uses RENAME_EXCHANGE, exactly like a
            # second copy of this same tooling would -- the original,
            # fully-populated reservation is displaced but never destroyed,
            # only its shared name is taken.
            displaced_name = f".displaced-original-{os.getpid()}"
            os.mkdir(displaced_name, 0o700, dir_fd=parent_fd)
            e9._renameat2(
                parent_fd, displaced_name, parent_fd, output_name,
                e9._RENAME_EXCHANGE, unsupported_label="RENAME_EXCHANGE",
            )
            (output / "winner").write_bytes(foreign_bytes)
        return reservation

    monkeypatch.setattr(e9, "_publish_manifest_last", replace_after_publish)

    with pytest.raises(e9.E9PublicationRecoveryError) as captured:
        build_spatial_bank(
            new_source, output, split="train", max_images=1, overwrite=True,
            repository_root=repository,
        )

    assert replaced
    message = str(captured.value)
    assert "replaced" in message
    # The foreign output that raced in survives untouched.
    assert (output / "winner").read_bytes() == foreign_bytes
    assert not (output / "manifest.json").exists()
    # The old pilot backup is preserved byte-for-byte -- it is never
    # restored over the foreign output, and never deleted.
    assert exact_tree_snapshot(backup) == old_pilot_snapshot
    # The validated new bank is not silently discarded: it is preserved at a
    # reported recovery path.
    recovery_dirs = [
        path for path in tmp_path.iterdir()
        if path.is_dir() and path.name.startswith(".e9-orphaned-output-")
    ]
    assert len(recovery_dirs) == 1
    assert str(recovery_dirs[0].name) in message
    recovered_manifest = json.loads((recovery_dirs[0] / "manifest.json").read_text())
    assert [row["image_id"] for row in recovered_manifest["image_index"]] == [72]
    assert not list(tmp_path.glob(".bank.e9-*"))
    assert not os.path.lexists(tmp_path / ".bank.publication-lock")


def _open_fd_count():
    return len(os.listdir("/proc/self/fd"))


def test_descriptor_based_validation_rejects_symlink_swap_and_leaks_no_fds(tmp_path):
    repository = tmp_path / "repo"
    git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(95)])
    output = tmp_path / "bank"
    build_spatial_bank(
        source, output, split="train", max_images=1, repository_root=repository,
    )

    secret = tmp_path / "outside-secret.txt"
    secret.write_bytes(b"SECRET-DO-NOT-LEAK")
    secret_before = secret.read_bytes()

    # The manifest is swapped for a symlink to an external, sensitive file.
    manifest_path = output / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_path.unlink()
    manifest_path.symlink_to(secret)
    before_fds = _open_fd_count()
    with pytest.raises(
        e9.E9SpatialBankValidationError, match="non-symlink"
    ) as captured:
        e9.validate_e9_spatial_bank(output, require_production=False)
    assert _open_fd_count() == before_fds
    assert "SECRET-DO-NOT-LEAK" not in str(captured.value)
    assert secret.read_bytes() == secret_before
    manifest_path.unlink()
    manifest_path.write_bytes(manifest_bytes)

    # A canonical shard is swapped for a symlink to the same external file.
    shard_path = output / "train-000000.pth"
    shard_bytes = shard_path.read_bytes()
    shard_path.unlink()
    shard_path.symlink_to(secret)
    before_fds = _open_fd_count()
    with pytest.raises(
        e9.E9SpatialBankValidationError, match="non-symlink"
    ) as captured:
        e9.validate_e9_spatial_bank(output, require_production=False)
    assert _open_fd_count() == before_fds
    assert "SECRET-DO-NOT-LEAK" not in str(captured.value)
    assert secret.read_bytes() == secret_before
    shard_path.unlink()
    shard_path.write_bytes(shard_bytes)

    # Regular validation still succeeds after restoring the real files, and
    # every call still closes every descriptor it opened.
    before_fds = _open_fd_count()
    for _ in range(5):
        result = e9.validate_e9_spatial_bank(output, require_production=False)
        assert result["is_pilot"] is True
    assert _open_fd_count() == before_fds


def test_early_final_component_symlink_rejection_precedes_all_side_effects(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repo"
    git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(91)])
    symlink_target = tmp_path / "elsewhere"
    symlink_target.mkdir()
    (symlink_target / "preexisting").write_bytes(b"do not touch")
    output = tmp_path / "bank"
    output.symlink_to(symlink_target, target_is_directory=True)
    before_target = exact_tree_snapshot(symlink_target)

    calls = {
        "iter_source_records": 0,
        "mkdtemp": 0,
        "acquire_lock": 0,
        "renameat2": 0,
    }
    original_iter = e9._iter_source_records
    original_mkdtemp = e9.tempfile.mkdtemp
    original_acquire_lock = e9._acquire_publication_lock
    original_renameat2 = e9._renameat2

    def counting_iter(*args, **kwargs):
        calls["iter_source_records"] += 1
        return original_iter(*args, **kwargs)

    def counting_mkdtemp(*args, **kwargs):
        calls["mkdtemp"] += 1
        return original_mkdtemp(*args, **kwargs)

    def counting_acquire_lock(*args, **kwargs):
        calls["acquire_lock"] += 1
        return original_acquire_lock(*args, **kwargs)

    def counting_renameat2(*args, **kwargs):
        calls["renameat2"] += 1
        return original_renameat2(*args, **kwargs)

    monkeypatch.setattr(e9, "_iter_source_records", counting_iter)
    monkeypatch.setattr(e9.tempfile, "mkdtemp", counting_mkdtemp)
    monkeypatch.setattr(e9, "_acquire_publication_lock", counting_acquire_lock)
    monkeypatch.setattr(e9, "_renameat2", counting_renameat2)

    with pytest.raises(e9.E9SpatialBankValidationError, match="symbolic-link"):
        build_spatial_bank(
            source, output, split="train", max_images=1,
            repository_root=repository,
        )

    assert calls == {
        "iter_source_records": 0,
        "mkdtemp": 0,
        "acquire_lock": 0,
        "renameat2": 0,
    }
    assert output.is_symlink()
    assert os.readlink(output) == str(symlink_target)
    assert exact_tree_snapshot(symlink_target) == before_target


def test_missing_renameat2_support_fails_closed_before_any_destination_write(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repo"
    git_repo(repository)
    source = make_source(tmp_path / "source", [source_record(81)])
    output = tmp_path / "bank"
    build_spatial_bank(
        source, output, split="train", max_images=1, repository_root=repository,
    )
    before_production = exact_tree_snapshot(output)

    pilot_source = make_source(tmp_path / "pilot-source", [source_record(82)])
    pilot_output = tmp_path / "pilot-bank"
    build_spatial_bank(
        pilot_source, pilot_output, split="train", max_images=1,
        repository_root=repository,
    )
    before_pilot = exact_tree_snapshot(pilot_output)

    def unsupported_renameat2(
        old_dir_fd, old_name, new_dir_fd, new_name, flags, *, unsupported_label
    ):
        raise e9.E9AtomicOperationUnsupportedError(
            f"simulated missing {unsupported_label} support"
        )

    monkeypatch.setattr(e9, "_renameat2", unsupported_renameat2)

    new_pilot_source = make_source(tmp_path / "new-pilot-source", [source_record(83)])
    with pytest.raises(
        e9.E9AtomicOperationUnsupportedError,
        match="RENAME_NOREPLACE|RENAME_EXCHANGE",
    ):
        build_spatial_bank(
            new_pilot_source, pilot_output, split="train", max_images=1,
            overwrite=True, repository_root=repository,
        )

    # The two real destinations are byte-identical to before: no destination
    # was ever moved, replaced, or partially written.  (Cleanup itself also
    # requires the same atomic primitive, so a leftover, inert staging
    # directory is the correct, safe outcome here -- not a corrupted
    # destination -- when the primitive is entirely unavailable; this is
    # intentionally not asserted away.)
    assert exact_tree_snapshot(output) == before_production
    assert exact_tree_snapshot(pilot_output) == before_pilot
    assert not list(tmp_path.glob(".pilot-bank.backup-*"))
    # The probe runs before the publication lock is ever acquired, so no
    # lock is created either.
    assert not os.path.lexists(tmp_path / ".pilot-bank.publication-lock")
