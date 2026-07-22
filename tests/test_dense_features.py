import io
import json
import tarfile

import pytest
import torch
from torch.utils.data import DataLoader

from src.dense_features import (
    DenseFeatureShardWriter,
    DenseFeatureStreamingDataset,
    DenseFeatureValidationError,
    iter_dense_shard_records,
    validate_dense_dataset,
    validate_dense_shard,
)


CONFIG = {
    "model": "dinov2_vitb14_reg",
    "patch_count": 1024,
    "attention_map_format": "probabilities",
}


@pytest.fixture(scope="module")
def dense_tensors():
    generator = torch.Generator().manual_seed(7)
    patches = torch.randn(1024, 768, generator=generator).to(torch.float16)
    maps = torch.rand(12, 1024, generator=generator)
    maps = (maps / maps.sum(dim=-1, keepdim=True)).to(torch.float16)
    heads = (
        patches.float().unsqueeze(0) * maps.float().unsqueeze(-1)
    ).mean(dim=1)
    return patches, maps, heads


def make_record(dense_tensors, image_id, caption_count=2):
    patches, maps, heads = dense_tensors
    return {
        "image_id": image_id,
        "file_name": f"{image_id:012d}.jpg",
        "disentangled_self_attn": heads,
        "patch_tokens": patches,
        "self_attn_maps": maps,
        "captions": [f"caption {image_id}-{index}" for index in range(caption_count)],
        "ann_feats": [
            torch.full((512,), image_id + index, dtype=torch.float32)
            for index in range(caption_count)
        ],
        "annotation_ids": [image_id * 10 + index for index in range(caption_count)],
    }


def writer(path, images_per_shard=2):
    return DenseFeatureShardWriter(
        path,
        "train",
        CONFIG,
        "9f314dc",
        images_per_shard=images_per_shard,
    )


def write_records(path, dense_tensors, count=4, captions=2, images_per_shard=2):
    with writer(path, images_per_shard) as output:
        for image_id in range(count):
            output.add(make_record(dense_tensors, image_id, captions))


def test_continuous_image_level_storage_and_atomic_finalization(tmp_path, dense_tensors):
    output = writer(tmp_path, images_per_shard=2)
    output.add(make_record(dense_tensors, 1, caption_count=3))
    assert not (tmp_path / "train-000000.tar").exists()
    assert (tmp_path / "train-000000.tar.tmp").exists()

    output.add(make_record(dense_tensors, 2, caption_count=4))
    finalized = tmp_path / "train-000000.tar"
    assert finalized.exists()  # Written before the complete input has been consumed.
    assert not (tmp_path / "train-000000.tar.tmp").exists()
    records = list(iter_dense_shard_records(finalized))
    assert len(records) == 2  # Dense tensors occur once per image, not per caption.
    assert records[0]["captions"] == ["caption 1-0", "caption 1-1", "caption 1-2"]
    assert records[1]["annotation_ids"] == [20, 21, 22, 23]
    assert records[0]["patch_tokens"].dtype == torch.float16
    assert records[0]["self_attn_maps"].dtype == torch.float16
    assert torch.allclose(
        records[0]["self_attn_maps"].float().sum(-1), torch.ones(12), atol=2e-3
    )
    output.add(make_record(dense_tensors, 3))
    output.close()
    assert (tmp_path / "train-000001.tar").exists()


def test_manifest_and_dataset_validation(tmp_path, dense_tensors):
    write_records(tmp_path, dense_tensors, count=3, captions=2)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["source_commit"] == "9f314dc"
    assert manifest["extraction_config"] == CONFIG
    assert manifest["images"] == 3
    assert manifest["annotations"] == 6
    assert [shard["images"] for shard in manifest["shards"]] == [2, 1]
    assert all(shard["sha256"] and shard["bytes"] > 0 for shard in manifest["shards"])

    result = validate_dense_dataset(tmp_path)
    assert result["images"] == 3
    assert result["annotations"] == 6
    assert result["failed_images"] == []
    assert result["cosine_min"] > 0.999
    assert result["row_sum_min"] == pytest.approx(1.0, abs=2e-3)


def test_partial_shard_is_rejected(tmp_path, dense_tensors):
    output = writer(tmp_path)
    output.add(make_record(dense_tensors, 1))
    partial = tmp_path / "train-000000.tar.tmp"
    with pytest.raises(DenseFeatureValidationError, match="partial"):
        validate_dense_shard(partial)
    output.abort()
    assert partial.exists()


def test_restart_skips_verified_images_without_overwrite(tmp_path, dense_tensors):
    with writer(tmp_path) as output:
        output.add(make_record(dense_tensors, 1))
        output.add(make_record(dense_tensors, 2))
    first = tmp_path / "train-000000.tar"
    original = first.read_bytes()

    # Simulate a crash after atomic shard rename but before manifest replacement.
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["images"] = 0
    manifest["annotations"] = 0
    manifest["shards"] = []
    manifest_path.write_text(json.dumps(manifest))

    with writer(tmp_path) as resumed:
        assert resumed.add(make_record(dense_tensors, 1)) is False
        assert resumed.add(make_record(dense_tensors, 3)) is True
    assert first.read_bytes() == original
    result = validate_dense_dataset(tmp_path)
    assert result["images"] == 3

    with pytest.raises(FileExistsError, match="different extraction configuration"):
        DenseFeatureShardWriter(tmp_path, "train", {"different": True}, "9f314dc", 2)


def test_reader_expands_annotations_without_dense_copies(tmp_path, dense_tensors):
    write_records(tmp_path, dense_tensors, count=3, captions=3)
    dataset = DenseFeatureStreamingDataset(tmp_path)
    samples = list(dataset)
    assert len(dataset) == len(samples) == 9
    assert [sample["metadata"]["annotation_id"] for sample in samples[:3]] == [0, 1, 2]
    assert all(sample["metadata"]["image_id"] == 0 for sample in samples[:3])
    assert samples[0]["patch_tokens"].data_ptr() == samples[1]["patch_tokens"].data_ptr()
    assert samples[0]["self_attn_maps"].data_ptr() == samples[2]["self_attn_maps"].data_ptr()
    assert set(samples[0]) == {
        "annotation", "image", "metadata", "caption", "patch_tokens", "self_attn_maps"
    }


def test_reader_is_deterministic_and_bounded_shuffle_is_reproducible(tmp_path, dense_tensors):
    write_records(tmp_path, dense_tensors, count=4, captions=2)
    ordered = [
        sample["metadata"]["annotation_id"]
        for sample in DenseFeatureStreamingDataset(tmp_path)
    ]
    repeated = [
        sample["metadata"]["annotation_id"]
        for sample in DenseFeatureStreamingDataset(tmp_path)
    ]
    shuffled_a = [
        sample["metadata"]["annotation_id"]
        for sample in DenseFeatureStreamingDataset(
            tmp_path, shuffle_shards=True, shuffle_buffer=3, seed=19
        )
    ]
    shuffled_b = [
        sample["metadata"]["annotation_id"]
        for sample in DenseFeatureStreamingDataset(
            tmp_path, shuffle_shards=True, shuffle_buffer=3, seed=19
        )
    ]
    assert ordered == repeated
    assert shuffled_a == shuffled_b
    assert shuffled_a != ordered
    assert sorted(shuffled_a) == sorted(ordered)


def test_multiple_workers_do_not_duplicate_samples(tmp_path, dense_tensors):
    shard_entries = []
    for shard_index in range(2):
        records = []
        for image_id in range(shard_index * 2, shard_index * 2 + 2):
            record = make_record(dense_tensors, image_id)
            # Reader partitioning does not depend on production tensor dimensions;
            # keep worker IPC small in constrained test environments.
            record["disentangled_self_attn"] = torch.ones(2, 3)
            record["patch_tokens"] = torch.ones(4, 3)
            record["self_attn_maps"] = torch.full((2, 4), 0.25)
            records.append(record)
        shard_path = tmp_path / f"train-{shard_index:06d}.tar"
        write_unchecked_records(shard_path, records)
        shard_entries.append(
            {"name": shard_path.name, "images": 2, "annotations": 4}
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "split": "train",
                "images_per_shard": 2,
                "images": 4,
                "annotations": 8,
                "shards": shard_entries,
            }
        )
    )
    dataset = DenseFeatureStreamingDataset(tmp_path)
    annotation_ids = [
        sample["metadata"]["annotation_id"]
        for sample in DataLoader(dataset, batch_size=None, num_workers=2)
    ]
    assert len(annotation_ids) == len(dataset)
    assert len(annotation_ids) == len(set(annotation_ids))
    assert sorted(annotation_ids) == [0, 1, 10, 11, 20, 21, 30, 31]


def write_unchecked_records(path, records):
    with tarfile.open(path, "w") as archive:
        for index, record in enumerate(records):
            payload = io.BytesIO()
            torch.save(record, payload)
            member = tarfile.TarInfo(f"{index:08d}.pth")
            member.size = len(payload.getvalue())
            archive.addfile(member, io.BytesIO(payload.getvalue()))


def write_unchecked_shard(path, record):
    write_unchecked_records(path, [record])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.pop("patch_tokens"), "missing keys"),
        (
            lambda record: record.__setitem__(
                "patch_tokens", record["patch_tokens"].float()
            ),
            "must be float16",
        ),
        (
            lambda record: record.__setitem__(
                "self_attn_maps", record["self_attn_maps"][:, :-1]
            ),
            "patch/map token counts",
        ),
        (lambda record: record["captions"].pop(), "alignment"),
    ],
)
def test_missing_corrupt_shape_dtype_and_alignment_errors(
    tmp_path, dense_tensors, mutation, message
):
    record = make_record(dense_tensors, 1)
    mutation(record)
    path = tmp_path / "corrupt.tar"
    write_unchecked_shard(path, record)
    with pytest.raises(DenseFeatureValidationError, match=message):
        validate_dense_shard(path)


def test_non_probability_map_is_rejected(tmp_path, dense_tensors):
    record = make_record(dense_tensors, 1)
    record["self_attn_maps"] = record["self_attn_maps"].clone()
    record["self_attn_maps"][0, 0] = -0.25
    path = tmp_path / "negative.tar"
    write_unchecked_shard(path, record)
    with pytest.raises(DenseFeatureValidationError, match="negative"):
        validate_dense_shard(path)


def test_head_reconstruction_failure_is_reported(tmp_path, dense_tensors):
    record = make_record(dense_tensors, 1)
    record["disentangled_self_attn"] = -record["disentangled_self_attn"]
    path = tmp_path / "mismatch.tar"
    write_unchecked_shard(path, record)
    result = validate_dense_shard(path)
    assert result["failed_images"] == [1]
    assert result["cosine_max"] < -0.999
