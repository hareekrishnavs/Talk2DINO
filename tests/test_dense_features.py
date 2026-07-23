import io
import json
import random
import tarfile

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from src.dataset import ImageAwareBatchLoader

from src.dense_features import (
    DenseFeatureShardWriter,
    DenseFeatureStreamingDataset,
    DenseFeatureValidationError,
    IncompleteDenseFeatureExtraction,
    iter_dense_shard_records,
    validate_dense_dataset,
    validate_dense_shard,
)


CONFIG = {
    "model": "dinov2_vitb14_reg",
    "patch_count": 1024,
    "attention_map_format": "probabilities",
}


class SyntheticImageRecordDataset(IterableDataset):
    """Tiny records exercise scheduling without allocating real dense tensors."""

    def __init__(self, image_count=1000, captions_per_image=5, seed=123):
        super().__init__()
        self.image_count = image_count
        self.captions_per_image = captions_per_image
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        image_ids = list(range(self.image_count))
        random.Random(self.seed + 1_000_003 * self.epoch).shuffle(image_ids)
        for image_id in image_ids[worker_id::worker_count]:
            annotation_ids = [
                image_id * self.captions_per_image + index
                for index in range(self.captions_per_image)
            ]
            yield {
                "image_id": image_id,
                "disentangled_self_attn": torch.tensor([image_id]),
                "patch_tokens": torch.tensor([image_id]),
                "self_attn_maps": torch.tensor([image_id]),
                "captions": [str(index) for index in annotation_ids],
                "ann_feats": [torch.tensor([index]) for index in annotation_ids],
                "annotation_ids": annotation_ids,
            }


def synthetic_image_aware_loader(dataset, num_workers):
    return ImageAwareBatchLoader(
        dataset,
        annotation_count=dataset.image_count * dataset.captions_per_image,
        batch_size=128,
        record_pool_size=256,
        seed=dataset.seed,
        num_workers=num_workers,
    )


def collect_batch_metadata(loader):
    sizes = []
    unique_image_counts = []
    annotation_ids = []
    image_order = []
    for batch in loader:
        image_ids = batch["metadata"]["image_id"].tolist()
        batch_annotation_ids = batch["metadata"]["annotation_id"].tolist()
        sizes.append(len(image_ids))
        unique_image_counts.append(len(set(image_ids)))
        annotation_ids.extend(batch_annotation_ids)
        image_order.extend(image_ids)
    return sizes, unique_image_counts, annotation_ids, image_order


def test_image_aware_batches_have_unique_images_and_complete_coverage():
    dataset = SyntheticImageRecordDataset(image_count=1000, captions_per_image=5)
    loader = synthetic_image_aware_loader(dataset, num_workers=0)
    sizes, unique_counts, annotation_ids, _ = collect_batch_metadata(loader)

    assert len(loader) == len(sizes) == 40
    assert sizes[:-1] == [128] * 39
    assert sizes[-1] == 8
    assert unique_counts == sizes
    assert len(annotation_ids) == len(set(annotation_ids)) == 5000
    assert sorted(annotation_ids) == list(range(5000))


def test_image_aware_batches_preserve_two_worker_disjointness():
    dataset = SyntheticImageRecordDataset(image_count=1000, captions_per_image=5)
    loader = synthetic_image_aware_loader(dataset, num_workers=2)
    sizes, unique_counts, annotation_ids, _ = collect_batch_metadata(loader)

    assert sizes[:-1] == [128] * 39
    assert sizes[-1] == 8
    assert unique_counts == sizes
    assert len(annotation_ids) == len(set(annotation_ids)) == 5000
    assert sorted(annotation_ids) == list(range(5000))


def test_image_aware_seed_and_epoch_are_deterministic():
    dataset = SyntheticImageRecordDataset(image_count=1000, captions_per_image=5)
    loader = synthetic_image_aware_loader(dataset, num_workers=2)
    first = collect_batch_metadata(loader)[3]
    repeated = collect_batch_metadata(loader)[3]
    assert repeated == first

    dataset.set_epoch(1)
    next_epoch = collect_batch_metadata(loader)[3]
    assert next_epoch != first
    dataset.set_epoch(0)
    assert collect_batch_metadata(loader)[3] == first


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


def writer(
    path,
    records,
    images_per_shard=2,
    *,
    source_images=None,
    source_annotations=None,
    max_images=None,
    overwrite=False,
):
    image_ids = [record["image_id"] for record in records]
    annotation_ids = [
        annotation_id
        for record in records
        for annotation_id in record["annotation_ids"]
    ]
    return DenseFeatureShardWriter(
        path,
        "train",
        CONFIG,
        "9f314dc",
        source_images=source_images if source_images is not None else len(image_ids),
        source_annotations=(
            source_annotations
            if source_annotations is not None
            else len(annotation_ids)
        ),
        selected_image_ids=image_ids,
        selected_annotation_ids=annotation_ids,
        max_images=max_images,
        images_per_shard=images_per_shard,
        overwrite=overwrite,
    )


def write_records(path, dense_tensors, count=2, captions=2, images_per_shard=2):
    records = [
        make_record(dense_tensors, image_id, captions) for image_id in range(count)
    ]
    output = writer(path, records, images_per_shard)
    for record in records:
        output.add(record)
    output.finish()
    return records


def test_continuous_image_level_storage_and_atomic_finalization(tmp_path, dense_tensors):
    records = [
        make_record(dense_tensors, 1, caption_count=3),
        make_record(dense_tensors, 2, caption_count=4),
    ]
    output = writer(tmp_path, records, images_per_shard=1)
    output.add(records[0])
    finalized = tmp_path / "train-000000.tar"
    assert finalized.exists()  # Written before the complete input has been consumed.
    assert not (tmp_path / "train-000000.tar.tmp").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["complete"] is False
    stored = list(iter_dense_shard_records(finalized))
    assert len(stored) == 1  # Dense tensors occur once per image, not per caption.
    assert stored[0]["captions"] == ["caption 1-0", "caption 1-1", "caption 1-2"]
    assert stored[0]["patch_tokens"].dtype == torch.float16
    assert stored[0]["self_attn_maps"].dtype == torch.float16
    assert torch.allclose(
        stored[0]["self_attn_maps"].float().sum(-1), torch.ones(12), atol=2e-3
    )
    output.add(records[1])
    output.finish()
    assert (tmp_path / "train-000001.tar").exists()


def test_manifest_and_dataset_validation(tmp_path, dense_tensors):
    write_records(tmp_path, dense_tensors, count=2, captions=2)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["source_commit"] == "9f314dc"
    assert manifest["extraction_config"] == CONFIG
    assert manifest["source_images"] == 2
    assert manifest["source_annotations"] == 4
    assert manifest["selected_images"] == 2
    assert manifest["selected_annotations"] == 4
    assert manifest["complete"] is True
    assert manifest["failed_image_ids"] == []
    assert manifest["images"] == 2
    assert manifest["annotations"] == 4
    assert [shard["images"] for shard in manifest["shards"]] == [2]
    assert all(shard["sha256"] and shard["bytes"] > 0 for shard in manifest["shards"])

    result = validate_dense_dataset(tmp_path)
    assert result["images"] == 2
    assert result["annotations"] == 4
    assert result["complete"] is True
    assert result["coverage_complete"] is True
    assert result["is_pilot"] is False
    assert validate_dense_dataset(tmp_path, require_complete=True)["complete"] is True
    assert result["failed_images"] == []
    assert result["cosine_min"] > 0.999
    assert result["row_sum_min"] == pytest.approx(1.0, abs=2e-3)


def test_partial_shard_is_rejected(tmp_path, dense_tensors):
    record = make_record(dense_tensors, 1)
    output = writer(tmp_path, [record])
    output.add(record)
    partial = tmp_path / "train-000000.tar.tmp"
    with pytest.raises(DenseFeatureValidationError, match="partial"):
        validate_dense_shard(partial)
    output.abort()
    assert partial.exists()


def test_restart_skips_verified_images_without_overwrite(tmp_path, dense_tensors):
    records = [make_record(dense_tensors, 1), make_record(dense_tensors, 2)]
    output = writer(tmp_path, records, images_per_shard=1)
    output.add(records[0])
    output.close()
    first = tmp_path / "train-000000.tar"
    original = first.read_bytes()

    # Simulate a crash after atomic shard rename but before manifest replacement.
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["images"] = 0
    manifest["annotations"] = 0
    manifest["shards"] = []
    manifest_path.write_text(json.dumps(manifest))

    resumed = writer(tmp_path, records, images_per_shard=1)
    assert resumed.add(records[0]) is False
    assert resumed.add(records[1]) is True
    resumed.finish()
    assert first.read_bytes() == original
    result = validate_dense_dataset(tmp_path)
    assert result["images"] == 2
    assert result["complete"] is True

    with pytest.raises(FileExistsError, match="immutable"):
        writer(tmp_path, records, images_per_shard=1)


def test_interrupted_extraction_and_reader_incomplete_guard(tmp_path, dense_tensors):
    records = [make_record(dense_tensors, 0), make_record(dense_tensors, 1)]
    output = writer(tmp_path, records, images_per_shard=1)
    output.add(records[0])
    output.close()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["complete"] is False
    assert manifest["images"] == 1
    with pytest.raises(DenseFeatureValidationError, match="incomplete"):
        validate_dense_dataset(tmp_path, require_complete=True)
    with pytest.raises(DenseFeatureValidationError, match="allow_incomplete"):
        DenseFeatureStreamingDataset(tmp_path)

    inspection = DenseFeatureStreamingDataset(tmp_path, allow_incomplete=True)
    assert len(inspection) == 2
    assert len(list(inspection)) == 2


def test_pilot_is_complete_but_cannot_be_reused_as_full(tmp_path, dense_tensors):
    pilot_record = make_record(dense_tensors, 0)
    full_records = [pilot_record, make_record(dense_tensors, 1)]
    pilot = writer(
        tmp_path,
        [pilot_record],
        source_images=2,
        source_annotations=4,
        max_images=1,
    )
    pilot.add(pilot_record)
    pilot.finish()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["is_pilot"] is True
    assert manifest["max_images"] == 1
    assert manifest["source_images"] == 2
    assert manifest["selected_images"] == 1
    assert validate_dense_dataset(tmp_path)["is_pilot"] is True
    with pytest.raises(DenseFeatureValidationError, match="is_pilot=True"):
        validate_dense_dataset(tmp_path, require_complete=True)
    with pytest.raises(DenseFeatureValidationError, match="allow_pilot"):
        DenseFeatureStreamingDataset(tmp_path)
    pilot_dataset = DenseFeatureStreamingDataset(tmp_path, allow_pilot=True)
    assert len(pilot_dataset) == 2
    assert len(list(pilot_dataset)) == 2

    with pytest.raises(FileExistsError, match="different extraction configuration"):
        writer(
            tmp_path,
            full_records,
            source_images=2,
            source_annotations=4,
            max_images=None,
        )


def test_incomplete_pilot_requires_both_reader_flags(tmp_path, dense_tensors):
    records = [make_record(dense_tensors, 0), make_record(dense_tensors, 1)]
    output = writer(
        tmp_path,
        records,
        images_per_shard=1,
        source_images=3,
        source_annotations=6,
        max_images=2,
    )
    output.add(records[0])
    output.close()

    with pytest.raises(DenseFeatureValidationError, match="allow_incomplete"):
        DenseFeatureStreamingDataset(tmp_path)
    with pytest.raises(DenseFeatureValidationError, match="allow_pilot"):
        DenseFeatureStreamingDataset(tmp_path, allow_incomplete=True)
    with pytest.raises(DenseFeatureValidationError, match="allow_incomplete"):
        DenseFeatureStreamingDataset(tmp_path, allow_pilot=True)

    inspection = DenseFeatureStreamingDataset(
        tmp_path,
        allow_incomplete=True,
        allow_pilot=True,
    )
    assert len(inspection) == 2
    assert len(list(inspection)) == 2


def test_failed_image_is_persisted_and_completion_fails(tmp_path, dense_tensors):
    record = make_record(dense_tensors, 7)
    output = writer(tmp_path, [record])
    output.record_failure(7)
    with pytest.raises(IncompleteDenseFeatureExtraction, match="failed_image_ids"):
        output.finish()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["complete"] is False
    assert manifest["failed_image_ids"] == [7]
    resumed = writer(tmp_path, [record])
    assert resumed.manifest["failed_image_ids"] == [7]
    resumed.abort()
    with pytest.raises(DenseFeatureValidationError, match="incomplete"):
        validate_dense_dataset(tmp_path, require_complete=True)


def test_missing_image_id_is_detected(tmp_path, dense_tensors):
    records = [make_record(dense_tensors, 0), make_record(dense_tensors, 1)]
    output = writer(tmp_path, records)
    output.add(records[0])
    with pytest.raises(IncompleteDenseFeatureExtraction, match="missing images=1"):
        output.finish()
    assert json.loads((tmp_path / "manifest.json").read_text())["complete"] is False


def test_missing_annotation_id_is_detected(tmp_path, dense_tensors):
    expected = make_record(dense_tensors, 0, caption_count=2)
    incomplete = dict(expected)
    incomplete["captions"] = expected["captions"][:1]
    incomplete["ann_feats"] = expected["ann_feats"][:1]
    incomplete["annotation_ids"] = expected["annotation_ids"][:1]
    output = writer(tmp_path, [expected])
    output.add(incomplete)
    with pytest.raises(IncompleteDenseFeatureExtraction, match="missing annotations=1"):
        output.finish()
    assert json.loads((tmp_path / "manifest.json").read_text())["complete"] is False


def test_reader_expands_annotations_without_dense_copies(tmp_path, dense_tensors):
    write_records(tmp_path, dense_tensors, count=2, captions=3)
    dataset = DenseFeatureStreamingDataset(tmp_path)
    samples = list(dataset)
    assert len(dataset) == len(samples) == 6
    assert [sample["metadata"]["annotation_id"] for sample in samples[:3]] == [0, 1, 2]
    assert all(sample["metadata"]["image_id"] == 0 for sample in samples[:3])
    assert samples[0]["patch_tokens"].data_ptr() == samples[1]["patch_tokens"].data_ptr()
    assert samples[0]["self_attn_maps"].data_ptr() == samples[2]["self_attn_maps"].data_ptr()
    assert set(samples[0]) == {
        "annotation", "image", "metadata", "caption", "patch_tokens", "self_attn_maps"
    }


def test_reader_is_deterministic_and_bounded_shuffle_is_reproducible(tmp_path, dense_tensors):
    write_records(tmp_path, dense_tensors, count=2, captions=2, images_per_shard=1)
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
    write_records(tmp_path, dense_tensors, count=2, captions=2, images_per_shard=1)
    dataset = DenseFeatureStreamingDataset(tmp_path)
    annotation_ids = [
        sample["metadata"]["annotation_id"]
        for sample in DataLoader(dataset, batch_size=None, num_workers=2)
    ]
    assert len(annotation_ids) == len(dataset)
    assert len(annotation_ids) == len(set(annotation_ids))
    assert sorted(annotation_ids) == [0, 1, 10, 11]


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
