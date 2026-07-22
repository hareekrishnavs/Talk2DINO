import pytest
import torch

from src.dataset import GroupedDinoClipDataset, multicaption_collate_fn


@pytest.fixture
def grouped_feature_file(tmp_path):
    path = tmp_path / "features.pth"
    torch.save(
        {
            "images": [
                {"id": 10, "heads": torch.full((2, 3), 10.0)},
                {"id": 20, "heads": torch.full((2, 3), 20.0)},
            ],
            "annotations": [
                {"id": 1, "image_id": 10, "ann_feats": torch.tensor([1.0, 0.0])},
                {"id": 2, "image_id": 20, "ann_feats": torch.tensor([2.0, 0.0])},
                {"id": 3, "image_id": 10, "ann_feats": torch.tensor([3.0, 0.0])},
                {"id": 4, "image_id": 10, "ann_feats": torch.tensor([4.0, 0.0])},
            ],
        },
        path,
    )
    return path


def test_groups_every_caption_by_image_without_duplication(grouped_feature_file):
    dataset = GroupedDinoClipDataset(
        grouped_feature_file, features_name="heads", text_features="ann_feats"
    )

    assert len(dataset) == 2
    assert [sample["image_id"] for sample in dataset] == [10, 20]
    assert [len(sample["annotation"]) for sample in dataset] == [3, 1]
    torch.testing.assert_close(
        torch.stack(dataset[0]["annotation"]),
        torch.tensor([[1.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
    )
    torch.testing.assert_close(dataset[0]["image"], torch.full((2, 3), 10.0))
    torch.testing.assert_close(dataset[1]["annotation"][0], torch.tensor([2.0, 0.0]))


def test_collate_pads_variable_caption_counts(grouped_feature_file):
    dataset = GroupedDinoClipDataset(
        grouped_feature_file, features_name="heads", text_features="ann_feats"
    )

    batch = multicaption_collate_fn([dataset[0], dataset[1]])

    assert batch["image"].shape == (2, 2, 3)
    assert batch["annotation"].shape == (2, 3, 2)
    assert batch["caption_mask"].dtype == torch.bool
    torch.testing.assert_close(
        batch["caption_mask"],
        torch.tensor([[True, True, True], [True, False, False]]),
    )
    torch.testing.assert_close(batch["annotation"][1, 1:], torch.zeros(2, 2))
    torch.testing.assert_close(batch["image_id"], torch.tensor([10, 20]))


def test_rejects_images_without_captions(tmp_path):
    path = tmp_path / "empty_caption_image.pth"
    torch.save(
        {
            "images": [{"id": 1, "heads": torch.randn(2, 3)}],
            "annotations": [],
        },
        path,
    )

    with pytest.raises(ValueError, match="at least one caption"):
        GroupedDinoClipDataset(path, features_name="heads")


def test_collate_rejects_empty_caption_sample():
    with pytest.raises(ValueError, match="at least one caption"):
        multicaption_collate_fn(
            [{"image": torch.randn(2, 3), "annotation": [], "image_id": 1}]
        )
