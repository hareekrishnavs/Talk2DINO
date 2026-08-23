"""Adversarial coverage for
``src.dataset_image_identity.reconcile_canonical_image_id`` -- the shared
helper that reconciles the canonical dataset-relative image ID
(``dataset.img_infos[index]["filename"]``) against the mmseg pipeline's
resolved physical path (``img_meta["filename"]``), fixing the real-dataset
regression exposed by SLURM job 20340482:

    dataset[0] image_id './data/coco_stuff164k/images/val2017/000000000139.jpg'
    disagrees with the precomputed canonical order '000000000139.jpg'

Requires no mmcv, mmseg, torch, or CUDA -- pure path-string reconciliation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset_image_identity import reconcile_canonical_image_id
from src.k11_k12_stability_gate_identity import K11K12StabilityGateError


# ---------------------------------------------------------------------------
# Passing cases
# ---------------------------------------------------------------------------


def test_basename_canonical_id_with_prefixed_pipeline_path():
    result = reconcile_canonical_image_id(
        canonical_relative_id="000000000139.jpg",
        image_root="/data/coco_stuff164k/images/val2017",
        pipeline_resolved_filename="/data/coco_stuff164k/images/val2017/000000000139.jpg",
    )
    assert result == "000000000139.jpg"


def test_nested_canonical_id():
    result = reconcile_canonical_image_id(
        canonical_relative_id="subdir/image.jpg",
        image_root="/data/images",
        pipeline_resolved_filename="/data/images/subdir/image.jpg",
    )
    assert result == "subdir/image.jpg"


def test_relative_image_root():
    result = reconcile_canonical_image_id(
        canonical_relative_id="img.jpg",
        image_root="./data/images",
        pipeline_resolved_filename="./data/images/img.jpg",
    )
    assert result == "img.jpg"


def test_absolute_image_root():
    result = reconcile_canonical_image_id(
        canonical_relative_id="img.jpg",
        image_root="/abs/data/images",
        pipeline_resolved_filename="/abs/data/images/img.jpg",
    )
    assert result == "img.jpg"


def test_redundant_dot_slash_in_pipeline_path():
    result = reconcile_canonical_image_id(
        canonical_relative_id="000000000139.jpg",
        image_root="./data/coco_stuff164k/images/val2017",
        pipeline_resolved_filename="./data/coco_stuff164k/images/val2017/000000000139.jpg",
    )
    assert result == "000000000139.jpg"


def test_path_objects_are_accepted():
    result = reconcile_canonical_image_id(
        canonical_relative_id=Path("img.jpg"),
        image_root=Path("/data/images"),
        pipeline_resolved_filename=Path("/data/images/img.jpg"),
    )
    assert result == "img.jpg"


def test_real_coco_stuff_index_0_metadata_structure():
    """Exact reproduction of job 20340482's real inputs (recorded from a
    CPU-only, no-CUDA, no-model dataset construction against the actual
    COCOStuffDataset -- see Section 7's smoke test for the live version of
    this same structure)."""
    result = reconcile_canonical_image_id(
        canonical_relative_id="000000000139.jpg",
        image_root="./data/coco_stuff164k/images/val2017",
        pipeline_resolved_filename="./data/coco_stuff164k/images/val2017/000000000139.jpg",
    )
    assert result == "000000000139.jpg"
    assert "/" not in result
    assert "data" not in result
    assert "coco_stuff164k" not in result


# ---------------------------------------------------------------------------
# Failing cases
# ---------------------------------------------------------------------------


def test_same_basename_different_directory_rejected():
    with pytest.raises(K11K12StabilityGateError, match="disagrees"):
        reconcile_canonical_image_id(
            canonical_relative_id="img.jpg",
            image_root="/data/dir_a",
            pipeline_resolved_filename="/data/dir_b/img.jpg",
        )


def test_a_image_versus_b_image_rejected():
    with pytest.raises(K11K12StabilityGateError, match="disagrees"):
        reconcile_canonical_image_id(
            canonical_relative_id="a/image.jpg",
            image_root="/data",
            pipeline_resolved_filename="/data/b/image.jpg",
        )


def test_canonical_absolute_path_rejected():
    with pytest.raises(K11K12StabilityGateError, match="absolute"):
        reconcile_canonical_image_id(
            canonical_relative_id="/etc/passwd",
            image_root="/data/images",
            pipeline_resolved_filename="/data/images/etc/passwd",
        )


def test_canonical_parent_traversal_rejected():
    with pytest.raises(K11K12StabilityGateError, match="traversal"):
        reconcile_canonical_image_id(
            canonical_relative_id="../image.jpg",
            image_root="/data/images",
            pipeline_resolved_filename="/data/image.jpg",
        )


def test_canonical_nested_traversal_rejected():
    with pytest.raises(K11K12StabilityGateError, match="traversal"):
        reconcile_canonical_image_id(
            canonical_relative_id="subdir/../../etc/passwd",
            image_root="/data/images",
            pipeline_resolved_filename="/etc/passwd",
        )


def test_pipeline_path_outside_expected_image_root_rejected():
    with pytest.raises(K11K12StabilityGateError, match="disagrees"):
        reconcile_canonical_image_id(
            canonical_relative_id="img.jpg",
            image_root="/data/images",
            pipeline_resolved_filename="/other/root/img.jpg",
        )


def test_wrong_filename_rejected():
    with pytest.raises(K11K12StabilityGateError, match="disagrees"):
        reconcile_canonical_image_id(
            canonical_relative_id="000000000139.jpg",
            image_root="/data/images",
            pipeline_resolved_filename="/data/images/000000000999.jpg",
        )


@pytest.mark.parametrize("bad_canonical", ["", "   ", "\t\n"])
def test_empty_and_whitespace_only_canonical_id_rejected(bad_canonical):
    with pytest.raises(K11K12StabilityGateError):
        reconcile_canonical_image_id(
            canonical_relative_id=bad_canonical, image_root="/data", pipeline_resolved_filename="/data/x.jpg",
        )


@pytest.mark.parametrize("bad_root", ["", "   "])
def test_empty_and_whitespace_only_image_root_rejected(bad_root):
    with pytest.raises(K11K12StabilityGateError):
        reconcile_canonical_image_id(
            canonical_relative_id="x.jpg", image_root=bad_root, pipeline_resolved_filename="/data/x.jpg",
        )


@pytest.mark.parametrize("bad_pipeline", ["", "   "])
def test_empty_and_whitespace_only_pipeline_filename_rejected(bad_pipeline):
    with pytest.raises(K11K12StabilityGateError):
        reconcile_canonical_image_id(
            canonical_relative_id="x.jpg", image_root="/data", pipeline_resolved_filename=bad_pipeline,
        )


@pytest.mark.parametrize("bad_value", [True, False, 42, ["x.jpg"], {"filename": "x.jpg"}, None])
def test_non_string_non_pathlike_canonical_id_rejected(bad_value):
    with pytest.raises(K11K12StabilityGateError):
        reconcile_canonical_image_id(
            canonical_relative_id=bad_value, image_root="/data", pipeline_resolved_filename="/data/x.jpg",
        )


@pytest.mark.parametrize("bad_value", [True, 1, ["/data"], None])
def test_non_string_non_pathlike_image_root_rejected(bad_value):
    with pytest.raises(K11K12StabilityGateError):
        reconcile_canonical_image_id(
            canonical_relative_id="x.jpg", image_root=bad_value, pipeline_resolved_filename="/data/x.jpg",
        )


@pytest.mark.parametrize("bad_value", [True, 1, ["/data/x.jpg"], None])
def test_non_string_non_pathlike_pipeline_filename_rejected(bad_value):
    with pytest.raises(K11K12StabilityGateError):
        reconcile_canonical_image_id(
            canonical_relative_id="x.jpg", image_root="/data", pipeline_resolved_filename=bad_value,
        )


def test_nul_byte_in_canonical_id_rejected():
    with pytest.raises(K11K12StabilityGateError, match="NUL"):
        reconcile_canonical_image_id(
            canonical_relative_id="x.jpg\x00evil", image_root="/data", pipeline_resolved_filename="/data/x.jpg",
        )


def test_nul_byte_in_image_root_rejected():
    with pytest.raises(K11K12StabilityGateError, match="NUL"):
        reconcile_canonical_image_id(
            canonical_relative_id="x.jpg", image_root="/data\x00evil", pipeline_resolved_filename="/data/x.jpg",
        )


def test_nul_byte_in_pipeline_filename_rejected():
    with pytest.raises(K11K12StabilityGateError, match="NUL"):
        reconcile_canonical_image_id(
            canonical_relative_id="x.jpg", image_root="/data", pipeline_resolved_filename="/data/x.jpg\x00evil",
        )


def test_pipeline_metadata_with_another_images_path_rejected():
    # Exactly the shape of a mis-shuffled-pipeline hazard: dataset index i's
    # canonical ID reconciled against a physically different image's
    # resolved path.
    with pytest.raises(K11K12StabilityGateError, match="disagrees"):
        reconcile_canonical_image_id(
            canonical_relative_id="000000000139.jpg",
            image_root="./data/coco_stuff164k/images/val2017",
            pipeline_resolved_filename="./data/coco_stuff164k/images/val2017/000000000285.jpg",
        )


# ---------------------------------------------------------------------------
# Returned ID is exactly the canonical one, never a private/full path
# ---------------------------------------------------------------------------


def test_returned_id_is_exactly_canonical_never_the_resolved_physical_path():
    result = reconcile_canonical_image_id(
        canonical_relative_id="000000000139.jpg",
        image_root="/opt/synthetic_cluster_home/example_user/example_project_data/coco_stuff164k/images/val2017",
        pipeline_resolved_filename="/opt/synthetic_cluster_home/example_user/example_project_data/coco_stuff164k/images/val2017/000000000139.jpg",
    )
    assert result == "000000000139.jpg"
    assert "synthetic_cluster_home" not in result
    assert "example_user" not in result
    assert "example_project_data" not in result


# ---------------------------------------------------------------------------
# Two nested files with the same basename remain distinct
# ---------------------------------------------------------------------------


def test_two_nested_files_with_same_basename_remain_distinct():
    result_a = reconcile_canonical_image_id(
        canonical_relative_id="a/img.jpg", image_root="/data", pipeline_resolved_filename="/data/a/img.jpg",
    )
    result_b = reconcile_canonical_image_id(
        canonical_relative_id="b/img.jpg", image_root="/data", pipeline_resolved_filename="/data/b/img.jpg",
    )
    assert result_a == "a/img.jpg"
    assert result_b == "b/img.jpg"
    assert result_a != result_b

    # cross-wiring must fail: b's canonical id against a's resolved path
    with pytest.raises(K11K12StabilityGateError, match="disagrees"):
        reconcile_canonical_image_id(
            canonical_relative_id="b/img.jpg", image_root="/data", pipeline_resolved_filename="/data/a/img.jpg",
        )
