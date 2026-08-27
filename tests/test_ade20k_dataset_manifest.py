"""Synthetic (no real data needed) tests for ADE20K dataset-root
resolution and split-scanning edge cases: missing paths, a regular file,
ambiguous roots, incomplete layouts, duplicate/invalid IDs, and
train/val disjointness. CPU-only, no torch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.ade20k_dataset_identity import Ade20kDatasetIdentityError, load_identity  # noqa: E402
from src.ade20k_dataset_manifest import (  # noqa: E402
    canonical_validation_ids,
    check_train_val_disjointness,
    resolve_dataset_root,
)

IDENTITY = load_identity(ROOT / "evaluation_identities/e12_ade20k_dataset_source.toml", repo_root=ROOT)


def _make_layout(base: Path, *, val_count: int = 2, train_count: int = 0, with_masks: bool = True) -> Path:
    ade = base / "ADEChallengeData2016"
    (ade / "images" / "training").mkdir(parents=True)
    (ade / "images" / "validation").mkdir(parents=True)
    (ade / "annotations" / "training").mkdir(parents=True)
    (ade / "annotations" / "validation").mkdir(parents=True)
    for i in range(val_count):
        stem = f"ADE_val_{i + 1:08d}"
        (ade / "images" / "validation" / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
        if with_masks:
            (ade / "annotations" / "validation" / f"{stem}.png").write_bytes(b"\x89PNG")
    for i in range(train_count):
        stem = f"ADE_train_{i + 1:08d}"
        (ade / "images" / "training" / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
    return ade


def test_resolve_missing_root_rejected(tmp_path):
    with pytest.raises(Ade20kDatasetIdentityError):
        resolve_dataset_root(tmp_path / "does-not-exist", IDENTITY)


def test_resolve_regular_file_rejected(tmp_path):
    f = tmp_path / "not-a-dir"
    f.write_text("x")
    with pytest.raises(Ade20kDatasetIdentityError):
        resolve_dataset_root(f, IDENTITY)


def test_resolve_incomplete_layout_rejected(tmp_path):
    ade = tmp_path / "ADEChallengeData2016"
    (ade / "images" / "validation").mkdir(parents=True)
    # annotations/validation and both training dirs missing
    with pytest.raises(Ade20kDatasetIdentityError):
        resolve_dataset_root(tmp_path, IDENTITY)


def test_resolve_direct_root(tmp_path):
    ade = _make_layout(tmp_path, val_count=0)
    resolved = resolve_dataset_root(ade, IDENTITY)
    assert resolved == ade


def test_resolve_parent_root(tmp_path):
    ade = _make_layout(tmp_path, val_count=0)
    resolved = resolve_dataset_root(tmp_path, IDENTITY)
    assert resolved == ade


def test_resolve_ambiguous_nested_roots_rejected(tmp_path):
    _make_layout(tmp_path / "copy_a", val_count=0)
    _make_layout(tmp_path / "copy_b", val_count=0)
    with pytest.raises(Ade20kDatasetIdentityError):
        resolve_dataset_root(tmp_path, IDENTITY)


def test_canonical_ids_wrong_count_rejected(tmp_path):
    ade = _make_layout(tmp_path, val_count=3)
    with pytest.raises(Ade20kDatasetIdentityError):
        canonical_validation_ids(ade, IDENTITY)


def test_canonical_ids_image_mask_mismatch_rejected(tmp_path):
    ade = tmp_path / "ADEChallengeData2016"
    (ade / "images" / "training").mkdir(parents=True)
    (ade / "images" / "validation").mkdir(parents=True)
    (ade / "annotations" / "training").mkdir(parents=True)
    (ade / "annotations" / "validation").mkdir(parents=True)
    (ade / "images" / "validation" / "ADE_val_00000001.jpg").write_bytes(b"x")
    # no matching mask
    with pytest.raises(Ade20kDatasetIdentityError):
        canonical_validation_ids(ade, IDENTITY)


def test_canonical_ids_invalid_syntax_rejected(tmp_path):
    ade = tmp_path / "ADEChallengeData2016"
    (ade / "images" / "training").mkdir(parents=True)
    (ade / "images" / "validation").mkdir(parents=True)
    (ade / "annotations" / "training").mkdir(parents=True)
    (ade / "annotations" / "validation").mkdir(parents=True)
    (ade / "images" / "validation" / "not_ade_syntax.jpg").write_bytes(b"x")
    (ade / "annotations" / "validation" / "not_ade_syntax.png").write_bytes(b"x")
    with pytest.raises(Ade20kDatasetIdentityError):
        canonical_validation_ids(ade, IDENTITY)


def test_train_val_disjointness_absent_training_is_fine(tmp_path):
    ade = _make_layout(tmp_path, val_count=1, train_count=0)
    ids = ["ADE_val_00000001"]
    assert check_train_val_disjointness(ade, IDENTITY, ids) == 0


def test_train_val_disjointness_overlap_rejected(tmp_path):
    ade = _make_layout(tmp_path, val_count=1, train_count=1)
    # force an overlap: a training image with the same id as the val image
    overlapping = ade / "images" / "training" / "ADE_val_00000001.jpg"
    overlapping.write_bytes(b"x")
    ids = ["ADE_val_00000001"]
    with pytest.raises(Ade20kDatasetIdentityError):
        check_train_val_disjointness(ade, IDENTITY, ids)


def test_train_val_disjointness_no_overlap_passes(tmp_path):
    ade = _make_layout(tmp_path, val_count=1, train_count=3)
    ids = ["ADE_val_00000001"]
    assert check_train_val_disjointness(ade, IDENTITY, ids) == 3
