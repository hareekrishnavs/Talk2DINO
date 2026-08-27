"""Proves the COCO train2017 and val2017 image-ID sets used by the
COCO-Object protocol confirmation are disjoint -- i.e. no evaluation
image also appears in the training split. This is a structural fact
about the underlying COCO-2017 release (train2017/ and val2017/ are
disjoint official splits, not something this project constructs), but
is verified directly against the real dataset here rather than assumed,
per this evaluation track's own documentation requirement to prove (not
merely claim) train/evaluation split disjointness.

CPU-only; gated on COCO_OBJECT_REAL_DATA_ROOT with no private path in
this tracked file."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _real_data_root() -> Path | None:
    raw = os.environ.get("COCO_OBJECT_REAL_DATA_ROOT")
    if raw is None or not raw.strip():
        return None
    return Path(raw).expanduser()


REAL_DATA_ROOT = _real_data_root()
requires_real_data = pytest.mark.skipif(
    REAL_DATA_ROOT is None or not REAL_DATA_ROOT.exists(),
    reason="requires COCO_OBJECT_REAL_DATA_ROOT to point at the real COCO-Stuff164k archive",
)


def _stems_with_suffix(directory: Path, *, suffix: str) -> set[str]:
    """Uses os.scandir (not Path.iterdir + Path.is_file) so the file-type
    check is read from the directory entry itself on POSIX filesystems,
    never a separate per-entry stat() syscall -- on a networked
    filesystem with 100k+ entries in one directory (train2017), the
    Path-based equivalent issues one extra round-trip per file and is
    unreliably slow under contention."""
    stems: set[str] = set()
    with os.scandir(directory) as it:
        for entry in it:
            if entry.name.endswith(suffix) and entry.is_file(follow_symlinks=False):
                stems.add(entry.name[: -len(suffix)])
    return stems


def _image_ids(split_dir: Path, *, suffix: str = ".jpg") -> set[str]:
    return _stems_with_suffix(split_dir, suffix=suffix)


@requires_real_data
def test_train2017_and_val2017_image_ids_are_disjoint():
    train_dir = REAL_DATA_ROOT / "images" / "train2017"
    val_dir = REAL_DATA_ROOT / "images" / "val2017"
    assert train_dir.is_dir(), f"expected train2017 image directory at {train_dir}"
    assert val_dir.is_dir(), f"expected val2017 image directory at {val_dir}"

    train_ids = _image_ids(train_dir)
    val_ids = _image_ids(val_dir)

    assert len(train_ids) > 0, "train2017 directory is empty"
    assert len(val_ids) == 5000, f"val2017 directory has {len(val_ids)} images, expected the canonical 5000"

    overlap = train_ids & val_ids
    assert overlap == set(), (
        f"train2017 and val2017 image IDs are NOT disjoint -- {len(overlap)} overlapping IDs found, "
        f"e.g. {sorted(overlap)[:5]}; this would mean an evaluation image also appears in the training "
        "split, invalidating any in-domain/held-out evaluation claim"
    )


@requires_real_data
def test_val2017_ids_have_no_duplicates():
    """A prerequisite for the disjointness check to be meaningful: the
    val split itself must not double-count an image under two names."""
    val_dir = REAL_DATA_ROOT / "images" / "val2017"
    all_stems: list[str] = []
    with os.scandir(val_dir) as it:
        for entry in it:
            if entry.name.endswith(".jpg") and entry.is_file(follow_symlinks=False):
                all_stems.append(entry.name[: -len(".jpg")])
    assert len(all_stems) == len(set(all_stems)), "val2017 contains duplicate image IDs"


@requires_real_data
def test_annotation_masks_present_only_for_val_split_not_train():
    """The materialized COCO-Object masks this evaluator reads are
    val2017-only by construction (materialize_coco_object_val.py never
    converts train2017) -- confirm the raw per-pixel mask source
    similarly separates train/val, so there is no path by which a
    training-split annotation could leak into the evaluation masks."""
    raw_masks_root = REAL_DATA_ROOT / "annotations"
    train_masks = raw_masks_root / "train2017"
    val_masks = raw_masks_root / "val2017"
    assert train_masks.is_dir()
    assert val_masks.is_dir()

    def _mask_ids(d: Path) -> set[str]:
        stems: set[str] = set()
        with os.scandir(d) as it:
            for entry in it:
                if not entry.name.endswith(".png"):
                    continue
                if "labelTrainIds" in entry.name or "instanceTrainIds" in entry.name:
                    continue
                if entry.is_file(follow_symlinks=False):
                    stems.add(entry.name[: -len(".png")])
        return stems

    train_mask_ids = _mask_ids(train_masks)
    val_mask_ids = _mask_ids(val_masks)
    overlap = train_mask_ids & val_mask_ids
    assert overlap == set(), f"train2017 and val2017 raw mask IDs are NOT disjoint -- {len(overlap)} overlapping IDs"
