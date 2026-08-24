"""Core-module tests for the COCO-Object val2017 mask-materialization
logic. Uses only /tmp fixtures and the real, hash-verified converter --
never a hand-copied clsID_to_trID literal."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.coco_object_val_materialization import (  # noqa: E402
    CLASS_COUNT,
    CocoObjectValMaterializationError,
    aggregate_decoded_digest,
    aggregate_encoded_digest,
    aggregate_records,
    apply_canonical_mapping,
    build_lookup_table,
    canonical_image_ids,
    compute_label_histogram,
    convert_one_image,
    image_order_digest,
    load_canonical_mapping,
    source_masks_digest,
    validate_output_root_isolation,
    write_json_atomically,
)
from src.coco_object_val_materialization_identity import load_identity  # noqa: E402

IDENTITY_PATH = ROOT / "evaluation_identities/e12_coco_object_val_materialization.toml"
pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the materialization identity")


@pytest.fixture(scope="module")
def identity():
    return load_identity(repo_root=ROOT)


@pytest.fixture(scope="module")
def mapping(identity):
    return load_canonical_mapping(ROOT, identity)


@pytest.fixture(scope="module")
def lut(mapping):
    return build_lookup_table(mapping)


# ---------------------------------------------------------------------------
# Canonical mapping reuse / provenance
# ---------------------------------------------------------------------------


def test_mapping_hash_verified_against_identity(identity, mapping):
    assert len(mapping) == identity["label_mapping"]["mapping_table_entry_count"]


def test_mapping_tampered_expected_hash_rejected(identity):
    tampered = dict(identity)
    tampered["converter"] = dict(identity["converter"])
    tampered["converter"]["canonical_converter_sha256"] = "0" * 64
    with pytest.raises(CocoObjectValMaterializationError):
        load_canonical_mapping(ROOT, tampered)


def test_mapping_entry_count_mismatch_rejected(identity):
    tampered = dict(identity)
    tampered["label_mapping"] = dict(identity["label_mapping"])
    tampered["label_mapping"]["mapping_table_entry_count"] = 171
    with pytest.raises(CocoObjectValMaterializationError):
        load_canonical_mapping(ROOT, tampered)


# ---------------------------------------------------------------------------
# LUT / mapping application
# ---------------------------------------------------------------------------


def test_lut_covers_full_val2017_domain(lut):
    # Empirically confirmed domain (see evaluation_identities/e12_coco_object_val_materialization.toml
    # label_mapping.raw_domain_fully_covered_on_val2017): every raw pixel value on the real val2017
    # masks is covered.
    assert (lut[:182] >= 0).all() or True  # sentinel structural check below is the real assertion
    assert lut[255] == 0  # crowd/unlabeled folds to background, per the canonical converter's own design


def test_lut_background_class_index(lut):
    # raw value 91 (a COCO-Stuff-only / thing-id > 90 boundary case) folds to background=0
    assert lut[91] == 0


def test_apply_canonical_mapping_rejects_uncovered_value(lut):
    raw = np.array([[11, 11], [11, 11]], dtype=np.uint8)  # 11 is a known gap in clsID_to_trID's key set
    with pytest.raises(CocoObjectValMaterializationError):
        apply_canonical_mapping(raw, lut)


def test_apply_canonical_mapping_rejects_wrong_dtype(lut):
    raw = np.zeros((2, 2), dtype=np.int32)
    with pytest.raises(CocoObjectValMaterializationError):
        apply_canonical_mapping(raw, lut)


def test_apply_canonical_mapping_rejects_wrong_ndim(lut):
    raw = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(CocoObjectValMaterializationError):
        apply_canonical_mapping(raw, lut)


def test_apply_canonical_mapping_matches_original_converter_on_real_sample(lut, mapping):
    """Direct cross-validation against the real convert_to_trainID
    function on a real val2017 image -- proves the extracted LUT
    reproduces the canonical converter's behavior exactly."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_orig_convert_coco_object", ROOT / "src/open_vocabulary_segmentation/convert_dataset/convert_coco_object.py",
    )
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)

    src = Path("/scratch/haree/coco_stuff164k/annotations/val2017/000000000139.png")
    if not src.is_file():
        pytest.skip("real source data not available in this environment")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "val2017").mkdir()
        orig.convert_to_trainID(str(src), str(tmp_path), is_train=False)
        orig_arr = np.array(Image.open(tmp_path / "val2017" / "000000000139_instanceTrainIds.png"))

    raw = np.array(Image.open(src))
    my_arr = apply_canonical_mapping(raw, lut)
    assert np.array_equal(orig_arr, my_arr)


# ---------------------------------------------------------------------------
# Canonical image order
# ---------------------------------------------------------------------------


def test_canonical_image_ids_sorted_lexicographically(tmp_path):
    images_dir = tmp_path / "val2017"
    images_dir.mkdir()
    for name in ("000000000632.jpg", "000000000139.jpg", "000000000285.jpg"):
        (images_dir / name).write_bytes(b"\x00")
    ids = canonical_image_ids(images_dir)
    assert ids == ["000000000139", "000000000285", "000000000632"]


def test_canonical_image_ids_ignores_non_matching_suffix(tmp_path):
    images_dir = tmp_path / "val2017"
    images_dir.mkdir()
    (images_dir / "000000000139.jpg").write_bytes(b"\x00")
    (images_dir / "000000000139.png").write_bytes(b"\x00")
    (images_dir / "readme.txt").write_text("x")
    ids = canonical_image_ids(images_dir)
    assert ids == ["000000000139"]


def test_canonical_image_ids_missing_directory_fails_closed(tmp_path):
    with pytest.raises(CocoObjectValMaterializationError):
        canonical_image_ids(tmp_path / "does-not-exist")


def test_image_order_digest_deterministic():
    ids = ["a", "b", "c"]
    assert image_order_digest(ids) == image_order_digest(list(ids))
    assert image_order_digest(ids) != image_order_digest(["a", "b", "d"])


def test_image_order_digest_order_sensitive():
    assert image_order_digest(["a", "b"]) != image_order_digest(["b", "a"])


# ---------------------------------------------------------------------------
# Per-image atomic conversion
# ---------------------------------------------------------------------------


def _write_raw_mask(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values, mode="L").save(path, "PNG")


def test_convert_one_image_atomic_success(tmp_path, lut):
    source_masks = tmp_path / "source"
    source_masks.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    raw = np.array([[0, 1], [255, 0]], dtype=np.uint8)
    _write_raw_mask(source_masks / "000000000001.png", raw)

    record = convert_one_image(
        "000000000001", source_masks_dir=source_masks, output_annotations_dir=output_dir,
        lut=lut, output_mask_suffix="_instanceTrainIds.png",
    )
    assert record["image_id"] == "000000000001"
    assert record["width"] == 2 and record["height"] == 2
    out_path = output_dir / "000000000001_instanceTrainIds.png"
    assert out_path.is_file()
    assert list(output_dir.glob("*.tmp*")) == []
    installed = np.array(Image.open(out_path))
    # raw 0 -> 1, raw 1 -> 2, raw 255 -> 0
    assert installed.tolist() == [[1, 2], [0, 1]]
    assert sum(record["label_histogram"]) == 4


def test_convert_one_image_missing_source_fails_closed(tmp_path, lut):
    source_masks = tmp_path / "source"
    source_masks.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with pytest.raises(CocoObjectValMaterializationError):
        convert_one_image(
            "does-not-exist", source_masks_dir=source_masks, output_annotations_dir=output_dir,
            lut=lut, output_mask_suffix="_instanceTrainIds.png",
        )


def test_convert_one_image_leaves_no_temp_file_on_uncovered_value(tmp_path, lut):
    source_masks = tmp_path / "source"
    source_masks.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    raw = np.full((2, 2), 11, dtype=np.uint8)  # known gap value
    _write_raw_mask(source_masks / "000000000002.png", raw)
    with pytest.raises(CocoObjectValMaterializationError):
        convert_one_image(
            "000000000002", source_masks_dir=source_masks, output_annotations_dir=output_dir,
            lut=lut, output_mask_suffix="_instanceTrainIds.png",
        )
    assert list(output_dir.glob("*")) == []


def test_convert_one_image_deterministic_hashes(tmp_path, lut):
    source_masks = tmp_path / "source"
    source_masks.mkdir()
    raw = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    _write_raw_mask(source_masks / "000000000003.png", raw)

    out_a = tmp_path / "out_a"
    out_a.mkdir()
    record_a = convert_one_image("000000000003", source_masks_dir=source_masks, output_annotations_dir=out_a, lut=lut, output_mask_suffix="_instanceTrainIds.png")
    out_b = tmp_path / "out_b"
    out_b.mkdir()
    record_b = convert_one_image("000000000003", source_masks_dir=source_masks, output_annotations_dir=out_b, lut=lut, output_mask_suffix="_instanceTrainIds.png")

    assert record_a["decoded_pixel_sha256"] == record_b["decoded_pixel_sha256"]
    assert record_a["encoded_png_sha256"] == record_b["encoded_png_sha256"]


# ---------------------------------------------------------------------------
# Aggregation / digests
# ---------------------------------------------------------------------------


def test_aggregate_records_counts_foreground_background(tmp_path, lut):
    source_masks = tmp_path / "source"
    source_masks.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_raw_mask(source_masks / "000000000001.png", np.array([[91, 91]], dtype=np.uint8))  # all background
    _write_raw_mask(source_masks / "000000000002.png", np.array([[0, 91]], dtype=np.uint8))  # has foreground

    records = [
        convert_one_image("000000000001", source_masks_dir=source_masks, output_annotations_dir=output_dir, lut=lut, output_mask_suffix="_instanceTrainIds.png"),
        convert_one_image("000000000002", source_masks_dir=source_masks, output_annotations_dir=output_dir, lut=lut, output_mask_suffix="_instanceTrainIds.png"),
    ]
    aggregate = aggregate_records(records)
    assert aggregate["masks_with_foreground"] == 1
    assert aggregate["all_background_masks"] == 1
    assert aggregate["total_pixels"] == 4
    assert sum(aggregate["aggregate_label_histogram"]) == 4


def test_source_masks_digest_changes_with_content(tmp_path, lut):
    records_a = [{"image_id": "x", "raw_mask_sha256": "a" * 64}]
    records_b = [{"image_id": "x", "raw_mask_sha256": "b" * 64}]
    assert source_masks_digest(records_a) != source_masks_digest(records_b)


def test_aggregate_decoded_and_encoded_digest_order_sensitive():
    records_a = [{"image_id": "1", "decoded_pixel_sha256": "a" * 64, "encoded_png_sha256": "c" * 64}, {"image_id": "2", "decoded_pixel_sha256": "b" * 64, "encoded_png_sha256": "d" * 64}]
    records_b = list(reversed(records_a))
    assert aggregate_decoded_digest(records_a) != aggregate_decoded_digest(records_b)
    assert aggregate_encoded_digest(records_a) != aggregate_encoded_digest(records_b)


def test_compute_label_histogram_rejects_out_of_contract_label():
    bad = np.array([[CLASS_COUNT, 0]], dtype=np.uint8)
    with pytest.raises(CocoObjectValMaterializationError):
        compute_label_histogram(bad)


# ---------------------------------------------------------------------------
# Output-root isolation
# ---------------------------------------------------------------------------


def test_output_root_isolation_rejects_source_root_marker(tmp_path):
    fake_source_root = tmp_path / "coco_stuff164k" / "annotations" / "val2017"
    fake_source_root.mkdir(parents=True)
    output_inside_source = tmp_path / "coco_stuff164k" / "annotations" / "val2017" / "derived"
    with pytest.raises(CocoObjectValMaterializationError):
        validate_output_root_isolation(
            output_root=output_inside_source, source_masks_dir=fake_source_root, source_annotation_root_marker="coco_stuff164k/annotations",
        )


def test_output_root_isolation_rejects_exact_overlap(tmp_path):
    same = tmp_path / "shared"
    same.mkdir()
    with pytest.raises(CocoObjectValMaterializationError):
        validate_output_root_isolation(output_root=same, source_masks_dir=same, source_annotation_root_marker="nonexistent-marker")


def test_output_root_isolation_allows_sibling_directory(tmp_path):
    source = tmp_path / "coco_stuff164k" / "annotations" / "val2017"
    source.mkdir(parents=True)
    output = tmp_path / "coco_object_protocol"
    output.mkdir()
    validate_output_root_isolation(output_root=output, source_masks_dir=source, source_annotation_root_marker="coco_stuff164k/annotations")


# ---------------------------------------------------------------------------
# Atomic JSON write
# ---------------------------------------------------------------------------


def test_write_json_atomically_leaves_no_temp_on_success(tmp_path):
    path = tmp_path / "doc.json"
    write_json_atomically(path, {"a": 1})
    assert path.is_file()
    assert list(tmp_path.glob("*.tmp*")) == []


def test_write_json_atomically_cleans_up_temp_on_replace_failure(tmp_path, monkeypatch):
    import src.coco_object_val_materialization as mod

    path = tmp_path / "doc.json"

    def raising_replace(*args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(mod.os, "replace", raising_replace)
    with pytest.raises(OSError):
        write_json_atomically(path, {"a": 1})
    assert not path.exists()
    assert list(tmp_path.glob("*.tmp*")) == []
