"""Tests for VOC2012 dataset-root resolution and manifest generation/
verification. Synthetic fixtures cover every adversarial split/scan
scenario cheaply; the real dataset is exercised separately (never as
the sole evidence), read from the VOC2012_REAL_DATA_ROOT environment
variable so no machine-specific path is hardcoded here."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.voc2012_dataset_identity import Voc2012DatasetIdentityError, load_identity  # noqa: E402
from src.voc2012_dataset_manifest import (  # noqa: E402
    build_manifest,
    canonical_validation_ids,
    image_order_digest,
    resolve_dataset_root,
    scan_validation_split,
    sha256_file,
    verify_manifest_against_identity,
)

IDENTITY_PATH = ROOT / "evaluation_identities/e12_voc2012_dataset_source.toml"


def _parse_real_data_root(raw: str | None) -> Path | None:
    """Resolve VOC2012_REAL_DATA_ROOT from its raw string value. Absent or
    whitespace-only text yields None -- never Path(""), which would
    silently resolve to the current working directory and could
    incorrectly activate real-data tests."""
    if raw is None or not raw.strip():
        return None
    return Path(raw).expanduser()


def _real_data_root_ready(root: Path | None, identity_path: Path) -> bool:
    """Exact readiness check: the resolved root must exist, be a
    directory, and satisfy the VOC2012 root contract (via the same
    resolution logic the CLI itself uses) before any real-data test may
    run."""
    if root is None or not identity_path.exists() or not root.is_dir():
        return False
    try:
        identity = load_identity(repo_root=ROOT)
        resolve_dataset_root(root, identity)
    except (Voc2012DatasetIdentityError, OSError):
        return False
    return True


REAL_DATA_ROOT = _parse_real_data_root(os.environ.get("VOC2012_REAL_DATA_ROOT"))
REAL_DATA_AVAILABLE = _real_data_root_ready(REAL_DATA_ROOT, IDENTITY_PATH)
pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the voc2012 dataset-source identity")


@pytest.fixture(scope="module")
def identity():
    return load_identity(repo_root=ROOT)


def _write_palette_png(path: Path, *, size, values):
    arr = np.array(values, dtype=np.uint8)
    im = Image.fromarray(arr, mode="P")
    im.putpalette([0, 0, 0] * 256)
    im.save(path)


def _make_synthetic_voc(tmp_path: Path, *, ids: list[str], corrupt_image: str | None = None, corrupt_mask: str | None = None,
                          mismatched_dims: str | None = None, non_p_mode: str | None = None, illegal_label: str | None = None):
    root = tmp_path / "VOCdevkit" / "VOC2012"
    (root / "JPEGImages").mkdir(parents=True)
    (root / "SegmentationClass").mkdir(parents=True)
    (root / "ImageSets" / "Segmentation").mkdir(parents=True)
    (root / "Annotations").mkdir()
    for name in ("train.txt", "val.txt", "trainval.txt"):
        (root / "ImageSets" / "Segmentation" / name).write_text("\n".join(ids) + "\n")

    for split_id in ids:
        img_path = root / "JPEGImages" / f"{split_id}.jpg"
        mask_path = root / "SegmentationClass" / f"{split_id}.png"
        h, w = (4, 4)
        if split_id == mismatched_dims:
            Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(img_path)
            _write_palette_png(mask_path, size=(h + 1, w), values=np.zeros((h + 1, w), dtype=np.uint8))
            continue
        if split_id == corrupt_image:
            img_path.write_bytes(b"not a real jpeg")
            _write_palette_png(mask_path, size=(h, w), values=np.zeros((h, w), dtype=np.uint8))
            continue
        if split_id == corrupt_mask:
            Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(img_path)
            mask_path.write_bytes(b"not a real png")
            continue
        Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(img_path)
        if split_id == non_p_mode:
            Image.fromarray(np.zeros((h, w), dtype=np.uint8), mode="L").save(mask_path)
            continue
        values = np.zeros((h, w), dtype=np.uint8)
        if split_id == illegal_label:
            values[0, 0] = 99
        _write_palette_png(mask_path, size=(h, w), values=values)

    return root


@pytest.fixture
def synthetic_identity(identity):
    patched = copy.deepcopy(identity)
    patched["protocol"] = dict(patched["protocol"])
    patched["protocol"]["expected_image_count"] = 3
    return patched


def test_resolve_dataset_root_direct(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"])
    resolved = resolve_dataset_root(voc_root, synthetic_identity)
    assert resolved == voc_root


def test_resolve_dataset_root_via_parent(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"])
    resolved = resolve_dataset_root(voc_root.parent, synthetic_identity)
    assert resolved == voc_root


def test_resolve_dataset_root_missing_rejected(tmp_path, synthetic_identity):
    with pytest.raises(Voc2012DatasetIdentityError):
        resolve_dataset_root(tmp_path / "does-not-exist", synthetic_identity)


def test_resolve_dataset_root_ambiguous_rejected(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"])
    # make the parent ALSO satisfy the root contract directly, by mirroring
    # the required subpaths one level up -- both `data_root` and
    # `data_root/VOC2012` now qualify as candidates.
    import shutil

    for sub in ("JPEGImages", "SegmentationClass"):
        shutil.copytree(voc_root / sub, voc_root.parent / sub)
    (voc_root.parent / "ImageSets" / "Segmentation").mkdir(parents=True)
    for name in ("train.txt", "val.txt", "trainval.txt"):
        (voc_root.parent / "ImageSets" / "Segmentation" / name).write_text("a\n")
    with pytest.raises(Voc2012DatasetIdentityError):
        resolve_dataset_root(voc_root.parent, synthetic_identity)


def _write_val_txt(voc_root: Path, ids: list[str]) -> None:
    (voc_root / "ImageSets" / "Segmentation" / "val.txt").write_text("\n".join(ids) + "\n")


def test_canonical_validation_ids_ordered(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["c", "a", "b"])
    _write_val_txt(voc_root, ["c", "a", "b"])
    ids = canonical_validation_ids(voc_root, synthetic_identity)
    assert ids == ["c", "a", "b"]  # file order preserved, never re-sorted


def test_canonical_validation_ids_duplicate_rejected(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"])
    _write_val_txt(voc_root, ["a", "a", "b"])
    with pytest.raises(Voc2012DatasetIdentityError):
        canonical_validation_ids(voc_root, synthetic_identity)


def test_canonical_validation_ids_absolute_entry_rejected(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"])
    _write_val_txt(voc_root, ["/etc/passwd", "b", "c"])
    with pytest.raises(Voc2012DatasetIdentityError):
        canonical_validation_ids(voc_root, synthetic_identity)


def test_canonical_validation_ids_traversal_entry_rejected(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"])
    _write_val_txt(voc_root, ["../a", "b", "c"])
    with pytest.raises(Voc2012DatasetIdentityError):
        canonical_validation_ids(voc_root, synthetic_identity)


def test_canonical_validation_ids_blank_line_rejected(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"])
    (voc_root / "ImageSets" / "Segmentation" / "val.txt").write_text("a\n\nb\nc\n")
    with pytest.raises(Voc2012DatasetIdentityError):
        canonical_validation_ids(voc_root, synthetic_identity)


def test_canonical_validation_ids_wrong_count_rejected(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b"])
    _write_val_txt(voc_root, ["a", "b"])
    with pytest.raises(Voc2012DatasetIdentityError):
        canonical_validation_ids(voc_root, synthetic_identity)


def test_scan_missing_image_rejected(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"])
    (voc_root / "JPEGImages" / "b.jpg").unlink()
    with pytest.raises(Voc2012DatasetIdentityError):
        scan_validation_split(voc_root, synthetic_identity, ["a", "b", "c"])


def test_scan_missing_mask_rejected(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"])
    (voc_root / "SegmentationClass" / "b.png").unlink()
    with pytest.raises(Voc2012DatasetIdentityError):
        scan_validation_split(voc_root, synthetic_identity, ["a", "b", "c"])


def test_scan_corrupted_jpeg_rejected(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"], corrupt_image="b")
    with pytest.raises(Voc2012DatasetIdentityError):
        scan_validation_split(voc_root, synthetic_identity, ["a", "b", "c"])


def test_scan_corrupted_png_rejected(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"], corrupt_mask="b")
    with pytest.raises(Voc2012DatasetIdentityError):
        scan_validation_split(voc_root, synthetic_identity, ["a", "b", "c"])


def test_scan_dimension_mismatch_rejected(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"], mismatched_dims="b")
    with pytest.raises(Voc2012DatasetIdentityError):
        scan_validation_split(voc_root, synthetic_identity, ["a", "b", "c"])


def test_scan_illegal_label_rejected(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"], illegal_label="b")
    with pytest.raises(Voc2012DatasetIdentityError):
        scan_validation_split(voc_root, synthetic_identity, ["a", "b", "c"])


def test_scan_non_single_channel_mask_rejected(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"], non_p_mode="b")
    with pytest.raises(Voc2012DatasetIdentityError):
        scan_validation_split(voc_root, synthetic_identity, ["a", "b", "c"])


def test_scan_clean_synthetic_dataset_passes(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"])
    scan = scan_validation_split(voc_root, synthetic_identity, ["a", "b", "c"])
    assert scan["dimension_reconciliation"]["all_agree"] is True
    assert scan["observed_labels"] == [0]


def test_build_manifest_deterministic_apart_from_timestamp(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"])
    ids = ["a", "b", "c"]
    scan = scan_validation_split(voc_root, synthetic_identity, ids)
    m1 = build_manifest(identity=synthetic_identity, identity_sha256="a" * 64, voc_root=voc_root, image_ids=ids, scan=scan, generated_at_utc="t1")
    m2 = build_manifest(identity=synthetic_identity, identity_sha256="a" * 64, voc_root=voc_root, image_ids=ids, scan=scan, generated_at_utc="t2")
    m1_no_ts = {k: v for k, v in m1.items() if k != "generated_at_utc"}
    m2_no_ts = {k: v for k, v in m2.items() if k != "generated_at_utc"}
    assert m1_no_ts == m2_no_ts
    assert m1["generated_at_utc"] != m2["generated_at_utc"]


def test_manifest_no_private_absolute_path(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"])
    ids = ["a", "b", "c"]
    scan = scan_validation_split(voc_root, synthetic_identity, ids)
    manifest = build_manifest(identity=synthetic_identity, identity_sha256="a" * 64, voc_root=voc_root, image_ids=ids, scan=scan, generated_at_utc="t")
    serialized = str(manifest)
    assert str(tmp_path) not in serialized
    assert "/scratch/" not in serialized
    assert manifest["source_root_logical_label"] == "VOC2012"


def test_verify_manifest_against_identity_tampered_image_count_rejected(synthetic_identity):
    manifest = {
        "schema": synthetic_identity["manifest"]["schema_name"],
        "identity_name": synthetic_identity["identity"]["name"],
        "identity_sha256": "a" * 64,
        "source_root_logical_label": "VOC2012",
        "split": synthetic_identity["protocol"]["split"],
        "image_count": synthetic_identity["protocol"]["expected_image_count"] + 1,
        "v21_class_count": 21, "v20_class_count": 20,
        "split_file_sha256": "b" * 64, "image_order_digest": "c" * 64, "image_content_digest": "d" * 64,
        "mask_encoded_content_digest": "e" * 64, "mask_decoded_content_digest": "f" * 64,
        "observed_label_set": [0], "label_histogram_v21": [0] * 21, "ignore_pixel_count": 0,
        "dimension_reconciliation": {"all_agree": True, "mismatches": []},
        "v20_v21_shared_source_assertion": True, "generated_at_utc": "t",
    }
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_dimension_disagreement_rejected(synthetic_identity):
    manifest = {
        "schema": synthetic_identity["manifest"]["schema_name"],
        "identity_name": synthetic_identity["identity"]["name"],
        "identity_sha256": "a" * 64,
        "source_root_logical_label": "VOC2012",
        "split": synthetic_identity["protocol"]["split"],
        "image_count": synthetic_identity["protocol"]["expected_image_count"],
        "v21_class_count": 21, "v20_class_count": 20,
        "split_file_sha256": "b" * 64, "image_order_digest": "c" * 64, "image_content_digest": "d" * 64,
        "mask_encoded_content_digest": "e" * 64, "mask_decoded_content_digest": "f" * 64,
        "observed_label_set": [0], "label_histogram_v21": [0] * 21, "ignore_pixel_count": 0,
        "dimension_reconciliation": {"all_agree": False, "mismatches": [{"image_id": "x"}]},
        "v20_v21_shared_source_assertion": True, "generated_at_utc": "t",
    }
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_malformed_manifest_missing_key_rejected(synthetic_identity):
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity({"schema": "x"}, synthetic_identity, identity_sha256="a" * 64)


def _valid_manifest(synthetic_identity):
    return {
        "schema": synthetic_identity["manifest"]["schema_name"],
        "identity_name": synthetic_identity["identity"]["name"],
        "identity_sha256": "a" * 64,
        "source_root_logical_label": "VOC2012",
        "split": synthetic_identity["protocol"]["split"],
        "image_count": synthetic_identity["protocol"]["expected_image_count"],
        "v21_class_count": synthetic_identity["class_contract"]["v21_class_count"],
        "v20_class_count": synthetic_identity["class_contract"]["v20_class_count"],
        "split_file_sha256": "b" * 64, "image_order_digest": "c" * 64, "image_content_digest": "d" * 64,
        "mask_encoded_content_digest": "e" * 64, "mask_decoded_content_digest": "f" * 64,
        "observed_label_set": [0, 1, 255], "label_histogram_v21": [0] * 21, "ignore_pixel_count": 0,
        "dimension_reconciliation": {"all_agree": True, "mismatches": []},
        "v20_v21_shared_source_assertion": True, "generated_at_utc": "2024-01-01T00:00:00+00:00",
    }


def test_verify_manifest_against_identity_accepts_well_typed_manifest(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_float_image_count_rejected(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    manifest["image_count"] = float(manifest["image_count"])
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_float_v20_class_count_rejected(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    manifest["v20_class_count"] = float(manifest["v20_class_count"])
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_float_v21_class_count_rejected(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    manifest["v21_class_count"] = float(manifest["v21_class_count"])
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_bool_as_int_count_rejected(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    manifest["ignore_pixel_count"] = True
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_numeric_string_count_rejected(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    manifest["image_count"] = str(manifest["image_count"])
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_float_in_observed_label_set_rejected(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    manifest["observed_label_set"] = [0, 1.0, 255]
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_bool_in_observed_label_set_rejected(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    manifest["observed_label_set"] = [0, True, 255]
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_wrong_container_type_for_observed_label_set_rejected(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    manifest["observed_label_set"] = (0, 1, 255)
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_wrong_container_type_for_label_histogram_rejected(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    manifest["label_histogram_v21"] = {str(i): 0 for i in range(21)}
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_float_in_label_histogram_rejected(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    manifest["label_histogram_v21"] = [0.0] * 21
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_wrong_dimension_reconciliation_type_rejected(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    manifest["dimension_reconciliation"] = {"all_agree": 1, "mismatches": []}
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_mismatches_wrong_container_type_rejected(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    manifest["dimension_reconciliation"] = {"all_agree": True, "mismatches": ()}
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_non_boolean_shared_source_assertion_rejected(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    manifest["v20_v21_shared_source_assertion"] = 1
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_non_string_digest_rejected(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    manifest["image_content_digest"] = 12345
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_malformed_timestamp_rejected(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    manifest["generated_at_utc"] = "not-a-timestamp"
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_verify_manifest_against_identity_non_string_timestamp_rejected(synthetic_identity):
    manifest = _valid_manifest(synthetic_identity)
    manifest["generated_at_utc"] = 1704067200
    with pytest.raises(Voc2012DatasetIdentityError):
        verify_manifest_against_identity(manifest, synthetic_identity, identity_sha256="a" * 64)


def test_input_split_and_masks_remain_immutable(tmp_path, synthetic_identity):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"])
    _write_val_txt(voc_root, ["a", "b", "c"])
    split_path = voc_root / "ImageSets" / "Segmentation" / "val.txt"
    mask_path = voc_root / "SegmentationClass" / "a.png"
    before_split = split_path.read_bytes()
    before_mask = mask_path.read_bytes()

    ids = canonical_validation_ids(voc_root, synthetic_identity)
    scan_validation_split(voc_root, synthetic_identity, ids)

    assert split_path.read_bytes() == before_split
    assert mask_path.read_bytes() == before_mask


# ---------------------------------------------------------------------
# VOC2012_REAL_DATA_ROOT resolution logic -- purely synthetic, no
# hardcoded private path, runs unconditionally.
# ---------------------------------------------------------------------


def test_parse_real_data_root_none_when_absent():
    assert _parse_real_data_root(None) is None


def test_parse_real_data_root_none_when_empty():
    assert _parse_real_data_root("") is None


@pytest.mark.parametrize("raw", ["   ", "\t", "\n", " \t\n "])
def test_parse_real_data_root_none_when_whitespace_only(raw):
    assert _parse_real_data_root(raw) is None


def test_parse_real_data_root_resolves_a_set_path(tmp_path):
    target = tmp_path / "some" / "dataset" / "root"
    assert _parse_real_data_root(str(target)) == target


def test_parse_real_data_root_expands_user():
    assert _parse_real_data_root("~/example-voc-root") == Path("~/example-voc-root").expanduser()


def test_parse_real_data_root_never_resolves_empty_string_to_cwd():
    # The defect this guards against: Path("") == Path(".") -- an empty
    # env value must resolve to None, never to the current directory.
    assert Path("") == Path(".")  # documents the footgun being avoided
    assert _parse_real_data_root("") is not Path(".")
    assert _parse_real_data_root("") is None


def test_real_data_root_ready_false_when_root_is_none():
    assert _real_data_root_ready(None, IDENTITY_PATH) is False


def test_real_data_root_ready_false_when_path_missing(tmp_path):
    assert _real_data_root_ready(tmp_path / "does-not-exist", IDENTITY_PATH) is False


def test_real_data_root_ready_false_when_path_is_regular_file(tmp_path):
    file_path = tmp_path / "not-a-directory.txt"
    file_path.write_text("x")
    assert _real_data_root_ready(file_path, IDENTITY_PATH) is False


def test_real_data_root_ready_false_when_directory_without_voc2012_structure(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert _real_data_root_ready(empty_dir, IDENTITY_PATH) is False


def test_real_data_root_ready_false_when_identity_path_missing(tmp_path):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"])
    assert _real_data_root_ready(voc_root.parent, tmp_path / "no-such-identity.toml") is False


def test_real_data_root_ready_true_when_structure_present(tmp_path):
    voc_root = _make_synthetic_voc(tmp_path, ids=["a", "b", "c"])
    assert _real_data_root_ready(voc_root.parent, IDENTITY_PATH) is True


# ---------------------------------------------------------------------
# Real-dataset evidence -- synthetic fixtures above are not the sole
# validation evidence for this module.
# ---------------------------------------------------------------------

pytestmark_real = pytest.mark.skipif(
    not REAL_DATA_AVAILABLE,
    reason="requires VOC2012_REAL_DATA_ROOT to point at a directory satisfying the VOC2012 root contract",
)


@pytestmark_real
def test_real_dataset_root_resolves(identity):
    resolved = resolve_dataset_root(REAL_DATA_ROOT, identity)
    assert resolved.name == "VOC2012"
    assert (resolved / "JPEGImages").is_dir()


@pytestmark_real
def test_real_dataset_canonical_ids_count_and_order(identity):
    voc_root = resolve_dataset_root(REAL_DATA_ROOT, identity)
    ids = canonical_validation_ids(voc_root, identity)
    assert len(ids) == 1449
    assert ids[0] == "2007_000033"
    assert len(set(ids)) == 1449
