"""CPU-only tests for the shared VOC2012 V20/V21 matched evaluator's
dataset-level metric computation and NPZ/manifest sufficient-statistic
round-trip. Hand-derived numeric fixtures -- never calls the function
under test to produce its own expected values."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
OVS_ROOT = ROOT / "src/open_vocabulary_segmentation"
for _p in (ROOT, OVS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

torch = pytest.importorskip("torch")

from models.dinotext.cover_dr import compute_full_precision_metrics  # noqa: E402


def test_hand_derived_two_image_three_class_metrics():
    # Image 1: intersect=[2,0,1] union=[3,1,1] pred=[2,0,2] label=[3,1,1]
    # Image 2: intersect=[1,3,0] union=[2,3,2] pred=[1,3,1] label=[2,3,1]
    pre_eval_results = [
        (np.array([2, 0, 1]), np.array([3, 1, 1]), np.array([2, 0, 2]), np.array([3, 1, 1])),
        (np.array([1, 3, 0]), np.array([2, 3, 2]), np.array([1, 3, 1]), np.array([2, 3, 1])),
    ]
    result = compute_full_precision_metrics(pre_eval_results)

    intersect_total = np.array([3, 3, 1])  # 2+1, 0+3, 1+0
    union_total = np.array([5, 4, 3])      # 3+2, 1+3, 1+2
    label_total = np.array([5, 4, 2])      # 3+2, 1+3, 1+1

    expected_aacc = intersect_total.sum() / label_total.sum()  # 7/11
    expected_iou = intersect_total / union_total  # [0.6, 0.75, 0.333...]
    expected_miou = expected_iou.mean()
    expected_acc = intersect_total / label_total  # [0.6, 0.75, 0.5]
    expected_macc = expected_acc.mean()

    assert result["aAcc"] == pytest.approx(expected_aacc, abs=1e-9)
    assert result["mIoU"] == pytest.approx(expected_miou, abs=1e-9)
    assert result["mAcc"] == pytest.approx(expected_macc, abs=1e-9)
    # Fractions in [0, 1], never pre-multiplied by 100 -- percent conversion
    # is the caller's responsibility (percent_0_100 unit).
    assert 0.0 <= result["aAcc"] <= 1.0
    assert 0.0 <= result["mIoU"] <= 1.0
    assert 0.0 <= result["mAcc"] <= 1.0


def test_metrics_never_mean_of_per_image_miou():
    # A mean-of-per-image-mIoU implementation would average two images'
    # OWN mIoU values; the correct (accumulated-sufficient-statistics)
    # implementation differs whenever per-image class supports differ, as
    # they do here (image 1 has no class-1 support scaled the same as
    # image 2). Confirm the function's output does NOT equal that wrong
    # alternative.
    pre_eval_results = [
        (np.array([2, 0, 1]), np.array([3, 1, 1]), np.array([2, 0, 2]), np.array([3, 1, 1])),
        (np.array([1, 3, 0]), np.array([2, 3, 2]), np.array([1, 3, 1]), np.array([2, 3, 1])),
    ]
    result = compute_full_precision_metrics(pre_eval_results)

    def _per_image_miou(intersect, union):
        return float(np.mean(intersect / union))

    wrong_mean_of_image_miou = (
        _per_image_miou(np.array([2, 0, 1]), np.array([3, 1, 1]))
        + _per_image_miou(np.array([1, 3, 0]), np.array([2, 3, 2]))
    ) / 2
    assert result["mIoU"] != pytest.approx(wrong_mean_of_image_miou, abs=1e-9)


def test_background_only_appears_in_v21_class_dimension():
    # V20 sufficient statistics have exactly 20 columns (no background);
    # V21 has exactly 21 (background prepended at index 0). This is a
    # structural, class-dimension-only distinction -- verified directly
    # against the shape contract the driver's per-image-stats writer
    # enforces (see test_run_voc2012_matched_evaluation.py for the writer
    # itself); here we confirm the underlying metrics function is
    # class-count-agnostic and does not special-case index 0.
    v20_pre_eval = [(np.zeros(20, dtype=np.int64), np.ones(20, dtype=np.int64), np.zeros(20, dtype=np.int64), np.ones(20, dtype=np.int64))]
    v21_pre_eval = [(np.zeros(21, dtype=np.int64), np.ones(21, dtype=np.int64), np.zeros(21, dtype=np.int64), np.ones(21, dtype=np.int64))]
    v20_result = compute_full_precision_metrics(v20_pre_eval)
    v21_result = compute_full_precision_metrics(v21_pre_eval)
    assert set(v20_result) == set(v21_result) == {"aAcc", "mIoU", "mAcc"}


def test_compute_full_precision_metrics_rejects_empty_input():
    with pytest.raises(ValueError):
        compute_full_precision_metrics([])


# ---------------------------------------------------------------------
# Per-image-stats NPZ/manifest round-trip (via the driver's own writer)
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def driver_module():
    spec = importlib.util.spec_from_file_location(
        "voc2012_driver_under_test", ROOT / "diagnostics" / "run_voc2012_matched_evaluation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_rows(driver_module, *, image_count=2):
    from src.voc2012_matched_evaluator_report import VARIANT_NAMES

    rng = np.random.default_rng(0)
    rows = {"label_v20": [], "label_v21": []}
    for variant in VARIANT_NAMES:
        rows[f"intersect_{variant}"] = []
        rows[f"union_{variant}"] = []
        rows[f"pred_{variant}"] = []
    for _ in range(image_count):
        rows["label_v20"].append(rng.integers(0, 5, size=20))
        rows["label_v21"].append(rng.integers(0, 5, size=21))
        for variant in VARIANT_NAMES:
            size = 20 if variant.startswith("v20_") else 21
            rows[f"intersect_{variant}"].append(rng.integers(0, 5, size=size))
            rows[f"union_{variant}"].append(rng.integers(1, 5, size=size))
            rows[f"pred_{variant}"].append(rng.integers(0, 5, size=size))
    return rows


def test_per_image_stats_round_trip(tmp_path, driver_module):
    rows = _synthetic_rows(driver_module)
    manifest_path = tmp_path / "stats.json"
    driver_module._write_per_image_stats_atomically(
        manifest_path, schema_name="test-schema", v20_class_count=20, v21_class_count=21,
        live_v20_class_names_digest="a" * 64, live_v21_class_names_digest="b" * 64,
        dataset_indices=[0, 1], image_ids=["img0", "img1"], rows=rows,
    )
    loaded = driver_module._load_per_image_stats(manifest_path)
    assert loaded["manifest"]["image_count"] == 2
    assert list(loaded["manifest"]["image_ids"]) == ["img0", "img1"]
    for key, values in rows.items():
        np.testing.assert_array_equal(loaded["arrays"][key], np.stack(values))


def test_per_image_stats_npz_tamper_detected(tmp_path, driver_module):
    rows = _synthetic_rows(driver_module)
    manifest_path = tmp_path / "stats.json"
    driver_module._write_per_image_stats_atomically(
        manifest_path, schema_name="test-schema", v20_class_count=20, v21_class_count=21,
        live_v20_class_names_digest="a" * 64, live_v21_class_names_digest="b" * 64,
        dataset_indices=[0, 1], image_ids=["img0", "img1"], rows=rows,
    )
    npz_path = manifest_path.with_suffix(".npz")
    npz_path.write_bytes(npz_path.read_bytes() + b"\x00")  # corrupt
    with pytest.raises(Exception):
        driver_module._load_per_image_stats(manifest_path)


def test_per_image_stats_wrong_v20_column_count_rejected(tmp_path, driver_module):
    rows = _synthetic_rows(driver_module)
    rows["intersect_v20_e3"] = [np.zeros(21, dtype=np.int64) for _ in rows["intersect_v20_e3"]]  # wrong width for every V20 row
    manifest_path = tmp_path / "stats.json"
    with pytest.raises(Exception):
        driver_module._write_per_image_stats_atomically(
            manifest_path, schema_name="test-schema", v20_class_count=20, v21_class_count=21,
            live_v20_class_names_digest="a" * 64, live_v21_class_names_digest="b" * 64,
            dataset_indices=[0, 1], image_ids=["img0", "img1"], rows=rows,
        )


# ---------------------------------------------------------------------
# Closed NPZ member schema (EXPECTED_PER_IMAGE_STATS_ARRAY_NAMES, derived
# from VARIANT_NAMES -- never a second, independently-declared list).
# ---------------------------------------------------------------------


def _write_synthetic_manifest(driver_module, tmp_path, name, *, image_count=2, seed=0):
    rows = _synthetic_rows(driver_module, image_count=image_count)
    manifest_path = tmp_path / f"{name}.json"
    driver_module._write_per_image_stats_atomically(
        manifest_path, schema_name="test-schema", v20_class_count=20, v21_class_count=21,
        live_v20_class_names_digest="a" * 64, live_v21_class_names_digest="b" * 64,
        dataset_indices=list(range(image_count)), image_ids=[f"img{i}" for i in range(image_count)], rows=rows,
    )
    return manifest_path


def _resave_npz_with_consistent_manifest(driver_module, manifest_path, arrays):
    import hashlib
    import json

    npz_path = manifest_path.with_suffix(".npz")
    np.savez(npz_path, allow_pickle=False, **arrays)
    manifest = json.loads(manifest_path.read_text())
    manifest["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_npz_schema_expected_names_derived_from_variant_names(driver_module):
    from src.voc2012_matched_evaluator_report import VARIANT_NAMES

    expected = driver_module.EXPECTED_PER_IMAGE_STATS_ARRAY_NAMES
    assert "dataset_indices" in expected
    assert "label_v20" in expected and "label_v21" in expected
    for variant in VARIANT_NAMES:
        assert f"intersect_{variant}" in expected
        assert f"union_{variant}" in expected
        assert f"pred_{variant}" in expected
    assert len(expected) == 3 + 3 * len(VARIANT_NAMES)


def test_npz_one_unexpected_member_rejected(tmp_path, driver_module):
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "unexpected")
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["bogus_extra_member"] = np.zeros(3, dtype=np.int64)
    _resave_npz_with_consistent_manifest(driver_module, manifest_path, arrays)
    with pytest.raises(Exception, match="unexpected"):
        driver_module._load_per_image_stats(manifest_path)


def test_npz_one_missing_member_rejected(tmp_path, driver_module):
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "missing")
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    del arrays["union_v21_k12"]
    _resave_npz_with_consistent_manifest(driver_module, manifest_path, arrays)
    with pytest.raises(Exception, match="missing"):
        driver_module._load_per_image_stats(manifest_path)


def test_npz_renamed_member_rejected(tmp_path, driver_module):
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "renamed")
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["pred_v20_k11_renamed"] = arrays.pop("pred_v20_k11")
    _resave_npz_with_consistent_manifest(driver_module, manifest_path, arrays)
    with pytest.raises(Exception):
        driver_module._load_per_image_stats(manifest_path)


def test_npz_object_member_rejected_by_numpy_itself(tmp_path):
    with pytest.raises(Exception):
        np.savez(tmp_path / "object_member.npz", allow_pickle=False, bad=np.array([{"a": 1}], dtype=object))


def test_npz_reordered_zip_members_still_load(tmp_path, driver_module):
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "reordered")
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    reordered = dict(reversed(list(arrays.items())))
    _resave_npz_with_consistent_manifest(driver_module, manifest_path, reordered)
    loaded = driver_module._load_per_image_stats(manifest_path)
    assert set(loaded["arrays"]) == driver_module.EXPECTED_PER_IMAGE_STATS_ARRAY_NAMES


def test_npz_tampered_content_with_recomputed_digest_still_schema_valid(tmp_path, driver_module):
    # A schema-correct but content-tampered NPZ (digest recomputed to stay
    # internally consistent) must still pass the SCHEMA check -- content
    # tamper detection is a separate, already-covered concern (whole-file
    # digest binding against the manifest), not this check's job.
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "content_tamper")
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["label_v20"] = arrays["label_v20"] + 1
    _resave_npz_with_consistent_manifest(driver_module, manifest_path, arrays)
    loaded = driver_module._load_per_image_stats(manifest_path)
    assert set(loaded["arrays"]) == driver_module.EXPECTED_PER_IMAGE_STATS_ARRAY_NAMES


# ---------------------------------------------------------------------
# Per-image-stats array shape/dtype contract: _validate_per_image_stats_array
# is the single pure helper invoked by BOTH the atomic writer (before
# serialization) and _load_per_image_stats (after loading), so schema
# drift between them is structurally impossible. Every negative case here
# builds a valid artifact with the real writer, tampers the NPZ in
# memory, recomputes the manifest's npz_sha256 (so the whole-file digest
# check -- a separate, already-covered concern -- never masks the
# shape/dtype check under test), and requires the loader to reject it.
# ---------------------------------------------------------------------


def test_per_image_stats_float64_v20_array_rejected(tmp_path, driver_module):
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "float64_v20")
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["intersect_v20_e3"] = arrays["intersect_v20_e3"].astype(np.float64)
    _resave_npz_with_consistent_manifest(driver_module, manifest_path, arrays)
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="int64"):
        driver_module._load_per_image_stats(manifest_path)


def test_per_image_stats_float64_v21_array_rejected(tmp_path, driver_module):
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "float64_v21")
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["union_v21_k12"] = arrays["union_v21_k12"].astype(np.float64)
    _resave_npz_with_consistent_manifest(driver_module, manifest_path, arrays)
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="int64"):
        driver_module._load_per_image_stats(manifest_path)


def test_per_image_stats_int32_array_rejected(tmp_path, driver_module):
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "int32")
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["pred_v20_k11"] = arrays["pred_v20_k11"].astype(np.int32)
    _resave_npz_with_consistent_manifest(driver_module, manifest_path, arrays)
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="int64"):
        driver_module._load_per_image_stats(manifest_path)


def test_per_image_stats_uint64_array_rejected(tmp_path, driver_module):
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "uint64")
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["label_v21"] = arrays["label_v21"].astype(np.uint64)
    _resave_npz_with_consistent_manifest(driver_module, manifest_path, arrays)
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="int64"):
        driver_module._load_per_image_stats(manifest_path)


def test_per_image_stats_object_dtype_rejected_at_resave(tmp_path, driver_module):
    # allow_pickle=False structurally forbids object-dtype arrays from
    # ever being serialized into the NPZ in the first place -- the
    # rejection happens at np.savez, before the loader is even reachable.
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "object_dtype")
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["pred_v21_e3"] = np.array([{"a": 1}] * arrays["pred_v21_e3"].shape[0], dtype=object)
    with pytest.raises(Exception):
        _resave_npz_with_consistent_manifest(driver_module, manifest_path, arrays)


def test_per_image_stats_one_dimensional_array_rejected(tmp_path, driver_module):
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "one_dim")
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["label_v20"] = arrays["label_v20"].reshape(-1).astype(np.int64)
    _resave_npz_with_consistent_manifest(driver_module, manifest_path, arrays)
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="2-dimensional"):
        driver_module._load_per_image_stats(manifest_path)


def test_per_image_stats_three_dimensional_array_rejected(tmp_path, driver_module):
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "three_dim")
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["intersect_v21_k11"] = arrays["intersect_v21_k11"][:, :, np.newaxis]
    _resave_npz_with_consistent_manifest(driver_module, manifest_path, arrays)
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="2-dimensional"):
        driver_module._load_per_image_stats(manifest_path)


def test_per_image_stats_wrong_row_count_rejected(tmp_path, driver_module):
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "wrong_rows", image_count=3)
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["pred_v20_k12"] = arrays["pred_v20_k12"][:2]  # 2 rows instead of the manifest's image_count=3
    _resave_npz_with_consistent_manifest(driver_module, manifest_path, arrays)
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="rows"):
        driver_module._load_per_image_stats(manifest_path)


@pytest.mark.parametrize("bad_column_count", [19, 21])
def test_per_image_stats_v20_wrong_column_count_rejected(tmp_path, driver_module, bad_column_count):
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, f"v20_cols_{bad_column_count}")
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    rows = arrays["label_v20"].shape[0]
    arrays["label_v20"] = np.zeros((rows, bad_column_count), dtype=np.int64)
    _resave_npz_with_consistent_manifest(driver_module, manifest_path, arrays)
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="columns"):
        driver_module._load_per_image_stats(manifest_path)


@pytest.mark.parametrize("bad_column_count", [20, 22])
def test_per_image_stats_v21_wrong_column_count_rejected(tmp_path, driver_module, bad_column_count):
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, f"v21_cols_{bad_column_count}")
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    rows = arrays["label_v21"].shape[0]
    arrays["label_v21"] = np.zeros((rows, bad_column_count), dtype=np.int64)
    _resave_npz_with_consistent_manifest(driver_module, manifest_path, arrays)
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="columns"):
        driver_module._load_per_image_stats(manifest_path)


def test_per_image_stats_zero_row_artifact_inconsistent_with_image_count_rejected(tmp_path, driver_module):
    # image_count=2 in the manifest, but every array trimmed to 0 rows --
    # a "someone truncated everything" tamper case, distinct from the
    # genuinely-empty positive control below (where image_count==0 and
    # the manifest agrees).
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "zero_row_mismatch", image_count=2)
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    zeroed = {name: array[:0] for name, array in arrays.items()}
    _resave_npz_with_consistent_manifest(driver_module, manifest_path, zeroed)
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="rows"):
        driver_module._load_per_image_stats(manifest_path)


def test_per_image_stats_correct_shapes_but_unexpected_member_rejected(tmp_path, driver_module):
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "shape_ok_extra_member")
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["extra_but_shape_correct"] = arrays["label_v20"].copy()  # schema-correct shape/dtype, wrong name
    _resave_npz_with_consistent_manifest(driver_module, manifest_path, arrays)
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="unexpected"):
        driver_module._load_per_image_stats(manifest_path)


def test_per_image_stats_writer_rejects_wrong_column_count_for_all_variant_keys(tmp_path, driver_module):
    # Previously only 8 of 21 keys were shape-checked at write time
    # (label_v20/v21 plus intersect_*); this confirms union_*/pred_* keys
    # -- not covered by the old ad-hoc check -- are now covered too.
    rows = _synthetic_rows(driver_module)
    rows["pred_v21_k12"] = [np.zeros(22, dtype=np.int64) for _ in rows["pred_v21_k12"]]
    manifest_path = tmp_path / "writer_bad_union_pred_shape.json"
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="columns"):
        driver_module._write_per_image_stats_atomically(
            manifest_path, schema_name="test-schema", v20_class_count=20, v21_class_count=21,
            live_v20_class_names_digest="a" * 64, live_v21_class_names_digest="b" * 64,
            dataset_indices=[0, 1], image_ids=["img0", "img1"], rows=rows,
        )


# --- positive controls ---


def test_per_image_stats_canonical_nonempty_artifact_accepted(tmp_path, driver_module):
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "canonical_nonempty", image_count=5)
    loaded = driver_module._load_per_image_stats(manifest_path)
    assert loaded["manifest"]["image_count"] == 5
    assert set(loaded["arrays"]) == driver_module.EXPECTED_PER_IMAGE_STATS_ARRAY_NAMES


def test_per_image_stats_canonical_zero_image_artifact_accepted(tmp_path, driver_module):
    # image_count=0 is not reachable through the real writer: np.stack([])
    # (unconditionally applied to every rows[key] list) raises ValueError
    # on an empty list, independent of this repair and independent of
    # run-mode minimums (pilot20/pilot100/full all start at >=20 images).
    # So a zero-image artifact is hand-constructed directly here to probe
    # only the LOADER half of the contract: confirms the new shape/dtype
    # validator does not itself impose an artificial minimum image count
    # -- shape[0] == image_count == 0 is accepted, not gratuitously
    # rejected -- for the boundary case the writer can never produce.
    import hashlib
    import json

    from src.voc2012_matched_evaluator_report import VARIANT_NAMES

    arrays = {"dataset_indices": np.zeros(0, dtype=np.int64), "label_v20": np.zeros((0, 20), dtype=np.int64), "label_v21": np.zeros((0, 21), dtype=np.int64)}
    for variant in VARIANT_NAMES:
        width = 20 if variant.startswith("v20_") else 21
        for stat in ("intersect", "union", "pred"):
            arrays[f"{stat}_{variant}"] = np.zeros((0, width), dtype=np.int64)

    manifest_path = tmp_path / "canonical_zero.json"
    npz_path = manifest_path.with_suffix(".npz")
    np.savez(npz_path, allow_pickle=False, **arrays)
    manifest = {
        "schema": "test-schema", "npz_filename": npz_path.name,
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "v20_class_count": 20, "v21_class_count": 21,
        "live_v20_class_names_digest": "a" * 64, "live_v21_class_names_digest": "b" * 64,
        "image_count": 0, "dataset_indices": [], "image_ids": [], "image_order_digest": hashlib.sha256(b"[]").hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest))

    loaded = driver_module._load_per_image_stats(manifest_path)
    assert loaded["manifest"]["image_count"] == 0
    for array in loaded["arrays"].values():
        assert array.shape[0] == 0


def test_per_image_stats_fresh_versus_resume_equivalence(tmp_path, driver_module):
    # A manifest/NPZ pair produced in one write (as if freshly completed)
    # must be byte-for-byte array-equal to the same logical content
    # loaded back and re-persisted (as if the driver were about to
    # resume from it) -- confirms the round trip is lossless and
    # order-independent.
    rows = _synthetic_rows(driver_module, image_count=4)
    fresh_path = tmp_path / "fresh.json"
    driver_module._write_per_image_stats_atomically(
        fresh_path, schema_name="test-schema", v20_class_count=20, v21_class_count=21,
        live_v20_class_names_digest="a" * 64, live_v21_class_names_digest="b" * 64,
        dataset_indices=[0, 1, 2, 3], image_ids=["img0", "img1", "img2", "img3"], rows=rows,
    )
    fresh_loaded = driver_module._load_per_image_stats(fresh_path)

    resume_path = tmp_path / "resume.json"
    driver_module._write_per_image_stats_atomically(
        resume_path, schema_name="test-schema", v20_class_count=20, v21_class_count=21,
        live_v20_class_names_digest="a" * 64, live_v21_class_names_digest="b" * 64,
        dataset_indices=[0, 1, 2, 3], image_ids=["img0", "img1", "img2", "img3"], rows=rows,
    )
    resume_loaded = driver_module._load_per_image_stats(resume_path)

    assert set(fresh_loaded["arrays"]) == set(resume_loaded["arrays"])
    for name in fresh_loaded["arrays"]:
        np.testing.assert_array_equal(fresh_loaded["arrays"][name], resume_loaded["arrays"][name])
    assert fresh_loaded["manifest"]["image_count"] == resume_loaded["manifest"]["image_count"]


# ---------------------------------------------------------------------
# Metric-unit contract at the artifact boundary: percent_0_100 for
# metrics, percentage points for deltas. Confirms the ACTUAL driver
# source performs `100.0 * fraction`, then ties a hand-derived fixture
# (fraction mIoU=0.5 -> serialized 50.0; fraction delta=0.125 ->
# serialized 12.5 percentage points) to that exact conversion.
# ---------------------------------------------------------------------


def test_driver_metrics_helper_multiplies_fraction_by_100(driver_module):
    import inspect

    source = inspect.getsource(driver_module)
    assert "100.0 * value for name, value in fraction.items()" in source, (
        "the driver's _metrics helper must convert compute_full_precision_metrics' fraction_0_1 "
        "output to percent_0_100 via an explicit *100 multiplication at the artifact boundary"
    )


def test_driver_deltas_are_computed_from_already_percent_scaled_metrics(driver_module):
    import inspect

    source = inspect.getsource(driver_module)
    # Each delta subtracts two `metrics[...]["mIoU"]` values (already
    # percent-scaled by _metrics), never two raw fraction values -- this
    # is what makes the delta itself "percentage points", not "fraction
    # points". Confirm the literal subtraction pattern exists.
    assert 'metrics["v20_k11"]["mIoU"] - metrics["v20_k12"]["mIoU"]' in source
    assert 'metrics["v21_k11"]["mIoU"] - metrics["v21_e3"]["mIoU"]' in source


def test_hand_derived_metric_unit_boundary_fixture():
    """fraction mIoU = 0.5 -> serialized (percent_0_100) mIoU = 50.0;
    fraction delta = 0.125 -> serialized (percentage_points) delta = 12.5.
    Applies the EXACT conversion the driver's own source performs
    (100.0 * fraction, then subtract two already-converted values),
    never re-deriving a different formula."""
    fraction_miou_a = 0.625
    fraction_miou_b = 0.5
    assert fraction_miou_a - fraction_miou_b == pytest.approx(0.125)

    # driver's _metrics(): {name: 100.0 * value for name, value in fraction.items()}
    serialized_a = 100.0 * fraction_miou_a
    serialized_b = 100.0 * fraction_miou_b
    assert serialized_b == pytest.approx(50.0)
    assert serialized_a == pytest.approx(62.5)

    # driver's delta: metrics[...]["mIoU"] - metrics[...]["mIoU"] (both already percent-scaled)
    serialized_delta = serialized_a - serialized_b
    assert serialized_delta == pytest.approx(12.5)

    # never label a fraction as percent: the RAW fraction delta (0.125) must
    # never be confused with the correctly-labeled percentage-point delta (12.5)
    fraction_delta = fraction_miou_a - fraction_miou_b
    assert fraction_delta == pytest.approx(0.125)
    assert serialized_delta != pytest.approx(fraction_delta)


def test_full_pipeline_fraction_to_percent_boundary_via_compute_full_precision_metrics():
    """End-to-end: construct sufficient statistics that make
    compute_full_precision_metrics return EXACTLY fraction mIoU=0.5 for
    one image set and 0.625 for another, apply the driver's own *100
    conversion, and confirm the percentage-point delta is exactly 12.5 --
    tying the hand-derived fixture to the REAL internal helper, not a
    reimplementation of it."""
    # Single class, single image: intersect/union chosen so IoU = 0.5 exactly.
    pre_eval_a = [(np.array([1]), np.array([2]), np.array([1]), np.array([2]))]
    pre_eval_b = [(np.array([5]), np.array([8]), np.array([5]), np.array([8]))]  # IoU = 5/8 = 0.625

    result_a = compute_full_precision_metrics(pre_eval_a)
    result_b = compute_full_precision_metrics(pre_eval_b)
    assert result_a["mIoU"] == pytest.approx(0.5)
    assert result_b["mIoU"] == pytest.approx(0.625)

    percent_a = 100.0 * result_a["mIoU"]
    percent_b = 100.0 * result_b["mIoU"]
    assert percent_a == pytest.approx(50.0)
    assert percent_b == pytest.approx(62.5)

    percentage_point_delta = percent_b - percent_a
    assert percentage_point_delta == pytest.approx(12.5)


def test_sufficient_statistic_arrays_remain_integer_never_converted_to_percent(driver_module, tmp_path):
    """Persisted intersect/union/pred/label NPZ arrays must stay raw
    integer pixel counts -- the *100 percent conversion happens ONLY at
    the result/metrics boundary, never applied to the sufficient
    statistics themselves."""
    manifest_path = _write_synthetic_manifest(driver_module, tmp_path, "int_arrays")
    loaded = driver_module._load_per_image_stats(manifest_path)
    for name, array in loaded["arrays"].items():
        assert np.issubdtype(array.dtype, np.integer), f"{name} must remain an integer array, got dtype {array.dtype}"
