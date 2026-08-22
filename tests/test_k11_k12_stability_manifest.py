"""Direct CPU coverage of the production manifest-building logic used by
the bounded k11/k12 finite-step stability gate.

Tests the actual ``build_bounded_manifest``/``ManifestGeometry``/
``ImageGeometryRecord``/``manifest_geometry_from_e3_identity``
implementation in ``src.k11_k12_stability_manifest`` -- never a
reimplementation of it. Expected window geometry below is hand-derived
from the legacy ``slide_inference`` grid formula (``grid_n = max(dim -
crop + stride - 1, 0) // stride + 1``), not produced by calling
``SlidingWindowPlan`` and comparing to itself.

``build_bounded_manifest`` consumes only already-verified
``ImageGeometryRecord`` values -- never a raw dataset object, and never
``dataset.img_infos``/``dataset.data_infos`` directly (which, for the
real COCOStuffDataset, never carry height/width at all: only ``filename``
and ``ann``). The adapter that derives authoritative per-image geometry
from the real, processed inference tensor lives in
``diagnostics.run_k11_k12_stability`` and is covered separately in
``tests/test_k11_k12_stability_prepared_image.py``.

Importing this module requires no mmcv, mmseg, torch, CUDA, model
construction, or dataset initialization: ``build_bounded_manifest`` only
ever imports ``segmentation.evaluation.sliding_window_geometry``, a
pure-Python module, via a safe loader that never executes the real
(mmcv-eager) ``segmentation/evaluation/__init__.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.k11_k12_stability_gate_identity import K11K12StabilityGateError
from src.k11_k12_stability_manifest import (
    ImageGeometryRecord,
    ManifestGeometry,
    build_bounded_manifest,
    manifest_geometry_from_e3_identity,
)

ROOT = Path(__file__).parents[1]

SMALL_GEOMETRY = ManifestGeometry(crop_height=100, crop_width=100, stride_height=50, stride_width=50)


def _record(dataset_index, image_id, height, width, provenance="tensor==img_shape==pad_shape"):
    return ImageGeometryRecord(
        dataset_index=dataset_index, image_id=image_id,
        inference_height=height, inference_width=width,
        source_shape_provenance=provenance,
    )


# ---------------------------------------------------------------------------
# ManifestGeometry validation
# ---------------------------------------------------------------------------


def test_manifest_geometry_accepts_positive_exact_integers():
    geometry = ManifestGeometry(crop_height=448, crop_width=448, stride_height=224, stride_width=224)
    assert geometry.crop_height == 448


@pytest.mark.parametrize("bad_value", [True, 1.0, "448", 0, -1])
def test_manifest_geometry_rejects_non_positive_exact_int(bad_value):
    with pytest.raises(K11K12StabilityGateError):
        ManifestGeometry(crop_height=bad_value, crop_width=448, stride_height=224, stride_width=224)


def test_manifest_geometry_is_frozen():
    geometry = ManifestGeometry(crop_height=100, crop_width=100, stride_height=50, stride_width=50)
    with pytest.raises(Exception):
        geometry.crop_height = 200  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ImageGeometryRecord validation
# ---------------------------------------------------------------------------


def test_image_geometry_record_accepts_valid_values():
    record = _record(0, "a.jpg", 448, 448)
    assert record.inference_height == 448
    assert record.source_shape_provenance == "tensor==img_shape==pad_shape"


@pytest.mark.parametrize("bad_index", [True, -1, 1.5, "0", None])
def test_image_geometry_record_rejects_bad_dataset_index(bad_index):
    with pytest.raises(K11K12StabilityGateError):
        ImageGeometryRecord(dataset_index=bad_index, image_id="a.jpg", inference_height=1, inference_width=1, source_shape_provenance="p")


@pytest.mark.parametrize("bad_id", ["", None, 0, True, ["a.jpg"]])
def test_image_geometry_record_rejects_bad_image_id(bad_id):
    with pytest.raises(K11K12StabilityGateError):
        ImageGeometryRecord(dataset_index=0, image_id=bad_id, inference_height=1, inference_width=1, source_shape_provenance="p")


@pytest.mark.parametrize("bad_dim", [0, -1, True, 1.5, "448", None])
def test_image_geometry_record_rejects_bad_dimensions(bad_dim):
    with pytest.raises(K11K12StabilityGateError):
        ImageGeometryRecord(dataset_index=0, image_id="a.jpg", inference_height=bad_dim, inference_width=1, source_shape_provenance="p")
    with pytest.raises(K11K12StabilityGateError):
        ImageGeometryRecord(dataset_index=0, image_id="a.jpg", inference_height=1, inference_width=bad_dim, source_shape_provenance="p")


@pytest.mark.parametrize("bad_provenance", ["", None, 0, True])
def test_image_geometry_record_rejects_bad_provenance(bad_provenance):
    with pytest.raises(K11K12StabilityGateError):
        ImageGeometryRecord(dataset_index=0, image_id="a.jpg", inference_height=1, inference_width=1, source_shape_provenance=bad_provenance)


def test_image_geometry_record_is_frozen():
    record = _record(0, "a.jpg", 448, 448)
    with pytest.raises(Exception):
        record.inference_height = 200  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_bounded_manifest: hand-derived single-image fixtures
# ---------------------------------------------------------------------------


def test_image_taller_than_crop_last_row_clamped():
    # height=220, width=100 (== crop_w), crop=100, stride=50.
    # grid_rows = max(220-100+50-1, 0)//50+1 = 4; grid_cols = 1.
    records = [_record(0, "tall.jpg", 220, 100)]
    manifest, digest = build_bounded_manifest(records, canonical_window_count=4, geometry=SMALL_GEOMETRY)
    assert len(manifest) == 4
    expected = [
        (0, 0, [0, 0], [100, 100], False),
        (0, 1, [50, 0], [150, 100], False),
        (0, 2, [100, 0], [200, 100], False),
        (0, 3, [120, 0], [220, 100], True),  # back-shifted from nominal origin 150 -> 120
    ]
    for entry, (dataset_index, flat_index, origin, end, clamped) in zip(manifest, expected):
        assert entry["dataset_index"] == dataset_index
        assert entry["window_flat_index"] == flat_index
        assert entry["crop_origin"] == origin
        assert entry["crop_end"] == end
        assert entry["clamped"] == clamped
        assert entry["source_shape_provenance"] == "tensor==img_shape==pad_shape"


def test_image_wider_than_crop_last_column_clamped():
    # height=100 (== crop_h), width=220, crop=100, stride=50.
    # grid_rows = 1; grid_cols = 4.
    records = [_record(0, "wide.jpg", 100, 220)]
    manifest, digest = build_bounded_manifest(records, canonical_window_count=4, geometry=SMALL_GEOMETRY)
    assert len(manifest) == 4
    expected = [
        (0, [0, 0], [100, 100], False),
        (1, [0, 50], [100, 150], False),
        (2, [0, 100], [100, 200], False),
        (3, [0, 120], [100, 220], True),  # back-shifted from nominal origin 150 -> 120
    ]
    for entry, (flat_index, origin, end, clamped) in zip(manifest, expected):
        assert entry["window_flat_index"] == flat_index
        assert entry["crop_origin"] == origin
        assert entry["crop_end"] == end
        assert entry["clamped"] == clamped


def test_image_smaller_than_crop_single_unclamped_window():
    # height=60, width=60, crop=100: grid = 1x1. The single window's origin
    # cannot be back-shifted below 0 (max(60-100, 0) == 0 == nominal
    # origin), so clamped_vertical/horizontal are correctly False even
    # though the window covers less than the full 100x100 crop -- this is
    # WindowGeometry's own established "origin was back-shifted" semantic,
    # not "extent is smaller than nominal crop".
    records = [_record(0, "small.jpg", 60, 60)]
    manifest, digest = build_bounded_manifest(records, canonical_window_count=1, geometry=SMALL_GEOMETRY)
    assert len(manifest) == 1
    entry = manifest[0]
    assert entry["crop_origin"] == [0, 0]
    assert entry["crop_end"] == [60, 60]
    assert entry["clamped"] is False


def test_image_exactly_crop_sized_single_unclamped_window():
    records = [_record(0, "exact.jpg", 100, 100)]
    manifest, digest = build_bounded_manifest(records, canonical_window_count=1, geometry=SMALL_GEOMETRY)
    assert len(manifest) == 1
    entry = manifest[0]
    assert entry["crop_origin"] == [0, 0]
    assert entry["crop_end"] == [100, 100]
    assert entry["clamped"] is False


def test_patch_grid_derived_from_geometry_not_hardcoded():
    records = [_record(0, "exact.jpg", 100, 100)]
    manifest, _ = build_bounded_manifest(records, canonical_window_count=1, geometry=SMALL_GEOMETRY)
    assert manifest[0]["patch_grid"] == [100 // 14, 100 // 14]


# ---------------------------------------------------------------------------
# Multi-image enumeration, stopping behavior, ordering
# ---------------------------------------------------------------------------


def _three_image_records():
    return [
        _record(0, "imgA.jpg", 220, 100),  # 4 windows
        _record(1, "imgB.jpg", 60, 60),  # 1 window
        _record(2, "imgC.jpg", 100, 220),  # 4 windows
    ]


def test_multiple_images_stopping_exactly_at_total_windows():
    manifest, _ = build_bounded_manifest(_three_image_records(), canonical_window_count=9, geometry=SMALL_GEOMETRY)
    assert len(manifest) == 9
    assert [e["dataset_index"] for e in manifest] == [0, 0, 0, 0, 1, 2, 2, 2, 2]


def test_stopping_partway_through_an_image():
    manifest, _ = build_bounded_manifest(_three_image_records(), canonical_window_count=6, geometry=SMALL_GEOMETRY)
    assert len(manifest) == 6
    assert [e["dataset_index"] for e in manifest] == [0, 0, 0, 0, 1, 2]
    assert manifest[-1]["window_flat_index"] == 0  # first window of the third image only


def test_row_major_ordering_within_and_across_images():
    manifest, _ = build_bounded_manifest(_three_image_records(), canonical_window_count=9, geometry=SMALL_GEOMETRY)
    for dataset_index in {0, 1, 2}:
        flat_indices = [e["window_flat_index"] for e in manifest if e["dataset_index"] == dataset_index]
        assert flat_indices == sorted(flat_indices)
    dataset_indices = [e["dataset_index"] for e in manifest]
    assert dataset_indices == sorted(dataset_indices)


def test_flat_index_and_sample_order_index_continuity():
    manifest, _ = build_bounded_manifest(_three_image_records(), canonical_window_count=9, geometry=SMALL_GEOMETRY)
    assert [e["sample_order_index"] for e in manifest] == list(range(9))


def test_image_ids_taken_from_records():
    manifest, _ = build_bounded_manifest(_three_image_records(), canonical_window_count=9, geometry=SMALL_GEOMETRY)
    assert {e["image_id"] for e in manifest if e["dataset_index"] == 0} == {"imgA.jpg"}
    assert {e["image_id"] for e in manifest if e["dataset_index"] == 1} == {"imgB.jpg"}


def test_generator_input_accepted_and_only_pulled_as_needed():
    pulled = []

    def _gen():
        for record in _three_image_records():
            pulled.append(record.dataset_index)
            yield record

    manifest, _ = build_bounded_manifest(_gen(), canonical_window_count=6, geometry=SMALL_GEOMETRY)
    assert len(manifest) == 6
    # only images 0, 1, 2 were needed to reach 6 windows (4+1+1) -- the
    # generator must never be pulled beyond that, proving no image beyond
    # what is needed is ever processed
    assert pulled == [0, 1, 2]


# ---------------------------------------------------------------------------
# Digest, mutation resistance, error handling
# ---------------------------------------------------------------------------


def test_manifest_digest_deterministic_across_calls():
    manifest1, digest1 = build_bounded_manifest(_three_image_records(), canonical_window_count=9, geometry=SMALL_GEOMETRY)
    manifest2, digest2 = build_bounded_manifest(_three_image_records(), canonical_window_count=9, geometry=SMALL_GEOMETRY)
    assert digest1 == digest2
    assert manifest1 == manifest2


def test_manifest_digest_changes_with_different_geometry():
    _, digest_a = build_bounded_manifest(_three_image_records(), canonical_window_count=9, geometry=SMALL_GEOMETRY)
    other_geometry = ManifestGeometry(crop_height=90, crop_width=90, stride_height=45, stride_width=45)
    _, digest_b = build_bounded_manifest(_three_image_records(), canonical_window_count=1, geometry=other_geometry)
    assert digest_a != digest_b


def test_non_image_geometry_record_input_rejected():
    with pytest.raises(K11K12StabilityGateError):
        build_bounded_manifest([{"dataset_index": 0, "image_id": "a", "inference_height": 1, "inference_width": 1}], canonical_window_count=1, geometry=SMALL_GEOMETRY)


def test_out_of_order_dataset_index_rejected():
    # canonical_window_count=2 (not 1) is required so the second record is
    # actually consumed before the target is reached -- with target=1 the
    # function would stop after the first (single-window) image and never
    # even look at the second, structurally unable to detect the ordering
    # violation.
    records = [_record(1, "b.jpg", 100, 100), _record(0, "a.jpg", 100, 100)]
    with pytest.raises(K11K12StabilityGateError, match="increasing"):
        build_bounded_manifest(records, canonical_window_count=2, geometry=SMALL_GEOMETRY)


def test_duplicate_dataset_index_rejected():
    records = [_record(0, "a.jpg", 100, 100), _record(0, "a2.jpg", 100, 100)]
    with pytest.raises(K11K12StabilityGateError, match="increasing"):
        build_bounded_manifest(records, canonical_window_count=2, geometry=SMALL_GEOMETRY)


def test_insufficient_total_windows_raises():
    records = [_record(0, "exact.jpg", 100, 100)]
    with pytest.raises(K11K12StabilityGateError, match="expected exactly"):
        build_bounded_manifest(records, canonical_window_count=5, geometry=SMALL_GEOMETRY)


def test_bad_canonical_window_count_rejected():
    records = [_record(0, "exact.jpg", 100, 100)]
    for bad_count in (0, -1, True, 1.5, "9"):
        with pytest.raises(K11K12StabilityGateError):
            build_bounded_manifest(records, canonical_window_count=bad_count, geometry=SMALL_GEOMETRY)


def test_bad_geometry_type_rejected():
    records = [_record(0, "exact.jpg", 100, 100)]
    with pytest.raises(K11K12StabilityGateError):
        build_bounded_manifest(records, canonical_window_count=1, geometry={"crop_height": 100})


def test_input_records_list_not_mutated():
    records = _three_image_records()
    before = list(records)
    build_bounded_manifest(records, canonical_window_count=9, geometry=SMALL_GEOMETRY)
    assert records == before


def test_no_executable_access_to_img_infos_height_or_width():
    # Regression test for the real-dataset defect: build_bounded_manifest
    # must never assume/access img_infos["height"]/["width"] -- it no
    # longer even receives a dataset object at all.
    import ast
    import inspect

    from src import k11_k12_stability_manifest

    source = inspect.getsource(k11_k12_stability_manifest.build_bounded_manifest)
    assert "img_infos" not in source
    assert "data_infos" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value in ("height", "width"):
            raise AssertionError("build_bounded_manifest must not subscript ['height']/['width'] from raw metadata")


# ---------------------------------------------------------------------------
# E3-identity-derived geometry: authority flow
# ---------------------------------------------------------------------------


def test_manifest_geometry_from_e3_identity_reads_only_crop_and_stride():
    e3_identity = {"evaluation": {"crop": [448, 448], "stride": [224, 224]}}
    geometry = manifest_geometry_from_e3_identity(e3_identity)
    assert geometry == ManifestGeometry(crop_height=448, crop_width=448, stride_height=224, stride_width=224)


def test_manifest_geometry_from_e3_identity_does_not_mutate_input():
    e3_identity = {"evaluation": {"crop": [448, 448], "stride": [224, 224]}}
    before = json.dumps(e3_identity, sort_keys=True)
    manifest_geometry_from_e3_identity(e3_identity)
    after = json.dumps(e3_identity, sort_keys=True)
    assert before == after


def test_manifest_geometry_from_e3_identity_changes_with_synthetic_e3_geometry():
    default_geometry = manifest_geometry_from_e3_identity({"evaluation": {"crop": [448, 448], "stride": [224, 224]}})
    custom_geometry = manifest_geometry_from_e3_identity({"evaluation": {"crop": [320, 320], "stride": [160, 160]}})
    assert default_geometry != custom_geometry
    assert custom_geometry.crop_height == 320


def test_canonical_real_e3_identity_produces_canonical_geometry():
    from src.e3_evaluation_identity import load_identity as load_e3_identity
    from src.matched_k11_k12_identity import load_identity as load_matched_identity
    from src.k11_k12_stability_gate_identity import load_identity as load_gate_identity

    gate_identity = load_gate_identity(repo_root=ROOT)
    matched_identity = load_matched_identity(
        ROOT / gate_identity["parent_identity"]["matched_identity_path"], repo_root=ROOT
    )
    e3_identity = load_e3_identity(
        ROOT / matched_identity["parent_identities"]["e3_identity_path"], repo_root=ROOT
    )
    geometry = manifest_geometry_from_e3_identity(e3_identity)
    assert (geometry.crop_height, geometry.crop_width) == tuple(e3_identity["evaluation"]["crop"])
    assert (geometry.stride_height, geometry.stride_width) == tuple(e3_identity["evaluation"]["stride"])


def test_no_hardcoded_canonical_crop_stride_literals_in_manifest_module():
    import ast
    import inspect

    from src import k11_k12_stability_manifest

    source = inspect.getsource(k11_k12_stability_manifest)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in (448, 224):
            raise AssertionError(
                f"src/k11_k12_stability_manifest.py contains a bare {node.value!r} literal "
                f"at line {node.lineno} -- crop/stride must come from the caller's geometry, never a hardcoded value"
            )
