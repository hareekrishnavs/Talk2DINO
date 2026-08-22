"""Direct CPU coverage of the production manifest-building logic used by
the bounded k11/k12 finite-step stability gate.

Tests the actual ``build_bounded_manifest``/``ManifestGeometry``/
``manifest_geometry_from_e3_identity`` implementation in
``src.k11_k12_stability_manifest`` -- never a reimplementation of it.
Expected window geometry below is hand-derived from the legacy
``slide_inference`` grid formula (``grid_n = max(dim - crop + stride - 1,
0) // stride + 1``), not produced by calling ``SlidingWindowPlan`` and
comparing to itself.

Importing this module requires no mmcv, mmseg, CUDA, model construction,
or dataset initialization: ``build_bounded_manifest`` only ever imports
``segmentation.evaluation.sliding_window_geometry``, a pure-Python module,
via a safe loader that never executes the real (mmcv-eager)
``segmentation/evaluation/__init__.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.k11_k12_stability_gate_identity import K11K12StabilityGateError
from src.k11_k12_stability_manifest import (
    ManifestGeometry,
    build_bounded_manifest,
    manifest_geometry_from_e3_identity,
)

ROOT = Path(__file__).parents[1]

SMALL_GEOMETRY = ManifestGeometry(crop_height=100, crop_width=100, stride_height=50, stride_width=50)


class FakeDataset:
    def __init__(self, img_infos):
        self.img_infos = img_infos

    def __len__(self):
        return len(self.img_infos)


class FakeDatasetDataInfos:
    """Some dataset implementations expose ``data_infos`` instead of
    ``img_infos``; the manifest builder must support both."""

    def __init__(self, data_infos):
        self.data_infos = data_infos

    def __len__(self):
        return len(self.data_infos)


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
# build_bounded_manifest: hand-derived single-image fixtures
# ---------------------------------------------------------------------------


def test_image_taller_than_crop_last_row_clamped():
    # height=220, width=100 (== crop_w), crop=100, stride=50.
    # grid_rows = max(220-100+50-1, 0)//50+1 = 4; grid_cols = 1.
    dataset = FakeDataset([{"height": 220, "width": 100, "filename": "tall.jpg"}])
    manifest, digest = build_bounded_manifest(dataset, canonical_window_count=4, geometry=SMALL_GEOMETRY)
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


def test_image_wider_than_crop_last_column_clamped():
    # height=100 (== crop_h), width=220, crop=100, stride=50.
    # grid_rows = 1; grid_cols = 4.
    dataset = FakeDataset([{"height": 100, "width": 220, "filename": "wide.jpg"}])
    manifest, digest = build_bounded_manifest(dataset, canonical_window_count=4, geometry=SMALL_GEOMETRY)
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
    dataset = FakeDataset([{"height": 60, "width": 60, "filename": "small.jpg"}])
    manifest, digest = build_bounded_manifest(dataset, canonical_window_count=1, geometry=SMALL_GEOMETRY)
    assert len(manifest) == 1
    entry = manifest[0]
    assert entry["crop_origin"] == [0, 0]
    assert entry["crop_end"] == [60, 60]
    assert entry["clamped"] is False


def test_image_exactly_crop_sized_single_unclamped_window():
    dataset = FakeDataset([{"height": 100, "width": 100, "filename": "exact.jpg"}])
    manifest, digest = build_bounded_manifest(dataset, canonical_window_count=1, geometry=SMALL_GEOMETRY)
    assert len(manifest) == 1
    entry = manifest[0]
    assert entry["crop_origin"] == [0, 0]
    assert entry["crop_end"] == [100, 100]
    assert entry["clamped"] is False


def test_patch_grid_derived_from_geometry_not_hardcoded():
    dataset = FakeDataset([{"height": 100, "width": 100, "filename": "exact.jpg"}])
    manifest, _ = build_bounded_manifest(dataset, canonical_window_count=1, geometry=SMALL_GEOMETRY)
    assert manifest[0]["patch_grid"] == [100 // 14, 100 // 14]


# ---------------------------------------------------------------------------
# Multi-image enumeration, stopping behavior, ordering
# ---------------------------------------------------------------------------


def _three_image_dataset():
    return FakeDataset(
        [
            {"height": 220, "width": 100, "filename": "imgA.jpg"},  # 4 windows
            {"height": 60, "width": 60, "filename": "imgB.jpg"},  # 1 window
            {"height": 100, "width": 220, "filename": "imgC.jpg"},  # 4 windows
        ]
    )


def test_multiple_images_stopping_exactly_at_total_windows():
    dataset = _three_image_dataset()
    manifest, _ = build_bounded_manifest(dataset, canonical_window_count=9, geometry=SMALL_GEOMETRY)
    assert len(manifest) == 9
    assert [e["dataset_index"] for e in manifest] == [0, 0, 0, 0, 1, 2, 2, 2, 2]


def test_stopping_partway_through_an_image():
    dataset = _three_image_dataset()
    manifest, _ = build_bounded_manifest(dataset, canonical_window_count=6, geometry=SMALL_GEOMETRY)
    assert len(manifest) == 6
    assert [e["dataset_index"] for e in manifest] == [0, 0, 0, 0, 1, 2]
    assert manifest[-1]["window_flat_index"] == 0  # first window of the third image only


def test_row_major_ordering_within_and_across_images():
    dataset = _three_image_dataset()
    manifest, _ = build_bounded_manifest(dataset, canonical_window_count=9, geometry=SMALL_GEOMETRY)
    for dataset_index in {0, 1, 2}:
        flat_indices = [e["window_flat_index"] for e in manifest if e["dataset_index"] == dataset_index]
        assert flat_indices == sorted(flat_indices)
    dataset_indices = [e["dataset_index"] for e in manifest]
    assert dataset_indices == sorted(dataset_indices)


def test_flat_index_and_sample_order_index_continuity():
    dataset = _three_image_dataset()
    manifest, _ = build_bounded_manifest(dataset, canonical_window_count=9, geometry=SMALL_GEOMETRY)
    assert [e["sample_order_index"] for e in manifest] == list(range(9))


def test_image_ids_taken_from_filename():
    dataset = _three_image_dataset()
    manifest, _ = build_bounded_manifest(dataset, canonical_window_count=9, geometry=SMALL_GEOMETRY)
    assert {e["image_id"] for e in manifest if e["dataset_index"] == 0} == {"imgA.jpg"}
    assert {e["image_id"] for e in manifest if e["dataset_index"] == 1} == {"imgB.jpg"}


def test_dataset_with_data_infos_attribute_supported():
    dataset = FakeDatasetDataInfos([{"height": 100, "width": 100, "filename": "d.jpg"}])
    manifest, _ = build_bounded_manifest(dataset, canonical_window_count=1, geometry=SMALL_GEOMETRY)
    assert manifest[0]["image_id"] == "d.jpg"


# ---------------------------------------------------------------------------
# Digest, mutation resistance, error handling
# ---------------------------------------------------------------------------


def test_manifest_digest_deterministic_across_calls():
    dataset = _three_image_dataset()
    manifest1, digest1 = build_bounded_manifest(dataset, canonical_window_count=9, geometry=SMALL_GEOMETRY)
    manifest2, digest2 = build_bounded_manifest(dataset, canonical_window_count=9, geometry=SMALL_GEOMETRY)
    assert digest1 == digest2
    assert manifest1 == manifest2


def test_manifest_digest_changes_with_different_geometry():
    dataset = _three_image_dataset()
    _, digest_a = build_bounded_manifest(dataset, canonical_window_count=9, geometry=SMALL_GEOMETRY)
    other_geometry = ManifestGeometry(crop_height=90, crop_width=90, stride_height=45, stride_width=45)
    _, digest_b = build_bounded_manifest(dataset, canonical_window_count=1, geometry=other_geometry)
    assert digest_a != digest_b


def test_wrong_metadata_missing_height_raises():
    dataset = FakeDataset([{"width": 100, "filename": "bad.jpg"}])
    with pytest.raises(KeyError):
        build_bounded_manifest(dataset, canonical_window_count=1, geometry=SMALL_GEOMETRY)


def test_insufficient_total_windows_raises():
    dataset = FakeDataset([{"height": 100, "width": 100, "filename": "exact.jpg"}])
    with pytest.raises(K11K12StabilityGateError, match="expected exactly"):
        build_bounded_manifest(dataset, canonical_window_count=5, geometry=SMALL_GEOMETRY)


def test_bad_canonical_window_count_rejected():
    dataset = FakeDataset([{"height": 100, "width": 100, "filename": "exact.jpg"}])
    for bad_count in (0, -1, True, 1.5, "9"):
        with pytest.raises(K11K12StabilityGateError):
            build_bounded_manifest(dataset, canonical_window_count=bad_count, geometry=SMALL_GEOMETRY)


def test_bad_geometry_type_rejected():
    dataset = FakeDataset([{"height": 100, "width": 100, "filename": "exact.jpg"}])
    with pytest.raises(K11K12StabilityGateError):
        build_bounded_manifest(dataset, canonical_window_count=1, geometry={"crop_height": 100})


def test_input_img_infos_not_mutated():
    img_infos = [{"height": 220, "width": 100, "filename": "imgA.jpg"}]
    before = json.dumps(img_infos, sort_keys=True)
    dataset = FakeDataset(img_infos)
    build_bounded_manifest(dataset, canonical_window_count=4, geometry=SMALL_GEOMETRY)
    after = json.dumps(img_infos, sort_keys=True)
    assert before == after


# ---------------------------------------------------------------------------
# E3-identity-derived geometry: authority flow (Finding 3)
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
