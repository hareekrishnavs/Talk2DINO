"""Tests for src/dfp_phase1a_dataset_view.py: the runtime adapter that
maps the frozen 500-image DFP Phase 1A manifest order onto the native
mmseg dataset's own (different) index order.

Covers: a synthetic 6-image fixture whose native order deliberately
differs from the selected order (the exact shape of SLURM job 21055181's
real failure), every adversarial malformed/unsafe/ambiguous input listed
in the repair task, a direct reproduction of the real mismatch without
GPU, and (when the real COCO-Stuff dataset root is available) a genuine
read-only mapping check against it -- metadata only, no model weights,
no inference.

CPU-only throughout. No CUDA, no model construction, no image pixel
data touched (metadata-only)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import src.dfp_phase1a_dataset_view as dv

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Minimal stand-in exposing exactly the mmseg CustomDataset surface
    this module reads: img_infos (list of {"filename": ...}), img_suffix,
    img_dir. Never touches real pixel data."""

    def __init__(self, native_ids, *, img_suffix=".jpg", img_dir="images", data_infos_only=False):
        infos = [{"filename": f"{i}{img_suffix}"} for i in native_ids]
        if data_infos_only:
            self.data_infos = infos
        else:
            self.img_infos = infos
        self.img_suffix = img_suffix
        self.img_dir = img_dir


def _six_image_fixture():
    native_order = ["000000000139", "000000000285", "000000000724", "000000000800", "000000000901", "000000000999"]
    selected_order = ["000000000724", "000000000139", "000000000901"]
    return _FakeDataset(native_order), selected_order


def test_task_example_geometry():
    dataset, selected = _six_image_fixture()
    view = dv.build_manifest_execution_view(dataset, selected)
    assert len(view) == 3
    assert view[0].canonical_id == "000000000724" and view[0].native_index == 2
    assert view[1].canonical_id == "000000000139" and view[1].native_index == 0
    assert view[2].canonical_id == "000000000901" and view[2].native_index == 4
    # checkpoint/row order remains exactly the selected (manifest) order
    assert [p.canonical_id for p in view.positions] == selected


def test_gt_lookup_uses_native_indices_not_positions():
    dataset, selected = _six_image_fixture()
    view = dv.build_manifest_execution_view(dataset, selected)
    native_indices_for_gt = [view[i].native_index for i in range(len(view))]
    assert native_indices_for_gt == [2, 0, 4]


def test_data_infos_fallback_when_img_infos_absent():
    dataset = _FakeDataset(["000000000139", "000000000724"], data_infos_only=True)
    view = dv.build_manifest_execution_view(dataset, ["000000000724"])
    assert view[0].native_index == 1


def test_mapping_digest_deterministic_across_pythonhashseed():
    import subprocess
    import sys

    script = (
        "import sys; sys.path.insert(0, 'tests'); "
        "from test_dfp_phase1a_dataset_view import _six_image_fixture; "
        "import src.dfp_phase1a_dataset_view as dv; "
        "dataset, selected = _six_image_fixture(); "
        "print(dv.build_manifest_execution_view(dataset, selected).mapping_digest)"
    )
    digests = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run([sys.executable, "-c", script], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        digests.add(result.stdout.strip())
    assert len(digests) == 1, f"digest varied across PYTHONHASHSEED: {digests}"


def test_repeated_construction_same_digest_in_process():
    dataset, selected = _six_image_fixture()
    d1 = dv.build_manifest_execution_view(dataset, selected).mapping_digest
    d2 = dv.build_manifest_execution_view(dataset, selected).mapping_digest
    assert d1 == d2


def test_native_index_map_never_requires_native_order_equal_selected_order():
    dataset, selected = _six_image_fixture()
    native_map = dv.build_native_image_index(dataset)
    assert len(native_map) == 6
    assert set(native_map) == {"000000000139", "000000000285", "000000000724", "000000000800", "000000000901", "000000000999"}


# ---------------------------------------------------------------------------
# Adversarial fixtures
# ---------------------------------------------------------------------------


def test_missing_selected_id_rejected():
    dataset, _ = _six_image_fixture()
    with pytest.raises(dv.DfpPhase1aDatasetViewError, match="no corresponding entry"):
        dv.build_manifest_execution_view(dataset, ["000000000724", "000000000001"])


def test_duplicate_native_id_rejected():
    dataset = _FakeDataset(["000000000139", "000000000139"])
    with pytest.raises(dv.DfpPhase1aDatasetViewError, match="duplicate canonical ID"):
        dv.build_native_image_index(dataset)


def test_duplicate_selected_id_rejected():
    dataset, _ = _six_image_fixture()
    with pytest.raises(dv.DfpPhase1aDatasetViewError, match="duplicate of an earlier manifest position"):
        dv.build_manifest_execution_view(dataset, ["000000000724", "000000000724"])


def test_wrong_image_suffix_rejected():
    dataset = _FakeDataset(["000000000139"], img_suffix=".jpg")
    dataset.img_infos = [{"filename": "000000000139.png"}]  # disagrees with dataset.img_suffix
    with pytest.raises(dv.DfpPhase1aDatasetViewError, match="does not end with"):
        dv.build_native_image_index(dataset)


def test_path_traversal_rejected():
    dataset = _FakeDataset(["dummy"])
    dataset.img_infos = [{"filename": "../../etc/passwd.jpg"}]
    with pytest.raises(Exception):  # K11K12StabilityGateError, propagated from reconcile_canonical_image_id
        dv.build_native_image_index(dataset)


def test_absolute_outside_root_path_rejected():
    dataset = _FakeDataset(["dummy"])
    dataset.img_infos = [{"filename": "/etc/passwd.jpg"}]
    with pytest.raises(Exception):
        dv.build_native_image_index(dataset)


def test_symlink_escape_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "000000000001.jpg").write_bytes(b"x")
    link = root / "000000000001.jpg"
    try:
        link.symlink_to(outside / "000000000001.jpg")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported in this environment")
    dataset = _FakeDataset(["000000000001"], img_dir=str(root))
    # Path.resolve() follows symlinks to their real target, so a symlink
    # planted inside image_root but pointing OUTSIDE it is correctly
    # rejected by the shared reconcile_canonical_image_id authority
    # (image_root/canonical_id no longer resolves to a location under
    # image_root once the symlink is followed).
    with pytest.raises(Exception):
        dv.build_manifest_execution_view(dataset, ["000000000001"])


def test_pipeline_filename_mismatch_at_runtime_reverification():
    dataset, selected = _six_image_fixture()
    view = dv.build_manifest_execution_view(dataset, selected)
    # simulate a mutated dataset (native index 2's filename changed) between
    # view construction and per-image re-verification
    dataset.img_infos[2]["filename"] = "000000000042.jpg"
    with pytest.raises(dv.DfpPhase1aDatasetViewError, match="dataset ordering changed"):
        dv.verify_position_identity(
            dataset, manifest_position=0, canonical_id=view[0].canonical_id, native_index=view[0].native_index,
        )


def test_image_correct_but_gt_from_wrong_native_index_is_the_exact_bug_this_module_fixes():
    """Demonstrates the OLD bug directly: using manifest position as the
    native index (rather than the mapped one) fetches the wrong image
    entirely for position 0."""
    dataset, selected = _six_image_fixture()
    view = dv.build_manifest_execution_view(dataset, selected)
    old_buggy_native_index = 0  # manifest position used directly (the bug)
    correct_native_index = view[0].native_index  # 2
    assert old_buggy_native_index != correct_native_index
    # the OLD behavior would silently fetch dataset.img_infos[0] (id 139)
    # while claiming to process selected_ids[0] (id 724) -- exactly what
    # verify_position_identity now catches:
    with pytest.raises(dv.DfpPhase1aDatasetViewError):
        dv.verify_position_identity(
            dataset, manifest_position=0, canonical_id=view[0].canonical_id, native_index=old_buggy_native_index,
        )
    # the mapped index is silently correct and passes:
    dv.verify_position_identity(
        dataset, manifest_position=0, canonical_id=view[0].canonical_id, native_index=correct_native_index,
    )


def test_metadata_list_shorter_than_dataset_length():
    dataset, selected = _six_image_fixture()
    view = dv.build_manifest_execution_view(dataset, selected)
    dataset.img_infos = dataset.img_infos[:1]  # truncate after construction
    with pytest.raises(dv.DfpPhase1aDatasetViewError, match="no longer"):
        dv.verify_position_identity(
            dataset, manifest_position=0, canonical_id=view[0].canonical_id, native_index=view[0].native_index,
        )


def test_unexpected_metadata_container_rejected():
    class NoMetadataDataset:
        img_suffix = ".jpg"
        img_dir = "images"

    with pytest.raises(dv.DfpPhase1aDatasetViewError, match="neither 'img_infos' nor 'data_infos'"):
        dv.build_native_image_index(NoMetadataDataset())


def test_non_string_filename_rejected():
    dataset = _FakeDataset(["000000000139"])
    dataset.img_infos = [{"filename": 12345}]
    with pytest.raises(dv.DfpPhase1aDatasetViewError, match="must be a non-empty exact str"):
        dv.build_native_image_index(dataset)


@pytest.mark.parametrize("bad_id", ["12", "0000000007240", "72400000000a", "", "000000000724 "])
def test_id_with_wrong_format_rejected(bad_id):
    dataset = _FakeDataset([bad_id])
    with pytest.raises(dv.DfpPhase1aDatasetViewError, match="frozen COCO identifier format"):
        dv.build_native_image_index(dataset)


def test_missing_img_suffix_attribute_rejected():
    dataset, _ = _six_image_fixture()
    del dataset.img_suffix
    with pytest.raises(dv.DfpPhase1aDatasetViewError, match="img_suffix"):
        dv.build_native_image_index(dataset)


def test_ambiguous_metadata_field_missing_filename_rejected():
    dataset = _FakeDataset(["000000000139"])
    dataset.img_infos = [{"not_filename": "x"}]
    with pytest.raises(dv.DfpPhase1aDatasetViewError, match="missing a 'filename'"):
        dv.build_native_image_index(dataset)


def test_selected_ids_not_a_list_or_tuple_rejected():
    dataset, selected = _six_image_fixture()
    with pytest.raises(dv.DfpPhase1aDatasetViewError, match="must be a list or tuple"):
        dv.build_manifest_execution_view(dataset, set(selected))


def test_does_not_mutate_selected_ids_or_dataset():
    dataset, selected = _six_image_fixture()
    selected_before = list(selected)
    infos_before = [dict(x) for x in dataset.img_infos]
    dv.build_manifest_execution_view(dataset, selected)
    assert selected == selected_before
    assert dataset.img_infos == infos_before


# ---------------------------------------------------------------------------
# Direct reproduction of the real job-21055181 mismatch, no GPU
# ---------------------------------------------------------------------------


def test_reproduce_real_mismatch_without_gpu():
    """selected_ids[0] = 000000000724, native index 0 = 000000000139 --
    exactly SLURM job 21055181's observed shape. Proves the OLD
    positional behavior fails, and the NEW mapping resolves correctly."""
    native_order = ["000000000139", "000000000724"]  # native index 0 is 139, matching the real dataset's own alphabetical-first file
    selected = ["000000000724"]
    dataset = _FakeDataset(native_order, img_dir="./data/coco_stuff164k/images/val2017")

    # OLD (buggy) behavior: manifest position 0 used directly as native index.
    old_native_index = 0
    with pytest.raises(dv.DfpPhase1aDatasetViewError):
        dv.verify_position_identity(dataset, manifest_position=0, canonical_id=selected[0], native_index=old_native_index)

    # NEW mapping:
    view = dv.build_manifest_execution_view(dataset, selected)
    assert view[0].canonical_id == "000000000724"
    assert view[0].native_index == 1
    assert view[0].relative_path == "000000000724.jpg"
    # image_root ("./data/coco_stuff164k/images/val2017") is itself a
    # symlink to the real scratch location in this environment;
    # Path.resolve() follows it, so only the filename suffix (not the
    # exact prefix) is asserted here.
    assert view[0].resolved_path.endswith("000000000724.jpg")
    dv.verify_position_identity(dataset, manifest_position=0, canonical_id=view[0].canonical_id, native_index=view[0].native_index)


# ---------------------------------------------------------------------------
# Real dataset, metadata-only, read-only (skipped if unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("COCO_STUFF_REAL_DATA_ROOT"), reason="requires COCO_STUFF_REAL_DATA_ROOT")
def test_real_dataset_metadata_only_mapping(monkeypatch):
    """Build ONLY the real mmseg dataset object (no model, no CUDA, no
    checkpoint load) and validate the complete 500-image mapping
    read-only."""
    import json

    from diagnostics.run_k11_k12_stability import _build_dataset_only
    from src.dfp_phase1a_identity import load_identity as load_protocol_identity
    from src.dfp_phase1a_execute import _load_e3_identity

    repo_root = ROOT
    protocol_identity = load_protocol_identity(repo_root=repo_root)
    e3_identity = _load_e3_identity(repo_root, protocol_identity)
    _, dataset, _ = _build_dataset_only(repo_root, e3_identity)

    manifest = json.loads((repo_root / "evaluation_subsets/dfp_phase1a_coco_stuff_500.json").read_text())
    selected_ids = manifest["selected_ids"]

    assert len(dataset) == 5000
    native_map = dv.build_native_image_index(dataset)
    assert len(native_map) == 5000

    view = dv.build_manifest_execution_view(dataset, selected_ids)
    assert len(view) == 500
    assert view[0].canonical_id == "000000000724"
    assert view[0].relative_path == "000000000724.jpg"
    print("mapping_digest:", view.mapping_digest)
    print("first 10:", [(p.canonical_id, p.native_index) for p in view.positions[:10]])
    print("last 10:", [(p.canonical_id, p.native_index) for p in view.positions[-10:]])
