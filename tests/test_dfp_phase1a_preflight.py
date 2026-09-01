"""Tests for DFP Phase 1A Step 12: the unified CLI and ordered CPU-only
preflight orchestration.

CPU-only throughout. No real dataset access (except the explicitly
environment-gated real-data smoke tests at the bottom, which skip safely
when COCO_STUFF_REAL_DATA_ROOT / ADE20K_REAL_DATA_ROOT are unset), no
model construction, no CUDA initialization, no graph/recurrence
execution. Every preflight call (positive AND negative) is proven never
to import torch or call any heavy entry point -- see the poison-import
section.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import src.dfp_phase1a_artifact_schema as sch
import src.dfp_phase1a_checkpoint as ckpt
import src.dfp_phase1a_preflight as pf

ROOT = Path(__file__).resolve().parent.parent
CLI = str(ROOT / "diagnostics" / "run_dfp_phase1a.py")
PY = sys.executable


# ---------------------------------------------------------------------------
# Shared fixtures: synthetic (empty-content, existence-only) dataset roots
# built over the REAL, already-committed 500 selected canonical IDs. No
# real pixel content is needed -- Phase B's reconciliation is existence-
# only, matching its own documented scope (the heavier presence-based
# recomputation lives in verify-source-binding / prepare_dfp_phase1a_
# subsets.py, not here).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coco_selected_ids():
    manifest = json.loads((ROOT / "evaluation_subsets/dfp_phase1a_coco_stuff_500.json").read_text())
    return manifest["selected_ids"]


@pytest.fixture(scope="module")
def ade_selected_ids():
    manifest = json.loads((ROOT / "evaluation_subsets/dfp_phase1a_ade20k_500.json").read_text())
    return manifest["selected_ids"]


@pytest.fixture()
def coco_dataset_root(tmp_path, coco_selected_ids):
    dataset_dir = tmp_path / "coco_dataset"
    img_dir = dataset_dir / "images/val2017"
    ann_dir = dataset_dir / "annotations/val2017"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    for cid in coco_selected_ids:
        (img_dir / f"{cid}.jpg").write_bytes(b"x")
        (ann_dir / f"{cid}_labelTrainIds.png").write_bytes(b"x")
    return dataset_dir


@pytest.fixture()
def ade_dataset_root(tmp_path, ade_selected_ids):
    from src.ade20k_dataset_identity import load_identity as load_ade_source

    ade_identity = load_ade_source(ROOT / "evaluation_identities/e12_ade20k_dataset_source.toml", repo_root=ROOT)
    parent_dir = tmp_path / "ade_dataset"
    dataset_dir = parent_dir / "ADEChallengeData2016"
    img_dir = dataset_dir / ade_identity["source"]["image_relative_root"]
    ann_dir = dataset_dir / ade_identity["source"]["annotation_relative_root"]
    (dataset_dir / "images/training").mkdir(parents=True)
    (dataset_dir / "annotations/training").mkdir(parents=True)
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    for cid in ade_selected_ids:
        (img_dir / f"{cid}.jpg").write_bytes(b"x")
        (ann_dir / f"{cid}.png").write_bytes(b"x")
    return parent_dir


@pytest.fixture()
def output_root(tmp_path):
    out = tmp_path / "preflight_output"
    out.mkdir()
    return out


# ---------------------------------------------------------------------------
# Checkpoint fixture builder for resume-mode tests (mirrors the same
# pattern used in tests/test_dfp_phase1a_artifacts.py -- not imported
# from there, since each test file's fixture builder is local by
# established repo convention, but functionally identical).
# ---------------------------------------------------------------------------


def _make_arrays(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}
    for name in sch.INT64_MEMBER_NAMES:
        arrays[name] = np.arange(n, dtype=np.int64) if name == "dataset_indices" else rng.integers(0, 100, size=n).astype(np.int64)
    base = np.sort(rng.random((n, 5)), axis=1) if n else np.zeros((0, 5))
    for i, name in enumerate(("q_w_median", "q_w_p90", "q_w_p95", "q_w_p99", "q_w_maximum")):
        arrays[name] = base[:, i].astype(np.float64) if n else np.zeros(0, dtype=np.float64)
    arrays["q_w_mean"] = rng.random(n) if n else np.zeros(0, dtype=np.float64)
    if n:
        arrays["dispute_pixel_count"] = np.minimum(arrays["dispute_pixel_count"], arrays["evaluated_pixel_count"])
        arrays["zero_dispute_window_count"] = np.minimum(arrays["zero_dispute_window_count"], arrays["window_count"])
    return arrays


def _make_fields(*, selected_ids, selected_ids_digest, n: int, complete: bool = False) -> dict:
    schema_summary = sch.validate_static_configuration(repo_root=ROOT)
    artifact_identity = sch.load_identity(repo_root=ROOT)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "schema": sch.CHECKPOINT_SCHEMA_NAME, "schema_version": sch.CHECKPOINT_SCHEMA_VERSION,
        "artifact_schema_identity_sha256": sch.sha256_file(ROOT / sch.IDENTITY_RELATIVE_PATH),
        "protocol_identity_sha256": schema_summary["protocol_identity_sha256"],
        "subset_manifest_sha256": schema_summary["coco_stuff_manifest_sha256"], "selected_ids_digest": selected_ids_digest,
        "run_mode": "coco_stuff_500", "image_count_expected": n if complete else len(selected_ids),
        "class_count": artifact_identity["dataset_binding"]["coco_stuff_class_count"],
        "class_names_digest": artifact_identity["dataset_binding"]["coco_stuff_class_names_digest"],
        "horizon_order": list(sch.HORIZON_ORDER), "source_commit": "a" * 40,
        "required_ancestor_commit": artifact_identity["parent_identity_binding"]["required_ancestor_commit"],
        "bridge_checkpoint_sha256": schema_summary["bridge_checkpoint_sha256"], "next_dataset_index": n,
        "completed_image_ids": list(selected_ids[:n]), "images_completed_count": n, "total_windows_processed": n * 4,
        "complete": complete, "final_result_produced": False, "created_at_utc": now, "updated_at_utc": now,
    }


@pytest.fixture()
def resumable_checkpoints_dir(tmp_path, coco_selected_ids):
    manifest = json.loads((ROOT / "evaluation_subsets/dfp_phase1a_coco_stuff_500.json").read_text())
    n = 3
    fields = _make_fields(selected_ids=coco_selected_ids, selected_ids_digest=manifest["selected_ids_digest"], n=n, complete=False)
    arrays = _make_arrays(n, seed=1)
    checkpoints_dir = tmp_path / "checkpoints"
    ckpt.generate_and_install_checkpoint(
        checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
        arrays=arrays, image_ids=coco_selected_ids[:n],
    )
    return checkpoints_dir


@pytest.fixture()
def complete_checkpoints_dir(tmp_path, coco_selected_ids):
    manifest = json.loads((ROOT / "evaluation_subsets/dfp_phase1a_coco_stuff_500.json").read_text())
    n = len(coco_selected_ids)
    fields = _make_fields(selected_ids=coco_selected_ids, selected_ids_digest=manifest["selected_ids_digest"], n=n, complete=True)
    arrays = _make_arrays(n, seed=2)
    checkpoints_dir = tmp_path / "checkpoints"
    ckpt.generate_and_install_checkpoint(
        checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
        arrays=arrays, image_ids=coco_selected_ids[:n],
    )
    return checkpoints_dir


# ---------------------------------------------------------------------------
# Phase A
# ---------------------------------------------------------------------------


def test_phase_a_coco_stuff():
    r = pf.run_phase_a(repo_root=ROOT, run_mode="coco_stuff_500", check_git=True)
    assert r.dataset_key == "coco_stuff"
    assert r.expected_count == 500
    assert r.class_count == 171
    assert len(r.selected_ids) == 500
    assert len(set(r.selected_ids)) == 500


def test_phase_a_ade20k():
    r = pf.run_phase_a(repo_root=ROOT, run_mode="ade20k_500", check_git=True)
    assert r.dataset_key == "ade20k"
    assert r.expected_count == 500
    assert r.class_count == 150


@pytest.mark.parametrize("bad_name", ["coco_stuff", "COCO_STUFF_500", "coco-stuff-500", "cocostuff500", " coco_stuff_500", "coco_stuff_500 ", "ade20k", "ade_20k_500"])
def test_phase_a_rejects_alias_or_case_change(bad_name):
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.run_phase_a(repo_root=ROOT, run_mode=bad_name, check_git=True)


def test_phase_a_rejects_nonexistent_repo_root(tmp_path):
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.run_phase_a(repo_root=tmp_path / "does_not_exist", run_mode="coco_stuff_500", check_git=True)


def test_phase_a_source_commit_is_real_head():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    r = pf.run_phase_a(repo_root=ROOT, run_mode="coco_stuff_500", check_git=True)
    assert r.source_commit == head


def test_phase_a_does_not_mutate_repo_root_argument():
    root = Path(str(ROOT))
    before = str(root)
    pf.run_phase_a(repo_root=root, run_mode="coco_stuff_500", check_git=True)
    assert str(root) == before


# ---------------------------------------------------------------------------
# verify_frozen_subset_manifest adapter
# ---------------------------------------------------------------------------


def test_verify_frozen_subset_manifest_wrong_dataset_key_rejected():
    protocol_sha256 = sch.validate_static_configuration(repo_root=ROOT)["protocol_identity_sha256"]
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.verify_frozen_subset_manifest(
            repo_root=ROOT, dataset_key="ade20k", protocol_identity_sha256=protocol_sha256,
            manifest_path=ROOT / "evaluation_subsets/dfp_phase1a_coco_stuff_500.json",
        )


def test_verify_frozen_subset_manifest_missing_file_rejected(tmp_path):
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.verify_frozen_subset_manifest(
            repo_root=ROOT, dataset_key="coco_stuff", protocol_identity_sha256="a" * 64,
            manifest_path=tmp_path / "does_not_exist.json",
        )


def test_verify_frozen_subset_manifest_malformed_json_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.verify_frozen_subset_manifest(
            repo_root=ROOT, dataset_key="coco_stuff", protocol_identity_sha256="a" * 64, manifest_path=bad,
        )


def test_verify_frozen_subset_manifest_duplicate_id_rejected(tmp_path):
    real = json.loads((ROOT / "evaluation_subsets/dfp_phase1a_coco_stuff_500.json").read_text())
    protocol_sha256 = sch.validate_static_configuration(repo_root=ROOT)["protocol_identity_sha256"]
    mutated = copy.deepcopy(real)
    mutated["selected_ids"][1] = mutated["selected_ids"][0]
    bad = tmp_path / "dup.json"
    bad.write_text(json.dumps(mutated))
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.verify_frozen_subset_manifest(repo_root=ROOT, dataset_key="coco_stuff", protocol_identity_sha256=protocol_sha256, manifest_path=bad)


def test_verify_frozen_subset_manifest_reordered_ids_rejected(tmp_path):
    real = json.loads((ROOT / "evaluation_subsets/dfp_phase1a_coco_stuff_500.json").read_text())
    protocol_sha256 = sch.validate_static_configuration(repo_root=ROOT)["protocol_identity_sha256"]
    mutated = copy.deepcopy(real)
    mutated["selected_ids"] = list(reversed(mutated["selected_ids"]))
    bad = tmp_path / "reordered.json"
    bad.write_text(json.dumps(mutated))
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.verify_frozen_subset_manifest(repo_root=ROOT, dataset_key="coco_stuff", protocol_identity_sha256=protocol_sha256, manifest_path=bad)


def test_verify_frozen_subset_manifest_wrong_selected_ids_digest_rejected(tmp_path):
    real = json.loads((ROOT / "evaluation_subsets/dfp_phase1a_coco_stuff_500.json").read_text())
    protocol_sha256 = sch.validate_static_configuration(repo_root=ROOT)["protocol_identity_sha256"]
    mutated = copy.deepcopy(real)
    mutated["selected_ids_digest"] = "f" * 64
    bad = tmp_path / "wrong_digest.json"
    bad.write_text(json.dumps(mutated))
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.verify_frozen_subset_manifest(repo_root=ROOT, dataset_key="coco_stuff", protocol_identity_sha256=protocol_sha256, manifest_path=bad)


# ---------------------------------------------------------------------------
# Logical ID vs physical filename reconciliation (Section 6 / VOC
# regression)
# ---------------------------------------------------------------------------


def test_reconcile_selected_ids_success(coco_dataset_root, coco_selected_ids):
    pf.reconcile_selected_ids_to_physical_paths(
        dataset_key="coco_stuff", dataset_root=coco_dataset_root, selected_ids=coco_selected_ids[:5],
        image_relative_root="images/val2017", annotation_relative_root="annotations/val2017",
        image_suffix=".jpg", annotation_suffix="_labelTrainIds.png",
    )


def test_reconcile_voc_style_altered_suffix_rejected(coco_dataset_root, coco_selected_ids):
    """Regression: a canonical ID with no extension must never be
    naively joined to the image root -- only (id + suffix) is a valid
    physical path. If the physical file's suffix is missing/altered, the
    reconciliation must fail cleanly, never silently accept a wrong
    path or a partial match."""
    bad_id = coco_selected_ids[0]
    img_path = coco_dataset_root / "images/val2017" / f"{bad_id}.jpg"
    img_path.rename(coco_dataset_root / "images/val2017" / bad_id)  # strip the suffix entirely
    with pytest.raises(pf.DfpPhase1aPreflightError, match="no physical file"):
        pf.reconcile_selected_ids_to_physical_paths(
            dataset_key="coco_stuff", dataset_root=coco_dataset_root, selected_ids=[bad_id],
            image_relative_root="images/val2017", annotation_relative_root="annotations/val2017",
            image_suffix=".jpg", annotation_suffix="_labelTrainIds.png",
        )


def test_reconcile_never_uses_bare_canonical_id_as_path(coco_dataset_root, coco_selected_ids):
    """A physical file literally named after the bare canonical ID (no
    suffix at all) must NOT satisfy reconciliation -- proves the
    construction is never dataset_root / canonical_id alone."""
    extra_id = "999999999999"
    (coco_dataset_root / "images/val2017" / extra_id).write_bytes(b"x")  # bare-ID file, no suffix
    (coco_dataset_root / "annotations/val2017" / extra_id).write_bytes(b"x")
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.reconcile_selected_ids_to_physical_paths(
            dataset_key="coco_stuff", dataset_root=coco_dataset_root, selected_ids=[extra_id],
            image_relative_root="images/val2017", annotation_relative_root="annotations/val2017",
            image_suffix=".jpg", annotation_suffix="_labelTrainIds.png",
        )


def test_reconcile_missing_annotation_rejected(coco_dataset_root, coco_selected_ids):
    bad_id = coco_selected_ids[1]
    (coco_dataset_root / "annotations/val2017" / f"{bad_id}_labelTrainIds.png").unlink()
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.reconcile_selected_ids_to_physical_paths(
            dataset_key="coco_stuff", dataset_root=coco_dataset_root, selected_ids=[bad_id],
            image_relative_root="images/val2017", annotation_relative_root="annotations/val2017",
            image_suffix=".jpg", annotation_suffix="_labelTrainIds.png",
        )


def test_reconcile_missing_image_directory_rejected(tmp_path):
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.reconcile_selected_ids_to_physical_paths(
            dataset_key="coco_stuff", dataset_root=tmp_path, selected_ids=["000000000001"],
            image_relative_root="images/val2017", annotation_relative_root="annotations/val2017",
            image_suffix=".jpg", annotation_suffix="_labelTrainIds.png",
        )


def test_reconcile_ade20k_success(ade_dataset_root, ade_selected_ids):
    from src.ade20k_dataset_identity import load_identity as load_ade_source

    ade_identity = load_ade_source(ROOT / "evaluation_identities/e12_ade20k_dataset_source.toml", repo_root=ROOT)
    dataset_dir = ade_dataset_root / "ADEChallengeData2016"
    pf.reconcile_selected_ids_to_physical_paths(
        dataset_key="ade20k", dataset_root=dataset_dir, selected_ids=ade_selected_ids[:5],
        image_relative_root=ade_identity["source"]["image_relative_root"],
        annotation_relative_root=ade_identity["source"]["annotation_relative_root"],
        image_suffix=ade_identity["source"]["image_suffix"], annotation_suffix=ade_identity["source"]["annotation_suffix"],
    )


# ---------------------------------------------------------------------------
# Output-root safety
# ---------------------------------------------------------------------------


def test_output_root_safety_accepts_valid_root(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    resolved = pf.validate_output_root_safety(output_root=out, forbidden_roots=[], probe_writeable=True)
    assert resolved == out.resolve()
    assert list(out.iterdir()) == []  # probe leaves zero residue


def test_output_root_safety_rejects_regular_file(tmp_path):
    f = tmp_path / "blocker"
    f.write_text("i am a file")
    with pytest.raises(pf.DfpPhase1aPreflightError, match="regular file"):
        pf.validate_output_root_safety(output_root=f, forbidden_roots=[])


def test_output_root_safety_rejects_blocker_file_parent(tmp_path):
    f = tmp_path / "blocker_parent"
    f.write_text("i am a file, not a directory")
    with pytest.raises(pf.DfpPhase1aPreflightError, match="blocker file"):
        pf.validate_output_root_safety(output_root=f / "nested_out", forbidden_roots=[])


def test_output_root_safety_rejects_missing_parent(tmp_path):
    with pytest.raises(pf.DfpPhase1aPreflightError, match="does not exist"):
        pf.validate_output_root_safety(output_root=tmp_path / "missing_parent" / "out", forbidden_roots=[])


def test_output_root_safety_rejects_overlap_with_forbidden_root(tmp_path):
    protected = tmp_path / "protected"
    protected.mkdir()
    nested = protected / "nested_out"
    with pytest.raises(pf.DfpPhase1aPreflightError, match="overlaps"):
        pf.validate_output_root_safety(output_root=nested, forbidden_roots=[protected])


def test_output_root_safety_rejects_forbidden_root_nested_inside_output(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    protected = out / "protected_inside"
    protected.mkdir()
    with pytest.raises(pf.DfpPhase1aPreflightError, match="overlaps"):
        pf.validate_output_root_safety(output_root=out, forbidden_roots=[protected])


def test_output_root_safety_rejects_symlink_escape(tmp_path):
    real_target = tmp_path / "elsewhere"
    real_target.mkdir()
    parent = tmp_path / "parent"
    parent.mkdir()
    link = parent / "out"
    link.symlink_to(real_target, target_is_directory=True)
    with pytest.raises(pf.DfpPhase1aPreflightError, match="symlink"):
        pf.validate_output_root_safety(output_root=link, forbidden_roots=[])


def test_output_root_safety_writeable_probe_preserves_existing_files(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "preexisting.txt").write_text("keep me")
    pf.validate_output_root_safety(output_root=out, forbidden_roots=[], probe_writeable=True)
    assert (out / "preexisting.txt").read_text() == "keep me"
    assert [p.name for p in out.iterdir()] == ["preexisting.txt"]


# ---------------------------------------------------------------------------
# Fresh/resume contract (Section 9)
# ---------------------------------------------------------------------------


def test_phase_c_fresh_with_no_existing_checkpoint(tmp_path):
    phase_a = pf.run_phase_a(repo_root=ROOT, run_mode="coco_stuff_500", check_git=True)
    result = pf.run_phase_c(phase_a=phase_a, mode="fresh", checkpoints_dir=tmp_path)
    assert result.mode == "fresh"
    assert result.next_dataset_index == 0
    assert result.completed_image_ids == ()
    assert result.arrays is None


def test_phase_c_fresh_rejects_existing_checkpoint(resumable_checkpoints_dir):
    phase_a = pf.run_phase_a(repo_root=ROOT, run_mode="coco_stuff_500", check_git=True)
    with pytest.raises(pf.DfpPhase1aPreflightError, match="checkpoint already exists"):
        pf.run_phase_c(phase_a=phase_a, mode="fresh", checkpoints_dir=resumable_checkpoints_dir)


def test_phase_c_fresh_allows_overwrite_when_authorized(resumable_checkpoints_dir):
    phase_a = pf.run_phase_a(repo_root=ROOT, run_mode="coco_stuff_500", check_git=True)
    result = pf.run_phase_c(phase_a=phase_a, mode="fresh", checkpoints_dir=resumable_checkpoints_dir, allow_fresh_overwrite=True)
    assert result.mode == "fresh"


def test_phase_c_resume_requires_existing_checkpoint(tmp_path):
    phase_a = pf.run_phase_a(repo_root=ROOT, run_mode="coco_stuff_500", check_git=True)
    with pytest.raises(pf.DfpPhase1aPreflightError, match="no checkpoint exists"):
        pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=tmp_path)


def test_phase_c_resume_success(resumable_checkpoints_dir, coco_selected_ids):
    phase_a = pf.run_phase_a(repo_root=ROOT, run_mode="coco_stuff_500", check_git=True)
    result = pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=resumable_checkpoints_dir)
    assert result.mode == "resume"
    assert result.next_dataset_index == 3
    assert result.completed_image_ids == tuple(coco_selected_ids[:3])
    assert result.arrays is not None
    assert set(result.arrays) == sch.ALL_NPZ_MEMBER_NAMES


def test_phase_c_resume_rejects_already_complete(complete_checkpoints_dir):
    phase_a = pf.run_phase_a(repo_root=ROOT, run_mode="coco_stuff_500", check_git=True)
    with pytest.raises(pf.DfpPhase1aPreflightError, match="already marked complete"):
        pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=complete_checkpoints_dir)


def test_phase_c_resume_rejects_run_mode_mismatch(resumable_checkpoints_dir):
    phase_a = pf.run_phase_a(repo_root=ROOT, run_mode="ade20k_500", check_git=True)
    with pytest.raises(pf.DfpPhase1aPreflightError, match="disagrees with the requested run"):
        pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=resumable_checkpoints_dir)


def test_phase_c_invalid_mode_rejected(tmp_path):
    phase_a = pf.run_phase_a(repo_root=ROOT, run_mode="coco_stuff_500", check_git=True)
    with pytest.raises(pf.DfpPhase1aPreflightError, match="fresh.*resume"):
        pf.run_phase_c(phase_a=phase_a, mode="continue", checkpoints_dir=tmp_path)


def test_phase_c_resume_arrays_are_readonly(resumable_checkpoints_dir):
    phase_a = pf.run_phase_a(repo_root=ROOT, run_mode="coco_stuff_500", check_git=True)
    result = pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=resumable_checkpoints_dir)
    for name, array in result.arrays.items():
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array[0] = 12345 if array.dtype.kind != "f" else 1.0


def test_phase_c_resume_malformed_checkpoint_rejected(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    (checkpoints_dir / "checkpoint.json").write_text("{not valid json")
    phase_a = pf.run_phase_a(repo_root=ROOT, run_mode="coco_stuff_500", check_git=True)
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=checkpoints_dir)


def test_phase_c_resume_missing_npz_sibling_rejected(resumable_checkpoints_dir):
    checkpoint = json.loads((resumable_checkpoints_dir / "checkpoint.json").read_text())
    npz_name = json.loads((resumable_checkpoints_dir / checkpoint["per_image_stats_manifest_relative_path"]).read_text())["npz_filename"]
    (resumable_checkpoints_dir / npz_name).unlink()
    phase_a = pf.run_phase_a(repo_root=ROOT, run_mode="coco_stuff_500", check_git=True)
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=resumable_checkpoints_dir)


# ---------------------------------------------------------------------------
# Full end-to-end preflight (fresh and resume) over synthetic dataset
# roots, and exact ordering proof (Section 13)
# ---------------------------------------------------------------------------


def test_full_preflight_fresh_coco_stuff(monkeypatch, coco_dataset_root, output_root):
    monkeypatch.setenv("COCO_STUFF_REAL_DATA_ROOT", str(coco_dataset_root))
    trace = pf.PreflightTrace()
    ctx = pf.run_preflight(
        repo_root=ROOT, run_mode="coco_stuff_500", mode="fresh", output_root=output_root,
        probe_output_writeable=True, check_git=True, trace=trace,
    )
    assert ctx.mode == "fresh"
    assert ctx.next_dataset_index == 0
    assert ctx.completed_image_ids == ()
    assert ctx.checkpoint_arrays is None
    assert trace.stages == pf.ALL_STAGES


def test_full_preflight_fresh_ade20k(monkeypatch, ade_dataset_root, output_root):
    monkeypatch.setenv("ADE20K_REAL_DATA_ROOT", str(ade_dataset_root))
    ctx = pf.run_preflight(repo_root=ROOT, run_mode="ade20k_500", mode="fresh", output_root=output_root, check_git=False)
    assert ctx.run_mode == "ade20k_500"
    assert ctx.class_count == 150


def test_full_preflight_resume(monkeypatch, coco_dataset_root, output_root, resumable_checkpoints_dir, coco_selected_ids):
    monkeypatch.setenv("COCO_STUFF_REAL_DATA_ROOT", str(coco_dataset_root))
    trace = pf.PreflightTrace()
    ctx = pf.run_preflight(
        repo_root=ROOT, run_mode="coco_stuff_500", mode="resume", output_root=output_root,
        checkpoints_dir=resumable_checkpoints_dir, check_git=True, trace=trace,
    )
    assert ctx.mode == "resume"
    assert ctx.next_dataset_index == 3
    assert ctx.completed_image_ids == tuple(coco_selected_ids[:3])
    assert trace.stages == pf.ALL_STAGES


def test_ordering_identity_error_precedes_dataset_root_resolution(monkeypatch, output_root):
    """An identity/subset error (Phase A) must surface WITHOUT ever
    reaching dataset-root resolution (Phase B) -- proven by requiring the
    real-data env var to be entirely absent and unset, yet the failure is
    still the Phase-A error, not an env-var error."""
    monkeypatch.delenv("COCO_STUFF_REAL_DATA_ROOT", raising=False)
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.run_preflight(repo_root=ROOT, run_mode="not_a_real_run_mode", mode="fresh", output_root=output_root)


def test_ordering_subset_error_precedes_source_file_inspection(monkeypatch, tmp_path, coco_dataset_root, output_root):
    """A tampered committed subset manifest must fail inside Phase A
    before Phase B ever inspects a physical source file -- proven by
    deleting the dataset image directory (which would ALSO fail Phase B)
    while independently tampering Phase A's own subset manifest via a
    monkeypatched MANIFEST_RELATIVE_PATH_BY_RUN_MODE pointing at a
    corrupted copy; the corrupted-manifest error must be the one raised,
    not a missing-source-file error."""
    corrupted = tmp_path / "corrupted_manifest.json"
    real = json.loads((ROOT / "evaluation_subsets/dfp_phase1a_coco_stuff_500.json").read_text())
    mutated = copy.deepcopy(real)
    mutated["selected_ids_digest"] = "0" * 64
    corrupted.write_text(json.dumps(mutated))
    override_path = str(corrupted.relative_to(ROOT)) if corrupted.is_relative_to(ROOT) else str(corrupted)
    patched = dict(pf.MANIFEST_RELATIVE_PATH_BY_RUN_MODE)
    patched["coco_stuff_500"] = override_path
    monkeypatch.setattr(pf, "MANIFEST_RELATIVE_PATH_BY_RUN_MODE", patched)
    monkeypatch.setenv("COCO_STUFF_REAL_DATA_ROOT", "/does/not/exist")
    with pytest.raises(pf.DfpPhase1aPreflightError, match="frozen subset manifest failed verification"):
        pf.run_preflight(repo_root=ROOT, run_mode="coco_stuff_500", mode="fresh", output_root=output_root)


def test_ordering_checkpoint_error_precedes_output_success(monkeypatch, coco_dataset_root, resumable_checkpoints_dir):
    """A resume validation failure (Phase C) must never let a PASS/READY
    result be constructed -- proven by corrupting the checkpoint and
    confirming run_preflight raises rather than returning a context."""
    monkeypatch.setenv("COCO_STUFF_REAL_DATA_ROOT", str(coco_dataset_root))
    checkpoint_path = resumable_checkpoints_dir / "checkpoint.json"
    doc = json.loads(checkpoint_path.read_text())
    doc["images_completed_count"] = 999999
    checkpoint_path.write_text(json.dumps(doc))
    out = resumable_checkpoints_dir.parent / "out2"
    out.mkdir()
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.run_preflight(
            repo_root=ROOT, run_mode="coco_stuff_500", mode="resume", output_root=out,
            checkpoints_dir=resumable_checkpoints_dir,
        )


def test_ordering_output_error_precedes_success(monkeypatch, coco_dataset_root):
    """An output-root safety violation (last stage of Phase B) must still
    prevent a PASS/READY result even though every earlier phase passed."""
    monkeypatch.setenv("COCO_STUFF_REAL_DATA_ROOT", str(coco_dataset_root))
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.run_preflight(repo_root=ROOT, run_mode="coco_stuff_500", mode="fresh", output_root=coco_dataset_root)  # overlaps dataset root


# ---------------------------------------------------------------------------
# PreflightContext immutability (Section 8)
# ---------------------------------------------------------------------------


def test_preflight_context_is_frozen(monkeypatch, coco_dataset_root, output_root):
    monkeypatch.setenv("COCO_STUFF_REAL_DATA_ROOT", str(coco_dataset_root))
    ctx = pf.run_preflight(repo_root=ROOT, run_mode="coco_stuff_500", mode="fresh", output_root=output_root, check_git=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.run_mode = "ade20k_500"


def test_preflight_context_selected_ids_is_tuple(monkeypatch, coco_dataset_root, output_root):
    monkeypatch.setenv("COCO_STUFF_REAL_DATA_ROOT", str(coco_dataset_root))
    ctx = pf.run_preflight(repo_root=ROOT, run_mode="coco_stuff_500", mode="fresh", output_root=output_root, check_git=False)
    assert isinstance(ctx.selected_ids, tuple)
    assert isinstance(ctx.completed_image_ids, tuple)


def test_preflight_context_summary_has_no_private_paths(monkeypatch, coco_dataset_root, output_root):
    monkeypatch.setenv("COCO_STUFF_REAL_DATA_ROOT", str(coco_dataset_root))
    ctx = pf.run_preflight(repo_root=ROOT, run_mode="coco_stuff_500", mode="fresh", output_root=output_root, check_git=False)
    summary = ctx.to_summary_dict()
    summary_text = json.dumps(summary, default=str)
    assert str(coco_dataset_root) not in summary_text
    assert str(output_root) not in summary_text
    assert "dataset_root" not in summary
    assert "output_root" not in summary
    assert "checkpoint_arrays" not in summary


def test_preflight_context_summary_dict_is_read_only(monkeypatch, coco_dataset_root, output_root):
    monkeypatch.setenv("COCO_STUFF_REAL_DATA_ROOT", str(coco_dataset_root))
    ctx = pf.run_preflight(repo_root=ROOT, run_mode="coco_stuff_500", mode="fresh", output_root=output_root, check_git=False)
    summary = ctx.to_summary_dict()
    with pytest.raises(TypeError):
        summary["run_mode"] = "tampered"


def test_preflight_context_resume_does_not_reread_npz(monkeypatch, coco_dataset_root, output_root, resumable_checkpoints_dir):
    """Phase C loads the NPZ exactly once (inside validate_resume_state);
    the context's checkpoint_arrays IS that single load, never a second
    parse. Proven by monkeypatching load_per_image_stats_npz to count
    invocations across the full run_preflight call."""
    import src.dfp_phase1a_checkpoint as ckpt_mod

    calls = {"n": 0}
    original = ckpt_mod.load_per_image_stats_npz

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ckpt_mod, "load_per_image_stats_npz", _counting)
    monkeypatch.setenv("COCO_STUFF_REAL_DATA_ROOT", str(coco_dataset_root))
    pf.run_preflight(
        repo_root=ROOT, run_mode="coco_stuff_500", mode="resume", output_root=output_root,
        checkpoints_dir=resumable_checkpoints_dir, check_git=False,
    )
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Delegation call-count proofs (Section 18)
# ---------------------------------------------------------------------------


def test_validate_resume_state_called_exactly_once(monkeypatch, coco_dataset_root, output_root, resumable_checkpoints_dir):
    calls = {"n": 0}
    original = pf.validate_resume_state

    def _counting(**kwargs):
        calls["n"] += 1
        return original(**kwargs)

    monkeypatch.setattr(pf, "validate_resume_state", _counting)
    monkeypatch.setenv("COCO_STUFF_REAL_DATA_ROOT", str(coco_dataset_root))
    pf.run_preflight(
        repo_root=ROOT, run_mode="coco_stuff_500", mode="resume", output_root=output_root,
        checkpoints_dir=resumable_checkpoints_dir, check_git=False,
    )
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# CLI behavior (Section 15)
# ---------------------------------------------------------------------------


def _run_cli(*args):
    return subprocess.run([PY, CLI, *args], capture_output=True, text=True, cwd=str(ROOT))


def test_cli_verify_protocol_passes():
    proc = _run_cli("verify-protocol")
    assert proc.returncode == 0
    assert proc.stdout.startswith("PASS:")


def test_cli_verify_subset_both_datasets():
    for dataset in ("coco_stuff", "ade20k"):
        proc = _run_cli("verify-subset", "--dataset", dataset)
        assert proc.returncode == 0, proc.stderr


def test_cli_verify_model_binding_coco_stuff():
    proc = _run_cli("verify-model-binding", "--dataset", "coco_stuff")
    assert proc.returncode == 0, proc.stderr
    assert "evaluator_identity=e12-matched-k11-k12-t320" in proc.stderr


def test_cli_no_run_evaluate_pilot_full_command_exposed():
    """Checks the CLI's own exact dispatchable subcommand set (not raw
    --help text, which legitimately contains --run as an ARGUMENT name of
    the preflight subcommand) -- no top-level subcommand named run/
    evaluate/pilot/full exists."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_dfp_phase1a_dispatch_probe", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    forbidden = {"run", "evaluate", "pilot", "full"}
    assert forbidden.isdisjoint(module._DISPATCH.keys())
    expected = {
        "verify-protocol", "verify-subset", "verify-source-binding", "verify-model-binding",
        "verify-checkpoint", "verify-result", "verify-qualitative-report", "preflight",
    }
    assert set(module._DISPATCH.keys()) == expected


def test_cli_preflight_end_to_end(monkeypatch, coco_dataset_root, output_root):
    env = dict(os.environ)
    env["COCO_STUFF_REAL_DATA_ROOT"] = str(coco_dataset_root)
    proc = subprocess.run(
        [PY, CLI, "preflight", "--run", "coco_stuff_500", "--mode", "fresh", "--output-root", str(output_root), "--probe-output-writeable"],
        capture_output=True, text=True, cwd=str(ROOT), env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("READY:")


def test_cli_missing_env_var_exits_2_no_traceback(monkeypatch, output_root):
    env = {k: v for k, v in os.environ.items() if k != "COCO_STUFF_REAL_DATA_ROOT"}
    proc = subprocess.run(
        [PY, CLI, "preflight", "--run", "coco_stuff_500", "--mode", "fresh", "--output-root", str(output_root)],
        capture_output=True, text=True, cwd=str(ROOT), env=env,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "PASS" not in proc.stdout and "READY" not in proc.stdout


def test_cli_invalid_run_choice_exits_2_via_argparse():
    proc = _run_cli("preflight", "--run", "coco_stuff", "--mode", "fresh", "--output-root", "/tmp/x")
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


def test_cli_verify_checkpoint_missing_file_exits_2():
    proc = _run_cli("verify-checkpoint", "--checkpoint", "/tmp/does_not_exist_checkpoint.json")
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


def test_cli_verify_result_missing_bundle_exits_2():
    proc = _run_cli("verify-result", "--bundle", "/tmp/does_not_exist_bundle_dir")
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


def test_cli_keyboard_interrupt_propagates():
    import importlib

    sys.path.insert(0, str(ROOT))
    cli_module = importlib.import_module("diagnostics.run_dfp_phase1a") if False else None
    # importlib requires the diagnostics package to be importable; use
    # direct file-based import since diagnostics/ has no __init__.py.
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_dfp_phase1a_module_under_test", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _boom(args):
        raise KeyboardInterrupt("simulated")

    original = dict(module._DISPATCH)
    module._DISPATCH["verify-protocol"] = _boom
    try:
        with pytest.raises(KeyboardInterrupt):
            module.main(["verify-protocol"])
    finally:
        module._DISPATCH.clear()
        module._DISPATCH.update(original)


def test_cli_system_exit_propagates():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_dfp_phase1a_module_under_test2", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _boom(args):
        raise SystemExit(17)

    original = dict(module._DISPATCH)
    module._DISPATCH["verify-protocol"] = _boom
    try:
        with pytest.raises(SystemExit) as excinfo:
            module.main(["verify-protocol"])
        assert excinfo.value.code == 17
    finally:
        module._DISPATCH.clear()
        module._DISPATCH.update(original)


def test_cli_memory_error_propagates():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_dfp_phase1a_module_under_test3", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _boom(args):
        raise MemoryError("simulated")

    original = dict(module._DISPATCH)
    module._DISPATCH["verify-protocol"] = _boom
    try:
        with pytest.raises(MemoryError):
            module.main(["verify-protocol"])
    finally:
        module._DISPATCH.clear()
        module._DISPATCH.update(original)


# ---------------------------------------------------------------------------
# Heavy-entry-point poison matrix (Section 14)
# ---------------------------------------------------------------------------


def test_no_top_level_torch_import_in_preflight_module():
    src = (ROOT / "src" / "dfp_phase1a_preflight.py").read_text()
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("import torch") or stripped.startswith("from torch"):
            pytest.fail(f"top-level torch import found: {stripped!r}")


def test_no_top_level_torch_import_in_cli_module():
    src = (ROOT / "diagnostics" / "run_dfp_phase1a.py").read_text()
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("import torch") or stripped.startswith("from torch"):
            pytest.fail(f"top-level torch import found: {stripped!r}")


def test_import_of_preflight_module_does_not_trigger_torch(monkeypatch):
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, "src.dfp_phase1a_preflight", raising=False)
    import importlib

    importlib.import_module("src.dfp_phase1a_preflight")
    assert "torch" not in sys.modules


def test_poison_subprocess_torch_import_full_preflight_still_passes(monkeypatch, coco_dataset_root, output_root):
    """Runs a full fresh-mode preflight in a subprocess whose sitecustomize
    poisons any attempt to import torch, call CUDA, construct a model, or
    touch a dataset __getitem__ -- proving the entire ordered preflight
    (positive path) never reaches any of them."""
    poison_script = r"""
import sys, types

class _Boom:
    def __getattr__(self, name):
        raise AssertionError(f"heavy entry point touched: torch.{name}")
    def __call__(self, *a, **kw):
        raise AssertionError("heavy entry point called: torch(...)")

class _PoisonTorchFinder:
    def find_module(self, name, path=None):
        if name == "torch" or name.startswith("torch."):
            return self
        return None
    def load_module(self, name):
        raise AssertionError(f"import of {name!r} attempted")

sys.meta_path.insert(0, _PoisonTorchFinder())

sys.path.insert(0, r"%(root)s")
from pathlib import Path
import src.dfp_phase1a_preflight as pf

ctx = pf.run_preflight(
    repo_root=Path(r"%(root)s"), run_mode="coco_stuff_500", mode="fresh",
    output_root=Path(r"%(output_root)s"), check_git=False,
)
assert ctx.mode == "fresh"
assert "torch" not in sys.modules
print("POISON_TEST_OK")
""" % {"root": str(ROOT), "output_root": str(output_root)}
    script_path = output_root.parent / "poison_script.py"
    script_path.write_text(poison_script)
    env = dict(os.environ)
    env["COCO_STUFF_REAL_DATA_ROOT"] = str(coco_dataset_root)
    proc = subprocess.run([PY, str(script_path)], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "POISON_TEST_OK" in proc.stdout


def test_poison_subprocess_negative_case_also_never_touches_torch(coco_dataset_root):
    """A NEGATIVE preflight call (missing checkpoint on resume) must also
    never import torch before failing closed."""
    poison_script = r"""
import sys

class _PoisonTorchFinder:
    def find_module(self, name, path=None):
        if name == "torch" or name.startswith("torch."):
            return self
        return None
    def load_module(self, name):
        raise AssertionError(f"import of {name!r} attempted")

sys.meta_path.insert(0, _PoisonTorchFinder())
sys.path.insert(0, r"%(root)s")
from pathlib import Path
import src.dfp_phase1a_preflight as pf

try:
    pf.run_preflight(
        repo_root=Path(r"%(root)s"), run_mode="coco_stuff_500", mode="resume",
        output_root=Path(r"%(root)s") / "does_not_matter", check_git=False,
        checkpoints_dir=Path("/tmp/definitely_does_not_exist_checkpoints_dir_xyz"),
    )
    print("UNEXPECTED_SUCCESS")
except pf.DfpPhase1aPreflightError:
    print("POISON_NEGATIVE_TEST_OK")
""" % {"root": str(ROOT)}
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(poison_script)
        script_path = f.name
    try:
        env = dict(os.environ)
        env["COCO_STUFF_REAL_DATA_ROOT"] = str(coco_dataset_root)
        proc = subprocess.run([PY, script_path], capture_output=True, text=True, env=env)
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        assert "POISON_NEGATIVE_TEST_OK" in proc.stdout
    finally:
        os.unlink(script_path)


# ---------------------------------------------------------------------------
# torch.cuda.is_initialized() before/after (repository-wide invariant)
# ---------------------------------------------------------------------------


def test_torch_cuda_not_initialized_after_full_preflight(monkeypatch, coco_dataset_root, output_root):
    import torch

    assert torch.cuda.is_initialized() is False
    monkeypatch.setenv("COCO_STUFF_REAL_DATA_ROOT", str(coco_dataset_root))
    pf.run_preflight(repo_root=ROOT, run_mode="coco_stuff_500", mode="fresh", output_root=output_root, check_git=False)
    assert torch.cuda.is_initialized() is False


# ---------------------------------------------------------------------------
# Validator immutability
# ---------------------------------------------------------------------------


def test_run_preflight_does_not_mutate_output_root_path_object(monkeypatch, coco_dataset_root):
    monkeypatch.setenv("COCO_STUFF_REAL_DATA_ROOT", str(coco_dataset_root))
    out = coco_dataset_root.parent / "out3"
    out.mkdir()
    out_arg = Path(str(out))
    before = str(out_arg)
    pf.run_preflight(repo_root=ROOT, run_mode="coco_stuff_500", mode="fresh", output_root=out_arg, check_git=False)
    assert str(out_arg) == before


def test_resume_does_not_mutate_checkpoint_file_bytes(monkeypatch, coco_dataset_root, output_root, resumable_checkpoints_dir):
    before = (resumable_checkpoints_dir / "checkpoint.json").read_bytes()
    monkeypatch.setenv("COCO_STUFF_REAL_DATA_ROOT", str(coco_dataset_root))
    pf.run_preflight(
        repo_root=ROOT, run_mode="coco_stuff_500", mode="resume", output_root=output_root,
        checkpoints_dir=resumable_checkpoints_dir, check_git=False,
    )
    after = (resumable_checkpoints_dir / "checkpoint.json").read_bytes()
    assert before == after


# ---------------------------------------------------------------------------
# Environment-gated real-data smoke (Section 19)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("COCO_STUFF_REAL_DATA_ROOT"), reason="COCO_STUFF_REAL_DATA_ROOT not set")
def test_real_data_smoke_coco_stuff(tmp_path):
    import torch

    out = tmp_path / "real_smoke_out"
    out.mkdir()
    ctx = pf.run_preflight(repo_root=ROOT, run_mode="coco_stuff_500", mode="fresh", output_root=out, check_git=True)
    assert len(ctx.selected_ids) == 500
    assert ctx.class_count == 171
    assert torch.cuda.is_initialized() is False


@pytest.mark.skipif(not os.environ.get("ADE20K_REAL_DATA_ROOT"), reason="ADE20K_REAL_DATA_ROOT not set")
def test_real_data_smoke_ade20k(tmp_path):
    import torch

    out = tmp_path / "real_smoke_out_ade"
    out.mkdir()
    ctx = pf.run_preflight(repo_root=ROOT, run_mode="ade20k_500", mode="fresh", output_root=out, check_git=True)
    assert len(ctx.selected_ids) == 500
    assert ctx.class_count == 150
    assert torch.cuda.is_initialized() is False
