"""Tests for the DFP Phase 1A preflight repair that allows a complete-
but-unfinalized schema-v2 checkpoint to reach the already-verified
finalize-only branch (closing the real gap SLURM job 21063877 exposed:
``execute --mode resume`` unconditionally refused an already-complete
checkpoint at Phase C, before ``execute_run`` ever got a chance to
inspect it).

Covers: the exact finalize-only eligibility predicate
(``src.dfp_phase1a_preflight.run_phase_c``), the synthetic eligibility
matrix (incomplete checkpoints keep their prior resume behavior exactly;
only complete+v2+unfinalized+absent-result checkpoints receive
``EXECUTION_ACTION_FINALIZE_ONLY``; every inconsistent/mismatched/
already-finalized state fails closed), the real unified CLI reaching
``READY_TO_FINALIZE``, the real CLI ``execute --mode resume`` reaching
the finalize-only branch with every heavy entry point poisoned, fresh-
mode protection against a complete checkpoint, and that
``validate_resume_state``'s ``resumable=False`` meaning for a complete
checkpoint is completely unchanged.

CPU-only throughout. No real dataset, no model construction, no CUDA
initialization, no GPU job. The real production checkpoint from job
21063877 is read-only inspected in one test, gated behind
COCO_STUFF_REAL_DATA_ROOT, and never modified."""

from __future__ import annotations

import datetime
import io
import contextlib
import json
import os
import sys
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

import src.dfp_phase1a_artifact_schema as sch
import src.dfp_phase1a_checkpoint as ckpt
import src.dfp_phase1a_execute as ex
import src.dfp_phase1a_preflight as pf
import src.dfp_phase1a_report as report
import diagnostics.run_dfp_phase1a as cli

ROOT = Path(__file__).resolve().parent.parent
# validate_resume_state cross-checks checkpoint.class_count against the
# LIVE frozen artifact identity's own coco_stuff_class_count -- unlike
# the execute-level fixtures in tests/test_dfp_phase1a_finalize.py
# (which never reach validate_resume_state), every fixture here must
# use the real, current value, never an arbitrary placeholder.
CLASS_COUNT = sch.load_identity(repo_root=ROOT)["dataset_binding"]["coco_stuff_class_count"]


# ---------------------------------------------------------------------------
# Real-identity-bound v2 checkpoint fixture builder (validate_resume_state
# checks against LIVE identity/manifest files, unlike the execute-level
# fixtures in tests/test_dfp_phase1a_finalize.py, which never reach it).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coco_selected_ids():
    manifest = json.loads((ROOT / "evaluation_subsets/dfp_phase1a_coco_stuff_500.json").read_text())
    return manifest["selected_ids"], manifest["selected_ids_digest"]


def _make_v2_arrays(n: int, class_count: int = CLASS_COUNT) -> dict[str, np.ndarray]:
    a = ckpt.build_zero_row_arrays_v2(class_count=class_count)
    a["dataset_indices"] = np.arange(n, dtype=np.int64)
    for name in sch.SCALAR_INT64_MEMBER_NAMES_V2:
        if name != "dataset_indices":
            a[name] = np.zeros(n, dtype=np.int64)
    a["evaluated_pixel_count"] = np.full(n, 10, dtype=np.int64)
    for name in sch.SCALAR_FLOAT64_MEMBER_NAMES_V2:
        a[name] = np.zeros(n, dtype=np.float64)
    for variant in sch.PREDICTION_VARIANT_NAMES_V2:
        for stat in ("intersection", "predicted_pixels", "ground_truth_pixels", "union"):
            arr = np.zeros((n, class_count), dtype=np.int64)
            arr[:, 0] = 10
            a[sch.confusion_member_name_v2(variant, stat)] = arr
    a["component_count"] = np.ones(n, dtype=np.int64)
    a["component_offsets"] = np.arange(n + 1, dtype=np.int64)
    a["component_class_indices_flat"] = np.zeros(n, dtype=np.int64)
    a["component_pixel_areas_flat"] = np.full(n, 10, dtype=np.int64)
    a["component_patch_equivalent_areas_flat"] = np.full(n, 10.0, dtype=np.float64)
    a["component_source_mass_flat"] = np.ones(n, dtype=np.float64)
    a["reversion_component_offsets"] = np.zeros(n + 1, dtype=np.int64)
    return a


def _build_v2_checkpoint(
    checkpoints_dir: Path, *, selected_ids: list[str], selected_ids_digest: str, n: int, complete: bool,
    class_count: int = CLASS_COUNT, final_result_produced: bool = False, corrupt_field: dict | None = None,
    corrupt_npz: bool = False, mislabel_v1: bool = False,
) -> Path:
    """Hand-build a real-identity-bound, schema-valid v2 checkpoint +
    per-image-stats manifest + NPZ under ``checkpoints_dir``, mirroring
    the driver's own real writer's on-disk shape closely enough for
    ``validate_resume_state`` to accept it. ``corrupt_field`` overrides
    one or more checkpoint top-level fields AFTER construction, to
    probe a specific inconsistent state."""
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    schema_summary = sch.validate_static_configuration(repo_root=ROOT)
    artifact_identity = sch.load_identity(repo_root=ROOT)

    arrays = _make_v2_arrays(n, class_count=class_count)
    if corrupt_npz:
        del arrays["component_source_mass_flat"]
    npz_filename = f"dfp_phase1a_per_image_stats_v2_{checkpoints_dir.name}_{n}.npz"
    npz_path = checkpoints_dir / npz_filename
    np.savez(npz_path, allow_pickle=False, **arrays)
    npz_sha256 = sch.sha256_file(npz_path)

    stats_schema = (sch.PER_IMAGE_STATS_SCHEMA_NAME, 1) if mislabel_v1 else (sch.PER_IMAGE_STATS_SCHEMA_NAME_V2, sch.PER_IMAGE_STATS_SCHEMA_VERSION_V2)
    stats_manifest = {
        "schema": stats_schema[0], "schema_version": stats_schema[1],
        "npz_filename": npz_filename, "npz_sha256": npz_sha256, "run_mode": "coco_stuff_500",
        "class_count": class_count, "class_names_digest": artifact_identity["dataset_binding"]["coco_stuff_class_names_digest"],
        "image_count": n, "image_ids": list(selected_ids[:n]),
    }
    manifest_path = checkpoints_dir / "stats_manifest.json"
    sch.write_json_atomically(manifest_path, stats_manifest, label="per-image-stats manifest")
    manifest_sha256 = sch.sha256_file(manifest_path)

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    checkpoint_fields = {
        "schema": sch.CHECKPOINT_SCHEMA_NAME, "schema_version": sch.CHECKPOINT_SCHEMA_VERSION,
        "artifact_schema_identity_sha256": sch.sha256_file(ROOT / sch.IDENTITY_RELATIVE_PATH),
        "protocol_identity_sha256": schema_summary["protocol_identity_sha256"],
        "subset_manifest_sha256": schema_summary["coco_stuff_manifest_sha256"], "selected_ids_digest": selected_ids_digest,
        "run_mode": "coco_stuff_500", "image_count_expected": n if complete else len(selected_ids),
        "class_count": class_count, "class_names_digest": artifact_identity["dataset_binding"]["coco_stuff_class_names_digest"],
        "horizon_order": list(sch.HORIZON_ORDER), "source_commit": "a" * 40,
        "required_ancestor_commit": artifact_identity["parent_identity_binding"]["required_ancestor_commit"],
        "bridge_checkpoint_sha256": schema_summary["bridge_checkpoint_sha256"], "next_dataset_index": n,
        "completed_image_ids": list(selected_ids[:n]), "images_completed_count": n, "total_windows_processed": max(n, 1),
        "complete": complete, "final_result_produced": final_result_produced, "created_at_utc": now, "updated_at_utc": now,
        "per_image_stats_manifest_relative_path": "stats_manifest.json", "per_image_stats_manifest_sha256": manifest_sha256,
        "generation_nonce": "deadbeef" * 4,
    }
    if corrupt_field:
        checkpoint_fields.update(corrupt_field)
    sch.write_json_atomically(checkpoints_dir / "checkpoint.json", checkpoint_fields, label="checkpoint")
    return checkpoints_dir


@pytest.fixture()
def phase_a():
    return pf.run_phase_a(repo_root=ROOT, run_mode="coco_stuff_500", check_git=True)


# ---------------------------------------------------------------------------
# Section 12: synthetic eligibility matrix.
# ---------------------------------------------------------------------------


def test_incomplete_checkpoints_keep_prior_resume_behavior(tmp_path, phase_a, coco_selected_ids):
    """Cases 1-4: incomplete checkpoints (0/500, 1/500, 250/500, 499/500)
    all still receive EXECUTION_ACTION_PROCESS, byte-for-byte the
    original resumable=True behavior."""
    ids, digest = coco_selected_ids
    for n in (0, 1, 250, 499):
        d = _build_v2_checkpoint(tmp_path / f"incomplete_{n}", selected_ids=ids, selected_ids_digest=digest, n=n, complete=False)
        result = pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=d)
        assert result.execution_action == pf.EXECUTION_ACTION_PROCESS
        assert result.next_dataset_index == n


def test_complete_unfinalized_absent_result_grants_finalize_only(tmp_path, phase_a, coco_selected_ids):
    """Case 5: the exact target state."""
    ids, digest = coco_selected_ids
    d = _build_v2_checkpoint(tmp_path / "complete", selected_ids=ids, selected_ids_digest=digest, n=len(ids), complete=True)
    result = pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=d)
    assert result.execution_action == pf.EXECUTION_ACTION_FINALIZE_ONLY
    assert result.next_dataset_index == len(ids)
    assert result.completed_image_ids == tuple(ids)


def test_complete_with_valid_final_result_is_refused_as_already_finalized(tmp_path, phase_a, coco_selected_ids):
    """Case 6: complete, result true, valid result -- refused, not
    reconciled (matches install_result_bundle's own non-idempotent
    contract: it always refuses over an existing destination)."""
    ids, digest = coco_selected_ids
    d = _build_v2_checkpoint(tmp_path / "already_final", selected_ids=ids, selected_ids_digest=digest, n=len(ids), complete=True)
    dest = ex._resolve_result_bundle_dir(d)
    report.install_result_bundle(destination_dir=dest, checkpoints_dir=d, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    with pytest.raises(pf.DfpPhase1aPreflightError, match="already-finalized"):
        pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=d)


def test_complete_unfinalized_with_stray_content_at_destination_is_refused(tmp_path, phase_a, coco_selected_ids):
    """Case 8 (partial/invalid result at the destination): a directory
    exists at the authoritative destination but is not a valid result
    bundle -- refused with a distinct, ambiguous-destination diagnostic,
    never silently treated as absent."""
    ids, digest = coco_selected_ids
    d = _build_v2_checkpoint(tmp_path / "stray_dest", selected_ids=ids, selected_ids_digest=digest, n=len(ids), complete=True)
    dest = ex._resolve_result_bundle_dir(d)
    dest.mkdir(parents=True)
    (dest / "not_a_result.txt").write_text("stray")
    with pytest.raises(pf.DfpPhase1aPreflightError, match="failed verification"):
        pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=d)


@pytest.mark.parametrize(
    "label, corrupt_field",
    [
        ("case9_next_index_mismatch", {"next_dataset_index": 499}),
        ("case10_images_completed_count_mismatch", {"images_completed_count": 499}),
    ],
)
def test_complete_with_inconsistent_counts_fails_not_finalize(tmp_path, phase_a, coco_selected_ids, label, corrupt_field):
    """Cases 9-10: a complete checkpoint with self-inconsistent progress
    counters must fail (via the pre-existing, unmodified
    validate_checkpoint_structure), never reach finalize_only."""
    ids, digest = coco_selected_ids
    d = _build_v2_checkpoint(tmp_path / label, selected_ids=ids, selected_ids_digest=digest, n=len(ids), complete=True, corrupt_field=corrupt_field)
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=d)


def test_complete_with_truncated_id_list_fails_not_finalize(tmp_path, phase_a, coco_selected_ids):
    """Case 11: completed_image_ids shorter than image_count_expected
    while complete=True and next_dataset_index=expected_count."""
    ids, digest = coco_selected_ids
    d = _build_v2_checkpoint(tmp_path / "case11", selected_ids=ids, selected_ids_digest=digest, n=len(ids), complete=True)
    d = tmp_path / "case11"
    ckpt_path = d / "checkpoint.json"
    doc = json.loads(ckpt_path.read_text())
    doc["completed_image_ids"] = doc["completed_image_ids"][:-1]
    sch.write_json_atomically(ckpt_path, doc, label="checkpoint")
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=d)


def test_complete_with_reordered_ids_fails_not_finalize(tmp_path, phase_a, coco_selected_ids):
    """Case 12."""
    ids, digest = coco_selected_ids
    d = _build_v2_checkpoint(tmp_path / "case12", selected_ids=ids, selected_ids_digest=digest, n=len(ids), complete=True)
    ckpt_path = d / "checkpoint.json"
    doc = json.loads(ckpt_path.read_text())
    doc["completed_image_ids"] = list(reversed(doc["completed_image_ids"]))
    sch.write_json_atomically(ckpt_path, doc, label="checkpoint")
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=d)


def test_complete_with_wrong_final_id_fails_not_finalize(tmp_path, phase_a, coco_selected_ids):
    """Case 13."""
    ids, digest = coco_selected_ids
    d = _build_v2_checkpoint(tmp_path / "case13", selected_ids=ids, selected_ids_digest=digest, n=len(ids), complete=True)
    ckpt_path = d / "checkpoint.json"
    doc = json.loads(ckpt_path.read_text())
    doc["completed_image_ids"][-1] = "000000000001"  # not the frozen final ID
    sch.write_json_atomically(ckpt_path, doc, label="checkpoint")
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=d)


def _make_v1_arrays(n: int, seed: int = 0) -> dict[str, np.ndarray]:
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


def test_complete_v1_checkpoint_fails_not_finalize(tmp_path, coco_selected_ids):
    """Case 14: complete, but the per-image-stats schema is v1 -- never
    eligible for finalize-only recovery. A genuinely v1-shaped checkpoint
    (the real v1 writer, real v1 NPZ member set) -- mirrors ``tests/
    test_dfp_phase1a_preflight.py``'s own ``complete_checkpoints_dir``
    fixture, which already independently proves the base rejection still
    fires (test_phase_c_resume_rejects_already_complete); this test
    additionally confirms the v2-specific finalize-only message."""
    ids, digest = coco_selected_ids
    n = len(ids)
    schema_summary = sch.validate_static_configuration(repo_root=ROOT)
    artifact_identity = sch.load_identity(repo_root=ROOT)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    fields = {
        "schema": sch.CHECKPOINT_SCHEMA_NAME, "schema_version": sch.CHECKPOINT_SCHEMA_VERSION,
        "artifact_schema_identity_sha256": sch.sha256_file(ROOT / sch.IDENTITY_RELATIVE_PATH),
        "protocol_identity_sha256": schema_summary["protocol_identity_sha256"],
        "subset_manifest_sha256": schema_summary["coco_stuff_manifest_sha256"], "selected_ids_digest": digest,
        "run_mode": "coco_stuff_500", "image_count_expected": n,
        "class_count": artifact_identity["dataset_binding"]["coco_stuff_class_count"],
        "class_names_digest": artifact_identity["dataset_binding"]["coco_stuff_class_names_digest"],
        "horizon_order": list(sch.HORIZON_ORDER), "source_commit": "a" * 40,
        "required_ancestor_commit": artifact_identity["parent_identity_binding"]["required_ancestor_commit"],
        "bridge_checkpoint_sha256": schema_summary["bridge_checkpoint_sha256"], "next_dataset_index": n,
        "completed_image_ids": list(ids[:n]), "images_completed_count": n, "total_windows_processed": n * 4,
        "complete": True, "final_result_produced": False, "created_at_utc": now, "updated_at_utc": now,
    }
    checkpoints_dir = tmp_path / "case14_v1"
    ckpt.generate_and_install_checkpoint(
        checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
        arrays=_make_v1_arrays(n, seed=9), image_ids=ids[:n],
    )
    phase_a = pf.run_phase_a(repo_root=ROOT, run_mode="coco_stuff_500", check_git=True)
    with pytest.raises(pf.DfpPhase1aPreflightError, match="never eligible for finalize-only"):
        pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=checkpoints_dir)


def test_complete_checkpoint_for_another_run_mode_fails(tmp_path, phase_a, coco_selected_ids):
    """Case 15 (approximation: run_mode mismatch is the closed-schema
    equivalent of 'checkpoint for another run' within one dataset's own
    manifest space)."""
    ids, digest = coco_selected_ids
    d = _build_v2_checkpoint(tmp_path / "case15", selected_ids=ids, selected_ids_digest=digest, n=len(ids), complete=True)
    ade_phase_a = pf.run_phase_a(repo_root=ROOT, run_mode="ade20k_500", check_git=True)
    with pytest.raises(pf.DfpPhase1aPreflightError, match="disagrees with the requested run"):
        pf.run_phase_c(phase_a=ade_phase_a, mode="resume", checkpoints_dir=d)


def test_complete_checkpoint_with_corrupted_npz_fails(tmp_path, phase_a, coco_selected_ids):
    """Case 16."""
    ids, digest = coco_selected_ids
    d = _build_v2_checkpoint(tmp_path / "case16", selected_ids=ids, selected_ids_digest=digest, n=len(ids), complete=True, corrupt_npz=True)
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=d)


def test_complete_checkpoint_with_stale_digest_fails(tmp_path, phase_a, coco_selected_ids):
    """Case 17: manifest digest binding tampered after construction."""
    ids, digest = coco_selected_ids
    d = _build_v2_checkpoint(tmp_path / "case17", selected_ids=ids, selected_ids_digest=digest, n=len(ids), complete=True)
    ckpt_path = d / "checkpoint.json"
    doc = json.loads(ckpt_path.read_text())
    doc["per_image_stats_manifest_sha256"] = "0" * 64
    sch.write_json_atomically(ckpt_path, doc, label="checkpoint")
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=d)


def test_complete_checkpoint_with_invalid_offsets_fails(tmp_path, phase_a, coco_selected_ids):
    """Case 18."""
    ids, digest = coco_selected_ids
    n = len(ids)
    checkpoints_dir = tmp_path / "case18"
    checkpoints_dir.mkdir(parents=True)
    schema_summary = sch.validate_static_configuration(repo_root=ROOT)
    artifact_identity = sch.load_identity(repo_root=ROOT)
    arrays = _make_v2_arrays(n, class_count=CLASS_COUNT)
    arrays["component_offsets"][-1] = 999999  # invalid terminal offset
    npz_path = checkpoints_dir / "stats.npz"
    np.savez(npz_path, allow_pickle=False, **arrays)
    npz_sha256 = sch.sha256_file(npz_path)
    stats_manifest = {
        "schema": sch.PER_IMAGE_STATS_SCHEMA_NAME_V2, "schema_version": sch.PER_IMAGE_STATS_SCHEMA_VERSION_V2,
        "npz_filename": "stats.npz", "npz_sha256": npz_sha256, "run_mode": "coco_stuff_500", "class_count": CLASS_COUNT,
        "class_names_digest": artifact_identity["dataset_binding"]["coco_stuff_class_names_digest"], "image_count": n,
        "image_ids": list(ids[:n]),
    }
    manifest_path = checkpoints_dir / "stats_manifest.json"
    sch.write_json_atomically(manifest_path, stats_manifest, label="m")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    checkpoint_fields = {
        "schema": sch.CHECKPOINT_SCHEMA_NAME, "schema_version": sch.CHECKPOINT_SCHEMA_VERSION,
        "artifact_schema_identity_sha256": sch.sha256_file(ROOT / sch.IDENTITY_RELATIVE_PATH),
        "protocol_identity_sha256": schema_summary["protocol_identity_sha256"],
        "subset_manifest_sha256": schema_summary["coco_stuff_manifest_sha256"], "selected_ids_digest": digest,
        "run_mode": "coco_stuff_500", "image_count_expected": n, "class_count": CLASS_COUNT,
        "class_names_digest": artifact_identity["dataset_binding"]["coco_stuff_class_names_digest"],
        "horizon_order": list(sch.HORIZON_ORDER), "source_commit": "a" * 40,
        "required_ancestor_commit": artifact_identity["parent_identity_binding"]["required_ancestor_commit"],
        "bridge_checkpoint_sha256": schema_summary["bridge_checkpoint_sha256"], "next_dataset_index": n,
        "completed_image_ids": list(ids[:n]), "images_completed_count": n, "total_windows_processed": n,
        "complete": True, "final_result_produced": False, "created_at_utc": now, "updated_at_utc": now,
        "per_image_stats_manifest_relative_path": "stats_manifest.json", "per_image_stats_manifest_sha256": sch.sha256_file(manifest_path),
        "generation_nonce": "deadbeef" * 4,
    }
    sch.write_json_atomically(checkpoints_dir / "checkpoint.json", checkpoint_fields, label="checkpoint")
    with pytest.raises(pf.DfpPhase1aPreflightError):
        pf.run_phase_c(phase_a=phase_a, mode="resume", checkpoints_dir=checkpoints_dir)


# ---------------------------------------------------------------------------
# Section 11: fresh-path protection.
# ---------------------------------------------------------------------------


def test_fresh_mode_refuses_over_complete_checkpoint(tmp_path, phase_a, coco_selected_ids):
    ids, digest = coco_selected_ids
    d = _build_v2_checkpoint(tmp_path / "fresh_refuse", selected_ids=ids, selected_ids_digest=digest, n=len(ids), complete=True)
    with pytest.raises(pf.DfpPhase1aPreflightError, match="checkpoint already exists"):
        pf.run_phase_c(phase_a=phase_a, mode="fresh", checkpoints_dir=d)


def test_fresh_mode_with_allow_overwrite_does_not_silently_resume_from_zero_alongside_complete_checkpoint(tmp_path, phase_a, coco_selected_ids):
    """--allow-fresh-overwrite is explicit authorization to start over --
    it must not be reachable through this repair's new logic in any way
    that pretends 0 remaining images coexist with the untouched complete
    checkpoint; the checkpoint itself is left completely alone (never
    deleted/replaced) by run_phase_c, which only decides an action."""
    ids, digest = coco_selected_ids
    d = _build_v2_checkpoint(tmp_path / "fresh_overwrite", selected_ids=ids, selected_ids_digest=digest, n=len(ids), complete=True)
    before = (d / "checkpoint.json").read_bytes()
    result = pf.run_phase_c(phase_a=phase_a, mode="fresh", checkpoints_dir=d, allow_fresh_overwrite=True)
    assert result.mode == "fresh"
    assert result.execution_action == pf.EXECUTION_ACTION_PROCESS
    assert result.next_dataset_index == 0
    # The checkpoint on disk is completely untouched by Phase C -- only
    # execute_run's own fresh-mode per-image loop could ever act on this
    # PhaseCResult, and that is unaffected by this repair.
    assert (d / "checkpoint.json").read_bytes() == before


# ---------------------------------------------------------------------------
# Section 13: real CLI preflight regression.
# ---------------------------------------------------------------------------


def test_real_cli_preflight_reaches_ready_to_finalize(tmp_path, coco_selected_ids, monkeypatch):
    ids, digest = coco_selected_ids
    d = _build_v2_checkpoint(tmp_path / "cli_ready", selected_ids=ids, selected_ids_digest=digest, n=len(ids), complete=True)
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        rc = cli.main(["--repo-root", str(ROOT), "preflight", "--run", "coco_stuff_500", "--mode", "resume", "--output-root", str(d)])
    assert rc == 0
    assert "READY_TO_FINALIZE" in stdout_buf.getvalue()
    assert "execution_action=finalize_only" in stderr_buf.getvalue()
    assert "images_to_process=0" in stderr_buf.getvalue()
    assert "Traceback" not in stderr_buf.getvalue()


def test_real_cli_preflight_malformed_variant_exits_2_no_false_ready(tmp_path, coco_selected_ids):
    ids, digest = coco_selected_ids
    d = _build_v2_checkpoint(tmp_path / "cli_bad", selected_ids=ids, selected_ids_digest=digest, n=len(ids), complete=True, corrupt_field={"next_dataset_index": 499})
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        try:
            rc = cli.main(["--repo-root", str(ROOT), "preflight", "--run", "coco_stuff_500", "--mode", "resume", "--output-root", str(d)])
        except SystemExit as e:
            rc = e.code
    assert rc == 2
    assert "READY" not in stdout_buf.getvalue()
    assert "Traceback" not in stderr_buf.getvalue()
    assert stderr_buf.getvalue().startswith("DFP PHASE 1A FAIL [preflight]:")


# ---------------------------------------------------------------------------
# Section 14: execute-path reachability proof via the REAL, unstubbed
# CLI preflight (the exact thing that was broken before this repair).
# ---------------------------------------------------------------------------


def test_real_cli_execute_resume_reaches_finalize_only_with_everything_poisoned(tmp_path, coco_selected_ids):
    ids, digest = coco_selected_ids
    d = _build_v2_checkpoint(tmp_path / "execute_reach", selected_ids=ids, selected_ids_digest=digest, n=len(ids), complete=True)
    source_ckpt = (d / "checkpoint.json").read_bytes()

    import diagnostics.run_k11_k12_stability as rk
    import src.dfp_phase1a_dataset_view as dv
    poisoned = {}

    def poison(name):
        def _raise(*a, **kw):
            poisoned[name] = poisoned.get(name, 0) + 1
            raise AssertionError(f"POISON: {name} must never be called on the finalize-only recovery path")
        return _raise

    install_calls = []
    real_install = report.install_result_bundle
    def spying_install(**kw):
        install_calls.append(kw)
        return real_install(**kw)

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(rk, "_build_inference", poison("_build_inference")))
        stack.enter_context(mock.patch.object(rk, "_extract_prepared_image", poison("_extract_prepared_image")))
        stack.enter_context(mock.patch.object(dv, "build_manifest_execution_view", poison("build_manifest_execution_view")))
        stack.enter_context(mock.patch.object(ex, "_process_one_image", poison("_process_one_image")))
        stack.enter_context(mock.patch.object(ex, "_write_checkpoint", poison("_write_checkpoint")))
        stack.enter_context(mock.patch.object(ex, "_V2Accumulator", poison("_V2Accumulator")))
        stack.enter_context(mock.patch("src.dfp_phase1a_report.install_result_bundle", spying_install))
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            rc = cli.main([
                "--repo-root", str(ROOT), "execute", "--run", "coco_stuff_500", "--mode", "resume",
                "--device", "cpu", "--output-root", str(d),
            ])

    assert rc == 0
    assert "COMPLETE" in stdout_buf.getvalue()
    assert poisoned == {}
    assert len(install_calls) == 1
    assert (d / "checkpoint.json").read_bytes() == source_ckpt
    dest = ex._resolve_result_bundle_dir(d)
    assert install_calls[0]["destination_dir"] == dest
    verified = report.verify_result_bundle(dest)
    assert verified["result"]["final"] is True


# ---------------------------------------------------------------------------
# resumable=False meaning is unchanged.
# ---------------------------------------------------------------------------


def test_validate_resume_state_resumable_still_false_for_complete_checkpoint(tmp_path, coco_selected_ids):
    ids, digest = coco_selected_ids
    d = _build_v2_checkpoint(tmp_path / "resumable_check", selected_ids=ids, selected_ids_digest=digest, n=len(ids), complete=True)
    result = ckpt.validate_resume_state(checkpoints_dir=d, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert result["resumable"] is False


def test_resume_dataset_index_still_rejects_complete_checkpoint():
    doc = {"complete": True, "next_dataset_index": 5}
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="already complete"):
        ckpt.resume_dataset_index(doc)


# ---------------------------------------------------------------------------
# Production checkpoint (job 21063877), read-only, gated.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.environ.get("COCO_STUFF_REAL_DATA_ROOT") and os.environ.get("DFP_PHASE1A_OUTPUT_ROOT")),
    reason="requires COCO_STUFF_REAL_DATA_ROOT and DFP_PHASE1A_OUTPUT_ROOT",
)
def test_real_production_checkpoint_reaches_ready_to_finalize():
    """Read-only: the actual job-21063877 checkpoint must now reach
    finalize_only readiness. Never installs a result, never mutates the
    production output root. The production path is taken ONLY from
    DFP_PHASE1A_OUTPUT_ROOT (never hardcoded here) -- matching the exact
    env var this whole verification task's own Section 5 setup uses."""
    production_dir = Path(os.environ["DFP_PHASE1A_OUTPUT_ROOT"]) / "coco_stuff_500"
    if not (production_dir / "checkpoint.json").exists():
        pytest.skip("production checkpoint not present in this environment")
    before = (production_dir / "checkpoint.json").read_bytes()
    result_dest = ex._resolve_result_bundle_dir(production_dir)
    assert not result_dest.exists()

    ctx = pf.run_preflight(repo_root=ROOT, run_mode="coco_stuff_500", mode="resume", output_root=production_dir, check_git=True)
    assert ctx.execution_action == pf.EXECUTION_ACTION_FINALIZE_ONLY
    assert ctx.next_dataset_index == 500
    assert ctx.expected_count == 500

    assert (production_dir / "checkpoint.json").read_bytes() == before
    assert not result_dest.exists()
