"""Regression tests for the DFP Phase 1A result-bundle v1/v2 dispatch
repair (src/dfp_phase1a_report.py's ``_load_per_image_stats_npz_dispatch``).

Covers: valid v1 and v2 install/verify round trips, v2 self-verification
before atomic installation, every adversarial mislabeling/malformed-
content case listed in the repair task, the standalone verifier CLI and
the unified CLI both exiting 2 with a concise diagnostic and no
traceback, the NPZ being loaded exactly once per physical file per
verification path, no temporary residue on failure, and an existing
valid bundle surviving a failed install attempt at the same destination.

CPU-only throughout. No real dataset, no model construction, no CUDA
initialization, no execution-driver code exercised (none exists to
exercise): v2 artifacts are hand-constructed directly, since no v2
writer is in scope for this repair."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import src.dfp_phase1a_artifact_schema as sch
import src.dfp_phase1a_checkpoint as ckpt
import src.dfp_phase1a_report as report

ROOT = Path(__file__).resolve().parent.parent
CLASS_COUNT = 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def schema_summary():
    return sch.validate_static_configuration(repo_root=ROOT)


@pytest.fixture(scope="module")
def artifact_identity():
    return sch.load_identity(repo_root=ROOT)


@pytest.fixture(scope="module")
def coco_selected_ids():
    manifest = json.loads((ROOT / "evaluation_subsets/dfp_phase1a_coco_stuff_500.json").read_text())
    return manifest["selected_ids"], manifest["selected_ids_digest"]


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


def _make_v1_fields(*, schema_summary, artifact_identity, selected_ids, selected_ids_digest, n: int, complete: bool = False) -> dict:
    import datetime

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


def _install_v1(tmp_path, *, schema_summary, artifact_identity, selected_ids, selected_ids_digest, n: int, complete: bool = False, seed: int = 0):
    fields = _make_v1_fields(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=selected_ids_digest, n=n, complete=complete)
    arrays = _make_v1_arrays(n, seed=seed)
    checkpoints_dir = tmp_path / "checkpoints_v1"
    ckpt.generate_and_install_checkpoint(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=fields, arrays=arrays, image_ids=selected_ids[:n])
    return checkpoints_dir


def _make_v2_arrays(n: int, class_count: int = CLASS_COUNT) -> dict[str, np.ndarray]:
    """A minimal, fully self-consistent v2 array set: all mass on class
    0, one component per image covering the whole (10-pixel) image, no
    dispute/persistence/reversion activity -- sufficient to exercise the
    v2 NPZ loader end to end without needing the full richness of the
    schema's own dedicated test suite."""
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


def _install_v2(tmp_path, *, schema_summary, artifact_identity, selected_ids, selected_ids_digest, n: int, complete: bool = False,
                 class_count: int = CLASS_COUNT, corrupt_npz_member: str | None = None, wrong_npz_class_count: int | None = None,
                 mislabel_schema: tuple[str, int] | None = None, dirname: str = "checkpoints_v2"):
    """Hand-build a complete, schema-valid v2 checkpoint + per-image-stats
    manifest + NPZ bundle -- there is no v2 writer, by design (this
    repair does not implement the execution driver), so every v2 fixture
    in this file is constructed directly against the frozen schema
    authority, exactly as a future driver would eventually do."""
    import datetime

    checkpoints_dir = tmp_path / dirname
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    arrays = _make_v2_arrays(n, class_count=wrong_npz_class_count or class_count)
    if corrupt_npz_member is not None:
        del arrays[corrupt_npz_member]
    npz_filename = f"dfp_phase1a_per_image_stats_v2_{n}.npz"
    npz_path = checkpoints_dir / npz_filename
    np.savez(npz_path, allow_pickle=False, **arrays)
    npz_sha256 = sch.sha256_file(npz_path)

    schema_name, schema_version = mislabel_schema if mislabel_schema is not None else (sch.PER_IMAGE_STATS_SCHEMA_NAME_V2, sch.PER_IMAGE_STATS_SCHEMA_VERSION_V2)
    stats_manifest = {
        "schema": schema_name, "schema_version": schema_version,
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
        "completed_image_ids": list(selected_ids[:n]), "images_completed_count": n, "total_windows_processed": n * 4,
        "complete": complete, "final_result_produced": False, "created_at_utc": now, "updated_at_utc": now,
        "per_image_stats_manifest_relative_path": "stats_manifest.json", "per_image_stats_manifest_sha256": manifest_sha256,
        "generation_nonce": "deadbeef" * 4,
    }
    sch.write_json_atomically(checkpoints_dir / "checkpoint.json", checkpoint_fields, label="checkpoint")
    return checkpoints_dir


# ---------------------------------------------------------------------------
# 1/2. Valid v1 and v2 install + verify round trips
# ---------------------------------------------------------------------------


def test_v1_install_and_verify_roundtrip_still_works(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v1(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3, complete=True)
    dest = tmp_path / "bundle_v1"
    report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    verified = report.verify_result_bundle(dest)
    assert verified["result"]["complete"] is True
    assert len(verified["arrays"]) == len(sch.ALL_NPZ_MEMBER_NAMES)


def test_v2_install_and_verify_roundtrip(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    n = 4
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=n, complete=True)
    dest = tmp_path / "bundle_v2"
    report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    verified = report.verify_result_bundle(dest)
    assert verified["result"]["complete"] is True
    assert verified["result"]["final"] is True
    assert set(verified["arrays"]) == sch.ALL_NPZ_MEMBER_NAMES_V2
    assert verified["arrays"]["dataset_indices"].shape == (n,)


# ---------------------------------------------------------------------------
# 3. v2 self-verification before atomic installation
# ---------------------------------------------------------------------------


def test_v2_staged_self_verification_catches_a_copy_corruption(tmp_path, schema_summary, artifact_identity, coco_selected_ids, monkeypatch):
    """Simulate a corrupted staging copy (e.g. a bit-flip during
    shutil.copyfile) and confirm install_result_bundle's own staged
    self-verification -- which now correctly re-loads the staged NPZ
    through the SAME version-aware dispatch as everywhere else -- catches
    it before any atomic rename, leaving no partial result and no
    residue."""
    selected_ids, digest = coco_selected_ids
    n = 2
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=n, complete=True, dirname="checkpoints_v2_corrupt_copy")
    dest = tmp_path / "bundle_v2_corrupt_copy"

    import shutil as _shutil
    original_copyfile = _shutil.copyfile

    def _corrupting_copyfile(src, dst, *a, **kw):
        result = original_copyfile(src, dst, *a, **kw)
        if str(src).endswith(".npz"):
            with open(dst, "r+b") as f:
                f.seek(0)
                byte = f.read(1)
                f.seek(0)
                f.write(bytes([byte[0] ^ 0xFF]))
        return result

    monkeypatch.setattr(report.shutil, "copyfile", _corrupting_copyfile)
    with pytest.raises(report.DfpPhase1aReportError, match="hash-match"):
        report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert not dest.exists()
    assert not any("staging" in p.name for p in dest.parent.iterdir())


# ---------------------------------------------------------------------------
# 4-10. Adversarial mislabeling / malformed-content matrix
# ---------------------------------------------------------------------------


def test_malformed_v2_member_set_rejected(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, complete=True, corrupt_npz_member="reversion_perimeter", dirname="checkpoints_v2_malformed")
    dest = tmp_path / "bundle_v2_malformed"
    with pytest.raises(report.DfpPhase1aReportError, match="missing="):
        report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert not dest.exists()


def test_v1_npz_mislabeled_v2_rejected(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """A manifest that CLAIMS the v2 schema name+version but whose NPZ
    actually has the v1 member set -- the closed-member-set check inside
    load_per_image_stats_npz_v2 must catch this; the manifest-level
    check alone cannot see the NPZ's true content."""
    selected_ids, digest = coco_selected_ids
    n = 2
    checkpoints_dir = tmp_path / "checkpoints_v1_shaped_npz_v2_manifest"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    v1_arrays = _make_v1_arrays(n, seed=7)
    npz_path = checkpoints_dir / "v1_shaped.npz"
    np.savez(npz_path, allow_pickle=False, **v1_arrays)
    npz_sha256 = sch.sha256_file(npz_path)
    stats_manifest = {
        "schema": sch.PER_IMAGE_STATS_SCHEMA_NAME_V2, "schema_version": sch.PER_IMAGE_STATS_SCHEMA_VERSION_V2,
        "npz_filename": "v1_shaped.npz", "npz_sha256": npz_sha256, "run_mode": "coco_stuff_500",
        "class_count": CLASS_COUNT, "class_names_digest": artifact_identity["dataset_binding"]["coco_stuff_class_names_digest"],
        "image_count": n, "image_ids": list(selected_ids[:n]),
    }
    with pytest.raises(report.DfpPhase1aReportError, match="missing="):
        report._load_per_image_stats_npz_dispatch(npz_path, stats_manifest)


def test_v2_npz_mislabeled_v1_rejected(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """The reverse: a manifest claiming v1, but an NPZ with the v2
    member set."""
    selected_ids, digest = coco_selected_ids
    n = 2
    checkpoints_dir = tmp_path / "checkpoints_v2_shaped_npz_v1_manifest"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    v2_arrays = _make_v2_arrays(n)
    npz_path = checkpoints_dir / "v2_shaped.npz"
    np.savez(npz_path, allow_pickle=False, **v2_arrays)
    npz_sha256 = sch.sha256_file(npz_path)
    stats_manifest = {
        "schema": sch.PER_IMAGE_STATS_SCHEMA_NAME, "schema_version": sch.PER_IMAGE_STATS_SCHEMA_VERSION,
        "npz_filename": "v2_shaped.npz", "npz_sha256": npz_sha256, "run_mode": "coco_stuff_500",
        "class_count": CLASS_COUNT, "class_names_digest": artifact_identity["dataset_binding"]["coco_stuff_class_names_digest"],
        "image_count": n, "image_ids": list(selected_ids[:n]),
    }
    with pytest.raises(report.DfpPhase1aReportError, match="unexpected="):
        report._load_per_image_stats_npz_dispatch(npz_path, stats_manifest)


def test_unknown_schema_name_version_rejected_never_guessed(tmp_path, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    n = 1
    checkpoints_dir = tmp_path / "checkpoints_unknown_schema"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    v2_arrays = _make_v2_arrays(n)
    npz_path = checkpoints_dir / "a.npz"
    np.savez(npz_path, allow_pickle=False, **v2_arrays)
    npz_sha256 = sch.sha256_file(npz_path)
    for schema_name, schema_version in (
        ("talk2dino-dfp-phase1a-per-image-stats-v3", 1),  # unknown name
        (sch.PER_IMAGE_STATS_SCHEMA_NAME_V2, 2),           # known name, wrong version number
        ("", 1),                                            # empty name
    ):
        stats_manifest = {
            "schema": schema_name, "schema_version": schema_version,
            "npz_filename": "a.npz", "npz_sha256": npz_sha256, "run_mode": "coco_stuff_500",
            "class_count": CLASS_COUNT, "class_names_digest": artifact_identity["dataset_binding"]["coco_stuff_class_names_digest"],
            "image_count": n, "image_ids": list(selected_ids[:n]),
        }
        with pytest.raises((report.DfpPhase1aReportError, ckpt.DfpPhase1aCheckpointError), match="does not match any known schema"):
            report._load_per_image_stats_npz_dispatch(npz_path, stats_manifest)


def test_wrong_v2_class_count_rejected(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(
        tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids,
        selected_ids_digest=digest, n=2, complete=True, class_count=CLASS_COUNT, wrong_npz_class_count=CLASS_COUNT + 1,
        dirname="checkpoints_v2_wrong_class_count",
    )
    dest = tmp_path / "bundle_v2_wrong_class_count"
    with pytest.raises(report.DfpPhase1aReportError):
        report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert not dest.exists()


def test_corrupted_v2_digest_rejected(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, complete=True, dirname="checkpoints_v2_corrupt_digest")
    npz_path = next(checkpoints_dir.glob("*.npz"))
    with open(npz_path, "r+b") as f:
        f.seek(0)
        byte = f.read(1)
        f.seek(0)
        f.write(bytes([byte[0] ^ 0xFF]))
    dest = tmp_path / "bundle_v2_corrupt_digest"
    with pytest.raises(report.DfpPhase1aReportError, match="SHA256 mismatch"):
        report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert not dest.exists()


def test_missing_v2_npz_rejected(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, complete=True, dirname="checkpoints_v2_missing_npz")
    npz_path = next(checkpoints_dir.glob("*.npz"))
    npz_path.unlink()
    dest = tmp_path / "bundle_v2_missing_npz"
    with pytest.raises(report.DfpPhase1aReportError):
        report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert not dest.exists()


# ---------------------------------------------------------------------------
# 11. Direct API domain error (never a raw underlying exception type)
# ---------------------------------------------------------------------------


def test_v2_failures_always_surface_as_report_domain_error(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, complete=True, corrupt_npz_member="component_count", dirname="checkpoints_v2_domain_error")
    dest = tmp_path / "bundle_v2_domain_error"
    try:
        report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
        pytest.fail("expected DfpPhase1aReportError")
    except report.DfpPhase1aReportError:
        pass
    except ckpt.DfpPhase1aCheckpointError:
        pytest.fail("a checkpoint-domain error leaked unwrapped through install_result_bundle's public API")


# ---------------------------------------------------------------------------
# 12. Both CLIs exit 2 with no traceback (valid + malformed v2 bundles)
# ---------------------------------------------------------------------------


def _run_cli(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, script, *args], cwd=ROOT, capture_output=True, text=True, timeout=60,
    )


def test_standalone_verifier_cli_accepts_a_valid_v2_bundle(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, complete=True, dirname="checkpoints_v2_cli_ok")
    dest = tmp_path / "bundle_v2_cli_ok"
    report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    proc = _run_cli("verify_dfp_phase1a_artifacts.py", "verify-result", "--bundle", str(dest))
    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr


def test_standalone_verifier_cli_rejects_a_malformed_v2_bundle_exit2_no_traceback(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, complete=True, dirname="checkpoints_v2_cli_bad")
    dest = tmp_path / "bundle_v2_cli_bad"
    report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    npz_files = list(dest.glob("*.npz"))
    assert len(npz_files) == 1
    with open(npz_files[0], "r+b") as f:
        f.seek(0)
        byte = f.read(1)
        f.seek(0)
        f.write(bytes([byte[0] ^ 0xFF]))
    proc = _run_cli("verify_dfp_phase1a_artifacts.py", "verify-result", "--bundle", str(dest))
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr
    assert "FAIL" in proc.stderr


def test_unified_cli_accepts_a_valid_v2_bundle(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, complete=True, dirname="checkpoints_v2_unified_cli_ok")
    dest = tmp_path / "bundle_v2_unified_cli_ok"
    report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    proc = _run_cli("diagnostics/run_dfp_phase1a.py", "verify-result", "--bundle", str(dest))
    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr


def test_unified_cli_rejects_a_malformed_v2_bundle_exit2_no_traceback(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, complete=True, dirname="checkpoints_v2_unified_cli_bad")
    dest = tmp_path / "bundle_v2_unified_cli_bad"
    report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    (dest / "stats_manifest.json").unlink()  # remove a required bundle member
    proc = _run_cli("diagnostics/run_dfp_phase1a.py", "verify-result", "--bundle", str(dest))
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr


# ---------------------------------------------------------------------------
# 13. NPZ loaded exactly once per physical file per verification path
# ---------------------------------------------------------------------------


def test_npz_loaded_exactly_once_per_file_in_verify(tmp_path, schema_summary, artifact_identity, coco_selected_ids, monkeypatch):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, complete=True, dirname="checkpoints_v2_count_verify")
    dest = tmp_path / "bundle_v2_count_verify"
    report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)

    calls = {"n": 0}
    original = report.load_per_image_stats_npz_v2

    def _counting(*a, **kw):
        calls["n"] += 1
        return original(*a, **kw)

    monkeypatch.setattr(report, "load_per_image_stats_npz_v2", _counting)
    report.verify_result_bundle(dest)
    assert calls["n"] == 1


def test_npz_loaded_exactly_once_per_file_in_install(tmp_path, schema_summary, artifact_identity, coco_selected_ids, monkeypatch):
    """install_result_bundle touches TWO distinct physical NPZ files (the
    source and its staged copy) -- each must be loaded exactly once, for
    a total of exactly two calls, never re-loading the same file twice
    and never skipping either file's own verification."""
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, complete=True, dirname="checkpoints_v2_count_install")
    dest = tmp_path / "bundle_v2_count_install"

    calls = []
    original = report.load_per_image_stats_npz_v2

    def _counting(path, *a, **kw):
        calls.append(Path(path).resolve())
        return original(path, *a, **kw)

    monkeypatch.setattr(report, "load_per_image_stats_npz_v2", _counting)
    report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert len(calls) == 2
    assert len(set(calls)) == 2  # two DISTINCT physical files, each loaded exactly once


# ---------------------------------------------------------------------------
# 14. No temporary residue on failure (v2-specific)
# ---------------------------------------------------------------------------


def test_v2_failed_install_leaves_no_staging_residue(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, complete=True, corrupt_npz_member="harmful_pixel_count", dirname="checkpoints_v2_residue")
    dest = tmp_path / "bundle_v2_residue"
    with pytest.raises(report.DfpPhase1aReportError):
        report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert not dest.exists()
    assert not any("staging" in p.name for p in dest.parent.iterdir())


# ---------------------------------------------------------------------------
# 15. Existing valid bundle preserved after a failed replacement attempt
# ---------------------------------------------------------------------------


def test_existing_valid_v1_bundle_survives_a_rejected_v2_reinstall_attempt(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    v1_checkpoints_dir = _install_v1(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, complete=True)
    dest = tmp_path / "bundle_shared_dest"
    report.install_result_bundle(destination_dir=dest, checkpoints_dir=v1_checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    before = report.verify_result_bundle(dest)

    v2_checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, complete=True, dirname="checkpoints_v2_for_reinstall_attempt")
    with pytest.raises(report.DfpPhase1aReportError, match="already exists"):
        report.install_result_bundle(destination_dir=dest, checkpoints_dir=v2_checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)

    after = report.verify_result_bundle(dest)
    assert before["result"] == after["result"]
    assert set(after["arrays"]) == sch.ALL_NPZ_MEMBER_NAMES  # still the original v1 bundle, untouched
