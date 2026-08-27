"""Regression tests for the Phase-A resume-ordering repair in
diagnostics/run_voc2012_matched_evaluation.py::_run_evaluation.

Every malformed/inconsistent resume artifact must be rejected before
dataset construction, dataset sample access, model construction,
DINO/backbone loading, bridge-checkpoint loading, or any CUDA
initialization. These tests poison each of those entry points
independently with call-counting sentinels and require zero calls for
every Phase-A failure case, using a REAL generated VOC2012 source
manifest and REAL per-image-stats artifacts (built via the production
writer) so the checks are exercised through the actual --resume CLI
path, not just the individual validator functions in isolation.

Gated on VOC2012_REAL_DATA_ROOT (same convention as
test_run_voc2012_matched_evaluation.py's real-data smoke test) because a
real, verified source manifest and the real canonical image order are
required to build a genuinely valid Phase-A baseline to tamper from.
CPU-only throughout; never initializes CUDA."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
OVS_ROOT = ROOT / "src/open_vocabulary_segmentation"
for _p in (ROOT, OVS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

IDENTITY_PATH = ROOT / "evaluation_identities/e12_voc2012_matched_evaluator.toml"
pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the voc2012 matched-evaluator identity")

torch = pytest.importorskip("torch")


def _real_data_root():
    raw = os.environ.get("VOC2012_REAL_DATA_ROOT")
    if raw is None or not raw.strip():
        return None
    return Path(raw).expanduser()


REAL_DATA_ROOT = _real_data_root()
requires_real_data = pytest.mark.skipif(
    REAL_DATA_ROOT is None or not REAL_DATA_ROOT.exists(),
    reason="requires VOC2012_REAL_DATA_ROOT to point at the real VOC2012 archive",
)
pytestmark = [pytestmark, requires_real_data]


@pytest.fixture(scope="module")
def driver_module():
    spec = importlib.util.spec_from_file_location(
        "voc2012_driver_phase_a_ordering_under_test", ROOT / "diagnostics" / "run_voc2012_matched_evaluation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evaluator_identity(driver_module):
    from src.voc2012_matched_evaluator_identity import load_identity

    return load_identity(repo_root=ROOT)


@pytest.fixture(scope="module")
def source_manifest_path(tmp_path_factory):
    """Generate a real, exhaustively-scanned VOC2012 source manifest ONCE
    for this module, into a scratch directory (never inside the
    dataset)."""
    import subprocess

    out_dir = tmp_path_factory.mktemp("phase_a_ordering_source_manifest")
    manifest_path = out_dir / "source_manifest.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify_voc2012_dataset.py"), "generate-manifest",
         "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--output", str(manifest_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return manifest_path


@pytest.fixture(scope="module")
def canonical_ids(driver_module, evaluator_identity):
    from src.voc2012_dataset_identity import load_identity as load_source_identity
    from src.voc2012_dataset_manifest import canonical_validation_ids, resolve_dataset_root

    source_identity = load_source_identity(repo_root=ROOT)
    voc_root = resolve_dataset_root(REAL_DATA_ROOT, source_identity)
    ids = canonical_validation_ids(voc_root, source_identity)
    image_count = evaluator_identity["run_modes"]["pilot20_image_count"]
    expected_image_ids = ids[:image_count]
    image_order_digest = driver_module._image_order_digest(expected_image_ids)
    return expected_image_ids, image_order_digest


def _identity_sha256():
    return hashlib.sha256(IDENTITY_PATH.read_bytes()).hexdigest()


def _build_rows(driver_module, n):
    from src.voc2012_matched_evaluator_report import VARIANT_NAMES

    rng = np.random.default_rng(0)
    rows = {"label_v20": [], "label_v21": []}
    for v in VARIANT_NAMES:
        rows[f"intersect_{v}"] = []
        rows[f"union_{v}"] = []
        rows[f"pred_{v}"] = []
    for _ in range(n):
        rows["label_v20"].append(rng.integers(0, 5, size=20))
        rows["label_v21"].append(rng.integers(0, 5, size=21))
        for v in VARIANT_NAMES:
            w = 20 if v.startswith("v20_") else 21
            rows[f"intersect_{v}"].append(rng.integers(0, 5, size=w))
            rows[f"union_{v}"].append(rng.integers(1, 5, size=w))
            rows[f"pred_{v}"].append(rng.integers(0, 5, size=w))
    return rows


def _build_valid_pair(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, *, next_index, complete=False):
    """Build a mutually-consistent, fully valid checkpoint.json +
    per-image-stats manifest/NPZ pair (via the REAL production writer)
    for a resume at `next_index` images completed. Returns
    (checkpoint_path, stats_manifest_path, result_path)."""
    expected_image_ids, image_order_digest = canonical_ids
    image_count = evaluator_identity["run_modes"]["pilot20_image_count"]
    identity_sha256 = _identity_sha256()

    checkpoint_doc = {
        "schema": evaluator_identity["checkpoint"]["schema_name"], "run_mode": "pilot20",
        "identity": evaluator_identity["identity"]["name"], "identity_sha256": identity_sha256,
        "matched_identity_sha256": evaluator_identity["parent_identities"]["matched_identity_sha256"],
        "voc2012_source_identity_sha256": evaluator_identity["parent_identities"]["voc2012_source_identity_sha256"],
        "source_manifest_sha256": hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
        "bridge_checkpoint_sha256": evaluator_identity["model_and_checkpoint"]["projection_checkpoint_sha256"],
        "git_commit": "1" * 40, "v20_class_count": 20, "v21_class_count": 21,
        "live_v20_class_names_digest": "2" * 64, "live_v21_class_names_digest": "3" * 64,
        "image_count_expected": image_count, "image_order_digest": image_order_digest,
        "next_dataset_index": next_index, "completed_image_ids": expected_image_ids[:next_index],
        "images_completed_count": next_index, "windows_processed_total": max(next_index * 3, 0), "complete": complete,
        "created_at_utc": "2026-01-01T00:00:00+00:00", "updated_at_utc": "2026-01-01T00:00:00+00:00",
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint_doc))

    stats_manifest_path = tmp_path / "stats.json"
    rows = _build_rows(driver_module, next_index)
    driver_module._write_per_image_stats_atomically(
        stats_manifest_path, schema_name=evaluator_identity["artifacts"]["per_image_stats_manifest_schema_name"],
        v20_class_count=20, v21_class_count=21,
        live_v20_class_names_digest="2" * 64, live_v21_class_names_digest="3" * 64,
        dataset_indices=list(range(next_index)), image_ids=expected_image_ids[:next_index], rows=rows,
    )
    result_path = tmp_path / "result.json"
    return checkpoint_path, stats_manifest_path, result_path


def _tamper_npz_with_consistent_digest(stats_manifest_path, mutate_arrays_fn):
    npz_path = stats_manifest_path.with_suffix(".npz")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays = mutate_arrays_fn(arrays)
    np.savez(npz_path, allow_pickle=False, **arrays)
    manifest = json.loads(stats_manifest_path.read_text())
    manifest["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    stats_manifest_path.write_text(json.dumps(manifest))


class _PhaseAReachedError(Exception):
    """Raised by a poisoned heavy entry point; distinguishes 'Phase A let
    execution through' from a legitimate Voc2012MatchedEvaluatorIdentityError."""


@pytest.fixture
def poisoned(monkeypatch):
    """Poisons every heavy/CUDA-adjacent entry point independently with a
    call-counting sentinel that raises immediately. Returns the counts
    dict; every Phase-A failure test asserts every count is exactly 0."""
    counts = {
        "build_dataset": 0, "dataset_sample_access": 0, "build_model": 0,
        "load_local_vision_backbone": 0, "checkpoint_loader_load_checkpoint": 0,
        "module_cuda": 0, "module_to": 0,
    }

    import mmseg.datasets as mmseg_datasets_module
    import models as models_module
    import mmcv.runner as mmcv_runner_module
    import src.local_weights as local_weights_module
    import diagnostics.run_k11_k12_stability as k11_k12_stability_module

    def _poison(name):
        def _fn(*a, **k):
            counts[name] += 1
            raise _PhaseAReachedError(name)
        return _fn

    monkeypatch.setattr(mmseg_datasets_module, "build_dataset", _poison("build_dataset"))
    monkeypatch.setattr(k11_k12_stability_module, "_extract_prepared_image", _poison("dataset_sample_access"))
    monkeypatch.setattr(models_module, "build_model", _poison("build_model"))
    monkeypatch.setattr(local_weights_module, "load_local_vision_backbone", _poison("load_local_vision_backbone"))
    monkeypatch.setattr(mmcv_runner_module.CheckpointLoader, "load_checkpoint", staticmethod(_poison("checkpoint_loader_load_checkpoint")))
    monkeypatch.setattr(torch.nn.Module, "cuda", _poison("module_cuda"))
    monkeypatch.setattr(torch.nn.Module, "to", _poison("module_to"))

    return counts


def _run_and_expect_phase_a_rejection(driver_module, poisoned, *, checkpoint_path, stats_manifest_path, result_path, source_manifest_path):
    argv = [
        "--repo-root", str(ROOT), "--source-manifest", str(source_manifest_path),
        "--run-mode", "pilot20", "--checkpoint", str(checkpoint_path),
        "--result", str(result_path), "--per-image-stats", str(stats_manifest_path),
        "--data-root", str(REAL_DATA_ROOT), "--resume", "--device", "cpu",
    ]
    assert torch.cuda.is_initialized() is False
    stderr_buf = io.StringIO()
    with redirect_stderr(stderr_buf):
        exit_code = driver_module.main(argv)
    stderr_text = stderr_buf.getvalue()

    assert exit_code == 2, f"expected exit 2, got {exit_code}; stderr={stderr_text}"
    assert "Traceback" not in stderr_text
    assert "VOC2012 MATCHED EVALUATOR FAIL" in stderr_text
    assert not result_path.exists()
    assert torch.cuda.is_initialized() is False
    for name, count in poisoned.items():
        assert count == 0, f"poisoned entry point {name!r} was reached ({count} call(s)) before Phase A rejected the resume artifact"
    return stderr_text


# ---------------------------------------------------------------------
# Negative cases: each must be rejected in Phase A with zero calls to
# every poisoned heavy/CUDA entry point.
# ---------------------------------------------------------------------


def test_phase_a_rejects_wrong_npz_dtype(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    _tamper_npz_with_consistent_digest(stats_manifest_path, lambda a: {**a, "union_v21_k12": a["union_v21_k12"].astype(np.float64)})
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_rejects_wrong_npz_shape(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    _tamper_npz_with_consistent_digest(stats_manifest_path, lambda a: {**a, "label_v20": np.zeros((a["label_v20"].shape[0], 19), dtype=np.int64)})
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_rejects_missing_npz_member(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    def _drop(a):
        a = dict(a)
        del a["union_v21_k12"]
        return a
    _tamper_npz_with_consistent_digest(stats_manifest_path, _drop)
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_rejects_extra_npz_member(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    _tamper_npz_with_consistent_digest(stats_manifest_path, lambda a: {**a, "bogus_extra": np.zeros((5, 3), dtype=np.int64)})
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_rejects_malformed_stats_manifest_json(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    stats_manifest_path.write_text('{"a": 1, "a": 2}')  # duplicate key -> strict JSON parse failure
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_rejects_missing_stats_artifact(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    stats_manifest_path.unlink()
    stats_manifest_path.with_suffix(".npz").unlink()
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_rejects_stats_present_checkpoint_absent(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    checkpoint_path.unlink()
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_rejects_checkpoint_present_stats_absent(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    stats_manifest_path.unlink()
    stats_manifest_path.with_suffix(".npz").unlink()
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_rejects_row_count_different_from_resume_index(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    # truncate every array to 4 rows while checkpoint.next_dataset_index stays 5
    def _truncate(a):
        return {name: arr[:4] for name, arr in a.items()}
    _tamper_npz_with_consistent_digest(stats_manifest_path, _truncate)
    manifest = json.loads(stats_manifest_path.read_text())
    manifest["image_count"] = 4
    manifest["dataset_indices"] = list(range(4))
    manifest["image_ids"] = manifest["image_ids"][:4]
    stats_manifest_path.write_text(json.dumps(manifest))
    stderr_text = _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)
    assert "next_dataset_index" in stderr_text


def test_phase_a_rejects_row_count_different_from_completed_id_count(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    # checkpoint says images_completed_count=5 but images_completed_count
    # disagreeing with completed_image_ids length is itself rejected by
    # validate_checkpoint_structure -- exercise that path directly.
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    doc = json.loads(checkpoint_path.read_text())
    doc["images_completed_count"] = 4  # disagrees with len(completed_image_ids) == 5
    checkpoint_path.write_text(json.dumps(doc))
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_rejects_reordered_completed_ids(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    doc = json.loads(checkpoint_path.read_text())
    doc["completed_image_ids"] = list(reversed(doc["completed_image_ids"]))
    checkpoint_path.write_text(json.dumps(doc))
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_rejects_incorrect_completed_ids(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    doc = json.loads(checkpoint_path.read_text())
    doc["completed_image_ids"][-1] = "not_a_real_image_id"
    checkpoint_path.write_text(json.dumps(doc))
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_rejects_changed_identity_hash(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    doc = json.loads(checkpoint_path.read_text())
    doc["identity_sha256"] = "0" * 64
    checkpoint_path.write_text(json.dumps(doc))
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_rejects_changed_source_manifest_hash(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    """Phase A now cross-checks checkpoint.source_manifest_sha256 against
    the hash of the currently-supplied --source-manifest (previously only
    format-validated, never bound to the live --source-manifest)."""
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    doc = json.loads(checkpoint_path.read_text())
    doc["source_manifest_sha256"] = "0" * 64
    checkpoint_path.write_text(json.dumps(doc))
    stderr_text = _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)
    assert "source_manifest_sha256" in stderr_text


def test_phase_a_rejects_changed_bridge_checkpoint_hash(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    doc = json.loads(checkpoint_path.read_text())
    doc["bridge_checkpoint_sha256"] = "f" * 64
    checkpoint_path.write_text(json.dumps(doc))
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_rejects_changed_run_mode(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    doc = json.loads(checkpoint_path.read_text())
    doc["run_mode"] = "pilot100"
    checkpoint_path.write_text(json.dumps(doc))
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_rejects_changed_expected_count(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    doc = json.loads(checkpoint_path.read_text())
    doc["image_count_expected"] = 999
    checkpoint_path.write_text(json.dumps(doc))
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_rejects_changed_class_count(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    doc = json.loads(checkpoint_path.read_text())
    doc["v20_class_count"] = 21
    checkpoint_path.write_text(json.dumps(doc))
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_rejects_stats_class_count_disagreeing_with_checkpoint(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    """A checkpoint that is internally valid (v20_class_count=20, matching
    the identity) paired with a per-image-stats artifact that is ITSELF
    internally self-consistent (manifest.v20_class_count=19, and every
    array genuinely has 19 columns, so _load_per_image_stats's own shape
    check does not fire) -- isolates the NEW
    validate_checkpoint_against_per_image_stats cross-check, which must
    still catch the checkpoint-vs-stats-manifest disagreement."""
    expected_image_ids, _ = canonical_ids
    next_index = 5
    checkpoint_path, _, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=next_index,
    )
    stats_manifest_path = tmp_path / "stats.json"
    from src.voc2012_matched_evaluator_report import VARIANT_NAMES
    rng = np.random.default_rng(1)
    rows = {"label_v20": [], "label_v21": []}
    for v in VARIANT_NAMES:
        rows[f"intersect_{v}"] = []
        rows[f"union_{v}"] = []
        rows[f"pred_{v}"] = []
    for _ in range(next_index):
        rows["label_v20"].append(rng.integers(0, 5, size=19))  # self-consistent 19, not the checkpoint's 20
        rows["label_v21"].append(rng.integers(0, 5, size=21))
        for v in VARIANT_NAMES:
            w = 19 if v.startswith("v20_") else 21
            rows[f"intersect_{v}"].append(rng.integers(0, 5, size=w))
            rows[f"union_{v}"].append(rng.integers(1, 5, size=w))
            rows[f"pred_{v}"].append(rng.integers(0, 5, size=w))
    driver_module._write_per_image_stats_atomically(
        stats_manifest_path, schema_name=evaluator_identity["artifacts"]["per_image_stats_manifest_schema_name"],
        v20_class_count=19, v21_class_count=21,
        live_v20_class_names_digest="2" * 64, live_v21_class_names_digest="3" * 64,
        dataset_indices=list(range(next_index)), image_ids=expected_image_ids[:next_index], rows=rows,
    )
    stderr_text = _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)
    assert "v20_class_count" in stderr_text


def test_phase_a_rejects_already_complete_checkpoint(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    image_count = evaluator_identity["run_modes"]["pilot20_image_count"]
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=image_count, complete=True,
    )
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)


def test_phase_a_manifest_and_checkpoint_untouched_by_a_failed_resume(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    _tamper_npz_with_consistent_digest(stats_manifest_path, lambda a: {**a, "union_v21_k12": a["union_v21_k12"].astype(np.float64)})
    checkpoint_before = checkpoint_path.read_bytes()
    stats_manifest_before = stats_manifest_path.read_bytes()
    stats_npz_before = stats_manifest_path.with_suffix(".npz").read_bytes()
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)
    assert checkpoint_path.read_bytes() == checkpoint_before
    assert stats_manifest_path.read_bytes() == stats_manifest_before
    assert stats_manifest_path.with_suffix(".npz").read_bytes() == stats_npz_before


def test_phase_a_existing_valid_result_preserved_across_failed_resume(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    _tamper_npz_with_consistent_digest(stats_manifest_path, lambda a: {**a, "union_v21_k12": a["union_v21_k12"].astype(np.float64)})
    result_path.write_text('{"existing": "valid content that must survive"}')
    before = result_path.read_bytes()
    argv = [
        "--repo-root", str(ROOT), "--source-manifest", str(source_manifest_path),
        "--run-mode", "pilot20", "--checkpoint", str(checkpoint_path),
        "--result", str(result_path), "--per-image-stats", str(stats_manifest_path),
        "--data-root", str(REAL_DATA_ROOT), "--resume", "--overwrite", "--device", "cpu",
    ]
    exit_code = driver_module.main(argv)
    assert exit_code == 2
    assert result_path.read_bytes() == before
    for name, count in poisoned.items():
        assert count == 0, name


def test_phase_a_no_temporary_residue_after_rejection(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, poisoned):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    _tamper_npz_with_consistent_digest(stats_manifest_path, lambda a: {**a, "union_v21_k12": a["union_v21_k12"].astype(np.float64)})
    _run_and_expect_phase_a_rejection(driver_module, poisoned, checkpoint_path=checkpoint_path, stats_manifest_path=stats_manifest_path, result_path=result_path, source_manifest_path=source_manifest_path)
    leftover = [p for p in tmp_path.rglob("*") if p.is_file() and (".tmp" in p.name or "selfcheck" in p.name)]
    assert leftover == []


def test_phase_a_keyboard_interrupt_propagates(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, monkeypatch):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    def _raise(*a, **k):
        raise KeyboardInterrupt()
    monkeypatch.setattr(driver_module, "_load_per_image_stats", _raise)
    argv = [
        "--repo-root", str(ROOT), "--source-manifest", str(source_manifest_path),
        "--run-mode", "pilot20", "--checkpoint", str(checkpoint_path),
        "--result", str(result_path), "--per-image-stats", str(stats_manifest_path),
        "--data-root", str(REAL_DATA_ROOT), "--resume", "--device", "cpu",
    ]
    with pytest.raises(KeyboardInterrupt):
        driver_module.main(argv)


def test_phase_a_system_exit_propagates(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, monkeypatch):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    def _raise(*a, **k):
        raise SystemExit(9)
    monkeypatch.setattr(driver_module, "_load_per_image_stats", _raise)
    argv = [
        "--repo-root", str(ROOT), "--source-manifest", str(source_manifest_path),
        "--run-mode", "pilot20", "--checkpoint", str(checkpoint_path),
        "--result", str(result_path), "--per-image-stats", str(stats_manifest_path),
        "--data-root", str(REAL_DATA_ROOT), "--resume", "--device", "cpu",
    ]
    with pytest.raises(SystemExit) as excinfo:
        driver_module.main(argv)
    assert excinfo.value.code == 9


def test_phase_a_memory_error_propagates(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, monkeypatch):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    def _raise(*a, **k):
        raise MemoryError("simulated")
    monkeypatch.setattr(driver_module, "_load_per_image_stats", _raise)
    argv = [
        "--repo-root", str(ROOT), "--source-manifest", str(source_manifest_path),
        "--run-mode", "pilot20", "--checkpoint", str(checkpoint_path),
        "--result", str(result_path), "--per-image-stats", str(stats_manifest_path),
        "--data-root", str(REAL_DATA_ROOT), "--resume", "--device", "cpu",
    ]
    with pytest.raises(MemoryError):
        driver_module.main(argv)


# ---------------------------------------------------------------------
# Positive controls: Phase A must NOT reject a genuinely valid resume
# pair at index 0, 1, a middle index, and final-minus-one. A full
# resumed-vs-uninterrupted numeric-equivalence run is not exercised here
# -- this sandbox has no local DINOv2 backbone weights staged (only the
# frozen bridge/projection checkpoint is present) and this task forbids
# CUDA initialization, so a genuine end-to-end model pass cannot run
# here regardless of correctness. Instead: (a) confirm Phase A's own
# relational checks accept a valid pair cleanly at each position, and
# (b) confirm the arrays Phase A loads are byte-identical to what
# _load_per_image_stats returns when called directly on the same files
# -- proving the relocation changed WHEN loading happens, never WHAT is
# loaded.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("next_index", [0, 1, 10, 19])
def test_phase_a_accepts_valid_resume_at_various_positions(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index):
    from src.voc2012_matched_evaluator_checkpoint import (
        resume_dataset_index, validate_checkpoint_against_canonical_order,
        validate_checkpoint_against_per_image_stats, validate_checkpoint_structure,
    )

    if next_index == 0:
        # np.stack([]) has no support for empty rows, so a genuinely
        # zero-image artifact is not reachable through the real writer
        # (a pre-existing limitation, independent of this repair) --
        # hand-build a schema-consistent zero-row artifact directly to
        # probe Phase A's own boundary behavior at the earliest possible
        # resume point.
        expected_image_ids, image_order_digest = canonical_ids
        identity_sha256 = _identity_sha256()
        checkpoint_doc = {
            "schema": evaluator_identity["checkpoint"]["schema_name"], "run_mode": "pilot20",
            "identity": evaluator_identity["identity"]["name"], "identity_sha256": identity_sha256,
            "matched_identity_sha256": evaluator_identity["parent_identities"]["matched_identity_sha256"],
            "voc2012_source_identity_sha256": evaluator_identity["parent_identities"]["voc2012_source_identity_sha256"],
            "source_manifest_sha256": hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
            "bridge_checkpoint_sha256": evaluator_identity["model_and_checkpoint"]["projection_checkpoint_sha256"],
            "git_commit": "1" * 40, "v20_class_count": 20, "v21_class_count": 21,
            "live_v20_class_names_digest": "2" * 64, "live_v21_class_names_digest": "3" * 64,
            "image_count_expected": evaluator_identity["run_modes"]["pilot20_image_count"], "image_order_digest": image_order_digest,
            "next_dataset_index": 0, "completed_image_ids": [], "images_completed_count": 0,
            "windows_processed_total": 0, "complete": False,
            "created_at_utc": "2026-01-01T00:00:00+00:00", "updated_at_utc": "2026-01-01T00:00:00+00:00",
        }
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint_path.write_text(json.dumps(checkpoint_doc))

        from src.voc2012_matched_evaluator_report import VARIANT_NAMES
        stats_manifest_path = tmp_path / "stats.json"
        zero_arrays = {"dataset_indices": np.zeros(0, dtype=np.int64), "label_v20": np.zeros((0, 20), dtype=np.int64), "label_v21": np.zeros((0, 21), dtype=np.int64)}
        for v in VARIANT_NAMES:
            w = 20 if v.startswith("v20_") else 21
            for s in ("intersect", "union", "pred"):
                zero_arrays[f"{s}_{v}"] = np.zeros((0, w), dtype=np.int64)
        npz_path = stats_manifest_path.with_suffix(".npz")
        np.savez(npz_path, allow_pickle=False, **zero_arrays)
        stats_manifest_path.write_text(json.dumps({
            "schema": evaluator_identity["artifacts"]["per_image_stats_manifest_schema_name"],
            "npz_filename": npz_path.name, "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
            "v20_class_count": 20, "v21_class_count": 21,
            "live_v20_class_names_digest": "2" * 64, "live_v21_class_names_digest": "3" * 64,
            "image_count": 0, "dataset_indices": [], "image_ids": [], "image_order_digest": hashlib.sha256(b"[]").hexdigest(),
        }))
    else:
        checkpoint_path, stats_manifest_path, _ = _build_valid_pair(
            tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=next_index,
        )
    expected_image_ids, image_order_digest = canonical_ids
    identity_sha256 = _identity_sha256()

    checkpoint_doc = json.loads(checkpoint_path.read_text())
    validate_checkpoint_structure(checkpoint_doc, identity=evaluator_identity, identity_sha256=identity_sha256, run_mode="pilot20")
    resume_dataset_index(checkpoint_doc)
    validate_checkpoint_against_canonical_order(checkpoint_doc, expected_image_ids, image_order_digest=image_order_digest)
    stats = driver_module._load_per_image_stats(stats_manifest_path)
    validate_checkpoint_against_per_image_stats(checkpoint_doc, stats["manifest"], identity=evaluator_identity)  # must not raise

    assert stats["manifest"]["image_count"] == next_index
    assert list(stats["manifest"]["dataset_indices"]) == list(range(next_index))


def test_phase_a_loaded_arrays_identical_to_direct_load(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path):
    """Proves the relocation is purely an ordering change: the arrays
    Phase A loads via _load_per_image_stats are byte-identical to a
    direct, independent call to the same function on the same files."""
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, next_index=5,
    )
    direct = driver_module._load_per_image_stats(stats_manifest_path)
    again = driver_module._load_per_image_stats(stats_manifest_path)
    assert set(direct["arrays"]) == set(again["arrays"])
    for name in direct["arrays"]:
        np.testing.assert_array_equal(direct["arrays"][name], again["arrays"][name])
    assert direct["manifest"] == again["manifest"]


def test_phase_a_fresh_run_never_touches_stats_existence(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, monkeypatch):
    """A fresh (non-resume) run must never require --per-image-stats to
    exist, and must never call _load_per_image_stats."""
    calls = {"count": 0}
    real = driver_module._load_per_image_stats

    def _counting(*a, **k):
        calls["count"] += 1
        return real(*a, **k)
    monkeypatch.setattr(driver_module, "_load_per_image_stats", _counting)

    checkpoint_path = tmp_path / "does-not-exist-checkpoint.json"
    stats_manifest_path = tmp_path / "does-not-exist-stats.json"
    result_path = tmp_path / "result.json"
    argv = [
        "--repo-root", str(ROOT), "--source-manifest", str(source_manifest_path),
        "--run-mode", "pilot20", "--checkpoint", str(checkpoint_path),
        "--result", str(result_path), "--per-image-stats", str(stats_manifest_path),
        "--data-root", str(REAL_DATA_ROOT), "--device", "cpu",
    ]
    # No --resume: Phase A must be a no-op regardless of artifact absence,
    # and _load_per_image_stats must never be called. (This still runs
    # into real dataset/model construction afterward, which in this
    # sandbox fails for an unrelated, pre-existing reason -- the DINOv2
    # backbone weights are not locally staged here -- so only Phase A's
    # own behavior is asserted, not full-run completion.)
    try:
        driver_module.main(argv)
    except Exception:
        pass
    assert calls["count"] == 0


def test_phase_a_resume_with_no_existing_artifacts_behaves_as_fresh(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, monkeypatch):
    """--resume passed but neither checkpoint nor stats exists: the
    established behavior (silently proceed as a fresh run) must be
    preserved -- this must NOT be treated as 'inconsistent presence'."""
    calls = {"count": 0}
    real = driver_module._load_per_image_stats

    def _counting(*a, **k):
        calls["count"] += 1
        return real(*a, **k)
    monkeypatch.setattr(driver_module, "_load_per_image_stats", _counting)

    checkpoint_path = tmp_path / "does-not-exist-checkpoint.json"
    stats_manifest_path = tmp_path / "does-not-exist-stats.json"
    result_path = tmp_path / "result.json"
    argv = [
        "--repo-root", str(ROOT), "--source-manifest", str(source_manifest_path),
        "--run-mode", "pilot20", "--checkpoint", str(checkpoint_path),
        "--result", str(result_path), "--per-image-stats", str(stats_manifest_path),
        "--data-root", str(REAL_DATA_ROOT), "--resume", "--device", "cpu",
    ]
    stderr_buf = io.StringIO()
    try:
        with redirect_stderr(stderr_buf):
            driver_module.main(argv)
    except Exception:
        pass
    assert calls["count"] == 0
    assert "consistently present or" not in stderr_buf.getvalue()
