"""Regression tests for the per-image-stats NPZ failure-contract repair:
_load_per_image_stats must convert every expected filesystem/parse
failure (missing file, directory-in-place-of-file, dangling symlink,
unreadable file, a race between manifest parsing and digest
computation, ValueError/OSError from _sha256_file or np.load, digest
mismatch, invalid dtype/shape) into Voc2012MatchedEvaluatorIdentityError
-- never a raw FileNotFoundError/OSError/ValueError -- and main() must
convert the same OSError family into a clean exit-2 diagnostic.

Direct-call tests need no real VOC2012 data. CLI-subprocess and
poisoned-entry-point tests need a real, verified source manifest (same
VOC2012_REAL_DATA_ROOT-gated convention as
test_voc2012_matched_evaluator_phase_a_ordering.py) since they exercise
the full --resume Phase-A path, not just the loader in isolation."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
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


@pytest.fixture(scope="module")
def driver_module():
    spec = importlib.util.spec_from_file_location(
        "voc2012_driver_npz_failure_contract_under_test", ROOT / "diagnostics" / "run_voc2012_matched_evaluation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_valid_artifact(driver_module, tmp_path, *, n=3):
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
    manifest_path = tmp_path / "stats.json"
    driver_module._write_per_image_stats_atomically(
        manifest_path, schema_name="test-schema", v20_class_count=20, v21_class_count=21,
        live_v20_class_names_digest="a" * 64, live_v21_class_names_digest="b" * 64,
        dataset_indices=list(range(n)), image_ids=[f"img{i}" for i in range(n)], rows=rows,
    )
    return manifest_path


# ---------------------------------------------------------------------
# Direct _load_per_image_stats calls: every case converts to the domain
# exception, never a raw filesystem/parse exception.
# ---------------------------------------------------------------------


def test_direct_manifest_exists_npz_missing(driver_module, tmp_path):
    manifest_path = _write_valid_artifact(driver_module, tmp_path)
    manifest_path.with_suffix(".npz").unlink()
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError) as excinfo:
        driver_module._load_per_image_stats(manifest_path)
    assert "per-image-stats" in str(excinfo.value).lower()
    assert not isinstance(excinfo.value, FileNotFoundError)


def test_direct_npz_path_is_a_directory(driver_module, tmp_path):
    manifest_path = _write_valid_artifact(driver_module, tmp_path)
    npz_path = manifest_path.with_suffix(".npz")
    npz_path.unlink()
    npz_path.mkdir()
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError):
        driver_module._load_per_image_stats(manifest_path)


def test_direct_dangling_npz_symlink(driver_module, tmp_path):
    manifest_path = _write_valid_artifact(driver_module, tmp_path)
    npz_path = manifest_path.with_suffix(".npz")
    npz_path.unlink()
    npz_path.symlink_to(tmp_path / "does-not-exist-target.npz")
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError):
        driver_module._load_per_image_stats(manifest_path)


def test_direct_unreadable_npz(driver_module, tmp_path):
    if os.geteuid() == 0:
        pytest.skip("running as root -- permission checks are not meaningful")
    manifest_path = _write_valid_artifact(driver_module, tmp_path)
    npz_path = manifest_path.with_suffix(".npz")
    npz_path.chmod(0o000)
    try:
        with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError):
            driver_module._load_per_image_stats(manifest_path)
    finally:
        npz_path.chmod(0o644)


def test_direct_npz_removed_between_manifest_validation_and_digest(driver_module, tmp_path, monkeypatch):
    """Controlled race: the manifest JSON has already been parsed
    successfully when the NPZ disappears immediately before the digest
    computation reads it."""
    manifest_path = _write_valid_artifact(driver_module, tmp_path)
    npz_path = manifest_path.with_suffix(".npz")
    real_sha256_file = driver_module._sha256_file

    def racy_sha256_file(path):
        if Path(path) == npz_path and npz_path.exists():
            npz_path.unlink()  # simulate disappearance right before the read
        return real_sha256_file(path)

    monkeypatch.setattr(driver_module, "_sha256_file", racy_sha256_file)
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError):
        driver_module._load_per_image_stats(manifest_path)


def test_direct_sha256_file_raises_filenotfounderror(driver_module, tmp_path, monkeypatch):
    manifest_path = _write_valid_artifact(driver_module, tmp_path)

    def raising(path):
        raise FileNotFoundError(f"simulated: {path}")

    monkeypatch.setattr(driver_module, "_sha256_file", raising)
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError):
        driver_module._load_per_image_stats(manifest_path)


def test_direct_sha256_file_raises_generic_oserror(driver_module, tmp_path, monkeypatch):
    manifest_path = _write_valid_artifact(driver_module, tmp_path)

    def raising(path):
        raise OSError("simulated generic OS failure")

    monkeypatch.setattr(driver_module, "_sha256_file", raising)
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError):
        driver_module._load_per_image_stats(manifest_path)


def test_direct_np_load_raises_oserror(driver_module, tmp_path, monkeypatch):
    manifest_path = _write_valid_artifact(driver_module, tmp_path)

    def raising_load(*a, **k):
        raise OSError("simulated np.load OS failure")

    monkeypatch.setattr(driver_module.np, "load", raising_load)
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError):
        driver_module._load_per_image_stats(manifest_path)


def test_direct_corrupted_npz_raises_valueerror_path(driver_module, tmp_path):
    """Overwrite the NPZ with non-zip garbage bytes -- numpy's np.load
    raises ValueError for this specific corruption shape (content that
    is not a valid zip container at all)."""
    manifest_path = _write_valid_artifact(driver_module, tmp_path)
    npz_path = manifest_path.with_suffix(".npz")
    npz_path.write_bytes(b"not a zip file, just garbage bytes 1234567890" * 10)
    manifest = json.loads(manifest_path.read_text())
    manifest["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError):
        driver_module._load_per_image_stats(manifest_path)


def test_direct_digest_mismatch(driver_module, tmp_path):
    manifest_path = _write_valid_artifact(driver_module, tmp_path)
    npz_path = manifest_path.with_suffix(".npz")
    npz_path.write_bytes(npz_path.read_bytes() + b"\x00")  # tamper without recomputing digest
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="SHA256"):
        driver_module._load_per_image_stats(manifest_path)


def test_direct_invalid_dtype(driver_module, tmp_path):
    manifest_path = _write_valid_artifact(driver_module, tmp_path)
    npz_path = manifest_path.with_suffix(".npz")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["label_v20"] = arrays["label_v20"].astype(np.float64)
    np.savez(npz_path, allow_pickle=False, **arrays)
    manifest = json.loads(manifest_path.read_text())
    manifest["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="int64"):
        driver_module._load_per_image_stats(manifest_path)


def test_direct_invalid_shape(driver_module, tmp_path):
    manifest_path = _write_valid_artifact(driver_module, tmp_path)
    npz_path = manifest_path.with_suffix(".npz")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["label_v21"] = np.zeros((arrays["label_v21"].shape[0], 20), dtype=np.int64)
    np.savez(npz_path, allow_pickle=False, **arrays)
    manifest = json.loads(manifest_path.read_text())
    manifest["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="columns"):
        driver_module._load_per_image_stats(manifest_path)


def test_direct_keyboardinterrupt_propagates(driver_module, tmp_path, monkeypatch):
    manifest_path = _write_valid_artifact(driver_module, tmp_path)

    def raising(path):
        raise KeyboardInterrupt()

    monkeypatch.setattr(driver_module, "_sha256_file", raising)
    with pytest.raises(KeyboardInterrupt):
        driver_module._load_per_image_stats(manifest_path)


def test_direct_systemexit_propagates(driver_module, tmp_path, monkeypatch):
    manifest_path = _write_valid_artifact(driver_module, tmp_path)

    def raising(path):
        raise SystemExit(11)

    monkeypatch.setattr(driver_module, "_sha256_file", raising)
    with pytest.raises(SystemExit) as excinfo:
        driver_module._load_per_image_stats(manifest_path)
    assert excinfo.value.code == 11


def test_direct_memoryerror_propagates(driver_module, tmp_path, monkeypatch):
    manifest_path = _write_valid_artifact(driver_module, tmp_path)

    def raising(path):
        raise MemoryError("simulated")

    monkeypatch.setattr(driver_module, "_sha256_file", raising)
    with pytest.raises(MemoryError):
        driver_module._load_per_image_stats(manifest_path)


def test_main_except_clause_now_includes_oserror(driver_module):
    import inspect

    source = inspect.getsource(driver_module.main)
    assert "OSError" in source, "main()'s except clause must include OSError per the failure contract"
    assert "except Exception" not in source
    assert "except:" not in source


# ---------------------------------------------------------------------
# Real-data-gated: CLI subprocess + in-process poisoned-entry-point
# confirmation for the headline missing-NPZ case, plus a representative
# subprocess subset for other cases.
# ---------------------------------------------------------------------


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


@pytest.fixture(scope="module")
def evaluator_identity():
    from src.voc2012_matched_evaluator_identity import load_identity

    return load_identity(repo_root=ROOT)


@pytest.fixture(scope="module")
def source_manifest_path(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("npz_failure_contract_source_manifest")
    manifest_path = out_dir / "source_manifest.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify_voc2012_dataset.py"), "generate-manifest",
         "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--output", str(manifest_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return manifest_path


@pytest.fixture(scope="module")
def canonical_ids(driver_module):
    from src.voc2012_dataset_identity import load_identity as load_source_identity
    from src.voc2012_dataset_manifest import canonical_validation_ids, resolve_dataset_root
    from src.voc2012_matched_evaluator_identity import load_identity as load_evaluator_identity

    source_identity = load_source_identity(repo_root=ROOT)
    voc_root = resolve_dataset_root(REAL_DATA_ROOT, source_identity)
    ids = canonical_validation_ids(voc_root, source_identity)
    image_count = load_evaluator_identity(repo_root=ROOT)["run_modes"]["pilot20_image_count"]
    expected_image_ids = ids[:image_count]
    image_order_digest = driver_module._image_order_digest(expected_image_ids)
    return expected_image_ids, image_order_digest


def _build_valid_pair(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, *, next_index=5):
    expected_image_ids, image_order_digest = canonical_ids
    identity_sha256 = hashlib.sha256(IDENTITY_PATH.read_bytes()).hexdigest()
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
        "next_dataset_index": next_index, "completed_image_ids": expected_image_ids[:next_index],
        "images_completed_count": next_index, "windows_processed_total": next_index * 3, "complete": False,
        "created_at_utc": "2026-01-01T00:00:00+00:00", "updated_at_utc": "2026-01-01T00:00:00+00:00",
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint_doc))
    stats_manifest_path = _write_valid_artifact(driver_module, tmp_path, n=next_index)
    # rebuild with the real image_ids/dataset_indices matching the checkpoint
    stats_manifest_path.unlink()
    stats_manifest_path.with_suffix(".npz").unlink()
    from src.voc2012_matched_evaluator_report import VARIANT_NAMES
    rng = np.random.default_rng(0)
    rows = {"label_v20": [], "label_v21": []}
    for v in VARIANT_NAMES:
        rows[f"intersect_{v}"] = []
        rows[f"union_{v}"] = []
        rows[f"pred_{v}"] = []
    for _ in range(next_index):
        rows["label_v20"].append(rng.integers(0, 5, size=20))
        rows["label_v21"].append(rng.integers(0, 5, size=21))
        for v in VARIANT_NAMES:
            w = 20 if v.startswith("v20_") else 21
            rows[f"intersect_{v}"].append(rng.integers(0, 5, size=w))
            rows[f"union_{v}"].append(rng.integers(1, 5, size=w))
            rows[f"pred_{v}"].append(rng.integers(0, 5, size=w))
    driver_module._write_per_image_stats_atomically(
        stats_manifest_path, schema_name=evaluator_identity["artifacts"]["per_image_stats_manifest_schema_name"],
        v20_class_count=20, v21_class_count=21,
        live_v20_class_names_digest="2" * 64, live_v21_class_names_digest="3" * 64,
        dataset_indices=list(range(next_index)), image_ids=expected_image_ids[:next_index], rows=rows,
    )
    result_path = tmp_path / "result.json"
    return checkpoint_path, stats_manifest_path, result_path


class _PoisonReached(Exception):
    pass


@requires_real_data
def test_missing_npz_full_cli_in_process_zero_heavy_calls(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path,
    )
    stats_manifest_path.with_suffix(".npz").unlink()

    counts = {k: 0 for k in ("build_dataset", "build_model", "backbone", "load_checkpoint", "cuda", "to")}
    import mmseg.datasets as mmseg_datasets_module
    import models as models_module
    import mmcv.runner as mmcv_runner_module
    import src.local_weights as local_weights_module

    def poison(name):
        def _fn(*a, **k):
            counts[name] += 1
            raise _PoisonReached(name)
        return _fn

    orig = {
        "build_dataset": mmseg_datasets_module.build_dataset, "build_model": models_module.build_model,
        "backbone": local_weights_module.load_local_vision_backbone,
        "load_checkpoint": mmcv_runner_module.CheckpointLoader.load_checkpoint,
        "cuda": torch.nn.Module.cuda, "to": torch.nn.Module.to,
    }
    mmseg_datasets_module.build_dataset = poison("build_dataset")
    models_module.build_model = poison("build_model")
    local_weights_module.load_local_vision_backbone = poison("backbone")
    mmcv_runner_module.CheckpointLoader.load_checkpoint = staticmethod(poison("load_checkpoint"))
    torch.nn.Module.cuda = poison("cuda")
    torch.nn.Module.to = poison("to")

    try:
        argv = [
            "--repo-root", str(ROOT), "--source-manifest", str(source_manifest_path),
            "--run-mode", "pilot20", "--checkpoint", str(checkpoint_path),
            "--result", str(result_path), "--per-image-stats", str(stats_manifest_path),
            "--data-root", str(REAL_DATA_ROOT), "--resume", "--device", "cpu",
        ]
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            exit_code = driver_module.main(argv)
        stderr_text = stderr_buf.getvalue()
    finally:
        mmseg_datasets_module.build_dataset = orig["build_dataset"]
        models_module.build_model = orig["build_model"]
        local_weights_module.load_local_vision_backbone = orig["backbone"]
        mmcv_runner_module.CheckpointLoader.load_checkpoint = orig["load_checkpoint"]
        torch.nn.Module.cuda = orig["cuda"]
        torch.nn.Module.to = orig["to"]

    assert exit_code == 2
    assert "Traceback" not in stderr_text
    assert "VOC2012 MATCHED EVALUATOR FAIL" in stderr_text
    assert not result_path.exists()
    assert torch.cuda.is_initialized() is False
    for name, count in counts.items():
        assert count == 0, f"{name} was reached ({count} calls) -- missing-NPZ must fail in Phase A"
    leftover = [p for p in tmp_path.rglob("*") if p.is_file() and (".tmp" in p.name or "selfcheck" in p.name)]
    assert leftover == []


@requires_real_data
def test_missing_npz_existing_valid_result_preserved(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path):
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path,
    )
    stats_manifest_path.with_suffix(".npz").unlink()
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


@requires_real_data
@pytest.mark.parametrize("tamper", ["missing_npz", "directory_npz", "corrupted_npz", "digest_mismatch"])
def test_cli_subprocess_failure_contract(tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path, tamper):
    """A genuine, separate-process CLI invocation (not driver.main() in
    the test process) -- the most faithful possible reproduction of the
    original uncaught-exception bug and its repair."""
    checkpoint_path, stats_manifest_path, result_path = _build_valid_pair(
        tmp_path, driver_module, evaluator_identity, canonical_ids, source_manifest_path,
    )
    npz_path = stats_manifest_path.with_suffix(".npz")
    if tamper == "missing_npz":
        npz_path.unlink()
    elif tamper == "directory_npz":
        npz_path.unlink()
        npz_path.mkdir()
    elif tamper == "corrupted_npz":
        npz_path.write_bytes(b"not a zip file, garbage bytes" * 20)
        manifest = json.loads(stats_manifest_path.read_text())
        manifest["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        stats_manifest_path.write_text(json.dumps(manifest))
    elif tamper == "digest_mismatch":
        npz_path.write_bytes(npz_path.read_bytes() + b"\x00")

    proc = subprocess.run(
        [sys.executable, str(ROOT / "diagnostics" / "run_voc2012_matched_evaluation.py"),
         "--repo-root", str(ROOT), "--source-manifest", str(source_manifest_path),
         "--run-mode", "pilot20", "--checkpoint", str(checkpoint_path),
         "--result", str(result_path), "--per-image-stats", str(stats_manifest_path),
         "--data-root", str(REAL_DATA_ROOT), "--resume", "--device", "cpu"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert proc.returncode == 2, f"tamper={tamper}: expected exit 2, got {proc.returncode}\nstderr={proc.stderr}"
    assert "Traceback" not in proc.stderr, f"tamper={tamper}: raw traceback leaked to stderr:\n{proc.stderr}"
    assert "VOC2012 MATCHED EVALUATOR FAIL" in proc.stderr, f"tamper={tamper}: missing concise diagnostic prefix"
    assert not result_path.exists()
