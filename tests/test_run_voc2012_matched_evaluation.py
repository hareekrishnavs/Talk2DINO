"""CLI/failure-safety tests for diagnostics/run_voc2012_matched_evaluation.py
and verify_voc2012_matched_evaluation.py. CPU-only for everything except
the explicitly real-data-gated smoke test at the bottom, which never
initializes CUDA."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import verify_voc2012_matched_evaluation as verify_cli  # noqa: E402

IDENTITY_PATH = ROOT / "evaluation_identities/e12_voc2012_matched_evaluator.toml"
pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the voc2012 matched-evaluator identity")


@pytest.fixture(scope="module")
def driver_module():
    spec = importlib.util.spec_from_file_location(
        "voc2012_driver_under_test_cli", ROOT / "diagnostics" / "run_voc2012_matched_evaluation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scratch(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------
# --data-root / VOC2012_REAL_DATA_ROOT resolution
# ---------------------------------------------------------------------


def test_resolve_data_root_uses_explicit_arg_over_env(driver_module, monkeypatch):
    monkeypatch.setenv("VOC2012_REAL_DATA_ROOT", "/should/not/be/used")
    result = driver_module._resolve_data_root(Path("/explicit/path"))
    assert result == Path("/explicit/path")


def test_resolve_data_root_falls_back_to_env(driver_module, monkeypatch, tmp_path):
    target = tmp_path / "voc-root"
    monkeypatch.setenv("VOC2012_REAL_DATA_ROOT", str(target))
    result = driver_module._resolve_data_root(None)
    assert result == target


@pytest.mark.parametrize("env_value", [None, "", "   ", "\t"])
def test_resolve_data_root_rejects_absent_empty_whitespace_env(driver_module, monkeypatch, env_value):
    if env_value is None:
        monkeypatch.delenv("VOC2012_REAL_DATA_ROOT", raising=False)
    else:
        monkeypatch.setenv("VOC2012_REAL_DATA_ROOT", env_value)
    with pytest.raises(Exception):
        driver_module._resolve_data_root(None)


# ---------------------------------------------------------------------
# Argparse structure
# ---------------------------------------------------------------------


def test_driver_argparser_requires_run_mode_and_choices(driver_module):
    parser = driver_module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--source-manifest", "x.json", "--checkpoint", "c.json", "--result", "r.json", "--per-image-stats", "p.json"])
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--source-manifest", "x.json", "--run-mode", "not-a-mode", "--checkpoint", "c.json",
            "--result", "r.json", "--per-image-stats", "p.json",
        ])


def test_driver_argparser_accepts_valid_minimal_invocation(driver_module):
    parser = driver_module.build_parser()
    args = parser.parse_args([
        "--source-manifest", "x.json", "--run-mode", "pilot20", "--checkpoint", "c.json",
        "--result", "r.json", "--per-image-stats", "p.json",
    ])
    assert args.run_mode == "pilot20"
    assert args.device == "cuda"  # default
    assert args.resume is False
    assert args.overwrite is False


def test_verify_cli_has_four_subcommands():
    parser = verify_cli.build_parser()
    subparsers_action = next(a for a in parser._actions if a.dest == "command")
    assert set(subparsers_action.choices) == {"preflight", "verify-source-binding", "verify-checkpoint", "verify-result"}


# ---------------------------------------------------------------------
# Failure safety: missing/malformed inputs, exit 2, no traceback, no
# partial output, existing-result preservation, no stray temp files
# ---------------------------------------------------------------------


def test_driver_missing_source_manifest_fails_closed(driver_module, scratch):
    args = driver_module.build_parser().parse_args([
        "--repo-root", str(ROOT), "--source-manifest", str(scratch / "does-not-exist.json"),
        "--run-mode", "pilot20", "--checkpoint", str(scratch / "c.json"), "--result", str(scratch / "r.json"),
        "--per-image-stats", str(scratch / "p.json"),
    ])
    exit_code = driver_module.main([
        "--repo-root", str(ROOT), "--source-manifest", str(scratch / "does-not-exist.json"),
        "--run-mode", "pilot20", "--checkpoint", str(scratch / "c.json"), "--result", str(scratch / "r.json"),
        "--per-image-stats", str(scratch / "p.json"), "--data-root", str(scratch / "fake-data-root"),
    ])
    assert exit_code == 2
    assert not (scratch / "r.json").exists()


def test_driver_existing_result_without_overwrite_preserved(driver_module, scratch):
    result_path = scratch / "r.json"
    result_path.write_text('{"existing": "content"}')
    before = result_path.read_bytes()

    exit_code = driver_module.main([
        "--repo-root", str(ROOT), "--source-manifest", str(scratch / "manifest.json"),
        "--run-mode", "pilot20", "--checkpoint", str(scratch / "c.json"), "--result", str(result_path),
        "--per-image-stats", str(scratch / "p.json"), "--data-root", str(scratch / "fake-data-root"),
    ])
    assert exit_code == 2
    assert result_path.read_bytes() == before
    assert list(scratch.glob("*.tmp*")) == []
    assert list(scratch.glob("*.selfcheck*")) == []


def test_driver_no_stray_temp_files_after_source_manifest_failure(driver_module, scratch):
    (scratch / "manifest.json").write_text("not valid json {{{")
    driver_module.main([
        "--repo-root", str(ROOT), "--source-manifest", str(scratch / "manifest.json"),
        "--run-mode", "pilot20", "--checkpoint", str(scratch / "c.json"), "--result", str(scratch / "r.json"),
        "--per-image-stats", str(scratch / "p.json"), "--data-root", str(scratch / "fake-data-root"),
    ])
    leftover = [p for p in scratch.iterdir() if p.suffix == ".tmp" or ".tmp" in p.name or "selfcheck" in p.name]
    assert leftover == []


def test_driver_keyboard_interrupt_not_swallowed(driver_module, monkeypatch, scratch):
    def raising(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(driver_module, "_verify_source_manifest_before_cuda", raising)
    with pytest.raises(KeyboardInterrupt):
        driver_module.main([
            "--repo-root", str(ROOT), "--source-manifest", str(scratch / "manifest.json"),
            "--run-mode", "pilot20", "--checkpoint", str(scratch / "c.json"), "--result", str(scratch / "r.json"),
            "--per-image-stats", str(scratch / "p.json"), "--data-root", str(scratch / "fake-data-root"),
        ])


def test_driver_system_exit_not_swallowed(driver_module, monkeypatch, scratch):
    def raising(*a, **k):
        raise SystemExit(7)

    monkeypatch.setattr(driver_module, "_verify_source_manifest_before_cuda", raising)
    with pytest.raises(SystemExit) as excinfo:
        driver_module.main([
            "--repo-root", str(ROOT), "--source-manifest", str(scratch / "manifest.json"),
            "--run-mode", "pilot20", "--checkpoint", str(scratch / "c.json"), "--result", str(scratch / "r.json"),
            "--per-image-stats", str(scratch / "p.json"), "--data-root", str(scratch / "fake-data-root"),
        ])
    assert excinfo.value.code == 7


def test_driver_memory_error_not_swallowed(driver_module, monkeypatch, scratch):
    def raising(*a, **k):
        raise MemoryError("simulated OOM")

    monkeypatch.setattr(driver_module, "_verify_source_manifest_before_cuda", raising)
    with pytest.raises(MemoryError):
        driver_module.main([
            "--repo-root", str(ROOT), "--source-manifest", str(scratch / "manifest.json"),
            "--run-mode", "pilot20", "--checkpoint", str(scratch / "c.json"), "--result", str(scratch / "r.json"),
            "--per-image-stats", str(scratch / "p.json"), "--data-root", str(scratch / "fake-data-root"),
        ])


def test_verify_cli_malformed_json_checkpoint_fails_closed(scratch):
    bad = scratch / "bad.json"
    bad.write_text('{"a": 1, "a": 2}')  # duplicate key
    exit_code = verify_cli.main(["verify-checkpoint", "--repo-root", str(ROOT), "--checkpoint", str(bad)])
    assert exit_code == 2


def test_verify_cli_missing_result_file_fails_closed(scratch):
    exit_code = verify_cli.main(["verify-result", "--repo-root", str(ROOT), "--result", str(scratch / "missing.json")])
    assert exit_code == 2


def test_verify_cli_preflight_passes_against_real_repo():
    exit_code = verify_cli.main(["preflight", "--repo-root", str(ROOT)])
    assert exit_code == 0


# ---------------------------------------------------------------------
# Real-data CPU smoke test (VOC2012_REAL_DATA_ROOT-gated, no CUDA)
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


@requires_real_data
def test_real_data_first_canonical_samples_cpu_only_no_cuda():
    """Resolves the real VOC2012 root, the canonical validation-split
    order, and confirms the first few pilot20 image IDs are stable and
    well-formed -- entirely CPU-side (no torch.cuda call anywhere in this
    test), matching the task's 'CPU smoke test' requirement."""
    from src.voc2012_dataset_identity import load_identity as load_source_identity
    from src.voc2012_dataset_manifest import canonical_validation_ids, resolve_dataset_root
    from src.voc2012_matched_evaluator_identity import load_identity as load_matched_evaluator_identity

    source_identity = load_source_identity(repo_root=ROOT)
    matched_evaluator_identity = load_matched_evaluator_identity(repo_root=ROOT)

    voc_root = resolve_dataset_root(REAL_DATA_ROOT, source_identity)
    canonical_ids = canonical_validation_ids(voc_root, source_identity)
    assert len(canonical_ids) == matched_evaluator_identity["protocol"]["expected_image_count"]

    pilot20 = canonical_ids[: matched_evaluator_identity["run_modes"]["pilot20_image_count"]]
    assert len(pilot20) == 20
    assert len(set(pilot20)) == 20  # no duplicates in the prefix
    for image_id in pilot20:
        assert (voc_root / "JPEGImages" / f"{image_id}.jpg").is_file()
        assert (voc_root / "SegmentationClass" / f"{image_id}.png").is_file()

    import torch

    assert torch.cuda.is_initialized() is False
