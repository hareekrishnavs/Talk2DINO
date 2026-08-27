"""Regression test for a real GPU-job failure (job 20642885): the driver
passed --data-root straight through to the mmseg config override without
resolving it through the same acceptance logic verify_ade20k_dataset.py's
preflight/verify-manifest already applied (either the ADEChallengeData2016
directory itself or its unambiguous parent). Preflight passed with
--data-root=.../ADEChallengeData2016 (a form it legitimately accepts), but
the driver then built a config path with 'ADEChallengeData2016' appended
TWICE, since ade20k.py's own img_dir/ann_dir already embed that prefix --
raising a real, job-ending FileNotFoundError.

This test proves the driver's own module-level import wiring resolves
--data-root identically for both accepted forms, matching
src.ade20k_dataset_manifest.resolve_dataset_root exactly -- CPU-only, no
torch/CUDA/model import needed since only the resolution helpers are
exercised."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.ade20k_dataset_identity import load_identity as load_source_identity  # noqa: E402
from src.ade20k_dataset_manifest import resolve_dataset_root  # noqa: E402


def _load_driver_module():
    """Import the driver module by file path without triggering its own
    sys.path/CUDA-adjacent side effects beyond what module-level imports
    already require (identity/checkpoint/report modules only -- torch,
    mmcv, and mmseg are imported lazily inside _run_evaluation, never at
    module import time)."""
    spec = importlib.util.spec_from_file_location(
        "run_ade20k_matched_evaluation", ROOT / "diagnostics/run_ade20k_matched_evaluation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_driver_imports_the_same_resolver_verify_ade20k_dataset_uses():
    driver = _load_driver_module()
    assert driver.resolve_ade20k_dataset_root is resolve_dataset_root


def test_driver_data_root_resolution_agrees_for_both_accepted_forms(tmp_path):
    """Synthetic layout, both forms of --data-root must resolve to the
    SAME ADEChallengeData2016 directory (and therefore the same parent,
    which is what the mmseg config's data_root must become)."""
    ade = tmp_path / "ADEChallengeData2016"
    for sub in ("images/training", "images/validation", "annotations/training", "annotations/validation"):
        (ade / sub).mkdir(parents=True)

    identity = load_source_identity(repo_root=ROOT)

    resolved_from_ade_dir_itself = resolve_dataset_root(ade, identity)
    resolved_from_parent = resolve_dataset_root(tmp_path, identity)

    assert resolved_from_ade_dir_itself == resolved_from_parent == ade
    assert resolved_from_ade_dir_itself.parent == resolved_from_parent.parent == tmp_path


def test_real_data_root_form_that_failed_job_20642885_now_resolves_correctly():
    """The exact real path shape that caused the job to fail:
    --data-root pointed directly at the ADEChallengeData2016 directory.
    No private path is hardcoded here beyond what test_ade20k_dataset_identity.py
    already establishes is safe to read from -- this test is gated on the
    real dataset via the same pattern as the other real-data tests."""
    import os

    raw = os.environ.get("ADE20K_REAL_DATA_ROOT")
    if raw is None or not raw.strip():
        pytest.skip("requires ADE20K_REAL_DATA_ROOT to point at a real ADEChallengeData2016 root")
    real_ade_dir = Path(raw).expanduser()
    if not real_ade_dir.is_dir():
        pytest.skip("ADE20K_REAL_DATA_ROOT does not exist")

    identity = load_source_identity(repo_root=ROOT)
    resolved = resolve_dataset_root(real_ade_dir, identity)
    config_data_root = resolved.parent

    # This is exactly what the mmseg config's img_dir/ann_dir get joined
    # against; it must NOT contain a doubled 'ADEChallengeData2016'.
    assert (config_data_root / "ADEChallengeData2016" / "images" / "validation").is_dir()
    assert "ADEChallengeData2016/ADEChallengeData2016" not in str(config_data_root / "ADEChallengeData2016")
