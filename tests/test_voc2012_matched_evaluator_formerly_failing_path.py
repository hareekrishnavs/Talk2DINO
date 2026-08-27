"""Integration-style CPU regression proving the VOC2012 matched
evaluator's real production path reaches, passes, and continues beyond
the exact line that failed on real GPU pilot job 20626350 (the bare-ID
filename-reconciliation bug). Exercises Phase A, real dataset
construction, and real model/backbone construction for the first real
canonical image -- only the heavy per-window snapshot/graph/propagation
work (stitch_one_image_with_e3, i.e. everything AFTER the formerly
failing call site) is stubbed, so this test stays CPU-bounded while
still proving the fix through the real code path, not a mock of it.

Must be run as its own process (not combined with other test files in
the same pytest session) because TALK2DINO_WEIGHT_DIR must already be
set in the environment before src.local_weights.DEFAULT_WEIGHT_DIR is
evaluated at that module's import time. This file never hardcodes a
private path for it -- the test skips cleanly if the caller hasn't set
one, exactly like VOC2012_REAL_DATA_ROOT below."""

from __future__ import annotations

import os

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from contextlib import redirect_stderr
from pathlib import Path

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


def _weight_dir():
    raw = os.environ.get("TALK2DINO_WEIGHT_DIR")
    if raw is None or not raw.strip():
        return None
    return Path(raw).expanduser()


REAL_DATA_ROOT = _real_data_root()
WEIGHT_DIR = _weight_dir()
requires_real_data_and_weights = pytest.mark.skipif(
    REAL_DATA_ROOT is None or not REAL_DATA_ROOT.exists()
    or WEIGHT_DIR is None or not (WEIGHT_DIR / "dinov2_vitb14_reg4_pretrain.pth").exists(),
    reason="requires VOC2012_REAL_DATA_ROOT and TALK2DINO_WEIGHT_DIR (pointing at a staged DINOv2 backbone) to be set",
)
pytestmark = [pytestmark, requires_real_data_and_weights]


class _StubReached(Exception):
    """Raised by the stubbed heavy per-window entry point -- reaching it
    proves execution got past the formerly-failing filename
    reconciliation, real Phase A, real dataset construction, and real
    model/backbone/bridge-checkpoint construction, entirely on CPU."""


@pytest.fixture(scope="module")
def driver_module():
    spec = importlib.util.spec_from_file_location(
        "voc2012_driver_formerly_failing_path_under_test", ROOT / "diagnostics" / "run_voc2012_matched_evaluation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_first_sample_reaches_and_passes_the_formerly_failing_line(tmp_path, driver_module):
    from src.voc2012_matched_evaluator_identity import load_identity

    identity = load_identity(repo_root=ROOT)

    source_manifest_path = tmp_path / "source_manifest.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify_voc2012_dataset.py"), "generate-manifest",
         "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--output", str(source_manifest_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    checkpoint_path = tmp_path / "checkpoint.json"
    result_path = tmp_path / "result.json"
    per_image_stats_path = tmp_path / "stats.json"

    # Poison ONLY the heavy per-window snapshot/graph/propagation entry
    # point -- everything up to and including the first image's
    # _extract_prepared_voc_image call (the formerly failing line) runs
    # for real. `import main` first (side-effect only, registers the
    # FloatImage pipeline transform) matches the established import-order
    # requirement _run_evaluation's own source already documents. The
    # module is reached via the same `from ... import name` form the
    # driver itself uses -- `import models.dinotext.cover_dr....` as a
    # dotted `import x.y.z as name` statement hits an unrelated,
    # pre-existing package/module name collision (models/dinotext/
    # dinotext.py sharing a name with the models.dinotext package) that
    # only affects that import FORM, not the driver's own style.
    import main  # noqa: F401
    from models.dinotext.cover_dr.coco_object_evaluator import stitch_one_image_with_e3 as _unused  # noqa: F401
    coe_module = sys.modules["models.dinotext.cover_dr.coco_object_evaluator"]
    real_stitch = coe_module.stitch_one_image_with_e3
    reached = {"flag": False}

    def poisoned_stitch(*a, **k):
        reached["flag"] = True
        raise _StubReached("reached stitch_one_image_with_e3 -- formerly failing line already passed")

    # The driver does `from ...coco_object_evaluator import
    # stitch_one_image_with_e3` LOCALLY inside _run_evaluation, freshly
    # on each call -- patching the source module attribute here is
    # picked up by that fresh import at call time.
    coe_module.stitch_one_image_with_e3 = poisoned_stitch

    argv = [
        "--repo-root", str(ROOT), "--source-manifest", str(source_manifest_path),
        "--run-mode", "pilot20", "--checkpoint", str(checkpoint_path), "--result", str(result_path),
        "--per-image-stats", str(per_image_stats_path), "--data-root", str(REAL_DATA_ROOT), "--device", "cpu",
    ]

    stderr_buf = io.StringIO()
    caught = None
    try:
        with redirect_stderr(stderr_buf):
            try:
                driver_module.main(argv)
            except _StubReached:
                pass
    except Exception as e:  # pragma: no cover -- diagnostic aid only
        caught = e
    finally:
        coe_module.stitch_one_image_with_e3 = real_stitch

    assert caught is None, f"unexpected exception (not the expected stub) reached the top: {caught!r}"
    assert reached["flag"], (
        "stitch_one_image_with_e3 (the heavy per-window entry point, immediately after image preparation) "
        "was never reached -- means Phase A, dataset construction, model/backbone construction, or the "
        "formerly-failing filename reconciliation did not complete successfully"
    )
    assert torch.cuda.is_initialized() is False
    # No checkpoint/result/stats were ever written -- the stub fires
    # before the first _write_per_image_stats_atomically/checkpoint call.
    assert not result_path.exists()
