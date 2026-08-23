"""Regression tests for the evaluator CLI's top-level exception boundary
(``diagnostics/run_matched_k11_k12_evaluation.py::main``).

Confirmed defect (independent audit): legitimate, reachable ``ValueError``s
from already-verified, unmodified building blocks --
``models.dinotext.cover_dr.compute_full_precision_metrics`` rejecting
non-finite/degenerate/out-of-range sufficient statistics, and
``write_checkpoint_atomically``'s own strict ``allow_nan=False`` JSON
serialization -- were not caught by ``main()``'s exception tuple, producing
an uncaught traceback and exit code 1 instead of the evaluator's normal
fail-closed contract (exit 2, concise ``K11/K12 POWER EVALUATION FAIL:``
diagnostic, no traceback).

The repair adds ``ValueError`` to ``main()``'s existing exception tuple --
never a broad ``except Exception``/``except BaseException`` -- so
``KeyboardInterrupt``, ``SystemExit``, and ``MemoryError`` (none of which
are ``ValueError`` subclasses) remain completely untouched by this change.

Reuses the exact synthetic dataset/model/harness from
``test_matched_power_evaluator_e2e.py`` (never redefined here) so these
tests exercise the real ``cli.main()`` entry point end to end, on CPU,
without CUDA or the real E3 checkpoint/dataset.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # already set by a prior test module in this process

ROOT = Path(__file__).parents[1]
TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/open_vocabulary_segmentation"))
sys.path.insert(0, str(TESTS_DIR))

import diagnostics.run_matched_k11_k12_evaluation as cli  # noqa: E402
from test_matched_power_evaluator_e2e import (  # noqa: E402 -- reused, never redefined
    NUM_CLASSES,
    NUM_IMAGES,
    REAL_STABILITY_RESULT,
    run_synthetic_evaluation,
)

pytestmark = pytest.mark.skipif(
    not REAL_STABILITY_RESULT.exists(), reason="real GPU stability-gate result not present on this machine"
)


def _warm_up_cover_dr():
    """The evaluator imports models.dinotext.cover_dr lazily, inside
    _run_evaluation; a real run must happen once before sys.modules
    actually holds it, so tests can monkeypatch its attributes. Uses a
    real temporary directory (never a path inside the repository) so no
    stray artifact is left behind regardless of how this module is run."""
    if "models.dinotext.cover_dr" in sys.modules:
        return
    import tempfile

    with tempfile.TemporaryDirectory() as warm_dir_str:
        warm_dir = Path(warm_dir_str)
        exit_code, _, _ = run_synthetic_evaluation(
            checkpoint_path=warm_dir / "c.json", result_path=warm_dir / "r.json", stats_path=warm_dir / "s.json",
        )
    assert exit_code == 0, "warm-up run itself failed, cannot proceed"


def _current_cover_dr_module():
    """Fetch models.dinotext.cover_dr FRESH from sys.modules every time --
    never cache the module object at collection time. Several sibling test
    files in this suite (e.g. test_image_window_cache.py,
    test_finite_step_regime.py) load their own copy of this same package
    under the identical dotted name via importlib.util.spec_from_file_
    location, which replaces the sys.modules entry; a module reference
    captured once at import time would silently go stale in a combined
    test run, making a monkeypatch on it a no-op against whatever object
    diagnostics.run_matched_k11_k12_evaluation's own local `from
    models.dinotext.cover_dr import ...` actually resolves at call time."""
    _warm_up_cover_dr()
    return sys.modules["models.dinotext.cover_dr"]


_real_write_checkpoint_atomically = cli.write_checkpoint_atomically


import contextlib  # noqa: E402


@contextlib.contextmanager
def _patched_nan_metrics():
    """Monkeypatch compute_full_precision_metrics, on whichever module
    object is CURRENTLY live in sys.modules, so its first call returns a
    non-finite mIoU; later calls behave normally. Always restores the
    original function on that same object afterward."""
    module = _current_cover_dr_module()
    real_compute = module.compute_full_precision_metrics
    call_count = {"n": 0}

    def nan_first_call(pre_eval_results):
        call_count["n"] += 1
        result = real_compute(pre_eval_results)
        if call_count["n"] == 1:
            result = dict(result)
            result["mIoU"] = float("nan")
        return result

    module.compute_full_precision_metrics = nan_first_call
    try:
        yield
    finally:
        module.compute_full_precision_metrics = real_compute


# ---------------------------------------------------------------------------
# A. Metric computation raises ValueError (non-finite/degenerate metric)
# ---------------------------------------------------------------------------


def test_A_nonfinite_metric_exits_2_with_clean_diagnostic_no_traceback(tmp_path, capsys):
    with _patched_nan_metrics():
        exit_code, dataset, model = run_synthetic_evaluation(
            checkpoint_path=tmp_path / "c.json", result_path=tmp_path / "r.json", stats_path=tmp_path / "s.json",
        )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "K11/K12 POWER EVALUATION FAIL" in captured.err
    assert "not JSON compliant" in captured.err or "nan" in captured.err.lower()
    assert "Traceback" not in captured.err
    assert not (tmp_path / "r.json").exists()


def test_A_checkpoint_not_marked_complete_after_nonfinite_metric(tmp_path):
    with _patched_nan_metrics():
        run_synthetic_evaluation(
            checkpoint_path=tmp_path / "c.json", result_path=tmp_path / "r.json", stats_path=tmp_path / "s.json",
        )

    import json

    checkpoint = json.loads((tmp_path / "c.json").read_text())
    assert checkpoint["complete"] is False


def test_A_no_stray_temp_output_after_nonfinite_metric(tmp_path):
    with _patched_nan_metrics():
        run_synthetic_evaluation(
            checkpoint_path=tmp_path / "c.json", result_path=tmp_path / "r.json", stats_path=tmp_path / "s.json",
        )

    assert not any(tmp_path.glob("*.selfcheck-*"))
    assert not any(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# B. Strict JSON serialization raises ValueError (independent of the
# metrics path -- the serializer itself is monkeypatched to fail)
# ---------------------------------------------------------------------------


def test_B_json_serialization_valueerror_exits_2_with_clean_diagnostic(tmp_path, capsys):
    def raising_write(path, record):
        if path == tmp_path / "r.json" or str(path).startswith(str(tmp_path / "r.json")):
            raise ValueError("simulated: Out of range float values are not JSON compliant: nan")
        return _real_write_checkpoint_atomically(path, record)

    cli.write_checkpoint_atomically = raising_write
    try:
        exit_code, dataset, model = run_synthetic_evaluation(
            checkpoint_path=tmp_path / "c.json", result_path=tmp_path / "r.json", stats_path=tmp_path / "s.json",
        )
    finally:
        cli.write_checkpoint_atomically = _real_write_checkpoint_atomically

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "K11/K12 POWER EVALUATION FAIL" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "r.json").exists()


def test_B_write_checkpoint_atomically_itself_raises_valueerror_on_nan():
    """Direct, minimal confirmation of the underlying contract this repair
    relies on: the strict (allow_nan=False) JSON writer really does raise
    plain ValueError on a NaN, and this is deliberately NOT changed by the
    repair (see module docstring: do not modify
    compute_full_precision_metrics or the writer's own contract)."""
    import tempfile

    with pytest.raises(ValueError, match="not JSON compliant"):
        with tempfile.TemporaryDirectory() as d:
            _real_write_checkpoint_atomically(Path(d) / "x.json", {"a": float("nan")})


# ---------------------------------------------------------------------------
# C. Existing valid result preservation across both ValueError paths
# ---------------------------------------------------------------------------


def test_C_existing_valid_result_bytes_and_mtime_preserved_after_metric_failure(tmp_path):
    # first, a genuinely successful run establishes a valid sentinel result
    exit_code, _, _ = run_synthetic_evaluation(
        checkpoint_path=tmp_path / "c1.json", result_path=tmp_path / "r.json", stats_path=tmp_path / "s1.json",
    )
    assert exit_code == 0
    original_bytes = (tmp_path / "r.json").read_bytes()
    original_mtime = (tmp_path / "r.json").stat().st_mtime_ns

    with _patched_nan_metrics():
        exit_code2, _, _ = run_synthetic_evaluation(
            checkpoint_path=tmp_path / "c2.json", result_path=tmp_path / "r.json", stats_path=tmp_path / "s2.json",
        )

    assert exit_code2 == 2
    assert (tmp_path / "r.json").read_bytes() == original_bytes
    assert (tmp_path / "r.json").stat().st_mtime_ns == original_mtime


def test_C_existing_valid_result_preserved_after_serialization_failure(tmp_path):
    exit_code, _, _ = run_synthetic_evaluation(
        checkpoint_path=tmp_path / "c1.json", result_path=tmp_path / "r.json", stats_path=tmp_path / "s1.json",
    )
    assert exit_code == 0
    original_bytes = (tmp_path / "r.json").read_bytes()

    def raising_write(path, record):
        if str(path).startswith(str(tmp_path / "r.json")):
            raise ValueError("simulated serialization failure")
        return _real_write_checkpoint_atomically(path, record)

    cli.write_checkpoint_atomically = raising_write
    try:
        exit_code2, _, _ = run_synthetic_evaluation(
            checkpoint_path=tmp_path / "c2.json", result_path=tmp_path / "r.json", stats_path=tmp_path / "s2.json",
        )
    finally:
        cli.write_checkpoint_atomically = _real_write_checkpoint_atomically

    assert exit_code2 == 2
    assert (tmp_path / "r.json").read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# D. Control-flow exclusions: KeyboardInterrupt/SystemExit/MemoryError
# must still propagate; the repair must not widen to a broad catch
# ---------------------------------------------------------------------------


def test_D_keyboardinterrupt_propagates_through_main():
    real_run = cli._run_evaluation

    def raising(args):
        raise KeyboardInterrupt("simulated ctrl-c")

    cli._run_evaluation = raising
    try:
        with pytest.raises(KeyboardInterrupt):
            cli.main([
                "--run-mode", "pilot20", "--checkpoint", "/tmp/c.json", "--result", "/tmp/r.json",
                "--per-image-stats", "/tmp/s.json", "--stability-result", "/tmp/x.json", "--device", "cpu",
            ])
    finally:
        cli._run_evaluation = real_run


def test_D_systemexit_propagates_through_main():
    real_run = cli._run_evaluation

    def raising(args):
        raise SystemExit(7)

    cli._run_evaluation = raising
    try:
        with pytest.raises(SystemExit) as excinfo:
            cli.main([
                "--run-mode", "pilot20", "--checkpoint", "/tmp/c.json", "--result", "/tmp/r.json",
                "--per-image-stats", "/tmp/s.json", "--stability-result", "/tmp/x.json", "--device", "cpu",
            ])
        assert excinfo.value.code == 7
    finally:
        cli._run_evaluation = real_run


def test_D_memoryerror_propagates_through_main():
    real_run = cli._run_evaluation

    def raising(args):
        raise MemoryError("simulated OOM")

    cli._run_evaluation = raising
    try:
        with pytest.raises(MemoryError):
            cli.main([
                "--run-mode", "pilot20", "--checkpoint", "/tmp/c.json", "--result", "/tmp/r.json",
                "--per-image-stats", "/tmp/s.json", "--stability-result", "/tmp/x.json", "--device", "cpu",
            ])
    finally:
        cli._run_evaluation = real_run


def test_D_exception_tuple_is_not_broadened_to_exception_or_baseexception():
    """Static guard on the repair itself: confirm main()'s except clause
    names an exact, bounded tuple of types -- never bare Exception or
    BaseException -- so this test fails loudly if a future edit widens it
    carelessly."""
    import ast
    import inspect

    source = inspect.getsource(cli.main)
    tree = ast.parse(source)
    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
    assert len(handlers) == 1, "expected exactly one except clause in main()"
    handler = handlers[0]
    assert handler.type is not None, "except clause must name explicit types, never a bare except:"
    if isinstance(handler.type, ast.Tuple):
        names = {elt.id for elt in handler.type.elts if isinstance(elt, ast.Name)}
    else:
        names = {handler.type.id} if isinstance(handler.type, ast.Name) else set()
    assert "Exception" not in names
    assert "BaseException" not in names
    assert names == {
        "K11K12PowerEvaluationError", "MatchedPowerEvaluatorError", "K11K12StabilityGateError", "ValueError",
    }, names


# ---------------------------------------------------------------------------
# Subprocess-level confirmation: exit code and absence of a traceback as
# observed from an actual separate process boundary, not just in-process.
# ---------------------------------------------------------------------------


_SUBPROCESS_SCRIPT_TEMPLATE = textwrap.dedent(
    """
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import sys
    sys.path.insert(0, {root!r})
    sys.path.insert(0, {ovs_root!r})
    sys.path.insert(0, {tests_dir!r})
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    import diagnostics.run_matched_k11_k12_evaluation as cli
    from test_matched_power_evaluator_e2e import run_synthetic_evaluation

    # warm up cover_dr, then patch its metrics function to inject NaN
    exit_code, _, _ = run_synthetic_evaluation(
        checkpoint_path={warmup_ckpt!r}, result_path={warmup_res!r}, stats_path={warmup_stats!r},
    )
    assert exit_code == 0

    cover_dr = sys.modules["models.dinotext.cover_dr"]
    real_compute = cover_dr.compute_full_precision_metrics
    call_count = {{"n": 0}}
    def nan_first_call(pre_eval_results):
        call_count["n"] += 1
        result = real_compute(pre_eval_results)
        if call_count["n"] == 1:
            result = dict(result)
            result["mIoU"] = float("nan")
        return result
    cover_dr.compute_full_precision_metrics = nan_first_call

    exit_code, _, _ = run_synthetic_evaluation(
        checkpoint_path={final_ckpt!r}, result_path={final_res!r}, stats_path={final_stats!r},
    )
    raise SystemExit(exit_code)
    """
)


def test_subprocess_nonfinite_metric_exits_2_without_traceback(tmp_path):
    script = tmp_path / "run_subprocess_case.py"
    script.write_text(
        _SUBPROCESS_SCRIPT_TEMPLATE.format(
            root=str(ROOT), ovs_root=str(ROOT / "src/open_vocabulary_segmentation"), tests_dir=str(TESTS_DIR),
            warmup_ckpt=str(tmp_path / "warmup_c.json"), warmup_res=str(tmp_path / "warmup_r.json"),
            warmup_stats=str(tmp_path / "warmup_s.json"),
            final_ckpt=str(tmp_path / "final_c.json"), final_res=str(tmp_path / "final_r.json"),
            final_stats=str(tmp_path / "final_s.json"),
        )
    )
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 2, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "K11/K12 POWER EVALUATION FAIL" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "final_r.json").exists()
