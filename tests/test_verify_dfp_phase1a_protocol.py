"""CLI tests for verify_dfp_phase1a_protocol.py. CPU-only, no CUDA, no
dataset access, no model loading. Exercises the real subprocess and the
in-process main() to confirm the exact failure contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import verify_dfp_phase1a_protocol as cli  # noqa: E402

IDENTITY_PATH = ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "verify_dfp_phase1a_protocol.py"), *args],
        capture_output=True, text=True,
    )


def test_help_does_not_crash():
    proc = _run_cli("--help")
    assert proc.returncode == 0
    assert "--identity" in proc.stdout


def test_positive_verification_exact_pass_line():
    proc = _run_cli("--identity", str(IDENTITY_PATH))
    assert proc.returncode == 0
    assert "PASS: DFP Phase 1A protocol identity verified" in proc.stdout


def test_positive_verification_no_traceback():
    proc = _run_cli("--identity", str(IDENTITY_PATH))
    assert "Traceback" not in proc.stdout
    assert "Traceback" not in proc.stderr


def test_missing_identity_file_exit_2_no_traceback(tmp_path):
    proc = _run_cli("--identity", str(tmp_path / "does-not-exist.toml"))
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "PASS" not in proc.stdout


def test_malformed_toml_exit_2_no_traceback(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("this is not [valid toml", encoding="utf-8")
    proc = _run_cli("--identity", str(bad))
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


def test_duplicate_key_toml_exit_2(tmp_path):
    bad = tmp_path / "dup.toml"
    bad.write_text('format_version = "x"\nformat_version = "y"\n', encoding="utf-8")
    proc = _run_cli("--identity", str(bad))
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


def test_wrong_repo_root_exit_2(tmp_path):
    proc = _run_cli("--identity", str(IDENTITY_PATH), "--repo-root", str(tmp_path))
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


def test_no_false_pass_on_failure(tmp_path):
    proc = _run_cli("--identity", str(tmp_path / "does-not-exist.toml"))
    assert "PASS: DFP Phase 1A protocol identity verified" not in proc.stdout


# ---------------------------------------------------------------------------
# In-process: exception propagation contract (main() must catch only
# DfpPhase1aIdentityError/OSError/ValueError; KeyboardInterrupt,
# SystemExit, MemoryError must propagate).
# ---------------------------------------------------------------------------


def test_keyboard_interrupt_not_swallowed(monkeypatch):
    def _raise(*a, **kw):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "validate_static_configuration", _raise)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["--identity", str(IDENTITY_PATH)])


def test_system_exit_not_swallowed(monkeypatch):
    def _raise(*a, **kw):
        raise SystemExit(3)

    monkeypatch.setattr(cli, "validate_static_configuration", _raise)
    with pytest.raises(SystemExit):
        cli.main(["--identity", str(IDENTITY_PATH)])


def test_memory_error_not_swallowed(monkeypatch):
    def _raise(*a, **kw):
        raise MemoryError()

    monkeypatch.setattr(cli, "validate_static_configuration", _raise)
    with pytest.raises(MemoryError):
        cli.main(["--identity", str(IDENTITY_PATH)])


def test_os_error_caught_and_exit_2(monkeypatch):
    def _raise(*a, **kw):
        raise OSError("simulated I/O failure")

    monkeypatch.setattr(cli, "validate_static_configuration", _raise)
    assert cli.main(["--identity", str(IDENTITY_PATH)]) == 2


def test_value_error_caught_and_exit_2(monkeypatch):
    def _raise(*a, **kw):
        raise ValueError("simulated validation failure")

    monkeypatch.setattr(cli, "validate_static_configuration", _raise)
    assert cli.main(["--identity", str(IDENTITY_PATH)]) == 2


def test_generic_exception_not_caught(monkeypatch):
    """The CLI must not catch bare Exception -- only the documented
    domain error, ValueError, and OSError."""

    class _Unrelated(RuntimeError):
        pass

    def _raise(*a, **kw):
        raise _Unrelated("should propagate")

    monkeypatch.setattr(cli, "validate_static_configuration", _raise)
    with pytest.raises(_Unrelated):
        cli.main(["--identity", str(IDENTITY_PATH)])


def test_no_git_check_flag_skips_ancestry(monkeypatch):
    """--no-git-check must still validate everything else."""
    exit_code = cli.main(["--identity", str(IDENTITY_PATH), "--no-git-check"])
    assert exit_code == 0


# ---------------------------------------------------------------------------
# Confirmed-defect regressions at the CLI level: each of these four
# mutations was independently reproduced as an incorrect PASS/exit 0
# before canonical-string pinning was added. Verified here through the
# real subprocess entry point, not just the in-process loader.
# ---------------------------------------------------------------------------


def _write_mutated_identity(tmp_path: Path, old_fragment: str, new_fragment: str) -> Path:
    text = IDENTITY_PATH.read_text(encoding="utf-8")
    assert old_fragment in text, "fixture fragment not found in the real identity file"
    mutated = text.replace(old_fragment, new_fragment, 1)
    path = tmp_path / "mutated_identity.toml"
    path.write_text(mutated, encoding="utf-8")
    return path


def test_cli_rejects_image_global_q_w_definition(tmp_path):
    path = _write_mutated_identity(
        tmp_path,
        'q_w_definition = "Q_w = { a(x), b(x) : x is a stitched T=320 dispute and window w contributes to x }"',
        'q_w_definition = "Q_image = { a(x), b(x) : x is a stitched T=320 dispute in the whole image }"',
    )
    proc = _run_cli("--identity", str(path), "--no-git-check")
    assert proc.returncode == 2
    assert "q_w_definition disagrees" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "PASS" not in proc.stdout


def test_cli_rejects_harmful_beneficial_category_swap(tmp_path):
    path = _write_mutated_identity(
        tmp_path,
        'harmful_washout = "GT(x) == a(x) and GT(x) != b(x)"',
        'harmful_washout = "GT(x) == b(x) and GT(x) != a(x)"',
    )
    proc = _run_cli("--identity", str(path), "--no-git-check")
    assert proc.returncode == 2
    assert "harmful_washout disagrees" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "PASS" not in proc.stdout


def test_cli_rejects_g_recover_sign_reversal(tmp_path):
    path = _write_mutated_identity(
        tmp_path,
        'g_recover_definition = "G_recover = metric(O_P) - metric(P)"',
        'g_recover_definition = "G_recover = metric(P) - metric(O_P)"',
    )
    proc = _run_cli("--identity", str(path), "--no-git-check")
    assert proc.returncode == 2
    assert "g_recover_definition disagrees" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "PASS" not in proc.stdout


def test_cli_rejects_whitespace_padded_q_w_definition(tmp_path):
    path = _write_mutated_identity(
        tmp_path,
        'q_w_definition = "Q_w = { a(x), b(x) : x is a stitched T=320 dispute and window w contributes to x }"',
        'q_w_definition = "  Q_w = { a(x), b(x) : x is a stitched T=320 dispute and window w contributes to x }  "',
    )
    proc = _run_cli("--identity", str(path), "--no-git-check")
    assert proc.returncode == 2
    assert "q_w_definition disagrees" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "PASS" not in proc.stdout
