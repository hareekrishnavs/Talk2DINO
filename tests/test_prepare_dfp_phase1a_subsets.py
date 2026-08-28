"""CLI failure-contract tests for prepare_dfp_phase1a_subsets.py.
CPU-only, no CUDA, no real dataset access. Exercises the real subprocess
where the exact exit/stdout/stderr contract must be verified, and the
in-process functions for exception-propagation and preservation checks.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import prepare_dfp_phase1a_subsets as cli  # noqa: E402
import src.dfp_phase1a_subset_manifest as m  # noqa: E402


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "prepare_dfp_phase1a_subsets.py"), *args],
        capture_output=True, text=True,
    )


def test_help_does_not_crash():
    proc = _run_cli("--help")
    assert proc.returncode == 0


def test_generate_missing_env_root_exit_2(tmp_path, monkeypatch):
    monkeypatch.delenv("COCO_STUFF_REAL_DATA_ROOT", raising=False)
    proc = _run_cli("generate", "--dataset", "coco_stuff", "--output", str(tmp_path / "out.json"), "--no-git-check")
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "PASS" not in proc.stdout


def test_generate_root_is_a_file_exit_2(tmp_path):
    f = tmp_path / "not_a_dir"
    f.write_text("x")
    proc = _run_cli(
        "generate", "--dataset", "coco_stuff", "--output", str(tmp_path / "out.json"),
        "--data-root", str(f), "--no-git-check",
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


def test_generate_missing_root_exit_2(tmp_path):
    proc = _run_cli(
        "generate", "--dataset", "coco_stuff", "--output", str(tmp_path / "out.json"),
        "--data-root", str(tmp_path / "does-not-exist"), "--no-git-check",
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


def test_generate_incomplete_root_exit_2(tmp_path):
    # a directory that exists but lacks images/val2017 and annotations/val2017
    (tmp_path / "some_other_dir").mkdir()
    proc = _run_cli(
        "generate", "--dataset", "coco_stuff", "--output", str(tmp_path / "out.json"),
        "--data-root", str(tmp_path), "--no-git-check",
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


def test_generate_refuses_overwrite_without_flag(tmp_path):
    existing = tmp_path / "out.json"
    existing.write_text('{"already": "there"}\n', encoding="utf-8")
    before = existing.read_bytes()
    proc = _run_cli(
        "generate", "--dataset", "coco_stuff", "--output", str(existing),
        "--data-root", str(tmp_path), "--no-git-check",
    )
    assert proc.returncode == 2
    assert "--overwrite" in proc.stderr
    assert existing.read_bytes() == before


def test_verify_missing_manifest_exit_2(tmp_path):
    proc = _run_cli("verify", "--dataset", "coco_stuff", "--manifest", str(tmp_path / "nope.json"), "--no-git-check")
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "PASS" not in proc.stdout


def test_verify_malformed_json_exit_2(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("this is not { valid json", encoding="utf-8")
    proc = _run_cli("verify", "--dataset", "coco_stuff", "--manifest", str(bad), "--no-git-check")
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


def test_verify_duplicate_key_json_exit_2(tmp_path):
    bad = tmp_path / "dup.json"
    bad.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    proc = _run_cli("verify", "--dataset", "coco_stuff", "--manifest", str(bad), "--no-git-check")
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


def test_verify_wrong_schema_exit_2(tmp_path):
    bad = tmp_path / "wrong.json"
    bad.write_text('{"x": 1}\n', encoding="utf-8")
    proc = _run_cli("verify", "--dataset", "coco_stuff", "--manifest", str(bad), "--no-git-check")
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


def test_verify_source_binding_missing_env_root_exit_2(tmp_path, monkeypatch):
    monkeypatch.delenv("COCO_STUFF_REAL_DATA_ROOT", raising=False)
    bad = tmp_path / "wrong.json"
    bad.write_text('{"x": 1}\n', encoding="utf-8")
    proc = _run_cli("verify-source-binding", "--dataset", "coco_stuff", "--manifest", str(bad), "--no-git-check")
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


# ---------------------------------------------------------------------------
# In-process: exception propagation contract
# ---------------------------------------------------------------------------


def test_keyboard_interrupt_not_swallowed(monkeypatch, tmp_path):
    def _raise(*a, **kw):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "_build_manifest_for_dataset", _raise)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["generate", "--dataset", "coco_stuff", "--output", str(tmp_path / "out.json"), "--no-git-check"])


def test_system_exit_not_swallowed(monkeypatch, tmp_path):
    def _raise(*a, **kw):
        raise SystemExit(3)

    monkeypatch.setattr(cli, "_build_manifest_for_dataset", _raise)
    with pytest.raises(SystemExit):
        cli.main(["generate", "--dataset", "coco_stuff", "--output", str(tmp_path / "out.json"), "--no-git-check"])


def test_memory_error_not_swallowed(monkeypatch, tmp_path):
    def _raise(*a, **kw):
        raise MemoryError()

    monkeypatch.setattr(cli, "_build_manifest_for_dataset", _raise)
    with pytest.raises(MemoryError):
        cli.main(["generate", "--dataset", "coco_stuff", "--output", str(tmp_path / "out.json"), "--no-git-check"])


def test_generic_exception_not_caught(monkeypatch, tmp_path):
    class _Unrelated(RuntimeError):
        pass

    def _raise(*a, **kw):
        raise _Unrelated("should propagate")

    monkeypatch.setattr(cli, "_build_manifest_for_dataset", _raise)
    with pytest.raises(_Unrelated):
        cli.main(["generate", "--dataset", "coco_stuff", "--output", str(tmp_path / "out.json"), "--no-git-check"])


# ---------------------------------------------------------------------------
# Existing-result preservation / no temporary residue on a failed generate
# ---------------------------------------------------------------------------


def test_failed_generate_preserves_existing_output_and_leaves_no_residue(monkeypatch, tmp_path):
    existing = tmp_path / "out.json"
    existing.write_text('{"already": "there", "valid": true}\n', encoding="utf-8")
    before = existing.read_bytes()

    def _raise(*a, **kw):
        raise m.DfpPhase1aSubsetManifestError("simulated failure after --overwrite was accepted")

    monkeypatch.setattr(cli, "_build_manifest_for_dataset", _raise)
    exit_code = cli.main([
        "generate", "--dataset", "coco_stuff", "--output", str(existing), "--overwrite", "--no-git-check",
    ])
    assert exit_code == 2
    assert existing.read_bytes() == before
    assert list(tmp_path.glob("*.tmp-*")) == []


# ---------------------------------------------------------------------------
# Narrow repair regressions
# ---------------------------------------------------------------------------


def _real_manifest_path(dataset_key: str) -> Path:
    return {
        "coco_stuff": ROOT / "evaluation_subsets/dfp_phase1a_coco_stuff_500.json",
        "ade20k": ROOT / "evaluation_subsets/dfp_phase1a_ade20k_500.json",
    }[dataset_key]


def test_cli_rejects_wrong_canonical_source_order_assertion(tmp_path):
    import json

    with open(_real_manifest_path("coco_stuff")) as f:
        doc = json.load(f)
    doc["canonical_source_order_assertion"] = "a completely different assertion"
    mutated = tmp_path / "mutated.json"
    with open(mutated, "w") as f:
        json.dump(doc, f)
    proc = _run_cli("verify", "--dataset", "coco_stuff", "--manifest", str(mutated), "--no-git-check")
    assert proc.returncode == 2
    assert "canonical_source_order_assertion disagrees" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "PASS" not in proc.stdout


def test_cli_rejects_reversed_selected_ids_with_recomputed_digest(tmp_path):
    import hashlib
    import json

    with open(_real_manifest_path("coco_stuff")) as f:
        doc = json.load(f)
    reversed_ids = list(reversed(doc["selected_ids"]))
    assert reversed_ids != doc["selected_ids"]
    doc["selected_ids"] = reversed_ids
    doc["selected_ids_digest"] = hashlib.sha256(json.dumps(reversed_ids, ensure_ascii=True).encode("utf-8")).hexdigest()
    mutated = tmp_path / "mutated.json"
    with open(mutated, "w") as f:
        json.dump(doc, f)
    proc = _run_cli("verify", "--dataset", "coco_stuff", "--manifest", str(mutated), "--no-git-check")
    assert proc.returncode == 2
    assert "selected_ids" in proc.stderr
    assert "canonical source order" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "PASS" not in proc.stdout


def test_cli_verify_still_accepts_real_frozen_manifests():
    proc = _run_cli("verify", "--dataset", "coco_stuff", "--manifest", str(_real_manifest_path("coco_stuff")), "--no-git-check")
    assert proc.returncode == 0
    assert "PASS" in proc.stdout
    proc2 = _run_cli("verify", "--dataset", "ade20k", "--manifest", str(_real_manifest_path("ade20k")), "--no-git-check")
    assert proc2.returncode == 0
    assert "PASS" in proc2.stdout


# ---------------------------------------------------------------------------
# Blocker-file output-parent handling
# ---------------------------------------------------------------------------


def test_blocker_file_parent_exits_2_clean_no_artifact_no_residue(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    proc = _run_cli(
        "generate", "--dataset", "coco_stuff", "--output", str(blocker / "out.json"),
        "--data-root", str(tmp_path), "--no-git-check",
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "PASS" not in proc.stdout
    # no output artifact was created (blocker is still a plain file)
    assert blocker.is_file()
    assert not blocker.is_dir()
    # no temp residue anywhere under tmp_path
    assert list(tmp_path.rglob("*.tmp-*")) == []


def test_blocker_file_parent_preserves_unrelated_existing_manifest(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text('{"already": "there"}\n', encoding="utf-8")
    before = unrelated.read_bytes()
    _run_cli(
        "generate", "--dataset", "coco_stuff", "--output", str(blocker / "out.json"),
        "--data-root", str(tmp_path), "--no-git-check",
    )
    assert unrelated.read_bytes() == before


def test_missing_output_parent_directory_is_still_created(monkeypatch, tmp_path):
    target = tmp_path / "does" / "not" / "exist" / "yet" / "out.json"
    assert not target.parent.exists()

    def _fake_build(**kw):
        return {"fake": "manifest"}

    monkeypatch.setattr(cli, "_build_manifest_for_dataset", lambda **kw: {"schema": "x", "selected_count": 500, "source_validation_count": 5000, "selected_ids_digest": "a" * 64})
    monkeypatch.setattr(cli, "verify_manifest_document", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_protocol_and_source_identity_sha256s", lambda *a, **kw: ("a" * 64, "b" * 64, "c" * 64))
    exit_code = cli.main(["generate", "--dataset", "coco_stuff", "--output", str(target), "--no-git-check"])
    assert exit_code == 0
    assert target.parent.is_dir()
    assert target.is_file()


def test_normal_existing_output_directory_succeeds(monkeypatch, tmp_path):
    target = tmp_path / "out.json"
    assert tmp_path.is_dir()
    monkeypatch.setattr(cli, "_build_manifest_for_dataset", lambda **kw: {"schema": "x", "selected_count": 500, "source_validation_count": 5000, "selected_ids_digest": "a" * 64})
    monkeypatch.setattr(cli, "verify_manifest_document", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_protocol_and_source_identity_sha256s", lambda *a, **kw: ("a" * 64, "b" * 64, "c" * 64))
    exit_code = cli.main(["generate", "--dataset", "coco_stuff", "--output", str(target), "--no-git-check"])
    assert exit_code == 0
    assert target.is_file()


def test_permission_denied_output_parent_fails_cleanly(monkeypatch, tmp_path):
    readonly_parent = tmp_path / "readonly"
    readonly_parent.mkdir(mode=0o500)
    try:
        proc = _run_cli(
            "generate", "--dataset", "coco_stuff", "--output", str(readonly_parent / "sub" / "out.json"),
            "--data-root", str(tmp_path), "--no-git-check",
        )
        assert proc.returncode == 2
        assert "Traceback" not in proc.stderr
        assert "PASS" not in proc.stdout
        assert list(readonly_parent.rglob("*.tmp-*")) == []
    finally:
        readonly_parent.chmod(0o700)


def test_serialization_value_error_remains_clean(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_build_manifest_for_dataset", lambda **kw: {"nan_value": float("nan")})
    monkeypatch.setattr(cli, "verify_manifest_document", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_protocol_and_source_identity_sha256s", lambda *a, **kw: ("a" * 64, "b" * 64, "c" * 64))
    exit_code = cli.main(["generate", "--dataset", "coco_stuff", "--output", str(tmp_path / "out.json"), "--no-git-check"])
    assert exit_code == 2
    assert not (tmp_path / "out.json").exists()
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_keyboard_interrupt_still_propagates_through_mkdir_and_write(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_build_manifest_for_dataset", lambda **kw: {"schema": "x", "selected_count": 500, "source_validation_count": 5000, "selected_ids_digest": "a" * 64})
    monkeypatch.setattr(cli, "verify_manifest_document", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_protocol_and_source_identity_sha256s", lambda *a, **kw: ("a" * 64, "b" * 64, "c" * 64))

    def _boom(*a, **kw):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "write_json_atomically", _boom)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["generate", "--dataset", "coco_stuff", "--output", str(tmp_path / "out.json"), "--no-git-check"])


def test_system_exit_still_propagates_through_mkdir_and_write(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_build_manifest_for_dataset", lambda **kw: {"schema": "x", "selected_count": 500, "source_validation_count": 5000, "selected_ids_digest": "a" * 64})
    monkeypatch.setattr(cli, "verify_manifest_document", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_protocol_and_source_identity_sha256s", lambda *a, **kw: ("a" * 64, "b" * 64, "c" * 64))

    def _boom(*a, **kw):
        raise SystemExit(9)

    monkeypatch.setattr(cli, "write_json_atomically", _boom)
    with pytest.raises(SystemExit):
        cli.main(["generate", "--dataset", "coco_stuff", "--output", str(tmp_path / "out.json"), "--no-git-check"])


def test_memory_error_still_propagates_through_mkdir_and_write(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_build_manifest_for_dataset", lambda **kw: {"schema": "x", "selected_count": 500, "source_validation_count": 5000, "selected_ids_digest": "a" * 64})
    monkeypatch.setattr(cli, "verify_manifest_document", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_protocol_and_source_identity_sha256s", lambda *a, **kw: ("a" * 64, "b" * 64, "c" * 64))

    def _boom(*a, **kw):
        raise MemoryError()

    monkeypatch.setattr(cli, "write_json_atomically", _boom)
    with pytest.raises(MemoryError):
        cli.main(["generate", "--dataset", "coco_stuff", "--output", str(tmp_path / "out.json"), "--no-git-check"])
