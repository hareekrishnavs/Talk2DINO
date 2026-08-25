"""CLI tests for diagnostics/run_coco_object_protocol_confirmation.py.
Exercises the entire pre-CUDA path (identity binding, materialization
verify-output subprocess, checkpoint validation) against the REAL,
already-verified production materialization artifact -- CUDA is
unavailable in this CI/dev environment, so --device cuda deterministically
fails closed at the availability check immediately afterward, giving
full coverage of every step this task's implementation phase is allowed
to exercise without ever running model inference."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import diagnostics.run_coco_object_protocol_confirmation as cli  # noqa: E402
from src.coco_object_protocol_confirmation_identity import CocoObjectProtocolConfirmationIdentityError  # noqa: E402

IDENTITY_PATH = ROOT / "evaluation_identities/e12_coco_object_protocol_confirmation.toml"
MANIFEST_PATH = Path("/scratch/haree/coco_object_protocol/manifests/manifest-20443250.json")
DATA_ROOT = Path("/scratch/haree/coco_object_protocol")
SOURCE_MASKS = Path("/scratch/haree/coco_stuff164k/annotations/val2017")
SOURCE_IMAGES = Path("/scratch/haree/coco_stuff164k/images/val2017")

pytestmark = pytest.mark.skipif(
    not (IDENTITY_PATH.exists() and MANIFEST_PATH.exists() and DATA_ROOT.exists()),
    reason="requires the protocol-confirmation identity and the real materialized COCO-Object data",
)


@pytest.fixture
def scratch(tmp_path):
    return tmp_path


def _base_args(scratch, **overrides):
    args = {
        "--repo-root": str(ROOT),
        "--materialization-manifest": str(MANIFEST_PATH),
        "--data-root": str(DATA_ROOT),
        "--source-masks": str(SOURCE_MASKS),
        "--source-images": str(SOURCE_IMAGES),
        "--run-mode": "pilot20",
        "--checkpoint": str(scratch / "checkpoint.json"),
        "--result": str(scratch / "result.json"),
        "--per-image-stats": str(scratch / "per_image_stats.json"),
        "--device": "cuda",
    }
    args.update(overrides)
    flat: list[str] = []
    for key, value in args.items():
        flat.extend([key, value])
    return flat


def test_help_does_not_crash():
    proc = subprocess.run([sys.executable, str(ROOT / "diagnostics/run_coco_object_protocol_confirmation.py"), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "--materialization-manifest" in proc.stdout


def test_full_pre_cuda_path_against_real_production_data_fails_closed_at_cuda_check(scratch, capsys):
    """CUDA is unavailable in this environment: this proves identity
    loading, matched/materialization parent binding, and the real
    verify_coco_object_val_materialization.py verify-output subprocess
    call (re-scanning all 5000 real masks) all succeed against production
    data, with execution stopping cleanly at the CUDA-availability check
    -- never reaching model construction or inference."""
    exit_code = cli.main(_base_args(scratch))
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "CUDA is not available" in captured.err
    assert "Traceback" not in captured.err
    assert not (scratch / "checkpoint.json").exists()
    assert not (scratch / "result.json").exists()


def test_tampered_materialization_manifest_rejected_before_cuda(scratch, capsys):
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest["complete"] = False
    tampered = scratch / "tampered_manifest.json"
    tampered.write_text(json.dumps(manifest))

    exit_code = cli.main(_base_args(scratch, **{"--materialization-manifest": str(tampered)}))
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "materialization manifest.complete" in captured.err
    assert "Traceback" not in captured.err


def test_malformed_checkpoint_on_resume_rejected_before_cuda(scratch, capsys):
    checkpoint_path = scratch / "checkpoint.json"
    checkpoint_path.write_text('{"a": 1, "a": 2}')
    args = _base_args(scratch, **{"--checkpoint": str(checkpoint_path)}) + ["--resume"]
    exit_code = cli.main(args)
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Traceback" not in captured.err


def test_no_stray_output_after_cuda_failure(scratch):
    cli.main(_base_args(scratch))
    assert list(scratch.glob("*")) == []


def test_keyboard_interrupt_not_swallowed(scratch, monkeypatch):
    def raising(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "_verify_materialization_before_cuda", raising)
    with pytest.raises(KeyboardInterrupt):
        cli.main(_base_args(scratch))


def test_system_exit_not_swallowed(scratch, monkeypatch):
    def raising(*a, **k):
        raise SystemExit(11)

    monkeypatch.setattr(cli, "_verify_materialization_before_cuda", raising)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(_base_args(scratch))
    assert excinfo.value.code == 11


def test_source_and_manifest_files_never_modified(scratch):
    before_manifest = MANIFEST_PATH.read_bytes()
    cli.main(_base_args(scratch))
    assert MANIFEST_PATH.read_bytes() == before_manifest


# --- E/F: materialization-identity tampering, exercised through the real
# CLI entrypoint. The mutated materialization identity must live under the
# repository root (parent_identities.materialization_identity_path is
# validated as a safe repo-relative path), so a repo-local scratch
# directory is used and removed unconditionally afterwards; no real
# identity/manifest/mask is ever modified. ---


def _toml_emit(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_emit(v) for v in value) + "]"
    raise TypeError(f"unsupported TOML value type: {type(value)}")


def _dump_flat_toml(document):
    lines = []
    for key, value in document.items():
        if isinstance(value, dict):
            lines.append(f"\n[{key}]")
            for k, v in value.items():
                lines.append(f"{k} = {_toml_emit(v)}")
        else:
            lines.append(f"{key} = {_toml_emit(value)}")
    return "\n".join(lines) + "\n"


def _dump_confirmation_toml(document):
    lines = [f"format_version = {_toml_emit(document['format_version'])}"]
    for section, block in document.items():
        if section == "format_version":
            continue
        if section == "e3_config":
            lines.append("\n[e3_config]")
            for key, value in block.items():
                if key == "resolved_configuration":
                    continue
                lines.append(f"{key} = {_toml_emit(value)}")
            lines.append("\n[e3_config.resolved_configuration]")
            for key, value in block["resolved_configuration"].items():
                lines.append(f"{key} = {_toml_emit(value)}")
            continue
        lines.append(f"\n[{section}]")
        for key, value in block.items():
            lines.append(f"{key} = {_toml_emit(value)}")
    return "\n".join(lines) + "\n"


@pytest.fixture
def repo_scratch():
    scratch_dir = ROOT / "tests" / "_scratch_cli_materialization_binding"
    scratch_dir.mkdir(exist_ok=True)
    try:
        yield scratch_dir
    finally:
        import shutil

        shutil.rmtree(scratch_dir, ignore_errors=True)


def _confirmation_identity_with_materialization_override(scratch_dir, *, materialization_path, materialization_sha256):
    import hashlib
    import tomllib

    with IDENTITY_PATH.open("rb") as handle:
        document = tomllib.load(handle)
    document["parent_identities"] = dict(document["parent_identities"])
    document["parent_identities"]["materialization_identity_path"] = str(
        materialization_path.relative_to(ROOT)
    )
    document["parent_identities"]["materialization_identity_sha256"] = materialization_sha256
    out_path = scratch_dir / "confirmation_materialization_override.toml"
    out_path.write_text(_dump_confirmation_toml(document))
    return out_path


def test_materialization_identity_schema_changed_confirmation_canonical_rejected(scratch, repo_scratch, capsys):
    """E. The materialization identity's OWN manifest.schema_name is
    changed while the confirmation identity stays exactly canonical --
    must be rejected before CUDA/model work, through the real CLI."""
    import hashlib
    import tomllib

    real_materialization_path = ROOT / "evaluation_identities/e12_coco_object_val_materialization.toml"
    real_materialization_bytes_before = real_materialization_path.read_bytes()
    with real_materialization_path.open("rb") as handle:
        mat_document = tomllib.load(handle)
    mat_document["manifest"]["schema_name"] = "talk2dino-coco-object-val-materialization-manifest-v2"
    tampered_mat_path = repo_scratch / "materialization_schema_changed.toml"
    tampered_mat_path.write_text(_dump_flat_toml(mat_document))
    tampered_sha256 = hashlib.sha256(tampered_mat_path.read_bytes()).hexdigest()

    confirmation_path = _confirmation_identity_with_materialization_override(
        repo_scratch, materialization_path=tampered_mat_path, materialization_sha256=tampered_sha256,
    )

    exit_code = cli.main(_base_args(scratch, **{"--identity": str(confirmation_path)}))
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.startswith("COCO-OBJECT PROTOCOL CONFIRMATION FAIL:")
    assert "materialization" in captured.err
    assert "CUDA is not available" not in captured.err
    assert "Traceback" not in captured.err
    assert not (scratch / "checkpoint.json").exists()
    assert not (scratch / "result.json").exists()
    # the real materialization identity/manifest were never touched
    assert real_materialization_path.read_bytes() == real_materialization_bytes_before


def test_malformed_materialization_identity_toml_rejected(scratch, repo_scratch, capsys):
    """F. Malformed TOML for the materialization identity -- exit 2, no
    traceback, through the real CLI, before CUDA/model work."""
    tampered_mat_path = repo_scratch / "materialization_malformed.toml"
    tampered_mat_path.write_text("this is not [valid toml")
    import hashlib

    tampered_sha256 = hashlib.sha256(tampered_mat_path.read_bytes()).hexdigest()

    confirmation_path = _confirmation_identity_with_materialization_override(
        repo_scratch, materialization_path=tampered_mat_path, materialization_sha256=tampered_sha256,
    )

    exit_code = cli.main(_base_args(scratch, **{"--identity": str(confirmation_path)}))
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.startswith("COCO-OBJECT PROTOCOL CONFIRMATION FAIL:")
    assert "Traceback" not in captured.err
    assert not (scratch / "checkpoint.json").exists()
    assert not (scratch / "result.json").exists()


def test_materialization_identity_missing_manifest_section_rejected(scratch, repo_scratch, capsys):
    """F. A materialization identity missing the required [manifest]
    schema-name section -- exit 2, no traceback, through the real CLI."""
    import hashlib
    import tomllib

    real_materialization_path = ROOT / "evaluation_identities/e12_coco_object_val_materialization.toml"
    with real_materialization_path.open("rb") as handle:
        mat_document = tomllib.load(handle)
    mat_document.pop("manifest")
    tampered_mat_path = repo_scratch / "materialization_missing_manifest_section.toml"
    tampered_mat_path.write_text(_dump_flat_toml(mat_document))
    tampered_sha256 = hashlib.sha256(tampered_mat_path.read_bytes()).hexdigest()

    confirmation_path = _confirmation_identity_with_materialization_override(
        repo_scratch, materialization_path=tampered_mat_path, materialization_sha256=tampered_sha256,
    )

    exit_code = cli.main(_base_args(scratch, **{"--identity": str(confirmation_path)}))
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.startswith("COCO-OBJECT PROTOCOL CONFIRMATION FAIL:")
    assert "Traceback" not in captured.err
    assert not (scratch / "checkpoint.json").exists()
    assert not (scratch / "result.json").exists()
