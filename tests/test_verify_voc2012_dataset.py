"""CLI tests for verify_voc2012_dataset.py. Exercises preflight,
generate-manifest, and verify-manifest against the REAL VOC2012 dataset
(never solely synthetic), plus adversarial/failure-contract behavior.
The real dataset root is read from the VOC2012_REAL_DATA_ROOT
environment variable so no machine-specific path is hardcoded here."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import verify_voc2012_dataset as cli  # noqa: E402
from src.voc2012_dataset_identity import Voc2012DatasetIdentityError, load_identity  # noqa: E402
from src.voc2012_dataset_manifest import resolve_dataset_root  # noqa: E402

IDENTITY_PATH = ROOT / "evaluation_identities/e12_voc2012_dataset_source.toml"


def _parse_real_data_root(raw: str | None) -> Path | None:
    """Resolve VOC2012_REAL_DATA_ROOT from its raw string value. Absent or
    whitespace-only text yields None -- never Path(""), which would
    silently resolve to the current working directory and could
    incorrectly activate real-data tests."""
    if raw is None or not raw.strip():
        return None
    return Path(raw).expanduser()


def _real_data_root_ready(root: Path | None, identity_path: Path) -> bool:
    """Exact readiness check: the resolved root must exist, be a
    directory, and satisfy the VOC2012 root contract (via the same
    resolution logic the CLI itself uses) before any real-data test may
    run."""
    if root is None or not identity_path.exists() or not root.is_dir():
        return False
    try:
        identity = load_identity(repo_root=ROOT)
        resolve_dataset_root(root, identity)
    except (Voc2012DatasetIdentityError, OSError):
        return False
    return True


REAL_DATA_ROOT = _parse_real_data_root(os.environ.get("VOC2012_REAL_DATA_ROOT"))
REAL_DATA_AVAILABLE = _real_data_root_ready(REAL_DATA_ROOT, IDENTITY_PATH)
# Applied individually (not as a module-level pytestmark) so the
# env-var-resolution regression tests below run unconditionally --
# a module-level pytestmark would skip them too, defeating their purpose.
requires_real_data = pytest.mark.skipif(
    not REAL_DATA_AVAILABLE,
    reason="requires VOC2012_REAL_DATA_ROOT to point at a directory satisfying the VOC2012 root contract",
)


def test_parse_real_data_root_none_when_absent():
    assert _parse_real_data_root(None) is None


def test_parse_real_data_root_none_when_empty():
    assert _parse_real_data_root("") is None


@pytest.mark.parametrize("raw", ["   ", "\t", "\n"])
def test_parse_real_data_root_none_when_whitespace_only(raw):
    assert _parse_real_data_root(raw) is None


def test_parse_real_data_root_never_resolves_empty_string_to_cwd():
    assert _parse_real_data_root("") is None


@pytest.fixture
def scratch(tmp_path):
    return tmp_path


@requires_real_data
def test_help_does_not_crash():
    proc = subprocess.run([sys.executable, str(ROOT / "verify_voc2012_dataset.py"), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "preflight" in proc.stdout


@requires_real_data
def test_real_preflight_passes(capsys):
    exit_code = cli.main(["preflight", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PREFLIGHT PASS" in captured.out
    assert "image_count=1449" in captured.out


@requires_real_data
def test_real_generate_then_verify_manifest_roundtrip(scratch, capsys):
    output = scratch / "manifest.json"
    exit_code = cli.main([
        "generate-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--output", str(output),
    ])
    assert exit_code == 0
    assert output.exists()

    manifest = json.loads(output.read_text())
    assert manifest["image_count"] == 1449
    assert manifest["observed_label_set"] == list(range(21)) + [255]
    assert "/scratch/" not in json.dumps(manifest)
    assert manifest["source_root_logical_label"] == "VOC2012"

    capsys.readouterr()
    exit_code = cli.main([
        "verify-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--manifest", str(output),
    ])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "VERIFY PASS" in captured.out


@requires_real_data
def test_generate_manifest_refuses_overwrite_without_flag(scratch, capsys):
    output = scratch / "manifest.json"
    cli.main(["generate-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--output", str(output)])
    before = output.read_bytes()

    exit_code = cli.main(["generate-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--output", str(output)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Traceback" not in captured.err
    assert output.read_bytes() == before  # existing valid output preserved
    assert list(scratch.glob("*.tmp*")) == []  # no partial/temp output left


@requires_real_data
def test_generate_manifest_overwrite_flag_allows_regeneration(scratch):
    output = scratch / "manifest.json"
    cli.main(["generate-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--output", str(output)])
    exit_code = cli.main([
        "generate-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--output", str(output), "--overwrite",
    ])
    assert exit_code == 0
    assert list(scratch.glob("*.tmp*")) == []


@requires_real_data
def test_verify_manifest_tampered_content_rejected(scratch, capsys):
    output = scratch / "manifest.json"
    cli.main(["generate-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--output", str(output)])
    manifest = json.loads(output.read_text())
    manifest["image_content_digest"] = "0" * 64
    tampered = scratch / "tampered.json"
    tampered.write_text(json.dumps(manifest))

    exit_code = cli.main(["verify-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--manifest", str(tampered)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Traceback" not in captured.err


@requires_real_data
def test_verify_manifest_float_image_count_rejected(scratch, capsys):
    output = scratch / "manifest.json"
    cli.main(["generate-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--output", str(output)])
    manifest = json.loads(output.read_text())
    manifest["image_count"] = float(manifest["image_count"])
    tampered = scratch / "tampered_float_count.json"
    tampered.write_text(json.dumps(manifest))

    exit_code = cli.main(["verify-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--manifest", str(tampered)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Traceback" not in captured.err


@requires_real_data
def test_verify_manifest_bool_as_int_ignore_pixel_count_rejected(scratch, capsys):
    output = scratch / "manifest.json"
    cli.main(["generate-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--output", str(output)])
    manifest = json.loads(output.read_text())
    manifest["ignore_pixel_count"] = True
    tampered = scratch / "tampered_bool_count.json"
    tampered.write_text(json.dumps(manifest))

    exit_code = cli.main(["verify-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--manifest", str(tampered)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Traceback" not in captured.err


@requires_real_data
def test_verify_manifest_numeric_string_class_count_rejected(scratch, capsys):
    output = scratch / "manifest.json"
    cli.main(["generate-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--output", str(output)])
    manifest = json.loads(output.read_text())
    manifest["v20_class_count"] = str(manifest["v20_class_count"])
    tampered = scratch / "tampered_string_count.json"
    tampered.write_text(json.dumps(manifest))

    exit_code = cli.main(["verify-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--manifest", str(tampered)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Traceback" not in captured.err


@requires_real_data
def test_verify_manifest_float_in_observed_label_set_rejected(scratch, capsys):
    output = scratch / "manifest.json"
    cli.main(["generate-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--output", str(output)])
    manifest = json.loads(output.read_text())
    manifest["observed_label_set"] = [float(v) for v in manifest["observed_label_set"]]
    tampered = scratch / "tampered_float_labels.json"
    tampered.write_text(json.dumps(manifest))

    exit_code = cli.main(["verify-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--manifest", str(tampered)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Traceback" not in captured.err


@requires_real_data
def test_verify_manifest_wrong_container_type_for_dimension_reconciliation_rejected(scratch, capsys):
    output = scratch / "manifest.json"
    cli.main(["generate-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--output", str(output)])
    manifest = json.loads(output.read_text())
    manifest["dimension_reconciliation"]["all_agree"] = 1
    tampered = scratch / "tampered_dim_recon.json"
    tampered.write_text(json.dumps(manifest))

    exit_code = cli.main(["verify-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--manifest", str(tampered)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Traceback" not in captured.err


@requires_real_data
def test_verify_manifest_malformed_json_duplicate_keys_rejected(scratch, capsys):
    bad = scratch / "bad.json"
    bad.write_text('{"a": 1, "a": 2}')
    exit_code = cli.main(["verify-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--manifest", str(bad)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Traceback" not in captured.err


@requires_real_data
def test_verify_manifest_non_finite_json_rejected(scratch, capsys):
    bad = scratch / "bad_nan.json"
    bad.write_text('{"a": NaN}')
    exit_code = cli.main(["verify-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--manifest", str(bad)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Traceback" not in captured.err


@requires_real_data
def test_missing_data_root_rejected(scratch, capsys):
    exit_code = cli.main(["preflight", "--repo-root", str(ROOT), "--data-root", str(scratch / "does-not-exist")])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Traceback" not in captured.err


@requires_real_data
def test_ambiguous_data_root_rejected(scratch, capsys):
    import shutil

    voc_root = scratch / "VOCdevkit" / "VOC2012"
    voc_root.mkdir(parents=True)
    for sub in ("JPEGImages", "SegmentationClass"):
        (voc_root / sub).mkdir()
    (voc_root / "ImageSets" / "Segmentation").mkdir(parents=True)
    for name in ("train.txt", "val.txt", "trainval.txt"):
        (voc_root / "ImageSets" / "Segmentation" / name).write_text("x\n")
    # make the parent ALSO satisfy the root contract
    for sub in ("JPEGImages", "SegmentationClass"):
        shutil.copytree(voc_root / sub, voc_root.parent / sub)
    (voc_root.parent / "ImageSets" / "Segmentation").mkdir(parents=True)
    for name in ("train.txt", "val.txt", "trainval.txt"):
        (voc_root.parent / "ImageSets" / "Segmentation" / name).write_text("x\n")

    exit_code = cli.main(["preflight", "--repo-root", str(ROOT), "--data-root", str(voc_root.parent)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Traceback" not in captured.err


@requires_real_data
def test_keyboard_interrupt_not_swallowed(monkeypatch):
    def raising(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "resolve_dataset_root", raising)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["preflight", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT)])


@requires_real_data
def test_system_exit_not_swallowed(monkeypatch):
    def raising(*a, **k):
        raise SystemExit(7)

    monkeypatch.setattr(cli, "resolve_dataset_root", raising)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["preflight", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT)])
    assert excinfo.value.code == 7


@requires_real_data
def test_real_dataset_and_split_never_modified(scratch):
    split_path = REAL_DATA_ROOT / "VOC2012" / "ImageSets" / "Segmentation" / "val.txt"
    if not split_path.exists():
        split_path = REAL_DATA_ROOT / "ImageSets" / "Segmentation" / "val.txt"
    before = split_path.read_bytes()
    cli.main(["preflight", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT)])
    output = scratch / "manifest.json"
    cli.main(["generate-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--output", str(output)])
    cli.main(["verify-manifest", "--repo-root", str(ROOT), "--data-root", str(REAL_DATA_ROOT), "--manifest", str(output)])
    assert split_path.read_bytes() == before


@requires_real_data
def test_no_torch_cuda_import_in_cli_module():
    """AST-based, not a substring search: confirms verify_voc2012_dataset.py
    and its two source modules never import torch/CUDA/model machinery."""
    for path in (
        ROOT / "verify_voc2012_dataset.py",
        ROOT / "src" / "voc2012_dataset_identity.py",
        ROOT / "src" / "voc2012_dataset_manifest.py",
    ):
        tree = ast.parse(path.read_text(), filename=str(path))
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module.split(".")[0])
        assert "torch" not in imported_names, f"{path} imports torch at module scope"
