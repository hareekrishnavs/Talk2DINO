"""Tests for verify_coco_object_val_materialization.py -- the independent,
manifest-distrusting verifier. Reuses the same tiny from-scratch identity
pattern as test_run_materialize_coco_object_val.py so the full
verify-output path (including the independent spot-check re-rasterization)
can be exercised without ever running the real 5000-image conversion."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import materialize_coco_object_val as materialize_cli  # noqa: E402
import verify_coco_object_val_materialization as verify_cli  # noqa: E402
from src.coco_object_val_materialization_identity import IDENTITY_RELATIVE_PATH  # noqa: E402
import src.coco_object_val_materialization_identity as identity_module  # noqa: E402

REAL_IDENTITY_PATH = ROOT / IDENTITY_RELATIVE_PATH
pytestmark = pytest.mark.skipif(not REAL_IDENTITY_PATH.exists(), reason="requires the materialization identity")

RAW_ID_TO_BACKGROUND = 91
RAW_ID_TO_CLASS_ONE = 0
RAW_ID_TO_CLASS_TWO = 1


def _manual_toml_dump(document: dict) -> str:
    lines: list[str] = []

    def emit_value(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, list):
            return "[" + ", ".join(emit_value(v) for v in value) + "]"
        raise TypeError(f"unsupported TOML value type: {type(value)}")

    lines.append(f"format_version = {emit_value(document['format_version'])}")
    for section, block in document.items():
        if section == "format_version":
            continue
        lines.append(f"\n[{section}]")
        for key, value in block.items():
            lines.append(f"{key} = {emit_value(value)}")
    return "\n".join(lines) + "\n"


@pytest.fixture
def tiny_identity(tmp_path, monkeypatch, request):
    image_count = getattr(request, "param", 5)
    train_count = 7
    with REAL_IDENTITY_PATH.open("rb") as handle:
        document = tomllib.load(handle)
    document["protocol"]["expected_image_count"] = image_count
    document["source"]["val_mask_count_expected"] = image_count
    document["source"]["train_mask_count_expected"] = train_count
    document["source"]["coco_len_total"] = train_count + image_count
    path = tmp_path / "tiny_identity.toml"
    path.write_text(_manual_toml_dump(document))
    monkeypatch.setattr(identity_module, "SUPPORTED_EXPECTED_IMAGE_COUNT", image_count)
    monkeypatch.setattr(identity_module, "SUPPORTED_TRAIN_MASK_COUNT", train_count)
    monkeypatch.setattr(identity_module, "SUPPORTED_COCO_LEN_TOTAL", train_count + image_count)
    return path, image_count


def _write_synthetic_source(tmp_path, image_count: int):
    source_masks = tmp_path / "annotations" / "val2017"
    source_masks.mkdir(parents=True)
    source_images = tmp_path / "images" / "val2017"
    source_images.mkdir(parents=True)
    ids = [f"{i:012d}" for i in range(1, image_count + 1)]
    for i, image_id in enumerate(ids):
        raw = np.full((3, 3), RAW_ID_TO_BACKGROUND, dtype=np.uint8)
        raw[0, 0] = RAW_ID_TO_CLASS_ONE if i % 2 == 0 else RAW_ID_TO_CLASS_TWO
        Image.fromarray(raw, mode="L").save(source_masks / f"{image_id}.png", "PNG")
        Image.new("RGB", (3, 3), color=(10, 20, 30)).save(source_images / f"{image_id}.jpg", "JPEG")
    return source_masks, source_images, ids


@pytest.fixture
def materialized(tmp_path, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, ids = _write_synthetic_source(tmp_path, image_count)
    output_root = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint.json"
    manifest = tmp_path / "manifest.json"
    exit_code = materialize_cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(source_masks), "--source-images", str(source_images),
        "--output-root", str(output_root), "--checkpoint", str(checkpoint), "--manifest", str(manifest),
    ])
    assert exit_code == 0
    return {
        "identity_path": identity_path, "source_masks": source_masks, "source_images": source_images,
        "output_root": output_root, "checkpoint": checkpoint, "manifest": manifest, "ids": ids,
    }


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def test_preflight_pass_against_real_repo():
    exit_code = verify_cli.main(["preflight", "--repo-root", str(ROOT)])
    assert exit_code == 0


def test_preflight_fails_on_wrong_source_masks_count(tmp_path, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, _ = _write_synthetic_source(tmp_path, image_count)
    (source_masks / "extra_999999999999.png").write_bytes(b"\x00")
    exit_code = verify_cli.main([
        "--identity", str(identity_path), "preflight", "--repo-root", str(ROOT), "--source-masks", str(source_masks),
    ])
    assert exit_code == 2


def test_preflight_fails_on_output_overlap(tmp_path, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, _ = _write_synthetic_source(tmp_path, image_count)
    exit_code = verify_cli.main([
        "--identity", str(identity_path), "preflight", "--repo-root", str(ROOT),
        "--source-masks", str(source_masks), "--output-root", str(source_masks),
    ])
    assert exit_code == 2


# ---------------------------------------------------------------------------
# verify-checkpoint
# ---------------------------------------------------------------------------


def test_verify_checkpoint_pass_on_valid_checkpoint(materialized):
    exit_code = verify_cli.main([
        "--identity", str(materialized["identity_path"]), "verify-checkpoint",
        "--repo-root", str(ROOT), "--checkpoint", str(materialized["checkpoint"]),
        "--source-images", str(materialized["source_images"]),
    ])
    assert exit_code == 0


def test_verify_checkpoint_fails_on_tampered_order_digest(materialized):
    doc = json.loads(materialized["checkpoint"].read_text())
    doc["image_order_digest"] = "0" * 64
    materialized["checkpoint"].write_text(json.dumps(doc))
    exit_code = verify_cli.main([
        "--identity", str(materialized["identity_path"]), "verify-checkpoint",
        "--repo-root", str(ROOT), "--checkpoint", str(materialized["checkpoint"]),
        "--source-images", str(materialized["source_images"]),
    ])
    assert exit_code == 2


# ---------------------------------------------------------------------------
# verify-output -- must not trust the manifest's own claims
# ---------------------------------------------------------------------------


def test_verify_output_pass_on_valid_materialization(materialized):
    exit_code = verify_cli.main([
        "--identity", str(materialized["identity_path"]), "verify-output",
        "--repo-root", str(ROOT), "--manifest", str(materialized["manifest"]),
        "--output-root", str(materialized["output_root"]), "--source-masks", str(materialized["source_masks"]),
        "--source-images", str(materialized["source_images"]),
    ])
    assert exit_code == 0


def test_verify_output_detects_missing_mask(materialized):
    mask_path = materialized["output_root"] / "annotations" / "val2017" / f"{materialized['ids'][0]}_instanceTrainIds.png"
    mask_path.unlink()
    exit_code = verify_cli.main([
        "--identity", str(materialized["identity_path"]), "verify-output",
        "--repo-root", str(ROOT), "--manifest", str(materialized["manifest"]),
        "--output-root", str(materialized["output_root"]), "--source-masks", str(materialized["source_masks"]),
        "--source-images", str(materialized["source_images"]),
    ])
    assert exit_code == 2


def test_verify_output_detects_extra_train_like_file(materialized):
    extra = materialized["output_root"] / "annotations" / "val2017" / "999999999999_train_instanceTrainIds.png"
    extra.write_bytes(b"\x00")
    exit_code = verify_cli.main([
        "--identity", str(materialized["identity_path"]), "verify-output",
        "--repo-root", str(ROOT), "--manifest", str(materialized["manifest"]),
        "--output-root", str(materialized["output_root"]), "--source-masks", str(materialized["source_masks"]),
        "--source-images", str(materialized["source_images"]),
    ])
    assert exit_code == 2


def test_verify_output_detects_tampered_pixel_content(materialized):
    """The manifest's own claims (histogram/digests) are stale after this
    tamper -- verify-output must catch it by independent recomputation,
    not by trusting the manifest."""
    mask_path = materialized["output_root"] / "annotations" / "val2017" / f"{materialized['ids'][0]}_instanceTrainIds.png"
    arr = np.array(Image.open(mask_path))
    arr[0, 0] = (arr[0, 0] + 1) % 81
    Image.fromarray(arr, mode="L").save(mask_path, "PNG")
    exit_code = verify_cli.main([
        "--identity", str(materialized["identity_path"]), "verify-output",
        "--repo-root", str(ROOT), "--manifest", str(materialized["manifest"]),
        "--output-root", str(materialized["output_root"]), "--source-masks", str(materialized["source_masks"]),
        "--source-images", str(materialized["source_images"]),
    ])
    assert exit_code == 2


def test_verify_output_detects_tampered_source_mask(materialized):
    """If a source raw mask changes after materialization, the
    independent spot-check re-rasterization must catch the drift."""
    raw_path = materialized["source_masks"] / f"{materialized['ids'][0]}.png"
    raw = np.array(Image.open(raw_path))
    raw[:] = RAW_ID_TO_CLASS_TWO if raw[0, 0] != RAW_ID_TO_CLASS_TWO else RAW_ID_TO_CLASS_ONE
    Image.fromarray(raw, mode="L").save(raw_path, "PNG")
    exit_code = verify_cli.main([
        "--identity", str(materialized["identity_path"]), "verify-output",
        "--repo-root", str(ROOT), "--manifest", str(materialized["manifest"]),
        "--output-root", str(materialized["output_root"]), "--source-masks", str(materialized["source_masks"]),
        "--source-images", str(materialized["source_images"]),
    ])
    assert exit_code == 2


def test_verify_output_rejects_incomplete_manifest(materialized):
    doc = json.loads(materialized["manifest"].read_text())
    doc["complete"] = False
    materialized["manifest"].write_text(json.dumps(doc))
    exit_code = verify_cli.main([
        "--identity", str(materialized["identity_path"]), "verify-output",
        "--repo-root", str(ROOT), "--manifest", str(materialized["manifest"]),
        "--output-root", str(materialized["output_root"]), "--source-masks", str(materialized["source_masks"]),
        "--source-images", str(materialized["source_images"]),
    ])
    assert exit_code == 2


def test_verify_output_independent_spot_check_runs(materialized, capsys):
    exit_code = verify_cli.main([
        "--identity", str(materialized["identity_path"]), "verify-output",
        "--repo-root", str(ROOT), "--manifest", str(materialized["manifest"]),
        "--output-root", str(materialized["output_root"]), "--source-masks", str(materialized["source_masks"]),
        "--source-images", str(materialized["source_images"]),
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "spot_checked=" in captured.out


# ---------------------------------------------------------------------------
# CLI hygiene
# ---------------------------------------------------------------------------


def test_verify_no_traceback_on_expected_failure(materialized, capsys):
    materialized["manifest"].write_text('{"a": 1, "a": 2}')
    exit_code = verify_cli.main([
        "--identity", str(materialized["identity_path"]), "verify-output",
        "--repo-root", str(ROOT), "--manifest", str(materialized["manifest"]),
        "--output-root", str(materialized["output_root"]), "--source-masks", str(materialized["source_masks"]),
        "--source-images", str(materialized["source_images"]),
    ])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
