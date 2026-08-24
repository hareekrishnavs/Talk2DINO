"""End-to-end CLI tests for materialize_coco_object_val.py. Never runs the
real 5000-image conversion (forbidden during implementation) -- the
'full production-like run' tests instead build a tiny, from-scratch
identity (same real converter/dataset-class/dataset-config hashes, a
small expected_image_count) and monkeypatch the identity module's
SUPPORTED_* constants to match, exercising the exact same code path a
real 5000-image run would take."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import materialize_coco_object_val as cli  # noqa: E402
from src.coco_object_val_materialization_identity import IDENTITY_RELATIVE_PATH  # noqa: E402
import src.coco_object_val_materialization_identity as identity_module  # noqa: E402

REAL_IDENTITY_PATH = ROOT / IDENTITY_RELATIVE_PATH
pytestmark = pytest.mark.skipif(not REAL_IDENTITY_PATH.exists(), reason="requires the materialization identity")

# clsID_to_trID raw values (from the real, hash-verified converter table)
RAW_ID_TO_BACKGROUND = 91  # -> mapped 0
RAW_ID_TO_CLASS_ONE = 0  # -> mapped 1
RAW_ID_TO_CLASS_TWO = 1  # -> mapped 2


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


def _tiny_identity_document(image_count: int, *, train_count: int) -> dict:
    with REAL_IDENTITY_PATH.open("rb") as handle:
        document = tomllib.load(handle)
    document["protocol"]["expected_image_count"] = image_count
    document["source"]["val_mask_count_expected"] = image_count
    document["source"]["train_mask_count_expected"] = train_count
    document["source"]["coco_len_total"] = train_count + image_count
    return document


@pytest.fixture
def tiny_identity(tmp_path, monkeypatch, request):
    image_count = getattr(request, "param", 3)
    train_count = 7  # arbitrary small stand-in for the real 118287, consistently patched below
    document = _tiny_identity_document(image_count, train_count=train_count)
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
def scratch(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# Full production-like run (tiny canonical count) -- manifest path
# ---------------------------------------------------------------------------


def test_full_run_produces_valid_manifest(scratch, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, ids = _write_synthetic_source(scratch, image_count)
    output_root = scratch / "output"
    checkpoint = scratch / "checkpoint.json"
    manifest = scratch / "manifest.json"

    exit_code = cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(source_masks), "--source-images", str(source_images),
        "--output-root", str(output_root), "--checkpoint", str(checkpoint), "--manifest", str(manifest),
    ])
    assert exit_code == 0
    assert manifest.is_file()
    document = json.loads(manifest.read_text())
    assert document["complete"] is True
    assert document["final"] is True
    assert document["image_count"] == image_count
    assert document["no_train_masks_generated"] is True
    assert document["no_source_mutation"] is True

    for image_id in ids:
        assert (output_root / "annotations" / "val2017" / f"{image_id}_instanceTrainIds.png").is_file()
    assert (output_root / "images" / "val2017").is_symlink()
    assert Path(output_root / "images" / "val2017").resolve() == source_images.resolve()


def test_deterministic_manifest_digest_two_independent_runs(scratch, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, ids = _write_synthetic_source(scratch, image_count)

    def run(tag):
        output_root = scratch / f"output_{tag}"
        checkpoint = scratch / f"checkpoint_{tag}.json"
        manifest = scratch / f"manifest_{tag}.json"
        exit_code = cli.main([
            "--repo-root", str(ROOT), "--identity", str(identity_path),
            "--source-masks", str(source_masks), "--source-images", str(source_images),
            "--output-root", str(output_root), "--checkpoint", str(checkpoint), "--manifest", str(manifest),
        ])
        assert exit_code == 0
        return json.loads(manifest.read_text())

    manifest_a = run("a")
    manifest_b = run("b")
    diff_keys = {k for k in manifest_a if manifest_a[k] != manifest_b.get(k)}
    assert diff_keys <= {"created_at_utc"}


# ---------------------------------------------------------------------------
# Val-only / split safety
# ---------------------------------------------------------------------------


def test_train_split_source_masks_rejected(scratch, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, _ = _write_synthetic_source(scratch, image_count)
    train_masks = scratch / "annotations" / "train2017"
    train_masks.mkdir()
    exit_code = cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(train_masks), "--source-images", str(source_images),
        "--output-root", str(scratch / "output"), "--checkpoint", str(scratch / "c.json"), "--manifest", str(scratch / "m.json"),
    ])
    assert exit_code == 2


def test_arbitrary_split_name_rejected(scratch, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, _ = _write_synthetic_source(scratch, image_count)
    weird = scratch / "annotations" / "test2099"
    weird.mkdir()
    exit_code = cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(weird), "--source-images", str(source_images),
        "--output-root", str(scratch / "output"), "--checkpoint", str(scratch / "c.json"), "--manifest", str(scratch / "m.json"),
    ])
    assert exit_code == 2


def test_wrong_image_count_rejected(scratch, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, ids = _write_synthetic_source(scratch, image_count)
    # add one extra image so the count no longer matches the identity's expectation
    extra_id = f"{image_count + 1:012d}"
    Image.new("RGB", (3, 3)).save(source_images / f"{extra_id}.jpg", "JPEG")
    exit_code = cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(source_masks), "--source-images", str(source_images),
        "--output-root", str(scratch / "output"), "--checkpoint", str(scratch / "c.json"), "--manifest", str(scratch / "m.json"),
    ])
    assert exit_code == 2


def test_missing_raw_mask_for_canonical_image_rejected(scratch, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, ids = _write_synthetic_source(scratch, image_count)
    (source_masks / f"{ids[0]}.png").unlink()
    exit_code = cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(source_masks), "--source-images", str(source_images),
        "--output-root", str(scratch / "output"), "--checkpoint", str(scratch / "c.json"), "--manifest", str(scratch / "m.json"),
    ])
    assert exit_code == 2


# ---------------------------------------------------------------------------
# Output-root / overlap safety
# ---------------------------------------------------------------------------


def test_output_root_overlapping_source_rejected(scratch, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, _ = _write_synthetic_source(scratch, image_count)
    exit_code = cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(source_masks), "--source-images", str(source_images),
        "--output-root", str(source_masks), "--checkpoint", str(scratch / "c.json"), "--manifest", str(scratch / "m.json"),
    ])
    assert exit_code == 2


def test_test_limit_requires_explicit_flag(scratch, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, _ = _write_synthetic_source(scratch, image_count)
    exit_code = cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(source_masks), "--source-images", str(source_images),
        "--output-root", str(scratch / "output"), "--checkpoint", str(scratch / "c.json"), "--manifest", str(scratch / "m.json"),
        "--test-limit", "2",
    ])
    assert exit_code == 2


# ---------------------------------------------------------------------------
# Checkpoint / resume / existing-output safety
# ---------------------------------------------------------------------------


def test_existing_result_preserved_without_resume(scratch, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, ids = _write_synthetic_source(scratch, image_count)
    output_root = scratch / "output"
    checkpoint = scratch / "checkpoint.json"
    manifest = scratch / "manifest.json"
    exit0 = cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(source_masks), "--source-images", str(source_images),
        "--output-root", str(output_root), "--checkpoint", str(checkpoint), "--manifest", str(manifest),
    ])
    assert exit0 == 0
    original = manifest.read_text()

    exit1 = cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(source_masks), "--source-images", str(source_images),
        "--output-root", str(output_root), "--checkpoint", str(checkpoint), "--manifest", str(manifest),
    ])
    assert exit1 == 2
    assert manifest.read_text() == original


def test_nonempty_unproven_output_rejected_without_overwrite_flag(scratch, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, ids = _write_synthetic_source(scratch, image_count)
    output_root = scratch / "output"
    (output_root / "annotations" / "val2017").mkdir(parents=True)
    (output_root / "annotations" / "val2017" / f"{ids[0]}_instanceTrainIds.png").write_bytes(b"\x00")
    exit_code = cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(source_masks), "--source-images", str(source_images),
        "--output-root", str(output_root), "--checkpoint", str(scratch / "c.json"), "--manifest", str(scratch / "m.json"),
    ])
    assert exit_code == 2


@pytest.mark.parametrize("tiny_identity", [4], indirect=True)
def test_resume_from_interrupted_checkpoint_reaches_same_final_state(scratch, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, ids = _write_synthetic_source(scratch, image_count)
    output_root = scratch / "output"
    checkpoint = scratch / "checkpoint.json"
    manifest = scratch / "manifest.json"

    full_exit = cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(source_masks), "--source-images", str(source_images),
        "--output-root", str(output_root), "--checkpoint", str(checkpoint), "--manifest", str(manifest),
    ])
    assert full_exit == 0
    reference_manifest = json.loads(manifest.read_text())

    # Now simulate an interruption from scratch: truncate a fresh checkpoint/output pair to 1 image.
    output_root2 = scratch / "output2"
    checkpoint2 = scratch / "checkpoint2.json"
    manifest2 = scratch / "manifest2.json"
    resumable = dict(json.loads(checkpoint.read_text()))
    from src.coco_object_val_materialization import aggregate_records, write_json_atomically
    truncated = resumable["per_image_records"][:1]
    agg = aggregate_records(truncated)
    resumable["per_image_records"] = truncated
    resumable["completed_image_ids"] = [r["image_id"] for r in truncated]
    resumable["next_index"] = 1
    resumable["complete"] = False
    resumable["aggregate_label_histogram"] = agg["aggregate_label_histogram"]
    resumable["total_pixels"] = agg["total_pixels"]
    resumable["masks_with_foreground"] = agg["masks_with_foreground"]
    resumable["all_background_masks"] = agg["all_background_masks"]

    (output_root2 / "annotations" / "val2017").mkdir(parents=True)
    (output_root2 / "images").mkdir(parents=True)
    (output_root2 / "images" / "val2017").symlink_to(source_images.resolve())
    (output_root2 / "manifests").mkdir()
    (output_root2 / "checkpoints").mkdir()
    src_mask = output_root / "annotations" / "val2017" / f"{truncated[0]['image_id']}_instanceTrainIds.png"
    (output_root2 / "annotations" / "val2017" / src_mask.name).write_bytes(src_mask.read_bytes())
    write_json_atomically(checkpoint2, resumable)

    resumed_exit = cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(source_masks), "--source-images", str(source_images),
        "--output-root", str(output_root2), "--checkpoint", str(checkpoint2), "--manifest", str(manifest2),
        "--resume",
    ])
    assert resumed_exit == 0
    resumed_manifest = json.loads(manifest2.read_text())
    diff_keys = {k for k in reference_manifest if reference_manifest[k] != resumed_manifest.get(k)}
    assert diff_keys <= {"created_at_utc"}


def test_resume_rejects_tampered_installed_mask(scratch, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, ids = _write_synthetic_source(scratch, image_count)
    output_root = scratch / "output"
    checkpoint = scratch / "checkpoint.json"
    manifest = scratch / "manifest.json"
    cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(source_masks), "--source-images", str(source_images),
        "--output-root", str(output_root), "--checkpoint", str(checkpoint), "--manifest", str(manifest),
    ])
    from src.coco_object_val_materialization import aggregate_records, write_json_atomically
    doc = json.loads(checkpoint.read_text())
    doc["next_index"] = image_count - 1
    doc["complete"] = False
    doc["completed_image_ids"] = doc["completed_image_ids"][:-1]
    truncated = doc["per_image_records"][:-1]
    doc["per_image_records"] = truncated
    agg = aggregate_records(truncated)
    doc.update({
        "aggregate_label_histogram": agg["aggregate_label_histogram"], "total_pixels": agg["total_pixels"],
        "masks_with_foreground": agg["masks_with_foreground"], "all_background_masks": agg["all_background_masks"],
    })
    write_json_atomically(checkpoint, doc)
    # Tamper with an already-"completed" mask referenced by the checkpoint.
    tampered_id = truncated[0]["image_id"]
    mask_path = output_root / "annotations" / "val2017" / f"{tampered_id}_instanceTrainIds.png"
    mask_path.write_bytes(b"not a real png")

    exit_code = cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(source_masks), "--source-images", str(source_images),
        "--output-root", str(output_root), "--checkpoint", str(checkpoint), "--manifest", str(scratch / "m2.json"),
        "--resume",
    ])
    assert exit_code == 2


def test_malformed_checkpoint_json_exit_2_no_traceback(scratch, tiny_identity, capsys):
    identity_path, image_count = tiny_identity
    source_masks, source_images, _ = _write_synthetic_source(scratch, image_count)
    checkpoint = scratch / "checkpoint.json"
    checkpoint.write_text('{"a": 1, "a": 2}')
    exit_code = cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(source_masks), "--source-images", str(source_images),
        "--output-root", str(scratch / "output"), "--checkpoint", str(checkpoint), "--manifest", str(scratch / "m.json"),
        "--resume",
    ])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err


# ---------------------------------------------------------------------------
# Control-flow exclusion
# ---------------------------------------------------------------------------


def test_keyboard_interrupt_not_swallowed(scratch, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, _ = _write_synthetic_source(scratch, image_count)
    real_convert = cli.convert_one_image

    def raising(*a, **k):
        raise KeyboardInterrupt()

    cli.convert_one_image = raising
    try:
        with pytest.raises(KeyboardInterrupt):
            cli.main([
                "--repo-root", str(ROOT), "--identity", str(identity_path),
                "--source-masks", str(source_masks), "--source-images", str(source_images),
                "--output-root", str(scratch / "output"), "--checkpoint", str(scratch / "c.json"), "--manifest", str(scratch / "m.json"),
            ])
    finally:
        cli.convert_one_image = real_convert


def test_system_exit_not_swallowed(scratch, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, _ = _write_synthetic_source(scratch, image_count)
    real_convert = cli.convert_one_image

    def raising(*a, **k):
        raise SystemExit(9)

    cli.convert_one_image = raising
    try:
        with pytest.raises(SystemExit) as excinfo:
            cli.main([
                "--repo-root", str(ROOT), "--identity", str(identity_path),
                "--source-masks", str(source_masks), "--source-images", str(source_images),
                "--output-root", str(scratch / "output"), "--checkpoint", str(scratch / "c.json"), "--manifest", str(scratch / "m.json"),
            ])
        assert excinfo.value.code == 9
    finally:
        cli.convert_one_image = real_convert


# ---------------------------------------------------------------------------
# Source immutability
# ---------------------------------------------------------------------------


def test_source_files_never_modified(scratch, tiny_identity):
    identity_path, image_count = tiny_identity
    source_masks, source_images, ids = _write_synthetic_source(scratch, image_count)
    before = {p: p.read_bytes() for p in list(source_masks.iterdir()) + list(source_images.iterdir())}

    cli.main([
        "--repo-root", str(ROOT), "--identity", str(identity_path),
        "--source-masks", str(source_masks), "--source-images", str(source_images),
        "--output-root", str(scratch / "output"), "--checkpoint", str(scratch / "c.json"), "--manifest", str(scratch / "m.json"),
    ])

    for path, content in before.items():
        assert path.read_bytes() == content
