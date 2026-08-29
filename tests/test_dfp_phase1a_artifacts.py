"""Tests for DFP Phase 1A Step 10 strict checkpoint/artifact
infrastructure: the artifact-schema authority, checkpoint/per-image-stats
NPZ schema and invariants, atomic generation/install, ordered resume
validation, final-result schema/bundle install, and the standalone
verifier CLI.

CPU-only throughout. No real dataset, no model construction, no CUDA
initialization, no graph/recurrence execution."""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

import src.dfp_phase1a_artifact_schema as sch
import src.dfp_phase1a_checkpoint as ckpt
import src.dfp_phase1a_report as report

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def schema_summary():
    return sch.validate_static_configuration(repo_root=ROOT)


@pytest.fixture(scope="module")
def artifact_identity():
    return sch.load_identity(repo_root=ROOT)


@pytest.fixture(scope="module")
def coco_selected_ids():
    manifest = json.loads((ROOT / "evaluation_subsets/dfp_phase1a_coco_stuff_500.json").read_text())
    return manifest["selected_ids"], manifest["selected_ids_digest"]


def _make_arrays(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}
    for name in sch.INT64_MEMBER_NAMES:
        arrays[name] = np.arange(n, dtype=np.int64) if name == "dataset_indices" else rng.integers(0, 100, size=n).astype(np.int64)
    base = np.sort(rng.random((n, 5)), axis=1) if n else np.zeros((0, 5))
    for i, name in enumerate(("q_w_median", "q_w_p90", "q_w_p95", "q_w_p99", "q_w_maximum")):
        arrays[name] = base[:, i].astype(np.float64) if n else np.zeros(0, dtype=np.float64)
    arrays["q_w_mean"] = rng.random(n) if n else np.zeros(0, dtype=np.float64)
    if n:
        arrays["dispute_pixel_count"] = np.minimum(arrays["dispute_pixel_count"], arrays["evaluated_pixel_count"])
        arrays["zero_dispute_window_count"] = np.minimum(arrays["zero_dispute_window_count"], arrays["window_count"])
    return arrays


def _make_fields(*, schema_summary, artifact_identity, selected_ids, selected_ids_digest, n: int, complete: bool = False) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "schema": sch.CHECKPOINT_SCHEMA_NAME, "schema_version": sch.CHECKPOINT_SCHEMA_VERSION,
        "artifact_schema_identity_sha256": sch.sha256_file(ROOT / sch.IDENTITY_RELATIVE_PATH),
        "protocol_identity_sha256": schema_summary["protocol_identity_sha256"],
        "subset_manifest_sha256": schema_summary["coco_stuff_manifest_sha256"], "selected_ids_digest": selected_ids_digest,
        "run_mode": "coco_stuff_500", "image_count_expected": n if complete else len(selected_ids),
        "class_count": artifact_identity["dataset_binding"]["coco_stuff_class_count"],
        "class_names_digest": artifact_identity["dataset_binding"]["coco_stuff_class_names_digest"],
        "horizon_order": list(sch.HORIZON_ORDER), "source_commit": "a" * 40,
        "required_ancestor_commit": artifact_identity["parent_identity_binding"]["required_ancestor_commit"],
        "bridge_checkpoint_sha256": schema_summary["bridge_checkpoint_sha256"], "next_dataset_index": n,
        "completed_image_ids": list(selected_ids[:n]), "images_completed_count": n, "total_windows_processed": n * 4,
        "complete": complete, "final_result_produced": False, "created_at_utc": now, "updated_at_utc": now,
    }


def _install(tmp_path, *, schema_summary, artifact_identity, selected_ids, selected_ids_digest, n: int, complete: bool = False, seed: int = 0):
    fields = _make_fields(
        schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids,
        selected_ids_digest=selected_ids_digest, n=n, complete=complete,
    )
    arrays = _make_arrays(n, seed=seed)
    checkpoints_dir = tmp_path / "checkpoints"
    ckpt.generate_and_install_checkpoint(
        checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
        arrays=arrays, image_ids=selected_ids[:n],
    )
    return checkpoints_dir


# ---------------------------------------------------------------------------
# Artifact-schema authority
# ---------------------------------------------------------------------------


def test_artifact_schema_validates(schema_summary):
    assert schema_summary["identity_name"] == "e13-dfp-phase1a-artifacts"


def test_artifact_schema_immutable_after_load(artifact_identity):
    before = copy.deepcopy(artifact_identity)
    sch.load_identity(repo_root=ROOT)
    assert artifact_identity == before


@pytest.mark.parametrize(
    "mutator",
    [
        lambda d: d["identity"].__setitem__("name", "wrong"),
        lambda d: d["protocol_binding"].__setitem__("protocol_identity_sha256", "not-a-valid-hash"),
        lambda d: d["checkpoint_schema"].__setitem__("schema_version", 2),
        lambda d: d["per_image_stats_schema"]["int64_member_names"].pop(),
        lambda d: d.pop("path_policy"),
        lambda d: d["dataset_binding"].__setitem__("horizon_order", [0, 16, 32, 64]),
        lambda d: d.__setitem__("extra_unknown_section", {}),
    ],
)
def test_artifact_schema_mutation_matrix_rejected(tmp_path, mutator):
    identity = sch.load_identity(repo_root=ROOT)
    mutated = copy.deepcopy(identity)
    mutator(mutated)

    # load_identity re-parses TOML from disk; here we exercise the same
    # section validators it calls directly against the mutated in-memory
    # dict, so no TOML round-trip is needed.
    with pytest.raises(sch.DfpPhase1aArtifactSchemaError):
        if set(mutated) != sch.IDENTITY_TOP_KEYS:
            raise sch.DfpPhase1aArtifactSchemaError("top-level mismatch")
        for section, keys in sch.SECTION_KEYS.items():
            block = mutated.get(section)
            if not isinstance(block, dict) or set(block) != keys:
                raise sch.DfpPhase1aArtifactSchemaError(f"{section} mismatch")
        sch._validate_identity_block(mutated["identity"])
        sch._validate_protocol_binding_block(mutated["protocol_binding"])
        sch._validate_checkpoint_schema_block(mutated["checkpoint_schema"])
        sch._validate_per_image_stats_schema_block(mutated["per_image_stats_schema"])
        sch._validate_dataset_binding_block(mutated["dataset_binding"])


def test_artifact_schema_relational_hash_mismatch_rejected(tmp_path):
    """A syntactically valid (well-formed 64-hex) but WRONG protocol hash
    must be rejected relationally by validate_static_configuration, even
    though it passes the format-only structural check in load_identity."""
    wrong_but_well_formed_hash = "0" * 64
    original_text = (ROOT / sch.IDENTITY_RELATIVE_PATH).read_text()
    tampered_text = original_text.replace(
        'protocol_identity_sha256 = "d53cfe6b9632b0fb24a90ed1c1482a00b83dd64255aaa595b0af141cab7febe7"',
        f'protocol_identity_sha256 = "{wrong_but_well_formed_hash}"',
    )
    # Ensure the replacement actually changed the text (guards against a
    # future rewording of the pinned literal silently no-opping this test).
    assert tampered_text != original_text
    tampered_path = tmp_path / "tampered_artifacts.toml"
    tampered_path.write_text(tampered_text)
    with pytest.raises(sch.DfpPhase1aArtifactSchemaError, match="mismatch"):
        sch.validate_static_configuration(repo_root=ROOT, identity_path=tampered_path)


def test_safe_relative_path_matrix():
    ok = ["a/b.json", "a.npz", "sub/dir/file.txt"]
    for path in ok:
        assert sch.require_safe_relative_path(path, "x") == path
    bad = ["/abs/path", "../escape", "a/../b", "a/./b", "C:\\win\\path", "a\\b", "a\x00b", ""]
    for path in bad:
        with pytest.raises(sch.DfpPhase1aArtifactSchemaError):
            sch.require_safe_relative_path(path, "x")


def test_safe_basename_rejects_directories():
    assert sch.require_safe_basename("file.json", "x") == "file.json"
    with pytest.raises(sch.DfpPhase1aArtifactSchemaError):
        sch.require_safe_basename("dir/file.json", "x")


def test_resolve_within_root_rejects_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(sch.DfpPhase1aArtifactSchemaError):
        sch.resolve_within_root(root, "../outside.txt", label="x")


# ---------------------------------------------------------------------------
# Strict JSON I/O
# ---------------------------------------------------------------------------


def test_strict_json_rejects_duplicate_key(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text('{"a": 1, "a": 2}')
    with pytest.raises(sch.DfpPhase1aArtifactSchemaError):
        sch.parse_strict_json_document(path, label="x")


def test_strict_json_rejects_nan_infinity(tmp_path):
    for token in ("NaN", "Infinity", "-Infinity"):
        path = tmp_path / "bad.json"
        path.write_text('{"a": %s}' % token)
        with pytest.raises(sch.DfpPhase1aArtifactSchemaError):
            sch.parse_strict_json_document(path, label="x")


def test_strict_json_rejects_malformed_utf8(tmp_path):
    path = tmp_path / "bad_utf8.json"
    path.write_bytes(b'{"a": "\xff\xfe"}')
    with pytest.raises(sch.DfpPhase1aArtifactSchemaError):
        sch.parse_strict_json_document(path, label="x")


def test_strict_json_rejects_trailing_garbage(tmp_path):
    path = tmp_path / "trailing.json"
    path.write_text('{"a": 1} garbage')
    with pytest.raises(sch.DfpPhase1aArtifactSchemaError):
        sch.parse_strict_json_document(path, label="x")


def test_strict_json_rejects_non_object_root(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(sch.DfpPhase1aArtifactSchemaError):
        sch.parse_strict_json_document(path, label="x")


def test_strict_json_rejects_missing_and_directory(tmp_path):
    with pytest.raises(sch.DfpPhase1aArtifactSchemaError):
        sch.parse_strict_json_document(tmp_path / "missing.json", label="x")
    with pytest.raises(sch.DfpPhase1aArtifactSchemaError):
        sch.parse_strict_json_document(tmp_path, label="x")


def test_write_json_atomically_no_orphans(tmp_path):
    path = tmp_path / "doc.json"
    sch.write_json_atomically(path, {"a": 1}, label="doc")
    assert path.read_text().endswith("\n")
    assert json.loads(path.read_text()) == {"a": 1}
    remaining = list(tmp_path.iterdir())
    assert remaining == [path]


def test_write_json_atomically_rejects_nan():
    with pytest.raises(sch.DfpPhase1aArtifactSchemaError):
        sch.write_json_atomically(Path("/tmp/unused.json"), {"a": float("nan")}, label="doc")


def test_write_json_atomically_does_not_mutate_input(tmp_path):
    doc = {"b": 2, "a": 1}
    before = copy.deepcopy(doc)
    sch.write_json_atomically(tmp_path / "doc.json", doc, label="doc")
    assert doc == before


# ---------------------------------------------------------------------------
# Checkpoint structure invariants
# ---------------------------------------------------------------------------


def test_checkpoint_end_to_end_install_and_resume(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    result = ckpt.validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert result["next_dataset_index"] == 3
    assert result["resumable"] is True
    assert result["run_mode"] == "coco_stuff_500"


def test_zero_row_checkpoint_valid(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=0)
    result = ckpt.validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert result["next_dataset_index"] == 0
    for name in sch.ALL_NPZ_MEMBER_NAMES:
        assert result["arrays"][name].shape == (0,)


def test_build_zero_row_arrays_never_uses_np_stack_empty():
    arrays = ckpt.build_zero_row_arrays()
    assert set(arrays) == sch.ALL_NPZ_MEMBER_NAMES
    for name, array in arrays.items():
        assert array.shape == (0,)
        expected_dtype = np.int64 if name in sch.INT64_MEMBER_NAMES else np.float64
        assert array.dtype == np.dtype(expected_dtype)


def test_progress_invariant_next_index_mismatch_rejected():
    checkpoint = _bare_checkpoint(next_index=3, completed=["a", "b"])
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="next_dataset_index"):
        _structural_validate(checkpoint)


def test_duplicate_completed_ids_rejected():
    checkpoint = _bare_checkpoint(next_index=2, completed=["a", "a"])
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="duplicate"):
        _structural_validate(checkpoint)


def test_complete_flag_requires_full_count():
    checkpoint = _bare_checkpoint(next_index=2, completed=["a", "b"], complete=True, image_count_expected=5)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="complete"):
        _structural_validate(checkpoint)


def test_final_result_produced_requires_complete():
    checkpoint = _bare_checkpoint(next_index=2, completed=["a", "b"], complete=False, final_result_produced=True)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="final_result_produced"):
        _structural_validate(checkpoint)


def test_resume_dataset_index_refuses_complete_checkpoint():
    checkpoint = _bare_checkpoint(next_index=2, completed=["a", "b"], complete=True, image_count_expected=2)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="already complete"):
        ckpt.resume_dataset_index(checkpoint)


def test_completed_id_prefix_skip_reorder_rejected(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    checkpoint_path = checkpoints_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    reordered = dict(checkpoint)
    reordered["completed_image_ids"] = list(reversed(checkpoint["completed_image_ids"]))
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="canonical"):
        ckpt.validate_checkpoint_against_canonical_order(reordered, selected_ids, selected_ids_digest=digest)


def _bare_checkpoint(*, next_index, completed, complete=False, image_count_expected=500, final_result_produced=False) -> dict:
    now = "2026-01-01T00:00:00+00:00"
    return {
        "schema": sch.CHECKPOINT_SCHEMA_NAME, "schema_version": sch.CHECKPOINT_SCHEMA_VERSION,
        "artifact_schema_identity_sha256": "0" * 64, "protocol_identity_sha256": "0" * 64,
        "subset_manifest_sha256": "0" * 64, "selected_ids_digest": "0" * 64, "run_mode": "coco_stuff_500",
        "image_count_expected": image_count_expected, "class_count": 171, "class_names_digest": "0" * 64,
        "horizon_order": list(sch.HORIZON_ORDER), "source_commit": "a" * 40, "required_ancestor_commit": "b" * 40,
        "bridge_checkpoint_sha256": "0" * 64, "next_dataset_index": next_index, "completed_image_ids": completed,
        "images_completed_count": len(completed), "total_windows_processed": len(completed), "complete": complete,
        "final_result_produced": final_result_produced, "generation_nonce": "n" * 8,
        "per_image_stats_manifest_relative_path": "stats.json", "per_image_stats_manifest_sha256": "0" * 64,
        "created_at_utc": now, "updated_at_utc": now,
    }


def _structural_validate(checkpoint):
    ckpt.validate_checkpoint_structure(
        checkpoint, artifact_schema_identity_sha256="0" * 64, protocol_identity_sha256="0" * 64,
        subset_manifest_sha256="0" * 64, bridge_checkpoint_sha256="0" * 64, required_ancestor_commit="b" * 40,
    )


# ---------------------------------------------------------------------------
# Per-image-stats NPZ schema
# ---------------------------------------------------------------------------


def test_npz_wrong_dtype_rejected():
    array = np.arange(3, dtype=np.float32)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="int64"):
        ckpt.validate_per_image_stats_array("window_count", array, image_count=3)


def test_npz_wrong_rank_rejected():
    array = np.zeros((3, 1), dtype=np.int64)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="1-dimensional"):
        ckpt.validate_per_image_stats_array("window_count", array, image_count=3)


def test_npz_wrong_row_count_rejected():
    array = np.zeros((3,), dtype=np.int64)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="image_count"):
        ckpt.validate_per_image_stats_array("window_count", array, image_count=4)


def test_npz_nan_float_rejected():
    array = np.array([1.0, float("nan")], dtype=np.float64)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="non-finite"):
        ckpt.validate_per_image_stats_array("q_w_mean", array, image_count=2)


def test_npz_negative_count_rejected():
    array = np.array([1, -1], dtype=np.int64)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="non-negative"):
        ckpt.validate_per_image_stats_array("window_count", array, image_count=2)


def test_npz_unknown_name_rejected():
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="not part of"):
        ckpt.validate_per_image_stats_array("not_a_real_member", np.zeros(2, dtype=np.int64), image_count=2)


def test_npz_percentile_monotonicity_violation_rejected():
    arrays = {
        "q_w_median": np.array([0.9]), "q_w_p90": np.array([0.5]), "q_w_p95": np.array([0.6]),
        "q_w_p99": np.array([0.7]), "q_w_maximum": np.array([0.8]),
        "dispute_pixel_count": np.array([0]), "evaluated_pixel_count": np.array([1]),
        "zero_dispute_window_count": np.array([0]), "window_count": np.array([1]),
    }
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="monotonic"):
        ckpt._validate_percentile_monotonicity(arrays)


def test_npz_missing_member_rejected(tmp_path):
    arrays = _make_arrays(2)
    del arrays["q_w_mean"]
    npz_path = tmp_path / "bad.npz"
    np.savez(npz_path, allow_pickle=False, **arrays)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="missing"):
        ckpt.load_per_image_stats_npz(npz_path, expected_sha256=sch.sha256_file(npz_path), image_count=2)


def test_npz_unexpected_member_rejected(tmp_path):
    arrays = _make_arrays(2)
    arrays["extra_unauthorized_member"] = np.zeros(2, dtype=np.int64)
    npz_path = tmp_path / "bad.npz"
    np.savez(npz_path, allow_pickle=False, **arrays)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="unexpected"):
        ckpt.load_per_image_stats_npz(npz_path, expected_sha256=sch.sha256_file(npz_path), image_count=2)


def test_npz_digest_mismatch_rejected(tmp_path):
    arrays = _make_arrays(2)
    npz_path = tmp_path / "good.npz"
    np.savez(npz_path, allow_pickle=False, **arrays)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="SHA256 mismatch"):
        ckpt.load_per_image_stats_npz(npz_path, expected_sha256="0" * 64, image_count=2)


def test_npz_changed_bytes_after_digest_recorded_rejected(tmp_path):
    arrays = _make_arrays(2)
    npz_path = tmp_path / "good.npz"
    np.savez(npz_path, allow_pickle=False, **arrays)
    original_sha256 = sch.sha256_file(npz_path)
    with npz_path.open("ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="SHA256 mismatch"):
        ckpt.load_per_image_stats_npz(npz_path, expected_sha256=original_sha256, image_count=2)


def test_npz_recomputed_digest_with_invalid_schema_still_rejected(tmp_path):
    """A tampered NPZ whose digest is recomputed AFTER the tamper (so the
    digest check alone would pass) must still be rejected by the schema
    checks."""
    arrays = _make_arrays(2)
    arrays["window_count"] = arrays["window_count"].astype(np.float32)  # invalid dtype
    npz_path = tmp_path / "bad.npz"
    np.savez(npz_path, allow_pickle=False, **arrays)
    recomputed_sha256 = sch.sha256_file(npz_path)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="int64"):
        ckpt.load_per_image_stats_npz(npz_path, expected_sha256=recomputed_sha256, image_count=2)


def test_npz_truncated_archive_rejected(tmp_path):
    arrays = _make_arrays(2)
    npz_path = tmp_path / "good.npz"
    np.savez(npz_path, allow_pickle=False, **arrays)
    raw = npz_path.read_bytes()
    truncated_path = tmp_path / "truncated.npz"
    truncated_path.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(ckpt.DfpPhase1aCheckpointError):
        ckpt.load_per_image_stats_npz(truncated_path, expected_sha256=sch.sha256_file(truncated_path), image_count=2)


def test_npz_duplicate_zip_member_rejected(tmp_path):
    arrays = _make_arrays(1)
    npz_path = tmp_path / "dup_members.npz"
    with zipfile.ZipFile(npz_path, "w") as archive:
        for name, array in arrays.items():
            import io

            buf = io.BytesIO()
            np.save(buf, array)
            archive.writestr(f"{name}.npy", buf.getvalue())
        # duplicate the first member's entry
        first_name = next(iter(arrays))
        buf = io.BytesIO()
        np.save(buf, arrays[first_name])
        archive.writestr(f"{first_name}.npy", buf.getvalue())
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="duplicate ZIP member"):
        ckpt.load_per_image_stats_npz(npz_path, expected_sha256=sch.sha256_file(npz_path), image_count=1)


def test_npz_missing_sibling_manifest_or_npz_rejected(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2)
    checkpoint = json.loads((checkpoints_dir / "checkpoint.json").read_text())
    npz_path = checkpoints_dir / json.loads((checkpoints_dir / checkpoint["per_image_stats_manifest_relative_path"]).read_text())["npz_filename"]
    npz_path.unlink()
    with pytest.raises(ckpt.DfpPhase1aCheckpointError):
        ckpt.validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)


# ---------------------------------------------------------------------------
# Atomic generation/install failure-preservation
# ---------------------------------------------------------------------------


def test_generation_failure_preserves_previous_checkpoint(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    checkpoint_path = checkpoints_dir / "checkpoint.json"
    before_bytes = checkpoint_path.read_bytes()
    before_files = sorted(p.name for p in checkpoints_dir.iterdir())

    broken_arrays = _make_arrays(2)
    broken_arrays["window_count"] = broken_arrays["window_count"].astype(np.float32)
    fields = _make_fields(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError):
        ckpt.generate_and_install_checkpoint(
            checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
            arrays=broken_arrays, image_ids=selected_ids[:2],
        )

    assert checkpoint_path.read_bytes() == before_bytes
    assert sorted(p.name for p in checkpoints_dir.iterdir()) == before_files
    assert not any(".tmp" in name for name in (p.name for p in checkpoints_dir.iterdir()))


def test_generate_and_install_does_not_mutate_input_fields(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    fields = _make_fields(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2)
    before = copy.deepcopy(fields)
    arrays = _make_arrays(2)
    ckpt.generate_and_install_checkpoint(
        checkpoints_dir=tmp_path / "checkpoints", checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
        arrays=arrays, image_ids=selected_ids[:2],
    )
    assert fields == before


# ---------------------------------------------------------------------------
# Final result schema / atomic bundle install
# ---------------------------------------------------------------------------


def test_result_install_requires_complete_checkpoint(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    with pytest.raises(report.DfpPhase1aReportError, match="not complete"):
        report.install_result_bundle(destination_dir=tmp_path / "bundle", checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)


def test_result_install_and_verify_roundtrip(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3, complete=True)
    dest = tmp_path / "bundle"
    report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    verified = report.verify_result_bundle(dest)
    assert verified["result"]["complete"] is True
    assert verified["result"]["final"] is True


def test_result_install_rejects_existing_destination(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, complete=True)
    dest = tmp_path / "bundle"
    report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    with pytest.raises(report.DfpPhase1aReportError, match="already exists"):
        report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)


def test_result_install_leaves_no_staging_residue_on_failure(tmp_path, schema_summary, artifact_identity, coco_selected_ids, monkeypatch):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, complete=True)
    dest = tmp_path / "bundle"

    original = report.validate_result_structure
    call_count = {"n": 0}

    def _boom(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 2:  # let the first (pre-stage) call pass, fail the staged self-verify
            raise report.DfpPhase1aReportError("simulated self-verification failure")
        return original(*a, **kw)

    monkeypatch.setattr(report, "validate_result_structure", _boom)
    with pytest.raises(report.DfpPhase1aReportError, match="simulated"):
        report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert not dest.exists()
    assert not any("staging" in p.name for p in dest.parent.iterdir())


def test_pilot_result_never_labeled_as_another_run(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, complete=True)
    checkpoint_path = checkpoints_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint_sha256 = sch.sha256_file(checkpoint_path)
    result = {
        "schema": sch.RESULT_SCHEMA_NAME, "schema_version": sch.RESULT_SCHEMA_VERSION,
        "artifact_schema_identity_sha256": checkpoint["artifact_schema_identity_sha256"],
        "protocol_identity_sha256": checkpoint["protocol_identity_sha256"], "subset_manifest_sha256": checkpoint["subset_manifest_sha256"],
        "selected_ids_digest": checkpoint["selected_ids_digest"], "run_mode": "ade20k_500",  # WRONG on purpose
        "image_count_expected": checkpoint["image_count_expected"], "checkpoint_sha256": checkpoint_sha256,
        "per_image_stats_manifest_sha256": checkpoint["per_image_stats_manifest_sha256"], "complete": True, "final": True,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
    }
    with pytest.raises(report.DfpPhase1aReportError, match="another dataset/run"):
        report.validate_result_structure(result, checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha256)


# ---------------------------------------------------------------------------
# Purity: no dataset/model/CUDA entry point reachable from resume validation
# ---------------------------------------------------------------------------


def test_resume_validation_never_initializes_cuda(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    import torch

    assert torch.cuda.is_initialized() is False
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2)
    ckpt.validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert torch.cuda.is_initialized() is False


def test_malformed_resume_state_poison_test(tmp_path, schema_summary, artifact_identity, coco_selected_ids, monkeypatch):
    """Poison dataset/model/CUDA entry points; a malformed checkpoint must
    be rejected before any of them could possibly be reached."""
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2)
    checkpoint_path = checkpoints_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["completed_image_ids"] = list(reversed(checkpoint["completed_image_ids"]))  # corrupt the prefix
    sch.write_json_atomically(checkpoint_path, checkpoint, label="checkpoint")

    def _boom(*a, **kw):
        raise AssertionError("forbidden entry point reached")

    import torch

    monkeypatch.setattr(torch.cuda, "init", _boom)
    monkeypatch.setattr(torch.cuda, "is_available", _boom)

    with pytest.raises(ckpt.DfpPhase1aCheckpointError):
        ckpt.validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert torch.cuda.is_initialized() is False


def test_no_files_written_by_read_only_validation(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2)
    before = sorted(p.name for p in checkpoints_dir.iterdir())
    ckpt.validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    after = sorted(p.name for p in checkpoints_dir.iterdir())
    assert before == after


# ---------------------------------------------------------------------------
# CLI failure contract
# ---------------------------------------------------------------------------


def test_cli_verify_schema_pass():
    proc = subprocess.run([sys.executable, str(ROOT / "verify_dfp_phase1a_artifacts.py"), "verify-schema"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "PASS" in proc.stdout
    assert "Traceback" not in proc.stderr


def test_cli_missing_checkpoint_exits_2_no_traceback(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify_dfp_phase1a_artifacts.py"), "verify-checkpoint", "--checkpoint", str(tmp_path / "missing.json")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "DFP PHASE 1A ARTIFACTS VERIFY FAIL" in proc.stderr


def test_cli_corrupted_json_exits_2_no_traceback(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify_dfp_phase1a_artifacts.py"), "verify-checkpoint", "--checkpoint", str(bad)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


def test_cli_verify_result_on_missing_bundle_exits_2(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify_dfp_phase1a_artifacts.py"), "verify-result", "--bundle", str(tmp_path / "nope")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


# ---------------------------------------------------------------------------
# Purity: source-level forbidden-call audit
# ---------------------------------------------------------------------------


def _strip_comments_and_strings(source: str) -> str:
    """Return ``source`` with every COMMENT and STRING token blanked out,
    so a forbidden-call audit only sees actual code, never a docstring or
    comment that merely (correctly) describes what must never be called."""
    import io
    import tokenize

    pieces: list[str] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(tok.string)
    except tokenize.TokenError:
        return source
    return " ".join(pieces)


def test_zero_forbidden_calls_in_source():
    forbidden = ("finite_step_propagate", "DirectedTopKGraph", "torch.load", "create_model", ".cuda(", "CocoStuffDataset(", "ADE20KDataset(")
    for path in (
        ROOT / "src/dfp_phase1a_artifact_schema.py", ROOT / "src/dfp_phase1a_checkpoint.py",
        ROOT / "src/dfp_phase1a_report.py", ROOT / "verify_dfp_phase1a_artifacts.py",
    ):
        code_only = _strip_comments_and_strings(path.read_text())
        for token in forbidden:
            assert token not in code_only, f"{path} contains forbidden token {token!r} in actual code"


# ---------------------------------------------------------------------------
# Repair regression: unsafe-path dependency errors must normalize to
# DfpPhase1aCheckpointError (never let DfpPhase1aArtifactSchemaError escape
# validate_checkpoint_structure / validate_per_image_stats_manifest_structure).
# ---------------------------------------------------------------------------

UNSAFE_PATH_CASES = [
    ("parent-traversal", "../escape.json"),
    ("absolute-path", "/etc/passwd"),
    ("backslash-traversal", "..\\..\\escape.json"),
    ("whitespace-only", "   "),
]


@pytest.mark.parametrize("label,unsafe_value", UNSAFE_PATH_CASES)
def test_checkpoint_unsafe_manifest_path_normalizes_to_checkpoint_error(label, unsafe_value):
    """Direct validator: an unsafe checkpoint.per_image_stats_manifest_relative_path
    must raise exactly DfpPhase1aCheckpointError, chained from the underlying
    DfpPhase1aArtifactSchemaError, with the field name preserved in the message."""
    checkpoint = _bare_checkpoint(next_index=2, completed=["a", "b"])
    checkpoint = dict(checkpoint)
    checkpoint["per_image_stats_manifest_relative_path"] = unsafe_value
    with pytest.raises(ckpt.DfpPhase1aCheckpointError) as excinfo:
        _structural_validate(checkpoint)
    assert type(excinfo.value) is ckpt.DfpPhase1aCheckpointError
    assert isinstance(excinfo.value.__cause__, sch.DfpPhase1aArtifactSchemaError)
    assert "per_image_stats_manifest_relative_path" in str(excinfo.value)


@pytest.mark.parametrize("label,unsafe_value", UNSAFE_PATH_CASES)
def test_per_image_stats_manifest_unsafe_npz_filename_normalizes_to_checkpoint_error(label, unsafe_value):
    """Direct validator: an unsafe per-image-stats manifest.npz_filename
    must raise exactly DfpPhase1aCheckpointError, chained from
    DfpPhase1aArtifactSchemaError, with the field name preserved."""
    manifest = {
        "schema": sch.PER_IMAGE_STATS_SCHEMA_NAME, "schema_version": sch.PER_IMAGE_STATS_SCHEMA_VERSION,
        "npz_filename": unsafe_value, "npz_sha256": "0" * 64, "run_mode": "coco_stuff_500",
        "class_count": 171, "class_names_digest": "0" * 64, "image_count": 2, "image_ids": ["a", "b"],
    }
    with pytest.raises(ckpt.DfpPhase1aCheckpointError) as excinfo:
        ckpt.validate_per_image_stats_manifest_structure(manifest)
    assert type(excinfo.value) is ckpt.DfpPhase1aCheckpointError
    assert isinstance(excinfo.value.__cause__, sch.DfpPhase1aArtifactSchemaError)
    assert "npz_filename" in str(excinfo.value)


def test_unsafe_manifest_path_rejected_via_resume_validation(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """Resume-validation entry point: the same unsafe-path checkpoint must
    be rejected as DfpPhase1aCheckpointError, matching validate_resume_state's
    own documented contract ("raises DfpPhase1aCheckpointError... chained
    from any dependency error")."""
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2)
    checkpoint_path = checkpoints_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["per_image_stats_manifest_relative_path"] = "../escape.json"
    sch.write_json_atomically(checkpoint_path, checkpoint, label="checkpoint")
    with pytest.raises(ckpt.DfpPhase1aCheckpointError) as excinfo:
        ckpt.validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert isinstance(excinfo.value.__cause__, sch.DfpPhase1aArtifactSchemaError)


def test_unsafe_manifest_path_rejected_via_cli(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """CLI boundary: the same case exits 2 with a concise diagnostic and
    no traceback (regardless of exception type, since the CLI already
    catches both domain error types -- this test locks in that the
    normalized exception type doesn't change that outcome)."""
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2)
    checkpoint_path = checkpoints_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["per_image_stats_manifest_relative_path"] = "../escape.json"
    sch.write_json_atomically(checkpoint_path, checkpoint, label="checkpoint")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify_dfp_phase1a_artifacts.py"), "verify-resume", "--checkpoints-dir", str(checkpoints_dir), "--checkpoint-filename", "checkpoint.json"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "DFP PHASE 1A ARTIFACTS VERIFY FAIL" in proc.stderr
    assert "per_image_stats_manifest_relative_path" in proc.stderr


def test_require_safe_basename_call_sites_all_wrapped():
    """Audit: every require_safe_basename call in dfp_phase1a_checkpoint.py
    must be inside a try/except DfpPhase1aArtifactSchemaError that
    re-raises DfpPhase1aCheckpointError -- catches a future unwrapped
    call site being added without the established pattern."""
    import ast

    source = (ROOT / "src/dfp_phase1a_checkpoint.py").read_text()
    tree = ast.parse(source)
    call_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "require_safe_basename":
            call_lines.append(node.lineno)
    assert len(call_lines) == 2, f"expected exactly 2 require_safe_basename call sites, found {len(call_lines)} at lines {call_lines}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            try_call_lines = [
                n.lineno for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "require_safe_basename"
            ]
            handler_types = set()
            for handler in node.handlers:
                if isinstance(handler.type, ast.Name):
                    handler_types.add(handler.type.id)
            if try_call_lines:
                assert "DfpPhase1aArtifactSchemaError" in handler_types, (
                    f"require_safe_basename call(s) at {try_call_lines} inside a try block whose handlers "
                    f"are {handler_types}, not DfpPhase1aArtifactSchemaError"
                )
                for line in try_call_lines:
                    call_lines.remove(line)
    assert call_lines == [], f"require_safe_basename call(s) at {call_lines} are NOT wrapped in any try/except at all"


# ---------------------------------------------------------------------------
# Repair regression: the final checkpoint-pointer write must be inside the
# same cleanup scope as every earlier generation step -- a failure there
# must remove only the current attempt's own new generation files, never a
# pre-existing file, and must preserve the old checkpoint byte-for-byte.
# ---------------------------------------------------------------------------


def _snapshot(cp_dir):
    files = sorted(p.name for p in cp_dir.iterdir())
    hashes = {name: hashlib.sha256((cp_dir / name).read_bytes()).hexdigest() for name in files}
    return files, hashes


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: OSError("simulated disk-full during checkpoint-pointer install"),
        lambda: ValueError("simulated serialization ValueError during checkpoint-pointer install"),
        lambda: ckpt.DfpPhase1aCheckpointError("simulated domain error during checkpoint-pointer install"),
        lambda: MemoryError("simulated MemoryError during checkpoint-pointer install"),
    ],
    ids=["OSError", "ValueError", "DomainError", "MemoryError"],
)
def test_pointer_write_failure_matrix_preserves_old_and_cleans_new(tmp_path, schema_summary, artifact_identity, coco_selected_ids, exc_factory):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=1)
    before_files, before_hashes = _snapshot(checkpoints_dir)

    new_fields = _make_fields(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    new_fields_before = copy.deepcopy(new_fields)
    new_arrays = _make_arrays(3, seed=2)

    original = ckpt.write_json_atomically
    call_count = {"n": 0}
    exc_instance = exc_factory()

    def selective_fail(path, doc, *, label):
        call_count["n"] += 1
        if call_count["n"] == 2:  # 1st call = per-image-stats manifest (succeeds); 2nd = checkpoint pointer
            raise exc_instance
        return original(path, doc, label=label)

    ckpt.write_json_atomically = selective_fail
    try:
        with pytest.raises(type(exc_instance)) as excinfo:
            ckpt.generate_and_install_checkpoint(
                checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=new_fields,
                arrays=new_arrays, image_ids=selected_ids[:3],
            )
        assert excinfo.value is exc_instance  # re-raised unchanged, not wrapped/replaced
    finally:
        ckpt.write_json_atomically = original

    after_files, after_hashes = _snapshot(checkpoints_dir)
    assert before_files == after_files, "old checkpoint's referenced file set must be unchanged"
    assert before_hashes == after_hashes, "old checkpoint and its referenced generation must remain byte-identical"
    assert not any(".tmp" in name for name in after_files), "no .tmp residue"
    assert new_fields == new_fields_before, "input checkpoint_fields must not be mutated"

    # The old checkpoint must still resolve/validate cleanly (fully valid, not just present).
    old_checkpoint = json.loads((checkpoints_dir / "checkpoint.json").read_text())
    stats_manifest_path = checkpoints_dir / old_checkpoint["per_image_stats_manifest_relative_path"]
    stats_manifest = json.loads(stats_manifest_path.read_text())
    ckpt.validate_checkpoint_against_per_image_stats(old_checkpoint, stats_manifest)
    npz_path = checkpoints_dir / stats_manifest["npz_filename"]
    ckpt.load_per_image_stats_npz(npz_path, expected_sha256=stats_manifest["npz_sha256"], image_count=stats_manifest["image_count"])


@pytest.mark.parametrize(
    "target",
    ["np.savez", "write-manifest", "candidate-validation", "self-verification", "os.replace-npz"],
)
def test_generation_failure_injection_matrix_full_coverage(tmp_path, schema_summary, artifact_identity, coco_selected_ids, target, monkeypatch):
    """Broader injection matrix: new-NPZ write, new-manifest write,
    candidate validation, self-verification, and the NPZ install's own
    os.replace -- every one must preserve the old checkpoint and leave no
    orphan/.tmp residue."""
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=3)
    before_files, before_hashes = _snapshot(checkpoints_dir)

    new_fields = _make_fields(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    new_arrays = _make_arrays(3, seed=4)

    if target == "np.savez":
        monkeypatch.setattr(ckpt.np, "savez", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated np.savez failure")))
    elif target == "write-manifest":
        original = ckpt.write_json_atomically
        state = {"n": 0}

        def fail_first(path, doc, *, label):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("simulated manifest write failure")
            return original(path, doc, label=label)

        monkeypatch.setattr(ckpt, "write_json_atomically", fail_first)
    elif target == "candidate-validation":
        monkeypatch.setattr(ckpt, "validate_checkpoint_structure", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated candidate validation failure")))
    elif target == "self-verification":
        monkeypatch.setattr(ckpt, "load_per_image_stats_npz", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated self-verification failure")))
    elif target == "os.replace-npz":
        import os as os_module

        original_replace = os_module.replace
        state = {"n": 0}

        def fail_first_replace(*a, **kw):
            state["n"] += 1
            if state["n"] == 1:
                raise OSError("simulated os.replace failure during NPZ install")
            return original_replace(*a, **kw)

        # generate_and_install_checkpoint does a local `import os` and calls
        # os.replace on the same shared module object, so patching the
        # global os.replace attribute affects it regardless of where it
        # was imported.
        monkeypatch.setattr(os_module, "replace", fail_first_replace)

    with pytest.raises(Exception):
        ckpt.generate_and_install_checkpoint(
            checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=new_fields,
            arrays=new_arrays, image_ids=selected_ids[:3],
        )

    after_files, after_hashes = _snapshot(checkpoints_dir)
    assert before_files == after_files
    assert before_hashes == after_hashes
    assert not any(".tmp" in name for name in after_files)


def test_generation_success_installs_exactly_one_new_generation(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """Positive control: a successful generation installs exactly one new
    NPZ + manifest generation, a checkpoint referencing that exact
    generation, and leaves no temporary residue."""
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=5)
    before_files, _ = _snapshot(checkpoints_dir)

    new_fields = _make_fields(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    new_arrays = _make_arrays(3, seed=6)
    ckpt.generate_and_install_checkpoint(
        checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=new_fields,
        arrays=new_arrays, image_ids=selected_ids[:3],
    )

    after_files, _ = _snapshot(checkpoints_dir)
    new_files = set(after_files) - set(before_files)
    # exactly one new generation (manifest + npz) plus the checkpoint pointer already existed as a filename
    npz_new = [f for f in new_files if f.endswith(".npz")]
    json_new = [f for f in new_files if f.endswith(".json")]
    assert len(npz_new) == 1
    assert len(json_new) == 1
    assert not any(".tmp" in name for name in after_files)

    checkpoint = json.loads((checkpoints_dir / "checkpoint.json").read_text())
    assert checkpoint["per_image_stats_manifest_relative_path"] == json_new[0]
    stats_manifest = json.loads((checkpoints_dir / json_new[0]).read_text())
    assert stats_manifest["npz_filename"] == npz_new[0]


# ---------------------------------------------------------------------------
# Transaction-reconciliation regression matrix (named tests A-F): the final
# checkpoint-pointer write's own os.replace may already have durably
# succeeded before write_json_atomically raises from its own post-replace
# bookkeeping. generate_and_install_checkpoint must reconcile the actually-
# installed checkpoint's generation_nonce before deciding whether to clean
# up the current attempt's new generation, never inferring installation
# from file existence alone, and never masking the original exception.
# ---------------------------------------------------------------------------


def _fully_reverify(checkpoints_dir, repo_root=ROOT):
    """Independent-style full re-verification of whatever checkpoint is
    currently installed: structural validity, per-image-stats binding,
    and a full NPZ reload."""
    checkpoint = json.loads((checkpoints_dir / "checkpoint.json").read_text())
    stats_manifest_path = checkpoints_dir / checkpoint["per_image_stats_manifest_relative_path"]
    stats_manifest = json.loads(stats_manifest_path.read_text())
    assert sch.sha256_file(stats_manifest_path) == checkpoint["per_image_stats_manifest_sha256"]
    ckpt.validate_checkpoint_against_per_image_stats(checkpoint, stats_manifest)
    npz_path = checkpoints_dir / stats_manifest["npz_filename"]
    ckpt.load_per_image_stats_npz(npz_path, expected_sha256=stats_manifest["npz_sha256"], image_count=stats_manifest["image_count"])
    return checkpoint, stats_manifest


def test_A_pre_replacement_final_pointer_failure(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """A. The final write_json_atomically call fails BEFORE its own
    os.replace ever runs (simulated by never calling through to the real
    writer at all): old checkpoint byte-identical, old generation valid,
    new generation removed, original exception propagates."""
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=10)
    before_files, before_hashes = _snapshot(checkpoints_dir)
    old_checkpoint, _ = _fully_reverify(checkpoints_dir)

    new_fields = _make_fields(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    new_arrays = _make_arrays(3, seed=11)

    original = ckpt.write_json_atomically
    call_count = {"n": 0}
    injected = OSError("simulated pre-replace failure (never reaches real os.replace)")

    def fail_before_replace(path, doc, *, label):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise injected  # never calls `original` -- the real os.replace never runs
        return original(path, doc, label=label)

    ckpt.write_json_atomically = fail_before_replace
    try:
        with pytest.raises(OSError) as excinfo:
            ckpt.generate_and_install_checkpoint(
                checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=new_fields,
                arrays=new_arrays, image_ids=selected_ids[:3],
            )
        assert excinfo.value is injected
    finally:
        ckpt.write_json_atomically = original

    after_files, after_hashes = _snapshot(checkpoints_dir)
    assert before_files == after_files
    assert before_hashes == after_hashes
    assert not any(".tmp" in n for n in after_files)
    reverified_checkpoint, _ = _fully_reverify(checkpoints_dir)
    assert reverified_checkpoint == old_checkpoint


def test_B_post_replacement_memory_error(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """B. The final pointer write's os.replace actually succeeds (the
    real writer is called through), then MemoryError is raised as if from
    write_json_atomically's own post-replace bookkeeping. MemoryError
    must propagate; the installed checkpoint must carry the CURRENT
    generation nonce; the new NPZ+manifest must remain present and fully
    verify; nothing referenced may be deleted; no dangling checkpoint."""
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=20)
    old_checkpoint, _ = _fully_reverify(checkpoints_dir)
    old_generation_files = {p.name for p in checkpoints_dir.iterdir()} - {"checkpoint.json"}

    new_fields = _make_fields(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    new_arrays = _make_arrays(3, seed=21)

    original = ckpt.write_json_atomically
    call_count = {"n": 0}

    def succeed_then_memory_error(path, doc, *, label):
        call_count["n"] += 1
        if call_count["n"] == 2:
            original(path, doc, label=label)  # the real install genuinely happens
            raise MemoryError("simulated MemoryError from post-replace bookkeeping")
        return original(path, doc, label=label)

    ckpt.write_json_atomically = succeed_then_memory_error
    try:
        with pytest.raises(MemoryError):
            ckpt.generate_and_install_checkpoint(
                checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=new_fields,
                arrays=new_arrays, image_ids=selected_ids[:3],
            )
    finally:
        ckpt.write_json_atomically = original

    installed_checkpoint = json.loads((checkpoints_dir / "checkpoint.json").read_text())
    # the checkpoint genuinely advanced to the new generation
    assert installed_checkpoint["next_dataset_index"] == 3
    assert installed_checkpoint != old_checkpoint

    all_files = {p.name for p in checkpoints_dir.iterdir()}
    assert old_generation_files <= all_files, "old generation files must never be deleted"
    manifest_ref = installed_checkpoint["per_image_stats_manifest_relative_path"]
    assert (checkpoints_dir / manifest_ref).exists()
    # the installed checkpoint's generation_nonce is exactly the nonce
    # embedded in the manifest/NPZ filenames it references (proving the
    # CURRENT attempt's generation, not a stale/foreign one, is installed)
    assert installed_checkpoint["generation_nonce"] in manifest_ref
    stats_manifest = json.loads((checkpoints_dir / manifest_ref).read_text())
    assert (checkpoints_dir / stats_manifest["npz_filename"]).exists()
    assert installed_checkpoint["generation_nonce"] in stats_manifest["npz_filename"]

    # installed checkpoint + its generation fully verify
    _fully_reverify(checkpoints_dir)
    result = ckpt.validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert result["next_dataset_index"] == 3
    assert not any(".tmp" in n for n in all_files)


def test_C_post_replacement_ordinary_exception(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """C. Same as B but with a plain non-OSError/non-ValueError exception
    (RuntimeError) injected after a genuine successful replace -- same
    reconciliation and preservation behavior required."""
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=30)

    new_fields = _make_fields(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    new_arrays = _make_arrays(3, seed=31)

    original = ckpt.write_json_atomically
    call_count = {"n": 0}
    injected = RuntimeError("simulated ordinary post-replace exception")

    def succeed_then_runtime_error(path, doc, *, label):
        call_count["n"] += 1
        if call_count["n"] == 2:
            original(path, doc, label=label)
            raise injected
        return original(path, doc, label=label)

    ckpt.write_json_atomically = succeed_then_runtime_error
    try:
        with pytest.raises(RuntimeError) as excinfo:
            ckpt.generate_and_install_checkpoint(
                checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=new_fields,
                arrays=new_arrays, image_ids=selected_ids[:3],
            )
        assert excinfo.value is injected
    finally:
        ckpt.write_json_atomically = original

    installed_checkpoint = json.loads((checkpoints_dir / "checkpoint.json").read_text())
    assert installed_checkpoint["next_dataset_index"] == 3
    _fully_reverify(checkpoints_dir)
    all_files = {p.name for p in checkpoints_dir.iterdir()}
    assert not any(".tmp" in n for n in all_files)


def test_D_installed_pointer_still_references_old_generation(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """D. Pointer write fails and the OLD checkpoint is left intact
    (never replaced) -- reconciliation must detect the old nonce, remove
    the new generation, and leave the old checkpoint/generation valid."""
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=40)
    before_files, before_hashes = _snapshot(checkpoints_dir)
    old_checkpoint, _ = _fully_reverify(checkpoints_dir)

    new_fields = _make_fields(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    new_arrays = _make_arrays(3, seed=41)

    original = ckpt.write_json_atomically
    call_count = {"n": 0}
    injected = OSError("simulated failure, old pointer never replaced")

    def fail_leaving_old_intact(path, doc, *, label):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise injected  # old checkpoint.json on disk is left completely untouched
        return original(path, doc, label=label)

    ckpt.write_json_atomically = fail_leaving_old_intact
    try:
        with pytest.raises(OSError):
            ckpt.generate_and_install_checkpoint(
                checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=new_fields,
                arrays=new_arrays, image_ids=selected_ids[:3],
            )
    finally:
        ckpt.write_json_atomically = original

    after_files, after_hashes = _snapshot(checkpoints_dir)
    assert before_files == after_files, "the old generation's file set must be exactly preserved (new generation removed)"
    assert before_hashes == after_hashes
    reverified_checkpoint, _ = _fully_reverify(checkpoints_dir)
    assert reverified_checkpoint == old_checkpoint


def test_E_reconciliation_itself_fails(tmp_path, schema_summary, artifact_identity, coco_selected_ids, monkeypatch):
    """E. The pointer write fails with one exception, and the
    reconciliation re-read (parse_strict_json_document, as used inside
    dfp_phase1a_checkpoint's own namespace) raises a DIFFERENT exception
    while handling it. The ORIGINAL exception, not the reconciliation
    failure, must propagate, and the new generation must be conservatively
    preserved (never deleted when the installed state cannot be proven)."""
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=50)

    new_fields = _make_fields(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    new_arrays = _make_arrays(3, seed=51)

    original_write = ckpt.write_json_atomically
    call_count = {"n": 0}
    original_error = ValueError("original pointer-write failure")

    def fail_pointer_write(path, doc, *, label):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise original_error
        return original_write(path, doc, label=label)

    def broken_reconciliation_read(*a, **kw):
        raise RuntimeError("simulated reconciliation read failure -- must never be what propagates")

    monkeypatch.setattr(ckpt, "write_json_atomically", fail_pointer_write)
    monkeypatch.setattr(ckpt, "parse_strict_json_document", broken_reconciliation_read)

    with pytest.raises(ValueError) as excinfo:
        ckpt.generate_and_install_checkpoint(
            checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=new_fields,
            arrays=new_arrays, image_ids=selected_ids[:3],
        )
    assert excinfo.value is original_error, "the ORIGINAL exception must propagate, never the reconciliation failure"

    # New generation must be conservatively preserved (state indeterminate).
    all_files = {p.name for p in checkpoints_dir.iterdir()}
    npz_files = [f for f in all_files if f.endswith(".npz")]
    json_gen_files = [f for f in all_files if f.endswith(".json") and f != "checkpoint.json"]
    assert len(npz_files) == 2, "both the old and the (conservatively preserved) new generation NPZ must remain"
    assert len(json_gen_files) == 2


def test_F_fresh_checkpoint_no_previous_pointer_pre_replace(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """F (pre-replace half). No checkpoint.json exists at all yet. The
    very first generate_and_install_checkpoint call's pointer write fails
    before ever creating checkpoint.json. Non-existence is sound proof of
    non-installation: the new generation must be cleaned up, and no
    checkpoint.json must exist afterward."""
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = tmp_path / "fresh_pre_replace"
    checkpoints_dir.mkdir()

    fields = _make_fields(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2)
    arrays = _make_arrays(2, seed=60)

    original = ckpt.write_json_atomically
    call_count = {"n": 0}
    injected = OSError("simulated failure before checkpoint.json is ever created")

    def fail_before_creation(path, doc, *, label):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise injected
        return original(path, doc, label=label)

    ckpt.write_json_atomically = fail_before_creation
    try:
        with pytest.raises(OSError):
            ckpt.generate_and_install_checkpoint(
                checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
                arrays=arrays, image_ids=selected_ids[:2],
            )
    finally:
        ckpt.write_json_atomically = original

    assert not (checkpoints_dir / "checkpoint.json").exists()
    remaining = list(checkpoints_dir.iterdir())
    assert remaining == [], f"a fresh directory with no prior checkpoint must be fully cleaned up on a pre-replace failure, found {remaining}"


def test_F_fresh_checkpoint_no_previous_pointer_post_replace(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """F (post-replace half). No checkpoint.json exists at all yet. The
    very first generate_and_install_checkpoint call's pointer write
    genuinely succeeds (checkpoint.json is created), then a post-replace
    exception fires. The freshly-created checkpoint must survive fully
    installed and referencing the new generation."""
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = tmp_path / "fresh_post_replace"
    checkpoints_dir.mkdir()

    fields = _make_fields(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2)
    arrays = _make_arrays(2, seed=61)

    original = ckpt.write_json_atomically
    call_count = {"n": 0}

    def succeed_then_fail(path, doc, *, label):
        call_count["n"] += 1
        if call_count["n"] == 2:
            original(path, doc, label=label)
            raise MemoryError("simulated post-replace failure on the very first checkpoint")
        return original(path, doc, label=label)

    ckpt.write_json_atomically = succeed_then_fail
    try:
        with pytest.raises(MemoryError):
            ckpt.generate_and_install_checkpoint(
                checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
                arrays=arrays, image_ids=selected_ids[:2],
            )
    finally:
        ckpt.write_json_atomically = original

    assert (checkpoints_dir / "checkpoint.json").exists()
    _fully_reverify(checkpoints_dir)
    result = ckpt.validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert result["next_dataset_index"] == 2
    assert not any(".tmp" in p.name for p in checkpoints_dir.iterdir())
