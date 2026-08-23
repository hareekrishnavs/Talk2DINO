"""Tests for the single authoritative checkpoint loader/validator
(``src.k11_k12_power_evaluation_checkpoint``) shared by the evaluator's
resume path and ``verify_k11_k12_power_evaluation.py verify-checkpoint``.

Covers both confirmed findings from the independent verification of
``feat: add GPU matched k11-k12 power evaluator``:

- Finding 1: fail-closed checkpoint resume invariants (including the exact
  regression -- next_dataset_index=11 with only 10 completed images).
- Finding 2: strict checkpoint JSON loading (missing/malformed/corrupt
  documents fail closed with a concise domain error, never a traceback).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.k11_k12_power_evaluation_identity import K11K12PowerEvaluationError, load_identity  # noqa: E402
from src.k11_k12_power_evaluation_checkpoint import (  # noqa: E402
    parse_strict_json_document,
    resume_dataset_index,
    validate_checkpoint_against_artifact,
    validate_checkpoint_against_canonical_order,
    validate_checkpoint_structure,
)


def _identity():
    return load_identity(repo_root=ROOT)


IDENTITY_SHA = "a" * 64


def _checkpoint(*, next_index=0, completed=None, complete=False, class_count=171, run_mode="pilot20", **overrides):
    identity = _identity()
    completed = completed if completed is not None else [f"img{i}.jpg" for i in range(next_index)]
    record = {
        "schema": "talk2dino-k11-k12-power-evaluation-checkpoint-v1",
        "run_mode": run_mode,
        "identity": identity["identity"]["name"],
        "identity_sha256": IDENTITY_SHA,
        "matched_identity_sha256": identity["parent_identity"]["matched_identity_sha256"],
        "stability_result_sha256": "f" * 64,
        "finite_step_kernel_sha256": "2" * 64,
        "git_commit": "b" * 40,
        "class_count": class_count,
        "image_count_expected": identity["run_modes"]["pilot20_image_count"],
        "image_order_digest": "c" * 64,
        "next_dataset_index": next_index,
        "completed_image_ids": completed,
        "images_completed_count": len(completed),
        "windows_processed_total": 3 * next_index,
        "complete": complete,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "updated_at_utc": "2026-01-01T00:00:01+00:00",
    }
    record.update(overrides)
    return record


def _validate(record):
    validate_checkpoint_structure(record, identity=_identity(), identity_sha256=IDENTITY_SHA)


# ---------------------------------------------------------------------------
# Finding 1: checkpoint corruption tests 1-25
# ---------------------------------------------------------------------------


def test_1_next_index_greater_than_id_count_rejected():
    record = _checkpoint(next_index=10)
    record["next_dataset_index"] = 11
    with pytest.raises(K11K12PowerEvaluationError, match="next_dataset_index"):
        _validate(record)


def test_2_next_index_less_than_id_count_rejected():
    record = _checkpoint(next_index=10)
    record["next_dataset_index"] = 9
    with pytest.raises(K11K12PowerEvaluationError, match="next_dataset_index"):
        _validate(record)


def test_3_duplicate_ids_rejected():
    record = _checkpoint(next_index=5, completed=["a.jpg", "b.jpg", "a.jpg", "c.jpg", "d.jpg"])
    with pytest.raises(K11K12PowerEvaluationError, match="duplicate"):
        _validate(record)


def test_4_missing_dataset_index_rejected_via_artifact_prefix():
    """A checkpoint that's internally self-consistent (next_index ==
    len(completed_image_ids)) can still hide a missing/reordered/future
    dataset index -- that's caught by the artifact cross-check, which
    independently re-derives the expected index prefix."""
    record = _checkpoint(next_index=5)
    _validate(record)  # passes structural check alone
    manifest = {"image_ids": record["completed_image_ids"], "dataset_indices": [0, 1, 2, 4, 5], "class_count": 171}  # index 3 missing, 5 instead
    arrays = _valid_arrays(5)
    with pytest.raises(K11K12PowerEvaluationError, match="canonical prefix"):
        validate_checkpoint_against_artifact(record, stats_manifest=manifest, stats_arrays=arrays)


def test_5_reordered_indices_rejected():
    record = _checkpoint(next_index=4)
    manifest = {"image_ids": record["completed_image_ids"], "dataset_indices": [0, 2, 1, 3], "class_count": 171}
    arrays = _valid_arrays(4)
    with pytest.raises(K11K12PowerEvaluationError, match="canonical prefix"):
        validate_checkpoint_against_artifact(record, stats_manifest=manifest, stats_arrays=arrays)


def test_6_skipped_middle_index_rejected():
    record = _checkpoint(next_index=4)
    manifest = {"image_ids": record["completed_image_ids"], "dataset_indices": [0, 1, 3, 4], "class_count": 171}  # 2 skipped
    arrays = _valid_arrays(4)
    with pytest.raises(K11K12PowerEvaluationError, match="canonical prefix"):
        validate_checkpoint_against_artifact(record, stats_manifest=manifest, stats_arrays=arrays)


def test_7_future_index_rejected():
    record = _checkpoint(next_index=3)
    manifest = {"image_ids": record["completed_image_ids"], "dataset_indices": [0, 1, 99], "class_count": 171}
    arrays = _valid_arrays(3)
    with pytest.raises(K11K12PowerEvaluationError, match="canonical prefix"):
        validate_checkpoint_against_artifact(record, stats_manifest=manifest, stats_arrays=arrays)


def _valid_arrays(n, class_count=171):
    return {
        key: np.zeros((n, class_count), dtype=np.int64)
        for key in ("label", "intersect_k11", "union_k11", "pred_k11", "intersect_k12", "union_k12", "pred_k12")
    }


def _valid_manifest(record, class_count=171):
    return {
        "image_ids": record["completed_image_ids"],
        "dataset_indices": list(range(record["next_dataset_index"])),
        "class_count": class_count,
    }


def test_8_k11_k12_row_count_mismatch_rejected():
    record = _checkpoint(next_index=3)
    manifest = _valid_manifest(record)
    arrays = _valid_arrays(3)
    arrays["intersect_k12"] = np.zeros((2, 171), dtype=np.int64)  # one fewer row than k11
    with pytest.raises(K11K12PowerEvaluationError):
        validate_checkpoint_against_artifact(record, stats_manifest=manifest, stats_arrays=arrays)


def test_9_row_count_differs_from_id_count_rejected():
    record = _checkpoint(next_index=3)
    manifest = _valid_manifest(record)
    arrays = _valid_arrays(2)  # only 2 rows for 3 completed images
    with pytest.raises(K11K12PowerEvaluationError, match="rows"):
        validate_checkpoint_against_artifact(record, stats_manifest=manifest, stats_arrays=arrays)


def test_10_negative_aggregate_values_rejected():
    """Stand-in for 'aggregate arrays disagree with rows': a negative
    sufficient-statistic value can never arise from real histogram counts,
    so it is rejected as an internal-consistency violation."""
    record = _checkpoint(next_index=2)
    manifest = _valid_manifest(record)
    arrays = _valid_arrays(2)
    arrays["intersect_k11"][0, 0] = -1
    with pytest.raises(K11K12PowerEvaluationError, match="negative or non-finite"):
        validate_checkpoint_against_artifact(record, stats_manifest=manifest, stats_arrays=arrays)


def test_10b_intersect_exceeding_union_rejected():
    record = _checkpoint(next_index=2)
    manifest = _valid_manifest(record)
    arrays = _valid_arrays(2)
    arrays["intersect_k11"][0, 0] = 5
    arrays["union_k11"][0, 0] = 3
    with pytest.raises(K11K12PowerEvaluationError, match="exceeds union"):
        validate_checkpoint_against_artifact(record, stats_manifest=manifest, stats_arrays=arrays)


def test_11_gt_duplicated_per_variant_rejected():
    """GT ('label') is architecturally shared, never duplicated per
    variant -- an artifact carrying label_k11/label_k12 is rejected."""
    record = _checkpoint(next_index=2)
    manifest = _valid_manifest(record)
    arrays = _valid_arrays(2)
    arrays["label_k11"] = np.zeros((2, 171), dtype=np.int64)
    with pytest.raises(K11K12PowerEvaluationError, match="shared"):
        validate_checkpoint_against_artifact(record, stats_manifest=manifest, stats_arrays=arrays)


def test_12_image_digest_mismatch_rejected():
    identity = _identity()
    record = _checkpoint(next_index=0)
    with pytest.raises(K11K12PowerEvaluationError, match="image_order_digest"):
        validate_checkpoint_against_canonical_order(record, ["a.jpg", "b.jpg"], image_order_digest="d" * 64)


def test_12b_canonical_prefix_mismatch_rejected():
    record = _checkpoint(next_index=2, completed=["a.jpg", "WRONG.jpg"])
    with pytest.raises(K11K12PowerEvaluationError, match="canonical dataset image-ID prefix"):
        validate_checkpoint_against_canonical_order(
            record, ["a.jpg", "b.jpg", "c.jpg"], image_order_digest=record["image_order_digest"]
        )


def test_12c_canonical_prefix_match_passes():
    record = _checkpoint(next_index=2, completed=["a.jpg", "b.jpg"])
    validate_checkpoint_against_canonical_order(
        record, ["a.jpg", "b.jpg", "c.jpg"], image_order_digest=record["image_order_digest"]
    )  # must not raise


def test_13_and_14_telemetry_reconciliation_via_deterministic_derivation():
    """Telemetry image/window-count reconciliation is enforced by deriving
    operation_telemetry directly from windows_processed_total (see
    diagnostics/run_matched_k11_k12_evaluation.py
    ._operation_telemetry_for_window_count) rather than from a per-run
    list that would silently under-report after a resume -- exercised end
    to end in test_matched_power_evaluator_integration.py's resume tests."""
    sys.path.insert(0, str(ROOT))
    from diagnostics.run_matched_k11_k12_evaluation import _operation_telemetry_for_window_count

    totals_10 = _operation_telemetry_for_window_count(10)
    totals_20 = _operation_telemetry_for_window_count(20)
    assert totals_20["backbone_snapshot_calls"] == 20
    assert totals_20["k11_updates"] == 320 * 20
    assert totals_20["k12_updates"] == 320 * 20
    # linear in window count -- confirms this is a pure derivation, not an
    # accidentally-cached or resume-unsafe accumulation
    assert totals_20["backbone_snapshot_calls"] == 2 * totals_10["backbone_snapshot_calls"]


def test_15_artifact_sha_mismatch_detected(tmp_path):
    from diagnostics.run_matched_k11_k12_evaluation import _write_per_image_stats_atomically, _load_per_image_stats

    manifest_path = tmp_path / "stats.json"
    arr = np.zeros((1, 3), dtype=np.int64)
    _write_per_image_stats_atomically(
        manifest_path, schema_name="x", class_count=3, dataset_indices=[0], image_ids=["a.jpg"],
        label=arr, intersect_k11=arr, union_k11=arr, pred_k11=arr,
        intersect_k12=arr, union_k12=arr, pred_k12=arr,
    )
    npz_path = manifest_path.with_suffix(".npz")
    npz_path.write_bytes(npz_path.read_bytes() + b"corruption")
    with pytest.raises(K11K12PowerEvaluationError, match="SHA256"):
        _load_per_image_stats(manifest_path)


def test_16_run_mode_expected_count_mismatch_rejected():
    record = _checkpoint(next_index=0, run_mode="pilot100")
    record["image_count_expected"] = 20  # pilot20's count, not pilot100's
    with pytest.raises(K11K12PowerEvaluationError, match="image_count_expected"):
        _validate(record)


def test_16b_run_mode_disagrees_with_caller_rejected():
    record = _checkpoint(next_index=0, run_mode="pilot100")
    with pytest.raises(K11K12PowerEvaluationError, match="run_mode"):
        validate_checkpoint_structure(
            record, identity=_identity(), identity_sha256=IDENTITY_SHA, run_mode="pilot20"
        )


def test_17_incomplete_checkpoint_marked_complete_rejected():
    record = _checkpoint(next_index=5, complete=True)  # 5 != pilot20's 20
    with pytest.raises(K11K12PowerEvaluationError):
        _validate(record)


def test_18_complete_checkpoint_missing_final_row_rejected():
    """complete=true with completed_image_ids one short of the expected
    count."""
    record = _checkpoint(next_index=19, completed=[f"img{i}.jpg" for i in range(19)], complete=True)
    with pytest.raises(K11K12PowerEvaluationError):
        _validate(record)


def test_19_canonical_image_id_prefix_mismatch_rejected():
    record = _checkpoint(next_index=3, completed=["a.jpg", "b.jpg", "c.jpg"])
    with pytest.raises(K11K12PowerEvaluationError):
        validate_checkpoint_against_canonical_order(
            record, ["a.jpg", "DIFFERENT.jpg", "c.jpg"], image_order_digest=record["image_order_digest"]
        )


def test_20_resume_from_valid_zero_image_checkpoint():
    record = _checkpoint(next_index=0, completed=[])
    _validate(record)  # must not raise
    assert resume_dataset_index(record) == 0


def test_21_resume_from_valid_partial_checkpoint():
    record = _checkpoint(next_index=7)
    _validate(record)
    assert resume_dataset_index(record) == 7


def test_22_resume_from_completed_checkpoint_raises_per_documented_behavior():
    record = _checkpoint(next_index=20, completed=[f"img{i}.jpg" for i in range(20)], complete=True)
    _validate(record)  # structurally valid
    with pytest.raises(K11K12PowerEvaluationError, match="already complete"):
        resume_dataset_index(record)


# --- class_count invariants ---


def test_class_count_type_errors_rejected():
    for bad in (True, 3.0, "171", None):
        record = _checkpoint(next_index=0, class_count=bad)
        with pytest.raises(K11K12PowerEvaluationError):
            _validate(record)


def test_class_count_cross_check_against_live_value():
    record = _checkpoint(next_index=0, class_count=5)
    with pytest.raises(K11K12PowerEvaluationError, match="class_count"):
        validate_checkpoint_structure(
            record, identity=_identity(), identity_sha256=IDENTITY_SHA, class_count=171
        )
    validate_checkpoint_structure(
        record, identity=_identity(), identity_sha256=IDENTITY_SHA, class_count=5
    )  # matching value passes


def test_manifest_class_count_disagreement_rejected():
    record = _checkpoint(next_index=2, class_count=171)
    manifest = _valid_manifest(record, class_count=5)  # disagrees with checkpoint.class_count
    arrays = _valid_arrays(2)
    with pytest.raises(K11K12PowerEvaluationError, match="class_count"):
        validate_checkpoint_against_artifact(record, stats_manifest=manifest, stats_arrays=arrays)


# --- hash/identity binding cross-checks ---


def test_stability_result_sha_mismatch_rejected():
    record = _checkpoint(next_index=0)
    with pytest.raises(K11K12PowerEvaluationError, match="stability_result_sha256"):
        validate_checkpoint_structure(
            record, identity=_identity(), identity_sha256=IDENTITY_SHA, stability_result_sha256="0" * 64
        )


def test_kernel_sha_mismatch_rejected():
    record = _checkpoint(next_index=0)
    with pytest.raises(K11K12PowerEvaluationError, match="finite_step_kernel_sha256"):
        validate_checkpoint_structure(
            record, identity=_identity(), identity_sha256=IDENTITY_SHA, finite_step_kernel_sha256="0" * 64
        )


def test_wrong_schema_rejected():
    record = _checkpoint(next_index=0)
    record["schema"] = "wrong-schema-name"
    with pytest.raises(K11K12PowerEvaluationError, match="schema"):
        _validate(record)


def test_extra_key_rejected():
    record = _checkpoint(next_index=0)
    record["extra_unexpected_key"] = 1
    with pytest.raises(K11K12PowerEvaluationError):
        _validate(record)


def test_missing_key_rejected():
    record = _checkpoint(next_index=0)
    del record["windows_processed_total"]
    with pytest.raises(K11K12PowerEvaluationError):
        _validate(record)


# ---------------------------------------------------------------------------
# Finding 2: strict JSON loading tests 1-16
# ---------------------------------------------------------------------------


def test_json_1_missing_checkpoint(tmp_path):
    with pytest.raises(K11K12PowerEvaluationError, match="file does not exist"):
        parse_strict_json_document(tmp_path / "missing.json", label="checkpoint")


def test_json_2_directory_checkpoint(tmp_path):
    directory = tmp_path / "adir"
    directory.mkdir()
    with pytest.raises(K11K12PowerEvaluationError, match="directory"):
        parse_strict_json_document(directory, label="checkpoint")


def test_json_3_invalid_utf8(tmp_path):
    path = tmp_path / "bad.json"
    path.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(K11K12PowerEvaluationError, match="UTF-8"):
        parse_strict_json_document(path, label="checkpoint")


def test_json_4_empty_file(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("")
    with pytest.raises(K11K12PowerEvaluationError, match="empty"):
        parse_strict_json_document(path, label="checkpoint")


def test_json_5_truncated_json(tmp_path):
    path = tmp_path / "truncated.json"
    path.write_text('{"a": 1, "b": ')
    with pytest.raises(K11K12PowerEvaluationError):
        parse_strict_json_document(path, label="checkpoint")


def test_json_6_trailing_garbage(tmp_path):
    path = tmp_path / "trailing.json"
    path.write_text('{"a": 1} extra')
    with pytest.raises(K11K12PowerEvaluationError):
        parse_strict_json_document(path, label="checkpoint")


def test_json_7_duplicate_key(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text('{"a": 1, "a": 2}')
    with pytest.raises(K11K12PowerEvaluationError, match="duplicate"):
        parse_strict_json_document(path, label="checkpoint")


def test_json_8_nan(tmp_path):
    path = tmp_path / "nan.json"
    path.write_text('{"a": NaN}')
    with pytest.raises(K11K12PowerEvaluationError, match="non-finite"):
        parse_strict_json_document(path, label="checkpoint")


def test_json_9_infinity(tmp_path):
    path = tmp_path / "inf.json"
    path.write_text('{"a": Infinity}')
    with pytest.raises(K11K12PowerEvaluationError, match="non-finite"):
        parse_strict_json_document(path, label="checkpoint")


def test_json_10_list_root(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2]")
    with pytest.raises(K11K12PowerEvaluationError, match="root"):
        parse_strict_json_document(path, label="checkpoint")


def test_json_11_string_root(tmp_path):
    path = tmp_path / "str.json"
    path.write_text('"hello"')
    with pytest.raises(K11K12PowerEvaluationError, match="root"):
        parse_strict_json_document(path, label="checkpoint")


def test_json_12_wrong_schema_after_valid_parse(tmp_path):
    path = tmp_path / "wrong_schema.json"
    path.write_text(json.dumps({"schema": "not-a-real-schema"}))
    doc = parse_strict_json_document(path, label="checkpoint")  # parses fine
    assert doc["schema"] == "not-a-real-schema"  # schema validity is a separate, later check


def test_json_13_and_14_exit_code_and_no_traceback_via_cli(tmp_path):
    import subprocess

    broken = tmp_path / "broken.json"
    broken.write_text("{not valid")
    result = subprocess.run(
        [sys.executable, str(ROOT / "verify_k11_k12_power_evaluation.py"), "verify-checkpoint",
         "--checkpoint", str(broken), "--repo-root", str(ROOT)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "K11/K12 POWER EVALUATION VERIFICATION FAIL" in result.stderr


def test_json_15_input_bytes_unchanged(tmp_path):
    path = tmp_path / "broken.json"
    original = b"{not valid json"
    path.write_bytes(original)
    try:
        parse_strict_json_document(path, label="checkpoint")
    except K11K12PowerEvaluationError:
        pass
    assert path.read_bytes() == original


def test_json_16_no_output_created_on_malformed_checkpoint(tmp_path):
    import subprocess

    broken = tmp_path / "broken.json"
    broken.write_text("{not valid")
    result_path = tmp_path / "result.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "diagnostics/run_matched_k11_k12_evaluation.py"),
         "--run-mode", "pilot20", "--checkpoint", str(broken), "--result", str(result_path),
         "--per-image-stats", str(tmp_path / "stats.json"),
         "--stability-result", "/scratch/haree/e12_k11_k12_stability/result-20300858.json",
         "--device", "cpu", "--resume"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert not result_path.exists()


# ---------------------------------------------------------------------------
# Checkpoint input-immutability regression tests
#
# validate_checkpoint_structure is documented (and was independently
# verified) to never mutate its input mapping -- these are the permanent
# regression tests that guard that property, for both a valid checkpoint
# (the success path) and a rejected checkpoint (the failure path, since a
# validator mutating its input while raising would be at least as
# concerning as mutating on success). The checkpoint schema itself is
# flat (completed_image_ids is its only list field; there are no nested
# mappings), so "deeply nested" here means a realistically large,
# structurally rich completed_image_ids list -- never inventing fields
# (tensors/arrays, extra nested objects) that are not part of the actual
# schema merely to make the fixture look more elaborate.
# ---------------------------------------------------------------------------


def _deeply_populated_checkpoint(*, next_index=17, complete=False):
    """A checkpoint fixture with every field populated as richly as the
    real schema allows: a long, varied completed_image_ids list (the only
    nested/collection field in the schema), never fabricating fields the
    schema does not define."""
    completed = [f"images/val2017/subset_{i:02d}/img_{i:05d}.jpg" for i in range(next_index)]
    return _checkpoint(next_index=next_index, completed=completed, complete=complete)


def test_valid_checkpoint_deep_equality_preserved_after_validation():
    import copy

    checkpoint = _deeply_populated_checkpoint(next_index=17, complete=False)
    reference = copy.deepcopy(checkpoint)

    validate_checkpoint_structure(checkpoint, identity=_identity(), identity_sha256=IDENTITY_SHA)

    assert checkpoint == reference
    # explicit nested-collection checks, not just top-level dict equality
    assert checkpoint["completed_image_ids"] == reference["completed_image_ids"]
    assert checkpoint["completed_image_ids"] is not None
    for original_id, current_id in zip(reference["completed_image_ids"], checkpoint["completed_image_ids"]):
        assert original_id == current_id
    assert len(checkpoint) == len(reference)
    assert set(checkpoint.keys()) == set(reference.keys())


def test_valid_checkpoint_no_fields_added_removed_or_reordered_by_validation():
    checkpoint = _deeply_populated_checkpoint(next_index=5, complete=False)
    keys_before = list(checkpoint.keys())
    values_before = {k: (v if not isinstance(v, list) else list(v)) for k, v in checkpoint.items()}

    validate_checkpoint_structure(checkpoint, identity=_identity(), identity_sha256=IDENTITY_SHA)

    assert list(checkpoint.keys()) == keys_before, "validation must not add, remove, or reorder top-level keys"
    for key, original_value in values_before.items():
        assert checkpoint[key] == original_value, f"validation must not coerce/normalize field {key!r}"


def test_rejected_checkpoint_deep_equality_preserved_after_validation():
    """A checkpoint that WILL be rejected (next_dataset_index inconsistent
    with completed_image_ids -- the exact originally-confirmed regression
    invariant) must still be left byte-for-byte unmodified: a validator
    that mutates its input on the way to raising is its own hazard,
    independent of whether the rejection itself is correct."""
    import copy

    checkpoint = _deeply_populated_checkpoint(next_index=17, complete=False)
    checkpoint["next_dataset_index"] = 999  # forces rejection
    reference = copy.deepcopy(checkpoint)

    with pytest.raises(K11K12PowerEvaluationError):
        validate_checkpoint_structure(checkpoint, identity=_identity(), identity_sha256=IDENTITY_SHA)

    assert checkpoint == reference
    assert checkpoint["completed_image_ids"] == reference["completed_image_ids"]
    assert list(checkpoint.keys()) == list(reference.keys())


def test_rejected_checkpoint_no_fields_added_removed_or_normalized():
    checkpoint = _deeply_populated_checkpoint(next_index=8, complete=False)
    checkpoint["completed_image_ids"] = checkpoint["completed_image_ids"] + ["duplicate_free_but_wrong_count.jpg"]
    # now images_completed_count (8) disagrees with len(completed_image_ids) (9) -- rejected
    keys_before = list(checkpoint.keys())
    completed_before = list(checkpoint["completed_image_ids"])

    with pytest.raises(K11K12PowerEvaluationError):
        validate_checkpoint_structure(checkpoint, identity=_identity(), identity_sha256=IDENTITY_SHA)

    assert list(checkpoint.keys()) == keys_before
    assert checkpoint["completed_image_ids"] == completed_before
    assert checkpoint["completed_image_ids"] is not None


def test_validation_against_artifact_also_does_not_mutate_checkpoint_or_manifest():
    """validate_checkpoint_against_artifact takes both the checkpoint and
    the per-image-stats manifest -- confirm neither is mutated."""
    import copy

    checkpoint = _deeply_populated_checkpoint(next_index=3, complete=False)
    checkpoint_reference = copy.deepcopy(checkpoint)
    manifest = {
        "image_ids": list(checkpoint["completed_image_ids"]),
        "dataset_indices": [0, 1, 2],
        "class_count": 171,
    }
    manifest_reference = copy.deepcopy(manifest)
    arrays = {
        key: np.zeros((3, 171), dtype=np.int64)
        for key in ("label", "intersect_k11", "union_k11", "pred_k11", "intersect_k12", "union_k12", "pred_k12")
    }

    validate_checkpoint_against_artifact(checkpoint, stats_manifest=manifest, stats_arrays=arrays)

    assert checkpoint == checkpoint_reference
    assert manifest == manifest_reference
