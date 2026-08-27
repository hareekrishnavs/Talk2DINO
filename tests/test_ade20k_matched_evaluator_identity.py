"""Static/mutation tests for the ADE20K matched-evaluator identity, its
checkpoint schema, and its result schema. CPU-only."""

from __future__ import annotations

import copy
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.ade20k_matched_evaluator_identity import (  # noqa: E402
    Ade20kMatchedEvaluatorIdentityError,
    load_identity,
    validate_bridge_checkpoint_binding,
    validate_static_configuration,
)
from src.ade20k_matched_evaluator_checkpoint import (  # noqa: E402
    validate_checkpoint_against_canonical_order,
    validate_checkpoint_structure,
)
from src.ade20k_matched_evaluator_report import verify_record  # noqa: E402

IDENTITY_PATH = ROOT / "evaluation_identities/e12_ade20k_matched_evaluator.toml"


def _load_raw() -> dict:
    with IDENTITY_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _fmt(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    raise TypeError(f"unsupported TOML value type: {type(value)}")


def _dump_toml(document: dict) -> str:
    lines: list[str] = [f'format_version = {_fmt(document["format_version"])}']
    for section, body in document.items():
        if section == "format_version":
            continue
        lines.append("")
        lines.append(f"[{section}]")
        for key, value in body.items():
            lines.append(f"{key} = {_fmt(value)}")
    return "\n".join(lines) + "\n"


def _write_and_load(tmp_path: Path, document: dict):
    path = tmp_path / "identity.toml"
    path.write_text(_dump_toml(document), encoding="utf-8")
    return load_identity(path, repo_root=ROOT)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_real_identity_loads_and_validates():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    assert identity["identity"]["name"] == "e12-ade20k-matched-evaluator"
    result = validate_static_configuration(repo_root=ROOT, check_git=True)
    assert result["expected_image_count"] == 2000
    assert result["matched_identity_name"] == "e12-matched-k11-k12-t320"
    assert result["source_identity_name"] == "e12-ade20k-dataset-source"


def test_bridge_checkpoint_binding_accepts_real_checkpoint():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    observed = validate_bridge_checkpoint_binding(ROOT, identity)
    assert observed == identity["e3_config"]["projection_checkpoint_sha256"]


def test_bridge_checkpoint_binding_rejects_wrong_hash():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    tampered = copy.deepcopy(identity)
    tampered["e3_config"]["projection_checkpoint_sha256"] = "f" * 64
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError, match="SHA256 mismatch"):
        validate_bridge_checkpoint_binding(ROOT, tampered)


def test_no_private_path_in_identity_file():
    text = IDENTITY_PATH.read_text(encoding="utf-8")
    for fragment in ("/scratch/", "/project/", "/home/"):
        assert fragment not in text


# ---------------------------------------------------------------------------
# Mutation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("protocol", "expected_image_count", 2001),
        ("dataset", "class_count", 151),
        ("dataset", "background_class_evaluated", True),
        ("dataset", "ignore_index", 254),
        ("dataset_root_override", "canonical_configured_root", "./data/ade2020"),
        ("e3_config", "pamr", True),
        ("background_protocol", "mechanism", "constant_threshold_channel"),
        ("background_protocol", "background_class_evaluated", True),
        ("comparisons", "metric_unit", "percent_0_100"),
        ("run_modes", "pilot20_images", 21),
        ("run_modes", "pilot100_images", 101),
        ("run_modes", "full_images", 2001),
    ],
)
def test_mutated_field_rejected(tmp_path, section, key, value):
    document = _load_raw()
    document[section][key] = value
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_matched_identity_name_rejected():
    """matched_identity_name's correctness against the real loaded parent
    identity is checked relationally in validate_static_configuration
    (via _validate_matched_parent), not by load_identity's own schema
    validation, which only requires a non-empty string."""
    from src.ade20k_matched_evaluator_identity import _validate_matched_parent

    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    tampered = copy.deepcopy(identity)
    tampered["parent_identities"]["matched_identity_name"] = "e12-matched-k11-k12-t160"
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        _validate_matched_parent(ROOT, tampered)


def test_wrong_source_identity_sha256_rejected():
    """load_identity accepts any well-formed SHA256 string for the parent
    identity fields; the RELATIONAL cross-check (does it match the real
    file's actual hash) happens in validate_static_configuration, matching
    the split already proven for the ADE20K source identity's own
    class_names_digest test."""
    from src.ade20k_matched_evaluator_identity import _validate_ade20k_source_parent

    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    tampered = copy.deepcopy(identity)
    tampered["parent_identities"]["ade20k_source_identity_sha256"] = "f" * 64
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        _validate_ade20k_source_parent(ROOT, tampered)


# ---------------------------------------------------------------------------
# Checkpoint schema
# ---------------------------------------------------------------------------


@pytest.fixture
def identity():
    return load_identity(IDENTITY_PATH, repo_root=ROOT)


@pytest.fixture
def identity_sha256():
    import hashlib

    return hashlib.sha256(IDENTITY_PATH.read_bytes()).hexdigest()


def _valid_checkpoint(identity, identity_sha256, *, run_mode="pilot20", next_index=0, completed=None, complete=False):
    completed = completed if completed is not None else []
    now = "2026-01-01T00:00:00+00:00"
    return {
        "schema": identity["checkpoint"]["schema_name"],
        "run_mode": run_mode,
        "identity": identity["identity"]["name"],
        "identity_sha256": identity_sha256,
        "matched_identity_sha256": identity["parent_identities"]["matched_identity_sha256"],
        "ade20k_source_identity_sha256": identity["parent_identities"]["ade20k_source_identity_sha256"],
        "ade20k_source_manifest_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "class_count": identity["dataset"]["class_count"],
        "live_class_names_digest": identity["dataset"]["class_names_digest"],
        "image_count_expected": identity["run_modes"][f"{run_mode}_images"],
        "image_order_digest": "c" * 64,
        "next_dataset_index": next_index,
        "completed_image_ids": completed,
        "images_completed_count": len(completed),
        "windows_processed_total": max(len(completed), 1) if completed else 0,
        "complete": complete,
        "created_at_utc": now,
        "updated_at_utc": now,
    }


def test_valid_checkpoint_passes(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256)
    validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_checkpoint_wrong_identity_sha256_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256)
    checkpoint["identity_sha256"] = "f" * 64
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_checkpoint_next_index_completed_mismatch_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, next_index=5, completed=["a", "b"])
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_checkpoint_duplicate_completed_id_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, next_index=2, completed=["x", "x"])
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_checkpoint_wrong_run_mode_image_count_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, run_mode="pilot20")
    checkpoint["image_count_expected"] = 100
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_checkpoint_complete_but_wrong_next_index_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, next_index=5, completed=[f"id{i}" for i in range(5)], complete=True)
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_checkpoint_unknown_key_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256)
    checkpoint["extra_field"] = 1
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_checkpoint_against_canonical_order_prefix_mismatch(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, next_index=2, completed=["a", "b"])
    checkpoint["image_order_digest"] = "known-digest"
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        validate_checkpoint_against_canonical_order(
            checkpoint, ["a", "different"], image_order_digest="known-digest",
        )


def test_checkpoint_against_canonical_order_digest_mismatch(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, next_index=2, completed=["a", "b"])
    checkpoint["image_order_digest"] = "stale-digest"
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        validate_checkpoint_against_canonical_order(
            checkpoint, ["a", "b"], image_order_digest="fresh-digest",
        )


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


def _valid_result(identity, identity_sha256, *, run_mode="pilot20"):
    windows_total = 3
    metrics = {"aAcc": 50.0, "mIoU": 30.0, "mAcc": 40.0}
    metrics_k11 = {"aAcc": 51.0, "mIoU": 31.0, "mAcc": 41.0}
    metrics_k12 = {"aAcc": 52.0, "mIoU": 32.0, "mAcc": 42.0}
    return {
        "schema": identity["run_modes"][f"{run_mode}_schema_name"],
        "run_mode": run_mode,
        "identity": identity["identity"]["name"],
        "identity_sha256": identity_sha256,
        "matched_identity_sha256": identity["parent_identities"]["matched_identity_sha256"],
        "ade20k_source_identity_sha256": identity["parent_identities"]["ade20k_source_identity_sha256"],
        "ade20k_source_manifest_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "complete": True,
        "final": run_mode == "full",
        "device": "cuda",
        "gpu_model": "H100",
        "torch_version": "2.0.0",
        "cuda_version": "12.1",
        "image_count_expected": identity["run_modes"][f"{run_mode}_images"],
        "image_count_processed": identity["run_modes"][f"{run_mode}_images"],
        "image_order_digest": "c" * 64,
        "windows_processed_total": windows_total,
        "class_count": identity["dataset"]["class_count"],
        "live_class_names_digest": identity["dataset"]["class_names_digest"],
        "metrics_E3": metrics,
        "metrics_k11": metrics_k11,
        "metrics_k12": metrics_k12,
        "delta_mIoU_k11_minus_k12_percentage_points": metrics_k11["mIoU"] - metrics_k12["mIoU"],
        "delta_mIoU_k11_minus_E3_percentage_points": metrics_k11["mIoU"] - metrics["mIoU"],
        "delta_mIoU_k12_minus_E3_percentage_points": metrics_k12["mIoU"] - metrics["mIoU"],
        "metric_unit": "percent_0_100",
        "metric_source": "full_precision_area_statistics_from_mmseg_pre_eval",
        "per_image_stats_manifest_path": "/tmp/x.json",
        "per_image_stats_manifest_sha256": "d" * 64,
        "per_image_stats_npz_sha256": "e" * 64,
        "operation_telemetry": {
            "backbone_snapshot_calls": windows_total, "dino_feature_extractions": windows_total,
            "topk_selection_calls": windows_total, "graph_normalizations": windows_total * 2,
            "finite_step_propagations": windows_total * 2, "e3_propagations": 0,
            "k11_updates": windows_total * 320, "k12_updates": windows_total * 320,
            "sigmoid_calls": windows_total * 3, "interpolation_calls": windows_total * 3,
        },
        "phase_runtime_seconds": {"total": 12.5},
        "peak_gpu_memory_bytes": 1000,
        "resumed_from_checkpoint": False,
        "source_git_branch": "e12-connectivity-analysis",
        "failure_reason": None,
    }


def test_valid_result_passes(identity, identity_sha256):
    verify_record(_valid_result(identity, identity_sha256), identity, identity_sha256=identity_sha256)


def test_pilot_result_rejected_as_full_schema(identity, identity_sha256):
    record = _valid_result(identity, identity_sha256, run_mode="pilot20")
    record["schema"] = identity["run_modes"]["full_schema_name"]
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        verify_record(record, identity, identity_sha256=identity_sha256)


def test_final_flag_disagreeing_with_run_mode_rejected(identity, identity_sha256):
    record = _valid_result(identity, identity_sha256, run_mode="pilot20")
    record["final"] = True
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        verify_record(record, identity, identity_sha256=identity_sha256)


def test_nonzero_e3_propagations_rejected(identity, identity_sha256):
    record = _valid_result(identity, identity_sha256)
    record["operation_telemetry"]["e3_propagations"] = 1
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        verify_record(record, identity, identity_sha256=identity_sha256)


def test_delta_disagreeing_with_metrics_rejected(identity, identity_sha256):
    record = _valid_result(identity, identity_sha256)
    record["delta_mIoU_k11_minus_k12_percentage_points"] = 999.0
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        verify_record(record, identity, identity_sha256=identity_sha256)


def test_incomplete_result_rejected(identity, identity_sha256):
    record = _valid_result(identity, identity_sha256)
    record["complete"] = False
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        verify_record(record, identity, identity_sha256=identity_sha256)


def test_wrong_class_count_rejected(identity, identity_sha256):
    record = _valid_result(identity, identity_sha256)
    record["class_count"] = 151
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        verify_record(record, identity, identity_sha256=identity_sha256)


def test_failure_reason_must_be_null(identity, identity_sha256):
    record = _valid_result(identity, identity_sha256)
    record["failure_reason"] = "oops"
    with pytest.raises(Ade20kMatchedEvaluatorIdentityError):
        verify_record(record, identity, identity_sha256=identity_sha256)
