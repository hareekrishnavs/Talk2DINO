"""Strict tests for the DFP Phase 1B conservation-calibration candidate
encoding, selection algorithm, digest functions, and artifact-schema
validator (`src.dfp_phase1b_calibration_artifact`).

CPU-only synthetic tests. No real dataset, no real image, no GT, no
torch, no CUDA, no model/checkpoint. Every candidate/artifact fixture
is hand-built."""

from __future__ import annotations

import copy
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import src.dfp_phase1b_calibration_artifact as m  # noqa: E402

E = m.DfpPhase1bCalibrationArtifactError


def _candidate(image_ordinal, window_ordinal, *, image_id=None, top=0, left=0, bottom=448, right=448,
               canvas_h=896, canvas_w=896, crop_h=448, crop_w=448, patch_h=32, patch_w=32):
    resolved_image_id = image_id if image_id is not None else f"{image_ordinal:012d}"
    return {
        "run_mode": "coco_stuff_500",
        "image_id": resolved_image_id,
        "image_path": f"val2017/{resolved_image_id}.jpg",
        "image_manifest_ordinal": image_ordinal,
        "inference_canvas_height": canvas_h,
        "inference_canvas_width": canvas_w,
        "canonical_window_ordinal": window_ordinal,
        "top": top, "left": left, "bottom": bottom, "right": right,
        "crop_height": crop_h, "crop_width": crop_w,
        "patch_grid_height": patch_h, "patch_grid_width": patch_w,
    }


def _universe(image_count=50, windows_per_image=5):
    return [_candidate(img, win) for img in range(image_count) for win in range(windows_per_image)]


def _residual_summary(sample_count=10, value=1e-9):
    if sample_count == 0:
        return {
            "sample_count": 0, "finite_count": 0, "non_finite_count": 0,
            "signed_minimum": None, "signed_maximum": None,
            "maximum_absolute_residual": None, "mean_absolute_residual": None, "root_mean_square_residual": None,
            "absolute_residual_sum": None, "squared_residual_sum": None,
        }
    abs_sum = value * sample_count
    sq_sum = (value ** 2) * sample_count
    return {
        "sample_count": sample_count, "finite_count": sample_count, "non_finite_count": 0,
        "signed_minimum": -value, "signed_maximum": value,
        "maximum_absolute_residual": value, "mean_absolute_residual": abs_sum / sample_count,
        "root_mean_square_residual": math.sqrt(sq_sum / sample_count),
        "absolute_residual_sum": abs_sum, "squared_residual_sum": sq_sum,
    }


def _selected_rows(result):
    rank_by_index = {t["canonical_candidate_index"]: t["selection_rank"] for t in result.selection_trace}
    key_by_index = {t["canonical_candidate_index"]: t["selection_key"] for t in result.selection_trace}
    rows = []
    for index, record in zip(result.selected_canonical_indices, result.selected_records):
        row = {
            **dict(record),
            "canonical_candidate_index": index,
            "selection_rank": rank_by_index[index],
            "selection_key": key_by_index[index],
            "completed": True,
        }
        for family in m.RESIDUAL_FAMILIES:
            row[family] = _residual_summary()
        rows.append(row)
    return rows


def _artifact_document(universe, *, complete=True, final=True):
    result = m.select_calibration_windows(universe)
    rows = _selected_rows(result)
    if not complete:
        rows = rows[:50]
        for row in rows[25:]:
            row["completed"] = False
    aggregates = {
        family: m.aggregate_residual_summaries([row[family] for row in rows if row["completed"]], label=family)
        for family in m.RESIDUAL_FAMILIES
    }
    return {
        "schema_name": m.ARTIFACT_SCHEMA_NAME, "schema_version": m.ARTIFACT_SCHEMA_VERSION,
        "complete": complete, "final": final,
        "calibration_identity_sha256": "0" * 64,
        "bound_e15_protocol_sha256": "1" * 64,
        "bound_e15_artifacts_sha256": "2" * 64,
        "bound_masked_forward_sha256": "3" * 64,
        "repository_commit": "a" * 40,
        "run_mode": "coco_stuff_500",
        "bound_subset_manifest_sha256": "4" * 64,
        "selected_ids_digest": "5" * 64,
        "candidate_window_count": len(universe),
        "expected_selected_window_count": m.WINDOW_COUNT,
        "completed_selected_window_count": sum(1 for row in rows if row["completed"]),
        "candidate_universe_digest": result.candidate_universe_digest,
        "selected_windows_digest": result.selected_windows_digest,
        "selection_trace_digest": result.selection_trace_digest,
        "device_record": {
            "device_type": "cuda", "gpu_model_name": "NVIDIA A100", "cuda_compute_capability": "8.0",
            "total_device_memory_bytes": 42949672960, "cuda_driver_version": "535.104.05", "cuda_runtime_version": "12.2",
        },
        "software_record": {
            "repository_commit": "a" * 40, "python_version": "3.11.5", "torch_version": "2.12.0",
            "numpy_version": "2.4.2", "cuda_version": "12.2", "cudnn_version": "8.9",
            "deterministic_algorithms_enabled": True, "tf32_enabled": False, "autocast_enabled": False,
        },
        "propagation_precision_record": {
            "graph_storage_dtype": "float32", "canonical_recurrence_dtype": "float64",
            "masked_recurrence_dtype": "float64", "residual_accumulation_dtype": "float64",
            "alpha": 0.98, "steps": 320, "k": 12, "chunk_size": 4,
        },
        "selected_window_records": rows,
        "aggregate_residual_summaries": aggregates,
    }, result


# ---------------------------------------------------------------------
# Candidate validation.
# ---------------------------------------------------------------------


def test_valid_candidate_record_accepted():
    m.validate_candidate_record(_candidate(0, 0))


@pytest.mark.parametrize("field", m.CANDIDATE_FIELDS)
def test_missing_candidate_field_rejected(field):
    record = _candidate(0, 0)
    del record[field]
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_unknown_candidate_field_rejected():
    record = _candidate(0, 0)
    record["extra"] = 1
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_wrong_run_mode_rejected():
    record = _candidate(0, 0)
    record["run_mode"] = "ade20k_500"
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_path_traversal_rejected():
    record = _candidate(0, 0)
    record["image_path"] = "../../etc/passwd"
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_absolute_path_rejected():
    record = _candidate(0, 0)
    record["image_path"] = "/etc/passwd"
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_bool_as_int_field_rejected():
    record = _candidate(0, 0)
    record["top"] = True
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_negative_int_field_rejected():
    record = _candidate(0, 0)
    record["top"] = -1
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_inverted_top_bottom_rejected():
    record = _candidate(0, 0, top=100, bottom=50)
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_out_of_canvas_bottom_rejected():
    record = _candidate(0, 0, bottom=1000, canvas_h=896)
    with pytest.raises(E):
        m.validate_candidate_record(record)


# ---------------------------------------------------------------------
# Encoding.
# ---------------------------------------------------------------------


def test_encode_candidate_record_matches_frozen_json_dumps_equivalent():
    import json

    record = _candidate(3, 2)
    encoded = m.encode_candidate_record(record)
    expected = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    assert encoded == expected


def test_encoding_is_alphabetically_key_ordered():
    record = _candidate(3, 2)
    encoded = m.encode_candidate_record(record).decode("utf-8")
    assert encoded.startswith('{"bottom"')  # first alphabetical key


def test_encoding_has_no_surrounding_whitespace():
    record = _candidate(3, 2)
    encoded = m.encode_candidate_record(record).decode("utf-8")
    assert ", " not in encoded and ": " not in encoded


def test_encoding_has_no_trailing_newline():
    record = _candidate(3, 2)
    encoded = m.encode_candidate_record(record)
    assert not encoded.endswith(b"\n")


# ---------------------------------------------------------------------
# Selection determinism.
# ---------------------------------------------------------------------


def test_fewer_than_200_candidates_rejected():
    with pytest.raises(E):
        m.select_calibration_windows(_universe(image_count=39, windows_per_image=5))  # 195 total


def test_exactly_200_candidates_all_selected():
    universe = _universe(image_count=40, windows_per_image=5)  # exactly 200
    result = m.select_calibration_windows(universe)
    assert len(result.selected_records) == 200
    assert set(result.selected_canonical_indices) == set(range(200))


def test_more_than_200_candidates_selects_exactly_200():
    result = m.select_calibration_windows(_universe(image_count=50, windows_per_image=5))  # 250 total
    assert len(result.selected_records) == 200


def test_identical_image_ids_different_window_ordinals_are_distinct_candidates():
    universe = _universe(image_count=41, windows_per_image=5)
    # two candidates sharing an image_id string but different manifest ordinal (a malformed
    # fixture) must still be treated as distinct by (image_manifest_ordinal, window_ordinal).
    universe.append(_candidate(999, 0, image_id="000000000000"))
    result = m.select_calibration_windows(universe)
    assert result.candidate_universe_size == len(universe)


def test_duplicate_candidate_rejected():
    universe = _universe(image_count=40, windows_per_image=5)
    universe.append(universe[0])
    with pytest.raises(E):
        m.select_calibration_windows(universe)


def test_duplicate_order_key_with_different_geometry_rejected():
    universe = _universe(image_count=40, windows_per_image=5)
    clashing = _candidate(0, 0, top=10, bottom=458)  # same (0,0) order key, different geometry
    universe.append(clashing)
    with pytest.raises(E):
        m.select_calibration_windows(universe)


@pytest.mark.parametrize("windows_per_image", [1, 5])
def test_flush_right_and_flush_bottom_windows_participate_normally(windows_per_image):
    image_count = max(40, 205 // windows_per_image + 1)
    universe = _universe(image_count=image_count, windows_per_image=windows_per_image)
    universe.append(_candidate(100000, 0, top=448, left=448, bottom=896, right=896))  # flush bottom-right
    result = m.select_calibration_windows(universe)
    assert len(result.selected_records) == 200


def test_input_order_independence_shuffled():
    import random

    universe = _universe(image_count=50, windows_per_image=5)
    baseline = m.select_calibration_windows(universe)
    shuffled = universe[:]
    random.Random(7).shuffle(shuffled)
    shuffled_result = m.select_calibration_windows(shuffled)
    assert baseline.selected_windows_digest == shuffled_result.selected_windows_digest
    assert baseline.candidate_universe_digest == shuffled_result.candidate_universe_digest
    assert baseline.selection_trace_digest == shuffled_result.selection_trace_digest


def test_reversed_filesystem_style_order_independence():
    universe = _universe(image_count=50, windows_per_image=5)
    baseline = m.select_calibration_windows(universe)
    reversed_result = m.select_calibration_windows(list(reversed(universe)))
    assert baseline.selected_windows_digest == reversed_result.selected_windows_digest


def test_dictionary_insertion_order_independence():
    universe = _universe(image_count=50, windows_per_image=5)
    reordered_dicts = [{k: record[k] for k in reversed(list(record.keys()))} for record in universe]
    baseline = m.select_calibration_windows(universe)
    reordered_result = m.select_calibration_windows(reordered_dicts)
    assert baseline.selected_windows_digest == reordered_result.selected_windows_digest


def test_pythonhashseed_independence_across_subprocesses():
    script = (
        "import sys; sys.path.insert(0, %r); "
        "import src.dfp_phase1b_calibration_artifact as m; "
        "universe = [{'run_mode': 'coco_stuff_500', 'image_id': f'{i:012d}', 'image_path': f'val2017/{i:012d}.jpg', "
        "'image_manifest_ordinal': i, 'inference_canvas_height': 896, 'inference_canvas_width': 896, "
        "'canonical_window_ordinal': w, 'top': 0, 'left': 0, 'bottom': 448, 'right': 448, "
        "'crop_height': 448, 'crop_width': 448, 'patch_grid_height': 32, 'patch_grid_width': 32} "
        "for i in range(50) for w in range(5)]; "
        "result = m.select_calibration_windows(universe); "
        "print(result.selected_windows_digest)"
    ) % str(ROOT)
    digests = set()
    for seed in ("0", "1", "12345"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        )
        digests.add(completed.stdout.strip())
    assert len(digests) == 1, f"selection digest varied across PYTHONHASHSEED values: {digests}"


def test_selection_key_reconstructed_independently_in_test():
    """Independently compute the SHA-256 selection key -- never via the
    production helper as the oracle."""
    import hashlib
    import json

    record = _candidate(3, 2)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    expected_key = hashlib.sha256(m.WINDOW_SELECTION_NAMESPACE.encode("utf-8") + b"\x00" + encoded).hexdigest()
    assert m.compute_selection_key(m.WINDOW_SELECTION_NAMESPACE, m.encode_candidate_record(record)) == expected_key


def _claimed_rows_with_trace(result):
    trace_by_index = {t["canonical_candidate_index"]: t for t in result.selection_trace}
    return [
        {
            **dict(rec), "canonical_candidate_index": idx,
            "selection_key": trace_by_index[idx]["selection_key"],
            "selection_rank": trace_by_index[idx]["selection_rank"],
        }
        for idx, rec in zip(result.selected_canonical_indices, result.selected_records)
    ]


def test_verify_selection_against_universe_accepts_correct_selection():
    universe = _universe(image_count=50, windows_per_image=5)
    result = m.select_calibration_windows(universe)
    claimed = _claimed_rows_with_trace(result)
    m.verify_selection_against_universe(
        universe, claimed,
        claimed_candidate_universe_digest=result.candidate_universe_digest,
        claimed_selected_windows_digest=result.selected_windows_digest,
        claimed_selection_trace_digest=result.selection_trace_digest,
    )


def test_verify_selection_rejects_reordered_with_recomputed_digest():
    """A reordered-but-otherwise-correct selection, even if its own
    digest were recomputed to match the reordering, must still be
    rejected because rows must be in canonical candidate order."""
    universe = _universe(image_count=50, windows_per_image=5)
    result = m.select_calibration_windows(universe)
    claimed = [{**dict(rec), "canonical_candidate_index": idx} for idx, rec in zip(result.selected_canonical_indices, result.selected_records)]
    claimed[0], claimed[-1] = claimed[-1], claimed[0]
    with pytest.raises(E):
        m.verify_selection_against_universe(
            universe, claimed,
            claimed_candidate_universe_digest=result.candidate_universe_digest,
            claimed_selected_windows_digest=result.selected_windows_digest,
            claimed_selection_trace_digest=result.selection_trace_digest,
        )


def test_verify_selection_rejects_altered_geometry():
    universe = _universe(image_count=50, windows_per_image=5)
    result = m.select_calibration_windows(universe)
    claimed = [{**dict(rec), "canonical_candidate_index": idx} for idx, rec in zip(result.selected_canonical_indices, result.selected_records)]
    claimed[0] = dict(claimed[0])
    claimed[0]["top"] = claimed[0]["top"] + 1
    with pytest.raises(E):
        m.verify_selection_against_universe(
            universe, claimed,
            claimed_candidate_universe_digest=result.candidate_universe_digest,
            claimed_selected_windows_digest=result.selected_windows_digest,
            claimed_selection_trace_digest=result.selection_trace_digest,
        )


def test_verify_selection_rejects_wrong_subset_via_altered_universe_digest():
    universe = _universe(image_count=50, windows_per_image=5)
    result = m.select_calibration_windows(universe)
    claimed = _claimed_rows_with_trace(result)
    with pytest.raises(E):
        m.verify_selection_against_universe(
            universe, claimed,
            claimed_candidate_universe_digest="f" * 64,
            claimed_selected_windows_digest=result.selected_windows_digest,
            claimed_selection_trace_digest=result.selection_trace_digest,
        )


# ---------------------------------------------------------------------
# Residual summary / aggregate reconstruction.
# ---------------------------------------------------------------------


def test_residual_summary_zero_sample_requires_null_floats():
    m.validate_residual_summary(_residual_summary(sample_count=0), label="test")


def test_residual_summary_zero_sample_with_nonnull_float_rejected():
    summary = _residual_summary(sample_count=0)
    summary["mean_absolute_residual"] = 0.0
    with pytest.raises(E):
        m.validate_residual_summary(summary, label="test")


def test_residual_summary_nan_rejected():
    summary = _residual_summary()
    summary["signed_maximum"] = float("nan")
    with pytest.raises(E):
        m.validate_residual_summary(summary, label="test")


def test_residual_summary_infinity_rejected():
    summary = _residual_summary()
    summary["maximum_absolute_residual"] = float("inf")
    with pytest.raises(E):
        m.validate_residual_summary(summary, label="test")


def test_residual_summary_inconsistent_mean_rejected():
    summary = _residual_summary()
    summary["mean_absolute_residual"] += 1.0
    with pytest.raises(E):
        m.validate_residual_summary(summary, label="test")


def test_residual_summary_inconsistent_rms_rejected():
    summary = _residual_summary()
    summary["root_mean_square_residual"] += 1.0
    with pytest.raises(E):
        m.validate_residual_summary(summary, label="test")


def test_residual_summary_count_mismatch_rejected():
    summary = _residual_summary()
    summary["finite_count"] -= 1
    with pytest.raises(E):
        m.validate_residual_summary(summary, label="test")


def test_residual_summary_non_finite_count_nonzero_rejected():
    summary = _residual_summary()
    summary["sample_count"] += 1
    summary["non_finite_count"] = 1
    with pytest.raises(E):
        m.validate_residual_summary(summary, label="test")


def test_very_small_finite_residual_preserved_nonzero():
    tiny = 1e-300
    summary = _residual_summary(sample_count=1, value=tiny)
    m.validate_residual_summary(summary, label="test")
    assert summary["maximum_absolute_residual"] == tiny
    assert summary["maximum_absolute_residual"] != 0.0


def test_aggregate_reconstructable_from_sums():
    summaries = [_residual_summary(sample_count=n, value=v) for n, v in [(3, 1e-8), (5, 2e-8), (0, 0.0)]]
    aggregate = m.aggregate_residual_summaries(summaries, label="test")
    total_n = 3 + 5
    expected_abs_sum = 3 * 1e-8 + 5 * 2e-8
    expected_sq_sum = 3 * (1e-8 ** 2) + 5 * (2e-8 ** 2)
    assert aggregate["sample_count"] == total_n
    assert aggregate["absolute_residual_sum"] == expected_abs_sum
    assert aggregate["mean_absolute_residual"] == expected_abs_sum / total_n
    assert aggregate["root_mean_square_residual"] == math.sqrt(expected_sq_sum / total_n)


def test_aggregate_of_all_zero_sample_summaries_is_null():
    summaries = [_residual_summary(sample_count=0), _residual_summary(sample_count=0)]
    aggregate = m.aggregate_residual_summaries(summaries, label="test")
    assert aggregate["sample_count"] == 0
    assert aggregate["mean_absolute_residual"] is None


def test_perturbing_one_window_shifts_aggregate_by_analytically_expected_amount():
    base = [_residual_summary(sample_count=4, value=1e-6) for _ in range(3)]
    baseline_aggregate = m.aggregate_residual_summaries(base, label="test")

    perturbed = copy.deepcopy(base)
    perturbed[0] = _residual_summary(sample_count=4, value=1e-6 + 5e-7)
    perturbed_aggregate = m.aggregate_residual_summaries(perturbed, label="test")

    expected_abs_sum_delta = 4 * 5e-7
    actual_abs_sum_delta = perturbed_aggregate["absolute_residual_sum"] - baseline_aggregate["absolute_residual_sum"]
    assert math.isclose(actual_abs_sum_delta, expected_abs_sum_delta, rel_tol=1e-12)


# ---------------------------------------------------------------------
# Full artifact validation.
# ---------------------------------------------------------------------


def test_complete_200_window_artifact_validates():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    m.validate_calibration_artifact(doc, candidate_universe=universe)


def test_zero_progress_artifact_validates():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe, complete=False, final=False)
    doc["selected_window_records"] = []
    doc["completed_selected_window_count"] = 0
    doc["aggregate_residual_summaries"] = {
        family: m.aggregate_residual_summaries([], label=family) for family in m.RESIDUAL_FAMILIES
    }
    m.validate_calibration_artifact(doc)


def test_one_row_partial_artifact_validates():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, result = _artifact_document(universe)
    rows = _selected_rows(result)[:1]
    doc["selected_window_records"] = rows
    doc["completed_selected_window_count"] = 1
    doc["complete"] = False
    doc["final"] = False
    doc["aggregate_residual_summaries"] = {
        family: m.aggregate_residual_summaries([rows[0][family]], label=family) for family in m.RESIDUAL_FAMILIES
    }
    m.validate_calibration_artifact(doc)


def test_partial_progress_artifact_validates():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe, complete=False, final=False)
    m.validate_calibration_artifact(doc)


def test_missing_top_level_key_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    del doc["schema_version"]
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_unknown_top_level_key_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["conservation_tolerance"] = 1e-6
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_wrong_type_schema_version_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["schema_version"] = True
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_wrong_shape_selected_window_records_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["selected_window_records"] = {"not": "a list"}
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_nan_infinity_rejected_in_full_artifact():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["selected_window_records"][0]["rho_M"]["signed_minimum"] = float("nan")
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_corrupted_candidate_universe_digest_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["candidate_universe_digest"] = "f" * 64
    with pytest.raises(E):
        m.validate_calibration_artifact(doc, candidate_universe=universe)


def test_corrupted_selected_windows_digest_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["selected_windows_digest"] = "f" * 64
    with pytest.raises(E):
        m.validate_calibration_artifact(doc, candidate_universe=universe)


def test_corrupted_selection_trace_digest_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["selection_trace_digest"] = "f" * 64
    with pytest.raises(E):
        m.validate_calibration_artifact(doc, candidate_universe=universe)


def test_duplicate_rank_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["selected_window_records"][1]["selection_rank"] = doc["selected_window_records"][0]["selection_rank"]
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_duplicate_canonical_candidate_index_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["selected_window_records"][1]["canonical_candidate_index"] = doc["selected_window_records"][0]["canonical_candidate_index"]
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_duplicate_selection_key_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["selected_window_records"][1]["selection_key"] = doc["selected_window_records"][0]["selection_key"]
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_reordered_rows_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    rows = doc["selected_window_records"]
    rows[0], rows[-1] = rows[-1], rows[0]
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_incomplete_but_marked_complete_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["selected_window_records"][0]["completed"] = False
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_complete_final_inconsistency_row_count_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["selected_window_records"] = doc["selected_window_records"][:199]
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_added_tolerance_field_in_residual_summary_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["selected_window_records"][0]["rho_g"]["tolerance"] = 1e-6
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_aggregate_summary_inconsistent_with_rows_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["aggregate_residual_summaries"]["rho_g"]["mean_absolute_residual"] *= 2
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_malformed_device_record_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["device_record"]["device_type"] = "cpu"
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_wrong_type_alpha_via_direct_python_value_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["propagation_precision_record"]["alpha"] = 1  # int, not float
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_wrong_k_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["propagation_precision_record"]["k"] = 11
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_deterministic_serialization_of_two_independent_encodings():
    record_a = _candidate(3, 2)
    record_b = dict(reversed(list(_candidate(3, 2).items())))  # same content, different insertion order
    assert m.encode_candidate_record(record_a) == m.encode_candidate_record(record_b)


def test_source_immutability_selection_never_mutates_input_universe():
    universe = _universe(image_count=50, windows_per_image=5)
    before = copy.deepcopy(universe)
    m.select_calibration_windows(universe)
    assert universe == before


# ---------------------------------------------------------------------
# Exceptional bases.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_select_calibration_windows_propagates_fatal_exceptions(monkeypatch, exc_type):
    def raiser(*args, **kwargs):
        raise exc_type("must propagate")

    monkeypatch.setattr(m.hashlib, "sha256", raiser)
    with pytest.raises(exc_type):
        m.select_calibration_windows(_universe(image_count=40, windows_per_image=5))


# ---------------------------------------------------------------------
# Isolation.
# ---------------------------------------------------------------------


def test_module_never_imports_torch():
    import ast

    source = Path(m.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("torch")
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("torch")


def test_module_source_contains_no_gt_shaped_identifier():
    source = Path(m.__file__).read_text(encoding="utf-8").lower()
    for fragment in ("ground_truth", "annotation_load", "harmful_label", "beneficial_label", "oracle_map"):
        assert fragment not in source


def test_module_writes_no_files_outside_explicit_temp_destinations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m.select_calibration_windows(_universe(image_count=40, windows_per_image=5))
    m.validate_calibration_artifact(_artifact_document(_universe(image_count=40, windows_per_image=5))[0])
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------
# Repair regression: relational selection_key/selection_rank checks
# (independent-verification Finding 1). Expected keys/ranks below are
# computed with our own hashlib call against the frozen byte formula --
# never via compute_selection_key/select_calibration_windows as the
# test's own oracle for the EXPECTED value, only as the thing under
# test for the OBSERVED (production) behavior.
# ---------------------------------------------------------------------


def _independent_selection_key(namespace, record):
    """Independently computed selection key, built without calling
    m.compute_selection_key or m.encode_candidate_record."""
    import hashlib as _hashlib
    import json as _json

    encoded = _json.dumps(
        {field: record[field] for field in m.CANDIDATE_FIELDS},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return _hashlib.sha256(namespace.encode("utf-8") + b"\x00" + encoded).hexdigest()


def test_valid_200_row_artifact_still_accepted():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    m.validate_calibration_artifact(doc, candidate_universe=universe)


def test_wellformed_but_wrong_selection_key_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    original = doc["selected_window_records"][0]["selection_key"]
    tampered = ("0" if original[0] != "0" else "1") + original[1:]
    doc["selected_window_records"][0]["selection_key"] = tampered
    with pytest.raises(E):
        m.validate_calibration_artifact(doc, candidate_universe=universe)


def test_selection_rank_99999_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["selected_window_records"][0]["selection_rank"] = 99999
    with pytest.raises(E):
        m.validate_calibration_artifact(doc, candidate_universe=universe)


def test_swapped_valid_ranks_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    rows = doc["selected_window_records"]
    rows[0]["selection_rank"], rows[1]["selection_rank"] = rows[1]["selection_rank"], rows[0]["selection_rank"]
    with pytest.raises(E):
        m.validate_calibration_artifact(doc, candidate_universe=universe)


def test_duplicate_ranks_rejected_relationally():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["selected_window_records"][1]["selection_rank"] = doc["selected_window_records"][0]["selection_rank"]
    with pytest.raises(E):
        m.validate_calibration_artifact(doc, candidate_universe=universe)


def test_duplicate_keys_rejected_relationally():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["selected_window_records"][1]["selection_key"] = doc["selected_window_records"][0]["selection_key"]
    with pytest.raises(E):
        m.validate_calibration_artifact(doc, candidate_universe=universe)


def test_rank_assigned_to_wrong_candidate_rejected():
    """Both ranks are individually valid (0..199, unique), but each is
    attached to the WRONG candidate row -- must still be rejected."""
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    rows = doc["selected_window_records"]
    # Rotate ranks by one position across all rows -- every rank value
    # remains valid and unique, but none matches its own row's candidate.
    ranks = [row["selection_rank"] for row in rows]
    rotated = ranks[1:] + ranks[:1]
    for row, new_rank in zip(rows, rotated):
        row["selection_rank"] = new_rank
    with pytest.raises(E):
        m.validate_calibration_artifact(doc, candidate_universe=universe)


def test_key_copied_from_another_selected_candidate_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    rows = doc["selected_window_records"]
    rows[0]["selection_key"] = rows[1]["selection_key"]
    rows[1]["selection_key"] = rows[0]["selection_key"]  # already-mutated rows[0], so both now equal original rows[1]
    with pytest.raises(E):
        m.validate_calibration_artifact(doc, candidate_universe=universe)


def test_changed_key_with_recomputed_trace_digest_still_rejected():
    """The whole point of the repair: tampering with one row's key while
    recomputing the document's own selection_trace_digest to match the
    TAMPERED rows must still be rejected, because the digest is
    re-derived from the CANDIDATE UNIVERSE, not from the untrusted rows,
    and the per-row key is independently checked against that
    re-derivation too."""
    universe = _universe(image_count=50, windows_per_image=5)
    doc, result = _artifact_document(universe)
    rows = doc["selected_window_records"]
    tampered_key = ("0" if rows[0]["selection_key"][0] != "0" else "1") + rows[0]["selection_key"][1:]
    rows[0]["selection_key"] = tampered_key
    # Recompute a trace digest FROM THE TAMPERED ROWS (an attacker's best attempt).
    trace_records = [
        {"canonical_candidate_index": row["canonical_candidate_index"], "selection_key": row["selection_key"], "selection_rank": row["selection_rank"]}
        for row in sorted(rows, key=lambda r: r["selection_rank"])
    ]
    import json as _json
    encoded_traces = [
        _json.dumps(t, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") for t in trace_records
    ]
    doc["selection_trace_digest"] = m.digest_over_records(encoded_traces)
    with pytest.raises(E):
        m.validate_calibration_artifact(doc, candidate_universe=universe)


def test_changed_rank_with_recomputed_trace_digest_still_rejected():
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    rows = doc["selected_window_records"]
    rows[0]["selection_rank"], rows[1]["selection_rank"] = rows[1]["selection_rank"], rows[0]["selection_rank"]
    trace_records = [
        {"canonical_candidate_index": row["canonical_candidate_index"], "selection_key": row["selection_key"], "selection_rank": row["selection_rank"]}
        for row in sorted(rows, key=lambda r: r["selection_rank"])
    ]
    import json as _json
    encoded_traces = [
        _json.dumps(t, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") for t in trace_records
    ]
    doc["selection_trace_digest"] = m.digest_over_records(encoded_traces)
    with pytest.raises(E):
        m.validate_calibration_artifact(doc, candidate_universe=universe)


def test_changed_key_and_rank_with_all_global_digests_recomputed_still_rejected():
    """The strongest attacker: tamper with one row's key AND rank, then
    recompute candidate_universe_digest, selected_windows_digest, AND
    selection_trace_digest, all from the tampered document's own rows.
    Because the candidate-identity fields themselves are untouched, the
    global digests over candidate *identity* bytes remain valid -- only
    the per-row key/rank relational check (this repair) can catch it."""
    universe = _universe(image_count=50, windows_per_image=5)
    doc, result = _artifact_document(universe)
    rows = doc["selected_window_records"]
    tampered_key = ("0" if rows[0]["selection_key"][0] != "0" else "1") + rows[0]["selection_key"][1:]
    rows[0]["selection_key"] = tampered_key
    rows[0]["selection_rank"] = 99999
    # Global candidate-identity digests are untouched (candidate fields
    # themselves were never modified), so they remain individually
    # "valid" -- only recompute the trace digest, which is all an
    # attacker controlling just key/rank fields could hope to fix up.
    trace_records = [
        {"canonical_candidate_index": row["canonical_candidate_index"], "selection_key": row["selection_key"], "selection_rank": row["selection_rank"]}
        for row in sorted(rows, key=lambda r: r["selection_rank"])
    ]
    import json as _json
    encoded_traces = [
        _json.dumps(t, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") for t in trace_records
    ]
    doc["selection_trace_digest"] = m.digest_over_records(encoded_traces)
    with pytest.raises(E):
        m.validate_calibration_artifact(doc, candidate_universe=universe)


def test_reordered_rows_with_stale_ranks_rejected():
    """Reorder two rows (breaking canonical candidate order) while each
    row keeps carrying its OWN original rank -- must still be rejected,
    both for the ordering violation and (independently) because
    verify_selection_against_universe compares position-by-position."""
    universe = _universe(image_count=50, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    rows = doc["selected_window_records"]
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(E):
        m.validate_calibration_artifact(doc, candidate_universe=universe)


def test_artificial_selection_key_collision_still_tie_broken_by_canonical_index(monkeypatch):
    """Force two specific candidates to an identical selection key and
    confirm canonical_candidate_index still breaks the tie
    deterministically, exactly per the frozen ranking_key formula."""
    universe = _universe(image_count=50, windows_per_image=5)
    ordered = m.sort_candidate_universe(universe)
    forced_index_a, forced_index_b = 10, 20
    real_compute_selection_key = m.compute_selection_key

    def colliding(namespace, encoded_record):
        real_key = real_compute_selection_key(namespace, encoded_record)
        for forced_index in (forced_index_a, forced_index_b):
            if encoded_record == m.encode_candidate_record(ordered[forced_index]):
                return "0" * 64  # identical, artificially low key for both
        return real_key

    monkeypatch.setattr(m, "compute_selection_key", colliding)
    result = m.select_calibration_windows(universe)
    rank_a = next(t["selection_rank"] for t in result.selection_trace if t["canonical_candidate_index"] == forced_index_a)
    rank_b = next(t["selection_rank"] for t in result.selection_trace if t["canonical_candidate_index"] == forced_index_b)
    assert rank_a < rank_b  # forced_index_a (10) < forced_index_b (20): lower canonical index breaks the tie first


def test_verify_selection_against_universe_directly_rejects_tampered_key():
    universe = _universe(image_count=50, windows_per_image=5)
    result = m.select_calibration_windows(universe)
    claimed = []
    for index, record in zip(result.selected_canonical_indices, result.selected_records):
        trace = next(t for t in result.selection_trace if t["canonical_candidate_index"] == index)
        claimed.append({
            **dict(record), "canonical_candidate_index": index,
            "selection_key": trace["selection_key"], "selection_rank": trace["selection_rank"],
        })
    claimed[0]["selection_key"] = ("0" if claimed[0]["selection_key"][0] != "0" else "1") + claimed[0]["selection_key"][1:]
    with pytest.raises(E):
        m.verify_selection_against_universe(
            universe, claimed,
            claimed_candidate_universe_digest=result.candidate_universe_digest,
            claimed_selected_windows_digest=result.selected_windows_digest,
            claimed_selection_trace_digest=result.selection_trace_digest,
        )


def test_independent_key_computation_matches_production_before_tampering():
    """Sanity check that our OWN independent key formula (not calling
    compute_selection_key) agrees with production, so the tampering
    tests above are genuinely mutating a correct baseline."""
    universe = _universe(image_count=50, windows_per_image=5)
    result = m.select_calibration_windows(universe)
    for index, record in zip(result.selected_canonical_indices, result.selected_records):
        trace = next(t for t in result.selection_trace if t["canonical_candidate_index"] == index)
        assert _independent_selection_key(m.WINDOW_SELECTION_NAMESPACE, record) == trace["selection_key"]


# ---------------------------------------------------------------------
# Repair regression: exact alpha/dtype pinning (Finding 2).
# ---------------------------------------------------------------------


def test_correct_propagation_precision_record_accepted():
    universe = _universe(image_count=40, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    m.validate_calibration_artifact(doc)


@pytest.mark.parametrize("bad_alpha", [0.99, 0.0, -0.98, 1.0])
def test_wrong_alpha_value_rejected(bad_alpha):
    universe = _universe(image_count=40, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["propagation_precision_record"]["alpha"] = bad_alpha
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


@pytest.mark.parametrize("bad_alpha", [True, False, 1, 0, "0.98", float("nan"), float("inf"), float("-inf")])
def test_wrong_alpha_type_or_nonfinite_rejected(bad_alpha):
    universe = _universe(image_count=40, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["propagation_precision_record"]["alpha"] = bad_alpha
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


@pytest.mark.parametrize("field", ["graph_storage_dtype", "canonical_recurrence_dtype", "masked_recurrence_dtype", "residual_accumulation_dtype"])
@pytest.mark.parametrize("bad_value", ["float64", "float32", "fp32", "fp64", "FLOAT32", " float32", "float32 ", "bfloat16", ""])
def test_wrong_dtype_value_rejected(field, bad_value):
    universe = _universe(image_count=40, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    correct = doc["propagation_precision_record"][field]
    if bad_value == correct:
        pytest.skip("mutation coincides with the correct value for this field")
    doc["propagation_precision_record"][field] = bad_value
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_expected_alpha_constant_matches_e13_frozen_value():
    assert m.EXPECTED_ALPHA == 0.98


def test_expected_dtype_constants_match_e16_execution_contract():
    assert m._PROPAGATION_PRECISION_EXACT_DTYPES == {
        "graph_storage_dtype": "float32",
        "canonical_recurrence_dtype": "float64",
        "masked_recurrence_dtype": "float64",
        "residual_accumulation_dtype": "float64",
    }


# ---------------------------------------------------------------------
# Repair regression: candidate image_id/image_path hardening (Finding 3).
# ---------------------------------------------------------------------


@pytest.mark.parametrize("field", ["image_id", "image_path"])
@pytest.mark.parametrize("bad_value", ["", "   ", "\t", " leading", "trailing ", "embedded\x00null"])
def test_candidate_string_field_rejects_noncanonical_value(field, bad_value):
    record = _candidate(0, 0)
    record[field] = bad_value
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_candidate_image_id_empty_rejected():
    record = _candidate(0, 0)
    record["image_id"] = ""
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_candidate_image_id_whitespace_only_rejected():
    record = _candidate(0, 0)
    record["image_id"] = "   "
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_candidate_image_id_nul_rejected():
    record = _candidate(0, 0)
    record["image_id"] = "abc\x00def"
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_candidate_image_id_padded_rejected_not_normalized():
    record = _candidate(0, 0)
    record["image_id"] = " 000000000000 "
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_candidate_image_path_empty_rejected():
    record = _candidate(0, 0)
    record["image_path"] = ""
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_candidate_image_path_whitespace_only_rejected():
    record = _candidate(0, 0)
    record["image_path"] = "   "
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_candidate_image_path_nul_rejected():
    record = _candidate(0, 0)
    record["image_path"] = "val2017/abc\x00.jpg"
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_candidate_image_path_empty_segment_rejected():
    record = _candidate(0, 0)
    record["image_path"] = "val2017//000000000000.jpg"
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_candidate_image_path_trailing_slash_rejected():
    record = _candidate(0, 0)
    record["image_path"] = "val2017/000000000000.jpg/"
    with pytest.raises(E):
        m.validate_candidate_record(record)


def test_candidate_string_hardening_reached_through_encoding():
    record = _candidate(0, 0)
    record["image_id"] = ""
    with pytest.raises(E):
        m.encode_candidate_record(record)


def test_candidate_string_hardening_reached_through_selection():
    universe = _universe(image_count=40, windows_per_image=5)
    universe[0]["image_id"] = ""
    with pytest.raises(E):
        m.select_calibration_windows(universe)


def test_candidate_string_hardening_reached_through_artifact_validation():
    universe = _universe(image_count=40, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    doc["selected_window_records"][0]["image_id"] = "   "
    with pytest.raises(E):
        m.validate_calibration_artifact(doc)


def test_candidate_windows_and_traversal_checks_still_reject():
    """Regression: the hardening must not have weakened the pre-existing
    path-safety checks."""
    for bad_path in ("/abs/path.jpg", "..\\..\\etc", "a/../b.jpg", "a\\b.jpg"):
        record = _candidate(0, 0)
        record["image_path"] = bad_path
        with pytest.raises(E):
            m.validate_candidate_record(record)


# ---------------------------------------------------------------------
# Repair regression: read-only Mapping / MappingProxyType (Finding 4).
# ---------------------------------------------------------------------


def test_mappingproxytype_candidate_encodes_byte_identically_to_plain_dict():
    import types

    record = _candidate(3, 2)
    proxy = types.MappingProxyType(dict(record))
    assert m.encode_candidate_record(proxy) == m.encode_candidate_record(record)


def test_mappingproxytype_input_left_unchanged():
    import types

    record = _candidate(3, 2)
    proxy = types.MappingProxyType(dict(record))
    before = dict(proxy)
    m.encode_candidate_record(proxy)
    assert dict(proxy) == before


def test_mappingproxytype_dict_insertion_order_does_not_affect_bytes():
    import types

    record = _candidate(3, 2)
    reordered = {k: record[k] for k in reversed(list(record.keys()))}
    proxy_a = types.MappingProxyType(dict(record))
    proxy_b = types.MappingProxyType(reordered)
    assert m.encode_candidate_record(proxy_a) == m.encode_candidate_record(proxy_b)


def test_malformed_mappingproxytype_raises_domain_exception_not_typeerror():
    import types

    record = _candidate(3, 2)
    del record["image_id"]  # malformed: missing a required key
    proxy = types.MappingProxyType(record)
    with pytest.raises(E):
        m.encode_candidate_record(proxy)


def test_no_raw_typeerror_escapes_encode_candidate_record_for_valid_proxy():
    import types

    record = _candidate(3, 2)
    proxy = types.MappingProxyType(dict(record))
    try:
        m.encode_candidate_record(proxy)
    except TypeError:
        pytest.fail("encode_candidate_record leaked a raw TypeError for a valid MappingProxyType")
    except E:
        pytest.fail("a VALID MappingProxyType must not be rejected")


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_encode_candidate_record_propagates_fatal_exceptions(monkeypatch, exc_type):
    def raiser(*args, **kwargs):
        raise exc_type("must propagate")

    monkeypatch.setattr(m.json, "dumps", raiser)
    record = _candidate(3, 2)
    with pytest.raises(exc_type):
        m.encode_candidate_record(record)


# ---------------------------------------------------------------------
# Repair regression: unchanged scientific semantics (Finding 5 and
# general non-regression).
# ---------------------------------------------------------------------


def test_selection_semantics_and_digest_algorithms_unchanged():
    """Same universe, independently reconstructed selection/digests,
    still match production exactly after the repair -- proves the
    repair did not alter the frozen selection algorithm."""
    universe = _universe(image_count=50, windows_per_image=5)
    result = m.select_calibration_windows(universe)
    ordered = m.sort_candidate_universe(universe)
    encoded = [m.encode_candidate_record(r) for r in ordered]
    keys = [_independent_selection_key(m.WINDOW_SELECTION_NAMESPACE, r) for r in ordered]
    ranked = sorted(range(len(ordered)), key=lambda i: (keys[i], i))
    selected = tuple(sorted(ranked[: m.WINDOW_COUNT]))
    assert selected == result.selected_canonical_indices
    assert m.digest_over_records(encoded) == result.candidate_universe_digest


def test_artifact_schema_field_sets_unchanged():
    assert len(m.TOP_LEVEL_ARTIFACT_FIELDS) == 23
    assert len(m.SELECTED_WINDOW_RECORD_FIELDS) == 23
    assert len(m.CANDIDATE_FIELDS) == 15
    assert len(m.RESIDUAL_SUMMARY_FIELDS) == 10


def test_no_tolerance_field_introduced_by_repair():
    import ast

    source = Path(m.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstring_ids.add(id(first.value))
    forbidden_fragments = ("tolerance", "epsilon", "atol", "rtol", "safety_factor", "percentile_cutoff")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            lowered = node.id.lower()
            assert not any(fragment in lowered for fragment in forbidden_fragments), node.id


def test_no_pass_fail_verdict_field_introduced_by_repair():
    universe = _universe(image_count=40, windows_per_image=5)
    doc, _ = _artifact_document(universe)
    for forbidden in ("tolerance", "verdict", "passed", "acceptable"):
        bad = copy.deepcopy(doc)
        bad[forbidden] = "x"
        with pytest.raises(E):
            m.validate_calibration_artifact(bad)
