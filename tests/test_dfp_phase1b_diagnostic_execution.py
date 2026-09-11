"""Adversarial tests for the DFP Phase 1B diagnostic-execution
successor identity (E20) loader/validators + CPU-only CLI.

CPU/data-read-only. Never imports torch, never initializes CUDA, never
loads a model/dataset/checkpoint-tensor, never executes recurrence/
graph/masked propagation, never submits SLURM. Every mutation test
writes its own temporary copy -- the committed TOML is never edited by
any test."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import src.dfp_phase1b_diagnostic_execution as m  # noqa: E402

IDENTITY_PATH = ROOT / m.IDENTITY_RELATIVE_PATH
CLI_PATH = ROOT / "diagnostics/run_dfp_phase1b_diagnostic_execution.py"
CALIBRATION_ARTIFACT_PATH = Path(
    "/scratch/haree/dfp_phase1b_conservation_calibration/coco_stuff_200/phase1b_conservation_calibration.json"
)
TASK4_ARTIFACT_PATH = Path("/scratch/haree/dfp_phase1b_chunk_benchmark/21331697/phase1b_chunk_selection.json")
TASK4_REPORT_PATH = Path("/scratch/haree/dfp_phase1b_chunk_benchmark/21331697/phase1b_chunk_selection_report.md")
_EXTERNAL_AVAILABLE = CALIBRATION_ARTIFACT_PATH.exists() and TASK4_ARTIFACT_PATH.exists() and TASK4_REPORT_PATH.exists()
requires_external = pytest.mark.skipif(not _EXTERNAL_AVAILABLE, reason="scratch artifacts not present on this host")

E = m.DfpPhase1bDiagnosticExecutionError


def _identity_text() -> str:
    return IDENTITY_PATH.read_text()


def _write(tmp_path: Path, text: str, name: str = "e20.toml") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def _mutate_line(text: str, old_line: str, new_line: str) -> str:
    assert old_line in text, f"anchor not found: {old_line!r}"
    return text.replace(old_line, new_line, 1)


# =====================================================================
# 1. Baseline / authority and bindings
# =====================================================================

def test_committed_identity_loads_clean():
    identity = m.load_identity()
    assert identity["identity"]["name"] == "e20-dfp-phase1b-diagnostic-execution"
    assert identity["population"]["image_count"] == 500
    assert identity["population"]["window_count"] == 1090
    assert identity["population"]["component_chunk_size"] == 16
    assert identity["population"]["gpu_count"] == 1
    assert identity["population"]["gpu_model"] == "NVIDIA H100 80GB HBM3"
    assert identity["population"]["maximum_job_wall_time_seconds"] == 28800
    assert identity["scientific_execution_binding"]["conservation_tolerance"] == 9.5367431640625e-07
    assert identity["scientific_execution_binding"]["conservation_tolerance"].hex() == "0x1.0000000000000p-20"


def test_e19_tolerance_binding_matches_live_e19():
    identity = m.load_identity()
    import src.dfp_phase1b_conservation_tolerance as t19
    live_e19_hash = t19.load_identity()
    import hashlib
    live_hash = hashlib.sha256((ROOT / "evaluation_identities/e19_dfp_phase1b_conservation_tolerance.toml").read_bytes()).hexdigest()
    assert identity["dependency_bindings"]["e19_tolerance_identity"]["sha256"] == live_hash
    assert identity["scientific_execution_binding"]["conservation_tolerance"] == live_e19_hash["tolerance"]["conservation_tolerance"]


def test_e18_chunk_size_binding_matches_live_e18():
    identity = m.load_identity()
    import hashlib
    live_hash = hashlib.sha256((ROOT / "evaluation_identities/e18_dfp_phase1b_chunk_selection.toml").read_bytes()).hexdigest()
    assert identity["dependency_bindings"]["e18_chunk_selection_identity"]["sha256"] == live_hash
    assert identity["population"]["component_chunk_size"] == 16
    assert identity["scientific_execution_binding"]["component_chunk_size"] == 16


@requires_external
def test_external_artifact_bindings_verify():
    r = m.verify_external_artifact_bindings(
        calibration_artifact_path=CALIBRATION_ARTIFACT_PATH,
        task4_selection_artifact_path=TASK4_ARTIFACT_PATH,
        task4_selection_report_path=TASK4_REPORT_PATH,
    )
    assert r["calibration_artifact"] == "f065b143933cf7881e5a31dd663783719a684e6d430e9e7b2bb314ed7e792d22"
    assert r["task4_selection_artifact"] == "dacf48460f335c603e631c4e15e875ba439d4336a3fe9a57afca9809cf20e6b8"


@requires_external
def test_window_universe_reconstruction_from_calibration_artifact():
    identity = m.load_identity()
    doc = json.loads(CALIBRATION_ARTIFACT_PATH.read_bytes())
    assert doc["candidate_window_count"] == 1090
    assert doc["candidate_universe_digest"] == identity["population"]["window_universe_binding"]["candidate_universe_digest"]
    assert identity["population"]["window_count"] == doc["candidate_window_count"]


def test_h100_one_gpu_eight_hour_contract():
    identity = m.load_identity()
    pop = identity["population"]
    assert pop["gpu_count"] == 1
    assert pop["gpu_model"] == "NVIDIA H100 80GB HBM3"
    assert pop["maximum_job_wall_time_seconds"] == 28800
    assert pop["sharding_across_jobs_or_gpus_forbidden"] is True
    assert identity["authorization_boundary"]["sharding_across_jobs_or_gpus_authorized"] is False


def test_authorization_boundary_all_false_where_required():
    identity = m.load_identity()
    ab = identity["authorization_boundary"]
    for key in (
        "execution_driver_implemented_by_task6b", "profiler_implemented_by_task6b",
        "checkpoint_writer_implemented_by_task6b", "result_writer_implemented_by_task6b",
        "cli_execution_command_implemented_by_task6b", "slurm_launcher_implemented_by_task6b",
        "real_inference_executed_by_task6b", "gpu_execution_performed_by_task6b",
        "phase1b_500_image_execution_authorized", "phase1b_5000_image_exact_provenance_authorized",
        "full_5000_image_miou_evaluation_authorized", "lightweight_final_method_selected",
        "sharding_across_jobs_or_gpus_authorized",
    ):
        assert ab[key] is False, key
    assert ab["authority_only_task"] is True


# =====================================================================
# 2. Closed-schema mutation matrix + path safety (temp copies only)
# =====================================================================

@pytest.mark.parametrize(
    "old,new",
    [
        ("window_count = 1090", "window_count = 1089"),
        ("image_count = 500\nwindow_count", "image_count = 501\nwindow_count"),
        ("component_chunk_size = 16", "component_chunk_size = 32"),
        ('gpu_model = "NVIDIA H100 80GB HBM3"', 'gpu_model = "NVIDIA A100 80GB"'),
        ("gpu_count = 1", "gpu_count = 2"),
        ("maximum_job_wall_time_seconds = 28800", "maximum_job_wall_time_seconds = 36000"),
        ("conservation_tolerance = 9.5367431640625e-07", "conservation_tolerance = 1.0e-06"),
        ('comparison_operator = "<="', 'comparison_operator = "<"'),
        ("sharding_across_jobs_or_gpus_forbidden = true", "sharding_across_jobs_or_gpus_forbidden = false"),
        ("sharding_across_jobs_or_gpus_authorized = false", "sharding_across_jobs_or_gpus_authorized = true"),
        ("phase1b_500_image_execution_authorized = false", "phase1b_500_image_execution_authorized = true"),
        ("phase1b_5000_image_exact_provenance_authorized = false", "phase1b_5000_image_exact_provenance_authorized = true"),
        ("full_5000_image_miou_evaluation_authorized = false", "full_5000_image_miou_evaluation_authorized = true"),
        ("lightweight_final_method_selected = false", "lightweight_final_method_selected = true"),
        ("execution_driver_implemented_by_task6b = false", "execution_driver_implemented_by_task6b = true"),
        ('stage_count = 18', 'stage_count = 17'),
        ("percentile_convention = \"nearest_rank\"", "percentile_convention = \"linear_interpolation\""),
    ],
)
def test_e20_relational_mutation_matrix_rejected(tmp_path, old, new):
    text = _mutate_line(_identity_text(), old, new)
    p = _write(tmp_path, text)
    with pytest.raises(E):
        m.load_identity(path=p, repo_root=ROOT)


def test_e20_missing_field_rejected(tmp_path):
    text = _mutate_line(_identity_text(), "stage_count = 18\n", "")
    p = _write(tmp_path, text)
    with pytest.raises(E):
        m.load_identity(path=p, repo_root=ROOT)


def test_e20_extra_field_rejected(tmp_path):
    text = _mutate_line(_identity_text(), "stage_count = 18", "stage_count = 18\nsurprise_field = 1")
    p = _write(tmp_path, text)
    with pytest.raises(E):
        m.load_identity(path=p, repo_root=ROOT)


@pytest.mark.parametrize(
    "old,new",
    [
        ('relative_path = "evaluation_identities/e19_dfp_phase1b_conservation_tolerance.toml"',
         'relative_path = "../evaluation_identities/e19_dfp_phase1b_conservation_tolerance.toml"'),
        ('relative_path = "evaluation_identities/e19_dfp_phase1b_conservation_tolerance.toml"',
         'relative_path = "/etc/passwd"'),
        ('relative_path = "src/dfp_phase1b_masked_forward.py"',
         'relative_path = "evaluation_identities/e15_dfp_phase1b_protocol.toml"'),
    ],
)
def test_e20_path_traversal_and_substitution_rejected(tmp_path, old, new):
    text = _mutate_line(_identity_text(), old, new)
    p = _write(tmp_path, text)
    with pytest.raises(E):
        m.load_identity(path=p, repo_root=ROOT)


def test_e20_wrong_dependency_hash_rejected(tmp_path):
    text = _mutate_line(
        _identity_text(),
        '[dependency_bindings.masked_forward_core]\nrelative_path = "src/dfp_phase1b_masked_forward.py"\nsha256 = "72f9b7093379cdbde14691e0dcc97c54eb2563d8bc8690da027409d6c81b56b3"',
        '[dependency_bindings.masked_forward_core]\nrelative_path = "src/dfp_phase1b_masked_forward.py"\nsha256 = "0000000000000000000000000000000000000000000000000000000000000000"',
    )
    p = _write(tmp_path, text)
    with pytest.raises(E):
        m.load_identity(path=p, repo_root=ROOT)


def test_not_toml_wrapped_in_domain_exception(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("{{{ not toml")
    with pytest.raises(E):
        m.load_identity(path=p, repo_root=ROOT)


# =====================================================================
# 3. Checkpoint document tests
# =====================================================================

def _base_checkpoint(*, complete=False, final=False, completed_ids=(), completed_windows=0, final_produced=False, generation=0):
    return {
        "schema_name": m.CHECKPOINT_SCHEMA_NAME,
        "schema_version": m.CHECKPOINT_SCHEMA_VERSION,
        "identity_bindings": {
            "e20_identity_name": "e20-dfp-phase1b-diagnostic-execution",
            "e20_identity_schema_version": 1,
            "e20_identity_sha256": "a" * 64,
            "e15_protocol_sha256": "b" * 64,
            "e15_artifacts_sha256": "c" * 64,
            "e18_chunk_selection_sha256": "d" * 64,
            "e19_tolerance_sha256": "e" * 64,
            "subset_manifest_sha256": "f" * 64,
            "window_universe_digest": "0" * 64,
            "scientific_schema_name": "talk2dino-dfp-phase1b-stage-a-features-v1",
            "scientific_schema_version": 1,
            "run_mode": "coco_stuff_500",
            "device_type": "cuda",
            "gpu_model": "NVIDIA H100 80GB HBM3",
            "gpu_count": 1,
            "component_chunk_size": 16,
            "conservation_tolerance": 9.5367431640625e-07,
        },
        "progress": {
            "expected_image_count": 500,
            "expected_window_count": 1090,
            "next_canonical_image_index": len(completed_ids),
            "completed_image_ids": list(completed_ids),
            "completed_window_count": completed_windows,
            "total_processed_component_chunks": 0,
            "generation": generation,
            "complete": complete,
            "final_result_produced": final_produced,
        },
        "scientific_state": {
            "stage_a_artifact_filename": "phase1b_diagnostic_stage_a.npz",
            "stage_a_schema_name": "talk2dino-dfp-phase1b-stage-a-features-v1",
            "stage_a_schema_version": 1,
            "stage_a_row_count": 0,
            "stage_a_byte_sha256": "1" * 64,
            "scientific_collection_digest": "2" * 64,
        },
        "profiling_state": {
            "profiling_mode": "off",
            "profiling_record_filename": "phase1b_diagnostic_profiling.json",
            "profiling_schema_name": m.PROFILING_SCHEMA_NAME,
            "profiling_schema_version": 1,
            "profiling_record_count": 0,
            "profiling_digest": "3" * 64,
        },
        "complete": complete,
        "final": final,
    }


def test_zero_progress_checkpoint_valid():
    doc = _base_checkpoint()
    m.validate_checkpoint_document(doc)


def test_partial_progress_checkpoint_valid():
    doc = _base_checkpoint(completed_ids=["im_0001", "im_0002"], completed_windows=5)
    m.validate_checkpoint_document(doc)


def test_complete_unfinalized_checkpoint_valid():
    doc = _base_checkpoint(complete=True, completed_ids=[f"im_{i:04d}" for i in range(500)], completed_windows=1090)
    m.validate_checkpoint_document(doc)
    m.validate_resume_eligibility(doc)  # complete-but-unfinalized may enter finalize-only path


def test_finalized_checkpoint_rejected_for_resume():
    doc = _base_checkpoint(complete=True, final=True, final_produced=True,
                            completed_ids=[f"im_{i:04d}" for i in range(500)], completed_windows=1090)
    m.validate_checkpoint_document(doc)
    with pytest.raises(E):
        m.validate_resume_eligibility(doc)


def test_checkpoint_wrong_prefix_rejected():
    doc = _base_checkpoint(completed_ids=["im_0002", "im_0001"], completed_windows=0)
    m.validate_checkpoint_document(doc)
    with pytest.raises(E):
        m.validate_resume_prefix(doc, ["im_0001", "im_0002", "im_0003"])


def test_checkpoint_duplicate_ids_rejected():
    doc = _base_checkpoint(completed_ids=["im_0001", "im_0001"], completed_windows=0)
    with pytest.raises(E):
        m.validate_checkpoint_document(doc)


def test_checkpoint_reordered_or_skipped_prefix_rejected():
    doc = _base_checkpoint(completed_ids=["im_0001", "im_0003"], completed_windows=0)
    m.validate_checkpoint_document(doc)
    with pytest.raises(E):
        m.validate_resume_prefix(doc, ["im_0001", "im_0002", "im_0003"])


def test_checkpoint_window_count_disagreement_rejected():
    doc = _base_checkpoint(complete=True, completed_ids=[f"im_{i:04d}" for i in range(500)], completed_windows=1089)
    with pytest.raises(E):
        m.validate_checkpoint_document(doc)


def test_checkpoint_corrupt_artifact_hash_rejected():
    doc = _base_checkpoint()
    doc["scientific_state"]["stage_a_byte_sha256"] = "not-a-hash"
    with pytest.raises(E):
        m.validate_checkpoint_document(doc)


def test_checkpoint_unknown_schema_version_rejected():
    doc = _base_checkpoint()
    doc["schema_version"] = 2
    with pytest.raises(E):
        m.validate_checkpoint_document(doc)


def test_checkpoint_cross_version_feed_rejected():
    doc = _base_checkpoint()
    doc["schema_name"] = "talk2dino-dfp-phase1b-diagnostic-checkpoint-v0"
    with pytest.raises(E):
        m.validate_checkpoint_document(doc)


def test_checkpoint_partial_generation_rejected():
    doc = _base_checkpoint(complete=False, completed_ids=[f"im_{i:04d}" for i in range(500)], completed_windows=1090)
    with pytest.raises(E):
        m.validate_checkpoint_document(doc)


def test_checkpoint_final_requires_complete_rejected():
    doc = _base_checkpoint(complete=False, final=True)
    with pytest.raises(E):
        m.validate_checkpoint_document(doc)


def test_fresh_start_requires_no_existing_checkpoint():
    m.validate_fresh_start(False)
    with pytest.raises(E):
        m.validate_fresh_start(True)
    m.validate_fresh_start(True, overwrite_authorized=True)


# =====================================================================
# 4. Profiling document tests
# =====================================================================

def _base_profiling_record(stage="graph_construction", scope="window", key="img0/win0", *, cpu_ns=1000, incomplete=False):
    return {
        "stage_name": stage, "scope": scope, "scope_covariate_key": key, "incomplete": incomplete,
        "cpu_elapsed_ns": None if incomplete else cpu_ns, "cuda_elapsed_ms": None,
        "host_rss_bytes_before": 1000, "host_rss_bytes_after": 1100,
        "process_peak_rss_bytes_after_stage": 1200,
        "gpu_allocated_bytes_before": None, "gpu_reserved_bytes_before": None,
        "gpu_allocated_bytes_after": None, "gpu_reserved_bytes_after": None,
        "gpu_peak_allocated_bytes_during_stage": None, "gpu_peak_reserved_bytes_during_stage": None,
        "gpu_device_index": None, "gpu_device_model": None,
    }


def _base_profiling_doc(records=()):
    return {
        "schema_name": m.PROFILING_SCHEMA_NAME, "schema_version": m.PROFILING_SCHEMA_VERSION,
        "e20_identity_sha256": "a" * 64, "profiling_mode": "off", "complete": True, "final": True,
        "run_identity": {}, "device_record": {}, "software_record": {}, "timing_method": {},
        "host_memory_method": {}, "gpu_memory_method": {},
        "stage_name_enumeration": list(m.STAGE_NAME_ENUMERATION),
        "records": list(records), "aggregates": {}, "workload_covariates": {},
        "total_run_wall_time_ns": 5000, "total_scientific_processing_wall_time_ns": 3000,
        "checkpoint_io_wall_time_ns": 1000, "unattributed_overhead_ns": 1000,
        "runtime_budget": {
            "maximum_job_wall_time_seconds": 28800.0, "elapsed_job_wall_time_seconds": 10.0,
            "remaining_budget_seconds": 28790.0,
        },
        "warnings": [], "source_scientific_collection_digest": "2" * 64,
    }


@pytest.mark.parametrize("stage", list(m.STAGE_NAME_ENUMERATION))
def test_every_stage_name_and_scope_valid(stage):
    scope = m.STAGE_SCOPES[stage]
    rec = _base_profiling_record(stage=stage, scope=scope)
    m.validate_profiling_record(rec)


def test_stage_scope_mismatch_rejected():
    rec = _base_profiling_record(stage="graph_construction", scope="run")
    with pytest.raises(E):
        m.validate_profiling_record(rec)


def test_unique_record_ids_enforced():
    r1 = _base_profiling_record()
    r2 = _base_profiling_record()  # identical identity
    doc = _base_profiling_doc([r1, r2])
    with pytest.raises(E):
        m.validate_profiling_document(doc)


def test_distinct_record_ids_accepted():
    r1 = _base_profiling_record(key="img0/win0")
    r2 = _base_profiling_record(key="img0/win1")
    doc = _base_profiling_doc([r1, r2])
    m.validate_profiling_document(doc)


def test_cpu_timing_nonnegative_int():
    rec = _base_profiling_record()
    rec["cpu_elapsed_ns"] = -1
    with pytest.raises(E):
        m.validate_profiling_record(rec)


def test_cuda_timing_unavailable_semantics():
    rec = _base_profiling_record()
    rec["cuda_elapsed_ms"] = None  # CPU-only stage -> unavailable, not 0.0
    m.validate_profiling_record(rec)
    rec["cuda_elapsed_ms"] = 0.0
    m.validate_profiling_record(rec)  # 0.0 is a legitimate measured value, also accepted
    rec["cuda_elapsed_ms"] = -1.0
    with pytest.raises(E):
        m.validate_profiling_record(rec)


def test_host_memory_field_semantics():
    rec = _base_profiling_record()
    rec["host_rss_bytes_before"] = None  # unavailable is allowed
    m.validate_profiling_record(rec)
    rec["host_rss_bytes_before"] = -5
    with pytest.raises(E):
        m.validate_profiling_record(rec)


def test_gpu_allocated_reserved_peak_relations():
    rec = _base_profiling_record()
    rec.update(gpu_allocated_bytes_before=100, gpu_reserved_bytes_before=200,
               gpu_allocated_bytes_after=150, gpu_reserved_bytes_after=250,
               gpu_peak_allocated_bytes_during_stage=180, gpu_peak_reserved_bytes_during_stage=260,
               gpu_device_index=0, gpu_device_model="NVIDIA H100 80GB HBM3")
    m.validate_profiling_record(rec)
    bad = dict(rec); bad["gpu_allocated_bytes_before"] = 300  # > reserved_before
    with pytest.raises(E):
        m.validate_profiling_record(bad)
    bad2 = dict(rec); bad2["gpu_peak_allocated_bytes_during_stage"] = 10  # < after
    with pytest.raises(E):
        m.validate_profiling_record(bad2)


def test_nearest_rank_percentile_reconstruction():
    vals = [float(x) for x in range(1, 11)]  # 1..10
    assert m.nearest_rank_percentile(vals, 0.5) == 5.0
    assert m.nearest_rank_percentile(vals, 0.90) == 9.0
    assert m.nearest_rank_percentile(vals, 0.95) == 10.0
    assert m.nearest_rank_percentile(vals, 0.99) == 10.0
    assert m.nearest_rank_percentile([42.0], 0.5) == 42.0


def test_aggregate_zero_one_many_records():
    zero = m.aggregate_numeric_records([])
    assert zero["count"] == 0
    for k in ("sum", "mean", "min", "max", "median", "p90", "p95", "p99"):
        assert zero[k] is None
    one = m.aggregate_numeric_records([7.0])
    assert one == {"count": 1, "sum": 7.0, "mean": 7.0, "min": 7.0, "max": 7.0, "median": 7.0, "p90": 7.0, "p95": 7.0, "p99": 7.0}
    many = m.aggregate_numeric_records([1.0, 2.0, 3.0])
    assert many["count"] == 3 and many["sum"] == 6.0 and many["mean"] == 2.0


def test_aggregate_rejects_nonfinite():
    with pytest.raises(E):
        m.aggregate_numeric_records([1.0, float("nan")])
    with pytest.raises(E):
        m.aggregate_numeric_records([float("inf")])


def test_overhead_relation_negative_rejected():
    doc = _base_profiling_doc()
    doc["unattributed_overhead_ns"] = -1
    doc["total_run_wall_time_ns"] = doc["total_scientific_processing_wall_time_ns"] + doc["checkpoint_io_wall_time_ns"] + doc["unattributed_overhead_ns"]
    with pytest.raises(E):
        m.validate_profiling_document(doc)


def test_overhead_relation_must_sum_exactly():
    doc = _base_profiling_doc()
    doc["total_run_wall_time_ns"] = 9999  # disagrees with component sum
    with pytest.raises(E):
        m.validate_profiling_document(doc)


def test_mixed_scope_records_not_averaged_together():
    r_window = _base_profiling_record(stage="graph_construction", scope="window", key="a", cpu_ns=100)
    r_chunk = _base_profiling_record(stage="masked_k12_recurrence", scope="component_chunk", key="b", cpu_ns=200)
    aggs = m.reconstruct_aggregates_from_records([r_window, r_chunk])
    assert ("graph_construction", "window") in aggs
    assert ("masked_k12_recurrence", "component_chunk") in aggs
    assert aggs[("graph_construction", "window")]["count"] == 1


def test_incomplete_record_excluded_from_aggregate_reconstruction():
    completed = _base_profiling_record(cpu_ns=500)
    interrupted = _base_profiling_record(key="other", incomplete=True)
    aggs = m.reconstruct_aggregates_from_records([completed, interrupted])
    key = (completed["stage_name"], completed["scope"])
    assert aggs[key]["count"] == 1  # the interrupted record never counted as completed


def test_profiling_excluded_from_scientific_digest_flag():
    identity = m.load_identity()
    assert identity["profiling_artifact_schema"]["operational_only"] is True
    assert identity["profiling_artifact_schema"]["included_in_scientific_digest"] is False


def test_stage_name_enumeration_must_match_exactly():
    doc = _base_profiling_doc()
    doc["stage_name_enumeration"] = list(m.STAGE_NAME_ENUMERATION)[:-1]
    with pytest.raises(E):
        m.validate_profiling_document(doc)


# =====================================================================
# 5. Result document tests
# =====================================================================

def _base_result(*, image_count=500, window_count=1090, wall_s=100.0, within=None, digest="9" * 64, profiling_mode="off"):
    if within is None:
        within = wall_s <= 28800.0
    return {
        "schema_name": m.RESULT_SCHEMA_NAME, "schema_version": m.RESULT_SCHEMA_VERSION,
        "e20_identity_sha256": "a" * 64, "run_mode": "coco_stuff_500",
        "image_count": image_count, "window_count": window_count, "component_chunk_size": 16,
        "conservation_tolerance": 9.5367431640625e-07,
        "scientific_artifacts": [{
            "filename": "phase1b_diagnostic_stage_a.npz", "schema_name": "talk2dino-dfp-phase1b-stage-a-features-v1",
            "schema_version": 1, "row_count": 12345, "size_bytes": 999999, "sha256": "4" * 64,
        }],
        "scientific_collection_digest": digest,
        "profiling_artifact_reference": "phase1b_diagnostic_profiling.json",
        "profiling_mode": profiling_mode,
        "checkpoint_source": {}, "finalization_state": {}, "device_record": {}, "software_record": {},
        "total_job_wall_time_seconds": wall_s, "within_8_hour_allocation": within,
        "complete": True, "final": True,
        "offline_mechanistic_analysis": True, "deployed_inference_method": False, "full_5000_image_evaluation": False,
    }


def test_result_exactly_500_images_1090_windows():
    m.validate_result_document(_base_result())
    with pytest.raises(E):
        m.validate_result_document(_base_result(image_count=499))
    with pytest.raises(E):
        m.validate_result_document(_base_result(window_count=1091))


def test_result_complete_final_required():
    doc = _base_result()
    doc["complete"] = False
    with pytest.raises(E):
        m.validate_result_document(doc)


@pytest.mark.parametrize("wall_s,expected", [(28799.0, True), (28800.0, True), (28801.0, False)])
def test_eight_hour_flag_below_at_above(wall_s, expected):
    doc = _base_result(wall_s=wall_s, within=expected)
    m.validate_result_document(doc)
    bad = _base_result(wall_s=wall_s, within=not expected)
    with pytest.raises(E):
        m.validate_result_document(bad)


def test_offline_analysis_disclosure_required():
    doc = _base_result()
    doc["offline_mechanistic_analysis"] = False
    with pytest.raises(E):
        m.validate_result_document(doc)


def test_5000_image_prohibition_enforced():
    doc = _base_result()
    doc["full_5000_image_evaluation"] = True
    with pytest.raises(E):
        m.validate_result_document(doc)
    doc2 = _base_result()
    doc2["deployed_inference_method"] = True
    with pytest.raises(E):
        m.validate_result_document(doc2)


# =====================================================================
# 6. Equivalence contracts (synthetic documents)
# =====================================================================

def test_profiling_off_on_scientific_equivalence():
    off = _base_result(profiling_mode="off")
    on = copy.deepcopy(off)
    on["profiling_mode"] = "detailed"
    on["total_job_wall_time_seconds"] = 150.0  # operational field may differ
    m.assert_profiling_scientific_equivalence(off, on)


def test_profiling_off_on_scientific_digest_mismatch_rejected():
    off = _base_result(profiling_mode="off", digest="9" * 64)
    on = _base_result(profiling_mode="detailed", digest="8" * 64)
    with pytest.raises(E):
        m.assert_profiling_scientific_equivalence(off, on)


def test_profiling_off_on_nonoperational_field_mismatch_rejected():
    off = _base_result(profiling_mode="off")
    on = copy.deepcopy(off)
    on["profiling_mode"] = "detailed"
    on["image_count"] = 499  # non-operational field must not differ
    with pytest.raises(E):
        m.assert_profiling_scientific_equivalence(off, on)


def test_fresh_resume_scientific_equivalence():
    fresh = _base_result()
    resumed = copy.deepcopy(fresh)
    resumed["total_job_wall_time_seconds"] = 200.0  # operational; wall time legitimately differs
    m.assert_fresh_resume_equivalence(fresh, resumed)


def test_fresh_resume_digest_mismatch_rejected():
    fresh = _base_result(digest="9" * 64)
    resumed = _base_result(digest="8" * 64)
    with pytest.raises(E):
        m.assert_fresh_resume_equivalence(fresh, resumed)


def test_checkpoint_finalization_does_not_mutate_source_artifacts():
    # a complete-but-unfinalized checkpoint's scientific_state must be
    # bit-identical before and after a (simulated) finalize-only pass --
    # this module performs no writes, so "before" and "after" validation
    # of the same document object is the applicable non-mutation check.
    doc = _base_checkpoint(complete=True, completed_ids=[f"im_{i:04d}" for i in range(500)], completed_windows=1090)
    before = copy.deepcopy(doc)
    m.validate_checkpoint_document(doc)
    m.validate_resume_eligibility(doc)
    assert doc == before


# =====================================================================
# 7. Isolation -- zero forbidden imports / calls
# =====================================================================

_HERMETIC_ISOLATION_CHILD = r'''
import sys, os, builtins
FORBIDDEN = ("torch", "torchvision", "cv2", "mmcv", "mmengine", "mmseg")
_pre = sorted(n for n in FORBIDDEN if n in sys.modules)
if _pre:
    sys.stderr.write("FORBIDDEN PRESENT BEFORE IMPORT: %r\n" % (_pre,)); os._exit(11)
_real_import = builtins.__import__
def _guard(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in FORBIDDEN:
        sys.stderr.write("FORBIDDEN IMPORT ATTEMPTED: %s\n" % name); os._exit(12)
    return _real_import(name, globals, locals, fromlist, level)
builtins.__import__ = _guard
sys.path.insert(0, os.environ["IV_ROOT"])
import src.dfp_phase1b_diagnostic_execution as t
ident = t.load_identity(repo_root=os.environ["IV_ROOT"])
_post = sorted(n for n in FORBIDDEN if n in sys.modules)
if _post:
    sys.stderr.write("FORBIDDEN PRESENT AFTER: %r\n" % (_post,)); os._exit(13)
print("HERMETIC_OK")
sys.exit(0)
'''


def test_no_torch_or_heavy_modules_hermetic(tmp_path):
    child = tmp_path / "iv_hermetic.py"
    child.write_text(_HERMETIC_ISOLATION_CHILD)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "IV_ROOT": str(ROOT)}
    r = subprocess.run([sys.executable, "-I", str(child)], capture_output=True, text=True, cwd=str(tmp_path), env=env)
    assert r.returncode == 0, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    assert "HERMETIC_OK" in r.stdout


def test_module_source_has_no_forbidden_tokens():
    # NOTE: the module's frozen contract data legitimately contains
    # documentation *strings* describing what a FUTURE execution driver
    # must call (e.g. "torch.cuda.reset_peak_memory_stats(...)" inside
    # gpu_memory_semantics.stage_local_peak_mechanism) -- those are data,
    # not executable imports. Real isolation is proven separately by
    # test_no_torch_or_heavy_modules_hermetic. This scan checks only for
    # an actual executable import/call, never for documentation prose.
    src_text = (ROOT / "src/dfp_phase1b_diagnostic_execution.py").read_text()
    cli_text = CLI_PATH.read_text()
    for token in ("import torch", "import cv2", "import mmcv", "import mmengine", "import mmseg",
                  "sbatch", "srun", "salloc", "load_checkpoint", "build_model"):
        assert token not in src_text, f"{token!r} in diagnostic-execution module"
        assert token not in cli_text, f"{token!r} in diagnostic-execution CLI"


def test_no_sbatch_launcher_created_for_task6b():
    assert not (ROOT / "run_dfp_phase1b_diagnostic_execution.sbatch").exists()
    assert not (ROOT / "dfp_phase1b_diagnostic_execution.sbatch").exists()


# =====================================================================
# 8. CLI
# =====================================================================

def test_cli_has_only_the_five_subcommands():
    r = subprocess.run([sys.executable, str(CLI_PATH), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    import re
    blocks = re.findall(r"\{([a-z0-9,\-]+)\}", r.stdout)
    assert blocks
    subs = set(blocks[0].split(","))
    assert subs == {"verify-protocol", "verify-checkpoint-document", "verify-profiling-document", "verify-result-document", "show-contract"}
    for forbidden in ("execute", "run", "resume", "submit", "benchmark"):
        assert forbidden not in subs


def test_cli_verify_protocol_from_repo_root_and_fresh_cwd(tmp_path):
    r1 = subprocess.run([sys.executable, str(CLI_PATH), "verify-protocol"], capture_output=True, text=True, cwd=str(ROOT))
    assert r1.returncode == 0, r1.stderr
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}
    r2 = subprocess.run([sys.executable, str(CLI_PATH), "--repo-root", str(ROOT), "verify-protocol"],
                         capture_output=True, text=True, cwd=str(tmp_path), env=env)
    assert r2.returncode == 0, r2.stderr
    assert "Traceback" not in r2.stderr


def test_cli_verify_document_commands_exit_0_on_valid(tmp_path):
    ck = tmp_path / "checkpoint.json"
    ck.write_text(json.dumps(_base_checkpoint()))
    r = subprocess.run([sys.executable, str(CLI_PATH), "verify-checkpoint-document", "--document", str(ck)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    pr = tmp_path / "profiling.json"
    pr.write_text(json.dumps(_base_profiling_doc()))
    r = subprocess.run([sys.executable, str(CLI_PATH), "verify-profiling-document", "--document", str(pr)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    rs = tmp_path / "result.json"
    rs.write_text(json.dumps(_base_result()))
    r = subprocess.run([sys.executable, str(CLI_PATH), "verify-result-document", "--document", str(rs)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_cli_ordinary_failure_exit_2_no_traceback(tmp_path):
    bad = tmp_path / "bad_checkpoint.json"
    doc = _base_checkpoint()
    doc["progress"]["completed_image_ids"] = ["dup", "dup"]
    bad.write_text(json.dumps(doc))
    r = subprocess.run([sys.executable, str(CLI_PATH), "verify-checkpoint-document", "--document", str(bad)], capture_output=True, text=True)
    assert r.returncode == 2
    assert "Traceback" not in r.stderr
    assert "DFP PHASE 1B DIAGNOSTIC EXECUTION FAIL" in r.stderr


def test_cli_missing_file_exit_2_no_traceback():
    r = subprocess.run([sys.executable, str(CLI_PATH), "verify-result-document", "--document", "/no/such/file.json"], capture_output=True, text=True)
    assert r.returncode == 2
    assert "Traceback" not in r.stderr


def test_cli_malformed_json_exit_2_no_traceback(tmp_path):
    p = tmp_path / "malformed.json"
    p.write_text("{ not json")
    r = subprocess.run([sys.executable, str(CLI_PATH), "verify-result-document", "--document", str(p)], capture_output=True, text=True)
    assert r.returncode == 2
    assert "Traceback" not in r.stderr


def test_cli_show_contract_exit_0():
    r = subprocess.run([sys.executable, str(CLI_PATH), "show-contract"], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    assert "OFFLINE MECHANISTIC" in r.stdout
    assert "500" in r.stdout and "1090" in r.stdout


# =====================================================================
# 9. Fatal exceptions propagate unchanged
# =====================================================================

@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_fatal_exceptions_not_swallowed_in_load_identity(monkeypatch, exc_type):
    import tomllib as _tomllib
    def boom(*a, **k):
        raise exc_type("fatal during load")
    monkeypatch.setattr(_tomllib, "load", boom)
    with pytest.raises(exc_type):
        m.load_identity()


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_fatal_exceptions_not_swallowed_in_percentile(monkeypatch, exc_type):
    real_sorted = sorted
    def boom(*a, **k):
        raise exc_type("fatal during sort")
    monkeypatch.setattr("builtins.sorted", boom)
    try:
        with pytest.raises(exc_type):
            m.nearest_rank_percentile([1.0, 2.0], 0.5)
    finally:
        monkeypatch.setattr("builtins.sorted", real_sorted)
