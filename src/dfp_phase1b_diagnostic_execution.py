"""Strict loading/validation for the DFP Phase 1B diagnostic-execution
successor identity
(`evaluation_identities/e20_dfp_phase1b_diagnostic_execution.toml`) and
for the checkpoint / profiling / final-result DOCUMENTS a future
(not-yet-implemented) execution driver would produce under its contract.

This module is a NEW LEAF authority (E20). It is an ADDITIVE SUCCESSOR
to E19: it NEVER rewrites, re-binds, or cascades new hashes through
E13-E19, the verified calibration artifact, the installed Task 4
selection artifact, or any scientific kernel. It only relationally
re-verifies their SHA-256 against current file bytes on every load.

It implements no execution driver, profiler, checkpoint writer, result
writer, CLI execution command, or SLURM launcher. It performs no torch /
CUDA / model / dataset / image access -- pure TOML/JSON/filesystem/hash
validation and exact arithmetic only. `KeyboardInterrupt`, `SystemExit`,
and `MemoryError` always propagate unmodified; ordinary parse / type /
path / relational failures are wrapped in
`DfpPhase1bDiagnosticExecutionError`.

Document validators (checkpoint / profiling / result) check STRUCTURE,
TYPES, and CROSS-DOCUMENT RELATIONS -- they validate documents a future
writer would produce, they do not measure or execute anything
themselves, and this module never writes one in production use (only
tests build small synthetic documents).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.dfp_phase1b_chunk_benchmark_identity import (
    DfpPhase1bChunkBenchmarkIdentityError,
    resolve_repository_relative_path,
)


class DfpPhase1bDiagnosticExecutionError(ValueError):
    """Raised on any DFP Phase 1B diagnostic-execution identity or
    document invariant violation. Always fail closed. Never raised for,
    and never catches, KeyboardInterrupt / SystemExit / MemoryError --
    those always propagate unmodified."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e20_dfp_phase1b_diagnostic_execution.toml")
FORMAT_VERSION = "Talk2Dino-DFP-Phase1B-Diagnostic-Execution-Identity-V1"
IDENTITY_SCHEMA_NAME = "talk2dino-dfp-phase1b-diagnostic-execution-identity-v1"
IDENTITY_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_NAME = "talk2dino-dfp-phase1b-diagnostic-checkpoint-v1"
CHECKPOINT_SCHEMA_VERSION = 1
PROFILING_SCHEMA_NAME = "talk2dino-dfp-phase1b-diagnostic-profiling-v1"
PROFILING_SCHEMA_VERSION = 1
RESULT_SCHEMA_NAME = "talk2dino-dfp-phase1b-diagnostic-result-v1"
RESULT_SCHEMA_VERSION = 1
IMAGE_COUNT = 500
WINDOW_COUNT = 1090
COMPONENT_CHUNK_SIZE = 16
CONSERVATION_TOLERANCE = 9.5367431640625e-07
MAXIMUM_JOB_WALL_TIME_SECONDS = 28800

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

STAGE_NAME_ENUMERATION = (
    "authority_preflight_verification",
    "dataset_index_and_manifest_adapter_construction",
    "model_construction_and_checkpoint_loading",
    "final_result_validation_and_installation",
    "per_image_checkpoint_serialization_and_atomic_installation",
    "input_preparation",
    "unary_model_forward",
    "graph_construction",
    "canonical_k12_recurrence",
    "final_resolution_stitching_rescale_for_canonical_field",
    "global_unary_component_construction",
    "final_resolution_component_membership_pullback_construction",
    "component_masked_seed_construction",
    "masked_k12_recurrence",
    "signed_component_contribution_computation",
    "component_score_stitch_final_rescale",
    "conservation_residual_computation_and_validation",
    "scientific_row_array_assembly",
)
STAGE_SCOPES: dict[str, str] = {
    "authority_preflight_verification": "run",
    "dataset_index_and_manifest_adapter_construction": "run",
    "model_construction_and_checkpoint_loading": "run",
    "final_result_validation_and_installation": "run",
    "per_image_checkpoint_serialization_and_atomic_installation": "image",
    "input_preparation": "window",
    "unary_model_forward": "window",
    "graph_construction": "window",
    "canonical_k12_recurrence": "window",
    "final_resolution_stitching_rescale_for_canonical_field": "window",
    "global_unary_component_construction": "window",
    "final_resolution_component_membership_pullback_construction": "window",
    "component_masked_seed_construction": "component_chunk",
    "masked_k12_recurrence": "component_chunk",
    "signed_component_contribution_computation": "component_chunk",
    "component_score_stitch_final_rescale": "component_chunk",
    "conservation_residual_computation_and_validation": "window",
    "scientific_row_array_assembly": "window",
}
VALID_SCOPES = frozenset({"run", "image", "window", "component_chunk"})


# ---------------------------------------------------------------------
# Primitive checks (fail closed; reject bool-as-int, NaN/inf, blanks).
# ---------------------------------------------------------------------

def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1bDiagnosticExecutionError(f"{label} must be an exact bool, observed {type(value).__name__}")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise DfpPhase1bDiagnosticExecutionError(f"{label} must be an exact int, observed {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise DfpPhase1bDiagnosticExecutionError(f"{label} must be >= {minimum}, observed {value}")
    return value


def _require_exact_float(value: Any, label: str, *, minimum: float | None = None, allow_none: bool = False) -> float | None:
    if allow_none and value is None:
        return None
    if type(value) is not float:
        raise DfpPhase1bDiagnosticExecutionError(f"{label} must be an exact float, observed {type(value).__name__}")
    if not math.isfinite(value):
        raise DfpPhase1bDiagnosticExecutionError(f"{label} must be finite (no NaN/Infinity), observed {value!r}")
    if minimum is not None and value < minimum:
        raise DfpPhase1bDiagnosticExecutionError(f"{label} must be >= {minimum}, observed {value!r}")
    return value


def _require_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str:
        raise DfpPhase1bDiagnosticExecutionError(f"{label} must be a string, observed {type(value).__name__}")
    if nonempty and not value:
        raise DfpPhase1bDiagnosticExecutionError(f"{label} must not be empty")
    if value != value.strip():
        raise DfpPhase1bDiagnosticExecutionError(f"{label} must not have leading/trailing whitespace")
    if "\x00" in value:
        raise DfpPhase1bDiagnosticExecutionError(f"{label} must not contain a NUL byte")
    return value


def _require_sha256(value: Any, label: str) -> str:
    s = _require_string(value, label)
    if not _HEX64.match(s):
        raise DfpPhase1bDiagnosticExecutionError(f"{label} must be lowercase 64-hex, observed {value!r}")
    return s


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DfpPhase1bDiagnosticExecutionError(f"{label} must be a table")
    return value


def _require_closed(node: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    node = _require_mapping(node, label)
    actual = frozenset(node.keys())
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
    if missing or unknown:
        raise DfpPhase1bDiagnosticExecutionError(f"{label} has the wrong key set -- missing={missing} unknown={unknown}")
    return node


def _walk_leaves(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out.extend(_walk_leaves(v, f"{prefix}.{k}" if prefix else k))
    else:
        out.append((prefix, node))
    return out


# ---------------------------------------------------------------------
# Mechanically-flattened exact-value table for every E20 leaf OUTSIDE
# [dependency_bindings] -- generated once from the committed TOML.
# ---------------------------------------------------------------------

_CANONICAL: dict[str, Any] = {
    'atomicity_and_path_policy.atomic_rename_required': True,
    'atomicity_and_path_policy.binds_e15_atomic_write_policy': True,
    'atomicity_and_path_policy.binds_e15_path_policy': True,
    'atomicity_and_path_policy.canonical_filenames': ['phase1b_diagnostic_checkpoint.json', 'phase1b_diagnostic_stage_a.npz', 'phase1b_diagnostic_profiling.json', 'phase1b_diagnostic_result.json'],
    'atomicity_and_path_policy.exclusive_fresh_creation_by_default': True,
    'atomicity_and_path_policy.fsync_requirements': 'consistent with e15-dfp-phase1b-artifacts.[atomic_write_policy] (fsync staged files and the destination directory entry before the atomic rename that publishes them)',
    'atomicity_and_path_policy.installers_not_implemented_by_this_task': True,
    'atomicity_and_path_policy.lexical_dotdot_rejected': True,
    'atomicity_and_path_policy.no_orphaned_partial_final_bundle': True,
    'atomicity_and_path_policy.physical_containment_required': True,
    'atomicity_and_path_policy.run_specific_output_root': 'one single output root directory per run; every artifact for that run is a safe basename under it -- never an absolute or private path recorded in any authority document',
    'atomicity_and_path_policy.same_filesystem_staging_required': True,
    'atomicity_and_path_policy.source_checkpoint_and_scientific_artifacts_immutable_during_finalization': True,
    'atomicity_and_path_policy.symlink_escape_rejected': True,
    'atomicity_and_path_policy.verified_rollback_for_multi_file_replacement_required': True,
    'authorization_boundary.authority_only_task': True,
    'authorization_boundary.checkpoint_writer_implemented_by_task6b': False,
    'authorization_boundary.cli_execution_command_implemented_by_task6b': False,
    'authorization_boundary.execution_driver_implemented_by_task6b': False,
    'authorization_boundary.full_5000_image_miou_evaluation_authorized': False,
    'authorization_boundary.gpu_execution_performed_by_task6b': False,
    'authorization_boundary.lightweight_final_method_selected': False,
    'authorization_boundary.phase1b_5000_image_exact_provenance_authorized': False,
    'authorization_boundary.phase1b_500_image_execution_authorized': False,
    'authorization_boundary.profiler_implemented_by_task6b': False,
    'authorization_boundary.real_inference_executed_by_task6b': False,
    'authorization_boundary.result_writer_implemented_by_task6b': False,
    'authorization_boundary.sharding_across_jobs_or_gpus_authorized': False,
    'authorization_boundary.slurm_launcher_implemented_by_task6b': False,
    'checkpoint_schema.binds_e15_atomic_write_policy': True,
    'checkpoint_schema.binds_e15_checkpoint_resume_schema': True,
    'checkpoint_schema.binds_e15_path_policy': True,
    'checkpoint_schema.identity_bindings.e20_identity_sha256_circularity_note': "the checkpoint records THIS identity's own file hash for provenance, exactly as e19's own module binds itself by hash elsewhere in this repository's convention; the identity file itself never records a checkpoint's hash, so no cycle exists",
    'checkpoint_schema.identity_bindings.keys': ['e20_identity_name', 'e20_identity_schema_version', 'e20_identity_sha256', 'e15_protocol_sha256', 'e15_artifacts_sha256', 'e18_chunk_selection_sha256', 'e19_tolerance_sha256', 'subset_manifest_sha256', 'window_universe_digest', 'scientific_schema_name', 'scientific_schema_version', 'run_mode', 'device_type', 'gpu_model', 'gpu_count', 'component_chunk_size', 'conservation_tolerance'],
    'checkpoint_schema.profiling_state.incomplete_final_stage_handling': 'an in-progress (unterminated) stage record at interruption time is persisted with an explicit incomplete=true marker and excluded from all aggregate reconstruction; it is never promoted to a completed timing sample',
    'checkpoint_schema.profiling_state.keys': ['profiling_mode', 'profiling_record_filename', 'profiling_schema_name', 'profiling_schema_version', 'profiling_record_count', 'profiling_digest', 'incomplete_final_stage_handling'],
    'checkpoint_schema.profiling_state.profiling_state_excluded_from_scientific_digest': True,
    'checkpoint_schema.progress.completed_image_ids_order': 'exact prefix, in the frozen canonical (lexicographic fixed-width-ID) manifest order -- never reordered',
    'checkpoint_schema.progress.expected_image_count_value': 500,
    'checkpoint_schema.progress.expected_window_count_value': 1090,
    'checkpoint_schema.progress.keys': ['expected_image_count', 'expected_window_count', 'next_canonical_image_index', 'completed_image_ids', 'completed_window_count', 'total_processed_component_chunks', 'generation', 'complete', 'final_result_produced'],
    'checkpoint_schema.schema_name': 'talk2dino-dfp-phase1b-diagnostic-checkpoint-v1',
    'checkpoint_schema.schema_version': 1,
    'checkpoint_schema.scientific_state.binds_alternative_scientific_data': False,
    'checkpoint_schema.scientific_state.invents_no_new_scientific_array': True,
    'checkpoint_schema.scientific_state.keys': ['stage_a_artifact_filename', 'stage_a_schema_name', 'stage_a_schema_version', 'stage_a_row_count', 'stage_a_byte_sha256', 'scientific_collection_digest', 'progress_to_rows_relation'],
    'checkpoint_schema.scientific_state.progress_to_rows_relation': "stage_a_row_count must equal the flat disputed-pixel row count implied by completed_image_ids' dispute_offsets prefix -- an exact relation, never approximate",
    'checkpoint_schema.top_keys': ['schema_name', 'schema_version', 'identity_bindings', 'progress', 'scientific_state', 'profiling_state', 'complete', 'final'],
    'disclosure.chunk_size_is_frozen_not_reselected': True,
    'disclosure.conservation_tolerance_is_frozen_not_reselected': True,
    'disclosure.eight_hour_allocation_is_operational_disclosure_not_scientific_gate': True,
    'disclosure.future_5000_image_exact_provenance_requires_separate_explicit_authorization_and_is_not_implied_here': True,
    'disclosure.future_500_image_execution_requires_separate_explicit_authorization': True,
    'disclosure.not_a_5000_image_paper_evaluation': True,
    'disclosure.not_a_claim_of_efficient_inference': True,
    'disclosure.not_a_lightweight_final_method': True,
    'disclosure.not_a_replacement_for_full_validation_set_miou_evaluation': True,
    'disclosure.not_the_deployed_test_time_method': True,
    'disclosure.offline_mechanistic_provenance_analysis': True,
    'disclosure.population_is_500_images_1090_windows': True,
    'disclosure.profiling_on_off_scientific_equivalence_required': True,
    'disclosure.single_h100_single_gpu_contract': True,
    'disclosure.task6b_creates_no_execution_output': True,
    'disclosure.task6b_performs_no_inference': True,
    'final_result_schema.complete_required': True,
    'final_result_schema.deployed_inference_method_required_value': False,
    'final_result_schema.final_required': True,
    'final_result_schema.full_5000_image_evaluation_required_value': False,
    'final_result_schema.image_count_required_value': 500,
    'final_result_schema.offline_mechanistic_analysis_required_value': True,
    'final_result_schema.partial_generations_forbidden': True,
    'final_result_schema.schema_name': 'talk2dino-dfp-phase1b-diagnostic-result-v1',
    'final_result_schema.schema_version': 1,
    'final_result_schema.scientific_artifacts.keys_per_artifact': ['filename', 'schema_name', 'schema_version', 'row_count', 'size_bytes', 'sha256'],
    'final_result_schema.scientific_artifacts.references_e15_stage_a_schema': True,
    'final_result_schema.scientific_validity_independent_of_runtime_budget': True,
    'final_result_schema.top_keys': ['schema_name', 'schema_version', 'e20_identity_sha256', 'run_mode', 'image_count', 'window_count', 'component_chunk_size', 'conservation_tolerance', 'scientific_artifacts', 'scientific_collection_digest', 'profiling_artifact_reference', 'profiling_mode', 'checkpoint_source', 'finalization_state', 'device_record', 'software_record', 'total_job_wall_time_seconds', 'within_8_hour_allocation', 'complete', 'final', 'offline_mechanistic_analysis', 'deployed_inference_method', 'full_5000_image_evaluation'],
    'final_result_schema.window_count_required_value': 1090,
    'final_result_schema.within_8_hour_allocation_definition': 'total_job_wall_time_seconds <= 28800',
    'final_result_schema.within_8_hour_allocation_is_disclosure_not_acceptance_gate': True,
    'format_version': 'Talk2Dino-DFP-Phase1B-Diagnostic-Execution-Identity-V1',
    'fresh_resume_equivalence.interrupted_partial_stage_record_disposition': 'stored as an explicit incomplete record (never discarded silently, never treated as completed) -- matches checkpoint_schema.profiling_state.incomplete_final_stage_handling',
    'fresh_resume_equivalence.profiling_may_differ_in': ['wall_time', 'host_memory_measurements', 'gpu_memory_measurements'],
    'fresh_resume_equivalence.profiling_must_preserve': ['unique_record_identities', 'no_duplicate_completed_stage_records_after_resume', 'correct_interrupted_attempt_labeling', 'aggregate_reconstruction_from_accepted_records_only'],
    'fresh_resume_equivalence.required_equal_fields': ['row_order', 'selected_ids', 'window_order', 'offsets', 'flat_arrays', 'scientific_collection_digest', 'final_scientific_result_document_excluding_operational_fields'],
    'fresh_resume_equivalence.scientific_arrays_bit_identical_where_schema_promises_it': True,
    'future_step_bindings.description': 'Fields explicitly deferred to a later task, never guessed or invented here.',
    'future_step_bindings.next_step_items': ['independently re-verify this E20 identity and its loader/CLI module against the committed HEAD', 'implement the execution driver, profiler, checkpoint writer, and result writer bound by this contract -- a separate, later, separately-authorized task', 'explicitly authorize the 500-image diagnostic execution itself -- never this identity', 'the 5,000-image exact masked-forward provenance question and the full 5,000-image mIoU evaluation -- never this identity'],
    'gpu_memory_semantics.exact_device_requirement': 'device index is fixed and recorded; a run touching more than one CUDA device is a contract violation (single_gpu_only, per population.gpu_count=1)',
    'gpu_memory_semantics.fields': ['gpu_allocated_bytes_before', 'gpu_reserved_bytes_before', 'gpu_allocated_bytes_after', 'gpu_reserved_bytes_after', 'gpu_peak_allocated_bytes_during_stage', 'gpu_peak_reserved_bytes_during_stage', 'gpu_device_index', 'gpu_device_model'],
    'gpu_memory_semantics.integer_bytes_required': True,
    'gpu_memory_semantics.non_negative_validation_required': True,
    'gpu_memory_semantics.nvidia_smi_process_wide_values_forbidden_as_silent_replacement': 'if nvidia-smi (or any other process-wide/device-wide source) values are ever also recorded, they must be a distinct, separately-labeled field -- never silently substituted for the allocator-reported (torch.cuda.*) values above',
    'gpu_memory_semantics.reset_does_not_free_or_move_memory': True,
    'gpu_memory_semantics.reset_placement': "at stage entry, after any before-stage snapshot is taken, before the stage's own CUDA work begins",
    'gpu_memory_semantics.stage_local_peak_mechanism': "torch.cuda.reset_peak_memory_stats(device) immediately before stage entry; torch.cuda.max_memory_allocated(device) and torch.cuda.max_memory_reserved(device) immediately after the stage's defined CUDA measurement boundary (after the same synchronization point used for cuda_synchronization_policy)",
    'gpu_memory_semantics.synchronization_placement': 'identical to timing_semantics.cuda_synchronization_policy -- query peak stats only at the defined measurement boundary',
    'gpu_memory_semantics.unavailable_for_cpu_only_stages': 'gpu_* fields = null for a stage with no CUDA work, never fabricated as 0',
    'gpu_memory_semantics.values_are_operational_metadata_only': True,
    'host_memory_semantics.background_sampling_interval': 'not authorized by this identity -- no background memory sampler is frozen; only before/after and cumulative-peak measurements at defined stage boundaries',
    'host_memory_semantics.cumulative_high_water_mark_naming_forbidden_as_stage_peak': True,
    'host_memory_semantics.fields': ['host_rss_bytes_before', 'host_rss_bytes_after', 'process_peak_rss_bytes_after_stage'],
    'host_memory_semantics.final_process_peak_rss_source': 'resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, with an explicitly frozen platform conversion (Linux: kibibytes; the conversion factor used must itself be recorded alongside the raw value)',
    'host_memory_semantics.host_rss_source': '/proc/self/status VmRSS line',
    'host_memory_semantics.malformed_value_handling': 'a negative, non-finite, or non-integer parsed value is rejected as malformed, not silently coerced',
    'host_memory_semantics.proc_status_parsing_rule': "match the line beginning 'VmRSS:' or 'VmHWM:' (exact prefix, case-sensitive), take the sole embedded integer token as kibibytes",
    'host_memory_semantics.proc_status_to_bytes_conversion': 'bytes = KiB * 1024, exact integer arithmetic, no floating point',
    'host_memory_semantics.proc_status_unit': 'integer_kibibytes_as_reported_by_the_kernel',
    'host_memory_semantics.process_peak_rss_source': '/proc/self/status VmHWM line (cumulative high-water mark -- never labeled a per-stage-local peak)',
    'host_memory_semantics.unavailable_vs_zero': 'a field that could not be read (missing /proc, parse failure, permission denied) is recorded as null (unavailable), never as 0',
    'identity.does_not_modify_e13_through_e19': True,
    'identity.name': 'e20-dfp-phase1b-diagnostic-execution',
    'identity.purpose': 'Freezes the execution population (500 images / 1090 canonical windows), the exact scientific-execution binding (unmodified masked-forward provenance core), the operational profiling contract, the checkpoint/resume schema, and the final-result bundle schema for one future single-GPU offline mechanistic/provenance diagnostic run. Implements no execution driver, profiler, checkpoint writer, result writer, CLI execution command, or SLURM launcher. Authorizes no inference, no GPU job, and no exact masked-forward provenance on the full 5,000-image validation set.',
    'identity.required_ancestor_commit': '90bcf71671d69523bdc686fc695fdfb02b083a68',
    'identity.required_ancestor_relation': "required_ancestor_commit must be an ancestor of (or equal to) the executing repository_commit -- verified via 'git merge-base --is-ancestor <required_ancestor_commit> <repository_commit>', exactly like E17/E18/E19's own convention. Every repository-file SHA-256 bound below is valid only for a worktree at or descended from this commit.",
    'identity.schema_name': 'talk2dino-dfp-phase1b-diagnostic-execution-identity-v1',
    'identity.schema_version': 1,
    'identity.status': 'frozen_additive_successor_authority',
    'identity.successor_of': 'e19-dfp-phase1b-conservation-tolerance',
    'identity.successor_relation': 'additive_only: binds but never edits, reinterprets, or re-hashes any E13-E19 file, the verified calibration artifact, or the installed Task 4 selection artifact; packages the already-frozen E15 Stage-A/Stage-B scientific schemas, E16 candidate-window-universe geometry, E18 operational chunk size, and E19 conservation tolerance into one execution/checkpoint/profiling/result contract for a not-yet-authorized future run',
    'identity.working_branch': 'dfp-phase1a-diagnostics',
    'operational_scope.authority_only_task': True,
    'operational_scope.checkpoint_writer_implemented_by_task6b': False,
    'operational_scope.cli_execution_command_implemented_by_task6b': False,
    'operational_scope.execution_driver_implemented_by_task6b': False,
    'operational_scope.full_5000_image_miou_evaluation_authorized': False,
    'operational_scope.gpu_execution_performed_by_task6b': False,
    'operational_scope.lightweight_final_method_selected': False,
    'operational_scope.phase1b_5000_image_exact_provenance_authorized': False,
    'operational_scope.phase1b_500_image_execution_authorized': False,
    'operational_scope.profiler_implemented_by_task6b': False,
    'operational_scope.purpose': 'post_tolerance_500_image_diagnostic_execution_contract_freeze',
    'operational_scope.real_inference_executed_by_task6b': False,
    'operational_scope.result_writer_implemented_by_task6b': False,
    'operational_scope.scientific_accuracy_metric': False,
    'operational_scope.sharding_across_jobs_or_gpus_authorized': False,
    'operational_scope.slurm_launcher_implemented_by_task6b': False,
    'population.component_chunk_size': 16,
    'population.device_type': 'cuda',
    'population.gpu_count': 1,
    'population.gpu_model': 'NVIDIA H100 80GB HBM3',
    'population.image_count': 500,
    'population.image_universe_binding.canonical_image_order_source': "subset_manifest's own 'selected_ids' field, taken verbatim in its own already-frozen canonical (lexicographic fixed-width-ID) order -- identical to e16-dfp-phase1b-conservation-calibration's [candidate_scope].image_order_source -- never re-sorted, re-derived, or filtered here",
    'population.image_universe_binding.canonical_relative_paths_include_suffixes': True,
    'population.image_universe_binding.image_count_field': 'selected_count',
    'population.image_universe_binding.image_count_value': 500,
    'population.image_universe_binding.native_dataset_index_reconciliation_source': "dependency_bindings.dataset_view_module's build_native_image_index(...) / build_manifest_execution_view(...) / compute_mapping_digest(...) -- the same frozen adapter e18-dfp-phase1b-chunk-selection's own execution-context relies on for canonical_id <-> native filesystem index reconciliation; never a fresh ad hoc mapping",
    'population.image_universe_binding.no_dict_order_dependence': True,
    'population.image_universe_binding.no_filesystem_order_dependence': True,
    'population.image_universe_binding.no_hash_seed_dependence': True,
    'population.image_universe_binding.selected_ids_digest': '38c61b8f830475944ae94e3dba88ea62905159a46f30cfd2a951bee67540e770',
    'population.image_universe_binding.source': 'dependency_bindings.subset_manifest',
    'population.maximum_job_wall_time_seconds': 28800,
    'population.run_mode': 'coco_stuff_500',
    'population.sharding_across_jobs_or_gpus_forbidden': True,
    'population.single_logical_run_with_checkpoint_resume': True,
    'population.window_count': 1090,
    'population.window_universe_binding.candidate_universe_digest': '2163e2aeab0f7bf4e86cd3a33505388024d6cfb5ce7c1596c4d49691e364b390',
    'population.window_universe_binding.candidate_windows_per_image_source': "identical provenance chain to e16-dfp-phase1b-conservation-calibration.[candidate_scope].candidate_windows_per_image_source -- dependency_bindings.window_plan_module's SlidingWindowPlan.build(...)",
    'population.window_universe_binding.consistency_note': '175 distinct images (of 500) touched by 200 sampled calibration windows, and candidate_window_count=1090 spanning all 500 images, are mutually consistent (mean ~2.18 candidate windows per image); no contradiction with any other repository-recorded window count',
    'population.window_universe_binding.distinct_images_touched_by_200_calibration_windows': 175,
    'population.window_universe_binding.gt_annotation_prediction_dispute_or_model_output_influence_forbidden': True,
    'population.window_universe_binding.independent_recomputation_from_raw_images_performed_by_this_identity': False,
    'population.window_universe_binding.independent_recomputation_from_raw_images_required': False,
    'population.window_universe_binding.reconstruction_note': "the verified calibration artifact's own [candidate_scope]-governed candidate-window construction (e16-dfp-phase1b-conservation-calibration.[candidate_scope]) already ran SlidingWindowPlan.build(...) once per image, over the SAME 500-image coco_stuff_500 subset bound above, using the frozen crop=[448,448]/stride=[224,224] geometry from e13-dfp-phase1a-protocol's [score_interpolation_stitching] -- a pure per-image-canvas-dimension geometric computation with gt_annotation_prediction_dispute_or_model_output_influence_forbidden=true (e16.[candidate_scope]). candidate_window_count=1090 is that already-computed, already-independently-verified, digest-protected total candidate population; the 200 calibration windows are a size-200 SAMPLE drawn from it. This identity binds that persisted fact by content hash rather than re-deriving it from raw images (which would require dataset access this task never performs).",
    'population.window_universe_binding.source': 'dependency_bindings.calibration_artifact',
    'population.window_universe_binding.window_count_field': 'candidate_window_count',
    'population.window_universe_binding.window_count_value': 1090,
    'profiling_aggregation.aggregation_group_key': '(stage_name, scope)',
    'profiling_aggregation.duplicate_record_identity_rejection': "two records sharing the same canonical record identity (stage_name, scope, and the scope's covariate key: run_id / canonical_image_id / (canonical_image_id, window_ordinal) / (canonical_image_id, window_ordinal, component_chunk_index)) are rejected as a duplicate, never averaged or last-write-wins merged",
    'profiling_aggregation.integer_overflow_handling': "aggregate sums accumulate in Python's arbitrary-precision int/float; a persisted integer field exceeding int64 range is rejected rather than silently wrapped",
    'profiling_aggregation.mixed_scope_aggregation_forbidden': 'image-scoped and window-scoped (and component_chunk-scoped) records for the same stage name are never averaged or combined into one aggregate; aggregation groups strictly by (stage_name, scope)',
    'profiling_aggregation.non_finite_value_rejection': 'NaN or +-infinity in any profiling numeric field is rejected at record-validation time, never aggregated',
    'profiling_aggregation.one_record_behavior': 'count=1; sum=mean=min=max=median=p90=p95=p99 = that single value',
    'profiling_aggregation.percentile_convention': 'nearest_rank',
    'profiling_aggregation.percentile_rule': 'sort ascending; rank = max(1, ceil(p * n)); value = sorted_values[rank - 1] (0-indexed)',
    'profiling_aggregation.statistics': ['count', 'sum', 'mean', 'min', 'max', 'median', 'p90', 'p95', 'p99'],
    'profiling_aggregation.unavailable_cuda_field_behavior': 'a record with cuda_elapsed_ms = null is excluded from CUDA-timing aggregates but still counted in count and in any CPU-side aggregate for that same stage; excluding a null-valued record from a numeric aggregate must never silently zero-fill it',
    'profiling_aggregation.workload_covariates.disputed_pixel_count_authorization_note': 'disputed_pixel_count is already an E15/E16-authorized scientific-adjacent count (e15-dfp-phase1b-artifacts.[stage_a_schema].manifest_fields.dispute_count_per_image); recorded here only as a profiling covariate, never recomputed with a different definition',
    'profiling_aggregation.workload_covariates.list': ['canonical_image_index', 'canonical_image_id', 'window_index', 'window_canonical_identity', 'windows_per_image', 'component_count', 'query_class_count', 'component_chunk_count', 'node_count', 'disputed_pixel_count', 'component_chunk_size'],
    'profiling_aggregation.workload_covariates.no_new_scientific_statistic_introduced_for_profiling': True,
    'profiling_aggregation.zero_records_behavior': 'count=0; sum/mean/min/max/median/p90/p95/p99 = null (undefined), never 0.0 or NaN',
    'profiling_artifact_schema.included_in_scientific_digest': False,
    'profiling_artifact_schema.operational_only': True,
    'profiling_artifact_schema.record_identity_uniqueness_key': '(stage_name, scope, scope_covariate_key)',
    'profiling_artifact_schema.runtime_budget_fields': ['maximum_job_wall_time_seconds', 'elapsed_job_wall_time_seconds', 'remaining_budget_seconds'],
    'profiling_artifact_schema.schema_name': 'talk2dino-dfp-phase1b-diagnostic-profiling-v1',
    'profiling_artifact_schema.schema_version': 1,
    'profiling_artifact_schema.top_keys': ['schema_name', 'schema_version', 'e20_identity_sha256', 'profiling_mode', 'complete', 'final', 'run_identity', 'device_record', 'software_record', 'timing_method', 'host_memory_method', 'gpu_memory_method', 'stage_name_enumeration', 'records', 'aggregates', 'workload_covariates', 'total_run_wall_time_ns', 'total_scientific_processing_wall_time_ns', 'checkpoint_io_wall_time_ns', 'unattributed_overhead_ns', 'runtime_budget', 'warnings', 'source_scientific_collection_digest'],
    'profiling_stage_contract.function_bindings.authority_preflight_verification': "dependency_bindings.diagnostic_execution_module's own verify_protocol(...) / verify_calibration_binding-style relational checks (this identity's loader), mirroring diagnostics/run_dfp_phase1b_calibration.py's preflight Phase A+B pattern",
    'profiling_stage_contract.function_bindings.canonical_k12_recurrence': "dependency_bindings.recurrence_capture_module: capture_dfp_phase1a_trajectory(...), which calls dependency_bindings.finite_step_regime_module's finite_step_propagate(...) unmodified",
    'profiling_stage_contract.function_bindings.component_masked_seed_construction': 'dependency_bindings.masked_forward_core: build_masked_seed_chunk(...)',
    'profiling_stage_contract.function_bindings.component_score_stitch_final_rescale': 'dependency_bindings.masked_forward_core: stitch_and_rescale_component_scores(...)',
    'profiling_stage_contract.function_bindings.conservation_residual_computation_and_validation': 'dependency_bindings.masked_forward_core: build_conservation_measurements(...), compute_signed_margin_conservation_residual(...); dependency_bindings.phase1b_artifact_schema_module: validate_stage_a_flat_arrays(..., conservation_tolerance=...)',
    'profiling_stage_contract.function_bindings.dataset_index_and_manifest_adapter_construction': 'dependency_bindings.dataset_view_module: build_native_image_index(...), build_manifest_execution_view(...), compute_mapping_digest(...)',
    'profiling_stage_contract.function_bindings.final_resolution_component_membership_pullback_construction': 'dependency_bindings.final_rescale_pullback_module: build_final_stitching_pullback_plan(...), compute_window_final_soft_memberships(...)',
    'profiling_stage_contract.function_bindings.final_resolution_stitching_rescale_for_canonical_field': 'dependency_bindings.stitching_pullback_module: apply_window_stitching_forward(...); dependency_bindings.final_rescale_pullback_module: apply_final_rescale_forward(...)',
    'profiling_stage_contract.function_bindings.final_result_validation_and_installation': 'AUTHORITY/POLICY-BOUND, not yet function-bound: e15-dfp-phase1b-artifacts.[atomic_write_policy].result_install convention; the concrete installer is not implemented by Task 6B (Section 20)',
    'profiling_stage_contract.function_bindings.global_unary_component_construction': 'dependency_bindings.global_unary_components_module: compute_global_unary_components(...)',
    'profiling_stage_contract.function_bindings.graph_construction': 'dependency_bindings.graph_module: build_directed_topk_graph(...)',
    'profiling_stage_contract.function_bindings.input_preparation': "dependency_bindings.window_plan_module's SlidingWindowPlan.build(...) per-window crop extraction, composed with dependency_bindings.dataset_view_module's native image resolution",
    'profiling_stage_contract.function_bindings.masked_k12_recurrence': 'dependency_bindings.masked_forward_core: propagate_masked_seed_chunk(...)',
    'profiling_stage_contract.function_bindings.model_construction_and_checkpoint_loading': "AUTHORITY-BOUND, not yet function-bound: dependency_bindings.e15_protocol's authority_bindings.k12_graph_recurrence (e12-matched-k11-k12-t320) and authority_bindings.text_prototype_projection (e3-paired-soft-routing); the concrete model-construction/checkpoint-loading function is defined by the future (not-yet-implemented) execution driver and must resolve within this authority chain -- Task 6B implements no driver, so no concrete function exists yet",
    'profiling_stage_contract.function_bindings.per_image_checkpoint_serialization_and_atomic_installation': 'POLICY-BOUND, not yet function-bound: e15-dfp-phase1b-artifacts.[checkpoint_resume_schema] + [atomic_write_policy].checkpoint_install; checkpoint_after_every_completed_image=true per e15-dfp-phase1b-protocol.[resource_ceiling]; the concrete writer is not implemented by Task 6B',
    'profiling_stage_contract.function_bindings.scientific_row_array_assembly': 'SCHEMA-BOUND, not a single function: assembly into e15-dfp-phase1b-artifacts.[stage_a_schema].flat_disputed_pixel_members member layout, validated by dependency_bindings.phase1b_artifact_schema_module',
    'profiling_stage_contract.function_bindings.signed_component_contribution_computation': 'dependency_bindings.masked_forward_core: compute_signed_contributions(...)',
    'profiling_stage_contract.function_bindings.unary_model_forward': "AUTHORITY-BOUND, not yet function-bound: same e12/e3 authority chain as model_construction_and_checkpoint_loading; produces the s0 tensor consumed as an INPUT by dependency_bindings.recurrence_capture_module's capture_dfp_phase1a_trajectory(graph, s0, ...) -- the model forward itself precedes and is external to every frozen kernel module bound by this identity",
    'profiling_stage_contract.non_overlapping_decomposition_note': 'matches the 4 run-level + 14 per-image/per-window/per-component-chunk stages named in the Task 6B specification exactly, in the same order; no stage was renamed, merged, or split beyond that list',
    'profiling_stage_contract.overhead_accounting.double_counting_forbidden': True,
    'profiling_stage_contract.overhead_accounting.leaf_stage_scopes': ['window', 'component_chunk', 'image', 'run'],
    'profiling_stage_contract.overhead_accounting.leaf_stages_non_overlapping': True,
    'profiling_stage_contract.overhead_accounting.parent_aggregate_labeling': 'inclusive; a parent/group aggregate (e.g. a window total, an image total, the run total) must never be summed together with its own child leaf-stage totals -- it already contains them',
    'profiling_stage_contract.overhead_accounting.unattributed_overhead_definition': "total wall time for the enclosing scope minus the sum of that scope's own non-overlapping leaf-stage elapsed times; negative overhead is invalid and must be rejected, never clamped to zero",
    'profiling_stage_contract.overhead_accounting.window_total_definition': 'sum of the 9 window-scoped leaf stages (input_preparation .. scientific_row_array_assembly, excluding the 4 component_chunk-scoped stages) plus, for each component chunk of that window, the sum of the 4 component_chunk-scoped leaf stages for that chunk',
    'profiling_stage_contract.scopes.authority_preflight_verification': 'run',
    'profiling_stage_contract.scopes.canonical_k12_recurrence': 'window',
    'profiling_stage_contract.scopes.component_masked_seed_construction': 'component_chunk',
    'profiling_stage_contract.scopes.component_score_stitch_final_rescale': 'component_chunk',
    'profiling_stage_contract.scopes.conservation_residual_computation_and_validation': 'window',
    'profiling_stage_contract.scopes.dataset_index_and_manifest_adapter_construction': 'run',
    'profiling_stage_contract.scopes.final_resolution_component_membership_pullback_construction': 'window',
    'profiling_stage_contract.scopes.final_resolution_stitching_rescale_for_canonical_field': 'window',
    'profiling_stage_contract.scopes.final_result_validation_and_installation': 'run',
    'profiling_stage_contract.scopes.global_unary_component_construction': 'window',
    'profiling_stage_contract.scopes.graph_construction': 'window',
    'profiling_stage_contract.scopes.input_preparation': 'window',
    'profiling_stage_contract.scopes.masked_k12_recurrence': 'component_chunk',
    'profiling_stage_contract.scopes.model_construction_and_checkpoint_loading': 'run',
    'profiling_stage_contract.scopes.per_image_checkpoint_serialization_and_atomic_installation': 'image',
    'profiling_stage_contract.scopes.scientific_row_array_assembly': 'window',
    'profiling_stage_contract.scopes.signed_component_contribution_computation': 'component_chunk',
    'profiling_stage_contract.scopes.unary_model_forward': 'window',
    'profiling_stage_contract.stage_count': 18,
    'profiling_stage_contract.stage_name_enumeration': ['authority_preflight_verification', 'dataset_index_and_manifest_adapter_construction', 'model_construction_and_checkpoint_loading', 'final_result_validation_and_installation', 'per_image_checkpoint_serialization_and_atomic_installation', 'input_preparation', 'unary_model_forward', 'graph_construction', 'canonical_k12_recurrence', 'final_resolution_stitching_rescale_for_canonical_field', 'global_unary_component_construction', 'final_resolution_component_membership_pullback_construction', 'component_masked_seed_construction', 'masked_k12_recurrence', 'signed_component_contribution_computation', 'component_score_stitch_final_rescale', 'conservation_residual_computation_and_validation', 'scientific_row_array_assembly'],
    'prohibited.list': ['modifying_e13_through_e19_identities', 'modifying_the_verified_calibration_artifact_or_installed_task4_selection_artifact', 'modifying_masked_forward_recurrence_graph_stitching_pullback_or_component_code', 'redefining_component_definition_query_class_selection_prediction_variants_recurrence_contribution_sign_resizing_stitching_conservation_formula_tolerance_or_chunk_size', 'implementing_an_execution_driver_profiler_checkpoint_writer_result_writer_or_cli_execution_command', 'implementing_a_slurm_launcher', 'running_real_inference_or_initializing_cuda', 'accessing_coco_stuff_images_or_annotations', 'sharding_the_500_image_run_across_multiple_jobs_or_gpus', 'authorizing_the_500_image_diagnostic_execution', 'authorizing_exact_masked_forward_provenance_on_the_5000_image_validation_set', 'authorizing_the_full_5000_image_miou_evaluation', 'selecting_a_lightweight_final_method', 'adding_a_new_scientific_statistic_only_to_aid_profiling', 'creating_a_circular_hash_dependency'],
    'resume_rules.already_finalized_runs_rejected_for_resume': True,
    'resume_rules.bindings_must_match_exactly': True,
    'resume_rules.complete_unfinalized_checkpoint_may_enter_finalize_only_path': True,
    'resume_rules.completed_ids_must_be_exact_prefix_of_frozen_manifest': True,
    'resume_rules.completed_windows_must_match_that_prefixs_canonical_windows': True,
    'resume_rules.corrupted_hashes_rejected': True,
    'resume_rules.fresh_overwrite_requires_separate_explicit_authorization': True,
    'resume_rules.fresh_requires_no_existing_checkpoint_or_artifacts': True,
    'resume_rules.partial_generations_rejected': True,
    'resume_rules.profiling_record_disagreement_rejected_note': "a resumed run's profiling records for an already-completed image must not be regenerated or altered; only new records for not-yet-completed work may be appended",
    'resume_rules.resume_must_not_reprocess_completed_images': True,
    'resume_rules.resume_requires_valid_incomplete_checkpoint': True,
    'resume_rules.scientific_state_restoration_exact': True,
    'resume_rules.skipped_duplicated_reordered_or_cross_fed_rows_forbidden': True,
    'resume_rules.v1_or_unknown_schema_rejected': True,
    'scientific_execution_binding.acceptance_relation': 'absolute_residual <= conservation_tolerance',
    'scientific_execution_binding.canonical_recurrence_dtype': 'float64',
    'scientific_execution_binding.comparison_operator': '<=',
    'scientific_execution_binding.component_chunk_size': 16,
    'scientific_execution_binding.component_chunk_size_source': "dependency_bindings.e18_chunk_selection_identity's frozen selected_chunk_size, cross-checked against the installed task4_selection_artifact",
    'scientific_execution_binding.conservation_tolerance': 9.5367431640625e-07,
    'scientific_execution_binding.conservation_tolerance_binary_exponent': -20,
    'scientific_execution_binding.deterministic_scientific_serialization': 'identical array member ordering, dtype, and byte layout to e15-dfp-phase1b-artifacts.[stage_a_schema].flat_disputed_pixel_members; no new scientific member is introduced by this identity',
    'scientific_execution_binding.final_resolution_pullback_operator': "A_final,w = R o A_stitch,w (dependency_bindings.stitching_pullback_module's apply_window_stitching_forward composed with dependency_bindings.final_rescale_pullback_module's apply_final_rescale_forward), exactly as reused by compute_signed_contributions / stitch_and_rescale_component_scores in dependency_bindings.masked_forward_core",
    'scientific_execution_binding.governed_residual_family': 'rho_g',
    'scientific_execution_binding.governed_residual_formula': 'rho_g^(w)(x) = sum_r g^(w)_{x,r} - delta_m^(w)(x), unchanged from e16-dfp-phase1b-conservation-calibration.[residual_definitions].rho_g.formula',
    'scientific_execution_binding.graph_k': 12,
    'scientific_execution_binding.graph_storage_dtype': 'float32',
    'scientific_execution_binding.masked_recurrence_dtype': 'float64',
    'scientific_execution_binding.non_finite_value_failure_behavior': 'reject: a non-finite scientific array member is a fatal E15/E16/masked-forward-core-defined invariant violation, never silently dropped, clamped, or imputed',
    'scientific_execution_binding.precision_source': "verified calibration artifact's own recorded propagation_precision_record, produced under e16-dfp-phase1b-conservation-calibration's same_propagation_precision_required=true constraint -- never re-selected or re-guessed here",
    'scientific_execution_binding.recurrence_alpha': 0.98,
    'scientific_execution_binding.recurrence_alpha_source': "e13-dfp-phase1a-protocol.[propagation].alpha, reused verbatim via e15-dfp-phase1b-protocol's authority_bindings.phase1a_protocol -- never redefined here",
    'scientific_execution_binding.recurrence_steps': 320,
    'scientific_execution_binding.residual_accumulation_dtype': 'float64',
    'scientific_execution_binding.scientific_collection_digest_definition': "sha256 over the canonical concatenation of every Stage-A flat array member's raw native-endian bytes, in the exact member order fixed by e15-dfp-phase1b-artifacts.[stage_a_schema].flat_disputed_pixel_members, preceded by image_count/dispute_offsets/window_count_per_image encoded the same way stage_a_schema.manifest_fields already requires; profiling fields are never included",
    'scientific_execution_binding.signed_contribution_convention': "unchanged from e16-dfp-phase1b-conservation-calibration.[residual_definitions].rho_g.g_definition -- g^(w)_{x,r}: the selected window's own signed component contribution, computed from S0/S_final AFTER applying A_final,w",
    'scientific_execution_binding.stage_a_schema_binding': 'e15-dfp-phase1b-artifacts.[stage_a_schema] (talk2dino-dfp-phase1b-stage-a-features-v1, schema_version 1) -- bound and reused verbatim, never redefined',
    'scientific_execution_binding.stage_b_schema_binding': 'e15-dfp-phase1b-artifacts.[stage_b_schema] (talk2dino-dfp-phase1b-stage-b-labels-v1, schema_version 1) -- bound and reused verbatim, never redefined; Stage B is out of scope for the 500-image mechanistic diagnostic itself (Stage B evaluation labels are a downstream Phase 1B step) and this identity authorizes no Stage B computation',
    'scientific_execution_binding.tolerance_source': 'dependency_bindings.e19_tolerance_identity -- frozen, not re-derived or re-selected here',
    'scientific_execution_binding.tolerance_violation_failure_behavior': "reject: absolute_residual > conservation_tolerance raises the committed src/dfp_phase1b_artifact_schema.py::validate_stage_a_flat_arrays domain error, exactly as already enforced for e15-dfp-phase1b-artifacts's stage_a_schema; this identity adds no alternative acceptance path",
    'scientific_execution_binding.unmodified_list.list': ['component_definition', 'query_class_selection', 'prediction_variants', 'recurrence', 'contribution_sign', 'resizing', 'stitching', 'conservation_formula', 'tolerance', 'chunk_size'],
    'timing_semantics.cpu_clock': 'time.perf_counter_ns',
    'timing_semantics.cpu_end_definition': "immediately after the bound stage's required completion (its defined exit point, not merely enqueue when the stage is itself CPU-only)",
    'timing_semantics.cpu_start_definition': "immediately before the bound stage's entry",
    'timing_semantics.cpu_unit': 'integer_nanoseconds',
    'timing_semantics.cuda_elapsed_unit': 'float64_milliseconds_as_returned_by_the_cuda_event_elapsed_time_api',
    'timing_semantics.cuda_event_end_definition': "recorded immediately after enqueueing that stage's final CUDA operation",
    'timing_semantics.cuda_event_start_definition': 'recorded immediately before the bound CUDA stage begins',
    'timing_semantics.cuda_synchronization_overhead_disclosure': "any CPU time spent blocked in a synchronization call at a measurement boundary must be disclosed as a distinct field, never silently folded into the adjacent stage's elapsed time",
    'timing_semantics.cuda_synchronization_policy': 'synchronize only at the defined measurement boundary (elapsed-time query), never mid-stage; unsynchronized CPU wall time must never be substituted for CUDA event timing',
    'timing_semantics.cuda_timing_mechanism': 'cuda_events_on_the_active_execution_stream',
    'timing_semantics.cuda_timing_unavailable_for_cpu_only_stages': 'a CPU-only stage (e.g. authority_preflight_verification, dataset_index_and_manifest_adapter_construction, scientific_row_array_assembly) records cuda_elapsed_ms = null (unavailable), never a fabricated 0.0',
    'timing_semantics.cuda_to_persisted_unit_conversion': 'persist the float64 milliseconds value directly, unrounded; if a nanosecond integer field is also required for cross-comparison with CPU timings, convert via nanoseconds = round_half_even(milliseconds * 1e6) and record both the original float64 milliseconds and the derived integer nanoseconds, never only the derived value',
    'timing_semantics.cuda_warmup_labeling': "initialization/warm-up CUDA work (first-call kernel compilation, cuDNN/cuBLAS algorithm selection, context creation) must be recorded under a separate warmup-labeled record, never merged into a scientific stage's steady-state timing",
    'timing_semantics.total_run_wall_time_definition': 'measured independently of any stage-total reconstruction, via one time.perf_counter_ns() pair spanning the entire run process from authority_preflight_verification entry to final_result_validation_and_installation exit (or to the last checkpoint write on an incomplete run)',
}
# total leaves: 291


TOP_KEYS: frozenset[str] = frozenset({
    "format_version", "identity", "operational_scope", "dependency_bindings", "population",
    "scientific_execution_binding", "profiling_stage_contract", "timing_semantics",
    "host_memory_semantics", "gpu_memory_semantics", "profiling_aggregation", "checkpoint_schema",
    "resume_rules", "fresh_resume_equivalence", "profiling_artifact_schema", "final_result_schema",
    "atomicity_and_path_policy", "authorization_boundary", "disclosure", "prohibited",
    "future_step_bindings",
})

_DEP_REPO_NAMES = (
    "e15_protocol", "e15_artifacts", "e16_calibration_identity", "e17_chunk_benchmark_identity",
    "e18_chunk_selection_identity", "e19_tolerance_identity", "subset_manifest", "window_plan_module",
    "graph_module", "finite_step_regime_module", "recurrence_capture_module", "stitching_pullback_module",
    "final_rescale_pullback_module", "horizon_stitching_module", "global_unary_components_module",
    "soft_membership_module", "dataset_view_module", "masked_forward_core", "phase1b_artifact_schema_module",
    "calibration_artifact_validator_module", "tolerance_module", "chunk_selection_module",
    "chunk_benchmark_identity_module", "diagnostic_execution_module", "diagnostic_execution_cli_module",
)
_SYMBOLIC_DEP_NAMES = ("calibration_artifact", "task4_selection_artifact", "task4_selection_report")
_DEP_BINDING_KEYS = frozenset({"relative_path", "sha256", "untouched_by_this_identity"})
_DEP_BINDING_KEYS_SELFREF = frozenset({"relative_path", "sha256", "untouched_by_this_identity", "self_reference_note"})
_SYMBOLIC_BINDING_KEYS = frozenset({
    "logical_identifier", "relative_path_is_symbolic_not_repository_relative", "relative_path_note",
    "sha256", "untouched_by_this_identity",
})


# ---------------------------------------------------------------------
# load_identity: closed schema + exact values + dependency hash verify.
# ---------------------------------------------------------------------

def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load, closed-schema-validate, exact-value-validate, and
    relationally hash-verify the E20 diagnostic-execution identity.
    Never mutates or re-derives any bound dependency -- only re-verifies
    SHA-256 against current bytes. Performs no torch/CUDA/model/dataset
    access."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DfpPhase1bDiagnosticExecutionError(f"cannot load E20 diagnostic-execution identity {source}: {error}") from error

    _require_closed(identity, TOP_KEYS, "E20 identity")

    dep = _require_mapping(identity.get("dependency_bindings"), "dependency_bindings")
    expected_dep = frozenset(_DEP_REPO_NAMES) | frozenset(_SYMBOLIC_DEP_NAMES)
    if frozenset(dep.keys()) != expected_dep:
        raise DfpPhase1bDiagnosticExecutionError(
            f"dependency_bindings has the wrong key set -- missing={sorted(expected_dep - frozenset(dep))} "
            f"unknown={sorted(frozenset(dep) - expected_dep)}"
        )

    # every leaf outside [dependency_bindings] must match _CANONICAL exactly
    observed_leaves: dict[str, Any] = {}
    for dotted, value in _walk_leaves(identity):
        if dotted.startswith("dependency_bindings."):
            continue
        observed_leaves[dotted] = value
    canon_keys = frozenset(_CANONICAL)
    obs_keys = frozenset(observed_leaves)
    if canon_keys != obs_keys:
        raise DfpPhase1bDiagnosticExecutionError(
            f"E20 leaf set drifted -- missing={sorted(canon_keys - obs_keys)[:5]} extra={sorted(obs_keys - canon_keys)[:5]}"
        )
    for dotted, expected in _CANONICAL.items():
        observed = observed_leaves[dotted]
        if type(observed) is not type(expected) or observed != expected:
            raise DfpPhase1bDiagnosticExecutionError(
                f"{dotted}: frozen value drift -- expected {expected!r} ({type(expected).__name__}), "
                f"observed {observed!r} ({type(observed).__name__})"
            )

    _verify_dependency_hashes(root, dep)
    return identity


def _verify_dependency_hashes(root: Path, dependency_bindings: Mapping[str, Any]) -> None:
    for name in _DEP_REPO_NAMES:
        binding = dependency_bindings[name]
        keys = _DEP_BINDING_KEYS_SELFREF if name == "diagnostic_execution_module" else _DEP_BINDING_KEYS
        _require_closed(binding, keys, f"dependency_bindings.{name}")
        sha = binding["sha256"]
        if name in ("diagnostic_execution_module", "diagnostic_execution_cli_module"):
            if sha == "PLACEHOLDER_RECOMPUTED_AFTER_MODULE_IS_FINAL":
                raise DfpPhase1bDiagnosticExecutionError(
                    f"dependency_bindings.{name}.sha256 is still a placeholder -- recompute and freeze it"
                )
        expected = _require_sha256(sha, f"dependency_bindings.{name}.sha256")
        try:
            target = resolve_repository_relative_path(root, binding["relative_path"], binding_name=name)
        except DfpPhase1bChunkBenchmarkIdentityError as exc:
            raise DfpPhase1bDiagnosticExecutionError(
                f"dependency_bindings.{name}: unsafe or unresolvable relative_path -- {exc}"
            ) from exc
        try:
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as exc:
            raise DfpPhase1bDiagnosticExecutionError(
                f"dependency_bindings.{name}: cannot read bound file {binding['relative_path']}: {exc}"
            ) from exc
        if actual != expected:
            raise DfpPhase1bDiagnosticExecutionError(
                f"dependency_bindings.{name}: {binding['relative_path']} SHA-256 mismatch -- "
                f"pinned {expected}, observed {actual} (bound dependency drifted since E20 was frozen)"
            )
    for name in _SYMBOLIC_DEP_NAMES:
        _require_closed(dependency_bindings[name], _SYMBOLIC_BINDING_KEYS, f"dependency_bindings.{name}")
        _require_sha256(dependency_bindings[name]["sha256"], f"dependency_bindings.{name}.sha256")


def verify_external_artifact_bindings(
    identity: Mapping[str, Any] | None = None,
    *,
    calibration_artifact_path: Path | str,
    task4_selection_artifact_path: Path | str,
    task4_selection_report_path: Path | str,
    repo_root: Path | None = None,
) -> dict[str, str]:
    """Hash-verify the three scratch-hosted (symbolic) dependencies
    E20 binds by content digest only: the verified calibration artifact
    and the installed Task 4 selection artifact + report. Does not
    re-run the E16/E18 official validators itself (callers that need
    the full family-reconstruction / winner-recomputation should call
    ``src.dfp_phase1b_calibration_artifact`` / ``src.dfp_phase1b_chunk_selection``
    directly) -- this function only confirms E20's own pinned hashes
    still match the live bytes."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = identity if identity is not None else load_identity(repo_root=root)
    dep = identity["dependency_bindings"]
    results: dict[str, str] = {}
    for logical_name, path, binding_name in (
        ("calibration_artifact", calibration_artifact_path, "calibration_artifact"),
        ("task4_selection_artifact", task4_selection_artifact_path, "task4_selection_artifact"),
        ("task4_selection_report", task4_selection_report_path, "task4_selection_report"),
    ):
        p = Path(path)
        try:
            raw = p.read_bytes()
        except OSError as exc:
            raise DfpPhase1bDiagnosticExecutionError(f"cannot read {logical_name} at {p}: {exc}") from exc
        actual = hashlib.sha256(raw).hexdigest()
        expected = dep[binding_name]["sha256"]
        if actual != expected:
            raise DfpPhase1bDiagnosticExecutionError(
                f"{logical_name}: SHA-256 mismatch -- E20 pinned {expected}, observed {actual}"
            )
        results[logical_name] = actual
    return results


# ---------------------------------------------------------------------
# Nearest-rank percentile + deterministic aggregation.
# ---------------------------------------------------------------------

def nearest_rank_percentile(values: Sequence[float], p: float) -> float:
    """sort ascending; rank = max(1, ceil(p * n)); value = sorted[rank-1]."""
    if type(p) is not float or not (0.0 < p <= 1.0):
        raise DfpPhase1bDiagnosticExecutionError(f"percentile p must be an exact float in (0, 1], observed {p!r}")
    if not values:
        raise DfpPhase1bDiagnosticExecutionError("nearest_rank_percentile requires at least one value")
    for v in values:
        if type(v) is not float or not math.isfinite(v):
            raise DfpPhase1bDiagnosticExecutionError(f"nearest_rank_percentile: non-finite or non-float value {v!r}")
    n = len(values)
    rank = max(1, math.ceil(p * n))
    return sorted(values)[rank - 1]


def aggregate_numeric_records(values: Sequence[float]) -> dict[str, Any]:
    """count/sum/mean/min/max/median/p90/p95/p99 over a homogeneous
    (single stage_name, single scope) list of finite float values.
    Zero records -> count=0 and every statistic is None. One record ->
    every statistic equals that value."""
    for v in values:
        if type(v) is not float or not math.isfinite(v):
            raise DfpPhase1bDiagnosticExecutionError(f"aggregate_numeric_records: non-finite or non-float value {v!r}")
    n = len(values)
    if n == 0:
        return {"count": 0, "sum": None, "mean": None, "min": None, "max": None, "median": None, "p90": None, "p95": None, "p99": None}
    total = math.fsum(values)
    return {
        "count": n,
        "sum": total,
        "mean": total / n,
        "min": min(values),
        "max": max(values),
        "median": nearest_rank_percentile(values, 0.5),
        "p90": nearest_rank_percentile(values, 0.90),
        "p95": nearest_rank_percentile(values, 0.95),
        "p99": nearest_rank_percentile(values, 0.99),
    }


# ---------------------------------------------------------------------
# Checkpoint document validator.
# ---------------------------------------------------------------------

_CHECKPOINT_TOP_KEYS = frozenset({"schema_name", "schema_version", "identity_bindings", "progress", "scientific_state", "profiling_state", "complete", "final"})
_CHECKPOINT_IDENTITY_KEYS = frozenset({
    "e20_identity_name", "e20_identity_schema_version", "e20_identity_sha256", "e15_protocol_sha256",
    "e15_artifacts_sha256", "e18_chunk_selection_sha256", "e19_tolerance_sha256", "subset_manifest_sha256",
    "window_universe_digest", "scientific_schema_name", "scientific_schema_version", "run_mode",
    "device_type", "gpu_model", "gpu_count", "component_chunk_size", "conservation_tolerance",
})
_CHECKPOINT_PROGRESS_KEYS = frozenset({
    "expected_image_count", "expected_window_count", "next_canonical_image_index", "completed_image_ids",
    "completed_window_count", "total_processed_component_chunks", "generation", "complete", "final_result_produced",
})
_CHECKPOINT_SCIENTIFIC_STATE_KEYS = frozenset({
    "stage_a_artifact_filename", "stage_a_schema_name", "stage_a_schema_version", "stage_a_row_count",
    "stage_a_byte_sha256", "scientific_collection_digest",
})
_CHECKPOINT_PROFILING_STATE_KEYS = frozenset({
    "profiling_mode", "profiling_record_filename", "profiling_schema_name", "profiling_schema_version",
    "profiling_record_count", "profiling_digest",
})


def validate_checkpoint_document(doc: Any) -> Mapping[str, Any]:
    """Structural/relational validation of a (real or synthetic)
    diagnostic-checkpoint document against the closed checkpoint schema.
    Never reads a file, never writes one."""
    doc = _require_closed(doc, _CHECKPOINT_TOP_KEYS, "checkpoint")
    if _require_string(doc["schema_name"], "checkpoint.schema_name") != CHECKPOINT_SCHEMA_NAME:
        raise DfpPhase1bDiagnosticExecutionError(f"checkpoint.schema_name must be {CHECKPOINT_SCHEMA_NAME!r}")
    if _require_exact_int(doc["schema_version"], "checkpoint.schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise DfpPhase1bDiagnosticExecutionError(f"checkpoint.schema_version must be {CHECKPOINT_SCHEMA_VERSION} (v1/unknown schemas are rejected)")

    ib = _require_closed(doc["identity_bindings"], _CHECKPOINT_IDENTITY_KEYS, "checkpoint.identity_bindings")
    _require_string(ib["e20_identity_name"], "identity_bindings.e20_identity_name")
    _require_exact_int(ib["e20_identity_schema_version"], "identity_bindings.e20_identity_schema_version", minimum=1)
    _require_sha256(ib["e20_identity_sha256"], "identity_bindings.e20_identity_sha256")
    for k in ("e15_protocol_sha256", "e15_artifacts_sha256", "e18_chunk_selection_sha256", "e19_tolerance_sha256", "subset_manifest_sha256", "window_universe_digest"):
        _require_sha256(ib[k], f"identity_bindings.{k}")
    _require_string(ib["scientific_schema_name"], "identity_bindings.scientific_schema_name")
    _require_exact_int(ib["scientific_schema_version"], "identity_bindings.scientific_schema_version", minimum=1)
    if ib["run_mode"] != "coco_stuff_500":
        raise DfpPhase1bDiagnosticExecutionError("identity_bindings.run_mode must be 'coco_stuff_500'")
    if ib["device_type"] != "cuda":
        raise DfpPhase1bDiagnosticExecutionError("identity_bindings.device_type must be 'cuda'")
    if ib["gpu_model"] != "NVIDIA H100 80GB HBM3":
        raise DfpPhase1bDiagnosticExecutionError("identity_bindings.gpu_model must be 'NVIDIA H100 80GB HBM3'")
    if _require_exact_int(ib["gpu_count"], "identity_bindings.gpu_count") != 1:
        raise DfpPhase1bDiagnosticExecutionError("identity_bindings.gpu_count must be exactly 1 (single-GPU contract; sharding forbidden)")
    if _require_exact_int(ib["component_chunk_size"], "identity_bindings.component_chunk_size") != COMPONENT_CHUNK_SIZE:
        raise DfpPhase1bDiagnosticExecutionError(f"identity_bindings.component_chunk_size must be {COMPONENT_CHUNK_SIZE}")
    tol = _require_exact_float(ib["conservation_tolerance"], "identity_bindings.conservation_tolerance", minimum=0.0)
    if tol != CONSERVATION_TOLERANCE:
        raise DfpPhase1bDiagnosticExecutionError(f"identity_bindings.conservation_tolerance must be exactly {CONSERVATION_TOLERANCE!r}")

    pr = _require_closed(doc["progress"], _CHECKPOINT_PROGRESS_KEYS, "checkpoint.progress")
    if _require_exact_int(pr["expected_image_count"], "progress.expected_image_count") != IMAGE_COUNT:
        raise DfpPhase1bDiagnosticExecutionError(f"progress.expected_image_count must be {IMAGE_COUNT}")
    if _require_exact_int(pr["expected_window_count"], "progress.expected_window_count") != WINDOW_COUNT:
        raise DfpPhase1bDiagnosticExecutionError(f"progress.expected_window_count must be {WINDOW_COUNT}")
    next_idx = _require_exact_int(pr["next_canonical_image_index"], "progress.next_canonical_image_index", minimum=0)
    ids = pr["completed_image_ids"]
    if type(ids) is not list or not all(type(x) is str for x in ids):
        raise DfpPhase1bDiagnosticExecutionError("progress.completed_image_ids must be a list[str]")
    if len(set(ids)) != len(ids):
        raise DfpPhase1bDiagnosticExecutionError("progress.completed_image_ids contains a duplicate id")
    if len(ids) != next_idx:
        raise DfpPhase1bDiagnosticExecutionError("progress.next_canonical_image_index must equal len(completed_image_ids)")
    if next_idx > IMAGE_COUNT:
        raise DfpPhase1bDiagnosticExecutionError("progress.next_canonical_image_index exceeds expected_image_count")
    completed_windows = _require_exact_int(pr["completed_window_count"], "progress.completed_window_count", minimum=0)
    if completed_windows > WINDOW_COUNT:
        raise DfpPhase1bDiagnosticExecutionError("progress.completed_window_count exceeds expected_window_count")
    _require_exact_int(pr["total_processed_component_chunks"], "progress.total_processed_component_chunks", minimum=0)
    generation = _require_exact_int(pr["generation"], "progress.generation", minimum=0)
    complete = _require_exact_bool(pr["complete"], "progress.complete")
    final_produced = _require_exact_bool(pr["final_result_produced"], "progress.final_result_produced")
    doc_complete = _require_exact_bool(doc["complete"], "checkpoint.complete")
    doc_final = _require_exact_bool(doc["final"], "checkpoint.final")
    if complete != doc_complete:
        raise DfpPhase1bDiagnosticExecutionError("progress.complete must equal checkpoint.complete")
    if doc_final and not doc_complete:
        raise DfpPhase1bDiagnosticExecutionError("checkpoint.final=true requires checkpoint.complete=true (complete=false,final=true is not a valid combination)")
    if final_produced and not doc_final:
        raise DfpPhase1bDiagnosticExecutionError("progress.final_result_produced=true requires checkpoint.final=true")
    if doc_complete and next_idx != IMAGE_COUNT:
        raise DfpPhase1bDiagnosticExecutionError("checkpoint.complete=true requires next_canonical_image_index == expected_image_count")
    if doc_complete and completed_windows != WINDOW_COUNT:
        raise DfpPhase1bDiagnosticExecutionError("checkpoint.complete=true requires completed_window_count == expected_window_count")
    if not doc_complete and (next_idx == IMAGE_COUNT and completed_windows == WINDOW_COUNT):
        raise DfpPhase1bDiagnosticExecutionError("progress reports full completion but checkpoint.complete=false -- partial generation is invalid")

    ss = _require_closed(doc["scientific_state"], _CHECKPOINT_SCIENTIFIC_STATE_KEYS, "checkpoint.scientific_state")
    _require_string(ss["stage_a_artifact_filename"], "scientific_state.stage_a_artifact_filename")
    _require_no_unsafe_path(ss["stage_a_artifact_filename"], "scientific_state.stage_a_artifact_filename")
    _require_string(ss["stage_a_schema_name"], "scientific_state.stage_a_schema_name")
    _require_exact_int(ss["stage_a_schema_version"], "scientific_state.stage_a_schema_version", minimum=1)
    _require_exact_int(ss["stage_a_row_count"], "scientific_state.stage_a_row_count", minimum=0)
    _require_sha256(ss["stage_a_byte_sha256"], "scientific_state.stage_a_byte_sha256")
    _require_sha256(ss["scientific_collection_digest"], "scientific_state.scientific_collection_digest")

    ps = _require_closed(doc["profiling_state"], _CHECKPOINT_PROFILING_STATE_KEYS, "checkpoint.profiling_state")
    if ps["profiling_mode"] not in ("off", "detailed"):
        raise DfpPhase1bDiagnosticExecutionError("profiling_state.profiling_mode must be 'off' or 'detailed'")
    _require_string(ps["profiling_record_filename"], "profiling_state.profiling_record_filename")
    _require_no_unsafe_path(ps["profiling_record_filename"], "profiling_state.profiling_record_filename")
    _require_string(ps["profiling_schema_name"], "profiling_state.profiling_schema_name")
    _require_exact_int(ps["profiling_schema_version"], "profiling_state.profiling_schema_version", minimum=1)
    _require_exact_int(ps["profiling_record_count"], "profiling_state.profiling_record_count", minimum=0)
    _require_sha256(ps["profiling_digest"], "profiling_state.profiling_digest")

    return doc


def _require_no_unsafe_path(value: str, label: str) -> None:
    if "/" in value or "\\" in value:
        raise DfpPhase1bDiagnosticExecutionError(f"{label} must be a safe basename with no directory component, observed {value!r}")
    if value in (".", "..") or value.startswith(".."):
        raise DfpPhase1bDiagnosticExecutionError(f"{label} must not be a dot-component, observed {value!r}")
    if "\x00" in value:
        raise DfpPhase1bDiagnosticExecutionError(f"{label} must not contain a NUL byte")


def validate_resume_prefix(checkpoint_doc: Mapping[str, Any], frozen_manifest_ids: Sequence[str]) -> None:
    """Require completed_image_ids to be an EXACT prefix (same order,
    no gaps, no reordering, no duplicates, no foreign ids) of the frozen
    canonical manifest order."""
    validate_checkpoint_document(checkpoint_doc)
    completed = checkpoint_doc["progress"]["completed_image_ids"]
    if type(frozen_manifest_ids) not in (list, tuple):
        raise DfpPhase1bDiagnosticExecutionError("frozen_manifest_ids must be a list or tuple")
    if len(completed) > len(frozen_manifest_ids):
        raise DfpPhase1bDiagnosticExecutionError("completed_image_ids is longer than the frozen manifest")
    prefix = list(frozen_manifest_ids[: len(completed)])
    if completed != prefix:
        raise DfpPhase1bDiagnosticExecutionError(
            "completed_image_ids is not an exact prefix of the frozen canonical manifest order "
            "(skipped, duplicated, reordered, or cross-fed ids are all rejected here)"
        )


def validate_fresh_start(existing_checkpoint_present: bool, *, overwrite_authorized: bool = False) -> None:
    if existing_checkpoint_present and not overwrite_authorized:
        raise DfpPhase1bDiagnosticExecutionError(
            "fresh start requires no existing checkpoint/artifacts unless a separately explicit overwrite authorization is supplied"
        )


def validate_resume_eligibility(checkpoint_doc: Mapping[str, Any]) -> None:
    validate_checkpoint_document(checkpoint_doc)
    if checkpoint_doc["final"]:
        raise DfpPhase1bDiagnosticExecutionError("resume rejected: this checkpoint is already finalized")
    if not checkpoint_doc["complete"]:
        return  # ordinary incomplete resume
    if not checkpoint_doc["progress"]["final_result_produced"]:
        return  # complete-but-unfinalized -> may enter a finalize-only path


# ---------------------------------------------------------------------
# Profiling document validator.
# ---------------------------------------------------------------------

_PROFILING_TOP_KEYS = frozenset({
    "schema_name", "schema_version", "e20_identity_sha256", "profiling_mode", "complete", "final",
    "run_identity", "device_record", "software_record", "timing_method", "host_memory_method",
    "gpu_memory_method", "stage_name_enumeration", "records", "aggregates", "workload_covariates",
    "total_run_wall_time_ns", "total_scientific_processing_wall_time_ns", "checkpoint_io_wall_time_ns",
    "unattributed_overhead_ns", "runtime_budget", "warnings", "source_scientific_collection_digest",
})
_PROFILING_RECORD_KEYS = frozenset({
    "stage_name", "scope", "scope_covariate_key", "incomplete", "cpu_elapsed_ns", "cuda_elapsed_ms",
    "host_rss_bytes_before", "host_rss_bytes_after", "process_peak_rss_bytes_after_stage",
    "gpu_allocated_bytes_before", "gpu_reserved_bytes_before", "gpu_allocated_bytes_after",
    "gpu_reserved_bytes_after", "gpu_peak_allocated_bytes_during_stage", "gpu_peak_reserved_bytes_during_stage",
    "gpu_device_index", "gpu_device_model",
})
_RUNTIME_BUDGET_KEYS = frozenset({"maximum_job_wall_time_seconds", "elapsed_job_wall_time_seconds", "remaining_budget_seconds"})


def validate_profiling_record(record: Any) -> Mapping[str, Any]:
    record = _require_closed(record, _PROFILING_RECORD_KEYS, "profiling record")
    stage = record["stage_name"]
    if stage not in STAGE_SCOPES:
        raise DfpPhase1bDiagnosticExecutionError(f"profiling record: unknown stage_name {stage!r}")
    scope = record["scope"]
    if scope not in VALID_SCOPES:
        raise DfpPhase1bDiagnosticExecutionError(f"profiling record: unknown scope {scope!r}")
    if scope != STAGE_SCOPES[stage]:
        raise DfpPhase1bDiagnosticExecutionError(f"profiling record: stage {stage!r} must have scope {STAGE_SCOPES[stage]!r}, observed {scope!r}")
    _require_string(record["scope_covariate_key"], "profiling record.scope_covariate_key")
    incomplete = _require_exact_bool(record["incomplete"], "profiling record.incomplete")
    cpu_ns = record["cpu_elapsed_ns"]
    if not incomplete:
        if type(cpu_ns) is not int or cpu_ns < 0:
            raise DfpPhase1bDiagnosticExecutionError("profiling record.cpu_elapsed_ns must be a non-negative exact int for a completed record")
    elif cpu_ns is not None and (type(cpu_ns) is not int or cpu_ns < 0):
        raise DfpPhase1bDiagnosticExecutionError("profiling record.cpu_elapsed_ns must be null or a non-negative exact int")
    cuda_ms = _require_exact_float(record["cuda_elapsed_ms"], "profiling record.cuda_elapsed_ms", minimum=0.0, allow_none=True)
    for k in ("host_rss_bytes_before", "host_rss_bytes_after", "process_peak_rss_bytes_after_stage"):
        v = record[k]
        if v is not None and (type(v) is not int or v < 0):
            raise DfpPhase1bDiagnosticExecutionError(f"profiling record.{k} must be null or a non-negative exact int")
    for k in ("gpu_allocated_bytes_before", "gpu_reserved_bytes_before", "gpu_allocated_bytes_after",
              "gpu_reserved_bytes_after", "gpu_peak_allocated_bytes_during_stage", "gpu_peak_reserved_bytes_during_stage",
              "gpu_device_index"):
        v = record[k]
        if v is not None and (type(v) is not int or v < 0):
            raise DfpPhase1bDiagnosticExecutionError(f"profiling record.{k} must be null or a non-negative exact int")
    if record["gpu_device_model"] is not None:
        _require_string(record["gpu_device_model"], "profiling record.gpu_device_model")
    ga_before, gr_before = record["gpu_allocated_bytes_before"], record["gpu_reserved_bytes_before"]
    ga_after, gr_after = record["gpu_allocated_bytes_after"], record["gpu_reserved_bytes_after"]
    ga_peak, gr_peak = record["gpu_peak_allocated_bytes_during_stage"], record["gpu_peak_reserved_bytes_during_stage"]
    if ga_before is not None and gr_before is not None and ga_before > gr_before:
        raise DfpPhase1bDiagnosticExecutionError("profiling record: gpu_allocated_bytes_before must be <= gpu_reserved_bytes_before")
    if ga_after is not None and gr_after is not None and ga_after > gr_after:
        raise DfpPhase1bDiagnosticExecutionError("profiling record: gpu_allocated_bytes_after must be <= gpu_reserved_bytes_after")
    if ga_peak is not None and ga_after is not None and ga_peak < ga_after:
        raise DfpPhase1bDiagnosticExecutionError("profiling record: gpu_peak_allocated_bytes_during_stage must be >= gpu_allocated_bytes_after")
    if gr_peak is not None and gr_after is not None and gr_peak < gr_after:
        raise DfpPhase1bDiagnosticExecutionError("profiling record: gpu_peak_reserved_bytes_during_stage must be >= gpu_reserved_bytes_after")
    return record


def validate_profiling_document(doc: Any) -> Mapping[str, Any]:
    doc = _require_closed(doc, _PROFILING_TOP_KEYS, "profiling")
    if _require_string(doc["schema_name"], "profiling.schema_name") != PROFILING_SCHEMA_NAME:
        raise DfpPhase1bDiagnosticExecutionError(f"profiling.schema_name must be {PROFILING_SCHEMA_NAME!r}")
    if _require_exact_int(doc["schema_version"], "profiling.schema_version") != PROFILING_SCHEMA_VERSION:
        raise DfpPhase1bDiagnosticExecutionError(f"profiling.schema_version must be {PROFILING_SCHEMA_VERSION}")
    _require_sha256(doc["e20_identity_sha256"], "profiling.e20_identity_sha256")
    if doc["profiling_mode"] not in ("off", "detailed"):
        raise DfpPhase1bDiagnosticExecutionError("profiling.profiling_mode must be 'off' or 'detailed'")
    _require_exact_bool(doc["complete"], "profiling.complete")
    _require_exact_bool(doc["final"], "profiling.final")
    stages = doc["stage_name_enumeration"]
    if list(stages) != list(STAGE_NAME_ENUMERATION):
        raise DfpPhase1bDiagnosticExecutionError("profiling.stage_name_enumeration must exactly equal the frozen 18-stage enumeration, in order")

    records = doc["records"]
    if type(records) is not list:
        raise DfpPhase1bDiagnosticExecutionError("profiling.records must be a list")
    seen_ids: set[tuple[str, str, str]] = set()
    for i, rec in enumerate(records):
        validate_profiling_record(rec)
        ident = (rec["stage_name"], rec["scope"], rec["scope_covariate_key"])
        if ident in seen_ids:
            raise DfpPhase1bDiagnosticExecutionError(f"profiling.records[{i}]: duplicate record identity {ident}")
        seen_ids.add(ident)

    rb = _require_closed(doc["runtime_budget"], _RUNTIME_BUDGET_KEYS, "profiling.runtime_budget")
    max_s = _require_exact_float(rb["maximum_job_wall_time_seconds"], "runtime_budget.maximum_job_wall_time_seconds", minimum=0.0)
    if max_s != float(MAXIMUM_JOB_WALL_TIME_SECONDS):
        raise DfpPhase1bDiagnosticExecutionError(f"runtime_budget.maximum_job_wall_time_seconds must be {float(MAXIMUM_JOB_WALL_TIME_SECONDS)!r}")
    elapsed_s = _require_exact_float(rb["elapsed_job_wall_time_seconds"], "runtime_budget.elapsed_job_wall_time_seconds", minimum=0.0)
    remaining_s = _require_exact_float(rb["remaining_budget_seconds"], "runtime_budget.remaining_budget_seconds")
    if abs((max_s - elapsed_s) - remaining_s) > 1e-6:
        raise DfpPhase1bDiagnosticExecutionError("runtime_budget.remaining_budget_seconds must equal maximum - elapsed")

    for k in ("total_run_wall_time_ns", "total_scientific_processing_wall_time_ns", "checkpoint_io_wall_time_ns", "unattributed_overhead_ns"):
        _require_exact_int(doc[k], f"profiling.{k}", minimum=0)
    if doc["total_scientific_processing_wall_time_ns"] + doc["checkpoint_io_wall_time_ns"] + doc["unattributed_overhead_ns"] != doc["total_run_wall_time_ns"]:
        raise DfpPhase1bDiagnosticExecutionError(
            "profiling: total_scientific_processing_wall_time_ns + checkpoint_io_wall_time_ns + unattributed_overhead_ns "
            "must equal total_run_wall_time_ns exactly (negative overhead is folded in as a violation of this identity)"
        )
    if doc["unattributed_overhead_ns"] < 0:
        raise DfpPhase1bDiagnosticExecutionError("profiling.unattributed_overhead_ns must be non-negative (negative overhead is invalid)")

    _require_sha256(doc["source_scientific_collection_digest"], "profiling.source_scientific_collection_digest")
    if type(doc["warnings"]) is not list or not all(type(w) is str for w in doc["warnings"]):
        raise DfpPhase1bDiagnosticExecutionError("profiling.warnings must be a list[str]")
    for k in ("run_identity", "device_record", "software_record", "timing_method", "host_memory_method", "gpu_memory_method", "aggregates", "workload_covariates"):
        _require_mapping(doc[k], f"profiling.{k}")
    return doc


def reconstruct_aggregates_from_records(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Group accepted (non-incomplete) records by (stage_name, scope)
    and independently recompute their cpu_elapsed_ns aggregate via
    ``aggregate_numeric_records`` (as floats), to be compared against a
    profiling document's own stored aggregates."""
    groups: dict[tuple[str, str], list[float]] = {}
    for rec in records:
        if rec.get("incomplete"):
            continue
        key = (rec["stage_name"], rec["scope"])
        groups.setdefault(key, []).append(float(rec["cpu_elapsed_ns"]))
    return {key: aggregate_numeric_records(vals) for key, vals in groups.items()}


# ---------------------------------------------------------------------
# Final-result document validator.
# ---------------------------------------------------------------------

_RESULT_TOP_KEYS = frozenset({
    "schema_name", "schema_version", "e20_identity_sha256", "run_mode", "image_count", "window_count",
    "component_chunk_size", "conservation_tolerance", "scientific_artifacts", "scientific_collection_digest",
    "profiling_artifact_reference", "profiling_mode", "checkpoint_source", "finalization_state",
    "device_record", "software_record", "total_job_wall_time_seconds", "within_8_hour_allocation",
    "complete", "final", "offline_mechanistic_analysis", "deployed_inference_method", "full_5000_image_evaluation",
})
_RESULT_SCIENTIFIC_ARTIFACT_KEYS = frozenset({"filename", "schema_name", "schema_version", "row_count", "size_bytes", "sha256"})


def validate_result_document(doc: Any) -> Mapping[str, Any]:
    doc = _require_closed(doc, _RESULT_TOP_KEYS, "result")
    if _require_string(doc["schema_name"], "result.schema_name") != RESULT_SCHEMA_NAME:
        raise DfpPhase1bDiagnosticExecutionError(f"result.schema_name must be {RESULT_SCHEMA_NAME!r}")
    if _require_exact_int(doc["schema_version"], "result.schema_version") != RESULT_SCHEMA_VERSION:
        raise DfpPhase1bDiagnosticExecutionError(f"result.schema_version must be {RESULT_SCHEMA_VERSION}")
    _require_sha256(doc["e20_identity_sha256"], "result.e20_identity_sha256")
    if doc["run_mode"] != "coco_stuff_500":
        raise DfpPhase1bDiagnosticExecutionError("result.run_mode must be 'coco_stuff_500'")
    if _require_exact_int(doc["image_count"], "result.image_count") != IMAGE_COUNT:
        raise DfpPhase1bDiagnosticExecutionError(f"result.image_count must be exactly {IMAGE_COUNT}")
    if _require_exact_int(doc["window_count"], "result.window_count") != WINDOW_COUNT:
        raise DfpPhase1bDiagnosticExecutionError(f"result.window_count must be exactly {WINDOW_COUNT}")
    if _require_exact_int(doc["component_chunk_size"], "result.component_chunk_size") != COMPONENT_CHUNK_SIZE:
        raise DfpPhase1bDiagnosticExecutionError(f"result.component_chunk_size must be {COMPONENT_CHUNK_SIZE}")
    tol = _require_exact_float(doc["conservation_tolerance"], "result.conservation_tolerance", minimum=0.0)
    if tol != CONSERVATION_TOLERANCE:
        raise DfpPhase1bDiagnosticExecutionError(f"result.conservation_tolerance must be exactly {CONSERVATION_TOLERANCE!r}")

    artifacts = doc["scientific_artifacts"]
    if type(artifacts) is not list or not artifacts:
        raise DfpPhase1bDiagnosticExecutionError("result.scientific_artifacts must be a non-empty list")
    for i, art in enumerate(artifacts):
        art = _require_closed(art, _RESULT_SCIENTIFIC_ARTIFACT_KEYS, f"result.scientific_artifacts[{i}]")
        _require_string(art["filename"], f"scientific_artifacts[{i}].filename")
        _require_no_unsafe_path(art["filename"], f"scientific_artifacts[{i}].filename")
        _require_string(art["schema_name"], f"scientific_artifacts[{i}].schema_name")
        _require_exact_int(art["schema_version"], f"scientific_artifacts[{i}].schema_version", minimum=1)
        _require_exact_int(art["row_count"], f"scientific_artifacts[{i}].row_count", minimum=0)
        _require_exact_int(art["size_bytes"], f"scientific_artifacts[{i}].size_bytes", minimum=0)
        _require_sha256(art["sha256"], f"scientific_artifacts[{i}].sha256")

    _require_sha256(doc["scientific_collection_digest"], "result.scientific_collection_digest")
    _require_string(doc["profiling_artifact_reference"], "result.profiling_artifact_reference")
    _require_no_unsafe_path(doc["profiling_artifact_reference"], "result.profiling_artifact_reference")
    if doc["profiling_mode"] not in ("off", "detailed"):
        raise DfpPhase1bDiagnosticExecutionError("result.profiling_mode must be 'off' or 'detailed'")
    _require_mapping(doc["checkpoint_source"], "result.checkpoint_source")
    _require_mapping(doc["finalization_state"], "result.finalization_state")
    _require_mapping(doc["device_record"], "result.device_record")
    _require_mapping(doc["software_record"], "result.software_record")

    wall_s = _require_exact_float(doc["total_job_wall_time_seconds"], "result.total_job_wall_time_seconds", minimum=0.0)
    within_flag = _require_exact_bool(doc["within_8_hour_allocation"], "result.within_8_hour_allocation")
    expected_within = wall_s <= float(MAXIMUM_JOB_WALL_TIME_SECONDS)
    if within_flag != expected_within:
        raise DfpPhase1bDiagnosticExecutionError(
            f"result.within_8_hour_allocation ({within_flag}) disagrees with the derived "
            f"total_job_wall_time_seconds <= {MAXIMUM_JOB_WALL_TIME_SECONDS} ({expected_within})"
        )

    complete = _require_exact_bool(doc["complete"], "result.complete")
    final = _require_exact_bool(doc["final"], "result.final")
    if not complete or not final:
        raise DfpPhase1bDiagnosticExecutionError("result.complete and result.final must both be true -- no partial generation is a valid final-result document")

    if _require_exact_bool(doc["offline_mechanistic_analysis"], "result.offline_mechanistic_analysis") is not True:
        raise DfpPhase1bDiagnosticExecutionError("result.offline_mechanistic_analysis must be true")
    if _require_exact_bool(doc["deployed_inference_method"], "result.deployed_inference_method") is not False:
        raise DfpPhase1bDiagnosticExecutionError("result.deployed_inference_method must be false")
    if _require_exact_bool(doc["full_5000_image_evaluation"], "result.full_5000_image_evaluation") is not False:
        raise DfpPhase1bDiagnosticExecutionError("result.full_5000_image_evaluation must be false")
    return doc


# ---------------------------------------------------------------------
# Profiling on/off scientific-equivalence check (over two synthetic or
# real documents; never executes anything itself).
# ---------------------------------------------------------------------

_OPERATIONAL_RESULT_FIELDS = frozenset({
    "profiling_artifact_reference", "profiling_mode", "total_job_wall_time_seconds",
    "within_8_hour_allocation", "device_record", "software_record", "checkpoint_source", "finalization_state",
})


def assert_profiling_scientific_equivalence(result_off: Mapping[str, Any], result_on: Mapping[str, Any]) -> None:
    """Require two final-result documents that differ only in
    profiling_mode ('off' vs 'detailed') to carry an IDENTICAL
    scientific_collection_digest and identical scientific_artifacts,
    and to differ ONLY in explicitly operational fields."""
    validate_result_document(result_off)
    validate_result_document(result_on)
    if result_off["profiling_mode"] == result_on["profiling_mode"]:
        raise DfpPhase1bDiagnosticExecutionError("assert_profiling_scientific_equivalence requires two documents with different profiling_mode")
    if result_off["scientific_collection_digest"] != result_on["scientific_collection_digest"]:
        raise DfpPhase1bDiagnosticExecutionError("scientific_collection_digest differs between profiling_mode=off and profiling_mode=detailed -- profiling must never change scientific output")
    if result_off["scientific_artifacts"] != result_on["scientific_artifacts"]:
        raise DfpPhase1bDiagnosticExecutionError("scientific_artifacts differ between profiling modes -- profiling must never change scientific output")
    for key in result_off:
        if key in _OPERATIONAL_RESULT_FIELDS:
            continue
        if result_off[key] != result_on[key]:
            raise DfpPhase1bDiagnosticExecutionError(f"non-operational field {key!r} differs between profiling modes -- only operational fields may differ")


def assert_fresh_resume_equivalence(fresh_result: Mapping[str, Any], resumed_result: Mapping[str, Any]) -> None:
    """Require a fresh-run final-result document and a
    checkpoint-then-resume final-result document (same deterministic
    fixture) to be identical except for the operational fields."""
    validate_result_document(fresh_result)
    validate_result_document(resumed_result)
    if fresh_result["scientific_collection_digest"] != resumed_result["scientific_collection_digest"]:
        raise DfpPhase1bDiagnosticExecutionError("scientific_collection_digest differs between fresh and resumed runs")
    if fresh_result["scientific_artifacts"] != resumed_result["scientific_artifacts"]:
        raise DfpPhase1bDiagnosticExecutionError("scientific_artifacts differ between fresh and resumed runs")
    for key in fresh_result:
        if key in _OPERATIONAL_RESULT_FIELDS:
            continue
        if fresh_result[key] != resumed_result[key]:
            raise DfpPhase1bDiagnosticExecutionError(f"non-operational field {key!r} differs between fresh and resumed runs")


__all__ = [
    "CHECKPOINT_SCHEMA_NAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "COMPONENT_CHUNK_SIZE",
    "CONSERVATION_TOLERANCE",
    "DfpPhase1bDiagnosticExecutionError",
    "FORMAT_VERSION",
    "IDENTITY_RELATIVE_PATH",
    "IDENTITY_SCHEMA_NAME",
    "IDENTITY_SCHEMA_VERSION",
    "IMAGE_COUNT",
    "MAXIMUM_JOB_WALL_TIME_SECONDS",
    "PROFILING_SCHEMA_NAME",
    "PROFILING_SCHEMA_VERSION",
    "RESULT_SCHEMA_NAME",
    "RESULT_SCHEMA_VERSION",
    "STAGE_NAME_ENUMERATION",
    "STAGE_SCOPES",
    "TOP_KEYS",
    "VALID_SCOPES",
    "WINDOW_COUNT",
    "aggregate_numeric_records",
    "assert_fresh_resume_equivalence",
    "assert_profiling_scientific_equivalence",
    "load_identity",
    "nearest_rank_percentile",
    "reconstruct_aggregates_from_records",
    "repository_root",
    "validate_checkpoint_document",
    "validate_fresh_start",
    "validate_profiling_document",
    "validate_profiling_record",
    "validate_resume_eligibility",
    "validate_resume_prefix",
    "validate_result_document",
    "verify_external_artifact_bindings",
]
