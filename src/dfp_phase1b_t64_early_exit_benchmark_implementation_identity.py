"""Strict loading/validation for the DFP Phase 1B T64-early-exit
benchmark IMPLEMENTATION authority
(`evaluation_identities/e26_dfp_phase1b_t64_early_exit_benchmark_implementation.toml`,
"E26").

E26 authorizes IMPLEMENTATION ONLY of the T64_early_exit-versus-
T320_reference benchmark runner, CLI, tests, documentation, and SLURM
launcher whose scientific/operational contract is already frozen,
unmodified, by E25
(`src.dfp_phase1b_t64_early_exit_benchmark_identity`). E25 itself
explicitly authorizes no runner implementation and no real GPU
execution -- E26 closes that gap for exactly five named future files
and no others. E26 introduces no new scientific value, no new
schedule, no new threshold, and no new schema.

This module implements no runner, no CLI, no launcher, and performs no
torch/CUDA/model/dataset access -- pure TOML/filesystem/hash validation
and exact comparison only.

`KeyboardInterrupt`, `SystemExit`, and `MemoryError` always propagate
unmodified; ordinary parse/type/path/relational failures are wrapped in
`DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError`.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping


class DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(ValueError):
    """Raised on any E26 identity invariant violation. Always fail
    closed. Never raised for, and never catches, KeyboardInterrupt /
    SystemExit / MemoryError -- those always propagate unmodified."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e26_dfp_phase1b_t64_early_exit_benchmark_implementation.toml")
SCHEMA_NAME = "talk2dino-dfp-phase1b-t64-early-exit-benchmark-implementation-identity-v1"
SCHEMA_VERSION = 1

REQUIRED_ANCESTOR_COMMIT = "99da1be639932b0e34a96f2f32d23ea72e696f0a"
EXPECTED_WORKING_BRANCH = "dfp-phase1a-diagnostics"
EXPECTED_SUCCESSOR_OF = "e25-dfp-phase1b-t64-early-exit-benchmark"

AUTHORIZED_FUTURE_FILES = (
    "src/dfp_phase1b_t64_early_exit_benchmark.py",
    "diagnostics/run_dfp_phase1b_t64_early_exit_benchmark.py",
    "tests/test_dfp_phase1b_t64_early_exit_benchmark.py",
    "docs/dfp_phase1b_t64_early_exit_benchmark_run.md",
    "dfp_phase1b_t64_early_exit_benchmark_h100.sbatch",
)
RUNNER_SUBCOMMANDS = ("verify-protocol", "preflight", "run-trial", "verify-trial", "merge", "verify-result", "verify-decision")
MANIFEST_SHA256 = "be2c77466a2c1d64f0f39f17f4d073ce1de92dc526a59e47ae8c3dabb8c644ab"
WINDOW_COUNT = 24
T64_STEPS = 64
T320_STEPS = 320
T64_RECURRENCE_STEPS_TOTAL = WINDOW_COUNT * T64_STEPS
T320_RECURRENCE_STEPS_TOTAL = WINDOW_COUNT * T320_STEPS

COUNTERBALANCED_ORDER = (
    {"trial_index": 0, "variant": "T320_reference", "repeat": 0},
    {"trial_index": 1, "variant": "T64_early_exit", "repeat": 0},
    {"trial_index": 2, "variant": "T64_early_exit", "repeat": 1},
    {"trial_index": 3, "variant": "T320_reference", "repeat": 1},
    {"trial_index": 4, "variant": "T320_reference", "repeat": 2},
    {"trial_index": 5, "variant": "T64_early_exit", "repeat": 2},
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------
# Small closed-schema primitives -- the same convention every other
# E-identity loader in this repository uses.
# ---------------------------------------------------------------------


def _require_string(value: Any, label: str) -> str:
    if type(value) is not str:
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"{label} must be a string, observed {type(value).__name__}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    s = _require_string(value, label)
    if not _HEX64.match(s):
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"{label} must be lowercase 64-hex, observed {value!r}")
    return s


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"{label} must be a mapping, observed {type(value).__name__}")
    return value


def _require_closed(node: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    node = _require_mapping(node, label)
    actual = frozenset(node.keys())
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
    if missing or unknown:
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"{label} has the wrong key set -- missing={missing} unknown={unknown}")
    return node


def _require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"{label} must be an int (not bool/float), observed {type(value).__name__}: {value!r}")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"{label} must be a bool, observed {type(value).__name__}: {value!r}")
    return value


def _require_exact_value(observed: Any, expected: Any, label: str) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(
            f"{label}: frozen value drift -- expected {expected!r} ({type(expected).__name__}), observed {observed!r} ({type(observed).__name__})"
        )


def _resolve_repo_relative(root: Path, relative_path: str, *, label: str) -> Path:
    candidate = (root / relative_path).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"{label}: relative_path escapes the repository root: {relative_path!r}") from exc
    return candidate


# ---------------------------------------------------------------------
# [identity]
# ---------------------------------------------------------------------

_IDENTITY_KEYS = frozenset({
    "name", "schema_name", "schema_version", "description", "required_ancestor_commit",
    "working_branch", "successor_of", "mutates_or_version_bumps_earlier_identity",
})


def _verify_identity_block(node: Any) -> None:
    node = _require_closed(node, _IDENTITY_KEYS, "identity")
    _require_exact_value(_require_string(node["name"], "identity.name"), "e26-dfp-phase1b-t64-early-exit-benchmark-implementation", "identity.name")
    _require_exact_value(_require_string(node["schema_name"], "identity.schema_name"), SCHEMA_NAME, "identity.schema_name")
    _require_exact_value(_require_exact_int(node["schema_version"], "identity.schema_version"), SCHEMA_VERSION, "identity.schema_version")
    _require_string(node["description"], "identity.description")
    commit = _require_string(node["required_ancestor_commit"], "identity.required_ancestor_commit")
    if not _HEX40.match(commit):
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"identity.required_ancestor_commit must be 40-hex, observed {commit!r}")
    _require_exact_value(commit, REQUIRED_ANCESTOR_COMMIT, "identity.required_ancestor_commit")
    _require_exact_value(_require_string(node["working_branch"], "identity.working_branch"), EXPECTED_WORKING_BRANCH, "identity.working_branch")
    _require_exact_value(_require_string(node["successor_of"], "identity.successor_of"), EXPECTED_SUCCESSOR_OF, "identity.successor_of")
    _require_exact_value(_require_bool(node["mutates_or_version_bumps_earlier_identity"], "identity.mutates_or_version_bumps_earlier_identity"), False, "identity.mutates_or_version_bumps_earlier_identity")


# ---------------------------------------------------------------------
# [dependency_bindings]: 14 ordinary repository bindings, all
# physically contained, none symbolic.
# ---------------------------------------------------------------------

_ORDINARY_BINDING_KEYS = frozenset({"relative_path", "sha256", "binds"})
_ALL_BINDING_NAMES = frozenset({
    "e25_identity", "e25_artifacts", "e25_loader", "e25_docs", "e17_manifest", "e13_protocol",
    "matched_evaluator", "graph_module", "finite_step_regime_module", "horizon_stitching_module",
    "stitching_pullback_module", "final_rescale_module", "production_provider_module", "artifact_schema_module",
})


def _verify_dependency_bindings(root: Path, node: Any) -> None:
    node = _require_closed(node, _ALL_BINDING_NAMES, "dependency_bindings")
    for name in _ALL_BINDING_NAMES:
        binding = _require_closed(node[name], _ORDINARY_BINDING_KEYS, f"dependency_bindings.{name}")
        rel = _require_string(binding["relative_path"], f"dependency_bindings.{name}.relative_path")
        expected_hash = _require_sha256(binding["sha256"], f"dependency_bindings.{name}.sha256")
        _require_string(binding["binds"], f"dependency_bindings.{name}.binds")
        target = _resolve_repo_relative(root, rel, label=f"dependency_bindings.{name}")
        try:
            observed_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as exc:
            raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"dependency_bindings.{name}: cannot read bound file {rel}: {exc}") from exc
        if observed_hash != expected_hash:
            raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"dependency_bindings.{name}: bound file {rel} sha256 drift -- expected {expected_hash}, observed {observed_hash}")
        # No E26 file may be self-referenced by hash here (no circularity).
        if Path(rel).name in ("e26_dfp_phase1b_t64_early_exit_benchmark_implementation.toml", "dfp_phase1b_t64_early_exit_benchmark_implementation_identity.py"):
            raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"dependency_bindings.{name}: E26 must not bind its own identity/loader file (circular dependency)")


# ---------------------------------------------------------------------
# [authorized_future_files]
# ---------------------------------------------------------------------

_AUTHORIZED_FUTURE_FILES_KEYS = frozenset({
    "list", "runner_relative_path", "cli_relative_path", "tests_relative_path", "docs_relative_path",
    "launcher_relative_path", "no_other_new_production_file_authorized", "implemented_in_this_task",
})


def _verify_authorized_future_files(node: Any) -> None:
    node = _require_closed(node, _AUTHORIZED_FUTURE_FILES_KEYS, "authorized_future_files")
    _require_exact_value(tuple(node["list"]), AUTHORIZED_FUTURE_FILES, "authorized_future_files.list")
    _require_exact_value(_require_string(node["runner_relative_path"], "..runner"), AUTHORIZED_FUTURE_FILES[0], "authorized_future_files.runner_relative_path")
    _require_exact_value(_require_string(node["cli_relative_path"], "..cli"), AUTHORIZED_FUTURE_FILES[1], "authorized_future_files.cli_relative_path")
    _require_exact_value(_require_string(node["tests_relative_path"], "..tests"), AUTHORIZED_FUTURE_FILES[2], "authorized_future_files.tests_relative_path")
    _require_exact_value(_require_string(node["docs_relative_path"], "..docs"), AUTHORIZED_FUTURE_FILES[3], "authorized_future_files.docs_relative_path")
    _require_exact_value(_require_string(node["launcher_relative_path"], "..launcher"), AUTHORIZED_FUTURE_FILES[4], "authorized_future_files.launcher_relative_path")
    _require_exact_value(_require_bool(node["no_other_new_production_file_authorized"], "..no_other"), True, "authorized_future_files.no_other_new_production_file_authorized")
    _require_exact_value(_require_bool(node["implemented_in_this_task"], "..implemented"), False, "authorized_future_files.implemented_in_this_task")


# ---------------------------------------------------------------------
# [runner_subcommands]
# ---------------------------------------------------------------------

_RUNNER_SUBCOMMANDS_KEYS = frozenset({
    "list", "verify_protocol_torch_free_required", "preflight_torch_free_required",
    "verify_protocol_cuda_free_required", "preflight_cuda_free_required",
    "run_trial_may_contain_real_execution_path", "run_trial_invoked_by_this_task",
})


def _verify_runner_subcommands(node: Any) -> None:
    node = _require_closed(node, _RUNNER_SUBCOMMANDS_KEYS, "runner_subcommands")
    _require_exact_value(tuple(node["list"]), RUNNER_SUBCOMMANDS, "runner_subcommands.list")
    for key in ("verify_protocol_torch_free_required", "preflight_torch_free_required", "verify_protocol_cuda_free_required", "preflight_cuda_free_required", "run_trial_may_contain_real_execution_path"):
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"runner_subcommands.{key}")
    _require_exact_value(_require_bool(node["run_trial_invoked_by_this_task"], "..invoked"), False, "runner_subcommands.run_trial_invoked_by_this_task")


# ---------------------------------------------------------------------
# [runner_population_contract]
# ---------------------------------------------------------------------

_RUNNER_POPULATION_CONTRACT_KEYS = frozenset({
    "manifest_relative_path", "manifest_sha256", "window_count", "alternate_manifest_rejected",
    "reordered_windows_rejected", "missing_windows_rejected", "duplicated_windows_rejected",
    "extra_windows_rejected", "gt_bearing_input_rejected",
})


def _verify_runner_population_contract(node: Any) -> None:
    node = _require_closed(node, _RUNNER_POPULATION_CONTRACT_KEYS, "runner_population_contract")
    _require_exact_value(_require_string(node["manifest_relative_path"], "..path"), "evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json", "runner_population_contract.manifest_relative_path")
    _require_exact_value(_require_sha256(node["manifest_sha256"], "..sha"), MANIFEST_SHA256, "runner_population_contract.manifest_sha256")
    _require_exact_value(_require_exact_int(node["window_count"], "..count"), WINDOW_COUNT, "runner_population_contract.window_count")
    for key in ("alternate_manifest_rejected", "reordered_windows_rejected", "missing_windows_rejected", "duplicated_windows_rejected", "extra_windows_rejected", "gt_bearing_input_rejected"):
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"runner_population_contract.{key}")


# ---------------------------------------------------------------------
# [runner_construction_contract]
# ---------------------------------------------------------------------

_RUNNER_CONSTRUCTION_CONTRACT_KEYS = frozenset({
    "dataset_constructed_exactly_once_per_trial", "model_constructed_exactly_once_per_trial",
    "model_forward_count_per_window", "graph_construction_count_per_window",
    "t64_recurrence_steps", "t320_recurrence_steps", "t64_via_320_step_trajectory_then_slice_forbidden",
    "candidate_specific_recurrence_forbidden", "exact_masked_forward_component_provenance_call_forbidden",
    "trial_outputs_validated_against_e25_schemas",
})


def _verify_runner_construction_contract(node: Any) -> None:
    node = _require_closed(node, _RUNNER_CONSTRUCTION_CONTRACT_KEYS, "runner_construction_contract")
    for key in ("dataset_constructed_exactly_once_per_trial", "model_constructed_exactly_once_per_trial", "t64_via_320_step_trajectory_then_slice_forbidden", "candidate_specific_recurrence_forbidden", "exact_masked_forward_component_provenance_call_forbidden", "trial_outputs_validated_against_e25_schemas"):
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"runner_construction_contract.{key}")
    _require_exact_value(_require_exact_int(node["model_forward_count_per_window"], "..fwd"), 1, "runner_construction_contract.model_forward_count_per_window")
    _require_exact_value(_require_exact_int(node["graph_construction_count_per_window"], "..graph"), 1, "runner_construction_contract.graph_construction_count_per_window")
    _require_exact_value(_require_exact_int(node["t64_recurrence_steps"], "..t64"), T64_STEPS, "runner_construction_contract.t64_recurrence_steps")
    _require_exact_value(_require_exact_int(node["t320_recurrence_steps"], "..t320"), T320_STEPS, "runner_construction_contract.t320_recurrence_steps")


# ---------------------------------------------------------------------
# [process_isolation]
# ---------------------------------------------------------------------

_PROCESS_ISOLATION_KEYS = frozenset({
    "fresh_process_per_trial_required", "warmup_artifacts_validated", "warmup_excluded_from_summaries",
    "launcher_fails_immediately_on_any_trial_or_verification_failure", "no_partial_merged_result_authorized",
    "warmup", "measured_order",
})
_ORDERED_TRIAL_ENTRY_KEYS = frozenset({"trial_index", "variant", "repeat"})


def _verify_process_isolation(node: Any) -> None:
    node = _require_closed(node, _PROCESS_ISOLATION_KEYS, "process_isolation")
    for key in ("fresh_process_per_trial_required", "warmup_artifacts_validated", "warmup_excluded_from_summaries", "launcher_fails_immediately_on_any_trial_or_verification_failure", "no_partial_merged_result_authorized"):
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"process_isolation.{key}")

    warmup = _require_closed(node["warmup"], frozenset({"list"}), "process_isolation.warmup")
    _require_exact_value(list(warmup["list"]), ["T64_early_exit", "T320_reference"], "process_isolation.warmup.list")

    order = _require_closed(node["measured_order"], frozenset({"ordered_trial_list"}), "process_isolation.measured_order")
    ordered = order["ordered_trial_list"]
    if not isinstance(ordered, list) or len(ordered) != 6:
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError("process_isolation.measured_order.ordered_trial_list must have exactly 6 entries")
    for position, (entry, expected) in enumerate(zip(ordered, COUNTERBALANCED_ORDER)):
        entry = _require_closed(entry, _ORDERED_TRIAL_ENTRY_KEYS, f"process_isolation.measured_order.ordered_trial_list[{position}]")
        if (
            _require_exact_int(entry["trial_index"], f"..[{position}].trial_index") != expected["trial_index"]
            or _require_string(entry["variant"], f"..[{position}].variant") != expected["variant"]
            or _require_exact_int(entry["repeat"], f"..[{position}].repeat") != expected["repeat"]
        ):
            raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"process_isolation.measured_order.ordered_trial_list[{position}] disagrees with the frozen required schedule")


# ---------------------------------------------------------------------
# [launcher_resource_contract]
# ---------------------------------------------------------------------

_LAUNCHER_RESOURCE_CONTRACT_KEYS = frozenset({
    "gpu_count", "gpu_model", "cpus_per_task", "host_memory", "max_wall_time", "account",
    "self_submission_forbidden", "literal_sbatch_invocation_forbidden", "required_environment_variables",
    "preflight_before_any_model_or_cuda_operation_required", "fresh_output_destination_required",
    "overwrite_authorized", "force_mode_authorized", "verify_each_trial_immediately_after_creation_required",
    "merge_exactly_six_measured_trials_plus_warmup_evidence", "verify_final_result_and_decision_after_installation_required",
    "submitted_by_this_task",
})


def _verify_launcher_resource_contract(node: Any) -> None:
    node = _require_closed(node, _LAUNCHER_RESOURCE_CONTRACT_KEYS, "launcher_resource_contract")
    _require_exact_value(_require_exact_int(node["gpu_count"], "..gpu"), 1, "launcher_resource_contract.gpu_count")
    _require_exact_value(_require_string(node["gpu_model"], "..model"), "h100_80gb", "launcher_resource_contract.gpu_model")
    _require_exact_value(_require_exact_int(node["cpus_per_task"], "..cpu"), 8, "launcher_resource_contract.cpus_per_task")
    _require_exact_value(_require_string(node["host_memory"], "..mem"), "80G", "launcher_resource_contract.host_memory")
    _require_exact_value(_require_string(node["max_wall_time"], "..wall"), "02:00:00", "launcher_resource_contract.max_wall_time")
    _require_exact_value(_require_string(node["account"], "..account"), "rrg-yangw", "launcher_resource_contract.account")
    for key in ("self_submission_forbidden", "literal_sbatch_invocation_forbidden", "preflight_before_any_model_or_cuda_operation_required", "fresh_output_destination_required", "verify_each_trial_immediately_after_creation_required", "merge_exactly_six_measured_trials_plus_warmup_evidence", "verify_final_result_and_decision_after_installation_required"):
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"launcher_resource_contract.{key}")
    for key in ("overwrite_authorized", "force_mode_authorized", "submitted_by_this_task"):
        _require_exact_value(_require_bool(node[key], f"..{key}"), False, f"launcher_resource_contract.{key}")
    envs = node["required_environment_variables"]
    if not isinstance(envs, list) or set(envs) != {"repository_root", "venv_activate_path", "dataset_root", "output_root"}:
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError("launcher_resource_contract.required_environment_variables disagrees with the frozen required set")


# ---------------------------------------------------------------------
# [timing_implementation_contract]
# ---------------------------------------------------------------------

_TIMING_IMPLEMENTATION_CONTRACT_KEYS = frozenset({
    "fields", "accounting_identity_required", "primary_timing_field", "clock_source",
    "synchronize_cuda_before_measured_loop_required", "synchronize_cuda_after_measured_loop_required",
    "process_windows_in_canonical_order_required", "stage_timing_boundaries_identical_across_variants_required",
    "cuda_events_may_measure_gpu_stage_elapsed_time", "stage_totals_must_not_replace_primary_wall_time",
})
_REQUIRED_TIMING_FIELDS = (
    "process_total_wall_ns", "preflight_wall_ns", "dataset_construction_wall_ns",
    "model_construction_and_checkpoint_loading_wall_ns", "measured_window_loop_wall_ns",
    "artifact_validation_and_write_wall_ns", "unattributed_overhead_ns",
)


def _verify_timing_implementation_contract(node: Any) -> None:
    node = _require_closed(node, _TIMING_IMPLEMENTATION_CONTRACT_KEYS, "timing_implementation_contract")
    _require_exact_value(tuple(node["fields"]), _REQUIRED_TIMING_FIELDS, "timing_implementation_contract.fields")
    _require_exact_value(_require_string(node["primary_timing_field"], "..primary"), "measured_window_loop_wall_ns", "timing_implementation_contract.primary_timing_field")
    _require_exact_value(_require_string(node["clock_source"], "..clock"), "time.perf_counter_ns", "timing_implementation_contract.clock_source")
    for key in ("accounting_identity_required", "synchronize_cuda_before_measured_loop_required", "synchronize_cuda_after_measured_loop_required", "process_windows_in_canonical_order_required", "stage_timing_boundaries_identical_across_variants_required", "cuda_events_may_measure_gpu_stage_elapsed_time", "stage_totals_must_not_replace_primary_wall_time"):
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"timing_implementation_contract.{key}")


# ---------------------------------------------------------------------
# [telemetry_implementation_contract]
# ---------------------------------------------------------------------

_TELEMETRY_IMPLEMENTATION_CONTRACT_KEYS = frozenset({
    "fields", "allocated_le_reserved_le_total_required", "cuda_peak_stats_reset_immediately_before_measured_loop_required",
    "missing_or_malformed_cuda_telemetry_fails_closed",
})
_REQUIRED_TELEMETRY_FIELDS = ("host_peak_rss_bytes", "gpu_peak_memory_allocated_bytes", "gpu_peak_memory_reserved_bytes", "gpu_total_memory_bytes")


def _verify_telemetry_implementation_contract(node: Any) -> None:
    node = _require_closed(node, _TELEMETRY_IMPLEMENTATION_CONTRACT_KEYS, "telemetry_implementation_contract")
    _require_exact_value(tuple(node["fields"]), _REQUIRED_TELEMETRY_FIELDS, "telemetry_implementation_contract.fields")
    for key in ("allocated_le_reserved_le_total_required", "cuda_peak_stats_reset_immediately_before_measured_loop_required", "missing_or_malformed_cuda_telemetry_fails_closed"):
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"telemetry_implementation_contract.{key}")


# ---------------------------------------------------------------------
# [output_count_verification_contract]
# ---------------------------------------------------------------------

_OUTPUT_COUNT_VERIFICATION_CONTRACT_KEYS = frozenset({
    "model_forward_count_required_value", "graph_construction_count_required_value",
    "t64_recurrence_steps_executed_required_value", "t320_recurrence_steps_executed_required_value",
    "per_window_digest_deterministic_within_variant_required", "collection_digest_deterministic_within_variant_required",
    "cross_variant_digest_equality_required", "all_outputs_finite_required", "profiling_must_not_affect_scientific_digest",
})


def _verify_output_count_verification_contract(node: Any) -> None:
    node = _require_closed(node, _OUTPUT_COUNT_VERIFICATION_CONTRACT_KEYS, "output_count_verification_contract")
    _require_exact_value(_require_exact_int(node["model_forward_count_required_value"], "..fwd"), WINDOW_COUNT, "output_count_verification_contract.model_forward_count_required_value")
    _require_exact_value(_require_exact_int(node["graph_construction_count_required_value"], "..graph"), WINDOW_COUNT, "output_count_verification_contract.graph_construction_count_required_value")
    _require_exact_value(_require_exact_int(node["t64_recurrence_steps_executed_required_value"], "..t64steps"), T64_RECURRENCE_STEPS_TOTAL, "output_count_verification_contract.t64_recurrence_steps_executed_required_value")
    _require_exact_value(_require_exact_int(node["t320_recurrence_steps_executed_required_value"], "..t320steps"), T320_RECURRENCE_STEPS_TOTAL, "output_count_verification_contract.t320_recurrence_steps_executed_required_value")
    for key in ("per_window_digest_deterministic_within_variant_required", "collection_digest_deterministic_within_variant_required", "all_outputs_finite_required", "profiling_must_not_affect_scientific_digest"):
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"output_count_verification_contract.{key}")
    _require_exact_value(_require_bool(node["cross_variant_digest_equality_required"], "..cross"), False, "output_count_verification_contract.cross_variant_digest_equality_required")


# ---------------------------------------------------------------------
# [implementation_test_requirements]
# ---------------------------------------------------------------------

_IMPLEMENTATION_TEST_REQUIREMENTS_KEYS = frozenset({
    "list", "synthetic_cpu_fixtures_authorized", "real_dataset_access_in_tests_forbidden",
    "real_model_access_in_tests_forbidden", "real_weights_access_in_tests_forbidden", "cuda_in_tests_forbidden",
})
_REQUIRED_IMPLEMENTATION_TEST_LIST = (
    "genuine_t64_early_exit", "step_64_equivalence_to_controlled_320_step_reference_prefix", "step_65_unreachable_in_t64",
    "exact_counts", "exact_schedule", "fresh_process_orchestration", "deterministic_outputs",
    "within_variant_digest_equality", "cross_variant_equality_not_required", "timing_accounting_validation",
    "memory_validation", "strict_artifact_schemas", "atomic_installation", "no_overwrite",
    "public_post_install_verification", "clean_failure_on_malformed_inputs", "fatal_exception_propagation",
    "no_gt_path", "no_confirmation_or_full_5000_path", "no_exact_masked_forward_dependency",
)


def _verify_implementation_test_requirements(node: Any) -> None:
    node = _require_closed(node, _IMPLEMENTATION_TEST_REQUIREMENTS_KEYS, "implementation_test_requirements")
    _require_exact_value(tuple(node["list"]), _REQUIRED_IMPLEMENTATION_TEST_LIST, "implementation_test_requirements.list")
    for key in ("synthetic_cpu_fixtures_authorized",):
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"implementation_test_requirements.{key}")
    for key in ("real_dataset_access_in_tests_forbidden", "real_model_access_in_tests_forbidden", "real_weights_access_in_tests_forbidden", "cuda_in_tests_forbidden"):
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"implementation_test_requirements.{key}")


# ---------------------------------------------------------------------
# [authorization_boundary]
# ---------------------------------------------------------------------

_AUTHORIZATION_BOUNDARY_TRUE_KEYS = frozenset({
    "runner_implementation_authorized", "cli_implementation_authorized", "launcher_implementation_authorized",
    "cpu_only_synthetic_testing_authorized",
})
_AUTHORIZATION_BOUNDARY_FALSE_KEYS = frozenset({
    "real_dataset_access_authorized", "real_model_construction_authorized", "real_weight_loading_authorized",
    "cuda_initialization_authorized", "real_gpu_trial_authorized", "slurm_submission_authorized",
    "confirmation_execution_authorized", "full_5000_execution_authorized", "gt_access_authorized",
    "scientific_kernel_modification_authorized", "e25_modification_authorized", "exact_masked_forward_provenance_authorized",
})
_AUTHORIZATION_BOUNDARY_KEYS = _AUTHORIZATION_BOUNDARY_TRUE_KEYS | _AUTHORIZATION_BOUNDARY_FALSE_KEYS


def _verify_authorization_boundary(node: Any) -> None:
    node = _require_closed(node, _AUTHORIZATION_BOUNDARY_KEYS, "authorization_boundary")
    for key in _AUTHORIZATION_BOUNDARY_TRUE_KEYS:
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"authorization_boundary.{key}")
    for key in _AUTHORIZATION_BOUNDARY_FALSE_KEYS:
        _require_exact_value(_require_bool(node[key], f"..{key}"), False, f"authorization_boundary.{key}")


# ---------------------------------------------------------------------
# [deferred_next_steps], [prohibited]
# ---------------------------------------------------------------------

_REQUIRED_DEFERRED_NEXT_STEPS = (
    "implement_src_dfp_phase1b_t64_early_exit_benchmark_py",
    "implement_diagnostics_run_dfp_phase1b_t64_early_exit_benchmark_py",
    "implement_tests_test_dfp_phase1b_t64_early_exit_benchmark_py",
    "implement_docs_dfp_phase1b_t64_early_exit_benchmark_run_md",
    "implement_dfp_phase1b_t64_early_exit_benchmark_h100_sbatch",
    "independently_verify_e26",
    "commit_e26",
    "implement_the_five_authorized_files",
    "independently_verify_the_implementation",
    "perform_committed_head_production_readiness_verification",
    "user_manually_submits_the_h100_job",
    "independently_verify_benchmark_artifacts",
    "only_if_outcome_is_proceed_to_freeze_t64_confirmation_protocol_create_a_separate_confirmation_authority",
)


def _verify_deferred_next_steps(node: Any) -> None:
    node = _require_closed(node, frozenset({"list"}), "deferred_next_steps")
    if tuple(node["list"]) != _REQUIRED_DEFERRED_NEXT_STEPS:
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError("deferred_next_steps.list disagrees with the frozen required order")


_REQUIRED_PROHIBITED = (
    "real_gpu_benchmark_execution_in_this_task", "gpu_use_in_this_task", "slurm_submission_in_this_task",
    "dataset_access_in_this_task", "weight_loading_in_this_task", "cuda_initialization_in_this_task",
    "implementing_the_runner_cli_launcher_or_runner_tests_in_this_task", "confirmation_evaluation",
    "full_5000_image_evaluation", "candidate_selection_as_a_final_paper_method", "modifying_e13_through_e25_identities",
    "modifying_the_e17_manifest", "modifying_the_balp_scientific_kernel_or_production_runner",
    "modifying_graph_recurrence_stitching_or_pullback_mathematics", "modifying_the_completed_development_result_checkpoint_or_log",
    "modifying_the_e24_closure", "treating_a_320_step_snapshot_as_a_genuine_t64_early_exit",
    "candidate_specific_recurrence", "exact_masked_forward_provenance", "adaptive_or_relaxed_speedup_threshold",
    "new_scientific_value_schedule_threshold_or_schema", "authorizing_a_new_production_file_beyond_the_five_listed",
)


def _verify_prohibited(node: Any) -> None:
    node = _require_closed(node, frozenset({"list"}), "prohibited")
    if tuple(node["list"]) != _REQUIRED_PROHIBITED:
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError("prohibited.list disagrees with the frozen required list")


def verify_repository_gate(repo_root: Path, repository_commit: str) -> None:
    commit = _require_string(repository_commit, "repository_commit")
    if not _HEX40.match(commit):
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"repository_commit must be a 40-hex commit, observed {commit!r}")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", REQUIRED_ANCESTOR_COMMIT, commit],
            capture_output=True, text=True,
        )
    except OSError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"cannot invoke git for ancestor verification: {exc}") from exc
    if result.returncode != 0:
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"required_ancestor_commit {REQUIRED_ANCESTOR_COMMIT} is not an ancestor of (or equal to) {commit}")


# ---------------------------------------------------------------------
# Top-level load/verify.
# ---------------------------------------------------------------------

_TOP_LEVEL_KEYS = frozenset({
    "format_version", "identity", "dependency_bindings", "authorized_future_files", "runner_subcommands",
    "runner_population_contract", "runner_construction_contract", "process_isolation", "launcher_resource_contract",
    "timing_implementation_contract", "telemetry_implementation_contract", "output_count_verification_contract",
    "implementation_test_requirements", "authorization_boundary", "deferred_next_steps", "prohibited",
})


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load, closed-schema-validate, exact-value-validate, and
    relationally hash-verify the E26 implementation-authority identity.
    Never mutates or re-derives any bound dependency -- only
    re-verifies SHA-256/content against current bytes. Performs no
    torch/CUDA/model/dataset access."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with open(source, "rb") as fh:
            document = tomllib.load(fh)
    except OSError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"cannot read E26 identity {source}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError(f"E26 identity {source} is not valid TOML: {exc}") from exc

    _require_closed(document, _TOP_LEVEL_KEYS, "<top level>")
    _require_exact_value(
        _require_string(document["format_version"], "format_version"),
        "talk2dino-dfp-phase1b-t64-early-exit-benchmark-implementation-identity-v1", "format_version",
    )
    _verify_identity_block(document["identity"])
    _verify_dependency_bindings(root, document["dependency_bindings"])
    _verify_authorized_future_files(document["authorized_future_files"])
    _verify_runner_subcommands(document["runner_subcommands"])
    _verify_runner_population_contract(document["runner_population_contract"])
    _verify_runner_construction_contract(document["runner_construction_contract"])
    _verify_process_isolation(document["process_isolation"])
    _verify_launcher_resource_contract(document["launcher_resource_contract"])
    _verify_timing_implementation_contract(document["timing_implementation_contract"])
    _verify_telemetry_implementation_contract(document["telemetry_implementation_contract"])
    _verify_output_count_verification_contract(document["output_count_verification_contract"])
    _verify_implementation_test_requirements(document["implementation_test_requirements"])
    _verify_authorization_boundary(document["authorization_boundary"])
    _verify_deferred_next_steps(document["deferred_next_steps"])
    _verify_prohibited(document["prohibited"])

    return document
