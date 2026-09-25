"""Torch-free loader/validator for research_specs/semantic_evidence_discovery_execution_v1.toml.

Standard-library only. Never imports torch, torchvision, mmcv, mmseg, PIL,
or any GPU/CUDA API. Never opens a dataset image or a GT/annotation file.
Every dependency hash is re-verified against `git show <required_head>:<path>`
at call time -- this module never trusts a hash recorded in the TOML alone,
and never trusts this module's own prior output as an oracle for itself.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import tomllib
from pathlib import Path
from typing import Any


class DiscoveryExecutionContractError(Exception):
    pass


def _reraise_if_fatal(exc: BaseException) -> None:
    if isinstance(exc, (KeyboardInterrupt, SystemExit, MemoryError)):
        raise exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiscoveryExecutionContractError(message)


def _require_str(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value != "", f"{label} must be a non-empty string")
    return value


def _require_sha256_hex(value: Any, label: str) -> str:
    s = _require_str(value, label)
    _require(len(s) == 64 and all(c in "0123456789abcdef" for c in s), f"{label} must be lowercase 64-hex sha256")
    return s


def _require_git_sha_hex(value: Any, label: str) -> str:
    s = _require_str(value, label)
    _require(len(s) == 40 and all(c in "0123456789abcdef" for c in s), f"{label} must be a lowercase 40-hex git object id")
    return s


def _require_bool(value: Any, label: str) -> bool:
    _require(isinstance(value, bool), f"{label} must be a real bool, got {type(value).__name__}")
    return value


def _require_int_not_bool(value: Any, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be a real int, got {type(value).__name__}")
    return value


def _require_list(value: Any, label: str) -> list:
    _require(isinstance(value, list), f"{label} must be a list")
    return value


def _require_dict(value: Any, label: str) -> dict:
    _require(isinstance(value, dict), f"{label} must be a table")
    return value


# ---------------------------------------------------------------------------
# Generic recursive closed-schema validation
#
# SCHEMA nodes:
#   frozenset(...)            -- a table; exact key set required at this level
#   (frozenset(...), {sub})   -- a table with the given exact key set, where
#                                 `sub` maps some of those keys to a nested
#                                 SCHEMA node describing that key's own value
#   ("list_of_table", frozenset(...)) -- a list; every element must be a
#                                 table with exactly this key set
#   None                      -- a leaf; validated elsewhere (value/type
#                                 checks), not further closed here
# ---------------------------------------------------------------------------

def _validate_closed_recursive(obj: Any, schema: Any, path: str) -> None:
    if schema is None:
        return
    if isinstance(schema, tuple) and schema and schema[0] == "list_of_table":
        _, entry_keys = schema
        _require(isinstance(obj, list), f"{path} must be a list")
        for i, entry in enumerate(obj):
            _require(isinstance(entry, dict), f"{path}[{i}] must be a table")
            actual = frozenset(entry.keys())
            _require(
                actual == entry_keys,
                f"{path}[{i}] key mismatch: missing={sorted(entry_keys - actual)}, extra={sorted(actual - entry_keys)}",
            )
        return
    if isinstance(schema, tuple):
        expected_keys, sub = schema
    else:
        expected_keys, sub = schema, {}
    _require(isinstance(obj, dict), f"{path} must be a table")
    actual_keys = frozenset(obj.keys())
    _require(
        actual_keys == expected_keys,
        f"{path} key mismatch: missing={sorted(expected_keys - actual_keys)}, extra={sorted(actual_keys - expected_keys)}",
    )
    for key, child_schema in sub.items():
        _validate_closed_recursive(obj[key], child_schema, f"{path}.{key}")


_DEPENDENCY_RECORD_KEYS = frozenset({"role", "path", "sha256", "must_be_committed_at_required_head", "reason"})
_ARTIFACT_SCHEMA_ENTRY = frozenset({"schema_name", "schema_version", "required_fields"})

_TOML_SCHEMA: dict = {
    "identity": frozenset({"created_at_utc", "purpose", "schema_name", "schema_version", "task"}),
    "prerequisite": frozenset({
        "required_branch", "required_head", "required_head_subject",
        "required_head_tree", "working_tree_must_be_clean_at_freeze_time",
    }),
    "dependency": ("list_of_table", _DEPENDENCY_RECORD_KEYS),
    "smoke_prerequisite": (
        frozenset({
            "execution_head", "execution_tree", "job_id", "outcome_required",
            "path_disclosure_is_operational_not_scientific_identity",
            "public_verification_required", "result_path", "result_schema_name",
            "result_schema_version", "result_sha256", "scientific_facts_required",
            "scientific_smoke_status", "smoke_profiling_defects",
            "smoke_profiling_defects_inherited_by_this_authority",
            "smoke_profiling_defects_repair_authorized_by_this_task",
            "smoke_profiling_status", "smoke_rerun_authorized",
        }),
        {
            "scientific_facts_required": frozenset({
                "counts_match", "gt_accessed", "interventions_executed", "m", "n",
                "non_finite_counts_all_zero", "r", "t0_bitwise_preservation_all_windows",
                "t0_t320_repeat_determinism",
            }),
        },
    ),
    "population": (
        frozenset({
            "canonical_source_digest", "digest_definition", "discovery_count",
            "discovery_digest", "discovery_holdout_overlap", "discovery_rank_range",
            "execution_population", "holdout_protection", "locked_holdout_count",
            "locked_holdout_digest", "locked_holdout_rank_range", "namespace",
            "selected_record_digest", "smoke_images_are_discovery_rank_0_and_1",
            "smoke_images_confirm_discovery_split_is_real_model_executable",
            "source_manifest_role", "total_selected_count",
        }),
        {
            "holdout_protection": frozenset({
                "future_runner_must_accept_only_discovery_ids",
                "future_runner_prohibited_selectable_populations",
                "holdout_ids_bound_by_digest_only_in_this_authority",
                "holdout_ids_not_enumerated_in_this_document",
            }),
        },
    ),
    "scientific_inheritance": (
        frozenset({
            "anchor_selection", "bootstrap", "controls", "e3_geometry", "gaussian_blur",
            "gt_targets", "inheritance_rule", "interventions", "model_fits",
            "predecessor", "scores", "statistics",
        }),
        {
            "e3_geometry": frozenset({
                "border_excluded_patch_rings", "crop_size_pixels", "eligible_grid_range",
                "pamr_active", "patch_grid_side", "patch_size_pixels", "predecessor_table",
                "resize_dim_pixels", "stride_pixels", "tie_behavior",
            }),
            "anchor_selection": frozenset({
                "anchors_per_stratum", "gt_or_correctness_forbidden_in_selection",
                "predecessor_table", "queries_per_anchor", "strata", "target_anchors_per_window",
            }),
            "interventions": frozenset({
                "core_patch_count", "core_pixel_size", "predecessor_table", "view_count", "views",
            }),
            "gaussian_blur": frozenset({
                "deterministic", "kernel_size", "padding_mode", "predecessor_table", "sigma", "source",
            }),
            "scores": frozenset({
                "formula_names", "input_failure_prohibited_actions", "input_finiteness_required",
                "input_range_required", "intervention_reliability_range", "output_finiteness_required",
                "output_range_required", "predecessor_table", "validity_predecessor_table",
            }),
            "controls": frozenset({
                "control_7_status", "control_7_used_by_any_task", "count", "ids", "predecessor_table",
            }),
            "statistics": frozenset({
                "C", "class_weight", "estimator", "hyperparameter_search", "max_iter",
                "model_settings_predecessor_table", "penalty", "predecessor_table",
                "random_state", "solver", "standardization", "task_identifiers",
            }),
            "model_fits": frozenset({"count", "evaluate_partition", "fit_partition", "ids", "predecessor_table"}),
            "bootstrap": frozenset({
                "confidence_level", "interval_type", "min_valid_replicate_fraction", "paired",
                "predecessor_table", "refit_inside_replicates", "replicates", "resample_unit", "seed",
            }),
            "gt_targets": frozenset({
                "ambiguous_label", "ignore_gt_value", "min_majority_share", "min_valid_footprint_fraction",
                "predecessor_table", "propagation_categories", "propagation_outcome_predecessor_table",
            }),
        },
    ),
    "t0_t320_target_authority": (
        frozenset({
            "frozen_constants", "modification_boundary", "t0_predecessor_table", "t0_reuses",
            "t0_status", "t320_required_successor_action", "t320_required_successor_action_fulfilled_by",
            "t320_status_in_feasibility_contract", "t320_status_supersedes_feasibility_contract_deferred_flag",
            "t320_status_supersedes_note", "t320_status_under_this_authority",
        }),
        {
            "frozen_constants": frozenset({
                "affinity_power", "alpha", "contiguous_grid_before_sigmoid_interpolation",
                "directed_row_stochastic_graph", "early_stopping", "final_align_corners",
                "final_interpolation_mode", "internal_align_corners", "internal_interpolation_mode",
                "internal_sigmoid_count", "k", "prediction_rule", "recurrence_input",
                "stable_neighbor_tie_breaking", "steps",
            }),
            "modification_boundary": frozenset({
                "e3_adapter_modification_authorized", "future_runner_must_reuse_directly",
                "reimplementation_of_graph_recurrence_interpolation_stitching_finalization_authorized",
                "t320_implementation_modification_authorized",
            }),
        },
    ),
    "discovery_unit_structure": (
        frozenset({
            "anchors_per_accepted_window", "intervention_view_count", "intervention_view_ids_in_order",
            "logistic_fits", "propagation_outcome_task", "queries_per_accepted_anchor", "query_correctness_task",
        }),
        {
            "query_correctness_task": frozenset({
                "computable_from_discovery_features", "eligibility", "model_fit_ids",
                "predecessor_table", "rows_per_accepted_anchor", "unit_of_analysis",
            }),
            "propagation_outcome_task": frozenset({
                "computable_from_discovery_features", "model_fit_ids", "no_query_2_leakage_required",
                "note", "predecessor_table", "query_used", "rows_per_eligible_anchor", "unit_of_analysis",
            }),
            "logistic_fits": frozenset({
                "count", "evaluate_partition_for_all_four", "evaluate_partition_out_of_scope_for_this_authority",
                "fit_ids", "fit_partition_for_all_four", "non_finite_rejection_policy",
                "refit_inside_bootstrap_authorized", "rejected_or_invalid_anchor_handling",
            }),
        },
    ),
    "gt_isolation_boundary": (
        frozenset({
            "digest_separation", "gt_affects_intervention_reliability", "gt_affects_selection",
            "phase_a_gt_free", "phase_b_gt_dependent", "predecessor_table",
        }),
        {
            "phase_a_gt_free": frozenset({"gt_accessible_during_phase_a", "steps"}),
            "phase_b_gt_dependent": frozenset({"gt_may_open_only_after_phase_a_complete", "steps"}),
            "digest_separation": frozenset({
                "gt_dependent_label_digest_required", "gt_free_feature_digest_required",
                "gt_must_not_affect", "gt_mutation_must_not_change_gt_free_feature_digest",
            }),
        },
    ),
    "one_forward_reuse_contract": frozenset({
        "accidental_second_backbone_forward_prohibited_for", "additional_model_evaluations_for_interventions",
        "balp_path_prohibited", "candidate_specific_t320_recurrence_prohibited", "cgls_rwr_path_prohibited",
        "intervention_processing_reuses_frozen_feature_scoring_boundary", "invented_caching_policy_prohibited",
        "masked_forward_component_provenance_path_prohibited", "one_e3_snapshot_per_canonical_source_window",
        "one_real_e3_forward_per_canonical_source_window", "pamr_explicitly_included_by_predecessor",
        "pamr_path_prohibited_unless_explicit", "t0_and_t320_reuse_that_snapshot",
    }),
    "artifact_contract": (
        frozenset({
            "checkpoint_metadata", "closed_schema_requirements", "decision_document",
            "detailed_runtime_profiling_required_for_discovery_decision", "final_result",
            "gt_dependent_labels", "gt_free_features",
            "independently_verifiable_without_importing_the_real_model", "job_elapsed_time_status",
            "physical_representation", "smoke_profiling_included_as_scientific_field",
        }),
        {
            "checkpoint_metadata": _ARTIFACT_SCHEMA_ENTRY,
            "gt_free_features": _ARTIFACT_SCHEMA_ENTRY,
            "gt_dependent_labels": _ARTIFACT_SCHEMA_ENTRY,
            "final_result": _ARTIFACT_SCHEMA_ENTRY,
            "decision_document": _ARTIFACT_SCHEMA_ENTRY,
            "closed_schema_requirements": frozenset({
                "boolean_as_integer_rejected", "duplicate_ids_rejected", "missing_keys_rejected",
                "nan_infinity_rejected", "nested_mappings_closed", "strict_scalar_container_types",
                "unknown_keys_rejected", "unknown_model_metric_gate_id_rejected", "wrong_population_order_rejected",
            }),
        },
    ),
    "lifecycle_contract": (
        frozenset({
            "atomic_installation_method", "checkpoint_arrays_hash_bound_to_checkpoint_metadata",
            "completed_id_prefix_validation_exact", "determinism_identity", "force_mode_prohibited",
            "modes", "no_orphaned_temporary_files", "no_success_before_post_install_verification",
            "overwrite_prohibited", "post_install_reopen_required",
            "previous_valid_generation_preserved_on_failure", "public_reverification_required",
            "resume_must_not_alter_scientific_content", "scope", "staged_reopen_and_validation_required",
            "staged_write_required", "zero_reprocessing_of_completed_images",
        }),
        {
            "determinism_identity": frozenset({"requirement"}),
            "scope": frozenset({"elaborate_generation_system_beyond_48_image_run_prohibited"}),
        },
    ),
    "statistical_execution_contract": frozenset({
        "adaptive_thresholds_prohibited", "degenerate_discovery_sample_behavior",
        "holdout_inspection_during_discovery_fitting_or_evaluation_prohibited",
        "min_eligible_query_correctness_rows_discovery", "minimum_data_requirements_predecessor_table",
        "predecessor_defines_tuning_on_discovery_outcomes", "predecessor_table",
        "tuning_on_discovery_outcomes_prohibited_unless_predecessor_defines_it",
    }),
    "gate_contract": (
        frozenset({
            "A", "B", "C", "D", "E", "inconclusive_handling", "outcome",
            "post_hoc_threshold_changes_authorized", "predecessor_table", "proceed_value",
            "stop_value", "use_predecessor_outcome_strings_exactly",
        }),
        {
            "A": frozenset({
                "computable_under_this_contract", "computed_over", "max_gt_dependent_intervention_records",
                "max_non_finite_scores", "min_structural_validity_fraction", "name", "predecessor_table",
                "reliability_range",
            }),
            "B": frozenset({
                "computable_note", "computable_under_this_contract", "min_auroc_improvement_holdout",
                "name", "paired_bootstrap_95_lower_bound_required", "predecessor_table", "status",
            }),
            "C": frozenset({
                "computable_note", "computable_under_this_contract", "min_auroc_improvement_holdout",
                "name", "paired_bootstrap_95_lower_bound_required", "predecessor_table", "status",
            }),
            "D": frozenset({
                "computable_note", "computable_under_this_contract", "max_negative_auroc_difference",
                "min_subgroup_size_for_check", "name", "predecessor_table", "status", "subgroups",
            }),
            "E": frozenset({
                "computable_under_this_contract", "computed_over", "deployment_runtime_claim_authorized",
                "max_h100_hours_1000_image_projection", "max_h100_hours_96_image_feasibility_run",
                "name", "predecessor_table", "projection_formula_and_uncertainty_disclosure_required",
            }),
            "outcome": frozenset({
                "discovery_decision_definition", "discovery_decision_is_not_the_predecessor_gates_outcome",
                "discovery_decision_values", "discovery_pass_authorizes_only",
                "discovery_pass_does_not_authorize", "discovery_stop_ends_research_direction_under_frozen_hypothesis",
                "overall_status", "overall_status_reachable_by_this_authority",
            }),
        },
    ),
    "resource_contract": (
        frozenset({
            "gpu_count", "gpu_memory_gb", "gpu_type", "inherits_compute_ceiling_from",
            "larger_budget_prohibited", "precedent", "profiling_scope", "prohibited",
            "stage_level_profiling_requirement_introduced", "this_task",
        }),
        {
            "prohibited": frozenset({
                "alternate_gpu_type", "arbitrary_subset_execution", "distributed_execution",
                "full_5000_image_execution", "full_96_image_execution", "holdout_execution",
                "launcher_self_submission", "login_node_execution", "multi_gpu_execution",
                "overwrite_force", "parameter_updates", "training",
            }),
            "this_task": frozenset({"submits_anything"}),
        },
    ),
    "authorization": (
        frozenset({"current_task", "frozen_flags", "future_execution_conditions", "successor_implementation"}),
        {
            "current_task": frozenset({
                "authority_creation_authorized", "checkpoint_loading_in_this_task",
                "cuda_initialization_in_this_task", "dataset_access_in_this_task",
                "discovery_execution_in_this_task", "gpu_submission_in_this_task",
                "gt_access_in_this_task", "implementation_in_this_task", "model_construction_in_this_task",
            }),
            "successor_implementation": frozenset({
                "cli_implementation_authorized_after_commit", "discovery_runner_implementation_authorized_after_commit",
                "launcher_implementation_authorized_after_commit",
            }),
            "future_execution_conditions": frozenset({
                "requires_committed_head_production_readiness_pass", "requires_discovery_implementation_committed",
                "requires_discovery_implementation_independently_verified_pass",
                "requires_operator_explicit_job_submission", "requires_this_authority_committed",
                "requires_this_authority_independently_verified_pass",
            }),
            "frozen_flags": frozenset({
                "discovery_execution_authorized_now", "e3_modification_authorized",
                "full_5000_image_execution_authorized", "full_96_image_execution_authorized",
                "holdout_execution_authorized", "scientific_kernel_modification_authorized",
                "smoke_rerun_authorized", "t320_modification_authorized", "training_authorized",
            }),
        },
    ),
    "future_implementation_files": frozenset({
        "collision_checked_at_freeze_time", "collision_found",
        "modification_of_existing_scientific_files_authorized", "no_other_new_production_file_authorized",
        "paths", "this_task_creates_none_of_these_files",
    }),
    "this_authoritys_own_files": frozenset({
        "docs_path", "exactly_four_files_authorized", "identity_loader_path", "identity_tests_path",
        "no_existing_file_modification_authorized", "toml_path",
    }),
    "closure": frozenset({"complete", "final", "self_referential_hash_created"}),
}


# ---------------------------------------------------------------------------
# Path containment (never accepts traversal, absolute paths, or symlink escape)
# ---------------------------------------------------------------------------

def _require_repo_contained_relative_path(repo_root: Path, relative_path: str, label: str) -> Path:
    _require(isinstance(relative_path, str) and relative_path != "", f"{label} must be a non-empty string")
    _require(not os.path.isabs(relative_path), f"{label} must not be an absolute path: {relative_path!r}")
    normalized = os.path.normpath(relative_path)
    _require(not normalized.startswith(".."), f"{label} must not traverse outside the repository: {relative_path!r}")
    _require(not normalized.startswith("/"), f"{label} must not normalize to an absolute path: {relative_path!r}")
    resolved = (repo_root / relative_path).resolve()
    repo_resolved = repo_root.resolve()
    _require(
        resolved == repo_resolved or repo_resolved in resolved.parents,
        f"{label} resolves outside the repository root: {relative_path!r}",
    )
    return resolved


def _git_show_bytes(repo_root: Path, commit: str, relative_path: str) -> bytes:
    proc = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=str(repo_root), capture_output=True,
    )
    if proc.returncode != 0:
        raise DiscoveryExecutionContractError(
            f"git show failed for {commit}:{relative_path}: {proc.stderr.decode(errors='replace')[:300]}"
        )
    return proc.stdout


def _git_rev_parse(repo_root: Path, ref: str) -> str:
    proc = subprocess.run(["git", "rev-parse", ref], cwd=str(repo_root), capture_output=True, text=True)
    _require(proc.returncode == 0, f"git rev-parse {ref!r} failed: {proc.stderr[:300]}")
    return proc.stdout.strip()


def _git_tree_id(repo_root: Path, commit: str) -> str:
    proc = subprocess.run(["git", "cat-file", "-p", commit], cwd=str(repo_root), capture_output=True, text=True)
    _require(proc.returncode == 0, f"git cat-file -p {commit!r} failed: {proc.stderr[:300]}")
    first_line = proc.stdout.splitlines()[0]
    _require(first_line.startswith("tree "), f"unexpected git cat-file output for {commit!r}")
    return first_line[len("tree "):]


def _git_status_porcelain(repo_root: Path) -> str:
    proc = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo_root), capture_output=True, text=True)
    _require(proc.returncode == 0, f"git status failed: {proc.stderr[:300]}")
    return proc.stdout


# ---------------------------------------------------------------------------
# TOML loading
# ---------------------------------------------------------------------------

def load_authority(toml_path: Path) -> dict:
    try:
        with open(toml_path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DiscoveryExecutionContractError(f"failed to load/parse {toml_path}: {exc}") from exc

    _TOP_LEVEL_KEYS = frozenset(_TOML_SCHEMA.keys())
    actual_keys = frozenset(data.keys())
    _require(
        actual_keys == _TOP_LEVEL_KEYS,
        f"top-level key mismatch: missing={sorted(_TOP_LEVEL_KEYS - actual_keys)}, extra={sorted(actual_keys - _TOP_LEVEL_KEYS)}",
    )

    for key, schema in _TOML_SCHEMA.items():
        _validate_closed_recursive(data[key], schema, key)

    identity = _require_dict(data["identity"], "identity")
    _require(identity.get("schema_name") == "semantic_evidence_discovery_execution_authority", "identity.schema_name mismatch")
    _require(identity.get("schema_version") == "1.0.0", "identity.schema_version mismatch")

    return data


# ---------------------------------------------------------------------------
# Section 1: prerequisite gate
# ---------------------------------------------------------------------------

def verify_prerequisite_gate(repo_root: Path, data: dict) -> dict:
    prereq = _require_dict(data["prerequisite"], "prerequisite")
    required_branch = _require_str(prereq["required_branch"], "prerequisite.required_branch")
    required_head = _require_git_sha_hex(prereq["required_head"], "prerequisite.required_head")
    required_head_tree = _require_git_sha_hex(prereq["required_head_tree"], "prerequisite.required_head_tree")

    proc = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo_root), capture_output=True, text=True)
    _require(proc.returncode == 0, "git rev-parse --abbrev-ref HEAD failed")
    current_branch = proc.stdout.strip()
    current_head = _git_rev_parse(repo_root, "HEAD")
    current_tree = _git_tree_id(repo_root, "HEAD")

    branch_ok = current_branch == required_branch
    head_ok = current_head == required_head
    tree_ok = current_tree == required_head_tree

    return {
        "current_branch": current_branch, "required_branch": required_branch, "branch_ok": branch_ok,
        "current_head": current_head, "required_head": required_head, "head_ok": head_ok,
        "current_tree": current_tree, "required_head_tree": required_head_tree, "tree_ok": tree_ok,
        "all_ok": branch_ok and head_ok and tree_ok,
    }


# ---------------------------------------------------------------------------
# Section 2: dependency hash verification
# ---------------------------------------------------------------------------

_EXPECTED_DEPENDENCY_ROLES = frozenset({
    "feasibility_contract", "feasibility_loader", "feasibility_tests", "feasibility_docs",
    "population_manifest", "t320_source_authority", "t320_source_manifest", "e3_adapter",
    "t320_implementation", "e3_adapter_tests", "t320_implementation_tests",
    "t320_implementation_docs", "smoke_authority", "smoke_authority_loader",
    "smoke_runner", "smoke_cli", "smoke_launcher", "smoke_two_image_manifest",
})


def verify_dependency_hashes(repo_root: Path, data: dict) -> dict:
    prereq = _require_dict(data["prerequisite"], "prerequisite")
    required_head = _require_git_sha_hex(prereq["required_head"], "prerequisite.required_head")

    deps = _require_list(data["dependency"], "dependency")
    _require(len(deps) >= 1, "dependency list must be non-empty")

    declared_roles = frozenset(e.get("role") for e in deps if isinstance(e, dict))
    _require(
        declared_roles == _EXPECTED_DEPENDENCY_ROLES,
        f"dependency role set mismatch: missing={sorted(_EXPECTED_DEPENDENCY_ROLES - declared_roles)}, "
        f"extra={sorted(declared_roles - _EXPECTED_DEPENDENCY_ROLES)}",
    )

    seen_roles: dict = {}
    seen_paths: dict = {}
    results = []
    for entry in deps:
        entry = _require_dict(entry, "dependency entry")
        actual_keys = frozenset(entry.keys())
        _require(
            actual_keys == _DEPENDENCY_RECORD_KEYS,
            f"dependency entry key mismatch: missing={_DEPENDENCY_RECORD_KEYS - actual_keys}, extra={actual_keys - _DEPENDENCY_RECORD_KEYS}",
        )
        role = _require_str(entry["role"], "dependency.role")
        path = _require_str(entry["path"], "dependency.path")
        declared_sha = _require_sha256_hex(entry["sha256"], f"dependency[{role!r}].sha256")
        must_committed = _require_bool(entry["must_be_committed_at_required_head"], f"dependency[{role!r}].must_be_committed_at_required_head")
        _require_str(entry["reason"], f"dependency[{role!r}].reason")

        _require(role not in seen_roles, f"duplicate dependency role: {role!r}")
        _require(path not in seen_paths, f"duplicate dependency path: {path!r}")
        seen_roles[role] = path
        seen_paths[path] = role

        _require_repo_contained_relative_path(repo_root, path, f"dependency[{role!r}].path")
        _require(must_committed is True, f"dependency[{role!r}].must_be_committed_at_required_head must be true")

        actual_bytes = _git_show_bytes(repo_root, required_head, path)
        actual_sha = hashlib.sha256(actual_bytes).hexdigest()
        match = actual_sha == declared_sha

        results.append({"role": role, "path": path, "declared_sha256": declared_sha, "actual_sha256": actual_sha, "match": match})

    all_match = all(r["match"] for r in results)
    return {"records": results, "all_match": all_match, "role_count": len(results)}


# ---------------------------------------------------------------------------
# Section 4: population verification
# ---------------------------------------------------------------------------

def _population_digest(ids: list) -> str:
    data = "".join(i + "\n" for i in ids).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def verify_population(repo_root: Path, data: dict) -> dict:
    pop = _require_dict(data["population"], "population")
    required_head = _require_git_sha_hex(_require_dict(data["prerequisite"], "prerequisite")["required_head"], "prerequisite.required_head")

    _require_int_not_bool(pop["total_selected_count"], "population.total_selected_count")
    _require_int_not_bool(pop["discovery_count"], "population.discovery_count")
    _require_int_not_bool(pop["locked_holdout_count"], "population.locked_holdout_count")
    _require_int_not_bool(pop["discovery_holdout_overlap"], "population.discovery_holdout_overlap")
    for bound in list(pop["discovery_rank_range"]) + list(pop["locked_holdout_rank_range"]):
        _require_int_not_bool(bound, "population rank range bound")

    manifest_dep = None
    for entry in _require_list(data["dependency"], "dependency"):
        if entry.get("role") == "population_manifest":
            manifest_dep = entry
            break
    _require(manifest_dep is not None, "population_manifest dependency entry is required")
    manifest_path = manifest_dep["path"]

    raw = _git_show_bytes(repo_root, required_head, manifest_path)
    manifest = json.loads(raw)

    _require(manifest["selected_count"] == pop["total_selected_count"], "selected_count mismatch")
    _require(manifest["discovery_count"] == pop["discovery_count"], "discovery_count mismatch")
    _require(manifest["locked_holdout_count"] == pop["locked_holdout_count"], "locked_holdout_count mismatch")

    records = manifest["records"]
    _require(len(records) == pop["total_selected_count"], f"expected {pop['total_selected_count']} records, got {len(records)}")

    recs_sorted = sorted(records, key=lambda r: r["selection_rank"])
    ranks = [r["selection_rank"] for r in recs_sorted]
    _require(ranks == list(range(pop["total_selected_count"])), "selection_rank is not an exact 0..N-1 permutation")

    dr_lo, dr_hi = pop["discovery_rank_range"]
    hr_lo, hr_hi = pop["locked_holdout_rank_range"]
    discovery_ids = [r["canonical_image_id"] for r in recs_sorted if dr_lo <= r["selection_rank"] <= dr_hi]
    holdout_ids = [r["canonical_image_id"] for r in recs_sorted if hr_lo <= r["selection_rank"] <= hr_hi]

    _require(len(discovery_ids) == pop["discovery_count"], "discovery id count mismatch")
    _require(len(holdout_ids) == pop["locked_holdout_count"], "locked_holdout id count mismatch")

    overlap = set(discovery_ids) & set(holdout_ids)
    _require(len(overlap) == pop["discovery_holdout_overlap"], f"discovery/holdout overlap mismatch: found {len(overlap)}")

    duplicate_discovery = len(discovery_ids) != len(set(discovery_ids))
    _require(not duplicate_discovery, "duplicate discovery image id found")

    computed_discovery_digest = _population_digest(discovery_ids)
    computed_holdout_digest = _population_digest(holdout_ids)

    discovery_digest_ok = computed_discovery_digest == pop["discovery_digest"]
    holdout_digest_ok = computed_holdout_digest == pop["locked_holdout_digest"]

    return {
        "discovery_count": len(discovery_ids),
        "holdout_count": len(holdout_ids),
        "overlap_count": len(overlap),
        "discovery_digest_ok": discovery_digest_ok,
        "holdout_digest_ok": holdout_digest_ok,
        "all_ok": discovery_digest_ok and holdout_digest_ok and len(overlap) == 0 and not duplicate_discovery,
    }


# ---------------------------------------------------------------------------
# Section 5: scientific inheritance equality verification
# ---------------------------------------------------------------------------

def verify_scientific_inheritance(repo_root: Path, data: dict) -> dict:
    prereq = _require_dict(data["prerequisite"], "prerequisite")
    required_head = _require_git_sha_hex(prereq["required_head"], "prerequisite.required_head")

    feas_dep = None
    for entry in _require_list(data["dependency"], "dependency"):
        if entry.get("role") == "feasibility_contract":
            feas_dep = entry
            break
    _require(feas_dep is not None, "feasibility_contract dependency entry is required")

    raw = _git_show_bytes(repo_root, required_head, feas_dep["path"])
    predecessor = tomllib.loads(raw.decode("utf-8"))

    si = _require_dict(data["scientific_inheritance"], "scientific_inheritance")
    checks = []

    def check(label: str, mine: Any, theirs: Any) -> None:
        # if the reference (expected) value is a real int, reject a bool
        # standing in for it before comparing -- True == 1 / False == 0
        # would otherwise silently launder a wrong-typed value through an
        # equality check whenever the frozen value happens to be 0 or 1.
        if isinstance(theirs, int) and not isinstance(theirs, bool) and isinstance(mine, bool):
            checks.append({"label": label, "ok": False, "mine": mine, "theirs": theirs, "reason": "bool given where int expected"})
            return
        checks.append({"label": label, "ok": mine == theirs, "mine": mine, "theirs": theirs})

    geo = si["e3_geometry"]
    p_geo = predecessor["e3_geometry"]
    check("e3_geometry.resize_dim_pixels", geo["resize_dim_pixels"], p_geo["resize_dim_pixels"])
    check("e3_geometry.patch_size_pixels", geo["patch_size_pixels"], p_geo["patch_size_pixels"])
    check("e3_geometry.patch_grid_side", geo["patch_grid_side"], p_geo["patch_grid_side"])
    check("e3_geometry.crop_size_pixels", geo["crop_size_pixels"], p_geo["crop_size_pixels"])
    check("e3_geometry.stride_pixels", geo["stride_pixels"], p_geo["stride_pixels"])
    check("e3_geometry.eligible_grid_range", geo["eligible_grid_range"], p_geo["eligible_grid_range"])
    check("e3_geometry.pamr_active", geo["pamr_active"], p_geo["pamr_active"])

    anc = si["anchor_selection"]
    p_anc = predecessor["anchor_selection"]
    check("anchor_selection.target_anchors_per_window", anc["target_anchors_per_window"], p_anc["target_anchors_per_window"])
    check("anchor_selection.anchors_per_stratum", anc["anchors_per_stratum"], p_anc["anchors_per_stratum"])
    check("anchor_selection.queries_per_anchor", anc["queries_per_anchor"], p_anc["queries_per_anchor"])
    check("anchor_selection.gt_or_correctness_forbidden_in_selection", anc["gt_or_correctness_forbidden_in_selection"], p_anc["gt_or_correctness_forbidden_in_selection"])

    iv = si["interventions"]
    p_iv = predecessor["interventions"]
    check("interventions.views", iv["views"], p_iv["views"])
    check("interventions.core_patch_count", iv["core_patch_count"], predecessor["region"]["core_patch_count"])
    check("interventions.core_pixel_size", iv["core_pixel_size"], predecessor["region"]["core_pixel_size"])

    gb = si["gaussian_blur"]
    p_gb = predecessor["gaussian_blur"]
    check("gaussian_blur.kernel_size", gb["kernel_size"], p_gb["kernel_size"])
    check("gaussian_blur.sigma", gb["sigma"], p_gb["sigma"])
    check("gaussian_blur.padding_mode", gb["padding_mode"], p_gb["padding_mode"])
    check("gaussian_blur.deterministic", gb["deterministic"], p_gb["deterministic"])

    sc = si["scores"]
    p_sc_v = predecessor["scores"]["validity"]
    check("scores.input_range_required", sc["input_range_required"], p_sc_v["input_range_required"])
    check("scores.output_range_required", sc["output_range_required"], p_sc_v["output_range_required"])
    check("scores.intervention_reliability_range", sc["intervention_reliability_range"], predecessor["scores"]["formulas"]["intervention_reliability_range"])

    ms = si["statistics"]["model_settings"] if "model_settings" in si.get("statistics", {}) else None
    p_ms = predecessor["statistics"]["model_settings"]
    stat = si["statistics"]
    check("statistics.C", stat["C"], p_ms["C"])
    check("statistics.penalty", stat["penalty"], p_ms["penalty"])
    check("statistics.class_weight", stat["class_weight"], p_ms["class_weight"])
    check("statistics.solver", stat["solver"], p_ms["solver"])
    check("statistics.max_iter", stat["max_iter"], p_ms["max_iter"])
    check("statistics.random_state", stat["random_state"], p_ms["random_state"])
    check("statistics.hyperparameter_search", stat["hyperparameter_search"], p_ms["hyperparameter_search"])

    bs = si["bootstrap"]
    p_bs = predecessor["statistics"]["bootstrap"]
    check("bootstrap.replicates", bs["replicates"], p_bs["replicates"])
    check("bootstrap.seed", bs["seed"], p_bs["seed"])
    check("bootstrap.confidence_level", bs["confidence_level"], p_bs["confidence_level"])
    check("bootstrap.paired", bs["paired"], p_bs["paired"])
    check("bootstrap.refit_inside_replicates", bs["refit_inside_replicates"], p_bs["refit_inside_replicates"])
    check("bootstrap.min_valid_replicate_fraction", bs["min_valid_replicate_fraction"], p_bs["min_valid_replicate_fraction"])

    gt = si["gt_targets"]
    p_gt = predecessor["gt_targets"]["propagation_outcome"]
    check("gt_targets.propagation_categories", gt["propagation_categories"], p_gt["categories"])
    check("gt_targets.ambiguous_label", gt["ambiguous_label"], predecessor["gt_targets"]["primary_correctness"]["ambiguous_label"])
    check("gt_targets.ignore_gt_value", gt["ignore_gt_value"], predecessor["gt_targets"]["primary_correctness"]["ignore_gt_value"])
    check("gt_targets.min_valid_footprint_fraction", gt["min_valid_footprint_fraction"], predecessor["gt_targets"]["primary_correctness"]["min_valid_footprint_fraction"])
    check("gt_targets.min_majority_share", gt["min_majority_share"], predecessor["gt_targets"]["primary_correctness"]["min_majority_share"])

    gates = _require_dict(data["gate_contract"], "gate_contract")
    p_gates_a = predecessor["gates"]["A"]
    check("gate_contract.A.min_structural_validity_fraction", gates["A"]["min_structural_validity_fraction"], p_gates_a["min_structural_validity_fraction"])
    check("gate_contract.A.max_non_finite_scores", gates["A"]["max_non_finite_scores"], p_gates_a["max_non_finite_scores"])
    p_gates_b = predecessor["gates"]["B"]
    check("gate_contract.B.min_auroc_improvement_holdout", gates["B"]["min_auroc_improvement_holdout"], p_gates_b["min_auroc_improvement_holdout"])
    p_gates_c = predecessor["gates"]["C"]
    check("gate_contract.C.min_auroc_improvement_holdout", gates["C"]["min_auroc_improvement_holdout"], p_gates_c["min_auroc_improvement_holdout"])
    p_gates_d = predecessor["gates"]["D"]
    check("gate_contract.D.min_subgroup_size_for_check", gates["D"]["min_subgroup_size_for_check"], p_gates_d["min_subgroup_size_for_check"])
    check("gate_contract.D.max_negative_auroc_difference", gates["D"]["max_negative_auroc_difference"], p_gates_d["max_negative_auroc_difference"])
    check("gate_contract.D.subgroups", gates["D"]["subgroups"], p_gates_d["subgroups"])
    p_gates_e = predecessor["gates"]["E"]
    check("gate_contract.E.max_h100_hours_96_image_feasibility_run", gates["E"]["max_h100_hours_96_image_feasibility_run"], p_gates_e["max_h100_hours_96_image_feasibility_run"])
    check("gate_contract.E.max_h100_hours_1000_image_projection", gates["E"]["max_h100_hours_1000_image_projection"], p_gates_e["max_h100_hours_1000_image_projection"])

    check("gate_contract.proceed_value", gates["proceed_value"], predecessor["gates"]["outcome"]["proceed_value"])
    check("gate_contract.stop_value", gates["stop_value"], predecessor["gates"]["outcome"]["stop_value"])

    t320_status_in_feasibility = predecessor["propagation_target_authority"]["t320_status"]
    t0t320 = _require_dict(data["t0_t320_target_authority"], "t0_t320_target_authority")
    check("t0_t320_target_authority.t320_status_in_feasibility_contract", t0t320["t320_status_in_feasibility_contract"], t320_status_in_feasibility)

    all_ok = all(c["ok"] for c in checks)
    return {"checks": checks, "all_ok": all_ok, "check_count": len(checks)}


# ---------------------------------------------------------------------------
# T320 frozen-constant cross-verification against committed source (AST-based)
# ---------------------------------------------------------------------------

def _module_level_literal_assignments(tree: ast.Module) -> dict:
    """Extracts NAME = <literal> assignments at module (top) level only."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass
    return out


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise DiscoveryExecutionContractError(f"required function {name!r} not found in committed T320 source")


def _call_full_name(call_node: ast.Call) -> str:
    """Renders a Call's func as a dotted name string, e.g. 'torch.sigmoid'."""
    func = call_node.func
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def _analyze_internal_upsample(func_node: ast.FunctionDef) -> dict:
    """Independently walks internal_upsample's own statement list (in source
    order) to confirm: exactly one torch.sigmoid call; a .contiguous() call
    chained into the value assigned to the variable later passed to
    torch.sigmoid; that sigmoid-assignment statement precedes the
    F.interpolate statement; the interpolate call's mode/align_corners."""
    stmts = func_node.body

    sigmoid_calls = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call) and _call_full_name(node) == "torch.sigmoid":
            sigmoid_calls.append(node)
    sigmoid_count = len(sigmoid_calls)

    contiguous_var = None
    contiguous_stmt_index = None
    sigmoid_var = None
    sigmoid_stmt_index = None
    interpolate_stmt_index = None
    interpolate_mode = None
    interpolate_align_corners_name = None

    for i, stmt in enumerate(stmts):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue
        target_name = stmt.targets[0].id
        value = stmt.value

        # does this statement's value expression end in a .contiguous() call?
        node = value
        has_contiguous = False
        while isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "contiguous":
                has_contiguous = True
                break
            node = node.func.value if isinstance(node.func, ast.Attribute) else None
            if node is None:
                break
        if has_contiguous and contiguous_var is None:
            contiguous_var = target_name
            contiguous_stmt_index = i

        if isinstance(value, ast.Call) and _call_full_name(value) == "torch.sigmoid":
            arg_names = [a.id for a in value.args if isinstance(a, ast.Name)]
            if contiguous_var is not None and contiguous_var in arg_names:
                sigmoid_var = target_name
                sigmoid_stmt_index = i

        if isinstance(value, ast.Call) and _call_full_name(value) == "F.interpolate":
            interpolate_stmt_index = i
            for kw in value.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    interpolate_mode = kw.value.value
                if kw.arg == "align_corners" and isinstance(kw.value, ast.Name):
                    interpolate_align_corners_name = kw.value.id
            first_arg = value.args[0] if value.args else None
            interpolate_arg_is_sigmoid_var = isinstance(first_arg, ast.Name) and first_arg.id == sigmoid_var

    return {
        "sigmoid_count": sigmoid_count,
        "contiguous_found": contiguous_var is not None,
        "contiguous_before_sigmoid": (
            contiguous_stmt_index is not None and sigmoid_stmt_index is not None
            and contiguous_stmt_index < sigmoid_stmt_index
        ),
        "sigmoid_before_interpolate": (
            sigmoid_stmt_index is not None and interpolate_stmt_index is not None
            and sigmoid_stmt_index < interpolate_stmt_index
        ),
        "interpolate_uses_sigmoid_output": locals().get("interpolate_arg_is_sigmoid_var", False),
        "interpolate_mode": interpolate_mode,
        "interpolate_align_corners_name": interpolate_align_corners_name,
    }


def _analyze_recurrence_loop(func_node: ast.FunctionDef) -> dict:
    """Confirms _finite_step_propagate's `for` loop contains no Break and no
    conditional (If) early-exit construct, and that torch.sigmoid is never
    called anywhere inside this function (sigmoid must happen only after
    recurrence, inside internal_upsample, never during/before it)."""
    for_nodes = [n for n in ast.walk(func_node) if isinstance(n, ast.For)]
    _require(len(for_nodes) == 1, "expected exactly one for-loop in _finite_step_propagate")
    loop = for_nodes[0]

    has_break = any(isinstance(n, ast.Break) for n in ast.walk(loop))
    has_conditional_in_loop = any(isinstance(n, ast.If) for n in ast.walk(loop))
    sigmoid_calls_in_function = [
        n for n in ast.walk(func_node)
        if isinstance(n, ast.Call) and _call_full_name(n) == "torch.sigmoid"
    ]

    return {
        "early_stop_break_absent": not has_break,
        "early_stop_conditional_absent": not has_conditional_in_loop,
        "no_sigmoid_during_recurrence": len(sigmoid_calls_in_function) == 0,
    }


def _analyze_public_propagate_entry_point(func_node: ast.FunctionDef) -> dict:
    """Confirms propagate_t320's own parameter list exposes no alpha/steps
    override -- exactly (graph, s0)."""
    arg_names = [a.arg for a in func_node.args.args]
    return {"no_public_override": arg_names == ["graph", "s0"]}


def verify_t0_t320_frozen_constants(repo_root: Path, data: dict) -> dict:
    prereq = _require_dict(data["prerequisite"], "prerequisite")
    required_head = _require_git_sha_hex(prereq["required_head"], "prerequisite.required_head")

    t320_dep = None
    for entry in _require_list(data["dependency"], "dependency"):
        if entry.get("role") == "t320_implementation":
            t320_dep = entry
            break
    _require(t320_dep is not None, "t320_implementation dependency entry is required")

    # verify the dependency hash FIRST, before trusting/parsing the source bytes
    raw_bytes = _git_show_bytes(repo_root, required_head, t320_dep["path"])
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    _require(actual_sha == t320_dep["sha256"], "t320_implementation source hash does not match its own dependency binding; refusing to parse untrusted bytes")

    try:
        tree = ast.parse(raw_bytes.decode("utf-8"), filename=t320_dep["path"])
    except SyntaxError as exc:
        raise DiscoveryExecutionContractError(f"committed T320 source failed to parse: {exc}") from exc

    module_consts = _module_level_literal_assignments(tree)

    internal_upsample_fn = _find_function(tree, "internal_upsample")
    recurrence_fn = _find_function(tree, "_finite_step_propagate")
    public_entry_fn = _find_function(tree, "propagate_t320")

    upsample_analysis = _analyze_internal_upsample(internal_upsample_fn)
    recurrence_analysis = _analyze_recurrence_loop(recurrence_fn)
    public_entry_analysis = _analyze_public_propagate_entry_point(public_entry_fn)

    fc = _require_dict(data["t0_t320_target_authority"], "t0_t320_target_authority")["frozen_constants"]

    checks = []

    def check(label: str, actual: Any, expected: Any) -> None:
        if isinstance(expected, int) and not isinstance(expected, bool) and isinstance(actual, bool):
            checks.append({"label": label, "ok": False, "actual": actual, "expected": expected, "reason": "bool given where int expected"})
            return
        checks.append({"label": label, "ok": actual == expected, "actual": actual, "expected": expected})

    check("K", module_consts.get("K"), fc["k"])
    check("ALPHA", module_consts.get("ALPHA"), fc["alpha"])
    check("AFFINITY_POWER", module_consts.get("AFFINITY_POWER"), fc["affinity_power"])
    check("STEPS", module_consts.get("STEPS"), fc["steps"])
    check("INTERNAL_UPSAMPLE_ALIGN_CORNERS", module_consts.get("INTERNAL_UPSAMPLE_ALIGN_CORNERS"), fc["internal_align_corners"])
    check("FINAL_RESCALE_ALIGN_CORNERS", module_consts.get("FINAL_RESCALE_ALIGN_CORNERS"), fc["final_align_corners"])

    check("internal_sigmoid_count", upsample_analysis["sigmoid_count"], fc["internal_sigmoid_count"])
    check("contiguous_grid_before_sigmoid_interpolation",
          upsample_analysis["contiguous_found"] and upsample_analysis["contiguous_before_sigmoid"] and upsample_analysis["sigmoid_before_interpolate"],
          fc["contiguous_grid_before_sigmoid_interpolation"])
    check("internal_interpolation_mode", upsample_analysis["interpolate_mode"], fc["internal_interpolation_mode"])
    check("internal_align_corners_wired_to_module_constant", upsample_analysis["interpolate_align_corners_name"], "INTERNAL_UPSAMPLE_ALIGN_CORNERS")
    check("interpolate_consumes_sigmoid_output", upsample_analysis["interpolate_uses_sigmoid_output"], True)

    check("early_stopping_absent", recurrence_analysis["early_stop_break_absent"] and recurrence_analysis["early_stop_conditional_absent"], not fc["early_stopping"])
    check("no_sigmoid_during_recurrence", recurrence_analysis["no_sigmoid_during_recurrence"], True)

    check("no_public_scientific_override", public_entry_analysis["no_public_override"], True)

    all_ok = all(c["ok"] for c in checks)
    return {"source_hash_verified": True, "checks": checks, "all_ok": all_ok, "check_count": len(checks)}


# ---------------------------------------------------------------------------
# Section 3: smoke prerequisite verification
# ---------------------------------------------------------------------------

_REQUIRED_SMOKE_PROFILING_STATUS = "NOT_AUTHORITY_FOR_RUNTIME_OR_THROUGHPUT_CLAIMS"

_FORBIDDEN_PROFILING_TOKENS = (
    "smoke profiling", "smoke timing", "smoke memory", "gpu peak memory",
    "runtime projection", "throughput", "full-dataset runtime", "wall-clock timing",
    "wall_time", "peak_gpu", "cuda_driver_version", "cuda_runtime_version",
)


def verify_smoke_prerequisite(repo_root: Path, data: dict) -> dict:
    sp = _require_dict(data["smoke_prerequisite"], "smoke_prerequisite")
    result_path_str = _require_str(sp["result_path"], "smoke_prerequisite.result_path")
    expected_sha = _require_sha256_hex(sp["result_sha256"], "smoke_prerequisite.result_sha256")

    _require(
        sp["smoke_profiling_status"] == _REQUIRED_SMOKE_PROFILING_STATUS,
        f"smoke_prerequisite.smoke_profiling_status must be exactly {_REQUIRED_SMOKE_PROFILING_STATUS!r}",
    )
    _require(sp["smoke_profiling_defects_repair_authorized_by_this_task"] is False, "smoke profiling defects must not be marked repair-authorized")
    _require(sp["smoke_profiling_defects_inherited_by_this_authority"] is False, "smoke profiling defects must not be marked inherited")
    _require(len(sp["smoke_profiling_defects"]) >= 5, "smoke_profiling_defects disclosure list looks truncated")

    result_path = Path(result_path_str)
    if not result_path.exists():
        return {
            "external_artifact_available": False,
            "note": "the external smoke result path does not exist in this environment; treated as UNVERIFIED, never as silently verified",
            "hash_match": None,
        }

    actual_bytes = result_path.read_bytes()
    actual_sha = hashlib.sha256(actual_bytes).hexdigest()
    hash_match = actual_sha == expected_sha

    try:
        parsed = json.loads(actual_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {
            "external_artifact_available": True, "hash_match": hash_match,
            "actual_sha256": actual_sha, "expected_sha256": expected_sha,
            "outcome_ok": False, "parse_error": True,
        }

    def field_ok(actual, expected):
        return actual == expected

    tree = parsed.get("repository_commit_tree_cleanliness", {})
    counts = parsed.get("forward_snapshot_graph_trajectory_update_counts", {})
    preservation = parsed.get("t0_preservation_comparisons_per_window", [])
    non_finite = parsed.get("non_finite_counts_by_stage", {})
    scope = parsed.get("scope_disclosures", {})
    schema = parsed.get("schema", {})
    telemetry = parsed.get("timing_telemetry", {})

    checks = {
        "outcome_ok": field_ok(parsed.get("final_smoke_outcome"), sp["outcome_required"]),
        "job_id_ok": field_ok(str(telemetry.get("slurm_job_id")), str(sp["job_id"])),
        "execution_head_ok": field_ok(tree.get("head"), sp["execution_head"]),
        "execution_tree_ok": field_ok(tree.get("tree"), sp["execution_tree"]),
        "clean_ok": tree.get("clean") is True,
        "complete_ok": parsed.get("complete") is True,
        "final_ok": parsed.get("final") is True,
        "schema_name_ok": field_ok(schema.get("name"), sp["result_schema_name"]),
        "schema_version_ok": field_ok(schema.get("version"), sp["result_schema_version"]),
        "counts_match_ok": counts.get("match") is True,
        "repeat_equality_ok": parsed.get("prediction_equality_results", {}).get("prediction_equal_between_repeats") is True,
        "t0_preservation_all_true": (
            len(preservation) == 4
            and all(r.get("bitwise_equal") is True and r.get("shape_match") is True and r.get("dtype_match") is True for r in preservation)
        ),
        "non_finite_all_zero": len(non_finite) > 0 and all(v == 0 for v in non_finite.values()),
        "scope_disclosures_all_false": len(scope) > 0 and all(v is False for v in scope.values()),
    }

    # selected image population cross-check against the smoke_two_image_manifest dependency
    prereq = _require_dict(data["prerequisite"], "prerequisite")
    required_head = _require_git_sha_hex(prereq["required_head"], "prerequisite.required_head")
    manifest_dep = None
    for entry in _require_list(data["dependency"], "dependency"):
        if entry.get("role") == "smoke_two_image_manifest":
            manifest_dep = entry
            break
    if manifest_dep is not None:
        manifest_raw = _git_show_bytes(repo_root, required_head, manifest_dep["path"])
        manifest = json.loads(manifest_raw)
        manifest_ids = manifest.get("selected_image_ids_ordered") if isinstance(manifest, dict) else manifest
        checks["selected_population_matches_manifest"] = parsed.get("selected_image_ids") == manifest_ids
    else:
        checks["selected_population_matches_manifest"] = False

    all_ok = hash_match and all(checks.values())

    return {
        "external_artifact_available": True,
        "hash_match": hash_match,
        "actual_sha256": actual_sha,
        "expected_sha256": expected_sha,
        "outcome_ok": checks["outcome_ok"],
        "checks": checks,
        "all_ok": all_ok,
    }


# ---------------------------------------------------------------------------
# Gate-input profiling exclusion (defense in depth against a free-form prose
# field being rewritten to reference smoke timing/memory as a gate input)
# ---------------------------------------------------------------------------

# Primary enforcement: exact frozen reference strings, taken verbatim from
# the unmodified, independently-verified-correct authority TOML at repair
# time. These are the ONLY values `verify_gate_inputs_exclude_profiling`
# will accept for each gate's prose reference field -- not derived from the
# mutable field being checked, not pattern-matched, not normalized.
_EXPECTED_GATE_COMPUTED_OVER = {
    "A": "discovery-phase generated records only (48 images); the same Gate A is independently re-evaluated over locked_holdout by a future successor holdout authority before the overall [gates.outcome] decision is reached",
    "E": "discovery-phase (48-image) Slurm elapsed time only, projected; job elapsed time is operational metadata per [artifact_contract], never a scientific field, and the smoke's own profiling defects (see [smoke_prerequisite]) must not be inherited into this projection's methodology",
}

_EXPECTED_GATE_COMPUTABLE_NOTE = {
    "B": "requires locked_holdout evaluation ([gates.B].auroc_improvement_definition explicitly evaluates on locked_holdout anchor_query rows); out of scope for this discovery-only authority; model fits 1/2 ARE fit on discovery by this authority, but their holdout evaluation is deferred",
    "C": "T320 is now bound (see [t0_t320_target_authority]), resolving the feasibility contract's original blocking reason; Gate C nonetheless still requires locked_holdout evaluation and remains out of scope for this discovery-only authority; model fits 3/4 ARE fit on discovery by this authority, but their holdout evaluation is deferred",
    "D": "the rule compares AUROC differences on the same evaluated population as Gates B/C, i.e. locked_holdout; out of scope for this discovery-only authority",
}


def verify_gate_inputs_exclude_profiling(data: dict) -> dict:
    gates = _require_dict(data["gate_contract"], "gate_contract")
    declared_gate_ids = frozenset(gates.keys()) & frozenset({"A", "B", "C", "D", "E"})
    _require(
        declared_gate_ids == frozenset({"A", "B", "C", "D", "E"}),
        f"gate_contract must define exactly gates A-E: missing={frozenset({'A','B','C','D','E'}) - declared_gate_ids}",
    )

    exact_match_checks = []
    for gate_id, expected in _EXPECTED_GATE_COMPUTED_OVER.items():
        actual = gates[gate_id].get("computed_over")
        exact_match_checks.append({"gate": gate_id, "field": "computed_over", "ok": actual == expected, "actual": actual})
    for gate_id, expected in _EXPECTED_GATE_COMPUTABLE_NOTE.items():
        actual = gates[gate_id].get("computable_note")
        exact_match_checks.append({"gate": gate_id, "field": "computable_note", "ok": actual == expected, "actual": actual})

    exact_match_ok = all(c["ok"] for c in exact_match_checks)

    # Defense in depth only: a forbidden-token scan over every prose field
    # actually present, regardless of whether it passed the exact-match
    # check above. Never the primary mechanism.
    prose_fields = []
    for gate_id in ("A", "B", "C", "D", "E"):
        gate = gates[gate_id]
        for key in ("computed_over", "computable_note"):
            if key in gate:
                prose_fields.append((gate_id, key, gate[key]))

    violations = []
    for gate_id, key, text in prose_fields:
        lowered = str(text).lower()
        for token in _FORBIDDEN_PROFILING_TOKENS:
            if token in lowered:
                violations.append((gate_id, key, token))

    all_ok = exact_match_ok and len(violations) == 0

    return {
        "exact_match_checks": exact_match_checks,
        "exact_match_ok": exact_match_ok,
        "prose_fields_scanned": len(prose_fields),
        "blocklist_violations": violations,
        "all_ok": all_ok,
    }


# ---------------------------------------------------------------------------
# Future implementation file-set verification
# ---------------------------------------------------------------------------

_EXPECTED_FUTURE_FILES = (
    "src/semantic_evidence_intervention_features.py",
    "src/semantic_evidence_discovery_execute.py",
    "diagnostics/run_semantic_evidence_discovery.py",
    "semantic_evidence_discovery_h100.sbatch",
    "tests/test_semantic_evidence_intervention_features.py",
    "tests/test_semantic_evidence_discovery_execute.py",
    "docs/semantic_evidence_discovery_run.md",
)


# ---------------------------------------------------------------------------
# Artifact required-field exact-set verification. Frozen verbatim from the
# unmodified, independently-verified-correct authority TOML at repair time --
# never derived from the mutable required_fields list being checked.
# ---------------------------------------------------------------------------

_EXPECTED_ARTIFACT_REQUIRED_FIELDS = {
    "checkpoint_metadata": (
        "schema", "complete", "final", "mode", "discovery_image_count",
        "completed_image_ids_ordered", "completed_image_prefix_digest",
        "identity_and_dependency_bindings", "checkpoint_array_hashes",
    ),
    "gt_free_features": (
        "schema", "image_id", "window_ids_ordered", "anchor_ids_ordered",
        "query_ranks", "intervention_view_scores", "control_features",
        "intervention_reliability", "structural_validity_flags", "gt_free_feature_digest",
    ),
    "gt_dependent_labels": (
        "schema", "image_id", "query_correctness_labels", "propagation_outcome_categories",
        "ambiguous_gt_anchor_ids", "gt_dependent_label_digest",
    ),
    "final_result": (
        "schema", "complete", "final", "identity_and_dependency_bindings",
        "population_binding", "implementation_binding", "discovery_image_ids_ordered",
        "gt_free_feature_digest", "gt_dependent_label_digest", "scientific_collection_digest",
        "model_fit_summaries", "primary_metrics", "bootstrap_intervals", "gate_inputs",
        "gate_pass_fail", "final_frozen_outcome", "holdout_accessed", "full_evaluation_accessed",
    ),
    "decision_document": (
        "schema", "gate_a_discovery_side", "gate_e_discovery_side", "gate_b_status",
        "gate_c_status", "gate_d_status", "overall_gates_outcome_status",
        "discovery_decision", "next_step",
    ),
}


def verify_artifact_required_fields(data: dict) -> dict:
    ac = _require_dict(data["artifact_contract"], "artifact_contract")

    declared_schemas = frozenset(_EXPECTED_ARTIFACT_REQUIRED_FIELDS.keys())
    _require(
        declared_schemas <= frozenset(ac.keys()),
        f"artifact_contract is missing expected schema tables: {declared_schemas - frozenset(ac.keys())}",
    )

    results = {}
    for schema_id, expected_fields in _EXPECTED_ARTIFACT_REQUIRED_FIELDS.items():
        actual_fields = ac[schema_id]["required_fields"]
        _require(isinstance(actual_fields, list), f"artifact_contract.{schema_id}.required_fields must be a list")
        exact_match = tuple(actual_fields) == tuple(expected_fields)
        no_duplicates = len(actual_fields) == len(set(actual_fields))
        results[schema_id] = {
            "exact_match": exact_match, "no_duplicates": no_duplicates,
            "declared_count": len(actual_fields), "expected_count": len(expected_fields),
        }

    all_ok = all(r["exact_match"] and r["no_duplicates"] for r in results.values())
    return {"schemas": results, "all_ok": all_ok}


def verify_future_implementation_files(repo_root: Path, data: dict) -> dict:
    fif = _require_dict(data["future_implementation_files"], "future_implementation_files")
    paths = _require_list(fif["paths"], "future_implementation_files.paths")

    declared = tuple(paths)
    expected = _EXPECTED_FUTURE_FILES

    duplicates = len(declared) != len(set(declared))
    exact_set_match = frozenset(declared) == frozenset(expected)
    exact_count = len(declared) == 7

    for p in declared:
        _require_repo_contained_relative_path(repo_root, p, "future_implementation_files.paths entry")

    currently_absent = {p: not (repo_root / p).exists() for p in declared}
    all_absent = all(currently_absent.values())

    all_ok = (not duplicates) and exact_set_match and exact_count and all_absent

    return {
        "declared_count": len(declared),
        "duplicates_found": duplicates,
        "exact_set_match": exact_set_match,
        "currently_absent": currently_absent,
        "all_ok": all_ok,
    }


# ---------------------------------------------------------------------------
# Section 15: authorization flag verification
# ---------------------------------------------------------------------------

_EXPECTED_CURRENT_TASK_FLAGS = {
    "authority_creation_authorized": True,
    "implementation_in_this_task": False,
    "dataset_access_in_this_task": False,
    "gt_access_in_this_task": False,
    "model_construction_in_this_task": False,
    "checkpoint_loading_in_this_task": False,
    "cuda_initialization_in_this_task": False,
    "gpu_submission_in_this_task": False,
    "discovery_execution_in_this_task": False,
}

_EXPECTED_FROZEN_FLAGS = {
    "discovery_execution_authorized_now": False,
    "holdout_execution_authorized": False,
    "full_96_image_execution_authorized": False,
    "full_5000_image_execution_authorized": False,
    "training_authorized": False,
    "scientific_kernel_modification_authorized": False,
    "e3_modification_authorized": False,
    "t320_modification_authorized": False,
    "smoke_rerun_authorized": False,
}


# ---------------------------------------------------------------------------
# Resource contract exact-value verification
# ---------------------------------------------------------------------------

_EXPECTED_RESOURCE_VALUES = {"gpu_count": 1, "gpu_memory_gb": 80, "gpu_type": "H100"}


def verify_resource_contract(data: dict) -> dict:
    rc = _require_dict(data["resource_contract"], "resource_contract")
    for field in ("gpu_count", "gpu_memory_gb"):
        _require_int_not_bool(rc[field], f"resource_contract.{field}")

    checks = {k: (rc.get(k) == v) for k, v in _EXPECTED_RESOURCE_VALUES.items()}
    all_ok = all(checks.values())
    return {"checks": checks, "all_ok": all_ok}


def verify_authorization_flags(data: dict) -> dict:
    auth = _require_dict(data["authorization"], "authorization")
    current_task = _require_dict(auth["current_task"], "authorization.current_task")
    frozen = _require_dict(auth["frozen_flags"], "authorization.frozen_flags")

    mismatches = []
    for key, expected in _EXPECTED_CURRENT_TASK_FLAGS.items():
        actual = _require_bool(current_task.get(key), f"authorization.current_task.{key}")
        if actual != expected:
            mismatches.append(("current_task", key, expected, actual))
    for key, expected in _EXPECTED_FROZEN_FLAGS.items():
        actual = _require_bool(frozen.get(key), f"authorization.frozen_flags.{key}")
        if actual != expected:
            mismatches.append(("frozen_flags", key, expected, actual))

    return {"mismatches": mismatches, "all_ok": len(mismatches) == 0}


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def verify_all(repo_root: Path, toml_path: Path) -> dict:
    try:
        data = load_authority(toml_path)

        gate = verify_prerequisite_gate(repo_root, data)
        deps = verify_dependency_hashes(repo_root, data)
        pop = verify_population(repo_root, data)
        sci = verify_scientific_inheritance(repo_root, data)
        t320_constants = verify_t0_t320_frozen_constants(repo_root, data)
        smoke = verify_smoke_prerequisite(repo_root, data)
        gate_profiling = verify_gate_inputs_exclude_profiling(data)
        artifact_fields = verify_artifact_required_fields(data)
        future_files = verify_future_implementation_files(repo_root, data)
        resource = verify_resource_contract(data)
        auth = verify_authorization_flags(data)

        working_tree_clean = _git_status_porcelain(repo_root) == ""

        smoke_ok = (
            smoke["external_artifact_available"] is False
            or (smoke["hash_match"] and smoke.get("all_ok", False))
        )

        overall_ok = (
            gate["all_ok"] and deps["all_match"] and pop["all_ok"] and sci["all_ok"]
            and t320_constants["all_ok"] and gate_profiling["all_ok"] and artifact_fields["all_ok"]
            and future_files["all_ok"] and resource["all_ok"]
            and auth["all_ok"] and working_tree_clean and smoke_ok
        )

        return {
            "verified": overall_ok,
            "prerequisite_gate": gate,
            "dependency_hashes": deps,
            "population": pop,
            "scientific_inheritance": sci,
            "t0_t320_frozen_constants": t320_constants,
            "smoke_prerequisite": smoke,
            "gate_inputs_exclude_profiling": gate_profiling,
            "artifact_required_fields": artifact_fields,
            "future_implementation_files": future_files,
            "resource_contract": resource,
            "authorization_flags": auth,
            "working_tree_clean": working_tree_clean,
        }
    except BaseException as exc:
        _reraise_if_fatal(exc)
        raise
