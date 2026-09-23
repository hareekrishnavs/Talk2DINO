"""Torch-free, CUDA-free loader/validator for the T320 real-smoke
integration contract (research_specs/semantic_evidence_t320_real_smoke_v1.toml)
and its companion two-image population manifest
(evaluation_subsets/semantic_evidence_t320_real_smoke_2.json).

This module implements NO model construction, NO checkpoint loading, NO
CUDA initialization, NO real inference, NO graph construction, NO
recurrence, NO stitching, and NO GT access. It only loads, strictly
validates, and independently re-derives (from Git objects, via
argument-vector subprocess calls, and from the committed 96-image manifest)
the provenance, ancestry, population, resource, and scope claims frozen by
those two files. It must never import Torch, MMCV, any model/dataset code
from this repository, or OpenCV, and never checks out a branch or opens a
checkpoint tensor.

Fatal exceptions (KeyboardInterrupt, SystemExit, MemoryError) are never
swallowed by this module: every broad except-clause below re-raises them.

TASK: FREEZE_ISOLATED_T320_REAL_SMOKE_CONTRACT. This module authorizes
NOTHING beyond what [status]/[closure] in the TOML itself declares -- in
particular it never authorizes real execution; it validates a contract that
a SEPARATE, not-yet-implemented smoke runner must satisfy before an operator
may submit one bounded job.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tomllib
from pathlib import Path
from typing import Any


class T320RealSmokeContractError(Exception):
    """Raised for any structural, value, hash, ancestry, population, scope,
    or resource violation encountered while loading or verifying this
    contract."""


_FATAL = (KeyboardInterrupt, SystemExit, MemoryError)


def _reraise_if_fatal(exc: BaseException) -> None:
    if isinstance(exc, _FATAL):
        raise exc


_LEAKABLE = (KeyError, TypeError, AttributeError, ValueError, AssertionError, IndexError)


def _no_leak(func):
    """Wraps a validator so a malformed-contract input never leaks a raw
    KeyError/TypeError/AttributeError/ValueError/AssertionError/IndexError --
    every such failure is re-raised as the single declared domain exception.
    T320RealSmokeContractError itself, and KeyboardInterrupt/SystemExit/
    MemoryError, always pass through unchanged."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except T320RealSmokeContractError:
            raise
        except _LEAKABLE as exc:
            _reraise_if_fatal(exc)
            raise T320RealSmokeContractError(f"{func.__name__}: malformed contract input ({type(exc).__name__}: {exc})") from exc
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def _reject_unsafe_relative_path(relative_path: str) -> None:
    if not isinstance(relative_path, str) or relative_path == "":
        raise T320RealSmokeContractError(f"path must be a non-empty string, got {relative_path!r}")
    if os.path.isabs(relative_path):
        raise T320RealSmokeContractError(f"path must be repository-relative, got absolute path: {relative_path!r}")
    if relative_path.startswith("-"):
        raise T320RealSmokeContractError(f"option-like path rejected: {relative_path!r}")
    for part in relative_path.split("/"):
        if part in ("..", ""):
            raise T320RealSmokeContractError(f"lexical path traversal or empty segment rejected: {relative_path!r}")
        if part == ".":
            raise T320RealSmokeContractError(f"lexical '.' segment rejected: {relative_path!r}")
        if part.startswith("-"):
            raise T320RealSmokeContractError(f"option-like path segment rejected: {relative_path!r}")


def resolve_repo_relative_path(repo_root: Path, relative_path: str) -> Path:
    """Resolve a repository-relative path, rejecting lexical '..', absolute
    paths, option-like paths, and any symlink that would escape repo_root."""
    _reject_unsafe_relative_path(relative_path)
    repo_root_real = Path(os.path.realpath(repo_root))
    candidate = repo_root / relative_path
    candidate_real = Path(os.path.realpath(candidate))
    try:
        candidate_real.relative_to(repo_root_real)
    except ValueError as exc:
        raise T320RealSmokeContractError(
            f"path escapes repository root after symlink resolution: {relative_path!r}"
        ) from exc
    return candidate


_FULL_HEX40 = frozenset("0123456789abcdef")


def _require_full_commit_hex(commit: Any, label: str) -> str:
    if not isinstance(commit, str):
        raise T320RealSmokeContractError(f"{label} must be a string, got {type(commit).__name__}")
    if len(commit) != 40:
        raise T320RealSmokeContractError(f"{label} must be a full 40-hex-character commit id, got {commit!r} (length {len(commit)})")
    if any(ch not in _FULL_HEX40 for ch in commit.lower()) or commit != commit.lower():
        raise T320RealSmokeContractError(f"{label} must be lowercase hex, got {commit!r}")
    return commit


def _require_sha256_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in _FULL_HEX40 for ch in value.lower()) or value != value.lower():
        raise T320RealSmokeContractError(f"{label} must be a lowercase 64-hex-character SHA-256 digest, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Strict TOML / JSON loading (reject NaN/Infinity, reject bool-as-int)
# ---------------------------------------------------------------------------

def _reject_nan_inf(token: str) -> float:
    raise T320RealSmokeContractError(f"NaN/Infinity is not permitted in this JSON document, got token {token!r}")


def load_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except T320RealSmokeContractError:
        raise
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise T320RealSmokeContractError(f"failed to parse TOML at {path}: {exc}") from exc


def load_json_strict(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        return json.loads(text, parse_constant=_reject_nan_inf)
    except T320RealSmokeContractError:
        raise
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise T320RealSmokeContractError(f"failed to parse JSON at {path}: {exc}") from exc


def _require_keys(data: Any, expected: frozenset, label: str) -> None:
    if not isinstance(data, dict):
        raise T320RealSmokeContractError(f"{label} must be a table/object, got {type(data).__name__}")
    actual = frozenset(data.keys())
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise T320RealSmokeContractError(f"{label} is missing required keys: {sorted(missing)}")
    if extra:
        raise T320RealSmokeContractError(f"{label} has unexpected keys: {sorted(extra)}")


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise T320RealSmokeContractError(f"{label} must be a bool, got {type(value).__name__}: {value!r}")
    return value


def _require_exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise T320RealSmokeContractError(f"{label} must be an int (not bool), got {type(value).__name__}: {value!r}")
    return value


def _require_finite_number(value: Any, label: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise T320RealSmokeContractError(f"{label} must be a number, got {type(value).__name__}: {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise T320RealSmokeContractError(f"{label} must be finite, got {value!r}")
    return value


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise T320RealSmokeContractError(f"{label} must be a string, got {type(value).__name__}: {value!r}")
    return value


def _require_list(value: Any, label: str) -> list:
    if not isinstance(value, list):
        raise T320RealSmokeContractError(f"{label} must be a list, got {type(value).__name__}: {value!r}")
    return value


# ---------------------------------------------------------------------------
# Git access (argument-vector subprocess only; never shell=True; never a
# checkout)
# ---------------------------------------------------------------------------

def _run_git_text(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise T320RealSmokeContractError(f"git command failed: git -C {repo_root} {' '.join(args)}: {exc}") from exc
    return result.stdout.strip()


def _run_git_bytes(repo_root: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            check=True,
        )
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise T320RealSmokeContractError(f"git command failed: git -C {repo_root} {' '.join(args)}: {exc}") from exc
    return result.stdout


def git_current_branch(repo_root: Path) -> str:
    return _run_git_text(repo_root, "rev-parse", "--abbrev-ref", "HEAD")


def git_head_commit(repo_root: Path) -> str:
    return _run_git_text(repo_root, "rev-parse", "HEAD")


def git_is_ancestor(repo_root: Path, ancestor_commit: str, descendant_commit: str) -> bool:
    _require_full_commit_hex(ancestor_commit, "ancestor_commit")
    _require_full_commit_hex(descendant_commit, "descendant_commit")
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", ancestor_commit, descendant_commit],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise T320RealSmokeContractError(f"failed to check ancestor relationship: {exc}") from exc


def git_blob_sha256(repo_root: Path, commit: str, path: str) -> str:
    _require_full_commit_hex(commit, "commit")
    _reject_unsafe_relative_path(path)
    content = _run_git_bytes(repo_root, "show", f"{commit}:{path}")
    return hashlib.sha256(content).hexdigest()


def git_tracked_tree_clean(repo_root: Path) -> bool:
    """True iff no tracked file is modified and nothing is staged. Untracked
    files (e.g. this very authority's own not-yet-committed files) are
    permitted -- this checks tracked-file cleanliness, not zero-untracked-
    residue, since that is a task-level report concern, not a permanent
    contract invariant enforced by this loader on every future load."""
    status = _run_git_text(repo_root, "status", "--porcelain")
    for line in status.splitlines():
        if not line:
            continue
        if line.startswith("??"):
            continue  # untracked: permitted
        return False
    return True


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# TOML schema key-set closure (exact key sets per table)
# ---------------------------------------------------------------------------

_TOP_LEVEL_KEYS = frozenset({
    "identity", "status", "implementation_binding", "dependency",
    "external_weights", "smoke_population", "window_geometry",
    "one_forward_contract", "preservation_contract", "t320_execution_contract",
    "determinism_contract", "validity_contract", "gt_isolation",
    "scope_isolation", "resource_contract", "telemetry_contract",
    "result_schema", "artifact_policy", "closure",
})

_IDENTITY_KEYS = frozenset({"schema_name", "schema_version", "created_by_task", "successor_to"})

_STATUS_KEYS = frozenset({
    "contract_only", "smoke_runner_implemented", "smoke_runner_committed",
    "smoke_execution_authorized_by_this_task", "smoke_execution_authorized_after",
    "operator_submission_only", "self_submission_prohibited",
})

_IMPLEMENTATION_BINDING_KEYS = frozenset({
    "introducing_commit", "introducing_commit_parent", "introducing_commit_tree",
    "required_branch", "ancestry_requirement", "file", "required_properties",
})

_IMPLEMENTATION_FILE_KEYS = frozenset({"path", "sha256", "role"})

_IMPLEMENTATION_REQUIRED_PROPERTIES_KEYS = frozenset({
    "sigmoid_before_internal_interpolation", "internal_align_corners",
    "final_align_corners", "build_t320_graph_public_signature",
    "public_k_override_present", "public_affinity_power_override_present",
    "dataset_or_model_or_checkpoint_or_gt_or_cli_path_present",
})

_DEPENDENCY_KEYS = frozenset({"name", "path", "sha256", "role"})

_CLOSURE_KEYS = frozenset({
    "complete", "final", "contract_only", "implements_smoke_runner",
    "executes_real_inference", "accesses_image_pixels",
    "accesses_annotations_or_gt", "loads_checkpoint_tensors",
    "initializes_cuda", "submits_job", "implements_interventions",
    "executes_96_image_feasibility_study", "trains_anything",
    "evaluates_accuracy_or_miou", "stages_or_commits_files",
})

_RESOURCE_CONTRACT_KEYS = frozenset({
    "gpu", "cpus", "host_memory_max_gb", "wall_time_limit", "account",
    "fresh_process_required", "job_count_max", "multi_job_sharding_prohibited",
    "resume_requirement", "overwrite_or_force_mode_prohibited",
    "self_submission_prohibited", "preflight_before", "submission_gate",
})

_SMOKE_POPULATION_KEYS = frozenset({
    "selected_count", "selection_rule", "selected_image_ids_ordered",
    "holdout_overlap_count", "duplicate_count", "image_pixel_access_by_this_task",
    "image_metadata_access_by_this_task", "label_or_annotation_access_by_this_task",
})

_WINDOW_GEOMETRY_KEYS = frozenset({
    "crop_pixels", "stride_pixels", "patch_size_pixels", "crop_stride_patch_source",
    "canonical_window_builder_source", "canonical_window_ordering",
    "flush_right_flush_bottom_behavior", "window_count_status",
    "window_count_fabrication_prohibited", "debug_subset_or_max_window_override_prohibited",
    "future_preflight_requirements",
})

_RESULT_SCHEMA_KEYS = frozenset({"example_filename", "required_top_level_fields", "outcomes"})

_RESULT_SCHEMA_OUTCOMES_KEYS = frozenset({
    "pass_value", "fail_value", "pass_requires_all_of", "warning_only_bypass_to_pass_prohibited",
})

_EXPECTED_RESULT_SCHEMA_FIELDS = frozenset({
    "schema", "complete", "final", "identity_and_dependency_bindings",
    "repository_commit_tree_cleanliness", "implementation_hashes",
    "external_weight_bindings", "selected_image_ids",
    "runtime_derived_image_dimensions", "ordered_window_records",
    "window_count_and_digest", "forward_snapshot_graph_trajectory_update_counts",
    "image_scoped_stitch_finalize_counts",
    "t0_preservation_comparisons_per_window", "t0_t320_repeat_digests",
    "prediction_equality_results", "non_finite_counts_by_stage",
    "tensor_shapes_dtypes", "timing_telemetry", "rss_vram_telemetry",
    "scope_disclosures", "no_gt_no_metric_fields", "final_smoke_outcome",
})

# --- determinism_contract nested key sets (Defect 1 repair) ---

_DETERMINISM_CONTRACT_KEYS = frozenset({
    "description", "target_generation_repeats", "window_scoped_expected_counts",
    "image_scoped_expected_counts", "required_between_repeats", "claims",
})

_WINDOW_SCOPED_COUNTS_KEYS = frozenset({
    "unit", "model_forwards", "snapshots", "t0_sigmoid_conversions",
    "t320_graph_constructions", "t320_trajectories", "t320_recurrence_updates",
    "t320_sigmoid_conversions",
})

_IMAGE_SCOPED_COUNTS_KEYS = frozenset({
    "unit", "selected_image_count", "target_generation_repeat_count", "expression",
    "t0_stitch_calls", "t0_finalize_calls", "t320_stitch_calls", "t320_finalize_calls",
})

_REQUIRED_BETWEEN_REPEATS_KEYS = frozenset({
    "identical_window_order", "identical_t0_scientific_digest",
    "identical_t320_scientific_digest", "exact_prediction_equality",
    "zero_non_finite_values", "snapshot_bytes_tensors_unchanged",
})

_DETERMINISM_CLAIMS_KEYS = frozenset({"cross_device_bitwise_determinism_claim", "determinism_scope"})

# --- the four previously-unvalidated tables (Defect 2 repair) ---

_VALIDITY_CONTRACT_KEYS = frozenset({
    "require_finite", "require_valid_class_dimension", "require_valid_image_geometry",
    "require_valid_prediction_shape", "require_no_input_mutation",
    "require_no_empty_result", "require_no_out_of_range_window",
})

_EXPECTED_REQUIRE_FINITE_STAGES = frozenset({
    "raw S0", "graph values", "propagated scores", "sigmoid masks",
    "stitched scores", "final scores",
})

_SCOPE_ISOLATION_KEYS = frozenset({"explicit_prohibitions"})

_SCOPE_ISOLATION_PROHIBITION_KEYS = frozenset({
    "any_holdout_image", "remaining_46_discovery_images",
    "complete_96_image_population", "balp_350_image_development_subset",
    "confirmation_150_image_subset", "other_validation_images",
    "training_images", "training", "interventions", "adapter_learning",
    "confirmation_evaluation", "full_5000_image_evaluation",
    "publication_level_accuracy_claims",
})

_TELEMETRY_CONTRACT_KEYS = frozenset({
    "required_fields", "wall_timing_clock", "cuda_sync_requirement",
    "gpu_memory_stat_reset_point", "memory_ordering_requirement",
    "timing_non_negativity_requirement", "profiling_must_not_alter_scientific_digests",
    "full_dataset_runtime_estimation_from_this_smoke_prohibited",
})

_EXPECTED_TELEMETRY_REQUIRED_FIELDS = frozenset({
    "slurm_job_id", "node", "gpu_model", "cuda_runtime_version",
    "cuda_driver_version", "total_wall_time_ns", "preflight_wall_time_ns",
    "model_construction_and_checkpoint_loading_wall_time_ns",
    "dataset_image_preparation_wall_time_ns", "model_forward_wall_time_ns",
    "graph_construction_wall_time_ns", "recurrence_wall_time_ns",
    "stitching_finalization_wall_time_ns",
    "artifact_validation_install_wall_time_ns", "peak_host_rss_bytes",
    "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes", "gpu_total_memory_bytes",
})

_ARTIFACT_POLICY_KEYS = frozenset({
    "output_root", "fresh_destination_required", "overwrite_prohibited",
    "force_mode_prohibited", "staging_location",
    "staged_validation_before_installation_required", "installation_method",
    "reopen_after_installation_required",
    "public_verifier_rerun_after_installation_required",
    "success_message_before_post_install_verification_prohibited",
    "partial_artifact_as_final_prohibited", "failure_preserves_prior_state",
    "temporary_residue_removed_or_reported", "result_artifact_committed_to_git",
})

_EXPECTED_PASS_VALUE = "PASS_REAL_E3_T320_INTEGRATION_SMOKE"
_EXPECTED_FAIL_VALUE = "FAIL_REAL_E3_T320_INTEGRATION_SMOKE"

_EXPECTED_CROP = [448, 448]
_EXPECTED_STRIDE = [224, 224]
_EXPECTED_PATCH = [14, 14]
_EXPECTED_GPU = "1 x H100 80GB"
_EXPECTED_CPUS = 8
_EXPECTED_HOST_MEM_GB = 80
_EXPECTED_WALL_TIME = "00:30:00"
_EXPECTED_ACCOUNT = "rrg-yangw"

_SMOKE_JSON_KEYS = frozenset({
    "schema", "namespace", "source_manifest", "source_population_digest",
    "source_discovery_digest", "source_locked_holdout_digest",
    "source_discovery_rank_range", "source_locked_holdout_rank_range",
    "selection_rule", "selected_count", "selected_records",
    "selected_image_ids_ordered", "both_are_discovery_ids",
    "holdout_overlap_count", "duplicate_count", "canonical_record_encoding",
    "canonical_ordered_record_digest", "complete", "final",
})

_SMOKE_JSON_RECORD_KEYS = frozenset({"canonical_image_id", "selection_key", "selection_rank", "partition"})


# ---------------------------------------------------------------------------
# Top-level loading
# ---------------------------------------------------------------------------

def load_authority(repo_root: Path) -> dict:
    toml_path = resolve_repo_relative_path(repo_root, "research_specs/semantic_evidence_t320_real_smoke_v1.toml")
    data = load_toml(toml_path)
    _require_keys(data, _TOP_LEVEL_KEYS, "toml top-level")
    _require_keys(data["identity"], _IDENTITY_KEYS, "identity")
    _require_keys(data["status"], _STATUS_KEYS, "status")
    _require_keys(data["implementation_binding"], _IMPLEMENTATION_BINDING_KEYS, "implementation_binding")
    for index, entry in enumerate(_require_list(data["implementation_binding"]["file"], "implementation_binding.file")):
        _require_keys(entry, _IMPLEMENTATION_FILE_KEYS, f"implementation_binding.file[{index}]")
    _require_keys(
        data["implementation_binding"]["required_properties"],
        _IMPLEMENTATION_REQUIRED_PROPERTIES_KEYS,
        "implementation_binding.required_properties",
    )
    for index, entry in enumerate(_require_list(data["dependency"], "dependency")):
        _require_keys(entry, _DEPENDENCY_KEYS, f"dependency[{index}]")
    _require_keys(data["smoke_population"], _SMOKE_POPULATION_KEYS, "smoke_population")
    _require_keys(data["window_geometry"], _WINDOW_GEOMETRY_KEYS, "window_geometry")
    _require_keys(data["resource_contract"], _RESOURCE_CONTRACT_KEYS, "resource_contract")
    _require_keys(data["result_schema"], _RESULT_SCHEMA_KEYS, "result_schema")
    _require_keys(data["result_schema"]["outcomes"], _RESULT_SCHEMA_OUTCOMES_KEYS, "result_schema.outcomes")
    _require_keys(data["closure"], _CLOSURE_KEYS, "closure")

    _require_keys(data["determinism_contract"], _DETERMINISM_CONTRACT_KEYS, "determinism_contract")
    _require_keys(
        data["determinism_contract"]["window_scoped_expected_counts"],
        _WINDOW_SCOPED_COUNTS_KEYS, "determinism_contract.window_scoped_expected_counts",
    )
    _require_keys(
        data["determinism_contract"]["image_scoped_expected_counts"],
        _IMAGE_SCOPED_COUNTS_KEYS, "determinism_contract.image_scoped_expected_counts",
    )
    _require_keys(
        data["determinism_contract"]["required_between_repeats"],
        _REQUIRED_BETWEEN_REPEATS_KEYS, "determinism_contract.required_between_repeats",
    )
    _require_keys(data["determinism_contract"]["claims"], _DETERMINISM_CLAIMS_KEYS, "determinism_contract.claims")

    _require_keys(data["validity_contract"], _VALIDITY_CONTRACT_KEYS, "validity_contract")
    _require_keys(data["scope_isolation"], _SCOPE_ISOLATION_KEYS, "scope_isolation")
    _require_keys(
        data["scope_isolation"]["explicit_prohibitions"],
        _SCOPE_ISOLATION_PROHIBITION_KEYS, "scope_isolation.explicit_prohibitions",
    )
    _require_keys(data["telemetry_contract"], _TELEMETRY_CONTRACT_KEYS, "telemetry_contract")
    _require_keys(data["artifact_policy"], _ARTIFACT_POLICY_KEYS, "artifact_policy")

    for key in _STATUS_KEYS - {"smoke_execution_authorized_after"}:
        _require_bool(data["status"][key], f"status.{key}")
    for key in _CLOSURE_KEYS:
        _require_bool(data["closure"][key], f"closure.{key}")

    return data


def load_smoke_population(repo_root: Path) -> dict:
    json_path = resolve_repo_relative_path(repo_root, "evaluation_subsets/semantic_evidence_t320_real_smoke_2.json")
    data = load_json_strict(json_path)
    _require_keys(data, _SMOKE_JSON_KEYS, "smoke_2.json top-level")
    for index, record in enumerate(_require_list(data["selected_records"], "smoke_2.json selected_records")):
        _require_keys(record, _SMOKE_JSON_RECORD_KEYS, f"smoke_2.json selected_records[{index}]")
    _require_bool(data["both_are_discovery_ids"], "smoke_2.json both_are_discovery_ids")
    _require_bool(data["complete"], "smoke_2.json complete")
    _require_bool(data["final"], "smoke_2.json final")
    _require_exact_int(data["selected_count"], "smoke_2.json selected_count")
    _require_exact_int(data["holdout_overlap_count"], "smoke_2.json holdout_overlap_count")
    _require_exact_int(data["duplicate_count"], "smoke_2.json duplicate_count")
    return data


# ---------------------------------------------------------------------------
# Repository state / ancestry verification
# ---------------------------------------------------------------------------

def verify_repository_state(repo_root: Path, toml_data: dict) -> dict:
    branch = git_current_branch(repo_root)
    head = git_head_commit(repo_root)
    required_branch = _require_str(toml_data["implementation_binding"]["required_branch"], "implementation_binding.required_branch")
    introducing_commit = _require_full_commit_hex(
        toml_data["implementation_binding"]["introducing_commit"], "implementation_binding.introducing_commit"
    )

    if branch != required_branch:
        raise T320RealSmokeContractError(f"branch mismatch: expected {required_branch!r}, got {branch!r}")

    is_same = head == introducing_commit
    is_descendant = (not is_same) and git_is_ancestor(repo_root, introducing_commit, head)
    if not (is_same or is_descendant):
        raise T320RealSmokeContractError(
            f"committed implementation commit {introducing_commit} is not HEAD ({head}) and not an ancestor of HEAD"
        )

    if not git_tracked_tree_clean(repo_root):
        raise T320RealSmokeContractError("tracked working tree is not clean (tracked file modified or a change is staged)")

    return {"branch": branch, "head": head, "introducing_commit": introducing_commit, "ancestry_ok": True, "clean": True}


def verify_implementation_hashes(repo_root: Path, toml_data: dict) -> dict:
    head = git_head_commit(repo_root)
    results = []
    for entry in toml_data["implementation_binding"]["file"]:
        path = _require_str(entry["path"], "implementation_binding.file.path")
        expected_sha256 = _require_sha256_hex(entry["sha256"], f"implementation_binding.file[{path}].sha256")
        resolved = resolve_repo_relative_path(repo_root, path)
        if not resolved.is_file():
            raise T320RealSmokeContractError(f"implementation file does not exist: {path}")
        working_tree_hash = sha256_file(resolved)
        committed_hash = git_blob_sha256(repo_root, head, path)
        if working_tree_hash != expected_sha256:
            raise T320RealSmokeContractError(f"implementation file hash mismatch (working tree) for {path}: expected {expected_sha256}, got {working_tree_hash}")
        if committed_hash != expected_sha256:
            raise T320RealSmokeContractError(f"implementation file hash mismatch (committed HEAD) for {path}: expected {expected_sha256}, got {committed_hash}")
        results.append({"path": path, "sha256": expected_sha256, "working_tree_matches": True, "committed_matches": True})
    return {"files": results, "all_match": True}


def verify_dependency_hashes(repo_root: Path, toml_data: dict) -> dict:
    """Verifies every [[dependency]] entry's working-tree hash. Most
    dependency entries are pre-existing, already-committed files (their
    working-tree hash matching is sufficient given verify_repository_state
    already confirmed the tracked tree is clean, i.e. working tree ==
    committed content for every tracked path). One dependency
    (t320_real_smoke_2_population) is a sibling file of this very authority,
    created and verified together with it but not yet committed at contract-
    authoring time -- so committed-blob verification is intentionally not
    required here (unlike verify_implementation_hashes, which binds an
    already-committed prior task's output and DOES require committed-blob
    proof)."""
    results = []
    for entry in toml_data["dependency"]:
        name = _require_str(entry["name"], "dependency.name")
        path = _require_str(entry["path"], f"dependency[{name}].path")
        expected_sha256 = _require_sha256_hex(entry["sha256"], f"dependency[{name}].sha256")
        resolved = resolve_repo_relative_path(repo_root, path)
        if not resolved.is_file():
            raise T320RealSmokeContractError(f"dependency file does not exist: {path} ({name})")
        working_tree_hash = sha256_file(resolved)
        if working_tree_hash != expected_sha256:
            raise T320RealSmokeContractError(
                f"dependency hash mismatch for {name} ({path}): expected {expected_sha256}, got {working_tree_hash}"
            )
        results.append({"name": name, "path": path, "sha256": expected_sha256})
    return {"dependencies": results, "all_match": True}


# ---------------------------------------------------------------------------
# Two-image population verification
# ---------------------------------------------------------------------------

def _compute_selection_digest(image_ids: list) -> str:
    data = b"".join(i.encode("ascii") + b"\n" for i in image_ids)
    return hashlib.sha256(data).hexdigest()


def verify_two_image_population(repo_root: Path, toml_data: dict, population_data: dict) -> dict:
    source_manifest_rel = _require_str(population_data["source_manifest"]["path"], "smoke_2.json source_manifest.path")
    expected_source_sha256 = _require_sha256_hex(population_data["source_manifest"]["sha256"], "smoke_2.json source_manifest.sha256")
    source_path = resolve_repo_relative_path(repo_root, source_manifest_rel)
    if not source_path.is_file():
        raise T320RealSmokeContractError(f"source 96-image manifest does not exist: {source_manifest_rel}")
    actual_source_sha256 = sha256_file(source_path)
    if actual_source_sha256 != expected_source_sha256:
        raise T320RealSmokeContractError(
            f"source 96-image manifest hash mismatch: expected {expected_source_sha256}, got {actual_source_sha256}"
        )

    source_data = load_json_strict(source_path)
    records = _require_list(source_data["records"], "source manifest records")
    if len(records) != 96:
        raise T320RealSmokeContractError(f"source manifest must contain exactly 96 records, got {len(records)}")

    ranks = [r["selection_rank"] for r in records]
    if ranks != sorted(ranks) or sorted(ranks) != list(range(96)):
        raise T320RealSmokeContractError("source manifest records are not in canonical ascending selection_rank order 0..95")

    by_rank = {r["selection_rank"]: r for r in records}
    expected_ids = [by_rank[0]["canonical_image_id"], by_rank[1]["canonical_image_id"]]
    for rank in (0, 1):
        if by_rank[rank]["partition"] != "discovery":
            raise T320RealSmokeContractError(f"selection_rank {rank} record is not partition=='discovery'")

    declared_ids = _require_list(population_data["selected_image_ids_ordered"], "smoke_2.json selected_image_ids_ordered")
    if declared_ids != expected_ids:
        raise T320RealSmokeContractError(
            f"declared selected_image_ids_ordered {declared_ids!r} does not match reconstructed first-two-discovery-order ids {expected_ids!r}"
        )

    if len(set(declared_ids)) != len(declared_ids):
        raise T320RealSmokeContractError("duplicate image id present in selected_image_ids_ordered")

    holdout_ids = {r["canonical_image_id"] for r in records if r["partition"] == "locked_holdout"}
    if holdout_ids & set(declared_ids):
        raise T320RealSmokeContractError("a holdout image id is present in the smoke population")

    if _require_exact_int(population_data["selected_count"], "smoke_2.json selected_count") != 2:
        raise T320RealSmokeContractError("smoke_2.json selected_count must be exactly 2")
    if _require_exact_int(population_data["holdout_overlap_count"], "smoke_2.json holdout_overlap_count") != 0:
        raise T320RealSmokeContractError("smoke_2.json holdout_overlap_count must be exactly 0")
    if _require_exact_int(population_data["duplicate_count"], "smoke_2.json duplicate_count") != 0:
        raise T320RealSmokeContractError("smoke_2.json duplicate_count must be exactly 0")
    if population_data["both_are_discovery_ids"] is not True:
        raise T320RealSmokeContractError("smoke_2.json both_are_discovery_ids must be true")

    expected_digest = _compute_selection_digest(expected_ids)
    declared_digest = _require_sha256_hex(population_data["canonical_ordered_record_digest"], "smoke_2.json canonical_ordered_record_digest")
    if declared_digest != expected_digest:
        raise T320RealSmokeContractError(
            f"canonical_ordered_record_digest mismatch: recomputed {expected_digest}, declared {declared_digest}"
        )

    toml_ids = _require_list(toml_data["smoke_population"]["selected_image_ids_ordered"], "toml smoke_population.selected_image_ids_ordered")
    if toml_ids != expected_ids:
        raise T320RealSmokeContractError(f"toml smoke_population.selected_image_ids_ordered {toml_ids!r} does not match reconstructed ids {expected_ids!r}")
    if _require_exact_int(toml_data["smoke_population"]["selected_count"], "toml smoke_population.selected_count") != 2:
        raise T320RealSmokeContractError("toml smoke_population.selected_count must be exactly 2")
    for flag in ("image_pixel_access_by_this_task", "image_metadata_access_by_this_task", "label_or_annotation_access_by_this_task"):
        if toml_data["smoke_population"][flag] is not False:
            raise T320RealSmokeContractError(f"toml smoke_population.{flag} must be false (this contract task never accesses image content)")

    return {"reconstructed_ids": expected_ids, "declared_ids": declared_ids, "match": True, "digest": expected_digest}


# ---------------------------------------------------------------------------
# Window geometry verification
# ---------------------------------------------------------------------------

def verify_window_geometry(toml_data: dict) -> dict:
    wg = toml_data["window_geometry"]
    if _require_list(wg["crop_pixels"], "window_geometry.crop_pixels") != _EXPECTED_CROP:
        raise T320RealSmokeContractError(f"crop_pixels must be {_EXPECTED_CROP}, got {wg['crop_pixels']!r}")
    if _require_list(wg["stride_pixels"], "window_geometry.stride_pixels") != _EXPECTED_STRIDE:
        raise T320RealSmokeContractError(f"stride_pixels must be {_EXPECTED_STRIDE}, got {wg['stride_pixels']!r}")
    if _require_list(wg["patch_size_pixels"], "window_geometry.patch_size_pixels") != _EXPECTED_PATCH:
        raise T320RealSmokeContractError(f"patch_size_pixels must be {_EXPECTED_PATCH}, got {wg['patch_size_pixels']!r}")
    if wg["window_count_status"] != "RUNTIME_DERIVED_AND_RECORDED":
        raise T320RealSmokeContractError(f"window_count_status must be 'RUNTIME_DERIVED_AND_RECORDED', got {wg['window_count_status']!r}")
    if _require_bool(wg["window_count_fabrication_prohibited"], "window_geometry.window_count_fabrication_prohibited") is not True:
        raise T320RealSmokeContractError("window_count_fabrication_prohibited must be true")
    if _require_bool(wg["debug_subset_or_max_window_override_prohibited"], "window_geometry.debug_subset_or_max_window_override_prohibited") is not True:
        raise T320RealSmokeContractError("debug_subset_or_max_window_override_prohibited must be true")
    return {"crop": wg["crop_pixels"], "stride": wg["stride_pixels"], "patch": wg["patch_size_pixels"], "window_count_status": wg["window_count_status"]}


# ---------------------------------------------------------------------------
# Resource / scope boundary verification
# ---------------------------------------------------------------------------

def verify_resource_contract(toml_data: dict) -> dict:
    rc = toml_data["resource_contract"]
    checks = {
        "gpu": (_require_str(rc["gpu"], "resource_contract.gpu"), _EXPECTED_GPU),
        "cpus": (_require_exact_int(rc["cpus"], "resource_contract.cpus"), _EXPECTED_CPUS),
        "host_memory_max_gb": (_require_exact_int(rc["host_memory_max_gb"], "resource_contract.host_memory_max_gb"), _EXPECTED_HOST_MEM_GB),
        "wall_time_limit": (_require_str(rc["wall_time_limit"], "resource_contract.wall_time_limit"), _EXPECTED_WALL_TIME),
        "account": (_require_str(rc["account"], "resource_contract.account"), _EXPECTED_ACCOUNT),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            raise T320RealSmokeContractError(f"resource_contract.{field} must be {expected!r}, got {actual!r}")

    if _require_exact_int(rc["job_count_max"], "resource_contract.job_count_max") != 1:
        raise T320RealSmokeContractError("resource_contract.job_count_max must be exactly 1")
    for flag in ("fresh_process_required", "multi_job_sharding_prohibited",
                 "overwrite_or_force_mode_prohibited", "self_submission_prohibited"):
        if _require_bool(rc[flag], f"resource_contract.{flag}") is not True:
            raise T320RealSmokeContractError(f"resource_contract.{flag} must be true")
    if _require_bool(rc["resume_requirement"], "resource_contract.resume_requirement") is not False:
        raise T320RealSmokeContractError("resource_contract.resume_requirement must be false")
    return {"resource_contract_ok": True}


def verify_scope_boundaries(toml_data: dict) -> dict:
    gt = toml_data["gt_isolation"]
    scope_forbidden_bools = [
        "coco_annotations_access_authorized", "segmentation_labels_access_authorized",
        "ignore_masks_access_authorized", "class_correctness_computation_authorized",
        "iou_computation_authorized", "miou_computation_authorized",
        "macc_computation_authorized", "pixel_accuracy_computation_authorized",
        "discovery_or_holdout_performance_computation_authorized",
        "t0_vs_t320_accuracy_improvement_claim_authorized",
    ]
    for field in scope_forbidden_bools:
        if _require_bool(gt[field], f"gt_isolation.{field}") is not False:
            raise T320RealSmokeContractError(f"gt_isolation.{field} must be false")

    closure = toml_data["closure"]
    for field in (
        "implements_smoke_runner", "executes_real_inference", "accesses_image_pixels",
        "accesses_annotations_or_gt", "loads_checkpoint_tensors", "initializes_cuda",
        "submits_job", "implements_interventions", "executes_96_image_feasibility_study",
        "trains_anything", "evaluates_accuracy_or_miou", "stages_or_commits_files",
    ):
        if closure[field] is not False:
            raise T320RealSmokeContractError(f"closure.{field} must be false")
    for field in ("complete", "final", "contract_only"):
        if closure[field] is not True:
            raise T320RealSmokeContractError(f"closure.{field} must be true")

    if toml_data["status"]["smoke_execution_authorized_by_this_task"] is not False:
        raise T320RealSmokeContractError("status.smoke_execution_authorized_by_this_task must be false")

    return {"scope_ok": True}


def verify_result_schema(toml_data: dict) -> dict:
    rs = toml_data["result_schema"]
    fields = frozenset(_require_list(rs["required_top_level_fields"], "result_schema.required_top_level_fields"))
    missing = _EXPECTED_RESULT_SCHEMA_FIELDS - fields
    if missing:
        raise T320RealSmokeContractError(f"result_schema.required_top_level_fields is missing: {sorted(missing)}")
    outcomes = rs["outcomes"]
    if outcomes["pass_value"] != _EXPECTED_PASS_VALUE:
        raise T320RealSmokeContractError(f"result_schema.outcomes.pass_value must be {_EXPECTED_PASS_VALUE!r}")
    if outcomes["fail_value"] != _EXPECTED_FAIL_VALUE:
        raise T320RealSmokeContractError(f"result_schema.outcomes.fail_value must be {_EXPECTED_FAIL_VALUE!r}")
    if _require_bool(outcomes["warning_only_bypass_to_pass_prohibited"], "result_schema.outcomes.warning_only_bypass_to_pass_prohibited") is not True:
        raise T320RealSmokeContractError("result_schema.outcomes.warning_only_bypass_to_pass_prohibited must be true")
    return {"result_schema_ok": True}


@_no_leak
def verify_determinism_and_count_contract(toml_data: dict) -> dict:
    dc = toml_data["determinism_contract"]
    if _require_exact_int(dc["target_generation_repeats"], "determinism_contract.target_generation_repeats") != 2:
        raise T320RealSmokeContractError("determinism_contract.target_generation_repeats must be exactly 2")

    # --- window-scoped counts (unchanged formulas, expressed in terms of N and R) ---
    window_counts = dc["window_scoped_expected_counts"]
    expected_window_counts = {
        "model_forwards": "N",
        "snapshots": "N",
        "t0_sigmoid_conversions": "N*R",
        "t320_graph_constructions": "N*R",
        "t320_trajectories": "N*R",
        "t320_recurrence_updates": "N*R*320",
        "t320_sigmoid_conversions": "N*R",
    }
    for field, expected in expected_window_counts.items():
        actual = window_counts[field]
        if actual != expected:
            raise T320RealSmokeContractError(
                f"determinism_contract.window_scoped_expected_counts.{field} must be {expected!r}, got {actual!r}"
            )

    # --- image-scoped counts (Defect 1 repair): DERIVE the expected total from
    # M (selected_image_count) and R (repeat count), never trust a copied
    # literal. M and R are independently cross-checked against
    # smoke_population.selected_count and determinism_contract's own
    # target_generation_repeats (Part 8 relational checks). ---
    M = _require_exact_int(toml_data["smoke_population"]["selected_count"], "smoke_population.selected_count")
    R = _require_exact_int(dc["target_generation_repeats"], "determinism_contract.target_generation_repeats")

    image_counts = dc["image_scoped_expected_counts"]
    declared_M = _require_exact_int(image_counts["selected_image_count"], "determinism_contract.image_scoped_expected_counts.selected_image_count")
    declared_R = _require_exact_int(image_counts["target_generation_repeat_count"], "determinism_contract.image_scoped_expected_counts.target_generation_repeat_count")
    if declared_M != M:
        raise T320RealSmokeContractError(
            f"determinism_contract.image_scoped_expected_counts.selected_image_count ({declared_M}) "
            f"must equal smoke_population.selected_count ({M})"
        )
    if declared_R != R:
        raise T320RealSmokeContractError(
            f"determinism_contract.image_scoped_expected_counts.target_generation_repeat_count ({declared_R}) "
            f"must equal determinism_contract.target_generation_repeats ({R})"
        )
    if image_counts["expression"] != "selected_image_count * target_generation_repeat_count":
        raise T320RealSmokeContractError(
            "determinism_contract.image_scoped_expected_counts.expression must be exactly "
            "'selected_image_count * target_generation_repeat_count'"
        )
    if image_counts["unit"] != "per_image_per_repeat":
        raise T320RealSmokeContractError("determinism_contract.image_scoped_expected_counts.unit must be 'per_image_per_repeat'")

    derived_total = M * R  # the ONLY source of truth for the expected campaign total
    for field in ("t0_stitch_calls", "t0_finalize_calls", "t320_stitch_calls", "t320_finalize_calls"):
        declared = _require_exact_int(image_counts[field], f"determinism_contract.image_scoped_expected_counts.{field}")
        if declared != derived_total:
            raise T320RealSmokeContractError(
                f"determinism_contract.image_scoped_expected_counts.{field} must equal "
                f"selected_image_count * target_generation_repeat_count = {M}*{R} = {derived_total}, got {declared}"
            )
    # T0 and T320 counts, and stitch vs. finalize counts, must be independently
    # present as distinct fields (not merged/aliased) -- enforced structurally
    # by _IMAGE_SCOPED_COUNTS_KEYS already requiring all four distinct keys.

    for field in (
        "identical_window_order", "identical_t0_scientific_digest",
        "identical_t320_scientific_digest", "exact_prediction_equality",
        "zero_non_finite_values", "snapshot_bytes_tensors_unchanged",
    ):
        if _require_bool(dc["required_between_repeats"][field], f"determinism_contract.required_between_repeats.{field}") is not True:
            raise T320RealSmokeContractError(f"determinism_contract.required_between_repeats.{field} must be true")
    if _require_bool(dc["claims"]["cross_device_bitwise_determinism_claim"], "determinism_contract.claims.cross_device_bitwise_determinism_claim") is not False:
        raise T320RealSmokeContractError("determinism_contract.claims.cross_device_bitwise_determinism_claim must be false")

    t320c = toml_data["t320_execution_contract"]
    if _require_exact_int(t320c["updates_per_trajectory"], "t320_execution_contract.updates_per_trajectory") != 320:
        raise T320RealSmokeContractError("t320_execution_contract.updates_per_trajectory must be exactly 320")
    if _require_exact_int(t320c["sigmoid_occurrences_per_window_per_repeat"], "t320_execution_contract.sigmoid_occurrences_per_window_per_repeat") != 1:
        raise T320RealSmokeContractError("t320_execution_contract.sigmoid_occurrences_per_window_per_repeat must be exactly 1")
    if _require_bool(t320c["internal_interpolation_align_corners"], "t320_execution_contract.internal_interpolation_align_corners") is not True:
        raise T320RealSmokeContractError("t320_execution_contract.internal_interpolation_align_corners must be true")
    if _require_bool(t320c["final_rescale_align_corners"], "t320_execution_contract.final_rescale_align_corners") is not False:
        raise T320RealSmokeContractError("t320_execution_contract.final_rescale_align_corners must be false")

    preservation = toml_data["preservation_contract"]
    if preservation["comparison_strength"] != "bitwise_equal":
        raise T320RealSmokeContractError("preservation_contract.comparison_strength must be 'bitwise_equal'")

    one_forward = toml_data["one_forward_contract"]
    for field in ("real_e3_model_forwards_equals_N", "adapter_snapshots_equals_N"):
        if _require_bool(one_forward[field], f"one_forward_contract.{field}") is not True:
            raise T320RealSmokeContractError(f"one_forward_contract.{field} must be true")
    for field in ("second_e3_forward_prohibited", "candidate_specific_or_t320_specific_forward_prohibited", "masked_forward_or_component_path_prohibited"):
        if _require_bool(one_forward[field], f"one_forward_contract.{field}") is not True:
            raise T320RealSmokeContractError(f"one_forward_contract.{field} must be true")

    return {"determinism_and_counts_ok": True}


def verify_external_weight_bindings(toml_data: dict) -> dict:
    ew = toml_data["external_weights"]
    for key in ("dino_backbone_checkpoint", "clip_text_checkpoint", "e3_projection_checkpoint"):
        entry = ew[key]
        _require_str(entry["filename"], f"external_weights.{key}.filename")
        _require_sha256_hex(entry["sha256"], f"external_weights.{key}.sha256")
    if ew["e3_projection_checkpoint"]["sha256"] != "9632117c18613674eeddcb35f86193d37da203a9cbd837cf500c4121b5fa9942":
        raise T320RealSmokeContractError("e3_projection_checkpoint.sha256 does not match the known frozen value")
    if _require_bool(ew["checkpoint_tensor_loading_by_this_task"], "external_weights.checkpoint_tensor_loading_by_this_task") is not False:
        raise T320RealSmokeContractError("external_weights.checkpoint_tensor_loading_by_this_task must be false")
    if _require_bool(ew["cuda_initialization_by_this_task"], "external_weights.cuda_initialization_by_this_task") is not False:
        raise T320RealSmokeContractError("external_weights.cuda_initialization_by_this_task must be false")
    if _require_exact_int(ew["torch_load_calls_by_this_task"], "external_weights.torch_load_calls_by_this_task") != 0:
        raise T320RealSmokeContractError("external_weights.torch_load_calls_by_this_task must be exactly 0")
    if _require_bool(ew["preflight_requirement"]["rehash_before_model_construction"], "external_weights.preflight_requirement.rehash_before_model_construction") is not True:
        raise T320RealSmokeContractError("external_weights.preflight_requirement.rehash_before_model_construction must be true")
    return {"external_weights_ok": True}


@_no_leak
def verify_identity(toml_data: dict) -> dict:
    """identity's key-set is checked by load_authority; this additionally
    checks its VALUES, which were previously never validated."""
    idn = toml_data["identity"]
    if idn["schema_name"] != "semantic_evidence_t320_real_smoke_authority":
        raise T320RealSmokeContractError(f"identity.schema_name must be 'semantic_evidence_t320_real_smoke_authority', got {idn['schema_name']!r}")
    if idn["schema_version"] != "1.0.0":
        raise T320RealSmokeContractError(f"identity.schema_version must be '1.0.0', got {idn['schema_version']!r}")
    _require_str(idn["created_by_task"], "identity.created_by_task")
    successors = _require_list(idn["successor_to"], "identity.successor_to")
    if "research_specs/semantic_evidence_feasibility_v1.toml" not in successors:
        raise T320RealSmokeContractError("identity.successor_to must include the feasibility contract")
    if "research_specs/semantic_evidence_t320_source_v1.toml" not in successors:
        raise T320RealSmokeContractError("identity.successor_to must include the T320 source authority")
    return {"identity_ok": True}


# ---------------------------------------------------------------------------
# Defect 2 repair: dedicated validators for the four tables that were
# previously unvalidated (present-as-a-key only, content never read).
# ---------------------------------------------------------------------------

@_no_leak
def _verify_validity_contract(toml_data: dict) -> dict:
    vc = toml_data["validity_contract"]
    stages = frozenset(_require_list(vc["require_finite"], "validity_contract.require_finite"))
    if stages != _EXPECTED_REQUIRE_FINITE_STAGES:
        raise T320RealSmokeContractError(
            f"validity_contract.require_finite must be exactly {sorted(_EXPECTED_REQUIRE_FINITE_STAGES)}, got {sorted(stages)}"
        )
    for field in (
        "require_valid_class_dimension", "require_valid_image_geometry",
        "require_valid_prediction_shape", "require_no_input_mutation",
        "require_no_empty_result", "require_no_out_of_range_window",
    ):
        if _require_bool(vc[field], f"validity_contract.{field}") is not True:
            raise T320RealSmokeContractError(f"validity_contract.{field} must be true (a failed mandatory validity check must prevent PASS)")

    # Cross-check against the result schema: the stages this table requires
    # finite must be observable in the frozen result schema's
    # non_finite_counts_by_stage field, and the zero-non-finite requirement
    # here must agree with determinism_contract's own zero_non_finite_values
    # requirement between repeats (both describe the same invariant).
    if "non_finite_counts_by_stage" not in toml_data["result_schema"]["required_top_level_fields"]:
        raise T320RealSmokeContractError(
            "validity_contract.require_finite stages are declared but result_schema.required_top_level_fields "
            "has no non_finite_counts_by_stage field to report them in"
        )
    if toml_data["determinism_contract"]["required_between_repeats"]["zero_non_finite_values"] is not True:
        raise T320RealSmokeContractError(
            "validity_contract's finite-value requirements are inconsistent with "
            "determinism_contract.required_between_repeats.zero_non_finite_values"
        )
    return {"validity_contract_ok": True, "required_finite_stages": sorted(stages)}


@_no_leak
def _verify_scope_isolation(toml_data: dict) -> dict:
    si = toml_data["scope_isolation"]["explicit_prohibitions"]
    for field in _SCOPE_ISOLATION_PROHIBITION_KEYS:
        value = _require_str(si[field], f"scope_isolation.explicit_prohibitions.{field}")
        if value != "prohibited":
            raise T320RealSmokeContractError(
                f"scope_isolation.explicit_prohibitions.{field} must be exactly 'prohibited', got {value!r}"
            )

    # Cross-check against gt_isolation, closure, status, and result_schema so
    # no table may say "prohibited" while another silently permits the same
    # action.
    gt = toml_data["gt_isolation"]
    closure = toml_data["closure"]
    status = toml_data["status"]
    cross_checks = [
        # (scope_isolation leaf implying the same restriction, other table[field], must equal)
        (gt["coco_annotations_access_authorized"], "gt_isolation.coco_annotations_access_authorized", False),
        (gt["miou_computation_authorized"], "gt_isolation.miou_computation_authorized", False),
        (gt["iou_computation_authorized"], "gt_isolation.iou_computation_authorized", False),
        (gt["macc_computation_authorized"], "gt_isolation.macc_computation_authorized", False),
        (gt["pixel_accuracy_computation_authorized"], "gt_isolation.pixel_accuracy_computation_authorized", False),
        (gt["discovery_or_holdout_performance_computation_authorized"], "gt_isolation.discovery_or_holdout_performance_computation_authorized", False),
        (closure["implements_interventions"], "closure.implements_interventions", False),
        (closure["executes_96_image_feasibility_study"], "closure.executes_96_image_feasibility_study", False),
        (closure["trains_anything"], "closure.trains_anything", False),
        (closure["evaluates_accuracy_or_miou"], "closure.evaluates_accuracy_or_miou", False),
        (closure["accesses_annotations_or_gt"], "closure.accesses_annotations_or_gt", False),
        (status["smoke_execution_authorized_by_this_task"], "status.smoke_execution_authorized_by_this_task", False),
    ]
    for actual, label, expected in cross_checks:
        if actual is not expected:
            raise T320RealSmokeContractError(
                f"{label} ({actual!r}) contradicts scope_isolation.explicit_prohibitions (expected {expected!r})"
            )
    if "no_gt_no_metric_fields" not in toml_data["result_schema"]["required_top_level_fields"]:
        raise T320RealSmokeContractError(
            "scope_isolation prohibits GT/metric access but result_schema.required_top_level_fields "
            "has no no_gt_no_metric_fields field to disclose it"
        )
    return {"scope_isolation_ok": True}


@_no_leak
def _verify_telemetry_contract(toml_data: dict) -> dict:
    tc = toml_data["telemetry_contract"]
    fields = frozenset(_require_list(tc["required_fields"], "telemetry_contract.required_fields"))
    if fields != _EXPECTED_TELEMETRY_REQUIRED_FIELDS:
        missing = _EXPECTED_TELEMETRY_REQUIRED_FIELDS - fields
        extra = fields - _EXPECTED_TELEMETRY_REQUIRED_FIELDS
        raise T320RealSmokeContractError(
            f"telemetry_contract.required_fields mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if tc["wall_timing_clock"] != "time.perf_counter_ns":
        raise T320RealSmokeContractError("telemetry_contract.wall_timing_clock must be exactly 'time.perf_counter_ns'")
    if "torch.cuda.synchronize()" not in _require_str(tc["cuda_sync_requirement"], "telemetry_contract.cuda_sync_requirement"):
        raise T320RealSmokeContractError("telemetry_contract.cuda_sync_requirement must name torch.cuda.synchronize()")
    if "before" not in tc["cuda_sync_requirement"] or "after" not in tc["cuda_sync_requirement"]:
        raise T320RealSmokeContractError("telemetry_contract.cuda_sync_requirement must require sync both before AND after each measured GPU region")
    if "reset_peak_memory_stats" not in _require_str(tc["gpu_memory_stat_reset_point"], "telemetry_contract.gpu_memory_stat_reset_point"):
        raise T320RealSmokeContractError("telemetry_contract.gpu_memory_stat_reset_point must name reset_peak_memory_stats")
    if "<=" not in _require_str(tc["memory_ordering_requirement"], "telemetry_contract.memory_ordering_requirement"):
        raise T320RealSmokeContractError("telemetry_contract.memory_ordering_requirement must state an allocated<=reserved<=total ordering")
    if "non-negative" not in _require_str(tc["timing_non_negativity_requirement"], "telemetry_contract.timing_non_negativity_requirement"):
        raise T320RealSmokeContractError("telemetry_contract.timing_non_negativity_requirement must require non-negative timings")
    if _require_bool(tc["profiling_must_not_alter_scientific_digests"], "telemetry_contract.profiling_must_not_alter_scientific_digests") is not True:
        raise T320RealSmokeContractError("telemetry_contract.profiling_must_not_alter_scientific_digests must be true")
    if _require_bool(tc["full_dataset_runtime_estimation_from_this_smoke_prohibited"], "telemetry_contract.full_dataset_runtime_estimation_from_this_smoke_prohibited") is not True:
        raise T320RealSmokeContractError("telemetry_contract.full_dataset_runtime_estimation_from_this_smoke_prohibited must be true")

    if "timing_telemetry" not in toml_data["result_schema"]["required_top_level_fields"]:
        raise T320RealSmokeContractError("telemetry_contract requires timing fields but result_schema has no timing_telemetry field")
    if "rss_vram_telemetry" not in toml_data["result_schema"]["required_top_level_fields"]:
        raise T320RealSmokeContractError("telemetry_contract requires memory fields but result_schema has no rss_vram_telemetry field")
    return {"telemetry_contract_ok": True}


@_no_leak
def _verify_artifact_policy(toml_data: dict) -> dict:
    ap = toml_data["artifact_policy"]
    if ap["output_root"] != "scratch-only":
        raise T320RealSmokeContractError("artifact_policy.output_root must be exactly 'scratch-only'")
    if ap["staging_location"] != "sibling temporary directory":
        raise T320RealSmokeContractError("artifact_policy.staging_location must be exactly 'sibling temporary directory'")
    if ap["installation_method"] != "atomic directory rename":
        raise T320RealSmokeContractError("artifact_policy.installation_method must be exactly 'atomic directory rename'")
    for field in (
        "fresh_destination_required", "overwrite_prohibited", "force_mode_prohibited",
        "staged_validation_before_installation_required", "reopen_after_installation_required",
        "public_verifier_rerun_after_installation_required",
        "success_message_before_post_install_verification_prohibited",
        "partial_artifact_as_final_prohibited", "failure_preserves_prior_state",
        "temporary_residue_removed_or_reported",
    ):
        if _require_bool(ap[field], f"artifact_policy.{field}") is not True:
            raise T320RealSmokeContractError(f"artifact_policy.{field} must be true")
    if _require_bool(ap["result_artifact_committed_to_git"], "artifact_policy.result_artifact_committed_to_git") is not False:
        raise T320RealSmokeContractError("artifact_policy.result_artifact_committed_to_git must be false")

    # Cross-check against resource_contract's own overwrite/force prohibition
    # -- both tables express the same restriction and must agree.
    if toml_data["resource_contract"]["overwrite_or_force_mode_prohibited"] is not True:
        raise T320RealSmokeContractError(
            "artifact_policy.overwrite_prohibited/force_mode_prohibited contradicts "
            "resource_contract.overwrite_or_force_mode_prohibited"
        )
    return {"artifact_policy_ok": True}


class _TopLevelAccessTrackingDict(dict):
    """A dict subclass that records every top-level key actually read via
    __getitem__/get during a real verify_all() run. This is RUNTIME PROOF
    that a table's contents were consumed by some validator -- not merely a
    static claim -- and is what the coverage test in Part 7 inspects (via
    `accessed_top_level_keys`), rather than grepping this module's source
    text for table names."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed_top_level_keys = set()

    def __getitem__(self, key):
        self.accessed_top_level_keys.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.accessed_top_level_keys.add(key)
        return super().get(key, default)


# ---------------------------------------------------------------------------
# Top-level table validation registry (Part 7): proves every one of the 19
# required top-level tables is consumed by some validator, via an explicit
# mapping rather than a fragile source-text grep.
# ---------------------------------------------------------------------------

_TABLE_VALIDATOR_REGISTRY = {
    "identity": "load_authority (key-set + presence)",
    "status": "load_authority (key-set + bool checks) / verify_scope_boundaries",
    "implementation_binding": "load_authority (key-set) / verify_repository_state / verify_implementation_hashes",
    "dependency": "load_authority (key-set) / verify_dependency_hashes",
    "external_weights": "load_authority (top-level presence) / verify_external_weight_bindings",
    "smoke_population": "load_authority (key-set) / verify_two_image_population / verify_determinism_and_count_contract",
    "window_geometry": "load_authority (key-set) / verify_window_geometry",
    "one_forward_contract": "load_authority (top-level presence) / verify_determinism_and_count_contract",
    "preservation_contract": "load_authority (top-level presence) / verify_determinism_and_count_contract",
    "t320_execution_contract": "load_authority (top-level presence) / verify_determinism_and_count_contract",
    "determinism_contract": "load_authority (key-set, nested) / verify_determinism_and_count_contract",
    "validity_contract": "load_authority (key-set) / _verify_validity_contract",
    "gt_isolation": "load_authority (top-level presence) / verify_scope_boundaries / _verify_scope_isolation",
    "scope_isolation": "load_authority (key-set, nested) / _verify_scope_isolation",
    "resource_contract": "load_authority (key-set) / verify_resource_contract / _verify_artifact_policy",
    "telemetry_contract": "load_authority (key-set) / _verify_telemetry_contract",
    "result_schema": "load_authority (key-set) / verify_result_schema",
    "artifact_policy": "load_authority (key-set) / _verify_artifact_policy",
    "closure": "load_authority (key-set + bool checks) / verify_scope_boundaries / _verify_scope_isolation",
}


def get_validated_table_registry() -> dict:
    """Returns a copy of the explicit table-name -> validator-description
    mapping. Used by the coverage test to prove every required top-level
    table (see _TOP_LEVEL_KEYS) has at least one real validator responsible
    for its contents -- not merely a presence check."""
    return dict(_TABLE_VALIDATOR_REGISTRY)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def verify_all(repo_root: Path) -> dict:
    """Runs every check in this module and returns a combined result dict.
    Raises T320RealSmokeContractError on the first violation found (fail
    closed); never authorizes real execution regardless of outcome.

    verify_all does NOT return verified=True unless every one of the 19
    required top-level tables (_TOP_LEVEL_KEYS) was actually read by some
    validator during THIS run -- proven at runtime via
    _TopLevelAccessTrackingDict, not merely claimed by the static
    _TABLE_VALIDATOR_REGISTRY."""
    toml_data = _TopLevelAccessTrackingDict(load_authority(repo_root))
    population_data = load_smoke_population(repo_root)

    identity = verify_identity(toml_data)
    repo_state = verify_repository_state(repo_root, toml_data)
    implementation = verify_implementation_hashes(repo_root, toml_data)
    dependencies = verify_dependency_hashes(repo_root, toml_data)
    population = verify_two_image_population(repo_root, toml_data, population_data)
    window_geometry = verify_window_geometry(toml_data)
    resource = verify_resource_contract(toml_data)
    scope = verify_scope_boundaries(toml_data)
    result_schema = verify_result_schema(toml_data)
    determinism = verify_determinism_and_count_contract(toml_data)
    external_weights = verify_external_weight_bindings(toml_data)
    validity = _verify_validity_contract(toml_data)
    scope_isolation = _verify_scope_isolation(toml_data)
    telemetry = _verify_telemetry_contract(toml_data)
    artifact_policy = _verify_artifact_policy(toml_data)

    missing_coverage = _TOP_LEVEL_KEYS - toml_data.accessed_top_level_keys
    if missing_coverage:
        raise T320RealSmokeContractError(
            f"verify_all did not actually read the following required top-level tables during this "
            f"run, so it cannot report verified=True: {sorted(missing_coverage)}"
        )
    if set(_TABLE_VALIDATOR_REGISTRY) != _TOP_LEVEL_KEYS:
        raise T320RealSmokeContractError(
            "the static _TABLE_VALIDATOR_REGISTRY is out of sync with _TOP_LEVEL_KEYS"
        )

    return {
        "identity": identity,
        "repository_state": repo_state,
        "implementation": implementation,
        "dependencies": dependencies,
        "population": population,
        "window_geometry": window_geometry,
        "resource": resource,
        "scope": scope,
        "result_schema": result_schema,
        "determinism": determinism,
        "external_weights": external_weights,
        "validity": validity,
        "scope_isolation": scope_isolation,
        "telemetry": telemetry,
        "artifact_policy": artifact_policy,
        "accessed_top_level_tables": sorted(toml_data.accessed_top_level_keys),
        "verified": True,
    }
