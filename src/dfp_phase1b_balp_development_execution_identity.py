"""Strict loading/validation for the DFP Phase 1B BALP development-
execution authorization successor identity
(`evaluation_identities/e23_dfp_phase1b_balp_development_execution.toml`,
"E23").

E23 authorizes REAL EXECUTION of the already-frozen BALP method (E22)
over the already-frozen 350-image development split, via the
already-committed BALP runner, on exactly one H100 80GB GPU, wall time
at most 05:00:00, in fresh/resume/finalize-only-via-resume modes. It
introduces no new scientific formula, no new metric, and no competing
artifact schema -- every scientific definition is re-pinned by exact
path + SHA-256 to E22 and reused unmodified. E23's own additions are
the real production dataset/geometry/model provider binding, the
frozen 350-image development window universe (image count, window
count, canonical digest), the GPU/wall-time resource contract, the
checkpoint/resume mode authorization, and an explicit prohibition list.

This module implements no method, runs no inference, submits no GPU
job, and performs no torch/CUDA/model/dataset access -- pure
TOML/JSON/filesystem/hash validation and exact comparison only.

`KeyboardInterrupt`, `SystemExit`, and `MemoryError` always propagate
unmodified; ordinary parse/type/path/relational failures are wrapped in
`DfpPhase1bBalpDevelopmentExecutionIdentityError`.
"""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping

import src.dfp_phase1b_lightweight_boundary_identity as e22
from src.dfp_phase1b_chunk_benchmark_identity import (
    DfpPhase1bChunkBenchmarkIdentityError,
    resolve_repository_relative_path,
)


class DfpPhase1bBalpDevelopmentExecutionIdentityError(ValueError):
    """Raised on any E23 identity invariant violation. Always fail
    closed. Never raised for, and never catches, KeyboardInterrupt /
    SystemExit / MemoryError -- those always propagate unmodified."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e23_dfp_phase1b_balp_development_execution.toml")
SCHEMA_NAME = "talk2dino-dfp-phase1b-balp-development-execution-identity-v1"
SCHEMA_VERSION = 1
REQUIRED_ANCESTOR_COMMIT = "f0bb0eadd8daf63b82f2b9c93f6c15afd78fb890"
EXPECTED_WORKING_BRANCH = "dfp-phase1a-diagnostics"
EXPECTED_SUCCESSOR_OF = "e22-dfp-phase1b-lightweight-boundary-preservation"

DEVELOPMENT_IMAGE_COUNT = 350
CONFIRMATION_IMAGE_COUNT_ACCESSIBLE = 0
OTHER_VALIDATION_IMAGES_ACCESSIBLE = 0
CANDIDATE_COUNT = 16
CONTROL_COUNT = 5
WINDOW_COUNT = 769
WINDOW_UNIVERSE_DIGEST = "b4672854e38199deef26c154a30e45c1835c201a4dbcf046aa29e7fdf12df9e6"

GPU_COUNT = 1
GPU_MODEL = "h100_80gb"
DEVICE_TYPE = "cuda"
MAX_WALL_TIME = "05:00:00"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT40 = re.compile(r"^[0-9a-f]{40}$")


# ---------------------------------------------------------------------
# Small closed-schema primitives -- a self-contained copy of the same
# convention every other E-identity loader in this repository uses
# (see e.g. src/dfp_phase1b_lightweight_boundary_identity.py), never a
# cross-module import of another identity's private validators.
# ---------------------------------------------------------------------


def _require_string(value: Any, label: str) -> str:
    if type(value) is not str:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(f"{label} must be a string, observed {type(value).__name__}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    s = _require_string(value, label)
    if not _HEX64.match(s):
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(f"{label} must be lowercase 64-hex, observed {value!r}")
    return s


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(f"{label} must be a table")
    return value


def _require_closed(node: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    node = _require_mapping(node, label)
    try:
        actual = frozenset(node.keys())
    except TypeError as exc:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(f"{label} has an unhashable key") from exc
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
    if missing or unknown:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(f"{label} has the wrong key set -- missing={missing} unknown={unknown}")
    return node


def _require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            f"{label} must be an int (not bool/float), observed {type(value).__name__}: {value!r}"
        )
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(f"{label} must be a bool, observed {type(value).__name__}: {value!r}")
    return value


def _require_exact_value(observed: Any, expected: Any, label: str) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            f"{label}: frozen value drift -- expected {expected!r} ({type(expected).__name__}), "
            f"observed {observed!r} ({type(observed).__name__})"
        )


# ---------------------------------------------------------------------
# [dependency_bindings]: closed key set (23 members). Ordinary bindings
# carry {relative_path, sha256, binds}; development_window_universe
# additionally carries {image_count, window_count, canonical_record_digest}.
# ---------------------------------------------------------------------

_ORDINARY_BINDING_NAMES = frozenset({
    "e22_preservation", "e22_artifacts", "e13_protocol", "matched_evaluator", "subset_manifest",
    "balp_split_artifact", "graph_module", "finite_step_regime_module", "recurrence_capture_module",
    "horizon_stitching_module", "stitching_pullback_module", "final_rescale_module", "analysis_module",
    "identity_module", "balp_scientific_kernel", "balp_artifact_validator", "balp_execution_runner",
    "balp_cli", "production_provider_module", "dataset_view_module", "sliding_window_geometry_module",
    "window_universe_materializer_module",
})
_WINDOW_UNIVERSE_BINDING_NAME = "development_window_universe"
_ALL_BINDING_NAMES = _ORDINARY_BINDING_NAMES | {_WINDOW_UNIVERSE_BINDING_NAME}

_ORDINARY_BINDING_KEYS = frozenset({"relative_path", "sha256", "binds"})
_WINDOW_UNIVERSE_BINDING_KEYS = frozenset({
    "relative_path", "sha256", "image_count", "window_count", "canonical_record_digest", "binds",
})


def _resolve_and_hash(root: Path, relative_path: Any, *, binding_name: str) -> str:
    import hashlib

    try:
        target = resolve_repository_relative_path(root, relative_path, binding_name=binding_name)
    except DfpPhase1bChunkBenchmarkIdentityError as exc:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            f"dependency_bindings.{binding_name}: unsafe or unresolvable relative_path -- {exc}"
        ) from exc
    try:
        return hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError as exc:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            f"dependency_bindings.{binding_name}: cannot read bound file {relative_path}: {exc}"
        ) from exc


def _verify_window_universe_binding(root: Path, binding: Mapping[str, Any]) -> None:
    image_count = _require_exact_int(binding["image_count"], "dependency_bindings.development_window_universe.image_count")
    window_count = _require_exact_int(binding["window_count"], "dependency_bindings.development_window_universe.window_count")
    digest = _require_sha256(binding["canonical_record_digest"], "dependency_bindings.development_window_universe.canonical_record_digest")
    if image_count != DEVELOPMENT_IMAGE_COUNT:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            f"dependency_bindings.development_window_universe.image_count must be exactly {DEVELOPMENT_IMAGE_COUNT}, observed {image_count}"
        )
    if window_count != WINDOW_COUNT:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            f"dependency_bindings.development_window_universe.window_count must be exactly {WINDOW_COUNT}, observed {window_count}"
        )
    if digest != WINDOW_UNIVERSE_DIGEST:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            f"dependency_bindings.development_window_universe.canonical_record_digest must equal {WINDOW_UNIVERSE_DIGEST}, observed {digest}"
        )

    relative_path = binding["relative_path"]
    target = resolve_repository_relative_path(root, relative_path, binding_name="development_window_universe")
    try:
        manifest = json.loads(target.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(f"cannot load bound development window-universe manifest {target}: {exc}") from exc

    if manifest.get("development_count") != image_count:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            f"development window-universe manifest development_count {manifest.get('development_count')!r} disagrees with the pinned image_count {image_count}"
        )
    if manifest.get("aggregate_window_count") != window_count:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            f"development window-universe manifest aggregate_window_count {manifest.get('aggregate_window_count')!r} disagrees with the pinned window_count {window_count}"
        )
    if manifest.get("canonical_record_digest") != digest:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            "development window-universe manifest canonical_record_digest disagrees with the pinned digest"
        )


def _verify_dependency_bindings(root: Path, dependency_bindings: Mapping[str, Any]) -> None:
    observed_names = frozenset(dependency_bindings.keys())
    if observed_names != _ALL_BINDING_NAMES:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            f"dependency_bindings has the wrong key set -- missing={sorted(_ALL_BINDING_NAMES - observed_names)} "
            f"unknown={sorted(observed_names - _ALL_BINDING_NAMES)}"
        )
    for name in _ALL_BINDING_NAMES:
        binding = dependency_bindings[name]
        keys = _WINDOW_UNIVERSE_BINDING_KEYS if name == _WINDOW_UNIVERSE_BINDING_NAME else _ORDINARY_BINDING_KEYS
        _require_closed(binding, keys, f"dependency_bindings.{name}")
        expected_hash = _require_sha256(binding["sha256"], f"dependency_bindings.{name}.sha256")
        relative_path = _require_string(binding["relative_path"], f"dependency_bindings.{name}.relative_path")
        actual_hash = _resolve_and_hash(root, relative_path, binding_name=name)
        if actual_hash != expected_hash:
            raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
                f"dependency_bindings.{name}: {relative_path} SHA-256 mismatch -- pinned {expected_hash}, observed {actual_hash} "
                f"(bound dependency drifted since E23 was frozen)"
            )
        if name == _WINDOW_UNIVERSE_BINDING_NAME:
            _verify_window_universe_binding(root, binding)

    # Cross-check E22's own two hashes agree with what E22's OWN loader
    # independently verifies -- E23 never trusts its own copy of these
    # hashes in isolation.
    e22_identity = e22.load_identity(repo_root=root)
    e22_artifacts_identity = e22.load_artifacts_identity(repo_root=root)
    import hashlib

    e22_preservation_path = resolve_repository_relative_path(
        root, dependency_bindings["e22_preservation"]["relative_path"], binding_name="e22_preservation",
    )
    e22_artifacts_path = resolve_repository_relative_path(
        root, dependency_bindings["e22_artifacts"]["relative_path"], binding_name="e22_artifacts",
    )
    if hashlib.sha256(e22_preservation_path.read_bytes()).hexdigest() != dependency_bindings["e22_preservation"]["sha256"]:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError("e22_preservation hash disagreement (unreachable -- already checked above)")
    if e22_artifacts_identity["protocol_binding"]["protocol_identity_sha256"] != dependency_bindings["e22_preservation"]["sha256"]:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            "dependency_bindings.e22_preservation.sha256 disagrees with E22 artifacts' own protocol_binding.protocol_identity_sha256"
        )
    if e22_identity["identity"]["name"] != EXPECTED_SUCCESSOR_OF:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            f"bound E22 identity name {e22_identity['identity']['name']!r} disagrees with EXPECTED_SUCCESSOR_OF {EXPECTED_SUCCESSOR_OF!r}"
        )


# ---------------------------------------------------------------------
# [authorization_boundary]: the safety-critical section -- every field
# checked to its exact required value, never merely "present".
# ---------------------------------------------------------------------

_AUTHORIZATION_BOUNDARY_KEYS = frozenset({
    "gpu_execution_authorized_by_this_identity", "device_type_authorized", "gpu_count_authorized",
    "gpu_model_authorized", "max_wall_time_authorized", "development_execution_authorized",
    "confirmation_split_access_authorized", "other_4500_image_access_authorized",
    "full_5000_image_evaluation_authorized", "exact_masked_forward_provenance_authorized",
    "component_construction_or_masked_propagation_authorized", "candidate_specific_recurrence_authorized",
    "confirmation_based_tuning_authorized", "final_paper_claims_authorized", "confirmation_execution_authorized",
    "full_operational_benchmarking_authorized", "overwrite_of_completed_development_result_authorized",
    "real_execute_cli_implementation_authorized_by_this_identity", "real_launcher_submission_authorized_by_this_identity",
    "candidate_selected_by_this_identity",
})

_REQUIRED_TRUE_AUTHORIZATION_FIELDS = ("gpu_execution_authorized_by_this_identity", "development_execution_authorized")
_REQUIRED_FALSE_AUTHORIZATION_FIELDS = (
    "confirmation_split_access_authorized", "other_4500_image_access_authorized",
    "full_5000_image_evaluation_authorized", "exact_masked_forward_provenance_authorized",
    "component_construction_or_masked_propagation_authorized", "candidate_specific_recurrence_authorized",
    "confirmation_based_tuning_authorized", "final_paper_claims_authorized", "confirmation_execution_authorized",
    "full_operational_benchmarking_authorized", "overwrite_of_completed_development_result_authorized",
    "real_execute_cli_implementation_authorized_by_this_identity", "real_launcher_submission_authorized_by_this_identity",
    "candidate_selected_by_this_identity",
)


def _verify_authorization_boundary(node: Any) -> None:
    node = _require_closed(node, _AUTHORIZATION_BOUNDARY_KEYS, "authorization_boundary")
    for field in _REQUIRED_TRUE_AUTHORIZATION_FIELDS:
        _require_exact_value(_require_bool(node[field], f"authorization_boundary.{field}"), True, f"authorization_boundary.{field}")
    for field in _REQUIRED_FALSE_AUTHORIZATION_FIELDS:
        _require_exact_value(_require_bool(node[field], f"authorization_boundary.{field}"), False, f"authorization_boundary.{field}")
    _require_exact_value(_require_string(node["device_type_authorized"], "authorization_boundary.device_type_authorized"), DEVICE_TYPE, "authorization_boundary.device_type_authorized")
    _require_exact_value(_require_exact_int(node["gpu_count_authorized"], "authorization_boundary.gpu_count_authorized"), GPU_COUNT, "authorization_boundary.gpu_count_authorized")
    _require_exact_value(_require_string(node["gpu_model_authorized"], "authorization_boundary.gpu_model_authorized"), GPU_MODEL, "authorization_boundary.gpu_model_authorized")
    _require_exact_value(_require_string(node["max_wall_time_authorized"], "authorization_boundary.max_wall_time_authorized"), MAX_WALL_TIME, "authorization_boundary.max_wall_time_authorized")


# ---------------------------------------------------------------------
# [resource_contract], [population], [checkpoint_resume_contract].
# ---------------------------------------------------------------------

_RESOURCE_CONTRACT_KEYS = frozenset({
    "device_type", "gpu_count", "gpu_model", "max_wall_time", "cpus_per_task", "host_memory",
    "account_convention", "timeout_leaves_valid_checkpoint_required", "timeout_must_not_silently_relax_limit",
    "runtime_claim_before_real_job_executed_forbidden",
})

_WALL_TIME_RE = re.compile(r"^(\d{2}):([0-5]\d):([0-5]\d)$")


def _wall_time_to_seconds(value: str, *, label: str) -> int:
    match = _WALL_TIME_RE.match(value)
    if not match:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(f"{label} must be HH:MM:SS, observed {value!r}")
    h, m, s = (int(part) for part in match.groups())
    return h * 3600 + m * 60 + s


def _verify_resource_contract(node: Any) -> None:
    node = _require_closed(node, _RESOURCE_CONTRACT_KEYS, "resource_contract")
    _require_exact_value(_require_string(node["device_type"], "resource_contract.device_type"), DEVICE_TYPE, "resource_contract.device_type")
    _require_exact_value(_require_exact_int(node["gpu_count"], "resource_contract.gpu_count"), GPU_COUNT, "resource_contract.gpu_count")
    _require_exact_value(_require_string(node["gpu_model"], "resource_contract.gpu_model"), GPU_MODEL, "resource_contract.gpu_model")
    max_wall_time = _require_string(node["max_wall_time"], "resource_contract.max_wall_time")
    _require_exact_value(max_wall_time, MAX_WALL_TIME, "resource_contract.max_wall_time")
    if _wall_time_to_seconds(max_wall_time, label="resource_contract.max_wall_time") > 5 * 3600:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError("resource_contract.max_wall_time exceeds the authorized five-hour ceiling")
    _require_exact_int(node["cpus_per_task"], "resource_contract.cpus_per_task")
    _require_string(node["host_memory"], "resource_contract.host_memory")
    _require_string(node["account_convention"], "resource_contract.account_convention")
    for field in ("timeout_leaves_valid_checkpoint_required", "timeout_must_not_silently_relax_limit", "runtime_claim_before_real_job_executed_forbidden"):
        _require_exact_value(_require_bool(node[field], f"resource_contract.{field}"), True, f"resource_contract.{field}")


_POPULATION_KEYS = frozenset({
    "development_image_count", "development_image_ids_source", "confirmation_image_count_accessible",
    "other_validation_images_accessible", "window_count", "window_count_source", "window_universe_digest",
    "window_universe_digest_source", "candidate_count", "control_count", "candidate_ids_source", "control_ids_source",
})


def _verify_population(node: Any) -> None:
    node = _require_closed(node, _POPULATION_KEYS, "population")
    _require_exact_value(_require_exact_int(node["development_image_count"], "population.development_image_count"), DEVELOPMENT_IMAGE_COUNT, "population.development_image_count")
    _require_exact_value(_require_exact_int(node["confirmation_image_count_accessible"], "population.confirmation_image_count_accessible"), CONFIRMATION_IMAGE_COUNT_ACCESSIBLE, "population.confirmation_image_count_accessible")
    _require_exact_value(_require_exact_int(node["other_validation_images_accessible"], "population.other_validation_images_accessible"), OTHER_VALIDATION_IMAGES_ACCESSIBLE, "population.other_validation_images_accessible")
    _require_exact_value(_require_exact_int(node["window_count"], "population.window_count"), WINDOW_COUNT, "population.window_count")
    _require_exact_value(_require_sha256(node["window_universe_digest"], "population.window_universe_digest"), WINDOW_UNIVERSE_DIGEST, "population.window_universe_digest")
    _require_exact_value(_require_exact_int(node["candidate_count"], "population.candidate_count"), CANDIDATE_COUNT, "population.candidate_count")
    _require_exact_value(_require_exact_int(node["control_count"], "population.control_count"), CONTROL_COUNT, "population.control_count")
    for field in ("development_image_ids_source", "window_count_source", "window_universe_digest_source", "candidate_ids_source", "control_ids_source"):
        _require_string(node[field], f"population.{field}")


_ACCEPTED_MODES = ("fresh", "resume", "finalize_only_via_resume")
_FORBIDDEN_MODES = frozenset({
    "debug_subset", "max_images_override", "alternate_id_order",
    "force_overwrite_completed_result", "manual_generation_skipping", "silent_checkpoint_reset",
})
_CHECKPOINT_RESUME_CONTRACT_KEYS = frozenset({
    "accepted_modes", "checkpoint_state_machine_source", "fresh_resume_scientific_equivalence_required",
    "zero_completed_image_reprocessing_required", "forbidden",
})


def _verify_checkpoint_resume_contract(node: Any) -> None:
    node = _require_closed(node, _CHECKPOINT_RESUME_CONTRACT_KEYS, "checkpoint_resume_contract")
    accepted = node["accepted_modes"]
    if not isinstance(accepted, list) or [type(m) for m in accepted] != [str] * len(accepted) or list(accepted) != list(_ACCEPTED_MODES):
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            f"checkpoint_resume_contract.accepted_modes must be exactly {list(_ACCEPTED_MODES)}, observed {accepted!r}"
        )
    for field in ("fresh_resume_scientific_equivalence_required", "zero_completed_image_reprocessing_required"):
        _require_exact_value(_require_bool(node[field], f"checkpoint_resume_contract.{field}"), True, f"checkpoint_resume_contract.{field}")
    _require_string(node["checkpoint_state_machine_source"], "checkpoint_resume_contract.checkpoint_state_machine_source")
    forbidden_node = _require_mapping(node["forbidden"], "checkpoint_resume_contract.forbidden")
    _require_closed(forbidden_node, frozenset({"list"}), "checkpoint_resume_contract.forbidden")
    forbidden_list = forbidden_node["list"]
    if not isinstance(forbidden_list, list) or frozenset(forbidden_list) != _FORBIDDEN_MODES or len(forbidden_list) != len(_FORBIDDEN_MODES):
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            f"checkpoint_resume_contract.forbidden.list must be exactly {sorted(_FORBIDDEN_MODES)} (no duplicates), observed {forbidden_list!r}"
        )


# ---------------------------------------------------------------------
# Remaining sections: [precision], [scientific_execution_contract],
# [historical_exclusions], [development_result_authority],
# [profiling_contract], [output_path_policy] -- closed-key validation
# at every nested level, plus exact-value checks for the fields that
# gate scientific behavior (precision, recurrence-per-window counts,
# fail-closed triggers, competing-schema prohibition).
# ---------------------------------------------------------------------

_PRECISION_KEYS = frozenset({
    "compute_dtype_source", "compute_dtype", "output_dtype_source", "output_dtype",
    "precision_source_description", "new_precision_setting_introduced",
})


def _verify_precision(node: Any) -> None:
    node = _require_closed(node, _PRECISION_KEYS, "precision")
    _require_exact_value(_require_string(node["compute_dtype"], "precision.compute_dtype"), "float32", "precision.compute_dtype")
    _require_exact_value(_require_string(node["output_dtype"], "precision.output_dtype"), "float32", "precision.output_dtype")
    _require_exact_value(_require_bool(node["new_precision_setting_introduced"], "precision.new_precision_setting_introduced"), False, "precision.new_precision_setting_introduced")
    for field in ("compute_dtype_source", "output_dtype_source", "precision_source_description"):
        _require_string(node[field], f"precision.{field}")


_SCIENTIFIC_EXECUTION_CONTRACT_KEYS = frozenset({
    "model_forwards_per_window", "graph_constructions_per_window", "recurrence_trajectories_per_window",
    "horizon_snapshots_reused_from_single_trajectory", "candidates_evaluated_per_image_without_additional_recurrence",
    "controls_evaluated_per_image_without_additional_recurrence", "second_recurrence_trajectory_per_candidate_forbidden",
    "fail_closed_on",
})
_REQUIRED_FAIL_CLOSED_TRIGGERS = frozenset({
    "model_or_source_binding_mismatch", "dataset_order_mismatch", "confirmation_or_non_development_image_requested",
    "filename_or_relative_path_identity_mismatch", "second_recurrence_trajectory_triggered_by_any_candidate",
    "development_window_universe_binding_mismatch",
})


def _verify_scientific_execution_contract(node: Any) -> None:
    node = _require_closed(node, _SCIENTIFIC_EXECUTION_CONTRACT_KEYS, "scientific_execution_contract")
    _require_exact_value(_require_exact_int(node["model_forwards_per_window"], "scientific_execution_contract.model_forwards_per_window"), 1, "scientific_execution_contract.model_forwards_per_window")
    _require_exact_value(_require_exact_int(node["graph_constructions_per_window"], "scientific_execution_contract.graph_constructions_per_window"), 1, "scientific_execution_contract.graph_constructions_per_window")
    _require_exact_value(_require_exact_int(node["recurrence_trajectories_per_window"], "scientific_execution_contract.recurrence_trajectories_per_window"), 1, "scientific_execution_contract.recurrence_trajectories_per_window")
    horizons = node["horizon_snapshots_reused_from_single_trajectory"]
    if not isinstance(horizons, list) or [type(h) for h in horizons] != [int] * len(horizons) or horizons != [0, 16, 32, 64, 320]:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(f"scientific_execution_contract.horizon_snapshots_reused_from_single_trajectory must be exactly [0,16,32,64,320], observed {horizons!r}")
    _require_exact_value(_require_exact_int(node["candidates_evaluated_per_image_without_additional_recurrence"], "scientific_execution_contract.candidates_evaluated_per_image_without_additional_recurrence"), CANDIDATE_COUNT, "scientific_execution_contract.candidates_evaluated_per_image_without_additional_recurrence")
    _require_exact_value(_require_exact_int(node["controls_evaluated_per_image_without_additional_recurrence"], "scientific_execution_contract.controls_evaluated_per_image_without_additional_recurrence"), CONTROL_COUNT, "scientific_execution_contract.controls_evaluated_per_image_without_additional_recurrence")
    _require_exact_value(_require_bool(node["second_recurrence_trajectory_per_candidate_forbidden"], "scientific_execution_contract.second_recurrence_trajectory_per_candidate_forbidden"), True, "scientific_execution_contract.second_recurrence_trajectory_per_candidate_forbidden")

    fail_closed_node = _require_mapping(node["fail_closed_on"], "scientific_execution_contract.fail_closed_on")
    _require_closed(fail_closed_node, frozenset({"list"}), "scientific_execution_contract.fail_closed_on")
    fail_closed_list = fail_closed_node["list"]
    if not isinstance(fail_closed_list, list) or frozenset(fail_closed_list) != _REQUIRED_FAIL_CLOSED_TRIGGERS or len(fail_closed_list) != len(_REQUIRED_FAIL_CLOSED_TRIGGERS):
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            f"scientific_execution_contract.fail_closed_on.list must be exactly {sorted(_REQUIRED_FAIL_CLOSED_TRIGGERS)} (no duplicates), observed {fail_closed_list!r}"
        )


_HISTORICAL_EXCLUSIONS_KEYS = frozenset({
    "list", "e19_e20_e21_required_by_balp_execution", "e15_e16_required_by_balp_execution",
    "note_on_shared_adapter_identity_loads",
})


def _verify_historical_exclusions(node: Any) -> None:
    node = _require_closed(node, _HISTORICAL_EXCLUSIONS_KEYS, "historical_exclusions")
    exclusions = node["list"]
    if not isinstance(exclusions, list) or not all(type(x) is str for x in exclusions) or len(exclusions) == 0:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError("historical_exclusions.list must be a nonempty list of strings")
    for field in ("e19_e20_e21_required_by_balp_execution", "e15_e16_required_by_balp_execution"):
        _require_exact_value(_require_bool(node[field], f"historical_exclusions.{field}"), False, f"historical_exclusions.{field}")
    _require_string(node["note_on_shared_adapter_identity_loads"], "historical_exclusions.note_on_shared_adapter_identity_loads")


_DEVELOPMENT_RESULT_AUTHORITY_KEYS = frozenset({
    "result_schema_source", "competing_result_schema_forbidden", "required_hash_bindings_in_result",
    "selected_candidate_field_authorized", "selected_candidate_field_requires",
})


def _verify_development_result_authority(node: Any) -> None:
    node = _require_closed(node, _DEVELOPMENT_RESULT_AUTHORITY_KEYS, "development_result_authority")
    _require_string(node["result_schema_source"], "development_result_authority.result_schema_source")
    _require_exact_value(_require_bool(node["competing_result_schema_forbidden"], "development_result_authority.competing_result_schema_forbidden"), True, "development_result_authority.competing_result_schema_forbidden")
    required = node["required_hash_bindings_in_result"]
    if not isinstance(required, list) or not all(type(x) is str for x in required) or len(required) == 0:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError("development_result_authority.required_hash_bindings_in_result must be a nonempty list of strings")
    _require_exact_value(_require_bool(node["selected_candidate_field_authorized"], "development_result_authority.selected_candidate_field_authorized"), False, "development_result_authority.selected_candidate_field_authorized")
    _require_string(node["selected_candidate_field_requires"], "development_result_authority.selected_candidate_field_requires")


_PROFILING_CONTRACT_KEYS = frozenset({
    "schema_source", "affects_scientific_digest", "profiling_on_off_scientific_output_equivalence_required",
})


def _verify_profiling_contract(node: Any) -> None:
    node = _require_closed(node, _PROFILING_CONTRACT_KEYS, "profiling_contract")
    _require_string(node["schema_source"], "profiling_contract.schema_source")
    _require_exact_value(_require_bool(node["affects_scientific_digest"], "profiling_contract.affects_scientific_digest"), False, "profiling_contract.affects_scientific_digest")
    _require_exact_value(_require_bool(node["profiling_on_off_scientific_output_equivalence_required"], "profiling_contract.profiling_on_off_scientific_output_equivalence_required"), True, "profiling_contract.profiling_on_off_scientific_output_equivalence_required")


_OUTPUT_PATH_POLICY_KEYS = frozenset({
    "root_convention", "lexical_traversal_forbidden", "symlink_containment_required",
    "fresh_output_for_first_execution_required", "overwrite_of_existing_completed_result_forbidden",
    "overwrite_requires_separate_explicit_later_authority", "e23_grants_overwrite_authority",
    "atomic_final_installation_required", "staged_validation_required",
    "post_install_reopen_and_reverification_required", "artifact_write_policy_source",
})
_OUTPUT_PATH_POLICY_REQUIRED_TRUE = (
    "lexical_traversal_forbidden", "symlink_containment_required", "fresh_output_for_first_execution_required",
    "overwrite_of_existing_completed_result_forbidden", "overwrite_requires_separate_explicit_later_authority",
    "atomic_final_installation_required", "staged_validation_required", "post_install_reopen_and_reverification_required",
)


def _verify_output_path_policy(node: Any) -> None:
    node = _require_closed(node, _OUTPUT_PATH_POLICY_KEYS, "output_path_policy")
    _require_string(node["root_convention"], "output_path_policy.root_convention")
    _require_string(node["artifact_write_policy_source"], "output_path_policy.artifact_write_policy_source")
    for field in _OUTPUT_PATH_POLICY_REQUIRED_TRUE:
        _require_exact_value(_require_bool(node[field], f"output_path_policy.{field}"), True, f"output_path_policy.{field}")
    _require_exact_value(_require_bool(node["e23_grants_overwrite_authority"], "output_path_policy.e23_grants_overwrite_authority"), False, "output_path_policy.e23_grants_overwrite_authority")


# ---------------------------------------------------------------------
# [identity] and repository gate.
# ---------------------------------------------------------------------

_IDENTITY_KEYS = frozenset({
    "name", "schema_name", "schema_version", "description", "required_ancestor_commit",
    "working_branch", "successor_of", "mutates_or_version_bumps_earlier_identity",
})


def _verify_identity_block(node: Any) -> None:
    node = _require_closed(node, _IDENTITY_KEYS, "identity")
    _require_exact_value(_require_string(node["name"], "identity.name"), "e23-dfp-phase1b-balp-development-execution", "identity.name")
    _require_exact_value(_require_string(node["schema_name"], "identity.schema_name"), SCHEMA_NAME, "identity.schema_name")
    _require_exact_value(_require_exact_int(node["schema_version"], "identity.schema_version"), SCHEMA_VERSION, "identity.schema_version")
    _require_string(node["description"], "identity.description")
    commit = _require_string(node["required_ancestor_commit"], "identity.required_ancestor_commit")
    if not _COMMIT40.match(commit):
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(f"identity.required_ancestor_commit must be 40-hex, observed {commit!r}")
    _require_exact_value(commit, REQUIRED_ANCESTOR_COMMIT, "identity.required_ancestor_commit")
    _require_exact_value(_require_string(node["working_branch"], "identity.working_branch"), EXPECTED_WORKING_BRANCH, "identity.working_branch")
    _require_exact_value(_require_string(node["successor_of"], "identity.successor_of"), EXPECTED_SUCCESSOR_OF, "identity.successor_of")
    _require_exact_value(_require_bool(node["mutates_or_version_bumps_earlier_identity"], "identity.mutates_or_version_bumps_earlier_identity"), False, "identity.mutates_or_version_bumps_earlier_identity")


def verify_repository_gate(repo_root: Path, repository_commit: str) -> None:
    """Verifies REQUIRED_ANCESTOR_COMMIT is an ancestor of (or equal to)
    `repository_commit`, identical to every other E-identity's own
    convention (e.g. e22's verify_repository_gate)."""
    commit = _require_string(repository_commit, "repository_commit")
    if not _COMMIT40.match(commit):
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(f"repository_commit must be a 40-hex commit, observed {commit!r}")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", REQUIRED_ANCESTOR_COMMIT, commit],
            capture_output=True, text=True,
        )
    except OSError as exc:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(f"cannot invoke git for ancestor verification: {exc}") from exc
    if result.returncode != 0:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(
            f"required_ancestor_commit {REQUIRED_ANCESTOR_COMMIT} is not an ancestor of (or equal to) {commit}"
        )


# ---------------------------------------------------------------------
# Top-level load/verify.
# ---------------------------------------------------------------------

_TOP_LEVEL_KEYS = frozenset({
    "format_version", "identity", "dependency_bindings", "historical_exclusions", "precision",
    "scientific_execution_contract", "population", "resource_contract", "checkpoint_resume_contract",
    "development_result_authority", "profiling_contract", "output_path_policy", "authorization_boundary",
    "prohibited",
})


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load, closed-schema-validate, exact-value-validate, and
    relationally hash-verify the E23 BALP development-execution
    identity. Never mutates or re-derives any bound dependency -- only
    re-verifies SHA-256/content against current bytes. Performs no
    torch/CUDA/model/dataset access."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with open(source, "rb") as fh:
            document = tomllib.load(fh)
    except OSError as exc:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(f"cannot read E23 identity {source}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError(f"E23 identity {source} is not valid TOML: {exc}") from exc

    _require_closed(document, _TOP_LEVEL_KEYS, "<top level>")
    _require_exact_value(
        _require_string(document["format_version"], "format_version"),
        "talk2dino-dfp-phase1b-balp-development-execution-identity-v1",
        "format_version",
    )
    _verify_identity_block(document["identity"])
    _verify_dependency_bindings(root, _require_mapping(document["dependency_bindings"], "dependency_bindings"))
    _verify_historical_exclusions(document["historical_exclusions"])
    _verify_precision(document["precision"])
    _verify_scientific_execution_contract(document["scientific_execution_contract"])
    _verify_population(document["population"])
    _verify_resource_contract(document["resource_contract"])
    _verify_checkpoint_resume_contract(document["checkpoint_resume_contract"])
    _verify_development_result_authority(document["development_result_authority"])
    _verify_profiling_contract(document["profiling_contract"])
    _verify_output_path_policy(document["output_path_policy"])
    _verify_authorization_boundary(document["authorization_boundary"])

    prohibited = _require_mapping(document["prohibited"], "prohibited")
    _require_closed(prohibited, frozenset({"list"}), "prohibited")
    prohibited_list = prohibited["list"]
    if not isinstance(prohibited_list, list) or not all(type(x) is str for x in prohibited_list) or len(set(prohibited_list)) != len(prohibited_list):
        raise DfpPhase1bBalpDevelopmentExecutionIdentityError("prohibited.list must be a list of unique strings")

    return document
