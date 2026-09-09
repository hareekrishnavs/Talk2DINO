"""DFP Phase 1B Task 4: deterministic operational chunk-size selection
(E18-authorized, additive successor to E17).

This module implements NO scientific mathematics of its own. It
consumes the completed, independently-verified E17 benchmark output
(the real ``merged_result.json`` and its six raw trial files, produced
by ``src.dfp_phase1b_chunk_benchmark``) as a read-only input, applies a
deterministic, frozen-before-being-applied post-hoc selection rule,
and records the resulting OPERATIONAL ``component_chunk_size`` choice.

This module never imports torch, mmcv, or any model/dataset module --
CPU-only, pure TOML/JSON/hash/git-metadata/arithmetic. It selects no
scientific metric, defines no conservation tolerance, and authorizes
no full Phase 1B or 5,000-image execution.

Always fail closed: never silently substitute a default, coerce a
type, or normalize a string. Never trusts a recorded winner without
recomputing it fresh from the six independently-verified raw trials.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.dfp_phase1b_chunk_benchmark import (
    DfpPhase1bChunkBenchmarkError,
    merge_trials,
    strict_structural_equal,
    verify_trial_document,
)
from src.dfp_phase1b_chunk_benchmark_identity import (
    BENCHMARK_WINDOW_COUNT,
    CHUNK_SIZES,
    TRIAL_SCHEDULE,
    DfpPhase1bChunkBenchmarkIdentityError,
    load_identity as load_e17_identity_document,
    require_clean_worktree,
    resolve_repository_relative_path,
    verify_ancestor_relation,
)


class DfpPhase1bChunkSelectionError(RuntimeError):
    """Raised on any DFP Phase 1B chunk-selection invariant violation.
    Always fail closed. Never raised for, and never catches,
    KeyboardInterrupt/SystemExit/MemoryError."""


IDENTITY_RELATIVE_PATH = "evaluation_identities/e18_dfp_phase1b_chunk_selection.toml"
FORMAT_VERSION = "Talk2Dino-DFP-Phase1B-Chunk-Selection-Identity-V1"

CANDIDATE_CHUNK_SIZES: tuple[int, ...] = tuple(CHUNK_SIZES)
REQUIRED_REPEATS_PER_CANDIDATE = 2
REQUIRED_MEASURED_WINDOWS_PER_TRIAL = BENCHMARK_WINDOW_COUNT
TIE_BREAK_SEQUENCE: tuple[str, ...] = (
    "lower_max_host_process_lifetime_peak_rss_bytes",
    "lower_max_gpu_peak_reserved_bytes",
    "smaller_numerical_chunk_size",
)
_ALLOWED_TIE_BREAK_PATHS: frozenset[str] = frozenset({
    "unique_minimum_no_tie_break_needed",
    "resolved_by_host_memory_tie_break",
    "resolved_by_gpu_memory_tie_break",
    "resolved_by_smaller_chunk_size_tie_break",
    "no_safe_candidate",
})

SELECTION_SCHEMA_NAME = "talk2dino-dfp-phase1b-chunk-selection-artifact-v1"
SELECTION_SCHEMA_VERSION = 1

REQUIRED_GPU_MODEL = "NVIDIA H100 80GB HBM3"
REQUIRED_DEVICE_TYPE = "cuda"

# ---------------------------------------------------------------------
# Closed schema for E18. Every table's key set is checked exactly;
# unknown or missing keys fail closed.
# ---------------------------------------------------------------------

_TOP_KEYS = frozenset({
    "format_version", "identity", "operational_scope", "dependency_bindings",
    "source_benchmark_run", "selection_contract", "safety_contract",
    "selection_result", "authorization_boundary", "disclosure", "prohibited",
    "future_step_bindings",
})
_IDENTITY_KEYS = frozenset({
    "name", "schema_name", "schema_version", "purpose", "creation_status",
    "successor_of", "successor_relation", "required_ancestor_commit",
    "required_ancestor_relation", "working_branch",
})
_OPERATIONAL_SCOPE_KEYS = frozenset({
    "purpose", "scientific_metric", "preregistered_hypothesis_test",
    "statistical_significance_claimed", "post_hoc_engineering_selection",
    "selection_rule_frozen_before_being_applied_to_trials",
    "conservation_tolerance_defined", "full_phase1b_execution_authorized",
    "five_thousand_image_evaluation_authorized", "gpu_execution_performed_by_this_step",
    "operational_chunk_size_selected",
})
_DEPENDENCY_BINDING_NAMES = frozenset({
    "e17_identity", "benchmark_manifest", "benchmark_runner_module", "benchmark_cli_module",
    "merged_result", "selector_module", "analysis_cli_module",
})
_DEPENDENCY_BINDING_KEYS_MIN = frozenset({"relative_path", "sha256"})
_SOURCE_RUN_KEYS = frozenset({
    "slurm_job_id", "repository_commit", "device_type", "gpu_model_name",
    "trial_count", "scheduler_state", "scheduler_exit_code",
})
_SELECTION_CONTRACT_KEYS = frozenset({
    "candidate_chunk_sizes", "required_repeats_per_candidate", "required_measured_windows_per_trial",
    "elapsed_time_field", "elapsed_time_field_note", "primary_statistic", "tie_definition",
    "tie_break_sequence", "eligibility_conditions",
})
_SAFETY_CONTRACT_KEYS = frozenset({
    "description", "host_rss_abort_fraction_of_allocation", "gpu_reserved_abort_fraction_of_allocation",
    "allocated_host_memory_gib", "allocated_gpu_memory_gib", "comparison_operator", "boundary_semantics",
    "host_threshold_bytes", "gpu_threshold_bytes",
})
_SELECTION_RESULT_KEYS = frozenset({
    "scientific_output_digest", "all_candidates_scientifically_equal", "selected_chunk_size",
    "selected_mean_elapsed_ns", "tie_break_path", "candidates",
})
_CANDIDATE_KEYS = frozenset({
    "chunk_size", "repeat0_trial_index", "repeat1_trial_index", "repeat0_primary_elapsed_ns",
    "repeat1_primary_elapsed_ns", "mean_elapsed_ns", "max_host_process_lifetime_peak_rss_bytes",
    "max_gpu_peak_reserved_bytes", "safe", "safety_violations",
})
_AUTHORIZATION_BOUNDARY_KEYS = frozenset({
    "operational_chunk_size_selected", "conservation_tolerance_defined",
    "full_phase1b_execution_authorized", "five_thousand_image_execution_authorized",
    "gpu_execution_performed_by_this_step",
})
_DISCLOSURE_KEYS = frozenset({
    "is_operational_engineering_selection", "selection_rule_frozen_after_observing_benchmark",
    "not_a_preregistered_scientific_hypothesis_test", "two_repeats_do_not_support_statistical_significance",
    "symmetric_trial_order_reduces_but_does_not_eliminate_temporal_cache_order_effects",
    "selection_does_not_change_scientific_results", "selection_does_not_define_conservation_tolerance",
    "selection_does_not_authorize_full_phase1b", "selection_does_not_authorize_or_implement_5000_image_evaluation",
    "chunk_16_not_claimed_scientifically_superior",
    "chunk_16_is_the_selected_operational_configuration_for_this_hardware_software_binding_only",
})

_CANONICAL_AUTHORIZATION_BOUNDARY = {
    "operational_chunk_size_selected": True,
    "conservation_tolerance_defined": False,
    "full_phase1b_execution_authorized": False,
    "five_thousand_image_execution_authorized": False,
    "gpu_execution_performed_by_this_step": False,
}
_CANONICAL_OPERATIONAL_SCOPE_BOOLEANS = {
    "scientific_metric": False,
    "preregistered_hypothesis_test": False,
    "statistical_significance_claimed": False,
    "post_hoc_engineering_selection": True,
    "selection_rule_frozen_before_being_applied_to_trials": True,
    "conservation_tolerance_defined": False,
    "full_phase1b_execution_authorized": False,
    "five_thousand_image_evaluation_authorized": False,
    "gpu_execution_performed_by_this_step": False,
    "operational_chunk_size_selected": True,
}


def _check_keys(node: Any, expected: frozenset, label: str) -> None:
    if not isinstance(node, dict):
        raise DfpPhase1bChunkSelectionError(f"{label} must be a table, observed {type(node).__name__}")
    actual = frozenset(node.keys())
    if actual != expected:
        raise DfpPhase1bChunkSelectionError(
            f"{label} has the wrong key set -- missing={sorted(expected - actual)} unknown={sorted(actual - expected)}"
        )


def _require_bool(node: Mapping[str, Any], key: str, expected: bool, label: str) -> None:
    value = node.get(key)
    if type(value) is not bool:
        raise DfpPhase1bChunkSelectionError(f"{label}.{key} must be a bool, observed {value!r} ({type(value).__name__})")
    if value is not expected:
        raise DfpPhase1bChunkSelectionError(f"{label}.{key} must be {expected!r}, observed {value!r}")


def _require_nonneg_int(node: Mapping[str, Any], key: str, label: str) -> None:
    """Exact ``int``, non-negative -- rejects ``bool`` (not ``int`` under
    ``type() is``), ``float`` (including NaN/Infinity/integral-looking
    floats), negative integers, and any other non-exact-int value."""
    value = node.get(key)
    if type(value) is not int or value < 0:
        raise DfpPhase1bChunkSelectionError(
            f"{label}.{key} must be an exact non-negative int, observed {value!r} ({type(value).__name__})"
        )


def _require_finite_nonneg_number(node: Mapping[str, Any], key: str, label: str) -> None:
    """``int`` or ``float`` (never ``bool``), finite, non-negative --
    used for the one field (``mean_elapsed_ns``/``selected_mean_elapsed_ns``)
    that is legitimately a half-integer average and so may be a float."""
    value = node.get(key)
    if type(value) not in (int, float) or isinstance(value, bool):
        raise DfpPhase1bChunkSelectionError(
            f"{label}.{key} must be an int or float, observed {value!r} ({type(value).__name__})"
        )
    if not math.isfinite(value):
        raise DfpPhase1bChunkSelectionError(f"{label}.{key} must be finite, observed {value!r}")
    if value < 0:
        raise DfpPhase1bChunkSelectionError(f"{label}.{key} must be non-negative, observed {value!r}")


def sha256_file(path: Path) -> str:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise DfpPhase1bChunkSelectionError(f"cannot read {path}: {exc}") from exc
    return hashlib.sha256(raw).hexdigest()


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except OSError as exc:
        raise DfpPhase1bChunkSelectionError(f"cannot read E18 identity {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise DfpPhase1bChunkSelectionError(f"E18 identity {path} is not valid TOML: {exc}") from exc


def load_e18_identity(repo_root: Path, *, path: Path | None = None) -> dict[str, Any]:
    """Loads and closed-schema-validates E18. Performs NO filesystem
    hashing of dependency bindings here -- that is
    ``verify_e18_dependency_bindings``'s job, kept separate so schema
    validation never silently depends on file I/O succeeding."""
    root = Path(repo_root)
    identity_path = Path(path) if path is not None else (root / IDENTITY_RELATIVE_PATH)
    document = _load_toml(identity_path)

    _check_keys(document, _TOP_KEYS - {"format_version"} | {"format_version"}, "E18 identity")
    if document.get("format_version") != FORMAT_VERSION:
        raise DfpPhase1bChunkSelectionError(
            f"E18 identity format_version must be {FORMAT_VERSION!r}, observed {document.get('format_version')!r}"
        )

    _check_keys(document["identity"], _IDENTITY_KEYS, "E18 identity.identity")
    ident = document["identity"]
    if ident.get("name") != "e18-dfp-phase1b-chunk-selection":
        raise DfpPhase1bChunkSelectionError("E18 identity.identity.name must be 'e18-dfp-phase1b-chunk-selection'")
    if ident.get("schema_name") != "talk2dino-dfp-phase1b-chunk-selection-identity-v1":
        raise DfpPhase1bChunkSelectionError("E18 identity.identity.schema_name must be 'talk2dino-dfp-phase1b-chunk-selection-identity-v1'")
    if ident.get("schema_version") != 1:
        raise DfpPhase1bChunkSelectionError("E18 identity.identity.schema_version must be 1")
    if ident.get("successor_of") != "e17-dfp-phase1b-chunk-benchmark":
        raise DfpPhase1bChunkSelectionError("E18 identity.identity.successor_of must be 'e17-dfp-phase1b-chunk-benchmark'")
    if not isinstance(ident.get("required_ancestor_commit"), str) or len(ident["required_ancestor_commit"]) != 40:
        raise DfpPhase1bChunkSelectionError("E18 identity.identity.required_ancestor_commit must be a 40-character commit hash")
    if ident.get("working_branch") != "dfp-phase1a-diagnostics":
        raise DfpPhase1bChunkSelectionError("E18 identity.identity.working_branch must be 'dfp-phase1a-diagnostics'")

    _check_keys(document["operational_scope"], _OPERATIONAL_SCOPE_KEYS, "E18 identity.operational_scope")
    scope = document["operational_scope"]
    if scope.get("purpose") != "operational_chunk_size_selection":
        raise DfpPhase1bChunkSelectionError("E18 identity.operational_scope.purpose must be 'operational_chunk_size_selection'")
    for key, expected in _CANONICAL_OPERATIONAL_SCOPE_BOOLEANS.items():
        _require_bool(scope, key, expected, "E18 identity.operational_scope")

    deps = document["dependency_bindings"]
    if frozenset(deps.keys()) != _DEPENDENCY_BINDING_NAMES:
        raise DfpPhase1bChunkSelectionError(
            f"E18 identity.dependency_bindings has the wrong binding set -- "
            f"missing={sorted(_DEPENDENCY_BINDING_NAMES - frozenset(deps.keys()))} "
            f"unknown={sorted(frozenset(deps.keys()) - _DEPENDENCY_BINDING_NAMES)}"
        )
    for name, binding in deps.items():
        if not _DEPENDENCY_BINDING_KEYS_MIN.issubset(frozenset(binding.keys())):
            raise DfpPhase1bChunkSelectionError(f"E18 identity.dependency_bindings.{name} is missing relative_path/sha256")
        relative_path = binding["relative_path"]
        if type(relative_path) is not str or len(relative_path) == 0:
            raise DfpPhase1bChunkSelectionError(f"E18 identity.dependency_bindings.{name}.relative_path must be a nonempty string")
        sha = binding["sha256"]
        if type(sha) is not str or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise DfpPhase1bChunkSelectionError(f"E18 identity.dependency_bindings.{name}.sha256 must be a 64-hex-char lowercase string")

    _check_keys(document["source_benchmark_run"], _SOURCE_RUN_KEYS, "E18 identity.source_benchmark_run")
    run = document["source_benchmark_run"]
    if run.get("gpu_model_name") != REQUIRED_GPU_MODEL:
        raise DfpPhase1bChunkSelectionError(f"E18 identity.source_benchmark_run.gpu_model_name must be {REQUIRED_GPU_MODEL!r}")
    if run.get("device_type") != REQUIRED_DEVICE_TYPE:
        raise DfpPhase1bChunkSelectionError(f"E18 identity.source_benchmark_run.device_type must be {REQUIRED_DEVICE_TYPE!r}")
    if run.get("trial_count") != len(TRIAL_SCHEDULE):
        raise DfpPhase1bChunkSelectionError(f"E18 identity.source_benchmark_run.trial_count must be {len(TRIAL_SCHEDULE)}")

    _check_keys(document["selection_contract"], _SELECTION_CONTRACT_KEYS, "E18 identity.selection_contract")
    sc = document["selection_contract"]
    if list(sc.get("candidate_chunk_sizes", [])) != list(CANDIDATE_CHUNK_SIZES):
        raise DfpPhase1bChunkSelectionError(f"E18 identity.selection_contract.candidate_chunk_sizes must be {list(CANDIDATE_CHUNK_SIZES)}")
    if sc.get("required_repeats_per_candidate") != REQUIRED_REPEATS_PER_CANDIDATE:
        raise DfpPhase1bChunkSelectionError("E18 identity.selection_contract.required_repeats_per_candidate must be 2")
    if sc.get("required_measured_windows_per_trial") != REQUIRED_MEASURED_WINDOWS_PER_TRIAL:
        raise DfpPhase1bChunkSelectionError("E18 identity.selection_contract.required_measured_windows_per_trial must be 24")
    if sc.get("elapsed_time_field") != "timing.primary_elapsed_ns":
        raise DfpPhase1bChunkSelectionError("E18 identity.selection_contract.elapsed_time_field must be 'timing.primary_elapsed_ns'")
    if list(sc.get("tie_break_sequence", [])) != list(TIE_BREAK_SEQUENCE):
        raise DfpPhase1bChunkSelectionError(f"E18 identity.selection_contract.tie_break_sequence must be {list(TIE_BREAK_SEQUENCE)}")
    if not isinstance(sc.get("eligibility_conditions"), dict) or not isinstance(sc["eligibility_conditions"].get("list"), list) \
            or len(sc["eligibility_conditions"]["list"]) == 0:
        raise DfpPhase1bChunkSelectionError("E18 identity.selection_contract.eligibility_conditions.list must be a nonempty list")

    _check_keys(document["safety_contract"], _SAFETY_CONTRACT_KEYS, "E18 identity.safety_contract")
    safety = document["safety_contract"]
    if safety.get("comparison_operator") != "strictly_greater_than":
        raise DfpPhase1bChunkSelectionError("E18 identity.safety_contract.comparison_operator must be 'strictly_greater_than'")
    host_threshold = float(safety["allocated_host_memory_gib"]) * (1024 ** 3) * float(safety["host_rss_abort_fraction_of_allocation"])
    gpu_threshold = float(safety["allocated_gpu_memory_gib"]) * (1024 ** 3) * float(safety["gpu_reserved_abort_fraction_of_allocation"])
    if float(safety["host_threshold_bytes"]) != host_threshold:
        raise DfpPhase1bChunkSelectionError("E18 identity.safety_contract.host_threshold_bytes disagrees with its own fraction*allocation")
    if float(safety["gpu_threshold_bytes"]) != gpu_threshold:
        raise DfpPhase1bChunkSelectionError("E18 identity.safety_contract.gpu_threshold_bytes disagrees with its own fraction*allocation")

    _check_keys(document["selection_result"], _SELECTION_RESULT_KEYS, "E18 identity.selection_result")
    result = document["selection_result"]
    if result.get("selected_chunk_size") not in CANDIDATE_CHUNK_SIZES:
        raise DfpPhase1bChunkSelectionError("E18 identity.selection_result.selected_chunk_size must be one of the candidate chunk sizes")
    if type(result.get("all_candidates_scientifically_equal")) is not bool:
        raise DfpPhase1bChunkSelectionError("E18 identity.selection_result.all_candidates_scientifically_equal must be a bool")
    digest = result.get("scientific_output_digest")
    if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise DfpPhase1bChunkSelectionError("E18 identity.selection_result.scientific_output_digest must be a 64-hex-char lowercase string")
    _require_finite_nonneg_number(result, "selected_mean_elapsed_ns", "E18 identity.selection_result")
    if result.get("tie_break_path") not in _ALLOWED_TIE_BREAK_PATHS:
        raise DfpPhase1bChunkSelectionError(
            f"E18 identity.selection_result.tie_break_path must be one of {sorted(_ALLOWED_TIE_BREAK_PATHS)}, observed {result.get('tie_break_path')!r}"
        )
    candidates = result["candidates"]
    expected_candidate_names = {f"chunk_{cs}" for cs in CANDIDATE_CHUNK_SIZES}
    if frozenset(candidates.keys()) != frozenset(expected_candidate_names):
        raise DfpPhase1bChunkSelectionError("E18 identity.selection_result.candidates has the wrong candidate set")
    for name, candidate in candidates.items():
        _check_keys(candidate, _CANDIDATE_KEYS, f"E18 identity.selection_result.candidates.{name}")
        if candidate["chunk_size"] not in CANDIDATE_CHUNK_SIZES:
            raise DfpPhase1bChunkSelectionError(f"E18 identity.selection_result.candidates.{name}.chunk_size must be a candidate chunk size")
        label = f"E18 identity.selection_result.candidates.{name}"
        _require_nonneg_int(candidate, "repeat0_trial_index", label)
        _require_nonneg_int(candidate, "repeat1_trial_index", label)
        if candidate["repeat0_trial_index"] >= len(TRIAL_SCHEDULE) or candidate["repeat1_trial_index"] >= len(TRIAL_SCHEDULE):
            raise DfpPhase1bChunkSelectionError(f"{label} trial index out of range")
        _require_nonneg_int(candidate, "repeat0_primary_elapsed_ns", label)
        _require_nonneg_int(candidate, "repeat1_primary_elapsed_ns", label)
        _require_finite_nonneg_number(candidate, "mean_elapsed_ns", label)
        _require_nonneg_int(candidate, "max_host_process_lifetime_peak_rss_bytes", label)
        _require_nonneg_int(candidate, "max_gpu_peak_reserved_bytes", label)
        if type(candidate["safe"]) is not bool:
            raise DfpPhase1bChunkSelectionError(f"{label}.safe must be a bool")
        if not isinstance(candidate["safety_violations"], list) or not all(type(r) is str for r in candidate["safety_violations"]):
            raise DfpPhase1bChunkSelectionError(f"{label}.safety_violations must be a list of strings")

    _check_keys(document["authorization_boundary"], _AUTHORIZATION_BOUNDARY_KEYS, "E18 identity.authorization_boundary")
    for key, expected in _CANONICAL_AUTHORIZATION_BOUNDARY.items():
        _require_bool(document["authorization_boundary"], key, expected, "E18 identity.authorization_boundary")

    _check_keys(document["disclosure"], _DISCLOSURE_KEYS, "E18 identity.disclosure")
    for key in _DISCLOSURE_KEYS:
        _require_bool(document["disclosure"], key, True, "E18 identity.disclosure")

    if not isinstance(document["prohibited"], dict) or not isinstance(document["prohibited"].get("list"), list) \
            or len(document["prohibited"]["list"]) == 0:
        raise DfpPhase1bChunkSelectionError("E18 identity.prohibited.list must be a nonempty list")
    if not isinstance(document["future_step_bindings"], dict) or not isinstance(document["future_step_bindings"].get("next_step_items"), list):
        raise DfpPhase1bChunkSelectionError("E18 identity.future_step_bindings.next_step_items must be a list")

    return document


def verify_e18_dependency_bindings(
    repo_root: Path, document: Mapping[str, Any], *, merged_result_path: Path,
) -> None:
    """Relationally re-verifies every E18 dependency binding's SHA-256
    against live bytes. Repository-relative bindings are resolved
    through the same safe path resolver E17 uses (rejects absolute
    paths, lexical ``..``, and symlink escape). The scratch-resident
    ``merged_result`` binding is verified against the caller-supplied
    real path."""
    root = Path(repo_root)
    deps = document["dependency_bindings"]

    for name in ("e17_identity", "benchmark_manifest", "benchmark_runner_module", "benchmark_cli_module",
                 "selector_module", "analysis_cli_module"):
        binding = deps[name]
        try:
            target = resolve_repository_relative_path(root, binding["relative_path"], binding_name=name)
        except DfpPhase1bChunkBenchmarkIdentityError as exc:
            raise DfpPhase1bChunkSelectionError(f"E18 dependency_bindings.{name}.relative_path is unsafe or unresolvable: {exc}") from exc
        actual = sha256_file(target)
        if actual != binding["sha256"]:
            raise DfpPhase1bChunkSelectionError(
                f"E18 dependency_bindings.{name} SHA-256 mismatch for {target} -- pinned {binding['sha256']}, observed {actual}"
            )

    merged_binding = deps["merged_result"]
    actual_merged = sha256_file(Path(merged_result_path))
    if actual_merged != merged_binding["sha256"]:
        raise DfpPhase1bChunkSelectionError(
            f"E18 dependency_bindings.merged_result SHA-256 mismatch for {merged_result_path} -- "
            f"pinned {merged_binding['sha256']}, observed {actual_merged}"
        )


def verify_e18_against_e17(repo_root: Path, e18_document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Loads and fully verifies the live E17 identity, and relationally
    cross-checks E18's safety_contract against E17's own
    [measurement.safety_guard] values -- E18 never merely hand-copies
    these, it re-derives them from the live E17 document every time."""
    e17_doc = load_e17_identity_document(repo_root=repo_root)
    guard = e17_doc["measurement"]["safety_guard"]
    safety = e18_document["safety_contract"]
    for field, e18_key in (
        ("host_rss_abort_fraction_of_allocation", "host_rss_abort_fraction_of_allocation"),
        ("gpu_reserved_abort_fraction_of_allocation", "gpu_reserved_abort_fraction_of_allocation"),
        ("allocated_host_memory_gib", "allocated_host_memory_gib"),
        ("allocated_gpu_memory_gib", "allocated_gpu_memory_gib"),
        ("comparison_operator", "comparison_operator"),
    ):
        if guard[field] != safety[e18_key]:
            raise DfpPhase1bChunkSelectionError(
                f"E18 safety_contract.{e18_key} ({safety[e18_key]!r}) disagrees with the live "
                f"E17 measurement.safety_guard.{field} ({guard[field]!r})"
            )
    e17_binding = e18_document["dependency_bindings"]["e17_identity"]
    e17_path = resolve_repository_relative_path(repo_root, e17_binding["relative_path"], binding_name="e17_identity")
    if sha256_file(e17_path) != e17_binding["sha256"]:
        raise DfpPhase1bChunkSelectionError("E18's bound E17 hash disagrees with the live E17 file")
    return e17_doc


# ---------------------------------------------------------------------
# Benchmark-output loading: independently re-verifies all six raw
# trials (via the real, committed Task 3 verifier) and independently
# reconstructs the merged result (via the real, committed Task 3
# merger), then requires the reconstruction to structurally equal the
# merged_result.json actually on disk -- catching drift in either
# direction (a tampered merged file, or a tampered/replaced raw trial
# file).
# ---------------------------------------------------------------------


def load_and_verify_benchmark_output(
    *, repo_root: Path, output_dir: Path, merged_result_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if merged_result_path is None:
        merged_result_path = output_dir / "merged_result.json"
    else:
        merged_result_path = Path(merged_result_path)

    trial_paths = sorted(output_dir.glob("trial_*.json"))
    if len(trial_paths) != len(TRIAL_SCHEDULE):
        raise DfpPhase1bChunkSelectionError(
            f"expected exactly {len(TRIAL_SCHEDULE)} trial_*.json files in {output_dir}, found {len(trial_paths)}: "
            f"{[p.name for p in trial_paths]}"
        )

    trials: list[dict[str, Any]] = []
    for p in trial_paths:
        try:
            doc = json.loads(p.read_text())
        except OSError as exc:
            raise DfpPhase1bChunkSelectionError(f"cannot read trial file {p}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise DfpPhase1bChunkSelectionError(f"trial file {p} is not valid JSON: {exc}") from exc
        try:
            verify_trial_document(doc, verify_repo_root=Path(repo_root))
        except DfpPhase1bChunkBenchmarkError as exc:
            raise DfpPhase1bChunkSelectionError(f"trial file {p} failed Task 3 verification: {exc}") from exc
        trials.append(doc)
    trials.sort(key=lambda t: t["trial_index"])

    try:
        merged_from_disk = json.loads(merged_result_path.read_text())
    except OSError as exc:
        raise DfpPhase1bChunkSelectionError(f"cannot read merged result {merged_result_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DfpPhase1bChunkSelectionError(f"merged result {merged_result_path} is not valid JSON: {exc}") from exc

    try:
        merged_reconstructed = merge_trials(trials, verify_repo_root=Path(repo_root))
    except DfpPhase1bChunkBenchmarkError as exc:
        raise DfpPhase1bChunkSelectionError(f"independent reconstruction of the merged result from its six raw trial files failed: {exc}") from exc

    if not strict_structural_equal(merged_reconstructed, merged_from_disk):
        raise DfpPhase1bChunkSelectionError(
            "merged_result.json on disk does not exactly structurally equal the independent reconstruction "
            "from its own six raw trial files -- a stored summary, raw trial, or digest may have been tampered"
        )

    return {"trials": trials, "merged": merged_from_disk, "merged_result_path": merged_result_path, "output_dir": output_dir}


# ---------------------------------------------------------------------
# Candidate measurement / eligibility / winner selection.
# ---------------------------------------------------------------------


def compute_candidate_measurements(trials: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    by_chunk: dict[int, list[Mapping[str, Any]]] = {}
    for t in trials:
        by_chunk.setdefault(int(t["chunk_size"]), []).append(t)

    measurements: dict[int, dict[str, Any]] = {}
    for chunk_size in CANDIDATE_CHUNK_SIZES:
        group = by_chunk.get(chunk_size, [])
        if len(group) != REQUIRED_REPEATS_PER_CANDIDATE:
            raise DfpPhase1bChunkSelectionError(
                f"chunk_size={chunk_size} has {len(group)} trial(s) in this benchmark output, required exactly {REQUIRED_REPEATS_PER_CANDIDATE}"
            )
        group_sorted = sorted(group, key=lambda t: t["repeat_index"])
        if [t["repeat_index"] for t in group_sorted] != list(range(REQUIRED_REPEATS_PER_CANDIDATE)):
            raise DfpPhase1bChunkSelectionError(
                f"chunk_size={chunk_size} does not have exactly one trial for each repeat_index 0..{REQUIRED_REPEATS_PER_CANDIDATE - 1}"
            )

        elapsed: list[int] = []
        host_peaks: list[int] = []
        gpu_peaks: list[int] = []
        for t in group_sorted:
            e = t["timing"]["primary_elapsed_ns"]
            if type(e) is not int or e < 0:
                raise DfpPhase1bChunkSelectionError(f"trial_index={t['trial_index']} has a non-finite/negative primary_elapsed_ns: {e!r}")
            elapsed.append(e)
            for label, key, sink in (
                ("host_process_lifetime_peak_rss_bytes", "host_process_lifetime_peak_rss_bytes", host_peaks),
                ("gpu_peak_reserved_bytes", "gpu_peak_reserved_bytes", gpu_peaks),
            ):
                v = t["memory"][key]
                if type(v) is not int or v < 0:
                    raise DfpPhase1bChunkSelectionError(f"trial_index={t['trial_index']} has a non-finite/negative memory.{key}: {v!r}")
                sink.append(v)

        measurements[chunk_size] = {
            "chunk_size": chunk_size,
            "trial_indices": [t["trial_index"] for t in group_sorted],
            "elapsed_ns_by_repeat": elapsed,
            "mean_elapsed_ns": sum(elapsed) / len(elapsed),
            "max_host_process_lifetime_peak_rss_bytes": max(host_peaks),
            "max_gpu_peak_reserved_bytes": max(gpu_peaks),
        }
    return measurements


def evaluate_candidate_safety(
    trials: Sequence[Mapping[str, Any]], measurements: Mapping[int, Mapping[str, Any]], *,
    host_threshold_bytes: float, gpu_threshold_bytes: float, common_digest: str,
) -> dict[int, dict[str, Any]]:
    by_chunk: dict[int, list[Mapping[str, Any]]] = {}
    for t in trials:
        by_chunk.setdefault(int(t["chunk_size"]), []).append(t)

    safety: dict[int, dict[str, Any]] = {}
    for chunk_size in CANDIDATE_CHUNK_SIZES:
        reasons: list[str] = []
        group = by_chunk.get(chunk_size, [])
        if len(group) != REQUIRED_REPEATS_PER_CANDIDATE:
            reasons.append("wrong_repeat_count")
        for t in group:
            ti = t.get("trial_index")
            key = (t.get("trial_index"), t.get("repeat_index"), t.get("chunk_size"))
            if key not in TRIAL_SCHEDULE:
                reasons.append(f"trial_{ti}_not_in_frozen_schedule")
            if t.get("complete") is not True or t.get("final") is not True:
                reasons.append(f"trial_{ti}_not_complete_final")
            if t.get("abort_status", {}).get("aborted"):
                reasons.append(f"trial_{ti}_aborted")
            if len(t.get("window_identity", [])) != REQUIRED_MEASURED_WINDOWS_PER_TRIAL:
                reasons.append(f"trial_{ti}_wrong_window_count")
            if t.get("device_record", {}).get("gpu_model_name") != REQUIRED_GPU_MODEL:
                reasons.append(f"trial_{ti}_wrong_gpu_model")
            if t.get("device_record", {}).get("device_type") != REQUIRED_DEVICE_TYPE:
                reasons.append(f"trial_{ti}_wrong_device_type")
            if t.get("scientific_output_digests", {}).get("collection_digest") != common_digest:
                reasons.append(f"trial_{ti}_scientific_digest_mismatch")
            elapsed = t.get("timing", {}).get("primary_elapsed_ns")
            if type(elapsed) is not int or elapsed < 0:
                reasons.append(f"trial_{ti}_non_finite_elapsed")
            for mkey, mval in t.get("memory", {}).items():
                if type(mval) is not int or mval < 0:
                    reasons.append(f"trial_{ti}_non_finite_or_negative_memory_{mkey}")

        m = measurements.get(chunk_size)
        if m is not None:
            if m["max_host_process_lifetime_peak_rss_bytes"] > host_threshold_bytes:
                reasons.append("host_memory_exceeds_safety_threshold")
            if m["max_gpu_peak_reserved_bytes"] > gpu_threshold_bytes:
                reasons.append("gpu_memory_exceeds_safety_threshold")

        safety[chunk_size] = {"safe": len(reasons) == 0, "reasons": reasons}
    return safety


def select_winner(
    measurements: Mapping[int, Mapping[str, Any]], safety: Mapping[int, Mapping[str, Any]],
) -> tuple[int | None, str, list[int]]:
    """Applies the frozen tie-break sequence via a single composite
    lexicographic sort key -- mathematically identical to applying the
    three tie-breakers in order, deterministic regardless of input
    iteration order, dict order, or PYTHONHASHSEED (no hashing is
    involved anywhere in this function)."""
    eligible = [cs for cs in CANDIDATE_CHUNK_SIZES if safety.get(cs, {}).get("safe")]
    if not eligible:
        return None, "no_safe_candidate", []

    def sort_key(cs: int) -> tuple[float, int, int, int]:
        m = measurements[cs]
        return (
            m["mean_elapsed_ns"],
            m["max_host_process_lifetime_peak_rss_bytes"],
            m["max_gpu_peak_reserved_bytes"],
            cs,
        )

    ranked = sorted(eligible, key=sort_key)
    winner = ranked[0]

    tied_on_mean = [cs for cs in eligible if measurements[cs]["mean_elapsed_ns"] == measurements[winner]["mean_elapsed_ns"]]
    if len(tied_on_mean) == 1:
        path = "unique_minimum_no_tie_break_needed"
    else:
        tied_on_host = [
            cs for cs in tied_on_mean
            if measurements[cs]["max_host_process_lifetime_peak_rss_bytes"] == measurements[winner]["max_host_process_lifetime_peak_rss_bytes"]
        ]
        if len(tied_on_host) == 1:
            path = "resolved_by_host_memory_tie_break"
        else:
            tied_on_gpu = [
                cs for cs in tied_on_host
                if measurements[cs]["max_gpu_peak_reserved_bytes"] == measurements[winner]["max_gpu_peak_reserved_bytes"]
            ]
            path = "resolved_by_gpu_memory_tie_break" if len(tied_on_gpu) == 1 else "resolved_by_smaller_chunk_size_tie_break"

    return winner, path, eligible


# ---------------------------------------------------------------------
# Top-level orchestration.
# ---------------------------------------------------------------------


def analyze_benchmark(
    *, repo_root: Path, output_dir: Path, merged_result_path: Path | None = None,
) -> dict[str, Any]:
    """CPU-only, no E18 required. Independently verifies the six raw
    trials + merged result, computes candidate measurements/safety,
    and applies the frozen selection rule. Returns a plain dict report
    (not a closed-schema artifact -- see ``build_selection_artifact``
    for that)."""
    loaded = load_and_verify_benchmark_output(repo_root=repo_root, output_dir=output_dir, merged_result_path=merged_result_path)
    trials = loaded["trials"]
    merged = loaded["merged"]

    digests = {t["scientific_output_digests"]["collection_digest"] for t in trials}
    if len(digests) != 1:
        raise DfpPhase1bChunkSelectionError(f"SCIENTIFIC_OUTPUT_MISMATCH: trials do not share a single common digest, observed {sorted(digests)}")
    common_digest = next(iter(digests))
    if common_digest != merged["scientific_output_digest"]:
        raise DfpPhase1bChunkSelectionError("common per-trial scientific digest disagrees with merged_result.json's own scientific_output_digest")

    measurements = compute_candidate_measurements(trials)
    host_threshold = 0.90 * 80 * (1024 ** 3)
    gpu_threshold = 0.90 * 80 * (1024 ** 3)
    safety = evaluate_candidate_safety(
        trials, measurements, host_threshold_bytes=host_threshold, gpu_threshold_bytes=gpu_threshold, common_digest=common_digest,
    )
    winner, tie_break_path, eligible = select_winner(measurements, safety)

    return {
        "trials": trials, "merged": merged, "common_digest": common_digest,
        "measurements": measurements, "safety": safety,
        "winner": winner, "tie_break_path": tie_break_path, "eligible": eligible,
        "output_dir": str(loaded["output_dir"]), "merged_result_path": str(loaded["merged_result_path"]),
    }


def build_expected_selection_result(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """The CANONICAL, fully independently-recomputed decision payload,
    shaped identically to E18's own ``[selection_result]`` table
    (including every candidate's full record). This is the single
    source of truth ``verify_selection`` compares E18's recorded
    ``selection_result`` against -- every field here is derived
    exclusively from the six independently re-verified raw trials
    (via ``analysis``, itself built by ``analyze_benchmark`` /
    ``load_and_verify_benchmark_output``), never copied from any
    previously-recorded E18 value. Adding a new persisted candidate
    field to E18's schema requires adding it here too -- there is
    exactly one place that defines what a correct decision payload
    looks like."""
    measurements = analysis["measurements"]
    safety = analysis["safety"]
    winner = analysis["winner"]

    candidates: dict[str, Any] = {}
    for cs in CANDIDATE_CHUNK_SIZES:
        m = measurements[cs]
        ti0, ti1 = m["trial_indices"]
        e0, e1 = m["elapsed_ns_by_repeat"]
        candidates[f"chunk_{cs}"] = {
            "chunk_size": cs,
            "repeat0_trial_index": ti0,
            "repeat1_trial_index": ti1,
            "repeat0_primary_elapsed_ns": e0,
            "repeat1_primary_elapsed_ns": e1,
            "mean_elapsed_ns": m["mean_elapsed_ns"],
            "max_host_process_lifetime_peak_rss_bytes": m["max_host_process_lifetime_peak_rss_bytes"],
            "max_gpu_peak_reserved_bytes": m["max_gpu_peak_reserved_bytes"],
            "safe": safety[cs]["safe"],
            "safety_violations": list(safety[cs]["reasons"]),
        }

    digests = {t["scientific_output_digests"]["collection_digest"] for t in analysis["trials"]}
    return {
        "scientific_output_digest": analysis["common_digest"],
        "all_candidates_scientifically_equal": len(digests) == 1,
        "selected_chunk_size": winner,
        "selected_mean_elapsed_ns": measurements[winner]["mean_elapsed_ns"] if winner is not None else None,
        "tie_break_path": analysis["tie_break_path"],
        "candidates": candidates,
    }


def _diff_mapping(expected: Any, actual: Any, path: str = "selection_result") -> list[str]:
    """Recursively finds every mismatching leaf path between the
    canonical recomputed payload and E18's recorded payload, using the
    identical type-strict equality standard as
    ``src.dfp_phase1b_chunk_benchmark.strict_structural_equal`` (never
    treats ``600 == 600.0`` or ``True == 1`` as equal). Returns a list
    of human-readable ``"path: expected=... observed=..."`` strings --
    empty if the two payloads are fully identical."""
    if type(expected) is not type(actual):
        return [f"{path}: type mismatch -- expected {type(expected).__name__} ({expected!r}), observed {type(actual).__name__} ({actual!r})"]
    if isinstance(expected, dict):
        mismatches: list[str] = []
        expected_keys, actual_keys = set(expected.keys()), set(actual.keys())
        if expected_keys != actual_keys:
            mismatches.append(
                f"{path}: key set mismatch -- missing={sorted(expected_keys - actual_keys)} unexpected={sorted(actual_keys - expected_keys)}"
            )
        for key in sorted(expected_keys & actual_keys):
            mismatches.extend(_diff_mapping(expected[key], actual[key], f"{path}.{key}"))
        return mismatches
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [f"{path}: length mismatch -- expected {len(expected)}, observed {len(actual)}"]
        mismatches = []
        for i, (e, a) in enumerate(zip(expected, actual)):
            mismatches.extend(_diff_mapping(e, a, f"{path}[{i}]"))
        return mismatches
    if expected != actual:
        return [f"{path}: expected {expected!r}, observed {actual!r}"]
    return []


def verify_selection(
    *, repo_root: Path, e18_path: Path | None = None, output_dir: Path | None = None,
    merged_result_path: Path | None = None,
) -> dict[str, Any]:
    """Full verify-selection boundary: loads+validates E18, verifies
    every dependency binding, cross-checks the safety contract against
    the live E17, independently re-verifies/reconstructs the benchmark
    output, recomputes the ENTIRE decision payload from scratch via
    ``build_expected_selection_result``, and requires it to exactly
    equal E18's recorded ``selection_result`` in full -- every
    candidate, both repeats, every memory maximum, every safety
    reason, the selected chunk, and the selected mean -- never trusting
    any individual recorded field without recomputing it."""
    root = Path(repo_root)
    e18 = load_e18_identity(root, path=e18_path)
    verify_ancestor_relation(
        root, required_ancestor_commit=e18["identity"]["required_ancestor_commit"],
        repository_commit=_current_head(root),
    )
    verify_e18_against_e17(root, e18)

    binding = e18["dependency_bindings"]["merged_result"]
    resolved_output_dir = Path(output_dir) if output_dir is not None else None
    resolved_merged_path = Path(merged_result_path) if merged_result_path is not None else None
    if resolved_merged_path is None and resolved_output_dir is not None:
        resolved_merged_path = resolved_output_dir / "merged_result.json"
    if resolved_merged_path is None:
        raise DfpPhase1bChunkSelectionError("verify_selection requires --output-dir or --merged-result to locate the real scratch benchmark output")
    if resolved_output_dir is None:
        resolved_output_dir = resolved_merged_path.parent

    verify_e18_dependency_bindings(root, e18, merged_result_path=resolved_merged_path)

    analysis = analyze_benchmark(repo_root=root, output_dir=resolved_output_dir, merged_result_path=resolved_merged_path)

    # Cross-check the merged result's OWN authority/manifest bindings
    # against the E18-bound E17 identity/benchmark-manifest hashes --
    # never merely trusting that a merged result which passes Task 3's
    # own internal consistency checks was also produced against the
    # exact E17/manifest this E18 identity binds.
    e17_binding = e18["dependency_bindings"]["e17_identity"]
    manifest_binding = e18["dependency_bindings"]["benchmark_manifest"]
    if analysis["merged"]["authority_sha256"] != e17_binding["sha256"]:
        raise DfpPhase1bChunkSelectionError(
            "merged result authority_sha256 disagrees with E18's bound E17 identity hash -- "
            "this benchmark output was not produced against the E17 identity this E18 identity binds"
        )
    if analysis["merged"]["manifest_sha256"] != manifest_binding["sha256"]:
        raise DfpPhase1bChunkSelectionError(
            "merged result manifest_sha256 disagrees with E18's bound benchmark manifest hash -- "
            "this benchmark output was not produced against the manifest this E18 identity binds"
        )

    recorded = e18["selection_result"]
    expected = build_expected_selection_result(analysis)
    mismatches = _diff_mapping(expected, recorded, path="selection_result")
    if mismatches:
        raise DfpPhase1bChunkSelectionError(
            "E18's recorded selection_result disagrees with the independently recomputed decision payload "
            "-- refusing to trust the recorded value(s):\n  " + "\n  ".join(mismatches)
        )

    analysis["e18"] = e18
    return analysis


def _current_head(repo_root: Path) -> str:
    import subprocess
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        raise DfpPhase1bChunkSelectionError(f"cannot determine repository HEAD via 'git rev-parse HEAD': {result.stderr.strip()}")
    return result.stdout.strip()


# ---------------------------------------------------------------------
# Deterministic selection artifact + Markdown report.
# ---------------------------------------------------------------------


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def resolve_analysis_generation_commit(repo_root: Path, *, e18_document: Mapping[str, Any]) -> str:
    """The SOLE way an ``analysis_generation_commit`` value may be
    obtained anywhere in this module. Requires the repository to be
    completely clean (tracked and untracked) via
    ``require_clean_worktree`` -- the identical gate Task 3's own
    ``run-trial`` uses before any execution -- so the resulting commit
    genuinely identifies the exact code that generated the artifact.
    Reads the commit via a SINGLE ``git rev-parse HEAD`` call (never
    re-read elsewhere, so a concurrent repository change cannot cause
    two different commit values to disagree within one generation).

    Validates the result is exactly 40 lowercase hex characters, that
    the commit object actually exists (``git cat-file -e``), and that
    it satisfies the E18-required ancestor relationship (an ancestor
    of, or equal to, ``e18_document``'s own
    ``identity.required_ancestor_commit`` -- NOT required to equal any
    particular future HEAD, so that artifact verification remains
    possible from any later descendant commit; this mirrors E17/E18's
    own established ancestor-not-equality convention exactly)."""
    try:
        commit = require_clean_worktree(repo_root)
    except DfpPhase1bChunkBenchmarkIdentityError as exc:
        raise DfpPhase1bChunkSelectionError(
            f"cannot resolve analysis_generation_commit: repository worktree is not clean -- {exc}"
        ) from exc

    if not _HEX40.match(commit):
        raise DfpPhase1bChunkSelectionError(f"analysis_generation_commit must be a 40-lowercase-hex commit id, observed {commit!r}")

    exists = _run_git_analysis_commit_check(repo_root, "cat-file", "-e", f"{commit}^{{commit}}")
    if exists.returncode != 0:
        raise DfpPhase1bChunkSelectionError(f"analysis_generation_commit {commit!r} does not exist as a commit object in this repository")

    required_ancestor = e18_document["identity"]["required_ancestor_commit"]
    try:
        verify_ancestor_relation(repo_root, required_ancestor_commit=required_ancestor, repository_commit=commit)
    except DfpPhase1bChunkBenchmarkIdentityError as exc:
        raise DfpPhase1bChunkSelectionError(
            f"analysis_generation_commit {commit!r} does not satisfy E18's required-ancestor relationship "
            f"(required_ancestor_commit={required_ancestor!r}): {exc}"
        ) from exc

    return commit


def _run_git_analysis_commit_check(repo_root: Path, *args: str):
    import subprocess
    return subprocess.run(
        ["git", "-C", str(repo_root), *args], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def build_selection_artifact(
    analysis: Mapping[str, Any], *, e18_sha256: str, e17_sha256: str, manifest_sha256: str,
    repository_commit: str, analysis_generation_commit: str, gpu_execution_performed_by_task4: bool, command: str,
) -> dict[str, Any]:
    if type(e18_sha256) is not str or not _HEX64.match(e18_sha256):
        raise DfpPhase1bChunkSelectionError(
            f"e18_sha256 must be a 64-lowercase-hex SHA-256 string, observed {e18_sha256!r} -- "
            "refusing to build an artifact with a blank, malformed, or placeholder E18 identity hash"
        )
    if type(analysis_generation_commit) is not str or not _HEX40.match(analysis_generation_commit):
        raise DfpPhase1bChunkSelectionError(
            f"analysis_generation_commit must be a 40-lowercase-hex commit id, observed {analysis_generation_commit!r}"
        )
    if type(gpu_execution_performed_by_task4) is not bool or gpu_execution_performed_by_task4 is not False:
        raise DfpPhase1bChunkSelectionError(
            f"gpu_execution_performed_by_task4 must be the exact bool False, observed {gpu_execution_performed_by_task4!r} "
            f"({type(gpu_execution_performed_by_task4).__name__}) -- Task 4 selection analysis never performs GPU execution"
        )
    measurements = analysis["measurements"]
    safety = analysis["safety"]
    winner = analysis["winner"]

    def candidate_block(cs: int) -> dict[str, Any]:
        m = measurements[cs]
        return {
            "chunk_size": cs,
            "trial_indices": list(m["trial_indices"]),
            "elapsed_ns_by_repeat": list(m["elapsed_ns_by_repeat"]),
            "mean_elapsed_ns": m["mean_elapsed_ns"],
            "max_host_process_lifetime_peak_rss_bytes": m["max_host_process_lifetime_peak_rss_bytes"],
            "max_gpu_peak_reserved_bytes": m["max_gpu_peak_reserved_bytes"],
            "safe": safety[cs]["safe"],
            "safety_reasons": list(safety[cs]["reasons"]),
            "difference_vs_selected_ns": (m["mean_elapsed_ns"] - measurements[winner]["mean_elapsed_ns"]) if winner is not None else None,
            "difference_vs_selected_percent": (
                100.0 * (m["mean_elapsed_ns"] - measurements[winner]["mean_elapsed_ns"]) / measurements[winner]["mean_elapsed_ns"]
            ) if winner is not None else None,
        }

    return {
        "schema_name": SELECTION_SCHEMA_NAME, "schema_version": SELECTION_SCHEMA_VERSION,
        "identity_name": "e18-dfp-phase1b-chunk-selection",
        "e18_identity_sha256": e18_sha256,
        "e17_identity_sha256": e17_sha256,
        "benchmark_manifest_sha256": manifest_sha256,
        "source_merged_result_path": analysis["merged_result_path"],
        "source_merged_result_sha256": sha256_file(Path(analysis["merged_result_path"])),
        "repository_commit": repository_commit,
        "repository_commit_role": "benchmark_source_commit -- the Task 3 commit whose committed code produced the six raw trials and merged_result.json this artifact analyzes; NOT the commit that generated this artifact",
        "analysis_generation_commit": analysis_generation_commit,
        "analysis_generation_commit_role": "the exact Task 4 repository commit whose committed selector/CLI code produced this artifact",
        "common_scientific_output_digest": analysis["common_digest"],
        "all_candidates_scientifically_equal": len({t["scientific_output_digests"]["collection_digest"] for t in analysis["trials"]}) == 1,
        "candidates": {str(cs): candidate_block(cs) for cs in CANDIDATE_CHUNK_SIZES},
        "eligible_chunk_sizes": list(analysis["eligible"]),
        "selection_rule": {
            "primary_statistic": "arithmetic_mean_of_the_two_complete_measured_trial_primary_elapsed_ns_values",
            "tie_break_sequence": list(TIE_BREAK_SEQUENCE),
        },
        "selected_chunk_size": winner,
        "selected_mean_elapsed_ns": measurements[winner]["mean_elapsed_ns"] if winner is not None else None,
        "tie_break_path": analysis["tie_break_path"],
        "limitations": {
            "is_operational_engineering_selection": True,
            "selection_rule_frozen_after_observing_benchmark": True,
            "not_a_preregistered_scientific_hypothesis_test": True,
            "two_repeats_do_not_support_statistical_significance": True,
            "symmetric_trial_order_reduces_but_does_not_eliminate_temporal_cache_order_effects": True,
            "selection_does_not_change_scientific_results": True,
            "selection_does_not_define_conservation_tolerance": True,
            "selection_does_not_authorize_full_phase1b": True,
            "selection_does_not_authorize_or_implement_5000_image_evaluation": True,
            "gpu_execution_performed_by_task4": gpu_execution_performed_by_task4,
        },
        "complete": winner is not None,
        "final": winner is not None,
        "command": command,
    }


def render_markdown_report(artifact: Mapping[str, Any]) -> str:
    lines = [
        "# DFP Phase 1B Task 4 -- Chunk-Size Selection Report",
        "",
        "Operational engineering selection only. Not a scientific metric, not a preregistered "
        "hypothesis test, does not define a conservation tolerance, and does not authorize full "
        "Phase 1B execution or the 5,000-image evaluation.",
        "",
        f"- Benchmark-source commit (Task 3, produced the raw trials on a real H100 GPU): `{artifact['repository_commit']}`",
        f"- Analysis-generation commit (Task 4, CPU-only, produced THIS artifact): `{artifact['analysis_generation_commit']}`",
        f"- Source merged result: `{artifact['source_merged_result_path']}`",
        f"- Source merged result SHA-256: `{artifact['source_merged_result_sha256']}`",
        f"- E18 identity SHA-256: `{artifact['e18_identity_sha256']}`",
        f"- Common scientific-output digest: `{artifact['common_scientific_output_digest']}`",
        f"- All candidates scientifically equal: **{artifact['all_candidates_scientifically_equal']}**",
        f"- GPU execution performed by Task 4 (this analysis step): **{artifact['limitations']['gpu_execution_performed_by_task4']}** "
        "-- the Task 3 benchmark itself ran on a real NVIDIA H100; this Task 4 selection/regeneration step is CPU-only and ran no inference.",
        "",
        "## Candidates",
        "",
        "| chunk_size | repeat elapsed (ns) | mean elapsed (s) | vs selected | host peak RSS max | GPU reserved max | safe |",
        "|---|---|---|---|---|---|---|",
    ]
    for cs_str, c in artifact["candidates"].items():
        mean_s = c["mean_elapsed_ns"] / 1e9
        diff = c["difference_vs_selected_percent"]
        diff_str = "selected" if c["chunk_size"] == artifact["selected_chunk_size"] else f"{diff:+.3f}%"
        lines.append(
            f"| {c['chunk_size']} | {c['elapsed_ns_by_repeat']} | {mean_s:.3f} | {diff_str} | "
            f"{c['max_host_process_lifetime_peak_rss_bytes']} | {c['max_gpu_peak_reserved_bytes']} | {c['safe']} |"
        )
    lines += [
        "",
        f"## Selected chunk size: **{artifact['selected_chunk_size']}**",
        "",
        f"- Tie-break path: `{artifact['tie_break_path']}`",
        f"- Eligible (safe) candidates: {artifact['eligible_chunk_sizes']}",
        "",
        "## Limitations and disclosure",
        "",
    ]
    for key in artifact["limitations"]:
        lines.append(f"- {key.replace('_', ' ')}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Findings-repair round 2: independent verification of the standalone
# JSON artifact itself (distinct from ``verify_selection``, which
# verifies E18's own ``[selection_result]`` table). Never trusts any
# artifact field without either relational re-verification against
# live bytes/repository state, or full recomputation from the six raw
# trials.
# ---------------------------------------------------------------------

_ARTIFACT_TOP_KEYS = frozenset({
    "schema_name", "schema_version", "identity_name", "e18_identity_sha256", "e17_identity_sha256",
    "benchmark_manifest_sha256", "source_merged_result_path", "source_merged_result_sha256",
    "repository_commit", "repository_commit_role", "analysis_generation_commit", "analysis_generation_commit_role",
    "common_scientific_output_digest", "all_candidates_scientifically_equal", "candidates", "eligible_chunk_sizes",
    "selection_rule", "selected_chunk_size", "selected_mean_elapsed_ns", "tie_break_path", "limitations",
    "complete", "final", "command",
})
_ARTIFACT_LIMITATIONS_KEYS = frozenset({
    "is_operational_engineering_selection", "selection_rule_frozen_after_observing_benchmark",
    "not_a_preregistered_scientific_hypothesis_test", "two_repeats_do_not_support_statistical_significance",
    "symmetric_trial_order_reduces_but_does_not_eliminate_temporal_cache_order_effects",
    "selection_does_not_change_scientific_results", "selection_does_not_define_conservation_tolerance",
    "selection_does_not_authorize_full_phase1b", "selection_does_not_authorize_or_implement_5000_image_evaluation",
    "gpu_execution_performed_by_task4",
})


def verify_selection_artifact(artifact_path: Path, *, repo_root: Path) -> dict[str, Any]:
    """Independently re-verifies a generated ``phase1b_chunk_selection.json``
    artifact end to end: closed schema (including the two Findings-
    repair-round-2 fields), relational hash/commit verification against
    live repository state, and full recomputation of every candidate/
    selection field from the six raw trials -- never trusting any
    artifact field without independently checking it. Returns the
    parsed artifact on success."""
    root = Path(repo_root)
    try:
        artifact = json.loads(Path(artifact_path).read_text())
    except OSError as exc:
        raise DfpPhase1bChunkSelectionError(f"cannot read artifact {artifact_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DfpPhase1bChunkSelectionError(f"artifact {artifact_path} is not valid JSON: {exc}") from exc

    if not isinstance(artifact, dict):
        raise DfpPhase1bChunkSelectionError(f"artifact {artifact_path} must be a JSON object")
    actual_keys = frozenset(artifact.keys())
    if actual_keys != _ARTIFACT_TOP_KEYS:
        raise DfpPhase1bChunkSelectionError(
            f"artifact has the wrong key set -- missing={sorted(_ARTIFACT_TOP_KEYS - actual_keys)} "
            f"unknown={sorted(actual_keys - _ARTIFACT_TOP_KEYS)}"
        )

    if artifact["schema_name"] != SELECTION_SCHEMA_NAME or artifact["schema_version"] != SELECTION_SCHEMA_VERSION:
        raise DfpPhase1bChunkSelectionError("artifact schema_name/schema_version mismatch")

    # --- e18_identity_sha256: format + relational match against the live E18 file ---
    e18_hash = artifact["e18_identity_sha256"]
    if type(e18_hash) is not str or not _HEX64.match(e18_hash):
        raise DfpPhase1bChunkSelectionError(f"artifact e18_identity_sha256 must be a 64-lowercase-hex string, observed {e18_hash!r}")
    live_e18_path = root / IDENTITY_RELATIVE_PATH
    live_e18_hash = sha256_file(live_e18_path)
    if e18_hash != live_e18_hash:
        raise DfpPhase1bChunkSelectionError(
            f"artifact e18_identity_sha256 ({e18_hash}) disagrees with the live E18 file's actual hash ({live_e18_hash})"
        )
    e18_document = load_e18_identity(root)

    # --- repository_commit (benchmark-source) and analysis_generation_commit: format + relational ---
    for field, length, pattern in (("repository_commit", 40, _HEX40), ("analysis_generation_commit", 40, _HEX40)):
        value = artifact[field]
        if type(value) is not str or not pattern.match(value):
            raise DfpPhase1bChunkSelectionError(f"artifact.{field} must be a {length}-lowercase-hex commit id, observed {value!r}")
    if artifact["repository_commit"] == artifact["analysis_generation_commit"]:
        # Not inherently an error (a generation commit COULD equal the
        # benchmark-source commit in principle), but Task 4's own code
        # cannot exist at the exact benchmark-source commit (E17 predates
        # it), so this combination is never legitimate in practice --
        # fail closed rather than silently accept a swapped/duplicated pair.
        raise DfpPhase1bChunkSelectionError(
            "artifact.repository_commit and artifact.analysis_generation_commit must not be identical -- "
            "the benchmark-source commit predates any commit containing the Task 4 analysis code"
        )
    exists = _run_git_analysis_commit_check(root, "cat-file", "-e", f"{artifact['analysis_generation_commit']}^{{commit}}")
    if exists.returncode != 0:
        raise DfpPhase1bChunkSelectionError(f"artifact.analysis_generation_commit {artifact['analysis_generation_commit']!r} does not exist in this repository")
    try:
        verify_ancestor_relation(
            root, required_ancestor_commit=e18_document["identity"]["required_ancestor_commit"],
            repository_commit=artifact["analysis_generation_commit"],
        )
    except DfpPhase1bChunkBenchmarkIdentityError as exc:
        raise DfpPhase1bChunkSelectionError(f"artifact.analysis_generation_commit fails E18's required-ancestor relationship: {exc}") from exc

    # --- e17/manifest hash fields: relational against live files ---
    if artifact["e17_identity_sha256"] != e18_document["dependency_bindings"]["e17_identity"]["sha256"]:
        raise DfpPhase1bChunkSelectionError("artifact e17_identity_sha256 disagrees with E18's own bound E17 hash")
    if artifact["benchmark_manifest_sha256"] != e18_document["dependency_bindings"]["benchmark_manifest"]["sha256"]:
        raise DfpPhase1bChunkSelectionError("artifact benchmark_manifest_sha256 disagrees with E18's own bound manifest hash")

    # --- source_merged_result_sha256: relational against the live source file ---
    source_path = Path(artifact["source_merged_result_path"])
    actual_source_hash = sha256_file(source_path)
    if artifact["source_merged_result_sha256"] != actual_source_hash:
        raise DfpPhase1bChunkSelectionError(
            f"artifact source_merged_result_sha256 ({artifact['source_merged_result_sha256']}) disagrees with the live "
            f"source file's actual hash ({actual_source_hash})"
        )

    # --- gpu_execution_performed_by_task4: exact bool False ---
    _check_keys(artifact["limitations"], _ARTIFACT_LIMITATIONS_KEYS, "artifact.limitations")
    gpu_flag = artifact["limitations"]["gpu_execution_performed_by_task4"]
    if type(gpu_flag) is not bool or gpu_flag is not False:
        raise DfpPhase1bChunkSelectionError(
            f"artifact.limitations.gpu_execution_performed_by_task4 must be the exact bool False, observed {gpu_flag!r}"
        )
    for key in _ARTIFACT_LIMITATIONS_KEYS - {"gpu_execution_performed_by_task4"}:
        if artifact["limitations"].get(key) is not True:
            raise DfpPhase1bChunkSelectionError(f"artifact.limitations.{key} must be the exact bool True")

    # --- full decision-content recomputation from the six raw trials ---
    analysis = analyze_benchmark(repo_root=root, output_dir=source_path.parent, merged_result_path=source_path)

    # repository_commit (benchmark-source commit) is NEVER trusted from
    # the artifact -- it must equal the commit the six raw trials'
    # OWN code_provenance actually recorded, independently of whatever
    # the artifact claims (this is what catches a repository_commit /
    # analysis_generation_commit swap that would otherwise pass a
    # naive self-referential comparison).
    true_source_commit = analysis["merged"]["code_provenance"]["repository_commit"]
    if artifact["repository_commit"] != true_source_commit:
        raise DfpPhase1bChunkSelectionError(
            f"artifact.repository_commit ({artifact['repository_commit']}) disagrees with the benchmark-source "
            f"commit independently recorded in the raw trials' own code_provenance ({true_source_commit})"
        )
    if artifact["analysis_generation_commit"] == true_source_commit:
        raise DfpPhase1bChunkSelectionError(
            "artifact.analysis_generation_commit must not equal the benchmark-source commit -- "
            "possible repository_commit/analysis_generation_commit swap"
        )

    expected = build_selection_artifact(
        analysis, e18_sha256=e18_hash, e17_sha256=artifact["e17_identity_sha256"], manifest_sha256=artifact["benchmark_manifest_sha256"],
        repository_commit=artifact["repository_commit"], analysis_generation_commit=artifact["analysis_generation_commit"],
        gpu_execution_performed_by_task4=False, command=artifact["command"],
    )
    # source_merged_result_path/command are legitimate per-invocation
    # provenance and are excluded from the decision-content comparison;
    # everything else must match exactly.
    excluded = {"source_merged_result_path", "command"}
    mismatches = _diff_mapping(
        {k: v for k, v in expected.items() if k not in excluded},
        {k: v for k, v in artifact.items() if k not in excluded},
        path="artifact",
    )
    if mismatches:
        raise DfpPhase1bChunkSelectionError(
            "artifact decision content disagrees with the independently recomputed payload -- refusing to trust it:\n  "
            + "\n  ".join(mismatches)
        )

    return artifact


# ---------------------------------------------------------------------
# Output-path safety (Findings-repair Section 2): a single shared
# resolver/validator used by the CLI, the JSON writer, the Markdown
# writer, and the combined bundle installer. ``output_dir`` is the
# ONE allowed destination root (the same directory ``analyze``/
# ``verify-selection`` read the real benchmark trials/merged result
# from); ``--output``/``--markdown-output`` (and the direct API's
# ``basename`` arguments) are SAFE BASENAMES ONLY, never arbitrary
# paths -- this is what closes Finding 2 (traversal via ``..``,
# absolute paths, backslash/Windows/UNC paths, symlink escape).
# ---------------------------------------------------------------------

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_JSON_SUFFIX = ".json"
_MARKDOWN_SUFFIX = ".md"


def resolve_output_dir(output_dir: Path | str) -> Path:
    """Validates and resolves the operator-selected destination root.
    May be absolute or relative, anywhere on the filesystem (no
    repository- or /scratch-only restriction is imposed -- Task 4
    never invents a narrower policy than it already needs). Rejects a
    NUL byte or a lexical ``..`` path COMPONENT (an bare ``..``
    substring inside a longer component, e.g. a directory literally
    named ``a..b``, is not rejected -- only the exact traversal
    component). Fails closed if the resolved path does not exist or is
    not a directory; never silently substitutes or redirects
    elsewhere."""
    raw = str(output_dir)
    if "\x00" in raw:
        raise DfpPhase1bChunkSelectionError("output_dir must not contain a NUL byte")
    if ".." in Path(raw).parts:
        raise DfpPhase1bChunkSelectionError(f"output_dir must not contain a lexical '..' path component: {raw!r}")
    try:
        resolved = Path(output_dir).resolve(strict=True)
    except OSError as exc:
        raise DfpPhase1bChunkSelectionError(f"cannot resolve output_dir {output_dir!r}: {exc}") from exc
    if not resolved.is_dir():
        raise DfpPhase1bChunkSelectionError(f"output_dir is not a directory: {resolved}")
    return resolved


def resolve_safe_output_basename(basename: str, *, output_dir: Path, label: str, required_suffix: str) -> Path:
    """Validates ``basename`` as a single, safe path COMPONENT (never a
    path) and resolves it directly under the already-validated,
    resolved ``output_dir``. Rejects: non-string values, a NUL byte, an
    empty or whitespace-only value, leading/trailing whitespace, any
    ``/`` or ``\\`` (which would make it a path rather than a bare
    name -- this alone rules out lexical traversal, absolute paths, and
    UNC paths), the exact components ``.``/``..``, a Windows drive
    prefix (``C:...``), and the wrong required suffix. The substring
    ``..`` inside an otherwise-safe basename (e.g. ``run..v2.json``) is
    explicitly PERMITTED -- only the exact path component ``..`` is
    rejected, never a mere substring match. Also rejects a target that
    is an existing directory or an existing symlink (symlink-escape
    defense: writing through a symlink could otherwise land the file
    outside ``output_dir`` even though the referencing path itself
    looks contained)."""
    if type(basename) is not str:
        raise DfpPhase1bChunkSelectionError(f"{label} must be a string, observed {basename!r} ({type(basename).__name__})")
    if "\x00" in basename:
        raise DfpPhase1bChunkSelectionError(f"{label} must not contain a NUL byte")
    if basename.strip() == "":
        raise DfpPhase1bChunkSelectionError(f"{label} must not be empty or whitespace-only")
    if basename != basename.strip():
        raise DfpPhase1bChunkSelectionError(f"{label} must not have leading/trailing whitespace, observed {basename!r}")
    if "/" in basename or "\\" in basename:
        raise DfpPhase1bChunkSelectionError(f"{label} must be a bare filename with no directory separators, observed {basename!r}")
    if basename in (".", ".."):
        raise DfpPhase1bChunkSelectionError(f"{label} must not be '.' or '..', observed {basename!r}")
    if _WINDOWS_DRIVE_RE.match(basename):
        raise DfpPhase1bChunkSelectionError(f"{label} must not look like a Windows drive path, observed {basename!r}")
    if not basename.endswith(required_suffix):
        raise DfpPhase1bChunkSelectionError(f"{label} must end with {required_suffix!r}, observed {basename!r}")

    resolved_dir = Path(output_dir)  # already validated/resolved by the caller via resolve_output_dir
    candidate = resolved_dir / basename
    # Algebraically guaranteed contained (basename has no separator),
    # but re-verify explicitly rather than trusting that invariant silently.
    if candidate.parent != resolved_dir:
        raise DfpPhase1bChunkSelectionError(f"{label} does not resolve directly under output_dir: {candidate}")
    if candidate.is_symlink():
        raise DfpPhase1bChunkSelectionError(f"{label} target is a symlink -- refusing (symlink escape)")
    if candidate.exists() and candidate.is_dir():
        raise DfpPhase1bChunkSelectionError(f"{label} names an existing directory, not a file: {candidate}")
    return candidate


def write_selection_artifact_atomically(
    document: Mapping[str, Any], *, output_dir: Path | str, basename: str = "phase1b_chunk_selection.json",
    allow_overwrite: bool = False,
) -> Path:
    """Validates the destination (output_dir + safe basename) BEFORE
    any serialization, directory creation, or filesystem write, then
    installs atomically via write-to-temp-then-rename. Raises before
    ever touching the filesystem if the destination is unsafe or
    already exists without ``allow_overwrite``."""
    resolved_dir = resolve_output_dir(output_dir)
    dest = resolve_safe_output_basename(basename, output_dir=resolved_dir, label="artifact basename", required_suffix=_JSON_SUFFIX)
    if dest.exists() and not allow_overwrite:
        raise DfpPhase1bChunkSelectionError(f"refusing to overwrite existing selection artifact {dest} (allow_overwrite is False)")
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    # Staged self-verification: the just-serialized bytes must parse
    # back to an equal document before anything is written to disk.
    if json.loads(payload) != dict(document):
        raise DfpPhase1bChunkSelectionError("internal error: serialized selection artifact does not round-trip -- refusing to install")
    tmp_path = dest.with_name(f".{dest.name}.tmp")
    tmp_path.write_text(payload)
    tmp_path.replace(dest)
    return dest


def write_markdown_report(
    text: str, *, output_dir: Path | str, basename: str = "phase1b_chunk_selection_report.md", allow_overwrite: bool = False,
) -> Path:
    """Markdown counterpart of ``write_selection_artifact_atomically``
    -- identical destination-safety and atomicity contract."""
    resolved_dir = resolve_output_dir(output_dir)
    dest = resolve_safe_output_basename(basename, output_dir=resolved_dir, label="markdown basename", required_suffix=_MARKDOWN_SUFFIX)
    if dest.exists() and not allow_overwrite:
        raise DfpPhase1bChunkSelectionError(f"refusing to overwrite existing markdown report {dest} (allow_overwrite is False)")
    if type(text) is not str:
        raise DfpPhase1bChunkSelectionError(f"markdown report content must be a string, observed {type(text).__name__}")
    tmp_path = dest.with_name(f".{dest.name}.tmp")
    tmp_path.write_text(text)
    tmp_path.replace(dest)
    return dest


def install_selection_bundle(
    artifact: Mapping[str, Any], markdown_text: str, *, output_dir: Path | str,
    json_basename: str = "phase1b_chunk_selection.json", markdown_basename: str = "phase1b_chunk_selection_report.md",
    allow_overwrite: bool = False,
) -> tuple[Path, Path]:
    """Installs the JSON artifact and Markdown report as one atomic
    pair: BOTH destinations are validated (safety + existing-without-
    overwrite) and BOTH payloads are fully generated/serialized/staged-
    self-verified BEFORE either temp file is written, and both renames
    happen back-to-back -- a failure at any point before both renames
    complete leaves NEITHER destination installed/modified and no
    ``.tmp`` residue behind."""
    resolved_dir = resolve_output_dir(output_dir)
    json_dest = resolve_safe_output_basename(json_basename, output_dir=resolved_dir, label="--output", required_suffix=_JSON_SUFFIX)
    md_dest = resolve_safe_output_basename(markdown_basename, output_dir=resolved_dir, label="--markdown-output", required_suffix=_MARKDOWN_SUFFIX)
    if json_dest == md_dest:
        raise DfpPhase1bChunkSelectionError(f"--output and --markdown-output must not resolve to the same destination: {json_dest}")
    if json_dest.exists() and not allow_overwrite:
        raise DfpPhase1bChunkSelectionError(f"refusing to overwrite existing selection artifact {json_dest} (allow_overwrite is False)")
    if md_dest.exists() and not allow_overwrite:
        raise DfpPhase1bChunkSelectionError(f"refusing to overwrite existing markdown report {md_dest} (allow_overwrite is False)")
    if type(markdown_text) is not str:
        raise DfpPhase1bChunkSelectionError(f"markdown report content must be a string, observed {type(markdown_text).__name__}")

    json_payload = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if json.loads(json_payload) != dict(artifact):
        raise DfpPhase1bChunkSelectionError("internal error: serialized selection artifact does not round-trip -- refusing to install")

    json_tmp = json_dest.with_name(f".{json_dest.name}.tmp")
    md_tmp = md_dest.with_name(f".{md_dest.name}.tmp")
    try:
        json_tmp.write_text(json_payload)
        md_tmp.write_text(markdown_text)
        json_tmp.replace(json_dest)
        try:
            md_tmp.replace(md_dest)
        except OSError:
            # Best-effort rollback: undo the first rename so neither
            # destination ends up half-installed.
            if json_dest.exists():
                json_dest.unlink()
            raise
    finally:
        for stray in (json_tmp, md_tmp):
            if stray.exists():
                stray.unlink()
    return json_dest, md_dest


__all__ = [
    "DfpPhase1bChunkSelectionError",
    "IDENTITY_RELATIVE_PATH", "FORMAT_VERSION",
    "CANDIDATE_CHUNK_SIZES", "REQUIRED_REPEATS_PER_CANDIDATE", "REQUIRED_MEASURED_WINDOWS_PER_TRIAL",
    "TIE_BREAK_SEQUENCE", "SELECTION_SCHEMA_NAME", "SELECTION_SCHEMA_VERSION",
    "REQUIRED_GPU_MODEL", "REQUIRED_DEVICE_TYPE",
    "sha256_file", "load_e18_identity", "verify_e18_dependency_bindings", "verify_e18_against_e17",
    "load_and_verify_benchmark_output", "compute_candidate_measurements", "evaluate_candidate_safety",
    "select_winner", "analyze_benchmark", "verify_selection",
    "build_expected_selection_result",
    "resolve_analysis_generation_commit",
    "build_selection_artifact", "render_markdown_report", "verify_selection_artifact",
    "resolve_output_dir", "resolve_safe_output_basename",
    "write_selection_artifact_atomically", "write_markdown_report", "install_selection_bundle",
]
