"""Strict loading/validation for the DFP Phase 1B T64-early-exit
benchmark EXECUTION authority
(`evaluation_identities/e27_dfp_phase1b_t64_early_exit_benchmark_execution.toml`,
"E27").

E27 is a narrow EXECUTION-ONLY successor to E26
(`src.dfp_phase1b_t64_early_exit_benchmark_implementation_identity`).
It authorizes exactly one future H100 wall-clock benchmark campaign
using the already-implemented, already-independently-verified,
already-committed runner/CLI/launcher bound below by exact SHA-256. It
redefines no science (E25 remains the sole source of variants,
population, schedule, timing/memory/digest contracts, and the decision
rule), no artifact schema (E25's artifacts identity remains the sole
source), and no implementation requirement (E26 remains the sole
source). It narrowly supersedes E25's and E26's own "execution not
authorized yet" restrictions for THIS benchmark only -- E13 through E26
remain completely unmodified.

CRITICAL: the committed CLI and launcher this file describes never
import this module and never read this file at runtime -- there is no
code-level gate inside the existing, unmodified launcher that consults
E27 or refuses execution based on any authorization flag. E27's own
`verify_execution_readiness` is therefore a MANDATORY, SEPARATE,
READ-ONLY, torch-free command an operator must run (and receive a PASS
from) before manually invoking the unmodified launcher -- never a
runtime check embedded in that launcher itself.

This module implements no runner, no CLI, no launcher, and performs no
torch/CUDA/model/dataset access -- pure TOML/filesystem/hash/git
validation and exact comparison only.

`KeyboardInterrupt`, `SystemExit`, and `MemoryError` always propagate
unmodified; ordinary parse/type/path/relational failures are wrapped in
`DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError`.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping


class DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(ValueError):
    """Raised on any E27 identity invariant violation. Always fail
    closed. Never raised for, and never catches, KeyboardInterrupt /
    SystemExit / MemoryError -- those always propagate unmodified."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e27_dfp_phase1b_t64_early_exit_benchmark_execution.toml")
SCHEMA_NAME = "talk2dino-dfp-phase1b-t64-early-exit-benchmark-execution-identity-v1"
SCHEMA_VERSION = 1

REQUIRED_ANCESTOR_COMMIT = "af92d4acc64254674d2156a13b55bf4f5d368d3d"
EXPECTED_WORKING_BRANCH = "dfp-phase1a-diagnostics"
EXPECTED_SUCCESSOR_OF = "e26-dfp-phase1b-t64-early-exit-benchmark-implementation"

MANIFEST_SHA256 = "be2c77466a2c1d64f0f39f17f4d073ce1de92dc526a59e47ae8c3dabb8c644ab"
CANONICAL_MANIFEST_DIGEST = "4b962001ccab863b4c7a156cdbde5f55b66b3ff26a9d429a8b6f148550551847"
BENCHMARK_RECORD_DIGEST = "8193ef03f5d1dee843098ded6ce896f0f5fe072d49f37d2657ad9b6507fec142"

VARIANTS = ("T64_early_exit", "T320_reference")
T64_STEPS = 64
T320_STEPS = 320
WINDOW_COUNT = 24
MANIFEST_AUXILIARY_WARMUP_RECORD_COUNT = 1

# REPAIR (REPAIR_E27_WARMUP_SCOPE_DISCLOSURE): every trial -- warm-up or
# measured -- processes the FULL WINDOW_COUNT (24) windows; `run_trial`
# calls `benchmark_records_in_canonical_order(manifest)` unconditionally
# and its per-window loop never branches on `is_warmup`. There are 2
# warm-up trials (one per variant) and 6 measured trials (3 per variant
# x 2 variants), so 4 trials per variant total (1 warm-up + 3 measured).
WARMUP_TRIAL_COUNT = 2
MEASURED_TRIAL_COUNT = 6
WARMUP_WINDOW_EXECUTIONS = WARMUP_TRIAL_COUNT * WINDOW_COUNT
MEASURED_WINDOW_EXECUTIONS = MEASURED_TRIAL_COUNT * WINDOW_COUNT
TOTAL_WINDOW_EXECUTIONS = WARMUP_WINDOW_EXECUTIONS + MEASURED_WINDOW_EXECUTIONS
TRIALS_PER_VARIANT = (WARMUP_TRIAL_COUNT // len(VARIANTS)) + (MEASURED_TRIAL_COUNT // len(VARIANTS))
T64_CAMPAIGN_RECURRENCE_STEPS = TRIALS_PER_VARIANT * WINDOW_COUNT * T64_STEPS
T320_CAMPAIGN_RECURRENCE_STEPS = TRIALS_PER_VARIANT * WINDOW_COUNT * T320_STEPS
TOTAL_CAMPAIGN_RECURRENCE_STEPS = T64_CAMPAIGN_RECURRENCE_STEPS + T320_CAMPAIGN_RECURRENCE_STEPS

DEVELOPMENT_SPLIT_SIZE = 350
CONFIRMATION_SPLIT_SIZE = 150

COUNTERBALANCED_ORDER = (
    {"trial_index": 0, "variant": "T320_reference", "repeat": 0},
    {"trial_index": 1, "variant": "T64_early_exit", "repeat": 0},
    {"trial_index": 2, "variant": "T64_early_exit", "repeat": 1},
    {"trial_index": 3, "variant": "T320_reference", "repeat": 1},
    {"trial_index": 4, "variant": "T320_reference", "repeat": 2},
    {"trial_index": 5, "variant": "T64_early_exit", "repeat": 2},
)

END_TO_END_SPEEDUP_THRESHOLD = 1.25
DECISION_OUTCOMES = ("PROCEED_TO_FREEZE_T64_CONFIRMATION_PROTOCOL", "STOP_T64_EARLY_EXIT_TRACK")

ACTIVATION_PREREQUISITES = (
    "e27_independently_verified", "e27_committed", "clean_committed_head_production_readiness_pass",
    "verified_implementation_and_dependency_hashes", "fresh_operator_selected_destination", "explicit_operator_submission",
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------
# Small closed-schema primitives -- the same convention every other
# E-identity loader in this repository uses.
# ---------------------------------------------------------------------


def _require_string(value: Any, label: str) -> str:
    if type(value) is not str:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"{label} must be a string, observed {type(value).__name__}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    s = _require_string(value, label)
    if not _HEX64.match(s):
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"{label} must be lowercase 64-hex, observed {value!r}")
    return s


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"{label} must be a mapping, observed {type(value).__name__}")
    return value


def _require_closed(node: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    node = _require_mapping(node, label)
    actual = frozenset(node.keys())
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
    if missing or unknown:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"{label} has the wrong key set -- missing={missing} unknown={unknown}")
    return node


def _require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"{label} must be an int (not bool/float), observed {type(value).__name__}: {value!r}")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"{label} must be a bool, observed {type(value).__name__}: {value!r}")
    return value


def _require_exact_value(observed: Any, expected: Any, label: str) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"{label} must equal {expected!r}, observed {observed!r}")


def _resolve_repo_relative(root: Path, relative_path: str, *, label: str) -> Path:
    candidate = (root / relative_path).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"{label}: relative_path escapes the repository root: {relative_path!r}") from exc
    return candidate


def _import_verified_module(path: Path, *, module_name: str, label: str) -> Any:
    """REPAIR (REPAIR_E27_SPLIT_DEPENDENCY_BINDINGS): loads the Python
    module at EXACTLY ``path`` -- the same physical file whose bytes
    ``_verify_dependency_bindings`` just hash-verified -- via
    ``importlib``, never via an ordinary ``import`` statement. An
    ordinary ``import`` resolves through ``sys.path``/``sys.modules``,
    which is anchored at the real repository root regardless of the
    ``repo_root`` a caller is verifying (e.g. a fixture root in tests):
    that would let hash verification check one copy of a file while a
    completely different copy (or an already-cached module) is the one
    actually executed. Loading by exact path here guarantees the
    verified file and the executed file are always the same file. The
    loaded module is never registered under its canonical dotted name
    in ``sys.modules``, so it can never shadow or leak into the real
    cached module other code in this process relies on."""
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"{label}: cannot construct an import spec for the verified module at {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except MemoryError:
        raise
    except DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError:
        raise
    except Exception as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"{label}: the verified module at {path} raised while being imported: {exc}") from exc
    return module


# ---------------------------------------------------------------------
# [identity]
# ---------------------------------------------------------------------

_IDENTITY_KEYS = frozenset({
    "name", "schema_name", "schema_version", "description", "required_ancestor_commit", "working_branch",
    "successor_of", "supersedes_predecessor_execution_restriction", "supersedes_scope",
    "mutates_or_version_bumps_earlier_identity", "execution_gate_is_operator_process_not_runtime_launcher_check",
})


def _verify_identity_block(node: Any) -> None:
    node = _require_closed(node, _IDENTITY_KEYS, "identity")
    _require_exact_value(_require_string(node["name"], "identity.name"), "e27-dfp-phase1b-t64-early-exit-benchmark-execution", "identity.name")
    _require_exact_value(_require_string(node["schema_name"], "identity.schema_name"), SCHEMA_NAME, "identity.schema_name")
    _require_exact_value(_require_exact_int(node["schema_version"], "identity.schema_version"), SCHEMA_VERSION, "identity.schema_version")
    _require_string(node["description"], "identity.description")
    commit = _require_string(node["required_ancestor_commit"], "identity.required_ancestor_commit")
    if not _HEX40.match(commit):
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"identity.required_ancestor_commit must be 40-hex, observed {commit!r}")
    _require_exact_value(commit, REQUIRED_ANCESTOR_COMMIT, "identity.required_ancestor_commit")
    _require_exact_value(_require_string(node["working_branch"], "identity.working_branch"), EXPECTED_WORKING_BRANCH, "identity.working_branch")
    _require_exact_value(_require_string(node["successor_of"], "identity.successor_of"), EXPECTED_SUCCESSOR_OF, "identity.successor_of")
    _require_exact_value(_require_bool(node["supersedes_predecessor_execution_restriction"], "..supersedes"), True, "identity.supersedes_predecessor_execution_restriction")
    _require_string(node["supersedes_scope"], "identity.supersedes_scope")
    _require_exact_value(_require_bool(node["mutates_or_version_bumps_earlier_identity"], "..mutates"), False, "identity.mutates_or_version_bumps_earlier_identity")
    _require_exact_value(_require_bool(node["execution_gate_is_operator_process_not_runtime_launcher_check"], "..gate"), True, "identity.execution_gate_is_operator_process_not_runtime_launcher_check")


# ---------------------------------------------------------------------
# [dependency_bindings]: 20 ordinary repository bindings, all
# physically contained, none symbolic. e17_manifest additionally
# carries the two population digests.
# ---------------------------------------------------------------------

_ORDINARY_BINDING_KEYS = frozenset({"relative_path", "sha256", "binds"})
_MANIFEST_BINDING_KEYS = frozenset({"relative_path", "sha256", "binds", "canonical_manifest_digest", "benchmark_record_digest"})
_ALL_BINDING_NAMES = frozenset({
    "implementation_runner", "implementation_cli", "implementation_launcher",
    "e25_identity", "e25_artifacts", "e25_loader", "e26_identity", "e26_loader",
    "e17_manifest", "e13_protocol", "matched_evaluator",
    "graph_module", "finite_step_regime_module", "horizon_stitching_module", "stitching_pullback_module",
    "final_rescale_module", "production_provider_module", "artifact_schema_module",
    # REPAIR (REPAIR_E27_SPLIT_DEPENDENCY_BINDINGS): dataset_access_scope's
    # own population-overlap verification directly executes
    # balp_split_loader's load_and_verify_split -- unlike e22/e23/e24
    # (pinned only as opaque TOML blobs inside E25, never executed by
    # anyone in E27's chain), this module and the two data files it
    # reads are DIRECTLY bound and hash-verified here, exactly like
    # e25_loader/e26_loader, because E27 itself is the one executing it.
    "balp_split_loader", "balp_split_artifact", "balp_subset_manifest",
})
_SELF_REFERENCE_NAMES = ("e27_dfp_phase1b_t64_early_exit_benchmark_execution.toml", "dfp_phase1b_t64_early_exit_benchmark_execution_identity.py")

# REPAIR (REPAIR_E27_SPLIT_DEPENDENCY_BINDINGS): the three new BALP
# bindings additionally pin their EXACT canonical relative_path (never
# only their bytes), so a substituted path cannot be accepted merely
# because byte-identical content was copied to an unauthorized
# location. Pre-existing bindings are unchanged by this repair and are
# left as they were (out of this narrow repair's scope).
_CANONICAL_RELATIVE_PATHS = {
    "balp_split_loader": "src/dfp_phase1b_lightweight_boundary_identity.py",
    "balp_split_artifact": "evaluation_subsets/dfp_phase1b_lightweight_boundary_500_split.json",
    "balp_subset_manifest": "evaluation_subsets/dfp_phase1a_coco_stuff_500.json",
}


def _verify_dependency_bindings(root: Path, node: Any) -> None:
    node = _require_closed(node, _ALL_BINDING_NAMES, "dependency_bindings")
    for name in _ALL_BINDING_NAMES:
        expected_keys = _MANIFEST_BINDING_KEYS if name == "e17_manifest" else _ORDINARY_BINDING_KEYS
        binding = _require_closed(node[name], expected_keys, f"dependency_bindings.{name}")
        rel = _require_string(binding["relative_path"], f"dependency_bindings.{name}.relative_path")
        if name in _CANONICAL_RELATIVE_PATHS:
            _require_exact_value(rel, _CANONICAL_RELATIVE_PATHS[name], f"dependency_bindings.{name}.relative_path (must be the exact canonical path -- a substituted location is rejected even if its bytes match)")
        expected_hash = _require_sha256(binding["sha256"], f"dependency_bindings.{name}.sha256")
        _require_string(binding["binds"], f"dependency_bindings.{name}.binds")
        if name == "e17_manifest":
            _require_exact_value(_require_sha256(binding["canonical_manifest_digest"], "..cmd"), CANONICAL_MANIFEST_DIGEST, f"dependency_bindings.{name}.canonical_manifest_digest")
            _require_exact_value(_require_sha256(binding["benchmark_record_digest"], "..brd"), BENCHMARK_RECORD_DIGEST, f"dependency_bindings.{name}.benchmark_record_digest")
        target = _resolve_repo_relative(root, rel, label=f"dependency_bindings.{name}")
        try:
            observed_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as exc:
            raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"dependency_bindings.{name}: cannot read bound file {rel}: {exc}") from exc
        if observed_hash != expected_hash:
            raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"dependency_bindings.{name}: bound file {rel} sha256 drift -- expected {expected_hash}, observed {observed_hash}")
        if Path(rel).name in _SELF_REFERENCE_NAMES:
            raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"dependency_bindings.{name}: E27 must not bind its own identity/loader file (circular dependency)")


# ---------------------------------------------------------------------
# [execution_scope.resources]
# ---------------------------------------------------------------------

_RESOURCES_KEYS = frozenset({"gpu_count", "gpu_model", "gpu_model_string_match", "cpus_per_task", "host_memory", "max_wall_time", "account", "single_node_required"})


def _verify_resources(node: Any) -> None:
    node = _require_closed(node, _RESOURCES_KEYS, "execution_scope.resources")
    _require_exact_value(_require_exact_int(node["gpu_count"], "..gpu"), 1, "execution_scope.resources.gpu_count")
    _require_exact_value(_require_string(node["gpu_model"], "..model"), "h100_80gb", "execution_scope.resources.gpu_model")
    _require_exact_value(_require_string(node["gpu_model_string_match"], "..match"), "NVIDIA H100 80GB HBM3", "execution_scope.resources.gpu_model_string_match")
    _require_exact_value(_require_exact_int(node["cpus_per_task"], "..cpu"), 8, "execution_scope.resources.cpus_per_task")
    _require_exact_value(_require_string(node["host_memory"], "..mem"), "80G", "execution_scope.resources.host_memory")
    _require_exact_value(_require_string(node["max_wall_time"], "..wall"), "02:00:00", "execution_scope.resources.max_wall_time")
    _require_exact_value(_require_string(node["account"], "..account"), "rrg-yangw", "execution_scope.resources.account")
    _require_exact_value(_require_bool(node["single_node_required"], "..node"), True, "execution_scope.resources.single_node_required")


# ---------------------------------------------------------------------
# [execution_scope.variants]
# ---------------------------------------------------------------------

_VARIANTS_KEYS = frozenset({"list", "t64_early_exit", "t320_reference"})


def _verify_variants(node: Any) -> None:
    node = _require_closed(node, _VARIANTS_KEYS, "execution_scope.variants")
    _require_exact_value(list(node["list"]), list(VARIANTS), "execution_scope.variants.list")
    t64 = _require_closed(node["t64_early_exit"], frozenset({"recurrence_steps"}), "execution_scope.variants.t64_early_exit")
    _require_exact_value(_require_exact_int(t64["recurrence_steps"], "..steps"), T64_STEPS, "execution_scope.variants.t64_early_exit.recurrence_steps")
    t320 = _require_closed(node["t320_reference"], frozenset({"recurrence_steps"}), "execution_scope.variants.t320_reference")
    _require_exact_value(_require_exact_int(t320["recurrence_steps"], "..steps"), T320_STEPS, "execution_scope.variants.t320_reference.recurrence_steps")


# ---------------------------------------------------------------------
# [execution_scope.population]
# ---------------------------------------------------------------------

_POPULATION_KEYS = frozenset({
    "measured_window_count", "manifest_auxiliary_warmup_record_count", "warmup_trial_window_count",
    "measured_trial_window_count", "warmup_trials_process_canonical_benchmark_records", "deterministic_order",
    "manifest_override_authorized", "alternate_population_authorized", "debug_subset_authorized",
    "max_images_flag_authorized",
})


def _verify_population(node: Any) -> None:
    """REPAIR (REPAIR_E27_WARMUP_SCOPE_DISCLOSURE): `manifest_auxiliary_warmup_record_count`
    is the E17 manifest's own single auxiliary `warmup_record` field
    (metadata-resolution only, never processed as its own window) --
    never a count of "warm-up windows processed". Both
    `warmup_trial_window_count` and `measured_trial_window_count` equal
    the full `measured_window_count`, since every trial (warm-up or
    measured) processes the same full window population."""
    node = _require_closed(node, _POPULATION_KEYS, "execution_scope.population")
    _require_exact_value(_require_exact_int(node["measured_window_count"], "..count"), WINDOW_COUNT, "execution_scope.population.measured_window_count")
    _require_exact_value(_require_exact_int(node["manifest_auxiliary_warmup_record_count"], "..aux"), MANIFEST_AUXILIARY_WARMUP_RECORD_COUNT, "execution_scope.population.manifest_auxiliary_warmup_record_count")
    _require_exact_value(_require_exact_int(node["warmup_trial_window_count"], "..wtw"), WINDOW_COUNT, "execution_scope.population.warmup_trial_window_count")
    _require_exact_value(_require_exact_int(node["measured_trial_window_count"], "..mtw"), WINDOW_COUNT, "execution_scope.population.measured_trial_window_count")
    _require_exact_value(_require_bool(node["warmup_trials_process_canonical_benchmark_records"], "..wpcb"), True, "execution_scope.population.warmup_trials_process_canonical_benchmark_records")
    _require_string(node["deterministic_order"], "execution_scope.population.deterministic_order")
    for key in ("manifest_override_authorized", "alternate_population_authorized", "debug_subset_authorized", "max_images_flag_authorized"):
        _require_exact_value(_require_bool(node[key], f"..{key}"), False, f"execution_scope.population.{key}")


# ---------------------------------------------------------------------
# [execution_scope.campaign_workload]: SCHEDULED repeated-processing
# counts -- never evidence execution has occurred, never unique-window
# or unique-image counts. Every field is verified RELATIONALLY against
# execution_scope.population/schedule/variants, never as a bare literal
# echo of the TOML's own declared numbers.
# ---------------------------------------------------------------------

_CAMPAIGN_WORKLOAD_KEYS = frozenset({
    "warmup_trial_count", "measured_trial_count", "warmup_window_executions", "measured_window_executions",
    "total_window_executions", "trials_per_variant", "t64_campaign_recurrence_steps",
    "t320_campaign_recurrence_steps", "total_campaign_recurrence_steps",
    "counts_are_scheduled_workload_not_evidence_of_execution",
})


def _verify_campaign_workload(
    node: Any, *, window_count: int, schedule_warmup_trial_count: int, schedule_measured_trial_count: int,
    t64_steps: int, t320_steps: int, variant_count: int,
) -> None:
    node = _require_closed(node, _CAMPAIGN_WORKLOAD_KEYS, "execution_scope.campaign_workload")

    warmup_trial_count = _require_exact_int(node["warmup_trial_count"], "..warmup_trial_count")
    measured_trial_count = _require_exact_int(node["measured_trial_count"], "..measured_trial_count")
    _require_exact_value(warmup_trial_count, schedule_warmup_trial_count, "execution_scope.campaign_workload.warmup_trial_count (must equal execution_scope.schedule.warmup's own trial count)")
    _require_exact_value(measured_trial_count, schedule_measured_trial_count, "execution_scope.campaign_workload.measured_trial_count (must equal execution_scope.schedule.measured.ordered_trial_list's own length)")

    warmup_window_executions = _require_exact_int(node["warmup_window_executions"], "..wwe")
    measured_window_executions = _require_exact_int(node["measured_window_executions"], "..mwe")
    _require_exact_value(warmup_window_executions, warmup_trial_count * window_count, "execution_scope.campaign_workload.warmup_window_executions (must equal warmup_trial_count * measured_window_count -- every warm-up trial processes the full window population, never a single-window pass)")
    _require_exact_value(measured_window_executions, measured_trial_count * window_count, "execution_scope.campaign_workload.measured_window_executions (must equal measured_trial_count * measured_window_count)")

    total_window_executions = _require_exact_int(node["total_window_executions"], "..twe")
    _require_exact_value(total_window_executions, warmup_window_executions + measured_window_executions, "execution_scope.campaign_workload.total_window_executions")

    trials_per_variant = _require_exact_int(node["trials_per_variant"], "..tpv")
    if warmup_trial_count % variant_count != 0 or measured_trial_count % variant_count != 0:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError("execution_scope.campaign_workload: warmup/measured trial counts must divide evenly across execution_scope.variants.list")
    _require_exact_value(trials_per_variant, (warmup_trial_count // variant_count) + (measured_trial_count // variant_count), "execution_scope.campaign_workload.trials_per_variant (must equal one warm-up plus the per-variant measured-repeat count)")

    t64_recurrence = _require_exact_int(node["t64_campaign_recurrence_steps"], "..t64crs")
    t320_recurrence = _require_exact_int(node["t320_campaign_recurrence_steps"], "..t320crs")
    _require_exact_value(t64_recurrence, trials_per_variant * window_count * t64_steps, "execution_scope.campaign_workload.t64_campaign_recurrence_steps (must equal trials_per_variant * measured_window_count * the T64 variant's own recurrence_steps)")
    _require_exact_value(t320_recurrence, trials_per_variant * window_count * t320_steps, "execution_scope.campaign_workload.t320_campaign_recurrence_steps (must equal trials_per_variant * measured_window_count * the T320 variant's own recurrence_steps)")

    total_recurrence = _require_exact_int(node["total_campaign_recurrence_steps"], "..tcrs")
    _require_exact_value(total_recurrence, t64_recurrence + t320_recurrence, "execution_scope.campaign_workload.total_campaign_recurrence_steps")

    _require_exact_value(_require_bool(node["counts_are_scheduled_workload_not_evidence_of_execution"], "..flag"), True, "execution_scope.campaign_workload.counts_are_scheduled_workload_not_evidence_of_execution")


# ---------------------------------------------------------------------
# [execution_scope.schedule]
# ---------------------------------------------------------------------

_SCHEDULE_KEYS = frozenset({"warmup", "measured", "process_isolation"})
_SCHEDULE_WARMUP_KEYS = frozenset({"ordered_list", "runs_before_measured_matrix", "excluded_from_summary_statistics"})
_SCHEDULE_MEASURED_KEYS = frozenset({"ordered_trial_list"})
_ORDERED_TRIAL_ENTRY_KEYS = frozenset({"trial_index", "variant", "repeat"})
_PROCESS_ISOLATION_KEYS = frozenset({
    "fresh_python_process_per_trial_required", "same_allocated_gpu_node_required", "identical_precision_and_settings_required",
    "immediate_trial_verification_required", "fail_fast_required", "automatic_retry_authorized", "extra_repeats_authorized",
    "adaptive_horizons_authorized", "threshold_change_authorized",
})


def _verify_schedule(node: Any) -> None:
    node = _require_closed(node, _SCHEDULE_KEYS, "execution_scope.schedule")

    warmup = _require_closed(node["warmup"], _SCHEDULE_WARMUP_KEYS, "execution_scope.schedule.warmup")
    _require_exact_value(list(warmup["ordered_list"]), ["T64_early_exit", "T320_reference"], "execution_scope.schedule.warmup.ordered_list")
    _require_exact_value(_require_bool(warmup["runs_before_measured_matrix"], "..runs"), True, "execution_scope.schedule.warmup.runs_before_measured_matrix")
    _require_exact_value(_require_bool(warmup["excluded_from_summary_statistics"], "..excl"), True, "execution_scope.schedule.warmup.excluded_from_summary_statistics")

    measured = _require_closed(node["measured"], _SCHEDULE_MEASURED_KEYS, "execution_scope.schedule.measured")
    ordered = measured["ordered_trial_list"]
    if not isinstance(ordered, list) or len(ordered) != 6:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError("execution_scope.schedule.measured.ordered_trial_list must have exactly 6 entries")
    for position, (entry, expected) in enumerate(zip(ordered, COUNTERBALANCED_ORDER)):
        entry = _require_closed(entry, _ORDERED_TRIAL_ENTRY_KEYS, f"execution_scope.schedule.measured.ordered_trial_list[{position}]")
        if (
            _require_exact_int(entry["trial_index"], f"..[{position}].trial_index") != expected["trial_index"]
            or _require_string(entry["variant"], f"..[{position}].variant") != expected["variant"]
            or _require_exact_int(entry["repeat"], f"..[{position}].repeat") != expected["repeat"]
        ):
            raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"execution_scope.schedule.measured.ordered_trial_list[{position}] disagrees with the frozen required schedule")

    iso = _require_closed(node["process_isolation"], _PROCESS_ISOLATION_KEYS, "execution_scope.schedule.process_isolation")
    for key in ("fresh_python_process_per_trial_required", "same_allocated_gpu_node_required", "identical_precision_and_settings_required", "immediate_trial_verification_required", "fail_fast_required"):
        _require_exact_value(_require_bool(iso[key], f"..{key}"), True, f"execution_scope.schedule.process_isolation.{key}")
    for key in ("automatic_retry_authorized", "extra_repeats_authorized", "adaptive_horizons_authorized", "threshold_change_authorized"):
        _require_exact_value(_require_bool(iso[key], f"..{key}"), False, f"execution_scope.schedule.process_isolation.{key}")


_EXECUTION_SCOPE_KEYS = frozenset({"resources", "variants", "population", "schedule", "campaign_workload"})


def _verify_execution_scope(node: Any) -> None:
    node = _require_closed(node, _EXECUTION_SCOPE_KEYS, "execution_scope")
    _verify_resources(node["resources"])
    _verify_variants(node["variants"])
    _verify_population(node["population"])
    _verify_schedule(node["schedule"])
    _verify_campaign_workload(
        node["campaign_workload"],
        window_count=WINDOW_COUNT,
        schedule_warmup_trial_count=len(node["schedule"]["warmup"]["ordered_list"]),
        schedule_measured_trial_count=len(node["schedule"]["measured"]["ordered_trial_list"]),
        t64_steps=T64_STEPS,
        t320_steps=T320_STEPS,
        variant_count=len(VARIANTS),
    )


# ---------------------------------------------------------------------
# [dataset_access_scope]
# ---------------------------------------------------------------------

_DATASET_ACCESS_SCOPE_KEYS = frozenset({
    "authorized_access", "gt_access_authorized", "gt_label_use_authorized", "confirmation_accuracy_evaluation_authorized",
    "decoding_or_inference_on_unrelated_images_authorized", "metadata_indexing_scope", "population_overlap_disclosure",
})
_POPULATION_OVERLAP_KEYS = frozenset({
    "development_split_size", "confirmation_split_size", "disclosure_note",
    "processed_ids", "resolution_only_ids", "provider_resolution_ids",
})
_ID_POPULATION_KEYS = frozenset({"count", "development_overlap_count", "confirmation_overlap_count", "outside_frozen_population_count"})
_RESOLUTION_ONLY_ID_POPULATION_KEYS = _ID_POPULATION_KEYS | frozenset({"disjoint_from_processed_ids"})


def _verify_id_population_block(node: Any, keys: frozenset[str], *, label: str, expected_count: int, expected_dev: int, expected_conf: int, expected_outside: int) -> Mapping[str, Any]:
    node = _require_closed(node, keys, label)
    _require_exact_value(_require_exact_int(node["count"], f"{label}.count"), expected_count, f"{label}.count")
    _require_exact_value(_require_exact_int(node["development_overlap_count"], f"{label}.dev"), expected_dev, f"{label}.development_overlap_count")
    _require_exact_value(_require_exact_int(node["confirmation_overlap_count"], f"{label}.conf"), expected_conf, f"{label}.confirmation_overlap_count")
    _require_exact_value(_require_exact_int(node["outside_frozen_population_count"], f"{label}.outside"), expected_outside, f"{label}.outside_frozen_population_count")
    return node


def _verify_dataset_access_scope(root: Path, e17_manifest_relative_path: str, balp_split_loader_relative_path: str, node: Any) -> None:
    """REPAIR (REPAIR_E27_WARMUP_SCOPE_DISCLOSURE): independently
    reconstructs three DISTINCT image-ID populations from source (the
    committed BALP split loader plus a direct read of the E17 manifest
    JSON) rather than merely checking the TOML's own internal
    consistency:
      processed_ids            -- the 24 canonical benchmark_records'
                                   image IDs, actually decoded and run
                                   through inference every trial.
      resolution_only_ids      -- the manifest's own single auxiliary
                                   warmup_record image ID, metadata-
                                   resolved only (confirmed disjoint
                                   from processed_ids; never decoded or
                                   inferred by the committed runner).
      provider_resolution_ids  -- their union, exactly the committed
                                   CLI's own `selected_ids` construction
                                   (`diagnostics/run_dfp_phase1b_t64_
                                   early_exit_benchmark.py`:
                                   `sorted({warmup_record["image_id"]} |
                                   {r["image_id"] for r in
                                   benchmark_records})`).
    """
    node = _require_closed(node, _DATASET_ACCESS_SCOPE_KEYS, "dataset_access_scope")
    _require_string(node["authorized_access"], "dataset_access_scope.authorized_access")
    for key in ("gt_access_authorized", "gt_label_use_authorized", "confirmation_accuracy_evaluation_authorized", "decoding_or_inference_on_unrelated_images_authorized"):
        _require_exact_value(_require_bool(node[key], f"..{key}"), False, f"dataset_access_scope.{key}")
    _require_string(node["metadata_indexing_scope"], "dataset_access_scope.metadata_indexing_scope")

    overlap = _require_closed(node["population_overlap_disclosure"], _POPULATION_OVERLAP_KEYS, "dataset_access_scope.population_overlap_disclosure")
    _require_exact_value(_require_exact_int(overlap["development_split_size"], "..dev"), DEVELOPMENT_SPLIT_SIZE, "..development_split_size")
    _require_exact_value(_require_exact_int(overlap["confirmation_split_size"], "..conf"), CONFIRMATION_SPLIT_SIZE, "..confirmation_split_size")
    _require_string(overlap["disclosure_note"], "dataset_access_scope.population_overlap_disclosure.disclosure_note")

    import json as _json

    # REPAIR (REPAIR_E27_SPLIT_DEPENDENCY_BINDINGS): balp_split_loader_relative_path
    # was already SHA-256 hash-verified by _verify_dependency_bindings,
    # called earlier in load_identity, BEFORE this function ever runs --
    # so by the time control reaches here the module bytes at this exact
    # path are already known-good. It is loaded by exact verified path
    # (never a bare `import src.dfp_phase1b_lightweight_boundary_identity`,
    # which would resolve through sys.path instead of the verified
    # `root`) so the file that was hash-checked and the file that is
    # executed can never diverge.
    balp_module_path = _resolve_repo_relative(root, balp_split_loader_relative_path, label="dataset_access_scope (balp_split_loader)")
    balp = _import_verified_module(balp_module_path, module_name="_e27_verified_balp_split_loader", label="dataset_access_scope.balp_split_loader")

    try:
        split = balp.load_and_verify_split(repo_root=root)
    except balp.DfpPhase1bLightweightBoundaryIdentityError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"dataset_access_scope: independent BALP split reverification failed: {exc}") from exc
    development_ids = set(split["development_image_ids"])
    confirmation_ids = set(split["confirmation_image_ids"])
    if len(development_ids) != DEVELOPMENT_SPLIT_SIZE or len(confirmation_ids) != CONFIRMATION_SPLIT_SIZE:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError("dataset_access_scope: independently loaded split sizes disagree with the frozen 350/150 split")

    manifest_path = _resolve_repo_relative(root, e17_manifest_relative_path, label="dataset_access_scope (e17 manifest)")
    try:
        manifest_doc = _json.loads(manifest_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"dataset_access_scope: cannot read/parse E17 manifest {manifest_path}: {exc}") from exc

    processed_ids = {r["image_id"] for r in manifest_doc["benchmark_records"]}
    resolution_only_ids = {manifest_doc["warmup_record"]["image_id"]}
    if len(processed_ids) != WINDOW_COUNT:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"dataset_access_scope: E17 manifest processed_ids count disagrees with the frozen {WINDOW_COUNT}-window population (observed {len(processed_ids)} distinct)")
    if resolution_only_ids & processed_ids:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError("dataset_access_scope: the manifest's auxiliary warmup_record image ID is not disjoint from processed_ids")
    provider_resolution_ids = processed_ids | resolution_only_ids
    if len(provider_resolution_ids) != len(processed_ids) + len(resolution_only_ids):
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError("dataset_access_scope: provider_resolution_ids is not the disjoint union of processed_ids and resolution_only_ids")

    def overlap_counts(ids: set) -> tuple[int, int, int]:
        return len(ids & development_ids), len(ids & confirmation_ids), len(ids - development_ids - confirmation_ids)

    processed_dev, processed_conf, processed_outside = overlap_counts(processed_ids)
    _verify_id_population_block(
        overlap["processed_ids"], _ID_POPULATION_KEYS, label="dataset_access_scope.population_overlap_disclosure.processed_ids",
        expected_count=len(processed_ids), expected_dev=processed_dev, expected_conf=processed_conf, expected_outside=processed_outside,
    )

    resolution_dev, resolution_conf, resolution_outside = overlap_counts(resolution_only_ids)
    resolution_node = _verify_id_population_block(
        overlap["resolution_only_ids"], _RESOLUTION_ONLY_ID_POPULATION_KEYS, label="dataset_access_scope.population_overlap_disclosure.resolution_only_ids",
        expected_count=len(resolution_only_ids), expected_dev=resolution_dev, expected_conf=resolution_conf, expected_outside=resolution_outside,
    )
    _require_exact_value(_require_bool(resolution_node["disjoint_from_processed_ids"], "..disjoint"), True, "dataset_access_scope.population_overlap_disclosure.resolution_only_ids.disjoint_from_processed_ids")

    provider_dev, provider_conf, provider_outside = overlap_counts(provider_resolution_ids)
    _verify_id_population_block(
        overlap["provider_resolution_ids"], _ID_POPULATION_KEYS, label="dataset_access_scope.population_overlap_disclosure.provider_resolution_ids",
        expected_count=len(provider_resolution_ids), expected_dev=provider_dev, expected_conf=provider_conf, expected_outside=provider_outside,
    )


# ---------------------------------------------------------------------
# [output_lifecycle_contract]
# ---------------------------------------------------------------------

_OUTPUT_LIFECYCLE_KEYS = frozenset({
    "launcher_required_environment_variables", "launcher_optional_environment_variables", "run_output_root_pattern",
    "trial_artifact_naming", "result_bundle_directory_name", "result_bundle_contents",
    "explicit_absolute_output_root_selected_and_recorded_at_readiness_required", "fresh_absent_final_destination_required",
    "lexical_traversal_authorized", "protected_source_overlap_authorized", "overwrite_authorized", "force_mode_authorized",
    "unique_campaign_destination_required", "single_verified_campaign_per_destination_required",
    "staged_self_verification_required", "atomic_installation_required", "result_and_decision_published_together_required",
    "installed_pair_reopened_and_publicly_verified_required", "success_before_verification_authorized",
    "automatic_deletion_or_replacement_of_existing_output_authorized", "new_artifact_schema_or_sidecar_authorized",
    "unimplemented_cli_flag_authorized", "campaign_provenance_recording_limitation",
})

_REQUIRED_LAUNCHER_ENV_VARS = ("DFP_REPO_ROOT", "DFP_VENV_ACTIVATE_PATH", "COCO_STUFF_REAL_DATA_ROOT", "TALK2DINO_WEIGHT_DIR", "DFP_BENCHMARK_OUTPUT_ROOT")
_OPTIONAL_LAUNCHER_ENV_VARS = ("DFP_OPENCV_MODULE",)


def _verify_output_lifecycle_contract(node: Any) -> None:
    node = _require_closed(node, _OUTPUT_LIFECYCLE_KEYS, "output_lifecycle_contract")
    if set(node["launcher_required_environment_variables"]) != set(_REQUIRED_LAUNCHER_ENV_VARS):
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError("output_lifecycle_contract.launcher_required_environment_variables disagrees with the frozen required set")
    if set(node["launcher_optional_environment_variables"]) != set(_OPTIONAL_LAUNCHER_ENV_VARS):
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError("output_lifecycle_contract.launcher_optional_environment_variables disagrees with the frozen optional set")
    _require_string(node["run_output_root_pattern"], "output_lifecycle_contract.run_output_root_pattern")
    _require_string(node["trial_artifact_naming"], "output_lifecycle_contract.trial_artifact_naming")
    _require_exact_value(_require_string(node["result_bundle_directory_name"], "..dirname"), "result", "output_lifecycle_contract.result_bundle_directory_name")
    _require_exact_value(list(node["result_bundle_contents"]), ["merged_result.json", "decision.json"], "output_lifecycle_contract.result_bundle_contents")
    for key in (
        "explicit_absolute_output_root_selected_and_recorded_at_readiness_required", "fresh_absent_final_destination_required",
        "unique_campaign_destination_required", "single_verified_campaign_per_destination_required",
        "staged_self_verification_required", "atomic_installation_required", "result_and_decision_published_together_required",
        "installed_pair_reopened_and_publicly_verified_required",
    ):
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"output_lifecycle_contract.{key}")
    for key in (
        "lexical_traversal_authorized", "protected_source_overlap_authorized", "overwrite_authorized", "force_mode_authorized",
        "success_before_verification_authorized", "automatic_deletion_or_replacement_of_existing_output_authorized",
        "new_artifact_schema_or_sidecar_authorized", "unimplemented_cli_flag_authorized",
    ):
        _require_exact_value(_require_bool(node[key], f"..{key}"), False, f"output_lifecycle_contract.{key}")
    _require_string(node["campaign_provenance_recording_limitation"], "output_lifecycle_contract.campaign_provenance_recording_limitation")


# ---------------------------------------------------------------------
# [measurement_bindings]
# ---------------------------------------------------------------------

_MEASUREMENT_BINDINGS_KEYS = frozenset({
    "primary_timing_field", "timing_phase_fields", "stage_timing_fields", "derived_fields", "memory_fields",
    "scientific_output_fields", "expected_counts_per_measured_trial", "window_throughput_disclosure",
})
_TIMING_PHASE_FIELDS = (
    "process_total_wall_ns", "preflight_wall_ns", "dataset_construction_wall_ns",
    "model_construction_and_checkpoint_loading_wall_ns", "measured_window_loop_wall_ns",
    "artifact_validation_and_write_wall_ns", "unattributed_overhead_ns",
)
_STAGE_TIMING_FIELDS = ("input_preparation", "unary_model_forward", "graph_construction", "canonical_recurrence", "stitching_and_rescale", "output_validation")
_DERIVED_FIELDS = ("measured_window_loop_seconds", "window_throughput_per_second", "mean_seconds_per_window")
_MEMORY_FIELDS = ("host_peak_rss_bytes", "gpu_peak_memory_allocated_bytes", "gpu_peak_memory_reserved_bytes", "gpu_total_memory_bytes")
_SCIENTIFIC_OUTPUT_FIELDS = ("scientific_output_collection_digest", "per_window_output_digests", "finite_value_count", "non_finite_value_count")
_EXPECTED_COUNTS_KEYS = frozenset({"window_count", "model_forward_count", "graph_construction_count", "t64_recurrence_steps_executed", "t320_recurrence_steps_executed"})
_WINDOW_THROUGHPUT_DISCLOSURE_KEYS = frozenset({"represents_image_throughput", "represents_full_5000_image_runtime", "disclosure_note"})


def _verify_measurement_bindings(node: Any) -> None:
    node = _require_closed(node, _MEASUREMENT_BINDINGS_KEYS, "measurement_bindings")
    _require_exact_value(_require_string(node["primary_timing_field"], "..primary"), "measured_window_loop_wall_ns", "measurement_bindings.primary_timing_field")
    _require_exact_value(list(node["timing_phase_fields"]), list(_TIMING_PHASE_FIELDS), "measurement_bindings.timing_phase_fields")
    _require_exact_value(list(node["stage_timing_fields"]), list(_STAGE_TIMING_FIELDS), "measurement_bindings.stage_timing_fields")
    _require_exact_value(list(node["derived_fields"]), list(_DERIVED_FIELDS), "measurement_bindings.derived_fields")
    _require_exact_value(list(node["memory_fields"]), list(_MEMORY_FIELDS), "measurement_bindings.memory_fields")
    _require_exact_value(list(node["scientific_output_fields"]), list(_SCIENTIFIC_OUTPUT_FIELDS), "measurement_bindings.scientific_output_fields")

    counts = _require_closed(node["expected_counts_per_measured_trial"], _EXPECTED_COUNTS_KEYS, "measurement_bindings.expected_counts_per_measured_trial")
    _require_exact_value(_require_exact_int(counts["window_count"], "..w"), WINDOW_COUNT, "..window_count")
    _require_exact_value(_require_exact_int(counts["model_forward_count"], "..f"), WINDOW_COUNT, "..model_forward_count")
    _require_exact_value(_require_exact_int(counts["graph_construction_count"], "..g"), WINDOW_COUNT, "..graph_construction_count")
    _require_exact_value(_require_exact_int(counts["t64_recurrence_steps_executed"], "..t64"), WINDOW_COUNT * T64_STEPS, "..t64_recurrence_steps_executed")
    _require_exact_value(_require_exact_int(counts["t320_recurrence_steps_executed"], "..t320"), WINDOW_COUNT * T320_STEPS, "..t320_recurrence_steps_executed")

    disclosure = _require_closed(node["window_throughput_disclosure"], _WINDOW_THROUGHPUT_DISCLOSURE_KEYS, "measurement_bindings.window_throughput_disclosure")
    _require_exact_value(_require_bool(disclosure["represents_image_throughput"], "..img"), False, "..represents_image_throughput")
    _require_exact_value(_require_bool(disclosure["represents_full_5000_image_runtime"], "..full"), False, "..represents_full_5000_image_runtime")
    _require_string(disclosure["disclosure_note"], "measurement_bindings.window_throughput_disclosure.disclosure_note")


# ---------------------------------------------------------------------
# [decision_binding]
# ---------------------------------------------------------------------

_DECISION_BINDING_KEYS = frozenset({
    "primary_speedup_formula", "end_to_end_speedup_threshold", "threshold_comparison", "outcomes",
    "proceed_outcome_authorizes_confirmation_execution", "proceed_outcome_authorizes_full_5000_evaluation",
    "proceed_outcome_authorizes_only",
})


def _verify_decision_binding(node: Any) -> None:
    node = _require_closed(node, _DECISION_BINDING_KEYS, "decision_binding")
    _require_string(node["primary_speedup_formula"], "decision_binding.primary_speedup_formula")
    if node["end_to_end_speedup_threshold"] != END_TO_END_SPEEDUP_THRESHOLD:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError("decision_binding.end_to_end_speedup_threshold disagrees with the frozen E25 threshold")
    _require_exact_value(_require_string(node["threshold_comparison"], "..cmp"), ">=", "decision_binding.threshold_comparison")
    _require_exact_value(list(node["outcomes"]), list(DECISION_OUTCOMES), "decision_binding.outcomes")
    _require_exact_value(_require_bool(node["proceed_outcome_authorizes_confirmation_execution"], "..conf"), False, "..proceed_outcome_authorizes_confirmation_execution")
    _require_exact_value(_require_bool(node["proceed_outcome_authorizes_full_5000_evaluation"], "..full"), False, "..proceed_outcome_authorizes_full_5000_evaluation")
    _require_string(node["proceed_outcome_authorizes_only"], "decision_binding.proceed_outcome_authorizes_only")


# ---------------------------------------------------------------------
# [authorization_boundary]: 20 keys, exactly 5 true, 15 false.
# ---------------------------------------------------------------------

_AUTHORIZATION_BOUNDARY_TRUE_KEYS = (
    "scoped_dataset_image_access_authorized", "bound_model_construction_and_checkpoint_loading_authorized",
    "cuda_initialization_authorized", "h100_benchmark_campaign_authorized", "artifact_verification_and_merge_authorized",
)
_AUTHORIZATION_BOUNDARY_FALSE_KEYS = (
    "agent_job_submission_during_this_task_authorized", "gt_access_authorized", "training_authorized",
    "weight_modification_authorized", "scientific_kernel_modification_authorized", "alternate_population_authorized",
    "extra_trials_authorized", "confirmation_accuracy_evaluation_authorized", "full_5000_image_evaluation_authorized",
    "balp_candidate_evaluation_authorized", "exact_masked_forward_provenance_authorized", "overwrite_authorized",
    "force_mode_authorized", "e25_modification_authorized", "e26_modification_authorized",
    "new_authority_to_bypass_activation_prerequisites_authorized",
)
_AUTHORIZATION_BOUNDARY_KEYS = frozenset(_AUTHORIZATION_BOUNDARY_TRUE_KEYS) | frozenset(_AUTHORIZATION_BOUNDARY_FALSE_KEYS)


def _verify_authorization_boundary(node: Any) -> None:
    node = _require_closed(node, _AUTHORIZATION_BOUNDARY_KEYS, "authorization_boundary")
    for key in _AUTHORIZATION_BOUNDARY_TRUE_KEYS:
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"authorization_boundary.{key}")
    for key in _AUTHORIZATION_BOUNDARY_FALSE_KEYS:
        _require_exact_value(_require_bool(node[key], f"..{key}"), False, f"authorization_boundary.{key}")


# ---------------------------------------------------------------------
# [activation_prerequisites]
# ---------------------------------------------------------------------

_ACTIVATION_PREREQUISITES_KEYS = frozenset({"list", "all_required", "this_task_satisfies_none_of_these_by_itself"})


def _verify_activation_prerequisites(node: Any) -> None:
    node = _require_closed(node, _ACTIVATION_PREREQUISITES_KEYS, "activation_prerequisites")
    _require_exact_value(tuple(node["list"]), ACTIVATION_PREREQUISITES, "activation_prerequisites.list")
    _require_exact_value(_require_bool(node["all_required"], "..all"), True, "activation_prerequisites.all_required")
    _require_exact_value(_require_bool(node["this_task_satisfies_none_of_these_by_itself"], "..none"), True, "activation_prerequisites.this_task_satisfies_none_of_these_by_itself")


# ---------------------------------------------------------------------
# [execution_readiness_contract]
# ---------------------------------------------------------------------

_EXECUTION_READINESS_CONTRACT_KEYS = frozenset({
    "verification_entry_point", "torch_free_required", "checks", "never_checks_gpu_dataset_model_or_cuda",
    "launcher_does_not_import_this_module", "operator_must_run_this_separately_before_submission",
})


def _verify_execution_readiness_contract(node: Any) -> None:
    node = _require_closed(node, _EXECUTION_READINESS_CONTRACT_KEYS, "execution_readiness_contract")
    _require_string(node["verification_entry_point"], "execution_readiness_contract.verification_entry_point")
    for key in ("torch_free_required", "never_checks_gpu_dataset_model_or_cuda", "launcher_does_not_import_this_module", "operator_must_run_this_separately_before_submission"):
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"execution_readiness_contract.{key}")
    checks = node["checks"]
    if not isinstance(checks, list) or len(checks) != 4:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError("execution_readiness_contract.checks must be a 4-entry list")
    for entry in checks:
        _require_string(entry, "execution_readiness_contract.checks[]")


# ---------------------------------------------------------------------
# [deferred_next_steps]
# ---------------------------------------------------------------------

_DEFERRED_NEXT_STEPS_KEYS = frozenset({"sequence", "no_step_in_this_sequence_is_performed_by_this_task"})
_REQUIRED_SEQUENCE = (
    "independent_e27_verification", "commit_e27", "committed_head_production_readiness",
    "operator_manually_submits_the_verified_launcher", "independent_benchmark_result_verification",
)


def _verify_deferred_next_steps(node: Any) -> None:
    node = _require_closed(node, _DEFERRED_NEXT_STEPS_KEYS, "deferred_next_steps")
    _require_exact_value(tuple(node["sequence"]), _REQUIRED_SEQUENCE, "deferred_next_steps.sequence")
    _require_exact_value(_require_bool(node["no_step_in_this_sequence_is_performed_by_this_task"], "..none"), True, "deferred_next_steps.no_step_in_this_sequence_is_performed_by_this_task")


# ---------------------------------------------------------------------
# [prohibited]
# ---------------------------------------------------------------------

_REQUIRED_PROHIBITED = (
    "agent_job_submission_during_this_task", "gt_access", "training", "weight_modification",
    "scientific_kernel_modification", "alternate_populations", "extra_trials", "confirmation_accuracy_evaluation",
    "full_5000_image_evaluation", "balp_candidate_evaluation", "exact_masked_forward_provenance", "overwrite_or_force",
    "creating_another_authority_merely_to_authorize_the_readiness_check",
)


def _verify_prohibited(node: Any) -> None:
    node = _require_closed(node, frozenset({"list"}), "prohibited")
    _require_exact_value(tuple(node["list"]), _REQUIRED_PROHIBITED, "prohibited.list")


# ---------------------------------------------------------------------
# Repository gate: current HEAD must have required_ancestor_commit as
# an ancestor (never exact equality) -- so E27 can be verified from any
# later clean descendant HEAD, not only from the implementation commit
# itself.
# ---------------------------------------------------------------------


def verify_repository_gate(repo_root: Path, repository_commit: str) -> None:
    commit = _require_string(repository_commit, "repository_commit")
    if not _HEX40.match(commit):
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"repository_commit must be a 40-hex commit, observed {commit!r}")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", REQUIRED_ANCESTOR_COMMIT, commit],
            capture_output=True, text=True,
        )
    except OSError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"cannot invoke git for ancestor verification: {exc}") from exc
    if result.returncode != 0:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"required_ancestor_commit {REQUIRED_ANCESTOR_COMMIT} is not an ancestor of (or equal to) {commit}")


# ---------------------------------------------------------------------
# Top-level load/verify.
# ---------------------------------------------------------------------

_TOP_LEVEL_KEYS = frozenset({
    "format_version", "identity", "dependency_bindings", "execution_scope", "dataset_access_scope",
    "output_lifecycle_contract", "measurement_bindings", "decision_binding", "authorization_boundary",
    "activation_prerequisites", "execution_readiness_contract", "deferred_next_steps", "prohibited",
})


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load, closed-schema-validate, exact-value-validate, and
    relationally hash-verify the E27 execution-authority identity.
    Never mutates or re-derives any bound dependency -- only
    re-verifies SHA-256/content against current bytes. Performs no
    torch/CUDA/model/dataset access, and does NOT itself check git
    ancestry or working-tree cleanliness -- see `verify_repository_gate`
    and `verify_execution_readiness` for that."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with open(source, "rb") as fh:
            document = tomllib.load(fh)
    except OSError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"cannot read E27 identity {source}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"E27 identity {source} is not valid TOML: {exc}") from exc

    _require_closed(document, _TOP_LEVEL_KEYS, "<top level>")
    _require_exact_value(
        _require_string(document["format_version"], "format_version"),
        "talk2dino-dfp-phase1b-t64-early-exit-benchmark-execution-identity-v1", "format_version",
    )
    _verify_identity_block(document["identity"])
    _verify_dependency_bindings(root, document["dependency_bindings"])
    _verify_execution_scope(document["execution_scope"])
    _verify_dataset_access_scope(
        root,
        document["dependency_bindings"]["e17_manifest"]["relative_path"],
        document["dependency_bindings"]["balp_split_loader"]["relative_path"],
        document["dataset_access_scope"],
    )
    _verify_output_lifecycle_contract(document["output_lifecycle_contract"])
    _verify_measurement_bindings(document["measurement_bindings"])
    _verify_decision_binding(document["decision_binding"])
    _verify_authorization_boundary(document["authorization_boundary"])
    _verify_activation_prerequisites(document["activation_prerequisites"])
    _verify_execution_readiness_contract(document["execution_readiness_contract"])
    _verify_deferred_next_steps(document["deferred_next_steps"])
    _verify_prohibited(document["prohibited"])
    return document


# ---------------------------------------------------------------------
# Execution readiness: the mandatory, separate, read-only, torch-free
# command an operator must run (and receive a PASS from) before
# manually invoking the existing, unmodified launcher. This function is
# never called automatically by the launcher or CLI -- it is a
# standalone entry point this module alone exposes.
# ---------------------------------------------------------------------


def verify_execution_readiness(repo_root: Path | None = None) -> dict[str, Any]:
    """Ordinary public loading (`load_identity`) can inspect an
    uncommitted E27 authority's own internal consistency at any time
    (including before it is committed) -- that alone never means
    execution is "activated". This function additionally requires: E25
    loads cleanly (torch-free, via its own unmodified loader,
    transitively re-verifying its full dependency chain), E26 loads
    cleanly (ditto), the current repository HEAD has
    required_ancestor_commit as an ancestor, and the current working
    tree is completely clean. It still performs no GPU/dataset/model/
    CUDA access of any kind -- passing this check is necessary, but
    never by itself sufficient, for real execution (see
    `activation_prerequisites` in the identity document itself)."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    document = load_identity(repo_root=root)

    import src.dfp_phase1b_t64_early_exit_benchmark_identity as e25_identity
    import src.dfp_phase1b_t64_early_exit_benchmark_implementation_identity as e26_identity

    try:
        e25_identity.load_identity(repo_root=root)
        e25_identity.load_artifacts_identity(repo_root=root)
        e26_identity.load_identity(repo_root=root)
    except (e25_identity.DfpPhase1bT64EarlyExitBenchmarkIdentityError, e26_identity.DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError) as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(f"E25/E26 re-verification failed: {exc}") from exc

    from src.dfp_phase1b_chunk_benchmark_identity import DfpPhase1bChunkBenchmarkIdentityError, require_clean_worktree

    try:
        repository_commit = require_clean_worktree(root)
        verify_repository_gate(root, repository_commit)
    except DfpPhase1bChunkBenchmarkIdentityError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError(str(exc)) from exc

    return {"e27": document, "repository_commit": repository_commit}


if __name__ == "__main__":
    import sys

    try:
        result = verify_execution_readiness()
    except DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError as exc:
        print(f"E27 EXECUTION READINESS FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print("E27 EXECUTION READINESS PASS -- no benchmark execution has occurred.")
    print(f"repository_commit={result['repository_commit']}")
    raise SystemExit(0)
