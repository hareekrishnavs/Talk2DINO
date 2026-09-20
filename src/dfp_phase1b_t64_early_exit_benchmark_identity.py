"""Strict loading/validation for the DFP Phase 1B T64-early-exit
benchmark identities:
`evaluation_identities/e25_dfp_phase1b_t64_early_exit_benchmark.toml`
("E25", the scientific/operational contract) and
`evaluation_identities/e25_dfp_phase1b_t64_early_exit_benchmark_artifacts.toml`
("E25 artifacts", the closed trial/merged/decision artifact schemas).

E25 freezes the contract for a future T64_early_exit-versus-
T320_reference wall-clock benchmark on the already-frozen E17 24-window
population. It implements no benchmark runner, no CLI execute path, no
sbatch launcher, no model/dataset provider, and no trial-merging code
beyond the pure closed-schema/relational validation helpers in this
module. It runs no inference, submits no GPU job, accesses no dataset
image or GT annotation, loads no model weight, and initializes no
CUDA.

This module implements no method, runs no inference, and performs no
torch/CUDA/model/dataset access -- pure TOML/filesystem/hash validation
and exact comparison only.

`KeyboardInterrupt`, `SystemExit`, and `MemoryError` always propagate
unmodified; ordinary parse/type/path/relational failures are wrapped in
`DfpPhase1bT64EarlyExitBenchmarkIdentityError`.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping


class DfpPhase1bT64EarlyExitBenchmarkIdentityError(ValueError):
    """Raised on any E25 (or E25 artifacts) identity invariant
    violation. Always fail closed. Never raised for, and never
    catches, KeyboardInterrupt / SystemExit / MemoryError -- those
    always propagate unmodified."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e25_dfp_phase1b_t64_early_exit_benchmark.toml")
ARTIFACTS_IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e25_dfp_phase1b_t64_early_exit_benchmark_artifacts.toml")

SCHEMA_NAME = "talk2dino-dfp-phase1b-t64-early-exit-benchmark-identity-v1"
SCHEMA_VERSION = 1
ARTIFACTS_SCHEMA_NAME = "talk2dino-dfp-phase1b-t64-early-exit-benchmark-artifacts-identity-v1"
ARTIFACTS_SCHEMA_VERSION = 1

REQUIRED_ANCESTOR_COMMIT = "453b2d2dec90dcf0623ac94eb177aa5be64db40e"
EXPECTED_WORKING_BRANCH = "dfp-phase1a-diagnostics"
EXPECTED_SUCCESSOR_OF = "e24-dfp-phase1b-balp-development-closure"
EXPECTED_ARTIFACTS_SUCCESSOR_OF = "e25-dfp-phase1b-t64-early-exit-benchmark"

VARIANTS = ("T64_early_exit", "T320_reference")
T64_STEPS = 64
T320_STEPS = 320
MEASURED_WINDOW_COUNT = 24
WARMUP_RECORD_COUNT = 1
MANIFEST_SHA256 = "be2c77466a2c1d64f0f39f17f4d073ce1de92dc526a59e47ae8c3dabb8c644ab"
CANONICAL_MANIFEST_DIGEST = "4b962001ccab863b4c7a156cdbde5f55b66b3ff26a9d429a8b6f148550551847"
BENCHMARK_RECORD_DIGEST = "8193ef03f5d1dee843098ded6ce896f0f5fe072d49f37d2657ad9b6507fec142"

WARMUP_TRIALS = {"T64_early_exit": 1, "T320_reference": 1}
MEASURED_REPEATS = {"T64_early_exit": 3, "T320_reference": 3}
TOTAL_MEASURED_TRIALS = 6
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
DEVELOPMENT_ACCURACY_DISCLOSURE_PP = -0.074828045106592
T64_DEVELOPMENT_MIOU = 30.859592058320246
T320_DEVELOPMENT_MIOU = 30.934420103426838

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------
# Small closed-schema primitives -- the same convention every other
# E-identity loader in this repository uses.
# ---------------------------------------------------------------------


def _require_string(value: Any, label: str) -> str:
    if type(value) is not str:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"{label} must be a string, observed {type(value).__name__}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    s = _require_string(value, label)
    if not _HEX64.match(s):
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"{label} must be lowercase 64-hex, observed {value!r}")
    return s


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"{label} must be a mapping, observed {type(value).__name__}")
    return value


def _require_closed(node: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    node = _require_mapping(node, label)
    actual = frozenset(node.keys())
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
    if missing or unknown:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"{label} has the wrong key set -- missing={missing} unknown={unknown}")
    return node


def _require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"{label} must be an int (not bool/float), observed {type(value).__name__}: {value!r}")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"{label} must be a bool, observed {type(value).__name__}: {value!r}")
    return value


def _require_exact_value(observed: Any, expected: Any, label: str) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(
            f"{label}: frozen value drift -- expected {expected!r} ({type(expected).__name__}), observed {observed!r} ({type(observed).__name__})"
        )


def _resolve_repo_relative(root: Path, relative_path: str, *, label: str) -> Path:
    candidate = (root / relative_path).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"{label}: relative_path escapes the repository root: {relative_path!r}") from exc
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
    _require_exact_value(_require_string(node["name"], "identity.name"), "e25-dfp-phase1b-t64-early-exit-benchmark", "identity.name")
    _require_exact_value(_require_string(node["schema_name"], "identity.schema_name"), SCHEMA_NAME, "identity.schema_name")
    _require_exact_value(_require_exact_int(node["schema_version"], "identity.schema_version"), SCHEMA_VERSION, "identity.schema_version")
    _require_string(node["description"], "identity.description")
    commit = _require_string(node["required_ancestor_commit"], "identity.required_ancestor_commit")
    if not _HEX40.match(commit):
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"identity.required_ancestor_commit must be 40-hex, observed {commit!r}")
    _require_exact_value(commit, REQUIRED_ANCESTOR_COMMIT, "identity.required_ancestor_commit")
    _require_exact_value(_require_string(node["working_branch"], "identity.working_branch"), EXPECTED_WORKING_BRANCH, "identity.working_branch")
    _require_exact_value(_require_string(node["successor_of"], "identity.successor_of"), EXPECTED_SUCCESSOR_OF, "identity.successor_of")
    _require_exact_value(_require_bool(node["mutates_or_version_bumps_earlier_identity"], "identity.mutates_or_version_bumps_earlier_identity"), False, "identity.mutates_or_version_bumps_earlier_identity")


# ---------------------------------------------------------------------
# [scientific_motivation]
# ---------------------------------------------------------------------

_SCIENTIFIC_MOTIVATION_KEYS = frozenset({
    "source", "t64_development_miou", "t320_development_miou", "t64_minus_t320_development_miou_pp",
    "t64_recurrence_steps", "t320_recurrence_steps", "recurrence_step_reduction_fraction", "balp_status",
    "t64_label", "t64_is_preregistered_method", "t64_is_new_accuracy_winner",
    "t64_validated_on_confirmation_split", "t64_validated_on_full_validation_set",
    "this_task_is_balp_candidate_continuation",
})


def _verify_scientific_motivation(node: Any) -> None:
    node = _require_closed(node, _SCIENTIFIC_MOTIVATION_KEYS, "scientific_motivation")
    _require_exact_value(node["t64_development_miou"], T64_DEVELOPMENT_MIOU, "scientific_motivation.t64_development_miou")
    _require_exact_value(node["t320_development_miou"], T320_DEVELOPMENT_MIOU, "scientific_motivation.t320_development_miou")
    _require_exact_value(node["t64_minus_t320_development_miou_pp"], DEVELOPMENT_ACCURACY_DISCLOSURE_PP, "scientific_motivation.t64_minus_t320_development_miou_pp")
    _require_exact_value(_require_exact_int(node["t64_recurrence_steps"], "..t64_recurrence_steps"), T64_STEPS, "scientific_motivation.t64_recurrence_steps")
    _require_exact_value(_require_exact_int(node["t320_recurrence_steps"], "..t320_recurrence_steps"), T320_STEPS, "scientific_motivation.t320_recurrence_steps")
    _require_exact_value(_require_string(node["t64_label"], "..t64_label"), "post_development_selected_operational_candidate", "scientific_motivation.t64_label")
    for key in ("t64_is_preregistered_method", "t64_is_new_accuracy_winner", "t64_validated_on_confirmation_split", "t64_validated_on_full_validation_set", "this_task_is_balp_candidate_continuation"):
        _require_exact_value(_require_bool(node[key], f"..{key}"), False, f"scientific_motivation.{key}")


# ---------------------------------------------------------------------
# [dependency_bindings]: closed key set, hash re-verification for every
# ordinary binding. The two symbolic (non-repository-relative) bindings
# (e17_manifest is repository-relative; e24_closure_v2 is symbolic) are
# handled specially.
# ---------------------------------------------------------------------

_ORDINARY_BINDING_KEYS = frozenset({"relative_path", "sha256", "binds"})
_E17_MANIFEST_BINDING_KEYS = frozenset({"relative_path", "sha256", "window_count", "canonical_manifest_digest", "benchmark_record_digest", "binds"})
_E24_CLOSURE_V2_BINDING_KEYS = frozenset({"relative_path_is_symbolic_not_repository_relative", "relative_path_note", "closure_json_sha256", "closure_md_sha256", "binds"})

_ORDINARY_BINDING_NAMES = frozenset({
    "e13_protocol", "matched_evaluator", "subset_manifest", "e17_identity", "e17_identity_loader_module",
    "e22_preservation", "e22_artifacts", "e23", "e24", "graph_module", "finite_step_regime_module",
    "recurrence_capture_module", "horizon_stitching_module", "stitching_pullback_module", "final_rescale_module",
    "identity_module", "production_provider_module", "artifact_schema_module",
})
_ALL_BINDING_NAMES = _ORDINARY_BINDING_NAMES | {"e17_manifest", "e24_closure_v2"}


def _recompute_e17_manifest_digests(manifest_bytes: bytes, *, label: str) -> tuple[str, str]:
    """Independently recomputes ``canonical_manifest_digest`` and
    ``benchmark_record_digest`` directly from the manifest's own
    canonical content (``benchmark_records`` / ``warmup_record`` /
    ``positions``), using the already-committed E17 canonical-encoding
    primitives (``src.dfp_phase1b_chunk_benchmark.canonical_encode`` /
    ``digest_over_records``) -- never reimplemented, and never trusting
    any digest string stored anywhere (including inside the manifest
    itself) as an input to this recomputation. Returns
    ``(recomputed_canonical_manifest_digest, recomputed_benchmark_record_digest)``."""
    import src.dfp_phase1b_chunk_benchmark as cb

    try:
        document = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"{label}: manifest is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"{label}: manifest must decode to a JSON object")
    missing = [k for k in ("benchmark_records", "warmup_record", "positions") if k not in document]
    if missing:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"{label}: manifest is missing required content field(s) for digest recomputation: {missing}")
    records = document["benchmark_records"]
    if not isinstance(records, list):
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"{label}: manifest 'benchmark_records' must be a list")
    try:
        encoded = [cb.canonical_encode(r) for r in records]
        recomputed_record_digest = cb.digest_over_records(encoded)
        recomputed_manifest_digest = cb.digest_over_records(
            encoded + [cb.canonical_encode(document["warmup_record"])] + [cb.canonical_encode({"positions": document["positions"]})]
        )
    except cb.DfpPhase1bChunkBenchmarkError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"{label}: cannot canonically encode manifest content for digest recomputation: {exc}") from exc

    if document.get("canonical_manifest_digest") != recomputed_manifest_digest:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(
            f"{label}: manifest's own internal canonical_manifest_digest disagrees with the independently recomputed value -- "
            f"expected {recomputed_manifest_digest!r}, observed {document.get('canonical_manifest_digest')!r}"
        )
    if document.get("benchmark_record_digest") != recomputed_record_digest:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(
            f"{label}: manifest's own internal benchmark_record_digest disagrees with the independently recomputed value -- "
            f"expected {recomputed_record_digest!r}, observed {document.get('benchmark_record_digest')!r}"
        )
    if recomputed_manifest_digest != CANONICAL_MANIFEST_DIGEST:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(
            f"{label}: recomputed canonical_manifest_digest disagrees with E25's frozen pinned value -- "
            f"expected {CANONICAL_MANIFEST_DIGEST!r}, observed {recomputed_manifest_digest!r} (manifest content drift)"
        )
    if recomputed_record_digest != BENCHMARK_RECORD_DIGEST:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(
            f"{label}: recomputed benchmark_record_digest disagrees with E25's frozen pinned value -- "
            f"expected {BENCHMARK_RECORD_DIGEST!r}, observed {recomputed_record_digest!r} (manifest content drift)"
        )
    return recomputed_manifest_digest, recomputed_record_digest


def _verify_dependency_bindings(root: Path, node: Any) -> tuple[str, str]:
    node = _require_closed(node, _ALL_BINDING_NAMES, "dependency_bindings")
    for name in _ORDINARY_BINDING_NAMES:
        binding = _require_closed(node[name], _ORDINARY_BINDING_KEYS, f"dependency_bindings.{name}")
        rel = _require_string(binding["relative_path"], f"dependency_bindings.{name}.relative_path")
        expected_hash = _require_sha256(binding["sha256"], f"dependency_bindings.{name}.sha256")
        _require_string(binding["binds"], f"dependency_bindings.{name}.binds")
        target = _resolve_repo_relative(root, rel, label=f"dependency_bindings.{name}")
        try:
            observed_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as exc:
            raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"dependency_bindings.{name}: cannot read bound file {rel}: {exc}") from exc
        if observed_hash != expected_hash:
            raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"dependency_bindings.{name}: bound file {rel} sha256 drift -- expected {expected_hash}, observed {observed_hash}")

    wu = _require_closed(node["e17_manifest"], _E17_MANIFEST_BINDING_KEYS, "dependency_bindings.e17_manifest")
    rel = _require_string(wu["relative_path"], "dependency_bindings.e17_manifest.relative_path")
    expected_hash = _require_sha256(wu["sha256"], "dependency_bindings.e17_manifest.sha256")
    target = _resolve_repo_relative(root, rel, label="dependency_bindings.e17_manifest")
    try:
        manifest_bytes = target.read_bytes()
    except OSError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"dependency_bindings.e17_manifest: cannot read {rel}: {exc}") from exc
    observed_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if observed_hash != expected_hash:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("dependency_bindings.e17_manifest: bound file sha256 drift")
    _require_exact_value(_require_exact_int(wu["window_count"], "..window_count"), MEASURED_WINDOW_COUNT, "dependency_bindings.e17_manifest.window_count")
    dep_canonical_digest = _require_sha256(wu["canonical_manifest_digest"], "dependency_bindings.e17_manifest.canonical_manifest_digest")
    dep_record_digest = _require_sha256(wu["benchmark_record_digest"], "dependency_bindings.e17_manifest.benchmark_record_digest")
    _require_string(wu["binds"], "dependency_bindings.e17_manifest.binds")

    # Relational digest verification (the narrow repair): the manifest
    # file's SHA-256 byte-hash check above proves the FILE wasn't
    # substituted -- it never proves anything about the two CONTENT
    # digests. Those are independently recomputed here from the real
    # manifest's own canonical records/warmup/positions (never trusting
    # any digest string stored anywhere as an input), then every
    # storage location's copy is checked against that single
    # recomputed pair, which is itself pinned against E25's frozen
    # constants (catching a manifest whose content and internal digest
    # fields were coherently, but incorrectly, tampered together).
    recomputed_manifest_digest, recomputed_record_digest = _recompute_e17_manifest_digests(manifest_bytes, label="dependency_bindings.e17_manifest")
    if dep_canonical_digest != recomputed_manifest_digest:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(
            f"dependency_bindings.e17_manifest.canonical_manifest_digest disagrees with the independently recomputed manifest digest -- "
            f"expected {recomputed_manifest_digest!r}, observed {dep_canonical_digest!r}"
        )
    if dep_record_digest != recomputed_record_digest:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(
            f"dependency_bindings.e17_manifest.benchmark_record_digest disagrees with the independently recomputed record digest -- "
            f"expected {recomputed_record_digest!r}, observed {dep_record_digest!r}"
        )

    cv2 = _require_closed(node["e24_closure_v2"], _E24_CLOSURE_V2_BINDING_KEYS, "dependency_bindings.e24_closure_v2")
    _require_exact_value(_require_bool(cv2["relative_path_is_symbolic_not_repository_relative"], "..relative_path_is_symbolic_not_repository_relative"), True, "dependency_bindings.e24_closure_v2.relative_path_is_symbolic_not_repository_relative")
    _require_string(cv2["relative_path_note"], "dependency_bindings.e24_closure_v2.relative_path_note")
    _require_sha256(cv2["closure_json_sha256"], "dependency_bindings.e24_closure_v2.closure_json_sha256")
    _require_sha256(cv2["closure_md_sha256"], "dependency_bindings.e24_closure_v2.closure_md_sha256")
    _require_string(cv2["binds"], "dependency_bindings.e24_closure_v2.binds")

    return recomputed_manifest_digest, recomputed_record_digest


# ---------------------------------------------------------------------
# [recurrence_execution_contract]
# ---------------------------------------------------------------------

_RECURRENCE_EXECUTION_CONTRACT_KEYS = frozenset({
    "mechanism", "t64_call_steps", "t320_call_steps", "t64_snapshot_steps", "t320_snapshot_steps",
    "loop_structure_note", "capture_dfp_phase1a_trajectory_usable_for_t64", "capture_dfp_phase1a_trajectory_usable_for_t320",
    "snapshot_of_a_320_step_run_forbidden_for_t64", "graph_reused_across_variants", "graph_constructed_fresh_per_window_per_trial",
    "unary_model_forward_count_per_window_per_variant", "graph_construction_count_per_window_per_variant",
    "candidate_specific_recurrence_authorized", "exact_masked_forward_provenance_authorized", "component_construction_authorized",
})


def _verify_recurrence_execution_contract(node: Any) -> None:
    node = _require_closed(node, _RECURRENCE_EXECUTION_CONTRACT_KEYS, "recurrence_execution_contract")
    _require_exact_value(_require_exact_int(node["t64_call_steps"], "..t64_call_steps"), T64_STEPS, "recurrence_execution_contract.t64_call_steps")
    _require_exact_value(_require_exact_int(node["t320_call_steps"], "..t320_call_steps"), T320_STEPS, "recurrence_execution_contract.t320_call_steps")
    _require_exact_value(list(node["t64_snapshot_steps"]), [T64_STEPS], "recurrence_execution_contract.t64_snapshot_steps")
    _require_exact_value(list(node["t320_snapshot_steps"]), [T320_STEPS], "recurrence_execution_contract.t320_snapshot_steps")
    _require_exact_value(_require_bool(node["capture_dfp_phase1a_trajectory_usable_for_t64"], "..usable_for_t64"), False, "recurrence_execution_contract.capture_dfp_phase1a_trajectory_usable_for_t64")
    _require_exact_value(_require_bool(node["capture_dfp_phase1a_trajectory_usable_for_t320"], "..usable_for_t320"), True, "recurrence_execution_contract.capture_dfp_phase1a_trajectory_usable_for_t320")
    _require_exact_value(_require_bool(node["snapshot_of_a_320_step_run_forbidden_for_t64"], "..forbidden"), True, "recurrence_execution_contract.snapshot_of_a_320_step_run_forbidden_for_t64")
    _require_exact_value(_require_exact_int(node["unary_model_forward_count_per_window_per_variant"], "..forward_count"), 1, "recurrence_execution_contract.unary_model_forward_count_per_window_per_variant")
    _require_exact_value(_require_exact_int(node["graph_construction_count_per_window_per_variant"], "..graph_count"), 1, "recurrence_execution_contract.graph_construction_count_per_window_per_variant")
    for key in ("candidate_specific_recurrence_authorized", "exact_masked_forward_provenance_authorized", "component_construction_authorized"):
        _require_exact_value(_require_bool(node[key], f"..{key}"), False, f"recurrence_execution_contract.{key}")
    _require_string(node["mechanism"], "recurrence_execution_contract.mechanism")
    _require_string(node["loop_structure_note"], "recurrence_execution_contract.loop_structure_note")
    _require_bool(node["graph_reused_across_variants"], "recurrence_execution_contract.graph_reused_across_variants")
    _require_bool(node["graph_constructed_fresh_per_window_per_trial"], "recurrence_execution_contract.graph_constructed_fresh_per_window_per_trial")


# ---------------------------------------------------------------------
# [benchmark_design]
# ---------------------------------------------------------------------

_BENCHMARK_DESIGN_KEYS = frozenset({"variants", "variant_count", "t64_early_exit", "t320_reference", "held_identical_between_variants", "per_window_call_budget"})
_T64_VARIANT_KEYS = frozenset({"recurrence_start", "graph", "k", "alpha", "affinity_function", "affinity_power", "maximum_recurrence_step", "output_step", "forbidden_steps"})
_T320_VARIANT_KEYS = frozenset({"recurrence_start", "graph", "k", "alpha", "affinity_function", "affinity_power", "maximum_recurrence_step", "output_step"})
_HELD_IDENTICAL_KEYS = frozenset({"list"})
_PER_WINDOW_CALL_BUDGET_KEYS = frozenset({"unary_model_forward_calls", "graph_construction_calls", "recurrence_passes", "candidate_specific_recurrence_calls"})

_REQUIRED_HELD_IDENTICAL = (
    "selected_windows_and_order", "dataset_records", "image_preprocessing", "unary_model", "checkpoint",
    "text_embeddings", "graph_construction", "graph_instance_semantics", "numerical_precision",
    "stitching_and_rescaling", "device", "software_environment", "model_mode", "deterministic_settings",
    "output_validation",
)


def _verify_benchmark_design(node: Any) -> None:
    node = _require_closed(node, _BENCHMARK_DESIGN_KEYS, "benchmark_design")
    _require_exact_value(list(node["variants"]), list(VARIANTS), "benchmark_design.variants")
    _require_exact_value(_require_exact_int(node["variant_count"], "..variant_count"), 2, "benchmark_design.variant_count")

    t64 = _require_closed(node["t64_early_exit"], _T64_VARIANT_KEYS, "benchmark_design.t64_early_exit")
    _require_exact_value(_require_exact_int(t64["k"], "..k"), 12, "benchmark_design.t64_early_exit.k")
    _require_exact_value(t64["alpha"], 0.98, "benchmark_design.t64_early_exit.alpha")
    _require_exact_value(t64["affinity_power"], 3.0, "benchmark_design.t64_early_exit.affinity_power")
    _require_exact_value(_require_exact_int(t64["maximum_recurrence_step"], "..max_step"), T64_STEPS, "benchmark_design.t64_early_exit.maximum_recurrence_step")
    _require_exact_value(_require_exact_int(t64["output_step"], "..output_step"), T64_STEPS, "benchmark_design.t64_early_exit.output_step")

    t320 = _require_closed(node["t320_reference"], _T320_VARIANT_KEYS, "benchmark_design.t320_reference")
    _require_exact_value(_require_exact_int(t320["k"], "..k"), 12, "benchmark_design.t320_reference.k")
    _require_exact_value(t320["alpha"], 0.98, "benchmark_design.t320_reference.alpha")
    _require_exact_value(t320["affinity_power"], 3.0, "benchmark_design.t320_reference.affinity_power")
    _require_exact_value(_require_exact_int(t320["maximum_recurrence_step"], "..max_step"), T320_STEPS, "benchmark_design.t320_reference.maximum_recurrence_step")
    _require_exact_value(_require_exact_int(t320["output_step"], "..output_step"), T320_STEPS, "benchmark_design.t320_reference.output_step")

    held = _require_closed(node["held_identical_between_variants"], _HELD_IDENTICAL_KEYS, "benchmark_design.held_identical_between_variants")
    held_list = held["list"]
    if not isinstance(held_list, list) or tuple(held_list) != _REQUIRED_HELD_IDENTICAL:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("benchmark_design.held_identical_between_variants.list disagrees with the frozen required list")

    budget = _require_closed(node["per_window_call_budget"], _PER_WINDOW_CALL_BUDGET_KEYS, "benchmark_design.per_window_call_budget")
    _require_exact_value(_require_exact_int(budget["unary_model_forward_calls"], "..calls"), 1, "benchmark_design.per_window_call_budget.unary_model_forward_calls")
    _require_exact_value(_require_exact_int(budget["graph_construction_calls"], "..calls"), 1, "benchmark_design.per_window_call_budget.graph_construction_calls")
    _require_exact_value(_require_exact_int(budget["recurrence_passes"], "..passes"), 1, "benchmark_design.per_window_call_budget.recurrence_passes")
    _require_exact_value(_require_exact_int(budget["candidate_specific_recurrence_calls"], "..calls"), 0, "benchmark_design.per_window_call_budget.candidate_specific_recurrence_calls")


# ---------------------------------------------------------------------
# [population_binding]
# ---------------------------------------------------------------------

_POPULATION_BINDING_KEYS = frozenset({
    "manifest_relative_path", "manifest_sha256", "measured_window_count", "warmup_record_count",
    "canonical_manifest_digest", "benchmark_record_digest", "deterministic_order", "warmup_record_distinction",
    "gt_annotation_field_present_in_manifest", "population_fixed_before_execution",
    "same_population_required_across_all_variants_and_trials",
})


def _verify_population_binding(node: Any, *, recomputed_manifest_digest: str, recomputed_record_digest: str) -> None:
    node = _require_closed(node, _POPULATION_BINDING_KEYS, "population_binding")
    _require_exact_value(_require_string(node["manifest_relative_path"], "..path"), "evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json", "population_binding.manifest_relative_path")
    _require_exact_value(_require_sha256(node["manifest_sha256"], "..sha256"), MANIFEST_SHA256, "population_binding.manifest_sha256")
    _require_exact_value(_require_exact_int(node["measured_window_count"], "..count"), MEASURED_WINDOW_COUNT, "population_binding.measured_window_count")
    _require_exact_value(_require_exact_int(node["warmup_record_count"], "..count"), WARMUP_RECORD_COUNT, "population_binding.warmup_record_count")
    # Checked against BOTH the independently recomputed digest pair
    # (established once, from the real manifest content, in
    # `_verify_dependency_bindings`) AND E25's own frozen pinned
    # constants -- the two are already required to agree with each
    # other there, so this is belt-and-suspenders, not redundant: it
    # is what makes population_binding's own copy of these fields
    # unable to silently disagree with the dependency_bindings copy.
    population_canonical_digest = _require_sha256(node["canonical_manifest_digest"], "..digest")
    population_record_digest = _require_sha256(node["benchmark_record_digest"], "..digest")
    _require_exact_value(population_canonical_digest, CANONICAL_MANIFEST_DIGEST, "population_binding.canonical_manifest_digest")
    _require_exact_value(population_record_digest, BENCHMARK_RECORD_DIGEST, "population_binding.benchmark_record_digest")
    if population_canonical_digest != recomputed_manifest_digest:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(
            f"population_binding.canonical_manifest_digest disagrees with the independently recomputed manifest digest -- "
            f"expected {recomputed_manifest_digest!r}, observed {population_canonical_digest!r}"
        )
    if population_record_digest != recomputed_record_digest:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(
            f"population_binding.benchmark_record_digest disagrees with the independently recomputed record digest -- "
            f"expected {recomputed_record_digest!r}, observed {population_record_digest!r}"
        )
    _require_exact_value(_require_bool(node["gt_annotation_field_present_in_manifest"], "..gt_present"), False, "population_binding.gt_annotation_field_present_in_manifest")
    _require_exact_value(_require_bool(node["population_fixed_before_execution"], "..fixed"), True, "population_binding.population_fixed_before_execution")
    _require_exact_value(_require_bool(node["same_population_required_across_all_variants_and_trials"], "..same"), True, "population_binding.same_population_required_across_all_variants_and_trials")
    _require_string(node["deterministic_order"], "population_binding.deterministic_order")
    _require_string(node["warmup_record_distinction"], "population_binding.warmup_record_distinction")


# ---------------------------------------------------------------------
# [execution_schedule]
# ---------------------------------------------------------------------

_EXECUTION_SCHEDULE_KEYS = frozenset({
    "required_device_type", "required_gpu_model", "gpu_count", "cpus_per_task", "host_memory", "max_wall_time",
    "fresh_process_per_trial_required", "same_node_required", "same_gpu_model_required", "same_requested_resources_required",
    "same_software_environment_required", "same_committed_head_required", "repository_worktree_clean_required",
    "warmup", "measured_repeats", "counterbalanced_order",
})
_WARMUP_KEYS = frozenset({"t64_early_exit", "t320_reference", "runs_before_measured_matrix", "excluded_from_summary_statistics", "schema_valid_required"})
_MEASURED_REPEATS_KEYS = frozenset({"t64_early_exit", "t320_reference", "total_measured_trials"})
_COUNTERBALANCED_ORDER_KEYS = frozenset({"description", "ordered_trial_list"})
_ORDERED_TRIAL_ENTRY_KEYS = frozenset({"trial_index", "variant", "repeat"})


def _verify_execution_schedule(node: Any) -> None:
    node = _require_closed(node, _EXECUTION_SCHEDULE_KEYS, "execution_schedule")
    _require_exact_value(_require_string(node["required_device_type"], "..type"), "cuda", "execution_schedule.required_device_type")
    _require_exact_value(_require_string(node["required_gpu_model"], "..model"), "h100_80gb", "execution_schedule.required_gpu_model")
    _require_exact_value(_require_exact_int(node["gpu_count"], "..count"), 1, "execution_schedule.gpu_count")
    _require_exact_value(_require_exact_int(node["cpus_per_task"], "..cpus"), 8, "execution_schedule.cpus_per_task")
    _require_exact_value(_require_string(node["host_memory"], "..mem"), "80G", "execution_schedule.host_memory")
    _require_exact_value(_require_string(node["max_wall_time"], "..wall"), "02:00:00", "execution_schedule.max_wall_time")
    for key in (
        "fresh_process_per_trial_required", "same_node_required", "same_gpu_model_required",
        "same_requested_resources_required", "same_software_environment_required",
        "same_committed_head_required", "repository_worktree_clean_required",
    ):
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"execution_schedule.{key}")

    warmup = _require_closed(node["warmup"], _WARMUP_KEYS, "execution_schedule.warmup")
    _require_exact_value(_require_exact_int(warmup["t64_early_exit"], "..t64"), WARMUP_TRIALS["T64_early_exit"], "execution_schedule.warmup.t64_early_exit")
    _require_exact_value(_require_exact_int(warmup["t320_reference"], "..t320"), WARMUP_TRIALS["T320_reference"], "execution_schedule.warmup.t320_reference")
    _require_exact_value(_require_bool(warmup["runs_before_measured_matrix"], "..before"), True, "execution_schedule.warmup.runs_before_measured_matrix")
    _require_exact_value(_require_bool(warmup["excluded_from_summary_statistics"], "..excluded"), True, "execution_schedule.warmup.excluded_from_summary_statistics")
    _require_exact_value(_require_bool(warmup["schema_valid_required"], "..valid"), True, "execution_schedule.warmup.schema_valid_required")

    repeats = _require_closed(node["measured_repeats"], _MEASURED_REPEATS_KEYS, "execution_schedule.measured_repeats")
    _require_exact_value(_require_exact_int(repeats["t64_early_exit"], "..t64"), MEASURED_REPEATS["T64_early_exit"], "execution_schedule.measured_repeats.t64_early_exit")
    _require_exact_value(_require_exact_int(repeats["t320_reference"], "..t320"), MEASURED_REPEATS["T320_reference"], "execution_schedule.measured_repeats.t320_reference")
    _require_exact_value(_require_exact_int(repeats["total_measured_trials"], "..total"), TOTAL_MEASURED_TRIALS, "execution_schedule.measured_repeats.total_measured_trials")

    order = _require_closed(node["counterbalanced_order"], _COUNTERBALANCED_ORDER_KEYS, "execution_schedule.counterbalanced_order")
    _require_string(order["description"], "execution_schedule.counterbalanced_order.description")
    ordered = order["ordered_trial_list"]
    if not isinstance(ordered, list) or len(ordered) != TOTAL_MEASURED_TRIALS:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"execution_schedule.counterbalanced_order.ordered_trial_list must have exactly {TOTAL_MEASURED_TRIALS} entries")
    for position, (entry, expected) in enumerate(zip(ordered, COUNTERBALANCED_ORDER)):
        entry = _require_closed(entry, _ORDERED_TRIAL_ENTRY_KEYS, f"execution_schedule.counterbalanced_order.ordered_trial_list[{position}]")
        if (
            _require_exact_int(entry["trial_index"], f"..[{position}].trial_index") != expected["trial_index"]
            or _require_string(entry["variant"], f"..[{position}].variant") != expected["variant"]
            or _require_exact_int(entry["repeat"], f"..[{position}].repeat") != expected["repeat"]
        ):
            raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"execution_schedule.counterbalanced_order.ordered_trial_list[{position}] disagrees with the frozen required schedule")


# ---------------------------------------------------------------------
# [timing_definitions], [stage_profiling_contract], [memory_telemetry_contract]
# ---------------------------------------------------------------------

_TIMING_PRIMARY_KEYS = frozenset({"name", "clock_source", "unit", "definition_steps"})
_TIMING_DERIVED_KEYS = frozenset({"measured_window_loop_seconds", "window_throughput_per_second", "mean_seconds_per_window"})
_TIMING_PROCESS_PHASE_KEYS = frozenset({"list", "accounting_identity", "all_values_non_negative_integers", "unknown_timing_silently_recorded_as_zero_forbidden"})
_TIMING_DEFINITIONS_KEYS = frozenset({"primary", "derived", "process_phase_fields"})

_REQUIRED_PROCESS_PHASE_FIELDS = (
    "process_total_wall_ns", "preflight_wall_ns", "dataset_construction_wall_ns",
    "model_construction_and_checkpoint_loading_wall_ns", "measured_window_loop_wall_ns",
    "artifact_validation_and_write_wall_ns", "unattributed_overhead_ns",
)


def _verify_timing_definitions(node: Any) -> None:
    node = _require_closed(node, _TIMING_DEFINITIONS_KEYS, "timing_definitions")
    primary = _require_closed(node["primary"], _TIMING_PRIMARY_KEYS, "timing_definitions.primary")
    _require_exact_value(_require_string(primary["name"], "..name"), "measured_window_loop_wall_ns", "timing_definitions.primary.name")
    _require_exact_value(_require_string(primary["clock_source"], "..clock"), "time.perf_counter_ns", "timing_definitions.primary.clock_source")
    if not isinstance(primary["definition_steps"], list) or len(primary["definition_steps"]) != 6:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("timing_definitions.primary.definition_steps must have exactly 6 entries")

    derived = _require_closed(node["derived"], _TIMING_DERIVED_KEYS, "timing_definitions.derived")
    for key in _TIMING_DERIVED_KEYS:
        _require_string(derived[key], f"timing_definitions.derived.{key}")

    phases = _require_closed(node["process_phase_fields"], _TIMING_PROCESS_PHASE_KEYS, "timing_definitions.process_phase_fields")
    if tuple(phases["list"]) != _REQUIRED_PROCESS_PHASE_FIELDS:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("timing_definitions.process_phase_fields.list disagrees with the frozen required list")
    _require_exact_value(_require_bool(phases["all_values_non_negative_integers"], "..nonneg"), True, "timing_definitions.process_phase_fields.all_values_non_negative_integers")
    _require_exact_value(_require_bool(phases["unknown_timing_silently_recorded_as_zero_forbidden"], "..forbidden"), True, "timing_definitions.process_phase_fields.unknown_timing_silently_recorded_as_zero_forbidden")


_STAGE_PROFILING_KEYS = frozenset({
    "stages", "primary_runtime_decision_field", "sum_of_stage_timings_forbidden_as_primary",
    "gpu_event_boundaries_identical_across_variants_required", "per_small_operation_synchronize_forbidden",
    "single_outer_synchronize_boundary_required", "profiling_must_not_alter_scientific_output",
    "count_fields", "required_relations",
})
_REQUIRED_STAGES = ("input_preparation", "unary_model_forward", "graph_construction", "canonical_recurrence", "stitching_and_rescale", "output_validation")
_REQUIRED_COUNT_FIELDS = ("recurrence_steps_executed", "model_forward_count", "graph_construction_count", "window_count")
_REQUIRED_RELATIONS_KEYS = frozenset({"t64_recurrence_steps_executed", "t320_recurrence_steps_executed", "model_forward_count", "graph_construction_count"})


def _verify_stage_profiling_contract(node: Any) -> None:
    node = _require_closed(node, _STAGE_PROFILING_KEYS, "stage_profiling_contract")
    if tuple(node["stages"]) != _REQUIRED_STAGES:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("stage_profiling_contract.stages disagrees with the frozen required list")
    _require_exact_value(_require_string(node["primary_runtime_decision_field"], "..field"), "measured_window_loop_wall_ns", "stage_profiling_contract.primary_runtime_decision_field")
    for key in (
        "sum_of_stage_timings_forbidden_as_primary", "gpu_event_boundaries_identical_across_variants_required",
        "per_small_operation_synchronize_forbidden", "single_outer_synchronize_boundary_required",
        "profiling_must_not_alter_scientific_output",
    ):
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"stage_profiling_contract.{key}")

    counts = _require_closed(node["count_fields"], frozenset({"list"}), "stage_profiling_contract.count_fields")
    if tuple(counts["list"]) != _REQUIRED_COUNT_FIELDS:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("stage_profiling_contract.count_fields.list disagrees with the frozen required list")

    relations = _require_closed(node["required_relations"], _REQUIRED_RELATIONS_KEYS, "stage_profiling_contract.required_relations")
    _require_exact_value(_require_string(relations["t64_recurrence_steps_executed"], "..t64"), "window_count * 64", "stage_profiling_contract.required_relations.t64_recurrence_steps_executed")
    _require_exact_value(_require_string(relations["t320_recurrence_steps_executed"], "..t320"), "window_count * 320", "stage_profiling_contract.required_relations.t320_recurrence_steps_executed")
    _require_exact_value(_require_string(relations["model_forward_count"], "..fwd"), "window_count", "stage_profiling_contract.required_relations.model_forward_count")
    _require_exact_value(_require_string(relations["graph_construction_count"], "..graph"), "window_count", "stage_profiling_contract.required_relations.graph_construction_count")


_MEMORY_TELEMETRY_KEYS = frozenset({
    "fields", "host_rss_source", "gpu_peak_reset_call", "gpu_peak_reset_timing",
    "synchronize_before_final_gpu_memory_collection_required", "all_values_non_negative_integers",
    "allocated_must_not_exceed_reserved", "reserved_must_not_exceed_total_device_memory",
    "missing_gpu_telemetry_on_cuda_trial_is_validation_failure",
})
_REQUIRED_MEMORY_FIELDS = ("host_peak_rss_bytes", "gpu_peak_memory_allocated_bytes", "gpu_peak_memory_reserved_bytes", "gpu_total_memory_bytes")


def _verify_memory_telemetry_contract(node: Any) -> None:
    node = _require_closed(node, _MEMORY_TELEMETRY_KEYS, "memory_telemetry_contract")
    if tuple(node["fields"]) != _REQUIRED_MEMORY_FIELDS:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("memory_telemetry_contract.fields disagrees with the frozen required list")
    _require_exact_value(_require_string(node["gpu_peak_reset_call"], "..reset_call"), "torch.cuda.reset_peak_memory_stats()", "memory_telemetry_contract.gpu_peak_reset_call")
    for key in (
        "synchronize_before_final_gpu_memory_collection_required", "all_values_non_negative_integers",
        "allocated_must_not_exceed_reserved", "reserved_must_not_exceed_total_device_memory",
        "missing_gpu_telemetry_on_cuda_trial_is_validation_failure",
    ):
        _require_exact_value(_require_bool(node[key], f"..{key}"), True, f"memory_telemetry_contract.{key}")
    _require_string(node["host_rss_source"], "memory_telemetry_contract.host_rss_source")
    _require_string(node["gpu_peak_reset_timing"], "memory_telemetry_contract.gpu_peak_reset_timing")


# ---------------------------------------------------------------------
# [scientific_output_contract], [summary_statistics]
# ---------------------------------------------------------------------

_SCIENTIFIC_OUTPUT_CONTRACT_KEYS = frozenset({
    "per_trial_fields", "within_variant_digest_equality_required", "within_variant_digest_equality_scope",
    "cross_variant_digest_equality_required", "warmup_output_schema_valid_required", "warmup_excluded_from_performance_summaries",
    "non_finite_value_count_must_be_zero", "performance_instrumentation_must_not_change_scientific_digest",
    "future_implementation_test_obligation",
})
_REQUIRED_SCIENTIFIC_OUTPUT_FIELDS = ("variant_id", "scientific_output_collection_digest", "per_window_output_digests", "finite_value_counts", "non_finite_value_counts")


def _verify_scientific_output_contract(node: Any) -> None:
    node = _require_closed(node, _SCIENTIFIC_OUTPUT_CONTRACT_KEYS, "scientific_output_contract")
    if tuple(node["per_trial_fields"]) != _REQUIRED_SCIENTIFIC_OUTPUT_FIELDS:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("scientific_output_contract.per_trial_fields disagrees with the frozen required list")
    _require_exact_value(_require_bool(node["within_variant_digest_equality_required"], "..required"), True, "scientific_output_contract.within_variant_digest_equality_required")
    _require_exact_value(_require_bool(node["cross_variant_digest_equality_required"], "..cross"), False, "scientific_output_contract.cross_variant_digest_equality_required")
    _require_exact_value(_require_bool(node["warmup_output_schema_valid_required"], "..warmup_valid"), True, "scientific_output_contract.warmup_output_schema_valid_required")
    _require_exact_value(_require_bool(node["warmup_excluded_from_performance_summaries"], "..warmup_excl"), True, "scientific_output_contract.warmup_excluded_from_performance_summaries")
    _require_exact_value(_require_bool(node["non_finite_value_count_must_be_zero"], "..nonfinite"), True, "scientific_output_contract.non_finite_value_count_must_be_zero")
    _require_exact_value(_require_bool(node["performance_instrumentation_must_not_change_scientific_digest"], "..perf"), True, "scientific_output_contract.performance_instrumentation_must_not_change_scientific_digest")
    _require_string(node["within_variant_digest_equality_scope"], "scientific_output_contract.within_variant_digest_equality_scope")
    obligation = _require_closed(node["future_implementation_test_obligation"], frozenset({"description"}), "scientific_output_contract.future_implementation_test_obligation")
    _require_string(obligation["description"], "scientific_output_contract.future_implementation_test_obligation.description")


_SUMMARY_STATISTICS_KEYS = frozenset({
    "per_variant_fields", "t64_speedup_over_t320_formula", "t64_recurrence_speedup_over_t320_formula",
    "recurrence_only_speedup_substitution_for_end_to_end_speedup_forbidden",
    "five_thousand_image_runtime_projection_from_24_window_benchmark_as_measured_fact_forbidden",
    "projection_rules",
})
_REQUIRED_SUMMARY_FIELDS = (
    "median_measured_window_loop_seconds", "minimum_measured_window_loop_seconds", "maximum_measured_window_loop_seconds",
    "median_window_throughput_per_second", "maximum_host_peak_rss_bytes",
    "maximum_gpu_peak_memory_allocated_bytes", "maximum_gpu_peak_memory_reserved_bytes",
)
_PROJECTION_RULES_KEYS = frozenset({
    "must_be_explicitly_labeled_unmeasured", "must_be_separated_from_benchmark_measurements", "excluded_from_proceed_decision",
    "must_include_exact_formula_and_assumptions", "historical_approximately_1h18m_baseline_usable_as_e25_performance_baseline",
    "historical_baseline_usable_as_context_only_if_log_commit_hardware_and_execution_contract_are_verified_and_bound",
})


def _verify_summary_statistics(node: Any) -> None:
    node = _require_closed(node, _SUMMARY_STATISTICS_KEYS, "summary_statistics")
    if tuple(node["per_variant_fields"]) != _REQUIRED_SUMMARY_FIELDS:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("summary_statistics.per_variant_fields disagrees with the frozen required list")
    _require_exact_value(
        _require_string(node["t64_speedup_over_t320_formula"], "..formula"),
        "median_T320_measured_window_loop_seconds / median_T64_measured_window_loop_seconds",
        "summary_statistics.t64_speedup_over_t320_formula",
    )
    _require_exact_value(_require_bool(node["recurrence_only_speedup_substitution_for_end_to_end_speedup_forbidden"], "..forbidden"), True, "summary_statistics.recurrence_only_speedup_substitution_for_end_to_end_speedup_forbidden")
    _require_exact_value(_require_bool(node["five_thousand_image_runtime_projection_from_24_window_benchmark_as_measured_fact_forbidden"], "..forbidden"), True, "summary_statistics.five_thousand_image_runtime_projection_from_24_window_benchmark_as_measured_fact_forbidden")
    _require_string(node["t64_recurrence_speedup_over_t320_formula"], "summary_statistics.t64_recurrence_speedup_over_t320_formula")

    rules = _require_closed(node["projection_rules"], _PROJECTION_RULES_KEYS, "summary_statistics.projection_rules")
    for key in ("must_be_explicitly_labeled_unmeasured", "must_be_separated_from_benchmark_measurements", "excluded_from_proceed_decision", "must_include_exact_formula_and_assumptions", "historical_baseline_usable_as_context_only_if_log_commit_hardware_and_execution_contract_are_verified_and_bound"):
        _require_exact_value(_require_bool(rules[key], f"..{key}"), True, f"summary_statistics.projection_rules.{key}")
    _require_exact_value(_require_bool(rules["historical_approximately_1h18m_baseline_usable_as_e25_performance_baseline"], "..hist"), False, "summary_statistics.projection_rules.historical_approximately_1h18m_baseline_usable_as_e25_performance_baseline")


# ---------------------------------------------------------------------
# [decision_rule], [authorization_boundary], [future_order], [prohibited]
# ---------------------------------------------------------------------

_DECISION_RULE_KEYS = frozenset({
    "outcomes", "outcome_set_closed", "adaptive_threshold_authorized", "end_to_end_speedup_threshold",
    "threshold_comparison_operator", "authorizes_confirmation_evaluation_directly", "authorizes_full_5000_image_evaluation_directly",
    "proceed_requires_all_of", "proceed_does_not_itself_authorize",
})
_REQUIRED_PROCEED_CONDITIONS = (
    "all warm-up and measured trials complete successfully", "all trial artifacts verify",
    "within-variant scientific digests are identical", "zero non-finite outputs",
    "exact model-forward, graph-construction, and recurrence-step counts", "no OOM", "all memory relations pass",
    "T64 median measured-window-loop time is lower than T320", "primary end-to-end speedup t64_speedup_over_t320 >= 1.25",
    "the frozen development accuracy disclosure remains T64 minus T320 = -0.074828045106592 percentage points",
)
_REQUIRED_PROCEED_DOES_NOT_AUTHORIZE = (
    "the_150_image_confirmation_evaluation", "candidate_selection_as_a_final_paper_method",
    "the_full_5000_image_evaluation", "a_deployment_claim", "a_latency_claim_outside_the_measured_h100_contract",
)


def _verify_decision_rule(node: Any) -> None:
    node = _require_closed(node, _DECISION_RULE_KEYS, "decision_rule")
    _require_exact_value(list(node["outcomes"]), list(DECISION_OUTCOMES), "decision_rule.outcomes")
    _require_exact_value(_require_bool(node["outcome_set_closed"], "..closed"), True, "decision_rule.outcome_set_closed")
    _require_exact_value(_require_bool(node["adaptive_threshold_authorized"], "..adaptive"), False, "decision_rule.adaptive_threshold_authorized")
    _require_exact_value(node["end_to_end_speedup_threshold"], END_TO_END_SPEEDUP_THRESHOLD, "decision_rule.end_to_end_speedup_threshold")
    _require_exact_value(_require_string(node["threshold_comparison_operator"], "..op"), "greater_than_or_equal", "decision_rule.threshold_comparison_operator")
    _require_exact_value(_require_bool(node["authorizes_confirmation_evaluation_directly"], "..conf"), False, "decision_rule.authorizes_confirmation_evaluation_directly")
    _require_exact_value(_require_bool(node["authorizes_full_5000_image_evaluation_directly"], "..full5000"), False, "decision_rule.authorizes_full_5000_image_evaluation_directly")

    proceed = _require_closed(node["proceed_requires_all_of"], frozenset({"list"}), "decision_rule.proceed_requires_all_of")
    if tuple(proceed["list"]) != _REQUIRED_PROCEED_CONDITIONS:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("decision_rule.proceed_requires_all_of.list disagrees with the frozen required list")

    does_not = _require_closed(node["proceed_does_not_itself_authorize"], frozenset({"list"}), "decision_rule.proceed_does_not_itself_authorize")
    if tuple(does_not["list"]) != _REQUIRED_PROCEED_DOES_NOT_AUTHORIZE:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("decision_rule.proceed_does_not_itself_authorize.list disagrees with the frozen required list")


_AUTHORIZATION_BOUNDARY_KEYS = frozenset({
    "real_gpu_benchmark_execution_authorized_by_e25", "runner_implementation_authorized_by_e25",
    "confirmation_evaluation_authorized", "full_5000_image_evaluation_authorized", "gt_access_authorized",
    "training_authorized", "model_weight_modification_authorized", "scientific_kernel_modification_authorized",
    "exact_masked_forward_provenance_authorized", "balp_candidate_evaluation_authorized",
})


def _verify_authorization_boundary(node: Any) -> None:
    node = _require_closed(node, _AUTHORIZATION_BOUNDARY_KEYS, "authorization_boundary")
    for key in _AUTHORIZATION_BOUNDARY_KEYS:
        _require_exact_value(_require_bool(node[key], f"..{key}"), False, f"authorization_boundary.{key}")


_REQUIRED_FUTURE_ORDER = (
    "independently_verify_e25", "commit_e25", "implement_the_benchmark_runner_and_launcher",
    "independently_verify_implementation", "perform_committed_head_production_readiness_verification",
    "user_manually_submits_the_h100_job", "independently_verify_benchmark_artifacts",
    "only_if_outcome_is_proceed_to_freeze_t64_confirmation_protocol_create_a_separate_confirmation_authority",
)


def _verify_future_order(node: Any) -> None:
    node = _require_closed(node, frozenset({"list"}), "future_order")
    if tuple(node["list"]) != _REQUIRED_FUTURE_ORDER:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("future_order.list disagrees with the frozen required order")


_REQUIRED_PROHIBITED = (
    "implementing_the_benchmark_runner_in_this_identity", "implementing_a_cli_execute_path_in_this_identity",
    "implementing_an_sbatch_launcher_in_this_identity", "implementing_model_or_dataset_providers_in_this_identity",
    "real_gpu_benchmark_execution_in_this_task", "gpu_use_in_this_task", "slurm_submission_in_this_task",
    "dataset_access_in_this_task", "weight_loading_in_this_task", "cuda_initialization_in_this_task",
    "confirmation_evaluation", "full_5000_image_evaluation", "candidate_selection_as_a_final_paper_method",
    "modifying_e13_through_e24_identities", "modifying_the_balp_scientific_kernel_or_production_runner",
    "modifying_graph_recurrence_stitching_or_pullback_mathematics", "modifying_the_completed_development_result_checkpoint_or_log",
    "modifying_the_e24_closure", "treating_a_320_step_snapshot_as_a_genuine_t64_early_exit",
    "candidate_specific_recurrence", "exact_masked_forward_provenance", "adaptive_or_relaxed_speedup_threshold",
    "projecting_a_5000_image_runtime_as_a_measured_fact",
    "using_the_historical_approximately_1h18m_baseline_as_e25s_performance_baseline_without_verified_binding",
)


def _verify_prohibited(node: Any) -> None:
    node = _require_closed(node, frozenset({"list"}), "prohibited")
    if tuple(node["list"]) != _REQUIRED_PROHIBITED:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("prohibited.list disagrees with the frozen required list")


def verify_repository_gate(repo_root: Path, repository_commit: str) -> None:
    commit = _require_string(repository_commit, "repository_commit")
    if not _HEX40.match(commit):
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"repository_commit must be a 40-hex commit, observed {commit!r}")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", REQUIRED_ANCESTOR_COMMIT, commit],
            capture_output=True, text=True,
        )
    except OSError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"cannot invoke git for ancestor verification: {exc}") from exc
    if result.returncode != 0:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"required_ancestor_commit {REQUIRED_ANCESTOR_COMMIT} is not an ancestor of (or equal to) {commit}")


# ---------------------------------------------------------------------
# Top-level load/verify: E25 (scientific/operational contract).
# ---------------------------------------------------------------------

_TOP_LEVEL_KEYS = frozenset({
    "format_version", "identity", "scientific_motivation", "dependency_bindings", "recurrence_execution_contract",
    "benchmark_design", "population_binding", "execution_schedule", "timing_definitions", "stage_profiling_contract",
    "memory_telemetry_contract", "scientific_output_contract", "summary_statistics", "decision_rule",
    "authorization_boundary", "future_order", "prohibited",
})


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load, closed-schema-validate, exact-value-validate, and
    relationally hash-verify the E25 T64-early-exit-benchmark identity.
    Never mutates or re-derives any bound dependency -- only re-
    verifies SHA-256/content against current bytes. Performs no
    torch/CUDA/model/dataset access."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with open(source, "rb") as fh:
            document = tomllib.load(fh)
    except OSError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"cannot read E25 identity {source}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"E25 identity {source} is not valid TOML: {exc}") from exc

    _require_closed(document, _TOP_LEVEL_KEYS, "<top level>")
    _require_exact_value(
        _require_string(document["format_version"], "format_version"),
        "talk2dino-dfp-phase1b-t64-early-exit-benchmark-identity-v1", "format_version",
    )
    _verify_identity_block(document["identity"])
    _verify_scientific_motivation(document["scientific_motivation"])
    recomputed_manifest_digest, recomputed_record_digest = _verify_dependency_bindings(root, document["dependency_bindings"])
    _verify_recurrence_execution_contract(document["recurrence_execution_contract"])
    _verify_benchmark_design(document["benchmark_design"])
    _verify_population_binding(document["population_binding"], recomputed_manifest_digest=recomputed_manifest_digest, recomputed_record_digest=recomputed_record_digest)
    _verify_execution_schedule(document["execution_schedule"])
    _verify_timing_definitions(document["timing_definitions"])
    _verify_stage_profiling_contract(document["stage_profiling_contract"])
    _verify_memory_telemetry_contract(document["memory_telemetry_contract"])
    _verify_scientific_output_contract(document["scientific_output_contract"])
    _verify_summary_statistics(document["summary_statistics"])
    _verify_decision_rule(document["decision_rule"])
    _verify_authorization_boundary(document["authorization_boundary"])
    _verify_future_order(document["future_order"])
    _verify_prohibited(document["prohibited"])

    return document


# ---------------------------------------------------------------------
# E25 artifacts identity (closed trial/merged/decision schemas).
# ---------------------------------------------------------------------

_ARTIFACTS_IDENTITY_KEYS = frozenset({
    "name", "schema_name", "schema_version", "description", "required_ancestor_commit",
    "working_branch", "successor_of", "storage_representation_only", "mutates_or_version_bumps_earlier_identity",
})
_ARTIFACTS_TOP_LEVEL_KEYS = frozenset({"format_version", "identity", "protocol_binding", "trial_schema", "result_schema", "decision_schema", "artifact_write_policy"})
_PROTOCOL_BINDING_KEYS = frozenset({"protocol_identity_relative_path", "protocol_identity_sha256"})


def load_artifacts_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load, closed-schema-validate, and relationally hash-verify the
    E25 artifacts identity, including cross-verifying its
    `protocol_binding.protocol_identity_sha256` against the ACTUAL E25
    identity file bytes (never trusting the recorded hash alone)."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / ARTIFACTS_IDENTITY_RELATIVE_PATH
    try:
        with open(source, "rb") as fh:
            document = tomllib.load(fh)
    except OSError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"cannot read E25 artifacts identity {source}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"E25 artifacts identity {source} is not valid TOML: {exc}") from exc

    _require_closed(document, _ARTIFACTS_TOP_LEVEL_KEYS, "<top level>")
    _require_exact_value(
        _require_string(document["format_version"], "format_version"),
        "talk2dino-dfp-phase1b-t64-early-exit-benchmark-artifacts-identity-v1", "format_version",
    )

    identity = _require_closed(document["identity"], _ARTIFACTS_IDENTITY_KEYS, "identity")
    _require_exact_value(_require_string(identity["name"], "identity.name"), "e25-dfp-phase1b-t64-early-exit-benchmark-artifacts", "identity.name")
    _require_exact_value(_require_string(identity["schema_name"], "identity.schema_name"), ARTIFACTS_SCHEMA_NAME, "identity.schema_name")
    _require_exact_value(_require_exact_int(identity["schema_version"], "identity.schema_version"), ARTIFACTS_SCHEMA_VERSION, "identity.schema_version")
    _require_string(identity["description"], "identity.description")
    commit = _require_string(identity["required_ancestor_commit"], "identity.required_ancestor_commit")
    if not _HEX40.match(commit):
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"identity.required_ancestor_commit must be 40-hex, observed {commit!r}")
    _require_exact_value(commit, REQUIRED_ANCESTOR_COMMIT, "identity.required_ancestor_commit")
    _require_exact_value(_require_string(identity["working_branch"], "identity.working_branch"), EXPECTED_WORKING_BRANCH, "identity.working_branch")
    _require_exact_value(_require_string(identity["successor_of"], "identity.successor_of"), EXPECTED_ARTIFACTS_SUCCESSOR_OF, "identity.successor_of")
    _require_exact_value(_require_bool(identity["storage_representation_only"], "identity.storage_representation_only"), True, "identity.storage_representation_only")
    _require_exact_value(_require_bool(identity["mutates_or_version_bumps_earlier_identity"], "identity.mutates_or_version_bumps_earlier_identity"), False, "identity.mutates_or_version_bumps_earlier_identity")

    binding = _require_closed(document["protocol_binding"], _PROTOCOL_BINDING_KEYS, "protocol_binding")
    rel = _require_string(binding["protocol_identity_relative_path"], "protocol_binding.protocol_identity_relative_path")
    expected_hash = _require_sha256(binding["protocol_identity_sha256"], "protocol_binding.protocol_identity_sha256")
    target = _resolve_repo_relative(root, rel, label="protocol_binding")
    try:
        observed_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"protocol_binding: cannot read {rel}: {exc}") from exc
    if observed_hash != expected_hash:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("protocol_binding.protocol_identity_sha256 drift against the actual E25 identity file bytes")

    _verify_trial_schema(document["trial_schema"])
    _verify_result_schema(document["result_schema"])
    _verify_decision_schema(document["decision_schema"])
    _verify_artifact_write_policy(document["artifact_write_policy"])

    return document


_TRIAL_SCHEMA_KEYS = frozenset({"schema_name", "schema_version", "top_keys", "variant_id_closed_set", "device_record_keys", "software_record_keys", "precision_record_keys", "deterministic_settings_record_keys", "counts_keys", "timing_keys", "memory_keys", "scientific_output_keys"})
_TRIAL_TOP_KEYS_REQUIRED = (
    "schema_name", "schema_version",
    "identity_sha256", "artifacts_identity_sha256", "manifest_sha256",
    "repository_commit", "required_ancestor_commit", "repository_worktree_clean",
    "identity_loader_sha256", "runner_sha256", "cli_sha256", "launcher_sha256",
    "variant_id", "trial_index", "repeat_index", "is_warmup",
    "device_record", "software_record", "precision_record", "deterministic_settings_record",
    "counts", "timing", "memory", "scientific_output",
    "complete", "final",
)


def _verify_trial_schema(node: Any) -> None:
    node = _require_closed(node, _TRIAL_SCHEMA_KEYS, "trial_schema")
    _require_exact_value(_require_string(node["schema_name"], "..name"), "talk2dino-dfp-phase1b-t64-early-exit-benchmark-trial-v1", "trial_schema.schema_name")
    _require_exact_value(_require_exact_int(node["schema_version"], "..version"), 1, "trial_schema.schema_version")

    top_keys = _require_closed(node["top_keys"], frozenset({"list", "no_gt_metric_field_permitted", "no_winner_or_recommendation_field_permitted"}), "trial_schema.top_keys")
    if tuple(top_keys["list"]) != _TRIAL_TOP_KEYS_REQUIRED:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("trial_schema.top_keys.list disagrees with the frozen required list")
    _require_exact_value(_require_bool(top_keys["no_gt_metric_field_permitted"], "..gt"), True, "trial_schema.top_keys.no_gt_metric_field_permitted")
    _require_exact_value(_require_bool(top_keys["no_winner_or_recommendation_field_permitted"], "..winner"), True, "trial_schema.top_keys.no_winner_or_recommendation_field_permitted")

    variant_set = _require_closed(node["variant_id_closed_set"], frozenset({"list"}), "trial_schema.variant_id_closed_set")
    _require_exact_value(list(variant_set["list"]), list(VARIANTS), "trial_schema.variant_id_closed_set.list")

    for key, count in (("counts_keys", 4), ("timing_keys", None), ("memory_keys", None), ("scientific_output_keys", None)):
        _require_mapping(node[key], f"trial_schema.{key}")

    counts = node["counts_keys"]
    _require_exact_value(tuple(counts["list"]), ("window_count", "recurrence_steps_executed", "model_forward_count", "graph_construction_count"), "trial_schema.counts_keys.list")
    _require_exact_value(_require_exact_int(counts["window_count_required_value"], "..w"), MEASURED_WINDOW_COUNT, "trial_schema.counts_keys.window_count_required_value")
    _require_exact_value(_require_exact_int(counts["t64_recurrence_steps_executed_required_value"], "..t64steps"), MEASURED_WINDOW_COUNT * T64_STEPS, "trial_schema.counts_keys.t64_recurrence_steps_executed_required_value")
    _require_exact_value(_require_exact_int(counts["t320_recurrence_steps_executed_required_value"], "..t320steps"), MEASURED_WINDOW_COUNT * T320_STEPS, "trial_schema.counts_keys.t320_recurrence_steps_executed_required_value")
    _require_exact_value(_require_exact_int(counts["model_forward_count_required_value"], "..fwd"), MEASURED_WINDOW_COUNT, "trial_schema.counts_keys.model_forward_count_required_value")
    _require_exact_value(_require_exact_int(counts["graph_construction_count_required_value"], "..graph"), MEASURED_WINDOW_COUNT, "trial_schema.counts_keys.graph_construction_count_required_value")

    memory = node["memory_keys"]
    _require_exact_value(tuple(memory["list"]), ("host_peak_rss_bytes", "gpu_peak_memory_allocated_bytes", "gpu_peak_memory_reserved_bytes", "gpu_total_memory_bytes"), "trial_schema.memory_keys.list")
    for key in ("all_non_negative_integers_required", "allocated_le_reserved_required", "reserved_le_total_required", "missing_gpu_field_on_cuda_trial_forbidden"):
        _require_exact_value(_require_bool(memory[key], f"..{key}"), True, f"trial_schema.memory_keys.{key}")

    sci = node["scientific_output_keys"]
    _require_exact_value(tuple(sci["list"]), ("scientific_output_collection_digest", "per_window_output_digests", "finite_value_count", "non_finite_value_count"), "trial_schema.scientific_output_keys.list")
    _require_exact_value(_require_exact_int(sci["per_window_output_digests_length_required"], "..len"), MEASURED_WINDOW_COUNT, "trial_schema.scientific_output_keys.per_window_output_digests_length_required")
    for key in ("scientific_output_collection_digest_hex64_required", "non_finite_value_count_must_be_zero", "finite_value_count_must_be_positive"):
        _require_exact_value(_require_bool(sci[key], f"..{key}"), True, f"trial_schema.scientific_output_keys.{key}")

    for key in ("device_record_keys", "software_record_keys", "precision_record_keys", "deterministic_settings_record_keys"):
        block = _require_closed(node[key], frozenset({"list"}), f"trial_schema.{key}")
        if not isinstance(block["list"], list) or not block["list"]:
            raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"trial_schema.{key}.list must be a non-empty list")

    timing = node["timing_keys"]
    if "measured_window_loop_wall_ns" not in timing["list"] or "process_total_wall_ns" not in timing["list"]:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("trial_schema.timing_keys.list is missing required primary timing fields")
    _require_exact_value(_require_bool(timing["integer_fields_strict_type_is_int_bool_rejected"], "..strict"), True, "trial_schema.timing_keys.integer_fields_strict_type_is_int_bool_rejected")
    _require_exact_value(_require_bool(timing["integer_fields_non_negative_required"], "..nonneg"), True, "trial_schema.timing_keys.integer_fields_non_negative_required")
    _require_exact_value(_require_bool(timing["accounting_identity_required"], "..acct"), True, "trial_schema.timing_keys.accounting_identity_required")
    _require_exact_value(_require_bool(timing["stage_timings_never_the_primary_runtime_decision_field"], "..stage"), True, "trial_schema.timing_keys.stage_timings_never_the_primary_runtime_decision_field")


_RESULT_SCHEMA_KEYS = frozenset({"schema_name", "schema_version", "top_keys", "measured_trials", "warmup_trials", "schedule_validation", "per_variant_summary_keys", "speedup_keys", "memory_summary_keys", "digest_equivalence_keys", "decision_rule_inputs_keys", "outcome", "no_extrapolation"})
_RESULT_TOP_KEYS_REQUIRED = (
    "schema_name", "schema_version", "identity_sha256", "artifacts_identity_sha256", "manifest_sha256", "repository_commit",
    "measured_trials", "warmup_trials", "schedule_validation", "per_variant_summary", "speedup", "memory_summary",
    "digest_equivalence", "decision_rule_inputs", "outcome", "complete", "final",
)


def _verify_result_schema(node: Any) -> None:
    node = _require_closed(node, _RESULT_SCHEMA_KEYS, "result_schema")
    _require_exact_value(_require_string(node["schema_name"], "..name"), "talk2dino-dfp-phase1b-t64-early-exit-benchmark-result-v1", "result_schema.schema_name")
    _require_exact_value(_require_exact_int(node["schema_version"], "..version"), 1, "result_schema.schema_version")

    top_keys = _require_closed(node["top_keys"], frozenset({"list"}), "result_schema.top_keys")
    if tuple(top_keys["list"]) != _RESULT_TOP_KEYS_REQUIRED:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("result_schema.top_keys.list disagrees with the frozen required list")

    measured = node["measured_trials"]
    _require_exact_value(_require_exact_int(measured["required_count"], "..count"), TOTAL_MEASURED_TRIALS, "result_schema.measured_trials.required_count")
    for key in (
        "requires_all_six_frozen_trials_present_exactly_once", "rejects_duplicate_or_missing_trials",
        "verifies_hashes_and_closed_schemas", "verifies_identical_hardware_software_precision_across_all_six_trials",
        "verifies_identical_code_provenance_across_all_six_trials", "verifies_within_variant_scientific_digest_equality",
        "preserves_all_raw_measurements",
    ):
        _require_exact_value(_require_bool(measured[key], f"..{key}"), True, f"result_schema.measured_trials.{key}")
    _require_exact_value(_require_bool(measured["cross_variant_digest_equality_not_required"], "..cross"), True, "result_schema.measured_trials.cross_variant_digest_equality_not_required")

    warmup = node["warmup_trials"]
    _require_exact_value(_require_exact_int(warmup["required_count"], "..count"), 2, "result_schema.warmup_trials.required_count")
    _require_exact_value(_require_bool(warmup["schema_valid_required"], "..valid"), True, "result_schema.warmup_trials.schema_valid_required")
    _require_exact_value(_require_bool(warmup["excluded_from_per_variant_summary"], "..excl"), True, "result_schema.warmup_trials.excluded_from_per_variant_summary")

    schedule = node["schedule_validation"]
    _require_exact_value(_require_bool(schedule["required_order_matches_e25_counterbalanced_order"], "..order"), True, "result_schema.schedule_validation.required_order_matches_e25_counterbalanced_order")
    _require_exact_value(_require_bool(schedule["wrong_order_rejected"], "..reject"), True, "result_schema.schedule_validation.wrong_order_rejected")

    per_variant = node["per_variant_summary_keys"]
    if tuple(per_variant["list"]) != _REQUIRED_SUMMARY_FIELDS:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("result_schema.per_variant_summary_keys.list disagrees with the frozen required list")

    speedup = node["speedup_keys"]
    _require_exact_value(tuple(speedup["list"]), ("t64_speedup_over_t320", "t64_recurrence_speedup_over_t320"), "result_schema.speedup_keys.list")
    _require_exact_value(_require_bool(speedup["t64_recurrence_speedup_over_t320_nullable_if_recurrence_timing_unavailable"], "..null"), True, "result_schema.speedup_keys.t64_recurrence_speedup_over_t320_nullable_if_recurrence_timing_unavailable")
    _require_exact_value(_require_bool(speedup["recurrence_speedup_substitution_for_end_to_end_speedup_forbidden"], "..forbidden"), True, "result_schema.speedup_keys.recurrence_speedup_substitution_for_end_to_end_speedup_forbidden")

    mem_summary = node["memory_summary_keys"]
    _require_exact_value(list(mem_summary["list"]), list(VARIANTS), "result_schema.memory_summary_keys.list")

    digest_eq = node["digest_equivalence_keys"]
    _require_exact_value(tuple(digest_eq["list"]), ("t64_within_variant_digest_equal", "t320_within_variant_digest_equal", "non_finite_value_count_total"), "result_schema.digest_equivalence_keys.list")
    _require_exact_value(_require_bool(digest_eq["non_finite_value_count_total_must_be_zero"], "..zero"), True, "result_schema.digest_equivalence_keys.non_finite_value_count_total_must_be_zero")

    decision_inputs = node["decision_rule_inputs_keys"]
    expected_decision_inputs = (
        "all_trials_complete", "all_artifacts_verified", "within_variant_digests_identical", "zero_non_finite_outputs",
        "exact_counts_verified", "no_oom", "memory_relations_pass", "t64_median_lower_than_t320", "end_to_end_speedup",
        "development_accuracy_disclosure_pp",
    )
    _require_exact_value(tuple(decision_inputs["list"]), expected_decision_inputs, "result_schema.decision_rule_inputs_keys.list")
    _require_exact_value(decision_inputs["development_accuracy_disclosure_pp_required_value"], DEVELOPMENT_ACCURACY_DISCLOSURE_PP, "result_schema.decision_rule_inputs_keys.development_accuracy_disclosure_pp_required_value")
    _require_exact_value(decision_inputs["end_to_end_speedup_threshold"], END_TO_END_SPEEDUP_THRESHOLD, "result_schema.decision_rule_inputs_keys.end_to_end_speedup_threshold")

    outcome = node["outcome"]
    _require_exact_value(list(outcome["closed_set"]), list(DECISION_OUTCOMES), "result_schema.outcome.closed_set")
    _require_exact_value(_require_bool(outcome["no_third_value_permitted"], "..third"), True, "result_schema.outcome.no_third_value_permitted")

    no_extrap = node["no_extrapolation"]
    _require_exact_value(_require_bool(no_extrap["five_thousand_image_projection_forbidden_in_this_schema"], "..forbidden"), True, "result_schema.no_extrapolation.five_thousand_image_projection_forbidden_in_this_schema")


_DECISION_SCHEMA_KEYS = frozenset({"schema_name", "schema_version", "top_keys", "proceed_does_not_authorize"})
_DECISION_TOP_KEYS_REQUIRED = (
    "schema_name", "schema_version", "merged_result_sha256", "decision_rule_inputs", "end_to_end_speedup_threshold",
    "outcome", "proceed_does_not_authorize", "complete", "final",
)


def _verify_decision_schema(node: Any) -> None:
    node = _require_closed(node, _DECISION_SCHEMA_KEYS, "decision_schema")
    _require_exact_value(_require_string(node["schema_name"], "..name"), "talk2dino-dfp-phase1b-t64-early-exit-decision-v1", "decision_schema.schema_name")
    _require_exact_value(_require_exact_int(node["schema_version"], "..version"), 1, "decision_schema.schema_version")

    top_keys = _require_closed(node["top_keys"], frozenset({"list", "merged_result_sha256_required_and_verified", "decision_rule_inputs_must_equal_merged_result_own_field", "outcome_must_equal_merged_result_own_field", "end_to_end_speedup_threshold_required_value"}), "decision_schema.top_keys")
    if tuple(top_keys["list"]) != _DECISION_TOP_KEYS_REQUIRED:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("decision_schema.top_keys.list disagrees with the frozen required list")
    for key in ("merged_result_sha256_required_and_verified", "decision_rule_inputs_must_equal_merged_result_own_field", "outcome_must_equal_merged_result_own_field"):
        _require_exact_value(_require_bool(top_keys[key], f"..{key}"), True, f"decision_schema.top_keys.{key}")
    _require_exact_value(top_keys["end_to_end_speedup_threshold_required_value"], END_TO_END_SPEEDUP_THRESHOLD, "decision_schema.top_keys.end_to_end_speedup_threshold_required_value")

    does_not = _require_closed(node["proceed_does_not_authorize"], frozenset({"list", "required_present_verbatim"}), "decision_schema.proceed_does_not_authorize")
    if tuple(does_not["list"]) != _REQUIRED_PROCEED_DOES_NOT_AUTHORIZE:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("decision_schema.proceed_does_not_authorize.list disagrees with the frozen required list")
    _require_exact_value(_require_bool(does_not["required_present_verbatim"], "..verbatim"), True, "decision_schema.proceed_does_not_authorize.required_present_verbatim")


_ARTIFACT_WRITE_POLICY_KEYS = frozenset({
    "atomic_write_required", "atomic_write_mechanism", "overwrite_authorized", "force_mode_authorized",
    "reopen_and_public_post_install_reverification_required", "unknown_top_level_or_nested_field_always_rejected",
})


def _verify_artifact_write_policy(node: Any) -> None:
    node = _require_closed(node, _ARTIFACT_WRITE_POLICY_KEYS, "artifact_write_policy")
    _require_exact_value(_require_bool(node["atomic_write_required"], "..atomic"), True, "artifact_write_policy.atomic_write_required")
    _require_exact_value(_require_bool(node["overwrite_authorized"], "..overwrite"), False, "artifact_write_policy.overwrite_authorized")
    _require_exact_value(_require_bool(node["force_mode_authorized"], "..force"), False, "artifact_write_policy.force_mode_authorized")
    _require_exact_value(_require_bool(node["reopen_and_public_post_install_reverification_required"], "..reopen"), True, "artifact_write_policy.reopen_and_public_post_install_reverification_required")
    _require_exact_value(_require_bool(node["unknown_top_level_or_nested_field_always_rejected"], "..unknown"), True, "artifact_write_policy.unknown_top_level_or_nested_field_always_rejected")
    _require_string(node["atomic_write_mechanism"], "artifact_write_policy.atomic_write_mechanism")


# ---------------------------------------------------------------------
# Pure schema/identity validation helpers (explicitly permitted by E25:
# "no trial merging code beyond pure schema/identity validation
# helpers"). These check INTERNAL CONSISTENCY of already-computed
# values a future merged-result artifact would carry -- they compute
# nothing from raw per-window data, orchestrate no trial execution,
# and read/write no file. No torch/CUDA/model/dataset access.
# ---------------------------------------------------------------------


def verify_timing_accounting_identity(timing: Mapping[str, Any]) -> None:
    """Requires `process_total_wall_ns` to exactly equal the sum of the
    other six frozen phase fields, and every one of the seven fields to
    be a strict, non-negative int (bool rejected)."""
    required = (
        "process_total_wall_ns", "preflight_wall_ns", "dataset_construction_wall_ns",
        "model_construction_and_checkpoint_loading_wall_ns", "measured_window_loop_wall_ns",
        "artifact_validation_and_write_wall_ns", "unattributed_overhead_ns",
    )
    for key in required:
        if key not in timing:
            raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"timing is missing required field {key!r}")
        value = timing[key]
        if isinstance(value, bool) or type(value) is not int:
            raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"timing[{key!r}] must be an exact int, observed {type(value).__name__}={value!r}")
        if value < 0:
            raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"timing[{key!r}] must be non-negative, observed {value}")
    total = timing["process_total_wall_ns"]
    parts_sum = sum(timing[k] for k in required[1:])
    if total != parts_sum:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(
            f"timing accounting identity violated: process_total_wall_ns={total} != sum of parts={parts_sum}"
        )


def verify_count_relations(counts: Mapping[str, Any], *, variant_id: str) -> None:
    """Requires exact window/forward/graph/recurrence-step count
    relations for one trial's `counts` block, per E25's frozen
    per-variant recurrence-step formula."""
    if variant_id not in VARIANTS:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"variant_id must be one of {VARIANTS}, observed {variant_id!r}")
    for key in ("window_count", "recurrence_steps_executed", "model_forward_count", "graph_construction_count"):
        value = counts.get(key)
        if isinstance(value, bool) or type(value) is not int:
            raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"counts[{key!r}] must be an exact int, observed {type(value).__name__ if key in counts else 'missing'}")
    window_count = counts["window_count"]
    if window_count != MEASURED_WINDOW_COUNT:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"counts['window_count'] must be {MEASURED_WINDOW_COUNT}, observed {window_count}")
    steps_per_window = T64_STEPS if variant_id == "T64_early_exit" else T320_STEPS
    expected_steps = window_count * steps_per_window
    if counts["recurrence_steps_executed"] != expected_steps:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(
            f"counts['recurrence_steps_executed'] must be {expected_steps} ({window_count} x {steps_per_window}) for {variant_id}, observed {counts['recurrence_steps_executed']}"
        )
    if counts["model_forward_count"] != window_count:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"counts['model_forward_count'] must equal window_count ({window_count}), observed {counts['model_forward_count']}")
    if counts["graph_construction_count"] != window_count:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"counts['graph_construction_count'] must equal window_count ({window_count}), observed {counts['graph_construction_count']}")


def verify_memory_relations(memory: Mapping[str, Any]) -> None:
    """Requires all four memory fields to be strict non-negative ints
    with allocated <= reserved <= total."""
    required = ("host_peak_rss_bytes", "gpu_peak_memory_allocated_bytes", "gpu_peak_memory_reserved_bytes", "gpu_total_memory_bytes")
    for key in required:
        if key not in memory:
            raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"memory is missing required field {key!r}")
        value = memory[key]
        if isinstance(value, bool) or type(value) is not int:
            raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"memory[{key!r}] must be an exact int, observed {type(value).__name__}={value!r}")
        if value < 0:
            raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"memory[{key!r}] must be non-negative, observed {value}")
    allocated, reserved, total = memory["gpu_peak_memory_allocated_bytes"], memory["gpu_peak_memory_reserved_bytes"], memory["gpu_total_memory_bytes"]
    if allocated > reserved:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"gpu_peak_memory_allocated_bytes ({allocated}) must not exceed gpu_peak_memory_reserved_bytes ({reserved})")
    if reserved > total:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"gpu_peak_memory_reserved_bytes ({reserved}) must not exceed gpu_total_memory_bytes ({total})")


def verify_scientific_output_finiteness(scientific_output: Mapping[str, Any]) -> None:
    """Requires non_finite_value_count == 0 and finite_value_count > 0,
    both strict non-negative ints."""
    for key in ("finite_value_count", "non_finite_value_count"):
        value = scientific_output.get(key)
        if isinstance(value, bool) or type(value) is not int or value < 0:
            raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"scientific_output[{key!r}] must be a non-negative exact int, observed {value!r}")
    if scientific_output["non_finite_value_count"] != 0:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"non_finite_value_count must be 0, observed {scientific_output['non_finite_value_count']}")
    if scientific_output["finite_value_count"] <= 0:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"finite_value_count must be positive, observed {scientific_output['finite_value_count']}")


def verify_within_variant_digest_equality(digests: "list[str] | tuple[str, ...]", *, variant_id: str) -> None:
    """Requires exactly 3 measured-trial scientific_output_collection_digest
    values for one variant, all bitwise identical hex-64 strings."""
    digests = tuple(digests)
    if len(digests) != MEASURED_REPEATS.get(variant_id, 3):
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"expected exactly {MEASURED_REPEATS.get(variant_id, 3)} measured digests for {variant_id}, observed {len(digests)}")
    for d in digests:
        _require_sha256(d, f"digest for {variant_id}")
    if len(set(digests)) != 1:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"within-variant scientific digests for {variant_id} are not all identical: {digests}")


def exact_median_of_three(values: "list[int | float] | tuple[int | float, ...]") -> "int | float":
    """Pure, deterministic median of exactly 3 numeric values -- the
    middle value after sorting. Never averages two values (Python's
    `statistics.median` would average for an even-length input; this
    helper only ever accepts exactly 3, so that ambiguity never
    arises)."""
    values = tuple(values)
    if len(values) != 3:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"exact_median_of_three requires exactly 3 values, observed {len(values)}")
    for v in values:
        if isinstance(v, bool) or type(v) not in (int, float):
            raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"exact_median_of_three values must be int or float (not bool), observed {type(v).__name__}={v!r}")
    return sorted(values)[1]


def verify_declared_median(values: "list[int | float] | tuple[int | float, ...]", declared_median: "int | float") -> None:
    """Requires `declared_median` to equal the exact median of exactly
    3 raw measured values, bit-for-bit (no tolerance)."""
    expected = exact_median_of_three(values)
    if declared_median != expected:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"declared median {declared_median!r} disagrees with the exact median of the 3 raw values {expected!r}")


def verify_declared_speedup(*, t320_median_seconds: float, t64_median_seconds: float, declared_speedup: float) -> None:
    """Requires `declared_speedup` to equal
    `t320_median_seconds / t64_median_seconds` exactly (float
    equality; both inputs must already be positive)."""
    for label, value in (("t320_median_seconds", t320_median_seconds), ("t64_median_seconds", t64_median_seconds)):
        if isinstance(value, bool) or type(value) not in (int, float) or value <= 0:
            raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"{label} must be a positive int or float, observed {value!r}")
    expected = t320_median_seconds / t64_median_seconds
    if declared_speedup != expected:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"declared speedup {declared_speedup!r} disagrees with the exact recomputation {expected!r}")


def apply_decision_rule(*, all_trials_complete: bool, all_artifacts_verified: bool, within_variant_digests_identical: bool,
                         zero_non_finite_outputs: bool, exact_counts_verified: bool, no_oom: bool, memory_relations_pass: bool,
                         t64_median_seconds: float, t320_median_seconds: float, end_to_end_speedup: float,
                         development_accuracy_disclosure_pp: float) -> str:
    """Pure application of E25's own frozen decision rule (Part L) to
    already-computed, already-verified inputs -- computes no raw
    measurement itself, orchestrates no trial. Every boolean input must
    itself already be an independently-verified fact; this function
    only combines them per the frozen rule."""
    for label, value in (
        ("all_trials_complete", all_trials_complete), ("all_artifacts_verified", all_artifacts_verified),
        ("within_variant_digests_identical", within_variant_digests_identical), ("zero_non_finite_outputs", zero_non_finite_outputs),
        ("exact_counts_verified", exact_counts_verified), ("no_oom", no_oom), ("memory_relations_pass", memory_relations_pass),
    ):
        if type(value) is not bool:
            raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(f"{label} must be a bool, observed {type(value).__name__}")
    if type(development_accuracy_disclosure_pp) is not float:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError("development_accuracy_disclosure_pp must be a float")
    if development_accuracy_disclosure_pp != DEVELOPMENT_ACCURACY_DISCLOSURE_PP:
        raise DfpPhase1bT64EarlyExitBenchmarkIdentityError(
            f"development_accuracy_disclosure_pp must remain {DEVELOPMENT_ACCURACY_DISCLOSURE_PP!r}, observed {development_accuracy_disclosure_pp!r}"
        )

    t64_lower = t64_median_seconds < t320_median_seconds
    all_gates = (
        all_trials_complete and all_artifacts_verified and within_variant_digests_identical and zero_non_finite_outputs
        and exact_counts_verified and no_oom and memory_relations_pass and t64_lower
        and end_to_end_speedup >= END_TO_END_SPEEDUP_THRESHOLD
    )
    return DECISION_OUTCOMES[0] if all_gates else DECISION_OUTCOMES[1]
