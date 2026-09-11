"""Strict loading/validation for the DFP Phase 1B Stage-A provenance-
feature successor identity
(`evaluation_identities/e21_dfp_phase1b_provenance_features.toml`).

This module is a NEW LEAF authority (E21). It is an ADDITIVE SUCCESSOR
to E20: it NEVER rewrites, re-binds, or cascades new hashes through
E13-E20, the verified calibration artifact, the installed Task 4
selection artifact, or the exact masked-forward core. It only
relationally re-verifies their SHA-256 against current file bytes on
every load.

It implements no execution driver, runner, profiler, checkpoint
writer, result writer, CLI execution command, or SLURM launcher. It
performs no torch / CUDA / model / dataset / image access -- pure
TOML/filesystem/hash validation and exact comparison only (it does
NOT import `src.dfp_phase1b_provenance_features` -- the numerical
kernel it hash-binds -- merely to hash-validate it). `KeyboardInterrupt`,
`SystemExit`, and `MemoryError` always propagate unmodified; ordinary
parse / type / path / relational failures are wrapped in
`DfpPhase1bProvenanceFeaturesIdentityError`.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path
from typing import Any, Mapping

from src.dfp_phase1b_chunk_benchmark_identity import (
    DfpPhase1bChunkBenchmarkIdentityError,
    resolve_repository_relative_path,
)


class DfpPhase1bProvenanceFeaturesIdentityError(ValueError):
    """Raised on any DFP Phase 1B provenance-features identity
    invariant violation. Always fail closed. Never raised for, and
    never catches, KeyboardInterrupt / SystemExit / MemoryError --
    those always propagate unmodified."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e21_dfp_phase1b_provenance_features.toml")
FORMAT_VERSION = "Talk2Dino-DFP-Phase1B-Provenance-Features-Identity-V1"
IDENTITY_SCHEMA_NAME = "talk2dino-dfp-phase1b-provenance-features-identity-v1"
IDENTITY_SCHEMA_VERSION = 1
REQUIRED_ANCESTOR_COMMIT = "38df8087abd3e0fbc163523c471c2cdd01e7d96c"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise DfpPhase1bProvenanceFeaturesIdentityError(f"{label} must be a string, observed {type(value).__name__}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    s = _require_string(value, label)
    if not _HEX64.match(s):
        raise DfpPhase1bProvenanceFeaturesIdentityError(f"{label} must be lowercase 64-hex, observed {value!r}")
    return s


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DfpPhase1bProvenanceFeaturesIdentityError(f"{label} must be a table")
    return value


def _require_closed(node: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    node = _require_mapping(node, label)
    actual = frozenset(node.keys())
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
    if missing or unknown:
        raise DfpPhase1bProvenanceFeaturesIdentityError(f"{label} has the wrong key set -- missing={missing} unknown={unknown}")
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
# Mechanically-flattened exact-value table for every E21 leaf OUTSIDE
# [dependency_bindings] -- generated once from the committed TOML.
# ---------------------------------------------------------------------

_CANONICAL: dict[str, Any] = {
    'authorization_boundary.authority_and_kernel_only_task': True,
    'authorization_boundary.checkpoint_writer_implemented_by_this_task': False,
    'authorization_boundary.cli_execution_command_implemented_by_this_task': False,
    'authorization_boundary.coco_stuff_dataset_accessed': False,
    'authorization_boundary.gpu_execution_performed': False,
    'authorization_boundary.phase1b_5000_image_exact_provenance_authorized': False,
    'authorization_boundary.phase1b_500_image_execution_authorized': False,
    'authorization_boundary.profiler_implemented_by_this_task': False,
    'authorization_boundary.real_inference_executed': False,
    'authorization_boundary.result_writer_implemented_by_this_task': False,
    'authorization_boundary.runner_implemented_by_this_task': False,
    'authorization_boundary.scientific_kernel_only': True,
    'authorization_boundary.slurm_launcher_implemented_by_this_task': False,
    'disclosure.5000_image_exact_provenance_requires_separate_explicit_authorization': True,
    'disclosure.500_image_execution_requires_separate_explicit_authorization': True,
    'disclosure.kernel_does_not_access_coco_stuff_images_or_annotations': True,
    'disclosure.kernel_does_not_compute_g_or_m320_itself': True,
    'disclosure.kernel_does_not_initialize_cuda_on_import': True,
    'disclosure.kernel_does_not_run_masked_propagation': True,
    'disclosure.no_e15_decision_re_derived_or_re_selected': True,
    'disclosure.no_new_scientific_formula_invented': True,
    'disclosure.runner_implementation_requires_separate_explicit_authorization': True,
    'disclosure.the_eight_stage_a_fields_were_previously_text_only_authority': True,
    'disclosure.this_identity_repairs_a_missing_implementation_gap': True,
    'disclosure.this_task_does_not_resume_runner_implementation': True,
    'domain_invariants.conservation_identity': 'sum_r G[x,r] = delta_m(x); e15 [component_evidence_decomposition].conservation_identity',
    'domain_invariants.delta_m_strictly_positive': 'delta_m(x) = m320(x) - m0(x) > 0; e15 [margins].delta_m_definition',
    'domain_invariants.k_star_always_exists_in_range': 'a k in [1, n_positive_components] satisfying the k_star crossing condition always exists for a valid disputed pixel, because positive_evidence_sum >= m320(x); k_star is therefore always finite and >= 1, matching e15 k_star_always_finite_for_valid_disputed_pixel = true',
    'domain_invariants.m0_non_positive': 'm0(x) <= 0 for every valid disputed pixel; e15 [margins].m0_definition',
    'domain_invariants.m320_strictly_positive': 'm320(x) > 0 for every valid disputed pixel; e15 [margins].m320_definition',
    'domain_invariants.n_positive_components_at_least_one': 'n_positive_components >= 1 for every valid disputed pixel; implied by positive_evidence_sum > 0',
    'domain_invariants.positive_evidence_sum_at_least_delta_m': 'positive_evidence_sum = delta_m(x) - negative_evidence_sum >= delta_m(x) >= m320(x), because negative_evidence_sum <= 0 and m0(x) <= 0 -- derived, not independently frozen',
    'domain_invariants.positive_evidence_sum_strictly_positive': 'positive_evidence_sum > 0 for every valid disputed pixel; also directly required by src/dfp_phase1b_artifact_schema.py [stage_a_schema.relational_validation]',
    'domain_invariants.violation_behavior': 'reject with DfpPhase1bProvenanceFeaturesError; never invent a fallback k_star, phi, or share value for a pixel that violates one of the invariants above',
    'format_version': 'Talk2Dino-DFP-Phase1B-Provenance-Features-Identity-V1',
    'future_step_bindings.description': 'Fields explicitly deferred to a later task, never guessed or invented here.',
    'future_step_bindings.next_step_items': ['independently re-verify this E21 identity, kernel, and its loader/CLI module against the committed HEAD', 'resume implementation of the profiled, resumable 500-image diagnostic runner bound by E20, now unblocked by this kernel', 'explicitly authorize the 500-image diagnostic execution itself -- never this identity', 'the 5,000-image exact masked-forward provenance question -- never this identity'],
    'identity.does_not_modify_e13_through_e20': True,
    'identity.name': 'e21-dfp-phase1b-provenance-features',
    'identity.purpose': "Freezes the numerical kernel that computes the eight E15 Stage-A component-evidence provenance features (n_positive_components, positive_evidence_sum, negative_evidence_sum, top_positive_component_id, top_positive_evidence, top1_positive_share, k_star, phi) from the signed component-evidence matrix G[x,r], exactly per e15-dfp-phase1b-protocol's [provenance_features.component_evidence] formulas. Repairs the gap in which these eight fields were frozen only as TEXT authority with no implementation, blocking the profiled 500-image diagnostic runner. Implements no execution driver, runner, profiler, checkpoint/result writer, CLI execution command, or SLURM launcher. Authorizes no inference, no GPU job, and no 500-image or 5000-image execution.",
    'identity.required_ancestor_commit': '38df8087abd3e0fbc163523c471c2cdd01e7d96c',
    'identity.required_ancestor_relation': "required_ancestor_commit must be an ancestor of (or equal to) the executing repository_commit -- verified via 'git merge-base --is-ancestor <required_ancestor_commit> <repository_commit>', exactly like E17/E18/E19/E20's own convention. Every repository-file SHA-256 bound below is valid only for a worktree at or descended from this commit.",
    'identity.schema_name': 'talk2dino-dfp-phase1b-provenance-features-identity-v1',
    'identity.schema_version': 1,
    'identity.status': 'frozen_additive_successor_authority',
    'identity.successor_of': 'e20-dfp-phase1b-diagnostic-execution',
    'identity.successor_relation': "additive_only: binds but never edits, reinterprets, or re-hashes any E13-E20 file, the verified calibration artifact, the installed Task 4 selection artifact, or the exact masked-forward core; adds exactly one new scientific-kernel module implementing formulas already frozen verbatim by e15-dfp-phase1b-protocol, plus the loader/CLI/test scaffolding a new authority requires under this repository's convention",
    'identity.working_branch': 'dfp-phase1a-diagnostics',
    'kernel_binding.accumulation_dtype': 'float64',
    'kernel_binding.accumulation_dtype_source': "e15 [component_evidence_decomposition].component_sum_accumulation_dtype = 'fp64'",
    'kernel_binding.component_ids_input_definition': 'the canonical, deterministic component identity for each column of G, independent of column position, per e15 [component_construction].ordering',
    'kernel_binding.component_ids_input_name': 'component_ids',
    'kernel_binding.formulas_re_derived_by_this_kernel': False,
    'kernel_binding.formulas_re_selected_by_this_kernel': False,
    'kernel_binding.function': 'compute_component_evidence_features',
    'kernel_binding.function_signature': 'compute_component_evidence_features(G: torch.Tensor, m320: torch.Tensor, *, component_ids: torch.Tensor) -> ComponentEvidenceFeatures',
    'kernel_binding.g_input_definition': 'the already-computed signed component-evidence matrix G[x,r], per e15 [component_evidence_decomposition].signed_component_evidence_formula; this kernel never computes G itself',
    'kernel_binding.g_input_name': 'G',
    'kernel_binding.gpu_or_cpu': "operates on whatever device the caller's input tensors are on; never silently transfers a CUDA tensor to CPU, never initializes CUDA merely to run",
    'kernel_binding.input_mutation': 'never mutates G, m320, or component_ids; every returned tensor is a freshly allocated buffer, never a view into an input tensor',
    'kernel_binding.m320_input_definition': 'the per-disputed-pixel propagated margin m320(x), per e15 [margins].m320_definition; e15 [provenance_features.margins_and_controls].propagated_margin_formula aliases propagated_margin = m320 -- this kernel uses the literal m320 name used by the k_star formula itself',
    'kernel_binding.m320_input_name': 'm320',
    'kernel_binding.module': 'src.dfp_phase1b_provenance_features',
    'kernel_binding.output_fields': ['k_star', 'phi', 'top1_positive_share', 'positive_evidence_sum', 'negative_evidence_sum', 'n_positive_components', 'top_positive_component_id', 'top_positive_evidence'],
    'kernel_binding.result_type': 'ComponentEvidenceFeatures',
    'kernel_binding.reused_e15_formulas_verbatim': ['g_plus_definition', 'sort_order', 'positive_evidence_sum_formula', 'negative_evidence_sum_formula', 'k_star_definition', 'phi_formula', 'top1_positive_share_formula'],
    'kernel_binding.tolerance_or_epsilon_forbidden_source': "e15 [provenance_features.component_evidence].phi_epsilon_forbidden = true; k_star crossing condition uses exact '<=' with no epsilon",
    'kernel_binding.tolerance_or_epsilon_used_in_feature_construction': False,
    'kernel_interface_schema.component_ids_dtype': 'int64',
    'kernel_interface_schema.component_ids_shape': '[component_count]',
    'kernel_interface_schema.device_mismatch_behavior': 'reject the whole call; never silently move a tensor to another device',
    'kernel_interface_schema.duplicate_component_id_behavior': 'reject the whole call',
    'kernel_interface_schema.g_dtype': 'float64',
    'kernel_interface_schema.g_shape': '[disputed_pixel_count, component_count]',
    'kernel_interface_schema.k_star_dtype': 'int64',
    'kernel_interface_schema.m320_dtype': 'float64',
    'kernel_interface_schema.m320_shape': '[disputed_pixel_count]',
    'kernel_interface_schema.n_positive_components_dtype': 'int64',
    'kernel_interface_schema.negative_evidence_sum_dtype': 'float64',
    'kernel_interface_schema.non_finite_input_behavior': 'reject the whole call (DfpPhase1bProvenanceFeaturesError), never a per-row silent skip',
    'kernel_interface_schema.output_dtype_source': 'src/dfp_phase1b_artifact_schema.py [stage_a_schema.flat_disputed_pixel_members] dtype declarations, already committed and unmodified by this identity',
    'kernel_interface_schema.phi_dtype': 'float64',
    'kernel_interface_schema.positive_evidence_sum_dtype': 'float64',
    'kernel_interface_schema.top1_positive_share_dtype': 'float64',
    'kernel_interface_schema.top_positive_component_id_dtype': 'int64',
    'kernel_interface_schema.top_positive_evidence_dtype': 'float64',
    'kernel_interface_schema.zero_disputed_pixel_rows_behavior': 'returns zero-length tensors for every field; a legitimate degenerate case, never fabricated',
    'operational_scope.checkpoint_writer_implemented_by_this_task': False,
    'operational_scope.cli_execution_command_implemented_by_this_task': False,
    'operational_scope.coco_stuff_dataset_accessed': False,
    'operational_scope.cuda_initialized_by_this_identity_on_load': False,
    'operational_scope.gpu_execution_performed': False,
    'operational_scope.new_scientific_convention_invented_by_this_task': False,
    'operational_scope.phase1b_5000_image_exact_provenance_authorized': False,
    'operational_scope.phase1b_500_image_execution_authorized': False,
    'operational_scope.profiler_implemented_by_this_task': False,
    'operational_scope.purpose': 'stage_a_component_evidence_provenance_feature_kernel_freeze',
    'operational_scope.real_inference_executed': False,
    'operational_scope.result_writer_implemented_by_this_task': False,
    'operational_scope.runner_implemented_by_this_task': False,
    'operational_scope.scientific_kernel_only': True,
    'operational_scope.slurm_launcher_implemented_by_this_task': False,
    'prohibited.list': ['modifying_e13_through_e20_identities', 'modifying_the_verified_calibration_artifact_or_installed_task4_selection_artifact', 'modifying_masked_forward_recurrence_graph_stitching_pullback_or_component_code', 'redefining_g_plus_sort_order_positive_evidence_sum_negative_evidence_sum_k_star_or_phi_formulas', 'inventing_a_missing_e15_decision_via_this_successor_authority', 'implementing_a_runner_profiler_checkpoint_writer_result_writer_or_cli_execution_command', 'implementing_a_slurm_launcher', 'running_real_inference_or_initializing_cuda', 'accessing_coco_stuff_images_or_annotations', 'authorizing_the_500_image_diagnostic_execution', 'authorizing_exact_masked_forward_provenance_on_the_5000_image_validation_set', 'adding_a_concentration_diagnostic_not_authorized_by_e15 (gini_entropy_or_similar)', 'creating_a_circular_hash_dependency'],
}

TOP_KEYS: frozenset[str] = frozenset(['authorization_boundary', 'dependency_bindings', 'disclosure', 'domain_invariants', 'format_version', 'future_step_bindings', 'identity', 'kernel_binding', 'kernel_interface_schema', 'operational_scope', 'prohibited'])

_DEP_KEYS: tuple[str, ...] = ['diagnostic_execution_module', 'e15_artifacts', 'e15_protocol', 'e20_diagnostic_execution_identity', 'masked_forward_core', 'phase1b_artifact_schema_module', 'provenance_features_cli_module', 'provenance_features_identity_module', 'provenance_features_module']

_DEP_BINDING_KEYS = frozenset({"relative_path", "sha256", "untouched_by_this_identity"})
_DEP_BINDING_KEYS_SELFREF = frozenset({"relative_path", "sha256", "untouched_by_this_identity", "self_reference_note"})
_SELFREF_DEP_NAMES = frozenset({"provenance_features_identity_module"})


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load, closed-schema-validate, exact-value-validate, and
    relationally hash-verify the E21 provenance-features identity.
    Never mutates or re-derives any bound dependency -- only
    re-verifies SHA-256 against current bytes. Performs no torch/CUDA/
    model/dataset access, and does not import the numerical kernel
    merely to hash-validate it."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DfpPhase1bProvenanceFeaturesIdentityError(f"cannot load E21 provenance-features identity {source}: {error}") from error

    _require_closed(identity, TOP_KEYS, "E21 identity")

    dep = _require_mapping(identity.get("dependency_bindings"), "dependency_bindings")
    expected_dep = frozenset(_DEP_KEYS)
    if frozenset(dep.keys()) != expected_dep:
        raise DfpPhase1bProvenanceFeaturesIdentityError(
            f"dependency_bindings has the wrong key set -- missing={sorted(expected_dep - frozenset(dep))} "
            f"unknown={sorted(frozenset(dep) - expected_dep)}"
        )

    observed_leaves: dict[str, Any] = {}
    for dotted, value in _walk_leaves(identity):
        if dotted.startswith("dependency_bindings."):
            continue
        observed_leaves[dotted] = value
    canon_keys = frozenset(_CANONICAL)
    obs_keys = frozenset(observed_leaves)
    if canon_keys != obs_keys:
        raise DfpPhase1bProvenanceFeaturesIdentityError(
            f"E21 leaf set drifted -- missing={sorted(canon_keys - obs_keys)[:5]} extra={sorted(obs_keys - canon_keys)[:5]}"
        )
    for dotted, expected in _CANONICAL.items():
        observed = observed_leaves[dotted]
        if type(observed) is not type(expected) or observed != expected:
            raise DfpPhase1bProvenanceFeaturesIdentityError(
                f"{dotted}: frozen value drift -- expected {expected!r} ({type(expected).__name__}), "
                f"observed {observed!r} ({type(observed).__name__})"
            )

    _verify_dependency_hashes(root, dep)
    return identity


def _verify_dependency_hashes(root: Path, dependency_bindings: Mapping[str, Any]) -> None:
    for name in _DEP_KEYS:
        binding = dependency_bindings[name]
        keys = _DEP_BINDING_KEYS_SELFREF if name in _SELFREF_DEP_NAMES else _DEP_BINDING_KEYS
        _require_closed(binding, keys, f"dependency_bindings.{name}")
        sha = binding["sha256"]
        if sha == "PLACEHOLDER_RECOMPUTED_AFTER_MODULE_IS_FINAL":
            raise DfpPhase1bProvenanceFeaturesIdentityError(
                f"dependency_bindings.{name}.sha256 is still a placeholder -- recompute and freeze it"
            )
        expected = _require_sha256(sha, f"dependency_bindings.{name}.sha256")
        try:
            target = resolve_repository_relative_path(root, binding["relative_path"], binding_name=name)
        except DfpPhase1bChunkBenchmarkIdentityError as exc:
            raise DfpPhase1bProvenanceFeaturesIdentityError(
                f"dependency_bindings.{name}: unsafe or unresolvable relative_path -- {exc}"
            ) from exc
        try:
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as exc:
            raise DfpPhase1bProvenanceFeaturesIdentityError(
                f"dependency_bindings.{name}: cannot read bound file {binding['relative_path']}: {exc}"
            ) from exc
        if actual != expected:
            raise DfpPhase1bProvenanceFeaturesIdentityError(
                f"dependency_bindings.{name}: {binding['relative_path']} SHA-256 mismatch -- "
                f"pinned {expected}, observed {actual} (bound dependency drifted since E21 was frozen)"
            )


def verify_repository_gate(repo_root: Path, repository_commit: str) -> None:
    """Verify REQUIRED_ANCESTOR_COMMIT is an ancestor of (or equal to)
    ``repository_commit``, exactly like E17/E18/E19/E20's own
    convention. Raises on any failure; performs no mutation."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR_COMMIT, repository_commit],
            cwd=str(repo_root), capture_output=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DfpPhase1bProvenanceFeaturesIdentityError(f"cannot verify repository ancestor gate: {exc}") from exc
    if result.returncode != 0:
        raise DfpPhase1bProvenanceFeaturesIdentityError(
            f"required_ancestor_commit {REQUIRED_ANCESTOR_COMMIT} is not an ancestor of {repository_commit}"
        )


__all__ = [
    "DfpPhase1bProvenanceFeaturesIdentityError",
    "FORMAT_VERSION",
    "IDENTITY_RELATIVE_PATH",
    "IDENTITY_SCHEMA_NAME",
    "IDENTITY_SCHEMA_VERSION",
    "REQUIRED_ANCESTOR_COMMIT",
    "TOP_KEYS",
    "load_identity",
    "repository_root",
    "verify_repository_gate",
]
