"""Strict loading/validation for the DFP Phase 1B artifact-schema
successor identity (`evaluation_identities/e15_dfp_phase1b_artifacts.toml`)
and pure structural/relational validation of Stage-A (GT-free
provenance-feature) and Stage-B (evaluation-label) arrays against that
schema.

This module performs NO scientific computation: it never computes a
component, a soft membership, a masked propagation, a signed component
evidence value, k_star, phi, an AUROC/AUPRC, or a Boundary IoU value.
Every function here either (a) loads and closed-schema/exact-value
validates the TOML identity itself, exactly like
`src.dfp_phase1b_identity`, or (b) validates that ALREADY-COMPUTED numpy
arrays satisfy the frozen dtype/shape/relational contract -- the same
kind of pure validation `src.dfp_phase1a_checkpoint` and
`src.dfp_phase1a_artifact_schema` already perform for the completed
Phase 1A v2 schema. This is a genuinely distinct schema: it neither
imports, reuses, nor mutates any Phase 1A v2 member name or dtype.

Always fail closed: never silently substitute a default, coerce a
type, cast, reshape, or pad an array that fails the contract. Performs
no torch, CUDA, model, dataset, or image access.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class DfpPhase1bArtifactSchemaError(ValueError):
    """Raised on any DFP Phase 1B artifact-schema invariant violation.
    Always fail closed. Never raised for, and never catches,
    KeyboardInterrupt, SystemExit, or MemoryError -- those always
    propagate unmodified."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e15_dfp_phase1b_artifacts.toml")
FORMAT_VERSION = "talk2dino-dfp-phase1b-artifacts-identity-v1"

# ---------------------------------------------------------------------
# Mechanically generated closed-schema + exact-value tables (see module
# docstring of src.dfp_phase1b_identity for the generation rationale;
# identical technique applied here to e15-dfp-phase1b-artifacts).
# ---------------------------------------------------------------------

CANONICAL_VALUES: dict[str, object] = {
    'atomic_write_policy.checkpoint_install': 'write_verified_candidate_files_to_unpredictable_task_specific_temp_names_in_the_destination_filesystem_then_os_replace_the_checkpoint_pointer_last_after_full_self_verification_previous_valid_checkpoint_preserved_until_the_new_candidate_verifies_orphan_temp_files_removed_on_any_failure',
    'atomic_write_policy.result_install': 'build_the_complete_result_in_a_fresh_sibling_staging_directory_verify_every_file_and_every_cross_file_hash_then_atomically_rename_the_directory_into_place_reject_cleanly_if_the_destination_already_exists_no_partial_multi_file_replacement',
    'boundary_confusion_schema.authority': 'e14-dfp-phase1b-boundary-iou',
    'boundary_confusion_schema.cross_variant_relation': 'all Boundary IoU variants must have identical per-class ground_truth_boundary_pixels when they use the same GT and the e14 authority',
    'boundary_confusion_schema.dtype': 'int64',
    'boundary_confusion_schema.member_name_pattern': 'boundary_{variant}_{stat}',
    'boundary_confusion_schema.relation': 'boundary_union = predicted_boundary_pixels + ground_truth_boundary_pixels - boundary_intersection',
    'boundary_confusion_schema.shape': '[image_count, class_count]',
    'boundary_confusion_schema.stat_names': ['boundary_intersection', 'boundary_union', 'predicted_boundary_pixels', 'ground_truth_boundary_pixels'],
    'boundary_confusion_schema.total_member_count': 48,
    'boundary_confusion_schema.total_member_count_formula': '12 variants * 4 stats',
    'checkpoint_resume_schema.append_only_generation': True,
    'checkpoint_resume_schema.atomic_pointer_update': True,
    'checkpoint_resume_schema.complete_final_boolean_combinations_exact': ['complete=false, final=false: incomplete', 'complete=true, final=false: complete-but-unfinalized', 'complete=true, final=true: finalized'],
    'checkpoint_resume_schema.each_installed_generation_self_verifies_before_pointer_replacement': True,
    'checkpoint_resume_schema.final_result_installation_cpu_only_no_model_rebuild': True,
    'checkpoint_resume_schema.orphan_generation_after_handled_failure_forbidden': True,
    'checkpoint_resume_schema.reconcile_ambiguous_post_replace_failure_before_deleting_files': True,
    'checkpoint_resume_schema.resume_equivalence_byte_identical_to_uninterrupted_execution': True,
    'checkpoint_resume_schema.skip_duplicate_or_reorder_forbidden': True,
    'checkpoint_resume_schema.source_manifest_prefix_must_match_first_completed_ids': True,
    'checkpoint_resume_schema.stage_a_stage_b_have_distinct_completion_states': True,
    'checkpoint_resume_schema.stage_b_cannot_start_until_stage_a_complete_and_sealed': True,
    'checkpoint_resume_schema.stage_b_checkpoint_binds_stage_a_final_digest': True,
    'checkpoint_resume_schema.writer_and_execution_driver_not_implemented_by_this_task': True,
    'checkpoint_resume_schema.zero_progress_arrays_valid_without_np_stack_of_empty': True,
    'identity.description': 'Storage-only artifact-schema authority for DFP Phase 1B: binds the frozen e15-dfp-phase1b-protocol identity, then defines the Stage-A GT-free provenance-feature artifact schema, the Stage-B evaluation-label artifact schema (binding the exact Stage-A digest), the fixed prediction-variant confusion-array naming convention (semantic and Boundary IoU), and the checkpoint/resume invariants for a future (not-yet-implemented) execution driver. Implements no evaluator, no inference, no masked propagation, and defines no scientific quantity.',
    'identity.name': 'e15-dfp-phase1b-artifacts',
    'identity.schema_name': 'talk2dino-dfp-phase1b-artifacts-identity-v1',
    'identity.schema_version': 1,
    'identity.storage_representation_only': True,
    'path_policy.allowed_path_kinds': ['repository_relative_logical_path', 'safe_basename_no_directory_component'],
    'path_policy.canonical_image_id_distinct_from_physical_filename': True,
    'path_policy.forbidden': ['absolute_path', 'parent_traversal_dotdot', 'empty_path_component', 'forbidden_dot_component', 'drive_qualified_path', 'backslash_traversal', 'nul_byte', 'path_escaping_artifact_root_after_resolution', 'unexpected_symlink_escape', 'private_cluster_or_user_path'],
    'prediction_variants.count': 12,
    'prediction_variants.names': ['t0', 't16', 't32', 't64', 't320', 'oracle_o_p', 'persist_h32_full', 'persist_h32_harmful_only', 'persist_h32_beneficial_only', 'persist_h32_third_label_only', 't0_pamr', 't320_pamr'],
    'prediction_variants.original_ten_order_preserved': True,
    'prediction_variants.pamr_variants_appended_last': True,
    'protocol_binding.protocol_identity_relative_path': 'evaluation_identities/e15_dfp_phase1b_protocol.toml',
    'protocol_binding.protocol_identity_sha256': 'b3f165c9030200e8d988c65f2cc984804eb0ba7f60765d0193ff06c0aa897e7f',
    'semantic_confusion_schema.cross_variant_relation': 'all variants must have identical per-class ground_truth_pixels',
    'semantic_confusion_schema.dtype': 'int64',
    'semantic_confusion_schema.member_name_pattern': 'semantic_{variant}_{stat}',
    'semantic_confusion_schema.relation': 'union = predicted_pixels + ground_truth_pixels - intersection',
    'semantic_confusion_schema.shape': '[image_count, class_count]',
    'semantic_confusion_schema.stat_names': ['intersection', 'union', 'predicted_pixels', 'ground_truth_pixels'],
    'semantic_confusion_schema.total_member_count': 48,
    'semantic_confusion_schema.total_member_count_formula': '12 variants * 4 stats',
    'stage_a_schema.flat_disputed_pixel_members.conservation_residual': 'float64',
    'stage_a_schema.flat_disputed_pixel_members.eta_net': 'float64',
    'stage_a_schema.flat_disputed_pixel_members.final_pixel_linear_index': 'int64',
    'stage_a_schema.flat_disputed_pixel_members.image_row_index': 'int64',
    'stage_a_schema.flat_disputed_pixel_members.k_star': 'int64',
    'stage_a_schema.flat_disputed_pixel_members.m0': 'float64',
    'stage_a_schema.flat_disputed_pixel_members.m320': 'float64',
    'stage_a_schema.flat_disputed_pixel_members.margin_change': 'float64',
    'stage_a_schema.flat_disputed_pixel_members.n_positive_components': 'int64',
    'stage_a_schema.flat_disputed_pixel_members.negative_evidence_sum': 'float64',
    'stage_a_schema.flat_disputed_pixel_members.phi': 'float64',
    'stage_a_schema.flat_disputed_pixel_members.positive_evidence_sum': 'float64',
    'stage_a_schema.flat_disputed_pixel_members.propagated_class_b': 'int64',
    'stage_a_schema.flat_disputed_pixel_members.sum_component_evidence': 'float64',
    'stage_a_schema.flat_disputed_pixel_members.t16_label': 'int64',
    'stage_a_schema.flat_disputed_pixel_members.t32_label': 'int64',
    'stage_a_schema.flat_disputed_pixel_members.t64_label': 'int64',
    'stage_a_schema.flat_disputed_pixel_members.text_cosine_ab': 'float64',
    'stage_a_schema.flat_disputed_pixel_members.top1_positive_share': 'float64',
    'stage_a_schema.flat_disputed_pixel_members.top_positive_component_id': 'int64',
    'stage_a_schema.flat_disputed_pixel_members.top_positive_evidence': 'float64',
    'stage_a_schema.flat_disputed_pixel_members.unary_class_a': 'int64',
    'stage_a_schema.gt_derived_array_forbidden': True,
    'stage_a_schema.manifest_fields.canonical_image_ids': 'list[str], length image_count',
    'stage_a_schema.manifest_fields.checkpoint_prefix': 'str, safe basename, no directory component',
    'stage_a_schema.manifest_fields.complete': 'bool',
    'stage_a_schema.manifest_fields.component_count_per_image': 'int64[image_count]',
    'stage_a_schema.manifest_fields.dispute_count_per_image': 'int64[image_count]',
    'stage_a_schema.manifest_fields.dispute_offsets': 'int64[image_count + 1]',
    'stage_a_schema.manifest_fields.fold_assignment': 'int64[image_count], values in [0,4]',
    'stage_a_schema.manifest_fields.fold_assignment_digest': 'str, sha256 hex over the canonical (image_id, fold_id) sequence',
    'stage_a_schema.manifest_fields.image_count': 'int, must equal 500',
    'stage_a_schema.manifest_fields.per_image_max_conservation_residual': 'float64[image_count]',
    'stage_a_schema.manifest_fields.per_image_mean_conservation_residual': 'float64[image_count]',
    'stage_a_schema.manifest_fields.per_image_projected_compute_fields': 'object, mirrors gate1_analysis.json compute_projection per-image field names',
    'stage_a_schema.manifest_fields.selected_ids_digest': 'str, sha256 hex',
    'stage_a_schema.manifest_fields.total_processed_windows': 'int64 scalar',
    'stage_a_schema.manifest_fields.window_count_per_image': 'int64[image_count]',
    'stage_a_schema.relational_validation.conservation_tolerance_source': 'phase1b_conservation_calibration.json, bound by [conservation_calibration] in e15-dfp-phase1b-protocol; this schema never guesses a numeric tolerance',
    'stage_a_schema.relational_validation.list': ['offset[0] = 0', 'offset is nondecreasing', 'offset[-1] = flat row count', 'diff(offsets) = per-image dispute count', 'a != b', 'm0 <= 0', 'm320 > 0', 'margin_change = m320 - m0', 'eta_net = m320 / margin_change', 'positive_evidence_sum > 0', 'negative_evidence_sum <= 0', '1 <= k_star <= n_positive_components', '0 < phi <= 1', '0 < top1_positive_share <= 1', 'sum_component_evidence ~= margin_change (within the separately bound calibration tolerance, never a guessed constant)'],
    'stage_a_schema.schema_name': 'talk2dino-dfp-phase1b-stage-a-features-v1',
    'stage_a_schema.schema_version': 1,
    'stage_b_schema.binds_exact_stage_a_digest': True,
    'stage_b_schema.diagnostic_prediction_members_optional.description': 'Absent until all five folds are complete; a Stage-B artifact with fewer than five folds fitted must not include these members at all.',
    'stage_b_schema.diagnostic_prediction_members_optional.out_of_fold_cheap_control_risk': 'float64',
    'stage_b_schema.diagnostic_prediction_members_optional.out_of_fold_provenance_augmented_risk': 'float64',
    'stage_b_schema.manifest_fields.bootstrap_result_summary': 'object: sufficient statistics or arrays per [bootstrap] in e15-dfp-phase1b-protocol',
    'stage_b_schema.manifest_fields.fixed_single_feature_risk_orientations': "object: {feature_name: '+' or '-'} for neg_phi, neg_k_star, top1_positive_share, matching [gate2_criteria.mandatory_conditions].c2_single_feature_candidates",
    'stage_b_schema.manifest_fields.per_fold_counts': 'object: harmful/beneficial/third_label/binary_eligible counts per fold',
    'stage_b_schema.manifest_fields.per_image_target_counts': 'object: harmful/beneficial/third_label counts per image',
    'stage_b_schema.per_pixel_members.binary_eligible': 'int64, 0 or 1; 1 exactly when target_category in {0,1}',
    'stage_b_schema.per_pixel_members.boundary_interior_flag': 'int64, 0 (interior) or 1 (boundary)',
    'stage_b_schema.per_pixel_members.fold_id': 'int64, in [0,4]; must equal the image-level frozen fold_assignment for the same image',
    'stage_b_schema.per_pixel_members.gt_class': 'int64',
    'stage_b_schema.per_pixel_members.harmful_binary_label': 'int64, 1 for harmful, 0 for beneficial, -1 sentinel (not-binary-eligible) for third_label -- never 0 or 1 for third_label',
    'stage_b_schema.per_pixel_members.target_category': 'int64, one of [0,1,2] per [stage_b_schema.target_category_codes]',
    'stage_b_schema.per_pixel_members.training_selection_flag': 'int64, 0 or 1; set by [training_subsampling] in e15-dfp-phase1b-protocol',
    'stage_b_schema.relational_validation.list': ['harmful + beneficial + third_label = disputes', 'binary_eligible = harmful + beneficial', 'third_label is never assigned harmful_binary_label 0 or 1', 'fold_id agrees with the image-level frozen fold'],
    'stage_b_schema.schema_name': 'talk2dino-dfp-phase1b-stage-b-labels-v1',
    'stage_b_schema.schema_version': 1,
    'stage_b_schema.target_category_codes.beneficial': 1,
    'stage_b_schema.target_category_codes.harmful': 0,
    'stage_b_schema.target_category_codes.third_label': 2,
}

TOP_KEYS = frozenset(['atomic_write_policy', 'boundary_confusion_schema', 'checkpoint_resume_schema', 'format_version', 'identity', 'path_policy', 'prediction_variants', 'protocol_binding', 'semantic_confusion_schema', 'stage_a_schema', 'stage_b_schema'])

SECTION_TREE: dict[str, frozenset[str]] = {
    'identity': frozenset(['description', 'name', 'schema_name', 'schema_version', 'storage_representation_only']),
    'protocol_binding': frozenset(['protocol_identity_relative_path', 'protocol_identity_sha256']),
    'stage_a_schema': frozenset(['flat_disputed_pixel_members', 'gt_derived_array_forbidden', 'manifest_fields', 'relational_validation', 'schema_name', 'schema_version']),
    'stage_a_schema.manifest_fields': frozenset(['canonical_image_ids', 'checkpoint_prefix', 'complete', 'component_count_per_image', 'dispute_count_per_image', 'dispute_offsets', 'fold_assignment', 'fold_assignment_digest', 'image_count', 'per_image_max_conservation_residual', 'per_image_mean_conservation_residual', 'per_image_projected_compute_fields', 'selected_ids_digest', 'total_processed_windows', 'window_count_per_image']),
    'stage_a_schema.flat_disputed_pixel_members': frozenset(['conservation_residual', 'eta_net', 'final_pixel_linear_index', 'image_row_index', 'k_star', 'm0', 'm320', 'margin_change', 'n_positive_components', 'negative_evidence_sum', 'phi', 'positive_evidence_sum', 'propagated_class_b', 'sum_component_evidence', 't16_label', 't32_label', 't64_label', 'text_cosine_ab', 'top1_positive_share', 'top_positive_component_id', 'top_positive_evidence', 'unary_class_a']),
    'stage_a_schema.relational_validation': frozenset(['conservation_tolerance_source', 'list']),
    'stage_b_schema': frozenset(['binds_exact_stage_a_digest', 'diagnostic_prediction_members_optional', 'manifest_fields', 'per_pixel_members', 'relational_validation', 'schema_name', 'schema_version', 'target_category_codes']),
    'stage_b_schema.target_category_codes': frozenset(['beneficial', 'harmful', 'third_label']),
    'stage_b_schema.per_pixel_members': frozenset(['binary_eligible', 'boundary_interior_flag', 'fold_id', 'gt_class', 'harmful_binary_label', 'target_category', 'training_selection_flag']),
    'stage_b_schema.diagnostic_prediction_members_optional': frozenset(['description', 'out_of_fold_cheap_control_risk', 'out_of_fold_provenance_augmented_risk']),
    'stage_b_schema.manifest_fields': frozenset(['bootstrap_result_summary', 'fixed_single_feature_risk_orientations', 'per_fold_counts', 'per_image_target_counts']),
    'stage_b_schema.relational_validation': frozenset(['list']),
    'prediction_variants': frozenset(['count', 'names', 'original_ten_order_preserved', 'pamr_variants_appended_last']),
    'semantic_confusion_schema': frozenset(['cross_variant_relation', 'dtype', 'member_name_pattern', 'relation', 'shape', 'stat_names', 'total_member_count', 'total_member_count_formula']),
    'boundary_confusion_schema': frozenset(['authority', 'cross_variant_relation', 'dtype', 'member_name_pattern', 'relation', 'shape', 'stat_names', 'total_member_count', 'total_member_count_formula']),
    'checkpoint_resume_schema': frozenset(['append_only_generation', 'atomic_pointer_update', 'complete_final_boolean_combinations_exact', 'each_installed_generation_self_verifies_before_pointer_replacement', 'final_result_installation_cpu_only_no_model_rebuild', 'orphan_generation_after_handled_failure_forbidden', 'reconcile_ambiguous_post_replace_failure_before_deleting_files', 'resume_equivalence_byte_identical_to_uninterrupted_execution', 'skip_duplicate_or_reorder_forbidden', 'source_manifest_prefix_must_match_first_completed_ids', 'stage_a_stage_b_have_distinct_completion_states', 'stage_b_cannot_start_until_stage_a_complete_and_sealed', 'stage_b_checkpoint_binds_stage_a_final_digest', 'writer_and_execution_driver_not_implemented_by_this_task', 'zero_progress_arrays_valid_without_np_stack_of_empty']),
    'path_policy': frozenset(['allowed_path_kinds', 'canonical_image_id_distinct_from_physical_filename', 'forbidden']),
    'atomic_write_policy': frozenset(['checkpoint_install', 'result_install']),
}

# ---------------------------------------------------------------------
# TOML identity loader (same generic closed-schema + exact-value
# technique as src.dfp_phase1b_identity).
# ---------------------------------------------------------------------


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str:
        raise DfpPhase1bArtifactSchemaError(f"{label} must be an exact string")
    if nonempty and not value.strip():
        raise DfpPhase1bArtifactSchemaError(f"{label} must be a non-empty, non-whitespace string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise DfpPhase1bArtifactSchemaError(f"{label} must be a lowercase 64-hex-character SHA-256")
    return token


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_equal(observed: Any, canonical: Any) -> bool:
    if type(observed) is not type(canonical):
        return False
    if isinstance(canonical, float):
        if not math.isfinite(observed):
            return False
        return observed == canonical
    if isinstance(canonical, list):
        if len(observed) != len(canonical):
            return False
        return all(_exact_equal(o, c) for o, c in zip(observed, canonical))
    return observed == canonical


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DfpPhase1bArtifactSchemaError(f"{label} must be a mapping")
    if set(value) != expected_keys:
        missing = expected_keys - set(value)
        unknown = set(value) - expected_keys
        raise DfpPhase1bArtifactSchemaError(f"{label} has an unexpected schema (missing={sorted(missing)}, unknown={sorted(unknown)})")
    return value


def _validate_node(node: Mapping[str, Any], path: str) -> None:
    expected_keys = SECTION_TREE[path] if path else TOP_KEYS
    _require_closed_mapping(node, expected_keys, f"identity.{path}" if path else "identity")
    for key, value in node.items():
        full_path = f"{path}.{key}" if path else key
        if full_path in SECTION_TREE:
            _validate_node(value, full_path)
        else:
            if full_path not in CANONICAL_VALUES:
                raise DfpPhase1bArtifactSchemaError(f"{full_path} has no registered canonical value or sub-schema -- refusing to validate an unrecognized leaf")
            canonical = CANONICAL_VALUES[full_path]
            if not _exact_equal(value, canonical):
                raise DfpPhase1bArtifactSchemaError(f"{full_path} disagrees with the frozen canonical value (observed {value!r})")
            if key == "sha256":
                _require_sha256(value, full_path)


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DfpPhase1bArtifactSchemaError(f"cannot load Phase 1B artifact-schema identity {source}: {error}") from error

    if set(identity) != TOP_KEYS:
        missing = TOP_KEYS - set(identity)
        unknown = set(identity) - TOP_KEYS
        raise DfpPhase1bArtifactSchemaError(f"identity has an unexpected top-level schema (missing={sorted(missing)}, unknown={sorted(unknown)})")

    format_version = identity.get("format_version")
    if format_version != FORMAT_VERSION:
        raise DfpPhase1bArtifactSchemaError(f"format_version disagrees with the frozen canonical value (observed {format_version!r})")

    for section in TOP_KEYS - {"format_version"}:
        _validate_node(identity[section], section)

    return identity


def verify_artifact_schema_bindings(*, repo_root: Path | None = None, identity_path: Path | None = None) -> str:
    """Loads AND relationally re-verifies the bound e15-dfp-phase1b-
    protocol identity's SHA-256 against the live file on disk. Never
    trusts an in-memory dict -- always re-reads from `repo_root`."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    binding = identity["protocol_binding"]
    protocol_path = root / binding["protocol_identity_relative_path"]
    if not protocol_path.is_file():
        raise DfpPhase1bArtifactSchemaError(f"bound Phase 1B protocol identity does not exist: {protocol_path}")
    observed = sha256_file(protocol_path)
    if observed != binding["protocol_identity_sha256"]:
        raise DfpPhase1bArtifactSchemaError(
            f"protocol_binding SHA256 mismatch: identity declares {binding['protocol_identity_sha256']}, observed {observed}"
        )
    return observed


# ---------------------------------------------------------------------
# Stage A / Stage B / prediction-variant member-name schemas.
# ---------------------------------------------------------------------

STAGE_A_SCHEMA_NAME = "talk2dino-dfp-phase1b-stage-a-features-v1"
STAGE_A_SCHEMA_VERSION = 1
STAGE_B_SCHEMA_NAME = "talk2dino-dfp-phase1b-stage-b-labels-v1"
STAGE_B_SCHEMA_VERSION = 1

STAGE_A_INT64_MEMBER_NAMES: tuple[str, ...] = (
    "image_row_index", "final_pixel_linear_index", "unary_class_a", "propagated_class_b",
    "t16_label", "t32_label", "t64_label", "n_positive_components", "k_star",
    "top_positive_component_id",
)
STAGE_A_FLOAT64_MEMBER_NAMES: tuple[str, ...] = (
    "m0", "m320", "margin_change", "eta_net", "text_cosine_ab", "positive_evidence_sum",
    "negative_evidence_sum", "sum_component_evidence", "conservation_residual", "phi",
    "top_positive_evidence", "top1_positive_share",
)
STAGE_A_FLAT_MEMBER_NAMES: frozenset[str] = frozenset(STAGE_A_INT64_MEMBER_NAMES) | frozenset(STAGE_A_FLOAT64_MEMBER_NAMES)

# Explicit, defense-in-depth denylist: no name here may ever appear in a
# Stage-A member set, even if some future edit to the allow-list above
# accidentally introduced one (belt-and-suspenders on top of exact
# closed-set membership checking).
STAGE_A_FORBIDDEN_GT_NAME_FRAGMENTS: tuple[str, ...] = (
    "gt", "ground_truth", "harmful", "beneficial", "third_label", "oracle", "boundary_label",
)

STAGE_B_INT64_PER_PIXEL_MEMBER_NAMES: tuple[str, ...] = (
    "gt_class", "target_category", "binary_eligible", "harmful_binary_label",
    "boundary_interior_flag", "fold_id", "training_selection_flag",
)
STAGE_B_OPTIONAL_FLOAT64_MEMBER_NAMES: tuple[str, ...] = (
    "out_of_fold_cheap_control_risk", "out_of_fold_provenance_augmented_risk",
)

TARGET_CATEGORY_HARMFUL = 0
TARGET_CATEGORY_BENEFICIAL = 1
TARGET_CATEGORY_THIRD_LABEL = 2
TARGET_CATEGORY_CODES: frozenset[int] = frozenset({TARGET_CATEGORY_HARMFUL, TARGET_CATEGORY_BENEFICIAL, TARGET_CATEGORY_THIRD_LABEL})
HARMFUL_BINARY_LABEL_NOT_ELIGIBLE_SENTINEL = -1

PREDICTION_VARIANT_NAMES: tuple[str, ...] = (
    "t0", "t16", "t32", "t64", "t320", "oracle_o_p",
    "persist_h32_full", "persist_h32_harmful_only", "persist_h32_beneficial_only", "persist_h32_third_label_only",
    "t0_pamr", "t320_pamr",
)
# The original ten variants (frozen order, never renamed/reordered) plus
# exactly two PAMR baseline variants appended last -- see
# [rgb_local_boundary_baseline] / [pamr_variants] in
# e15-dfp-phase1b-protocol. PAMR is bound (not GT-free provenance): these
# two variants carry only fixed RGB/local-boundary refinement statistics,
# never provenance features, never disputed-pixel selection.
ORIGINAL_TEN_PREDICTION_VARIANT_NAMES: tuple[str, ...] = PREDICTION_VARIANT_NAMES[:10]
PAMR_PREDICTION_VARIANT_NAMES: tuple[str, ...] = PREDICTION_VARIANT_NAMES[10:]
SEMANTIC_STAT_NAMES: tuple[str, ...] = ("intersection", "union", "predicted_pixels", "ground_truth_pixels")
BOUNDARY_STAT_NAMES: tuple[str, ...] = (
    "boundary_intersection", "boundary_union", "predicted_boundary_pixels", "ground_truth_boundary_pixels",
)


def semantic_member_name(variant: str, stat: str) -> str:
    if variant not in PREDICTION_VARIANT_NAMES:
        raise DfpPhase1bArtifactSchemaError(f"unknown prediction variant {variant!r}")
    if stat not in SEMANTIC_STAT_NAMES:
        raise DfpPhase1bArtifactSchemaError(f"unknown semantic stat {stat!r}")
    return f"semantic_{variant}_{stat}"


def boundary_member_name(variant: str, stat: str) -> str:
    if variant not in PREDICTION_VARIANT_NAMES:
        raise DfpPhase1bArtifactSchemaError(f"unknown prediction variant {variant!r}")
    if stat not in BOUNDARY_STAT_NAMES:
        raise DfpPhase1bArtifactSchemaError(f"unknown boundary stat {stat!r}")
    return f"boundary_{variant}_{stat}"


ALL_SEMANTIC_MEMBER_NAMES: frozenset[str] = frozenset(
    semantic_member_name(v, s) for v in PREDICTION_VARIANT_NAMES for s in SEMANTIC_STAT_NAMES
)
ALL_BOUNDARY_MEMBER_NAMES: frozenset[str] = frozenset(
    boundary_member_name(v, s) for v in PREDICTION_VARIANT_NAMES for s in BOUNDARY_STAT_NAMES
)

# ---------------------------------------------------------------------
# Generic array-shape/dtype validation (pure structural validation --
# no scientific computation).
# ---------------------------------------------------------------------


def _require_native_dtype_array(array: Any, name: str, expected_dtype: np.dtype) -> None:
    if not isinstance(array, np.ndarray):
        raise DfpPhase1bArtifactSchemaError(f"{name!r} must be a numpy.ndarray")
    if array.dtype.hasobject:
        raise DfpPhase1bArtifactSchemaError(f"{name!r} must not have an object dtype")
    if array.dtype.names is not None:
        raise DfpPhase1bArtifactSchemaError(f"{name!r} must not have a structured dtype")
    if array.dtype.byteorder == ">":
        raise DfpPhase1bArtifactSchemaError(f"{name!r} must use native (non-big-endian) byte order")
    if array.dtype != expected_dtype:
        raise DfpPhase1bArtifactSchemaError(f"{name!r} must have exact dtype {expected_dtype}, observed {array.dtype}")


def _check_zip_member_hygiene(npz_path: Path) -> None:
    """Inspect the NPZ as a raw ZIP container before trusting numpy's
    own loader: reject duplicate member names, directory entries, and
    path-traversal member names."""
    try:
        with zipfile.ZipFile(npz_path) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names):
                raise DfpPhase1bArtifactSchemaError("NPZ contains a duplicate ZIP member name")
            for name in names:
                if name.endswith("/"):
                    raise DfpPhase1bArtifactSchemaError(f"NPZ contains a directory entry {name!r}")
                if ".." in Path(name).parts or Path(name).is_absolute():
                    raise DfpPhase1bArtifactSchemaError(f"NPZ member name {name!r} is unsafe (path traversal)")
    except zipfile.BadZipFile as error:
        raise DfpPhase1bArtifactSchemaError(f"NPZ {npz_path} is not a valid ZIP archive: {error}") from error


# ---------------------------------------------------------------------
# Stage A validation.
# ---------------------------------------------------------------------


def validate_stage_a_member_names(member_names: Any) -> None:
    observed = set(member_names)
    if observed != STAGE_A_FLAT_MEMBER_NAMES:
        missing = STAGE_A_FLAT_MEMBER_NAMES - observed
        unknown = observed - STAGE_A_FLAT_MEMBER_NAMES
        raise DfpPhase1bArtifactSchemaError(f"Stage-A flat member set mismatch (missing={sorted(missing)}, unknown={sorted(unknown)})")
    for name in observed:
        lowered = name.lower()
        if any(fragment in lowered for fragment in STAGE_A_FORBIDDEN_GT_NAME_FRAGMENTS):
            raise DfpPhase1bArtifactSchemaError(f"Stage-A member name {name!r} contains a forbidden GT-shaped fragment")


def validate_stage_a_flat_arrays(arrays: Mapping[str, np.ndarray], *, row_count: int, conservation_tolerance: float) -> None:
    """Pure structural/relational validation of an already-populated
    Stage-A flat-array mapping. Never computes a feature; only checks
    that already-given values satisfy the frozen contract.
    `conservation_tolerance` MUST be supplied explicitly by the caller
    from the separately bound `phase1b_conservation_calibration.json`
    artifact -- this function has no default and never guesses one."""
    validate_stage_a_member_names(arrays.keys())
    if type(conservation_tolerance) is not float or not math.isfinite(conservation_tolerance) or conservation_tolerance < 0:
        raise DfpPhase1bArtifactSchemaError("conservation_tolerance must be an explicit non-negative finite float from the bound calibration artifact")
    if type(row_count) is not int or row_count < 0:
        raise DfpPhase1bArtifactSchemaError("row_count must be a non-negative exact int")

    for name in STAGE_A_INT64_MEMBER_NAMES:
        _require_native_dtype_array(arrays[name], name, np.dtype(np.int64))
    for name in STAGE_A_FLOAT64_MEMBER_NAMES:
        arr = arrays[name]
        _require_native_dtype_array(arr, name, np.dtype(np.float64))
        if arr.size and not np.isfinite(arr).all():
            raise DfpPhase1bArtifactSchemaError(f"{name!r} contains a non-finite value (NaN or Infinity)")

    for name in STAGE_A_FLAT_MEMBER_NAMES:
        arr = arrays[name]
        if arr.ndim != 1:
            raise DfpPhase1bArtifactSchemaError(f"{name!r} must be exactly 1-dimensional, observed ndim={arr.ndim}")
        if arr.shape[0] != row_count:
            raise DfpPhase1bArtifactSchemaError(f"{name!r} has {arr.shape[0]} rows, expected exactly row_count={row_count}")

    if row_count == 0:
        return

    a = arrays["unary_class_a"]
    b = arrays["propagated_class_b"]
    if np.any(a == b):
        raise DfpPhase1bArtifactSchemaError("unary_class_a equals propagated_class_b for at least one disputed pixel (a != b violated)")

    m0 = arrays["m0"]
    m320 = arrays["m320"]
    if np.any(m0 > 0):
        raise DfpPhase1bArtifactSchemaError("m0 > 0 for at least one disputed pixel (m0 <= 0 violated)")
    if np.any(m320 <= 0):
        raise DfpPhase1bArtifactSchemaError("m320 <= 0 for at least one disputed pixel (m320 > 0 violated)")

    margin_change = arrays["margin_change"]
    if not np.allclose(margin_change, m320 - m0, rtol=0.0, atol=0.0):
        raise DfpPhase1bArtifactSchemaError("margin_change != m320 - m0 for at least one disputed pixel")

    eta_net = arrays["eta_net"]
    if not np.allclose(eta_net, m320 / margin_change, rtol=0.0, atol=0.0):
        raise DfpPhase1bArtifactSchemaError("eta_net != m320 / margin_change for at least one disputed pixel")

    pos = arrays["positive_evidence_sum"]
    neg = arrays["negative_evidence_sum"]
    if np.any(pos <= 0):
        raise DfpPhase1bArtifactSchemaError("positive_evidence_sum <= 0 for at least one disputed pixel")
    if np.any(neg > 0):
        raise DfpPhase1bArtifactSchemaError("negative_evidence_sum > 0 for at least one disputed pixel")

    k_star = arrays["k_star"]
    n_pos = arrays["n_positive_components"]
    if np.any(k_star < 1) or np.any(k_star > n_pos):
        raise DfpPhase1bArtifactSchemaError("k_star out of range [1, n_positive_components] for at least one disputed pixel")

    phi = arrays["phi"]
    if np.any(phi <= 0) or np.any(phi > 1):
        raise DfpPhase1bArtifactSchemaError("phi out of range (0, 1] for at least one disputed pixel")

    top1_share = arrays["top1_positive_share"]
    if np.any(top1_share <= 0) or np.any(top1_share > 1):
        raise DfpPhase1bArtifactSchemaError("top1_positive_share out of range (0, 1] for at least one disputed pixel")

    sum_component_evidence = arrays["sum_component_evidence"]
    residual = np.abs(sum_component_evidence - margin_change)
    if np.any(residual > conservation_tolerance):
        worst = float(residual.max())
        raise DfpPhase1bArtifactSchemaError(
            f"sum_component_evidence disagrees with margin_change by up to {worst!r}, exceeding the bound calibration tolerance {conservation_tolerance!r}"
        )
    # conservation_residual, if present, must reflect the same quantity
    # this function just independently recomputed.
    stored_residual = arrays["conservation_residual"]
    if not np.allclose(stored_residual, residual, rtol=0.0, atol=1e-12):
        raise DfpPhase1bArtifactSchemaError("stored conservation_residual disagrees with |sum_component_evidence - margin_change|")


def validate_stage_a_offsets(offsets: np.ndarray, dispute_count_per_image: np.ndarray, *, flat_row_count: int) -> None:
    _require_native_dtype_array(offsets, "dispute_offsets", np.dtype(np.int64))
    _require_native_dtype_array(dispute_count_per_image, "dispute_count_per_image", np.dtype(np.int64))
    if offsets.ndim != 1 or offsets.shape[0] != dispute_count_per_image.shape[0] + 1:
        raise DfpPhase1bArtifactSchemaError("dispute_offsets must have length image_count + 1")
    if offsets.size and offsets[0] != 0:
        raise DfpPhase1bArtifactSchemaError("dispute_offsets[0] must be exactly 0")
    if offsets.size and np.any(np.diff(offsets) < 0):
        raise DfpPhase1bArtifactSchemaError("dispute_offsets must be nondecreasing")
    if offsets.size and offsets[-1] != flat_row_count:
        raise DfpPhase1bArtifactSchemaError("dispute_offsets[-1] must equal the flat disputed-pixel row count")
    if offsets.size > 1 and not np.array_equal(np.diff(offsets), dispute_count_per_image):
        raise DfpPhase1bArtifactSchemaError("diff(dispute_offsets) does not equal dispute_count_per_image")


# ---------------------------------------------------------------------
# Stage B validation.
# ---------------------------------------------------------------------


def validate_stage_b_member_names(member_names: Any, *, diagnostic_predictions_present: bool) -> None:
    observed = set(member_names)
    required = set(STAGE_B_INT64_PER_PIXEL_MEMBER_NAMES)
    if diagnostic_predictions_present:
        expected = required | set(STAGE_B_OPTIONAL_FLOAT64_MEMBER_NAMES)
        if observed != expected:
            raise DfpPhase1bArtifactSchemaError(
                f"Stage-B member set mismatch with diagnostic predictions present: missing={sorted(expected - observed)}, unknown={sorted(observed - expected)}"
            )
    else:
        if observed != required:
            unknown_diagnostic = observed & set(STAGE_B_OPTIONAL_FLOAT64_MEMBER_NAMES)
            if unknown_diagnostic:
                raise DfpPhase1bArtifactSchemaError(
                    f"Stage-B diagnostic prediction members {sorted(unknown_diagnostic)} present before all five folds are complete"
                )
            raise DfpPhase1bArtifactSchemaError(
                f"Stage-B member set mismatch: missing={sorted(required - observed)}, unknown={sorted(observed - required)}"
            )


def validate_stage_b_per_pixel_arrays(
    arrays: Mapping[str, np.ndarray], *, row_count: int, image_fold_assignment: np.ndarray, image_row_index: np.ndarray,
) -> None:
    """Pure structural/relational validation. `image_fold_assignment` is
    the frozen per-image fold array (index = image_row_index value);
    `image_row_index` is Stage-A's own array, supplied by the caller so
    this function never re-derives it."""
    for name in STAGE_B_INT64_PER_PIXEL_MEMBER_NAMES:
        arr = arrays[name]
        _require_native_dtype_array(arr, name, np.dtype(np.int64))
        if arr.ndim != 1 or arr.shape[0] != row_count:
            raise DfpPhase1bArtifactSchemaError(f"{name!r} must be 1-D with length row_count={row_count}")

    if row_count == 0:
        return

    target_category = arrays["target_category"]
    if np.any(~np.isin(target_category, list(TARGET_CATEGORY_CODES))):
        raise DfpPhase1bArtifactSchemaError(f"target_category contains a value outside {sorted(TARGET_CATEGORY_CODES)}")

    binary_eligible = arrays["binary_eligible"]
    expected_binary_eligible = np.isin(target_category, [TARGET_CATEGORY_HARMFUL, TARGET_CATEGORY_BENEFICIAL]).astype(np.int64)
    if not np.array_equal(binary_eligible, expected_binary_eligible):
        raise DfpPhase1bArtifactSchemaError("binary_eligible disagrees with target_category in {harmful, beneficial}")

    harmful_binary_label = arrays["harmful_binary_label"]
    third_label_mask = target_category == TARGET_CATEGORY_THIRD_LABEL
    if np.any(harmful_binary_label[third_label_mask] != HARMFUL_BINARY_LABEL_NOT_ELIGIBLE_SENTINEL):
        raise DfpPhase1bArtifactSchemaError("a third_label pixel was assigned a harmful_binary_label other than the not-eligible sentinel")
    harmful_mask = target_category == TARGET_CATEGORY_HARMFUL
    beneficial_mask = target_category == TARGET_CATEGORY_BENEFICIAL
    if np.any(harmful_binary_label[harmful_mask] != 1):
        raise DfpPhase1bArtifactSchemaError("a harmful pixel was not assigned harmful_binary_label == 1")
    if np.any(harmful_binary_label[beneficial_mask] != 0):
        raise DfpPhase1bArtifactSchemaError("a beneficial pixel was not assigned harmful_binary_label == 0")

    boundary_interior_flag = arrays["boundary_interior_flag"]
    if np.any(~np.isin(boundary_interior_flag, [0, 1])):
        raise DfpPhase1bArtifactSchemaError("boundary_interior_flag contains a value other than 0 or 1")

    training_selection_flag = arrays["training_selection_flag"]
    if np.any(~np.isin(training_selection_flag, [0, 1])):
        raise DfpPhase1bArtifactSchemaError("training_selection_flag contains a value other than 0 or 1")

    fold_id = arrays["fold_id"]
    if np.any((fold_id < 0) | (fold_id > 4)):
        raise DfpPhase1bArtifactSchemaError("fold_id contains a value outside [0,4]")
    _require_native_dtype_array(image_fold_assignment, "image_fold_assignment", np.dtype(np.int64))
    _require_native_dtype_array(image_row_index, "image_row_index", np.dtype(np.int64))
    if image_row_index.shape[0] != row_count:
        raise DfpPhase1bArtifactSchemaError("image_row_index length disagrees with row_count")
    expected_fold_id = image_fold_assignment[image_row_index]
    if not np.array_equal(fold_id, expected_fold_id):
        raise DfpPhase1bArtifactSchemaError("fold_id disagrees with the image-level frozen fold_assignment for at least one pixel")


def validate_stage_b_target_category_counts(target_category: np.ndarray, *, disputes_total: int) -> None:
    harmful = int(np.count_nonzero(target_category == TARGET_CATEGORY_HARMFUL))
    beneficial = int(np.count_nonzero(target_category == TARGET_CATEGORY_BENEFICIAL))
    third_label = int(np.count_nonzero(target_category == TARGET_CATEGORY_THIRD_LABEL))
    if harmful + beneficial + third_label != disputes_total:
        raise DfpPhase1bArtifactSchemaError(
            f"harmful({harmful}) + beneficial({beneficial}) + third_label({third_label}) != disputes({disputes_total})"
        )


def validate_stage_b_digest_binding(observed_stage_a_digest: str, declared_stage_a_digest: str) -> None:
    if observed_stage_a_digest != declared_stage_a_digest:
        raise DfpPhase1bArtifactSchemaError(
            f"Stage-B declares stage_a_digest={declared_stage_a_digest!r}, observed Stage-A artifact digest={observed_stage_a_digest!r}"
        )


# ---------------------------------------------------------------------
# Prediction-variant confusion-array validation (semantic + Boundary
# IoU), shared shape/relation logic.
# ---------------------------------------------------------------------


def validate_confusion_arrays(
    arrays: Mapping[str, np.ndarray], *, stat_names: tuple[str, ...], member_name_fn, image_count: int, class_count: int,
) -> None:
    if type(image_count) is not int or image_count <= 0:
        raise DfpPhase1bArtifactSchemaError("image_count must be a positive exact int")
    if type(class_count) is not int or class_count <= 0:
        raise DfpPhase1bArtifactSchemaError("class_count must be a positive exact int")

    expected_members = {member_name_fn(variant, stat) for variant in PREDICTION_VARIANT_NAMES for stat in stat_names}
    observed_members = set(arrays)
    if observed_members != expected_members:
        missing = sorted(expected_members - observed_members)
        unexpected = sorted(observed_members - expected_members)
        raise DfpPhase1bArtifactSchemaError(
            f"confusion array member set is not exactly the {len(expected_members)} frozen members "
            f"(missing={missing}, unexpected={unexpected})"
        )

    gt_by_variant: dict[str, np.ndarray] = {}
    for variant in PREDICTION_VARIANT_NAMES:
        per_stat: dict[str, np.ndarray] = {}
        for stat in stat_names:
            name = member_name_fn(variant, stat)
            if name not in arrays:
                raise DfpPhase1bArtifactSchemaError(f"missing confusion member {name!r}")
            arr = arrays[name]
            _require_native_dtype_array(arr, name, np.dtype(np.int64))
            if arr.shape != (image_count, class_count):
                raise DfpPhase1bArtifactSchemaError(f"{name!r} has shape {arr.shape}, expected ({image_count}, {class_count})")
            if arr.size and arr.min() < 0:
                raise DfpPhase1bArtifactSchemaError(f"{name!r} must be entirely non-negative")
            per_stat[stat] = arr

        intersection = per_stat[stat_names[0]]
        union = per_stat[stat_names[1]]
        predicted = per_stat[stat_names[2]]
        ground_truth = per_stat[stat_names[3]]
        if not np.array_equal(union, predicted + ground_truth - intersection):
            raise DfpPhase1bArtifactSchemaError(f"{variant!r}: union != predicted + ground_truth - intersection")
        gt_by_variant[variant] = ground_truth

    first_variant = PREDICTION_VARIANT_NAMES[0]
    reference_gt = gt_by_variant[first_variant]
    for variant in PREDICTION_VARIANT_NAMES[1:]:
        if not np.array_equal(gt_by_variant[variant], reference_gt):
            raise DfpPhase1bArtifactSchemaError(
                f"variant {variant!r} ground-truth counts disagree with {first_variant!r} -- all variants must share identical per-class ground-truth counts"
            )


def validate_semantic_confusion_arrays(arrays: Mapping[str, np.ndarray], *, image_count: int, class_count: int) -> None:
    validate_confusion_arrays(arrays, stat_names=SEMANTIC_STAT_NAMES, member_name_fn=semantic_member_name, image_count=image_count, class_count=class_count)


def validate_boundary_confusion_arrays(arrays: Mapping[str, np.ndarray], *, image_count: int, class_count: int) -> None:
    validate_confusion_arrays(arrays, stat_names=BOUNDARY_STAT_NAMES, member_name_fn=boundary_member_name, image_count=image_count, class_count=class_count)


__all__ = [
    "IDENTITY_RELATIVE_PATH", "FORMAT_VERSION", "TOP_KEYS", "SECTION_TREE", "CANONICAL_VALUES",
    "DfpPhase1bArtifactSchemaError", "repository_root", "sha256_file", "load_identity",
    "verify_artifact_schema_bindings",
    "STAGE_A_SCHEMA_NAME", "STAGE_A_SCHEMA_VERSION", "STAGE_B_SCHEMA_NAME", "STAGE_B_SCHEMA_VERSION",
    "STAGE_A_INT64_MEMBER_NAMES", "STAGE_A_FLOAT64_MEMBER_NAMES", "STAGE_A_FLAT_MEMBER_NAMES",
    "STAGE_B_INT64_PER_PIXEL_MEMBER_NAMES", "STAGE_B_OPTIONAL_FLOAT64_MEMBER_NAMES",
    "TARGET_CATEGORY_HARMFUL", "TARGET_CATEGORY_BENEFICIAL", "TARGET_CATEGORY_THIRD_LABEL", "TARGET_CATEGORY_CODES",
    "HARMFUL_BINARY_LABEL_NOT_ELIGIBLE_SENTINEL",
    "PREDICTION_VARIANT_NAMES", "ORIGINAL_TEN_PREDICTION_VARIANT_NAMES", "PAMR_PREDICTION_VARIANT_NAMES",
    "SEMANTIC_STAT_NAMES", "BOUNDARY_STAT_NAMES",
    "semantic_member_name", "boundary_member_name", "ALL_SEMANTIC_MEMBER_NAMES", "ALL_BOUNDARY_MEMBER_NAMES",
    "validate_stage_a_member_names", "validate_stage_a_flat_arrays", "validate_stage_a_offsets",
    "validate_stage_b_member_names", "validate_stage_b_per_pixel_arrays", "validate_stage_b_target_category_counts",
    "validate_stage_b_digest_binding",
    "validate_confusion_arrays", "validate_semantic_confusion_arrays", "validate_boundary_confusion_arrays",
]
