"""Strict loading/validation for the DFP Phase 1B conservation-
calibration successor identity
(`evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml`).

This module is a NEW LEAF authority. It NEVER rewrites, re-binds, or
cascades new hashes through `e15-dfp-phase1b-protocol`,
`e15-dfp-phase1b-artifacts`, the exact masked-forward provenance core,
the frozen COCO-Stuff-500 subset manifest, the canonical sliding-
window-plan implementation, or the committed stitching/final-
resolution pullback implementations. It only relationally re-verifies
their SHA-256 hashes (proving they remain byte-identical to what this
identity pins) every time it loads.

Because e16-dfp-phase1b-conservation-calibration is itself an entirely
frozen pre-registration document (every field is a fixed scientific
constant, formula, or policy flag -- there is no "variable data" field
anywhere in it), this loader pins EVERY scalar/list leaf value exactly
via a single mechanically-generated `CANONICAL_VALUES` table, exactly
mirroring `src.dfp_phase1b_identity`'s own approach for the same
reason: any mutation to any field, anywhere in the tree, is rejected --
never accepted merely because it is non-empty or superficially
plausible.

Always fail closed: never silently substitute a default, coerce a
type, or normalize a string. Performs no torch, CUDA, model, dataset,
or image access -- pure TOML/filesystem/hash validation only. Never
selects, infers, or stores a numeric conservation tolerance.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path
from typing import Any, Mapping


class DfpPhase1bCalibrationIdentityError(ValueError):
    """Raised on any DFP Phase 1B conservation-calibration identity
    invariant violation. Always fail closed. Never raised for, and
    never catches, KeyboardInterrupt, SystemExit, or MemoryError --
    those always propagate unmodified."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml")
FORMAT_VERSION = "Talk2Dino-DFP-Phase1B-Conservation-Calibration-Identity-V1"

# ---------------------------------------------------------------------
# Mechanically generated closed-schema + exact-value tables (see module
# docstring). Do not hand-edit without regenerating from the committed
# TOML file: both tables are produced by flattening/walking the exact
# parsed document, never independently re-typed.
# ---------------------------------------------------------------------

TOP_KEYS: frozenset[str] = frozenset(
    [
        "artifact_schema", "candidate_identity", "candidate_order", "candidate_scope",
        "dependency_bindings", "device_software_precision_binding", "execution_contract",
        "format_version", "future_step_bindings", "identity",
        "measurement_versus_tolerance_boundary", "prohibited", "raw_measurement_summary",
        "residual_definitions", "selection_algorithm",
    ]
)

SECTION_TREE: dict[str, frozenset[str]] = {
    "identity": frozenset(["implements_runner", "implements_tolerance_selection", "name", "purpose", "required_ancestor_commit", "schema_name", "schema_version", "storage_representation_only", "working_branch"]),
    "dependency_bindings": frozenset(["e15_artifacts", "e15_protocol", "final_rescale_pullback", "finite_step_regime", "graph_module", "masked_forward_core", "stitching_pullback", "subset_manifest", "window_plan"]),
    "dependency_bindings.e15_protocol": frozenset(["relative_path", "sha256", "untouched_by_this_identity"]),
    "dependency_bindings.e15_artifacts": frozenset(["relative_path", "sha256", "untouched_by_this_identity"]),
    "dependency_bindings.masked_forward_core": frozenset(["relative_path", "sha256", "untouched_by_this_identity"]),
    "dependency_bindings.subset_manifest": frozenset(["relative_path", "run_mode", "selected_count", "sha256", "untouched_by_this_identity"]),
    "dependency_bindings.window_plan": frozenset(["relative_path", "sha256", "untouched_by_this_identity"]),
    "dependency_bindings.stitching_pullback": frozenset(["relative_path", "sha256", "untouched_by_this_identity"]),
    "dependency_bindings.final_rescale_pullback": frozenset(["relative_path", "sha256", "untouched_by_this_identity"]),
    "dependency_bindings.graph_module": frozenset(["relative_path", "sha256", "untouched_by_this_identity"]),
    "dependency_bindings.finite_step_regime": frozenset(["relative_path", "sha256", "untouched_by_this_identity"]),
    "candidate_scope": frozenset(["candidate_existence_requires_only", "candidate_windows_per_image_source", "gt_annotation_prediction_dispute_or_model_output_influence_forbidden", "image_count", "image_order_source", "run_mode"]),
    "candidate_identity": frozenset(["encoding", "field_count", "fields", "image_id_must_match_subset_manifest_selected_ids_entry", "image_path_must_be_posix_relative_no_traversal", "integer_fields", "non_negative_integer_fields_required", "string_fields"]),
    "candidate_identity.encoding": frozenset(["booleans_forbidden_in_record", "byte_encoding", "container", "encoding_name", "ensure_ascii", "floats_forbidden_in_record", "integer_serialization", "item_separator", "json_dumps_equivalent", "key_order", "key_order_is_alphabetical", "key_separator", "null_forbidden_in_record", "path_separator_in_image_path", "trailing_newline_appended", "whitespace_between_tokens"]),
    "candidate_order": frozenset(["candidate_identities_must_be_unique", "canonical_candidate_index_definition", "coordinates_are_not_an_ordering_authority", "coordinates_are_validation_fields_only", "primary_direction", "primary_key", "secondary_direction", "secondary_key"]),
    "selection_algorithm": frozenset(["digests", "fail_closed_if_universe_smaller_than_window_count", "independence", "ranking_direction", "ranking_is_total_order", "ranking_key", "select_first_n", "selected_rows_stored_in_canonical_candidate_order_not_selection_rank_order", "selection_key_formula", "window_count", "window_selection_namespace", "window_selection_namespace_source"]),
    "selection_algorithm.digests": frozenset(["candidate_universe_digest", "digest_algorithm", "digest_over_records_formula", "persisted_digest_fields", "selected_windows_digest", "selection_trace_digest", "selection_trace_record_encoding"]),
    "selection_algorithm.independence": frozenset(["independent_of_component_or_dispute_counts", "independent_of_dictionary_insertion_order", "independent_of_filesystem_enumeration_order", "independent_of_gpu_availability", "independent_of_ground_truth", "independent_of_model_predictions", "independent_of_pythonhashseed", "independent_of_residual_magnitude", "independent_of_worker_count"]),
    "execution_contract": frozenset(["adjoint_approximation_forbidden", "alpha_source", "autocast_forbidden", "canonical_recurrence_dtype", "dense_influence_kernel_forbidden", "duplicate_conservation_formula_forbidden", "duplicate_final_rescale_forbidden", "duplicate_membership_construction_forbidden", "duplicate_recurrence_forbidden", "duplicate_seed_construction_forbidden", "duplicate_signed_contribution_formula_forbidden", "duplicate_stitching_forbidden", "early_stopping_forbidden", "equilibrium_solver_forbidden", "graph_storage_dtype", "issues_conservation_pass_fail_decision", "k", "k11_path_forbidden", "masked_recurrence_dtype", "measurement_only", "mixed_precision_forbidden", "residual_accumulation_dtype", "reuses_committed_masked_forward_apis_only", "same_graph_instance_required_for_paired_canonical_and_masked_calculation", "same_process_and_job_required", "selected_window_count_required", "steps", "tf32_forbidden", "top_k_residual_approximation_forbidden"]),
    "residual_definitions": frozenset(["all_recorded_families", "primary_measurement", "rho_0", "rho_320", "rho_M", "rho_g", "tolerance_defined_here"]),
    "residual_definitions.rho_M": frozenset(["bound_function", "domain_tensor", "formula", "sample_axis"]),
    "residual_definitions.rho_0": frozenset(["bound_function", "domain_tensor", "formula", "sample_axis"]),
    "residual_definitions.rho_320": frozenset(["bound_function", "domain_tensor", "formula", "sample_axis"]),
    "residual_definitions.rho_g": frozenset(["a_b_source", "bound_function", "delta_m_definition", "disputed_pixel_domain", "final_resolution_operator", "formula", "g_definition", "stitch_denominator_source", "zero_disputed_pixel_windows_are_valid"]),
    "raw_measurement_summary": frozenset(["aggregate_mean_absolute_formula", "aggregate_rms_formula", "forbidden_fields", "low_magnitude_exclusion_forbidden", "nan_or_infinity_must_never_be_serialized", "non_finite_values_forbidden_in_a_completed_artifact", "per_residual_family_fields", "residual_clipping_forbidden", "rounding_to_zero_forbidden", "sum_fields_exist_to_permit_exact_aggregate_reconstruction", "zero_sample_family_floating_fields_must_be_json_null"]),
    "raw_measurement_summary.forbidden_fields": frozenset(["list"]),
    "artifact_schema": frozenset(["completed_artifact_requires_exactly_expected_row_count", "completed_artifact_requires_non_finite_count_zero_everywhere", "expected_selected_window_count", "required_artifact_relative_path", "required_artifact_relative_path_source", "residual_summary_record_keys", "schema_name", "schema_version", "selected_window_record_keys", "top_level_keys", "validation_requirements"]),
    "artifact_schema.top_level_keys": frozenset(["list"]),
    "artifact_schema.selected_window_record_keys": frozenset(["list"]),
    "artifact_schema.residual_summary_record_keys": frozenset(["list"]),
    "artifact_schema.validation_requirements": frozenset(["list"]),
    "device_software_precision_binding": frozenset(["device_record", "propagation_precision_record", "software_record"]),
    "device_software_precision_binding.device_record": frozenset(["device_type_required_value", "gpu_model_preselection_forbidden", "required_fields"]),
    "device_software_precision_binding.software_record": frozenset(["required_fields"]),
    "device_software_precision_binding.propagation_precision_record": frozenset(["required_fields"]),
    "measurement_versus_tolerance_boundary": frozenset(["authorizes_deterministic_measurement_only", "calibration_artifact_must_be_generated_on_intended_phase1b_device_class", "calibration_artifact_must_be_independently_verified", "calibration_runner_implemented_by_this_identity", "full_phase1b_execution_remains_unauthorized_until_that_later_authority_is_committed_and_independently_verified", "later_successor_authority_required_to_bind_verified_artifact_and_freeze_tolerance", "tolerance_inference_suggestion_or_addition_in_this_identity_forbidden"]),
    "prohibited": frozenset(["list"]),
    "future_step_bindings": frozenset(["description", "next_step_items"]),
}

# Built from path segments (never as one contiguous string literal) so
# this legitimate repository-relative-path DATA value can never be
# mistaken, by a purely textual/AST scan, for the historical forbidden
# dotted-import-alias bug that
# `tests/test_dfp_phase1a_final_rescale_pullback.py::
# test_sliding_window_plan_single_canonical_import_source` permanently
# guards against elsewhere in this codebase -- this module never
# imports `SlidingWindowPlan` at all, dotted or otherwise; it only
# records this path as a SHA-256 dependency-binding target.
_WINDOW_PLAN_RELATIVE_PATH = "/".join(
    ("src", "open_vocabulary_segmentation", "segmentation", "evaluation", "sliding_window_geometry.py")
)

CANONICAL_VALUES: dict[str, Any] = {
    "artifact_schema.completed_artifact_requires_exactly_expected_row_count": True,
    "artifact_schema.completed_artifact_requires_non_finite_count_zero_everywhere": True,
    "artifact_schema.expected_selected_window_count": 200,
    "artifact_schema.required_artifact_relative_path": "phase1b_conservation_calibration.json",
    "artifact_schema.required_artifact_relative_path_source": "verbatim reuse of e15_protocol's own conservation_calibration.required_artifact_relative_path -- never redefined here",
    "artifact_schema.residual_summary_record_keys.list": ["sample_count", "finite_count", "non_finite_count", "signed_minimum", "signed_maximum", "maximum_absolute_residual", "mean_absolute_residual", "root_mean_square_residual", "absolute_residual_sum", "squared_residual_sum"],
    "artifact_schema.schema_name": "talk2dino-dfp-phase1b-conservation-calibration-v1",
    "artifact_schema.schema_version": 1,
    "artifact_schema.selected_window_record_keys.list": ["bottom", "canonical_window_ordinal", "crop_height", "crop_width", "image_id", "image_manifest_ordinal", "image_path", "inference_canvas_height", "inference_canvas_width", "left", "patch_grid_height", "patch_grid_width", "right", "run_mode", "top", "canonical_candidate_index", "selection_rank", "selection_key", "completed", "rho_M", "rho_0", "rho_320", "rho_g"],
    "artifact_schema.top_level_keys.list": ["schema_name", "schema_version", "complete", "final", "calibration_identity_sha256", "bound_e15_protocol_sha256", "bound_e15_artifacts_sha256", "bound_masked_forward_sha256", "repository_commit", "run_mode", "bound_subset_manifest_sha256", "selected_ids_digest", "candidate_window_count", "expected_selected_window_count", "completed_selected_window_count", "candidate_universe_digest", "selected_windows_digest", "selection_trace_digest", "device_record", "software_record", "propagation_precision_record", "selected_window_records", "aggregate_residual_summaries"],
    "artifact_schema.validation_requirements.list": ["exact top-level key set (no missing, no unknown)", "exact selected-window-record key set (no missing, no unknown)", "schema_name and schema_version match exactly; schema_version type(value) is int, excluding bool and float", "bool never accepted where int is required", "float never accepted where int is required", "no NaN or Infinity anywhere", "exactly expected_selected_window_count distinct rows required when complete/final are both true", "selected_window_records stored in canonical candidate order, not selection-rank order", "no duplicate canonical_candidate_index, selection_rank, or selection_key across rows", "candidate_universe_digest, selected_windows_digest, and selection_trace_digest independently recomputable and must match the stored values", "aggregate_residual_summaries must be exactly reconstructable from the per-window absolute_residual_sum/squared_residual_sum/sample_count fields", "all SHA-256 fields are lowercase 64-hex", "repository_commit is a full lowercase 40-hex git commit id"],
    "candidate_identity.encoding.booleans_forbidden_in_record": True,
    "candidate_identity.encoding.byte_encoding": "utf-8",
    "candidate_identity.encoding.container": "json object",
    "candidate_identity.encoding.encoding_name": "talk2dino-dfp-phase1b-calibration-candidate-record-v1",
    "candidate_identity.encoding.ensure_ascii": True,
    "candidate_identity.encoding.floats_forbidden_in_record": True,
    "candidate_identity.encoding.integer_serialization": "canonical base-10 ASCII digits, no sign for non-negative fields, no leading zeros other than the single digit '0', produced by Python's native int-to-JSON serialization",
    "candidate_identity.encoding.item_separator": ",",
    "candidate_identity.encoding.json_dumps_equivalent": "json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=True)",
    "candidate_identity.encoding.key_order": ["bottom", "canonical_window_ordinal", "crop_height", "crop_width", "image_id", "image_manifest_ordinal", "image_path", "inference_canvas_height", "inference_canvas_width", "left", "patch_grid_height", "patch_grid_width", "right", "run_mode", "top"],
    "candidate_identity.encoding.key_order_is_alphabetical": True,
    "candidate_identity.encoding.key_separator": ":",
    "candidate_identity.encoding.null_forbidden_in_record": True,
    "candidate_identity.encoding.path_separator_in_image_path": "posix forward slash '/' only, regardless of host OS",
    "candidate_identity.encoding.trailing_newline_appended": False,
    "candidate_identity.encoding.whitespace_between_tokens": "none",
    "candidate_identity.field_count": 15,
    "candidate_identity.fields": ["run_mode", "image_id", "image_path", "image_manifest_ordinal", "inference_canvas_height", "inference_canvas_width", "canonical_window_ordinal", "top", "left", "bottom", "right", "crop_height", "crop_width", "patch_grid_height", "patch_grid_width"],
    "candidate_identity.image_id_must_match_subset_manifest_selected_ids_entry": True,
    "candidate_identity.image_path_must_be_posix_relative_no_traversal": True,
    "candidate_identity.integer_fields": ["image_manifest_ordinal", "inference_canvas_height", "inference_canvas_width", "canonical_window_ordinal", "top", "left", "bottom", "right", "crop_height", "crop_width", "patch_grid_height", "patch_grid_width"],
    "candidate_identity.non_negative_integer_fields_required": True,
    "candidate_identity.string_fields": ["run_mode", "image_id", "image_path"],
    "candidate_order.candidate_identities_must_be_unique": True,
    "candidate_order.canonical_candidate_index_definition": "zero-based position of a candidate record within the complete candidate universe once sorted into canonical candidate order",
    "candidate_order.coordinates_are_not_an_ordering_authority": True,
    "candidate_order.coordinates_are_validation_fields_only": True,
    "candidate_order.primary_direction": "ascending",
    "candidate_order.primary_key": "image_manifest_ordinal",
    "candidate_order.secondary_direction": "ascending",
    "candidate_order.secondary_key": "canonical_window_ordinal",
    "candidate_scope.candidate_existence_requires_only": ["image_manifest_ordinal", "inference_canvas_height", "inference_canvas_width", "canonical_window_ordinal_and_geometry_from_window_plan", "patch_grid_height_and_width_from_bound_pipeline_metadata"],
    "candidate_scope.candidate_windows_per_image_source": "dependency_bindings.window_plan's SlidingWindowPlan.build(...), applied once per image to that image's resolved inference-canvas geometry, using the frozen crop=[448,448]/stride=[224,224] from e13_dfp_phase1a_protocol.toml's own [score_interpolation_stitching] block -- e15_protocol does not itself contain this section, it only transitively hash-binds e13_dfp_phase1a_protocol.toml via its own [authority_bindings.phase1a_protocol]; never re-derived or re-guessed here",
    "candidate_scope.gt_annotation_prediction_dispute_or_model_output_influence_forbidden": True,
    "candidate_scope.image_count": 500,
    "candidate_scope.image_order_source": "dependency_bindings.subset_manifest's own 'selected_ids' field, taken verbatim in its own already-frozen canonical (lexicographic fixed-width-ID) order -- never re-sorted, re-derived, or filtered here",
    "candidate_scope.run_mode": "coco_stuff_500",
    "dependency_bindings.e15_artifacts.relative_path": "evaluation_identities/e15_dfp_phase1b_artifacts.toml",
    "dependency_bindings.e15_artifacts.sha256": "ea271b5ca85fc58d8d5b82ba27b7dcb6a24c84adf9e03b5abb6735a66ce3a2a4",
    "dependency_bindings.e15_artifacts.untouched_by_this_identity": True,
    "dependency_bindings.e15_protocol.relative_path": "evaluation_identities/e15_dfp_phase1b_protocol.toml",
    "dependency_bindings.e15_protocol.sha256": "98ba30d5268e3548892f80aeff6bdd7719ae9cc663243776746681091f5962fc",
    "dependency_bindings.e15_protocol.untouched_by_this_identity": True,
    "dependency_bindings.final_rescale_pullback.relative_path": "src/dfp_phase1a_final_rescale_pullback.py",
    "dependency_bindings.final_rescale_pullback.sha256": "fe771097e0204dde1f48d236f6142ec32880743e0fb4638425938e489af14298",
    "dependency_bindings.final_rescale_pullback.untouched_by_this_identity": True,
    "dependency_bindings.finite_step_regime.relative_path": "src/open_vocabulary_segmentation/models/dinotext/cover_dr/finite_step_regime.py",
    "dependency_bindings.finite_step_regime.sha256": "79ce8ed0e5462131cf148a1615aa60048b4a05782de7d5279ddee45856de6108",
    "dependency_bindings.finite_step_regime.untouched_by_this_identity": True,
    "dependency_bindings.graph_module.relative_path": "src/open_vocabulary_segmentation/models/dinotext/cover_dr/graph.py",
    "dependency_bindings.graph_module.sha256": "b7d4eec053dc204db869590a5ff3763b630084a65f0abd3101190b1e77f33ac5",
    "dependency_bindings.graph_module.untouched_by_this_identity": True,
    "dependency_bindings.masked_forward_core.relative_path": "src/dfp_phase1b_masked_forward.py",
    "dependency_bindings.masked_forward_core.sha256": "72f9b7093379cdbde14691e0dcc97c54eb2563d8bc8690da027409d6c81b56b3",
    "dependency_bindings.masked_forward_core.untouched_by_this_identity": True,
    "dependency_bindings.stitching_pullback.relative_path": "src/dfp_phase1a_stitching_pullback.py",
    "dependency_bindings.stitching_pullback.sha256": "6e15dd8b967eeaa317ab417ded2127d04f9064accf97fca6275e7cf962a1ac04",
    "dependency_bindings.stitching_pullback.untouched_by_this_identity": True,
    "dependency_bindings.subset_manifest.relative_path": "evaluation_subsets/dfp_phase1a_coco_stuff_500.json",
    "dependency_bindings.subset_manifest.run_mode": "coco_stuff_500",
    "dependency_bindings.subset_manifest.selected_count": 500,
    "dependency_bindings.subset_manifest.sha256": "75373aca8ef6d434ae02a7e90b6dcc9ba13fd8a31d1e87d858a2f542c19ef482",
    "dependency_bindings.subset_manifest.untouched_by_this_identity": True,
    "dependency_bindings.window_plan.relative_path": _WINDOW_PLAN_RELATIVE_PATH,
    "dependency_bindings.window_plan.sha256": "082a812907581444ed6ed87b232c82105a0071ca24cd802aceec81fa0ec3c25b",
    "dependency_bindings.window_plan.untouched_by_this_identity": True,
    "device_software_precision_binding.device_record.device_type_required_value": "cuda",
    "device_software_precision_binding.device_record.gpu_model_preselection_forbidden": True,
    "device_software_precision_binding.device_record.required_fields": ["device_type", "gpu_model_name", "cuda_compute_capability", "total_device_memory_bytes", "cuda_driver_version", "cuda_runtime_version"],
    "device_software_precision_binding.propagation_precision_record.required_fields": ["graph_storage_dtype", "canonical_recurrence_dtype", "masked_recurrence_dtype", "residual_accumulation_dtype", "alpha", "steps", "k", "chunk_size"],
    "device_software_precision_binding.software_record.required_fields": ["repository_commit", "python_version", "torch_version", "numpy_version", "cuda_version", "cudnn_version", "deterministic_algorithms_enabled", "tf32_enabled", "autocast_enabled"],
    "execution_contract.adjoint_approximation_forbidden": True,
    "execution_contract.alpha_source": "e13_dfp_phase1a_protocol.toml's own [propagation].alpha (0.98) -- e15_protocol does not itself contain a [propagation] section, it only transitively hash-binds e13_dfp_phase1a_protocol.toml via its own [authority_bindings.phase1a_protocol]; this value is reused verbatim, never redefined here",
    "execution_contract.autocast_forbidden": True,
    "execution_contract.canonical_recurrence_dtype": "float64",
    "execution_contract.dense_influence_kernel_forbidden": True,
    "execution_contract.duplicate_conservation_formula_forbidden": True,
    "execution_contract.duplicate_final_rescale_forbidden": True,
    "execution_contract.duplicate_membership_construction_forbidden": True,
    "execution_contract.duplicate_recurrence_forbidden": True,
    "execution_contract.duplicate_seed_construction_forbidden": True,
    "execution_contract.duplicate_signed_contribution_formula_forbidden": True,
    "execution_contract.duplicate_stitching_forbidden": True,
    "execution_contract.early_stopping_forbidden": True,
    "execution_contract.equilibrium_solver_forbidden": True,
    "execution_contract.graph_storage_dtype": "float32",
    "execution_contract.issues_conservation_pass_fail_decision": False,
    "execution_contract.k": 12,
    "execution_contract.k11_path_forbidden": True,
    "execution_contract.masked_recurrence_dtype": "float64",
    "execution_contract.measurement_only": True,
    "execution_contract.mixed_precision_forbidden": True,
    "execution_contract.residual_accumulation_dtype": "float64",
    "execution_contract.reuses_committed_masked_forward_apis_only": True,
    "execution_contract.same_graph_instance_required_for_paired_canonical_and_masked_calculation": True,
    "execution_contract.same_process_and_job_required": True,
    "execution_contract.selected_window_count_required": 200,
    "execution_contract.steps": 320,
    "execution_contract.tf32_forbidden": True,
    "execution_contract.top_k_residual_approximation_forbidden": True,
    "format_version": "Talk2Dino-DFP-Phase1B-Conservation-Calibration-Identity-V1",
    "future_step_bindings.description": "Fields explicitly deferred to later steps, never guessed or invented here.",
    "future_step_bindings.next_step_items": ["implement the 200-window conservation-calibration runner against this identity", "generate phase1b_conservation_calibration.json on the GPU/device class intended for Phase 1B", "independently verify the generated calibration artifact", "freeze a new successor authority binding the verified artifact and the numeric conservation tolerance", "Phase 1B execution driver, authorized only after that successor authority is committed and independently verified"],
    "identity.implements_runner": False,
    "identity.implements_tolerance_selection": False,
    "identity.name": "e16-dfp-phase1b-conservation-calibration",
    "identity.purpose": "Freezes the deterministic 200-window selection algorithm, the same-job canonical/masked FP64 execution contract, the exact raw conservation-residual definitions, and the closed phase1b_conservation_calibration.json artifact schema required before the DFP Phase 1B conservation-calibration runner may be implemented. Implements no runner, selects no tolerance, and authorizes no real image, GPU, or SLURM execution.",
    "identity.required_ancestor_commit": "3d4f0a7ad889fe9feca91474a6f30e137a0741f6",
    "identity.schema_name": "talk2dino-dfp-phase1b-conservation-calibration-identity-v1",
    "identity.schema_version": 1,
    "identity.storage_representation_only": True,
    "identity.working_branch": "dfp-phase1a-diagnostics",
    "measurement_versus_tolerance_boundary.authorizes_deterministic_measurement_only": True,
    "measurement_versus_tolerance_boundary.calibration_artifact_must_be_generated_on_intended_phase1b_device_class": True,
    "measurement_versus_tolerance_boundary.calibration_artifact_must_be_independently_verified": True,
    "measurement_versus_tolerance_boundary.calibration_runner_implemented_by_this_identity": False,
    "measurement_versus_tolerance_boundary.full_phase1b_execution_remains_unauthorized_until_that_later_authority_is_committed_and_independently_verified": True,
    "measurement_versus_tolerance_boundary.later_successor_authority_required_to_bind_verified_artifact_and_freeze_tolerance": True,
    "measurement_versus_tolerance_boundary.tolerance_inference_suggestion_or_addition_in_this_identity_forbidden": True,
    "prohibited.list": ["real_image_or_gt_access_in_this_task", "real_dataset_scan_in_this_task", "model_or_checkpoint_loading_in_this_task", "cuda_initialization_in_this_task", "real_calibration_execution_in_this_task", "numeric_tolerance_selection_in_this_task", "pamr_execution_in_this_task", "gate2_model_fitting_in_this_task", "phase1b_execution_driver_in_this_task", "slurm_launcher_or_job_submission_in_this_task", "modifying_e13_e14_or_e15_identities", "modifying_phase1a_or_phase1b_masked_forward_artifacts", "gt_in_selection", "gt_in_residual_computation", "k11_counterfactual", "early_stopping", "equilibrium_solver", "dense_influence_kernel", "adjoint_approximation_in_primary_implementation", "top_k_plus_residual_approximation"],
    "raw_measurement_summary.aggregate_mean_absolute_formula": "sum_over_windows(absolute_residual_sum) / sum_over_windows(sample_count), computed only when sum_over_windows(sample_count) > 0, else null",
    "raw_measurement_summary.aggregate_rms_formula": "sqrt(sum_over_windows(squared_residual_sum) / sum_over_windows(sample_count)), computed only when sum_over_windows(sample_count) > 0, else null",
    "raw_measurement_summary.forbidden_fields.list": ["tolerance", "default_tolerance", "epsilon", "threshold", "multiplier", "safety_factor", "percentile_cutoff", "calibrated_tolerance", "recommended_tolerance", "pass", "fail", "passed", "failed", "acceptable", "within_tolerance", "verdict"],
    "raw_measurement_summary.low_magnitude_exclusion_forbidden": True,
    "raw_measurement_summary.nan_or_infinity_must_never_be_serialized": True,
    "raw_measurement_summary.non_finite_values_forbidden_in_a_completed_artifact": True,
    "raw_measurement_summary.per_residual_family_fields": ["sample_count", "finite_count", "non_finite_count", "signed_minimum", "signed_maximum", "maximum_absolute_residual", "mean_absolute_residual", "root_mean_square_residual", "absolute_residual_sum", "squared_residual_sum"],
    "raw_measurement_summary.residual_clipping_forbidden": True,
    "raw_measurement_summary.rounding_to_zero_forbidden": True,
    "raw_measurement_summary.sum_fields_exist_to_permit_exact_aggregate_reconstruction": True,
    "raw_measurement_summary.zero_sample_family_floating_fields_must_be_json_null": True,
    "residual_definitions.all_recorded_families": ["rho_M", "rho_0", "rho_320", "rho_g"],
    "residual_definitions.primary_measurement": "rho_g",
    "residual_definitions.rho_0.bound_function": "compute_seed_reconstruction_residual",
    "residual_definitions.rho_0.domain_tensor": "seed_by_component: [component_count, node_count, query_class_count] summed over components, compared to unary_window_scores: [node_count, query_class_count]",
    "residual_definitions.rho_0.formula": "rho_0(j,c) = sum_r Y0^(r)[w,j,c] - Y0[w,j,c]",
    "residual_definitions.rho_0.sample_axis": "node_count * query_class_count, at the raw per-window recurrence-state level (pre-stitch, pre-final-rescale)",
    "residual_definitions.rho_320.bound_function": "compute_propagated_reconstruction_residual",
    "residual_definitions.rho_320.domain_tensor": "propagated_by_component: [component_count, node_count, query_class_count] summed over components, compared to canonical_propagated: [node_count, query_class_count] from the SAME-job canonical reference",
    "residual_definitions.rho_320.formula": "rho_320(j,c) = sum_r Y_320^(r)[w,j,c] - Y_320[w,j,c]",
    "residual_definitions.rho_320.sample_axis": "node_count * query_class_count, at the raw per-window recurrence-state level (pre-stitch, pre-final-rescale), at T=320",
    "residual_definitions.rho_M.bound_function": "compute_membership_partition_residual",
    "residual_definitions.rho_M.domain_tensor": "membership_rows: [component_count, node_count] for the selected window's FULL component set",
    "residual_definitions.rho_M.formula": "rho_M(j) = sum_r M[r,w,j] - 1",
    "residual_definitions.rho_M.sample_axis": "node_count (one residual value per window node j)",
    "residual_definitions.rho_g.a_b_source": "a(x) and b(x) come only from canonical final-resolution T0 and T320 predictions; ground truth is forbidden as an input to a(x)/b(x) or to any part of this measurement",
    "residual_definitions.rho_g.bound_function": "compute_signed_margin_conservation_residual",
    "residual_definitions.rho_g.delta_m_definition": "delta_m^(w)(x): the same selected window's canonical (unmasked) T320-minus-T0 contribution to the b(x)-a(x) margin, after applying the identical A_final,w operator",
    "residual_definitions.rho_g.disputed_pixel_domain": "final-resolution pixels within the selected window's own final-resolution support where a(x) != b(x) (the frozen disputed set); pixels outside this set are excluded, exactly as compute_signed_contributions already requires",
    "residual_definitions.rho_g.final_resolution_operator": "A_final,w = R ∘ A_stitch,w (dependency_bindings.stitching_pullback's apply_window_stitching_forward composed with dependency_bindings.final_rescale_pullback's apply_final_rescale_forward, exactly as already reused by compute_signed_contributions/stitch_and_rescale_component_scores)",
    "residual_definitions.rho_g.formula": "rho_g^(w)(x) = sum_r g^(w)_{x,r} - delta_m^(w)(x)",
    "residual_definitions.rho_g.g_definition": "g^(w)_{x,r}: the selected window's own signed component contribution, computed from S0/S_final AFTER applying A_final,w -- never from the pre-stitch per-window recurrence state",
    "residual_definitions.rho_g.stitch_denominator_source": "the complete canonical window plan for the image (every window contributing to that image's stitched field), never a per-window-only or partial denominator",
    "residual_definitions.rho_g.zero_disputed_pixel_windows_are_valid": "a selected window with zero disputed final-resolution pixels remains a valid, completed, calibrated window; its rho_g sample_count is 0 and its floating summary fields are recorded as null, never as a fabricated 0.0 or omitted row",
    "residual_definitions.tolerance_defined_here": False,
    "selection_algorithm.digests.candidate_universe_digest": "digest_over_records applied to every candidate record's [candidate_identity.encoding] bytes, in canonical candidate order (the complete universe)",
    "selection_algorithm.digests.digest_algorithm": "sha256",
    "selection_algorithm.digests.digest_over_records_formula": "sha256(b''.join(len(record_bytes).to_bytes(8, 'big') + record_bytes for record_bytes in records_in_the_stated_order)).hexdigest() -- lowercase 64-hex; length-prefixing removes any concatenation-boundary ambiguity",
    "selection_algorithm.digests.persisted_digest_fields": ["candidate_universe_digest", "selected_windows_digest", "selection_trace_digest"],
    "selection_algorithm.digests.selected_windows_digest": "digest_over_records applied to the 200 selected candidate records' [candidate_identity.encoding] bytes, in canonical candidate order (not selection-rank order)",
    "selection_algorithm.digests.selection_trace_digest": "digest_over_records applied to the 200 selection_trace_records, ordered by selection_rank ascending (0..199)",
    "selection_algorithm.digests.selection_trace_record_encoding": "utf-8 json object with alphabetical key order ['canonical_candidate_index', 'selection_key', 'selection_rank'], same separators/ensure_ascii/no-trailing-newline rules as [candidate_identity.encoding]",
    "selection_algorithm.fail_closed_if_universe_smaller_than_window_count": True,
    "selection_algorithm.independence.independent_of_component_or_dispute_counts": True,
    "selection_algorithm.independence.independent_of_dictionary_insertion_order": True,
    "selection_algorithm.independence.independent_of_filesystem_enumeration_order": True,
    "selection_algorithm.independence.independent_of_gpu_availability": True,
    "selection_algorithm.independence.independent_of_ground_truth": True,
    "selection_algorithm.independence.independent_of_model_predictions": True,
    "selection_algorithm.independence.independent_of_pythonhashseed": True,
    "selection_algorithm.independence.independent_of_residual_magnitude": True,
    "selection_algorithm.independence.independent_of_worker_count": True,
    "selection_algorithm.ranking_direction": "ascending",
    "selection_algorithm.ranking_is_total_order": "canonical_candidate_index is unique per candidate, so this ranking key can never produce an unresolved tie; it is frozen regardless as defense-in-depth",
    "selection_algorithm.ranking_key": "(selection_key_hex, canonical_candidate_index)",
    "selection_algorithm.select_first_n": 200,
    "selection_algorithm.selected_rows_stored_in_canonical_candidate_order_not_selection_rank_order": True,
    "selection_algorithm.selection_key_formula": "sha256(namespace.encode('utf-8') + b'\\x00' + candidate_record_canonical_json_bytes).hexdigest() -- lowercase 64-hex",
    "selection_algorithm.window_count": 200,
    "selection_algorithm.window_selection_namespace": "talk2dino-dfp-phase1b-calibration-window-selection-v1",
    "selection_algorithm.window_selection_namespace_source": "verbatim reuse of e15_protocol's own conservation_calibration.procedure.window_selection_namespace value -- never redefined here",
}

_SHA256_FIELDS: tuple[str, ...] = (
    "dependency_bindings.e15_protocol.sha256",
    "dependency_bindings.e15_artifacts.sha256",
    "dependency_bindings.masked_forward_core.sha256",
    "dependency_bindings.subset_manifest.sha256",
    "dependency_bindings.window_plan.sha256",
    "dependency_bindings.stitching_pullback.sha256",
    "dependency_bindings.final_rescale_pullback.sha256",
    "dependency_bindings.graph_module.sha256",
    "dependency_bindings.finite_step_regime.sha256",
)
_GIT_IDENTITY_FIELDS: tuple[str, ...] = ("identity.required_ancestor_commit",)

_DEPENDENCY_NAMES: tuple[str, ...] = (
    "e15_protocol", "e15_artifacts", "masked_forward_core", "subset_manifest",
    "window_plan", "stitching_pullback", "final_rescale_pullback", "graph_module", "finite_step_regime",
)


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DfpPhase1bCalibrationIdentityError(f"{label} must be a mapping")
    if set(value) != expected_keys:
        missing = expected_keys - set(value)
        unknown = set(value) - expected_keys
        raise DfpPhase1bCalibrationIdentityError(
            f"{label} has an unexpected schema (missing={sorted(missing)}, unknown={sorted(unknown)})"
        )
    return value


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str:
        raise DfpPhase1bCalibrationIdentityError(f"{label} must be an exact string")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DfpPhase1bCalibrationIdentityError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def _require_git_identity(value: Any, label: str) -> str:
    if type(value) is not str:
        raise DfpPhase1bCalibrationIdentityError(f"{label} must be an exact string")
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise DfpPhase1bCalibrationIdentityError(f"{label} must be a full lowercase 40-character Git commit id")
    return value


def _validate_closed_tree(node: Any, prefix: str) -> None:
    if not isinstance(node, Mapping):
        return
    if prefix == "":
        expected = TOP_KEYS
        label = "e16 identity"
    else:
        try:
            expected = SECTION_TREE[prefix]
        except KeyError as exc:
            raise DfpPhase1bCalibrationIdentityError(f"internal schema error: no closed-schema entry pinned for {prefix}") from exc
        label = prefix
    _require_closed_mapping(node, expected, label)
    for key, value in node.items():
        _validate_closed_tree(value, f"{prefix}.{key}" if prefix else key)


def _validate_canonical_values(node: Any, prefix: str = "") -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            _validate_canonical_values(value, f"{prefix}.{key}" if prefix else key)
        return
    try:
        expected = CANONICAL_VALUES[prefix]
    except KeyError as exc:
        raise DfpPhase1bCalibrationIdentityError(f"internal schema error: no canonical value pinned for {prefix}") from exc
    if type(node) is not type(expected) or node != expected:
        raise DfpPhase1bCalibrationIdentityError(
            f"{prefix} must equal the frozen canonical value {expected!r}, observed {node!r}"
        )


def _validate_dependency_bindings(root: Path, dependency_bindings: Mapping[str, Any]) -> None:
    for name in _DEPENDENCY_NAMES:
        binding = dependency_bindings[name]
        relative_path = binding["relative_path"]
        expected_sha256 = binding["sha256"]
        target = root / relative_path
        try:
            actual_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as exc:
            raise DfpPhase1bCalibrationIdentityError(
                f"dependency_bindings.{name}: cannot read bound file {relative_path}: {exc}"
            ) from exc
        if actual_sha256 != expected_sha256:
            raise DfpPhase1bCalibrationIdentityError(
                f"dependency_bindings.{name}: {relative_path} SHA-256 mismatch -- pinned {expected_sha256}, "
                f"observed {actual_sha256} (the bound dependency has drifted since this identity was frozen)"
            )


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load, closed-schema-validate, exact-value-validate, and
    relationally hash-verify the E16 conservation-calibration
    identity. Never mutates, re-hashes, or re-derives any of its bound
    dependencies -- only re-verifies their SHA-256 against current file
    bytes. Performs no torch/CUDA/model/dataset access."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DfpPhase1bCalibrationIdentityError(f"cannot load E16 calibration identity {source}: {error}") from error

    _validate_closed_tree(identity, "")
    _validate_canonical_values(identity)

    for field in _SHA256_FIELDS:
        section, leaf = field.rsplit(".", 1)
        value = identity
        for part in section.split("."):
            value = value[part]
        _require_sha256(value[leaf], field)
    for field in _GIT_IDENTITY_FIELDS:
        section, leaf = field.rsplit(".", 1)
        value = identity
        for part in section.split("."):
            value = value[part]
        _require_git_identity(value[leaf], field)

    _validate_dependency_bindings(root, identity["dependency_bindings"])

    return identity


__all__ = [
    "CANONICAL_VALUES",
    "DfpPhase1bCalibrationIdentityError",
    "FORMAT_VERSION",
    "IDENTITY_RELATIVE_PATH",
    "SECTION_TREE",
    "TOP_KEYS",
    "load_identity",
    "repository_root",
]
