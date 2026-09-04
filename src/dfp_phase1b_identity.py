"""Strict loading/validation for the DFP Phase 1B scientific protocol
successor identity (`evaluation_identities/e15_dfp_phase1b_protocol.toml`).

This module NEVER rewrites, re-binds, or cascades new hashes through
the completed Phase 1A protocol identity
(`evaluation_identities/e13_dfp_phase1a_protocol.toml`), the completed
Phase 1A artifact-schema identity, the Boundary IoU successor identity
(`evaluation_identities/e14_dfp_phase1b_boundary_iou.toml`), or the
frozen 500-image subset manifest. It only relationally re-verifies
their SHA-256 hashes (proving they remain byte-identical to what this
identity pins) every time it loads.

Because e15-dfp-phase1b-protocol is itself an entirely frozen
pre-registration document (every field is a fixed scientific constant,
formula, or policy flag -- there is no "variable data" field anywhere
in it), this loader pins EVERY scalar/list leaf value exactly via a
single generated `CANONICAL_VALUES` table, rather than hand-selecting a
subset of "important" fields. This is stricter than (and a superset of)
the selective-pinning pattern used by `src.dfp_phase1a_identity` and
`src.dfp_phase1b_boundary_iou_identity`: any mutation to any field,
anywhere in the tree, is rejected. `CANONICAL_VALUES`, `TOP_KEYS`, and
`SECTION_TREE` below were mechanically generated from the exact
committed file at authoring time (see the generation script referenced
in the accompanying test file) -- they are not independently hand-typed
a second time, which removes transcription-error risk while preserving
the exact same protective semantics as the rest of this codebase's
identity loaders.

Always fail closed: never silently substitute a default, coerce a
type, or normalize a string. Performs no torch, CUDA, model, dataset,
or image access -- pure TOML/filesystem/hash validation only.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from pathlib import Path
from typing import Any, Mapping


class DfpPhase1bIdentityError(ValueError):
    """Raised on any DFP Phase 1B protocol identity invariant
    violation. Always fail closed. Never raised for, and never catches,
    KeyboardInterrupt, SystemExit, or MemoryError -- those always
    propagate unmodified."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e15_dfp_phase1b_protocol.toml")
FORMAT_VERSION = "talk2dino-dfp-phase1b-protocol-identity-v1"

# ---------------------------------------------------------------------
# Mechanically generated closed-schema + exact-value tables (see module
# docstring). Do not hand-edit without regenerating from the committed
# TOML.
# ---------------------------------------------------------------------

CANONICAL_VALUES: dict[str, object] = {
    'authority_bindings.boundary_iou_successor_identity.relative_path': 'evaluation_identities/e14_dfp_phase1b_boundary_iou.toml',
    'authority_bindings.boundary_iou_successor_identity.sha256': 'fe4c33d4376d40fcd53f7751405ffc60e5856ce9919f548df3c039809ebc0ab8',
    'authority_bindings.coco_stuff_500_subset_manifest.relative_path': 'evaluation_subsets/dfp_phase1a_coco_stuff_500.json',
    'authority_bindings.coco_stuff_500_subset_manifest.sha256': '75373aca8ef6d434ae02a7e90b6dcc9ba13fd8a31d1e87d858a2f542c19ef482',
    'authority_bindings.composed_final_resolution_pullback_code.relative_path': 'src/dfp_phase1a_final_rescale_pullback.py',
    'authority_bindings.composed_final_resolution_pullback_code.sha256': 'fe771097e0204dde1f48d236f6142ec32880743e0fb4638425938e489af14298',
    'authority_bindings.final_resolution_stitching_pullback_code.relative_path': 'src/dfp_phase1a_stitching_pullback.py',
    'authority_bindings.final_resolution_stitching_pullback_code.sha256': '6e15dd8b967eeaa317ab417ded2127d04f9064accf97fca6275e7cf962a1ac04',
    'authority_bindings.gate1_analysis.path_kind': 'logical_artifact_name_not_a_private_filesystem_path',
    'authority_bindings.gate1_analysis.relative_path': 'gate1_analysis.json',
    'authority_bindings.gate1_analysis.sha256': 'bd0cb034d6368e78ca8db19d0d86d3b785d0527fe4aef4c77b841edeb34769dd',
    'authority_bindings.gate1_analysis_provenance.path_kind': 'logical_artifact_name_not_a_private_filesystem_path',
    'authority_bindings.gate1_analysis_provenance.relative_path': 'analysis_provenance.json',
    'authority_bindings.gate1_analysis_provenance.sha256': '6e67c7cfa3ba4ab7f24e70f117823a91ef0e8ec435784d479e0117ead48d51a3',
    'authority_bindings.global_unary_component_code.relative_path': 'src/dfp_phase1a_global_unary_components.py',
    'authority_bindings.global_unary_component_code.sha256': '612403bcc5faa58be4dc54d1b941bc6989abd8e4f3214ff8c33e70926b03fb2f',
    'authority_bindings.k12_graph_recurrence.relative_path': 'evaluation_identities/e12_matched_k11_k12_t320.toml',
    'authority_bindings.k12_graph_recurrence.sha256': '1f4eea731ae7b803a5e2d572d0a1f59b3275ec240d36061d03540706e3bbc259',
    'authority_bindings.phase1a_artifacts.relative_path': 'evaluation_identities/e13_dfp_phase1a_artifacts.toml',
    'authority_bindings.phase1a_artifacts.sha256': '9b8827c77fc81a7e676dece6284f69c630d61eadf916c9c2adf72f0bb30584ce',
    'authority_bindings.phase1a_protocol.relative_path': 'evaluation_identities/e13_dfp_phase1a_protocol.toml',
    'authority_bindings.phase1a_protocol.sha256': '227549b03ec6b1262c465a18b7fc67529fa5a0b5deef419c33ec79a80feb2173',
    'authority_bindings.soft_membership_code.relative_path': 'src/dfp_phase1a_soft_membership.py',
    'authority_bindings.soft_membership_code.sha256': '7bef9b800aa2770166422f715456d994e67fa218347bba6bb9d5b66695d03ac9',
    'authority_bindings.text_prototype_projection.note': "governs the CLIP-text-to-shared-space projection used to derive per-class text prototypes; the class NAME strings themselves are governed transitively by [authority_bindings.phase1a_protocol]'s own coco_stuff_dataset.class_names_digest",
    'authority_bindings.text_prototype_projection.relative_path': 'evaluation_identities/e3_paired_soft_routing.toml',
    'authority_bindings.text_prototype_projection.sha256': 'db9cb8d407d6b5770d13c2b6546a33008206ae44ebcdc55f6b7384813eec477c',
    'baseline_variants.cheap_feature_controls_frozen': True,
    'baseline_variants.oracle_variants_are_gt_conditioned_analysis_tools_not_deployable_methods': True,
    'baseline_variants.original_ten_variant_order_preserved': True,
    'baseline_variants.pamr_variants_appended_last': True,
    'baseline_variants.required': ['t0', 't16', 't32', 't64', 't320', 'oracle_o_p', 'persist_h32_full', 'persist_h32_harmful_only', 'persist_h32_beneficial_only', 'persist_h32_third_label_only', 't0_pamr', 't320_pamr'],
    'bootstrap.interval': 'percentile_95_percent',
    'bootstrap.interval_endpoint_unit': 'absolute_fraction_0_1',
    'bootstrap.numpy_quantile_method': 'linear',
    'bootstrap.paired_comparison': 'identical sampled images for cheap and provenance models',
    'bootstrap.paired_difference_unit': 'absolute_fraction_0_1',
    'bootstrap.per_replicate_rules.compute_auroc_auprc_only_when_both_classes_present': True,
    'bootstrap.per_replicate_rules.duplicate_pixels_when_image_sampled_multiple_times': True,
    'bootstrap.per_replicate_rules.fail_below_minimum_valid_replicate_fraction': True,
    'bootstrap.per_replicate_rules.include_all_eligible_pixels_of_sampled_images': True,
    'bootstrap.per_replicate_rules.minimum_valid_replicate_fraction': 0.95,
    'bootstrap.per_replicate_rules.record_number_of_valid_replicates': True,
    'bootstrap.pixels_bootstrapped_independently_forbidden': True,
    'bootstrap.quantiles': [0.025, 0.975],
    'bootstrap.replicates': 10000,
    'bootstrap.resampling_unit': 'image',
    'bootstrap.sampling': '500 image IDs with replacement',
    'bootstrap.seed_derivation': "seed = int.from_bytes(SHA256(seed_namespace)[:8], byteorder='big') % (2**32)",
    'bootstrap.seed_namespace': 'talk2dino-dfp-phase1b-bootstrap-seed-v1',
    'boundary_iou_role.authority': 'e14-dfp-phase1b-boundary-iou',
    'boundary_iou_role.boundary_iou_from_completed_phase1a_sufficient_statistics_forbidden': 'raw prediction maps were not retained by the completed Phase 1A run',
    'boundary_iou_role.dataset_mBoundaryIoU_aggregation': 'sum_then_divide_using_e14_contract',
    'boundary_iou_role.parallel_semantic_confusion_stats': ['intersection', 'union', 'predicted_pixels', 'ground_truth_pixels'],
    'boundary_iou_role.per_variant_per_image_per_class_boundary_stats': ['boundary_intersection', 'boundary_union', 'predicted_boundary_pixels', 'ground_truth_boundary_pixels'],
    'boundary_iou_role.primary_segmentation_metric': 'mean Boundary IoU',
    'boundary_iou_role.rerun_phase1a_in_this_schema_freezing_task_forbidden': True,
    'boundary_iou_role.secondary_preservation_metric': 'mIoU',
    'canonical_predictions.a_definition': 'lowest_index_argmax_of_T0_stitched_prediction',
    'canonical_predictions.b_definition': 'lowest_index_argmax_of_T320_stitched_prediction',
    'canonical_predictions.dispute_set_definition': 'D = {x : a(x) != b(x), GT(x) != ignore_index}',
    'canonical_predictions.t0_definition': 'canonical stitched final ori-resolution unary field',
    'canonical_predictions.t320_definition': 'canonical stitched final ori-resolution propagated field',
    'canonical_predictions.tie_rule': 'exact_argmax_ties_select_lowest_class_index',
    'component_construction.component_merging_after_observing_labels_forbidden': True,
    'component_construction.connectivity': 8,
    'component_construction.domain': 'complete final image',
    'component_construction.gt_forbidden': True,
    'component_construction.morphological_closing_forbidden': True,
    'component_construction.ordering': 'deterministic, by first flat pixel index in canonical row-major order',
    'component_construction.per_window_components_forbidden': True,
    'component_construction.same_class_connected_components': True,
    'component_construction.semantic_supercohort_as_causal_unit_forbidden': True,
    'component_construction.source': 'final ori-resolution T0 argmax map',
    'component_construction.top_k_plus_residual_approximation_forbidden': True,
    'component_evidence_decomposition.adjoint_approximation_forbidden_in_primary_implementation': True,
    'component_evidence_decomposition.altered_precision_between_canonical_and_masked_runs_forbidden': True,
    'component_evidence_decomposition.component_sum_accumulation_dtype': 'fp64',
    'component_evidence_decomposition.conservation_identity': 'sum_r G[x,r] = m320(x) - m0(x) = delta_m(x)',
    'component_evidence_decomposition.dense_influence_kernel_forbidden': True,
    'component_evidence_decomposition.early_stopping_forbidden': True,
    'component_evidence_decomposition.equilibrium_solver_forbidden': True,
    'component_evidence_decomposition.exact_chunking_only': True,
    'component_evidence_decomposition.exact_masked_forward_propagation_required': True,
    'component_evidence_decomposition.k11_counterfactual_forbidden': True,
    'component_evidence_decomposition.masked_propagated_stitch_formula': 'S320^(r) = A_final(K320(P) Y0^(r))',
    'component_evidence_decomposition.masked_unary_seed_formula': 'Y0^(r)[w,j,c] = M[r,w,j] * Y0[w,j,c]',
    'component_evidence_decomposition.masked_unary_stitch_formula': 'S0^(r) = A_final(Y0^(r))',
    'component_evidence_decomposition.recurrence_variant_required': 'k12',
    'component_evidence_decomposition.residual_supercomponent_forbidden': True,
    'component_evidence_decomposition.same_graph_and_recurrence_as_canonical_t320_required': True,
    'component_evidence_decomposition.signed_component_evidence_formula': 'G[x,r] = [S320^(r)(x,b) - S320^(r)(x,a)] - [S0^(r)(x,b) - S0^(r)(x,a)]',
    'component_evidence_decomposition.storage_requirement': 'the artifact must store enough information to independently verify conservation without storing the full component-by-pixel evidence matrix',
    'component_evidence_decomposition.top_k_truncation_forbidden': True,
    'conservation_calibration.execution_authorization.full_phase1b_execution_authorized': False,
    'conservation_calibration.execution_authorization.requires_calibration_artifact_bound_into_execution_identity': True,
    'conservation_calibration.execution_authorization.requires_calibration_artifact_generated': True,
    'conservation_calibration.execution_authorization.requires_calibration_artifact_independently_verified': True,
    'conservation_calibration.must_be_produced_before_full_phase1b_run': True,
    'conservation_calibration.numeric_tolerance_invented_in_this_protocol': False,
    'conservation_calibration.procedure.accumulation_dtype': 'fp64',
    'conservation_calibration.procedure.canonical_and_masked_propagation_in_same_job_required': True,
    'conservation_calibration.procedure.measurement': 'stitched-field residual',
    'conservation_calibration.procedure.recorded_fields': ['device_version', 'software_versions'],
    'conservation_calibration.procedure.same_device_type_required': True,
    'conservation_calibration.procedure.same_gpu_model_required': True,
    'conservation_calibration.procedure.same_propagation_precision_required': True,
    'conservation_calibration.procedure.window_count': 200,
    'conservation_calibration.procedure.window_selection_namespace': 'talk2dino-dfp-phase1b-calibration-window-selection-v1',
    'conservation_calibration.procedure.window_selection_source': 'deterministically selected from the frozen 500 images',
    'conservation_calibration.required_artifact_relative_path': 'phase1b_conservation_calibration.json',
    'dependency_graph.acyclic_required': True,
    'dependency_graph.artifact_identity_binds_this_identity_hash': True,
    'dependency_graph.direction': 'phase1a_protocol_and_gate1_and_boundary_iou -> phase1b_protocol -> phase1b_artifacts',
    'dependency_graph.rationale': 'the artifact identity is generated after this protocol and must bind it for provenance; this protocol is frozen before the artifact identity exists and therefore cannot bind a hash that does not yet exist without creating a circular dependency',
    'dependency_graph.this_identity_binds_artifact_identity_hash': False,
    'diagnostic_feature_evaluation.cheap_control_model.inputs': ['unary_resistance', 'propagated_margin', 'eta_net', 't16_horizon_state', 't32_horizon_state', 't64_horizon_state', 'text_cosine_ab'],
    'diagnostic_feature_evaluation.forbidden_primary_model_inputs.forbidden': ['image_rgb', 'gt', 'class_frequency', 'class_id', 'thing_stuff_label', 'dataset_statistics', 'component_size'],
    'diagnostic_feature_evaluation.forbidden_primary_model_inputs.may_be_reported_as_subgroup_diagnostics_only_if_explicitly_listed': True,
    'diagnostic_feature_evaluation.gate2_uses_out_of_fold_predictions_only': True,
    'diagnostic_feature_evaluation.primary_binary_task': 'harmful_vs_beneficial',
    'diagnostic_feature_evaluation.provenance_augmented_model.inputs': ['unary_resistance', 'propagated_margin', 'eta_net', 't16_horizon_state', 't32_horizon_state', 't64_horizon_state', 'text_cosine_ab', 'k_star', 'n_positive_components', 'phi', 'top1_positive_share', 'positive_evidence_sum', 'negative_evidence_sum'],
    'diagnostic_feature_evaluation.third_label_excluded_from_model_fitting_and_binary_auroc_auprc': True,
    'diagnostic_feature_evaluation.third_label_feature_and_risk_distributions_reported_separately': True,
    'diagnostic_model.c': 1.0,
    'diagnostic_model.class_balancing_forbidden_unless_explicitly_frozen': True,
    'diagnostic_model.class_balancing_frozen': False,
    'diagnostic_model.convergence_tolerance': 0.0001,
    'diagnostic_model.diagnostic_only_never_final_selector': True,
    'diagnostic_model.early_model_selection_forbidden': True,
    'diagnostic_model.hyperparameter_search_forbidden': True,
    'diagnostic_model.implementation': 'scikit-learn LogisticRegression',
    'diagnostic_model.input_dtype': 'float64',
    'diagnostic_model.intercept_enabled': True,
    'diagnostic_model.kind': 'fixed_logistic_regression',
    'diagnostic_model.max_iterations': 1000,
    'diagnostic_model.regularization': 'l2',
    'diagnostic_model.solver': 'lbfgs',
    'diagnostic_model.standardization': 'training_fold_only',
    'diagnostic_model.test_fold_labels_in_standardization_or_fitting_forbidden': True,
    'fold_assignment.filesystem_order_forbidden': True,
    'fold_assignment.fold_count': 5,
    'fold_assignment.fold_id_formula': 'fold_id = sorted_rank mod 5',
    'fold_assignment.images_per_fold': 100,
    'fold_assignment.key_formula': 'key(image_id) = SHA256(namespace + NUL + canonical_image_id)',
    'fold_assignment.namespace': 'talk2dino-dfp-phase1b-fold-assignment-v1',
    'fold_assignment.pixels_never_split_across_folds': True,
    'fold_assignment.python_hash_forbidden': True,
    'fold_assignment.random_library_forbidden': True,
    'fold_assignment.required_checks': ['exactly_100_images_per_fold', 'every_dispute_pixel_inherits_its_images_fold'],
    'fold_assignment.required_stored_fields': ['fold_assignment', 'fold_assignment_digest'],
    'fold_assignment.sort_key': '(key, canonical_image_id)',
    'fold_safe_transformations.apply_training_statistics_to_held_out_fold': True,
    'fold_safe_transformations.binary_horizon_indicators_never_standardized': True,
    'fold_safe_transformations.descriptive_full_dataset_summaries_allowed_only_if_labelled_descriptive_and_never_fed_into_out_of_fold_fitting': True,
    'fold_safe_transformations.full_dataset_statistics_forbidden': True,
    'fold_safe_transformations.mean_std_from_training_folds_only': True,
    'fold_safe_transformations.raw_single_feature_aurocs_unstandardized': True,
    'fold_safe_transformations.zero_variance_training_feature_behavior': 'transform to zero and record it',
    'future_step_bindings.description': 'Fields explicitly deferred to later Phase 1B steps, never guessed or invented here.',
    'future_step_bindings.next_step_items': ['phase1b_conservation_calibration.json generation and independent verification', 'Phase 1B execution driver (masked propagation + provenance computation)', 'Phase 1B diagnostic model fitting on real data', 'Gate-2 decision on real data', 'Gate-3 protocol (if reached)'],
    'gate1_verified_facts.boundary_iou_authority_status_in_phase1a': 'not_yet_bound',
    'gate1_verified_facts.boundary_iou_successor_authority_now_bound_for_phase1b': True,
    'gate1_verified_facts.boundary_trigger_status': 'FIRED',
    'gate1_verified_facts.gate1_verdict': 'PROCEED_TO_PHASE_1B',
    'gate1_verified_facts.image_count': 500,
    'gate1_verified_facts.next_prerequisite_from_gate1': 'BIND_BOUNDARY_IOU',
    'gate1_verified_facts.run_mode': 'coco_stuff_500',
    'gate2_criteria.inconclusive_conditions.confidence_interval_crossing_zero': True,
    'gate2_criteria.inconclusive_conditions.incomplete_artifact': True,
    'gate2_criteria.inconclusive_conditions.insufficient_valid_bootstrap_replicates': True,
    'gate2_criteria.inconclusive_conditions.missing_authority': True,
    'gate2_criteria.inconclusive_conditions.missing_or_invalid_pamr_statistics': True,
    'gate2_criteria.inconclusive_conditions.missing_required_baseline': True,
    'gate2_criteria.inconclusive_conditions.positive_but_smaller_than_required_conditional_gain': True,
    'gate2_criteria.mandatory_conditions.c10_boundary_iou_sufficient_statistics_for_every_frozen_baseline_variant': True,
    'gate2_criteria.mandatory_conditions.c11_pamr_statistics_present_and_valid_for_both_pamr_variants': True,
    'gate2_criteria.mandatory_conditions.c1_conservation_and_artifact_validation_pass': True,
    'gate2_criteria.mandatory_conditions.c2_single_feature_auroc_lower_bound_threshold': 0.5,
    'gate2_criteria.mandatory_conditions.c2_single_feature_candidates': ['neg_phi', 'neg_k_star', 'top1_positive_share'],
    'gate2_criteria.mandatory_conditions.c3_provenance_auroc_minimum': 0.65,
    'gate2_criteria.mandatory_conditions.c4_auroc_improvement_minimum_absolute_fraction': 0.02,
    'gate2_criteria.mandatory_conditions.c5_auroc_improvement_bootstrap_lower_bound_strictly_positive': True,
    'gate2_criteria.mandatory_conditions.c6_auprc_improvement_minimum_absolute_fraction': 0.02,
    'gate2_criteria.mandatory_conditions.c7_auprc_improvement_bootstrap_lower_bound_strictly_positive': True,
    'gate2_criteria.mandatory_conditions.c8_minimum_valid_bootstrap_replicate_fraction': 0.95,
    'gate2_criteria.mandatory_conditions.c9_exact_computation_within_resource_ceiling_without_approximation': True,
    'gate2_criteria.outcomes': ['PROCEED_TO_GATE_3', 'STOP_DFP', 'INCONCLUSIVE'],
    'gate2_criteria.pamr_role.pamr_comparison_to_final_dfp_selector_deferred_to_gate3': True,
    'gate2_criteria.pamr_role.pamr_not_a_gate2_feature': True,
    'gate2_criteria.pamr_role.pamr_not_an_input_to_diagnostic_classifier': True,
    'gate2_criteria.pamr_role.pamr_numerical_superiority_threshold_at_gate2': False,
    'gate2_criteria.pamr_role.pamr_statistics_required_for_gate2_completion': True,
    'gate2_criteria.pamr_role.reported_pamr_statistics': ['mBoundaryIoU', 'mIoU', 'delta_vs_t0', 'delta_vs_t320', 'per_class_values', 'boundary_interior_diagnostics_where_supported'],
    'gate2_criteria.stop_conditions.auroc_improvement_upper_bound_at_or_below_zero': True,
    'gate2_criteria.stop_conditions.conservation_fails_after_confirmed_implementation_correctness': True,
    'gate2_criteria.stop_conditions.exact_computation_infeasible_under_resource_ceiling_requiring_unapproved_approximation': True,
    'gate2_criteria.stop_conditions.provenance_auroc_below': 0.6,
    'gate2_criteria.threshold_loosening_after_seeing_results_forbidden': True,
    'gate2_criteria.thresholds_frozen_before_any_phase1b_result': True,
    'gate2_metrics.also_reported': ['harmful_prevalence', 'per_image_auroc_where_both_classes_occur', 'macro_mean_across_valid_images', 'thing_stuff_subgroup_results', 'boundary_interior_subgroup_results', 'transition_text_similarity_bins', 'third_label_risk_distribution', 'coverage_of_gt_in_ab'],
    'gate2_metrics.primary': 'out_of_fold_pixel_level_auroc',
    'gate2_metrics.secondary': 'out_of_fold_pixel_level_auprc',
    'gate2_metrics.subgroup_metrics_are_descriptive_only': 'subgroup metrics are descriptive and do not independently pass the gate',
    'gate2_metrics.units.auprc_unit': 'fraction_0_1',
    'gate2_metrics.units.auroc_unit': 'fraction_0_1',
    'gate2_metrics.units.delta_auprc_unit': 'absolute_fraction_0_1',
    'gate2_metrics.units.delta_auroc_unit': 'absolute_fraction_0_1',
    'gate2_metrics.units.human_readable_equivalents': 'provenance AUROC minimum 0.65 = 65%; AUROC/AUPRC improvement minimum 0.02 absolute = 2.0 percentage points -- 0.02 on the fraction_0_1 scale must never itself be written as "0.02 pp"',
    'gate2_metrics.units.percent_0_100_forbidden_for_auroc_auprc': True,
    'gate2_metrics.units.percentage_points_wording_forbidden_for_a_fraction_0_1_value': True,
    'gate3_deferred_baselines.deferred': ['densecrf', 'zoom_and_recrop'],
    'gate3_deferred_baselines.reason': 'Gate-3 method baselines, not required for the Gate-2 feature-discrimination run',
    'gt_isolation.stage_a.driver_api_must_not_accept_gt_arrays_or_dataset_annotation_objects': True,
    'gt_isolation.stage_a.label': 'GT-free provenance feature artifact',
    'gt_isolation.stage_a.may_contain': ['image_identifiers', 'dispute_offsets', 'pixel_indices', 'a_b_classes', 'margins', 'horizon_states', 'text_cosine', 'component_evidence_features', 'conservation_diagnostics', 'compute_diagnostics'],
    'gt_isolation.stage_a.must_be_finalized_and_content_hashed_before_stage_b_begins': True,
    'gt_isolation.stage_a.must_not_contain': ['gt_class', 'harmful_beneficial_third_label_target', 'oracle_decision', 'gt_boundary_label', 'any_fitted_prediction_using_gt'],
    'gt_isolation.stage_a.tests_must_poison_gt_access_during_stage_a': True,
    'gt_isolation.stage_b.binds_exact_stage_a_digest': True,
    'gt_isolation.stage_b.label': 'evaluation-label artifact',
    'gt_isolation.stage_b.may_add': ['gt_class', 'harmful_beneficial_third_label_category', 'boundary_interior_indicator', 'deterministic_fold_id', 'confusion_sufficient_statistics', 'boundary_iou_sufficient_statistics', 'diagnostic_model_outputs'],
    'identity.description': 'Freezes the DFP Phase 1B scientific protocol: the component-evidence provenance hypothesis, GT-free feature decomposition, GT-isolation architecture (Stage A / Stage B), deterministic fold assignment, diagnostic feature-evaluation procedure (cheap-control vs provenance-augmented logistic diagnostic model), image-level paired bootstrap, and the Gate-2 PROCEED_TO_GATE_3/STOP_DFP/INCONCLUSIVE criteria. Implements no evaluator, no inference, no masked propagation, and computes no provenance value.',
    'identity.diagnostic_not_deployable_selector': True,
    'identity.name': 'e15-dfp-phase1b-protocol',
    'identity.schema_name': 'talk2dino-dfp-phase1b-protocol-identity-v1',
    'identity.schema_version': 1,
    'identity.storage_representation_only': True,
    'margins.delta_m_definition': 'delta_m(x) = m320(x) - m0(x); delta_m(x) > 0',
    'margins.m0_definition': 'm0(x) = S0(x,b) - S0(x,a); m0(x) <= 0',
    'margins.m320_definition': 'm320(x) = S320(x,b) - S320(x,a); m320(x) > 0',
    'pamr_variants.causal_comparison.historical_30_20_value_is_not_a_phase1b_result': True,
    'pamr_variants.causal_comparison.no_pamr_result_currently_known_for_frozen_500_image_phase1b_run': True,
    'pamr_variants.causal_comparison.pamr_does_not_affect_g_k_star_phi_eta_net_components_soft_membership_or_q_w': True,
    'pamr_variants.causal_comparison.pamr_does_not_redefine_a_or_b': True,
    'pamr_variants.causal_comparison.pamr_variants_are_fixed_rgb_local_boundary_baselines': True,
    'pamr_variants.causal_comparison.purpose': 'test whether conventional local image refinement (PAMR) explains boundary improvements that provenance also explains, without contaminating the causal T0/T320 provenance chain',
    'pamr_variants.causal_comparison.t0_t320_remain_the_source_pair_defining_disputes_and_provenance': True,
    'per_window_class_restriction.derivation_source': 'final stitched disputed pixels',
    'per_window_class_restriction.empty_q_w_requires_no_masked_propagation': True,
    'per_window_class_restriction.only_q_w_classes_carried_through_masked_propagation': True,
    'per_window_class_restriction.ordering': 'deterministic lowest-class-index ordering',
    'per_window_class_restriction.q_w_definition': 'Q_w = { a(x), b(x) : x is disputed and final window w contributes nonzero stitching weight to x }',
    'per_window_class_restriction.q_w_never_replaced_by_image_global_classes': True,
    'per_window_class_restriction.q_w_never_restricted_to_raw_patch_argmax_classes': True,
    'per_window_class_restriction.window_identifier_bare_or_guessed_forbidden': True,
    'per_window_class_restriction.window_identifier_field': 'position.relative_path',
    'prohibited.list': ['gt_in_component_construction', 'gt_in_soft_membership', 'gt_in_graph_construction', 'gt_in_recurrence', 'gt_in_contribution_calculation', 'gt_in_feature_extraction', 'fitted_diagnostic_model_as_final_selector', 'top_k_plus_residual_approximation', 'adjoint_approximation_in_primary_implementation', 'dense_influence_kernel', 'early_stopping', 'equilibrium_solver', 'k11_counterfactual', 'epsilon_in_eta_net_denominator', 'epsilon_in_phi', 'gini_or_entropy_features_without_successor_authorization', 'threshold_loosening_after_seeing_results', 'silent_pamr_parameter_choice', 'boundary_iou_from_phase1a_sufficient_statistics', 'rerunning_phase1a_in_this_task', 'masked_propagation_in_this_task', 'diagnostic_classifier_result_in_this_task', 'real_image_or_gt_access_in_this_task', 'real_boundary_iou_computation_in_this_task', 'final_selector_in_this_task', 'threshold_tuning_in_this_task', 'modifying_e13_or_e14_identities', 'modifying_phase1a_artifacts', 'phase1b_execution_driver_in_this_task', 'slurm_launcher_in_this_task', 'job_submission_in_this_task', 'cuda_initialization_in_this_task'],
    'provenance_features.component_evidence.additional_concentration_diagnostics_forbidden_without_successor_authorization': 'no Gini, entropy, or additional features may be added after observing results unless a later successor protocol explicitly authorizes them',
    'provenance_features.component_evidence.g_plus_definition': 'G_plus = { G[x,r] : G[x,r] > 0 }',
    'provenance_features.component_evidence.k_star_always_finite_for_valid_disputed_pixel': True,
    'provenance_features.component_evidence.k_star_definition': 'k_star(x) = minimum k such that m320(x) - sum_{q=1..k} G_positive_sorted[q] <= 0',
    'provenance_features.component_evidence.k_star_non_finite_case_behavior': 'reject an invalid non-finite case rather than inventing infinity',
    'provenance_features.component_evidence.n_positive_components_field': 'n_positive_components',
    'provenance_features.component_evidence.n_positive_components_stored_beside_k_star': 'because the integer count is fragmentation-confounded',
    'provenance_features.component_evidence.negative_evidence_sum_formula': 'negative_evidence_sum = sum_r min(G[x,r], 0)',
    'provenance_features.component_evidence.phi_epsilon_forbidden': True,
    'provenance_features.component_evidence.phi_formula': 'phi(x) = sum_{q=1..k_star} G_positive_sorted[q] / positive_evidence_sum',
    'provenance_features.component_evidence.positive_evidence_sum_formula': 'positive_evidence_sum = sum_r max(G[x,r], 0)',
    'provenance_features.component_evidence.sort_order': '1) descending contribution; 2) ascending deterministic component ID for exact ties',
    'provenance_features.component_evidence.top1_positive_share_field': 'top1_positive_share',
    'provenance_features.component_evidence.top1_positive_share_formula': 'top1_positive_share = top_positive_evidence / positive_evidence_sum',
    'provenance_features.component_evidence.top_positive_component_id_field': 'top_positive_component_id',
    'provenance_features.component_evidence.top_positive_evidence_field': 'top_positive_evidence',
    'provenance_features.horizon_controls.a_wins_at_h_formula': 'a_wins_at_h = 1[argmax S_h(x) = a(x)]',
    'provenance_features.horizon_controls.b_wins_at_h_formula': 'b_wins_at_h = 1[argmax S_h(x) = b(x)]',
    'provenance_features.horizon_controls.horizons_stored': [16, 32, 64],
    'provenance_features.horizon_controls.primary_cheap_horizon_baseline': 32,
    'provenance_features.horizon_controls.third_wins_at_h_formula': 'third_wins_at_h = 1 - b_wins_at_h - a_wins_at_h (otherwise case)',
    'provenance_features.margins_and_controls.epsilon_in_eta_net_denominator_forbidden': True,
    'provenance_features.margins_and_controls.eta_net_denominator_note': 'because m320 > 0 and m0 <= 0, the denominator (m320 - m0) is a sum of magnitudes and is always strictly positive',
    'provenance_features.margins_and_controls.eta_net_formula': 'eta_net = m320 / (m320 - m0)',
    'provenance_features.margins_and_controls.margin_change_formula': 'margin_change = delta_m',
    'provenance_features.margins_and_controls.propagated_margin_formula': 'propagated_margin = m320',
    'provenance_features.margins_and_controls.small_margin_exclusion_forbidden': True,
    'provenance_features.margins_and_controls.unary_resistance_formula': 'unary_resistance = -m0',
    'provenance_features.text_space_control.gt_dependent_standardization_forbidden_during_feature_generation': True,
    'provenance_features.text_space_control.text_cosine_ab_formula': 'text_cosine_ab = cosine(t_a, t_b)',
    'provenance_features.text_space_control.text_prototype_source': 'exact text prototypes bound to [authority_bindings.text_prototype_projection]',
    'resource_ceiling.checkpoint_after_every_completed_image': True,
    'resource_ceiling.chunk_size_selection': 'deterministic memory preflight only',
    'resource_ceiling.exact_chunking_required': True,
    'resource_ceiling.gpu_model_restriction_required_unless_by_calibration_identity': False,
    'resource_ceiling.multi_gpu_numerical_variation_forbidden': True,
    'resource_ceiling.oom_retry_that_silently_changes_scientific_computation_forbidden': True,
    'resource_ceiling.peak_allocated_memory_ceiling_fraction_of_physical_gpu_memory': 0.8,
    'resource_ceiling.planning_authority': 'gate1_analysis.json compute_projection (bound above by exact SHA-256)',
    'resource_ceiling.relative_projected_cost_vs_one_propagation': 19.831283867160256,
    'resource_ceiling.resume_must_reproduce_uninterrupted_arrays_exactly': True,
    'resource_ceiling.single_gpu_only': True,
    'resource_ceiling.values_reconstructed_from_bound_gate1_analysis_not_hardcoded_rounded': True,
    'resource_ceiling.wall_clock_estimates_are_diagnostic_not_gate_values_unless_measured_under_bound_calibration_environment': True,
    'resource_ceiling.worst_case_projected_peak_bytes': 113860608,
    'rgb_local_boundary_baseline.argmax_order': 'pamr_before_argmax',
    'rgb_local_boundary_baseline.authority_status': 'bound',
    'rgb_local_boundary_baseline.baseline_family': 'pamr',
    'rgb_local_boundary_baseline.configuration_relative_path': 'src/open_vocabulary_segmentation/models/dinotext/dinotext.py',
    'rgb_local_boundary_baseline.configuration_sha256': '94d920e7f9153e97c8c4948013beffd7533902c7d307fe15607a6b03508873ed',
    'rgb_local_boundary_baseline.determinism': True,
    'rgb_local_boundary_baseline.dilations': [1, 2, 4, 8, 12, 24],
    'rgb_local_boundary_baseline.ignore_handling': 'PAMR itself has no ignore-index awareness; ignore handling for T0_PAMR/T320_PAMR confusion statistics follows the same frozen valid-pixel definition (V = GT != ignore_index) already used for every other baseline variant',
    'rgb_local_boundary_baseline.image_normalization': "single-pass pipeline normalization (BGR-to-RGB channel reorder via index [2,1,0], then the model's own mean/std as already applied upstream); never re-normalized a second time and never restored to raw 0-255/0-1 pixel values before being passed to PAMR",
    'rgb_local_boundary_baseline.implementation_relative_path': 'src/open_vocabulary_segmentation/models/dinotext/pamr.py',
    'rgb_local_boundary_baseline.implementation_sha256': '5f3d6051e8e22aa2b0cfe9e091e3644d217b56716e152a654e2b6996c489f1d2',
    'rgb_local_boundary_baseline.inherited_configuration_paths': ['src/open_vocabulary_segmentation/configs/stuff/eval_stuff_pamr.yml'],
    'rgb_local_boundary_baseline.inherited_configuration_sha256s': ['d6043d83676fcb498e966fbaa0b90f68be3eed46dbb7002e3d332e13a69d9cb6'],
    'rgb_local_boundary_baseline.input_image_definition': "the evaluation pipeline's own image tensor at the point PAMR is invoked -- for T0_PAMR/T320_PAMR this is DFP's own already-loaded, already-normalized final-resolution image, never a separately re-decoded or re-fetched image",
    'rgb_local_boundary_baseline.input_score_definition': "the soft (pre-argmax), per-class stitched score tensor at DFP's final ori-resolution canonical domain -- T0 for t0_pamr, T320 for t320_pamr",
    'rgb_local_boundary_baseline.iterations': 10,
    'rgb_local_boundary_baseline.kernel': '3x3_local_affinity_shift_kernel_8_connectivity',
    'rgb_local_boundary_baseline.output_resolution': "identical to the input score tensor's resolution (DFP's final ori-resolution canonical domain); PAMR never changes resolution",
    'rgb_local_boundary_baseline.pamr_does_not_feed_provenance_features': True,
    'rgb_local_boundary_baseline.pamr_is_not_used_to_construct_components': True,
    'rgb_local_boundary_baseline.pamr_is_not_used_to_select_disputed_pixels': True,
    'rgb_local_boundary_baseline.pipeline_insertion_point': 'after bilinear upsample of the soft per-class score tensor to the final output resolution, strictly before argmax',
    'rgb_local_boundary_baseline.resize_and_interpolation': "bilinear, align_corners=true; identity in practice whenever the image and score tensor already share resolution, which holds when applied to DFP's own final-resolution T0/T320 tensors",
    'rgb_local_boundary_baseline.resolved_parameter_digest': '0cb1e75acb3a7e068ef10a41519e137910a64f2acdb201cf11caebf83360b14d',
    'rgb_local_boundary_baseline.resolved_parameter_digest_formula': "SHA256(json.dumps(resolved_params, sort_keys=True, separators=(',', ':')).encode('utf-8')), where resolved_params = {operator:'PAMR', num_iter:iterations, dilations:dilations, kernel:kernel, affinity:sigma_or_affinity_parameters, insertion_point:pipeline_insertion_point, image_input:input_image_definition, image_normalization:image_normalization, resize_interpolation:resize_and_interpolation, scope:window_or_global_scope, argmax_order:argmax_order, determinism:determinism, training_free:training_free} -- every value taken verbatim from this SAME [rgb_local_boundary_baseline] table, so the digest is mechanically self-verifying against this file (see src.dfp_phase1b_identity.compute_resolved_pamr_parameter_digest)",
    'rgb_local_boundary_baseline.same_argmax_convention_for_both': True,
    'rgb_local_boundary_baseline.same_output_geometry_for_both': True,
    'rgb_local_boundary_baseline.same_pamr_parameters_for_both': True,
    'rgb_local_boundary_baseline.same_rgb_input_for_both': True,
    'rgb_local_boundary_baseline.selection_did_not_use_phase1b_results': True,
    'rgb_local_boundary_baseline.selection_provenance': 'README.md\'s own officially-documented ViT-Base COCO-Stuff reproduction command (# COCOStuff section) uses --eval_cfg src/open_vocabulary_segmentation/configs/stuff/dinotext_stuff_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/stuff/eval_stuff_pamr.yml; grep -rn "PAMR(" across the whole repository confirms exactly one instantiation site (dinotext.py::apply_pamr), hardcoding num_iter=10 and dilations=[1,2,4,8,12,24] independent of which model/dataset config is active, and confirms no YAML anywhere in this repository overrides those two values',
    'rgb_local_boundary_baseline.sigma_or_affinity_parameters': 'softmax over the dilation/offset axis of -abs(local_affinity) / (1e-8 + 0.1 * local_stdev); no separate sigma parameter exists in this operator',
    'rgb_local_boundary_baseline.training_free': True,
    'rgb_local_boundary_baseline.window_or_global_scope': 'applied to the complete final-resolution stitched field; never per sliding window',
    'scientific_purpose.diagnostic_fitted_models_are_analysis_tools_only': True,
    'scientific_purpose.final_5000_image_evaluation_untouched': 'the final 5,000-image COCO evaluation remains untouched and confirmatory; this protocol governs only the frozen stratified 500-image development subset',
    'scientific_purpose.fitted_diagnostic_model_forbidden_as_final_method': 'no fitted diagnostic model produced under this protocol may be reported as the training-free final DFP method',
    'scientific_purpose.gt_forbidden_in_component_construction': True,
    'scientific_purpose.gt_forbidden_in_contribution_calculation': True,
    'scientific_purpose.gt_forbidden_in_feature_extraction': True,
    'scientific_purpose.gt_forbidden_in_graph_construction': True,
    'scientific_purpose.gt_forbidden_in_recurrence': True,
    'scientific_purpose.gt_forbidden_in_soft_membership': True,
    'scientific_purpose.gt_free_features_must_be_sealed_before_labeling': 'GT may define evaluation labels only after GT-free provenance features are sealed (Stage A digest finalized before Stage B begins)',
    'scientific_purpose.hypothesis': 'Does component-level diffusion evidence provenance provide statistically and practically meaningful information for distinguishing harmful T0->T320 washouts from beneficial T0->T320 corrections, beyond unary confidence, propagation margin, text similarity and early-horizon persistence?',
    'scientific_purpose.phase1b_is_diagnostic_not_final_selector': True,
    'scientific_purpose.subset_scope': 'coco_stuff_500 only; ade20k_500 was never run for Gate-1 and is out of scope for this protocol',
    'soft_membership_pullback.component_source_mass_is_diagnostic_only': True,
    'soft_membership_pullback.component_source_mass_not_patch_equivalent_area': True,
    'soft_membership_pullback.component_source_mass_not_pixel_area': True,
    'soft_membership_pullback.composition_formula': 'A_final = R composed with A_stitch',
    'soft_membership_pullback.membership_nonnegative': 'M[r,w,j] >= 0',
    'soft_membership_pullback.per_source_patch_copy_sum_to_one': 'sum_r M[r,w,j] = 1',
    'target_categories.beneficial_definition': 'GT(x) == b(x) and GT(x) != a(x)',
    'target_categories.binary_gate2_population_definition': 'GT(x) in {a(x), b(x)}',
    'target_categories.harmful_definition': 'GT(x) == a(x) and GT(x) != b(x)',
    'target_categories.ignore_pixels_excluded_from_every_target_population': True,
    'target_categories.negative_binary_label': 'beneficial = 0',
    'target_categories.positive_binary_label': 'harmful = 1',
    'target_categories.third_label_definition': 'GT(x) not in {a(x), b(x)}',
    'target_categories.third_label_excluded_from_binary_negative_class': True,
    'training_subsampling.fewer_available_uses_all': True,
    'training_subsampling.hash_algorithm': 'SHA256',
    'training_subsampling.held_out_evaluation_pixels_never_subsampled': True,
    'training_subsampling.key_components': ['frozen_namespace', 'canonical_image_id', 'final_pixel_linear_index', 'target_category'],
    'training_subsampling.key_namespace': 'talk2dino-dfp-phase1b-training-subsample-v1',
    'training_subsampling.maximum_per_image_per_binary_category': 4096,
    'training_subsampling.required_reports': ['training_sample_counts', 'full_held_out_evaluation_counts'],
    'training_subsampling.retain_rule': 'lowest keys, up to a fixed maximum per image per binary category',
}

TOP_KEYS = frozenset(['authority_bindings', 'baseline_variants', 'bootstrap', 'boundary_iou_role', 'canonical_predictions', 'component_construction', 'component_evidence_decomposition', 'conservation_calibration', 'dependency_graph', 'diagnostic_feature_evaluation', 'diagnostic_model', 'fold_assignment', 'fold_safe_transformations', 'format_version', 'future_step_bindings', 'gate1_verified_facts', 'gate2_criteria', 'gate2_metrics', 'gate3_deferred_baselines', 'gt_isolation', 'identity', 'margins', 'pamr_variants', 'per_window_class_restriction', 'prohibited', 'provenance_features', 'resource_ceiling', 'rgb_local_boundary_baseline', 'scientific_purpose', 'soft_membership_pullback', 'target_categories', 'training_subsampling'])

SECTION_TREE: dict[str, frozenset[str]] = {
    'identity': frozenset(['description', 'diagnostic_not_deployable_selector', 'name', 'schema_name', 'schema_version', 'storage_representation_only']),
    'dependency_graph': frozenset(['acyclic_required', 'artifact_identity_binds_this_identity_hash', 'direction', 'rationale', 'this_identity_binds_artifact_identity_hash']),
    'authority_bindings': frozenset(['boundary_iou_successor_identity', 'coco_stuff_500_subset_manifest', 'composed_final_resolution_pullback_code', 'final_resolution_stitching_pullback_code', 'gate1_analysis', 'gate1_analysis_provenance', 'global_unary_component_code', 'k12_graph_recurrence', 'phase1a_artifacts', 'phase1a_protocol', 'soft_membership_code', 'text_prototype_projection']),
    'authority_bindings.phase1a_protocol': frozenset(['relative_path', 'sha256']),
    'authority_bindings.phase1a_artifacts': frozenset(['relative_path', 'sha256']),
    'authority_bindings.coco_stuff_500_subset_manifest': frozenset(['relative_path', 'sha256']),
    'authority_bindings.boundary_iou_successor_identity': frozenset(['relative_path', 'sha256']),
    'authority_bindings.gate1_analysis': frozenset(['path_kind', 'relative_path', 'sha256']),
    'authority_bindings.gate1_analysis_provenance': frozenset(['path_kind', 'relative_path', 'sha256']),
    'authority_bindings.k12_graph_recurrence': frozenset(['relative_path', 'sha256']),
    'authority_bindings.text_prototype_projection': frozenset(['note', 'relative_path', 'sha256']),
    'authority_bindings.final_resolution_stitching_pullback_code': frozenset(['relative_path', 'sha256']),
    'authority_bindings.composed_final_resolution_pullback_code': frozenset(['relative_path', 'sha256']),
    'authority_bindings.global_unary_component_code': frozenset(['relative_path', 'sha256']),
    'authority_bindings.soft_membership_code': frozenset(['relative_path', 'sha256']),
    'gate1_verified_facts': frozenset(['boundary_iou_authority_status_in_phase1a', 'boundary_iou_successor_authority_now_bound_for_phase1b', 'boundary_trigger_status', 'gate1_verdict', 'image_count', 'next_prerequisite_from_gate1', 'run_mode']),
    'scientific_purpose': frozenset(['diagnostic_fitted_models_are_analysis_tools_only', 'final_5000_image_evaluation_untouched', 'fitted_diagnostic_model_forbidden_as_final_method', 'gt_forbidden_in_component_construction', 'gt_forbidden_in_contribution_calculation', 'gt_forbidden_in_feature_extraction', 'gt_forbidden_in_graph_construction', 'gt_forbidden_in_recurrence', 'gt_forbidden_in_soft_membership', 'gt_free_features_must_be_sealed_before_labeling', 'hypothesis', 'phase1b_is_diagnostic_not_final_selector', 'subset_scope']),
    'canonical_predictions': frozenset(['a_definition', 'b_definition', 'dispute_set_definition', 't0_definition', 't320_definition', 'tie_rule']),
    'margins': frozenset(['delta_m_definition', 'm0_definition', 'm320_definition']),
    'target_categories': frozenset(['beneficial_definition', 'binary_gate2_population_definition', 'harmful_definition', 'ignore_pixels_excluded_from_every_target_population', 'negative_binary_label', 'positive_binary_label', 'third_label_definition', 'third_label_excluded_from_binary_negative_class']),
    'component_construction': frozenset(['component_merging_after_observing_labels_forbidden', 'connectivity', 'domain', 'gt_forbidden', 'morphological_closing_forbidden', 'ordering', 'per_window_components_forbidden', 'same_class_connected_components', 'semantic_supercohort_as_causal_unit_forbidden', 'source', 'top_k_plus_residual_approximation_forbidden']),
    'soft_membership_pullback': frozenset(['component_source_mass_is_diagnostic_only', 'component_source_mass_not_patch_equivalent_area', 'component_source_mass_not_pixel_area', 'composition_formula', 'membership_nonnegative', 'per_source_patch_copy_sum_to_one']),
    'component_evidence_decomposition': frozenset(['adjoint_approximation_forbidden_in_primary_implementation', 'altered_precision_between_canonical_and_masked_runs_forbidden', 'component_sum_accumulation_dtype', 'conservation_identity', 'dense_influence_kernel_forbidden', 'early_stopping_forbidden', 'equilibrium_solver_forbidden', 'exact_chunking_only', 'exact_masked_forward_propagation_required', 'k11_counterfactual_forbidden', 'masked_propagated_stitch_formula', 'masked_unary_seed_formula', 'masked_unary_stitch_formula', 'recurrence_variant_required', 'residual_supercomponent_forbidden', 'same_graph_and_recurrence_as_canonical_t320_required', 'signed_component_evidence_formula', 'storage_requirement', 'top_k_truncation_forbidden']),
    'per_window_class_restriction': frozenset(['derivation_source', 'empty_q_w_requires_no_masked_propagation', 'only_q_w_classes_carried_through_masked_propagation', 'ordering', 'q_w_definition', 'q_w_never_replaced_by_image_global_classes', 'q_w_never_restricted_to_raw_patch_argmax_classes', 'window_identifier_bare_or_guessed_forbidden', 'window_identifier_field']),
    'provenance_features': frozenset(['component_evidence', 'horizon_controls', 'margins_and_controls', 'text_space_control']),
    'provenance_features.margins_and_controls': frozenset(['epsilon_in_eta_net_denominator_forbidden', 'eta_net_denominator_note', 'eta_net_formula', 'margin_change_formula', 'propagated_margin_formula', 'small_margin_exclusion_forbidden', 'unary_resistance_formula']),
    'provenance_features.horizon_controls': frozenset(['a_wins_at_h_formula', 'b_wins_at_h_formula', 'horizons_stored', 'primary_cheap_horizon_baseline', 'third_wins_at_h_formula']),
    'provenance_features.text_space_control': frozenset(['gt_dependent_standardization_forbidden_during_feature_generation', 'text_cosine_ab_formula', 'text_prototype_source']),
    'provenance_features.component_evidence': frozenset(['additional_concentration_diagnostics_forbidden_without_successor_authorization', 'g_plus_definition', 'k_star_always_finite_for_valid_disputed_pixel', 'k_star_definition', 'k_star_non_finite_case_behavior', 'n_positive_components_field', 'n_positive_components_stored_beside_k_star', 'negative_evidence_sum_formula', 'phi_epsilon_forbidden', 'phi_formula', 'positive_evidence_sum_formula', 'sort_order', 'top1_positive_share_field', 'top1_positive_share_formula', 'top_positive_component_id_field', 'top_positive_evidence_field']),
    'gt_isolation': frozenset(['stage_a', 'stage_b']),
    'gt_isolation.stage_a': frozenset(['driver_api_must_not_accept_gt_arrays_or_dataset_annotation_objects', 'label', 'may_contain', 'must_be_finalized_and_content_hashed_before_stage_b_begins', 'must_not_contain', 'tests_must_poison_gt_access_during_stage_a']),
    'gt_isolation.stage_b': frozenset(['binds_exact_stage_a_digest', 'label', 'may_add']),
    'fold_assignment': frozenset(['filesystem_order_forbidden', 'fold_count', 'fold_id_formula', 'images_per_fold', 'key_formula', 'namespace', 'pixels_never_split_across_folds', 'python_hash_forbidden', 'random_library_forbidden', 'required_checks', 'required_stored_fields', 'sort_key']),
    'diagnostic_feature_evaluation': frozenset(['cheap_control_model', 'forbidden_primary_model_inputs', 'gate2_uses_out_of_fold_predictions_only', 'primary_binary_task', 'provenance_augmented_model', 'third_label_excluded_from_model_fitting_and_binary_auroc_auprc', 'third_label_feature_and_risk_distributions_reported_separately']),
    'diagnostic_feature_evaluation.cheap_control_model': frozenset(['inputs']),
    'diagnostic_feature_evaluation.provenance_augmented_model': frozenset(['inputs']),
    'diagnostic_feature_evaluation.forbidden_primary_model_inputs': frozenset(['forbidden', 'may_be_reported_as_subgroup_diagnostics_only_if_explicitly_listed']),
    'diagnostic_model': frozenset(['c', 'class_balancing_forbidden_unless_explicitly_frozen', 'class_balancing_frozen', 'convergence_tolerance', 'diagnostic_only_never_final_selector', 'early_model_selection_forbidden', 'hyperparameter_search_forbidden', 'implementation', 'input_dtype', 'intercept_enabled', 'kind', 'max_iterations', 'regularization', 'solver', 'standardization', 'test_fold_labels_in_standardization_or_fitting_forbidden']),
    'training_subsampling': frozenset(['fewer_available_uses_all', 'hash_algorithm', 'held_out_evaluation_pixels_never_subsampled', 'key_components', 'key_namespace', 'maximum_per_image_per_binary_category', 'required_reports', 'retain_rule']),
    'fold_safe_transformations': frozenset(['apply_training_statistics_to_held_out_fold', 'binary_horizon_indicators_never_standardized', 'descriptive_full_dataset_summaries_allowed_only_if_labelled_descriptive_and_never_fed_into_out_of_fold_fitting', 'full_dataset_statistics_forbidden', 'mean_std_from_training_folds_only', 'raw_single_feature_aurocs_unstandardized', 'zero_variance_training_feature_behavior']),
    'gate2_metrics': frozenset(['also_reported', 'primary', 'secondary', 'subgroup_metrics_are_descriptive_only', 'units']),
    'gate2_metrics.units': frozenset(['auprc_unit', 'auroc_unit', 'delta_auprc_unit', 'delta_auroc_unit', 'human_readable_equivalents', 'percent_0_100_forbidden_for_auroc_auprc', 'percentage_points_wording_forbidden_for_a_fraction_0_1_value']),
    'bootstrap': frozenset(['interval', 'interval_endpoint_unit', 'numpy_quantile_method', 'paired_comparison', 'paired_difference_unit', 'per_replicate_rules', 'pixels_bootstrapped_independently_forbidden', 'quantiles', 'replicates', 'resampling_unit', 'sampling', 'seed_derivation', 'seed_namespace']),
    'bootstrap.per_replicate_rules': frozenset(['compute_auroc_auprc_only_when_both_classes_present', 'duplicate_pixels_when_image_sampled_multiple_times', 'fail_below_minimum_valid_replicate_fraction', 'include_all_eligible_pixels_of_sampled_images', 'minimum_valid_replicate_fraction', 'record_number_of_valid_replicates']),
    'gate2_criteria': frozenset(['inconclusive_conditions', 'mandatory_conditions', 'outcomes', 'pamr_role', 'stop_conditions', 'threshold_loosening_after_seeing_results_forbidden', 'thresholds_frozen_before_any_phase1b_result']),
    'gate2_criteria.mandatory_conditions': frozenset(['c10_boundary_iou_sufficient_statistics_for_every_frozen_baseline_variant', 'c11_pamr_statistics_present_and_valid_for_both_pamr_variants', 'c1_conservation_and_artifact_validation_pass', 'c2_single_feature_auroc_lower_bound_threshold', 'c2_single_feature_candidates', 'c3_provenance_auroc_minimum', 'c4_auroc_improvement_minimum_absolute_fraction', 'c5_auroc_improvement_bootstrap_lower_bound_strictly_positive', 'c6_auprc_improvement_minimum_absolute_fraction', 'c7_auprc_improvement_bootstrap_lower_bound_strictly_positive', 'c8_minimum_valid_bootstrap_replicate_fraction', 'c9_exact_computation_within_resource_ceiling_without_approximation']),
    'gate2_criteria.stop_conditions': frozenset(['auroc_improvement_upper_bound_at_or_below_zero', 'conservation_fails_after_confirmed_implementation_correctness', 'exact_computation_infeasible_under_resource_ceiling_requiring_unapproved_approximation', 'provenance_auroc_below']),
    'gate2_criteria.inconclusive_conditions': frozenset(['confidence_interval_crossing_zero', 'incomplete_artifact', 'insufficient_valid_bootstrap_replicates', 'missing_authority', 'missing_or_invalid_pamr_statistics', 'missing_required_baseline', 'positive_but_smaller_than_required_conditional_gain']),
    'gate2_criteria.pamr_role': frozenset(['pamr_comparison_to_final_dfp_selector_deferred_to_gate3', 'pamr_not_a_gate2_feature', 'pamr_not_an_input_to_diagnostic_classifier', 'pamr_numerical_superiority_threshold_at_gate2', 'pamr_statistics_required_for_gate2_completion', 'reported_pamr_statistics']),
    'boundary_iou_role': frozenset(['authority', 'boundary_iou_from_completed_phase1a_sufficient_statistics_forbidden', 'dataset_mBoundaryIoU_aggregation', 'parallel_semantic_confusion_stats', 'per_variant_per_image_per_class_boundary_stats', 'primary_segmentation_metric', 'rerun_phase1a_in_this_schema_freezing_task_forbidden', 'secondary_preservation_metric']),
    'baseline_variants': frozenset(['cheap_feature_controls_frozen', 'oracle_variants_are_gt_conditioned_analysis_tools_not_deployable_methods', 'original_ten_variant_order_preserved', 'pamr_variants_appended_last', 'required']),
    'rgb_local_boundary_baseline': frozenset(['argmax_order', 'authority_status', 'baseline_family', 'configuration_relative_path', 'configuration_sha256', 'determinism', 'dilations', 'ignore_handling', 'image_normalization', 'implementation_relative_path', 'implementation_sha256', 'inherited_configuration_paths', 'inherited_configuration_sha256s', 'input_image_definition', 'input_score_definition', 'iterations', 'kernel', 'output_resolution', 'pamr_does_not_feed_provenance_features', 'pamr_is_not_used_to_construct_components', 'pamr_is_not_used_to_select_disputed_pixels', 'pipeline_insertion_point', 'resize_and_interpolation', 'resolved_parameter_digest', 'resolved_parameter_digest_formula', 'same_argmax_convention_for_both', 'same_output_geometry_for_both', 'same_pamr_parameters_for_both', 'same_rgb_input_for_both', 'selection_did_not_use_phase1b_results', 'selection_provenance', 'sigma_or_affinity_parameters', 'training_free', 'window_or_global_scope']),
    'pamr_variants': frozenset(['causal_comparison']),
    'pamr_variants.causal_comparison': frozenset(['historical_30_20_value_is_not_a_phase1b_result', 'no_pamr_result_currently_known_for_frozen_500_image_phase1b_run', 'pamr_does_not_affect_g_k_star_phi_eta_net_components_soft_membership_or_q_w', 'pamr_does_not_redefine_a_or_b', 'pamr_variants_are_fixed_rgb_local_boundary_baselines', 'purpose', 't0_t320_remain_the_source_pair_defining_disputes_and_provenance']),
    'gate3_deferred_baselines': frozenset(['deferred', 'reason']),
    'conservation_calibration': frozenset(['execution_authorization', 'must_be_produced_before_full_phase1b_run', 'numeric_tolerance_invented_in_this_protocol', 'procedure', 'required_artifact_relative_path']),
    'conservation_calibration.procedure': frozenset(['accumulation_dtype', 'canonical_and_masked_propagation_in_same_job_required', 'measurement', 'recorded_fields', 'same_device_type_required', 'same_gpu_model_required', 'same_propagation_precision_required', 'window_count', 'window_selection_namespace', 'window_selection_source']),
    'conservation_calibration.execution_authorization': frozenset(['full_phase1b_execution_authorized', 'requires_calibration_artifact_bound_into_execution_identity', 'requires_calibration_artifact_generated', 'requires_calibration_artifact_independently_verified']),
    'resource_ceiling': frozenset(['checkpoint_after_every_completed_image', 'chunk_size_selection', 'exact_chunking_required', 'gpu_model_restriction_required_unless_by_calibration_identity', 'multi_gpu_numerical_variation_forbidden', 'oom_retry_that_silently_changes_scientific_computation_forbidden', 'peak_allocated_memory_ceiling_fraction_of_physical_gpu_memory', 'planning_authority', 'relative_projected_cost_vs_one_propagation', 'resume_must_reproduce_uninterrupted_arrays_exactly', 'single_gpu_only', 'values_reconstructed_from_bound_gate1_analysis_not_hardcoded_rounded', 'wall_clock_estimates_are_diagnostic_not_gate_values_unless_measured_under_bound_calibration_environment', 'worst_case_projected_peak_bytes']),
    'prohibited': frozenset(['list']),
    'future_step_bindings': frozenset(['description', 'next_step_items']),
}

# ---------------------------------------------------------------------
# Loader.
# ---------------------------------------------------------------------


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str:
        raise DfpPhase1bIdentityError(f"{label} must be an exact string")
    if nonempty and not value.strip():
        raise DfpPhase1bIdentityError(f"{label} must be a non-empty, non-whitespace string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise DfpPhase1bIdentityError(f"{label} must be a lowercase 64-hex-character SHA-256")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token or "\x00" in token:
        raise DfpPhase1bIdentityError(f"{label} must be a safe repository-relative path")
    for part in path.parts:
        if part in ("", "."):
            raise DfpPhase1bIdentityError(f"{label} must not contain empty or '.' path components")
    return token


def _require_no_symlink_escape(root: Path, relative: str, label: str) -> Path:
    resolved_root = root.resolve()
    candidate = root / relative
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError:
        raise DfpPhase1bIdentityError(f"{label} escapes the repository root after symlink resolution: {resolved_candidate}") from None
    return resolved_candidate


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_equal(observed: Any, canonical: Any) -> bool:
    """True iff `observed` matches `canonical` in both exact Python
    type and exact value, recursively through lists (never via bare
    `==`, which would silently accept e.g. `32.0` in place of the
    canonical int `32`, or accept a non-finite float that happens to
    compare unequal-but-not-obviously-so)."""
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
        raise DfpPhase1bIdentityError(f"{label} must be a mapping")
    if set(value) != expected_keys:
        missing = expected_keys - set(value)
        unknown = set(value) - expected_keys
        raise DfpPhase1bIdentityError(f"{label} has an unexpected schema (missing={sorted(missing)}, unknown={sorted(unknown)})")
    return value


def _validate_node(node: Mapping[str, Any], path: str) -> None:
    """Recursively validates one dict node against SECTION_TREE (shape)
    and, for every leaf, against CANONICAL_VALUES (exact type + exact
    value)."""
    expected_keys = SECTION_TREE[path] if path else TOP_KEYS
    _require_closed_mapping(node, expected_keys, f"identity.{path}" if path else "identity")
    for key, value in node.items():
        full_path = f"{path}.{key}" if path else key
        if full_path in SECTION_TREE:
            _validate_node(value, full_path)
        else:
            if full_path not in CANONICAL_VALUES:
                raise DfpPhase1bIdentityError(f"{full_path} has no registered canonical value or sub-schema -- refusing to validate an unrecognized leaf")
            canonical = CANONICAL_VALUES[full_path]
            if not _exact_equal(value, canonical):
                raise DfpPhase1bIdentityError(f"{full_path} disagrees with the frozen canonical value (observed {value!r})")
            # additional format-level checks for path/hash-shaped fields,
            # defense-in-depth beyond exact-value pinning.
            if key == "sha256":
                _require_sha256(value, full_path)
            if key == "relative_path":
                _require_relative_path(value, full_path)


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load and fully closed-schema/exact-type/exact-value validate the
    e15 identity TOML. Does NOT verify any live file hash on disk -- see
    `verify_authority_bindings` for the full relational check."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DfpPhase1bIdentityError(f"cannot load Phase 1B protocol identity {source}: {error}") from error

    if set(identity) != TOP_KEYS:
        missing = TOP_KEYS - set(identity)
        unknown = set(identity) - TOP_KEYS
        raise DfpPhase1bIdentityError(f"identity has an unexpected top-level schema (missing={sorted(missing)}, unknown={sorted(unknown)})")

    format_version = identity.get("format_version")
    if format_version != FORMAT_VERSION:
        raise DfpPhase1bIdentityError(f"format_version disagrees with the frozen canonical value (observed {format_version!r})")

    for section in TOP_KEYS - {"format_version"}:
        _validate_node(identity[section], section)

    return identity


def verify_authority_bindings(*, repo_root: Path | None = None, identity_path: Path | None = None) -> dict[str, str]:
    """Loads AND relationally re-verifies every file-backed authority
    binding's SHA-256 against the live file on disk (the Gate-1
    `gate1_analysis.json` / `analysis_provenance.json` bindings are
    logical-artifact-name entries -- their live re-verification is the
    responsibility of the caller that supplies their actual production
    path, since this module never stores a private /scratch path).
    Every referenced repository-relative path is checked for traversal-
    safety and symlink escape before any file is opened. Never trusts
    an in-memory dict for any of this -- always re-reads from
    `repo_root`."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)

    verified: dict[str, str] = {}
    file_backed = (
        "phase1a_protocol", "phase1a_artifacts", "coco_stuff_500_subset_manifest",
        "boundary_iou_successor_identity", "k12_graph_recurrence", "text_prototype_projection",
        "final_resolution_stitching_pullback_code", "composed_final_resolution_pullback_code",
        "global_unary_component_code", "soft_membership_code",
    )
    for name in file_backed:
        binding = identity["authority_bindings"][name]
        resolved = _require_no_symlink_escape(root, binding["relative_path"], f"authority_bindings.{name}.relative_path")
        if not resolved.is_file():
            raise DfpPhase1bIdentityError(f"authority_bindings.{name}: bound file does not exist: {resolved}")
        observed = sha256_file(resolved)
        if observed != binding["sha256"]:
            raise DfpPhase1bIdentityError(
                f"authority_bindings.{name} SHA256 mismatch: identity declares {binding['sha256']}, observed {observed}"
            )
        verified[name] = observed

    return verified


def compute_resolved_pamr_parameter_digest(binding: Mapping[str, Any]) -> str:
    """Recomputes the PAMR resolved-parameter digest directly from the
    SAME `[rgb_local_boundary_baseline]` table's own already-validated
    fields (never a separately maintained copy) -- see the identity's
    own `resolved_parameter_digest_formula` field for the authoritative
    description of this exact construction. Mechanically self-verifying:
    if any of the frozen descriptive fields the digest is built from is
    mutated, the recomputed digest changes and `verify_pamr_authority_
    bindings` below rejects the mismatch."""
    import json

    params = {
        "operator": "PAMR",
        "num_iter": binding["iterations"],
        "dilations": list(binding["dilations"]),
        "kernel": binding["kernel"],
        "affinity": binding["sigma_or_affinity_parameters"],
        "insertion_point": binding["pipeline_insertion_point"],
        "image_input": binding["input_image_definition"],
        "image_normalization": binding["image_normalization"],
        "resize_interpolation": binding["resize_and_interpolation"],
        "scope": binding["window_or_global_scope"],
        "argmax_order": binding["argmax_order"],
        "determinism": binding["determinism"],
        "training_free": binding["training_free"],
    }
    text = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_pamr_authority_bindings(*, repo_root: Path | None = None, identity_path: Path | None = None) -> dict[str, Any]:
    """Loads AND relationally re-verifies the bound PAMR (RGB/local-
    boundary) authority: the implementation file, the configuration
    file, every inherited/reference configuration file, and the
    resolved-parameter digest -- all against the live files on disk /
    the live identity content, never trusting an in-memory dict. Every
    referenced repository-relative path is checked for traversal-safety
    and symlink escape before any file is opened."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    binding = identity["rgb_local_boundary_baseline"]

    if binding["authority_status"] != "bound":
        raise DfpPhase1bIdentityError(f"rgb_local_boundary_baseline.authority_status is {binding['authority_status']!r}, not 'bound'")

    verified: dict[str, Any] = {}

    impl_path = _require_no_symlink_escape(root, binding["implementation_relative_path"], "rgb_local_boundary_baseline.implementation_relative_path")
    if not impl_path.is_file():
        raise DfpPhase1bIdentityError(f"PAMR implementation file does not exist: {impl_path}")
    observed_impl = sha256_file(impl_path)
    if observed_impl != binding["implementation_sha256"]:
        raise DfpPhase1bIdentityError(
            f"PAMR implementation SHA256 mismatch: identity declares {binding['implementation_sha256']}, observed {observed_impl}"
        )
    verified["implementation_sha256"] = observed_impl

    config_path = _require_no_symlink_escape(root, binding["configuration_relative_path"], "rgb_local_boundary_baseline.configuration_relative_path")
    if not config_path.is_file():
        raise DfpPhase1bIdentityError(f"PAMR configuration file does not exist: {config_path}")
    observed_config = sha256_file(config_path)
    if observed_config != binding["configuration_sha256"]:
        raise DfpPhase1bIdentityError(
            f"PAMR configuration SHA256 mismatch: identity declares {binding['configuration_sha256']}, observed {observed_config}"
        )
    verified["configuration_sha256"] = observed_config

    inherited_paths = binding["inherited_configuration_paths"]
    inherited_hashes = binding["inherited_configuration_sha256s"]
    if len(inherited_paths) != len(inherited_hashes):
        raise DfpPhase1bIdentityError("rgb_local_boundary_baseline inherited_configuration_paths/sha256s length mismatch")
    verified["inherited"] = []
    for rel, expected_sha in zip(inherited_paths, inherited_hashes):
        resolved = _require_no_symlink_escape(root, rel, "rgb_local_boundary_baseline.inherited_configuration_paths[]")
        if not resolved.is_file():
            raise DfpPhase1bIdentityError(f"PAMR inherited configuration file does not exist: {resolved}")
        observed = sha256_file(resolved)
        if observed != expected_sha:
            raise DfpPhase1bIdentityError(
                f"PAMR inherited configuration {rel!r} SHA256 mismatch: identity declares {expected_sha}, observed {observed}"
            )
        verified["inherited"].append(observed)

    recomputed_digest = compute_resolved_pamr_parameter_digest(binding)
    if recomputed_digest != binding["resolved_parameter_digest"]:
        raise DfpPhase1bIdentityError(
            f"PAMR resolved_parameter_digest mismatch: identity declares {binding['resolved_parameter_digest']}, recomputed {recomputed_digest}"
        )
    verified["resolved_parameter_digest"] = recomputed_digest

    return verified


__all__ = [
    "IDENTITY_RELATIVE_PATH",
    "FORMAT_VERSION",
    "TOP_KEYS",
    "SECTION_TREE",
    "CANONICAL_VALUES",
    "DfpPhase1bIdentityError",
    "repository_root",
    "sha256_file",
    "load_identity",
    "verify_authority_bindings",
    "compute_resolved_pamr_parameter_digest",
    "verify_pamr_authority_bindings",
]
