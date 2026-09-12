"""Strict loading/validation for the DFP Phase 1B lightweight Boundary-
Aware Late-Flip Preservation (BALP) successor identity
(`evaluation_identities/e22_dfp_phase1b_lightweight_boundary_preservation.toml`
and its paired artifact-schema authority
`evaluation_identities/e22_dfp_phase1b_lightweight_boundary_artifacts.toml`).

This module is a NEW LEAF authority (E22). It replaces the discontinued
exact masked-forward provenance runner with a lightweight method that
uses only canonical (unmasked) DFP Phase 1A horizon score fields. It
implements no method, runs no inference, selects no candidate, and
performs no torch / CUDA / model / dataset / image access -- pure
TOML / JSON / filesystem / hash validation and exact comparison only.
It does NOT bind E19 (conservation tolerance), E20 (diagnostic runner),
or E21 (provenance-feature kernel) as scientific dependencies -- BALP's
own inference path needs none of them (see [historical_exclusions] in
the preservation TOML).

`KeyboardInterrupt`, `SystemExit`, and `MemoryError` always propagate
unmodified; ordinary parse / type / path / relational failures are
wrapped in `DfpPhase1bLightweightBoundaryIdentityError`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from pathlib import Path
from typing import Any, Mapping

from src.dfp_phase1b_chunk_benchmark_identity import (
    DfpPhase1bChunkBenchmarkIdentityError,
    resolve_repository_relative_path,
)


class DfpPhase1bLightweightBoundaryIdentityError(ValueError):
    """Raised on any E22/BALP identity invariant violation. Always
    fail closed. Never raised for, and never catches, KeyboardInterrupt
    / SystemExit / MemoryError -- those always propagate unmodified."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


PRESERVATION_RELATIVE_PATH = Path("evaluation_identities/e22_dfp_phase1b_lightweight_boundary_preservation.toml")
ARTIFACTS_RELATIVE_PATH = Path("evaluation_identities/e22_dfp_phase1b_lightweight_boundary_artifacts.toml")
SPLIT_RELATIVE_PATH = Path("evaluation_subsets/dfp_phase1b_lightweight_boundary_500_split.json")
SUBSET_MANIFEST_RELATIVE_PATH = Path("evaluation_subsets/dfp_phase1a_coco_stuff_500.json")

PRESERVATION_SCHEMA_NAME = "talk2dino-dfp-phase1b-lightweight-boundary-preservation-identity-v1"
PRESERVATION_SCHEMA_VERSION = 1
ARTIFACTS_SCHEMA_NAME = "talk2dino-dfp-phase1b-lightweight-boundary-artifacts-identity-v1"
ARTIFACTS_SCHEMA_VERSION = 1
REQUIRED_ANCESTOR_COMMIT = "4032fe7330fff981f666097e8addf621b0964ae0"

BALP_SPLIT_NAMESPACE = "talk2dino-dfp-phase1b-balp-development-split-v1"
DEVELOPMENT_COUNT = 350
CONFIRMATION_COUNT = 150
STABILITY_STARTS: tuple[int, ...] = (0, 16, 32, 64)
RADII: tuple[int, ...] = (0, 1, 2, 4)
CANONICAL_CONTROLS: tuple[str, ...] = ("T0", "T16", "T32", "T64", "T320")
HORIZON_SET: tuple[int, ...] = (0, 16, 32, 64, 320)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _require_string(value: Any, label: str) -> str:
    if type(value) is not str:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"{label} must be a string, observed {type(value).__name__}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    s = _require_string(value, label)
    if not _HEX64.match(s):
        raise DfpPhase1bLightweightBoundaryIdentityError(f"{label} must be lowercase 64-hex, observed {value!r}")
    return s


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DfpPhase1bLightweightBoundaryIdentityError(f"{label} must be a table")
    return value


def _require_closed(node: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    node = _require_mapping(node, label)
    try:
        actual = frozenset(node.keys())
    except TypeError as exc:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"{label} has an unhashable key") from exc
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
    if missing or unknown:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"{label} has the wrong key set -- missing={missing} unknown={unknown}")
    return node


def _require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"{label} must be an int (not bool/float), observed {type(value).__name__}: {value!r}")
    return value


def _require_finite_float(value: Any, label: str) -> float:
    if type(value) is not float:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"{label} must be a float, observed {type(value).__name__}: {value!r}")
    if not math.isfinite(value):
        raise DfpPhase1bLightweightBoundaryIdentityError(f"{label} must be finite, observed {value!r}")
    return value


def _require_schema_pair(name: Any, version: Any, expected_name: str, expected_version: int, label: str) -> None:
    """Validates a (schema_name, schema_version) pair with the domain
    exception -- never a raw TypeError/KeyError -- rejecting a
    malformed (unhashable, wrong-typed) value before any comparison."""
    try:
        name_ok = type(name) is str and name == expected_name
    except TypeError as exc:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"{label}.schema_name is malformed/unhashable: {exc}") from exc
    if not name_ok:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"{label}.schema_name must be {expected_name!r}, observed {name!r}")
    try:
        version_ok = type(version) is int and version == expected_version
    except TypeError as exc:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"{label}.schema_version is malformed/unhashable: {exc}") from exc
    if not version_ok:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"{label}.schema_version must be {expected_version!r}, observed {version!r}")


def _walk_leaves(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out.extend(_walk_leaves(v, f"{prefix}.{k}" if prefix else k))
    else:
        out.append((prefix, node))
    return out


# =======================================================================
# Mechanically-flattened exact-value tables for every leaf OUTSIDE
# [dependency_bindings] / [protocol_binding] -- generated once from the
# committed TOML files (never hand-retyped from memory).
# =======================================================================

_CANONICAL_PRESERVATION = {
    'authorization_boundary.balp_presented_as': 'the deployable lightweight method',
    'authorization_boundary.candidate_selected_by_this_identity': False,
    'authorization_boundary.e19_e20_e21_part_of_balp_inference': False,
    'authorization_boundary.e22_authorizes_inference': False,
    'authorization_boundary.exact_dfp_provenance_status': 'discontinued exploratory infrastructure, not deployable',
    'authorization_boundary.exact_masked_forward_prohibited_for': ['development', 'confirmation', 'final_evaluation'],
    'authorization_boundary.full_5000_image_evaluation_authorized': False,
    'authorization_boundary.full_5000_image_evaluation_prerequisites': ['implementation', 'independent_implementation_verification', 'committed_head_verification', 'completed_350_image_development_result', 'independently_verified_selection', 'one_time_150_image_confirmation', 'operational_benchmark_passing', 'later_successor_authorization'],
    'authorization_boundary.gpu_execution_authorized_by_this_identity': False,
    'authorization_boundary.other_4500_image_access_authorized': False,
    'boundary_definition.boundary_formula': "B(y) = {x : at least one in-image 8-connected neighbor x' has y(x') != y(x)}",
    'boundary_definition.computed_at': "ori_shape (final resolution), identical to canonical_horizons.score_field_definition's own resolution",
    'boundary_definition.connectivity': 8,
    'boundary_definition.dilation_formula': 'D_r(B_union) = {x : min_{z in B_union} ||x-z||_inf <= r}',
    'boundary_definition.dilation_metric': 'chebyshev',
    'boundary_definition.edge_pixel_rule': 'an image-edge pixel is a boundary only if an EXISTING in-image neighbor differs; the image exterior contributes no comparison',
    'boundary_definition.erosion_used': False,
    'boundary_definition.gt_boundary_used': False,
    'boundary_definition.ignore_mask_used_in_boundary_construction': False,
    'boundary_definition.image_exterior_is_a_class': False,
    'boundary_definition.label_map_resize_during_boundary_construction': False,
    'boundary_definition.radii': [0, 1, 2, 4],
    'boundary_definition.union_boundary_formula': 'B_union = B(y_64) union B(y_320)',
    'candidate_family.candidate_count': 16,
    'candidate_family.candidate_id_examples': ['balp_s000_r00', 'balp_s000_r01', 'balp_s000_r02', 'balp_s000_r04'],
    'candidate_family.candidate_id_format': 'balp_s{s:03d}_r{r:02d}',
    'candidate_family.candidate_ordering_rule': 'row-major over (stability_starts, radii) in the exact list order given above -- s outer loop ascending [0,16,32,64], r inner loop ascending [0,1,2,4]',
    'candidate_family.canonical_controls.deployment_baseline': 'T320',
    'candidate_family.canonical_controls.list': ['T0', 'T16', 'T32', 'T64', 'T320'],
    'candidate_family.canonical_controls.no_addition_after_results_without_new_successor_authority': True,
    'candidate_family.ordered_candidate_ids.list': ['balp_s000_r00', 'balp_s000_r01', 'balp_s000_r02', 'balp_s000_r04', 'balp_s016_r00', 'balp_s016_r01', 'balp_s016_r02', 'balp_s016_r04', 'balp_s032_r00', 'balp_s032_r01', 'balp_s032_r02', 'balp_s032_r04', 'balp_s064_r00', 'balp_s064_r01', 'balp_s064_r02', 'balp_s064_r04'],
    'candidate_family.radii': [0, 1, 2, 4],
    'candidate_family.stability_starts': [0, 16, 32, 64],
    'candidate_selection.acceptance_gates': ['confirmation_miou_gain_versus_t320_at_least_plus_0_10_pp', 'no_non_finite_metric', 'no_gt_isolation_violation', 'operational_gates_pass'],
    'candidate_selection.acceptance_threshold_pp': 0.1,
    'candidate_selection.any_of_the_above_requires_new_successor_authority': True,
    'candidate_selection.confirmation_evaluation_rule': 'the one frozen winner (if any) is evaluated exactly once on the 150-image internal confirmation split, strictly after development-only selection is finalized',
    'candidate_selection.confirmation_gain_below_threshold_outcome': 'NO_BALP_CANDIDATE_PROCEEDS',
    'candidate_selection.confirmation_metrics_forbidden_for_selection_or_reranking': True,
    'candidate_selection.eligibility_rules': ['scientific_output_complete_and_schema_valid', 'no_gt_isolation_violation', 'finite_metrics', 'development_miou_strictly_exceeds_t320', 'no_recurrence_or_model_duplication', 'candidate_belongs_to_frozen_16_candidate_family'],
    'candidate_selection.no_eligible_development_candidate_outcome': 'NO_BALP_CANDIDATE_PROCEEDS',
    'candidate_selection.post_hoc_candidate_reselection_after_confirmation_forbidden': True,
    'candidate_selection.post_hoc_grid_expansion_forbidden': True,
    'candidate_selection.post_hoc_radii_alteration_forbidden': True,
    'candidate_selection.post_hoc_stability_weakening_forbidden': True,
    'candidate_selection.primary_ranking_order': ['highest_full_precision_development_dataset_level_miou', 'smallest_preservation_fraction', 'smaller_radius', 'longer_stability_requirement_ordered_s0_then_s16_then_s32_then_s64', 'lexicographically_smallest_candidate_id'],
    'candidate_selection.selection_image_domain': 'development (350 images) only',
    'canonical_horizons.argmax_definition': 'y_h(x) = argmax_q S_h(q,x)',
    'canonical_horizons.argmax_tie_rule': 'exact_argmax_ties_select_lowest_class_index',
    'canonical_horizons.argmax_tie_rule_source': "dependency_bindings.e13_protocol [dispute_protocol].tie_rule, identical to dependency_bindings.analysis_module's _lowest_index_argmax -- never a second, independently-derived tie convention",
    'canonical_horizons.argmax_tie_rule_source_key': 'tie_rule',
    'canonical_horizons.argmax_tie_rule_source_section': 'dispute_protocol',
    'canonical_horizons.class_order_source': 'dependency_bindings.e13_protocol [coco_stuff_dataset].class_names_source (mmseg.datasets.COCOStuffDataset.CLASSES), class_count=171, class_names_digest=0d1caff40f22c2f8de07c9d3016f2521dea096ae8b51889a12d3511bc33a18e7',
    'canonical_horizons.final_class_scores_relative_to_refinement': "S_h IS the final class score field BALP consumes; no refinement (PAMR or otherwise) exists between dependency_bindings.final_rescale_module's output and S_h",
    'canonical_horizons.horizon_set': [0, 16, 32, 64, 320],
    'canonical_horizons.horizon_set_source': 'dependency_bindings.e13_protocol [propagation].horizons -- never independently hardcoded',
    'canonical_horizons.ignore_index': 255,
    'canonical_horizons.ignore_index_source': 'dependency_bindings.e13_protocol [coco_stuff_dataset].ignore_index',
    'canonical_horizons.pamr_in_bound_baseline': False,
    'canonical_horizons.pamr_status': "PAMR is explicitly excluded from DFP's own graph/recurrence pipeline (dependency_bindings.matched_evaluator [prohibited] list includes 'pamr'); PAMR appears elsewhere ONLY as a separate historical comparison baseline (T0_PAMR/T320_PAMR in e15-dfp-phase1b-protocol), never as part of the canonical S_h fields BALP consumes. S_h is therefore the canonical (non-PAMR) DFP score field, read at its existing pre-refinement location -- there is no refinement stage between recurrence output and S_h.",
    'canonical_horizons.recurrence_passes_per_image': 1,
    'canonical_horizons.recurrence_passes_per_image_rule': "exactly one canonical k12 320-step recurrence, one model forward, one graph -- ALL FIVE horizon snapshots are read from that single pass via dependency_bindings.recurrence_capture_module's existing snapshot_steps mechanism; recurrence/model/graph must never be re-run per horizon or per BALP candidate",
    'canonical_horizons.score_field_definition': "S_h(q,x): the canonical evaluator-bound class-score field at final ori_shape resolution for horizon h and class q, produced by (1) the same model forward, (2) the same graph, (3) the same single k12 recurrence trajectory, (4) dependency_bindings.recurrence_capture_module's existing horizon capture, (5) dependency_bindings.horizon_stitching_module's exact existing stitching, (6) dependency_bindings.final_rescale_module's exact existing final crop/rescale (R) to ori_shape -- never a GT-derived shape",
    'development_split.confirmation_count': 150,
    'development_split.confirmation_label': 'internal_confirmation_split_not_an_independent_dataset_or_untouched_external_benchmark',
    'development_split.development_count': 350,
    'development_split.encoding': 'utf-8',
    'development_split.future_5000_image_disclosure_required': 'the eventual 5,000-image COCO-Stuff result will include these same 500 images (both development and confirmation subsets), and must explicitly disclose that development/confirmation used a frozen 500-image subset of the same validation universe',
    'development_split.hash_algorithm': 'sha256',
    'development_split.loader_must_reject': ['altered_selection_key', 'altered_rank', 'altered_membership', 'altered_ordering', 'duplicate_image_id', 'omitted_image_id', 'cross_split_overlap', 'wrong_350_150_counts'],
    'development_split.namespace': 'talk2dino-dfp-phase1b-balp-development-split-v1',
    'development_split.other_4500_images_accessed_in_this_task': False,
    'development_split.persisted_fields': ['source_selected_ids_digest', 'development_image_ids', 'development_selection_keys', 'development_ranks', 'confirmation_image_ids', 'confirmation_selection_keys', 'confirmation_ranks', 'development_digest', 'confirmation_digest', 'full_split_trace_digest'],
    'development_split.ranking_direction': 'ascending',
    'development_split.ranking_key': '(selection_key, canonical_image_id)',
    'development_split.selection_key_formula': "sha256(namespace.encode('utf-8') + b'\\x00' + canonical_image_id.encode('utf-8')).hexdigest() -- lowercase 64-hex",
    'development_split.source_subset': 'dependency_bindings.subset_manifest (coco_stuff_500, 500 images) -- the ONLY source of image IDs for this split; no other image is accessed by this identity',
    'development_split.split_rule': 'sort all 500 (selection_key, canonical_image_id) pairs ascending by ranking_key; the first 350 records (ranks 0..349) are development; the remaining 150 (ranks 350..499) are confirmation',
    'format_version': 'talk2dino-dfp-phase1b-lightweight-boundary-preservation-identity-v1',
    'gt_isolation.gt_forbidden_influences': ['horizon_scores', 'predicted_boundaries', 'stability', 'late_flip_detection', 'preservation_masks', 'candidate_score_fields', 'candidate_predictions', 'timing_control_flow'],
    'gt_isolation.gt_permitted_uses': ['confusion_statistic_accumulation', 'development_candidate_selection', 'internal_confirmation', 'final_metric_computation'],
    'gt_isolation.poisoned_gt_verification_definition': 'an implementation/verification task must demonstrate that replacing every GT array with a poisoned (e.g. all-ignore-index, or randomly-permuted) value leaves every horizon score, predicted boundary, stability flag, late-flip flag, preservation mask, candidate score field, and candidate prediction byte-identical -- only confusion statistics/metrics computed AFTER prediction finalization may differ',
    'gt_isolation.poisoned_gt_verification_required': True,
    'gt_isolation.predictions_finalized_before_gt_access_required': True,
    'historical_exclusions.e19_e20_e21_required_by_balp_inference': False,
    'historical_exclusions.list': ["e19-dfp-phase1b-conservation-tolerance: freezes a numeric conservation tolerance for exact masked-forward provenance's rho_g residual family; BALP performs no masked-forward propagation and computes no rho_g residual, so no real dependency exists", 'e20-dfp-phase1b-diagnostic-execution: freezes the 500-image profiled diagnostic RUNNER for exact masked-forward Stage-A extraction, whose multi-window label/margin domain ambiguity is exactly why this identity exists; BALP does not execute, resume, checkpoint through, or otherwise depend on that runner', 'e21-dfp-phase1b-provenance-features: freezes compute_component_evidence_features, the eight component-evidence Stage-A fields (k_star, phi, top1_positive_share, ...); BALP constructs no component, no membership, no signed component contribution, and consumes none of those eight fields', "e15-dfp-phase1b-protocol / e15-dfp-phase1b-artifacts / e16-dfp-phase1b-conservation-calibration: freeze the masked-forward Stage-A/Stage-B schemas and the 200-window conservation calibration; BALP's method, split, and artifact schemas are independent of all of these"],
    'identity.description': 'Freezes the lightweight Boundary-Aware Late-Flip Preservation (BALP) method family over already-frozen DFP Phase 1A canonical horizon score fields: the 16-candidate arbitration-rule family, the deterministic 350/150 development/confirmation split of the frozen coco_stuff_500 subset, candidate-selection and rejection rules, GT-isolation requirements, operational acceptance gates, and the authorization boundary for any future full 5,000-image evaluation. Implements no method, runs no inference, selects no candidate, and authorizes no GPU execution.',
    'identity.mutates_or_version_bumps_earlier_identity': False,
    'identity.name': 'e22-dfp-phase1b-lightweight-boundary-preservation',
    'identity.required_ancestor_commit': '4032fe7330fff981f666097e8addf621b0964ae0',
    'identity.schema_name': 'talk2dino-dfp-phase1b-lightweight-boundary-preservation-identity-v1',
    'identity.schema_version': 1,
    'identity.successor_of': 'none: BALP is a new, independent method family; it is not a version bump of E13/E15/E16/E19/E20/E21',
    'identity.working_branch': 'dfp-phase1a-diagnostics',
    'method.constructs_components': False,
    'method.introduces_learned_parameters': False,
    'method.is_exact_provenance': False,
    'method.is_learned_component_model': False,
    'method.is_oracle': False,
    'method.is_second_propagation_pass': False,
    'method.method_kind': 'inference_time_deterministic_arbitration_rule_over_canonical_horizon_fields',
    'method.performs_pullback_or_membership_computation': False,
    'method.requires_gt_at_inference': False,
    'method.runs_masked_forward_propagation': False,
    'method.scientific_name': 'Boundary-Aware Late-Flip Preservation',
    'method.short_identifier': 'BALP',
    'metric_contract.absent_class_convention': "identical to dependency_bindings.analysis_module's compute_semantic_metrics own convention -- never a new, independently-invented rule",
    'metric_contract.aggregation_level': 'dataset_level_confusion_aggregation',
    'metric_contract.arithmetic_mean_of_per_image_mious_forbidden': True,
    'metric_contract.class_count_and_order_source': 'dependency_bindings.e13_protocol [coco_stuff_dataset]',
    'metric_contract.evaluated_pixel_convention': 'V = (GT != ignore_index), identical to every other bound baseline variant in this repository',
    'metric_contract.ignore_index_source': 'dependency_bindings.e13_protocol [coco_stuff_dataset].ignore_index',
    'metric_contract.new_metric_formula_forbidden': 'no new metric formula is introduced when dependency_bindings.analysis_module already provides one',
    'metric_contract.precision_rule': 'full-precision value selected first (ranking, acceptance-threshold comparison); rounding, if any, is a display-only operation applied strictly after selection',
    'metric_contract.reporting_units': 'percentage_points',
    'metric_contract.required_for': 'every one of the 5 canonical controls and all 16 BALP candidates',
    'metric_contract.required_sufficient_statistics': ['dataset_level_miou', 'per_class_iou', 'mean_class_accuracy_if_already_governed', 'pixel_accuracy_if_already_governed', 'evaluated_pixel_count', 'ignored_pixel_count', 'candidate_versus_t320_changed_pixel_count', 'preservation_mask_pixel_count', 'preservation_mask_fraction', 'boundary_preservation_count', 'non_boundary_preservation_count', 'classwise_gain_loss_versus_t320'],
    'motivating_evidence.boundary_mass_reframe_threshold': 0.7,
    'motivating_evidence.boundary_reframe_trigger_status': 'FIRED',
    'motivating_evidence.does_not_authorize': ['oracle_label_use_at_inference', 'exact_component_provenance_use_at_inference', 'component_mask_use_at_inference', 'gt_use_at_inference'],
    'motivating_evidence.g_oracle_unary_pp': 4.393963458775502,
    'motivating_evidence.g_recover_pp': 2.544658559461869,
    'motivating_evidence.harmful_washout_boundary_fraction_of_harmful': 0.9765451809541299,
    'motivating_evidence.image_count': 500,
    'motivating_evidence.interpretation': "T64 already recovers most of T320's gain over T0 (31.167 vs 31.207 mIoU, i.e. T320-T64 = +0.040pp of the total +1.849pp T0->T320 gain), while unary-oracle headroom (+4.394pp) exceeds even T320's realized gain -- and harmful late-stage washout is overwhelmingly boundary-concentrated (97.65%, far above the 70% reframe threshold). This motivates selectively PRESERVING stable pre-T320 (specifically T64) predictions near predicted semantic boundaries, rather than trusting T320 uniformly. It does NOT authorize using oracle labels, exact provenance, component masks, or GT during inference -- BALP uses only the same canonical (non-oracle, non-GT) horizon score fields DFP Phase 1A already produces.",
    'motivating_evidence.miou_t0': 29.357779515515865,
    'motivating_evidence.miou_t320': 31.2070844148295,
    'motivating_evidence.miou_t64': 31.167212259591796,
    'motivating_evidence.run_mode': 'coco_stuff_500',
    'motivating_evidence.source_artifact_logical_identifier': 'gate1_analysis.json (coco_stuff_500, v2 schema)',
    'motivating_evidence.source_artifact_produced_by_relative_path': 'src/dfp_phase1a_gate1_analysis.py',
    'motivating_evidence.source_artifact_produced_by_sha256': '84cd90ec52bf40052163bb3382e663e58bd2acb941f2efd63fedd6aca5fb7ecb',
    'motivating_evidence.source_artifact_relative_path': 'N/A -- not committed to the repository; lives at /scratch/haree/dfp_phase1a_coco_stuff_500_analysis/coco_stuff_500_gate1_v2/gate1_analysis.json on the host this identity was authored on, exactly like e19-dfp-phase1b-conservation-tolerance binds its own scratch-resident calibration artifact by content hash rather than by committing it',
    'motivating_evidence.source_artifact_sha256': 'bd0cb034d6368e78ca8db19d0d86d3b785d0527fe4aef4c77b841edeb34769dd',
    'motivating_evidence.t320_minus_t0_pp': 1.8493048993136334,
    'operational_contract.acceptance_caps.caps_govern_only': 'whether BALP may proceed to a later full evaluation -- never a claim about the eventual 5,000-image result itself',
    'operational_contract.acceptance_caps.fabricated_missing_telemetry_forbidden': True,
    'operational_contract.acceptance_caps.host_rss_increase_policy': 'report_exactly_no_pass_fail_threshold_unless_already_governed',
    'operational_contract.acceptance_caps.mean_per_image_latency_overhead_pct_max': 15,
    'operational_contract.acceptance_caps.peak_allocated_vram_increase_pct_max': 10,
    'operational_contract.acceptance_caps.peak_reserved_vram_increase_pct_max': 10,
    'operational_contract.acceptance_caps.raw_measurements_required': True,
    'operational_contract.acceptance_caps.throughput_reduction_pct_max': 15,
    'operational_contract.acceptance_caps.total_wall_time_overhead_pct_max': 15,
    'operational_contract.deployed_candidate_requirements': ['reuse_one_model_forward_per_window', 'reuse_one_graph', 'reuse_one_320_step_recurrence', 'capture_existing_horizon_snapshots_in_that_recurrence', 'no_masked_forward_propagation', 'no_connected_component_construction', 'no_pullback_or_membership_computation', 'no_optimization_or_iterative_solve', 'no_cpu_gpu_synchronization_in_scientific_path', 'no_learned_parameters', 'no_gt_at_inference'],
    'operational_contract.formulas.mean_per_image_latency_overhead.baseline_field_must_be_positive': 'baseline_mean_per_image_latency_ns',
    'operational_contract.formulas.mean_per_image_latency_overhead.comparison_operator': '<=',
    'operational_contract.formulas.mean_per_image_latency_overhead.denominator_field': 'baseline_mean_per_image_latency_ns',
    'operational_contract.formulas.mean_per_image_latency_overhead.finite_value_required': True,
    'operational_contract.formulas.mean_per_image_latency_overhead.formula': 'mean_per_image_latency_overhead_pct = 100 * (candidate_mean_per_image_latency_ns - baseline_mean_per_image_latency_ns) / baseline_mean_per_image_latency_ns',
    'operational_contract.formulas.mean_per_image_latency_overhead.negative_value_meaning': 'the candidate had lower per-image latency than the baseline',
    'operational_contract.formulas.mean_per_image_latency_overhead.numerator_field': 'candidate_mean_per_image_latency_ns',
    'operational_contract.formulas.mean_per_image_latency_overhead.subtrahend_field': 'baseline_mean_per_image_latency_ns',
    'operational_contract.formulas.mean_per_image_latency_overhead.threshold': 15.0,
    'operational_contract.formulas.mean_per_image_latency_overhead.units': 'percent',
    'operational_contract.formulas.peak_allocated_vram_increase.baseline_field_must_be_positive': 'baseline_peak_allocated_vram_bytes',
    'operational_contract.formulas.peak_allocated_vram_increase.comparison_operator': '<=',
    'operational_contract.formulas.peak_allocated_vram_increase.denominator_field': 'baseline_peak_allocated_vram_bytes',
    'operational_contract.formulas.peak_allocated_vram_increase.finite_value_required': True,
    'operational_contract.formulas.peak_allocated_vram_increase.formula': 'peak_allocated_vram_increase_pct = 100 * (candidate_peak_allocated_vram_bytes - baseline_peak_allocated_vram_bytes) / baseline_peak_allocated_vram_bytes',
    'operational_contract.formulas.peak_allocated_vram_increase.negative_value_meaning': 'the candidate used less peak allocated VRAM than the baseline',
    'operational_contract.formulas.peak_allocated_vram_increase.numerator_field': 'candidate_peak_allocated_vram_bytes',
    'operational_contract.formulas.peak_allocated_vram_increase.subtrahend_field': 'baseline_peak_allocated_vram_bytes',
    'operational_contract.formulas.peak_allocated_vram_increase.threshold': 10.0,
    'operational_contract.formulas.peak_allocated_vram_increase.units': 'percent',
    'operational_contract.formulas.peak_reserved_vram_increase.baseline_field_must_be_positive': 'baseline_peak_reserved_vram_bytes',
    'operational_contract.formulas.peak_reserved_vram_increase.comparison_operator': '<=',
    'operational_contract.formulas.peak_reserved_vram_increase.denominator_field': 'baseline_peak_reserved_vram_bytes',
    'operational_contract.formulas.peak_reserved_vram_increase.finite_value_required': True,
    'operational_contract.formulas.peak_reserved_vram_increase.formula': 'peak_reserved_vram_increase_pct = 100 * (candidate_peak_reserved_vram_bytes - baseline_peak_reserved_vram_bytes) / baseline_peak_reserved_vram_bytes',
    'operational_contract.formulas.peak_reserved_vram_increase.negative_value_meaning': 'the candidate used less peak reserved VRAM than the baseline',
    'operational_contract.formulas.peak_reserved_vram_increase.numerator_field': 'candidate_peak_reserved_vram_bytes',
    'operational_contract.formulas.peak_reserved_vram_increase.subtrahend_field': 'baseline_peak_reserved_vram_bytes',
    'operational_contract.formulas.peak_reserved_vram_increase.threshold': 10.0,
    'operational_contract.formulas.peak_reserved_vram_increase.units': 'percent',
    'operational_contract.formulas.throughput_reduction.baseline_field_must_be_positive': 'baseline_throughput_images_per_second',
    'operational_contract.formulas.throughput_reduction.comparison_operator': '<=',
    'operational_contract.formulas.throughput_reduction.denominator_field': 'baseline_throughput_images_per_second',
    'operational_contract.formulas.throughput_reduction.finite_value_required': True,
    'operational_contract.formulas.throughput_reduction.formula': 'throughput_reduction_pct = 100 * (baseline_throughput_images_per_second - candidate_throughput_images_per_second) / baseline_throughput_images_per_second',
    'operational_contract.formulas.throughput_reduction.negative_value_meaning': 'the candidate had higher throughput than the baseline',
    'operational_contract.formulas.throughput_reduction.numerator_field': 'baseline_throughput_images_per_second',
    'operational_contract.formulas.throughput_reduction.subtrahend_field': 'candidate_throughput_images_per_second',
    'operational_contract.formulas.throughput_reduction.threshold': 15.0,
    'operational_contract.formulas.throughput_reduction.units': 'percent',
    'operational_contract.formulas.total_wall_time_overhead.baseline_field_must_be_positive': 'baseline_total_wall_time_ns',
    'operational_contract.formulas.total_wall_time_overhead.comparison_operator': '<=',
    'operational_contract.formulas.total_wall_time_overhead.denominator_field': 'baseline_total_wall_time_ns',
    'operational_contract.formulas.total_wall_time_overhead.finite_value_required': True,
    'operational_contract.formulas.total_wall_time_overhead.formula': 'total_wall_time_overhead_pct = 100 * (candidate_total_wall_time_ns - baseline_total_wall_time_ns) / baseline_total_wall_time_ns',
    'operational_contract.formulas.total_wall_time_overhead.negative_value_meaning': 'the candidate was faster than the baseline',
    'operational_contract.formulas.total_wall_time_overhead.numerator_field': 'candidate_total_wall_time_ns',
    'operational_contract.formulas.total_wall_time_overhead.subtrahend_field': 'baseline_total_wall_time_ns',
    'operational_contract.formulas.total_wall_time_overhead.threshold': 15.0,
    'operational_contract.formulas.total_wall_time_overhead.units': 'percent',
    'operational_contract.host_rss_policy.absolute_difference_field': 'host_rss_absolute_difference_bytes',
    'operational_contract.host_rss_policy.absolute_difference_formula': 'host_rss_absolute_difference_bytes = candidate_host_rss_bytes - baseline_host_rss_bytes',
    'operational_contract.host_rss_policy.must_record_baseline_and_candidate': True,
    'operational_contract.host_rss_policy.pass_fail_threshold_authorized': False,
    'operational_contract.host_rss_policy.percentage_difference_field': 'host_rss_percentage_difference_pct',
    'operational_contract.host_rss_policy.percentage_difference_field_nullable_if_baseline_not_positive': True,
    'operational_contract.host_rss_policy.percentage_difference_formula': 'host_rss_percentage_difference_pct = 100 * (candidate_host_rss_bytes - baseline_host_rss_bytes) / baseline_host_rss_bytes',
    'operational_contract.host_rss_policy.percentage_difference_requires_positive_baseline': True,
    'operational_contract.numeric_rules.arithmetic_dtype': 'float64',
    'operational_contract.numeric_rules.comparison_operator_all_caps': '<=',
    'operational_contract.numeric_rules.comparison_operator_strict_less_than_forbidden': True,
    'operational_contract.numeric_rules.denominator_authority': 'same_job_t320_control',
    'operational_contract.numeric_rules.display_rounding_never_affects_acceptance': True,
    'operational_contract.numeric_rules.historical_1h18m_runtime_status': 'contextual_only_never_authoritative',
    'operational_contract.numeric_rules.negative_overhead_or_reduction_is_a_valid_result': True,
    'operational_contract.numeric_rules.raw_measurements_retained': True,
    'operational_contract.numeric_rules.reject_bool_as_measurement': True,
    'operational_contract.numeric_rules.reject_infinity': True,
    'operational_contract.numeric_rules.reject_nan': True,
    'operational_contract.numeric_rules.reject_non_positive_baseline': True,
    'operational_contract.numeric_rules.reject_null_as_measurement': True,
    'operational_contract.numeric_rules.reject_string_as_measurement': True,
    'operational_contract.numeric_rules.units_are_percent_not_fraction': True,
    'operational_contract.paired_benchmark_comparison': ['canonical_T320_baseline', 'selected_BALP_candidate'],
    'operational_contract.paired_benchmark_conditions': 'same GPU and software environment, same frozen 500-image order, same model/checkpoint, same process-isolation/warm-up policy',
    'operational_contract.paired_benchmark_required': True,
    'prohibited.list': ['exact_masked_forward_propagation', 'connected_component_construction', 'component_membership_computation', 'pullback_computation', 'signed_component_contribution_computation', 'e19_tolerance_dependency', 'e20_runner_dependency', 'e21_feature_kernel_dependency', 'gt_at_inference', 'oracle_label_at_inference', 'blending_coefficient', 'confidence_threshold', 'temperature_parameter', 'learned_weight', 'random_tie_break', 'adaptive_iteration_count', 'second_recurrence_pass', 'per_candidate_model_rerun', 'candidate_selection_in_this_identity', 'gpu_execution_in_this_identity', 'full_5000_image_execution'],
    'prohibited_method_features.list': ['blending_coefficient', 'confidence_threshold', 'temperature', 'learned_weight', 'random_choice', 'component_feature', 'adaptive_iteration_count', 'masked_forward_propagation', 'component_construction', 'component_membership', 'pullback_computation', 'signed_component_contribution', 'second_recurrence_pass', 'per_candidate_recurrence_rerun', 'gt_input_at_inference'],
    'stability_and_late_flip.candidate_score_field_formula': 'S_BALP_{s,r}(:,x) = S_64(:,x) when P_{s,r}(x) is true, otherwise S_320(:,x)',
    'stability_and_late_flip.final_prediction_formula': 'y_BALP_{s,r}(x) = argmax_q S_BALP_{s,r}(q,x)',
    'stability_and_late_flip.late_flip_formula': 'late_flip(x) = [y_64(x) != y_320(x)]',
    'stability_and_late_flip.preservation_mask_formula': 'P_{s,r}(x) = stable_s(x) AND late_flip(x) AND [x in D_r(B_union)]',
    'stability_and_late_flip.preserved_label_source': 'the preserved label is always the stable T64 label (y_64(x)), never a blend, average, or recomputed value',
    'stability_and_late_flip.relational_equivalence_required': 'y_BALP_{s,r}(x) == y_64(x) when P_{s,r}(x) is true, and y_BALP_{s,r}(x) == y_320(x) otherwise -- the identity loader/tests must independently verify this equivalence, never merely trust the argmax-of-selected-field formula',
    'stability_and_late_flip.stability_start_horizons': [0, 16, 32, 64],
    'stability_and_late_flip.stability_start_horizons_source': 'S = [0,16,32,64] -- the four canonical_horizons.horizon_set members strictly preceding 320',
    'stability_and_late_flip.stable_formula': 'stable_s(x) = true iff all y_h(x), h in suffix(s), are identical',
    'stability_and_late_flip.suffix_definition': 'suffix(s): the ordered subset of [0,16,32,64] beginning at s (e.g. suffix(16) = [16,32,64])',
}

_CANONICAL_ARTIFACTS = {
    'artifact_write_policy.applies_to': ['development_checkpoint', 'development_result', 'selection_artifact', 'profiling_artifact', 'future_full_evaluation_result'],
    'artifact_write_policy.atomic_installation.multi_file_artifacts.applies_to': ['development_result', 'selection_artifact', 'profiling_artifact', 'future_full_evaluation_result', "a finalized development_checkpoint's result bundle"],
    'artifact_write_policy.atomic_installation.multi_file_artifacts.destination_must_be_absent_before_install': True,
    'artifact_write_policy.atomic_installation.multi_file_artifacts.install_operation': 'exactly one atomic directory rename (os.rename/os.replace) from the staging directory into an ABSENT final destination path',
    'artifact_write_policy.atomic_installation.multi_file_artifacts.staging_unit': 'one staging DIRECTORY containing every member file of the generation',
    'artifact_write_policy.atomic_installation.multi_file_artifacts.unguarded_sequential_final_renames_forbidden': True,
    'artifact_write_policy.atomic_installation.post_install_verification.applies_to': ['multi_file_artifacts', 'single_file_artifacts'],
    'artifact_write_policy.atomic_installation.post_install_verification.both_pre_install_and_post_install_verification_mandatory': True,
    'artifact_write_policy.atomic_installation.post_install_verification.cleanup_on_post_install_verification_failure': "reused_from_e20_atomic_write_stage_a_npz: NONE -- the newly-installed (already-renamed) artifact is left on disk exactly as installed, for forensic inspection; the exception propagates straight through with no automatic delete/quarantine/rollback of the just-installed bytes. This is a precise statement of E20's own actual behavior, not a stricter fabricated deletion policy.",
    'artifact_write_policy.atomic_installation.post_install_verification.downstream_advance_forbidden_before_post_install_verification_passes': ['current_generation_pointer', 'checkpoint.final_result_binding', 'checkpoint.final', 'selection_artifact.selected_candidate', 'any_current_or_latest_artifact_reference'],
    'artifact_write_policy.atomic_installation.post_install_verification.member_hashes_recomputed_from_installed_bytes': True,
    'artifact_write_policy.atomic_installation.post_install_verification.overwrite_authorized_by_this_requirement': False,
    'artifact_write_policy.atomic_installation.post_install_verification.post_install_failure_never_silently_converted_to_success': True,
    'artifact_write_policy.atomic_installation.post_install_verification.post_install_failure_reported_as_installation_failure': True,
    'artifact_write_policy.atomic_installation.post_install_verification.public_validator_required_against_installed_bytes': True,
    'artifact_write_policy.atomic_installation.post_install_verification.reopen_and_reverify_after_install_required': True,
    'artifact_write_policy.atomic_installation.post_install_verification.reopen_source': 'the installed destination path only -- never the staging/temporary path, which no longer exists (multi-file: renamed away; single-file: replaced/unlinked by os.replace)',
    'artifact_write_policy.atomic_installation.post_install_verification.reopen_timing': 'strictly after the atomic rename/replace has completed -- never before, and never substituting for the pre-install staged self-verification',
    'artifact_write_policy.atomic_installation.post_install_verification.replaces_pre_install_staged_verification': False,
    'artifact_write_policy.atomic_installation.post_install_verification.sibling_bindings_recomputed_from_installed_bytes': True,
    'artifact_write_policy.atomic_installation.post_install_verification.success_reported_only_after_post_install_verification_passes': True,
    'artifact_write_policy.atomic_installation.single_file_artifacts.applies_to': ['a checkpoint pointer/manifest file that is legitimately replaceable across generations'],
    'artifact_write_policy.atomic_installation.single_file_artifacts.install_operation': 'exactly one atomic os.replace of the validated temporary file onto the final path',
    'artifact_write_policy.atomic_installation.single_file_artifacts.prior_valid_file_preserved_on_validation_failure': True,
    'artifact_write_policy.atomic_installation.single_file_artifacts.staging_unit': 'one same-directory temporary file (same directory as the final destination, so the replace is on the same filesystem)',
    'artifact_write_policy.atomic_installation.single_file_artifacts.validate_before_replace_required': True,
    'artifact_write_policy.failure_behavior.existing_valid_artifacts_byte_identical_after_failed_install': True,
    'artifact_write_policy.failure_behavior.failure_before_final_rename_leaves_prior_state_untouched': True,
    'artifact_write_policy.failure_behavior.fatal_exceptions_propagate_after_cleanup': True,
    'artifact_write_policy.failure_behavior.interruption_between_computation_and_commit_reprocesses_at_most_one_image': 'since checkpoints commit after every completed development image (never mid-image), an interruption can cause at most the single in-flight (uncommitted) image to be reprocessed on resume -- never a committed one',
    'artifact_write_policy.failure_behavior.partial_generation_can_never_become_current': True,
    'artifact_write_policy.failure_behavior.rollback_failure_must_be_reported_explicitly': 'a failure during staging-residue cleanup is itself surfaced (never silently swallowed) -- it must never be reported as if the original write had succeeded, and must never be reported as if no error occurred',
    'artifact_write_policy.failure_behavior.schema_or_digest_failure_blocks_advancement': True,
    'artifact_write_policy.failure_behavior.source_artifacts_never_mutated': True,
    'artifact_write_policy.failure_behavior.staging_residue_removed_on_failure': True,
    'artifact_write_policy.freshness_and_overwrite.development_result_install_requires_absent_destination': True,
    'artifact_write_policy.freshness_and_overwrite.e22_grants_destructive_overwrite': False,
    'artifact_write_policy.freshness_and_overwrite.future_full_evaluation_result_install_requires_absent_destination': True,
    'artifact_write_policy.freshness_and_overwrite.immutable_checkpoint_generations_never_overwritten': True,
    'artifact_write_policy.freshness_and_overwrite.profiling_artifact_install_requires_absent_destination': True,
    'artifact_write_policy.freshness_and_overwrite.rerun_requires_new_output_destination_unless_later_authority_permits_replacement': True,
    'artifact_write_policy.freshness_and_overwrite.selection_artifact_install_requires_absent_destination': True,
    'artifact_write_policy.freshness_and_overwrite.temporary_staging_or_backup_entries_not_accepted_as_final_artifacts': True,
    'artifact_write_policy.general_path_policy.e22_grants_overwrite_authority': False,
    'artifact_write_policy.general_path_policy.filename_must_match_schema_name': True,
    'artifact_write_policy.general_path_policy.implicit_cwd_output_forbidden': True,
    'artifact_write_policy.general_path_policy.lexical_traversal_forbidden': True,
    'artifact_write_policy.general_path_policy.output_destination_must_be_explicit': True,
    'artifact_write_policy.general_path_policy.overwrite_by_default_forbidden': True,
    'artifact_write_policy.general_path_policy.overwrite_requires_separate_explicit_authority': True,
    'artifact_write_policy.general_path_policy.physical_containment_required': 'resolved (symlink-chased) target must remain inside the resolved (symlink-chased) authorized output root',
    'artifact_write_policy.general_path_policy.source_artifacts_are_read_only': True,
    'artifact_write_policy.general_path_policy.symlink_escape_forbidden': True,
    'artifact_write_policy.generation_protocol.design': "immutable generation directories plus an atomically-updated latest-generation pointer -- identical in spirit to e20-dfp-phase1b-diagnostic-execution's checkpoint-after-every-completed-image discipline, reused here because it is already proven safe under interruption in this repository and introduces no new safety mechanism",
    'artifact_write_policy.generation_protocol.immutable_generation_directory': 'each installed generation (e.g. each finalized checkpoint/result install) is written once, under its own generation identifier, and is NEVER modified or overwritten after a successful atomic install',
    'artifact_write_policy.generation_protocol.latest_pointer_update': "the 'current generation' pointer (if used) is itself updated via the single-file atomic os.replace rule above -- never by editing a pointer file in place",
    'artifact_write_policy.generation_protocol.why_this_protocol': "avoids ANY window where a reader can observe a partially-written generation, and avoids ANY window where a failed write leaves the 'current' pointer referencing a nonexistent or partial generation -- the same guarantee E20's own already-committed, already-exercised atomic_write_checkpoint/atomic_install_result provide",
    'artifact_write_policy.reused_pattern_source': "the already-proven staged-directory-then-atomic-rename pattern established by e20-dfp-phase1b-diagnostic-execution's atomic_write_stage_a_npz/atomic_install_result (stage under an unpredictable temp name on the SAME filesystem, self-verify via the unmodified schema validator, then os.replace/os.rename exactly once) -- reused by convention, never by code dependency (E22 does not import E20)",
    'artifact_write_policy.sibling_digest_bindings.digest_mismatch_behavior': 'reject: a sibling digest-binding mismatch is a fatal schema/relational violation, identical in severity to a closed-schema key-set violation -- never a warning',
    'artifact_write_policy.sibling_digest_bindings.list': ["checkpoint JSON metadata <-> checkpoint per-image/aggregate arrays: identity_bindings.development_digest and the checkpoint's own recorded array-derived hashes must agree", "development_result <-> its source (finalized) checkpoint: development_result_schema.identity_bindings (or equivalent) must record the exact finalized checkpoint's own generation/hash it was produced from", 'selection_artifact <-> development_result and confirmation_result: selection_schema.development_source_hashes must match the actual installed development_result bytes; confirmation_result must reference the same selected candidate identity', 'profiling_artifact <-> the selected candidate: profiling_schema records must be keyed to the SAME candidate_id the selection_artifact names as selected -- never a different or unselected candidate', 'future_full_evaluation_result <-> the selected E22 successor identity: upstream_hashes must include e22_protocol_sha256/e22_artifacts_sha256 of the EXACT E22 version whose selection_artifact authorized proceeding'],
    'artifact_write_policy.staging.flush_and_close_before_validation': True,
    'artifact_write_policy.staging.member_hashes_computed_from_staged_bytes': True,
    'artifact_write_policy.staging.one_generation_per_staging_unit': 'every member (JSON/NPZ/etc.) belonging to ONE logical generation is written into the SAME staging unit before any validation begins',
    'artifact_write_policy.staging.partial_final_bundle_forbidden': "the final destination must never contain a subset of a generation's members -- either all members are installed or none are",
    'artifact_write_policy.staging.self_verification_before_installation_required': True,
    'artifact_write_policy.staging.self_verification_scope': 'closed-schema validation, relational validation (checkpoint_schema.relational_validation / cross_generation_invariants where applicable), and sibling digest-binding validation, ALL run against staged contents before the final install step begins',
    'artifact_write_policy.staging.sibling_manifest_hashes_source': 'staged bytes only -- never a hash computed before the staged write completed, and never a hash of the eventual final-location bytes before they exist',
    'artifact_write_policy.staging.staging_location': 'a sibling of the final destination on the SAME filesystem (never a different filesystem/device, so the final install step can be a single atomic rename)',
    'artifact_write_policy.staging.staging_name_uniqueness': 'an unpredictable, collision-resistant staging name (e.g. a random hex suffix), never a fixed/reused temp name',
    'checkpoint_schema.aggregate_confusion.keys.description': "one confusion_sufficient_statistics_member per control (5) and per candidate (16), keyed by shared.control_ids/shared.candidate_ids -- exactly 21 arrays, append-only: each completed image's own confusion matrix is added in place, never replacing a prior partial sum",
    'checkpoint_schema.aggregate_confusion.keys.required_candidate_keys': ['balp_s000_r00', 'balp_s000_r01', 'balp_s000_r02', 'balp_s000_r04', 'balp_s016_r00', 'balp_s016_r01', 'balp_s016_r02', 'balp_s016_r04', 'balp_s032_r00', 'balp_s032_r01', 'balp_s032_r02', 'balp_s032_r04', 'balp_s064_r00', 'balp_s064_r01', 'balp_s064_r02', 'balp_s064_r04'],
    'checkpoint_schema.aggregate_confusion.keys.required_control_keys': ['T0', 'T16', 'T32', 'T64', 'T320'],
    'checkpoint_schema.cross_generation_invariants.description': 'rules relating an earlier checkpoint C_old to a later checkpoint C_new for the SAME development run (same identity_bindings)',
    'checkpoint_schema.cross_generation_invariants.list': ['C_new.progress.next_development_index >= C_old.progress.next_development_index', 'C_old.progress.completed_image_ids is an exact prefix of C_new.progress.completed_image_ids (never reordered, replaced, or reinterpreted)', 'for every completed image i < C_old.progress.next_development_index and every control/candidate key k, C_new.per_image_reconstruction_statistics[k][i] is byte-identical to C_old.per_image_reconstruction_statistics[k][i] (both the 6 scalar fields and per_image_confusion_matrix)', "C_new's candidate_family/control ordering and identity_bindings are byte-identical to C_old's", 'for every control/candidate key k, C_new.aggregate_confusion[k] == C_old.aggregate_confusion[k] plus the elementwise sum of per_image_confusion_matrix[i] for the newly-appended i in range(C_old.next_development_index, C_new.next_development_index)', 'resume never removes, reorders, replaces, or reinterprets a row C_old already completed', 'profiling_schema fields never appear inside a checkpoint document and can never affect scientific_output_digest (development_result_schema) or any aggregate_confusion/per_image value', 'a fresh run and a checkpoint-then-resumed run of the SAME development population produce byte-identical final (next_development_index==350) aggregate_confusion and per_image_reconstruction_statistics'],
    'checkpoint_schema.final_result_binding.keys.list': ['development_result_relative_path', 'development_result_sha256', 'installed_at_generation'],
    'checkpoint_schema.flags.complete_definition': 'next_development_index == 350 AND completed_image_ids == development_image_ids (the full frozen 350-image development order)',
    'checkpoint_schema.flags.final_definition': 'complete == true AND final_result_binding is present and valid',
    'checkpoint_schema.flags.final_implies_complete': True,
    'checkpoint_schema.flags.incomplete_cannot_bind_final_result': 'final_result_binding must be absent (or null) whenever complete == false; presence of a valid final_result_binding while complete == false is a fatal schema violation',
    'checkpoint_schema.identity_bindings.keys.list': ['e22_protocol_sha256', 'e22_artifacts_sha256', 'e13_protocol_sha256', 'matched_evaluator_sha256', 'subset_manifest_sha256', 'balp_split_sha256', 'development_digest', 'run_mode'],
    'checkpoint_schema.per_image_array_convention.completion_order_is_development_split_order': "completed_image_ids is always development_image_ids[:next_development_index] (development_split's own ranked order) -- position i in every per-image container corresponds to development_image_ids[i], never filesystem/dict/insertion order",
    'checkpoint_schema.per_image_array_convention.container_shape': 'dict keyed by canonical_image_id -- NOT a flat array with a separate offset array',
    'checkpoint_schema.per_image_array_convention.exact_length_invariant': 'for every control/candidate key, the per-image container holds EXACTLY next_development_index entries, one per completed image, and the entry for completed_image_ids[i] is populated if and only if i < next_development_index',
    'checkpoint_schema.per_image_array_convention.offset_array_needed': False,
    'checkpoint_schema.per_image_array_convention.offset_array_not_needed_reason': "every per_image_reconstruction_statistics member is exactly ONE scalar (or one [class_count,class_count] matrix) per completed image per control/candidate -- there is no variable-length per-image segment the way E15's Stage-A disputed-pixel rows require dispute_offsets, so there is nothing for an offset array to index",
    'checkpoint_schema.per_image_reconstruction_statistics.keys.list': ['candidate_versus_t320_changed_pixel_count', 'preservation_mask_pixel_count', 'boundary_preservation_pixel_count', 'non_boundary_preservation_pixel_count', 'evaluated_pixel_count', 'ignored_pixel_count', 'per_image_confusion_matrix'],
    'checkpoint_schema.per_image_reconstruction_statistics.keys.per_image_confusion_matrix_purpose': "the [class_count, class_count] confusion matrix for THIS ONE completed image alone (never a running sum) -- persisted specifically so checkpoint_schema.relational_validation's aggregate-equals-sum-of-per-image rule is independently checkable from persisted bytes alone, never re-derived by re-running inference",
    'checkpoint_schema.per_image_reconstruction_statistics.keys.required_for': 'every control (T64/T320 relevant subset) and every one of the 16 candidates, per completed image, accumulated append-only exactly like aggregate_confusion',
    'checkpoint_schema.per_image_scientific_files_required': False,
    'checkpoint_schema.prefer_append_only_aggregate_arrays': True,
    'checkpoint_schema.progress.keys.expected_development_image_count_value': 350,
    'checkpoint_schema.progress.keys.list': ['expected_development_image_count', 'completed_image_ids', 'next_development_index', 'generation'],
    'checkpoint_schema.relational_validation.list': ['progress.expected_development_image_count == 350', '0 <= progress.next_development_index <= 350', 'len(progress.completed_image_ids) == progress.next_development_index', 'progress.completed_image_ids == balp_split.development_image_ids[:progress.next_development_index] (exact frozen-split prefix, never a re-sorted or re-selected subset)', 'no duplicate id in progress.completed_image_ids', 'sorted(aggregate_confusion.keys()) covers exactly shared.control_ids union shared.candidate_ids (5 + 16 = 21 keys, no more, no fewer)', 'candidate keys in aggregate_confusion/per_image_reconstruction_statistics appear in exactly candidate_family.ordered_candidate_ids order', 'control keys in aggregate_confusion/per_image_reconstruction_statistics appear in exactly shared.control_ids order', 'identity_bindings.e22_protocol_sha256/e22_artifacts_sha256/e13_protocol_sha256/matched_evaluator_sha256/subset_manifest_sha256/balp_split_sha256/development_digest all equal the identical fields E22 itself pins (never independently re-chosen per checkpoint)', 'for every control/candidate key k and every completed image i < next_development_index, per_image_reconstruction_statistics[k] has exactly next_development_index entries (checkpoint_schema.per_image_array_convention.exact_length_invariant)', 'aggregate_confusion[k] == elementwise sum over i in range(next_development_index) of per_image_reconstruction_statistics[k].per_image_confusion_matrix[i], for every control/candidate key k', 'evaluated_pixel_count and ignored_pixel_count for a given completed image are identical across every control and every candidate (same image, same GT, same ignore mask -- prediction differences never change which pixels are evaluated/ignored)', 'every candidate key present anywhere in the checkpoint is a member of candidate_family.ordered_candidate_ids -- no foreign candidate id', 'no id in progress.completed_image_ids appears in balp_split.confirmation_image_ids', 'no representable state has completed_image_ids that is not a prefix of balp_split.development_image_ids (state_machine is exhaustive and closed)'],
    'checkpoint_schema.resume_never_changes_image_order_or_candidate_outputs': True,
    'checkpoint_schema.schema_name': 'talk2dino-dfp-phase1b-balp-development-checkpoint-v1',
    'checkpoint_schema.schema_version': 1,
    'checkpoint_schema.state_machine.complete_unfinalized.complete': True,
    'checkpoint_schema.state_machine.complete_unfinalized.completed_image_ids': 'exactly all 350 development_image_ids, in order',
    'checkpoint_schema.state_machine.complete_unfinalized.final': False,
    'checkpoint_schema.state_machine.complete_unfinalized.final_result_binding': 'absent',
    'checkpoint_schema.state_machine.complete_unfinalized.next_development_index': 350,
    'checkpoint_schema.state_machine.finalized.complete': True,
    'checkpoint_schema.state_machine.finalized.final': True,
    'checkpoint_schema.state_machine.finalized.final_result_binding': 'present and valid (sha256 of an existing, schema-valid development_result_schema document)',
    'checkpoint_schema.state_machine.finalized.next_development_index': 350,
    'checkpoint_schema.state_machine.partial_progress.complete': False,
    'checkpoint_schema.state_machine.partial_progress.final': False,
    'checkpoint_schema.state_machine.partial_progress.next_development_index': '0 < next_development_index < 350',
    'checkpoint_schema.state_machine.zero_progress.complete': False,
    'checkpoint_schema.state_machine.zero_progress.completed_image_ids': 'empty list',
    'checkpoint_schema.state_machine.zero_progress.final': False,
    'checkpoint_schema.state_machine.zero_progress.next_development_index': 0,
    'checkpoint_schema.supports_fresh_resume_without_reordering': True,
    'checkpoint_schema.top_keys.closed': True,
    'checkpoint_schema.top_keys.list': ['schema_name', 'schema_version', 'identity_bindings', 'progress', 'aggregate_confusion', 'per_image_reconstruction_statistics', 'complete', 'final', 'final_result_binding'],
    'development_result_schema.candidate_result_member.keys.candidate_id_closed_set': 'shared.candidate_ids',
    'development_result_schema.candidate_result_member.keys.list': ['candidate_id', 'stability_start', 'radius', 'dataset_level_miou', 'per_class_iou', 'mean_class_accuracy', 'pixel_accuracy', 'evaluated_pixel_count', 'ignored_pixel_count', 'candidate_versus_t320_changed_pixel_count', 'preservation_mask_pixel_count', 'preservation_mask_fraction', 'boundary_preservation_pixel_count', 'non_boundary_preservation_pixel_count', 'classwise_gain_loss_versus_t320'],
    'development_result_schema.candidate_result_member.keys.radius_closed_set': [0, 1, 2, 4],
    'development_result_schema.candidate_result_member.keys.stability_start_closed_set': [0, 16, 32, 64],
    'development_result_schema.control_result_member.keys.list': ['control_id', 'dataset_level_miou', 'per_class_iou', 'mean_class_accuracy', 'pixel_accuracy', 'evaluated_pixel_count', 'ignored_pixel_count'],
    'development_result_schema.schema_name': 'talk2dino-dfp-phase1b-balp-development-result-v1',
    'development_result_schema.schema_version': 1,
    'development_result_schema.top_keys.development_image_count_required_value': 350,
    'development_result_schema.top_keys.list': ['schema_name', 'schema_version', 'run_mode', 'development_image_count', 'identity_bindings', 'control_results', 'candidate_results', 'scientific_output_digest', 'complete', 'final'],
    'format_version': 'talk2dino-dfp-phase1b-lightweight-boundary-artifacts-identity-v1',
    'future_full_evaluation_schema.development_subset_disclosure.required_content': 'must state that the frozen 500-image coco_stuff_500 subset (both the 350-image development and 150-image internal-confirmation partitions) is included within the 5,000-image evaluation universe, and was used for BALP candidate development/confirmation prior to this full run',
    'future_full_evaluation_schema.development_subset_disclosure.required_field': 'development_subset_disclosure_text',
    'future_full_evaluation_schema.overhead_versus_historical_baseline.authoritative': False,
    'future_full_evaluation_schema.overhead_versus_historical_baseline.description': 'contextual only -- never authoritative for pass/fail',
    'future_full_evaluation_schema.overhead_versus_historical_baseline.historical_baseline_wall_time_reference': 'approximately 1h18m',
    'future_full_evaluation_schema.overhead_versus_same_job_t320_control.authoritative': True,
    'future_full_evaluation_schema.overhead_versus_same_job_t320_control.description': 'the same-job, same-environment T320 control measured in the SAME full-evaluation run is the authoritative overhead comparison',
    'future_full_evaluation_schema.schema_freezing_authorizes_execution': False,
    'future_full_evaluation_schema.schema_name': 'talk2dino-dfp-phase1b-balp-full-evaluation-result-v1',
    'future_full_evaluation_schema.schema_version': 1,
    'future_full_evaluation_schema.top_keys.image_count_required_value': 5000,
    'future_full_evaluation_schema.top_keys.list': ['schema_name', 'schema_version', 'image_count', 't320_control_aggregate_confusion', 'selected_balp_aggregate_confusion', 'miou_and_per_class_results', 'latency_throughput_host_rss_peak_vram', 'overhead_versus_historical_baseline', 'overhead_versus_same_job_t320_control', 'selected_candidate_identity', 'upstream_hashes', 'development_subset_disclosure', 'complete', 'final'],
    'identity.description': 'Closed, versioned artifact schemas for BALP: development checkpoint (fresh/resume), development result, selection artifact, profiling artifact, and the future 5,000-image result schema (schema only, no execution authorization). Storage-only; defines no scientific formula.',
    'identity.name': 'e22-dfp-phase1b-lightweight-boundary-artifacts',
    'identity.schema_name': 'talk2dino-dfp-phase1b-lightweight-boundary-artifacts-identity-v1',
    'identity.schema_version': 1,
    'identity.storage_representation_only': True,
    'profiling_schema.derived_ratios.keys.fabricated_missing_telemetry_forbidden': True,
    'profiling_schema.derived_ratios.keys.list': ['total_wall_time_overhead_pct', 'mean_latency_overhead_pct', 'throughput_reduction_pct', 'peak_allocated_vram_increase_pct', 'peak_reserved_vram_increase_pct', 'host_rss_increase_pct'],
    'profiling_schema.derived_ratios.keys.raw_values_required_alongside_ratios': True,
    'profiling_schema.per_variant_record.keys.list': ['variant_id', 'total_wall_time_seconds', 'per_image_latency_seconds', 'throughput_images_per_second', 'host_peak_rss_bytes', 'gpu_peak_allocated_bytes', 'gpu_peak_reserved_bytes', 'snapshot_overhead_seconds', 'stitching_overhead_seconds', 'boundary_overhead_seconds', 'arbitration_overhead_seconds'],
    'profiling_schema.per_variant_record.keys.overhead_fields_optional_if_unmeasurable': ['snapshot_overhead_seconds', 'stitching_overhead_seconds', 'boundary_overhead_seconds', 'arbitration_overhead_seconds'],
    'profiling_schema.per_variant_record.keys.variant_id_closed_set': ['T320_baseline', 'selected_balp_candidate'],
    'profiling_schema.schema_name': 'talk2dino-dfp-phase1b-balp-profiling-v1',
    'profiling_schema.schema_version': 1,
    'profiling_schema.top_keys.affects_scientific_digest_required_value': False,
    'profiling_schema.top_keys.list': ['schema_name', 'schema_version', 'device_record', 'software_record', 'baseline_record', 'balp_record', 'derived_ratios', 'affects_scientific_digest'],
    'selection_schema.acceptance_comparison.keys.comparison_operator_required_value': '>=',
    'selection_schema.acceptance_comparison.keys.list': ['threshold_pp', 'observed_gain_pp', 'comparison_operator', 'passed'],
    'selection_schema.acceptance_comparison.keys.threshold_pp_required_value': 0.1,
    'selection_schema.confirmation_result.keys.confirmation_evaluated_at_most_once': True,
    'selection_schema.confirmation_result.keys.confirmation_image_count_required_value': 150,
    'selection_schema.confirmation_result.keys.list': ['candidate_id', 'confirmation_dataset_level_miou', 'confirmation_miou_gain_versus_t320_pp', 'confirmation_image_count', 'no_gt_isolation_violation', 'all_metrics_finite'],
    'selection_schema.eligibility_outcomes.member_keys.list': ['candidate_id', 'scientific_output_complete_and_schema_valid', 'no_gt_isolation_violation', 'finite_metrics', 'development_miou_strictly_exceeds_t320', 'no_recurrence_or_model_duplication', 'belongs_to_frozen_16_candidate_family', 'eligible'],
    'selection_schema.schema_name': 'talk2dino-dfp-phase1b-balp-selection-v1',
    'selection_schema.schema_version': 1,
    'selection_schema.tie_break_trace.member_keys.list': ['candidate_id', 'development_dataset_level_miou', 'preservation_mask_fraction', 'radius', 'stability_start', 'rank_after_miou', 'rank_after_preservation_fraction', 'rank_after_radius', 'rank_after_stability', 'rank_after_candidate_id', 'final_rank'],
    'selection_schema.top_keys.evaluation_stage_label_required_value': 'post_gate1_pre_full_evaluation',
    'selection_schema.top_keys.list': ['schema_name', 'schema_version', 'development_source_hashes', 'candidate_development_metrics', 'eligibility_outcomes', 'tie_break_trace', 'selected_candidate', 'confirmation_result', 'acceptance_comparison', 'operational_benchmark_status', 'complete', 'final', 'evaluation_stage_label'],
    'selection_schema.top_keys.selected_candidate_closed_set': 'shared.candidate_ids union {"NO_BALP_CANDIDATE_PROCEEDS"}',
    'shared.candidate_ids.count': 16,
    'shared.candidate_ids.list': ['balp_s000_r00', 'balp_s000_r01', 'balp_s000_r02', 'balp_s000_r04', 'balp_s016_r00', 'balp_s016_r01', 'balp_s016_r02', 'balp_s016_r04', 'balp_s032_r00', 'balp_s032_r01', 'balp_s032_r02', 'balp_s032_r04', 'balp_s064_r00', 'balp_s064_r01', 'balp_s064_r02', 'balp_s064_r04'],
    'shared.confusion_sufficient_statistics_member.description': "row=GT class, col=predicted class, evaluated pixels only (GT != ignore_index); dataset-level aggregation is the elementwise sum of every image's own confusion matrix -- never an arithmetic mean of per-image mIoUs",
    'shared.confusion_sufficient_statistics_member.dtype': 'int64',
    'shared.confusion_sufficient_statistics_member.non_negative': True,
    'shared.confusion_sufficient_statistics_member.rank': 2,
    'shared.confusion_sufficient_statistics_member.shape': '[class_count, class_count]',
    'shared.control_ids.count': 5,
    'shared.control_ids.list': ['T0', 'T16', 'T32', 'T64', 'T320'],
}

_PRESERVATION_DEP_KEYS: tuple[str, ...] = (
    "e13_protocol", "matched_evaluator", "graph_module", "finite_step_regime_module",
    "recurrence_capture_module", "horizon_stitching_module", "stitching_pullback_module",
    "final_rescale_module", "analysis_module", "identity_module", "subset_manifest", "balp_split_artifact",
)
_DEP_BINDING_KEYS = frozenset({"relative_path", "sha256", "binds"})
_SUBSET_MANIFEST_DEP_KEYS = frozenset({"relative_path", "sha256", "selected_ids_digest", "binds"})

_PROHIBITED_DEPENDENCY_MODULE_FRAGMENTS: tuple[str, ...] = (
    "dfp_phase1b_conservation_tolerance",
    "dfp_phase1b_diagnostic_execution",
    "dfp_phase1b_diagnostic_execute",
    "dfp_phase1b_provenance_features",
    "dfp_phase1b_masked_forward",
    "e19_dfp_phase1b_conservation_tolerance",
    "e20_dfp_phase1b_diagnostic_execution",
    "e21_dfp_phase1b_provenance_features",
)


def _top_level_leaves(doc: Mapping[str, Any], *, exclude: frozenset[str]) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for key, value in doc.items():
        if key in exclude:
            continue
        for dotted, leaf in _walk_leaves(value, key):
            observed[dotted] = leaf
    return observed


def _verify_canonical_leaves(observed: Mapping[str, Any], canonical: Mapping[str, Any], *, label: str) -> None:
    canon_keys = frozenset(canonical)
    obs_keys = frozenset(observed)
    if canon_keys != obs_keys:
        missing = sorted(canon_keys - obs_keys)
        extra = sorted(obs_keys - canon_keys)
        raise DfpPhase1bLightweightBoundaryIdentityError(
            f"{label} leaf set drifted -- missing={missing[:8]} extra={extra[:8]}"
        )
    for dotted, expected in canonical.items():
        observed_value = observed[dotted]
        if type(observed_value) is not type(expected) or observed_value != expected:
            raise DfpPhase1bLightweightBoundaryIdentityError(
                f"{label}.{dotted}: frozen value drift -- expected {expected!r} ({type(expected).__name__}), "
                f"observed {observed_value!r} ({type(observed_value).__name__})"
            )


def _resolve_and_hash(root: Path, relative_path: Any, *, binding_name: str) -> str:
    try:
        target = resolve_repository_relative_path(root, relative_path, binding_name=binding_name)
    except DfpPhase1bChunkBenchmarkIdentityError as exc:
        raise DfpPhase1bLightweightBoundaryIdentityError(
            f"dependency_bindings.{binding_name}: unsafe or unresolvable relative_path -- {exc}"
        ) from exc
    try:
        return hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError as exc:
        raise DfpPhase1bLightweightBoundaryIdentityError(
            f"dependency_bindings.{binding_name}: cannot read bound file {relative_path}: {exc}"
        ) from exc


def _verify_dependency_hashes(root: Path, dependency_bindings: Mapping[str, Any]) -> None:
    observed_names = frozenset(dependency_bindings.keys())
    expected_names = frozenset(_PRESERVATION_DEP_KEYS)
    if observed_names != expected_names:
        raise DfpPhase1bLightweightBoundaryIdentityError(
            f"dependency_bindings has the wrong key set -- missing={sorted(expected_names - observed_names)} "
            f"unknown={sorted(observed_names - expected_names)}"
        )
    for name in _PRESERVATION_DEP_KEYS:
        binding = dependency_bindings[name]
        keys = _SUBSET_MANIFEST_DEP_KEYS if name == "subset_manifest" else _DEP_BINDING_KEYS
        _require_closed(binding, keys, f"dependency_bindings.{name}")
        expected = _require_sha256(binding["sha256"], f"dependency_bindings.{name}.sha256")
        relative_path = binding["relative_path"]
        for fragment in _PROHIBITED_DEPENDENCY_MODULE_FRAGMENTS:
            if fragment in str(relative_path):
                raise DfpPhase1bLightweightBoundaryIdentityError(
                    f"dependency_bindings.{name}: relative_path {relative_path!r} references a prohibited "
                    f"masked-forward/E19/E20/E21 module -- BALP inference must not depend on it"
                )
        actual = _resolve_and_hash(root, relative_path, binding_name=name)
        if actual != expected:
            raise DfpPhase1bLightweightBoundaryIdentityError(
                f"dependency_bindings.{name}: {relative_path} SHA-256 mismatch -- pinned {expected}, observed {actual} "
                f"(bound dependency drifted since E22 was frozen)"
            )
        if name == "subset_manifest":
            digest_expected = _require_sha256(binding["selected_ids_digest"], "dependency_bindings.subset_manifest.selected_ids_digest")
            manifest_doc = json.loads((root / relative_path).read_bytes())
            if manifest_doc.get("selected_ids_digest") != digest_expected:
                raise DfpPhase1bLightweightBoundaryIdentityError(
                    "dependency_bindings.subset_manifest.selected_ids_digest disagrees with the manifest's own recorded digest"
                )


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load, closed-schema-validate, exact-value-validate, and
    relationally hash-verify the E22 BALP preservation identity. Never
    mutates or re-derives any bound dependency -- only re-verifies
    SHA-256 against current bytes. Performs no torch/CUDA/model/dataset
    access."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / PRESERVATION_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"cannot load E22 preservation identity {source}: {error}") from error

    top_keys = frozenset(_CANONICAL_TOP_KEYS_PRESERVATION) | {"dependency_bindings"}
    _require_closed(identity, top_keys, "E22 preservation identity")
    _require_schema_pair(
        identity.get("identity", {}).get("schema_name") if isinstance(identity.get("identity"), dict) else None,
        identity.get("identity", {}).get("schema_version") if isinstance(identity.get("identity"), dict) else None,
        PRESERVATION_SCHEMA_NAME, PRESERVATION_SCHEMA_VERSION, "identity",
    )

    dep = _require_mapping(identity.get("dependency_bindings"), "dependency_bindings")
    observed = _top_level_leaves(identity, exclude=frozenset({"dependency_bindings"}))
    _verify_canonical_leaves(observed, _CANONICAL_PRESERVATION, label="E22 preservation")
    _verify_dependency_hashes(root, dep)

    if identity["authorization_boundary"]["full_5000_image_evaluation_authorized"] is not False:
        raise DfpPhase1bLightweightBoundaryIdentityError("authorization_boundary.full_5000_image_evaluation_authorized must be exactly False")
    if identity["historical_exclusions"]["e19_e20_e21_required_by_balp_inference"] is not False:
        raise DfpPhase1bLightweightBoundaryIdentityError("historical_exclusions.e19_e20_e21_required_by_balp_inference must be exactly False")

    _verify_tie_rule_citation(root, identity)
    _verify_operational_formula_consistency(identity)

    return identity


def load_artifacts_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load, closed-schema-validate, and exact-value-validate the E22
    BALP artifact-schema identity, then verify (one-directionally) that
    its own [protocol_binding].protocol_identity_sha256 matches the
    CURRENT bytes of the preservation identity file -- never the
    reverse (avoids a circular/impossible self-hash design)."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / ARTIFACTS_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"cannot load E22 artifacts identity {source}: {error}") from error

    top_keys = frozenset(_CANONICAL_TOP_KEYS_ARTIFACTS) | {"protocol_binding"}
    _require_closed(identity, top_keys, "E22 artifacts identity")
    _require_schema_pair(
        identity.get("identity", {}).get("schema_name") if isinstance(identity.get("identity"), dict) else None,
        identity.get("identity", {}).get("schema_version") if isinstance(identity.get("identity"), dict) else None,
        ARTIFACTS_SCHEMA_NAME, ARTIFACTS_SCHEMA_VERSION, "identity",
    )

    protocol_binding = _require_mapping(identity.get("protocol_binding"), "protocol_binding")
    _require_closed(protocol_binding, frozenset({"protocol_identity_relative_path", "protocol_identity_sha256"}), "protocol_binding")
    expected = _require_sha256(protocol_binding["protocol_identity_sha256"], "protocol_binding.protocol_identity_sha256")
    actual = _resolve_and_hash(root, protocol_binding["protocol_identity_relative_path"], binding_name="protocol_binding")
    if actual != expected:
        raise DfpPhase1bLightweightBoundaryIdentityError(
            f"protocol_binding.protocol_identity_sha256 mismatch -- pinned {expected}, observed {actual} "
            f"(the preservation identity drifted since the artifacts identity was frozen)"
        )

    observed = _top_level_leaves(identity, exclude=frozenset({"protocol_binding"}))
    _verify_canonical_leaves(observed, _CANONICAL_ARTIFACTS, label="E22 artifacts")
    return identity


_CANONICAL_TOP_KEYS_PRESERVATION = frozenset(
    dotted.split(".", 1)[0] for dotted in _CANONICAL_PRESERVATION
) | {"identity"}
_CANONICAL_TOP_KEYS_ARTIFACTS = frozenset(
    dotted.split(".", 1)[0] for dotted in _CANONICAL_ARTIFACTS
) | {"identity"}


# =======================================================================
# Candidate-family reconstruction (independent of the frozen list --
# regenerated from the (stability_start, radius) grid and compared).
# =======================================================================


def reconstruct_candidate_ids() -> tuple[str, ...]:
    """Independently regenerates the exact 16 ordered BALP candidate
    IDs from the frozen (stability_starts, radii) grid -- row-major,
    stability_starts outer ascending, radii inner ascending -- never
    trusting the frozen TOML list as its own oracle."""
    return tuple(f"balp_s{s:03d}_r{r:02d}" for s in STABILITY_STARTS for r in RADII)


def verify_candidate_family(identity: Mapping[str, Any]) -> None:
    reconstructed = reconstruct_candidate_ids()
    frozen = tuple(identity["candidate_family"]["ordered_candidate_ids"]["list"])
    if reconstructed != frozen:
        raise DfpPhase1bLightweightBoundaryIdentityError(
            f"candidate_family.ordered_candidate_ids.list disagrees with the independently-reconstructed grid -- "
            f"expected {reconstructed}, observed {frozen}"
        )
    if len(set(reconstructed)) != 16:
        raise DfpPhase1bLightweightBoundaryIdentityError("candidate family must contain exactly 16 distinct IDs")


def parse_candidate_id(candidate_id: str) -> tuple[int, int]:
    """Independently parses a candidate ID back into (stability_start,
    radius), rejecting anything outside the frozen grid."""
    m = re.match(r"^balp_s(\d{3})_r(\d{2})$", candidate_id)
    if m is None:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"malformed candidate ID: {candidate_id!r}")
    s, r = int(m.group(1)), int(m.group(2))
    if s not in STABILITY_STARTS or r not in RADII:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"candidate ID {candidate_id!r} decodes outside the frozen grid (s={s}, r={r})")
    return s, r


# =======================================================================
# Development/confirmation split: independent reconstruction from the
# subset manifest, verified against the persisted split artifact.
# =======================================================================


def _selection_key(namespace: str, canonical_image_id: str) -> str:
    if type(namespace) is not str or type(canonical_image_id) is not str:
        raise DfpPhase1bLightweightBoundaryIdentityError("namespace and canonical_image_id must be strings")
    return hashlib.sha256(namespace.encode("utf-8") + b"\x00" + canonical_image_id.encode("utf-8")).hexdigest()


def _digest_over_records(records: list[bytes]) -> str:
    h = hashlib.sha256()
    for record in records:
        h.update(len(record).to_bytes(8, "big"))
        h.update(record)
    return h.hexdigest()


def reconstruct_development_split(selected_ids: list[str]) -> dict[str, Any]:
    """Independently recomputes the full 350/150 split, solely from the
    frozen 500 canonical image IDs and the frozen namespace -- never
    reading the persisted split artifact as its own oracle."""
    if not isinstance(selected_ids, list) or len(selected_ids) != 500:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"selected_ids must be a 500-element list, observed {len(selected_ids) if isinstance(selected_ids, list) else type(selected_ids).__name__}")
    if len(set(selected_ids)) != 500:
        raise DfpPhase1bLightweightBoundaryIdentityError("selected_ids contains a duplicate canonical image ID")
    records = [(cid, _selection_key(BALP_SPLIT_NAMESPACE, cid)) for cid in selected_ids]
    records.sort(key=lambda t: (t[1], t[0]))
    development = records[:DEVELOPMENT_COUNT]
    confirmation = records[DEVELOPMENT_COUNT:]
    dev_ids = [cid for cid, _ in development]
    conf_ids = [cid for cid, _ in confirmation]
    development_digest = _digest_over_records([cid.encode("utf-8") for cid in dev_ids])
    confirmation_digest = _digest_over_records([cid.encode("utf-8") for cid in conf_ids])
    trace_records = []
    for rank, (cid, key) in enumerate(development + confirmation):
        trace_records.append(json.dumps({"canonical_image_id": cid, "rank": rank, "selection_key": key}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    full_split_trace_digest = _digest_over_records(trace_records)
    return {
        "development_image_ids": dev_ids, "confirmation_image_ids": conf_ids,
        "development_selection_keys": [k for _, k in development], "confirmation_selection_keys": [k for _, k in confirmation],
        "development_digest": development_digest, "confirmation_digest": confirmation_digest,
        "full_split_trace_digest": full_split_trace_digest,
    }


_SPLIT_FILE_KEYS = frozenset({
    "schema_name", "schema_version", "namespace", "source_subset_relative_path", "source_selected_ids_digest",
    "source_image_count", "development_count", "confirmation_count", "development_image_ids",
    "development_selection_keys", "development_ranks", "confirmation_image_ids", "confirmation_selection_keys",
    "confirmation_ranks", "development_digest", "confirmation_digest", "full_split_trace_digest",
})


def load_and_verify_split(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Loads the persisted BALP split artifact, independently
    reconstructs the split from the frozen subset manifest, and rejects
    any disagreement in keys, ranks, membership, ordering, duplication,
    omission, or cross-split overlap."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    split_path = root / SPLIT_RELATIVE_PATH
    try:
        persisted = json.loads(split_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"cannot load BALP split artifact {split_path}: {error}") from error

    _require_closed(persisted, _SPLIT_FILE_KEYS, "BALP split artifact")
    if persisted["namespace"] != BALP_SPLIT_NAMESPACE:
        raise DfpPhase1bLightweightBoundaryIdentityError("BALP split artifact namespace disagrees with the frozen namespace")
    dev_count = _require_exact_int(persisted["development_count"], "development_count")
    conf_count = _require_exact_int(persisted["confirmation_count"], "confirmation_count")
    if dev_count != DEVELOPMENT_COUNT or conf_count != CONFIRMATION_COUNT:
        raise DfpPhase1bLightweightBoundaryIdentityError(
            f"BALP split artifact counts must be exactly {DEVELOPMENT_COUNT}/{CONFIRMATION_COUNT}, observed {dev_count}/{conf_count}"
        )

    manifest_path = root / SUBSET_MANIFEST_RELATIVE_PATH
    manifest_doc = json.loads(manifest_path.read_bytes())
    selected_ids = manifest_doc["selected_ids"]
    if persisted["source_selected_ids_digest"] != manifest_doc.get("selected_ids_digest"):
        raise DfpPhase1bLightweightBoundaryIdentityError("BALP split artifact source_selected_ids_digest disagrees with the subset manifest's own digest")

    reconstructed = reconstruct_development_split(list(selected_ids))

    dev_ids_persisted = persisted["development_image_ids"]
    conf_ids_persisted = persisted["confirmation_image_ids"]
    if not isinstance(dev_ids_persisted, list) or not isinstance(conf_ids_persisted, list):
        raise DfpPhase1bLightweightBoundaryIdentityError("development_image_ids/confirmation_image_ids must be lists")
    if len(dev_ids_persisted) != DEVELOPMENT_COUNT or len(conf_ids_persisted) != CONFIRMATION_COUNT:
        raise DfpPhase1bLightweightBoundaryIdentityError("persisted split image-id list lengths disagree with the frozen 350/150 counts")
    if dev_ids_persisted != reconstructed["development_image_ids"]:
        raise DfpPhase1bLightweightBoundaryIdentityError("persisted development_image_ids disagrees with the independently reconstructed split (ordering, membership, or key drift)")
    if conf_ids_persisted != reconstructed["confirmation_image_ids"]:
        raise DfpPhase1bLightweightBoundaryIdentityError("persisted confirmation_image_ids disagrees with the independently reconstructed split (ordering, membership, or key drift)")
    if set(dev_ids_persisted) & set(conf_ids_persisted):
        raise DfpPhase1bLightweightBoundaryIdentityError("development and confirmation splits overlap")
    if set(dev_ids_persisted) | set(conf_ids_persisted) != set(selected_ids):
        raise DfpPhase1bLightweightBoundaryIdentityError("development+confirmation union disagrees with the full 500-image manifest (omission or extraneous ID)")

    for field, expected in (
        ("development_selection_keys", reconstructed["development_selection_keys"]),
        ("confirmation_selection_keys", reconstructed["confirmation_selection_keys"]),
        ("development_digest", reconstructed["development_digest"]),
        ("confirmation_digest", reconstructed["confirmation_digest"]),
        ("full_split_trace_digest", reconstructed["full_split_trace_digest"]),
    ):
        if persisted[field] != expected:
            raise DfpPhase1bLightweightBoundaryIdentityError(f"BALP split artifact field {field!r} disagrees with the independent reconstruction")

    if persisted["development_ranks"] != list(range(0, DEVELOPMENT_COUNT)):
        raise DfpPhase1bLightweightBoundaryIdentityError("development_ranks must be exactly range(0, 350)")
    if persisted["confirmation_ranks"] != list(range(DEVELOPMENT_COUNT, DEVELOPMENT_COUNT + CONFIRMATION_COUNT)):
        raise DfpPhase1bLightweightBoundaryIdentityError("confirmation_ranks must be exactly range(350, 500)")

    return persisted


# =======================================================================
# Metric / tie-break / boundary-formula relational verification.
# =======================================================================


def verify_stability_late_flip_equivalence(
    y0: Any, y16: Any, y32: Any, y64: Any, y320: Any, s: int, boundary_mask: Any,
) -> Any:
    """Independently reconstructs P_{s,r}(x) and the relational
    equivalence `y_BALP == y64 where P else y320` for one already-
    computed boundary/radius mask, using only plain Python sequences
    (no torch/numpy import) -- suitable for small adversarial-test
    fixtures. `y0..y320`/`boundary_mask` are same-length flat sequences
    of per-pixel values; `boundary_mask` is a sequence of bool."""
    horizons = {0: y0, 16: y16, 32: y32, 64: y64}
    if s not in STABILITY_STARTS:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"stability_start must be one of {STABILITY_STARTS}, observed {s!r}")
    suffix = [h for h in STABILITY_STARTS if h >= s]
    n = len(y320)
    for name, seq in (("y0", y0), ("y16", y16), ("y32", y32), ("y64", y64), ("y320", y320), ("boundary_mask", boundary_mask)):
        if len(seq) != n:
            raise DfpPhase1bLightweightBoundaryIdentityError(f"{name} length {len(seq)} disagrees with y320 length {n}")
    y_balp = []
    preservation_mask = []
    for i in range(n):
        stable = all(horizons[h][i] == horizons[suffix[0]][i] for h in suffix)
        late_flip = y64[i] != y320[i]
        preserved = bool(stable and late_flip and boundary_mask[i])
        preservation_mask.append(preserved)
        y_balp.append(y64[i] if preserved else y320[i])
    return y_balp, preservation_mask


def verify_no_masked_forward_dependency(identity: Mapping[str, Any]) -> None:
    if identity["authorization_boundary"]["e19_e20_e21_part_of_balp_inference"] is not False:
        raise DfpPhase1bLightweightBoundaryIdentityError("authorization_boundary.e19_e20_e21_part_of_balp_inference must be exactly False")
    if identity["method"]["runs_masked_forward_propagation"] is not False:
        raise DfpPhase1bLightweightBoundaryIdentityError("method.runs_masked_forward_propagation must be exactly False")
    if identity["method"]["constructs_components"] is not False:
        raise DfpPhase1bLightweightBoundaryIdentityError("method.constructs_components must be exactly False")
    if identity["method"]["performs_pullback_or_membership_computation"] is not False:
        raise DfpPhase1bLightweightBoundaryIdentityError("method.performs_pullback_or_membership_computation must be exactly False")


def verify_full_evaluation_not_authorized(identity: Mapping[str, Any]) -> None:
    ab = identity["authorization_boundary"]
    if ab["full_5000_image_evaluation_authorized"] is not False:
        raise DfpPhase1bLightweightBoundaryIdentityError("full_5000_image_evaluation_authorized must be exactly False")
    if ab["gpu_execution_authorized_by_this_identity"] is not False:
        raise DfpPhase1bLightweightBoundaryIdentityError("gpu_execution_authorized_by_this_identity must be exactly False")
    if ab["candidate_selected_by_this_identity"] is not False:
        raise DfpPhase1bLightweightBoundaryIdentityError("candidate_selected_by_this_identity must be exactly False")
    if ab["other_4500_image_access_authorized"] is not False:
        raise DfpPhase1bLightweightBoundaryIdentityError("other_4500_image_access_authorized must be exactly False")


def verify_repository_gate(repo_root: Path, repository_commit: str) -> None:
    """Verifies REQUIRED_ANCESTOR_COMMIT is an ancestor of (or equal
    to) `repository_commit`, exactly like E17-E21's own convention."""
    import subprocess

    commit = _require_string(repository_commit, "repository_commit")
    if not re.match(r"^[0-9a-f]{40}$", commit):
        raise DfpPhase1bLightweightBoundaryIdentityError(f"repository_commit must be a 40-hex commit, observed {commit!r}")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", REQUIRED_ANCESTOR_COMMIT, commit],
            capture_output=True, text=True,
        )
    except OSError as exc:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"cannot invoke git for ancestor verification: {exc}") from exc
    if result.returncode != 0:
        raise DfpPhase1bLightweightBoundaryIdentityError(
            f"required_ancestor_commit {REQUIRED_ANCESTOR_COMMIT} is not an ancestor of (or equal to) {commit}"
        )


# =======================================================================
# Repair F1: tie-rule citation relational verification. Independently
# resolves the E13 dependency and confirms the CITED section/key
# actually exists there and carries the exact frozen tie-rule value --
# never trusting the prose citation string alone.
# =======================================================================


def _verify_tie_rule_citation(root: Path, identity: Mapping[str, Any]) -> None:
    ch = identity["canonical_horizons"]
    section = _require_string(ch["argmax_tie_rule_source_section"], "canonical_horizons.argmax_tie_rule_source_section")
    key = _require_string(ch["argmax_tie_rule_source_key"], "canonical_horizons.argmax_tie_rule_source_key")
    expected_value = ch["argmax_tie_rule"]
    e13_binding = identity["dependency_bindings"]["e13_protocol"]
    try:
        target = resolve_repository_relative_path(root, e13_binding["relative_path"], binding_name="e13_protocol")
    except DfpPhase1bChunkBenchmarkIdentityError as exc:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"tie-rule citation: cannot resolve e13_protocol path -- {exc}") from exc
    try:
        with target.open("rb") as handle:
            e13_doc = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"tie-rule citation: cannot parse e13_protocol -- {exc}") from exc
    section_table = e13_doc.get(section)
    if not isinstance(section_table, dict):
        raise DfpPhase1bLightweightBoundaryIdentityError(
            f"tie-rule citation: E13 has no table [{section}] -- argmax_tie_rule_source_section is wrong"
        )
    if key not in section_table:
        raise DfpPhase1bLightweightBoundaryIdentityError(
            f"tie-rule citation: E13 [{section}] has no key {key!r} -- argmax_tie_rule_source_key is wrong"
        )
    observed_value = section_table[key]
    if observed_value != expected_value:
        raise DfpPhase1bLightweightBoundaryIdentityError(
            f"tie-rule citation: E13 [{section}].{key} == {observed_value!r}, but canonical_horizons.argmax_tie_rule == {expected_value!r} -- these must be identical"
        )


# =======================================================================
# Repair F2: exact operational formulas. Pure, independently callable
# computation functions (float64 arithmetic, no torch/numpy) plus a
# relational cross-check that the formula sub-tables' thresholds never
# drift from operational_contract.acceptance_caps.
# =======================================================================


_OPERATIONAL_FORMULA_TO_CAP_FIELD: dict[str, str] = {
    "total_wall_time_overhead": "total_wall_time_overhead_pct_max",
    "mean_per_image_latency_overhead": "mean_per_image_latency_overhead_pct_max",
    "throughput_reduction": "throughput_reduction_pct_max",
    "peak_allocated_vram_increase": "peak_allocated_vram_increase_pct_max",
    "peak_reserved_vram_increase": "peak_reserved_vram_increase_pct_max",
}


def _verify_operational_formula_consistency(identity: Mapping[str, Any]) -> None:
    formulas = identity["operational_contract"]["formulas"]
    caps = identity["operational_contract"]["acceptance_caps"]
    observed_names = frozenset(formulas.keys())
    expected_names = frozenset(_OPERATIONAL_FORMULA_TO_CAP_FIELD)
    if observed_names != expected_names:
        raise DfpPhase1bLightweightBoundaryIdentityError(
            f"operational_contract.formulas has the wrong key set -- missing={sorted(expected_names - observed_names)} "
            f"unknown={sorted(observed_names - expected_names)}"
        )
    for name, cap_field in _OPERATIONAL_FORMULA_TO_CAP_FIELD.items():
        block = formulas[name]
        if block["comparison_operator"] != "<=":
            raise DfpPhase1bLightweightBoundaryIdentityError(f"operational_contract.formulas.{name}.comparison_operator must be exactly '<='")
        threshold = _require_finite_float(block["threshold"], f"operational_contract.formulas.{name}.threshold")
        cap_value = caps[cap_field]
        if type(cap_value) not in (int, float) or isinstance(cap_value, bool):
            raise DfpPhase1bLightweightBoundaryIdentityError(f"operational_contract.acceptance_caps.{cap_field} must be a plain number")
        if float(threshold) != float(cap_value):
            raise DfpPhase1bLightweightBoundaryIdentityError(
                f"operational_contract.formulas.{name}.threshold ({threshold}) drifted from acceptance_caps.{cap_field} ({cap_value})"
            )
        if block["units"] != "percent":
            raise DfpPhase1bLightweightBoundaryIdentityError(f"operational_contract.formulas.{name}.units must be exactly 'percent'")


def _require_measurement(value: Any, label: str) -> float:
    """A raw telemetry measurement: exact int or float, finite,
    never bool/str/None/NaN/Infinity. Returned as float64."""
    if isinstance(value, bool) or type(value) not in (int, float):
        raise DfpPhase1bLightweightBoundaryIdentityError(f"{label} must be a plain int or float measurement, observed {type(value).__name__}: {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise DfpPhase1bLightweightBoundaryIdentityError(f"{label} must be finite, observed {value!r}")
    return value


def _require_positive_baseline(value: Any, label: str) -> float:
    v = _require_measurement(value, label)
    if not v > 0.0:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"{label} must be strictly positive, observed {v!r}")
    return v


def compute_total_wall_time_overhead_pct(candidate_total_wall_time_ns: Any, baseline_total_wall_time_ns: Any) -> float:
    baseline = _require_positive_baseline(baseline_total_wall_time_ns, "baseline_total_wall_time_ns")
    candidate = _require_measurement(candidate_total_wall_time_ns, "candidate_total_wall_time_ns")
    return 100.0 * (candidate - baseline) / baseline


def compute_mean_per_image_latency_overhead_pct(candidate_mean_per_image_latency_ns: Any, baseline_mean_per_image_latency_ns: Any) -> float:
    baseline = _require_positive_baseline(baseline_mean_per_image_latency_ns, "baseline_mean_per_image_latency_ns")
    candidate = _require_measurement(candidate_mean_per_image_latency_ns, "candidate_mean_per_image_latency_ns")
    return 100.0 * (candidate - baseline) / baseline


def compute_throughput_reduction_pct(candidate_throughput_images_per_second: Any, baseline_throughput_images_per_second: Any) -> float:
    baseline = _require_positive_baseline(baseline_throughput_images_per_second, "baseline_throughput_images_per_second")
    candidate = _require_measurement(candidate_throughput_images_per_second, "candidate_throughput_images_per_second")
    return 100.0 * (baseline - candidate) / baseline


def compute_peak_allocated_vram_increase_pct(candidate_peak_allocated_vram_bytes: Any, baseline_peak_allocated_vram_bytes: Any) -> float:
    baseline = _require_positive_baseline(baseline_peak_allocated_vram_bytes, "baseline_peak_allocated_vram_bytes")
    candidate = _require_measurement(candidate_peak_allocated_vram_bytes, "candidate_peak_allocated_vram_bytes")
    return 100.0 * (candidate - baseline) / baseline


def compute_peak_reserved_vram_increase_pct(candidate_peak_reserved_vram_bytes: Any, baseline_peak_reserved_vram_bytes: Any) -> float:
    baseline = _require_positive_baseline(baseline_peak_reserved_vram_bytes, "baseline_peak_reserved_vram_bytes")
    candidate = _require_measurement(candidate_peak_reserved_vram_bytes, "candidate_peak_reserved_vram_bytes")
    return 100.0 * (candidate - baseline) / baseline


def compute_host_rss_difference(candidate_host_rss_bytes: Any, baseline_host_rss_bytes: Any) -> dict[str, float | None]:
    """No pass/fail threshold is authorized for host RSS -- this
    returns the raw absolute difference always, and the percentage
    difference only when the baseline is strictly positive (None
    otherwise, never a fabricated 0.0 or a division by zero)."""
    baseline = _require_measurement(baseline_host_rss_bytes, "baseline_host_rss_bytes")
    candidate = _require_measurement(candidate_host_rss_bytes, "candidate_host_rss_bytes")
    absolute = candidate - baseline
    percentage = 100.0 * absolute / baseline if baseline > 0.0 else None
    return {"absolute_difference_bytes": absolute, "percentage_difference_pct": percentage}


_OPERATIONAL_ACCEPTANCE_FNS: dict[str, Any] = {
    "total_wall_time_overhead": compute_total_wall_time_overhead_pct,
    "mean_per_image_latency_overhead": compute_mean_per_image_latency_overhead_pct,
    "throughput_reduction": compute_throughput_reduction_pct,
    "peak_allocated_vram_increase": compute_peak_allocated_vram_increase_pct,
    "peak_reserved_vram_increase": compute_peak_reserved_vram_increase_pct,
}


def evaluate_operational_cap(name: str, candidate_value: Any, baseline_value: Any, *, threshold: float) -> tuple[float, bool]:
    """Computes the named metric's percentage and applies the frozen
    inclusive `<=` comparison against `threshold` -- never `<`."""
    if name not in _OPERATIONAL_ACCEPTANCE_FNS:
        raise DfpPhase1bLightweightBoundaryIdentityError(f"unknown operational metric {name!r}")
    pct = _OPERATIONAL_ACCEPTANCE_FNS[name](candidate_value, baseline_value)
    threshold = _require_finite_float(threshold, "threshold")
    return pct, bool(pct <= threshold)


# =======================================================================
# Repair F3: checkpoint relational validation. Pure structural/
# relational checks against a plain-dict checkpoint document shaped
# exactly per artifacts.toml's checkpoint_schema -- never a filesystem
# writer. Confusion matrices are represented as plain nested lists of
# int (no numpy), summed by hand.
# =======================================================================


def _matrix_add(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    if len(a) != len(b) or any(len(ra) != len(rb) for ra, rb in zip(a, b)):
        raise DfpPhase1bLightweightBoundaryIdentityError("confusion matrix shape mismatch during summation")
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def _zero_matrix(class_count: int) -> list[list[int]]:
    return [[0] * class_count for _ in range(class_count)]


def _matrix_equal(a: list[list[int]], b: list[list[int]]) -> bool:
    return a == b


CHECKPOINT_SCHEMA_NAME = _CANONICAL_ARTIFACTS['checkpoint_schema.schema_name']
CHECKPOINT_SCHEMA_VERSION = _CANONICAL_ARTIFACTS['checkpoint_schema.schema_version']
CHECKPOINT_TOP_LEVEL_KEYS = frozenset(_CANONICAL_ARTIFACTS['checkpoint_schema.top_keys.list'])
CHECKPOINT_PROGRESS_KEYS = frozenset(_CANONICAL_ARTIFACTS['checkpoint_schema.progress.keys.list'])
CHECKPOINT_IDENTITY_BINDINGS_KEYS = frozenset(_CANONICAL_ARTIFACTS['checkpoint_schema.identity_bindings.keys.list'])
CHECKPOINT_PER_IMAGE_STAT_KEYS = frozenset(_CANONICAL_ARTIFACTS['checkpoint_schema.per_image_reconstruction_statistics.keys.list'])
CHECKPOINT_FINAL_RESULT_BINDING_KEYS = frozenset(_CANONICAL_ARTIFACTS['checkpoint_schema.final_result_binding.keys.list'])
# ^ Every one of these is read directly from this module's own already-
# loaded, mechanically-generated _CANONICAL_ARTIFACTS table (itself
# generated from the committed artifacts TOML) -- never a second,
# independently hand-typed key list that could silently diverge from
# the authoritative schema.


def validate_checkpoint_single(
    doc: Mapping[str, Any], *, development_ids: Sequence[str], confirmation_ids: Sequence[str],
    candidate_ids: Sequence[str] = None, control_ids: Sequence[str] = None,
) -> None:
    """Independently checks every checkpoint_schema.relational_validation
    rule and the checkpoint_schema.state_machine against ONE checkpoint
    document (a plain dict shaped per the frozen schema). Raises on the
    first violated invariant.

    Enforces the EXACT closed top-level key set
    (checkpoint_schema.top_keys.list) and every declared-closed nested
    mapping (progress, identity_bindings, each per_image_reconstruction_
    statistics[control-or-candidate] block, and final_result_binding
    when present) BEFORE reading any individual field by indexing or
    `.get()` -- an extra key (e.g. a leaked `profiling`/`timing_ns`/
    `debug` field) is rejected exactly like a missing one, never
    silently ignored."""
    candidate_ids = tuple(candidate_ids) if candidate_ids is not None else reconstruct_candidate_ids()
    control_ids = tuple(control_ids) if control_ids is not None else CANONICAL_CONTROLS
    E = DfpPhase1bLightweightBoundaryIdentityError

    _require_closed(doc, CHECKPOINT_TOP_LEVEL_KEYS, "checkpoint")
    _require_schema_pair(doc.get("schema_name"), doc.get("schema_version"), CHECKPOINT_SCHEMA_NAME, CHECKPOINT_SCHEMA_VERSION, "checkpoint")

    progress = _require_mapping(doc.get("progress"), "checkpoint.progress")
    _require_closed(progress, CHECKPOINT_PROGRESS_KEYS, "checkpoint.progress")
    identity_bindings = _require_mapping(doc.get("identity_bindings"), "checkpoint.identity_bindings")
    _require_closed(identity_bindings, CHECKPOINT_IDENTITY_BINDINGS_KEYS, "checkpoint.identity_bindings")
    expected_count = _require_exact_int(progress["expected_development_image_count"], "progress.expected_development_image_count")
    if expected_count != DEVELOPMENT_COUNT:
        raise E(f"progress.expected_development_image_count must be {DEVELOPMENT_COUNT}, observed {expected_count}")
    next_idx = _require_exact_int(progress["next_development_index"], "progress.next_development_index")
    if not (0 <= next_idx <= DEVELOPMENT_COUNT):
        raise E(f"progress.next_development_index must be in [0,{DEVELOPMENT_COUNT}], observed {next_idx}")
    completed = progress["completed_image_ids"]
    if not isinstance(completed, list):
        raise E("progress.completed_image_ids must be a list")
    if len(completed) != next_idx:
        raise E(f"len(completed_image_ids)={len(completed)} disagrees with next_development_index={next_idx}")
    if len(set(completed)) != len(completed):
        raise E("progress.completed_image_ids contains a duplicate")
    expected_prefix = list(development_ids[:next_idx])
    if completed != expected_prefix:
        raise E("progress.completed_image_ids is not the exact frozen development-split prefix")
    if set(completed) & set(confirmation_ids):
        raise E("a confirmation image appears in a development checkpoint's completed_image_ids")

    agg = _require_mapping(doc.get("aggregate_confusion"), "checkpoint.aggregate_confusion")
    expected_keys = list(control_ids) + list(candidate_ids)
    if list(agg.keys()) != expected_keys:
        raise E(f"aggregate_confusion key order disagrees with frozen control+candidate order -- expected {expected_keys}, observed {list(agg.keys())}")
    for k in agg:
        if k not in control_ids and k not in candidate_ids:
            raise E(f"aggregate_confusion key {k!r} is not a member of the frozen control/candidate family")

    per_image = _require_mapping(doc.get("per_image_reconstruction_statistics"), "checkpoint.per_image_reconstruction_statistics")
    if list(per_image.keys()) != expected_keys:
        raise E("per_image_reconstruction_statistics key order disagrees with frozen control+candidate order")

    evaluated_by_image: dict[int, int] = {}
    ignored_by_image: dict[int, int] = {}
    for k in expected_keys:
        block = _require_mapping(per_image[k], f"checkpoint.per_image_reconstruction_statistics[{k}]")
        _require_closed(block, CHECKPOINT_PER_IMAGE_STAT_KEYS, f"checkpoint.per_image_reconstruction_statistics[{k}]")
        for field in ("candidate_versus_t320_changed_pixel_count", "preservation_mask_pixel_count",
                      "boundary_preservation_pixel_count", "non_boundary_preservation_pixel_count",
                      "evaluated_pixel_count", "ignored_pixel_count", "per_image_confusion_matrix"):
            arr = block.get(field)
            if not isinstance(arr, list) or len(arr) != next_idx:
                raise E(f"per_image_reconstruction_statistics[{k}][{field}] must be a list of exactly next_development_index={next_idx} entries, observed {len(arr) if isinstance(arr, list) else type(arr).__name__}")
        for i in range(next_idx):
            evaluated_by_image.setdefault(i, block["evaluated_pixel_count"][i])
            if block["evaluated_pixel_count"][i] != evaluated_by_image[i]:
                raise E(f"evaluated_pixel_count for image index {i} disagrees between control/candidate {k!r} and an earlier one")
            ignored_by_image.setdefault(i, block["ignored_pixel_count"][i])
            if block["ignored_pixel_count"][i] != ignored_by_image[i]:
                raise E(f"ignored_pixel_count for image index {i} disagrees between control/candidate {k!r} and an earlier one")

        summed = None
        for i in range(next_idx):
            m = block["per_image_confusion_matrix"][i]
            summed = m if summed is None else _matrix_add(summed, m)
        class_count = len(agg[k]) if next_idx > 0 else (len(summed) if summed else 0)
        expected_sum = summed if summed is not None else _zero_matrix(len(agg[k]) if agg[k] else 0)
        if next_idx > 0 and not _matrix_equal(agg[k], expected_sum):
            raise E(f"aggregate_confusion[{k}] does not equal the sum of persisted per-image confusion matrices")

    complete = doc.get("complete")
    final = doc.get("final")
    if not isinstance(complete, bool) or not isinstance(final, bool):
        raise E("complete/final must be booleans")
    if final and not complete:
        raise E("final == true requires complete == true")
    final_binding = doc.get("final_result_binding")
    if not complete and final_binding not in (None,):
        raise E("final_result_binding must be absent when complete == false")
    if final and final_binding is None:
        raise E("final == true requires a present final_result_binding")
    if final_binding is not None:
        _require_closed(_require_mapping(final_binding, "checkpoint.final_result_binding"), CHECKPOINT_FINAL_RESULT_BINDING_KEYS, "checkpoint.final_result_binding")

    if next_idx == 0:
        if completed or complete or final:
            raise E("zero-progress state violated: next_development_index==0 requires empty completed list, complete==false, final==false")
    elif next_idx < DEVELOPMENT_COUNT:
        if complete or final:
            raise E("partial-progress state violated: complete/final must both be false while next_development_index < 350")
    else:
        if completed != list(development_ids):
            raise E("next_development_index==350 requires completed_image_ids to be exactly the full frozen development order")
        if not complete:
            raise E("next_development_index==350 requires complete==true")


def validate_checkpoint_resume_pair(old: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    """Independently checks every checkpoint_schema.cross_generation_invariants
    rule between an earlier checkpoint `old` and a later checkpoint `new`."""
    E = DfpPhase1bLightweightBoundaryIdentityError
    old_progress, new_progress = old["progress"], new["progress"]
    old_next, new_next = old_progress["next_development_index"], new_progress["next_development_index"]
    if new_next < old_next:
        raise E("resume cannot decrease next_development_index")
    old_completed = old_progress["completed_image_ids"]
    new_completed = new_progress["completed_image_ids"]
    if new_completed[:len(old_completed)] != old_completed:
        raise E("resume: old completed_image_ids is not an exact prefix of the new checkpoint's completed_image_ids")

    old_bindings = old.get("identity_bindings")
    new_bindings = new.get("identity_bindings")
    if old_bindings != new_bindings:
        raise E("resume: identity_bindings changed across generations")

    old_agg, new_agg = old["aggregate_confusion"], new["aggregate_confusion"]
    if list(old_agg.keys()) != list(new_agg.keys()):
        raise E("resume: aggregate_confusion candidate/control ordering changed across generations")

    old_per_image, new_per_image = old["per_image_reconstruction_statistics"], new["per_image_reconstruction_statistics"]
    for k in old_agg:
        for field in ("candidate_versus_t320_changed_pixel_count", "preservation_mask_pixel_count",
                      "boundary_preservation_pixel_count", "non_boundary_preservation_pixel_count",
                      "evaluated_pixel_count", "ignored_pixel_count", "per_image_confusion_matrix"):
            old_vals = old_per_image[k][field]
            new_vals = new_per_image[k][field]
            if new_vals[:old_next] != old_vals[:old_next]:
                raise E(f"resume: per_image_reconstruction_statistics[{k}][{field}] changed a value for an already-completed image")

        newly_appended = new_per_image[k]["per_image_confusion_matrix"][old_next:new_next]
        expected_new_agg = old_agg[k]
        for m in newly_appended:
            expected_new_agg = _matrix_add(expected_new_agg, m)
        if not _matrix_equal(new_agg[k], expected_new_agg):
            raise E(f"resume: aggregate_confusion[{k}] does not equal the old aggregate plus the newly appended per-image confusion matrices")


# =======================================================================
# Repair F4: artifact write-policy structural verification.
# =======================================================================


_WRITE_POLICY_TOP_KEYS = frozenset({
    "applies_to", "reused_pattern_source", "general_path_policy", "staging",
    "atomic_installation", "generation_protocol", "failure_behavior",
    "freshness_and_overwrite", "sibling_digest_bindings",
})


def verify_write_policy_present(artifacts_identity: Mapping[str, Any]) -> None:
    """Confirms the closed [artifact_write_policy] section exists with
    every required sub-section and that it grants no overwrite
    authority -- structural leaf values are already exact-value-checked
    by load_artifacts_identity's own _CANONICAL_ARTIFACTS comparison;
    this is an explicit, named API surface for that same guarantee."""
    policy = artifacts_identity.get("artifact_write_policy")
    if policy is None:
        raise DfpPhase1bLightweightBoundaryIdentityError("artifacts identity is missing [artifact_write_policy]")
    _require_closed(policy, _WRITE_POLICY_TOP_KEYS, "artifact_write_policy")
    if policy["general_path_policy"]["e22_grants_overwrite_authority"] is not False:
        raise DfpPhase1bLightweightBoundaryIdentityError("artifact_write_policy must grant no overwrite authority")
    if policy["atomic_installation"]["multi_file_artifacts"]["unguarded_sequential_final_renames_forbidden"] is not True:
        raise DfpPhase1bLightweightBoundaryIdentityError("artifact_write_policy must forbid unguarded sequential final renames")
    if policy["freshness_and_overwrite"]["e22_grants_destructive_overwrite"] is not False:
        raise DfpPhase1bLightweightBoundaryIdentityError("artifact_write_policy must grant no destructive overwrite")
    post_install = policy["atomic_installation"]["post_install_verification"]
    if post_install["reopen_and_reverify_after_install_required"] is not True:
        raise DfpPhase1bLightweightBoundaryIdentityError("artifact_write_policy must require reopening and re-verifying the installed artifact after installation")
    if post_install["replaces_pre_install_staged_verification"] is not False:
        raise DfpPhase1bLightweightBoundaryIdentityError("post-install verification must never be described as replacing pre-install staged verification")
    if post_install["both_pre_install_and_post_install_verification_mandatory"] is not True:
        raise DfpPhase1bLightweightBoundaryIdentityError("both pre-install and post-install verification must be mandatory")
    if post_install["overwrite_authorized_by_this_requirement"] is not False:
        raise DfpPhase1bLightweightBoundaryIdentityError("post-install verification must not itself authorize overwrite")
