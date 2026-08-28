"""Static identity loading/validation for the DFP Phase 1A protocol
pre-registration. This is a diagnostic-protocol identity, not an
evaluator identity: it never loads CUDA, a model, or a dataset. It
records and exact-type validates the frozen T=0-vs-T=320 stitched-
dispute diagnostic contract, and binds relationally to the already-
frozen COCO-Stuff (e12-matched-k11-k12-t320) and ADE20K
(e12-ade20k-matched-evaluator) protocols this pre-registration reuses
unmodified.

Always fail closed: never silently substitute a default value, coerce a
type, or normalize a string.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping

IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e13_dfp_phase1a_protocol.toml")

REQUIRED_ANCESTOR_COMMIT = "00392527937253f93b9f3fe36b76347b4a6f1eb0"
WORKING_BRANCH = "dfp-phase1a-diagnostics"

IDENTITY_TOP_KEYS = frozenset(
    {
        "format_version", "identity", "parent_identities", "bridge_checkpoint", "coco_stuff_dataset",
        "ade20k_dataset", "graph", "propagation", "score_interpolation_stitching", "dispute_protocol",
        "dispute_categories", "oracle_definitions", "horizon_persistence", "metric_contract", "boundary_iou",
        "gate_contract", "gate_expectations", "run_modes", "subset_selection_policy", "structural_components",
        "structural_component_area", "query_compute_projection", "numerical_validation_policy", "prohibited",
        "future_step_bindings",
    }
)

SECTION_KEYS: dict[str, frozenset[str]] = {
    "identity": frozenset({"name", "schema_version", "description", "required_ancestor_commit", "working_branch"}),
    "parent_identities": frozenset(
        {
            "matched_k11_k12_identity_path", "matched_k11_k12_identity_name", "matched_k11_k12_identity_sha256",
            "e3_identity_path", "e3_identity_name", "e3_identity_sha256",
            "rwr_identity_path", "rwr_identity_name", "rwr_identity_sha256",
            "ade20k_source_identity_path", "ade20k_source_identity_name", "ade20k_source_identity_sha256",
            "ade20k_evaluator_identity_path", "ade20k_evaluator_identity_name", "ade20k_evaluator_identity_sha256",
            "required_relationship",
        }
    ),
    "bridge_checkpoint": frozenset({"projection_class", "relative_path", "sha256", "bound_via", "frozen"}),
    "coco_stuff_dataset": frozenset(
        {
            "name", "source_authority", "split", "images", "class_count", "class_names_digest",
            "class_names_digest_algorithm", "class_names_source", "dataset_type", "image_suffix",
            "annotation_suffix", "ignore_index", "background_class_evaluated", "reduce_zero_label",
        }
    ),
    "ade20k_dataset": frozenset(
        {
            "name", "source_authority", "split", "images", "class_count", "class_names_digest",
            "class_names_digest_algorithm", "class_names_source", "dataset_type", "image_suffix",
            "annotation_suffix", "ignore_index", "background_class_evaluated", "reduce_zero_label",
        }
    ),
    "graph": frozenset(
        {
            "directed", "row_stochastic", "maximum_rank", "affinity_function", "affinity_power",
            "tie_break_rule", "self_edge_policy", "fallback_row_policy", "graph_symmetrization",
            "artificial_tie_perturbation", "canonical_k_variant", "canonical_k_variant_note",
        }
    ),
    "propagation": frozenset(
        {
            "method", "alpha", "steps", "initial_iterate", "recurrence", "early_stopping",
            "convergence_tolerance", "fallback_solver", "horizons", "horizon_note",
            "primary_sensitivity_horizon", "secondary_sensitivity_horizons",
        }
    ),
    "score_interpolation_stitching": frozenset(
        {
            "raw_score_stage", "crop", "stride", "align_corners", "averaging", "window_enumeration",
            "clamping_policy", "hann", "majority_vote", "center_selection", "sparse_delta_restitching",
            "identical_across_horizons",
        }
    ),
    "dispute_protocol": frozenset(
        {"decision_stage", "a_definition", "b_definition", "tie_rule", "random_tie_break_allowed",
         "numpy_torch_tie_parity_required", "dispute_set_definition"}
    ),
    "dispute_categories": frozenset(
        {"harmful_washout", "beneficial_correction", "third_label_transition", "ignore",
         "auroc_auprc_population", "third_label_and_ignore_excluded_from_negative_class"}
    ),
    "oracle_definitions": frozenset(
        {
            "propagated_baseline_P", "unary_U", "oracle_O_P_rule", "g_recover_definition",
            "g_oracle_unary_definition", "relational_identity", "third_label_diagnostic_rule",
            "third_label_diagnostic_label",
        }
    ),
    "horizon_persistence": frozenset({"applies_to_horizons", "rule", "primary_horizon", "sensitivity_horizons"}),
    "metric_contract": frozenset(
        {
            "required_statistics", "required_metrics", "reported_metric_unit", "difference_unit",
            "aggregation_rule", "per_class_statistics_required", "prohibited_reductions",
        }
    ),
    "boundary_iou": frozenset(
        {
            "definition", "primary_width_convention", "sensitivity_pixel_widths",
            "implementation_source_status", "implementation_source_note",
        }
    ),
    "gate_contract": frozenset(
        {
            "oracle_unary_gate_threshold_percentage_points", "recover_gate_threshold_percentage_points",
            "gate_relation", "outcome_neither_dataset_passes_both", "outcome_exactly_one_dataset_passes_both",
            "outcome_both_datasets_pass_both", "cross_dataset_pass_requires_independent_evaluation",
        }
    ),
    "gate_expectations": frozenset(
        {
            "expected_dispute_fraction_low", "expected_dispute_fraction_high",
            "dispute_fraction_too_small_threshold", "dispute_fraction_too_small_consequence",
            "dispute_fraction_too_large_threshold", "dispute_fraction_too_large_consequence",
            "gt_in_ab_coverage_expected_minimum", "t32_persistence_recovery_expected_maximum_fraction",
            "t32_persistence_high_recovery_consequence", "boundary_mass_reframe_threshold",
            "boundary_mass_reframe_consequence", "expectations_are_soft_not_hard_validity_checks",
        }
    ),
    "run_modes": frozenset(
        {
            "coco_stuff_500_source_authority", "coco_stuff_500_required_unique_count",
            "ade20k_500_source_authority", "ade20k_500_required_unique_count", "run_mode_names",
        }
    ),
    "subset_selection_policy": frozenset(
        {
            "selection_method", "input_domain", "tie_break_method", "required_unique_count",
            "manifest_must_contain", "manifest_private_paths_forbidden",
            "manifest_must_be_immutably_committed_before_predictions",
            "concrete_manifest_ids_bound_in_this_identity", "concrete_manifest_binding_stage",
        }
    ),
    "structural_components": frozenset(
        {
            "component_source", "primary_connectivity", "sensitivity_connectivity", "tie_rule",
            "component_scope", "soft_membership_formula", "membership_operator",
            "membership_dense_matrix_forbidden", "adjoint_identity_required",
            "membership_nonnegative_required", "membership_sums_to_one_per_window_patch_source_required",
            "flush_edge_windows_included", "a_transpose_ones_cached_invariant",
        }
    ),
    "structural_component_area": frozenset(
        {
            "formula", "projected_pitch_source", "primary_microcomponent_threshold_patch_equivalents",
            "sensitivity_thresholds_patch_equivalents", "source_copy_mass_is_diagnostic_only",
            "source_copy_mass_is_not_physical_component_area", "phase1a_propagates_component_masks",
        }
    ),
    "query_compute_projection": frozenset(
        {
            "q_w_definition", "required_statistics", "required_statistics_subject",
            "components_with_nonzero_soft_membership_per_window_required", "projected_width_formula",
            "total_and_worst_window_projected_width_required", "projected_peak_memory_for_phase1b_required",
            "adjoint_or_masked_forward_router_defined_in_this_task",
        }
    ),
    "numerical_validation_policy": frozenset(
        {
            "canonical_and_decomposed_same_device_and_precision_required",
            "conservation_summary_accumulation_dtype", "validation_levels", "residual_kinds_required",
            "tolerance_source", "tolerance_invented_in_this_identity", "calibration_artifact_created_in_this_task",
            "calibration_metadata_and_tolerance_must_be_versioned_before_full_diagnostic_execution",
        }
    ),
    "prohibited": frozenset({"list", "ground_truth_permitted_uses"}),
    "future_step_bindings": frozenset({"description", "step2_items", "step1b_items"}),
}


# ---------------------------------------------------------------------------
# Canonical scientific text. Every string field that carries the actual
# scientific meaning of the frozen protocol (a definition, formula, rule,
# consequence, or policy that affects interpretation) is pinned here to its
# exact, byte-for-byte frozen value, independently of the TOML file being
# validated at runtime. This mapping is production code, not a fixture: it
# is embedded once, here, and never derived from the identity file it
# checks. A handful of string fields are deliberately NOT listed here
# because their exactness is already enforced by a stronger, independent
# mechanism -- a live SHA-256 file-content hash, a `git merge-base
# --is-ancestor` check, or a hardcoded module constant -- see
# `_RELATIONALLY_GOVERNED_STRING_FIELDS` below. Every other string-typed
# field declared in `SECTION_KEYS` (plus the top-level `format_version`)
# must appear in exactly one of these two registries; this is enforced by
# a coverage test, not merely by convention.
CANONICAL_SCIENTIFIC_STRINGS: dict[tuple[str, str], str] = {
    ("format_version", "format_version"): "talk2dino-dfp-phase1a-protocol-identity-v1",
    ("identity", "name"): "e13-dfp-phase1a-protocol",
    ("identity", "schema_version"): "talk2dino-dfp-phase1a-protocol-identity-v1",
    ("identity", "description"): "DFP Phase 1A pre-registration: freezes the T=0-vs-T=320 stitched-dispute diagnostic protocol (categories, oracle definitions, metrics, gates, structural-component preregistration, query/compute projection, numerical-validation policy) over the already-frozen COCO-Stuff (e12-matched-k11-k12-t320) and ADE20K (e12-ade20k-matched-evaluator) evaluation protocols. Implements no evaluator, no subset manifest, no result schema, and selects no inference method.",
    ("parent_identities", "matched_k11_k12_identity_name"): "e12-matched-k11-k12-t320",
    ("parent_identities", "e3_identity_name"): "e3-paired-soft-routing",
    ("parent_identities", "rwr_identity_name"): "e3-canonical-directed-rwr",
    ("parent_identities", "ade20k_source_identity_name"): "e12-ade20k-dataset-source",
    ("parent_identities", "ade20k_evaluator_identity_name"): "e12-ade20k-matched-evaluator",
    ("parent_identities", "required_relationship"): "freezes_a_diagnostic_over_the_unmodified_matched_k11_k12_and_ade20k_matched_evaluator_protocols",
    ("bridge_checkpoint", "projection_class"): "vitb_mlp_infonce_paired_soft_routing_tau010",
    ("bridge_checkpoint", "bound_via"): "e3-canonical-directed-rwr [checkpoint].sha256, cross-checked against e12-ade20k-matched-evaluator [e3_config].projection_checkpoint_sha256",
    ("coco_stuff_dataset", "name"): "COCO-Stuff 164k validation",
    ("coco_stuff_dataset", "source_authority"): "e12-matched-k11-k12-t320 [dataset] (no separate coco-stuff source-identity file exists in this repository)",
    ("coco_stuff_dataset", "split"): "val2017",
    ("coco_stuff_dataset", "class_names_digest_algorithm"): "sha256_json_dumps_ensure_ascii_list",
    ("coco_stuff_dataset", "class_names_source"): "mmseg.datasets.COCOStuffDataset.CLASSES",
    ("coco_stuff_dataset", "dataset_type"): "COCOStuffDataset",
    ("coco_stuff_dataset", "image_suffix"): ".jpg",
    ("coco_stuff_dataset", "annotation_suffix"): "_labelTrainIds.png",
    ("ade20k_dataset", "name"): "ADE20K (ADEChallengeData2016) validation",
    ("ade20k_dataset", "source_authority"): "e12-ade20k-dataset-source",
    ("ade20k_dataset", "split"): "validation",
    ("ade20k_dataset", "class_names_digest_algorithm"): "sha256_json_dumps_ensure_ascii_list",
    ("ade20k_dataset", "class_names_source"): "mmseg.datasets.ade.ADE20KDataset.CLASSES",
    ("ade20k_dataset", "dataset_type"): "ADE20KDataset",
    ("ade20k_dataset", "image_suffix"): ".jpg",
    ("ade20k_dataset", "annotation_suffix"): ".png",
    ("graph", "affinity_function"): "relu_cosine_power",
    ("graph", "tie_break_rule"): "stable_descending_argsort_lower_patch_index_wins",
    ("graph", "self_edge_policy"): "none_for_ordinary_rows",
    ("graph", "fallback_row_policy"): "zero_affinity_rows_receive_self_loop_weight_one_invariant_to_k",
    ("graph", "canonical_k_variant_note"): "DFP's propagated baseline P uses the unmodified canonical top-12 directed graph (k12) -- the prior, unablated graph from which k11 is itself derived as a matched prefix per e12-matched-k11-k12-t320's own [identity] description. k11 remains available only as the existing connectivity-sensitivity comparator; it is never DFP's primary propagated field.",
    ("propagation", "method"): "finite_power_iteration",
    ("propagation", "initial_iterate"): "S0",
    ("propagation", "recurrence"): "P_next = alpha * A_k12 @ P + (1 - alpha) * S0",
    ("propagation", "convergence_tolerance"): "not_applicable",
    ("propagation", "fallback_solver"): "none",
    ("propagation", "horizon_note"): "T=16, T=32, T=64 are snapshots captured from ONE continuous 320-step finite_step_propagate recurrence (via its existing snapshot_steps mechanism), never separate propagations restarted from S0. T=0 is the unpropagated unary snapshot (E3); T=320 is the final horizon already frozen by e12-matched-k11-k12-t320.",
    ("score_interpolation_stitching", "raw_score_stage"): "pre_sigmoid_pre_upsample_pre_stitch",
    ("score_interpolation_stitching", "averaging"): "uniform",
    ("score_interpolation_stitching", "window_enumeration"): "sliding_window_geometry.SlidingWindowPlan.build (legacy clamped slide_inference grid, row-major flat index order)",
    ("score_interpolation_stitching", "clamping_policy"): "back_shifted_clamp_to_image_bounds",
    ("dispute_protocol", "decision_stage"): "post_canonical_interpolation_and_stitching",
    ("dispute_protocol", "a_definition"): "lowest_index_argmax_of_S0_stitched_prediction",
    ("dispute_protocol", "b_definition"): "lowest_index_argmax_of_S320_stitched_prediction",
    ("dispute_protocol", "tie_rule"): "exact_argmax_ties_select_lowest_class_index",
    ("dispute_protocol", "dispute_set_definition"): "D = {x : a(x) != b(x)}, restricted to non-ignore evaluation pixels",
    ("dispute_categories", "harmful_washout"): "GT(x) == a(x) and GT(x) != b(x)",
    ("dispute_categories", "beneficial_correction"): "GT(x) == b(x) and GT(x) != a(x)",
    ("dispute_categories", "third_label_transition"): "GT(x) not in {a(x), b(x)}",
    ("dispute_categories", "ignore"): "GT(x) equals the dataset's authoritative ignore value",
    ("dispute_categories", "auroc_auprc_population"): "GT(x) in {a(x), b(x)}",
    ("oracle_definitions", "propagated_baseline_P"): "stitched final label map at T=320",
    ("oracle_definitions", "unary_U"): "stitched label map at T=0",
    ("oracle_definitions", "oracle_O_P_rule"): "begin with P; wherever U(x) == GT(x) and P(x) != GT(x), replace P(x) with U(x); retain P(x) everywhere else, including third-label-transition pixels",
    ("oracle_definitions", "g_recover_definition"): "G_recover = metric(O_P) - metric(P)",
    ("oracle_definitions", "g_oracle_unary_definition"): "G_oracle_unary = metric(O_P) - metric(U)",
    ("oracle_definitions", "relational_identity"): "G_oracle_unary = G_recover + (metric(P) - metric(U))",
    ("oracle_definitions", "third_label_diagnostic_rule"): "begin with P; only where GT(x) is neither a(x) nor b(x), replace b(x) with a(x); retain P(x) elsewhere",
    ("oracle_definitions", "third_label_diagnostic_label"): "GT-dependent diagnostic map, never a deployable method",
    ("horizon_persistence", "rule"): "for a T=320 dispute a->b: retain b if the stitched T=t winner is also b, otherwise use unary a; outside T=320 disputes, retain the T=320 prediction unchanged",
    ("metric_contract", "reported_metric_unit"): "percent_0_100",
    ("metric_contract", "difference_unit"): "percentage_points",
    ("metric_contract", "aggregation_rule"): "dataset_level_sum_then_divide_full_precision_integer_sufficient_statistics",
    ("boundary_iou", "definition"): "Cheng et al. 2021 Boundary IoU: IoU restricted to a normalized-width boundary band around the ground-truth/predicted region contours, computed via symmetric erosion of each mask by a pixel radius derived from a fraction of the region's diagonal (dilation-free erosion-based boundary extraction).",
    ("boundary_iou", "primary_width_convention"): "normalized_boundary_width_fraction_of_object_diagonal",
    ("boundary_iou", "implementation_source_status"): "not_yet_bound",
    ("boundary_iou", "implementation_source_note"): "No Boundary IoU implementation exists anywhere in this repository or its installed mmseg/mmcv dependencies as of the required_ancestor_commit (independently searched, zero hits). Per this identity's own no-guessing policy, Phase 1A freezes only the mathematical convention above; binding a concrete authoritative implementation (repository-local or a vetted external one) is an explicit Step-2 decision, exactly like the 500-image subset manifests below -- never invented or locally reimplemented in this step.",
    ("gate_contract", "gate_relation"): "both thresholds evaluated on the SAME dataset d: G_oracle_unary(d) >= 2.0pp AND G_recover(d) >= 1.0pp",
    ("gate_contract", "outcome_neither_dataset_passes_both"): "stop DFP as an inference-method project",
    ("gate_contract", "outcome_exactly_one_dataset_passes_both"): "continue only with a regime-scoped claim restricted to that dataset",
    ("gate_contract", "outcome_both_datasets_pass_both"): "a cross-dataset method claim remains possible",
    ("gate_expectations", "dispute_fraction_too_small_consequence"): "addressable set too small, stop",
    ("gate_expectations", "dispute_fraction_too_large_consequence"): "reframe beyond rare flips",
    ("gate_expectations", "t32_persistence_high_recovery_consequence"): "if T=32 persistence reaches >=60% of oracle G_recover, provenance-based Phase 1B must beat it decisively to justify additional complexity",
    ("gate_expectations", "boundary_mass_reframe_consequence"): "if >70% of harmful-washout mass lies in the primary boundary band, Boundary IoU becomes the primary improvement metric and mIoU becomes secondary/preservation",
    ("run_modes", "coco_stuff_500_source_authority"): "coco_stuff_dataset",
    ("run_modes", "ade20k_500_source_authority"): "ade20k_dataset",
    ("subset_selection_policy", "selection_method"): "deterministic_multilabel_stratification",
    ("subset_selection_policy", "input_domain"): "canonical_validation_ids_in_source_order",
    ("subset_selection_policy", "tie_break_method"): "sha256_based_deterministic",
    ("subset_selection_policy", "concrete_manifest_binding_stage"): "step_2",
    ("structural_components", "component_source"): "canonical_stitched_unary_(T=0)_field",
    ("structural_components", "tie_rule"): "lowest_class_index",
    ("structural_components", "component_scope"): "global_across_the_full_stitched_image_never_per_sliding_window",
    ("structural_components", "soft_membership_formula"): "M[r,w,j] = (sum over x in component r of A[w,x,j]) / (sum over all x of A[w,x,j])",
    ("structural_components", "membership_operator"): "existing_canonical_interpolation_and_stitching_operator_A",
    ("structural_components", "a_transpose_ones_cached_invariant"): "geometry_only",
    ("structural_component_area", "formula"): "area_patch_equivalent = component_pixel_area / projected_patch_pitch_area",
    ("structural_component_area", "projected_pitch_source"): "exact_production_geometry",
    ("query_compute_projection", "q_w_definition"): "Q_w = { a(x), b(x) : x is a stitched T=320 dispute and window w contributes to x }",
    ("query_compute_projection", "required_statistics_subject"): "|Q_w|_per_window",
    ("query_compute_projection", "projected_width_formula"): "R_w * |Q_w|",
    ("numerical_validation_policy", "conservation_summary_accumulation_dtype"): "float64",
    ("numerical_validation_policy", "tolerance_source"): "separate_frozen_200_window_numerical_calibration_artifact",
    ("future_step_bindings", "description"): "Fields explicitly deferred to later DFP steps, never guessed or invented here.",
}

# String fields whose exactness is already independently and at least as
# strongly enforced elsewhere: file-content SHA-256 (checked against the
# real bytes on disk, not merely pattern-matched), a full Git ancestry
# check, or a hardcoded Python module constant. Listed explicitly, rather
# than left as an implicit gap, so the coverage test below can prove every
# string field in the schema is accounted for by exactly one mechanism.
_RELATIONALLY_GOVERNED_STRING_FIELDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("identity", "required_ancestor_commit"),  # == REQUIRED_ANCESTOR_COMMIT, git-ancestry checked
        ("identity", "working_branch"),  # == WORKING_BRANCH module constant
        ("parent_identities", "matched_k11_k12_identity_path"),
        ("parent_identities", "matched_k11_k12_identity_sha256"),
        ("parent_identities", "e3_identity_path"),
        ("parent_identities", "e3_identity_sha256"),
        ("parent_identities", "rwr_identity_path"),
        ("parent_identities", "rwr_identity_sha256"),
        ("parent_identities", "ade20k_source_identity_path"),
        ("parent_identities", "ade20k_source_identity_sha256"),
        ("parent_identities", "ade20k_evaluator_identity_path"),
        ("parent_identities", "ade20k_evaluator_identity_sha256"),
        ("bridge_checkpoint", "relative_path"),
        ("bridge_checkpoint", "sha256"),
        ("coco_stuff_dataset", "class_names_digest"),
        ("ade20k_dataset", "class_names_digest"),
    }
)

# Frozen exact-membership content for the three list-of-string fields that
# enumerate a scientifically material closed set (what is prohibited, what
# is explicitly deferred) rather than a single sentence. Order does not
# carry meaning for these three; membership does, so equality is by set.
PROHIBITED_LIST_CANONICAL: frozenset[str] = frozenset(
    {
        "training_or_fine_tuning",
        "optimizer_or_gradient_use",
        "target_feature_extraction_as_separate_preprocessing_stage",
        "label_derived_prediction_construction",
        "masked_component_propagation",
        "semantic_supercohort_propagation",
        "phase1b_provenance_computation",
        "gate3_selection",
        "learned_gates",
        "morphology",
        "tuned_confidence_thresholds",
        "model_or_checkpoint_creation",
        "changing_graph_or_recurrence_mathematics",
        "cgls",
        "gmres",
        "dense_inverse_or_solve",
        "solver_fallback",
        "pamr",
        "creating_500_image_subsets",
        "implementing_phase1a_execution",
        "implementing_masked_propagation",
        "inference",
        "cuda_initialization",
        "dataset_access_or_modification",
    }
)

STEP2_ITEMS_CANONICAL: frozenset[str] = frozenset(
    {
        "coco_stuff_500 concrete manifest (ordered IDs, ordered-ID digest)",
        "ade20k_500 concrete manifest (ordered IDs, ordered-ID digest)",
        "boundary_iou concrete implementation source binding",
        "200-window numerical-calibration artifact and its derived tolerance",
    }
)

STEP1B_ITEMS_CANONICAL: frozenset[str] = frozenset(
    {
        "adjoint/masked-forward router",
        "Phase 1B provenance computation",
    }
)


class DfpPhase1aIdentityError(ValueError):
    """Raised on any DFP Phase 1A protocol identity invariant violation.
    Always fail closed."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str:
        raise DfpPhase1aIdentityError(f"{label} must be an exact string")
    if nonempty and not value.strip():
        raise DfpPhase1aIdentityError(f"{label} must be a non-empty, non-whitespace string")
    return value


def _require_canonical_string(value: Any, section: str, field: str) -> str:
    """Validate a scientifically material string field: exact type, exact
    non-empty content, AND exact byte-for-byte equality against its
    independently embedded canonical text in `CANONICAL_SCIENTIFIC_STRINGS`.
    Never strips, case-folds, or otherwise normalizes the input -- a
    prefixed/suffixed space, an internal whitespace change, a case change,
    or any other one-character deviation is rejected."""
    label = f"{section}.{field}"
    token = _require_exact_string(value, label)
    canonical = CANONICAL_SCIENTIFIC_STRINGS[(section, field)]
    if token != canonical:
        raise DfpPhase1aIdentityError(f"{label} disagrees with the frozen canonical scientific text")
    return token


def _require_canonical_string_set(value: Any, label: str, canonical: frozenset[str]) -> tuple[str, ...]:
    items = _require_exact_string_list(value, label)
    if set(items) != canonical:
        missing = canonical - set(items)
        unknown = set(items) - canonical
        raise DfpPhase1aIdentityError(f"{label} disagrees with the frozen canonical set (missing={sorted(missing)}, unknown={sorted(unknown)})")
    return items


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1aIdentityError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise DfpPhase1aIdentityError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise DfpPhase1aIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_exact_float(value: Any, label: str) -> float:
    import math

    if type(value) is not float:
        raise DfpPhase1aIdentityError(f"{label} must be an exact finite float")
    if not math.isfinite(value):
        raise DfpPhase1aIdentityError(f"{label} must be finite (no NaN/Infinity)")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise DfpPhase1aIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", token) is None:
        raise DfpPhase1aIdentityError(f"{label} must be a full Git identity")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token:
        raise DfpPhase1aIdentityError(f"{label} must be a safe repository-relative path")
    return token


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DfpPhase1aIdentityError(f"{label} must be a mapping")
    if set(value) != expected_keys:
        missing = expected_keys - set(value)
        unknown = set(value) - expected_keys
        raise DfpPhase1aIdentityError(f"{label} has an unexpected schema (missing={sorted(missing)}, unknown={sorted(unknown)})")
    return value


def _require_exact_string_list(value: Any, label: str, *, nonempty: bool = True) -> tuple[str, ...]:
    if type(value) is not list or (nonempty and not value):
        raise DfpPhase1aIdentityError(f"{label} must be a non-empty exact list")
    if any(type(item) is not str for item in value):
        raise DfpPhase1aIdentityError(f"{label} elements must be exact strings")
    return tuple(value)


def _require_exact_int_list(value: Any, label: str) -> tuple[int, ...]:
    if type(value) is not list or not value:
        raise DfpPhase1aIdentityError(f"{label} must be a non-empty exact list")
    for item in value:
        if type(item) is not int:
            raise DfpPhase1aIdentityError(f"{label} elements must be exact non-boolean integers")
    return tuple(value)


def _require_exact_float_list(value: Any, label: str) -> tuple[float, ...]:
    if type(value) is not list or not value:
        raise DfpPhase1aIdentityError(f"{label} must be a non-empty exact list")
    for item in value:
        _require_exact_float(item, f"{label} element")
    return tuple(value)


def _require_exact_int_pair(value: Any, label: str) -> tuple[int, int]:
    if type(value) is not list or len(value) != 2:
        raise DfpPhase1aIdentityError(f"{label} must be an exact two-element list")
    if any(type(item) is not int for item in value):
        raise DfpPhase1aIdentityError(f"{label} elements must be exact integers")
    return (value[0], value[1])


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DfpPhase1aIdentityError(f"cannot load DFP Phase 1A protocol identity {source}: {error}") from error

    if set(identity) != IDENTITY_TOP_KEYS:
        missing = IDENTITY_TOP_KEYS - set(identity)
        unknown = set(identity) - IDENTITY_TOP_KEYS
        raise DfpPhase1aIdentityError(f"identity has an unexpected top-level schema (missing={sorted(missing)}, unknown={sorted(unknown)})")
    _require_canonical_string(identity["format_version"], "format_version", "format_version")
    for section, keys in SECTION_KEYS.items():
        _require_closed_mapping(identity.get(section), keys, f"identity.{section}")

    _validate_identity_block(identity["identity"])
    _validate_parent_identities_block(identity["parent_identities"])
    _validate_bridge_checkpoint_block(identity["bridge_checkpoint"])
    _validate_dataset_block(identity["coco_stuff_dataset"], label="coco_stuff_dataset")
    _validate_dataset_block(identity["ade20k_dataset"], label="ade20k_dataset")
    _validate_graph_block(identity["graph"])
    _validate_propagation_block(identity["propagation"])
    _validate_score_interpolation_stitching_block(identity["score_interpolation_stitching"])
    _validate_dispute_protocol_block(identity["dispute_protocol"])
    _validate_dispute_categories_block(identity["dispute_categories"])
    _validate_oracle_definitions_block(identity["oracle_definitions"])
    _validate_horizon_persistence_block(identity["horizon_persistence"], identity["propagation"])
    _validate_metric_contract_block(identity["metric_contract"])
    _validate_boundary_iou_block(identity["boundary_iou"])
    _validate_gate_contract_block(identity["gate_contract"])
    _validate_gate_expectations_block(identity["gate_expectations"])
    _validate_run_modes_block(identity["run_modes"])
    _validate_subset_selection_policy_block(identity["subset_selection_policy"])
    _validate_structural_components_block(identity["structural_components"])
    _validate_structural_component_area_block(identity["structural_component_area"])
    _validate_query_compute_projection_block(identity["query_compute_projection"])
    _validate_numerical_validation_policy_block(identity["numerical_validation_policy"])
    _validate_prohibited_block(identity["prohibited"])
    _validate_future_step_bindings_block(identity["future_step_bindings"])

    return identity


def _validate_identity_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["name"], "identity", "name")
    _require_canonical_string(block["schema_version"], "identity", "schema_version")
    _require_canonical_string(block["description"], "identity", "description")
    if _require_git_identity(block["required_ancestor_commit"], "identity.required_ancestor_commit") != REQUIRED_ANCESTOR_COMMIT:
        raise DfpPhase1aIdentityError(f"identity.required_ancestor_commit must be exactly {REQUIRED_ANCESTOR_COMMIT}")
    if block["working_branch"] != WORKING_BRANCH:
        raise DfpPhase1aIdentityError(f"identity.working_branch must be exactly {WORKING_BRANCH!r}")


def _validate_parent_identities_block(block: Mapping[str, Any]) -> None:
    for prefix in ("matched_k11_k12", "e3", "rwr", "ade20k_source", "ade20k_evaluator"):
        _require_relative_path(block[f"{prefix}_identity_path"], f"parent_identities.{prefix}_identity_path")
        _require_canonical_string(block[f"{prefix}_identity_name"], "parent_identities", f"{prefix}_identity_name")
        _require_sha256(block[f"{prefix}_identity_sha256"], f"parent_identities.{prefix}_identity_sha256")
    _require_canonical_string(block["required_relationship"], "parent_identities", "required_relationship")


def _validate_bridge_checkpoint_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["projection_class"], "bridge_checkpoint", "projection_class")
    _require_relative_path(block["relative_path"], "bridge_checkpoint.relative_path")
    _require_sha256(block["sha256"], "bridge_checkpoint.sha256")
    _require_canonical_string(block["bound_via"], "bridge_checkpoint", "bound_via")
    if _require_exact_bool(block["frozen"], "bridge_checkpoint.frozen") is not True:
        raise DfpPhase1aIdentityError("bridge_checkpoint.frozen must be true")


def _validate_dataset_block(block: Mapping[str, Any], *, label: str) -> None:
    _require_canonical_string(block["name"], label, "name")
    _require_canonical_string(block["source_authority"], label, "source_authority")
    _require_canonical_string(block["split"], label, "split")
    _require_exact_int(block["images"], f"{label}.images", minimum=1)
    _require_exact_int(block["class_count"], f"{label}.class_count", minimum=1)
    _require_sha256(block["class_names_digest"], f"{label}.class_names_digest")
    _require_canonical_string(block["class_names_digest_algorithm"], label, "class_names_digest_algorithm")
    _require_canonical_string(block["class_names_source"], label, "class_names_source")
    _require_canonical_string(block["dataset_type"], label, "dataset_type")
    _require_canonical_string(block["image_suffix"], label, "image_suffix")
    _require_canonical_string(block["annotation_suffix"], label, "annotation_suffix")
    if _require_exact_int(block["ignore_index"], f"{label}.ignore_index") != 255:
        raise DfpPhase1aIdentityError(f"{label}.ignore_index must be exactly 255")
    if _require_exact_bool(block["background_class_evaluated"], f"{label}.background_class_evaluated") is not False:
        raise DfpPhase1aIdentityError(f"{label}.background_class_evaluated must be false")
    _require_exact_bool(block["reduce_zero_label"], f"{label}.reduce_zero_label")


def _validate_graph_block(block: Mapping[str, Any]) -> None:
    if _require_exact_bool(block["directed"], "graph.directed") is not True:
        raise DfpPhase1aIdentityError("graph.directed must be true")
    if _require_exact_bool(block["row_stochastic"], "graph.row_stochastic") is not True:
        raise DfpPhase1aIdentityError("graph.row_stochastic must be true")
    if _require_exact_int(block["maximum_rank"], "graph.maximum_rank") != 12:
        raise DfpPhase1aIdentityError("graph.maximum_rank must be exactly 12")
    _require_canonical_string(block["affinity_function"], "graph", "affinity_function")
    if _require_exact_float(block["affinity_power"], "graph.affinity_power") != 3.0:
        raise DfpPhase1aIdentityError("graph.affinity_power must be exactly 3.0")
    _require_canonical_string(block["tie_break_rule"], "graph", "tie_break_rule")
    _require_canonical_string(block["self_edge_policy"], "graph", "self_edge_policy")
    _require_canonical_string(block["fallback_row_policy"], "graph", "fallback_row_policy")
    if _require_exact_bool(block["graph_symmetrization"], "graph.graph_symmetrization") is not False:
        raise DfpPhase1aIdentityError("graph.graph_symmetrization must be false")
    if _require_exact_bool(block["artificial_tie_perturbation"], "graph.artificial_tie_perturbation") is not False:
        raise DfpPhase1aIdentityError("graph.artificial_tie_perturbation must be false")
    if _require_exact_int(block["canonical_k_variant"], "graph.canonical_k_variant") != 12:
        raise DfpPhase1aIdentityError("graph.canonical_k_variant must be exactly 12")
    _require_canonical_string(block["canonical_k_variant_note"], "graph", "canonical_k_variant_note")


def validate_graph_block(block: Mapping[str, Any]) -> None:
    """Public delegate to the canonical [graph] block validator, so other
    modules (e.g. a DFP-specific recurrence binding layer) can
    independently re-validate a caller-supplied graph block -- including
    ``canonical_k_variant`` -- against this single authority without
    duplicating any of its pinned values."""
    _validate_graph_block(block)


def _validate_propagation_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["method"], "propagation", "method")
    if _require_exact_float(block["alpha"], "propagation.alpha") != 0.98:
        raise DfpPhase1aIdentityError("propagation.alpha must be exactly 0.98")
    if _require_exact_int(block["steps"], "propagation.steps") != 320:
        raise DfpPhase1aIdentityError("propagation.steps must be exactly 320")
    _require_canonical_string(block["initial_iterate"], "propagation", "initial_iterate")
    _require_canonical_string(block["recurrence"], "propagation", "recurrence")
    if _require_exact_bool(block["early_stopping"], "propagation.early_stopping") is not False:
        raise DfpPhase1aIdentityError("propagation.early_stopping must be false")
    _require_canonical_string(block["convergence_tolerance"], "propagation", "convergence_tolerance")
    _require_canonical_string(block["fallback_solver"], "propagation", "fallback_solver")
    horizons = _require_exact_int_list(block["horizons"], "propagation.horizons")
    if horizons != (0, 16, 32, 64, 320):
        raise DfpPhase1aIdentityError(f"propagation.horizons must be exactly [0, 16, 32, 64, 320], observed {list(horizons)}")
    _require_canonical_string(block["horizon_note"], "propagation", "horizon_note")
    if _require_exact_int(block["primary_sensitivity_horizon"], "propagation.primary_sensitivity_horizon") != 32:
        raise DfpPhase1aIdentityError("propagation.primary_sensitivity_horizon must be exactly 32")
    secondary = _require_exact_int_list(block["secondary_sensitivity_horizons"], "propagation.secondary_sensitivity_horizons")
    if secondary != (16, 64):
        raise DfpPhase1aIdentityError(f"propagation.secondary_sensitivity_horizons must be exactly [16, 64], observed {list(secondary)}")


def validate_propagation_block(block: Mapping[str, Any]) -> None:
    """Public delegate to the canonical [propagation] block validator, so
    other modules (e.g. a DFP-specific recurrence binding layer) can
    independently re-validate a caller-supplied propagation block --
    alpha, steps, horizons, early_stopping, fallback_solver, and the rest
    -- against this single authority without duplicating any of its
    pinned values."""
    _validate_propagation_block(block)


def _validate_score_interpolation_stitching_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["raw_score_stage"], "score_interpolation_stitching", "raw_score_stage")
    if _require_exact_int_pair(block["crop"], "score_interpolation_stitching.crop") != (448, 448):
        raise DfpPhase1aIdentityError("score_interpolation_stitching.crop must be exactly [448, 448]")
    if _require_exact_int_pair(block["stride"], "score_interpolation_stitching.stride") != (224, 224):
        raise DfpPhase1aIdentityError("score_interpolation_stitching.stride must be exactly [224, 224]")
    if _require_exact_bool(block["align_corners"], "score_interpolation_stitching.align_corners") is not True:
        raise DfpPhase1aIdentityError("score_interpolation_stitching.align_corners must be true")
    _require_canonical_string(block["averaging"], "score_interpolation_stitching", "averaging")
    _require_canonical_string(block["window_enumeration"], "score_interpolation_stitching", "window_enumeration")
    _require_canonical_string(block["clamping_policy"], "score_interpolation_stitching", "clamping_policy")
    for flag in ("hann", "majority_vote", "center_selection", "sparse_delta_restitching"):
        if _require_exact_bool(block[flag], f"score_interpolation_stitching.{flag}") is not False:
            raise DfpPhase1aIdentityError(f"score_interpolation_stitching.{flag} must be false")
    if _require_exact_bool(block["identical_across_horizons"], "score_interpolation_stitching.identical_across_horizons") is not True:
        raise DfpPhase1aIdentityError("score_interpolation_stitching.identical_across_horizons must be true")


def validate_score_interpolation_stitching_block(block: Mapping[str, Any]) -> None:
    """Public delegate to the canonical [score_interpolation_stitching]
    block validator, so other modules (e.g. a DFP-specific horizon
    stitching binding layer) can independently re-validate a
    caller-supplied stitching block -- crop, stride, align_corners,
    averaging/hann/majority_vote/center_selection/sparse_delta_restitching,
    window_enumeration, raw_score_stage, identical_across_horizons --
    against this single authority without duplicating any of its pinned
    values."""
    _validate_score_interpolation_stitching_block(block)


def _validate_dispute_protocol_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["decision_stage"], "dispute_protocol", "decision_stage")
    _require_canonical_string(block["a_definition"], "dispute_protocol", "a_definition")
    _require_canonical_string(block["b_definition"], "dispute_protocol", "b_definition")
    _require_canonical_string(block["tie_rule"], "dispute_protocol", "tie_rule")
    if _require_exact_bool(block["random_tie_break_allowed"], "dispute_protocol.random_tie_break_allowed") is not False:
        raise DfpPhase1aIdentityError("dispute_protocol.random_tie_break_allowed must be false")
    if _require_exact_bool(block["numpy_torch_tie_parity_required"], "dispute_protocol.numpy_torch_tie_parity_required") is not True:
        raise DfpPhase1aIdentityError("dispute_protocol.numpy_torch_tie_parity_required must be true")
    _require_canonical_string(block["dispute_set_definition"], "dispute_protocol", "dispute_set_definition")


def _validate_dispute_categories_block(block: Mapping[str, Any]) -> None:
    for field in ("harmful_washout", "beneficial_correction", "third_label_transition", "ignore", "auroc_auprc_population"):
        _require_canonical_string(block[field], "dispute_categories", field)
    if _require_exact_bool(
        block["third_label_and_ignore_excluded_from_negative_class"],
        "dispute_categories.third_label_and_ignore_excluded_from_negative_class",
    ) is not True:
        raise DfpPhase1aIdentityError("dispute_categories.third_label_and_ignore_excluded_from_negative_class must be true")


def _validate_oracle_definitions_block(block: Mapping[str, Any]) -> None:
    for field in (
        "propagated_baseline_P", "unary_U", "oracle_O_P_rule", "g_recover_definition", "g_oracle_unary_definition",
        "relational_identity", "third_label_diagnostic_rule", "third_label_diagnostic_label",
    ):
        _require_canonical_string(block[field], "oracle_definitions", field)


def _validate_horizon_persistence_block(block: Mapping[str, Any], propagation_block: Mapping[str, Any]) -> None:
    applies_to = _require_exact_int_list(block["applies_to_horizons"], "horizon_persistence.applies_to_horizons")
    if applies_to != (16, 32, 64):
        raise DfpPhase1aIdentityError(f"horizon_persistence.applies_to_horizons must be exactly [16, 32, 64], observed {list(applies_to)}")
    _require_canonical_string(block["rule"], "horizon_persistence", "rule")
    if _require_exact_int(block["primary_horizon"], "horizon_persistence.primary_horizon") != propagation_block["primary_sensitivity_horizon"]:
        raise DfpPhase1aIdentityError("horizon_persistence.primary_horizon disagrees with propagation.primary_sensitivity_horizon")
    sensitivity = _require_exact_int_list(block["sensitivity_horizons"], "horizon_persistence.sensitivity_horizons")
    if sensitivity != tuple(propagation_block["secondary_sensitivity_horizons"]):
        raise DfpPhase1aIdentityError("horizon_persistence.sensitivity_horizons disagrees with propagation.secondary_sensitivity_horizons")


def _validate_metric_contract_block(block: Mapping[str, Any]) -> None:
    required_stats = _require_exact_string_list(block["required_statistics"], "metric_contract.required_statistics")
    if set(required_stats) != {"intersection", "union", "predicted_pixels", "ground_truth_pixels"}:
        raise DfpPhase1aIdentityError("metric_contract.required_statistics must be exactly the four sufficient statistics")
    required_metrics = _require_exact_string_list(block["required_metrics"], "metric_contract.required_metrics")
    if set(required_metrics) != {"aAcc", "mAcc", "mIoU", "boundary_iou"}:
        raise DfpPhase1aIdentityError("metric_contract.required_metrics must be exactly {aAcc, mAcc, mIoU, boundary_iou}")
    _require_canonical_string(block["reported_metric_unit"], "metric_contract", "reported_metric_unit")
    _require_canonical_string(block["difference_unit"], "metric_contract", "difference_unit")
    _require_canonical_string(block["aggregation_rule"], "metric_contract", "aggregation_rule")
    if _require_exact_bool(block["per_class_statistics_required"], "metric_contract.per_class_statistics_required") is not True:
        raise DfpPhase1aIdentityError("metric_contract.per_class_statistics_required must be true")
    prohibited = _require_exact_string_list(block["prohibited_reductions"], "metric_contract.prohibited_reductions")
    expected_prohibited = {
        "mean_of_per_image_miou", "summed_per_pixel_miou_contributions",
        "union_weighted_proxy_gains", "rounded_value_labeled_full_precision",
    }
    if set(prohibited) != expected_prohibited:
        raise DfpPhase1aIdentityError("metric_contract.prohibited_reductions must be exactly the four frozen prohibitions")


def _validate_boundary_iou_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["definition"], "boundary_iou", "definition")
    _require_canonical_string(block["primary_width_convention"], "boundary_iou", "primary_width_convention")
    widths = _require_exact_int_list(block["sensitivity_pixel_widths"], "boundary_iou.sensitivity_pixel_widths")
    if widths != (1, 2, 4):
        raise DfpPhase1aIdentityError(f"boundary_iou.sensitivity_pixel_widths must be exactly [1, 2, 4], observed {list(widths)}")
    _require_canonical_string(block["implementation_source_status"], "boundary_iou", "implementation_source_status")
    _require_canonical_string(block["implementation_source_note"], "boundary_iou", "implementation_source_note")


def _validate_gate_contract_block(block: Mapping[str, Any]) -> None:
    if _require_exact_float(block["oracle_unary_gate_threshold_percentage_points"], "gate_contract.oracle_unary_gate_threshold_percentage_points") != 2.0:
        raise DfpPhase1aIdentityError("gate_contract.oracle_unary_gate_threshold_percentage_points must be exactly 2.0")
    if _require_exact_float(block["recover_gate_threshold_percentage_points"], "gate_contract.recover_gate_threshold_percentage_points") != 1.0:
        raise DfpPhase1aIdentityError("gate_contract.recover_gate_threshold_percentage_points must be exactly 1.0")
    for field in ("gate_relation", "outcome_neither_dataset_passes_both", "outcome_exactly_one_dataset_passes_both", "outcome_both_datasets_pass_both"):
        _require_canonical_string(block[field], "gate_contract", field)
    if _require_exact_bool(
        block["cross_dataset_pass_requires_independent_evaluation"], "gate_contract.cross_dataset_pass_requires_independent_evaluation"
    ) is not True:
        raise DfpPhase1aIdentityError("gate_contract.cross_dataset_pass_requires_independent_evaluation must be true")


def _validate_gate_expectations_block(block: Mapping[str, Any]) -> None:
    low = _require_exact_float(block["expected_dispute_fraction_low"], "gate_expectations.expected_dispute_fraction_low")
    high = _require_exact_float(block["expected_dispute_fraction_high"], "gate_expectations.expected_dispute_fraction_high")
    if (low, high) != (0.05, 0.25):
        raise DfpPhase1aIdentityError("gate_expectations expected_dispute_fraction_low/high must be exactly 0.05/0.25")
    if _require_exact_float(block["dispute_fraction_too_small_threshold"], "gate_expectations.dispute_fraction_too_small_threshold") != 0.02:
        raise DfpPhase1aIdentityError("gate_expectations.dispute_fraction_too_small_threshold must be exactly 0.02")
    _require_canonical_string(block["dispute_fraction_too_small_consequence"], "gate_expectations", "dispute_fraction_too_small_consequence")
    if _require_exact_float(block["dispute_fraction_too_large_threshold"], "gate_expectations.dispute_fraction_too_large_threshold") != 0.40:
        raise DfpPhase1aIdentityError("gate_expectations.dispute_fraction_too_large_threshold must be exactly 0.40")
    _require_canonical_string(block["dispute_fraction_too_large_consequence"], "gate_expectations", "dispute_fraction_too_large_consequence")
    if _require_exact_float(block["gt_in_ab_coverage_expected_minimum"], "gate_expectations.gt_in_ab_coverage_expected_minimum") != 0.40:
        raise DfpPhase1aIdentityError("gate_expectations.gt_in_ab_coverage_expected_minimum must be exactly 0.40")
    if _require_exact_float(
        block["t32_persistence_recovery_expected_maximum_fraction"], "gate_expectations.t32_persistence_recovery_expected_maximum_fraction"
    ) != 0.60:
        raise DfpPhase1aIdentityError("gate_expectations.t32_persistence_recovery_expected_maximum_fraction must be exactly 0.60")
    _require_canonical_string(block["t32_persistence_high_recovery_consequence"], "gate_expectations", "t32_persistence_high_recovery_consequence")
    if _require_exact_float(block["boundary_mass_reframe_threshold"], "gate_expectations.boundary_mass_reframe_threshold") != 0.70:
        raise DfpPhase1aIdentityError("gate_expectations.boundary_mass_reframe_threshold must be exactly 0.70")
    _require_canonical_string(block["boundary_mass_reframe_consequence"], "gate_expectations", "boundary_mass_reframe_consequence")
    if _require_exact_bool(
        block["expectations_are_soft_not_hard_validity_checks"], "gate_expectations.expectations_are_soft_not_hard_validity_checks"
    ) is not True:
        raise DfpPhase1aIdentityError("gate_expectations.expectations_are_soft_not_hard_validity_checks must be true")


def _validate_run_modes_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["coco_stuff_500_source_authority"], "run_modes", "coco_stuff_500_source_authority")
    if _require_exact_int(block["coco_stuff_500_required_unique_count"], "run_modes.coco_stuff_500_required_unique_count") != 500:
        raise DfpPhase1aIdentityError("run_modes.coco_stuff_500_required_unique_count must be exactly 500")
    _require_canonical_string(block["ade20k_500_source_authority"], "run_modes", "ade20k_500_source_authority")
    if _require_exact_int(block["ade20k_500_required_unique_count"], "run_modes.ade20k_500_required_unique_count") != 500:
        raise DfpPhase1aIdentityError("run_modes.ade20k_500_required_unique_count must be exactly 500")
    names = _require_exact_string_list(block["run_mode_names"], "run_modes.run_mode_names")
    if names != ("coco_stuff_500", "ade20k_500"):
        raise DfpPhase1aIdentityError(f"run_modes.run_mode_names must be exactly ['coco_stuff_500', 'ade20k_500'], observed {list(names)}")


def _validate_subset_selection_policy_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["selection_method"], "subset_selection_policy", "selection_method")
    _require_canonical_string(block["input_domain"], "subset_selection_policy", "input_domain")
    _require_canonical_string(block["tie_break_method"], "subset_selection_policy", "tie_break_method")
    if _require_exact_int(block["required_unique_count"], "subset_selection_policy.required_unique_count") != 500:
        raise DfpPhase1aIdentityError("subset_selection_policy.required_unique_count must be exactly 500")
    must_contain = _require_exact_string_list(block["manifest_must_contain"], "subset_selection_policy.manifest_must_contain")
    if set(must_contain) != {"ordered_id_digest", "dataset_source_identity_sha256", "class_presence_summary"}:
        raise DfpPhase1aIdentityError("subset_selection_policy.manifest_must_contain must be exactly the three frozen fields")
    if _require_exact_bool(block["manifest_private_paths_forbidden"], "subset_selection_policy.manifest_private_paths_forbidden") is not True:
        raise DfpPhase1aIdentityError("subset_selection_policy.manifest_private_paths_forbidden must be true")
    if _require_exact_bool(
        block["manifest_must_be_immutably_committed_before_predictions"], "subset_selection_policy.manifest_must_be_immutably_committed_before_predictions"
    ) is not True:
        raise DfpPhase1aIdentityError("subset_selection_policy.manifest_must_be_immutably_committed_before_predictions must be true")
    if _require_exact_bool(
        block["concrete_manifest_ids_bound_in_this_identity"], "subset_selection_policy.concrete_manifest_ids_bound_in_this_identity"
    ) is not False:
        raise DfpPhase1aIdentityError("subset_selection_policy.concrete_manifest_ids_bound_in_this_identity must be false")
    _require_canonical_string(block["concrete_manifest_binding_stage"], "subset_selection_policy", "concrete_manifest_binding_stage")


def _validate_structural_components_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["component_source"], "structural_components", "component_source")
    if _require_exact_int(block["primary_connectivity"], "structural_components.primary_connectivity") != 8:
        raise DfpPhase1aIdentityError("structural_components.primary_connectivity must be exactly 8")
    if _require_exact_int(block["sensitivity_connectivity"], "structural_components.sensitivity_connectivity") != 4:
        raise DfpPhase1aIdentityError("structural_components.sensitivity_connectivity must be exactly 4")
    _require_canonical_string(block["tie_rule"], "structural_components", "tie_rule")
    _require_canonical_string(block["component_scope"], "structural_components", "component_scope")
    _require_canonical_string(block["soft_membership_formula"], "structural_components", "soft_membership_formula")
    _require_canonical_string(block["membership_operator"], "structural_components", "membership_operator")
    for field in (
        "membership_dense_matrix_forbidden", "adjoint_identity_required", "membership_nonnegative_required",
        "membership_sums_to_one_per_window_patch_source_required", "flush_edge_windows_included",
    ):
        if _require_exact_bool(block[field], f"structural_components.{field}") is not True:
            raise DfpPhase1aIdentityError(f"structural_components.{field} must be true")
    _require_canonical_string(block["a_transpose_ones_cached_invariant"], "structural_components", "a_transpose_ones_cached_invariant")


def _validate_structural_component_area_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["formula"], "structural_component_area", "formula")
    _require_canonical_string(block["projected_pitch_source"], "structural_component_area", "projected_pitch_source")
    if _require_exact_float(
        block["primary_microcomponent_threshold_patch_equivalents"], "structural_component_area.primary_microcomponent_threshold_patch_equivalents"
    ) != 4.0:
        raise DfpPhase1aIdentityError("structural_component_area.primary_microcomponent_threshold_patch_equivalents must be exactly 4.0")
    thresholds = _require_exact_float_list(block["sensitivity_thresholds_patch_equivalents"], "structural_component_area.sensitivity_thresholds_patch_equivalents")
    if thresholds != (1.0, 2.0, 4.0, 8.0):
        raise DfpPhase1aIdentityError(f"structural_component_area.sensitivity_thresholds_patch_equivalents must be exactly [1.0, 2.0, 4.0, 8.0], observed {list(thresholds)}")
    if _require_exact_bool(block["source_copy_mass_is_diagnostic_only"], "structural_component_area.source_copy_mass_is_diagnostic_only") is not True:
        raise DfpPhase1aIdentityError("structural_component_area.source_copy_mass_is_diagnostic_only must be true")
    if _require_exact_bool(
        block["source_copy_mass_is_not_physical_component_area"], "structural_component_area.source_copy_mass_is_not_physical_component_area"
    ) is not True:
        raise DfpPhase1aIdentityError("structural_component_area.source_copy_mass_is_not_physical_component_area must be true")
    if _require_exact_bool(block["phase1a_propagates_component_masks"], "structural_component_area.phase1a_propagates_component_masks") is not False:
        raise DfpPhase1aIdentityError("structural_component_area.phase1a_propagates_component_masks must be false")


def _validate_query_compute_projection_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["q_w_definition"], "query_compute_projection", "q_w_definition")
    stats = _require_exact_string_list(block["required_statistics"], "query_compute_projection.required_statistics")
    if stats != ("mean", "median", "p90", "p95", "p99", "maximum"):
        raise DfpPhase1aIdentityError(f"query_compute_projection.required_statistics must be exactly the six frozen statistics, observed {list(stats)}")
    _require_canonical_string(block["required_statistics_subject"], "query_compute_projection", "required_statistics_subject")
    if _require_exact_bool(
        block["components_with_nonzero_soft_membership_per_window_required"], "query_compute_projection.components_with_nonzero_soft_membership_per_window_required"
    ) is not True:
        raise DfpPhase1aIdentityError("query_compute_projection.components_with_nonzero_soft_membership_per_window_required must be true")
    _require_canonical_string(block["projected_width_formula"], "query_compute_projection", "projected_width_formula")
    if _require_exact_bool(
        block["total_and_worst_window_projected_width_required"], "query_compute_projection.total_and_worst_window_projected_width_required"
    ) is not True:
        raise DfpPhase1aIdentityError("query_compute_projection.total_and_worst_window_projected_width_required must be true")
    if _require_exact_bool(
        block["projected_peak_memory_for_phase1b_required"], "query_compute_projection.projected_peak_memory_for_phase1b_required"
    ) is not True:
        raise DfpPhase1aIdentityError("query_compute_projection.projected_peak_memory_for_phase1b_required must be true")
    if _require_exact_bool(
        block["adjoint_or_masked_forward_router_defined_in_this_task"], "query_compute_projection.adjoint_or_masked_forward_router_defined_in_this_task"
    ) is not False:
        raise DfpPhase1aIdentityError("query_compute_projection.adjoint_or_masked_forward_router_defined_in_this_task must be false")


def _validate_numerical_validation_policy_block(block: Mapping[str, Any]) -> None:
    if _require_exact_bool(
        block["canonical_and_decomposed_same_device_and_precision_required"], "numerical_validation_policy.canonical_and_decomposed_same_device_and_precision_required"
    ) is not True:
        raise DfpPhase1aIdentityError("numerical_validation_policy.canonical_and_decomposed_same_device_and_precision_required must be true")
    _require_canonical_string(block["conservation_summary_accumulation_dtype"], "numerical_validation_policy", "conservation_summary_accumulation_dtype")
    levels = _require_exact_string_list(block["validation_levels"], "numerical_validation_policy.validation_levels")
    if levels != ("seed", "window", "stitched_field", "margin"):
        raise DfpPhase1aIdentityError(f"numerical_validation_policy.validation_levels must be exactly the four frozen levels, observed {list(levels)}")
    residuals = _require_exact_string_list(block["residual_kinds_required"], "numerical_validation_policy.residual_kinds_required")
    if set(residuals) != {"absolute", "relative"}:
        raise DfpPhase1aIdentityError("numerical_validation_policy.residual_kinds_required must be exactly {absolute, relative}")
    _require_canonical_string(block["tolerance_source"], "numerical_validation_policy", "tolerance_source")
    if _require_exact_bool(block["tolerance_invented_in_this_identity"], "numerical_validation_policy.tolerance_invented_in_this_identity") is not False:
        raise DfpPhase1aIdentityError("numerical_validation_policy.tolerance_invented_in_this_identity must be false")
    if _require_exact_bool(block["calibration_artifact_created_in_this_task"], "numerical_validation_policy.calibration_artifact_created_in_this_task") is not False:
        raise DfpPhase1aIdentityError("numerical_validation_policy.calibration_artifact_created_in_this_task must be false")
    if _require_exact_bool(
        block["calibration_metadata_and_tolerance_must_be_versioned_before_full_diagnostic_execution"],
        "numerical_validation_policy.calibration_metadata_and_tolerance_must_be_versioned_before_full_diagnostic_execution",
    ) is not True:
        raise DfpPhase1aIdentityError("numerical_validation_policy.calibration_metadata_and_tolerance_must_be_versioned_before_full_diagnostic_execution must be true")


def _validate_prohibited_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string_set(block["list"], "prohibited.list", PROHIBITED_LIST_CANONICAL)
    permitted = _require_exact_string_list(block["ground_truth_permitted_uses"], "prohibited.ground_truth_permitted_uses")
    expected_permitted = {"evaluation", "stratification_or_class_presence_metadata", "oracle_diagnostics", "preregistered_analysis"}
    if set(permitted) != expected_permitted:
        raise DfpPhase1aIdentityError("prohibited.ground_truth_permitted_uses must be exactly the four frozen permitted uses")


def _validate_future_step_bindings_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["description"], "future_step_bindings", "description")
    _require_canonical_string_set(block["step2_items"], "future_step_bindings.step2_items", STEP2_ITEMS_CANONICAL)
    _require_canonical_string_set(block["step1b_items"], "future_step_bindings.step1b_items", STEP1B_ITEMS_CANONICAL)


def _check_git_ancestry(root: Path, commit: str, *, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"{label} is not an ancestor of HEAD"
        raise DfpPhase1aIdentityError(f"Git ancestry check failed for {label} ({commit}): {detail}")


def _validate_file_binding(root: Path, relative_path: str, expected_sha256: str, *, label: str) -> None:
    import hashlib

    path = root / relative_path
    try:
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise DfpPhase1aIdentityError(f"cannot read {label} {path}: {error}") from error
    if observed != expected_sha256:
        raise DfpPhase1aIdentityError(f"{label} {path} SHA256 mismatch: identity declares {expected_sha256}, observed {observed}")


def validate_bridge_checkpoint_binding(root: Path, identity: Mapping[str, Any]) -> str:
    block = identity["bridge_checkpoint"]
    _validate_file_binding(root, block["relative_path"], block["sha256"], label="bridge checkpoint")
    return block["sha256"]


def _validate_parent_file_bindings(root: Path, identity: Mapping[str, Any]) -> None:
    parents = identity["parent_identities"]
    for prefix in ("matched_k11_k12", "e3", "rwr", "ade20k_source", "ade20k_evaluator"):
        _validate_file_binding(
            root, parents[f"{prefix}_identity_path"], parents[f"{prefix}_identity_sha256"],
            label=f"parent identity ({prefix})",
        )


def _validate_matched_k11_k12_relational(root: Path, identity: Mapping[str, Any]) -> None:
    """Cross-check the frozen graph/propagation/geometry/dataset values in
    this identity against the LIVE values in e12_matched_k11_k12_t320.toml
    -- never merely hash-pinning the parent file while ignoring its
    contents."""
    from src.matched_k11_k12_identity import load_identity as load_matched_identity

    parents = identity["parent_identities"]
    matched = load_matched_identity(root / parents["matched_k11_k12_identity_path"], repo_root=root)

    graph = identity["graph"]
    if matched["graph"]["maximum_rank"] != graph["maximum_rank"]:
        raise DfpPhase1aIdentityError("graph.maximum_rank disagrees with the live matched_k11_k12 parent identity")
    if matched["graph"]["affinity_power"] != graph["affinity_power"]:
        raise DfpPhase1aIdentityError("graph.affinity_power disagrees with the live matched_k11_k12 parent identity")

    propagation = identity["propagation"]
    if matched["propagation"]["alpha"] != propagation["alpha"]:
        raise DfpPhase1aIdentityError("propagation.alpha disagrees with the live matched_k11_k12 parent identity")
    if matched["propagation"]["steps"] != propagation["steps"]:
        raise DfpPhase1aIdentityError("propagation.steps disagrees with the live matched_k11_k12 parent identity")

    geometry = identity["score_interpolation_stitching"]
    if tuple(matched["geometry"]["crop"]) != tuple(geometry["crop"]):
        raise DfpPhase1aIdentityError("score_interpolation_stitching.crop disagrees with the live matched_k11_k12 parent identity")
    if tuple(matched["geometry"]["stride"]) != tuple(geometry["stride"]):
        raise DfpPhase1aIdentityError("score_interpolation_stitching.stride disagrees with the live matched_k11_k12 parent identity")
    if matched["geometry"]["align_corners"] != geometry["align_corners"]:
        raise DfpPhase1aIdentityError("score_interpolation_stitching.align_corners disagrees with the live matched_k11_k12 parent identity")
    if matched["snapshot"]["raw_score_stage"] != geometry["raw_score_stage"]:
        raise DfpPhase1aIdentityError("score_interpolation_stitching.raw_score_stage disagrees with the live matched_k11_k12 parent identity")

    coco = identity["coco_stuff_dataset"]
    if matched["dataset"]["images"] != coco["images"]:
        raise DfpPhase1aIdentityError("coco_stuff_dataset.images disagrees with the live matched_k11_k12 parent identity")
    if matched["dataset"]["classes"] != coco["class_count"]:
        raise DfpPhase1aIdentityError("coco_stuff_dataset.class_count disagrees with the live matched_k11_k12 parent identity")
    if matched["dataset"]["background_class"] != coco["background_class_evaluated"]:
        raise DfpPhase1aIdentityError("coco_stuff_dataset.background_class_evaluated disagrees with the live matched_k11_k12 parent identity")


def _validate_rwr_checkpoint_relational(root: Path, identity: Mapping[str, Any]) -> None:
    """Cross-check the bridge checkpoint SHA-256 against the live
    e3_canonical_directed_rwr.toml [checkpoint].sha256 field -- the
    established authority this identity is bound_via."""
    with (root / identity["parent_identities"]["rwr_identity_path"]).open("rb") as handle:
        rwr_raw = tomllib.load(handle)
    live_checkpoint_sha256 = rwr_raw["checkpoint"]["sha256"]
    if live_checkpoint_sha256 != identity["bridge_checkpoint"]["sha256"]:
        raise DfpPhase1aIdentityError(
            "bridge_checkpoint.sha256 disagrees with the live e3-canonical-directed-rwr [checkpoint].sha256"
        )


def _validate_ade20k_relational(root: Path, identity: Mapping[str, Any]) -> None:
    """Cross-check the ADE20K dataset/checkpoint facts frozen here against
    the live ade20k source and evaluator identities."""
    from src.ade20k_dataset_identity import load_identity as load_ade_source_identity
    from src.ade20k_matched_evaluator_identity import load_identity as load_ade_evaluator_identity

    parents = identity["parent_identities"]
    ade_source = load_ade_source_identity(root / parents["ade20k_source_identity_path"], repo_root=root)
    ade_evaluator = load_ade_evaluator_identity(root / parents["ade20k_evaluator_identity_path"], repo_root=root)

    ade = identity["ade20k_dataset"]
    if ade_source["protocol"]["expected_image_count"] != ade["images"]:
        raise DfpPhase1aIdentityError("ade20k_dataset.images disagrees with the live ade20k source identity")
    if ade_source["class_contract"]["class_count"] != ade["class_count"]:
        raise DfpPhase1aIdentityError("ade20k_dataset.class_count disagrees with the live ade20k source identity")
    if ade_source["class_contract"]["class_names_digest"] != ade["class_names_digest"]:
        raise DfpPhase1aIdentityError("ade20k_dataset.class_names_digest disagrees with the live ade20k source identity")
    if ade_source["label_policy"]["reduce_zero_label"] != ade["reduce_zero_label"]:
        raise DfpPhase1aIdentityError("ade20k_dataset.reduce_zero_label disagrees with the live ade20k source identity")

    if ade_evaluator["e3_config"]["projection_checkpoint_sha256"] != identity["bridge_checkpoint"]["sha256"]:
        raise DfpPhase1aIdentityError("bridge_checkpoint.sha256 disagrees with the live ade20k evaluator identity's projection_checkpoint_sha256")


def validate_static_configuration(*, repo_root: Path, identity_path: Path | None = None, check_git: bool = True) -> dict[str, Any]:
    root = Path(repo_root)
    identity = load_identity(identity_path, repo_root=root)
    if check_git:
        _check_git_ancestry(root, identity["identity"]["required_ancestor_commit"], label="identity.required_ancestor_commit")

    _validate_parent_file_bindings(root, identity)
    validate_bridge_checkpoint_binding(root, identity)
    _validate_matched_k11_k12_relational(root, identity)
    _validate_rwr_checkpoint_relational(root, identity)
    _validate_ade20k_relational(root, identity)

    return {
        "identity_name": identity["identity"]["name"],
        "required_ancestor_commit": identity["identity"]["required_ancestor_commit"],
        "coco_stuff_images": identity["coco_stuff_dataset"]["images"],
        "ade20k_images": identity["ade20k_dataset"]["images"],
    }


__all__ = [
    "IDENTITY_RELATIVE_PATH",
    "REQUIRED_ANCESTOR_COMMIT",
    "WORKING_BRANCH",
    "DfpPhase1aIdentityError",
    "load_identity",
    "repository_root",
    "validate_bridge_checkpoint_binding",
    "validate_static_configuration",
    "validate_graph_block",
    "validate_propagation_block",
    "validate_score_interpolation_stitching_block",
]
