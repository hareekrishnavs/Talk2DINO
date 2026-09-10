"""Strict loading/validation for the DFP Phase 1B conservation-TOLERANCE
successor identity
(`evaluation_identities/e19_dfp_phase1b_conservation_tolerance.toml`).

This module is a NEW LEAF authority (E19). It is an ADDITIVE SUCCESSOR
to E16: it NEVER rewrites, re-binds, or cascades new hashes through
E13-E18, the verified calibration artifact, the exact masked-forward
provenance core, the committed E15 Stage-A/B artifact schema, the
frozen COCO-Stuff-500 subset manifest, or any recurrence / graph /
stitching / pullback / component code. It only relationally
re-verifies their SHA-256 against current file bytes on every load.

It freezes exactly ONE scalar: the numeric conservation tolerance that
the already-committed `src/dfp_phase1b_artifact_schema.py
::validate_stage_a_flat_arrays(*, conservation_tolerance: float)`
already expects to receive from `phase1b_conservation_calibration.json`
(see `e15-dfp-phase1b-artifacts`
 .stage_a_schema...conservation_tolerance_source). The governed
residual family is `rho_g` -- `e16-dfp-phase1b-conservation-calibration`
 .[residual_definitions].primary_measurement -- the residual
`|sum_component_evidence - margin_change|` that committed schema
compares. `rho_M` / `rho_0` / `rho_320` are recorded calibration
diagnostics; no committed authority binds a tolerance to them.

Deterministic rule: `tolerance = smallest exact power of two >= M`,
where `M` is the maximum absolute governed (`rho_g`) calibration
residual, reconstructed on every load from the 200 raw calibration
rows' per-window `maximum_absolute_residual` values -- never trusted
from the artifact's aggregate, and never trusted from the value
recorded in E19. `M == 0` -> `tolerance = 0`. `M` an exact power of
two -> `tolerance = M`. The acceptance relation is inclusive:
`absolute_residual <= tolerance`.

Always fail closed: never silently substitute a default, coerce a
type, or normalize a string. Performs no torch / CUDA / model /
dataset / image access -- pure TOML/JSON/filesystem/hash validation
and exact IEEE-754 arithmetic only. `KeyboardInterrupt`, `SystemExit`,
and `MemoryError` always propagate unmodified; ordinary parse / type /
path failures are wrapped in `DfpPhase1bConservationToleranceError`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from pathlib import Path
from typing import Any, Mapping

from src.dfp_phase1b_calibration_artifact import (
    RESIDUAL_FAMILIES,
    validate_calibration_artifact,
)
from src.dfp_phase1b_calibration_identity import load_identity as load_e16_identity
from src.dfp_phase1b_chunk_benchmark_identity import (
    DfpPhase1bChunkBenchmarkIdentityError,
    resolve_repository_relative_path,
)


class DfpPhase1bConservationToleranceError(ValueError):
    """Raised on any DFP Phase 1B conservation-tolerance identity
    invariant violation. Always fail closed. Never raised for, and
    never catches, KeyboardInterrupt / SystemExit / MemoryError --
    those always propagate unmodified."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e19_dfp_phase1b_conservation_tolerance.toml")
FORMAT_VERSION = "Talk2Dino-DFP-Phase1B-Conservation-Tolerance-Identity-V1"
IDENTITY_SCHEMA_NAME = "talk2dino-dfp-phase1b-conservation-tolerance-identity-v1"
IDENTITY_SCHEMA_VERSION = 1
GOVERNED_FAMILY = "rho_g"
CANONICAL_FAMILY_ORDER = ("rho_M", "rho_0", "rho_320", "rho_g")
CALIBRATION_ROW_COUNT = 200
COMPARISON_OPERATOR = "<="

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


# ---------------------------------------------------------------------
# Exact power-of-two ceiling. No floating-point boundary ambiguity:
# frexp splits x = m * 2**e with 0.5 <= m < 1 for x > 0, so x <= 2**e
# always, and x == 2**(e-1) exactly iff m == 0.5.
# ---------------------------------------------------------------------

def pow2_ceil(m_value: float) -> float:
    """Smallest exact power of two >= ``m_value`` for a finite
    ``m_value >= 0`` float. ``0.0 -> 0.0``. An exact power of two
    returns itself (no doubling). Handles subnormals exactly.

    Raises ``DfpPhase1bConservationToleranceError`` for a non-float,
    NaN, infinity, or negative input, and for any finite non-negative
    input whose power-of-two ceiling is not representable as a finite
    binary64 value (i.e. would require an exponent above ``2**1023``).
    The result is never clamped, rounded, or returned as infinity."""
    if type(m_value) is not float:
        raise DfpPhase1bConservationToleranceError(
            f"pow2_ceil requires an exact float, observed {type(m_value).__name__}"
        )
    if math.isnan(m_value) or math.isinf(m_value):
        raise DfpPhase1bConservationToleranceError("pow2_ceil requires a finite value (no NaN/Infinity)")
    if m_value < 0.0:
        raise DfpPhase1bConservationToleranceError("pow2_ceil requires a non-negative value")
    if m_value == 0.0:
        return 0.0
    mantissa, exponent = math.frexp(m_value)  # m_value = mantissa * 2**exponent, 0.5 <= mantissa < 1
    # 2**target_exponent is the smallest exact power of two >= m_value:
    # mantissa == 0.5 means m_value is exactly 2**(exponent - 1); otherwise
    # 2**(exponent - 1) < m_value < 2**exponent.
    target_exponent = exponent - 1 if mantissa == 0.5 else exponent
    # 2**1023 is the largest power of two representable as a finite binary64;
    # 2**1024 overflows. Refuse rather than clamp / round / return infinity.
    if target_exponent > 1023:
        raise DfpPhase1bConservationToleranceError(
            "pow2_ceil: the power-of-two ceiling of the input is not representable "
            f"as a finite binary64 value (would require 2**{target_exponent})"
        )
    try:
        return math.ldexp(1.0, target_exponent)
    except OverflowError as exc:  # defensive normalization; the guard above already covers this
        raise DfpPhase1bConservationToleranceError(
            "pow2_ceil: the power-of-two ceiling of the input is not representable "
            "as a finite binary64 value"
        ) from exc


def is_power_of_two(value: float) -> bool:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        return False
    return math.frexp(value)[0] == 0.5


# ---------------------------------------------------------------------
# Closed schema.
# ---------------------------------------------------------------------

TOP_KEYS: frozenset[str] = frozenset({
    "format_version", "identity", "operational_scope", "dependency_bindings",
    "calibration_source_binding", "governed_residual", "derivation_inputs",
    "rule", "tolerance", "authorization_boundary", "disclosure", "prohibited",
    "future_step_bindings",
})

_IDENTITY_KEYS = frozenset({
    "name", "schema_name", "schema_version", "status", "purpose", "successor_of",
    "successor_relation", "does_not_modify_e16", "required_ancestor_commit",
    "required_ancestor_relation", "working_branch",
})
_OPERATIONAL_SCOPE_KEYS = frozenset({
    "purpose", "scientific_accuracy_metric", "method_selection_criterion",
    "preregistered_hypothesis_test", "post_calibration", "pre_evaluation",
    "calibration_data_only", "numeric_tolerance_defined",
    "tolerance_derived_from_5000_image_evaluation", "real_inference_executed_by_task6a",
    "gpu_execution_performed_by_task6a", "full_phase1b_execution_authorized",
    "phase1b_500_image_diagnostic_authorized", "phase1b_5000_image_evaluation_authorized",
    "lightweight_method_selected",
})
_DEP_REPO_NAMES = (
    "e16_calibration_identity", "e15_protocol", "e15_artifacts",
    "calibration_identity_module", "calibration_artifact_validator_module",
    "calibration_cli_module", "masked_forward_core", "phase1b_artifact_schema_module",
    "tolerance_module", "tolerance_cli_module",
)
_DEP_BINDING_KEYS = frozenset({"relative_path", "sha256", "untouched_by_this_identity"})
_DEP_BINDING_KEYS_WITH_SELFREF = frozenset({"relative_path", "sha256", "untouched_by_this_identity", "self_reference_note"})
_CALIBRATION_ARTIFACT_BINDING_KEYS = frozenset({
    "logical_identifier", "relative_path_is_symbolic_not_repository_relative",
    "relative_path_note", "sha256", "untouched_by_this_identity",
})
_CALIBRATION_SOURCE_BINDING_KEYS = frozenset({
    "calibration_artifact_logical_identifier", "calibration_artifact_sha256",
    "e16_identity_relative_path", "e16_identity_sha256", "e16_artifact_schema_name",
    "e16_artifact_schema_version", "window_selection_namespace", "selected_windows_digest",
    "selected_ids_digest", "calibration_row_count", "calibration_run_mode",
    "calibration_device_type", "calibration_gpu_model_name",
    "propagation_precision_masked_recurrence_dtype",
    "propagation_precision_residual_accumulation_dtype", "recurrence_k",
    "recurrence_step_count", "recurrence_alpha", "recorded_component_chunk_size",
    "source_calibration_repository_commit",
})
_GOVERNED_RESIDUAL_KEYS = frozenset({
    "tolerance_structure", "tolerance_structure_source", "governed_family",
    "governed_family_source", "governed_family_formula", "governed_family_units",
    "governed_family_domain", "recorded_but_ungoverned_families",
    "recorded_but_ungoverned_note",
})
_DERIVATION_INPUTS_KEYS = frozenset({
    "residual_family_names_in_canonical_order", "reconstruction_source",
    "row_count_used", "rho_M", "rho_0", "rho_320", "rho_g", "governed_maximum",
})
_DERIVATION_FAMILY_KEYS = frozenset({
    "sample_count", "non_finite_count", "reconstructed_maximum_absolute_residual", "governed",
})
_GOVERNED_MAXIMUM_KEYS = frozenset({
    "family", "observed_maximum_absolute_residual", "observed_maximum_is_exact_power_of_two",
    "selected_binary_exponent", "selected_binary_exponent_definition",
    "previous_power_of_two", "previous_power_of_two_strictly_below_observed_maximum",
})
_RULE_KEYS = frozenset({
    "name", "aggregation_source", "absolute_value_operation", "maximum_operation",
    "binary_power_of_two_ceiling", "exact_power_of_two_input_behavior",
    "zero_input_behavior", "comparison_operator", "comparison_semantics", "scope",
    "manually_selected_multiplier_used", "percentile_used", "confidence_interval_used",
    "fitted_distribution_used", "decimal_rounding_used", "manually_typed_constant_used",
    "evaluation_result_input_used",
})
_TOLERANCE_KEYS = frozenset({
    "conservation_tolerance", "conservation_tolerance_is_power_of_two",
    "conservation_tolerance_binary_exponent", "conservation_tolerance_equals_two_pow_exponent",
    "governed_family", "comparison_operator", "acceptance_relation",
    "observed_maximum_absolute_residual", "observed_maximum_le_tolerance",
    "tolerance_over_observed_maximum_ratio_lt_two", "source_calibration_artifact_sha256",
    "label_post_calibration", "label_pre_evaluation",
})
_AUTHORIZATION_BOUNDARY_KEYS = frozenset({
    "post_calibration", "pre_evaluation", "calibration_data_only", "numeric_tolerance_defined",
    "tolerance_derived_from_5000_image_evaluation", "real_inference_executed_by_task6a",
    "gpu_execution_performed_by_task6a", "full_phase1b_execution_authorized",
    "phase1b_500_image_diagnostic_authorized", "phase1b_5000_image_evaluation_authorized",
    "lightweight_method_selected",
})
_DISCLOSURE_KEYS = frozenset({
    "is_numeric_conservation_threshold_not_accuracy_metric",
    "tolerance_selected_after_examining_calibration_artifact",
    "tolerance_frozen_before_500_image_diagnostic_and_before_5000_image_evaluation",
    "tolerance_is_deterministic_function_of_calibration_observations_only",
    "tolerance_never_smaller_than_largest_observed_governed_calibration_residual",
    "tolerance_envelope_below_factor_of_two_unless_maximum_is_exact_power_of_two",
    "no_fitted_percentile_confidence_interval_or_manual_safety_factor",
    "no_evaluation_derived_information_used",
    "passing_conservation_does_not_imply_good_miou",
    "passing_conservation_does_not_imply_good_boundary_iou",
    "passing_conservation_does_not_imply_good_provenance_quality_or_deployability",
    "task6a_performs_no_inference",
    "full_phase1b_and_5000_image_execution_remain_unauthorized",
    "future_evaluation_results_must_never_revise_this_tolerance_without_a_separately_disclosed_successor_protocol",
})

# Mechanically-generated exact-value table: EVERY scalar/list leaf
# outside [dependency_bindings] (which is hash-verified separately) is
# pinned. Any drift to any field, anywhere, is rejected -- E19 is a
# frozen pre-registration document with no "variable data" field.
# Regenerate by flattening the committed TOML; never hand-type.
_CANONICAL: dict[str, Any] = {
    'authorization_boundary.calibration_data_only': True,
    'authorization_boundary.full_phase1b_execution_authorized': False,
    'authorization_boundary.gpu_execution_performed_by_task6a': False,
    'authorization_boundary.lightweight_method_selected': False,
    'authorization_boundary.numeric_tolerance_defined': True,
    'authorization_boundary.phase1b_5000_image_evaluation_authorized': False,
    'authorization_boundary.phase1b_500_image_diagnostic_authorized': False,
    'authorization_boundary.post_calibration': True,
    'authorization_boundary.pre_evaluation': True,
    'authorization_boundary.real_inference_executed_by_task6a': False,
    'authorization_boundary.tolerance_derived_from_5000_image_evaluation': False,
    'calibration_source_binding.calibration_artifact_logical_identifier': 'phase1b_conservation_calibration.json',
    'calibration_source_binding.calibration_artifact_sha256': 'f065b143933cf7881e5a31dd663783719a684e6d430e9e7b2bb314ed7e792d22',
    'calibration_source_binding.calibration_device_type': 'cuda',
    'calibration_source_binding.calibration_gpu_model_name': 'NVIDIA H100 80GB HBM3',
    'calibration_source_binding.calibration_row_count': 200,
    'calibration_source_binding.calibration_run_mode': 'coco_stuff_500',
    'calibration_source_binding.e16_artifact_schema_name': 'talk2dino-dfp-phase1b-conservation-calibration-v1',
    'calibration_source_binding.e16_artifact_schema_version': 1,
    'calibration_source_binding.e16_identity_relative_path': 'evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml',
    'calibration_source_binding.e16_identity_sha256': '01add2460ad032055db2bd73dcc4e3f40b1b7e3dede07d25afbea4f268ecb679',
    'calibration_source_binding.propagation_precision_masked_recurrence_dtype': 'float64',
    'calibration_source_binding.propagation_precision_residual_accumulation_dtype': 'float64',
    'calibration_source_binding.recorded_component_chunk_size': 16,
    'calibration_source_binding.recurrence_alpha': 0.98,
    'calibration_source_binding.recurrence_k': 12,
    'calibration_source_binding.recurrence_step_count': 320,
    'calibration_source_binding.selected_ids_digest': '38c61b8f830475944ae94e3dba88ea62905159a46f30cfd2a951bee67540e770',
    'calibration_source_binding.selected_windows_digest': 'f7b31944cd438a987ff2f7e8d0dfb4e01edb85be4b8af1c77fe05f3a69a53974',
    'calibration_source_binding.source_calibration_repository_commit': 'ef738016f7c2d0b7d37cd85cb675854c18a8135c',
    'calibration_source_binding.window_selection_namespace': 'talk2dino-dfp-phase1b-calibration-window-selection-v1',
    'derivation_inputs.governed_maximum.family': 'rho_g',
    'derivation_inputs.governed_maximum.observed_maximum_absolute_residual': 5.889683609150431e-07,
    'derivation_inputs.governed_maximum.observed_maximum_is_exact_power_of_two': False,
    'derivation_inputs.governed_maximum.previous_power_of_two': 4.76837158203125e-07,
    'derivation_inputs.governed_maximum.previous_power_of_two_strictly_below_observed_maximum': True,
    'derivation_inputs.governed_maximum.selected_binary_exponent': -20,
    'derivation_inputs.governed_maximum.selected_binary_exponent_definition': 'e = frexp(M)[1] when frexp(M)[0] != 0.5 (M not an exact power of two); the frozen tolerance is ldexp(1.0, e) = 2**e',
    'derivation_inputs.reconstruction_source': "max over the 200 selected_window_records of record[family].maximum_absolute_residual; independently re-max'd, never read from aggregate_residual_summaries and trusted",
    'derivation_inputs.residual_family_names_in_canonical_order': ['rho_M', 'rho_0', 'rho_320', 'rho_g'],
    'derivation_inputs.rho_0.governed': False,
    'derivation_inputs.rho_0.non_finite_count': 0,
    'derivation_inputs.rho_0.reconstructed_maximum_absolute_residual': 3.4416913763379853e-15,
    'derivation_inputs.rho_0.sample_count': 4221952,
    'derivation_inputs.rho_320.governed': False,
    'derivation_inputs.rho_320.non_finite_count': 0,
    'derivation_inputs.rho_320.reconstructed_maximum_absolute_residual': 1.3007687971189341e-06,
    'derivation_inputs.rho_320.sample_count': 4221952,
    'derivation_inputs.rho_M.governed': False,
    'derivation_inputs.rho_M.non_finite_count': 0,
    'derivation_inputs.rho_M.reconstructed_maximum_absolute_residual': 7.771561172376096e-15,
    'derivation_inputs.rho_M.sample_count': 204800,
    'derivation_inputs.rho_g.governed': True,
    'derivation_inputs.rho_g.non_finite_count': 0,
    'derivation_inputs.rho_g.reconstructed_maximum_absolute_residual': 5.889683609150431e-07,
    'derivation_inputs.rho_g.sample_count': 7887805,
    'derivation_inputs.row_count_used': 200,
    'disclosure.full_phase1b_and_5000_image_execution_remain_unauthorized': True,
    'disclosure.future_evaluation_results_must_never_revise_this_tolerance_without_a_separately_disclosed_successor_protocol': True,
    'disclosure.is_numeric_conservation_threshold_not_accuracy_metric': True,
    'disclosure.no_evaluation_derived_information_used': True,
    'disclosure.no_fitted_percentile_confidence_interval_or_manual_safety_factor': True,
    'disclosure.passing_conservation_does_not_imply_good_boundary_iou': True,
    'disclosure.passing_conservation_does_not_imply_good_miou': True,
    'disclosure.passing_conservation_does_not_imply_good_provenance_quality_or_deployability': True,
    'disclosure.task6a_performs_no_inference': True,
    'disclosure.tolerance_envelope_below_factor_of_two_unless_maximum_is_exact_power_of_two': True,
    'disclosure.tolerance_frozen_before_500_image_diagnostic_and_before_5000_image_evaluation': True,
    'disclosure.tolerance_is_deterministic_function_of_calibration_observations_only': True,
    'disclosure.tolerance_never_smaller_than_largest_observed_governed_calibration_residual': True,
    'disclosure.tolerance_selected_after_examining_calibration_artifact': True,
    'format_version': 'Talk2Dino-DFP-Phase1B-Conservation-Tolerance-Identity-V1',
    'future_step_bindings.description': 'Fields explicitly deferred to a later task, never guessed or invented here.',
    'future_step_bindings.next_step_items': ['independently re-verify this E19 identity and its tolerance module against the committed HEAD', 'the 500-image Phase 1B provenance diagnostic (a separate, later, separately-authorized step -- never this identity)', 'the Phase 1B execution driver, authorized only after this identity is committed and independently verified', 'the 5,000-image Phase 1B evaluation (never this identity)', 'a separately disclosed successor protocol if a future evaluation result ever motivates revisiting this tolerance'],
    'governed_residual.governed_family': 'rho_g',
    'governed_residual.governed_family_domain': "final-resolution disputed pixels of the selected window's own final-resolution support where a(x) != b(x)",
    'governed_residual.governed_family_formula': 'rho_g^(w)(x) = sum_r g^(w)_{x,r} - delta_m^(w)(x)  (signed component-contribution sum minus the canonical T320-minus-T0 b(x)-a(x) margin change, after the identical final-resolution operator A_final,w)',
    'governed_residual.governed_family_source': "e16-dfp-phase1b-conservation-calibration.[residual_definitions].primary_measurement = 'rho_g'; the only calibration family any committed conservation-acceptance relation references",
    'governed_residual.governed_family_units': 'final-resolution class-score margin units (same units as margin_change = m320 - m0); real-valued, dimensionless model-score scale; no physical unit',
    'governed_residual.recorded_but_ungoverned_families': ['rho_M', 'rho_0', 'rho_320'],
    'governed_residual.recorded_but_ungoverned_note': 'rho_M (membership partition), rho_0 (seed reconstruction), rho_320 (propagated reconstruction) are recorded per E16 so the calibration run itself is independently auditable; no committed authority applies a numeric tolerance to them and this identity freezes none',
    'governed_residual.tolerance_structure': 'single_scalar',
    'governed_residual.tolerance_structure_source': "e15-dfp-phase1b-artifacts stage_a_schema (conservation_residual = float64; single 'the separately bound calibration tolerance') and its committed mirror src/dfp_phase1b_artifact_schema.py::validate_stage_a_flat_arrays(*, conservation_tolerance: float), which compares exactly |sum_component_evidence - margin_change| (the rho_g residual) against one scalar",
    'identity.does_not_modify_e16': True,
    'identity.name': 'e19-dfp-phase1b-conservation-tolerance',
    'identity.purpose': 'Binds the completed and independently-verified 200-window phase1b_conservation_calibration.json by content hash and freezes the single numeric conservation tolerance for DFP Phase 1B, derived deterministically as the smallest exact power of two greater than or equal to the reconstructed maximum absolute governed (rho_g) calibration residual. Implements no runner, runs no inference, defines no accuracy metric, and authorizes no 500-image diagnostic, full Phase 1B, or 5,000-image execution.',
    'identity.required_ancestor_commit': 'ae1c7d2daed2a8d77fb2f1fbb013501d5c722a28',
    'identity.required_ancestor_relation': "required_ancestor_commit must be an ancestor of (or equal to) the executing repository_commit -- verified via 'git merge-base --is-ancestor <required_ancestor_commit> <repository_commit>', exactly like E17/E18's own convention. Every repository-file SHA-256 bound below is valid only for a worktree at or descended from this commit.",
    'identity.schema_name': 'talk2dino-dfp-phase1b-conservation-tolerance-identity-v1',
    'identity.schema_version': 1,
    'identity.status': 'frozen_additive_successor_authority',
    'identity.successor_of': 'e16-dfp-phase1b-conservation-calibration',
    'identity.successor_relation': 'additive_only: binds but never edits, reinterprets, or re-hashes any E13-E18 file or the verified calibration artifact; consumes the calibration artifact as a read-only input and freezes exactly the one scalar tolerance that e15-dfp-phase1b-artifacts already sources from it',
    'identity.working_branch': 'dfp-phase1a-diagnostics',
    'operational_scope.calibration_data_only': True,
    'operational_scope.full_phase1b_execution_authorized': False,
    'operational_scope.gpu_execution_performed_by_task6a': False,
    'operational_scope.lightweight_method_selected': False,
    'operational_scope.method_selection_criterion': False,
    'operational_scope.numeric_tolerance_defined': True,
    'operational_scope.phase1b_5000_image_evaluation_authorized': False,
    'operational_scope.phase1b_500_image_diagnostic_authorized': False,
    'operational_scope.post_calibration': True,
    'operational_scope.pre_evaluation': True,
    'operational_scope.preregistered_hypothesis_test': False,
    'operational_scope.purpose': 'post_calibration_numeric_conservation_tolerance_freeze',
    'operational_scope.real_inference_executed_by_task6a': False,
    'operational_scope.scientific_accuracy_metric': False,
    'operational_scope.tolerance_derived_from_5000_image_evaluation': False,
    'prohibited.list': ['modifying_e13_e14_e15_e16_e17_or_e18_identities', 'modifying_or_regenerating_the_verified_calibration_artifact', 'modifying_masked_forward_recurrence_graph_stitching_pullback_or_component_code', 'modifying_task3_benchmark_results_or_task4_selection_artifacts', 'modifying_any_phase1a_result_or_dataset_manifest', 'deriving_the_tolerance_from_a_percentile_fitted_distribution_or_manual_multiplier', 'deriving_the_tolerance_from_any_data_outside_the_verified_200_window_artifact', 'deriving_or_revising_the_tolerance_from_500_image_or_5000_image_evaluation_results', 'decimal_rounding_or_a_manually_typed_tolerance_constant', 'trusting_the_recorded_tolerance_without_recomputation_from_the_200_rows', 'real_inference_model_or_dataset_access_in_this_task', 'cuda_initialization_in_this_task', 'slurm_submission_in_this_task', 'authorizing_the_500_image_phase1b_diagnostic', 'authorizing_full_phase1b_execution_or_the_5000_image_evaluation', 'selecting_a_lightweight_method'],
    'rule.absolute_value_operation': 'abs',
    'rule.aggregation_source': 'verified_200_window_calibration_artifact_row_level_maxima_only',
    'rule.binary_power_of_two_ceiling': 'tolerance = 2 ** ceil(log2(M)) for finite M > 0, computed exactly via math.frexp / math.ldexp with no floating-point boundary ambiguity',
    'rule.comparison_operator': '<=',
    'rule.comparison_semantics': 'conservation is accepted for a pixel when absolute_residual <= tolerance; inclusive; strictly identical to src/dfp_phase1b_artifact_schema.py which fails only when residual > tolerance',
    'rule.confidence_interval_used': False,
    'rule.decimal_rounding_used': False,
    'rule.evaluation_result_input_used': False,
    'rule.exact_power_of_two_input_behavior': 'if M is already an exact power of two, tolerance = M (no doubling)',
    'rule.fitted_distribution_used': False,
    'rule.manually_selected_multiplier_used': False,
    'rule.manually_typed_constant_used': False,
    'rule.maximum_operation': 'max_over_all_200_rows_of_per_window_maximum_absolute_residual_for_the_governed_family',
    'rule.name': 'smallest_exact_power_of_two_ge_observed_maximum_absolute_governed_residual',
    'rule.percentile_used': False,
    'rule.scope': 'single scalar governing the rho_g conservation residual only',
    'rule.zero_input_behavior': 'if M == 0, tolerance = 0',
    'tolerance.acceptance_relation': 'absolute_residual <= conservation_tolerance',
    'tolerance.comparison_operator': '<=',
    'tolerance.conservation_tolerance': 9.5367431640625e-07,
    'tolerance.conservation_tolerance_binary_exponent': -20,
    'tolerance.conservation_tolerance_equals_two_pow_exponent': True,
    'tolerance.conservation_tolerance_is_power_of_two': True,
    'tolerance.governed_family': 'rho_g',
    'tolerance.label_post_calibration': 'POST-CALIBRATION',
    'tolerance.label_pre_evaluation': 'PRE-EVALUATION',
    'tolerance.observed_maximum_absolute_residual': 5.889683609150431e-07,
    'tolerance.observed_maximum_le_tolerance': True,
    'tolerance.source_calibration_artifact_sha256': 'f065b143933cf7881e5a31dd663783719a684e6d430e9e7b2bb314ed7e792d22',
    'tolerance.tolerance_over_observed_maximum_ratio_lt_two': True,
}


def _walk_leaves(node: Any, prefix: str = "") -> "list[tuple[str, Any]]":
    out: list[tuple[str, Any]] = []
    if isinstance(node, Mapping):
        for k, v in node.items():
            out.extend(_walk_leaves(v, f"{prefix}.{k}" if prefix else k))
    else:
        out.append((prefix, node))
    return out


# ---------------------------------------------------------------------
# Primitive checks (fail closed; reject bool-as-int, NaN/inf, blanks).
# ---------------------------------------------------------------------

def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1bConservationToleranceError(f"{label} must be an exact bool, observed {type(value).__name__}")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:  # rejects bool (bool is not int here) and float
        raise DfpPhase1bConservationToleranceError(f"{label} must be an exact int, observed {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise DfpPhase1bConservationToleranceError(f"{label} must be >= {minimum}, observed {value}")
    return value


def _require_finite_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    if type(value) is not float:
        raise DfpPhase1bConservationToleranceError(f"{label} must be an exact float, observed {type(value).__name__}")
    if math.isnan(value) or math.isinf(value):
        raise DfpPhase1bConservationToleranceError(f"{label} must be finite (no NaN/Infinity), observed {value!r}")
    if minimum is not None and value < minimum:
        raise DfpPhase1bConservationToleranceError(f"{label} must be >= {minimum}, observed {value!r}")
    return value


def _require_clean_string(value: Any, label: str) -> str:
    if type(value) is not str:
        raise DfpPhase1bConservationToleranceError(f"{label} must be a string, observed {type(value).__name__}")
    if value == "":
        raise DfpPhase1bConservationToleranceError(f"{label} must not be empty")
    if value.strip() == "":
        raise DfpPhase1bConservationToleranceError(f"{label} must not be whitespace-only")
    if value != value.strip():
        raise DfpPhase1bConservationToleranceError(f"{label} must not have leading/trailing whitespace")
    if "\x00" in value:
        raise DfpPhase1bConservationToleranceError(f"{label} must not contain a NUL byte")
    return value


def _require_sha256(value: Any, label: str) -> str:
    _require_clean_string(value, label)
    if not _HEX64.match(value):
        raise DfpPhase1bConservationToleranceError(f"{label} must be lowercase 64-hex, observed {value!r}")
    return value


def _require_git_commit(value: Any, label: str) -> str:
    _require_clean_string(value, label)
    if not _HEX40.match(value):
        raise DfpPhase1bConservationToleranceError(f"{label} must be a full lowercase 40-hex commit id, observed {value!r}")
    return value


def _require_closed(node: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        raise DfpPhase1bConservationToleranceError(f"{label} must be a table")
    actual = frozenset(node.keys())
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        raise DfpPhase1bConservationToleranceError(
            f"{label} has the wrong key set -- missing={missing} unknown={unknown}"
        )
    return node


def _get(identity: Mapping[str, Any], dotted: str) -> Any:
    node: Any = identity
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise DfpPhase1bConservationToleranceError(f"missing required field {dotted!r}")
        node = node[part]
    return node


# ---------------------------------------------------------------------
# load_identity: closed schema + exact values + dependency hash verify.
# This is dependency-FORMAT validation only; the relational
# recomputation against the calibration artifact is a separate step.
# ---------------------------------------------------------------------

def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load, closed-schema-validate, exact-value-validate, and
    relationally hash-verify the E19 conservation-tolerance identity.
    Never mutates or re-derives any bound dependency -- only re-verifies
    SHA-256 against current bytes. Performs no torch/CUDA/model/dataset
    access. Does NOT itself read the calibration artifact -- call
    ``verify_calibration_binding`` for the full relational check."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DfpPhase1bConservationToleranceError(f"cannot load E19 tolerance identity {source}: {error}") from error

    _require_closed(identity, TOP_KEYS, "E19 identity")
    _require_closed(identity["identity"], _IDENTITY_KEYS, "identity")
    _require_closed(identity["operational_scope"], _OPERATIONAL_SCOPE_KEYS, "operational_scope")
    _require_closed(identity["calibration_source_binding"], _CALIBRATION_SOURCE_BINDING_KEYS, "calibration_source_binding")
    _require_closed(identity["governed_residual"], _GOVERNED_RESIDUAL_KEYS, "governed_residual")
    _require_closed(identity["derivation_inputs"], _DERIVATION_INPUTS_KEYS, "derivation_inputs")
    for fam in CANONICAL_FAMILY_ORDER:
        _require_closed(identity["derivation_inputs"][fam], _DERIVATION_FAMILY_KEYS, f"derivation_inputs.{fam}")
    _require_closed(identity["derivation_inputs"]["governed_maximum"], _GOVERNED_MAXIMUM_KEYS, "derivation_inputs.governed_maximum")
    _require_closed(identity["rule"], _RULE_KEYS, "rule")
    _require_closed(identity["tolerance"], _TOLERANCE_KEYS, "tolerance")
    _require_closed(identity["authorization_boundary"], _AUTHORIZATION_BOUNDARY_KEYS, "authorization_boundary")
    _require_closed(identity["disclosure"], _DISCLOSURE_KEYS, "disclosure")
    _require_closed(identity["prohibited"], frozenset({"list"}), "prohibited")
    _require_closed(identity["future_step_bindings"], frozenset({"description", "next_step_items"}), "future_step_bindings")

    # dependency_bindings: closed to the exact set of bindings
    dep = identity["dependency_bindings"]
    expected_dep = frozenset(_DEP_REPO_NAMES) | frozenset({"calibration_artifact"})
    if frozenset(dep.keys()) != expected_dep:
        raise DfpPhase1bConservationToleranceError(
            f"dependency_bindings key set wrong -- missing={sorted(expected_dep - set(dep))} "
            f"unknown={sorted(set(dep) - expected_dep)}"
        )

    # EVERY leaf outside [dependency_bindings] must match _CANONICAL
    # exactly, by type and value (bool != int, float compared exactly).
    observed_leaves = {}
    for dotted, value in _walk_leaves(identity):
        if dotted.startswith("dependency_bindings."):
            continue
        observed_leaves[dotted] = value
    canon_keys = frozenset(_CANONICAL)
    obs_keys = frozenset(observed_leaves)
    if obs_keys != canon_keys:
        raise DfpPhase1bConservationToleranceError(
            f"E19 leaf set drift -- missing={sorted(canon_keys - obs_keys)} unknown={sorted(obs_keys - canon_keys)}"
        )
    for dotted, expected in _CANONICAL.items():
        observed = observed_leaves[dotted]
        if type(observed) is not type(expected) or observed != expected:
            raise DfpPhase1bConservationToleranceError(
                f"{dotted}: frozen value drift -- expected {expected!r} ({type(expected).__name__}), "
                f"observed {observed!r} ({type(observed).__name__})"
            )

    # every disclosure / authorization_boundary flag must be an exact bool
    for k, v in identity["disclosure"].items():
        _require_exact_bool(v, f"disclosure.{k}")
        if v is not True:
            raise DfpPhase1bConservationToleranceError(f"disclosure.{k} must be the exact bool True")
    for k, v in identity["authorization_boundary"].items():
        _require_exact_bool(v, f"authorization_boundary.{k}")

    _require_git_commit(identity["identity"]["required_ancestor_commit"], "identity.required_ancestor_commit")

    # SHA-256-shaped fields
    _require_sha256(identity["calibration_source_binding"]["calibration_artifact_sha256"], "calibration_source_binding.calibration_artifact_sha256")
    _require_sha256(identity["calibration_source_binding"]["e16_identity_sha256"], "calibration_source_binding.e16_identity_sha256")
    _require_sha256(identity["calibration_source_binding"]["selected_windows_digest"], "calibration_source_binding.selected_windows_digest")
    _require_sha256(identity["calibration_source_binding"]["selected_ids_digest"], "calibration_source_binding.selected_ids_digest")
    _require_git_commit(identity["calibration_source_binding"]["source_calibration_repository_commit"], "calibration_source_binding.source_calibration_repository_commit")
    _require_sha256(identity["dependency_bindings"]["calibration_artifact"]["sha256"], "dependency_bindings.calibration_artifact.sha256")
    _require_sha256(identity["tolerance"]["source_calibration_artifact_sha256"], "tolerance.source_calibration_artifact_sha256")

    # numeric leaves
    tol = identity["tolerance"]
    _require_finite_float(tol["conservation_tolerance"], "tolerance.conservation_tolerance", minimum=0.0)
    _require_exact_int(tol["conservation_tolerance_binary_exponent"], "tolerance.conservation_tolerance_binary_exponent")
    _require_finite_float(tol["observed_maximum_absolute_residual"], "tolerance.observed_maximum_absolute_residual", minimum=0.0)
    _require_exact_bool(tol["conservation_tolerance_is_power_of_two"], "tolerance.conservation_tolerance_is_power_of_two")
    _require_exact_bool(tol["conservation_tolerance_equals_two_pow_exponent"], "tolerance.conservation_tolerance_equals_two_pow_exponent")
    _require_exact_bool(tol["observed_maximum_le_tolerance"], "tolerance.observed_maximum_le_tolerance")
    _require_exact_bool(tol["tolerance_over_observed_maximum_ratio_lt_two"], "tolerance.tolerance_over_observed_maximum_ratio_lt_two")

    gm = identity["derivation_inputs"]["governed_maximum"]
    _require_finite_float(gm["observed_maximum_absolute_residual"], "derivation_inputs.governed_maximum.observed_maximum_absolute_residual", minimum=0.0)
    _require_exact_int(gm["selected_binary_exponent"], "derivation_inputs.governed_maximum.selected_binary_exponent")
    _require_finite_float(gm["previous_power_of_two"], "derivation_inputs.governed_maximum.previous_power_of_two", minimum=0.0)
    _require_exact_bool(gm["observed_maximum_is_exact_power_of_two"], "derivation_inputs.governed_maximum.observed_maximum_is_exact_power_of_two")
    _require_exact_bool(gm["previous_power_of_two_strictly_below_observed_maximum"], "derivation_inputs.governed_maximum.previous_power_of_two_strictly_below_observed_maximum")

    for fam in CANONICAL_FAMILY_ORDER:
        node = identity["derivation_inputs"][fam]
        _require_exact_int(node["sample_count"], f"derivation_inputs.{fam}.sample_count", minimum=0)
        _require_exact_int(node["non_finite_count"], f"derivation_inputs.{fam}.non_finite_count", minimum=0)
        _require_finite_float(node["reconstructed_maximum_absolute_residual"], f"derivation_inputs.{fam}.reconstructed_maximum_absolute_residual", minimum=0.0)
        _require_exact_bool(node["governed"], f"derivation_inputs.{fam}.governed")
        if node["non_finite_count"] != 0:
            raise DfpPhase1bConservationToleranceError(f"derivation_inputs.{fam}.non_finite_count must be 0")
    if identity["derivation_inputs"]["rho_g"]["governed"] is not True:
        raise DfpPhase1bConservationToleranceError("derivation_inputs.rho_g.governed must be True")
    for fam in ("rho_M", "rho_0", "rho_320"):
        if identity["derivation_inputs"][fam]["governed"] is not False:
            raise DfpPhase1bConservationToleranceError(f"derivation_inputs.{fam}.governed must be False")

    _require_exact_int(identity["calibration_source_binding"]["calibration_row_count"], "calibration_source_binding.calibration_row_count")
    if identity["calibration_source_binding"]["calibration_row_count"] != CALIBRATION_ROW_COUNT:
        raise DfpPhase1bConservationToleranceError("calibration_source_binding.calibration_row_count must be exactly 200")
    if identity["calibration_source_binding"]["calibration_run_mode"] != "coco_stuff_500":
        raise DfpPhase1bConservationToleranceError("calibration_source_binding.calibration_run_mode must be 'coco_stuff_500'")

    # --- internal arithmetic self-consistency of the FROZEN values ---
    frozen_M = gm["observed_maximum_absolute_residual"]
    frozen_tol = tol["conservation_tolerance"]
    frozen_exp = tol["conservation_tolerance_binary_exponent"]
    recomputed_tol = pow2_ceil(frozen_M)
    if recomputed_tol != frozen_tol:
        raise DfpPhase1bConservationToleranceError(
            f"frozen tolerance {frozen_tol!r} != pow2_ceil(frozen M {frozen_M!r}) = {recomputed_tol!r}"
        )
    if frozen_M != identity["derivation_inputs"]["rho_g"]["reconstructed_maximum_absolute_residual"]:
        raise DfpPhase1bConservationToleranceError("governed_maximum M != derivation_inputs.rho_g reconstructed maximum")
    if frozen_M != tol["observed_maximum_absolute_residual"]:
        raise DfpPhase1bConservationToleranceError("tolerance.observed_maximum_absolute_residual != governed_maximum M")
    if frozen_tol != 0.0:
        if not is_power_of_two(frozen_tol):
            raise DfpPhase1bConservationToleranceError("frozen tolerance is not an exact power of two")
        if math.ldexp(1.0, frozen_exp) != frozen_tol:
            raise DfpPhase1bConservationToleranceError(f"2**{frozen_exp} != frozen tolerance {frozen_tol!r}")
        if not (frozen_M <= frozen_tol):
            raise DfpPhase1bConservationToleranceError("frozen M > frozen tolerance")
        exact_pow2_M = is_power_of_two(frozen_M)
        if gm["observed_maximum_is_exact_power_of_two"] != exact_pow2_M:
            raise DfpPhase1bConservationToleranceError("observed_maximum_is_exact_power_of_two flag disagrees with M")
        if not exact_pow2_M:
            prev = math.ldexp(1.0, math.frexp(frozen_M)[1] - 1)
            if prev != gm["previous_power_of_two"]:
                raise DfpPhase1bConservationToleranceError("frozen previous_power_of_two wrong")
            if not (prev < frozen_M):
                raise DfpPhase1bConservationToleranceError("previous power of two not strictly below M")
            if (frozen_tol / frozen_M) >= 2.0:
                raise DfpPhase1bConservationToleranceError("tolerance/M ratio must be < 2 when M is not an exact power of two")
    if tol["conservation_tolerance_is_power_of_two"] != (frozen_tol != 0.0 and is_power_of_two(frozen_tol)):
        raise DfpPhase1bConservationToleranceError("conservation_tolerance_is_power_of_two flag disagrees")

    # relational dependency hash verification (FORMAT stage)
    _verify_dependency_hashes(root, identity["dependency_bindings"])

    return identity


def _verify_dependency_hashes(root: Path, dependency_bindings: Mapping[str, Any]) -> None:
    for name in _DEP_REPO_NAMES:
        binding = dependency_bindings[name]
        if name in ("tolerance_module", "phase1b_artifact_schema_module"):
            _require_closed(binding, _DEP_BINDING_KEYS_WITH_SELFREF if name == "tolerance_module" else _DEP_BINDING_KEYS, f"dependency_bindings.{name}")
        else:
            _require_closed(binding, _DEP_BINDING_KEYS, f"dependency_bindings.{name}")
        expected = _require_sha256(binding["sha256"], f"dependency_bindings.{name}.sha256")
        try:
            target = resolve_repository_relative_path(root, binding["relative_path"], binding_name=name)
        except DfpPhase1bChunkBenchmarkIdentityError as exc:
            raise DfpPhase1bConservationToleranceError(
                f"dependency_bindings.{name}: unsafe or unresolvable relative_path -- {exc}"
            ) from exc
        try:
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as exc:
            raise DfpPhase1bConservationToleranceError(
                f"dependency_bindings.{name}: cannot read bound file {binding['relative_path']}: {exc}"
            ) from exc
        if actual != expected:
            raise DfpPhase1bConservationToleranceError(
                f"dependency_bindings.{name}: {binding['relative_path']} SHA-256 mismatch -- "
                f"pinned {expected}, observed {actual} (bound dependency drifted since E19 was frozen)"
            )
    _require_closed(dependency_bindings["calibration_artifact"], _CALIBRATION_ARTIFACT_BINDING_KEYS, "dependency_bindings.calibration_artifact")


# ---------------------------------------------------------------------
# Row-level reconstruction of every family maximum from the 200 rows.
# ---------------------------------------------------------------------

def reconstruct_family_maxima(calibration_document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Independently rebuild, for each residual family, the maximum
    absolute residual, total sample count, and total non-finite count
    from the per-window rows -- never from aggregate_residual_summaries.
    Also cross-checks each per-window maximum against
    max(|signed_minimum|, |signed_maximum|), and confirms the rebuilt
    values equal the artifact's own aggregate."""
    rows = calibration_document["selected_window_records"]
    if not isinstance(rows, list) or len(rows) != CALIBRATION_ROW_COUNT:
        raise DfpPhase1bConservationToleranceError(
            f"calibration artifact must have exactly {CALIBRATION_ROW_COUNT} selected_window_records, observed {len(rows) if isinstance(rows, list) else type(rows).__name__}"
        )
    aggregates = calibration_document["aggregate_residual_summaries"]
    out: dict[str, dict[str, Any]] = {}
    for fam in RESIDUAL_FAMILIES:
        row_maxabs: list[float] = []
        total_samples = 0
        total_nonfinite = 0
        for i, row in enumerate(rows):
            fam_summary = row[fam]
            m = fam_summary["maximum_absolute_residual"]
            sc = fam_summary["sample_count"]
            nf = fam_summary["non_finite_count"]
            smn = fam_summary["signed_minimum"]
            smx = fam_summary["signed_maximum"]
            if type(sc) is not int or type(nf) is not int:
                raise DfpPhase1bConservationToleranceError(f"row {i} {fam} sample_count/non_finite_count must be exact ints")
            if nf != 0:
                raise DfpPhase1bConservationToleranceError(f"row {i} {fam} non_finite_count must be 0 in a completed artifact")
            total_samples += sc
            total_nonfinite += nf
            if sc == 0:
                # zero-sample family: floating summary fields are JSON null
                if m is not None:
                    raise DfpPhase1bConservationToleranceError(f"row {i} {fam} has sample_count 0 but non-null maximum_absolute_residual")
                continue
            if type(m) is not float or not math.isfinite(m) or m < 0.0:
                raise DfpPhase1bConservationToleranceError(f"row {i} {fam} maximum_absolute_residual must be a finite non-negative float")
            if type(smn) is not float or type(smx) is not float or not math.isfinite(smn) or not math.isfinite(smx):
                raise DfpPhase1bConservationToleranceError(f"row {i} {fam} signed_minimum/signed_maximum must be finite floats")
            if m != max(abs(smn), abs(smx)):
                raise DfpPhase1bConservationToleranceError(
                    f"row {i} {fam} maximum_absolute_residual {m!r} != max(|signed_min|,|signed_max|) {max(abs(smn), abs(smx))!r}"
                )
            row_maxabs.append(m)
        m_f = max(row_maxabs) if row_maxabs else 0.0
        agg = aggregates[fam]
        if agg["maximum_absolute_residual"] != (m_f if row_maxabs else None) and not (not row_maxabs and agg["maximum_absolute_residual"] is None):
            raise DfpPhase1bConservationToleranceError(
                f"{fam}: row-reconstructed maximum {m_f!r} disagrees with aggregate {agg['maximum_absolute_residual']!r}"
            )
        if agg["sample_count"] != total_samples:
            raise DfpPhase1bConservationToleranceError(f"{fam}: row sample-count sum {total_samples} != aggregate {agg['sample_count']}")
        if agg["non_finite_count"] != total_nonfinite or total_nonfinite != 0:
            raise DfpPhase1bConservationToleranceError(f"{fam}: non-finite count nonzero or disagrees with aggregate")
        out[fam] = {
            "maximum_absolute_residual": m_f,
            "sample_count": total_samples,
            "non_finite_count": total_nonfinite,
        }
    return out


# ---------------------------------------------------------------------
# verify_calibration_binding: the full relational recomputation.
# ---------------------------------------------------------------------

def verify_calibration_binding(
    identity: Mapping[str, Any] | None = None,
    *,
    calibration_artifact_path: Path | str,
    repo_root: Path | None = None,
    candidate_universe: Any = None,
) -> dict[str, Any]:
    """Load the supplied calibration artifact, hash-verify it against
    E19's frozen ``calibration_artifact_sha256``, run the committed
    official E16 calibration-artifact validator, independently
    reconstruct every residual-family maximum from the 200 rows,
    recompute the tolerance from the governed (``rho_g``) maximum via
    the exact rule, and confirm EXACT agreement with the frozen E19
    tolerance / exponent / M. Also loads and validates E16 via its
    committed loader. Returns a summary dict on success."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = identity if identity is not None else load_identity(repo_root=root)

    # E16 official load (relational hash verification of E16's own tree)
    e16 = load_e16_identity(repo_root=root)
    e16_hash = hashlib.sha256((root / "evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml").read_bytes()).hexdigest()
    if e16_hash != identity["calibration_source_binding"]["e16_identity_sha256"]:
        raise DfpPhase1bConservationToleranceError("live E16 hash disagrees with E19's bound e16_identity_sha256")
    if e16["residual_definitions"]["primary_measurement"] != GOVERNED_FAMILY:
        raise DfpPhase1bConservationToleranceError(
            f"E16 primary_measurement is {e16['residual_definitions']['primary_measurement']!r}, not {GOVERNED_FAMILY!r} -- governed-family binding invalid"
        )

    artifact_path = Path(calibration_artifact_path)
    try:
        raw = artifact_path.read_bytes()
    except OSError as exc:
        raise DfpPhase1bConservationToleranceError(f"cannot read calibration artifact {artifact_path}: {exc}") from exc
    actual_hash = hashlib.sha256(raw).hexdigest()
    frozen_hash = identity["dependency_bindings"]["calibration_artifact"]["sha256"]
    if actual_hash != frozen_hash:
        raise DfpPhase1bConservationToleranceError(
            f"calibration artifact SHA-256 mismatch -- E19 pinned {frozen_hash}, observed {actual_hash}"
        )
    if actual_hash != identity["calibration_source_binding"]["calibration_artifact_sha256"]:
        raise DfpPhase1bConservationToleranceError("calibration_source_binding.calibration_artifact_sha256 disagrees with dependency binding")
    if actual_hash != identity["tolerance"]["source_calibration_artifact_sha256"]:
        raise DfpPhase1bConservationToleranceError("tolerance.source_calibration_artifact_sha256 disagrees with the bound artifact hash")

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DfpPhase1bConservationToleranceError(f"calibration artifact is not valid JSON: {exc}") from exc

    # official committed validator
    try:
        validate_calibration_artifact(document, candidate_universe=candidate_universe)
    except Exception as exc:  # noqa: BLE001 -- translate committed-validator error into our domain
        if isinstance(exc, (KeyboardInterrupt, SystemExit, MemoryError)):
            raise
        raise DfpPhase1bConservationToleranceError(
            f"official E16 calibration-artifact validator rejected the bound artifact: {exc}"
        ) from exc

    if not (document.get("complete") is True and document.get("final") is True):
        raise DfpPhase1bConservationToleranceError("bound calibration artifact is not complete/final")
    if len(document["selected_window_records"]) != CALIBRATION_ROW_COUNT:
        raise DfpPhase1bConservationToleranceError("bound calibration artifact does not have exactly 200 rows")

    # independent row-level reconstruction
    reconstructed = reconstruct_family_maxima(document)

    # canonical family order must match E16 + E19
    if tuple(RESIDUAL_FAMILIES) != CANONICAL_FAMILY_ORDER:
        raise DfpPhase1bConservationToleranceError("committed RESIDUAL_FAMILIES order drifted from the canonical order")
    if identity["derivation_inputs"]["residual_family_names_in_canonical_order"] != list(CANONICAL_FAMILY_ORDER):
        raise DfpPhase1bConservationToleranceError("E19 derivation_inputs family order drifted from the canonical order")

    # per-family: frozen vs reconstructed
    for fam in CANONICAL_FAMILY_ORDER:
        frozen = identity["derivation_inputs"][fam]
        rec = reconstructed[fam]
        if frozen["reconstructed_maximum_absolute_residual"] != rec["maximum_absolute_residual"]:
            raise DfpPhase1bConservationToleranceError(
                f"derivation_inputs.{fam}: frozen M {frozen['reconstructed_maximum_absolute_residual']!r} != "
                f"row-reconstructed {rec['maximum_absolute_residual']!r}"
            )
        if frozen["sample_count"] != rec["sample_count"]:
            raise DfpPhase1bConservationToleranceError(f"derivation_inputs.{fam}: frozen sample_count != reconstructed")
        if frozen["non_finite_count"] != rec["non_finite_count"]:
            raise DfpPhase1bConservationToleranceError(f"derivation_inputs.{fam}: frozen non_finite_count != reconstructed")

    # governed family + scalar rule
    M = reconstructed[GOVERNED_FAMILY]["maximum_absolute_residual"]
    if type(M) is not float or not math.isfinite(M) or M < 0.0:
        raise DfpPhase1bConservationToleranceError("reconstructed governed maximum is not a finite non-negative float")
    tolerance = pow2_ceil(M)
    frozen_tol = identity["tolerance"]["conservation_tolerance"]
    if tolerance != frozen_tol:
        raise DfpPhase1bConservationToleranceError(
            f"recomputed tolerance {tolerance!r} (from row-reconstructed M {M!r}) != frozen E19 tolerance {frozen_tol!r}"
        )
    frozen_exp = identity["tolerance"]["conservation_tolerance_binary_exponent"]
    if tolerance != 0.0:
        if math.ldexp(1.0, frozen_exp) != tolerance:
            raise DfpPhase1bConservationToleranceError(f"frozen exponent {frozen_exp} does not reproduce the tolerance")
        if not (M <= tolerance):
            raise DfpPhase1bConservationToleranceError("row-reconstructed M exceeds the frozen tolerance")
        if not is_power_of_two(M):
            prev = math.ldexp(1.0, math.frexp(M)[1] - 1)
            if not (prev < M):
                raise DfpPhase1bConservationToleranceError("previous power of two is not strictly below M")
            if identity["derivation_inputs"]["governed_maximum"]["previous_power_of_two"] != prev:
                raise DfpPhase1bConservationToleranceError("frozen previous_power_of_two disagrees with recomputation")

    return {
        "calibration_artifact_sha256": actual_hash,
        "governed_family": GOVERNED_FAMILY,
        "reconstructed_family_maxima": {f: reconstructed[f]["maximum_absolute_residual"] for f in CANONICAL_FAMILY_ORDER},
        "reconstructed_family_sample_counts": {f: reconstructed[f]["sample_count"] for f in CANONICAL_FAMILY_ORDER},
        "governed_observed_maximum": M,
        "conservation_tolerance": tolerance,
        "conservation_tolerance_binary_exponent": frozen_exp,
        "comparison_operator": COMPARISON_OPERATOR,
        "row_count": CALIBRATION_ROW_COUNT,
        "e16_identity_sha256": e16_hash,
    }


def frozen_tolerance(identity: Mapping[str, Any] | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Return the frozen tolerance and its governing metadata from a
    (schema-validated) E19 identity. Does not touch the calibration
    artifact -- use ``verify_calibration_binding`` for that."""
    identity = identity if identity is not None else load_identity(repo_root=repo_root)
    tol = identity["tolerance"]
    return {
        "conservation_tolerance": tol["conservation_tolerance"],
        "conservation_tolerance_binary_exponent": tol["conservation_tolerance_binary_exponent"],
        "governed_family": tol["governed_family"],
        "comparison_operator": tol["comparison_operator"],
        "acceptance_relation": tol["acceptance_relation"],
        "observed_maximum_absolute_residual": tol["observed_maximum_absolute_residual"],
        "source_calibration_artifact_sha256": tol["source_calibration_artifact_sha256"],
        "label_post_calibration": tol["label_post_calibration"],
        "label_pre_evaluation": tol["label_pre_evaluation"],
        "recorded_but_ungoverned_families": list(identity["governed_residual"]["recorded_but_ungoverned_families"]),
    }


__all__ = [
    "CALIBRATION_ROW_COUNT",
    "CANONICAL_FAMILY_ORDER",
    "COMPARISON_OPERATOR",
    "DfpPhase1bConservationToleranceError",
    "FORMAT_VERSION",
    "GOVERNED_FAMILY",
    "IDENTITY_RELATIVE_PATH",
    "IDENTITY_SCHEMA_NAME",
    "IDENTITY_SCHEMA_VERSION",
    "TOP_KEYS",
    "frozen_tolerance",
    "is_power_of_two",
    "load_identity",
    "pow2_ceil",
    "reconstruct_family_maxima",
    "repository_root",
    "verify_calibration_binding",
]
