"""Torch-free, CUDA-free loader/validator for the semantic-evidence
feasibility contract (research_specs/semantic_evidence_feasibility_v1.toml)
and its companion 96-image population
(evaluation_subsets/semantic_evidence_feasibility_96.json).

This module implements NO intervention logic and NO scientific inference. It
only loads, strictly validates, and independently reconstructs/cross-checks
the frozen contract described by those two files. It must never import
Torch, MMCV, any model/dataset code from this repository, or OpenCV.

Fatal exceptions (KeyboardInterrupt, SystemExit, MemoryError) are never
swallowed by this module: every broad except-clause below re-raises them.

REVISION NOTE (RESOLVE_SEMANTIC_EVIDENCE_CONTRACT_BLOCKERS): this revision
extends validation to cover five independent-verification findings: (1) the
[propagation_target_authority] table (T0 bound, T320 deferred); (2) the
two-task/four-model-fit [statistics] structure; (3) [scores.validity]
fail-closed input/output preconditions; (4) [gaussian_blur].padding_mode;
(5) the renamed external_untracked_checkout_relative_path field. See the
contract's own revision note for the full findings list.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tomllib
from pathlib import Path
from typing import Any


class ContractError(Exception):
    """Raised for any structural, value, hash, or authorization violation."""


# A tuple of exception types that must never be caught by a broad handler in
# this module; every `except Exception` below is written to re-raise these.
_FATAL = (KeyboardInterrupt, SystemExit, MemoryError)


def _reraise_if_fatal(exc: BaseException) -> None:
    if isinstance(exc, _FATAL):
        raise exc


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def _reject_lexical_traversal(relative_path: str) -> None:
    if relative_path == "":
        raise ContractError("empty path")
    if os.path.isabs(relative_path):
        raise ContractError(f"path must be repository-relative, got absolute path: {relative_path!r}")
    parts = relative_path.split("/")
    for part in parts:
        if part in ("..", ""):
            raise ContractError(f"lexical path traversal or empty segment rejected: {relative_path!r}")
        if part == ".":
            raise ContractError(f"lexical '.' segment rejected: {relative_path!r}")


def resolve_repo_relative_path(repo_root: Path, relative_path: str) -> Path:
    """Resolve a repository-relative path, rejecting lexical '..' and any
    symlink that would escape repo_root."""
    _reject_lexical_traversal(relative_path)
    repo_root_real = Path(os.path.realpath(repo_root))
    candidate = repo_root / relative_path
    candidate_real = Path(os.path.realpath(candidate))
    try:
        candidate_real.relative_to(repo_root_real)
    except ValueError:
        raise ContractError(
            f"path escapes repository root after symlink resolution: {relative_path!r} "
            f"-> {candidate_real} (repo_root_real={repo_root_real})"
        )
    return candidate_real


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# NaN/Infinity-safe JSON loading
# ---------------------------------------------------------------------------

def _reject_non_finite_constant(token: str):
    raise ContractError(f"non-finite JSON constant is not permitted: {token}")


def load_strict_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return json.loads(text, parse_constant=_reject_non_finite_constant)


def _check_finite(value: Any, where: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ContractError(f"non-finite float at {where}: {value}")
    if isinstance(value, dict):
        for k, v in value.items():
            _check_finite(v, f"{where}.{k}")
    if isinstance(value, list):
        for i, v in enumerate(value):
            _check_finite(v, f"{where}[{i}]")


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _require_keys(d: dict, required: set, where: str) -> None:
    if not isinstance(d, dict):
        raise ContractError(f"{where}: expected a table/object, got {type(d).__name__}")
    actual = set(d.keys())
    missing = required - actual
    extra = actual - required
    if missing:
        raise ContractError(f"{where}: missing required keys: {sorted(missing)}")
    if extra:
        raise ContractError(f"{where}: unexpected/unknown keys: {sorted(extra)}")


def _require_type(value: Any, expected_type: type, where: str, allow_bool: bool = True) -> None:
    if expected_type is int and not allow_bool and isinstance(value, bool):
        raise ContractError(f"{where}: expected int, got bool")
    if expected_type is float and isinstance(value, bool):
        raise ContractError(f"{where}: expected float/number, got bool")
    if expected_type is int and isinstance(value, bool):
        raise ContractError(f"{where}: expected int, got bool")
    if expected_type is float and isinstance(value, int) and not isinstance(value, bool):
        return  # TOML/JSON ints are acceptable where a float is expected
    if not isinstance(value, expected_type):
        raise ContractError(f"{where}: expected {expected_type.__name__}, got {type(value).__name__}")


def _require_bool(value: Any, where: str) -> None:
    if not isinstance(value, bool):
        raise ContractError(f"{where}: expected bool, got {type(value).__name__}")


def _require_exact_int(value: Any, expected: int, where: str) -> None:
    _require_type(value, int, where, allow_bool=False)
    if value != expected:
        raise ContractError(f"{where}: expected exact value {expected}, got {value}")


def _require_exact(value: Any, expected: Any, where: str) -> None:
    if isinstance(expected, bool):
        _require_bool(value, where)
    elif isinstance(expected, int):
        _require_exact_int(value, expected, where)
    if value != expected:
        raise ContractError(f"{where}: expected exact value {expected!r}, got {value!r}")


# ---------------------------------------------------------------------------
# Protocol (TOML) schema
# ---------------------------------------------------------------------------

_TOP_LEVEL_KEYS = {
    "schema", "research_question", "authorization", "repository_binding",
    "step0_closure", "external_weights", "dependency", "dataset",
    "e3_geometry", "population_selection", "anchor_selection", "region",
    "interventions", "gaussian_blur", "scores", "propagation_target_authority",
    "gt_isolation", "gt_targets", "control", "statistics", "gates", "closure",
}

_AUTHORIZATION_KEYS = {
    "protocol_definition_authorized", "subset_materialization_authorized",
    "loader_validator_implementation_authorized", "identity_tests_authorized",
    "intervention_runner_implementation_authorized",
    "real_image_intervention_execution_authorized", "gpu_execution_authorized",
    "model_construction_authorized", "checkpoint_tensor_loading_authorized",
    "graph_propagation_execution_authorized", "training_authorized",
    "deployable_predictor_implementation_authorized",
    "hyperparameter_search_authorized",
    "original_baseline_full_evaluation_authorized",
    "e3_full_evaluation_authorized", "full_5000_evaluation_authorized",
    "world_model_access_authorized", "sam_access_authorized",
    "gt_based_selection_authorized", "overwrite_existing_artifacts_authorized",
    "note",
}

_REQUIRED_TRUE_AUTHORIZATIONS = {
    "protocol_definition_authorized",
    "subset_materialization_authorized",
    "loader_validator_implementation_authorized",
    "identity_tests_authorized",
}

_REQUIRED_FALSE_AUTHORIZATIONS = _AUTHORIZATION_KEYS - _REQUIRED_TRUE_AUTHORIZATIONS - {"note"}

_DEPENDENCY_KEYS = {"name", "path", "sha256", "role"}

_GATES_A_KEYS = {"name", "min_structural_validity_fraction", "max_gt_dependent_intervention_records", "max_non_finite_scores", "reliability_range", "input_validity_reference"}
_GATES_B_KEYS = {"name", "task", "min_auroc_improvement_holdout", "auroc_improvement_definition", "paired_bootstrap_95_lower_bound_required", "average_precision_difference_requirement", "computable_under_this_contract"}
_GATES_C_KEYS = _GATES_B_KEYS | {"minimum_data_requirement_from_section_15", "computable_note"}
_GATES_D_KEYS = {"name", "subgroups", "min_subgroup_size_for_check", "max_negative_auroc_difference", "rule"}
_GATES_E_KEYS = {"name", "max_h100_hours_96_image_feasibility_run", "max_h100_hours_1000_image_projection", "projection_formula_and_uncertainty_disclosure_required", "deployment_runtime_claim_authorized"}
_GATES_OUTCOME_KEYS = {"proceed_value", "proceed_condition", "stop_value", "stop_condition", "inconclusive_handling", "post_hoc_threshold_changes_authorized", "feasibility_execution_authorized", "feasibility_execution_blocked_reason"}

_EXTERNAL_WEIGHT_E3_PROJECTION_KEYS = {
    "filename", "sha256", "resolved_via", "external_untracked_checkout_relative_path",
    "git_tracked", "git_tracked_evidence", "scientific_identity", "historical_binding",
}
_EXTERNAL_WEIGHT_OTHER_KEYS = {"filename", "sha256", "resolved_via", "git_tracked", "historical_binding"}

_PROPAGATION_TARGET_AUTHORITY_TOP_KEYS = {
    "t0_status", "t320_status", "t320_source_bound", "t320_execution_authorized",
    "graph_construction_authorized", "recurrence_execution_authorized",
    "cross_branch_source_use_authorized", "ignored_bytecode_is_authority",
    "feasibility_execution_blocked_until_t320_binding", "required_successor_action",
    "t0_definition", "t320_definition", "residue_disclosure",
}
_T0_DEFINITION_KEYS = {
    "concept", "computed_from", "score_field", "predicted_class_rule",
    "identity_with_query", "tie_breaking_source", "mechanisms_excluded",
}
_T320_DEFINITION_KEYS = {
    "concept", "specified_by_this_contract", "authorized_by_this_contract",
    "graph_construction_specified", "recurrence_specified", "normalization_specified",
    "stitching_specified", "target_extraction_specified", "note",
}
_RESIDUE_DISCLOSURE_KEYS = {
    "known_related_source_exists_on_other_branches", "other_branches", "other_branches_note",
    "present_as_source_on_this_branch", "present_as_bytecode_residue_on_this_branch",
    "bytecode_residue_location", "bytecode_residue_treated_as_authority",
    "cross_branch_source_copied_or_imported_by_this_revision",
}

_SCORES_VALIDITY_KEYS = {
    "input_finiteness_required", "input_range_required", "input_fields_checked",
    "input_check_order", "input_failure_behavior", "input_failure_prohibited_actions",
    "output_finiteness_required", "output_range_required", "output_fields_checked",
    "output_check_note", "nan_laundering_hazard_acknowledged", "nan_laundering_hazard_note",
    "loader_enforcement_requirement",
}

_STATS_TASK_QC_KEYS = {
    "unit_of_analysis", "row_definition", "query_ranks", "rows_per_accepted_anchor",
    "query_identity_source", "target_name", "target_definition",
    "computable_under_this_contract", "feature_order",
}
_STATS_TASK_PO_KEYS = {
    "unit_of_analysis", "row_definition", "rows_per_eligible_anchor", "query_used",
    "target_name", "target_definition", "computable_under_this_contract",
    "computable_note", "feature_order",
}
_FEATURE_ORDER_ENTRY_KEYS_REQUIRED = {"index", "control_id", "parameterization"}
_FEATURE_ORDER_ENTRY_KEYS_OPTIONAL = {"model_b_only"}

_MODEL_FIT_KEYS_MIN = {"id", "task", "variant", "model_name", "feature_order", "fit_partition", "evaluate_partition", "computable_under_this_contract"}
_MODEL_FIT_KEYS_MAX = _MODEL_FIT_KEYS_MIN | {"feature_order_note", "computable_note"}

_PRIMARY_METRIC_KEYS = {"id", "name", "task", "row_population", "target", "producing_model_fit_ids", "metric_kind", "computable_under_this_contract", "computable_note"}
_PRIMARY_METRIC_KEYS_MIN = {"id", "name", "task", "row_population", "target", "producing_model_fit_ids", "metric_kind", "computable_under_this_contract"}


_FROZEN_SCORE_FORMULAS = {
    "s_original_q": "pre-sigmoid cosine score at the center anchor patch in ORIGINAL, for query class q",
    "s_context_q": "pre-sigmoid cosine score at the center anchor patch in CONTEXT_BLURRED, for query class q",
    "s_region_q": "pre-sigmoid cosine score at the center anchor patch in REGION_BLURRED, for query class q",
    "s_control_q": "pre-sigmoid cosine score at the center anchor patch in AREA_MATCHED_CONTROL, for query class q",
    "context_change_q": "abs(s_context(q) - s_original(q))",
    "region_drop_q": "max(0, s_original(q) - s_region(q))",
    "control_change_q": "abs(s_control(q) - s_original(q))",
    "context_stability_q": "1 - min(1, context_change(q) / 2)",
    "specific_region_drop_q": "max(0, region_drop(q) - control_change(q))",
    "region_necessity_q": "min(1, specific_region_drop(q) / 2)",
    "intervention_reliability_q": "context_stability(q) * region_necessity(q)",
}

_FROZEN_CONTROL_FORMULAS = {
    1: "s_original(q) -- the raw pre-sigmoid cosine score of the anchor patch's query class q, in ORIGINAL. q is parameterized per the consuming task: for query_correctness rows q is the row's own query (q1 or q2); for propagation_outcome rows q is fixed at q1.",
    2: "unary_margin at the anchor patch (see [anchor_selection].unary_margin_definition), in ORIGINAL",
    3: "-sum(p_i * log(p_i)) over i in 1..171, where p = numerically_stable_softmax(raw 171-dim unary score vector at the anchor patch, ORIGINAL); numerically_stable_softmax subtracts the max logit before exponentiation",
    4: "cosine_similarity(anchor_patch_dino_feature, mean(eight_neighbor_dino_features)), using the raw (unprojected) DINOv2 patch embeddings, ORIGINAL view only. Per e3_geometry.eligible_patches_always_have_8_grid_neighbors, this is always defined for eligible anchors with no edge case.",
    5: "raw_distance = min over all boundary-flagged eligible patches p_b in the window of Euclidean distance, in patch-grid units, between the anchor patch coordinates and p_b's coordinates (0 if the anchor itself is boundary); normalized_distance = min(1, raw_distance / normalization_constant)",
    6: "(count of the patch's 8 grid neighbors whose unary_top1 class equals the anchor patch's own unary_top1 class) / 8",
    8: "control_change(q) as defined in [scores.formulas], parameterized by the SAME q as control 1 for the current row (the row's own query for query_correctness rows; fixed q1 for propagation_outcome rows) -- NOT hardcoded to q1, NOT averaged over both queries",
}
_FROZEN_CONTROL_7_STATUS = "NOT_AVAILABLE_IN_E3"
_FROZEN_CONTROL_7_USED_BY_ANY_TASK = False

_FROZEN_ANCHOR_DEFINITIONS = {
    "boundary_definition": "an eligible patch is 'boundary' if ANY of its (unconditionally all 8, per e3_geometry.eligible_patches_always_have_8_grid_neighbors) 8-connected grid neighbors has a different unary_top1 class than the patch itself",
    "interior_definition": "an eligible patch that is not 'boundary'",
    "unary_margin_definition": "top1_score - top2_score, using the raw (pre-sigmoid) unary scores of the patch's own unary_top1 and unary_top2 classes",
    "median_split_population": "computed separately within each window: the median of unary_margin is computed over the window's boundary-eligible patches and, separately, over the window's interior-eligible patches",
    "median_tie_rule": "a patch whose unary_margin equals its group's median value is assigned to the LOW stratum of that group (boundary_low or interior_low)",
    "stratum_selection_rule": "within each of the 4 strata, sort member patches ascending by (anchor_key, canonical_patch_index) and take the first 2",
    "fill_rule": "if a stratum has fewer than 2 member patches, fill the missing anchor slot(s) for that window from the remaining eligible patches NOT already selected as an anchor in this window, in ascending global (anchor_key, canonical_patch_index) order across strata, until 8 total anchors are reached for the window; no duplicate patch may be selected twice within one window",
}

_FROZEN_INTERVENTION_DEFINITIONS = {
    "ORIGINAL": "the unmodified E3 crop",
    "CONTEXT_BLURRED": "core pixels unchanged bit-exactly; pixels outside the core replaced by the deterministic Gaussian-blurred version of the same crop (see [gaussian_blur])",
    "REGION_BLURRED": "pixels outside the core unchanged bit-exactly; core pixels replaced by the deterministic Gaussian-blurred version of the same crop",
    "AREA_MATCHED_CONTROL": "anchor core unchanged; a non-overlapping 3x3-patch control region elsewhere in the same window is replaced by the deterministic Gaussian-blurred version of the same crop",
}

_FROZEN_GT_PRIMARY_CORRECTNESS = {
    "min_valid_footprint_fraction": 0.75,
    "min_majority_share": 0.80,
    "ambiguous_label": "AMBIGUOUS_GT",
}

_FROZEN_GT_PROPAGATION_OUTCOME_CATEGORIES = [
    "beneficial", "harmful", "unchanged_correct", "changed_but_same_correctness", "unchanged_incorrect",
]
_FROZEN_GT_PROPAGATION_OUTCOME = {
    "beneficial_definition": "T0 wrong AND T320 correct",
    "harmful_definition": "T0 correct AND T320 wrong",
    "unchanged_correct_definition": "T0 correct AND T320 correct (implies T0_predicted_class == T320_predicted_class == GT_majority_class)",
    "changed_but_same_correctness_definition": "T0 wrong AND T320 wrong AND T0_predicted_class != T320_predicted_class (the predicted class changed, but correctness status did not)",
    "unchanged_incorrect_definition": "T0 wrong AND T320 wrong AND T0_predicted_class == T320_predicted_class (the predicted class did not change, and it is wrong)",
    "beneficial_dominant_rule": "beneficial_count > harmful_count",
    "harmful_dominant_rule": "harmful_count > beneficial_count",
    "tied_rule": "beneficial_count == harmful_count",
    "changed_but_same_correctness_excluded_from_binary_fit": True,
    "changed_but_same_correctness_still_reported": True,
}

_FROZEN_GAUSSIAN_BLUR = {
    "padding_mode": "reflect",
    "padding_mode_prohibited_alternatives": ["constant", "zero", "replicate", "circular"],
}

_FROZEN_SCORES_VALIDITY = {
    "input_finiteness_required": True,
    "input_range_required": [-1.0, 1.0],
    "output_finiteness_required": True,
    "output_range_required": [0.0, 1.0],
    "nan_laundering_hazard_acknowledged": True,
}
_FROZEN_SCORES_VALIDITY_PROHIBITED_ACTIONS = {"clamp", "impute", "replace_with_default", "silently_launder_into_a_finite_output"}


_SCORES_TOP_KEYS = {"cosine_score_range", "score_location", "formulas", "validity", "constraints"}


def _verify_frozen_strings(data: dict) -> None:
    """Byte-exact verification of every frozen formula/definition string.
    Any deviation (a 'changed formula') is a contract violation."""
    _require_keys(data["scores"], _SCORES_TOP_KEYS, "protocol.scores")
    _require_keys(
        data["scores"]["formulas"],
        set(_FROZEN_SCORE_FORMULAS.keys()) | {"intervention_reliability_range"},
        "protocol.scores.formulas",
    )
    _require_keys(data["scores"]["validity"], _SCORES_VALIDITY_KEYS, "protocol.scores.validity")

    formulas = data["scores"]["formulas"]
    for key, expected in _FROZEN_SCORE_FORMULAS.items():
        if formulas.get(key) != expected:
            raise ContractError(f"protocol.scores.formulas.{key} does not match the frozen formula text")

    controls_by_id = {c["id"]: c for c in data["control"]}
    for cid, expected in _FROZEN_CONTROL_FORMULAS.items():
        actual = controls_by_id[cid].get("formula")
        if actual != expected:
            raise ContractError(f"protocol.control[id={cid}].formula does not match the frozen formula text")
    if controls_by_id[7].get("status") != _FROZEN_CONTROL_7_STATUS:
        raise ContractError("protocol.control[id=7].status must remain exactly 'NOT_AVAILABLE_IN_E3'")
    if controls_by_id[7].get("used_by_any_task") != _FROZEN_CONTROL_7_USED_BY_ANY_TASK:
        raise ContractError("protocol.control[id=7].used_by_any_task must remain exactly false")

    anchor = data["anchor_selection"]
    for key, expected in _FROZEN_ANCHOR_DEFINITIONS.items():
        if anchor.get(key) != expected:
            raise ContractError(f"protocol.anchor_selection.{key} does not match the frozen definition text")

    interventions = data["interventions"]
    for key, expected in _FROZEN_INTERVENTION_DEFINITIONS.items():
        if interventions.get(key, {}).get("definition") != expected:
            raise ContractError(f"protocol.interventions.{key}.definition does not match the frozen definition text")

    primary = data["gt_targets"]["primary_correctness"]
    for key, expected in _FROZEN_GT_PRIMARY_CORRECTNESS.items():
        if primary.get(key) != expected:
            raise ContractError(f"protocol.gt_targets.primary_correctness.{key} does not match the frozen value")

    propagation = data["gt_targets"]["propagation_outcome"]
    if propagation.get("categories") != _FROZEN_GT_PROPAGATION_OUTCOME_CATEGORIES:
        raise ContractError(
            f"protocol.gt_targets.propagation_outcome.categories must be exactly "
            f"{_FROZEN_GT_PROPAGATION_OUTCOME_CATEGORIES}, got {propagation.get('categories')}"
        )
    for key, expected in _FROZEN_GT_PROPAGATION_OUTCOME.items():
        if propagation.get(key) != expected:
            raise ContractError(f"protocol.gt_targets.propagation_outcome.{key} does not match the frozen definition text")
    if propagation.get("computable_under_this_contract") is not False:
        raise ContractError("protocol.gt_targets.propagation_outcome.computable_under_this_contract must be exactly false")

    blur = data["gaussian_blur"]
    for key, expected in _FROZEN_GAUSSIAN_BLUR.items():
        if blur.get(key) != expected:
            raise ContractError(f"protocol.gaussian_blur.{key} does not match the frozen value")

    validity = data["scores"]["validity"]
    for key, expected in _FROZEN_SCORES_VALIDITY.items():
        if validity.get(key) != expected:
            raise ContractError(f"protocol.scores.validity.{key} does not match the frozen value")
    if set(validity.get("input_failure_prohibited_actions", [])) != _FROZEN_SCORES_VALIDITY_PROHIBITED_ACTIONS:
        raise ContractError(
            "protocol.scores.validity.input_failure_prohibited_actions must contain exactly "
            f"{sorted(_FROZEN_SCORES_VALIDITY_PROHIBITED_ACTIONS)}"
        )

    ext = data["external_weights"]["e3_projection_checkpoint"]
    if "repository_relative_location" in ext:
        raise ContractError(
            "protocol.external_weights.e3_projection_checkpoint must not contain the old "
            "'repository_relative_location' field name; use external_untracked_checkout_relative_path"
        )
    if ext.get("git_tracked") is not False:
        raise ContractError("protocol.external_weights.e3_projection_checkpoint.git_tracked must be exactly false")


def _verify_propagation_target_authority(data: dict) -> None:
    pta = data["propagation_target_authority"]
    _require_keys(pta, _PROPAGATION_TARGET_AUTHORITY_TOP_KEYS, "protocol.propagation_target_authority")

    _require_exact(pta["t0_status"], "BOUND_TO_COMMITTED_E3_UNARY_INFERENCE", "protocol.propagation_target_authority.t0_status")
    _require_exact(pta["t320_status"], "DEFERRED_REQUIRES_SUCCESSOR_AUTHORITY", "protocol.propagation_target_authority.t320_status")
    _require_exact(pta["t320_source_bound"], False, "protocol.propagation_target_authority.t320_source_bound")
    _require_exact(pta["t320_execution_authorized"], False, "protocol.propagation_target_authority.t320_execution_authorized")
    _require_exact(pta["graph_construction_authorized"], False, "protocol.propagation_target_authority.graph_construction_authorized")
    _require_exact(pta["recurrence_execution_authorized"], False, "protocol.propagation_target_authority.recurrence_execution_authorized")
    _require_exact(pta["cross_branch_source_use_authorized"], False, "protocol.propagation_target_authority.cross_branch_source_use_authorized")
    _require_exact(pta["ignored_bytecode_is_authority"], False, "protocol.propagation_target_authority.ignored_bytecode_is_authority")
    _require_exact(pta["feasibility_execution_blocked_until_t320_binding"], True, "protocol.propagation_target_authority.feasibility_execution_blocked_until_t320_binding")
    _require_type(pta["required_successor_action"], str, "protocol.propagation_target_authority.required_successor_action")
    if not pta["required_successor_action"].strip():
        raise ContractError("protocol.propagation_target_authority.required_successor_action must not be empty")

    t0 = pta["t0_definition"]
    _require_keys(t0, _T0_DEFINITION_KEYS, "protocol.propagation_target_authority.t0_definition")
    if "q1" not in t0["identity_with_query"] and "query_1" not in t0["identity_with_query"]:
        raise ContractError("protocol.propagation_target_authority.t0_definition.identity_with_query must bind T0's predicted class to q1/query_1")
    if not isinstance(t0["mechanisms_excluded"], list) or not t0["mechanisms_excluded"]:
        raise ContractError("protocol.propagation_target_authority.t0_definition.mechanisms_excluded must be a non-empty list")

    t320 = pta["t320_definition"]
    _require_keys(t320, _T320_DEFINITION_KEYS, "protocol.propagation_target_authority.t320_definition")
    for key in ("specified_by_this_contract", "authorized_by_this_contract", "graph_construction_specified",
                "recurrence_specified", "normalization_specified", "stitching_specified", "target_extraction_specified"):
        _require_exact(t320[key], False, f"protocol.propagation_target_authority.t320_definition.{key}")

    residue = pta["residue_disclosure"]
    _require_keys(residue, _RESIDUE_DISCLOSURE_KEYS, "protocol.propagation_target_authority.residue_disclosure")
    _require_exact(residue["known_related_source_exists_on_other_branches"], True, "protocol.propagation_target_authority.residue_disclosure.known_related_source_exists_on_other_branches")
    _require_exact(residue["present_as_source_on_this_branch"], False, "protocol.propagation_target_authority.residue_disclosure.present_as_source_on_this_branch")
    _require_exact(residue["bytecode_residue_treated_as_authority"], False, "protocol.propagation_target_authority.residue_disclosure.bytecode_residue_treated_as_authority")
    _require_exact(residue["cross_branch_source_copied_or_imported_by_this_revision"], False, "protocol.propagation_target_authority.residue_disclosure.cross_branch_source_copied_or_imported_by_this_revision")
    if not isinstance(residue["other_branches"], list) or not residue["other_branches"]:
        raise ContractError("protocol.propagation_target_authority.residue_disclosure.other_branches must be a non-empty list")


def _verify_statistics(data: dict) -> None:
    stats = data["statistics"]
    _require_exact(
        stats["task_identifiers"],
        ["query_correctness", "propagation_outcome"],
        "protocol.statistics.task_identifiers",
    )

    tasks = stats["tasks"]
    _require_keys(tasks, {"query_correctness", "propagation_outcome"}, "protocol.statistics.tasks")

    qc = tasks["query_correctness"]
    _require_keys(qc, _STATS_TASK_QC_KEYS, "protocol.statistics.tasks.query_correctness")
    _require_exact(qc["unit_of_analysis"], "anchor_query", "protocol.statistics.tasks.query_correctness.unit_of_analysis")
    _require_exact(qc["query_ranks"], ["q1", "q2"], "protocol.statistics.tasks.query_correctness.query_ranks")
    _require_exact(qc["rows_per_accepted_anchor"], 2, "protocol.statistics.tasks.query_correctness.rows_per_accepted_anchor")
    _require_exact(qc["computable_under_this_contract"], True, "protocol.statistics.tasks.query_correctness.computable_under_this_contract")
    if not isinstance(qc["feature_order"], list) or len(qc["feature_order"]) != 8:
        raise ContractError("protocol.statistics.tasks.query_correctness.feature_order must have exactly 8 entries")
    _verify_feature_order(qc["feature_order"], "protocol.statistics.tasks.query_correctness.feature_order")

    po = tasks["propagation_outcome"]
    _require_keys(po, _STATS_TASK_PO_KEYS, "protocol.statistics.tasks.propagation_outcome")
    _require_exact(po["unit_of_analysis"], "anchor", "protocol.statistics.tasks.propagation_outcome.unit_of_analysis")
    _require_exact(po["rows_per_eligible_anchor"], 1, "protocol.statistics.tasks.propagation_outcome.rows_per_eligible_anchor")
    _require_exact(po["query_used"], "q1", "protocol.statistics.tasks.propagation_outcome.query_used")
    _require_exact(po["computable_under_this_contract"], False, "protocol.statistics.tasks.propagation_outcome.computable_under_this_contract")
    if not isinstance(po["feature_order"], list) or len(po["feature_order"]) != 8:
        raise ContractError("protocol.statistics.tasks.propagation_outcome.feature_order must have exactly 8 entries")
    _verify_feature_order(po["feature_order"], "protocol.statistics.tasks.propagation_outcome.feature_order")
    # Task P must never reference q2 anywhere in its own feature parameterization text.
    for entry in po["feature_order"]:
        if "q2" in str(entry.get("parameterization", "")):
            raise ContractError("protocol.statistics.tasks.propagation_outcome.feature_order must never reference q2")

    ms = stats["model_settings"]
    _require_exact(ms["C"], 1.0, "protocol.statistics.model_settings.C")
    _require_exact(ms["penalty"], "l2", "protocol.statistics.model_settings.penalty")
    _require_exact(ms["class_weight"], "balanced", "protocol.statistics.model_settings.class_weight")
    _require_exact(ms["solver"], "liblinear", "protocol.statistics.model_settings.solver")
    _require_exact(ms["max_iter"], 1000, "protocol.statistics.model_settings.max_iter")
    _require_exact(ms["random_state"], 0, "protocol.statistics.model_settings.random_state")
    _require_exact(ms["hyperparameter_search"], False, "protocol.statistics.model_settings.hyperparameter_search")

    fe = stats["fit_evaluate"]
    _require_exact(fe["fit_partition"], "discovery", "protocol.statistics.fit_evaluate.fit_partition")
    _require_exact(fe["fit_partition_images"], 48, "protocol.statistics.fit_evaluate.fit_partition_images")
    _require_exact(fe["evaluate_partition"], "locked_holdout", "protocol.statistics.fit_evaluate.evaluate_partition")
    _require_exact(fe["evaluate_partition_images"], 48, "protocol.statistics.fit_evaluate.evaluate_partition_images")
    _require_exact(fe["evaluate_once"], True, "protocol.statistics.fit_evaluate.evaluate_once")

    _verify_model_fits(stats["model_fits"])
    _verify_primary_metrics(stats["primary_metrics"])

    boot = stats["bootstrap"]
    _require_exact(boot["replicates"], 10000, "protocol.statistics.bootstrap.replicates")
    _require_exact(boot["seed"], 0, "protocol.statistics.bootstrap.seed")
    _require_exact(boot["confidence_level"], 0.95, "protocol.statistics.bootstrap.confidence_level")
    _require_exact(boot["paired"], True, "protocol.statistics.bootstrap.paired")
    _require_exact(boot["refit_inside_replicates"], False, "protocol.statistics.bootstrap.refit_inside_replicates")
    if "image" not in boot["resample_unit"]:
        raise ContractError("protocol.statistics.bootstrap.resample_unit must name 'image' as the resampling unit")
    if "anchor" in boot["resample_unit"].split("(")[0]:
        raise ContractError("protocol.statistics.bootstrap.resample_unit must not resample by anchor")

    min_data = stats["minimum_data_requirements"]
    _require_exact(min_data["min_eligible_query_correctness_rows_discovery"], 500, "protocol.statistics.minimum_data_requirements.min_eligible_query_correctness_rows_discovery")
    _require_exact(min_data["min_eligible_query_correctness_rows_holdout"], 500, "protocol.statistics.minimum_data_requirements.min_eligible_query_correctness_rows_holdout")
    _require_exact(min_data["min_beneficial_dominant_holdout_propagation_outcome_rows"], 100, "protocol.statistics.minimum_data_requirements.min_beneficial_dominant_holdout_propagation_outcome_rows")
    _require_exact(min_data["min_harmful_dominant_holdout_propagation_outcome_rows"], 100, "protocol.statistics.minimum_data_requirements.min_harmful_dominant_holdout_propagation_outcome_rows")
    _require_exact(min_data["population_enlargement_after_shortfall_authorized"], False, "protocol.statistics.minimum_data_requirements.population_enlargement_after_shortfall_authorized")


def _verify_feature_order(entries: list, where: str) -> None:
    seen_indices = set()
    model_b_only_seen = False
    for i, entry in enumerate(entries):
        actual_keys = set(entry.keys())
        if not (_FEATURE_ORDER_ENTRY_KEYS_REQUIRED <= actual_keys <= (_FEATURE_ORDER_ENTRY_KEYS_REQUIRED | _FEATURE_ORDER_ENTRY_KEYS_OPTIONAL)):
            raise ContractError(f"{where}[{i}]: keys must be exactly {sorted(_FEATURE_ORDER_ENTRY_KEYS_REQUIRED)} plus optional {sorted(_FEATURE_ORDER_ENTRY_KEYS_OPTIONAL)}, got {sorted(actual_keys)}")
        if entry["index"] != i + 1:
            raise ContractError(f"{where}[{i}].index must equal {i + 1}, got {entry['index']}")
        seen_indices.add(entry["index"])
        if entry["control_id"] == "intervention_reliability":
            if not entry.get("model_b_only"):
                raise ContractError(f"{where}[{i}]: the intervention_reliability feature must set model_b_only=true")
            model_b_only_seen = True
    if seen_indices != set(range(1, len(entries) + 1)):
        raise ContractError(f"{where}: indices must be exactly 1..{len(entries)}")
    if not model_b_only_seen:
        raise ContractError(f"{where}: must contain exactly one model_b_only intervention_reliability feature")


def _verify_model_fits(model_fits: list) -> None:
    if not isinstance(model_fits, list) or len(model_fits) != 4:
        raise ContractError("protocol.statistics.model_fits must be an array of exactly 4 tables")
    seen_ids = set()
    task_variant_seen = set()
    for i, mf in enumerate(model_fits):
        actual_keys = set(mf.keys())
        if not (_MODEL_FIT_KEYS_MIN <= actual_keys <= _MODEL_FIT_KEYS_MAX):
            raise ContractError(f"protocol.statistics.model_fits[{i}]: unexpected key set {sorted(actual_keys)}")
        if mf["id"] in seen_ids:
            raise ContractError(f"duplicate model_fits id: {mf['id']}")
        seen_ids.add(mf["id"])
        if mf["task"] not in ("query_correctness", "propagation_outcome"):
            raise ContractError(f"protocol.statistics.model_fits[{i}].task must be query_correctness or propagation_outcome, got {mf['task']!r}")
        if mf["variant"] not in ("control_only", "control_plus_intervention"):
            raise ContractError(f"protocol.statistics.model_fits[{i}].variant must be control_only or control_plus_intervention, got {mf['variant']!r}")
        key = (mf["task"], mf["variant"])
        if key in task_variant_seen:
            raise ContractError(f"duplicate (task, variant) in model_fits: {key}")
        task_variant_seen.add(key)
        if mf["fit_partition"] != "discovery" or mf["evaluate_partition"] != "locked_holdout":
            raise ContractError(f"protocol.statistics.model_fits[{i}] must use fit_partition=discovery, evaluate_partition=locked_holdout")
        expected_computable = mf["task"] == "query_correctness"
        if mf["computable_under_this_contract"] != expected_computable:
            raise ContractError(
                f"protocol.statistics.model_fits[{i}] (task={mf['task']!r}): "
                f"computable_under_this_contract must be {expected_computable}"
            )
        if mf["variant"] == "control_only" and mf["feature_order"] != [1, 2, 3, 4, 5, 6, 7]:
            raise ContractError(f"protocol.statistics.model_fits[{i}] control_only feature_order must be [1,2,3,4,5,6,7]")
        if mf["variant"] == "control_plus_intervention" and mf["feature_order"] != [1, 2, 3, 4, 5, 6, 7, "intervention_reliability"]:
            raise ContractError(f"protocol.statistics.model_fits[{i}] control_plus_intervention feature_order must end with intervention_reliability")
    if seen_ids != {1, 2, 3, 4}:
        raise ContractError(f"protocol.statistics.model_fits ids must be exactly 1..4, got {sorted(seen_ids)}")
    if task_variant_seen != {
        ("query_correctness", "control_only"), ("query_correctness", "control_plus_intervention"),
        ("propagation_outcome", "control_only"), ("propagation_outcome", "control_plus_intervention"),
    }:
        raise ContractError("protocol.statistics.model_fits must cover exactly the 2x2 (task, variant) grid")


_EXPECTED_METRIC_TASK = {1: "query_correctness", 2: "query_correctness", 3: "propagation_outcome", 4: "propagation_outcome"}
_EXPECTED_METRIC_FIT_IDS = {1: [1, 2], 2: [1, 2], 3: [3, 4], 4: [3, 4]}
_EXPECTED_METRIC_COMPUTABLE = {1: True, 2: True, 3: False, 4: False}


def _verify_primary_metrics(metrics: list) -> None:
    if not isinstance(metrics, list) or len(metrics) != 4:
        raise ContractError("protocol.statistics.primary_metrics must be an array of exactly 4 tables")
    seen_ids = set()
    for i, m in enumerate(metrics):
        actual_keys = set(m.keys())
        if not (_PRIMARY_METRIC_KEYS_MIN <= actual_keys <= _PRIMARY_METRIC_KEYS):
            raise ContractError(f"protocol.statistics.primary_metrics[{i}]: unexpected key set {sorted(actual_keys)}")
        mid = m["id"]
        if mid not in (1, 2, 3, 4):
            raise ContractError(f"protocol.statistics.primary_metrics[{i}].id must be in 1..4, got {mid}")
        if mid in seen_ids:
            raise ContractError(f"duplicate primary_metrics id: {mid}")
        seen_ids.add(mid)
        if m["task"] != _EXPECTED_METRIC_TASK[mid]:
            raise ContractError(f"protocol.statistics.primary_metrics[id={mid}].task must be {_EXPECTED_METRIC_TASK[mid]!r}")
        if m["producing_model_fit_ids"] != _EXPECTED_METRIC_FIT_IDS[mid]:
            raise ContractError(f"protocol.statistics.primary_metrics[id={mid}].producing_model_fit_ids must be {_EXPECTED_METRIC_FIT_IDS[mid]}")
        if m["computable_under_this_contract"] != _EXPECTED_METRIC_COMPUTABLE[mid]:
            raise ContractError(f"protocol.statistics.primary_metrics[id={mid}].computable_under_this_contract must be {_EXPECTED_METRIC_COMPUTABLE[mid]}")
    if seen_ids != {1, 2, 3, 4}:
        raise ContractError(f"protocol.statistics.primary_metrics ids must be exactly 1..4, got {sorted(seen_ids)}")
    # Gate B <-> query_correctness metrics only; Gate C <-> propagation_outcome metrics only
    # (checked here structurally; cross-referenced again against gates in load_protocol).


def load_protocol(path: Path) -> dict:
    """Load and strictly validate research_specs/semantic_evidence_feasibility_v1.toml."""
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise ContractError(f"failed to parse TOML at {path}: {exc}") from exc

    _require_keys(data, _TOP_LEVEL_KEYS, "protocol")

    _require_keys(data["authorization"], _AUTHORIZATION_KEYS, "protocol.authorization")
    for key in _REQUIRED_TRUE_AUTHORIZATIONS:
        _require_exact(data["authorization"][key], True, f"protocol.authorization.{key}")
    for key in _REQUIRED_FALSE_AUTHORIZATIONS:
        _require_exact(data["authorization"][key], False, f"protocol.authorization.{key}")

    ext = data["external_weights"]
    _require_keys(ext["e3_projection_checkpoint"], _EXTERNAL_WEIGHT_E3_PROJECTION_KEYS, "protocol.external_weights.e3_projection_checkpoint")
    _require_keys(ext["dino_backbone_checkpoint"], _EXTERNAL_WEIGHT_OTHER_KEYS, "protocol.external_weights.dino_backbone_checkpoint")
    _require_keys(ext["clip_text_checkpoint"], _EXTERNAL_WEIGHT_OTHER_KEYS, "protocol.external_weights.clip_text_checkpoint")
    _require_exact(ext["e3_projection_checkpoint"]["sha256"], "9632117c18613674eeddcb35f86193d37da203a9cbd837cf500c4121b5fa9942", "protocol.external_weights.e3_projection_checkpoint.sha256")

    if not isinstance(data["dependency"], list) or not data["dependency"]:
        raise ContractError("protocol.dependency must be a non-empty array of tables")
    seen_dep_names = set()
    for i, dep in enumerate(data["dependency"]):
        _require_keys(dep, _DEPENDENCY_KEYS, f"protocol.dependency[{i}]")
        if dep["name"] in seen_dep_names:
            raise ContractError(f"duplicate dependency name: {dep['name']}")
        seen_dep_names.add(dep["name"])
        if len(dep["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in dep["sha256"]):
            raise ContractError(f"protocol.dependency[{i}].sha256 is not a lowercase 64-hex-char string")

    if not isinstance(data["control"], list) or len(data["control"]) != 8:
        raise ContractError("protocol.control must be an array of exactly 8 tables")
    seen_ids = set()
    for i, c in enumerate(data["control"]):
        if "id" not in c or "name" not in c:
            raise ContractError(f"protocol.control[{i}] missing id/name")
        if c["id"] in seen_ids:
            raise ContractError(f"duplicate control id: {c['id']}")
        seen_ids.add(c["id"])
    if seen_ids != set(range(1, 9)):
        raise ContractError(f"protocol.control ids must be exactly 1..8, got {sorted(seen_ids)}")

    _verify_frozen_strings(data)
    _verify_propagation_target_authority(data)
    _verify_statistics(data)

    gates = data["gates"]
    _require_keys(gates, {"A", "B", "C", "D", "E", "outcome"}, "protocol.gates")
    _require_keys(gates["A"], _GATES_A_KEYS, "protocol.gates.A")
    _require_keys(gates["B"], _GATES_B_KEYS, "protocol.gates.B")
    _require_keys(gates["C"], _GATES_C_KEYS, "protocol.gates.C")
    _require_keys(gates["D"], _GATES_D_KEYS, "protocol.gates.D")
    _require_keys(gates["E"], _GATES_E_KEYS, "protocol.gates.E")
    _require_keys(gates["outcome"], _GATES_OUTCOME_KEYS, "protocol.gates.outcome")

    _require_exact(gates["B"]["task"], "query_correctness", "protocol.gates.B.task")
    _require_exact(gates["B"]["computable_under_this_contract"], True, "protocol.gates.B.computable_under_this_contract")
    _require_exact(gates["C"]["task"], "propagation_outcome", "protocol.gates.C.task")
    _require_exact(gates["C"]["computable_under_this_contract"], False, "protocol.gates.C.computable_under_this_contract")

    if gates["A"]["reliability_range"] != [0.0, 1.0]:
        raise ContractError("protocol.gates.A.reliability_range must be exactly [0.0, 1.0]")
    if gates["B"]["min_auroc_improvement_holdout"] != 0.03:
        raise ContractError("protocol.gates.B.min_auroc_improvement_holdout must be exactly 0.03")
    if gates["C"]["min_auroc_improvement_holdout"] != 0.03:
        raise ContractError("protocol.gates.C.min_auroc_improvement_holdout must be exactly 0.03")
    if gates["D"]["max_negative_auroc_difference"] != -0.01:
        raise ContractError("protocol.gates.D.max_negative_auroc_difference must be exactly -0.01")
    if gates["E"]["max_h100_hours_96_image_feasibility_run"] != 2.0:
        raise ContractError("protocol.gates.E.max_h100_hours_96_image_feasibility_run must be exactly 2.0")
    if gates["E"]["max_h100_hours_1000_image_projection"] != 8.0:
        raise ContractError("protocol.gates.E.max_h100_hours_1000_image_projection must be exactly 8.0")

    _require_exact(gates["outcome"]["feasibility_execution_authorized"], False, "protocol.gates.outcome.feasibility_execution_authorized")
    if not gates["outcome"]["feasibility_execution_blocked_reason"].strip():
        raise ContractError("protocol.gates.outcome.feasibility_execution_blocked_reason must not be empty")

    pop = data["population_selection"]
    _require_exact(pop["namespace"], "talk2dino-semantic-evidence-feasibility-v1", "protocol.population_selection.namespace")
    _require_exact(pop["selected_count"], 96, "protocol.population_selection.selected_count")
    _require_exact(pop["discovery_count"], 48, "protocol.population_selection.discovery_count")
    _require_exact(pop["locked_holdout_count"], 48, "protocol.population_selection.locked_holdout_count")
    _require_exact(pop["discovery_rank_range"], [0, 47], "protocol.population_selection.discovery_rank_range")
    _require_exact(pop["locked_holdout_rank_range"], [48, 95], "protocol.population_selection.locked_holdout_rank_range")

    anchor = data["anchor_selection"]
    _require_exact(anchor["anchor_namespace"], "talk2dino-semantic-evidence-anchor-selection-v1", "protocol.anchor_selection.anchor_namespace")
    _require_exact(anchor["target_anchors_per_window"], 8, "protocol.anchor_selection.target_anchors_per_window")
    _require_exact(anchor["anchors_per_stratum"], 2, "protocol.anchor_selection.anchors_per_stratum")
    _require_exact(
        anchor["strata"],
        ["boundary_low", "boundary_high", "interior_low", "interior_high"],
        "protocol.anchor_selection.strata",
    )

    interventions = data["interventions"]
    _require_exact(
        interventions["views"],
        ["ORIGINAL", "CONTEXT_BLURRED", "REGION_BLURRED", "AREA_MATCHED_CONTROL"],
        "protocol.interventions.views",
    )

    blur = data["gaussian_blur"]
    _require_exact(blur["kernel_size"], [31, 31], "protocol.gaussian_blur.kernel_size")
    _require_exact(blur["sigma"], [8.0, 8.0], "protocol.gaussian_blur.sigma")
    for flag in ("randomness", "color_jitter", "resize_applied", "crop_change_applied",
                 "inpainting_applied", "generative_model_used", "feathering_applied", "gt_used", "deterministic"):
        _require_exact(blur[flag], False if flag != "deterministic" else True, f"protocol.gaussian_blur.{flag}")

    geom = data["e3_geometry"]
    _require_exact(geom["resize_dim_pixels"], 448, "protocol.e3_geometry.resize_dim_pixels")
    _require_exact(geom["patch_size_pixels"], 14, "protocol.e3_geometry.patch_size_pixels")
    _require_exact(geom["patch_grid_side"], 32, "protocol.e3_geometry.patch_grid_side")
    _require_exact(geom["eligible_grid_range"], [1, 30], "protocol.e3_geometry.eligible_grid_range")
    _require_exact(geom["crop_size_pixels"], [448, 448], "protocol.e3_geometry.crop_size_pixels")
    _require_exact(geom["stride_pixels"], [224, 224], "protocol.e3_geometry.stride_pixels")
    _require_exact(geom["pamr_active"], False, "protocol.e3_geometry.pamr_active")
    _require_exact(geom["prompt_template_count"], 7, "protocol.e3_geometry.prompt_template_count")

    dataset = data["dataset"]
    _require_exact(dataset["source_validation_population_count"], 5000, "protocol.dataset.source_validation_population_count")
    _require_exact(dataset["class_count"], 171, "protocol.dataset.class_count")
    _require_exact(dataset["ignore_label"], 255, "protocol.dataset.ignore_label")

    closure = data["closure"]
    _require_exact(closure["complete"], True, "protocol.closure.complete")
    _require_exact(closure["final"], True, "protocol.closure.final")
    _require_exact(closure["self_referential_hash_created"], False, "protocol.closure.self_referential_hash_created")

    _check_finite(data, "protocol")

    return data


# ---------------------------------------------------------------------------
# Population selection reconstruction (independent of the JSON serializer)
# ---------------------------------------------------------------------------

def selection_key(namespace: str, canonical_image_id: str) -> str:
    h = hashlib.sha256()
    h.update(namespace.encode("utf-8"))
    h.update(b"\x00")
    h.update(canonical_image_id.encode("ascii"))
    return h.hexdigest()


def digest_of_id_list(ids: list[str]) -> str:
    h = hashlib.sha256()
    for cid in ids:
        h.update(cid.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def canonical_ids_from_filename_listing(text: str) -> list[str]:
    ids = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.endswith(".jpg"):
            raise ContractError(f"unexpected filename (not .jpg): {line!r}")
        cid = line[: -len(".jpg")]
        if len(cid) != 12 or not cid.isdigit():
            raise ContractError(f"canonical image id is not 12 ASCII digits: {cid!r}")
        ids.append(cid)
    return ids


def reconstruct_selection(canonical_ids: list[str], namespace: str) -> dict:
    """Independently reconstruct the 96-record selection and digests from a
    list of canonical image IDs (order-independent input)."""
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ContractError("duplicate canonical_image_id values in source population")

    records = sorted(
        ((selection_key(namespace, cid), cid) for cid in canonical_ids),
        key=lambda t: (t[0], t[1]),
    )
    selected = records[:96]

    selected_records = []
    for rank, (skey, cid) in enumerate(selected):
        partition = "discovery" if rank <= 47 else "locked_holdout"
        selected_records.append(
            {
                "canonical_image_id": cid,
                "selection_key": skey,
                "selection_rank": rank,
                "partition": partition,
            }
        )

    canonical_source_digest = digest_of_id_list(sorted(canonical_ids))
    selected_record_digest = digest_of_id_list([r["canonical_image_id"] for r in selected_records])
    discovery_ids = [r["canonical_image_id"] for r in selected_records if r["partition"] == "discovery"]
    holdout_ids = [r["canonical_image_id"] for r in selected_records if r["partition"] == "locked_holdout"]
    discovery_digest = digest_of_id_list(discovery_ids)
    locked_holdout_digest = digest_of_id_list(holdout_ids)

    return {
        "records": selected_records,
        "canonical_source_digest": canonical_source_digest,
        "selected_record_digest": selected_record_digest,
        "discovery_digest": discovery_digest,
        "locked_holdout_digest": locked_holdout_digest,
    }


# ---------------------------------------------------------------------------
# Subset (JSON) schema
# ---------------------------------------------------------------------------

_SUBSET_TOP_KEYS = {
    "schema", "namespace", "source_validation_population_count",
    "selected_count", "discovery_count", "locked_holdout_count",
    "canonical_source_digest", "selected_record_digest", "discovery_digest",
    "locked_holdout_digest", "complete", "final", "records",
}
_SUBSET_SCHEMA_KEYS = {"name", "version"}
_RECORD_KEYS = {"canonical_image_id", "selection_key", "selection_rank", "partition"}


def load_and_verify_subset(
    subset_path: Path,
    protocol: dict,
    canonical_ids: list[str] | None = None,
) -> dict:
    """Load and strictly validate the 96-image subset JSON. If canonical_ids
    (the full 5000-ID universe) is supplied, independently reconstruct the
    selection and verify every digest and every record."""
    data = load_strict_json(subset_path)

    _require_keys(data, _SUBSET_TOP_KEYS, "subset")
    _require_keys(data["schema"], _SUBSET_SCHEMA_KEYS, "subset.schema")

    _require_exact(data["namespace"], protocol["population_selection"]["namespace"], "subset.namespace")
    _require_exact(data["source_validation_population_count"], 5000, "subset.source_validation_population_count")
    _require_exact(data["selected_count"], 96, "subset.selected_count")
    _require_exact(data["discovery_count"], 48, "subset.discovery_count")
    _require_exact(data["locked_holdout_count"], 48, "subset.locked_holdout_count")
    _require_exact(data["complete"], True, "subset.complete")
    _require_exact(data["final"], True, "subset.final")

    if not isinstance(data["records"], list) or len(data["records"]) != 96:
        raise ContractError("subset.records must be an array of exactly 96 objects")

    seen_ids = set()
    seen_keys = set()
    discovery_ids = []
    holdout_ids = []
    for i, rec in enumerate(data["records"]):
        _require_keys(rec, _RECORD_KEYS, f"subset.records[{i}]")
        _require_type(rec["canonical_image_id"], str, f"subset.records[{i}].canonical_image_id")
        if len(rec["canonical_image_id"]) != 12 or not rec["canonical_image_id"].isdigit():
            raise ContractError(f"subset.records[{i}].canonical_image_id is not 12 ASCII digits")
        _require_type(rec["selection_key"], str, f"subset.records[{i}].selection_key")
        if len(rec["selection_key"]) != 64 or any(c not in "0123456789abcdef" for c in rec["selection_key"]):
            raise ContractError(f"subset.records[{i}].selection_key is not lowercase 64-hex-char")
        _require_type(rec["selection_rank"], int, f"subset.records[{i}].selection_rank", allow_bool=False)
        if rec["selection_rank"] != i:
            raise ContractError(f"subset.records[{i}].selection_rank must equal its array index {i}, got {rec['selection_rank']}")
        _require_type(rec["partition"], str, f"subset.records[{i}].partition")
        expected_partition = "discovery" if i <= 47 else "locked_holdout"
        if rec["partition"] != expected_partition:
            raise ContractError(f"subset.records[{i}].partition must be {expected_partition!r}, got {rec['partition']!r}")

        if rec["canonical_image_id"] in seen_ids:
            raise ContractError(f"duplicate canonical_image_id in subset.records: {rec['canonical_image_id']}")
        seen_ids.add(rec["canonical_image_id"])
        if rec["selection_key"] in seen_keys:
            raise ContractError(f"duplicate selection_key in subset.records: {rec['selection_key']}")
        seen_keys.add(rec["selection_key"])

        expected_key = selection_key(data["namespace"], rec["canonical_image_id"])
        if expected_key != rec["selection_key"]:
            raise ContractError(
                f"subset.records[{i}] selection_key does not match SHA256(namespace||NUL||id): "
                f"expected {expected_key}, got {rec['selection_key']}"
            )

        if rec["partition"] == "discovery":
            discovery_ids.append(rec["canonical_image_id"])
        else:
            holdout_ids.append(rec["canonical_image_id"])

    if set(discovery_ids) & set(holdout_ids):
        raise ContractError("discovery and locked_holdout partitions overlap")
    if len(discovery_ids) != 48 or len(holdout_ids) != 48:
        raise ContractError(f"expected 48/48 split, got {len(discovery_ids)}/{len(holdout_ids)}")

    computed_selected_digest = digest_of_id_list([r["canonical_image_id"] for r in data["records"]])
    if computed_selected_digest != data["selected_record_digest"]:
        raise ContractError("subset.selected_record_digest does not match records in file")
    computed_discovery_digest = digest_of_id_list(discovery_ids)
    if computed_discovery_digest != data["discovery_digest"]:
        raise ContractError("subset.discovery_digest does not match discovery records in file")
    computed_holdout_digest = digest_of_id_list(holdout_ids)
    if computed_holdout_digest != data["locked_holdout_digest"]:
        raise ContractError("subset.locked_holdout_digest does not match locked_holdout records in file")

    frozen = protocol["population_selection"]["frozen_digests"]
    for key, subset_key in (
        ("canonical_source_digest", "canonical_source_digest"),
        ("selected_record_digest", "selected_record_digest"),
        ("discovery_digest", "discovery_digest"),
        ("locked_holdout_digest", "locked_holdout_digest"),
    ):
        if data[subset_key] != frozen[key]:
            raise ContractError(
                f"subset.{subset_key} does not match protocol.population_selection.frozen_digests.{key}: "
                f"{data[subset_key]} != {frozen[key]}"
            )

    if canonical_ids is not None:
        reconstructed = reconstruct_selection(canonical_ids, data["namespace"])
        if reconstructed["records"] != data["records"]:
            raise ContractError("independently reconstructed selection does not match subset.records exactly")
        for key in ("canonical_source_digest", "selected_record_digest", "discovery_digest", "locked_holdout_digest"):
            if reconstructed[key] != data[key]:
                raise ContractError(f"independently reconstructed {key} does not match subset file")

    _check_finite(data, "subset")

    return data


# ---------------------------------------------------------------------------
# Repository / commit bindings
# ---------------------------------------------------------------------------

def _run_git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise ContractError(f"git command failed: git -C {repo_root} {' '.join(args)}: {exc}") from exc
    return result.stdout.strip()


def verify_commit_binding(repo_root: Path, protocol: dict) -> dict:
    required_commit = protocol["repository_binding"]["e3_commit"]
    required_branch = protocol["repository_binding"]["required_branch"]

    actual_head = _run_git(repo_root, "rev-parse", "HEAD")
    actual_branch = _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")

    if actual_branch != required_branch:
        raise ContractError(f"branch mismatch: required {required_branch!r}, got {actual_branch!r}")

    is_exact = actual_head == required_commit
    is_ancestor = False
    if not is_exact:
        try:
            subprocess.run(
                ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", required_commit, actual_head],
                check=True,
                capture_output=True,
            )
            is_ancestor = True
        except subprocess.CalledProcessError:
            is_ancestor = False
        except Exception as exc:
            _reraise_if_fatal(exc)
            raise ContractError(f"failed to check ancestor relationship: {exc}") from exc

    if not (is_exact or is_ancestor):
        raise ContractError(
            f"required commit {required_commit} is neither the current HEAD ({actual_head}) "
            f"nor an ancestor of it"
        )

    return {
        "branch": actual_branch,
        "head": actual_head,
        "required_commit": required_commit,
        "exact_match": is_exact,
        "ancestor_match": is_ancestor,
    }


def verify_repository_bindings(repo_root: Path, protocol: dict) -> dict:
    """Recompute SHA-256 for every [[dependency]] entry against the current
    repository-relative file and compare to the frozen value."""
    results = []
    for dep in protocol["dependency"]:
        resolved = resolve_repo_relative_path(repo_root, dep["path"])
        if not resolved.is_file():
            raise ContractError(f"dependency {dep['name']!r}: file does not exist: {resolved}")
        actual_sha256 = sha256_file(resolved)
        match = actual_sha256 == dep["sha256"]
        if not match:
            raise ContractError(
                f"dependency {dep['name']!r} hash mismatch at {dep['path']}: "
                f"expected {dep['sha256']}, got {actual_sha256}"
            )
        results.append({"name": dep["name"], "path": dep["path"], "sha256": actual_sha256, "match": match})
    return {"dependencies_verified": len(results), "results": results}


def verify_step0_closure(protocol: dict, closure_manifest_path: str | None = None) -> dict:
    """Validate the declared external hashes and, when an explicit closure
    manifest path is supplied, hash-verify that file too. This function does
    not implicitly read any path not given to it, and does not hard-fail
    when no path is supplied (Step 0 closure lives outside the repository,
    on a scratch filesystem, and its absence in a given environment is not
    itself a contract violation of THIS file's structure)."""
    declared = protocol["step0_closure"]
    out = {"declared_manifest_sha256": declared["closure_manifest_sha256"], "checked": False}
    if closure_manifest_path is None:
        return out
    p = Path(closure_manifest_path)
    if not p.is_file():
        raise ContractError(f"provided Step 0 closure manifest path does not exist: {p}")
    actual = sha256_file(p)
    if actual != declared["closure_manifest_sha256"]:
        raise ContractError(
            f"Step 0 closure manifest hash mismatch: expected {declared['closure_manifest_sha256']}, got {actual}"
        )
    out["checked"] = True
    out["actual_sha256"] = actual
    out["match"] = True
    return out


def report_gate_configuration(protocol: dict) -> dict:
    """Return the frozen Gate A-E configuration and outcome rule, verbatim."""
    return {
        "A": protocol["gates"]["A"],
        "B": protocol["gates"]["B"],
        "C": protocol["gates"]["C"],
        "D": protocol["gates"]["D"],
        "E": protocol["gates"]["E"],
        "outcome": protocol["gates"]["outcome"],
    }


def report_propagation_target_authority(protocol: dict) -> dict:
    """Return the frozen T0/T320 authority table verbatim. A convenience
    accessor so callers do not need to know the TOML's internal key path."""
    return protocol["propagation_target_authority"]


def is_feasibility_execution_authorized(protocol: dict) -> bool:
    """Single source of truth for whether the 96-image feasibility study may
    be executed. Always false while T320 is deferred, independent of every
    flag in [authorization]."""
    pta = protocol["propagation_target_authority"]
    gates_outcome = protocol["gates"]["outcome"]
    if pta["t320_status"] != "BOUND_TO_COMMITTED_SUCCESSOR_IMPLEMENTATION":
        return False
    return bool(gates_outcome["feasibility_execution_authorized"])


# ---------------------------------------------------------------------------
# Convenience top-level entry point
# ---------------------------------------------------------------------------

def load_and_verify_all(
    repo_root: Path,
    protocol_path: Path,
    subset_path: Path,
    canonical_ids: list[str] | None = None,
    closure_manifest_path: str | None = None,
) -> dict:
    protocol = load_protocol(protocol_path)
    subset = load_and_verify_subset(subset_path, protocol, canonical_ids=canonical_ids)
    commit_check = verify_commit_binding(repo_root, protocol)
    dep_check = verify_repository_bindings(repo_root, protocol)
    closure_check = verify_step0_closure(protocol, closure_manifest_path=closure_manifest_path)
    return {
        "protocol": protocol,
        "subset": subset,
        "commit_check": commit_check,
        "dependency_check": dep_check,
        "closure_check": closure_check,
        "gates": report_gate_configuration(protocol),
        "propagation_target_authority": report_propagation_target_authority(protocol),
        "feasibility_execution_authorized": is_feasibility_execution_authorized(protocol),
    }
