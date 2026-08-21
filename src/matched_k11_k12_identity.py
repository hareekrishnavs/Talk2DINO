"""Static preflight and strict result verification for the matched k=11
versus k=12 finite-step (T=320) RWR connectivity dose-response identity.

This module deliberately avoids importing MMCV, Torch, model code, or
dataset code. All scientific constants (alpha, step count, k values, crop,
stride, historical reference metrics, tolerances) live in the authoritative
TOML at :data:`IDENTITY_RELATIVE_PATH`; this module never hardcodes a
canonical scientific value -- it only defines schema names, key sets, and
validation logic, and reuses the existing E3/RWR identity validators rather
than re-implementing them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tomllib
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.e3_evaluation_identity import load_identity as _load_e3_identity
from src.rwr_reproduction_identity import (
    FULL_PRECISION_METRIC_SOURCE as _RWR_FULL_PRECISION_METRIC_SOURCE,
    HISTORICAL_EVIDENCE_KEYS as _E10_EVIDENCE_KEYS,
    HISTORICAL_EVIDENCE_PAYLOAD_KEYS as _E10_EVIDENCE_PAYLOAD_KEYS,
    HISTORICAL_EVIDENCE_ROW_KEYS as _E10_EVIDENCE_ROW_KEYS,
    load_identity as _load_rwr_identity,
)


IDENTITY_RELATIVE_PATH = Path(
    "evaluation_identities/e12_matched_k11_k12_t320.toml"
)
RESULT_SCHEMA_NAME = "talk2dino-matched-k11-k12-t320-result-v1"
PER_IMAGE_STATS_SCHEMA_NAME = "talk2dino-matched-k11-k12-per-image-stats-v1"
SUPPORTED_PROPAGATION_METHOD = "finite_power_iteration"
SUPPORTED_METRIC_UNIT = "percent_0_100"
SUPPORTED_COMPUTE_DTYPE = "float32"
SUPPORTED_OUTPUT_DTYPE = "float32"
# Reused directly from the sibling RWR module rather than re-declared here,
# so there is exactly one place in the codebase that names this value.
SUPPORTED_METRIC_PRECISION_SOURCE = _RWR_FULL_PRECISION_METRIC_SOURCE
SUPPORTED_SELF_EDGE_POLICY = "none_for_ordinary_rows"
SUPPORTED_FALLBACK_ROW_POLICY = "zero_affinity_rows_receive_self_loop_weight_one_invariant_to_k"
SUPPORTED_AFFINITY_FUNCTION = "relu_cosine_power"
SUPPORTED_TIE_BREAK_RULE = "stable_descending_argsort_lower_patch_index_wins"
SUPPORTED_INITIAL_ITERATE = "S0"
SUPPORTED_RECURRENCE = "P_next = alpha * A_k @ P + (1 - alpha) * S0"
SUPPORTED_CONVERGENCE_TOLERANCE = "not_applicable"
SUPPORTED_CHECKPOINT_RESUME_CONTRACT = "resume_at_next_unprocessed_image_boundary_no_partial_window_state"
SUPPORTED_STITCHING_AVERAGING = "uniform"
SUPPORTED_CROP_ORDER = "canonical_row_major_sliding_window_order"
SUPPORTED_DINO_FEATURE_STAGE = "frozen_backbone_patch_tokens_l2_normalized"
SUPPORTED_CLAMPING_POLICY = "back_shifted_clamp_to_image_bounds"
SUPPORTED_WINDOW_ENUMERATION = (
    "sliding_window_geometry.SlidingWindowPlan.build "
    "(legacy clamped slide_inference grid, row-major flat index order)"
)
VARIANT_KEYS = ("k11", "k12")
VARIANT_K_VALUES = {"k11": 11, "k12": 12}

IDENTITY_TOP_KEYS = frozenset(
    {
        "format_version", "identity", "parent_identities", "dataset",
        "geometry", "snapshot", "graph", "propagation", "execution",
        "stitching", "metrics", "historical_reference",
        "window_count_reference", "result_contract", "prohibited",
    }
)
IDENTITY_SECTION_KEYS = {
    "identity": frozenset(
        {"name", "schema_version", "description", "required_ancestor_commit"}
    ),
    "parent_identities": frozenset(
        {
            "e3_identity_path", "e3_identity_name", "e3_identity_sha256",
            "rwr_identity_path", "rwr_identity_name", "rwr_identity_sha256",
            "required_relationship",
        }
    ),
    "dataset": frozenset(
        {
            "name", "protocol", "images", "classes", "background_class",
            "evaluation_split_identity",
        }
    ),
    "geometry": frozenset(
        {
            "crop", "stride", "patch_grid", "patch_size", "align_corners",
            "window_enumeration", "clamping_policy",
        }
    ),
    "snapshot": frozenset(
        {
            "raw_score_stage", "dino_feature_stage",
            "immutability_requirement", "one_shared_snapshot_per_crop",
        }
    ),
    "graph": frozenset(
        {
            "directed", "row_stochastic", "maximum_rank", "variants",
            "matched_prefix_construction", "k11_construction_method",
            "self_edge_policy", "fallback_row_policy", "affinity_function",
            "affinity_power", "tie_break_rule", "artificial_tie_perturbation",
            "graph_symmetrization",
        }
    ),
    "propagation": frozenset(
        {
            "method", "alpha", "steps", "initial_iterate", "recurrence",
            "early_stopping", "convergence_tolerance", "fallback_solver",
            "compute_dtype", "output_dtype",
        }
    ),
    "execution": frozenset(
        {
            "shared_backbone_pass", "shared_scores", "shared_features",
            "shared_affinity", "shared_top12_selection",
            "graph_normalizations", "deterministic_variant_order",
            "second_backbone_pass", "checkpoint_resume_contract",
        }
    ),
    "stitching": frozenset(
        {
            "sigmoid_applications_per_variant_window",
            "interpolation_applications_per_variant_window", "averaging",
            "crop_order", "hann", "majority_vote", "center_selection",
            "sparse_delta_restitching",
        }
    ),
    "metrics": frozenset(
        {
            "metric_names", "unit", "precision_source",
            "rounded_prettytable_forbidden", "minimum_metric_decimal_places",
            "class_count", "per_image_per_class_artifact_required",
            "image_order_digest_required",
        }
    ),
    "historical_reference": frozenset(
        {
            "reference_kind", "source_commit", "source_path", "source_blob",
            "source_sha256", "k12_alpha", "k12_steps", "k12_evaluated_images",
            "k12_aAcc", "k12_mIoU", "k12_mAcc",
            "reproduction_tolerance_absolute_mIoU", "sanity_anchor_only",
        }
    ),
    "window_count_reference": frozenset(
        {"value", "source_commit", "source_path", "source_blob", "source_sha256"}
    ),
    "result_contract": frozenset(
        {"schema_name", "variant_k_values", "variant_order"}
    ),
    "prohibited": frozenset({"list"}),
}
GRAPH_CONTRACT_KEYS = frozenset(
    {
        "directed", "row_stochastic", "maximum_rank", "affinity_function",
        "affinity_power", "tie_break_rule", "fallback_row_policy",
        "self_edge_policy", "artificial_tie_perturbation",
        "graph_symmetrization", "k11_construction_method",
    }
)
SUPPORTED_K11_CONSTRUCTION_METHOD = "matched_prefix_of_canonical_top12"
PROPAGATION_CONTRACT_KEYS = frozenset(
    {
        "method", "alpha", "steps", "early_stopping", "fallback_solver",
        "initial_iterate", "compute_dtype", "output_dtype",
    }
)
STITCHING_CONTRACT_KEYS = frozenset(
    {
        "sigmoid_applications_per_variant_window",
        "interpolation_applications_per_variant_window", "averaging",
        "crop_order", "hann", "majority_vote", "center_selection",
        "sparse_delta_restitching",
    }
)
VARIANT_RECORD_KEYS = frozenset(
    {
        "k", "aAcc", "mIoU", "mAcc", "metric_unit", "metric_source",
        "intersection", "union", "predicted_pixels", "ground_truth_pixels",
        "score_or_label_digest", "completed_image_count",
        "completed_window_count", "min_steps", "max_steps",
        "fallback_row_count", "error_count", "non_finite_count",
    }
)
RUNTIME_TELEMETRY_KEYS = frozenset(
    {
        "backbone_forward_count", "snapshot_count", "affinity_build_count",
        "top12_selection_count", "k11_propagation_count",
        "k12_propagation_count", "tie_row_count",
        "k11_prefix_mismatch_count", "fallback_row_counts",
        "early_termination", "solver_fallback_used", "cgls_call_count",
        "second_backbone_pass_count", "per_phase_runtime_seconds",
        "peak_gpu_memory_bytes", "interrupted_resumed",
    }
)
PER_IMAGE_STATISTICS_ARTIFACT_KEYS = frozenset(
    {
        "path", "sha256", "image_count", "class_count", "schema",
        "image_id_digest", "additive_aggregation_verified",
    }
)
PROVENANCE_KEYS = frozenset(
    {
        "source_git_commit", "source_git_branch", "source_git_dirty",
        "elapsed_seconds", "gpu_model", "torch_version", "cuda_version",
    }
)
TOP_RESULT_KEYS = frozenset(
    {
        "schema", "identity", "identity_sha256", "git_commit", "complete",
        "final", "dataset_identity", "image_count", "unique_image_count",
        "window_count", "image_order_digest", "score_stage",
        "graph_contract", "propagation_contract", "stitching_contract",
        "variant_results", "paired_delta", "runtime_telemetry",
        "per_image_statistics_artifact", "provenance",
    }
)
PER_PHASE_RUNTIME_KEYS = frozenset(
    {"backbone", "affinity", "propagation_k11", "propagation_k12", "stitching", "metrics"}
)
PROHIBITED_LIST = frozenset(
    {
        "semantic_consensus", "t4_targeting", "dcr", "sur",
        "sherman_morrison", "edge_influence_gradients", "learned_weights",
        "dense_inverse", "cgls", "solver_fallback", "changed_crop_stride",
        "pamr", "independently_selected_k11_graph",
        "artificial_similarity_perturbation",
    }
)


class MatchedK11K12Error(ValueError):
    """Raised when the matched k11/k12 identity or a result fails closed."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Exact-type helpers
# ---------------------------------------------------------------------------


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise MatchedK11K12Error(f"{label} must be an exact non-empty string")
    return value


def _require_supported_value(value: Any, label: str, expected: str) -> str:
    """Closed-vocabulary check: ``value`` must be an exact string matching
    ``expected`` byte-for-byte -- no normalization, case-folding, or
    whitespace-stripping. Used for every schema-vocabulary field this
    identity closes to exactly one supported value."""
    token = _require_exact_string(value, label)
    if token != expected:
        raise MatchedK11K12Error(
            f"{label} must be exactly {expected!r}, observed {token!r}"
        )
    return token


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise MatchedK11K12Error(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise MatchedK11K12Error(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise MatchedK11K12Error(f"{label} must be at least {minimum}")
    return value


def _require_exact_float(value: Any, label: str) -> float:
    if type(value) is not float:
        raise MatchedK11K12Error(f"{label} must be an exact float")
    if not math.isfinite(value):
        raise MatchedK11K12Error(f"{label} must be finite")
    return value


def _require_exact_integer_pair(value: Any, label: str) -> tuple[int, int]:
    if type(value) is not list or len(value) != 2:
        raise MatchedK11K12Error(f"{label} must be an exact two-element list")
    if any(type(item) is not int for item in value):
        raise MatchedK11K12Error(f"{label} elements must be exact integers")
    return value[0], value[1]

def _require_exact_string_list(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise MatchedK11K12Error(f"{label} must be a non-empty exact list")
    if any(type(item) is not str for item in value):
        raise MatchedK11K12Error(f"{label} elements must be exact strings")
    return tuple(value)


def _require_exact_int_list(value: Any, label: str) -> tuple[int, ...]:
    if type(value) is not list or not value:
        raise MatchedK11K12Error(f"{label} must be a non-empty exact list")
    if any(type(item) is not int for item in value):
        raise MatchedK11K12Error(f"{label} elements must be exact integers")
    return tuple(value)


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise MatchedK11K12Error(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", token) is None:
        raise MatchedK11K12Error(f"{label} must be a full Git identity")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token:
        raise MatchedK11K12Error(f"{label} must be a safe repository-relative path")
    return token


def _require_closed_mapping(
    value: Any, expected_keys: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise MatchedK11K12Error(f"{label} has an unexpected schema")
    return value


def _require_json_float(value: Any, label: str) -> float:
    if not isinstance(value, Decimal):
        raise MatchedK11K12Error(f"{label} must be a JSON floating-point number")
    if not value.is_finite():
        raise MatchedK11K12Error(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise MatchedK11K12Error(f"{label} is outside finite float range")
    return result


def _require_json_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MatchedK11K12Error(f"{label} must be an exact JSON integer")
    if minimum is not None and value < minimum:
        raise MatchedK11K12Error(f"{label} must be at least {minimum}")
    return value


def _require_json_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise MatchedK11K12Error(f"{label} must be an exact JSON boolean")
    return value


def _require_json_int_array(value: Any, label: str, *, length: int) -> list[int]:
    if type(value) is not list or len(value) != length:
        raise MatchedK11K12Error(f"{label} must be an exact array of length {length}")
    return [
        _require_json_int(item, f"{label}[{index}]", minimum=0)
        for index, item in enumerate(value)
    ]


# ---------------------------------------------------------------------------
# Identity loading
# ---------------------------------------------------------------------------


def load_identity(
    path: Path | None = None, *, repo_root: Path | None = None
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise MatchedK11K12Error(
            f"cannot load matched k11/k12 identity {source}: {error}"
        ) from error

    if set(identity) != IDENTITY_TOP_KEYS:
        raise MatchedK11K12Error("matched k11/k12 identity has an unexpected top-level schema")
    _require_exact_string(identity["format_version"], "format_version")
    if identity["format_version"] != "talk2dino-matched-k11-k12-identity-v1":
        raise MatchedK11K12Error("unsupported matched k11/k12 identity format_version")
    for section, keys in IDENTITY_SECTION_KEYS.items():
        _require_closed_mapping(identity.get(section), keys, f"identity.{section}")

    block = identity["identity"]
    _require_exact_string(block["name"], "identity.name")
    _require_exact_string(block["schema_version"], "identity.schema_version")
    if block["schema_version"] != identity["format_version"]:
        raise MatchedK11K12Error("identity.schema_version disagrees with format_version")
    _require_exact_string(block["description"], "identity.description")
    _require_git_identity(block["required_ancestor_commit"], "identity.required_ancestor_commit")

    parents = identity["parent_identities"]
    _require_relative_path(parents["e3_identity_path"], "parent_identities.e3_identity_path")
    _require_exact_string(parents["e3_identity_name"], "parent_identities.e3_identity_name")
    _require_sha256(parents["e3_identity_sha256"], "parent_identities.e3_identity_sha256")
    _require_relative_path(parents["rwr_identity_path"], "parent_identities.rwr_identity_path")
    _require_exact_string(parents["rwr_identity_name"], "parent_identities.rwr_identity_name")
    _require_sha256(parents["rwr_identity_sha256"], "parent_identities.rwr_identity_sha256")
    _require_exact_string(parents["required_relationship"], "parent_identities.required_relationship")

    dataset = identity["dataset"]
    _require_exact_string(dataset["name"], "dataset.name")
    _require_exact_string(dataset["protocol"], "dataset.protocol")
    _require_exact_int(dataset["images"], "dataset.images", minimum=1)
    _require_exact_int(dataset["classes"], "dataset.classes", minimum=1)
    _require_exact_bool(dataset["background_class"], "dataset.background_class")
    _require_exact_string(dataset["evaluation_split_identity"], "dataset.evaluation_split_identity")

    geometry = identity["geometry"]
    _require_exact_integer_pair(geometry["crop"], "geometry.crop")
    _require_exact_integer_pair(geometry["stride"], "geometry.stride")
    grid = _require_exact_integer_pair(geometry["patch_grid"], "geometry.patch_grid")
    patch = _require_exact_integer_pair(geometry["patch_size"], "geometry.patch_size")
    crop = _require_exact_integer_pair(geometry["crop"], "geometry.crop")
    if crop[0] != patch[0] * grid[0] or crop[1] != patch[1] * grid[1]:
        raise MatchedK11K12Error("geometry.crop must equal patch_size * patch_grid on both axes")
    _require_exact_bool(geometry["align_corners"], "geometry.align_corners")
    if geometry["align_corners"] is not True:
        raise MatchedK11K12Error("geometry.align_corners must be true (canonical evaluation contract)")
    _require_supported_value(
        geometry["window_enumeration"], "geometry.window_enumeration", SUPPORTED_WINDOW_ENUMERATION
    )
    _require_supported_value(geometry["clamping_policy"], "geometry.clamping_policy", SUPPORTED_CLAMPING_POLICY)

    snapshot = identity["snapshot"]
    _require_exact_string(snapshot["raw_score_stage"], "snapshot.raw_score_stage")
    _require_supported_value(snapshot["dino_feature_stage"], "snapshot.dino_feature_stage", SUPPORTED_DINO_FEATURE_STAGE)
    _require_exact_bool(snapshot["immutability_requirement"], "snapshot.immutability_requirement")
    if snapshot["immutability_requirement"] is not True:
        raise MatchedK11K12Error("snapshot.immutability_requirement must be true")
    _require_exact_bool(snapshot["one_shared_snapshot_per_crop"], "snapshot.one_shared_snapshot_per_crop")
    if snapshot["one_shared_snapshot_per_crop"] is not True:
        raise MatchedK11K12Error("snapshot.one_shared_snapshot_per_crop must be true")

    graph = identity["graph"]
    for name in ("directed", "row_stochastic", "matched_prefix_construction"):
        _require_exact_bool(graph[name], f"graph.{name}")
        if graph[name] is not True:
            raise MatchedK11K12Error(f"graph.{name} must be true")
    for name in ("artificial_tie_perturbation", "graph_symmetrization"):
        _require_exact_bool(graph[name], f"graph.{name}")
        if graph[name] is not False:
            raise MatchedK11K12Error(f"graph.{name} must be false")
    _require_exact_int(graph["maximum_rank"], "graph.maximum_rank", minimum=1)
    variants = _require_exact_int_list(graph["variants"], "graph.variants")
    if variants != (11, 12):
        raise MatchedK11K12Error("graph.variants must be exactly [11, 12]")
    if graph["maximum_rank"] != 12:
        raise MatchedK11K12Error("graph.maximum_rank must be 12")
    _require_supported_value(graph["self_edge_policy"], "graph.self_edge_policy", SUPPORTED_SELF_EDGE_POLICY)
    _require_supported_value(graph["fallback_row_policy"], "graph.fallback_row_policy", SUPPORTED_FALLBACK_ROW_POLICY)
    _require_supported_value(graph["affinity_function"], "graph.affinity_function", SUPPORTED_AFFINITY_FUNCTION)
    _require_supported_value(graph["tie_break_rule"], "graph.tie_break_rule", SUPPORTED_TIE_BREAK_RULE)
    if _require_exact_float(graph["affinity_power"], "graph.affinity_power") <= 0:
        raise MatchedK11K12Error("graph.affinity_power must be positive")
    _require_exact_string(graph["k11_construction_method"], "graph.k11_construction_method")
    if graph["k11_construction_method"] != SUPPORTED_K11_CONSTRUCTION_METHOD:
        raise MatchedK11K12Error(
            "graph.k11_construction_method must be "
            f"{SUPPORTED_K11_CONSTRUCTION_METHOD!r} -- k11 is never an independently selected graph"
        )

    propagation = identity["propagation"]
    _require_exact_string(propagation["method"], "propagation.method")
    if propagation["method"] != SUPPORTED_PROPAGATION_METHOD:
        raise MatchedK11K12Error(
            f"propagation.method capability mismatch: expected "
            f"{SUPPORTED_PROPAGATION_METHOD!r}, observed {propagation['method']!r}"
        )
    alpha = _require_exact_float(propagation["alpha"], "propagation.alpha")
    if not 0 <= alpha < 1:
        raise MatchedK11K12Error("propagation.alpha must satisfy 0 <= alpha < 1")
    _require_exact_int(propagation["steps"], "propagation.steps", minimum=1)
    _require_supported_value(propagation["initial_iterate"], "propagation.initial_iterate", SUPPORTED_INITIAL_ITERATE)
    _require_supported_value(propagation["recurrence"], "propagation.recurrence", SUPPORTED_RECURRENCE)
    _require_exact_bool(propagation["early_stopping"], "propagation.early_stopping")
    if propagation["early_stopping"] is not False:
        raise MatchedK11K12Error("propagation.early_stopping must be false")
    _require_supported_value(
        propagation["convergence_tolerance"], "propagation.convergence_tolerance", SUPPORTED_CONVERGENCE_TOLERANCE
    )
    _require_exact_string(propagation["fallback_solver"], "propagation.fallback_solver")
    if propagation["fallback_solver"] != "none":
        raise MatchedK11K12Error("propagation.fallback_solver must be 'none'")
    _require_supported_value(propagation["compute_dtype"], "propagation.compute_dtype", SUPPORTED_COMPUTE_DTYPE)
    _require_supported_value(propagation["output_dtype"], "propagation.output_dtype", SUPPORTED_OUTPUT_DTYPE)

    execution = identity["execution"]
    for name in (
        "shared_backbone_pass", "shared_scores", "shared_features",
        "shared_affinity", "shared_top12_selection",
    ):
        _require_exact_bool(execution[name], f"execution.{name}")
        if execution[name] is not True:
            raise MatchedK11K12Error(f"execution.{name} must be true")
    _require_exact_bool(execution["second_backbone_pass"], "execution.second_backbone_pass")
    if execution["second_backbone_pass"] is not False:
        raise MatchedK11K12Error("execution.second_backbone_pass must be false")
    _require_exact_int(execution["graph_normalizations"], "execution.graph_normalizations", minimum=1)
    if execution["graph_normalizations"] != 2:
        raise MatchedK11K12Error("execution.graph_normalizations must be 2 (one per variant)")
    order = _require_exact_string_list(execution["deterministic_variant_order"], "execution.deterministic_variant_order")
    if order != VARIANT_KEYS:
        raise MatchedK11K12Error(f"execution.deterministic_variant_order must be {list(VARIANT_KEYS)}")
    _require_supported_value(
        execution["checkpoint_resume_contract"],
        "execution.checkpoint_resume_contract",
        SUPPORTED_CHECKPOINT_RESUME_CONTRACT,
    )

    stitching = identity["stitching"]
    _require_exact_int(
        stitching["sigmoid_applications_per_variant_window"],
        "stitching.sigmoid_applications_per_variant_window", minimum=1,
    )
    if stitching["sigmoid_applications_per_variant_window"] != 1:
        raise MatchedK11K12Error("stitching.sigmoid_applications_per_variant_window must be 1")
    _require_exact_int(
        stitching["interpolation_applications_per_variant_window"],
        "stitching.interpolation_applications_per_variant_window", minimum=1,
    )
    if stitching["interpolation_applications_per_variant_window"] != 1:
        raise MatchedK11K12Error("stitching.interpolation_applications_per_variant_window must be 1")
    _require_supported_value(stitching["averaging"], "stitching.averaging", SUPPORTED_STITCHING_AVERAGING)
    _require_supported_value(stitching["crop_order"], "stitching.crop_order", SUPPORTED_CROP_ORDER)
    for name in ("hann", "majority_vote", "center_selection", "sparse_delta_restitching"):
        _require_exact_bool(stitching[name], f"stitching.{name}")
        if stitching[name] is not False:
            raise MatchedK11K12Error(f"stitching.{name} must be false")

    metrics = identity["metrics"]
    names = _require_exact_string_list(metrics["metric_names"], "metrics.metric_names")
    if names != ("aAcc", "mIoU", "mAcc"):
        raise MatchedK11K12Error("metrics.metric_names must be exactly ['aAcc', 'mIoU', 'mAcc']")
    _require_supported_value(metrics["unit"], "metrics.unit", SUPPORTED_METRIC_UNIT)
    _require_supported_value(
        metrics["precision_source"], "metrics.precision_source", SUPPORTED_METRIC_PRECISION_SOURCE
    )
    _require_exact_bool(metrics["rounded_prettytable_forbidden"], "metrics.rounded_prettytable_forbidden")
    if metrics["rounded_prettytable_forbidden"] is not True:
        raise MatchedK11K12Error("metrics.rounded_prettytable_forbidden must be true")
    _require_exact_int(
        metrics["minimum_metric_decimal_places"], "metrics.minimum_metric_decimal_places", minimum=1
    )
    _require_exact_int(metrics["class_count"], "metrics.class_count", minimum=1)
    if metrics["class_count"] != dataset["classes"]:
        raise MatchedK11K12Error("metrics.class_count must equal dataset.classes")
    _require_exact_bool(
        metrics["per_image_per_class_artifact_required"], "metrics.per_image_per_class_artifact_required"
    )
    if metrics["per_image_per_class_artifact_required"] is not True:
        raise MatchedK11K12Error("metrics.per_image_per_class_artifact_required must be true")
    _require_exact_bool(metrics["image_order_digest_required"], "metrics.image_order_digest_required")
    if metrics["image_order_digest_required"] is not True:
        raise MatchedK11K12Error("metrics.image_order_digest_required must be true")

    historical = identity["historical_reference"]
    _require_exact_string(historical["reference_kind"], "historical_reference.reference_kind")
    if historical["reference_kind"] != "finite_step_reference":
        raise MatchedK11K12Error(
            "historical_reference.reference_kind must be 'finite_step_reference' "
            "(a finite-step T=320 propagation is never an exact equilibrium)"
        )
    _require_git_identity(historical["source_commit"], "historical_reference.source_commit")
    _require_relative_path(historical["source_path"], "historical_reference.source_path")
    _require_git_identity(historical["source_blob"], "historical_reference.source_blob")
    _require_sha256(historical["source_sha256"], "historical_reference.source_sha256")
    k12_alpha = _require_exact_float(historical["k12_alpha"], "historical_reference.k12_alpha")
    if k12_alpha != alpha:
        raise MatchedK11K12Error("historical_reference.k12_alpha must equal propagation.alpha")
    k12_steps = _require_exact_int(historical["k12_steps"], "historical_reference.k12_steps", minimum=1)
    if k12_steps != propagation["steps"]:
        raise MatchedK11K12Error("historical_reference.k12_steps must equal propagation.steps")
    _require_exact_int(historical["k12_evaluated_images"], "historical_reference.k12_evaluated_images", minimum=1)
    if historical["k12_evaluated_images"] != dataset["images"]:
        raise MatchedK11K12Error("historical_reference.k12_evaluated_images must equal dataset.images")
    for name in ("k12_aAcc", "k12_mIoU", "k12_mAcc"):
        _require_exact_float(historical[name], f"historical_reference.{name}")
    tolerance = _require_exact_float(
        historical["reproduction_tolerance_absolute_mIoU"],
        "historical_reference.reproduction_tolerance_absolute_mIoU",
    )
    if not 0 < tolerance <= 0.05:
        raise MatchedK11K12Error(
            "historical_reference.reproduction_tolerance_absolute_mIoU must lie in (0, 0.05]"
        )
    _require_exact_bool(historical["sanity_anchor_only"], "historical_reference.sanity_anchor_only")
    if historical["sanity_anchor_only"] is not True:
        raise MatchedK11K12Error(
            "historical_reference.sanity_anchor_only must be true -- the historical T=320 "
            "k=12 artifact is a reproduction sanity check, never authoritative over a new result"
        )

    window_ref = identity["window_count_reference"]
    _require_exact_int(window_ref["value"], "window_count_reference.value", minimum=1)
    _require_git_identity(window_ref["source_commit"], "window_count_reference.source_commit")
    _require_relative_path(window_ref["source_path"], "window_count_reference.source_path")
    _require_git_identity(window_ref["source_blob"], "window_count_reference.source_blob")
    _require_sha256(window_ref["source_sha256"], "window_count_reference.source_sha256")

    result_contract = identity["result_contract"]
    _require_exact_string(result_contract["schema_name"], "result_contract.schema_name")
    if result_contract["schema_name"] != RESULT_SCHEMA_NAME:
        raise MatchedK11K12Error("result_contract.schema_name does not match the supported result schema")
    contract_k_values = _require_exact_int_list(result_contract["variant_k_values"], "result_contract.variant_k_values")
    if contract_k_values != (11, 12):
        raise MatchedK11K12Error("result_contract.variant_k_values must be exactly [11, 12]")
    contract_order = _require_exact_string_list(result_contract["variant_order"], "result_contract.variant_order")
    if contract_order != VARIANT_KEYS:
        raise MatchedK11K12Error(f"result_contract.variant_order must be {list(VARIANT_KEYS)}")

    prohibited = identity["prohibited"]
    prohibited_list = _require_exact_string_list(prohibited["list"], "prohibited.list")
    if set(prohibited_list) != PROHIBITED_LIST:
        raise MatchedK11K12Error("prohibited.list does not match the supported prohibited-practice vocabulary")

    return identity


# ---------------------------------------------------------------------------
# Git ancestry and artifact provenance
# ---------------------------------------------------------------------------


def _check_git_ancestry(root: Path, identity: Mapping[str, Any]) -> None:
    command = [
        "git", "merge-base", "--is-ancestor",
        str(identity["identity"]["required_ancestor_commit"]), "HEAD",
    ]
    try:
        result = subprocess.run(
            command, cwd=root, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except OSError as error:
        raise MatchedK11K12Error(f"cannot execute Git ancestry check: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or "required ancestor is not an ancestor of HEAD"
        raise MatchedK11K12Error(
            f"Git ancestry check failed for {identity['identity']['required_ancestor_commit']}: {detail}"
        )


ArtifactReader = Callable[[str, str], bytes]


def _git_artifact_reader(root: Path) -> ArtifactReader:
    def read(commit: str, path: str) -> bytes:
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise MatchedK11K12Error("historical commit is not a full Git SHA")
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts or "\\" in path:
            raise MatchedK11K12Error("historical Git path is unsafe")
        try:
            return subprocess.run(
                ["git", "-C", str(root), "show", f"{commit}:{path}"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ).stdout
        except subprocess.CalledProcessError as error:
            raise MatchedK11K12Error(
                f"cannot read committed artifact {commit}:{path}: "
                f"{error.stderr.decode(errors='replace').strip()}"
            ) from error
    return read


def _validate_git_artifact(
    reader: ArtifactReader, *, commit: str, path: str, blob: str, sha256: str, label: str
) -> bytes:
    data = reader(commit, path)
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise MatchedK11K12Error(f"{label} SHA256 mismatch")
    header = f"blob {len(data)}\0".encode("ascii")
    observed_blob = hashlib.sha1(header + data).hexdigest()
    if observed_blob != blob:
        raise MatchedK11K12Error(f"{label} blob identity mismatch")
    return data


def _parse_strict_json(text: str, label: str) -> Mapping[str, Any]:
    def closed_object(pairs):
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MatchedK11K12Error(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            parse_float=Decimal,
            object_pairs_hook=closed_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                MatchedK11K12Error(f"{label} contains non-finite {token}")
            ),
        )
    except MatchedK11K12Error:
        raise
    except (json.JSONDecodeError, TypeError) as error:
        raise MatchedK11K12Error(f"cannot parse {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise MatchedK11K12Error(f"{label} must contain one JSON object")
    return value


def _validate_e10_evidence_document(
    evidence: Any, *, class_count: int
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate the E10 sweep-evidence document shape, reusing the exact
    committed-attestation key sets already established by the canonical RWR
    identity (:mod:`src.rwr_reproduction_identity`) rather than
    re-deriving a parallel schema."""
    document = _require_closed_mapping(evidence, _E10_EVIDENCE_KEYS, "historical evidence")
    for name in ("cache_manifest_sha256", "class_order_sha256", "dataset_config_sha256"):
        _require_sha256(document[name], f"historical evidence {name}")
    for name in ("command", "experiment", "format_version", "invocation"):
        _require_exact_string(document[name], f"historical evidence {name}")
    _require_git_identity(document["git_commit"], "historical evidence git_commit")
    _require_json_bool(document["git_dirty"], "historical evidence git_dirty")
    payload = _require_closed_mapping(
        document["payload"], _E10_EVIDENCE_PAYLOAD_KEYS, "historical evidence payload"
    )
    baked = _require_closed_mapping(
        payload["kappa_k_baked_at_construction_time"],
        frozenset({"affinity_power", "knn_k"}),
        "historical evidence baked graph settings",
    )
    if _require_json_bool(baked["knn_k"], "historical evidence baked knn_k") is not True:
        raise MatchedK11K12Error(
            "historical evidence does not attest that k=12 was baked into the cached graph "
            "at construction time"
        )
    _require_json_bool(baked["affinity_power"], "historical evidence baked affinity_power")
    rows = payload["rows"]
    if type(rows) is not list or len(rows) != 1:
        raise MatchedK11K12Error("historical propagation rows must be a one-element array")
    row = _require_closed_mapping(rows[0], _E10_EVIDENCE_ROW_KEYS, "historical propagation row")
    for name in ("aAcc", "alpha", "mIoU", "mAcc", "runtime_seconds"):
        _require_json_float(row[name], f"historical row {name}")
    for name in ("evaluated_images", "peak_cpu_ram_bytes", "peak_gpu_bytes", "steps"):
        _require_json_int(row[name], f"historical row {name}", minimum=0)
    for name in ("per_class_iou", "per_class_accuracy"):
        value = row[name]
        if type(value) is not list or len(value) != class_count:
            raise MatchedK11K12Error(f"historical row {name} must have length {class_count}")
    for name in ("intersection", "union", "predicted_pixels", "ground_truth_pixels"):
        _require_json_int_array(row[name], f"historical row {name}", length=class_count)
    return document, row


def validate_historical_provenance(
    identity: Mapping[str, Any], *, artifact_reader: ArtifactReader
) -> dict[str, Any]:
    historical = identity["historical_reference"]
    data = _validate_git_artifact(
        artifact_reader,
        commit=historical["source_commit"],
        path=historical["source_path"],
        blob=historical["source_blob"],
        sha256=historical["source_sha256"],
        label="historical finite-step reference artifact",
    )
    _, row = _validate_e10_evidence_document(
        _parse_strict_json(data.decode("utf-8", errors="strict"), "historical finite-step reference"),
        class_count=identity["dataset"]["classes"],
    )
    if _require_json_float(row["alpha"], "historical row alpha") != historical["k12_alpha"]:
        raise MatchedK11K12Error("historical row alpha disagrees with historical_reference.k12_alpha")
    if _require_json_int(row["steps"], "historical row steps") != historical["k12_steps"]:
        raise MatchedK11K12Error("historical row steps disagrees with historical_reference.k12_steps")
    if _require_json_int(row["evaluated_images"], "historical row evaluated_images") != historical["k12_evaluated_images"]:
        raise MatchedK11K12Error("historical row evaluated_images disagrees with historical_reference.k12_evaluated_images")
    for name, identity_name in (("aAcc", "k12_aAcc"), ("mIoU", "k12_mIoU"), ("mAcc", "k12_mAcc")):
        observed = _require_json_float(row[name], f"historical row {name}")
        if observed != historical[identity_name]:
            raise MatchedK11K12Error(
                f"historical row {name} disagrees with historical_reference.{identity_name}: "
                f"artifact says {observed}, identity says {historical[identity_name]}"
            )
    return {"source_sha256": historical["source_sha256"], "row_alpha": historical["k12_alpha"], "row_steps": historical["k12_steps"]}


def validate_window_count_reference(
    identity: Mapping[str, Any], *, artifact_reader: ArtifactReader
) -> int:
    window_ref = identity["window_count_reference"]
    _validate_git_artifact(
        artifact_reader,
        commit=window_ref["source_commit"],
        path=window_ref["source_path"],
        blob=window_ref["source_blob"],
        sha256=window_ref["source_sha256"],
        label="window-count reference artifact",
    )
    return window_ref["value"]


# ---------------------------------------------------------------------------
# Parent-identity relational validation
# ---------------------------------------------------------------------------


def _validate_parent_identities(
    root: Path, identity: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    parents = identity["parent_identities"]

    e3_path = root / parents["e3_identity_path"]
    try:
        e3_bytes = e3_path.read_bytes()
    except OSError as error:
        raise MatchedK11K12Error(f"cannot read E3 parent identity {e3_path}: {error}") from error
    if hashlib.sha256(e3_bytes).hexdigest() != parents["e3_identity_sha256"]:
        raise MatchedK11K12Error("E3 parent identity file SHA256 mismatch")
    e3_identity = _load_e3_identity(e3_path, repo_root=root)
    if e3_identity["identity_name"] != parents["e3_identity_name"]:
        raise MatchedK11K12Error("E3 parent identity name mismatch")

    rwr_path = root / parents["rwr_identity_path"]
    try:
        rwr_bytes = rwr_path.read_bytes()
    except OSError as error:
        raise MatchedK11K12Error(f"cannot read RWR parent identity {rwr_path}: {error}") from error
    if hashlib.sha256(rwr_bytes).hexdigest() != parents["rwr_identity_sha256"]:
        raise MatchedK11K12Error("RWR parent identity file SHA256 mismatch")
    rwr_identity = _load_rwr_identity(rwr_path, repo_root=root)
    if rwr_identity["identity_name"] != parents["rwr_identity_name"]:
        raise MatchedK11K12Error("RWR parent identity name mismatch")

    return e3_identity, rwr_identity


def _validate_relational_equality(
    identity: Mapping[str, Any], e3_identity: Mapping[str, Any], rwr_identity: Mapping[str, Any]
) -> None:
    comparisons = {
        "dataset.name": (identity["dataset"]["name"], e3_identity["dataset"]["name"]),
        "dataset.images": (identity["dataset"]["images"], e3_identity["dataset"]["images"]),
        "dataset.classes": (identity["dataset"]["classes"], e3_identity["dataset"]["classes"]),
        "dataset.background_class": (
            identity["dataset"]["background_class"], e3_identity["dataset"]["background_class"]
        ),
        "geometry.crop": (identity["geometry"]["crop"], list(e3_identity["evaluation"]["crop"])),
        "geometry.stride": (identity["geometry"]["stride"], list(e3_identity["evaluation"]["stride"])),
        "graph.affinity_power": (identity["graph"]["affinity_power"], rwr_identity["rwr"]["affinity_power"]),
        "graph.maximum_rank": (identity["graph"]["maximum_rank"], rwr_identity["rwr"]["top_k"]),
        "propagation.alpha": (identity["propagation"]["alpha"], rwr_identity["rwr"]["alpha"]),
        "snapshot.raw_score_stage": (identity["snapshot"]["raw_score_stage"], rwr_identity["rwr"]["score_stage"]),
    }
    for label, (observed, expected) in comparisons.items():
        if observed != expected:
            raise MatchedK11K12Error(
                f"matched k11/k12 identity {label} disagrees with parent identity: "
                f"observed {observed!r}, expected {expected!r}"
            )
    # The experiment must differ from canonical RWR only in backend + degree.
    if rwr_identity["solver"]["method"] == identity["propagation"]["method"]:
        raise MatchedK11K12Error(
            "matched k11/k12 propagation.method must differ from the canonical RWR solver method"
        )
    if rwr_identity["rwr"]["top_k"] != identity["graph"]["maximum_rank"]:
        raise MatchedK11K12Error(
            "canonical RWR top_k must equal graph.maximum_rank to match the k=12 variant"
        )


# ---------------------------------------------------------------------------
# Static preflight
# ---------------------------------------------------------------------------


def validate_static_configuration(
    *,
    repo_root: Path | None = None,
    identity_path: Path | None = None,
    check_checkpoint: bool = False,
    artifact_reader: ArtifactReader | None = None,
    check_git: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    if check_git:
        _check_git_ancestry(root, identity)

    e3_identity, rwr_identity = _validate_parent_identities(root, identity)
    _validate_relational_equality(identity, e3_identity, rwr_identity)

    reader = artifact_reader or _git_artifact_reader(root)
    historical = validate_historical_provenance(identity, artifact_reader=reader)
    window_count = validate_window_count_reference(identity, artifact_reader=reader)

    from src.e3_evaluation_identity import validate_static_configuration as _e3_preflight
    from src.rwr_reproduction_identity import validate_static_configuration as _rwr_preflight

    e3_preflight = _e3_preflight(
        repo_root=root,
        identity_path=root / identity["parent_identities"]["e3_identity_path"],
        check_checkpoint=check_checkpoint,
        check_git=check_git,
    )
    rwr_preflight = _rwr_preflight(
        repo_root=root,
        identity_path=root / identity["parent_identities"]["rwr_identity_path"],
        check_checkpoint=check_checkpoint,
        check_git=check_git,
    )

    return {
        "identity_name": identity["identity"]["name"],
        "e3_identity": e3_preflight["identity_name"],
        "rwr_identity": rwr_preflight["identity_name"],
        "required_ancestor_commit": identity["identity"]["required_ancestor_commit"],
        "historical_source_sha256": historical["source_sha256"],
        "window_count": window_count,
        "checkpoint_checked": check_checkpoint,
    }


# ---------------------------------------------------------------------------
# mmseg dataset-level metric recomputation (no torch/mmcv dependency)
# ---------------------------------------------------------------------------


def recompute_metrics(
    intersection: Sequence[int],
    union: Sequence[int],
    predicted_pixels: Sequence[int],
    ground_truth_pixels: Sequence[int],
) -> tuple[float, float, float]:
    """Recompute (aAcc, mIoU, mAcc) in percent from per-class sufficient
    statistics using the exact mmseg dataset-level definitions: aAcc is the
    micro-average over all classes (sum of intersection / sum of GT
    pixels); mIoU/mAcc are macro-averages over classes with a defined
    denominator (union>0 / GT>0 respectively), skipping classes without a
    defined ratio (nanmean semantics) rather than treating them as zero."""
    if not (len(intersection) == len(union) == len(predicted_pixels) == len(ground_truth_pixels)):
        raise MatchedK11K12Error("sufficient-statistic arrays must share one length")
    total_intersection = sum(intersection)
    total_ground_truth = sum(ground_truth_pixels)
    if total_ground_truth <= 0:
        raise MatchedK11K12Error("total ground-truth pixel count must be positive")
    aAcc = 100.0 * total_intersection / total_ground_truth

    iou_values = [
        i / u for i, u in zip(intersection, union) if u > 0
    ]
    if not iou_values:
        raise MatchedK11K12Error("no class has a defined IoU denominator")
    mIoU = 100.0 * (sum(iou_values) / len(iou_values))

    acc_values = [
        i / g for i, g in zip(intersection, ground_truth_pixels) if g > 0
    ]
    if not acc_values:
        raise MatchedK11K12Error("no class has a defined accuracy denominator")
    mAcc = 100.0 * (sum(acc_values) / len(acc_values))

    return aAcc, mIoU, mAcc


# ---------------------------------------------------------------------------
# Per-image statistics artifact schema validation (Section E)
# ---------------------------------------------------------------------------


def validate_per_image_statistics_artifact_metadata(
    value: Any, identity: Mapping[str, Any]
) -> Mapping[str, Any]:
    metadata = _require_closed_mapping(
        value, PER_IMAGE_STATISTICS_ARTIFACT_KEYS, "per_image_statistics_artifact"
    )
    _require_relative_path(metadata["path"], "per_image_statistics_artifact.path")
    _require_sha256(metadata["sha256"], "per_image_statistics_artifact.sha256")
    if _require_json_int(metadata["image_count"], "per_image_statistics_artifact.image_count", minimum=1) != identity["dataset"]["images"]:
        raise MatchedK11K12Error("per_image_statistics_artifact.image_count must equal dataset.images")
    if _require_json_int(metadata["class_count"], "per_image_statistics_artifact.class_count", minimum=1) != identity["dataset"]["classes"]:
        raise MatchedK11K12Error("per_image_statistics_artifact.class_count must equal dataset.classes")
    _require_exact_string(metadata["schema"], "per_image_statistics_artifact.schema")
    if metadata["schema"] != PER_IMAGE_STATS_SCHEMA_NAME:
        raise MatchedK11K12Error("per_image_statistics_artifact.schema does not match the supported schema")
    _require_sha256(metadata["image_id_digest"], "per_image_statistics_artifact.image_id_digest")
    if _require_json_bool(
        metadata["additive_aggregation_verified"], "per_image_statistics_artifact.additive_aggregation_verified"
    ) is not True:
        raise MatchedK11K12Error("per_image_statistics_artifact.additive_aggregation_verified must be true")
    return metadata


PER_IMAGE_ROW_KEYS = frozenset(
    {"image_id", "intersection", "union", "predicted_pixels", "ground_truth_pixels"}
)


def validate_per_image_statistics_row(value: Any, *, class_count: int) -> Mapping[str, Any]:
    row = _require_closed_mapping(value, PER_IMAGE_ROW_KEYS, "per-image statistics row")
    _require_exact_string(row["image_id"], "per-image statistics row.image_id")
    for name in ("intersection", "union", "predicted_pixels", "ground_truth_pixels"):
        _require_json_int_array(row[name], f"per-image statistics row.{name}", length=class_count)
    return row


def validate_per_image_statistics_additive_aggregation(
    rows: Sequence[Mapping[str, Any]],
    *,
    class_count: int,
    dataset_intersection: Sequence[int],
    dataset_union: Sequence[int],
    dataset_predicted_pixels: Sequence[int],
    dataset_ground_truth_pixels: Sequence[int],
) -> None:
    """Additively aggregate validated per-image rows and require the sums
    reproduce the supplied dataset-level aggregate arrays exactly -- this is
    the sufficient-statistic contract the future paired-bootstrap evaluator
    depends on, independent of any specific evaluator implementation."""
    totals = {
        "intersection": [0] * class_count,
        "union": [0] * class_count,
        "predicted_pixels": [0] * class_count,
        "ground_truth_pixels": [0] * class_count,
    }
    for row in rows:
        validated = validate_per_image_statistics_row(row, class_count=class_count)
        for name in totals:
            for index, value in enumerate(validated[name]):
                totals[name][index] += value
    expectations = {
        "intersection": dataset_intersection,
        "union": dataset_union,
        "predicted_pixels": dataset_predicted_pixels,
        "ground_truth_pixels": dataset_ground_truth_pixels,
    }
    for name, expected in expectations.items():
        if list(expected) != totals[name]:
            raise MatchedK11K12Error(
                f"additive aggregation of per-image {name} does not reproduce the dataset aggregate"
            )


# ---------------------------------------------------------------------------
# Structured result parsing and verification (Section D)
# ---------------------------------------------------------------------------


def parse_structured_result(path: Path) -> Mapping[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="strict")
    except OSError as error:
        raise MatchedK11K12Error(f"cannot read result {path}: {error}") from error
    return _parse_strict_json(text, "structured result")


def _verify_contract_section(
    observed: Any, expected_source: Mapping[str, Any], keys: frozenset[str], label: str
) -> None:
    section = _require_closed_mapping(observed, keys, label)
    for name in keys:
        expected = expected_source[name]
        actual = section[name]
        if type(expected) is float:
            if not isinstance(actual, Decimal) or float(actual) != expected:
                raise MatchedK11K12Error(f"{label}.{name} mismatch")
        elif type(expected) is bool:
            if _require_json_bool(actual, f"{label}.{name}") is not expected:
                raise MatchedK11K12Error(f"{label}.{name} mismatch")
        elif type(expected) is int:
            if _require_json_int(actual, f"{label}.{name}") != expected:
                raise MatchedK11K12Error(f"{label}.{name} mismatch")
        elif type(expected) is list:
            if type(actual) is not list or [_require_json_int(v, f"{label}.{name}[i]") for v in actual] != expected:
                raise MatchedK11K12Error(f"{label}.{name} mismatch")
        else:
            if _require_exact_string(actual, f"{label}.{name}") != expected:
                raise MatchedK11K12Error(f"{label}.{name} mismatch")


def _verify_variant_record(
    value: Any, *, variant_key: str, identity: Mapping[str, Any], image_count: int, window_count: int
) -> Mapping[str, Any]:
    record = _require_closed_mapping(value, VARIANT_RECORD_KEYS, f"variant_results.{variant_key}")
    if _require_json_int(record["k"], f"variant_results.{variant_key}.k") != VARIANT_K_VALUES[variant_key]:
        raise MatchedK11K12Error(f"variant_results.{variant_key}.k must equal {VARIANT_K_VALUES[variant_key]}")

    metrics = identity["metrics"]
    minimum_places = metrics["minimum_metric_decimal_places"]
    values: dict[str, float] = {}
    for name in ("aAcc", "mIoU", "mAcc"):
        token = record[name]
        if not isinstance(token, Decimal):
            raise MatchedK11K12Error(f"variant_results.{variant_key}.{name} must be a decimal JSON number")
        if max(0, -token.as_tuple().exponent) < minimum_places:
            raise MatchedK11K12Error(
                f"variant_results.{variant_key}.{name} is rounded-only; at least {minimum_places} decimals required"
            )
        values[name] = _require_json_float(token, f"variant_results.{variant_key}.{name}")
    if any(value < 0 or value > 100 for value in values.values()):
        raise MatchedK11K12Error(f"variant_results.{variant_key} metrics must use percentage units")

    if _require_exact_string(record["metric_unit"], f"variant_results.{variant_key}.metric_unit") != metrics["unit"]:
        raise MatchedK11K12Error(f"variant_results.{variant_key}.metric_unit mismatch")
    if _require_exact_string(record["metric_source"], f"variant_results.{variant_key}.metric_source") != metrics["precision_source"]:
        raise MatchedK11K12Error(f"variant_results.{variant_key}.metric_source mismatch")

    class_count = metrics["class_count"]
    intersection = _require_json_int_array(record["intersection"], f"variant_results.{variant_key}.intersection", length=class_count)
    union = _require_json_int_array(record["union"], f"variant_results.{variant_key}.union", length=class_count)
    predicted = _require_json_int_array(record["predicted_pixels"], f"variant_results.{variant_key}.predicted_pixels", length=class_count)
    ground_truth = _require_json_int_array(record["ground_truth_pixels"], f"variant_results.{variant_key}.ground_truth_pixels", length=class_count)
    for name, array in (("intersection", intersection), ("union", union)):
        for index, (value_i, value_u) in enumerate(zip(intersection, union)):
            if value_i > value_u:
                raise MatchedK11K12Error(
                    f"variant_results.{variant_key}.intersection[{index}] exceeds union[{index}]"
                )

    recomputed_aAcc, recomputed_mIoU, recomputed_mAcc = recompute_metrics(
        intersection, union, predicted, ground_truth
    )
    for name, reported, recomputed in (
        ("aAcc", values["aAcc"], recomputed_aAcc),
        ("mIoU", values["mIoU"], recomputed_mIoU),
        ("mAcc", values["mAcc"], recomputed_mAcc),
    ):
        if abs(reported - recomputed) > 1e-6:
            raise MatchedK11K12Error(
                f"variant_results.{variant_key}.{name} disagrees with recomputation from "
                f"confusion statistics: reported {reported}, recomputed {recomputed}"
            )

    _require_sha256(record["score_or_label_digest"], f"variant_results.{variant_key}.score_or_label_digest")
    if _require_json_int(record["completed_image_count"], f"variant_results.{variant_key}.completed_image_count", minimum=0) != image_count:
        raise MatchedK11K12Error(f"variant_results.{variant_key}.completed_image_count must equal top-level image_count")
    if _require_json_int(record["completed_window_count"], f"variant_results.{variant_key}.completed_window_count", minimum=0) != window_count:
        raise MatchedK11K12Error(f"variant_results.{variant_key}.completed_window_count must equal top-level window_count")
    steps = identity["propagation"]["steps"]
    if _require_json_int(record["min_steps"], f"variant_results.{variant_key}.min_steps") != steps:
        raise MatchedK11K12Error(f"variant_results.{variant_key}.min_steps must equal {steps} exactly")
    if _require_json_int(record["max_steps"], f"variant_results.{variant_key}.max_steps") != steps:
        raise MatchedK11K12Error(f"variant_results.{variant_key}.max_steps must equal {steps} exactly")
    _require_json_int(record["fallback_row_count"], f"variant_results.{variant_key}.fallback_row_count", minimum=0)
    if _require_json_int(record["error_count"], f"variant_results.{variant_key}.error_count", minimum=0) != 0:
        raise MatchedK11K12Error(f"variant_results.{variant_key}.error_count must be 0 for a complete/final result")
    if _require_json_int(record["non_finite_count"], f"variant_results.{variant_key}.non_finite_count", minimum=0) != 0:
        raise MatchedK11K12Error(f"variant_results.{variant_key}.non_finite_count must be 0")

    return record


def verify_record(record: Mapping[str, Any], identity: Mapping[str, Any], *, identity_sha256: str) -> str:
    if not hasattr(record, "get"):
        raise MatchedK11K12Error("structured result must be a mapping")
    _require_closed_mapping(record, TOP_RESULT_KEYS, "structured result")

    if _require_exact_string(record["schema"], "result.schema") != RESULT_SCHEMA_NAME:
        raise MatchedK11K12Error("structured result schema mismatch")
    if _require_exact_string(record["identity"], "result.identity") != identity["identity"]["name"]:
        raise MatchedK11K12Error("structured result identity mismatch")
    if _require_sha256(record["identity_sha256"], "result.identity_sha256") != identity_sha256:
        raise MatchedK11K12Error("structured result identity_sha256 does not match the loaded identity file")
    _require_git_identity(record["git_commit"], "result.git_commit")
    if _require_json_bool(record["complete"], "result.complete") is not True:
        raise MatchedK11K12Error("structured result is not complete")
    if _require_json_bool(record["final"], "result.final") is not True:
        raise MatchedK11K12Error("structured result is not final")

    if _require_exact_string(record["dataset_identity"], "result.dataset_identity") != identity["dataset"]["evaluation_split_identity"]:
        raise MatchedK11K12Error("structured result dataset_identity mismatch")
    image_count = _require_json_int(record["image_count"], "result.image_count", minimum=1)
    if image_count != identity["dataset"]["images"]:
        raise MatchedK11K12Error("structured result image_count mismatch")
    unique_image_count = _require_json_int(record["unique_image_count"], "result.unique_image_count", minimum=1)
    if unique_image_count != image_count:
        raise MatchedK11K12Error("structured result unique_image_count must equal image_count (no duplicate images)")
    window_count = _require_json_int(record["window_count"], "result.window_count", minimum=1)
    if window_count != identity["window_count_reference"]["value"]:
        raise MatchedK11K12Error("structured result window_count mismatch")
    _require_sha256(record["image_order_digest"], "result.image_order_digest")
    if _require_exact_string(record["score_stage"], "result.score_stage") != identity["snapshot"]["raw_score_stage"]:
        raise MatchedK11K12Error("structured result score_stage mismatch")

    _verify_contract_section(record["graph_contract"], identity["graph"], GRAPH_CONTRACT_KEYS, "result.graph_contract")
    _verify_contract_section(record["propagation_contract"], identity["propagation"], PROPAGATION_CONTRACT_KEYS, "result.propagation_contract")
    _verify_contract_section(record["stitching_contract"], identity["stitching"], STITCHING_CONTRACT_KEYS, "result.stitching_contract")

    variant_results = record["variant_results"]
    if not isinstance(variant_results, Mapping):
        raise MatchedK11K12Error("result.variant_results must be a mapping")
    if tuple(variant_results.keys()) != VARIANT_KEYS:
        raise MatchedK11K12Error(
            f"result.variant_results must declare exactly the variants {list(VARIANT_KEYS)} in that order"
        )
    variants = {
        key: _verify_variant_record(
            variant_results[key], variant_key=key, identity=identity,
            image_count=image_count, window_count=window_count,
        )
        for key in VARIANT_KEYS
    }
    if list(variants["k11"]["ground_truth_pixels"]) != list(variants["k12"]["ground_truth_pixels"]):
        raise MatchedK11K12Error("k11 and k12 ground_truth_pixels arrays must be identical (same GT, same images)")
    if variants["k11"]["fallback_row_count"] != variants["k12"]["fallback_row_count"]:
        raise MatchedK11K12Error(
            "k11 and k12 fallback_row_count must be identical -- the zero-affinity fallback set "
            "is invariant to k by construction"
        )

    reported_delta = record["paired_delta"]
    if not isinstance(reported_delta, Decimal):
        raise MatchedK11K12Error("result.paired_delta must be a decimal JSON number")
    reported_delta_value = _require_json_float(reported_delta, "result.paired_delta")
    recomputed_delta = float(variants["k11"]["mIoU"]) - float(variants["k12"]["mIoU"])
    if abs(reported_delta_value - recomputed_delta) > 1e-6:
        raise MatchedK11K12Error(
            "result.paired_delta is inconsistent with mIoU_k11 - mIoU_k12: "
            f"reported {reported_delta_value}, recomputed {recomputed_delta}"
        )

    telemetry = _require_closed_mapping(record["runtime_telemetry"], RUNTIME_TELEMETRY_KEYS, "result.runtime_telemetry")
    for name in (
        "backbone_forward_count", "snapshot_count", "affinity_build_count", "top12_selection_count",
    ):
        if _require_json_int(telemetry[name], f"runtime_telemetry.{name}", minimum=0) != window_count:
            raise MatchedK11K12Error(f"runtime_telemetry.{name} must equal window_count exactly (no second pass)")
    if _require_json_int(telemetry["k11_propagation_count"], "runtime_telemetry.k11_propagation_count", minimum=0) != window_count:
        raise MatchedK11K12Error("runtime_telemetry.k11_propagation_count must equal window_count")
    if _require_json_int(telemetry["k12_propagation_count"], "runtime_telemetry.k12_propagation_count", minimum=0) != window_count:
        raise MatchedK11K12Error("runtime_telemetry.k12_propagation_count must equal window_count")
    _require_json_int(telemetry["tie_row_count"], "runtime_telemetry.tie_row_count", minimum=0)
    if _require_json_int(telemetry["k11_prefix_mismatch_count"], "runtime_telemetry.k11_prefix_mismatch_count", minimum=0) != 0:
        raise MatchedK11K12Error(
            "runtime_telemetry.k11_prefix_mismatch_count must be 0 -- k11 must always be the "
            "matched prefix of the canonical k12 selection, never an independently selected graph"
        )
    fallback_counts = _require_closed_mapping(
        telemetry["fallback_row_counts"], frozenset(VARIANT_KEYS), "runtime_telemetry.fallback_row_counts"
    )
    for key in VARIANT_KEYS:
        if _require_json_int(fallback_counts[key], f"runtime_telemetry.fallback_row_counts.{key}", minimum=0) != variants[key]["fallback_row_count"]:
            raise MatchedK11K12Error(f"runtime_telemetry.fallback_row_counts.{key} disagrees with variant_results.{key}.fallback_row_count")
    if _require_json_bool(telemetry["early_termination"], "runtime_telemetry.early_termination") is not False:
        raise MatchedK11K12Error("runtime_telemetry.early_termination must be false")
    if _require_json_bool(telemetry["solver_fallback_used"], "runtime_telemetry.solver_fallback_used") is not False:
        raise MatchedK11K12Error("runtime_telemetry.solver_fallback_used must be false")
    if _require_json_int(telemetry["cgls_call_count"], "runtime_telemetry.cgls_call_count", minimum=0) != 0:
        raise MatchedK11K12Error("runtime_telemetry.cgls_call_count must be 0")
    if _require_json_int(telemetry["second_backbone_pass_count"], "runtime_telemetry.second_backbone_pass_count", minimum=0) != 0:
        raise MatchedK11K12Error("runtime_telemetry.second_backbone_pass_count must be 0")
    per_phase = _require_closed_mapping(
        telemetry["per_phase_runtime_seconds"], PER_PHASE_RUNTIME_KEYS, "runtime_telemetry.per_phase_runtime_seconds"
    )
    for name in PER_PHASE_RUNTIME_KEYS:
        value = _require_json_float(per_phase[name], f"runtime_telemetry.per_phase_runtime_seconds.{name}")
        if value < 0:
            raise MatchedK11K12Error(f"runtime_telemetry.per_phase_runtime_seconds.{name} must be non-negative")
    _require_json_int(telemetry["peak_gpu_memory_bytes"], "runtime_telemetry.peak_gpu_memory_bytes", minimum=0)
    if telemetry["interrupted_resumed"] is not None:
        _require_json_bool(telemetry["interrupted_resumed"], "runtime_telemetry.interrupted_resumed")

    validate_per_image_statistics_artifact_metadata(record["per_image_statistics_artifact"], identity)

    provenance = _require_closed_mapping(record["provenance"], PROVENANCE_KEYS, "result.provenance")
    _require_git_identity(provenance["source_git_commit"], "provenance.source_git_commit")
    _require_exact_string(provenance["source_git_branch"], "provenance.source_git_branch")
    _require_json_bool(provenance["source_git_dirty"], "provenance.source_git_dirty")
    for name in ("gpu_model", "torch_version", "cuda_version"):
        _require_exact_string(provenance[name], f"provenance.{name}")
    elapsed = _require_json_float(provenance["elapsed_seconds"], "provenance.elapsed_seconds")
    if elapsed < 0:
        raise MatchedK11K12Error("provenance.elapsed_seconds must be non-negative")

    return (
        "MATCHED K11/K12 RESULT PASS "
        f"images={image_count} windows={window_count} "
        f"mIoU_k11={float(variants['k11']['mIoU']):.12f} "
        f"mIoU_k12={float(variants['k12']['mIoU']):.12f} "
        f"paired_delta={reported_delta_value:.12f}"
    )


def verify_result(
    path: Path, *, identity_path: Path | None = None, repo_root: Path | None = None
) -> str:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    resolved_identity_path = Path(identity_path) if identity_path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        identity_sha256 = hashlib.sha256(resolved_identity_path.read_bytes()).hexdigest()
    except OSError as error:
        raise MatchedK11K12Error(f"cannot hash identity file {resolved_identity_path}: {error}") from error
    return verify_record(
        parse_structured_result(path), identity, identity_sha256=identity_sha256
    )


__all__ = [
    "IDENTITY_RELATIVE_PATH",
    "MatchedK11K12Error",
    "PER_IMAGE_STATS_SCHEMA_NAME",
    "RESULT_SCHEMA_NAME",
    "SUPPORTED_AFFINITY_FUNCTION",
    "SUPPORTED_CHECKPOINT_RESUME_CONTRACT",
    "SUPPORTED_CLAMPING_POLICY",
    "SUPPORTED_COMPUTE_DTYPE",
    "SUPPORTED_CONVERGENCE_TOLERANCE",
    "SUPPORTED_CROP_ORDER",
    "SUPPORTED_DINO_FEATURE_STAGE",
    "SUPPORTED_FALLBACK_ROW_POLICY",
    "SUPPORTED_INITIAL_ITERATE",
    "SUPPORTED_METRIC_PRECISION_SOURCE",
    "SUPPORTED_METRIC_UNIT",
    "SUPPORTED_OUTPUT_DTYPE",
    "SUPPORTED_PROPAGATION_METHOD",
    "SUPPORTED_RECURRENCE",
    "SUPPORTED_SELF_EDGE_POLICY",
    "SUPPORTED_STITCHING_AVERAGING",
    "SUPPORTED_TIE_BREAK_RULE",
    "SUPPORTED_WINDOW_ENUMERATION",
    "VARIANT_K_VALUES",
    "VARIANT_KEYS",
    "load_identity",
    "parse_structured_result",
    "recompute_metrics",
    "repository_root",
    "validate_historical_provenance",
    "validate_per_image_statistics_additive_aggregation",
    "validate_per_image_statistics_artifact_metadata",
    "validate_per_image_statistics_row",
    "validate_static_configuration",
    "validate_window_count_reference",
    "verify_record",
    "verify_result",
]
