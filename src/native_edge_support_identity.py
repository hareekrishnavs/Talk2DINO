"""Static identity loading/validation for the native cross-view
directed-edge support audit.

Deliberately avoids importing MMCV, Torch, model code, or dataset code.
Scientific values already registered in the parent chain (alpha, steps,
crop, stride, affinity_power, k=12, patch_size, grid_size) are cross-
checked against this identity's own recorded copies rather than read
dynamically at every use -- exactly the pattern
:mod:`src.stitching_control_identity` already established for its own
parent chain.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e12_native_edge_support_audit.toml")

RUN_MODES = ("mechanics20",)
CHECKPOINT_SCHEMA_NAME = "talk2dino-native-edge-support-audit-checkpoint-v1"
PER_IMAGE_STATS_SCHEMA_NAME = "talk2dino-native-edge-support-audit-per-image-stats-v1"
RUN_MODE_IMAGE_COUNT_KEYS = {"mechanics20": "mechanics20_image_count"}
RUN_MODE_SCHEMA_KEYS = {"mechanics20": "mechanics20_schema_name"}

SUPPORTED_K = 12
SUPPORTED_ALPHA = 0.98
SUPPORTED_STEPS = 320
SUPPORTED_GRAPH_MODE = "directed_topk"
SUPPORTED_AFFINITY_FUNCTION = "relu_cosine_power"
SUPPORTED_AFFINITY_POWER = 3.0
SUPPORTED_CROP = (448, 448)
SUPPORTED_STRIDE = (224, 224)
SUPPORTED_PATCH_SIZE = (14, 14)
SUPPORTED_GRID_SIZE = (32, 32)
SUPPORTED_CROP_ORDER = "row_major_flat_index"
SUPPORTED_INTERPOLATION_MODE = "bilinear"
SUPPORTED_CLASS_COUNT = 171
SUPPORTED_IMAGE_COUNT = 5000
SUPPORTED_CANONICAL_ALIGNED_STRIDE_OFFSET_PATCHES = 16
SUPPORTED_UNDEFINED_REASON_ENUM = (
    "single_window_image",
    "no_exactly_aligned_observer_covering_both_endpoints",
)
SUPPORTED_RANKING_CRITERIA = (
    "defined_before_undefined",
    "lower_support_fraction_first",
    "lower_support_count_first",
    "larger_observer_count_first",
    "lower_source_graph_weight_first",
    "larger_neighbor_rank_first",
    "lower_destination_node_index_last_tiebreak",
)
SUPPORTED_RUN_MODE_IMAGE_COUNTS = {"mechanics20_image_count": 20}

IDENTITY_TOP_KEYS = frozenset(
    {
        "format_version", "identity", "parent_identity", "propagation", "dataset", "geometry",
        "clamped_window_policy", "native_alignment", "support_definition", "undefined_reason",
        "ranking", "histograms", "decision", "bounded_debug", "gt_diagnostics", "metrics", "run_modes",
        "execution_contract", "prohibited",
    }
)
IDENTITY_SECTION_KEYS = {
    "identity": frozenset({"name", "schema_version", "description", "required_ancestor_commit"}),
    "parent_identity": frozenset(
        {
            "e3_identity_path", "e3_identity_name", "e3_identity_sha256",
            "power_evaluation_identity_path", "power_evaluation_identity_name", "power_evaluation_identity_sha256",
            "matched_identity_path", "matched_identity_name", "matched_identity_sha256",
            "stitching_control_identity_path", "stitching_control_identity_name", "stitching_control_identity_sha256",
            "required_relationship",
        }
    ),
    "propagation": frozenset(
        {
            "shared_variant", "k", "alpha", "steps", "graph_mode", "directed", "affinity_function",
            "affinity_power", "self_edge_policy", "fallback_row_policy", "propagation_required_for_diagnostics_only",
        }
    ),
    "dataset": frozenset({"name", "protocol", "images", "classes", "background_class"}),
    "geometry": frozenset(
        {
            "crop", "stride", "patch_size", "grid_size", "crop_order", "window_order_source",
            "interpolation_mode", "align_corners",
        }
    ),
    "clamped_window_policy": frozenset(
        {
            "aligned_if_origin_delta_divisible_by_patch_size", "unaligned_support_is_undefined",
            "never_round_or_interpolate", "never_quantize_or_nearest_patch",
            "never_bilinearly_transport_dino_tokens", "tolerance_radius_patches",
        }
    ),
    "native_alignment": frozenset(
        {"rule", "patch_size_source", "grid_size_source", "canonical_aligned_stride_offset_patches"}
    ),
    "support_definition": frozenset(
        {
            "observer_definition", "source_window_excluded_as_observer", "reverse_edge_counts_as_support",
            "weighting", "weight_by_confidence", "weight_by_crop_position", "weight_by_distance",
            "weight_by_graph_weight", "observer_count_field", "support_count_field", "support_fraction_field",
        }
    ),
    "undefined_reason": frozenset({"enum"}),
    "ranking": frozenset({"applies_edit", "criteria_in_order"}),
    "histograms": frozenset(
        {
            "support_fraction_bucket_count", "support_count_bucket_count", "crop_edge_bands_patch_units",
            "image_edge_bands_patch_units", "displacement_bands_patch_units", "affinity_rank_bucket_count",
        }
    ),
    "decision": frozenset(
        {
            "outcomes", "minimum_misclassified_rows_for_conclusive", "alignment_limited_min_undefined_edge_fraction",
            "reachable_min_risk_ratio", "rule",
        }
    ),
    "bounded_debug": frozenset({"enabled_by_default", "max_edges_per_image", "identity_locked"}),
    "gt_diagnostics": frozenset(
        {
            "computed_after_support_frozen", "influences_support_or_ranking", "sampling_rule",
            "rescale_convention", "respects_ignore_index", "source_row_class_rule", "canonical_stitched_source",
        }
    ),
    "metrics": frozenset({"count_unit", "fraction_unit", "distance_unit", "pixel_unit", "weight_unit"}),
    "run_modes": frozenset(
        {"mechanics20_image_count", "mechanics20_schema_name", "checkpoint_schema_name", "per_image_stats_manifest_schema_name"}
    ),
    "execution_contract": frozenset(
        {
            "sample_pulls_equal_images", "backbone_snapshot_calls_equal_windows", "graph_builds_equal_windows",
            "propagation_calls_at_most_windows", "cross_view_comparisons_never_invoke_model_or_graph_builder",
            "checkpoint_granularity",
        }
    ),
    "prohibited": frozenset({"list"}),
}


class NativeEdgeSupportAuditIdentityError(ValueError):
    """Raised when the native-edge-support-audit identity, a checkpoint,
    or a result fails closed. Always fail closed: never silently
    substitute a default alignment rule, ranking, or unverified
    assumption."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be an exact non-empty string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_exact_float(value: Any, label: str) -> float:
    if type(value) is not float:
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be an exact float")
    if not math.isfinite(value):
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be finite")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", token) is None:
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be a full Git identity")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token:
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be a safe repository-relative path")
    return token


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise NativeEdgeSupportAuditIdentityError(f"{label} has an unexpected schema")
    return value


def _require_exact_string_list(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be a non-empty exact list")
    if any(type(item) is not str for item in value):
        raise NativeEdgeSupportAuditIdentityError(f"{label} elements must be exact strings")
    return tuple(value)


def _require_int_pair(value: Any, label: str) -> tuple[int, int]:
    if type(value) is not list or len(value) != 2 or any(type(v) is not int for v in value):
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be an exact [int, int] pair")
    return (value[0], value[1])


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise NativeEdgeSupportAuditIdentityError(
            f"cannot load native-edge-support-audit identity {source}: {error}"
        ) from error

    if set(identity) != IDENTITY_TOP_KEYS:
        raise NativeEdgeSupportAuditIdentityError("native-edge-support-audit identity has an unexpected top-level schema")
    _require_exact_string(identity["format_version"], "format_version")
    if identity["format_version"] != "talk2dino-native-edge-support-audit-identity-v1":
        raise NativeEdgeSupportAuditIdentityError("unsupported native-edge-support-audit identity format_version")
    for section, keys in IDENTITY_SECTION_KEYS.items():
        _require_closed_mapping(identity.get(section), keys, f"identity.{section}")

    block = identity["identity"]
    _require_exact_string(block["name"], "identity.name")
    _require_exact_string(block["schema_version"], "identity.schema_version")
    if block["schema_version"] != identity["format_version"]:
        raise NativeEdgeSupportAuditIdentityError("identity.schema_version disagrees with format_version")
    _require_exact_string(block["description"], "identity.description")
    _require_git_identity(block["required_ancestor_commit"], "identity.required_ancestor_commit")

    parent = identity["parent_identity"]
    for prefix in ("e3", "power_evaluation", "matched", "stitching_control"):
        _require_relative_path(parent[f"{prefix}_identity_path"], f"parent_identity.{prefix}_identity_path")
        _require_exact_string(parent[f"{prefix}_identity_name"], f"parent_identity.{prefix}_identity_name")
        _require_sha256(parent[f"{prefix}_identity_sha256"], f"parent_identity.{prefix}_identity_sha256")
    _require_exact_string(parent["required_relationship"], "parent_identity.required_relationship")

    prop = identity["propagation"]
    if prop["shared_variant"] != "k12":
        raise NativeEdgeSupportAuditIdentityError("propagation.shared_variant must be 'k12' -- this audit never varies the graph")
    if _require_exact_int(prop["k"], "propagation.k") != SUPPORTED_K:
        raise NativeEdgeSupportAuditIdentityError(f"propagation.k must be exactly {SUPPORTED_K}")
    if _require_exact_float(prop["alpha"], "propagation.alpha") != SUPPORTED_ALPHA:
        raise NativeEdgeSupportAuditIdentityError(f"propagation.alpha must be exactly {SUPPORTED_ALPHA}")
    if _require_exact_int(prop["steps"], "propagation.steps") != SUPPORTED_STEPS:
        raise NativeEdgeSupportAuditIdentityError(f"propagation.steps must be exactly {SUPPORTED_STEPS}")
    if prop["graph_mode"] != SUPPORTED_GRAPH_MODE:
        raise NativeEdgeSupportAuditIdentityError(f"propagation.graph_mode must be {SUPPORTED_GRAPH_MODE!r}")
    if _require_exact_bool(prop["directed"], "propagation.directed") is not True:
        raise NativeEdgeSupportAuditIdentityError("propagation.directed must be true -- never symmetrized")
    if prop["affinity_function"] != SUPPORTED_AFFINITY_FUNCTION:
        raise NativeEdgeSupportAuditIdentityError(f"propagation.affinity_function must be {SUPPORTED_AFFINITY_FUNCTION!r}")
    if _require_exact_float(prop["affinity_power"], "propagation.affinity_power") != SUPPORTED_AFFINITY_POWER:
        raise NativeEdgeSupportAuditIdentityError(f"propagation.affinity_power must be exactly {SUPPORTED_AFFINITY_POWER}")
    _require_exact_string(prop["self_edge_policy"], "propagation.self_edge_policy")
    _require_exact_string(prop["fallback_row_policy"], "propagation.fallback_row_policy")
    if _require_exact_bool(prop["propagation_required_for_diagnostics_only"], "propagation.propagation_required_for_diagnostics_only") is not True:
        raise NativeEdgeSupportAuditIdentityError("propagation.propagation_required_for_diagnostics_only must be true -- support computation never needs propagation")

    dataset = identity["dataset"]
    _require_exact_string(dataset["name"], "dataset.name")
    _require_exact_string(dataset["protocol"], "dataset.protocol")
    if _require_exact_int(dataset["images"], "dataset.images") != SUPPORTED_IMAGE_COUNT:
        raise NativeEdgeSupportAuditIdentityError(f"dataset.images must be exactly {SUPPORTED_IMAGE_COUNT}")
    if _require_exact_int(dataset["classes"], "dataset.classes") != SUPPORTED_CLASS_COUNT:
        raise NativeEdgeSupportAuditIdentityError(f"dataset.classes must be exactly {SUPPORTED_CLASS_COUNT}")
    if _require_exact_bool(dataset["background_class"], "dataset.background_class") is not False:
        raise NativeEdgeSupportAuditIdentityError("dataset.background_class must be false")

    geometry = identity["geometry"]
    if _require_int_pair(geometry["crop"], "geometry.crop") != SUPPORTED_CROP:
        raise NativeEdgeSupportAuditIdentityError(f"geometry.crop must be exactly {list(SUPPORTED_CROP)}")
    if _require_int_pair(geometry["stride"], "geometry.stride") != SUPPORTED_STRIDE:
        raise NativeEdgeSupportAuditIdentityError(f"geometry.stride must be exactly {list(SUPPORTED_STRIDE)}")
    if _require_int_pair(geometry["patch_size"], "geometry.patch_size") != SUPPORTED_PATCH_SIZE:
        raise NativeEdgeSupportAuditIdentityError(f"geometry.patch_size must be exactly {list(SUPPORTED_PATCH_SIZE)}")
    if _require_int_pair(geometry["grid_size"], "geometry.grid_size") != SUPPORTED_GRID_SIZE:
        raise NativeEdgeSupportAuditIdentityError(f"geometry.grid_size must be exactly {list(SUPPORTED_GRID_SIZE)}")
    if geometry["crop"][0] != geometry["patch_size"][0] * geometry["grid_size"][0]:
        raise NativeEdgeSupportAuditIdentityError("geometry.crop height must equal patch_size.height * grid_size.height")
    if geometry["crop"][1] != geometry["patch_size"][1] * geometry["grid_size"][1]:
        raise NativeEdgeSupportAuditIdentityError("geometry.crop width must equal patch_size.width * grid_size.width")
    if geometry["crop_order"] != SUPPORTED_CROP_ORDER:
        raise NativeEdgeSupportAuditIdentityError(f"geometry.crop_order must be {SUPPORTED_CROP_ORDER!r}")
    _require_exact_string(geometry["window_order_source"], "geometry.window_order_source")
    if geometry["interpolation_mode"] != SUPPORTED_INTERPOLATION_MODE:
        raise NativeEdgeSupportAuditIdentityError(f"geometry.interpolation_mode must be {SUPPORTED_INTERPOLATION_MODE!r}")
    if _require_exact_bool(geometry["align_corners"], "geometry.align_corners") is not True:
        raise NativeEdgeSupportAuditIdentityError("geometry.align_corners must be true")

    clamped = identity["clamped_window_policy"]
    for name, expected in (
        ("aligned_if_origin_delta_divisible_by_patch_size", True),
        ("unaligned_support_is_undefined", True),
        ("never_round_or_interpolate", True),
        ("never_quantize_or_nearest_patch", True),
        ("never_bilinearly_transport_dino_tokens", True),
    ):
        if _require_exact_bool(clamped[name], f"clamped_window_policy.{name}") is not expected:
            raise NativeEdgeSupportAuditIdentityError(f"clamped_window_policy.{name} must be {expected}")
    if _require_exact_int(clamped["tolerance_radius_patches"], "clamped_window_policy.tolerance_radius_patches") != 0:
        raise NativeEdgeSupportAuditIdentityError("clamped_window_policy.tolerance_radius_patches must be exactly 0 -- no tolerance radius is permitted")

    native = identity["native_alignment"]
    _require_exact_string(native["rule"], "native_alignment.rule")
    _require_exact_string(native["patch_size_source"], "native_alignment.patch_size_source")
    _require_exact_string(native["grid_size_source"], "native_alignment.grid_size_source")
    if _require_exact_int(native["canonical_aligned_stride_offset_patches"], "native_alignment.canonical_aligned_stride_offset_patches") != SUPPORTED_CANONICAL_ALIGNED_STRIDE_OFFSET_PATCHES:
        raise NativeEdgeSupportAuditIdentityError(
            f"native_alignment.canonical_aligned_stride_offset_patches must be exactly {SUPPORTED_CANONICAL_ALIGNED_STRIDE_OFFSET_PATCHES}"
        )
    if geometry["stride"][0] // geometry["patch_size"][0] != SUPPORTED_CANONICAL_ALIGNED_STRIDE_OFFSET_PATCHES:
        raise NativeEdgeSupportAuditIdentityError("geometry.stride / geometry.patch_size disagrees with the canonical aligned stride offset")

    support = identity["support_definition"]
    _require_exact_string(support["observer_definition"], "support_definition.observer_definition")
    if _require_exact_bool(support["source_window_excluded_as_observer"], "support_definition.source_window_excluded_as_observer") is not True:
        raise NativeEdgeSupportAuditIdentityError("support_definition.source_window_excluded_as_observer must be true")
    if _require_exact_bool(support["reverse_edge_counts_as_support"], "support_definition.reverse_edge_counts_as_support") is not False:
        raise NativeEdgeSupportAuditIdentityError("support_definition.reverse_edge_counts_as_support must be false")
    if support["weighting"] != "topology_only":
        raise NativeEdgeSupportAuditIdentityError("support_definition.weighting must be 'topology_only'")
    for name in ("weight_by_confidence", "weight_by_crop_position", "weight_by_distance", "weight_by_graph_weight"):
        if _require_exact_bool(support[name], f"support_definition.{name}") is not False:
            raise NativeEdgeSupportAuditIdentityError(f"support_definition.{name} must be false -- support is topology-only")
    for name in ("observer_count_field", "support_count_field", "support_fraction_field"):
        _require_exact_string(support[name], f"support_definition.{name}")

    undefined_reason = identity["undefined_reason"]
    if _require_exact_string_list(undefined_reason["enum"], "undefined_reason.enum") != SUPPORTED_UNDEFINED_REASON_ENUM:
        raise NativeEdgeSupportAuditIdentityError(f"undefined_reason.enum must be exactly {list(SUPPORTED_UNDEFINED_REASON_ENUM)}")

    ranking = identity["ranking"]
    if _require_exact_bool(ranking["applies_edit"], "ranking.applies_edit") is not False:
        raise NativeEdgeSupportAuditIdentityError("ranking.applies_edit must be false -- descriptive only, never an edit")
    if _require_exact_string_list(ranking["criteria_in_order"], "ranking.criteria_in_order") != SUPPORTED_RANKING_CRITERIA:
        raise NativeEdgeSupportAuditIdentityError(f"ranking.criteria_in_order must be exactly {list(SUPPORTED_RANKING_CRITERIA)}")

    histograms = identity["histograms"]
    if _require_exact_int(histograms["support_fraction_bucket_count"], "histograms.support_fraction_bucket_count") != 11:
        raise NativeEdgeSupportAuditIdentityError("histograms.support_fraction_bucket_count must be exactly 11")
    if _require_exact_int(histograms["support_count_bucket_count"], "histograms.support_count_bucket_count") != 13:
        raise NativeEdgeSupportAuditIdentityError("histograms.support_count_bucket_count must be exactly 13")
    if _require_exact_string_list(histograms["crop_edge_bands_patch_units"], "histograms.crop_edge_bands_patch_units") != ("0", "1", "2", "3-4", "5-7", ">=8"):
        raise NativeEdgeSupportAuditIdentityError("histograms.crop_edge_bands_patch_units does not match the identity-locked bands")
    if _require_exact_string_list(histograms["image_edge_bands_patch_units"], "histograms.image_edge_bands_patch_units") != ("0", "1", "2", "3-4", "5-7", ">=8"):
        raise NativeEdgeSupportAuditIdentityError("histograms.image_edge_bands_patch_units does not match the identity-locked bands")
    if _require_exact_string_list(histograms["displacement_bands_patch_units"], "histograms.displacement_bands_patch_units") != ("0", "1", "2", "3-4", "5-7", ">=8"):
        raise NativeEdgeSupportAuditIdentityError("histograms.displacement_bands_patch_units does not match the identity-locked bands")
    if _require_exact_int(histograms["affinity_rank_bucket_count"], "histograms.affinity_rank_bucket_count") != 12:
        raise NativeEdgeSupportAuditIdentityError("histograms.affinity_rank_bucket_count must be exactly 12")

    decision = identity["decision"]
    if _require_exact_string_list(decision["outcomes"], "decision.outcomes") != ("REACHABLE", "STRUCTURALLY_UNREACHABLE", "ALIGNMENT_LIMITED", "INCONCLUSIVE"):
        raise NativeEdgeSupportAuditIdentityError("decision.outcomes does not match the identity-locked enum")
    _require_exact_int(decision["minimum_misclassified_rows_for_conclusive"], "decision.minimum_misclassified_rows_for_conclusive", minimum=1)
    fraction = _require_exact_float(decision["alignment_limited_min_undefined_edge_fraction"], "decision.alignment_limited_min_undefined_edge_fraction")
    if not 0.0 < fraction <= 1.0:
        raise NativeEdgeSupportAuditIdentityError("decision.alignment_limited_min_undefined_edge_fraction must be in (0, 1]")
    ratio = _require_exact_float(decision["reachable_min_risk_ratio"], "decision.reachable_min_risk_ratio")
    if ratio <= 0:
        raise NativeEdgeSupportAuditIdentityError("decision.reachable_min_risk_ratio must be strictly positive")
    _require_exact_string(decision["rule"], "decision.rule")

    bounded_debug = identity["bounded_debug"]
    if _require_exact_bool(bounded_debug["enabled_by_default"], "bounded_debug.enabled_by_default") is not False:
        raise NativeEdgeSupportAuditIdentityError("bounded_debug.enabled_by_default must be false")
    _require_exact_int(bounded_debug["max_edges_per_image"], "bounded_debug.max_edges_per_image", minimum=1)
    if _require_exact_bool(bounded_debug["identity_locked"], "bounded_debug.identity_locked") is not True:
        raise NativeEdgeSupportAuditIdentityError("bounded_debug.identity_locked must be true")

    gt = identity["gt_diagnostics"]
    if _require_exact_bool(gt["computed_after_support_frozen"], "gt_diagnostics.computed_after_support_frozen") is not True:
        raise NativeEdgeSupportAuditIdentityError("gt_diagnostics.computed_after_support_frozen must be true")
    if _require_exact_bool(gt["influences_support_or_ranking"], "gt_diagnostics.influences_support_or_ranking") is not False:
        raise NativeEdgeSupportAuditIdentityError("gt_diagnostics.influences_support_or_ranking must be false -- GT never feeds back into support or ranking")
    _require_exact_string(gt["sampling_rule"], "gt_diagnostics.sampling_rule")
    _require_exact_string(gt["rescale_convention"], "gt_diagnostics.rescale_convention")
    if _require_exact_bool(gt["respects_ignore_index"], "gt_diagnostics.respects_ignore_index") is not True:
        raise NativeEdgeSupportAuditIdentityError("gt_diagnostics.respects_ignore_index must be true")
    _require_exact_string(gt["source_row_class_rule"], "gt_diagnostics.source_row_class_rule")
    if gt["canonical_stitched_source"] != "uniform_probability_variant_finalize_prediction":
        raise NativeEdgeSupportAuditIdentityError("gt_diagnostics.canonical_stitched_source must be 'uniform_probability_variant_finalize_prediction'")

    metrics = identity["metrics"]
    for name, expected in (
        ("count_unit", "count"), ("fraction_unit", "fraction_0_1"), ("distance_unit", "patch_units"),
        ("pixel_unit", "pixel_units"), ("weight_unit", "dimensionless"),
    ):
        if metrics[name] != expected:
            raise NativeEdgeSupportAuditIdentityError(f"metrics.{name} must be {expected!r}")

    modes = identity["run_modes"]
    for name, expected_count in SUPPORTED_RUN_MODE_IMAGE_COUNTS.items():
        count = _require_exact_int(modes[name], f"run_modes.{name}", minimum=1)
        if count != expected_count:
            raise NativeEdgeSupportAuditIdentityError(f"run_modes.{name} must be exactly {expected_count}, observed {count!r}")
    _require_exact_string(modes["mechanics20_schema_name"], "run_modes.mechanics20_schema_name")
    if modes["checkpoint_schema_name"] != CHECKPOINT_SCHEMA_NAME:
        raise NativeEdgeSupportAuditIdentityError("run_modes.checkpoint_schema_name must match the implemented checkpoint schema")
    if modes["per_image_stats_manifest_schema_name"] != PER_IMAGE_STATS_SCHEMA_NAME:
        raise NativeEdgeSupportAuditIdentityError("run_modes.per_image_stats_manifest_schema_name must match the implemented schema")

    execution = identity["execution_contract"]
    for name in (
        "sample_pulls_equal_images", "backbone_snapshot_calls_equal_windows", "graph_builds_equal_windows",
        "propagation_calls_at_most_windows", "cross_view_comparisons_never_invoke_model_or_graph_builder",
    ):
        if _require_exact_bool(execution[name], f"execution_contract.{name}") is not True:
            raise NativeEdgeSupportAuditIdentityError(f"execution_contract.{name} must be true")
    if execution["checkpoint_granularity"] != "one_complete_image":
        raise NativeEdgeSupportAuditIdentityError("execution_contract.checkpoint_granularity must be 'one_complete_image'")

    _require_exact_string_list(identity["prohibited"]["list"], "prohibited.list")

    return identity


def _check_git_ancestry(root: Path, commit: str, *, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"{label} is not an ancestor of HEAD"
        raise NativeEdgeSupportAuditIdentityError(f"Git ancestry check failed for {label} ({commit}): {detail}")


def _validate_parent_identity(root: Path, identity: Mapping[str, Any]) -> Mapping[str, Any]:
    from src.stitching_control_identity import load_identity as _load_stitching_control_identity

    parent = identity["parent_identity"]
    stitching_control_path = root / parent["stitching_control_identity_path"]
    try:
        stitching_control_bytes = stitching_control_path.read_bytes()
    except OSError as error:
        raise NativeEdgeSupportAuditIdentityError(
            f"cannot read stitching-control parent identity {stitching_control_path}: {error}"
        ) from error
    if hashlib.sha256(stitching_control_bytes).hexdigest() != parent["stitching_control_identity_sha256"]:
        raise NativeEdgeSupportAuditIdentityError("stitching-control parent identity file SHA256 mismatch")
    stitching_control_identity = _load_stitching_control_identity(stitching_control_path, repo_root=root)
    if stitching_control_identity["identity"]["name"] != parent["stitching_control_identity_name"]:
        raise NativeEdgeSupportAuditIdentityError("stitching-control parent identity name mismatch")

    sc_parent = stitching_control_identity["parent_identity"]
    power_path = root / parent["power_evaluation_identity_path"]
    if power_path != root / sc_parent["power_evaluation_identity_path"]:
        raise NativeEdgeSupportAuditIdentityError("power_evaluation_identity_path disagrees with the stitching-control identity's own parent")
    if parent["power_evaluation_identity_sha256"] != sc_parent["power_evaluation_identity_sha256"]:
        raise NativeEdgeSupportAuditIdentityError("power_evaluation_identity_sha256 disagrees with the stitching-control identity's own parent")

    matched_path = root / parent["matched_identity_path"]
    if matched_path != root / sc_parent["matched_identity_path"]:
        raise NativeEdgeSupportAuditIdentityError("matched_identity_path disagrees with the stitching-control identity's own parent")
    if parent["matched_identity_sha256"] != sc_parent["matched_identity_sha256"]:
        raise NativeEdgeSupportAuditIdentityError("matched_identity_sha256 disagrees with the stitching-control identity's own parent")

    e3_path = root / parent["e3_identity_path"]
    if e3_path != root / sc_parent["e3_identity_path"]:
        raise NativeEdgeSupportAuditIdentityError("e3_identity_path disagrees with the stitching-control identity's own parent")
    if parent["e3_identity_sha256"] != sc_parent["e3_identity_sha256"]:
        raise NativeEdgeSupportAuditIdentityError("e3_identity_sha256 disagrees with the stitching-control identity's own parent")

    return stitching_control_identity


def validate_static_configuration(
    *, repo_root: Path | None = None, identity_path: Path | None = None, check_git: bool = True
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    if check_git:
        _check_git_ancestry(root, identity["identity"]["required_ancestor_commit"], label="identity.required_ancestor_commit")
    stitching_control_identity = _validate_parent_identity(root, identity)

    from src.stitching_control_identity import validate_static_configuration as _stitching_control_preflight

    stitching_control_result = _stitching_control_preflight(repo_root=root, check_git=check_git)

    return {
        "identity_name": identity["identity"]["name"],
        "stitching_control_identity": stitching_control_identity["identity"]["name"],
        "power_evaluation_identity": stitching_control_result["power_evaluation_identity"],
        "matched_identity": stitching_control_result["matched_identity"],
        "required_ancestor_commit": identity["identity"]["required_ancestor_commit"],
        "mechanics20_image_count": identity["run_modes"]["mechanics20_image_count"],
    }


__all__ = [
    "CHECKPOINT_SCHEMA_NAME",
    "IDENTITY_RELATIVE_PATH",
    "NativeEdgeSupportAuditIdentityError",
    "PER_IMAGE_STATS_SCHEMA_NAME",
    "RUN_MODES",
    "RUN_MODE_IMAGE_COUNT_KEYS",
    "RUN_MODE_SCHEMA_KEYS",
    "SUPPORTED_ALPHA",
    "SUPPORTED_CLASS_COUNT",
    "SUPPORTED_CROP",
    "SUPPORTED_GRID_SIZE",
    "SUPPORTED_IMAGE_COUNT",
    "SUPPORTED_K",
    "SUPPORTED_PATCH_SIZE",
    "SUPPORTED_RANKING_CRITERIA",
    "SUPPORTED_STEPS",
    "SUPPORTED_STRIDE",
    "SUPPORTED_UNDEFINED_REASON_ENUM",
    "load_identity",
    "repository_root",
    "validate_static_configuration",
]
