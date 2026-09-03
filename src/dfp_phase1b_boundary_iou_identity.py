"""Strict loading/validation for the DFP Phase 1B Boundary IoU successor
authority (`evaluation_identities/e14_dfp_phase1b_boundary_iou.toml`).

This module NEVER rewrites, re-binds, or cascades new hashes through the
completed Phase 1A protocol identity
(`evaluation_identities/e13_dfp_phase1a_protocol.toml`) or any Phase 1A
artifact schema, manifest, or result. It only relationally re-verifies
the parent protocol's SHA-256 (proving it is byte-identical to what this
identity pins) every time it loads.

A SEPARATE, pre-existing additive supplement
(`evaluation_identities/e13_dfp_phase1a_boundary_iou.toml`, loaded by
`src.dfp_phase1a_boundary_iou_identity`) already resolves the same parent
protocol's deferred `[boundary_iou].implementation_source_status` field
for its own (different, non-ignore-neutral) semantic contract. This
module does not read, import, or depend on that supplement in any way;
the two coexist by design (see the e14 identity's own module-level
docstring in the TOML for the full rationale). This loader is the SOLE
authority DFP Phase 1B code binds to.

Every scientifically material string/number is exact-validated against
an embedded canonical copy (`CANONICAL_SCIENTIFIC_STRINGS` /
literal-value checks below) -- never merely checked for non-emptiness or
type. Always fail closed: never silently substitute a default, coerce a
type, or normalize a string. Performs no torch, CUDA, model, dataset, or
image access -- pure TOML/filesystem/hash validation only.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import tomllib


class DfpPhase1bBoundaryIouIdentityError(ValueError):
    """Raised on any DFP Phase 1B Boundary IoU authority invariant
    violation. Always fail closed. Never raised for KeyboardInterrupt,
    SystemExit, or MemoryError -- those always propagate (this module
    never catches a bare Exception)."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e14_dfp_phase1b_boundary_iou.toml")
PARENT_PROTOCOL_RELATIVE_PATH = "evaluation_identities/e13_dfp_phase1a_protocol.toml"
REQUIRED_PARENT_DEFERRED_STATUS = "not_yet_bound"

IDENTITY_TOP_KEYS = frozenset(
    {
        "format_version", "identity", "parent_protocol_binding", "published_authority",
        "upstream_implementation_authority", "binary_boundary_definition", "semantic_adaptation",
        "evaluation_role",
    }
)

SECTION_KEYS: dict[str, frozenset[str]] = {
    "identity": frozenset({"name", "schema_name", "schema_version", "purpose", "storage_representation_only"}),
    "parent_protocol_binding": frozenset(
        {
            "parent_protocol_relative_path", "parent_protocol_sha256", "parent_protocol_untouched_by_this_identity",
            "parent_deferred_field", "parent_deferred_value_required", "retroactive_phase1a_claim_forbidden",
        }
    ),
    "published_authority": frozenset(
        {"paper_title", "paper_authors", "paper_venue", "paper_year", "paper_arxiv_id", "paper_canonical_url"}
    ),
    "upstream_implementation_authority": frozenset(
        {
            "official_repository_url", "official_repository_commit", "canonical_source_relative_path",
            "canonical_source_function_name", "upstream_source_sha256", "vendored_source_relative_path",
            "vendored_source_sha256", "upstream_license_relative_path", "upstream_license_sha256",
            "upstream_commit_pin_relative_path", "retrieval_date_utc_non_scientific_metadata",
        }
    ),
    "binary_boundary_definition": frozenset(
        {
            "dilation_ratio", "diagonal_formula", "dilation_formula", "rounding_convention", "minimum_dilation",
            "kernel_shape", "kernel_dtype", "kernel_value", "kernel_connectivity", "erosion_iterations",
            "padding_width", "padding_value", "image_edge_behavior", "inner_boundary_formula",
            "mask_dtype_requirement", "mask_value_requirement", "tunable_boundary_width_forbidden",
        }
    ),
    "semantic_adaptation": frozenset(
        {
            "label", "is_official_semantic_definition", "adaptation_statement", "valid_mask_formula",
            "class_gt_mask_formula", "class_pred_mask_formula", "ignore_semantics", "valid_domain_erosion_formula",
            "ignore_neutral_boundary_formula", "image_exterior_behavior", "all_valid_reduction_required",
            "boundary_intersection_formula", "boundary_union_formula", "gt_boundary_pixel_count_formula",
            "pred_boundary_pixel_count_formula", "per_image_sufficient_statistics", "dataset_aggregation_rule",
            "class_intersection_sum_formula", "class_union_sum_formula", "class_boundary_iou_formula",
            "class_inclusion_rule", "mean_formula", "undefined_aggregate_behavior", "all_ignore_image_behavior",
            "epsilon_in_denominator_forbidden", "metric_unit", "difference_unit",
        }
    ),
    "evaluation_role": frozenset(
        {
            "primary_phase1b_metric", "secondary_phase1b_metric", "gate1_boundary_trigger_source",
            "phase1a_retroactive_result_forbidden", "is_method_parameter", "is_training_signal",
            "threshold_fitting_forbidden", "phase1b_evaluation_still_requires_independent_verification",
        }
    ),
}

# Single production authority for every scientifically material string
# field's exact text (mirrors the pattern in src.dfp_phase1a_identity
# and src.dfp_phase1a_boundary_iou_identity).
CANONICAL_SCIENTIFIC_STRINGS: dict[tuple[str, str], str] = {
    ("format_version", "format_version"): "talk2dino-dfp-phase1b-boundary-iou-identity-v1",
    ("identity", "name"): "e14-dfp-phase1b-boundary-iou",
    ("identity", "schema_name"): "talk2dino-dfp-phase1b-boundary-iou-identity-v1",
    ("parent_protocol_binding", "parent_protocol_relative_path"): PARENT_PROTOCOL_RELATIVE_PATH,
    ("parent_protocol_binding", "parent_deferred_field"): "[boundary_iou].implementation_source_status",
    ("parent_protocol_binding", "parent_deferred_value_required"): REQUIRED_PARENT_DEFERRED_STATUS,
    ("published_authority", "paper_title"): "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation",
    ("published_authority", "paper_authors"): "Bowen Cheng, Ross Girshick, Piotr Dollar, Alexander C. Berg, Alexander Kirillov",
    ("published_authority", "paper_venue"): "CVPR",
    ("published_authority", "paper_arxiv_id"): "2103.16562",
    ("published_authority", "paper_canonical_url"): "https://arxiv.org/abs/2103.16562",
    ("upstream_implementation_authority", "official_repository_url"): "https://github.com/bowenc0221/boundary-iou-api",
    ("upstream_implementation_authority", "official_repository_commit"): "37d25586a677b043ed585f10e5c42d4e80176ea9",
    ("upstream_implementation_authority", "canonical_source_relative_path"): "boundary_iou/utils/boundary_utils.py",
    ("upstream_implementation_authority", "canonical_source_function_name"): "mask_to_boundary",
    ("upstream_implementation_authority", "upstream_source_sha256"): "a495cc15fdbded38681edef30b29c62014e81e864f0e68ca6062ce42db51fd9c",
    ("upstream_implementation_authority", "vendored_source_relative_path"): "third_party/boundary_iou_api/boundary_iou/utils/boundary_utils.py",
    ("upstream_implementation_authority", "vendored_source_sha256"): "a495cc15fdbded38681edef30b29c62014e81e864f0e68ca6062ce42db51fd9c",
    ("upstream_implementation_authority", "upstream_license_relative_path"): "third_party/boundary_iou_api/license.txt",
    ("upstream_implementation_authority", "upstream_license_sha256"): "6da6c74bb714b3f7fe7eb6b817e42dd59ebfca15f292f54a9a7ad33a6a872863",
    ("upstream_implementation_authority", "upstream_commit_pin_relative_path"): "third_party/boundary_iou_api/UPSTREAM_COMMIT",
    ("binary_boundary_definition", "diagonal_formula"): "image_diagonal = sqrt(height^2 + width^2)",
    ("binary_boundary_definition", "dilation_formula"): "dilation = int(round(dilation_ratio * image_diagonal))",
    ("binary_boundary_definition", "rounding_convention"): "python_builtin_round_half_to_even_then_int_cast",
    ("binary_boundary_definition", "kernel_shape"): "3x3",
    ("binary_boundary_definition", "kernel_dtype"): "uint8",
    ("binary_boundary_definition", "kernel_value"): "all_ones",
    ("binary_boundary_definition", "erosion_iterations"): "dilation",
    ("binary_boundary_definition", "image_edge_behavior"): (
        "a foreground mask truncated by the image border is treated as having a boundary at that edge, via zero "
        "padding before erosion"
    ),
    ("binary_boundary_definition", "inner_boundary_formula"): "boundary(mask) = mask - eroded(mask)",
    ("binary_boundary_definition", "mask_dtype_requirement"): "uint8",
    ("binary_boundary_definition", "mask_value_requirement"): "binary_0_or_1",
    ("semantic_adaptation", "label"): "DFP classwise semantic-segmentation adaptation of Cheng et al. Boundary IoU",
    ("semantic_adaptation", "adaptation_statement"): (
        "The official boundary-iou-api package provides only instance/panoptic integrations built on the "
        "mask_to_boundary primitive above; it does not supply a complete COCO-Stuff-style classwise semantic-mIoU "
        "aggregation contract with an ignore label. Every formula in this [semantic_adaptation] section and its "
        "subsections is a repository-defined (DFP) extension on top of the pinned official primitive, and must "
        "never be attributed to the upstream paper or repository."
    ),
    ("semantic_adaptation", "valid_mask_formula"): "V = (G != ignore_index)",
    ("semantic_adaptation", "class_gt_mask_formula"): "G_c = (G == c) AND V",
    ("semantic_adaptation", "class_pred_mask_formula"): "P_c = (P == c) AND V",
    ("semantic_adaptation", "ignore_semantics"): "ignore pixels are unknown, not class-negative",
    ("semantic_adaptation", "valid_domain_erosion_formula"): "E_V(M) = erosion(M OR NOT V)",
    ("semantic_adaptation", "ignore_neutral_boundary_formula"): "B_V(M) = M AND V AND NOT E_V(M)",
    ("semantic_adaptation", "image_exterior_behavior"): (
        "the image exterior remains zero-padded before erosion, so true image-edge boundaries are still counted; "
        "only INTERIOR invalid/ignore pixels are neutral for erosion"
    ),
    ("semantic_adaptation", "all_valid_reduction_required"): (
        "this construction must reduce exactly to the official boundary extraction B(M) = M - erosion(M) when V is "
        "all-true (no ignore pixels)"
    ),
    ("semantic_adaptation", "boundary_intersection_formula"): "boundary_intersection_c = count(B_V(G_c) AND B_V(P_c))",
    ("semantic_adaptation", "boundary_union_formula"): "boundary_union_c = count(B_V(G_c) OR B_V(P_c))",
    ("semantic_adaptation", "gt_boundary_pixel_count_formula"): "gt_boundary_pixels_c = count(B_V(G_c))",
    ("semantic_adaptation", "pred_boundary_pixel_count_formula"): "pred_boundary_pixels_c = count(B_V(P_c))",
    ("semantic_adaptation", "dataset_aggregation_rule"): "dataset_level_sum_then_divide_full_precision_integer_sufficient_statistics",
    ("semantic_adaptation", "class_intersection_sum_formula"): "I_c = sum over images of boundary_intersection_c",
    ("semantic_adaptation", "class_union_sum_formula"): "U_c = sum over images of boundary_union_c",
    ("semantic_adaptation", "class_boundary_iou_formula"): "BoundaryIoU_c = I_c / U_c",
    ("semantic_adaptation", "class_inclusion_rule"): "U_c > 0",
    ("semantic_adaptation", "mean_formula"): "mBoundaryIoU = mean over classes with U_c > 0 of BoundaryIoU_c",
    ("semantic_adaptation", "undefined_aggregate_behavior"): (
        "if no class has U_c > 0 across the complete evaluated dataset, the aggregate is rejected as undefined "
        "with a named error, never a fabricated value"
    ),
    ("semantic_adaptation", "all_ignore_image_behavior"): (
        "an all-ignore image contributes all-zero per-class sufficient statistics and does not independently fail "
        "a multi-image aggregate"
    ),
    ("semantic_adaptation", "metric_unit"): "percent_0_100",
    ("semantic_adaptation", "difference_unit"): "percentage_points",
    ("evaluation_role", "primary_phase1b_metric"): "mBoundaryIoU",
    ("evaluation_role", "secondary_phase1b_metric"): "mIoU preservation",
}

# String fields whose exactness is governed by a stronger, independent
# mechanism (a live SHA-256 file-content hash re-verified against a local
# file) rather than (only) a literal comparison against
# CANONICAL_SCIENTIFIC_STRINGS.
_RELATIONALLY_GOVERNED_HASH_FIELDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("parent_protocol_binding", "parent_protocol_sha256"),
        ("upstream_implementation_authority", "upstream_source_sha256"),
        ("upstream_implementation_authority", "vendored_source_sha256"),
        ("upstream_implementation_authority", "upstream_license_sha256"),
    }
)


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str:
        raise DfpPhase1bBoundaryIouIdentityError(f"{label} must be an exact string")
    if nonempty and not value.strip():
        raise DfpPhase1bBoundaryIouIdentityError(f"{label} must be a non-empty, non-whitespace string")
    return value


def _require_canonical_string(value: Any, section: str, field: str) -> str:
    label = f"{section}.{field}"
    token = _require_exact_string(value, label)
    canonical = CANONICAL_SCIENTIFIC_STRINGS[(section, field)]
    if token != canonical:
        raise DfpPhase1bBoundaryIouIdentityError(f"{label} disagrees with the frozen canonical scientific text")
    return token


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1bBoundaryIouIdentityError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise DfpPhase1bBoundaryIouIdentityError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise DfpPhase1bBoundaryIouIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_exact_float(value: Any, label: str) -> float:
    if type(value) is not float:
        raise DfpPhase1bBoundaryIouIdentityError(f"{label} must be an exact finite float")
    if not math.isfinite(value):
        raise DfpPhase1bBoundaryIouIdentityError(f"{label} must be finite (no NaN/Infinity)")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise DfpPhase1bBoundaryIouIdentityError(f"{label} must be a lowercase 64-hex-character SHA-256")
    return token


def _require_canonical_sha256(value: Any, section: str, field: str) -> str:
    label = f"{section}.{field}"
    token = _require_sha256(value, label)
    canonical = CANONICAL_SCIENTIFIC_STRINGS.get((section, field))
    if canonical is not None and token != canonical:
        raise DfpPhase1bBoundaryIouIdentityError(f"{label} disagrees with the frozen canonical hash")
    return token


def _require_git_commit(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 40 or any(c not in "0123456789abcdef" for c in token):
        raise DfpPhase1bBoundaryIouIdentityError(f"{label} must be a full lowercase 40-hex-character Git commit id")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token or "\x00" in token:
        raise DfpPhase1bBoundaryIouIdentityError(f"{label} must be a safe repository-relative path")
    for part in path.parts:
        if part in ("", "."):
            raise DfpPhase1bBoundaryIouIdentityError(f"{label} must not contain empty or '.' path components")
    return token


def _require_no_symlink_escape(root: Path, relative: str, label: str) -> Path:
    """Resolves `root / relative` and requires the resolved, symlink-
    followed path to remain strictly under the resolved `root` -- rejects
    both a symlinked component escaping the repository and (defensively)
    a path that resolves outside root for any other reason. Does not
    require the target to exist as a precondition for path-safety
    checking; existence is checked separately by the caller."""
    resolved_root = root.resolve()
    candidate = (root / relative)
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError:
        raise DfpPhase1bBoundaryIouIdentityError(f"{label} escapes the repository root after symlink resolution: {resolved_candidate}") from None
    return resolved_candidate


def _require_exact_int_list(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise DfpPhase1bBoundaryIouIdentityError(f"{label} must be a non-empty exact list")
    for item in value:
        if type(item) is not str:
            raise DfpPhase1bBoundaryIouIdentityError(f"{label} elements must be exact strings")
    return tuple(value)


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DfpPhase1bBoundaryIouIdentityError(f"{label} must be a mapping")
    if set(value) != expected_keys:
        missing = expected_keys - set(value)
        unknown = set(value) - expected_keys
        raise DfpPhase1bBoundaryIouIdentityError(f"{label} has an unexpected schema (missing={sorted(missing)}, unknown={sorted(unknown)})")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load and fully closed-schema/exact-type/exact-string validate the
    e14 identity TOML. Does NOT verify any live file hash on disk -- see
    `validate_static_configuration` for the full relational check."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DfpPhase1bBoundaryIouIdentityError(f"cannot load Boundary IoU successor identity {source}: {error}") from error

    if set(identity) != IDENTITY_TOP_KEYS:
        missing = IDENTITY_TOP_KEYS - set(identity)
        unknown = set(identity) - IDENTITY_TOP_KEYS
        raise DfpPhase1bBoundaryIouIdentityError(f"identity has an unexpected top-level schema (missing={sorted(missing)}, unknown={sorted(unknown)})")
    _require_canonical_string(identity["format_version"], "format_version", "format_version")
    for section, keys in SECTION_KEYS.items():
        _require_closed_mapping(identity.get(section), keys, f"identity.{section}")

    _validate_identity_block(identity["identity"])
    _validate_parent_protocol_binding_block(identity["parent_protocol_binding"])
    _validate_published_authority_block(identity["published_authority"])
    _validate_upstream_implementation_authority_block(identity["upstream_implementation_authority"])
    _validate_binary_boundary_definition_block(identity["binary_boundary_definition"])
    _validate_semantic_adaptation_block(identity["semantic_adaptation"])
    _validate_evaluation_role_block(identity["evaluation_role"])

    return identity


def _validate_identity_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["name"], "identity", "name")
    _require_canonical_string(block["schema_name"], "identity", "schema_name")
    if _require_exact_int(block["schema_version"], "identity.schema_version") != 1:
        raise DfpPhase1bBoundaryIouIdentityError("identity.schema_version must be exactly 1")
    _require_exact_string(block["purpose"], "identity.purpose")
    if _require_exact_bool(block["storage_representation_only"], "identity.storage_representation_only") is not True:
        raise DfpPhase1bBoundaryIouIdentityError("identity.storage_representation_only must be true")


def _validate_parent_protocol_binding_block(block: Mapping[str, Any]) -> None:
    path = _require_relative_path(block["parent_protocol_relative_path"], "parent_protocol_binding.parent_protocol_relative_path")
    if path != PARENT_PROTOCOL_RELATIVE_PATH:
        raise DfpPhase1bBoundaryIouIdentityError("parent_protocol_binding.parent_protocol_relative_path must be exactly the frozen Phase 1A protocol identity path")
    _require_sha256(block["parent_protocol_sha256"], "parent_protocol_binding.parent_protocol_sha256")
    if _require_exact_bool(block["parent_protocol_untouched_by_this_identity"], "parent_protocol_binding.parent_protocol_untouched_by_this_identity") is not True:
        raise DfpPhase1bBoundaryIouIdentityError("parent_protocol_binding.parent_protocol_untouched_by_this_identity must be true")
    _require_canonical_string(block["parent_deferred_field"], "parent_protocol_binding", "parent_deferred_field")
    _require_canonical_string(block["parent_deferred_value_required"], "parent_protocol_binding", "parent_deferred_value_required")
    if _require_exact_bool(block["retroactive_phase1a_claim_forbidden"], "parent_protocol_binding.retroactive_phase1a_claim_forbidden") is not True:
        raise DfpPhase1bBoundaryIouIdentityError("parent_protocol_binding.retroactive_phase1a_claim_forbidden must be true")


def _validate_published_authority_block(block: Mapping[str, Any]) -> None:
    for field in ("paper_title", "paper_authors", "paper_venue", "paper_arxiv_id", "paper_canonical_url"):
        _require_canonical_string(block[field], "published_authority", field)
    if _require_exact_int(block["paper_year"], "published_authority.paper_year") != 2021:
        raise DfpPhase1bBoundaryIouIdentityError("published_authority.paper_year must be exactly 2021")


def _validate_upstream_implementation_authority_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["official_repository_url"], "upstream_implementation_authority", "official_repository_url")
    commit = _require_git_commit(block["official_repository_commit"], "upstream_implementation_authority.official_repository_commit")
    canonical_commit = CANONICAL_SCIENTIFIC_STRINGS[("upstream_implementation_authority", "official_repository_commit")]
    if commit != canonical_commit:
        raise DfpPhase1bBoundaryIouIdentityError("upstream_implementation_authority.official_repository_commit disagrees with the frozen pinned commit")
    _require_canonical_string(block["canonical_source_relative_path"], "upstream_implementation_authority", "canonical_source_relative_path")
    _require_canonical_string(block["canonical_source_function_name"], "upstream_implementation_authority", "canonical_source_function_name")
    _require_canonical_sha256(block["upstream_source_sha256"], "upstream_implementation_authority", "upstream_source_sha256")
    vpath = _require_relative_path(block["vendored_source_relative_path"], "upstream_implementation_authority.vendored_source_relative_path")
    _require_canonical_string(vpath, "upstream_implementation_authority", "vendored_source_relative_path")
    _require_canonical_sha256(block["vendored_source_sha256"], "upstream_implementation_authority", "vendored_source_sha256")
    lpath = _require_relative_path(block["upstream_license_relative_path"], "upstream_implementation_authority.upstream_license_relative_path")
    _require_canonical_string(lpath, "upstream_implementation_authority", "upstream_license_relative_path")
    _require_canonical_sha256(block["upstream_license_sha256"], "upstream_implementation_authority", "upstream_license_sha256")
    cpath = _require_relative_path(block["upstream_commit_pin_relative_path"], "upstream_implementation_authority.upstream_commit_pin_relative_path")
    _require_canonical_string(cpath, "upstream_implementation_authority", "upstream_commit_pin_relative_path")
    _require_exact_string(block["retrieval_date_utc_non_scientific_metadata"], "upstream_implementation_authority.retrieval_date_utc_non_scientific_metadata")


def _validate_binary_boundary_definition_block(block: Mapping[str, Any]) -> None:
    if _require_exact_float(block["dilation_ratio"], "binary_boundary_definition.dilation_ratio") != 0.02:
        raise DfpPhase1bBoundaryIouIdentityError("binary_boundary_definition.dilation_ratio must be exactly 0.02")
    _require_canonical_string(block["diagonal_formula"], "binary_boundary_definition", "diagonal_formula")
    _require_canonical_string(block["dilation_formula"], "binary_boundary_definition", "dilation_formula")
    _require_canonical_string(block["rounding_convention"], "binary_boundary_definition", "rounding_convention")
    if _require_exact_int(block["minimum_dilation"], "binary_boundary_definition.minimum_dilation") != 1:
        raise DfpPhase1bBoundaryIouIdentityError("binary_boundary_definition.minimum_dilation must be exactly 1")
    _require_canonical_string(block["kernel_shape"], "binary_boundary_definition", "kernel_shape")
    _require_canonical_string(block["kernel_dtype"], "binary_boundary_definition", "kernel_dtype")
    _require_canonical_string(block["kernel_value"], "binary_boundary_definition", "kernel_value")
    if _require_exact_int(block["kernel_connectivity"], "binary_boundary_definition.kernel_connectivity") != 8:
        raise DfpPhase1bBoundaryIouIdentityError("binary_boundary_definition.kernel_connectivity must be exactly 8 (a full 3x3 all-ones kernel)")
    _require_canonical_string(block["erosion_iterations"], "binary_boundary_definition", "erosion_iterations")
    if _require_exact_int(block["padding_width"], "binary_boundary_definition.padding_width") != 1:
        raise DfpPhase1bBoundaryIouIdentityError("binary_boundary_definition.padding_width must be exactly 1")
    if _require_exact_int(block["padding_value"], "binary_boundary_definition.padding_value") != 0:
        raise DfpPhase1bBoundaryIouIdentityError("binary_boundary_definition.padding_value must be exactly 0")
    _require_canonical_string(block["image_edge_behavior"], "binary_boundary_definition", "image_edge_behavior")
    _require_canonical_string(block["inner_boundary_formula"], "binary_boundary_definition", "inner_boundary_formula")
    _require_canonical_string(block["mask_dtype_requirement"], "binary_boundary_definition", "mask_dtype_requirement")
    _require_canonical_string(block["mask_value_requirement"], "binary_boundary_definition", "mask_value_requirement")
    if _require_exact_bool(block["tunable_boundary_width_forbidden"], "binary_boundary_definition.tunable_boundary_width_forbidden") is not True:
        raise DfpPhase1bBoundaryIouIdentityError("binary_boundary_definition.tunable_boundary_width_forbidden must be true")


def _validate_semantic_adaptation_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["label"], "semantic_adaptation", "label")
    if _require_exact_bool(block["is_official_semantic_definition"], "semantic_adaptation.is_official_semantic_definition") is not False:
        raise DfpPhase1bBoundaryIouIdentityError("semantic_adaptation.is_official_semantic_definition must be false")
    for field in (
        "adaptation_statement", "valid_mask_formula", "class_gt_mask_formula", "class_pred_mask_formula",
        "ignore_semantics", "valid_domain_erosion_formula", "ignore_neutral_boundary_formula",
        "image_exterior_behavior", "all_valid_reduction_required", "boundary_intersection_formula",
        "boundary_union_formula", "gt_boundary_pixel_count_formula", "pred_boundary_pixel_count_formula",
        "dataset_aggregation_rule", "class_intersection_sum_formula", "class_union_sum_formula",
        "class_boundary_iou_formula", "class_inclusion_rule", "mean_formula", "undefined_aggregate_behavior",
        "all_ignore_image_behavior", "metric_unit", "difference_unit",
    ):
        _require_canonical_string(block[field], "semantic_adaptation", field)
    _require_exact_int_list(block["per_image_sufficient_statistics"], "semantic_adaptation.per_image_sufficient_statistics")
    expected_stats = ("boundary_intersection_c", "boundary_union_c", "gt_boundary_pixels_c", "pred_boundary_pixels_c")
    if tuple(block["per_image_sufficient_statistics"]) != expected_stats:
        raise DfpPhase1bBoundaryIouIdentityError(f"semantic_adaptation.per_image_sufficient_statistics must be exactly {list(expected_stats)}")
    if _require_exact_bool(block["epsilon_in_denominator_forbidden"], "semantic_adaptation.epsilon_in_denominator_forbidden") is not True:
        raise DfpPhase1bBoundaryIouIdentityError("semantic_adaptation.epsilon_in_denominator_forbidden must be true")


def _validate_evaluation_role_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["primary_phase1b_metric"], "evaluation_role", "primary_phase1b_metric")
    _require_canonical_string(block["secondary_phase1b_metric"], "evaluation_role", "secondary_phase1b_metric")
    if _require_exact_bool(block["gate1_boundary_trigger_source"], "evaluation_role.gate1_boundary_trigger_source") is not True:
        raise DfpPhase1bBoundaryIouIdentityError("evaluation_role.gate1_boundary_trigger_source must be true")
    if _require_exact_bool(block["phase1a_retroactive_result_forbidden"], "evaluation_role.phase1a_retroactive_result_forbidden") is not True:
        raise DfpPhase1bBoundaryIouIdentityError("evaluation_role.phase1a_retroactive_result_forbidden must be true")
    if _require_exact_bool(block["is_method_parameter"], "evaluation_role.is_method_parameter") is not False:
        raise DfpPhase1bBoundaryIouIdentityError("evaluation_role.is_method_parameter must be false")
    if _require_exact_bool(block["is_training_signal"], "evaluation_role.is_training_signal") is not False:
        raise DfpPhase1bBoundaryIouIdentityError("evaluation_role.is_training_signal must be false")
    if _require_exact_bool(block["threshold_fitting_forbidden"], "evaluation_role.threshold_fitting_forbidden") is not True:
        raise DfpPhase1bBoundaryIouIdentityError("evaluation_role.threshold_fitting_forbidden must be true")
    if _require_exact_bool(
        block["phase1b_evaluation_still_requires_independent_verification"],
        "evaluation_role.phase1b_evaluation_still_requires_independent_verification",
    ) is not True:
        raise DfpPhase1bBoundaryIouIdentityError("evaluation_role.phase1b_evaluation_still_requires_independent_verification must be true")


@dataclass(frozen=True)
class BoundaryIoUContract:
    """Independently verified, immutable facts from the e14 identity --
    never constructible except by `load_boundary_iou_contract`
    re-validating the identity AND every bound file hash from disk. The
    production metric module (`src.dfp_phase1b_boundary_iou`) requires
    one of these explicitly; there is no default and no silent
    fallback."""

    parent_protocol_sha256: str
    official_repository_commit: str
    upstream_source_sha256: str
    vendored_source_sha256: str
    upstream_license_sha256: str
    dilation_ratio: float
    minimum_dilation: int
    padding_width: int
    padding_value: int
    kernel_connectivity: int


def load_boundary_iou_contract(*, repo_root: Path | None = None, identity_path: Path | None = None) -> BoundaryIoUContract:
    """Loads AND fully relationally re-verifies the e14 identity against
    every live file it binds to: the parent Phase 1A protocol identity's
    SHA-256 (never re-read into the parent's own validated content --
    only its hash is checked, so this module never needs to import
    `src.dfp_phase1a_identity`'s full closed-schema validator), the
    vendored upstream source file's SHA-256, and the vendored upstream
    license file's SHA-256. Every referenced path is checked for
    traversal-safety and symlink escape before any file is opened. Never
    trusts an in-memory dict for any of this -- always re-reads from
    `repo_root`. Performs no torch, CUDA, model, dataset, or image
    access."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)

    parent_protocol_path = _require_no_symlink_escape(
        root, identity["parent_protocol_binding"]["parent_protocol_relative_path"], "parent_protocol_binding.parent_protocol_relative_path",
    )
    if not parent_protocol_path.is_file():
        raise DfpPhase1bBoundaryIouIdentityError(f"parent Phase 1A protocol identity does not exist: {parent_protocol_path}")
    observed_parent_sha256 = sha256_file(parent_protocol_path)
    declared_parent_sha256 = identity["parent_protocol_binding"]["parent_protocol_sha256"]
    if observed_parent_sha256 != declared_parent_sha256:
        raise DfpPhase1bBoundaryIouIdentityError(
            f"parent protocol identity {parent_protocol_path} SHA256 mismatch: e14 declares "
            f"{declared_parent_sha256}, observed {observed_parent_sha256} -- the completed Phase 1A protocol must "
            f"remain byte-identical; this identity refuses to bind against a changed parent"
        )

    # Confirm the parent protocol still declares the deferred status this
    # successor exists to eventually resolve, WITHOUT re-validating the
    # parent's entire closed schema here (that is
    # src.dfp_phase1a_identity.load_identity's sole responsibility, and is
    # exercised independently by the Phase 1A test suite; re-deriving it
    # here would risk silently drifting out of sync with that authority).
    import tomllib as _tomllib

    with parent_protocol_path.open("rb") as handle:
        parent_raw = _tomllib.load(handle)
    live_status = parent_raw.get("boundary_iou", {}).get("implementation_source_status")
    required_status = identity["parent_protocol_binding"]["parent_deferred_value_required"]
    if live_status != required_status:
        raise DfpPhase1bBoundaryIouIdentityError(
            f"live parent protocol's [boundary_iou].implementation_source_status is {live_status!r}, not "
            f"{required_status!r} -- refusing to bind against a parent whose deferred field no longer matches "
            f"what this successor identity expects"
        )

    vendored_source_path = _require_no_symlink_escape(
        root, identity["upstream_implementation_authority"]["vendored_source_relative_path"],
        "upstream_implementation_authority.vendored_source_relative_path",
    )
    if not vendored_source_path.is_file():
        raise DfpPhase1bBoundaryIouIdentityError(f"vendored upstream source does not exist: {vendored_source_path}")
    observed_vendored_sha256 = sha256_file(vendored_source_path)
    declared_vendored_sha256 = identity["upstream_implementation_authority"]["vendored_source_sha256"]
    declared_upstream_sha256 = identity["upstream_implementation_authority"]["upstream_source_sha256"]
    if observed_vendored_sha256 != declared_vendored_sha256:
        raise DfpPhase1bBoundaryIouIdentityError(
            f"vendored upstream source {vendored_source_path} SHA256 mismatch: identity declares "
            f"{declared_vendored_sha256}, observed {observed_vendored_sha256}"
        )
    if declared_vendored_sha256 != declared_upstream_sha256:
        raise DfpPhase1bBoundaryIouIdentityError(
            "upstream_implementation_authority.vendored_source_sha256 disagrees with "
            "upstream_implementation_authority.upstream_source_sha256 -- the vendored copy must be byte-identical "
            "to the pinned upstream file"
        )

    license_path = _require_no_symlink_escape(
        root, identity["upstream_implementation_authority"]["upstream_license_relative_path"],
        "upstream_implementation_authority.upstream_license_relative_path",
    )
    if not license_path.is_file():
        raise DfpPhase1bBoundaryIouIdentityError(f"vendored upstream license does not exist: {license_path}")
    observed_license_sha256 = sha256_file(license_path)
    declared_license_sha256 = identity["upstream_implementation_authority"]["upstream_license_sha256"]
    if observed_license_sha256 != declared_license_sha256:
        raise DfpPhase1bBoundaryIouIdentityError(
            f"vendored upstream license {license_path} SHA256 mismatch: identity declares "
            f"{declared_license_sha256}, observed {observed_license_sha256}"
        )

    commit_pin_path = _require_no_symlink_escape(
        root, identity["upstream_implementation_authority"]["upstream_commit_pin_relative_path"],
        "upstream_implementation_authority.upstream_commit_pin_relative_path",
    )
    if not commit_pin_path.is_file():
        raise DfpPhase1bBoundaryIouIdentityError(f"vendored upstream commit pin does not exist: {commit_pin_path}")
    observed_commit = commit_pin_path.read_text(encoding="utf-8").strip()
    declared_commit = identity["upstream_implementation_authority"]["official_repository_commit"]
    if observed_commit != declared_commit:
        raise DfpPhase1bBoundaryIouIdentityError(
            f"vendored UPSTREAM_COMMIT file {commit_pin_path} content {observed_commit!r} disagrees with the "
            f"identity's declared official_repository_commit {declared_commit!r}"
        )

    binary_def = identity["binary_boundary_definition"]
    return BoundaryIoUContract(
        parent_protocol_sha256=observed_parent_sha256,
        official_repository_commit=declared_commit,
        upstream_source_sha256=declared_upstream_sha256,
        vendored_source_sha256=observed_vendored_sha256,
        upstream_license_sha256=observed_license_sha256,
        dilation_ratio=binary_def["dilation_ratio"],
        minimum_dilation=binary_def["minimum_dilation"],
        padding_width=binary_def["padding_width"],
        padding_value=binary_def["padding_value"],
        kernel_connectivity=binary_def["kernel_connectivity"],
    )


__all__ = [
    "IDENTITY_RELATIVE_PATH",
    "PARENT_PROTOCOL_RELATIVE_PATH",
    "REQUIRED_PARENT_DEFERRED_STATUS",
    "IDENTITY_TOP_KEYS",
    "SECTION_KEYS",
    "CANONICAL_SCIENTIFIC_STRINGS",
    "DfpPhase1bBoundaryIouIdentityError",
    "BoundaryIoUContract",
    "repository_root",
    "sha256_file",
    "load_identity",
    "load_boundary_iou_contract",
]
