"""Deterministic, predeclared qualitative-example SELECTION for DFP Phase
1A Step 11.

This module is a closed schema/selection AUTHORITY, not an evaluator: it
defines no scientific quantity. Every dispute-category formula is reused
verbatim from the already-committed, already-frozen
``src/dfp_phase1a_analysis.py`` (Step 3) -- ``DisputeResult``,
``OracleMaps``, ``PersistenceMaps``, ``compute_reversion_coherence``, and
the pinned boundary primitive ``_official_mask_to_boundary`` -- never
redefined here.

Ground truth (GT) is used ONLY for post-hoc category labeling and panel
visualization within this module. Nothing in this module ever constructs
a prediction, touches the graph/recurrence/component/membership machinery,
or influences model input in any way -- see the purity tests in
``tests/test_dfp_phase1a_qualitative_report.py`` for an explicit poison
proof of this separation.

Selection is a pure function of already-verified per-image category
eligibility and the frozen subset manifest's own canonical image-ID
order: ``selection_key = SHA256(namespace + SEP + dataset_name + SEP +
category_name + SEP + canonical_image_id)``, sorted by
``(selection_key, canonical_image_id)``, first K taken. No RNG, no
Python ``hash()``, no wall-clock time, no manual ranking, no GT-derived
magnitude ever enters this formula.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.dfp_phase1a_artifact_schema import (
    DfpPhase1aArtifactSchemaError,
    parse_strict_json_document,
    require_safe_basename,
    require_safe_relative_path,
    sha256_file,
    write_json_atomically,
)


class DfpPhase1aQualitativeReportError(ValueError):
    """Raised on any qualitative-reporting identity, selection, input-
    contract, or manifest invariant violation. Always fail closed."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e13_dfp_phase1a_qualitative_report.toml")
PROTOCOL_IDENTITY_RELATIVE_PATH = "evaluation_identities/e13_dfp_phase1a_protocol.toml"

IDENTITY_TOP_KEYS = frozenset(
    {
        "format_version", "identity", "protocol_binding", "artifact_schema_binding", "boundary_iou_binding",
        "subset_manifest_binding", "category_binding", "selection_policy", "selection_schema", "bundle_schema",
        "panel_binding", "palette_binding", "rendering_policy", "future_step_bindings",
    }
)

SECTION_KEYS: dict[str, frozenset[str]] = {
    "identity": frozenset({"name", "schema_version", "description", "storage_representation_only"}),
    "protocol_binding": frozenset({"protocol_identity_relative_path", "protocol_identity_sha256"}),
    "artifact_schema_binding": frozenset({"artifact_schema_identity_relative_path", "artifact_schema_identity_sha256"}),
    "boundary_iou_binding": frozenset({"boundary_iou_identity_relative_path", "boundary_iou_identity_sha256", "boundary_interior_split_source"}),
    "subset_manifest_binding": frozenset(
        {"coco_stuff_manifest_relative_path", "coco_stuff_manifest_sha256", "ade20k_manifest_relative_path", "ade20k_manifest_sha256"}
    ),
    "category_binding": frozenset(
        {"category_order", "category_source", "eligibility_rule", "eligibility_pixel_count_threshold_forbidden", "tie_rule", "a_definition", "b_definition"}
    ),
    "selection_policy": frozenset(
        {
            "examples_per_category", "namespace", "separator", "encoding", "digest_algorithm", "selection_key_formula",
            "sort_key", "traversal_order", "allow_cross_category_duplicates", "shortage_policy", "zero_coverage_status",
        }
    ),
    "selection_schema": frozenset({"schema_name", "schema_version"}),
    "bundle_schema": frozenset({"schema_name", "schema_version", "overwrite_policy"}),
    "panel_binding": frozenset(
        {
            "panel_order", "panel_source", "rgb_interpolation", "label_interpolation", "antialiasing_for_labels_forbidden",
            "output_height", "output_width", "overlay_opacity", "caption_policy",
        }
    ),
    "palette_binding": frozenset(
        {
            "palette_source", "palette_formula", "palette_class_order", "ignore_color_rgb", "background_color_rgb",
            "category_overlay_colors",
        }
    ),
    "rendering_policy": frozenset(
        {
            "image_mode", "png_compression_level", "png_optimize", "strip_png_metadata", "strip_timestamps",
            "locale_independent_formatting", "numeric_caption_format", "decoded_digest_algorithm", "encoded_digest_algorithm",
        }
    ),
    "future_step_bindings": frozenset({"description", "excludes_in_this_step"}),
}

CATEGORY_ORDER: tuple[str, ...] = (
    "harmful_washout", "beneficial_correction", "third_label_transition",
    "disputed_boundary", "disputed_interior", "oracle_reverted",
    "horizon_persistence_reverting", "revert_set_fragmentation",
)
RUN_MODE_NAMES: tuple[str, ...] = ("coco_stuff_500", "ade20k_500")
SELECTION_SCHEMA_NAME = "talk2dino-dfp-phase1a-qualitative-selection-v1"
SELECTION_SCHEMA_VERSION = 1
SHORTAGE_STATUSES = ("complete", "shortage", "zero_coverage")

TOP_SELECTION_MANIFEST_KEYS = frozenset(
    {
        "schema", "schema_version", "reporting_identity_sha256", "protocol_identity_sha256", "artifact_schema_identity_sha256",
        "subset_manifest_sha256", "selected_ids_digest", "run_mode", "image_count_expected", "class_count", "class_names_digest",
        "category_order", "selection_namespace", "selection_algorithm", "requested_count_per_category", "categories",
        "shortage_status_by_category", "selection_trace_digest", "source_commit", "source_result_sha256",
    }
)

TOP_CATEGORY_RECORD_KEYS = frozenset(
    {"category_name", "requested_count", "eligible_count", "selected_count", "selected_ids", "selection_keys", "category_pixel_counts", "shortage_status"}
)


# ---------------------------------------------------------------------------
# Primitive validators.
# ---------------------------------------------------------------------------


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value.strip()):
        raise DfpPhase1aQualitativeReportError(f"{label} must be an exact non-empty string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1aQualitativeReportError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise DfpPhase1aQualitativeReportError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise DfpPhase1aQualitativeReportError(f"{label} must be at least {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise DfpPhase1aQualitativeReportError(f"{label} must be a lowercase 64-hex-character SHA-256")
    return token


def _require_string_list(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    if type(value) is not list or (nonempty and not value):
        raise DfpPhase1aQualitativeReportError(f"{label} must be an exact non-empty list of strings" if nonempty else f"{label} must be an exact list of strings")
    for item in value:
        if type(item) is not str:
            raise DfpPhase1aQualitativeReportError(f"{label} elements must be exact strings")
    return value


def _require_safe_relative_path(value: Any, label: str) -> str:
    """Delegates to the artifact-schema authority's own path validator,
    wrapping its `DfpPhase1aArtifactSchemaError` into this module's own
    `DfpPhase1aQualitativeReportError` -- never letting the dependency's
    exception type escape unwrapped (see Step 10's own repaired lesson
    on this exact pattern)."""
    try:
        return require_safe_relative_path(value, label)
    except DfpPhase1aArtifactSchemaError as error:
        raise DfpPhase1aQualitativeReportError(str(error)) from error


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        missing = expected_keys - set(value if isinstance(value, Mapping) else {})
        unknown = set(value if isinstance(value, Mapping) else {}) - expected_keys
        raise DfpPhase1aQualitativeReportError(f"{label} has an unexpected schema (missing={sorted(missing)}, unknown={sorted(unknown)})")
    return value


# ---------------------------------------------------------------------------
# Identity loading/validation.
# ---------------------------------------------------------------------------


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    import tomllib

    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DfpPhase1aQualitativeReportError(f"cannot load qualitative-report identity {source}: {error}") from error

    if set(identity) != IDENTITY_TOP_KEYS:
        missing = IDENTITY_TOP_KEYS - set(identity)
        unknown = set(identity) - IDENTITY_TOP_KEYS
        raise DfpPhase1aQualitativeReportError(f"identity has an unexpected top-level schema (missing={sorted(missing)}, unknown={sorted(unknown)})")
    for section, keys in SECTION_KEYS.items():
        block = identity.get(section)
        if not isinstance(block, Mapping) or set(block) != keys:
            missing = keys - set(block or {})
            unknown = set(block or {}) - keys
            raise DfpPhase1aQualitativeReportError(f"identity.{section} has an unexpected schema (missing={sorted(missing)}, unknown={sorted(unknown)})")

    _validate_identity_block(identity["identity"])
    _validate_protocol_binding_block(identity["protocol_binding"])
    _validate_artifact_schema_binding_block(identity["artifact_schema_binding"])
    _validate_boundary_iou_binding_block(identity["boundary_iou_binding"])
    _validate_subset_manifest_binding_block(identity["subset_manifest_binding"])
    _validate_category_binding_block(identity["category_binding"])
    _validate_selection_policy_block(identity["selection_policy"])
    _validate_selection_schema_block(identity["selection_schema"])
    _validate_bundle_schema_block(identity["bundle_schema"])
    _validate_panel_binding_block(identity["panel_binding"])
    _validate_palette_binding_block(identity["palette_binding"])
    _validate_rendering_policy_block(identity["rendering_policy"])
    return identity


def _validate_identity_block(block: Mapping[str, Any]) -> None:
    if _require_exact_string(block["name"], "identity.name") != "e13-dfp-phase1a-qualitative-report":
        raise DfpPhase1aQualitativeReportError("identity.name must be exactly e13-dfp-phase1a-qualitative-report")
    _require_exact_string(block["schema_version"], "identity.schema_version")
    _require_exact_string(block["description"], "identity.description")
    if _require_exact_bool(block["storage_representation_only"], "identity.storage_representation_only") is not True:
        raise DfpPhase1aQualitativeReportError("identity.storage_representation_only must be true")


def _validate_protocol_binding_block(block: Mapping[str, Any]) -> None:
    path = _require_safe_relative_path(block["protocol_identity_relative_path"], "protocol_binding.protocol_identity_relative_path")
    if path != PROTOCOL_IDENTITY_RELATIVE_PATH:
        raise DfpPhase1aQualitativeReportError("protocol_binding.protocol_identity_relative_path must be exactly the frozen protocol identity path")
    _require_sha256(block["protocol_identity_sha256"], "protocol_binding.protocol_identity_sha256")


def _validate_artifact_schema_binding_block(block: Mapping[str, Any]) -> None:
    _require_safe_relative_path(block["artifact_schema_identity_relative_path"], "artifact_schema_binding.artifact_schema_identity_relative_path")
    _require_sha256(block["artifact_schema_identity_sha256"], "artifact_schema_binding.artifact_schema_identity_sha256")


def _validate_boundary_iou_binding_block(block: Mapping[str, Any]) -> None:
    _require_safe_relative_path(block["boundary_iou_identity_relative_path"], "boundary_iou_binding.boundary_iou_identity_relative_path")
    _require_sha256(block["boundary_iou_identity_sha256"], "boundary_iou_binding.boundary_iou_identity_sha256")
    _require_exact_string(block["boundary_interior_split_source"], "boundary_iou_binding.boundary_interior_split_source")


def _validate_subset_manifest_binding_block(block: Mapping[str, Any]) -> None:
    _require_safe_relative_path(block["coco_stuff_manifest_relative_path"], "subset_manifest_binding.coco_stuff_manifest_relative_path")
    _require_sha256(block["coco_stuff_manifest_sha256"], "subset_manifest_binding.coco_stuff_manifest_sha256")
    _require_safe_relative_path(block["ade20k_manifest_relative_path"], "subset_manifest_binding.ade20k_manifest_relative_path")
    _require_sha256(block["ade20k_manifest_sha256"], "subset_manifest_binding.ade20k_manifest_sha256")


def _validate_category_binding_block(block: Mapping[str, Any]) -> None:
    categories = _require_string_list(block["category_order"], "category_binding.category_order", nonempty=True)
    if tuple(categories) != CATEGORY_ORDER:
        raise DfpPhase1aQualitativeReportError("category_binding.category_order disagrees with the authoritative Python CATEGORY_ORDER")
    _require_exact_string(block["category_source"], "category_binding.category_source")
    if _require_exact_string(block["eligibility_rule"], "category_binding.eligibility_rule") != "at_least_one_category_pixel_in_the_valid_region":
        raise DfpPhase1aQualitativeReportError("category_binding.eligibility_rule must be exactly 'at_least_one_category_pixel_in_the_valid_region'")
    if _require_exact_bool(block["eligibility_pixel_count_threshold_forbidden"], "category_binding.eligibility_pixel_count_threshold_forbidden") is not True:
        raise DfpPhase1aQualitativeReportError("category_binding.eligibility_pixel_count_threshold_forbidden must be true")
    if _require_exact_string(block["tie_rule"], "category_binding.tie_rule") != "exact_argmax_ties_select_lowest_class_index":
        raise DfpPhase1aQualitativeReportError("category_binding.tie_rule must match the frozen protocol tie rule exactly")
    _require_exact_string(block["a_definition"], "category_binding.a_definition")
    _require_exact_string(block["b_definition"], "category_binding.b_definition")


def _validate_selection_policy_block(block: Mapping[str, Any]) -> None:
    _require_exact_int(block["examples_per_category"], "selection_policy.examples_per_category", minimum=1)
    _require_exact_string(block["namespace"], "selection_policy.namespace")
    if type(block["separator"]) is not str:
        raise DfpPhase1aQualitativeReportError("selection_policy.separator must be an exact string")
    if _require_exact_string(block["encoding"], "selection_policy.encoding") != "utf-8":
        raise DfpPhase1aQualitativeReportError("selection_policy.encoding must be exactly 'utf-8'")
    if _require_exact_string(block["digest_algorithm"], "selection_policy.digest_algorithm") != "sha256":
        raise DfpPhase1aQualitativeReportError("selection_policy.digest_algorithm must be exactly 'sha256'")
    _require_exact_string(block["selection_key_formula"], "selection_policy.selection_key_formula")
    sort_key = _require_string_list(block["sort_key"], "selection_policy.sort_key", nonempty=True)
    if tuple(sort_key) != ("selection_key", "canonical_image_id"):
        raise DfpPhase1aQualitativeReportError("selection_policy.sort_key must be exactly ['selection_key', 'canonical_image_id']")
    if _require_exact_string(block["traversal_order"], "selection_policy.traversal_order") != "frozen_subset_manifest_canonical_selected_ids_order":
        raise DfpPhase1aQualitativeReportError("selection_policy.traversal_order must be exactly the frozen canonical order")
    _require_exact_bool(block["allow_cross_category_duplicates"], "selection_policy.allow_cross_category_duplicates")
    _require_exact_string(block["shortage_policy"], "selection_policy.shortage_policy")
    _require_exact_string(block["zero_coverage_status"], "selection_policy.zero_coverage_status")


def _validate_selection_schema_block(block: Mapping[str, Any]) -> None:
    if _require_exact_string(block["schema_name"], "selection_schema.schema_name") != SELECTION_SCHEMA_NAME:
        raise DfpPhase1aQualitativeReportError(f"selection_schema.schema_name must be exactly {SELECTION_SCHEMA_NAME!r}")
    if _require_exact_int(block["schema_version"], "selection_schema.schema_version", minimum=1) != SELECTION_SCHEMA_VERSION:
        raise DfpPhase1aQualitativeReportError(f"selection_schema.schema_version must be exactly {SELECTION_SCHEMA_VERSION}")


def _validate_bundle_schema_block(block: Mapping[str, Any]) -> None:
    _require_exact_string(block["schema_name"], "bundle_schema.schema_name")
    _require_exact_int(block["schema_version"], "bundle_schema.schema_version", minimum=1)
    _require_exact_string(block["overwrite_policy"], "bundle_schema.overwrite_policy")


def _validate_panel_binding_block(block: Mapping[str, Any]) -> None:
    panel_order = _require_string_list(block["panel_order"], "panel_binding.panel_order", nonempty=True)
    panel_source = _require_string_list(block["panel_source"], "panel_binding.panel_source", nonempty=True)
    if len(panel_order) != len(panel_source):
        raise DfpPhase1aQualitativeReportError("panel_binding.panel_order and panel_source must have the same length")
    if len(set(panel_order)) != len(panel_order):
        raise DfpPhase1aQualitativeReportError("panel_binding.panel_order must not contain a duplicate panel name")
    _require_exact_string(block["rgb_interpolation"], "panel_binding.rgb_interpolation")
    if _require_exact_string(block["label_interpolation"], "panel_binding.label_interpolation") != "nearest_neighbor":
        raise DfpPhase1aQualitativeReportError("panel_binding.label_interpolation must be exactly 'nearest_neighbor'")
    if _require_exact_bool(block["antialiasing_for_labels_forbidden"], "panel_binding.antialiasing_for_labels_forbidden") is not True:
        raise DfpPhase1aQualitativeReportError("panel_binding.antialiasing_for_labels_forbidden must be true")
    _require_exact_int(block["output_height"], "panel_binding.output_height", minimum=1)
    _require_exact_int(block["output_width"], "panel_binding.output_width", minimum=1)
    opacity = block["overlay_opacity"]
    if type(opacity) is not float or not (0.0 <= opacity <= 1.0):
        raise DfpPhase1aQualitativeReportError("panel_binding.overlay_opacity must be an exact float in [0, 1]")
    _require_exact_string(block["caption_policy"], "panel_binding.caption_policy")


def _validate_palette_binding_block(block: Mapping[str, Any]) -> None:
    _require_exact_string(block["palette_source"], "palette_binding.palette_source")
    _require_exact_string(block["palette_formula"], "palette_binding.palette_formula")
    if _require_exact_string(block["palette_class_order"], "palette_binding.palette_class_order") != "lowest_class_index_first":
        raise DfpPhase1aQualitativeReportError("palette_binding.palette_class_order must be exactly 'lowest_class_index_first'")
    for name in ("ignore_color_rgb", "background_color_rgb"):
        value = block[name]
        if type(value) is not list or len(value) != 3 or any(type(v) is not int or not (0 <= v <= 255) for v in value):
            raise DfpPhase1aQualitativeReportError(f"palette_binding.{name} must be an exact [R, G, B] list of ints in [0, 255]")
    overlay_colors = block["category_overlay_colors"]
    if not isinstance(overlay_colors, Mapping) or set(overlay_colors) != set(CATEGORY_ORDER):
        raise DfpPhase1aQualitativeReportError("palette_binding.category_overlay_colors must have exactly one entry per authoritative category")
    for name, value in overlay_colors.items():
        if type(value) is not list or len(value) != 3 or any(type(v) is not int or not (0 <= v <= 255) for v in value):
            raise DfpPhase1aQualitativeReportError(f"palette_binding.category_overlay_colors[{name!r}] must be an exact [R, G, B] list of ints in [0, 255]")


def _validate_rendering_policy_block(block: Mapping[str, Any]) -> None:
    if _require_exact_string(block["image_mode"], "rendering_policy.image_mode") != "RGB":
        raise DfpPhase1aQualitativeReportError("rendering_policy.image_mode must be exactly 'RGB'")
    _require_exact_int(block["png_compression_level"], "rendering_policy.png_compression_level", minimum=0)
    _require_exact_bool(block["png_optimize"], "rendering_policy.png_optimize")
    if _require_exact_bool(block["strip_png_metadata"], "rendering_policy.strip_png_metadata") is not True:
        raise DfpPhase1aQualitativeReportError("rendering_policy.strip_png_metadata must be true")
    if _require_exact_bool(block["strip_timestamps"], "rendering_policy.strip_timestamps") is not True:
        raise DfpPhase1aQualitativeReportError("rendering_policy.strip_timestamps must be true")
    if _require_exact_bool(block["locale_independent_formatting"], "rendering_policy.locale_independent_formatting") is not True:
        raise DfpPhase1aQualitativeReportError("rendering_policy.locale_independent_formatting must be true")
    _require_exact_string(block["numeric_caption_format"], "rendering_policy.numeric_caption_format")
    _require_exact_string(block["decoded_digest_algorithm"], "rendering_policy.decoded_digest_algorithm")
    _require_exact_string(block["encoded_digest_algorithm"], "rendering_policy.encoded_digest_algorithm")


def validate_static_configuration(*, repo_root: Path, identity_path: Path | None = None) -> dict[str, Any]:
    """Load the qualitative-report identity AND relationally verify every
    bound hash against the LIVE file content (protocol identity,
    Step 10 artifact-schema identity, the Boundary IoU authority
    supplement, and both subset manifests) -- never merely format-
    checked."""
    root = Path(repo_root)
    identity = load_identity(identity_path, repo_root=root)

    protocol_path = root / identity["protocol_binding"]["protocol_identity_relative_path"]
    observed_protocol_sha256 = sha256_file(protocol_path)
    if observed_protocol_sha256 != identity["protocol_binding"]["protocol_identity_sha256"]:
        raise DfpPhase1aQualitativeReportError(
            f"protocol identity {protocol_path} SHA256 mismatch: qualitative-report identity declares "
            f"{identity['protocol_binding']['protocol_identity_sha256']}, observed {observed_protocol_sha256}"
        )

    artifact_schema_path = root / identity["artifact_schema_binding"]["artifact_schema_identity_relative_path"]
    observed_artifact_schema_sha256 = sha256_file(artifact_schema_path)
    if observed_artifact_schema_sha256 != identity["artifact_schema_binding"]["artifact_schema_identity_sha256"]:
        raise DfpPhase1aQualitativeReportError("Step 10 artifact-schema identity SHA256 mismatch against the qualitative-report binding")

    boundary_path = root / identity["boundary_iou_binding"]["boundary_iou_identity_relative_path"]
    observed_boundary_sha256 = sha256_file(boundary_path)
    if observed_boundary_sha256 != identity["boundary_iou_binding"]["boundary_iou_identity_sha256"]:
        raise DfpPhase1aQualitativeReportError("Boundary IoU authority supplement SHA256 mismatch against the qualitative-report binding")

    coco_manifest_path = root / identity["subset_manifest_binding"]["coco_stuff_manifest_relative_path"]
    observed_coco_manifest_sha256 = sha256_file(coco_manifest_path)
    if observed_coco_manifest_sha256 != identity["subset_manifest_binding"]["coco_stuff_manifest_sha256"]:
        raise DfpPhase1aQualitativeReportError("coco_stuff subset manifest SHA256 mismatch against the qualitative-report binding")

    ade_manifest_path = root / identity["subset_manifest_binding"]["ade20k_manifest_relative_path"]
    observed_ade_manifest_sha256 = sha256_file(ade_manifest_path)
    if observed_ade_manifest_sha256 != identity["subset_manifest_binding"]["ade20k_manifest_sha256"]:
        raise DfpPhase1aQualitativeReportError("ade20k subset manifest SHA256 mismatch against the qualitative-report binding")

    from src.dfp_phase1a_identity import load_identity as load_protocol_identity

    protocol = load_protocol_identity(protocol_path, repo_root=root)
    live_dispute_categories = protocol["dispute_categories"]
    for formula_key, expected in (
        ("harmful_washout", "GT(x) == a(x) and GT(x) != b(x)"),
        ("beneficial_correction", "GT(x) == b(x) and GT(x) != a(x)"),
        ("third_label_transition", "GT(x) not in {a(x), b(x)}"),
    ):
        if live_dispute_categories[formula_key] != expected:
            raise DfpPhase1aQualitativeReportError(
                f"live protocol identity's [dispute_categories].{formula_key} disagrees with the formula this "
                "reporting identity's category set was bound against"
            )

    return {
        "identity_name": identity["identity"]["name"],
        "protocol_identity_sha256": observed_protocol_sha256,
        "artifact_schema_identity_sha256": observed_artifact_schema_sha256,
        "boundary_iou_identity_sha256": observed_boundary_sha256,
        "coco_stuff_manifest_sha256": observed_coco_manifest_sha256,
        "ade20k_manifest_sha256": observed_ade_manifest_sha256,
    }


# ---------------------------------------------------------------------------
# Deterministic selection.
# ---------------------------------------------------------------------------


def compute_selection_key(*, namespace: str, dataset_name: str, category_name: str, canonical_image_id: str, separator: str = "\x00") -> str:
    """``SHA256(namespace + SEP + dataset_name + SEP + category_name +
    SEP + canonical_image_id)``. Pure function of its four string
    arguments -- no RNG, no Python ``hash()``, no wall-clock time, no
    filesystem order."""
    for value, label in (
        (namespace, "namespace"), (dataset_name, "dataset_name"), (category_name, "category_name"),
        (canonical_image_id, "canonical_image_id"), (separator, "separator"),
    ):
        if type(value) is not str:
            raise DfpPhase1aQualitativeReportError(f"{label} must be an exact string")
    payload = separator.join((namespace, dataset_name, category_name, canonical_image_id)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CategorySelectionResult:
    category_name: str
    requested_count: int
    eligible_count: int
    selected_count: int
    selected_ids: tuple[str, ...]
    selection_keys: dict[str, str]
    shortage_status: str


def select_category_examples(
    *, category_name: str, dataset_name: str, canonical_ids_in_order: Sequence[str],
    eligibility_by_id: Mapping[str, bool], k: int, namespace: str, separator: str = "\x00",
) -> CategorySelectionResult:
    """Deterministically select up to ``k`` example image IDs for one
    category. Traverses ``canonical_ids_in_order`` (the frozen subset
    manifest's own order) to find eligible IDs, computes each one's
    selection key, sorts by ``(selection_key, canonical_image_id)``, and
    takes the first ``k``. On a shortage (fewer than ``k`` eligible),
    selects all eligible IDs -- never fills from another category, never
    duplicates, never weakens eligibility, never changes ``k``. Never
    mutates ``canonical_ids_in_order`` or ``eligibility_by_id``."""
    if type(k) is not int or isinstance(k, bool) or k < 1:
        raise DfpPhase1aQualitativeReportError("k must be an exact positive integer")
    if type(canonical_ids_in_order) not in (list, tuple):
        raise DfpPhase1aQualitativeReportError("canonical_ids_in_order must be a list or tuple")
    if len(set(canonical_ids_in_order)) != len(canonical_ids_in_order):
        raise DfpPhase1aQualitativeReportError("canonical_ids_in_order must not contain a duplicate ID")

    eligible = [cid for cid in canonical_ids_in_order if eligibility_by_id.get(cid, False) is True]
    keys = {
        cid: compute_selection_key(namespace=namespace, dataset_name=dataset_name, category_name=category_name, canonical_image_id=cid, separator=separator)
        for cid in eligible
    }
    ordered = sorted(eligible, key=lambda cid: (keys[cid], cid))
    selected = tuple(ordered[:k])

    if len(eligible) == 0:
        status = "zero_coverage"
    elif len(eligible) < k:
        status = "shortage"
    else:
        status = "complete"

    return CategorySelectionResult(
        category_name=category_name, requested_count=k, eligible_count=len(eligible), selected_count=len(selected),
        selected_ids=selected, selection_keys={cid: keys[cid] for cid in selected}, shortage_status=status,
    )


def selection_trace_digest(category_results: Sequence[CategorySelectionResult]) -> str:
    """A single SHA-256 digest binding the complete, ordered selection
    outcome (category name, requested/eligible/selected counts, selected
    IDs in selected order, and every selection key) -- sufficient to
    detect any change to the selection without re-running it."""
    pieces: list[str] = []
    for result in category_results:
        pieces.append(result.category_name)
        pieces.append(str(result.requested_count))
        pieces.append(str(result.eligible_count))
        pieces.append(str(result.selected_count))
        pieces.append(result.shortage_status)
        for cid in result.selected_ids:
            pieces.append(cid)
            pieces.append(result.selection_keys[cid])
    payload = "\0".join(pieces).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Category eligibility -- reuses Step 3's analysis outputs, never
# redefines a category formula.
# ---------------------------------------------------------------------------


def compute_category_pixel_counts(*, dispute, boundary_mask: np.ndarray, persistence_reverts: np.ndarray, reversion_component_count: int) -> dict[str, int]:
    """Pure aggregation of already-computed Step 3 category masks into
    per-category pixel counts for ONE image. `dispute` is a Step 3
    `DisputeResult`; `boundary_mask` is the dispute region's own boundary
    split (see `_official_mask_to_boundary`, reused unmodified);
    `persistence_reverts` is the primary-horizon reversion boolean mask.
    Computes no new category logic -- only counts pixels in masks the
    caller already derived from Step 3 primitives."""
    dispute_mask = dispute.dispute
    disputed_boundary = dispute_mask & boundary_mask
    disputed_interior = dispute_mask & (~boundary_mask)
    return {
        "harmful_washout": int(dispute.harmful.sum()),
        "beneficial_correction": int(dispute.beneficial.sum()),
        "third_label_transition": int(dispute.third_label.sum()),
        "disputed_boundary": int(disputed_boundary.sum()),
        "disputed_interior": int(disputed_interior.sum()),
        "oracle_reverted": int(dispute.harmful.sum()),
        "horizon_persistence_reverting": int(persistence_reverts.sum()),
        "revert_set_fragmentation": int(persistence_reverts.sum()) if reversion_component_count >= 2 else 0,
    }


def category_eligibility_from_pixel_counts(pixel_counts: Mapping[str, int]) -> dict[str, bool]:
    """`at_least_one_category_pixel_in_the_valid_region` -- exact, no
    threshold. Never mutates `pixel_counts`."""
    return {name: (pixel_counts.get(name, 0) > 0) for name in CATEGORY_ORDER}


# ---------------------------------------------------------------------------
# Strict selection manifest.
# ---------------------------------------------------------------------------


def build_selection_manifest(
    *, reporting_identity_sha256: str, protocol_identity_sha256: str, artifact_schema_identity_sha256: str,
    subset_manifest_sha256: str, selected_ids_digest: str, run_mode: str, image_count_expected: int, class_count: int,
    class_names_digest: str, requested_count_per_category: int, category_results: Sequence[CategorySelectionResult],
    category_pixel_counts_by_id: Mapping[str, Mapping[str, int]], source_commit: str, source_result_sha256: str,
) -> dict[str, Any]:
    """Pure construction of a strict, schema-closed selection-manifest
    document from already-computed selection results. Performs no I/O
    and mutates none of its inputs."""
    if run_mode not in RUN_MODE_NAMES:
        raise DfpPhase1aQualitativeReportError(f"run_mode must be one of {list(RUN_MODE_NAMES)}")
    categories: dict[str, Any] = {}
    shortage_status_by_category: dict[str, str] = {}
    for result in category_results:
        categories[result.category_name] = {
            "category_name": result.category_name, "requested_count": result.requested_count,
            "eligible_count": result.eligible_count, "selected_count": result.selected_count,
            "selected_ids": list(result.selected_ids), "selection_keys": dict(result.selection_keys),
            "category_pixel_counts": {cid: dict(category_pixel_counts_by_id[cid]) for cid in result.selected_ids},
            "shortage_status": result.shortage_status,
        }
        shortage_status_by_category[result.category_name] = result.shortage_status

    return {
        "schema": SELECTION_SCHEMA_NAME, "schema_version": SELECTION_SCHEMA_VERSION,
        "reporting_identity_sha256": reporting_identity_sha256, "protocol_identity_sha256": protocol_identity_sha256,
        "artifact_schema_identity_sha256": artifact_schema_identity_sha256, "subset_manifest_sha256": subset_manifest_sha256,
        "selected_ids_digest": selected_ids_digest, "run_mode": run_mode, "image_count_expected": image_count_expected,
        "class_count": class_count, "class_names_digest": class_names_digest, "category_order": list(CATEGORY_ORDER),
        "selection_namespace": "talk2dino-dfp-phase1a-qualitative-report-v1", "selection_algorithm": "category_conditioned_sha256_hash_selection_v1",
        "requested_count_per_category": requested_count_per_category, "categories": categories,
        "shortage_status_by_category": shortage_status_by_category,
        "selection_trace_digest": selection_trace_digest(category_results),
        "source_commit": source_commit, "source_result_sha256": source_result_sha256,
    }


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 40 or any(c not in "0123456789abcdef" for c in token):
        raise DfpPhase1aQualitativeReportError(f"{label} must be a full lowercase 40-hex-character Git commit id")
    return token


def validate_selection_manifest_structure(manifest: Mapping[str, Any]) -> None:
    """Sole authority for the selection manifest's schema/type/binding
    checks, including RELATIONAL reconstruction of the deterministic
    selection outcome itself (never trusting a stored key/order/trace-
    digest at face value): every selection key is independently
    recomputed via `compute_selection_key` using only the manifest's own
    authority-bound fields (`selection_namespace`, `run_mode`, each
    category name, each selected ID) and compared against the stored
    value; `selected_ids` is independently re-sorted by
    `(recomputed_key, canonical_image_id)` and compared for exact list
    equality (order included); `selection_trace_digest` is independently
    recomputed from the manifest's own category records USING the
    recomputed keys (never the stored ones) and compared. This proves
    internal ordering/key/trace consistency; it does not by itself prove
    the selected IDs are the true first-K among the complete eligible
    population -- that remains the responsibility of a stronger source-
    binding verifier with access to the live eligible-ID population.
    Never mutates `manifest` or any of its nested structures."""
    _require_closed_mapping(manifest, TOP_SELECTION_MANIFEST_KEYS, "selection manifest")
    if _require_exact_string(manifest["schema"], "selection_manifest.schema") != SELECTION_SCHEMA_NAME:
        raise DfpPhase1aQualitativeReportError(f"selection_manifest.schema must be exactly {SELECTION_SCHEMA_NAME!r}")
    if _require_exact_int(manifest["schema_version"], "selection_manifest.schema_version", minimum=1) != SELECTION_SCHEMA_VERSION:
        raise DfpPhase1aQualitativeReportError(f"selection_manifest.schema_version must be exactly {SELECTION_SCHEMA_VERSION}")
    for field in ("reporting_identity_sha256", "protocol_identity_sha256", "artifact_schema_identity_sha256", "subset_manifest_sha256", "selected_ids_digest", "selection_trace_digest"):
        _require_sha256(manifest[field], f"selection_manifest.{field}")
    run_mode = _require_exact_string(manifest["run_mode"], "selection_manifest.run_mode")
    if run_mode not in RUN_MODE_NAMES:
        raise DfpPhase1aQualitativeReportError(f"selection_manifest.run_mode must be one of {list(RUN_MODE_NAMES)}")
    _require_exact_int(manifest["image_count_expected"], "selection_manifest.image_count_expected", minimum=0)
    _require_exact_int(manifest["class_count"], "selection_manifest.class_count", minimum=1)
    _require_sha256(manifest["class_names_digest"], "selection_manifest.class_names_digest")
    category_order = _require_string_list(manifest["category_order"], "selection_manifest.category_order", nonempty=True)
    if tuple(category_order) != CATEGORY_ORDER:
        raise DfpPhase1aQualitativeReportError("selection_manifest.category_order disagrees with the authoritative CATEGORY_ORDER")
    namespace = _require_exact_string(manifest["selection_namespace"], "selection_manifest.selection_namespace")
    if _require_exact_string(manifest["selection_algorithm"], "selection_manifest.selection_algorithm") != "category_conditioned_sha256_hash_selection_v1":
        raise DfpPhase1aQualitativeReportError("selection_manifest.selection_algorithm must be exactly 'category_conditioned_sha256_hash_selection_v1'")
    k = _require_exact_int(manifest["requested_count_per_category"], "selection_manifest.requested_count_per_category", minimum=1)
    _require_git_identity(manifest["source_commit"], "selection_manifest.source_commit")
    _require_sha256(manifest["source_result_sha256"], "selection_manifest.source_result_sha256")

    categories = manifest["categories"]
    if not isinstance(categories, Mapping) or set(categories) != set(CATEGORY_ORDER):
        raise DfpPhase1aQualitativeReportError("selection_manifest.categories must have exactly one entry per authoritative category")
    shortage_status_by_category = manifest["shortage_status_by_category"]
    if not isinstance(shortage_status_by_category, Mapping) or set(shortage_status_by_category) != set(CATEGORY_ORDER):
        raise DfpPhase1aQualitativeReportError("selection_manifest.shortage_status_by_category must have exactly one entry per authoritative category")

    reconstructed_results: dict[str, CategorySelectionResult] = {}
    for name, record in categories.items():
        _require_closed_mapping(record, TOP_CATEGORY_RECORD_KEYS, f"selection_manifest.categories[{name!r}]")
        if record["category_name"] != name:
            raise DfpPhase1aQualitativeReportError(f"selection_manifest.categories[{name!r}].category_name disagrees with its own key")
        requested = _require_exact_int(record["requested_count"], f"categories[{name!r}].requested_count", minimum=1)
        if requested != k:
            raise DfpPhase1aQualitativeReportError(f"categories[{name!r}].requested_count disagrees with requested_count_per_category")
        eligible_count = _require_exact_int(record["eligible_count"], f"categories[{name!r}].eligible_count", minimum=0)
        selected_count = _require_exact_int(record["selected_count"], f"categories[{name!r}].selected_count", minimum=0)
        selected_ids = _require_string_list(record["selected_ids"], f"categories[{name!r}].selected_ids")
        if len(selected_ids) != selected_count:
            raise DfpPhase1aQualitativeReportError(f"categories[{name!r}].selected_ids length disagrees with selected_count")
        if len(set(selected_ids)) != len(selected_ids):
            raise DfpPhase1aQualitativeReportError(f"categories[{name!r}].selected_ids contains a duplicate ID")
        if selected_count > eligible_count:
            raise DfpPhase1aQualitativeReportError(f"categories[{name!r}].selected_count exceeds eligible_count")
        if selected_count > requested:
            raise DfpPhase1aQualitativeReportError(f"categories[{name!r}].selected_count exceeds requested_count")

        shortage_status = record["shortage_status"]
        if shortage_status not in SHORTAGE_STATUSES:
            raise DfpPhase1aQualitativeReportError(f"categories[{name!r}].shortage_status must be one of {SHORTAGE_STATUSES}")
        if eligible_count == 0 and shortage_status != "zero_coverage":
            raise DfpPhase1aQualitativeReportError(f"categories[{name!r}]: eligible_count == 0 requires shortage_status == 'zero_coverage'")
        if 0 < eligible_count < requested and shortage_status != "shortage":
            raise DfpPhase1aQualitativeReportError(f"categories[{name!r}]: eligible_count < requested requires shortage_status == 'shortage'")
        if eligible_count >= requested and shortage_status != "complete":
            raise DfpPhase1aQualitativeReportError(f"categories[{name!r}]: eligible_count >= requested requires shortage_status == 'complete'")
        if shortage_status_by_category[name] != shortage_status:
            raise DfpPhase1aQualitativeReportError(f"shortage_status_by_category[{name!r}] disagrees with categories[{name!r}].shortage_status")

        selection_keys = record["selection_keys"]
        if not isinstance(selection_keys, Mapping) or set(selection_keys) != set(selected_ids):
            raise DfpPhase1aQualitativeReportError(f"categories[{name!r}].selection_keys must have exactly one entry per selected ID")

        # Relationally reconstruct every selection key from the manifest's
        # own authority-bound fields via the canonical production
        # function -- never trust the stored key at face value. Iteration
        # over `selection_keys.items()` (a Mapping) never depends on
        # container/JSON key order.
        recomputed_keys: dict[str, str] = {}
        for cid, key in selection_keys.items():
            observed_key = _require_sha256(key, f"categories[{name!r}].selection_keys[{cid!r}]")
            recomputed_key = compute_selection_key(namespace=namespace, dataset_name=run_mode, category_name=name, canonical_image_id=cid)
            if observed_key != recomputed_key:
                raise DfpPhase1aQualitativeReportError(
                    f"categories[{name!r}].selection_keys[{cid!r}] disagrees with the independently recomputed "
                    "selection key -- refusing to trust an unverified key"
                )
            recomputed_keys[cid] = recomputed_key

        # Relationally reconstruct the expected selected-ID ORDER from the
        # recomputed keys and require exact list equality (order included)
        # -- never merely the right set in some order.
        expected_order = sorted(selected_ids, key=lambda cid: (recomputed_keys[cid], cid))
        if selected_ids != expected_order:
            raise DfpPhase1aQualitativeReportError(
                f"categories[{name!r}].selected_ids is not ordered by (selection_key, canonical_image_id) using "
                "the independently recomputed keys"
            )

        pixel_counts = record["category_pixel_counts"]
        if not isinstance(pixel_counts, Mapping) or set(pixel_counts) != set(selected_ids):
            raise DfpPhase1aQualitativeReportError(f"categories[{name!r}].category_pixel_counts must have exactly one entry per selected ID")

        # Collected for the trace-digest reconstruction below, USING the
        # recomputed keys (never the stored ones) -- this also catches a
        # trace digest computed consistently from tampered stored keys.
        reconstructed_results[name] = CategorySelectionResult(
            category_name=name, requested_count=requested, eligible_count=eligible_count, selected_count=selected_count,
            selected_ids=tuple(selected_ids), selection_keys=recomputed_keys, shortage_status=shortage_status,
        )

    expected_trace_digest = selection_trace_digest([reconstructed_results[name] for name in CATEGORY_ORDER])
    if manifest["selection_trace_digest"] != expected_trace_digest:
        raise DfpPhase1aQualitativeReportError(
            "selection_manifest.selection_trace_digest disagrees with the independently recomputed trace digest "
            "(derived from the recomputed keys/order, not the stored ones)"
        )


__all__ = [
    "DfpPhase1aQualitativeReportError",
    "IDENTITY_RELATIVE_PATH",
    "IDENTITY_TOP_KEYS",
    "SECTION_KEYS",
    "CATEGORY_ORDER",
    "RUN_MODE_NAMES",
    "SELECTION_SCHEMA_NAME",
    "SELECTION_SCHEMA_VERSION",
    "SHORTAGE_STATUSES",
    "TOP_SELECTION_MANIFEST_KEYS",
    "TOP_CATEGORY_RECORD_KEYS",
    "load_identity",
    "validate_static_configuration",
    "compute_selection_key",
    "CategorySelectionResult",
    "select_category_examples",
    "selection_trace_digest",
    "compute_category_pixel_counts",
    "category_eligibility_from_pixel_counts",
    "build_selection_manifest",
    "validate_selection_manifest_structure",
]
