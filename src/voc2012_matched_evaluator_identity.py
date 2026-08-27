"""Static identity loading/validation for the shared VOC2012 V20/V21
matched E3-vs-k11-vs-k12 evaluator.

Never redeclares alpha/steps/affinity_power/maximum_rank/crop/stride --
those are read from the parent matched-k11/k12 identity at runtime.
Never redeclares the VOC2012 dataset-source contract (split, image
count, class digests, label-transform description) -- those are read
from the parent VOC2012 source identity and cross-checked against the
real manifest. This module intentionally uses direct SHA256 file-hash
pinning (not the heavier typed-configuration resolution machinery) for
the plain YAML/Python leaf config files it binds against -- see
[model_and_checkpoint.provenance] in the TOML for the rationale.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e12_voc2012_matched_evaluator.toml")

SUPPORTED_SPLIT = "val"
SUPPORTED_EXPECTED_IMAGE_COUNT = 1449
SUPPORTED_V20_CLASS_COUNT = 20
SUPPORTED_V21_CLASS_COUNT = 21
SUPPORTED_BACKGROUND_CLASS_INDEX = 0
SUPPORTED_IGNORE_INDEX = 255
SUPPORTED_BG_MECHANISM = "canonical_constant_threshold_channel"
SUPPORTED_BG_STRATEGY = "base"
SUPPORTED_RUN_MODE_IMAGE_COUNTS = {"pilot20": 20, "pilot100": 100, "full": 1449}
SUPPORTED_MODEL_CONSTRUCTOR_CLASS = "DINOText"
SUPPORTED_VARIANT_ORDER = ("v20_e3", "v20_k11", "v20_k12", "v21_e3", "v21_k11", "v21_k12")
SUPPORTED_DELTA_FIELDS = (
    "delta_mIoU_v20_k11_minus_k12_percentage_points", "delta_mIoU_v20_k11_minus_e3_percentage_points",
    "delta_mIoU_v20_k12_minus_e3_percentage_points", "delta_mIoU_v21_k11_minus_k12_percentage_points",
    "delta_mIoU_v21_k11_minus_e3_percentage_points", "delta_mIoU_v21_k12_minus_e3_percentage_points",
)

IDENTITY_TOP_KEYS = frozenset(
    {
        "format_version", "identity", "parent_identities", "protocol", "v20_protocol", "v21_protocol",
        "shared_scores", "model_and_checkpoint", "reused_evaluator_modules", "background_protocol",
        "graph_and_propagation", "execution", "run_modes", "metrics", "checkpoint", "artifacts",
        "verification", "prohibited",
    }
)
IDENTITY_SECTION_KEYS = {
    "identity": frozenset({"name", "schema_version", "description", "required_ancestor_commit"}),
    "parent_identities": frozenset(
        {
            "voc2012_source_identity_path", "voc2012_source_identity_name", "voc2012_source_identity_sha256",
            "matched_identity_path", "matched_identity_name", "matched_identity_sha256", "required_relationship",
        }
    ),
    "protocol": frozenset({"label", "statement", "split", "expected_image_count"}),
    "v20_protocol": frozenset(
        {
            "dataset_type", "dataset_class_relative_path", "dataset_config_relative_path", "class_count",
            "background_included", "ignore_index", "class_names_source",
        }
    ),
    "v21_protocol": frozenset(
        {
            "dataset_type", "dataset_class_relative_path", "dataset_config_relative_path", "class_count",
            "background_included", "background_class_index", "ignore_index", "class_names_source",
        }
    ),
    "shared_scores": frozenset(
        {"statement", "text_query_source", "foreground_score_sharing", "second_backbone_pass_permitted"}
    ),
    "reused_evaluator_modules": frozenset(
        {
            "description", "coco_object_evaluator_relative_path", "coco_object_evaluator_sha256",
            "matched_power_evaluator_relative_path", "matched_power_evaluator_sha256",
            "graph_relative_path", "graph_sha256", "finite_step_regime_relative_path", "finite_step_regime_sha256",
            "sliding_window_geometry_relative_path", "sliding_window_geometry_sha256",
        }
    ),
    "background_protocol": frozenset(
        {
            "mechanism", "bg_thresh", "bg_strategy", "background_class_index", "competes_in_argmax", "formula",
            "formula_source", "reused_implementation", "injection_stage", "provenance_note", "v20_applies",
            "v21_applies", "ignore_pixels_excluded_from_argmax",
        }
    ),
    "graph_and_propagation": frozenset(
        {
            "statement", "maximum_rank_source", "alpha_source", "steps_source", "affinity_power_source",
            "crop_source", "stride_source", "window_enumeration_source",
        }
    ),
    "execution": frozenset(
        {
            "shared_backbone_pass", "shared_scores", "shared_features", "shared_affinity",
            "shared_top12_selection", "snapshot_calls_per_window", "topk_graph_calls_per_window",
            "k11_prefix_calls_per_window", "graph_normalizations_per_window", "k11_propagations_per_window",
            "k12_propagations_per_window", "k11_updates_per_window", "k12_updates_per_window",
            "e3_propagations_per_window", "second_model_pass_for_v20_v21", "variant_order",
        }
    ),
    "run_modes": frozenset(
        {
            "pilot20_image_count", "pilot100_image_count", "full_image_count", "pilot20_schema_name",
            "pilot100_schema_name", "full_schema_name", "checkpoint_schema_name", "image_order_source",
            "pilot_prefix_policy",
        }
    ),
    "metrics": frozenset(
        {
            "metric_names", "unit", "delta_unit", "precision_source", "sufficient_statistic_schema",
            "rounded_prettytable_forbidden", "minimum_metric_decimal_places", "delta_fields",
        }
    ),
    "checkpoint": frozenset(
        {"schema_name", "granularity", "serializes_gpu_tensors", "atomic_write_pattern", "corruption_detected_before_model_load"}
    ),
    "artifacts": frozenset({"per_image_stats_manifest_schema_name", "per_image_stats_array_format"}),
    "verification": frozenset({"require_source_manifest_verified_before_cuda", "require_checkpoint_bytes_verified_before_cuda"}),
    "prohibited": frozenset({"list"}),
}
MODEL_AND_CHECKPOINT_KEYS = frozenset(
    {
        "model_constructor_relative_path", "constructor_class", "backbone_name", "clip_model_name",
        "projection_class", "projection_config_relative_path", "projection_checkpoint_relative_path",
        "projection_checkpoint_sha256", "checkpoint_frozen", "target_dataset_training_or_feature_pre_extraction_permitted",
        "template", "pamr", "v20_eval_config", "v21_eval_config", "provenance",
    }
)
EVAL_CONFIG_KEYS = frozenset(
    {"eval_config_relative_path", "eval_config_sha256", "eval_base_config_relative_path", "eval_base_config_sha256"}
)
PROVENANCE_KEYS = frozenset(
    {
        "description", "projection_config_sha256", "model_constructor_sha256",
        "dataset_seg_inference_relative_path", "dataset_seg_inference_sha256",
        "dinotext_builder_relative_path", "dinotext_builder_sha256",
    }
)


class Voc2012MatchedEvaluatorIdentityError(ValueError):
    """Raised when the VOC2012 matched-evaluator identity, a checkpoint,
    or a result fails any exact-type/schema/provenance/binding check.
    Always fail closed: never silently substitute a default graph,
    propagation, background, or dataset-binding parameter."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value.strip()):
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be an exact non-empty string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_exact_float(value: Any, label: str) -> float:
    if type(value) is not float:
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be an exact float")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", token) is None:
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be a full Git identity")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token:
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be a safe repository-relative path")
    return token


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} has an unexpected schema")
    return value


def _require_exact_string_list(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be a non-empty exact list")
    if any(type(item) is not str for item in value):
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} elements must be exact strings")
    return tuple(value)


def _load_eval_config_block(value: Any, label: str) -> Mapping[str, Any]:
    block = _require_closed_mapping(value, EVAL_CONFIG_KEYS, label)
    _require_relative_path(block["eval_config_relative_path"], f"{label}.eval_config_relative_path")
    _require_sha256(block["eval_config_sha256"], f"{label}.eval_config_sha256")
    _require_relative_path(block["eval_base_config_relative_path"], f"{label}.eval_base_config_relative_path")
    _require_sha256(block["eval_base_config_sha256"], f"{label}.eval_base_config_sha256")
    return block


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise Voc2012MatchedEvaluatorIdentityError(f"cannot load voc2012 matched-evaluator identity {source}: {error}") from error

    if set(identity) != IDENTITY_TOP_KEYS:
        raise Voc2012MatchedEvaluatorIdentityError("voc2012 matched-evaluator identity has an unexpected top-level schema")
    _require_exact_string(identity["format_version"], "format_version")
    if identity["format_version"] != "talk2dino-voc2012-matched-evaluator-identity-v1":
        raise Voc2012MatchedEvaluatorIdentityError("unsupported voc2012 matched-evaluator identity format_version")
    for section, keys in IDENTITY_SECTION_KEYS.items():
        if section == "model_and_checkpoint":
            continue
        _require_closed_mapping(identity.get(section), keys, f"identity.{section}")

    block = identity["identity"]
    _require_exact_string(block["name"], "identity.name")
    _require_exact_string(block["schema_version"], "identity.schema_version")
    if block["schema_version"] != identity["format_version"]:
        raise Voc2012MatchedEvaluatorIdentityError("identity.schema_version disagrees with format_version")
    _require_exact_string(block["description"], "identity.description")
    _require_git_identity(block["required_ancestor_commit"], "identity.required_ancestor_commit")

    parents = identity["parent_identities"]
    _require_relative_path(parents["voc2012_source_identity_path"], "parent_identities.voc2012_source_identity_path")
    _require_exact_string(parents["voc2012_source_identity_name"], "parent_identities.voc2012_source_identity_name")
    _require_sha256(parents["voc2012_source_identity_sha256"], "parent_identities.voc2012_source_identity_sha256")
    _require_relative_path(parents["matched_identity_path"], "parent_identities.matched_identity_path")
    _require_exact_string(parents["matched_identity_name"], "parent_identities.matched_identity_name")
    _require_sha256(parents["matched_identity_sha256"], "parent_identities.matched_identity_sha256")
    _require_exact_string(parents["required_relationship"], "parent_identities.required_relationship")

    protocol = identity["protocol"]
    _require_exact_string(protocol["label"], "protocol.label")
    _require_exact_string(protocol["statement"], "protocol.statement")
    if protocol["split"] != SUPPORTED_SPLIT:
        raise Voc2012MatchedEvaluatorIdentityError(f"protocol.split must be exactly {SUPPORTED_SPLIT!r}")
    if _require_exact_int(protocol["expected_image_count"], "protocol.expected_image_count") != SUPPORTED_EXPECTED_IMAGE_COUNT:
        raise Voc2012MatchedEvaluatorIdentityError(f"protocol.expected_image_count must be exactly {SUPPORTED_EXPECTED_IMAGE_COUNT}")

    v20 = identity["v20_protocol"]
    if v20["dataset_type"] != "PascalVOCDataset20":
        raise Voc2012MatchedEvaluatorIdentityError("v20_protocol.dataset_type must be exactly 'PascalVOCDataset20'")
    _require_relative_path(v20["dataset_class_relative_path"], "v20_protocol.dataset_class_relative_path")
    _require_relative_path(v20["dataset_config_relative_path"], "v20_protocol.dataset_config_relative_path")
    if _require_exact_int(v20["class_count"], "v20_protocol.class_count") != SUPPORTED_V20_CLASS_COUNT:
        raise Voc2012MatchedEvaluatorIdentityError(f"v20_protocol.class_count must be exactly {SUPPORTED_V20_CLASS_COUNT}")
    if _require_exact_bool(v20["background_included"], "v20_protocol.background_included") is not False:
        raise Voc2012MatchedEvaluatorIdentityError("v20_protocol.background_included must be false")
    if _require_exact_int(v20["ignore_index"], "v20_protocol.ignore_index") != SUPPORTED_IGNORE_INDEX:
        raise Voc2012MatchedEvaluatorIdentityError(f"v20_protocol.ignore_index must be exactly {SUPPORTED_IGNORE_INDEX}")
    _require_exact_string(v20["class_names_source"], "v20_protocol.class_names_source")

    v21 = identity["v21_protocol"]
    if v21["dataset_type"] != "PascalVOCDataset":
        raise Voc2012MatchedEvaluatorIdentityError("v21_protocol.dataset_type must be exactly 'PascalVOCDataset'")
    _require_exact_string(v21["dataset_class_relative_path"], "v21_protocol.dataset_class_relative_path")
    _require_relative_path(v21["dataset_config_relative_path"], "v21_protocol.dataset_config_relative_path")
    if _require_exact_int(v21["class_count"], "v21_protocol.class_count") != SUPPORTED_V21_CLASS_COUNT:
        raise Voc2012MatchedEvaluatorIdentityError(f"v21_protocol.class_count must be exactly {SUPPORTED_V21_CLASS_COUNT}")
    if _require_exact_bool(v21["background_included"], "v21_protocol.background_included") is not True:
        raise Voc2012MatchedEvaluatorIdentityError("v21_protocol.background_included must be true")
    if _require_exact_int(v21["background_class_index"], "v21_protocol.background_class_index") != SUPPORTED_BACKGROUND_CLASS_INDEX:
        raise Voc2012MatchedEvaluatorIdentityError("v21_protocol.background_class_index must be exactly 0")
    if _require_exact_int(v21["ignore_index"], "v21_protocol.ignore_index") != SUPPORTED_IGNORE_INDEX:
        raise Voc2012MatchedEvaluatorIdentityError(f"v21_protocol.ignore_index must be exactly {SUPPORTED_IGNORE_INDEX}")
    _require_exact_string(v21["class_names_source"], "v21_protocol.class_names_source")
    if v21["class_count"] != v20["class_count"] + 1:
        raise Voc2012MatchedEvaluatorIdentityError("v21_protocol.class_count must equal v20_protocol.class_count + 1")

    shared = identity["shared_scores"]
    _require_exact_string(shared["statement"], "shared_scores.statement")
    _require_exact_string(shared["text_query_source"], "shared_scores.text_query_source")
    _require_exact_string(shared["foreground_score_sharing"], "shared_scores.foreground_score_sharing")
    if _require_exact_bool(shared["second_backbone_pass_permitted"], "shared_scores.second_backbone_pass_permitted") is not False:
        raise Voc2012MatchedEvaluatorIdentityError("shared_scores.second_backbone_pass_permitted must be false")

    mac = identity.get("model_and_checkpoint")
    if not isinstance(mac, Mapping) or set(mac) != MODEL_AND_CHECKPOINT_KEYS:
        raise Voc2012MatchedEvaluatorIdentityError("identity.model_and_checkpoint has an unexpected schema")
    _require_relative_path(mac["model_constructor_relative_path"], "model_and_checkpoint.model_constructor_relative_path")
    if mac["constructor_class"] != SUPPORTED_MODEL_CONSTRUCTOR_CLASS:
        raise Voc2012MatchedEvaluatorIdentityError(f"model_and_checkpoint.constructor_class must be exactly {SUPPORTED_MODEL_CONSTRUCTOR_CLASS!r}")
    _require_exact_string(mac["backbone_name"], "model_and_checkpoint.backbone_name")
    _require_exact_string(mac["clip_model_name"], "model_and_checkpoint.clip_model_name")
    _require_exact_string(mac["projection_class"], "model_and_checkpoint.projection_class")
    _require_relative_path(mac["projection_config_relative_path"], "model_and_checkpoint.projection_config_relative_path")
    _require_relative_path(mac["projection_checkpoint_relative_path"], "model_and_checkpoint.projection_checkpoint_relative_path")
    _require_sha256(mac["projection_checkpoint_sha256"], "model_and_checkpoint.projection_checkpoint_sha256")
    if _require_exact_bool(mac["checkpoint_frozen"], "model_and_checkpoint.checkpoint_frozen") is not True:
        raise Voc2012MatchedEvaluatorIdentityError("model_and_checkpoint.checkpoint_frozen must be true")
    if _require_exact_bool(
        mac["target_dataset_training_or_feature_pre_extraction_permitted"],
        "model_and_checkpoint.target_dataset_training_or_feature_pre_extraction_permitted",
    ) is not False:
        raise Voc2012MatchedEvaluatorIdentityError(
            "model_and_checkpoint.target_dataset_training_or_feature_pre_extraction_permitted must be false"
        )
    _require_exact_string(mac["template"], "model_and_checkpoint.template")
    if _require_exact_bool(mac["pamr"], "model_and_checkpoint.pamr") is not False:
        raise Voc2012MatchedEvaluatorIdentityError("model_and_checkpoint.pamr must be false")
    _load_eval_config_block(mac["v20_eval_config"], "model_and_checkpoint.v20_eval_config")
    _load_eval_config_block(mac["v21_eval_config"], "model_and_checkpoint.v21_eval_config")
    provenance = _require_closed_mapping(mac["provenance"], PROVENANCE_KEYS, "model_and_checkpoint.provenance")
    _require_exact_string(provenance["description"], "model_and_checkpoint.provenance.description")
    _require_sha256(provenance["projection_config_sha256"], "model_and_checkpoint.provenance.projection_config_sha256")
    _require_sha256(provenance["model_constructor_sha256"], "model_and_checkpoint.provenance.model_constructor_sha256")
    _require_relative_path(provenance["dataset_seg_inference_relative_path"], "model_and_checkpoint.provenance.dataset_seg_inference_relative_path")
    _require_sha256(provenance["dataset_seg_inference_sha256"], "model_and_checkpoint.provenance.dataset_seg_inference_sha256")
    _require_relative_path(provenance["dinotext_builder_relative_path"], "model_and_checkpoint.provenance.dinotext_builder_relative_path")
    _require_sha256(provenance["dinotext_builder_sha256"], "model_and_checkpoint.provenance.dinotext_builder_sha256")

    reused = identity["reused_evaluator_modules"]
    _require_exact_string(reused["description"], "reused_evaluator_modules.description")
    for name in (
        "coco_object_evaluator", "matched_power_evaluator", "graph", "finite_step_regime", "sliding_window_geometry",
    ):
        _require_relative_path(reused[f"{name}_relative_path"], f"reused_evaluator_modules.{name}_relative_path")
        _require_sha256(reused[f"{name}_sha256"], f"reused_evaluator_modules.{name}_sha256")

    bg = identity["background_protocol"]
    if bg["mechanism"] != SUPPORTED_BG_MECHANISM:
        raise Voc2012MatchedEvaluatorIdentityError(f"background_protocol.mechanism must be exactly {SUPPORTED_BG_MECHANISM!r}")
    bg_thresh = _require_exact_float(bg["bg_thresh"], "background_protocol.bg_thresh")
    if not (0.0 <= bg_thresh <= 1.0):
        raise Voc2012MatchedEvaluatorIdentityError("background_protocol.bg_thresh must be in [0, 1]")
    if bg["bg_strategy"] != SUPPORTED_BG_STRATEGY:
        raise Voc2012MatchedEvaluatorIdentityError(f"background_protocol.bg_strategy must be exactly {SUPPORTED_BG_STRATEGY!r}")
    if _require_exact_int(bg["background_class_index"], "background_protocol.background_class_index") != SUPPORTED_BACKGROUND_CLASS_INDEX:
        raise Voc2012MatchedEvaluatorIdentityError("background_protocol.background_class_index must be exactly 0")
    if _require_exact_bool(bg["competes_in_argmax"], "background_protocol.competes_in_argmax") is not True:
        raise Voc2012MatchedEvaluatorIdentityError("background_protocol.competes_in_argmax must be true")
    _require_exact_string(bg["formula"], "background_protocol.formula")
    _require_exact_string(bg["formula_source"], "background_protocol.formula_source")
    _require_exact_string(bg["reused_implementation"], "background_protocol.reused_implementation")
    _require_exact_string(bg["injection_stage"], "background_protocol.injection_stage")
    _require_exact_string(bg["provenance_note"], "background_protocol.provenance_note")
    if _require_exact_bool(bg["v20_applies"], "background_protocol.v20_applies") is not False:
        raise Voc2012MatchedEvaluatorIdentityError("background_protocol.v20_applies must be false")
    if _require_exact_bool(bg["v21_applies"], "background_protocol.v21_applies") is not True:
        raise Voc2012MatchedEvaluatorIdentityError("background_protocol.v21_applies must be true")
    if _require_exact_bool(bg["ignore_pixels_excluded_from_argmax"], "background_protocol.ignore_pixels_excluded_from_argmax") is not True:
        raise Voc2012MatchedEvaluatorIdentityError("background_protocol.ignore_pixels_excluded_from_argmax must be true")

    graph = identity["graph_and_propagation"]
    _require_exact_string(graph["statement"], "graph_and_propagation.statement")
    for field in (
        "maximum_rank_source", "alpha_source", "steps_source", "affinity_power_source",
        "crop_source", "stride_source", "window_enumeration_source",
    ):
        _require_exact_string(graph[field], f"graph_and_propagation.{field}")

    execution = identity["execution"]
    for field in ("shared_backbone_pass", "shared_scores", "shared_features", "shared_affinity", "shared_top12_selection"):
        if _require_exact_bool(execution[field], f"execution.{field}") is not True:
            raise Voc2012MatchedEvaluatorIdentityError(f"execution.{field} must be true")
    if _require_exact_int(execution["snapshot_calls_per_window"], "execution.snapshot_calls_per_window") != 1:
        raise Voc2012MatchedEvaluatorIdentityError("execution.snapshot_calls_per_window must be exactly 1")
    if _require_exact_int(execution["topk_graph_calls_per_window"], "execution.topk_graph_calls_per_window") != 1:
        raise Voc2012MatchedEvaluatorIdentityError("execution.topk_graph_calls_per_window must be exactly 1")
    if _require_exact_int(execution["k11_prefix_calls_per_window"], "execution.k11_prefix_calls_per_window") != 1:
        raise Voc2012MatchedEvaluatorIdentityError("execution.k11_prefix_calls_per_window must be exactly 1")
    if _require_exact_int(execution["graph_normalizations_per_window"], "execution.graph_normalizations_per_window") != 2:
        raise Voc2012MatchedEvaluatorIdentityError("execution.graph_normalizations_per_window must be exactly 2")
    if _require_exact_int(execution["k11_propagations_per_window"], "execution.k11_propagations_per_window") != 1:
        raise Voc2012MatchedEvaluatorIdentityError("execution.k11_propagations_per_window must be exactly 1")
    if _require_exact_int(execution["k12_propagations_per_window"], "execution.k12_propagations_per_window") != 1:
        raise Voc2012MatchedEvaluatorIdentityError("execution.k12_propagations_per_window must be exactly 1")
    if _require_exact_int(execution["k11_updates_per_window"], "execution.k11_updates_per_window") != 320:
        raise Voc2012MatchedEvaluatorIdentityError("execution.k11_updates_per_window must be exactly 320")
    if _require_exact_int(execution["k12_updates_per_window"], "execution.k12_updates_per_window") != 320:
        raise Voc2012MatchedEvaluatorIdentityError("execution.k12_updates_per_window must be exactly 320")
    if _require_exact_int(execution["e3_propagations_per_window"], "execution.e3_propagations_per_window") != 0:
        raise Voc2012MatchedEvaluatorIdentityError("execution.e3_propagations_per_window must be exactly 0")
    if _require_exact_bool(execution["second_model_pass_for_v20_v21"], "execution.second_model_pass_for_v20_v21") is not False:
        raise Voc2012MatchedEvaluatorIdentityError("execution.second_model_pass_for_v20_v21 must be false")
    if _require_exact_string_list(execution["variant_order"], "execution.variant_order") != SUPPORTED_VARIANT_ORDER:
        raise Voc2012MatchedEvaluatorIdentityError(f"execution.variant_order must be exactly {list(SUPPORTED_VARIANT_ORDER)}")

    run_modes = identity["run_modes"]
    for mode, count in SUPPORTED_RUN_MODE_IMAGE_COUNTS.items():
        if _require_exact_int(run_modes[f"{mode}_image_count"], f"run_modes.{mode}_image_count") != count:
            raise Voc2012MatchedEvaluatorIdentityError(f"run_modes.{mode}_image_count must be exactly {count}")
        _require_exact_string(run_modes[f"{mode}_schema_name"], f"run_modes.{mode}_schema_name")
    _require_exact_string(run_modes["checkpoint_schema_name"], "run_modes.checkpoint_schema_name")
    _require_exact_string(run_modes["image_order_source"], "run_modes.image_order_source")
    _require_exact_string(run_modes["pilot_prefix_policy"], "run_modes.pilot_prefix_policy")

    metrics = identity["metrics"]
    if _require_exact_string_list(metrics["metric_names"], "metrics.metric_names") != ("aAcc", "mIoU", "mAcc"):
        raise Voc2012MatchedEvaluatorIdentityError("metrics.metric_names must be exactly ['aAcc', 'mIoU', 'mAcc']")
    if metrics["unit"] != "percent_0_100":
        raise Voc2012MatchedEvaluatorIdentityError("metrics.unit must be exactly 'percent_0_100'")
    if metrics["delta_unit"] != "percentage_points":
        raise Voc2012MatchedEvaluatorIdentityError("metrics.delta_unit must be exactly 'percentage_points'")
    _require_exact_string(metrics["precision_source"], "metrics.precision_source")
    _require_exact_string(metrics["sufficient_statistic_schema"], "metrics.sufficient_statistic_schema")
    if _require_exact_bool(metrics["rounded_prettytable_forbidden"], "metrics.rounded_prettytable_forbidden") is not True:
        raise Voc2012MatchedEvaluatorIdentityError("metrics.rounded_prettytable_forbidden must be true")
    if _require_exact_int(metrics["minimum_metric_decimal_places"], "metrics.minimum_metric_decimal_places", minimum=1) < 6:
        raise Voc2012MatchedEvaluatorIdentityError("metrics.minimum_metric_decimal_places must be at least 6")
    if _require_exact_string_list(metrics["delta_fields"], "metrics.delta_fields") != SUPPORTED_DELTA_FIELDS:
        raise Voc2012MatchedEvaluatorIdentityError(f"metrics.delta_fields must be exactly {list(SUPPORTED_DELTA_FIELDS)}")

    _require_exact_string(identity["checkpoint"]["schema_name"], "checkpoint.schema_name")
    _require_exact_string(identity["checkpoint"]["granularity"], "checkpoint.granularity")
    if _require_exact_bool(identity["checkpoint"]["serializes_gpu_tensors"], "checkpoint.serializes_gpu_tensors") is not False:
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.serializes_gpu_tensors must be false")
    _require_exact_string(identity["checkpoint"]["atomic_write_pattern"], "checkpoint.atomic_write_pattern")
    if _require_exact_bool(
        identity["checkpoint"]["corruption_detected_before_model_load"], "checkpoint.corruption_detected_before_model_load"
    ) is not True:
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.corruption_detected_before_model_load must be true")

    _require_exact_string(identity["artifacts"]["per_image_stats_manifest_schema_name"], "artifacts.per_image_stats_manifest_schema_name")
    _require_exact_string(identity["artifacts"]["per_image_stats_array_format"], "artifacts.per_image_stats_array_format")

    verification = identity["verification"]
    if _require_exact_bool(
        verification["require_source_manifest_verified_before_cuda"], "verification.require_source_manifest_verified_before_cuda"
    ) is not True:
        raise Voc2012MatchedEvaluatorIdentityError("verification.require_source_manifest_verified_before_cuda must be true")
    if _require_exact_bool(
        verification["require_checkpoint_bytes_verified_before_cuda"], "verification.require_checkpoint_bytes_verified_before_cuda"
    ) is not True:
        raise Voc2012MatchedEvaluatorIdentityError("verification.require_checkpoint_bytes_verified_before_cuda must be true")

    _require_exact_string_list(identity["prohibited"]["list"], "prohibited.list")

    return identity


def _check_git_ancestry(root: Path, commit: str, *, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"{label} is not an ancestor of HEAD"
        raise Voc2012MatchedEvaluatorIdentityError(f"Git ancestry check failed for {label} ({commit}): {detail}")


def _validate_voc2012_source_parent(root: Path, identity: Mapping[str, Any]) -> Mapping[str, Any]:
    import hashlib

    from src.voc2012_dataset_identity import load_identity as _load_voc2012_source_identity

    parents = identity["parent_identities"]
    source_path = root / parents["voc2012_source_identity_path"]
    try:
        source_bytes = source_path.read_bytes()
    except OSError as error:
        raise Voc2012MatchedEvaluatorIdentityError(f"cannot read VOC2012 source parent identity {source_path}: {error}") from error
    if hashlib.sha256(source_bytes).hexdigest() != parents["voc2012_source_identity_sha256"]:
        raise Voc2012MatchedEvaluatorIdentityError("VOC2012 source parent identity file SHA256 mismatch")

    source_identity = _load_voc2012_source_identity(source_path, repo_root=root)
    if source_identity["identity"]["name"] != parents["voc2012_source_identity_name"]:
        raise Voc2012MatchedEvaluatorIdentityError("VOC2012 source parent identity name mismatch")
    if source_identity["protocol"]["split"] != identity["protocol"]["split"]:
        raise Voc2012MatchedEvaluatorIdentityError("VOC2012 source parent protocol.split disagrees with this identity's protocol.split")
    if source_identity["protocol"]["expected_image_count"] != identity["protocol"]["expected_image_count"]:
        raise Voc2012MatchedEvaluatorIdentityError(
            "VOC2012 source parent protocol.expected_image_count disagrees with this identity's protocol.expected_image_count"
        )
    class_contract = source_identity["class_contract"]
    if class_contract["v20_class_count"] != identity["v20_protocol"]["class_count"]:
        raise Voc2012MatchedEvaluatorIdentityError("VOC2012 source parent class_contract.v20_class_count disagrees with v20_protocol.class_count")
    if class_contract["v21_class_count"] != identity["v21_protocol"]["class_count"]:
        raise Voc2012MatchedEvaluatorIdentityError("VOC2012 source parent class_contract.v21_class_count disagrees with v21_protocol.class_count")
    return source_identity


def _validate_matched_parent(root: Path, identity: Mapping[str, Any]) -> Mapping[str, Any]:
    import hashlib

    from src.matched_k11_k12_identity import load_identity as _load_matched_identity

    parents = identity["parent_identities"]
    matched_path = root / parents["matched_identity_path"]
    try:
        matched_bytes = matched_path.read_bytes()
    except OSError as error:
        raise Voc2012MatchedEvaluatorIdentityError(f"cannot read matched parent identity {matched_path}: {error}") from error
    if hashlib.sha256(matched_bytes).hexdigest() != parents["matched_identity_sha256"]:
        raise Voc2012MatchedEvaluatorIdentityError("matched parent identity file SHA256 mismatch")

    matched_identity = _load_matched_identity(matched_path, repo_root=root)
    if matched_identity["identity"]["name"] != parents["matched_identity_name"]:
        raise Voc2012MatchedEvaluatorIdentityError("matched parent identity name mismatch")
    if matched_identity["graph"]["maximum_rank"] != 12:
        raise Voc2012MatchedEvaluatorIdentityError("matched parent identity graph.maximum_rank must be exactly 12")
    return matched_identity


def _load_yaml_no_duplicate_keys(path: Path) -> dict[str, Any]:
    """Strictly parse one YAML mapping document: rejects duplicate keys
    (PyYAML's default loader silently keeps the last one), never follows
    `_base_` inheritance (the pinned VOC background eval config is a leaf
    file with no `_base_` -- this is intentionally narrower than
    ``e3_evaluation_identity._load_yaml_with_bases``, which this function
    does not need to reuse since there is no inheritance chain here).
    Root must be a mapping."""
    try:
        import yaml
    except ImportError as error:
        raise Voc2012MatchedEvaluatorIdentityError(
            "PyYAML is required to strictly parse the pinned VOC background evaluation config"
        ) from error

    class _NoDuplicateKeysLoader(yaml.SafeLoader):
        pass

    def _construct_mapping(loader: "yaml.SafeLoader", node: "yaml.Node", deep: bool = False) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise Voc2012MatchedEvaluatorIdentityError(f"duplicate YAML key {key!r} in {path}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _NoDuplicateKeysLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
    )

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise Voc2012MatchedEvaluatorIdentityError(f"cannot read {path}: {error}") from error
    try:
        value = yaml.load(text, Loader=_NoDuplicateKeysLoader)
    except yaml.YAMLError as error:
        raise Voc2012MatchedEvaluatorIdentityError(f"cannot parse YAML {path}: {error}") from error
    if not isinstance(value, dict):
        raise Voc2012MatchedEvaluatorIdentityError(f"YAML document must be a mapping: {path}")
    return value


def validate_bg_thresh_binding(root: Path, identity: Mapping[str, Any]) -> float:
    """Strictly parse the identity-pinned V21 background evaluation YAML
    (``model_and_checkpoint.v21_eval_config.eval_base_config_relative_path``
    -- already SHA256-file-pinned by validate_model_and_checkpoint_binding,
    so this reads the exact bytes that pin authorizes, never a
    independently-chosen path) and require its authoritative
    ``evaluate.bg_thresh`` value to equal
    ``identity["background_protocol"]["bg_thresh"]`` EXACTLY -- same type
    (float, never bool/int/str), same value, no coercion. This is the
    live cross-check the file-hash pin alone cannot provide: hashing
    protects the CONFIG FILE from drifting; this protects the IDENTITY's
    OWN declared bg_thresh from drifting away from that file. CPU-only,
    safe before any CUDA/model work."""
    mac = identity["model_and_checkpoint"]
    config_path = root / mac["v21_eval_config"]["eval_base_config_relative_path"]
    document = _load_yaml_no_duplicate_keys(config_path)

    if "evaluate" not in document:
        raise Voc2012MatchedEvaluatorIdentityError(f"{config_path} is missing the required 'evaluate' section")
    evaluate_section = document["evaluate"]
    if not isinstance(evaluate_section, dict):
        raise Voc2012MatchedEvaluatorIdentityError(f"{config_path}: 'evaluate' must be a mapping")
    if "bg_thresh" not in evaluate_section:
        raise Voc2012MatchedEvaluatorIdentityError(f"{config_path}: 'evaluate.bg_thresh' is missing")

    live_value = evaluate_section["bg_thresh"]
    if type(live_value) is not float:
        raise Voc2012MatchedEvaluatorIdentityError(
            f"{config_path}: evaluate.bg_thresh must be an exact float, got {type(live_value).__name__} ({live_value!r})"
        )
    import math

    if not math.isfinite(live_value):
        raise Voc2012MatchedEvaluatorIdentityError(f"{config_path}: evaluate.bg_thresh must be finite, got {live_value!r}")

    declared = identity["background_protocol"]["bg_thresh"]
    if live_value != declared:
        raise Voc2012MatchedEvaluatorIdentityError(
            f"{config_path}: live evaluate.bg_thresh={live_value!r} disagrees with "
            f"identity.background_protocol.bg_thresh={declared!r} -- the identity's declared threshold has "
            "drifted from its own pinned source config"
        )
    return live_value


def validate_model_and_checkpoint_binding(root: Path, identity: Mapping[str, Any]) -> None:
    """Re-hash every plain leaf config/module/checkpoint file this identity
    pins and require it to match exactly. CPU-only, no mmcv/mmseg import
    required -- safe to run at preflight, before any CUDA/model work."""
    import hashlib

    def _sha256(relative_path: str) -> str:
        target = root / relative_path
        try:
            return hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as error:
            raise Voc2012MatchedEvaluatorIdentityError(f"cannot read {target}: {error}") from error

    mac = identity["model_and_checkpoint"]
    checks = (
        (mac["model_constructor_relative_path"], mac["provenance"]["model_constructor_sha256"]),
        (mac["projection_config_relative_path"], mac["provenance"]["projection_config_sha256"]),
        (mac["projection_checkpoint_relative_path"], mac["projection_checkpoint_sha256"]),
        (mac["provenance"]["dataset_seg_inference_relative_path"], mac["provenance"]["dataset_seg_inference_sha256"]),
        (mac["provenance"]["dinotext_builder_relative_path"], mac["provenance"]["dinotext_builder_sha256"]),
        (mac["v20_eval_config"]["eval_config_relative_path"], mac["v20_eval_config"]["eval_config_sha256"]),
        (mac["v20_eval_config"]["eval_base_config_relative_path"], mac["v20_eval_config"]["eval_base_config_sha256"]),
        (mac["v21_eval_config"]["eval_config_relative_path"], mac["v21_eval_config"]["eval_config_sha256"]),
        (mac["v21_eval_config"]["eval_base_config_relative_path"], mac["v21_eval_config"]["eval_base_config_sha256"]),
    )
    for relative_path, expected_sha256 in checks:
        observed = _sha256(relative_path)
        if observed != expected_sha256:
            raise Voc2012MatchedEvaluatorIdentityError(
                f"{relative_path} SHA256 mismatch: identity declares {expected_sha256}, observed {observed}"
            )

    reused = identity["reused_evaluator_modules"]
    for name in ("coco_object_evaluator", "matched_power_evaluator", "graph", "finite_step_regime", "sliding_window_geometry"):
        relative_path = reused[f"{name}_relative_path"]
        expected_sha256 = reused[f"{name}_sha256"]
        observed = _sha256(relative_path)
        if observed != expected_sha256:
            raise Voc2012MatchedEvaluatorIdentityError(
                f"{relative_path} SHA256 mismatch: identity declares {expected_sha256}, observed {observed}"
            )

    # File-hash pinning above protects the CONFIG FILE from drifting; this
    # additionally protects the IDENTITY's own declared bg_thresh from
    # drifting away from that (unchanged) file's actual live value.
    validate_bg_thresh_binding(root, identity)


def validate_static_configuration(
    *, repo_root: Path | None = None, identity_path: Path | None = None, check_git: bool = True
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    if check_git:
        _check_git_ancestry(root, identity["identity"]["required_ancestor_commit"], label="identity.required_ancestor_commit")
    source_identity = _validate_voc2012_source_parent(root, identity)
    matched_identity = _validate_matched_parent(root, identity)
    validate_model_and_checkpoint_binding(root, identity)

    return {
        "identity_name": identity["identity"]["name"],
        "voc2012_source_identity_name": source_identity["identity"]["name"],
        "matched_identity_name": matched_identity["identity"]["name"],
        "required_ancestor_commit": identity["identity"]["required_ancestor_commit"],
    }


__all__ = [
    "IDENTITY_RELATIVE_PATH",
    "Voc2012MatchedEvaluatorIdentityError",
    "SUPPORTED_BACKGROUND_CLASS_INDEX",
    "SUPPORTED_BG_MECHANISM",
    "SUPPORTED_DELTA_FIELDS",
    "SUPPORTED_EXPECTED_IMAGE_COUNT",
    "SUPPORTED_IGNORE_INDEX",
    "SUPPORTED_MODEL_CONSTRUCTOR_CLASS",
    "SUPPORTED_RUN_MODE_IMAGE_COUNTS",
    "SUPPORTED_SPLIT",
    "SUPPORTED_V20_CLASS_COUNT",
    "SUPPORTED_V21_CLASS_COUNT",
    "SUPPORTED_VARIANT_ORDER",
    "load_identity",
    "repository_root",
    "validate_bg_thresh_binding",
    "validate_model_and_checkpoint_binding",
    "validate_static_configuration",
]

