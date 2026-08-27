"""Static identity loading/validation for the ADE20K C150 matched
E3-vs-k11-vs-k12 evaluator: the sole new scientific authority this
adapter adds. Binds relationally to the shared matched k11/k12 identity
and the ADE20K dataset-source identity; duplicates no scientific default.

CPU-only until callers explicitly import torch/mmseg -- this module
itself never does.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping

from src.matched_k11_k12_identity import MatchedK11K12Error
from src.matched_k11_k12_identity import load_identity as load_matched_identity

IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e12_ade20k_matched_evaluator.toml")

IDENTITY_TOP_KEYS = frozenset(
    {
        "format_version", "identity", "parent_identities", "protocol", "dataset", "dataset_root_override",
        "e3_config", "background_protocol", "comparisons", "run_modes", "checkpoint", "verification", "prohibited",
    }
)
IDENTITY_SECTION_KEYS = {
    "identity": frozenset({"name", "schema_version", "description", "required_ancestor_commit"}),
    "parent_identities": frozenset(
        {
            "matched_identity_path", "matched_identity_name", "matched_identity_sha256",
            "ade20k_source_identity_path", "ade20k_source_identity_name", "ade20k_source_identity_sha256",
            "required_relationship",
        }
    ),
    "protocol": frozenset({"label", "statement", "split", "expected_image_count"}),
    "dataset": frozenset(
        {
            "dataset_type", "dataset_class_source", "dataset_config_relative_path", "dataset_config_sha256",
            "class_count", "class_names_digest", "class_names_digest_algorithm", "class_names_source",
            "annotation_suffix", "ignore_index", "background_class_evaluated",
        }
    ),
    "dataset_root_override": frozenset({"canonical_configured_root", "override_field", "override_policy", "description"}),
    "e3_config": frozenset(
        {
            "eval_config_relative_path", "eval_config_sha256", "eval_base_config_relative_path",
            "eval_base_config_sha256", "model_constructor_relative_path", "constructor_class",
            "projection_config_relative_path", "projection_checkpoint_relative_path",
            "projection_checkpoint_sha256", "template", "pamr", "checkpoint_choice_note",
        }
    ),
    "background_protocol": frozenset({"mechanism", "background_class_evaluated", "finalizer_source", "finalizer_note"}),
    "comparisons": frozenset(
        {"variants", "primary_delta_definition", "secondary_deltas", "metric_unit", "sufficient_statistic_schema"}
    ),
    "run_modes": frozenset(
        {
            "pilot20_images", "pilot20_schema_name", "pilot100_images", "pilot100_schema_name",
            "full_images", "full_schema_name",
        }
    ),
    "checkpoint": frozenset({"schema_name"}),
    "verification": frozenset({"require_source_manifest_verified_before_cuda", "source_manifest_schema_name", "required_manifest_image_count"}),
    "prohibited": frozenset({"list"}),
}


class Ade20kMatchedEvaluatorIdentityError(ValueError):
    """Raised on any ADE20K matched-evaluator identity invariant
    violation. Always fail closed."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value.strip()):
        raise Ade20kMatchedEvaluatorIdentityError(f"{label} must be an exact non-empty string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise Ade20kMatchedEvaluatorIdentityError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise Ade20kMatchedEvaluatorIdentityError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise Ade20kMatchedEvaluatorIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise Ade20kMatchedEvaluatorIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", token) is None:
        raise Ade20kMatchedEvaluatorIdentityError(f"{label} must be a full Git identity")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token:
        raise Ade20kMatchedEvaluatorIdentityError(f"{label} must be a safe repository-relative path")
    return token


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise Ade20kMatchedEvaluatorIdentityError(f"{label} has an unexpected schema")
    return value


def _require_exact_string_list(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise Ade20kMatchedEvaluatorIdentityError(f"{label} must be a non-empty exact list")
    if any(type(item) is not str for item in value):
        raise Ade20kMatchedEvaluatorIdentityError(f"{label} elements must be exact strings")
    return tuple(value)


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise Ade20kMatchedEvaluatorIdentityError(f"cannot load ade20k matched-evaluator identity {source}: {error}") from error

    if set(identity) != IDENTITY_TOP_KEYS:
        raise Ade20kMatchedEvaluatorIdentityError("ade20k matched-evaluator identity has an unexpected top-level schema")
    _require_exact_string(identity["format_version"], "format_version")
    if identity["format_version"] != "talk2dino-ade20k-matched-evaluator-identity-v1":
        raise Ade20kMatchedEvaluatorIdentityError("unsupported ade20k matched-evaluator identity format_version")
    for section, keys in IDENTITY_SECTION_KEYS.items():
        _require_closed_mapping(identity.get(section), keys, f"identity.{section}")

    block = identity["identity"]
    _require_exact_string(block["name"], "identity.name")
    if block["name"] != "e12-ade20k-matched-evaluator":
        raise Ade20kMatchedEvaluatorIdentityError("identity.name must be exactly 'e12-ade20k-matched-evaluator'")
    _require_exact_string(block["schema_version"], "identity.schema_version")
    if block["schema_version"] != identity["format_version"]:
        raise Ade20kMatchedEvaluatorIdentityError("identity.schema_version disagrees with format_version")
    _require_exact_string(block["description"], "identity.description")
    _require_git_identity(block["required_ancestor_commit"], "identity.required_ancestor_commit")

    parents = identity["parent_identities"]
    _require_relative_path(parents["matched_identity_path"], "parent_identities.matched_identity_path")
    _require_exact_string(parents["matched_identity_name"], "parent_identities.matched_identity_name")
    _require_sha256(parents["matched_identity_sha256"], "parent_identities.matched_identity_sha256")
    _require_relative_path(parents["ade20k_source_identity_path"], "parent_identities.ade20k_source_identity_path")
    _require_exact_string(parents["ade20k_source_identity_name"], "parent_identities.ade20k_source_identity_name")
    _require_sha256(parents["ade20k_source_identity_sha256"], "parent_identities.ade20k_source_identity_sha256")
    _require_exact_string(parents["required_relationship"], "parent_identities.required_relationship")

    protocol = identity["protocol"]
    if protocol["label"] != "ADE20K-C150":
        raise Ade20kMatchedEvaluatorIdentityError("protocol.label must be exactly 'ADE20K-C150'")
    _require_exact_string(protocol["statement"], "protocol.statement")
    if protocol["split"] != "validation":
        raise Ade20kMatchedEvaluatorIdentityError("protocol.split must be exactly 'validation'")
    if _require_exact_int(protocol["expected_image_count"], "protocol.expected_image_count") != 2000:
        raise Ade20kMatchedEvaluatorIdentityError("protocol.expected_image_count must be exactly 2000")

    dataset = identity["dataset"]
    if dataset["dataset_type"] != "ADE20KDataset":
        raise Ade20kMatchedEvaluatorIdentityError("dataset.dataset_type must be exactly 'ADE20KDataset'")
    _require_exact_string(dataset["dataset_class_source"], "dataset.dataset_class_source")
    _require_relative_path(dataset["dataset_config_relative_path"], "dataset.dataset_config_relative_path")
    _require_sha256(dataset["dataset_config_sha256"], "dataset.dataset_config_sha256")
    if _require_exact_int(dataset["class_count"], "dataset.class_count") != 150:
        raise Ade20kMatchedEvaluatorIdentityError("dataset.class_count must be exactly 150")
    _require_sha256(dataset["class_names_digest"], "dataset.class_names_digest")
    if dataset["class_names_digest_algorithm"] != "sha256_json_dumps_ensure_ascii_list":
        raise Ade20kMatchedEvaluatorIdentityError("dataset.class_names_digest_algorithm must be exactly 'sha256_json_dumps_ensure_ascii_list'")
    _require_exact_string(dataset["class_names_source"], "dataset.class_names_source")
    if dataset["annotation_suffix"] != ".png":
        raise Ade20kMatchedEvaluatorIdentityError("dataset.annotation_suffix must be exactly '.png'")
    if _require_exact_int(dataset["ignore_index"], "dataset.ignore_index") != 255:
        raise Ade20kMatchedEvaluatorIdentityError("dataset.ignore_index must be exactly 255")
    if _require_exact_bool(dataset["background_class_evaluated"], "dataset.background_class_evaluated") is not False:
        raise Ade20kMatchedEvaluatorIdentityError("dataset.background_class_evaluated must be false")

    root_override = identity["dataset_root_override"]
    if root_override["canonical_configured_root"] != "./data/ade":
        raise Ade20kMatchedEvaluatorIdentityError("dataset_root_override.canonical_configured_root must be exactly './data/ade'")
    if root_override["override_field"] != "data_root":
        raise Ade20kMatchedEvaluatorIdentityError("dataset_root_override.override_field must be exactly 'data_root'")
    if root_override["override_policy"] != "data_root_only":
        raise Ade20kMatchedEvaluatorIdentityError("dataset_root_override.override_policy must be exactly 'data_root_only'")
    _require_exact_string(root_override["description"], "dataset_root_override.description")

    e3_config = identity["e3_config"]
    _require_relative_path(e3_config["eval_config_relative_path"], "e3_config.eval_config_relative_path")
    _require_sha256(e3_config["eval_config_sha256"], "e3_config.eval_config_sha256")
    _require_relative_path(e3_config["eval_base_config_relative_path"], "e3_config.eval_base_config_relative_path")
    _require_sha256(e3_config["eval_base_config_sha256"], "e3_config.eval_base_config_sha256")
    _require_relative_path(e3_config["model_constructor_relative_path"], "e3_config.model_constructor_relative_path")
    if e3_config["constructor_class"] != "DINOText":
        raise Ade20kMatchedEvaluatorIdentityError("e3_config.constructor_class must be exactly 'DINOText'")
    _require_exact_string(e3_config["projection_config_relative_path"], "e3_config.projection_config_relative_path")
    _require_exact_string(e3_config["projection_checkpoint_relative_path"], "e3_config.projection_checkpoint_relative_path")
    _require_sha256(e3_config["projection_checkpoint_sha256"], "e3_config.projection_checkpoint_sha256")
    _require_exact_string(e3_config["template"], "e3_config.template")
    _require_exact_bool(e3_config["pamr"], "e3_config.pamr")
    if e3_config["pamr"] is not False:
        raise Ade20kMatchedEvaluatorIdentityError("e3_config.pamr must be false")
    _require_exact_string(e3_config["checkpoint_choice_note"], "e3_config.checkpoint_choice_note")

    background = identity["background_protocol"]
    if background["mechanism"] != "none":
        raise Ade20kMatchedEvaluatorIdentityError("background_protocol.mechanism must be exactly 'none'")
    if _require_exact_bool(background["background_class_evaluated"], "background_protocol.background_class_evaluated") is not False:
        raise Ade20kMatchedEvaluatorIdentityError("background_protocol.background_class_evaluated must be false")
    if background["finalizer_source"] != "models.dinotext.cover_dr.matched_power_evaluator.finalize_prediction":
        raise Ade20kMatchedEvaluatorIdentityError(
            "background_protocol.finalizer_source must be exactly "
            "'models.dinotext.cover_dr.matched_power_evaluator.finalize_prediction'"
        )
    _require_exact_string(background["finalizer_note"], "background_protocol.finalizer_note")

    comparisons = identity["comparisons"]
    if _require_exact_string_list(comparisons["variants"], "comparisons.variants") != ("E3", "k11", "k12"):
        raise Ade20kMatchedEvaluatorIdentityError("comparisons.variants must be exactly ['E3', 'k11', 'k12']")
    if comparisons["primary_delta_definition"] != "mIoU_k11_minus_mIoU_k12":
        raise Ade20kMatchedEvaluatorIdentityError("comparisons.primary_delta_definition must be exactly 'mIoU_k11_minus_mIoU_k12'")
    if set(_require_exact_string_list(comparisons["secondary_deltas"], "comparisons.secondary_deltas")) != {"k11_minus_E3", "k12_minus_E3"}:
        raise Ade20kMatchedEvaluatorIdentityError("comparisons.secondary_deltas must be exactly {'k11_minus_E3', 'k12_minus_E3'}")
    if comparisons["metric_unit"] != "percentage_points":
        raise Ade20kMatchedEvaluatorIdentityError("comparisons.metric_unit must be exactly 'percentage_points'")
    _require_exact_string(comparisons["sufficient_statistic_schema"], "comparisons.sufficient_statistic_schema")

    run_modes = identity["run_modes"]
    if _require_exact_int(run_modes["pilot20_images"], "run_modes.pilot20_images") != 20:
        raise Ade20kMatchedEvaluatorIdentityError("run_modes.pilot20_images must be exactly 20")
    _require_exact_string(run_modes["pilot20_schema_name"], "run_modes.pilot20_schema_name")
    if _require_exact_int(run_modes["pilot100_images"], "run_modes.pilot100_images") != 100:
        raise Ade20kMatchedEvaluatorIdentityError("run_modes.pilot100_images must be exactly 100")
    _require_exact_string(run_modes["pilot100_schema_name"], "run_modes.pilot100_schema_name")
    if _require_exact_int(run_modes["full_images"], "run_modes.full_images") != 2000:
        raise Ade20kMatchedEvaluatorIdentityError("run_modes.full_images must be exactly 2000")
    _require_exact_string(run_modes["full_schema_name"], "run_modes.full_schema_name")

    checkpoint = identity["checkpoint"]
    _require_exact_string(checkpoint["schema_name"], "checkpoint.schema_name")

    verification = identity["verification"]
    if _require_exact_bool(verification["require_source_manifest_verified_before_cuda"], "verification.require_source_manifest_verified_before_cuda") is not True:
        raise Ade20kMatchedEvaluatorIdentityError("verification.require_source_manifest_verified_before_cuda must be true")
    _require_exact_string(verification["source_manifest_schema_name"], "verification.source_manifest_schema_name")
    if _require_exact_int(verification["required_manifest_image_count"], "verification.required_manifest_image_count") != 2000:
        raise Ade20kMatchedEvaluatorIdentityError("verification.required_manifest_image_count must be exactly 2000")

    _require_exact_string_list(identity["prohibited"]["list"], "prohibited.list")

    return identity


def _check_git_ancestry(root: Path, commit: str, *, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"{label} is not an ancestor of HEAD"
        raise Ade20kMatchedEvaluatorIdentityError(f"Git ancestry check failed for {label} ({commit}): {detail}")


def _validate_file_binding(root: Path, relative_path: str, expected_sha256: str, *, label: str) -> None:
    path = root / relative_path
    try:
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise Ade20kMatchedEvaluatorIdentityError(f"cannot read {label} {path}: {error}") from error
    if observed != expected_sha256:
        raise Ade20kMatchedEvaluatorIdentityError(
            f"{label} {path} SHA256 mismatch: identity declares {expected_sha256}, observed {observed}"
        )


def validate_bridge_checkpoint_binding(root: Path, identity: Mapping[str, Any]) -> str:
    """Hash the actual bridge/projection checkpoint FILE BYTES on disk
    and require them to equal e3_config.projection_checkpoint_sha256 --
    a direct byte-level pin, not merely a format check. Mirrors the
    COCO-Object identity's bridge_checkpoint_binding hardening."""
    e3_config = identity["e3_config"]
    _validate_file_binding(
        root, e3_config["projection_checkpoint_relative_path"], e3_config["projection_checkpoint_sha256"],
        label="bridge checkpoint",
    )
    return e3_config["projection_checkpoint_sha256"]


def _validate_matched_parent(root: Path, identity: Mapping[str, Any]) -> Mapping[str, Any]:
    parents = identity["parent_identities"]
    matched_path = root / parents["matched_identity_path"]
    try:
        observed_sha256 = hashlib.sha256(matched_path.read_bytes()).hexdigest()
    except OSError as error:
        raise Ade20kMatchedEvaluatorIdentityError(f"cannot read matched parent identity {matched_path}: {error}") from error
    if observed_sha256 != parents["matched_identity_sha256"]:
        raise Ade20kMatchedEvaluatorIdentityError(
            f"matched parent identity {matched_path} SHA256 mismatch: identity declares "
            f"{parents['matched_identity_sha256']}, observed {observed_sha256}"
        )
    try:
        matched_identity = load_matched_identity(matched_path, repo_root=root)
    except MatchedK11K12Error as error:
        raise Ade20kMatchedEvaluatorIdentityError(f"matched parent identity failed validation: {error}") from error
    if matched_identity["identity"]["name"] != parents["matched_identity_name"]:
        raise Ade20kMatchedEvaluatorIdentityError("matched parent identity name mismatch")
    return matched_identity


def _validate_ade20k_source_parent(root: Path, identity: Mapping[str, Any]) -> Mapping[str, Any]:
    from src.ade20k_dataset_identity import Ade20kDatasetIdentityError, load_identity as load_source_identity

    parents = identity["parent_identities"]
    source_path = root / parents["ade20k_source_identity_path"]
    try:
        observed_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError as error:
        raise Ade20kMatchedEvaluatorIdentityError(f"cannot read ADE20K source parent identity {source_path}: {error}") from error
    if observed_sha256 != parents["ade20k_source_identity_sha256"]:
        raise Ade20kMatchedEvaluatorIdentityError(
            f"ADE20K source parent identity {source_path} SHA256 mismatch: identity declares "
            f"{parents['ade20k_source_identity_sha256']}, observed {observed_sha256}"
        )
    try:
        source_identity = load_source_identity(source_path, repo_root=root)
    except Ade20kDatasetIdentityError as error:
        raise Ade20kMatchedEvaluatorIdentityError(f"ADE20K source parent identity failed validation: {error}") from error
    if source_identity["identity"]["name"] != parents["ade20k_source_identity_name"]:
        raise Ade20kMatchedEvaluatorIdentityError("ADE20K source parent identity name mismatch")
    if source_identity["protocol"]["expected_image_count"] != identity["protocol"]["expected_image_count"]:
        raise Ade20kMatchedEvaluatorIdentityError("ADE20K source parent expected_image_count disagrees with this identity's protocol.expected_image_count")
    if source_identity["class_contract"]["class_count"] != identity["dataset"]["class_count"]:
        raise Ade20kMatchedEvaluatorIdentityError("ADE20K source parent class_count disagrees with this identity's dataset.class_count")
    if source_identity["class_contract"]["class_names_digest"] != identity["dataset"]["class_names_digest"]:
        raise Ade20kMatchedEvaluatorIdentityError("ADE20K source parent class_names_digest disagrees with this identity's dataset.class_names_digest")
    return source_identity


def validate_static_configuration(*, repo_root: Path, identity_path: Path | None = None, check_git: bool = True) -> dict[str, Any]:
    root = Path(repo_root)
    identity = load_identity(identity_path, repo_root=root)
    if check_git:
        _check_git_ancestry(root, identity["identity"]["required_ancestor_commit"], label="identity.required_ancestor_commit")

    matched_identity = _validate_matched_parent(root, identity)
    source_identity = _validate_ade20k_source_parent(root, identity)
    _validate_file_binding(root, identity["dataset"]["dataset_config_relative_path"], identity["dataset"]["dataset_config_sha256"], label="dataset config")
    _validate_file_binding(root, identity["e3_config"]["eval_config_relative_path"], identity["e3_config"]["eval_config_sha256"], label="e3 eval config")
    _validate_file_binding(root, identity["e3_config"]["eval_base_config_relative_path"], identity["e3_config"]["eval_base_config_sha256"], label="e3 eval base config")
    validate_bridge_checkpoint_binding(root, identity)

    return {
        "identity_name": identity["identity"]["name"],
        "matched_identity_name": matched_identity["identity"]["name"],
        "source_identity_name": source_identity["identity"]["name"],
        "split": identity["protocol"]["split"],
        "expected_image_count": identity["protocol"]["expected_image_count"],
    }


def validate_live_class_order(dataset: Any, identity: Mapping[str, Any]) -> str:
    """Bind the live-instantiated mmseg dataset's own CLASSES order/digest
    against the identity before any model inference."""
    import json

    classes = dataset.CLASSES
    if type(classes) is not tuple or any(type(name) is not str for name in classes):
        raise Ade20kMatchedEvaluatorIdentityError("dataset.CLASSES must be an exact tuple of exact strings")
    if len(classes) != identity["dataset"]["class_count"]:
        raise Ade20kMatchedEvaluatorIdentityError(
            f"live dataset.CLASSES has {len(classes)} entries, identity declares {identity['dataset']['class_count']}"
        )
    digest = hashlib.sha256(json.dumps(list(classes), ensure_ascii=True).encode("utf-8")).hexdigest()
    if digest != identity["dataset"]["class_names_digest"]:
        raise Ade20kMatchedEvaluatorIdentityError(
            f"live dataset.CLASSES digest {digest!r} disagrees with identity.dataset.class_names_digest "
            f"{identity['dataset']['class_names_digest']!r}"
        )
    reduce_zero_label = getattr(dataset, "reduce_zero_label", None)
    if reduce_zero_label is not True:
        raise Ade20kMatchedEvaluatorIdentityError(f"live dataset.reduce_zero_label must be True, observed {reduce_zero_label!r}")
    return digest


__all__ = [
    "IDENTITY_RELATIVE_PATH",
    "Ade20kMatchedEvaluatorIdentityError",
    "load_identity",
    "repository_root",
    "validate_bridge_checkpoint_binding",
    "validate_live_class_order",
    "validate_static_configuration",
]
