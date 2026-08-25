"""Static identity loading/validation for the COCO-Object protocol
confirmation evaluator (E3 vs k11 vs k12, T=320 finite-step).

Never redeclares alpha/steps/affinity_power/crop/stride/maximum_rank as
bare literals -- those are read from the parent matched-k11/k12 identity
at runtime. Never redeclares the materialization contract (image count,
converter hash, class digest, ...) -- those are read from the parent
materialization identity and cross-checked against the real manifest.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e12_coco_object_protocol_confirmation.toml")

SUPPORTED_SPLIT = "val2017"
SUPPORTED_EXPECTED_IMAGE_COUNT = 5000
SUPPORTED_DATASET_TYPE = "COCOObjectDataset"
SUPPORTED_CLASS_COUNT = 81
SUPPORTED_FOREGROUND_CLASS_COUNT = 80
SUPPORTED_BACKGROUND_CLASS_INDEX = 0
SUPPORTED_ANNOTATION_SUFFIX = "_instanceTrainIds.png"
SUPPORTED_IGNORE_INDEX = 255
SUPPORTED_VARIANTS = ("E3", "k11", "k12")
SUPPORTED_PRIMARY_DELTA = "mIoU_k11_minus_mIoU_k12"
SUPPORTED_SECONDARY_DELTAS = ("k11_minus_E3", "k12_minus_E3")
SUPPORTED_METRIC_UNIT = "percentage_points"
SUPPORTED_RUN_MODE_IMAGE_COUNTS = {"pilot20": 20, "pilot100": 100, "full": 5000}
SUPPORTED_BG_MECHANISM = "canonical_constant_threshold_channel"
SUPPORTED_BG_STRATEGY = "base"
SUPPORTED_CLASS_NAMES_DIGEST_ALGORITHM = "sha256_json_dumps_ensure_ascii_list"
SUPPORTED_MODEL_CONSTRUCTOR_CLASS = "DINOText"

IDENTITY_TOP_KEYS = frozenset(
    {
        "format_version", "identity", "parent_identities", "protocol", "dataset",
        "dataset_root_override", "e3_config", "background_protocol", "comparisons",
        "run_modes", "checkpoint", "verification", "prohibited",
    }
)
IDENTITY_SECTION_KEYS = {
    "identity": frozenset({"name", "schema_version", "description", "required_ancestor_commit"}),
    "parent_identities": frozenset(
        {
            "matched_identity_path", "matched_identity_name", "matched_identity_sha256",
            "materialization_identity_path", "materialization_identity_name", "materialization_identity_sha256",
            "required_relationship",
        }
    ),
    "protocol": frozenset({"label", "statement", "split", "expected_image_count"}),
    "dataset": frozenset(
        {
            "dataset_type", "dataset_class_relative_path", "dataset_config_relative_path", "class_count",
            "foreground_class_count", "background_class_index", "class_names_digest", "class_names_digest_algorithm",
            "class_names_source", "annotation_suffix", "ignore_index", "metric_includes_background",
        }
    ),
    "dataset_root_override": frozenset(
        {"canonical_configured_root", "override_field", "override_policy", "description"}
    ),
    "e3_config": frozenset(
        {
            "eval_config_relative_path", "eval_base_config_relative_path", "model_constructor_relative_path",
            "constructor_class", "projection_config_relative_path", "projection_checkpoint_relative_path",
            "template", "pamr", "with_bg_clean", "with_bg_clean_note", "resolved_configuration",
        }
    ),
    "background_protocol": frozenset(
        {
            "mechanism", "bg_thresh", "bg_strategy", "background_class_index", "competes_in_argmax", "formula",
            "formula_source", "injection_stage", "injection_stage_note", "ignore_pixels_excluded_from_argmax",
            "stuff_never_conflated_with_ignore",
        }
    ),
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
    "verification": frozenset(
        {
            "require_verify_output_before_cuda", "materialization_manifest_schema_name",
            "required_manifest_complete", "required_manifest_final", "required_manifest_image_count",
        }
    ),
    "prohibited": frozenset({"list"}),
}
RESOLVED_CONFIGURATION_KEYS = frozenset(
    {"encoding_version", "full_sha256", "dataset_sha256", "dataset_pipeline_sha256", "evaluation_sha256", "model_projection_sha256"}
)


class CocoObjectProtocolConfirmationIdentityError(ValueError):
    """Raised when the protocol-confirmation identity, a checkpoint, or a
    result fails any exact-type/schema/provenance/binding check. Always
    fail closed: never silently substitute a default graph/propagation
    parameter, dataset root, or background rule."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be an exact non-empty string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_exact_float(value: Any, label: str) -> float:
    if type(value) is not float:
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be an exact float")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", token) is None:
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be a full Git identity")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token:
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be a safe repository-relative path")
    return token


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} has an unexpected schema")
    return value


def _require_exact_string_list(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be a non-empty exact list")
    if any(type(item) is not str for item in value):
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} elements must be exact strings")
    return tuple(value)


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise CocoObjectProtocolConfirmationIdentityError(
            f"cannot load protocol-confirmation identity {source}: {error}"
        ) from error

    if set(identity) != IDENTITY_TOP_KEYS:
        raise CocoObjectProtocolConfirmationIdentityError("protocol-confirmation identity has an unexpected top-level schema")
    _require_exact_string(identity["format_version"], "format_version")
    if identity["format_version"] != "talk2dino-coco-object-protocol-confirmation-identity-v1":
        raise CocoObjectProtocolConfirmationIdentityError("unsupported protocol-confirmation identity format_version")
    for section, keys in IDENTITY_SECTION_KEYS.items():
        if section == "e3_config":
            block = identity.get(section)
            if not isinstance(block, Mapping) or set(block) != keys:
                raise CocoObjectProtocolConfirmationIdentityError("identity.e3_config has an unexpected schema")
            _require_closed_mapping(block["resolved_configuration"], RESOLVED_CONFIGURATION_KEYS, "identity.e3_config.resolved_configuration")
        else:
            _require_closed_mapping(identity.get(section), keys, f"identity.{section}")

    block = identity["identity"]
    _require_exact_string(block["name"], "identity.name")
    _require_exact_string(block["schema_version"], "identity.schema_version")
    if block["schema_version"] != identity["format_version"]:
        raise CocoObjectProtocolConfirmationIdentityError("identity.schema_version disagrees with format_version")
    _require_exact_string(block["description"], "identity.description")
    _require_git_identity(block["required_ancestor_commit"], "identity.required_ancestor_commit")

    parents = identity["parent_identities"]
    _require_relative_path(parents["matched_identity_path"], "parent_identities.matched_identity_path")
    _require_exact_string(parents["matched_identity_name"], "parent_identities.matched_identity_name")
    _require_sha256(parents["matched_identity_sha256"], "parent_identities.matched_identity_sha256")
    _require_relative_path(parents["materialization_identity_path"], "parent_identities.materialization_identity_path")
    _require_exact_string(parents["materialization_identity_name"], "parent_identities.materialization_identity_name")
    _require_sha256(parents["materialization_identity_sha256"], "parent_identities.materialization_identity_sha256")
    _require_exact_string(parents["required_relationship"], "parent_identities.required_relationship")

    protocol = identity["protocol"]
    if protocol["label"] != "COCO-Object":
        raise CocoObjectProtocolConfirmationIdentityError("protocol.label must be exactly 'COCO-Object'")
    _require_exact_string(protocol["statement"], "protocol.statement")
    if protocol["split"] != SUPPORTED_SPLIT:
        raise CocoObjectProtocolConfirmationIdentityError(f"protocol.split must be exactly {SUPPORTED_SPLIT!r}")
    if _require_exact_int(protocol["expected_image_count"], "protocol.expected_image_count") != SUPPORTED_EXPECTED_IMAGE_COUNT:
        raise CocoObjectProtocolConfirmationIdentityError(f"protocol.expected_image_count must be exactly {SUPPORTED_EXPECTED_IMAGE_COUNT}")

    dataset = identity["dataset"]
    if dataset["dataset_type"] != SUPPORTED_DATASET_TYPE:
        raise CocoObjectProtocolConfirmationIdentityError(f"dataset.dataset_type must be exactly {SUPPORTED_DATASET_TYPE!r}")
    _require_relative_path(dataset["dataset_class_relative_path"], "dataset.dataset_class_relative_path")
    _require_relative_path(dataset["dataset_config_relative_path"], "dataset.dataset_config_relative_path")
    if _require_exact_int(dataset["class_count"], "dataset.class_count") != SUPPORTED_CLASS_COUNT:
        raise CocoObjectProtocolConfirmationIdentityError(f"dataset.class_count must be exactly {SUPPORTED_CLASS_COUNT}")
    if _require_exact_int(dataset["foreground_class_count"], "dataset.foreground_class_count") != SUPPORTED_FOREGROUND_CLASS_COUNT:
        raise CocoObjectProtocolConfirmationIdentityError(f"dataset.foreground_class_count must be exactly {SUPPORTED_FOREGROUND_CLASS_COUNT}")
    if _require_exact_int(dataset["background_class_index"], "dataset.background_class_index") != SUPPORTED_BACKGROUND_CLASS_INDEX:
        raise CocoObjectProtocolConfirmationIdentityError("dataset.background_class_index must be exactly 0")
    _require_sha256(dataset["class_names_digest"], "dataset.class_names_digest")
    if dataset["class_names_digest_algorithm"] != SUPPORTED_CLASS_NAMES_DIGEST_ALGORITHM:
        raise CocoObjectProtocolConfirmationIdentityError(
            f"dataset.class_names_digest_algorithm must be exactly {SUPPORTED_CLASS_NAMES_DIGEST_ALGORITHM!r}"
        )
    _require_exact_string(dataset["class_names_source"], "dataset.class_names_source")
    if dataset["annotation_suffix"] != SUPPORTED_ANNOTATION_SUFFIX:
        raise CocoObjectProtocolConfirmationIdentityError(f"dataset.annotation_suffix must be exactly {SUPPORTED_ANNOTATION_SUFFIX!r}")
    if _require_exact_int(dataset["ignore_index"], "dataset.ignore_index") != SUPPORTED_IGNORE_INDEX:
        raise CocoObjectProtocolConfirmationIdentityError(f"dataset.ignore_index must be exactly {SUPPORTED_IGNORE_INDEX}")
    if _require_exact_bool(dataset["metric_includes_background"], "dataset.metric_includes_background") is not True:
        raise CocoObjectProtocolConfirmationIdentityError("dataset.metric_includes_background must be true")

    override = identity["dataset_root_override"]
    if override["canonical_configured_root"] != "./data/coco_stuff164k":
        raise CocoObjectProtocolConfirmationIdentityError("dataset_root_override.canonical_configured_root must match coco.py's own declared data_root")
    if override["override_field"] != "data_root":
        raise CocoObjectProtocolConfirmationIdentityError("dataset_root_override.override_field must be exactly 'data_root'")
    if override["override_policy"] != "data_root_only":
        raise CocoObjectProtocolConfirmationIdentityError("dataset_root_override.override_policy must be exactly 'data_root_only'")
    _require_exact_string(override["description"], "dataset_root_override.description")

    e3_config = identity["e3_config"]
    _require_relative_path(e3_config["eval_config_relative_path"], "e3_config.eval_config_relative_path")
    _require_relative_path(e3_config["eval_base_config_relative_path"], "e3_config.eval_base_config_relative_path")
    _require_relative_path(e3_config["model_constructor_relative_path"], "e3_config.model_constructor_relative_path")
    if e3_config["constructor_class"] != SUPPORTED_MODEL_CONSTRUCTOR_CLASS:
        raise CocoObjectProtocolConfirmationIdentityError(
            f"e3_config.constructor_class must be exactly {SUPPORTED_MODEL_CONSTRUCTOR_CLASS!r}"
        )
    _require_relative_path(e3_config["projection_config_relative_path"], "e3_config.projection_config_relative_path")
    _require_relative_path(e3_config["projection_checkpoint_relative_path"], "e3_config.projection_checkpoint_relative_path")
    _require_exact_string(e3_config["template"], "e3_config.template")
    if _require_exact_bool(e3_config["pamr"], "e3_config.pamr") is not False:
        raise CocoObjectProtocolConfirmationIdentityError("e3_config.pamr must be false")
    _require_exact_bool(e3_config["with_bg_clean"], "e3_config.with_bg_clean")
    _require_exact_string(e3_config["with_bg_clean_note"], "e3_config.with_bg_clean_note")
    resolved = e3_config["resolved_configuration"]
    _require_exact_string(resolved["encoding_version"], "e3_config.resolved_configuration.encoding_version")
    for name in ("full_sha256", "dataset_sha256", "dataset_pipeline_sha256", "evaluation_sha256", "model_projection_sha256"):
        _require_sha256(resolved[name], f"e3_config.resolved_configuration.{name}")

    bg = identity["background_protocol"]
    if bg["mechanism"] != SUPPORTED_BG_MECHANISM:
        raise CocoObjectProtocolConfirmationIdentityError(f"background_protocol.mechanism must be exactly {SUPPORTED_BG_MECHANISM!r}")
    _require_exact_float(bg["bg_thresh"], "background_protocol.bg_thresh")
    if bg["bg_strategy"] != SUPPORTED_BG_STRATEGY:
        raise CocoObjectProtocolConfirmationIdentityError(f"background_protocol.bg_strategy must be exactly {SUPPORTED_BG_STRATEGY!r}")
    if _require_exact_int(bg["background_class_index"], "background_protocol.background_class_index") != SUPPORTED_BACKGROUND_CLASS_INDEX:
        raise CocoObjectProtocolConfirmationIdentityError("background_protocol.background_class_index must be exactly 0")
    if _require_exact_bool(bg["competes_in_argmax"], "background_protocol.competes_in_argmax") is not True:
        raise CocoObjectProtocolConfirmationIdentityError("background_protocol.competes_in_argmax must be true")
    _require_exact_string(bg["formula"], "background_protocol.formula")
    _require_exact_string(bg["formula_source"], "background_protocol.formula_source")
    _require_exact_string(bg["injection_stage"], "background_protocol.injection_stage")
    _require_exact_string(bg["injection_stage_note"], "background_protocol.injection_stage_note")
    if _require_exact_bool(bg["ignore_pixels_excluded_from_argmax"], "background_protocol.ignore_pixels_excluded_from_argmax") is not True:
        raise CocoObjectProtocolConfirmationIdentityError("background_protocol.ignore_pixels_excluded_from_argmax must be true")
    if _require_exact_bool(bg["stuff_never_conflated_with_ignore"], "background_protocol.stuff_never_conflated_with_ignore") is not True:
        raise CocoObjectProtocolConfirmationIdentityError("background_protocol.stuff_never_conflated_with_ignore must be true")

    comparisons = identity["comparisons"]
    if _require_exact_string_list(comparisons["variants"], "comparisons.variants") != SUPPORTED_VARIANTS:
        raise CocoObjectProtocolConfirmationIdentityError(f"comparisons.variants must be exactly {list(SUPPORTED_VARIANTS)}")
    if comparisons["primary_delta_definition"] != SUPPORTED_PRIMARY_DELTA:
        raise CocoObjectProtocolConfirmationIdentityError(f"comparisons.primary_delta_definition must be exactly {SUPPORTED_PRIMARY_DELTA!r}")
    if _require_exact_string_list(comparisons["secondary_deltas"], "comparisons.secondary_deltas") != SUPPORTED_SECONDARY_DELTAS:
        raise CocoObjectProtocolConfirmationIdentityError(f"comparisons.secondary_deltas must be exactly {list(SUPPORTED_SECONDARY_DELTAS)}")
    if comparisons["metric_unit"] != SUPPORTED_METRIC_UNIT:
        raise CocoObjectProtocolConfirmationIdentityError(f"comparisons.metric_unit must be exactly {SUPPORTED_METRIC_UNIT!r}")
    _require_exact_string(comparisons["sufficient_statistic_schema"], "comparisons.sufficient_statistic_schema")

    run_modes = identity["run_modes"]
    for mode, count in SUPPORTED_RUN_MODE_IMAGE_COUNTS.items():
        if _require_exact_int(run_modes[f"{mode}_images"], f"run_modes.{mode}_images") != count:
            raise CocoObjectProtocolConfirmationIdentityError(f"run_modes.{mode}_images must be exactly {count}")
        _require_exact_string(run_modes[f"{mode}_schema_name"], f"run_modes.{mode}_schema_name")

    _require_exact_string(identity["checkpoint"]["schema_name"], "checkpoint.schema_name")

    verification = identity["verification"]
    if _require_exact_bool(verification["require_verify_output_before_cuda"], "verification.require_verify_output_before_cuda") is not True:
        raise CocoObjectProtocolConfirmationIdentityError("verification.require_verify_output_before_cuda must be true")
    _require_exact_string(verification["materialization_manifest_schema_name"], "verification.materialization_manifest_schema_name")
    if _require_exact_bool(verification["required_manifest_complete"], "verification.required_manifest_complete") is not True:
        raise CocoObjectProtocolConfirmationIdentityError("verification.required_manifest_complete must be true")
    if _require_exact_bool(verification["required_manifest_final"], "verification.required_manifest_final") is not True:
        raise CocoObjectProtocolConfirmationIdentityError("verification.required_manifest_final must be true")
    if _require_exact_int(verification["required_manifest_image_count"], "verification.required_manifest_image_count") != SUPPORTED_EXPECTED_IMAGE_COUNT:
        raise CocoObjectProtocolConfirmationIdentityError(f"verification.required_manifest_image_count must be exactly {SUPPORTED_EXPECTED_IMAGE_COUNT}")

    _require_exact_string_list(identity["prohibited"]["list"], "prohibited.list")

    return identity


def _check_git_ancestry(root: Path, commit: str, *, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"{label} is not an ancestor of HEAD"
        raise CocoObjectProtocolConfirmationIdentityError(f"Git ancestry check failed for {label} ({commit}): {detail}")


def _validate_matched_parent(root: Path, identity: Mapping[str, Any]) -> Mapping[str, Any]:
    import hashlib

    from src.matched_k11_k12_identity import load_identity as _load_matched_identity

    parents = identity["parent_identities"]
    matched_path = root / parents["matched_identity_path"]
    try:
        matched_bytes = matched_path.read_bytes()
    except OSError as error:
        raise CocoObjectProtocolConfirmationIdentityError(f"cannot read matched parent identity {matched_path}: {error}") from error
    if hashlib.sha256(matched_bytes).hexdigest() != parents["matched_identity_sha256"]:
        raise CocoObjectProtocolConfirmationIdentityError("matched parent identity file SHA256 mismatch")

    # The real loader performs every schema/type/value check (including
    # graph.maximum_rank == 12); never duplicate that validation here.
    matched_identity = _load_matched_identity(matched_path, repo_root=root)
    if matched_identity["identity"]["name"] != parents["matched_identity_name"]:
        raise CocoObjectProtocolConfirmationIdentityError("matched parent identity name mismatch")
    return matched_identity


def _validate_materialization_parent(root: Path, identity: Mapping[str, Any]) -> Mapping[str, Any]:
    import hashlib

    from src.coco_object_val_materialization_identity import load_identity as _load_materialization_identity

    parents = identity["parent_identities"]
    materialization_path = root / parents["materialization_identity_path"]
    try:
        materialization_bytes = materialization_path.read_bytes()
    except OSError as error:
        raise CocoObjectProtocolConfirmationIdentityError(f"cannot read materialization parent identity {materialization_path}: {error}") from error
    if hashlib.sha256(materialization_bytes).hexdigest() != parents["materialization_identity_sha256"]:
        raise CocoObjectProtocolConfirmationIdentityError("materialization parent identity file SHA256 mismatch")
    materialization_identity = _load_materialization_identity(materialization_path, repo_root=root)
    if materialization_identity["identity"]["name"] != parents["materialization_identity_name"]:
        raise CocoObjectProtocolConfirmationIdentityError("materialization parent identity name mismatch")

    # Cross-check the confirmation identity's own declared manifest schema
    # name (used at runtime to accept/reject the real manifest) against the
    # materialization identity's own authoritative manifest.schema_name --
    # these two fields must never be allowed to drift independently.
    verification = identity["verification"]
    manifest_schema_name = materialization_identity["manifest"]["schema_name"]
    if verification["materialization_manifest_schema_name"] != manifest_schema_name:
        raise CocoObjectProtocolConfirmationIdentityError(
            "verification.materialization_manifest_schema_name "
            f"{verification['materialization_manifest_schema_name']!r} does not match the materialization "
            f"parent identity's own manifest.schema_name {manifest_schema_name!r}"
        )
    materialization_protocol = materialization_identity["protocol"]
    if materialization_protocol["split"] != identity["protocol"]["split"]:
        raise CocoObjectProtocolConfirmationIdentityError(
            "materialization parent identity protocol.split disagrees with the confirmation identity's protocol.split"
        )
    if materialization_protocol["expected_image_count"] != verification["required_manifest_image_count"]:
        raise CocoObjectProtocolConfirmationIdentityError(
            "materialization parent identity protocol.expected_image_count disagrees with "
            "verification.required_manifest_image_count"
        )
    class_contract = materialization_identity["class_contract"]
    if class_contract["class_count"] != identity["dataset"]["class_count"]:
        raise CocoObjectProtocolConfirmationIdentityError(
            "materialization parent identity class_contract.class_count disagrees with dataset.class_count"
        )
    if class_contract["class_names_digest"] != identity["dataset"]["class_names_digest"]:
        raise CocoObjectProtocolConfirmationIdentityError(
            "materialization parent identity class_contract.class_names_digest disagrees with "
            "dataset.class_names_digest"
        )
    return materialization_identity


def validate_e3_configuration_binding(root: Path, identity: Mapping[str, Any]) -> Mapping[str, Any]:
    """Re-resolve the live COCO-Object E3 configuration (real YAML files,
    real inheritance, real DINOText constructor defaults -- never a
    hardcoded literal) and bind e3_config.template/with_bg_clean and the
    5 resolved_configuration hashes against it. CPU-only; safe before CUDA."""
    from src.e3_evaluation_identity import E3IdentityError, resolve_complete_e3_configuration
    from src.typed_configuration import typed_configuration_sha256

    e3_config = identity["e3_config"]
    adapter_identity = {
        "evaluation": {
            "config_path": e3_config["eval_config_relative_path"],
            "base_config_path": e3_config["eval_base_config_relative_path"],
        },
        "model": {
            "constructor_path": e3_config["model_constructor_relative_path"],
            "constructor_class": e3_config["constructor_class"],
        },
        "dataset": {"config_path": identity["dataset"]["dataset_config_relative_path"]},
        "projection": {"config_path": e3_config["projection_config_relative_path"]},
    }
    try:
        resolved = resolve_complete_e3_configuration(
            repo_root=root,
            identity=adapter_identity,
            eval_config=root / e3_config["eval_config_relative_path"],
            eval_base_config=root / e3_config["eval_base_config_relative_path"],
        )
    except E3IdentityError as error:
        raise CocoObjectProtocolConfirmationIdentityError(
            f"resolved E3 configuration mismatch: {error}"
        ) from error

    # Reuse the same hash computation resolve_complete_e3_configuration's own
    # sibling validator (validate_complete_e3_configuration) uses -- but skip
    # its configuration_sources check, which belongs to a different identity
    # schema this identity never records.
    expected = e3_config["resolved_configuration"]
    observed = {
        "full_sha256": typed_configuration_sha256(resolved["complete"]),
        "dataset_sha256": typed_configuration_sha256(resolved["dataset"]),
        "dataset_pipeline_sha256": typed_configuration_sha256(resolved["dataset_pipeline"]),
        "evaluation_sha256": typed_configuration_sha256(resolved["evaluation"]),
        "model_projection_sha256": typed_configuration_sha256(resolved["model_projection"]),
    }
    for name in observed:
        if observed[name] != expected[name]:
            raise CocoObjectProtocolConfirmationIdentityError(
                f"resolved E3 configuration mismatch at e3_config.resolved_configuration.{name}: "
                f"expected {expected[name]}, observed {observed[name]}"
            )

    runtime_template = resolved["evaluation"].get("template")
    if type(runtime_template) is not str or runtime_template != e3_config["template"]:
        raise CocoObjectProtocolConfirmationIdentityError(
            f"resolved runtime evaluate.template {runtime_template!r} does not match "
            f"identity e3_config.template {e3_config['template']!r}"
        )
    runtime_with_bg_clean = resolved["complete"]["runtime"]["model"].get("with_bg_clean")
    if (
        type(runtime_with_bg_clean) is not bool
        or type(e3_config["with_bg_clean"]) is not bool
        or runtime_with_bg_clean != e3_config["with_bg_clean"]
    ):
        raise CocoObjectProtocolConfirmationIdentityError(
            f"resolved runtime model.with_bg_clean {runtime_with_bg_clean!r} does not match "
            f"identity e3_config.with_bg_clean {e3_config['with_bg_clean']!r}"
        )
    return resolved


def validate_live_class_order(dataset: Any, identity: Mapping[str, Any]) -> str:
    """Validate the LIVE dataset's CLASSES against the identity's declared
    class count/background position/digest, before any model inference and
    before accepting the dataset for evaluation. Never normalizes, sorts,
    case-folds, or trims class names; never mutates the input sequence.
    Returns the recomputed digest for result/checkpoint/manifest provenance."""
    import hashlib
    import json

    resolved = dataset
    seen_ids: set[int] = set()
    for _ in range(8):
        if hasattr(resolved, "CLASSES"):
            break
        inner = getattr(resolved, "dataset", None)
        if inner is None or id(inner) in seen_ids:
            resolved = None
            break
        seen_ids.add(id(inner))
        resolved = inner
    if resolved is None or not hasattr(resolved, "CLASSES"):
        raise CocoObjectProtocolConfirmationIdentityError(
            "cannot resolve the underlying COCOObjectDataset (a CLASSES attribute) from the supplied dataset/wrapper"
        )

    classes = resolved.CLASSES
    if type(classes) is not tuple:
        raise CocoObjectProtocolConfirmationIdentityError(
            f"dataset.CLASSES must be an exact tuple, got {type(classes).__name__}"
        )
    canonical_classes = type(resolved).CLASSES
    if classes != canonical_classes:
        raise CocoObjectProtocolConfirmationIdentityError(
            "dataset.CLASSES has been overridden away from type(dataset).CLASSES -- refusing a "
            "custom/metadata-overridden class list"
        )
    for index, name in enumerate(classes):
        if type(name) is not str:
            raise CocoObjectProtocolConfirmationIdentityError(
                f"dataset.CLASSES[{index}] must be an exact str, got {type(name).__name__}"
            )

    dataset_block = identity["dataset"]
    if len(classes) != dataset_block["class_count"]:
        raise CocoObjectProtocolConfirmationIdentityError(
            f"live dataset.CLASSES has {len(classes)} entries; identity declares "
            f"dataset.class_count={dataset_block['class_count']}"
        )
    background_index = dataset_block["background_class_index"]
    if classes[background_index] != "background":
        raise CocoObjectProtocolConfirmationIdentityError(
            f"live dataset.CLASSES[{background_index}] must be exactly 'background'"
        )

    algorithm = dataset_block["class_names_digest_algorithm"]
    if algorithm != SUPPORTED_CLASS_NAMES_DIGEST_ALGORITHM:
        raise CocoObjectProtocolConfirmationIdentityError(
            f"dataset.class_names_digest_algorithm must be exactly {SUPPORTED_CLASS_NAMES_DIGEST_ALGORITHM!r}"
        )
    digest = hashlib.sha256(json.dumps(list(classes), ensure_ascii=True).encode("utf-8")).hexdigest()
    if digest != dataset_block["class_names_digest"]:
        raise CocoObjectProtocolConfirmationIdentityError(
            f"live dataset class-names digest {digest!r} disagrees with the registered "
            f"dataset.class_names_digest {dataset_block['class_names_digest']!r}"
        )
    return digest


def validate_static_configuration(
    *, repo_root: Path | None = None, identity_path: Path | None = None, check_git: bool = True
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    if check_git:
        _check_git_ancestry(root, identity["identity"]["required_ancestor_commit"], label="identity.required_ancestor_commit")
    matched_identity = _validate_matched_parent(root, identity)
    materialization_identity = _validate_materialization_parent(root, identity)
    validate_e3_configuration_binding(root, identity)

    return {
        "identity_name": identity["identity"]["name"],
        "matched_identity_name": matched_identity["identity"]["name"],
        "materialization_identity_name": materialization_identity["identity"]["name"],
        "required_ancestor_commit": identity["identity"]["required_ancestor_commit"],
    }


__all__ = [
    "IDENTITY_RELATIVE_PATH",
    "CocoObjectProtocolConfirmationIdentityError",
    "SUPPORTED_ANNOTATION_SUFFIX",
    "SUPPORTED_BACKGROUND_CLASS_INDEX",
    "SUPPORTED_CLASS_COUNT",
    "SUPPORTED_CLASS_NAMES_DIGEST_ALGORITHM",
    "SUPPORTED_DATASET_TYPE",
    "SUPPORTED_EXPECTED_IMAGE_COUNT",
    "SUPPORTED_FOREGROUND_CLASS_COUNT",
    "SUPPORTED_IGNORE_INDEX",
    "SUPPORTED_METRIC_UNIT",
    "SUPPORTED_MODEL_CONSTRUCTOR_CLASS",
    "SUPPORTED_PRIMARY_DELTA",
    "SUPPORTED_RUN_MODE_IMAGE_COUNTS",
    "SUPPORTED_SECONDARY_DELTAS",
    "SUPPORTED_SPLIT",
    "SUPPORTED_VARIANTS",
    "load_identity",
    "repository_root",
    "validate_e3_configuration_binding",
    "validate_live_class_order",
    "validate_static_configuration",
]
