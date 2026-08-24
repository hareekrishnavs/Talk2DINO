"""Static identity loading/validation for the COCO-Object val2017 mask
materialization stage.

This is a data-preparation identity, not a scientific-result identity: it
never loads CUDA, the model, or mmseg/mmcv dataset machinery. It records
and exact-type validates the sole authority for how the existing, already
-committed ``convert_dataset/convert_coco_object.py`` converter and
``COCOObjectDataset``/``coco.py`` dataset contract are reused to produce
the missing ``*_instanceTrainIds.png`` validation masks -- never a
redesigned mapping, threshold, or annotation-conversion algorithm.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e12_coco_object_val_materialization.toml")

SUPPORTED_SPLIT = "val2017"
SUPPORTED_EXPECTED_IMAGE_COUNT = 5000
SUPPORTED_COCO_LEN_TOTAL = 123287
SUPPORTED_TRAIN_MASK_COUNT = 118287
SUPPORTED_DATASET_CLASS_NAME = "COCOObjectDataset"
SUPPORTED_OUTPUT_MASK_SUFFIX = "_instanceTrainIds.png"
SUPPORTED_CLASS_COUNT = 81
SUPPORTED_BACKGROUND_CLASS_INDEX = 0
SUPPORTED_FOREGROUND_CLASS_COUNT = 80
SUPPORTED_DTYPE = "uint8"
SUPPORTED_DIRECTORY_STRUCTURE = ("images/val2017", "annotations/val2017", "manifests", "checkpoints")
SUPPORTED_IMAGES_RELATIONSHIP = "validated_symlink_to_canonical_source_images"
SUPPORTED_CHECKPOINT_SCHEMA_NAME = "talk2dino-coco-object-val-materialization-checkpoint-v1"
SUPPORTED_MANIFEST_SCHEMA_NAME = "talk2dino-coco-object-val-materialization-manifest-v1"
SUPPORTED_HASH_ALGORITHM = "sha256"
SUPPORTED_MAPPING_TABLE_SOURCE = "convert_dataset.convert_coco_object.clsID_to_trID"
SUPPORTED_MAPPING_TABLE_ENTRY_COUNT = 172
SUPPORTED_SOURCE_ANNOTATION_ROOT_MARKER = "coco_stuff164k/annotations"

IDENTITY_TOP_KEYS = frozenset(
    {
        "format_version", "identity", "protocol", "source", "converter", "class_contract",
        "label_mapping", "output", "policy", "checkpoint", "manifest", "hashing", "validation", "prohibited",
    }
)
IDENTITY_SECTION_KEYS = {
    "identity": frozenset({"name", "schema_version", "description", "required_ancestor_commit"}),
    "protocol": frozenset({"label", "statement", "split", "expected_image_count"}),
    "source": frozenset(
        {
            "description", "raw_mask_relative_root", "raw_mask_glob_suffix", "raw_mask_excluded_substrings",
            "canonical_images_relative_root", "image_suffix", "coco_len_total", "train_mask_count_expected",
            "val_mask_count_expected",
        }
    ),
    "converter": frozenset(
        {
            "canonical_converter_relative_path", "canonical_converter_sha256", "dataset_class_relative_path",
            "dataset_class_sha256", "dataset_config_relative_path", "dataset_config_sha256",
            "dataset_class_name", "output_mask_suffix",
        }
    ),
    "class_contract": frozenset(
        {"class_count", "background_class_index", "foreground_class_count", "class_names_digest", "class_names_source"}
    ),
    "label_mapping": frozenset(
        {
            "description", "mapping_table_source", "mapping_table_sha256", "mapping_table_entry_count",
            "raw_255_folds_to_background", "raw_domain_fully_covered_on_val2017", "crowd_or_overlap_handling",
        }
    ),
    "output": frozenset({"dtype", "directory_structure", "images_relationship"}),
    "policy": frozenset(
        {
            "train_conversion_prohibited", "writes_into_source_root_prohibited", "source_annotation_root_marker",
            "cuda_required", "model_required", "deterministic_output", "atomic_write_required",
            "resume_prefix_validation_required",
        }
    ),
    "checkpoint": frozenset({"schema_name"}),
    "manifest": frozenset({"schema_name"}),
    "hashing": frozenset({"algorithm"}),
    "validation": frozenset(
        {"require_decoded_pixel_hash", "require_encoded_png_hash", "require_ordered_image_digest", "overwrite_incomplete_requires_explicit_flag"}
    ),
    "prohibited": frozenset({"list"}),
}


class CocoObjectValMaterializationIdentityError(ValueError):
    """Raised when the materialization identity, checkpoint, or manifest
    fails any exact-type/schema/provenance check. Always fail closed:
    never silently substitute a default mapping, path, or split."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise CocoObjectValMaterializationIdentityError(f"{label} must be an exact non-empty string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise CocoObjectValMaterializationIdentityError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise CocoObjectValMaterializationIdentityError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise CocoObjectValMaterializationIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise CocoObjectValMaterializationIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", token) is None:
        raise CocoObjectValMaterializationIdentityError(f"{label} must be a full Git identity")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token:
        raise CocoObjectValMaterializationIdentityError(f"{label} must be a safe repository-relative path")
    return token


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise CocoObjectValMaterializationIdentityError(f"{label} has an unexpected schema")
    return value


def _require_exact_string_list(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise CocoObjectValMaterializationIdentityError(f"{label} must be a non-empty exact list")
    if any(type(item) is not str for item in value):
        raise CocoObjectValMaterializationIdentityError(f"{label} elements must be exact strings")
    return tuple(value)


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise CocoObjectValMaterializationIdentityError(
            f"cannot load coco-object-val-materialization identity {source}: {error}"
        ) from error

    if set(identity) != IDENTITY_TOP_KEYS:
        raise CocoObjectValMaterializationIdentityError("materialization identity has an unexpected top-level schema")
    _require_exact_string(identity["format_version"], "format_version")
    if identity["format_version"] != "talk2dino-coco-object-val-materialization-identity-v1":
        raise CocoObjectValMaterializationIdentityError("unsupported materialization identity format_version")
    for section, keys in IDENTITY_SECTION_KEYS.items():
        _require_closed_mapping(identity.get(section), keys, f"identity.{section}")

    block = identity["identity"]
    _require_exact_string(block["name"], "identity.name")
    _require_exact_string(block["schema_version"], "identity.schema_version")
    if block["schema_version"] != identity["format_version"]:
        raise CocoObjectValMaterializationIdentityError("identity.schema_version disagrees with format_version")
    _require_exact_string(block["description"], "identity.description")
    _require_git_identity(block["required_ancestor_commit"], "identity.required_ancestor_commit")

    protocol = identity["protocol"]
    _require_exact_string(protocol["label"], "protocol.label")
    if protocol["label"] != "COCO-Object":
        raise CocoObjectValMaterializationIdentityError("protocol.label must be exactly 'COCO-Object'")
    _require_exact_string(protocol["statement"], "protocol.statement")
    if protocol["split"] != SUPPORTED_SPLIT:
        raise CocoObjectValMaterializationIdentityError(f"protocol.split must be exactly {SUPPORTED_SPLIT!r}")
    if _require_exact_int(protocol["expected_image_count"], "protocol.expected_image_count") != SUPPORTED_EXPECTED_IMAGE_COUNT:
        raise CocoObjectValMaterializationIdentityError(f"protocol.expected_image_count must be exactly {SUPPORTED_EXPECTED_IMAGE_COUNT}")

    source_block = identity["source"]
    _require_exact_string(source_block["description"], "source.description")
    _require_relative_path(source_block["raw_mask_relative_root"], "source.raw_mask_relative_root")
    if source_block["raw_mask_glob_suffix"] != ".png":
        raise CocoObjectValMaterializationIdentityError("source.raw_mask_glob_suffix must be exactly '.png'")
    excluded = _require_exact_string_list(source_block["raw_mask_excluded_substrings"], "source.raw_mask_excluded_substrings")
    if set(excluded) != {"labelTrainIds", "instanceTrainIds"}:
        raise CocoObjectValMaterializationIdentityError("source.raw_mask_excluded_substrings must be exactly ['labelTrainIds', 'instanceTrainIds']")
    _require_relative_path(source_block["canonical_images_relative_root"], "source.canonical_images_relative_root")
    if source_block["image_suffix"] != ".jpg":
        raise CocoObjectValMaterializationIdentityError("source.image_suffix must be exactly '.jpg'")
    if _require_exact_int(source_block["coco_len_total"], "source.coco_len_total") != SUPPORTED_COCO_LEN_TOTAL:
        raise CocoObjectValMaterializationIdentityError(f"source.coco_len_total must be exactly {SUPPORTED_COCO_LEN_TOTAL}")
    if _require_exact_int(source_block["train_mask_count_expected"], "source.train_mask_count_expected") != SUPPORTED_TRAIN_MASK_COUNT:
        raise CocoObjectValMaterializationIdentityError(f"source.train_mask_count_expected must be exactly {SUPPORTED_TRAIN_MASK_COUNT}")
    if _require_exact_int(source_block["val_mask_count_expected"], "source.val_mask_count_expected") != SUPPORTED_EXPECTED_IMAGE_COUNT:
        raise CocoObjectValMaterializationIdentityError(f"source.val_mask_count_expected must be exactly {SUPPORTED_EXPECTED_IMAGE_COUNT}")
    if source_block["train_mask_count_expected"] + source_block["val_mask_count_expected"] != source_block["coco_len_total"]:
        raise CocoObjectValMaterializationIdentityError("source train/val mask counts must sum to coco_len_total")

    converter = identity["converter"]
    _require_relative_path(converter["canonical_converter_relative_path"], "converter.canonical_converter_relative_path")
    _require_sha256(converter["canonical_converter_sha256"], "converter.canonical_converter_sha256")
    _require_relative_path(converter["dataset_class_relative_path"], "converter.dataset_class_relative_path")
    _require_sha256(converter["dataset_class_sha256"], "converter.dataset_class_sha256")
    _require_relative_path(converter["dataset_config_relative_path"], "converter.dataset_config_relative_path")
    _require_sha256(converter["dataset_config_sha256"], "converter.dataset_config_sha256")
    if converter["dataset_class_name"] != SUPPORTED_DATASET_CLASS_NAME:
        raise CocoObjectValMaterializationIdentityError(f"converter.dataset_class_name must be exactly {SUPPORTED_DATASET_CLASS_NAME!r}")
    if converter["output_mask_suffix"] != SUPPORTED_OUTPUT_MASK_SUFFIX:
        raise CocoObjectValMaterializationIdentityError(f"converter.output_mask_suffix must be exactly {SUPPORTED_OUTPUT_MASK_SUFFIX!r}")

    class_contract = identity["class_contract"]
    if _require_exact_int(class_contract["class_count"], "class_contract.class_count") != SUPPORTED_CLASS_COUNT:
        raise CocoObjectValMaterializationIdentityError(f"class_contract.class_count must be exactly {SUPPORTED_CLASS_COUNT}")
    if _require_exact_int(class_contract["background_class_index"], "class_contract.background_class_index") != SUPPORTED_BACKGROUND_CLASS_INDEX:
        raise CocoObjectValMaterializationIdentityError("class_contract.background_class_index must be exactly 0")
    if _require_exact_int(class_contract["foreground_class_count"], "class_contract.foreground_class_count") != SUPPORTED_FOREGROUND_CLASS_COUNT:
        raise CocoObjectValMaterializationIdentityError(f"class_contract.foreground_class_count must be exactly {SUPPORTED_FOREGROUND_CLASS_COUNT}")
    if class_contract["background_class_index"] and class_contract["class_count"] != class_contract["foreground_class_count"] + 1:
        raise CocoObjectValMaterializationIdentityError("class_contract.class_count must equal foreground_class_count + 1")
    _require_sha256(class_contract["class_names_digest"], "class_contract.class_names_digest")
    _require_exact_string(class_contract["class_names_source"], "class_contract.class_names_source")

    label_mapping = identity["label_mapping"]
    _require_exact_string(label_mapping["description"], "label_mapping.description")
    if label_mapping["mapping_table_source"] != SUPPORTED_MAPPING_TABLE_SOURCE:
        raise CocoObjectValMaterializationIdentityError(f"label_mapping.mapping_table_source must be exactly {SUPPORTED_MAPPING_TABLE_SOURCE!r}")
    _require_sha256(label_mapping["mapping_table_sha256"], "label_mapping.mapping_table_sha256")
    if _require_exact_int(label_mapping["mapping_table_entry_count"], "label_mapping.mapping_table_entry_count") != SUPPORTED_MAPPING_TABLE_ENTRY_COUNT:
        raise CocoObjectValMaterializationIdentityError(f"label_mapping.mapping_table_entry_count must be exactly {SUPPORTED_MAPPING_TABLE_ENTRY_COUNT}")
    if _require_exact_bool(label_mapping["raw_255_folds_to_background"], "label_mapping.raw_255_folds_to_background") is not True:
        raise CocoObjectValMaterializationIdentityError("label_mapping.raw_255_folds_to_background must be true")
    if _require_exact_bool(label_mapping["raw_domain_fully_covered_on_val2017"], "label_mapping.raw_domain_fully_covered_on_val2017") is not True:
        raise CocoObjectValMaterializationIdentityError("label_mapping.raw_domain_fully_covered_on_val2017 must be true")
    _require_exact_string(label_mapping["crowd_or_overlap_handling"], "label_mapping.crowd_or_overlap_handling")

    output = identity["output"]
    if output["dtype"] != SUPPORTED_DTYPE:
        raise CocoObjectValMaterializationIdentityError(f"output.dtype must be exactly {SUPPORTED_DTYPE!r}")
    if _require_exact_string_list(output["directory_structure"], "output.directory_structure") != SUPPORTED_DIRECTORY_STRUCTURE:
        raise CocoObjectValMaterializationIdentityError(f"output.directory_structure must be exactly {list(SUPPORTED_DIRECTORY_STRUCTURE)}")
    if output["images_relationship"] != SUPPORTED_IMAGES_RELATIONSHIP:
        raise CocoObjectValMaterializationIdentityError(f"output.images_relationship must be exactly {SUPPORTED_IMAGES_RELATIONSHIP!r}")

    policy = identity["policy"]
    for flag in (
        "train_conversion_prohibited", "writes_into_source_root_prohibited", "deterministic_output",
        "atomic_write_required", "resume_prefix_validation_required",
    ):
        if _require_exact_bool(policy[flag], f"policy.{flag}") is not True:
            raise CocoObjectValMaterializationIdentityError(f"policy.{flag} must be true")
    for flag in ("cuda_required", "model_required"):
        if _require_exact_bool(policy[flag], f"policy.{flag}") is not False:
            raise CocoObjectValMaterializationIdentityError(f"policy.{flag} must be false -- this stage is offline data preparation")
    if policy["source_annotation_root_marker"] != SUPPORTED_SOURCE_ANNOTATION_ROOT_MARKER:
        raise CocoObjectValMaterializationIdentityError(f"policy.source_annotation_root_marker must be exactly {SUPPORTED_SOURCE_ANNOTATION_ROOT_MARKER!r}")

    checkpoint_block = identity["checkpoint"]
    if checkpoint_block["schema_name"] != SUPPORTED_CHECKPOINT_SCHEMA_NAME:
        raise CocoObjectValMaterializationIdentityError(f"checkpoint.schema_name must be exactly {SUPPORTED_CHECKPOINT_SCHEMA_NAME!r}")

    manifest_block = identity["manifest"]
    if manifest_block["schema_name"] != SUPPORTED_MANIFEST_SCHEMA_NAME:
        raise CocoObjectValMaterializationIdentityError(f"manifest.schema_name must be exactly {SUPPORTED_MANIFEST_SCHEMA_NAME!r}")

    hashing = identity["hashing"]
    if hashing["algorithm"] != SUPPORTED_HASH_ALGORITHM:
        raise CocoObjectValMaterializationIdentityError(f"hashing.algorithm must be exactly {SUPPORTED_HASH_ALGORITHM!r}")

    validation = identity["validation"]
    for flag in (
        "require_decoded_pixel_hash", "require_encoded_png_hash", "require_ordered_image_digest",
        "overwrite_incomplete_requires_explicit_flag",
    ):
        if _require_exact_bool(validation[flag], f"validation.{flag}") is not True:
            raise CocoObjectValMaterializationIdentityError(f"validation.{flag} must be true")

    _require_exact_string_list(identity["prohibited"]["list"], "prohibited.list")

    return identity


def _check_git_ancestry(root: Path, commit: str, *, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"{label} is not an ancestor of HEAD"
        raise CocoObjectValMaterializationIdentityError(f"Git ancestry check failed for {label} ({commit}): {detail}")


def _validate_converter_provenance(root: Path, identity: Mapping[str, Any]) -> None:
    import hashlib

    converter = identity["converter"]
    for field, sha_field in (
        ("canonical_converter_relative_path", "canonical_converter_sha256"),
        ("dataset_class_relative_path", "dataset_class_sha256"),
        ("dataset_config_relative_path", "dataset_config_sha256"),
    ):
        path = root / converter[field]
        try:
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise CocoObjectValMaterializationIdentityError(f"cannot read converter provenance file {path}: {error}") from error
        if observed != converter[sha_field]:
            raise CocoObjectValMaterializationIdentityError(
                f"{field} SHA256 mismatch: identity declares {converter[sha_field]}, observed {observed} at {path}"
            )


def validate_static_configuration(
    *, repo_root: Path | None = None, identity_path: Path | None = None, check_git: bool = True
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    if check_git:
        _check_git_ancestry(root, identity["identity"]["required_ancestor_commit"], label="identity.required_ancestor_commit")
    _validate_converter_provenance(root, identity)

    return {
        "identity_name": identity["identity"]["name"],
        "split": identity["protocol"]["split"],
        "expected_image_count": identity["protocol"]["expected_image_count"],
        "required_ancestor_commit": identity["identity"]["required_ancestor_commit"],
    }


__all__ = [
    "IDENTITY_RELATIVE_PATH",
    "CocoObjectValMaterializationIdentityError",
    "SUPPORTED_BACKGROUND_CLASS_INDEX",
    "SUPPORTED_CHECKPOINT_SCHEMA_NAME",
    "SUPPORTED_CLASS_COUNT",
    "SUPPORTED_COCO_LEN_TOTAL",
    "SUPPORTED_DATASET_CLASS_NAME",
    "SUPPORTED_DTYPE",
    "SUPPORTED_EXPECTED_IMAGE_COUNT",
    "SUPPORTED_FOREGROUND_CLASS_COUNT",
    "SUPPORTED_HASH_ALGORITHM",
    "SUPPORTED_MANIFEST_SCHEMA_NAME",
    "SUPPORTED_MAPPING_TABLE_ENTRY_COUNT",
    "SUPPORTED_MAPPING_TABLE_SOURCE",
    "SUPPORTED_OUTPUT_MASK_SUFFIX",
    "SUPPORTED_SPLIT",
    "SUPPORTED_TRAIN_MASK_COUNT",
    "load_identity",
    "repository_root",
    "validate_static_configuration",
]
