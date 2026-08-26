"""Static identity loading/validation for the shared PASCAL VOC 2012
dataset-source contract underlying Talk2DINO's V20 (no-background) and
V21 (with-background) protocols.

This is a data-source identity, not a scientific-result identity: it
never loads CUDA, the model, or a live dataset. It records and exact-
type validates the sole authority for the archive layout, the shared
validation split, the class contract, and each protocol's label
transform -- reusing the already-committed V20/V21 dataset-config/class
source files (and, for V21, the installed mmsegmentation package) as the
ground truth, never a redesigned or re-derived mapping.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e12_voc2012_dataset_source.toml")

SUPPORTED_SPLIT = "val"
SUPPORTED_EXPECTED_IMAGE_COUNT = 1449
SUPPORTED_V21_CLASS_COUNT = 21
SUPPORTED_V20_CLASS_COUNT = 20
SUPPORTED_BACKGROUND_CLASS_INDEX = 0
SUPPORTED_IGNORE_INDEX = 255
SUPPORTED_RAW_IGNORE_VALUE = 255
SUPPORTED_IMAGE_SUFFIX = ".jpg"
SUPPORTED_ANNOTATION_SUFFIX = ".png"
SUPPORTED_V20_DATASET_CLASS_NAME = "PascalVOCDataset20"
SUPPORTED_V21_DATASET_CLASS_NAME = "PascalVOCDataset"
SUPPORTED_MMSEGMENTATION_VERSION = "0.30.0"
SUPPORTED_MANIFEST_SCHEMA_NAME = "talk2dino-voc2012-dataset-manifest-v1"
SUPPORTED_HASH_ALGORITHM = "sha256"
SUPPORTED_CLASS_NAMES_DIGEST_ALGORITHM = "sha256_json_dumps_ensure_ascii_list"
SUPPORTED_CROP_SIZE = (448, 448)
SUPPORTED_STRIDE = (224, 224)
SUPPORTED_MODE = "slide"

IDENTITY_TOP_KEYS = frozenset(
    {
        "format_version", "identity", "upstream", "protocol", "expected_root_structure", "source",
        "dataset_loader", "class_contract", "label_policy", "evaluation_pipeline", "provenance_policy",
        "manifest", "hashing", "validation", "prohibited",
    }
)
IDENTITY_SECTION_KEYS = {
    "identity": frozenset({"name", "schema_version", "description", "required_ancestor_commit"}),
    "upstream": frozenset({"dataset_name", "dataset_version", "source_url", "source_reference", "archive_top_level_directory"}),
    "protocol": frozenset({"label", "statement", "split", "expected_image_count"}),
    "expected_root_structure": frozenset({"description", "voc2012_relative_root", "required_subpaths"}),
    "source": frozenset(
        {"image_relative_root", "annotation_relative_root", "split_relative_path", "image_suffix", "annotation_suffix", "split_order_authority"}
    ),
    "dataset_loader": frozenset(
        {
            "v21_dataset_config_relative_path", "v21_dataset_config_sha256", "v20_dataset_config_relative_path",
            "v20_dataset_config_sha256", "v20_dataset_class_relative_path", "v20_dataset_class_sha256",
            "v20_dataset_class_name", "v21_dataset_class_name", "v21_dataset_class_source",
            "mmsegmentation_required_version", "data_root_relative_path",
        }
    ),
    "class_contract": frozenset(
        {
            "v21_class_count", "v20_class_count", "background_class_index", "v21_class_names_digest",
            "v20_class_names_digest", "class_names_digest_algorithm", "class_names_relationship",
        }
    ),
    "label_policy": frozenset(
        {
            "raw_ignore_value", "ignore_index", "v21_uses_raw_labels_directly", "v21_reduce_zero_label",
            "v20_reduce_zero_label", "v20_transform_description", "v21_transform_description",
        }
    ),
    "evaluation_pipeline": frozenset({"description", "mode", "crop_size", "stride"}),
    "provenance_policy": frozenset({"v20_v21_share_same_source_files", "conversion_required", "no_bridge_training_or_feature_extraction_allowed", "statement"}),
    "manifest": frozenset({"schema_name"}),
    "hashing": frozenset({"algorithm"}),
    "validation": frozenset(
        {
            "require_decoded_pixel_hash", "require_encoded_png_hash", "require_ordered_image_digest",
            "require_dimension_reconciliation", "require_label_set_reconciliation",
            "overwrite_incomplete_requires_explicit_flag",
        }
    ),
    "prohibited": frozenset({"list"}),
}


class Voc2012DatasetIdentityError(ValueError):
    """Raised when the VOC2012 dataset-source identity, or the live
    dataset/loader it governs, fails any exact-type/schema/relational
    check. Always fail closed: never silently substitute a default
    split, class list, or label transform."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value.strip()):
        raise Voc2012DatasetIdentityError(f"{label} must be an exact non-empty string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise Voc2012DatasetIdentityError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise Voc2012DatasetIdentityError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise Voc2012DatasetIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise Voc2012DatasetIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", token) is None:
        raise Voc2012DatasetIdentityError(f"{label} must be a full Git identity")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token:
        raise Voc2012DatasetIdentityError(f"{label} must be a safe repository-relative path")
    return token


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise Voc2012DatasetIdentityError(f"{label} has an unexpected schema")
    return value


def _require_exact_string_list(value: Any, label: str, *, nonempty: bool = True) -> tuple[str, ...]:
    if type(value) is not list or (nonempty and not value):
        raise Voc2012DatasetIdentityError(f"{label} must be a non-empty exact list")
    if any(type(item) is not str for item in value):
        raise Voc2012DatasetIdentityError(f"{label} elements must be exact strings")
    return tuple(value)


def _require_exact_int_pair(value: Any, label: str) -> tuple[int, int]:
    if type(value) is not list or len(value) != 2:
        raise Voc2012DatasetIdentityError(f"{label} must be an exact two-element list")
    if any(type(item) is not int for item in value):
        raise Voc2012DatasetIdentityError(f"{label} elements must be exact integers")
    return (value[0], value[1])


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise Voc2012DatasetIdentityError(f"cannot load voc2012 dataset-source identity {source}: {error}") from error

    if set(identity) != IDENTITY_TOP_KEYS:
        raise Voc2012DatasetIdentityError("voc2012 dataset-source identity has an unexpected top-level schema")
    _require_exact_string(identity["format_version"], "format_version")
    if identity["format_version"] != "talk2dino-voc2012-dataset-source-identity-v1":
        raise Voc2012DatasetIdentityError("unsupported voc2012 dataset-source identity format_version")
    for section, keys in IDENTITY_SECTION_KEYS.items():
        _require_closed_mapping(identity.get(section), keys, f"identity.{section}")

    block = identity["identity"]
    _require_exact_string(block["name"], "identity.name")
    _require_exact_string(block["schema_version"], "identity.schema_version")
    if block["schema_version"] != identity["format_version"]:
        raise Voc2012DatasetIdentityError("identity.schema_version disagrees with format_version")
    _require_exact_string(block["description"], "identity.description")
    _require_git_identity(block["required_ancestor_commit"], "identity.required_ancestor_commit")

    upstream = identity["upstream"]
    if upstream["dataset_name"] != "PASCAL VOC 2012":
        raise Voc2012DatasetIdentityError("upstream.dataset_name must be exactly 'PASCAL VOC 2012'")
    _require_exact_string(upstream["dataset_version"], "upstream.dataset_version")
    _require_exact_string(upstream["source_url"], "upstream.source_url")
    _require_exact_string(upstream["source_reference"], "upstream.source_reference")
    if upstream["archive_top_level_directory"] != "VOCdevkit":
        raise Voc2012DatasetIdentityError("upstream.archive_top_level_directory must be exactly 'VOCdevkit'")

    protocol = identity["protocol"]
    if protocol["label"] != "VOC2012":
        raise Voc2012DatasetIdentityError("protocol.label must be exactly 'VOC2012'")
    _require_exact_string(protocol["statement"], "protocol.statement")
    if protocol["split"] != SUPPORTED_SPLIT:
        raise Voc2012DatasetIdentityError(f"protocol.split must be exactly {SUPPORTED_SPLIT!r}")
    if _require_exact_int(protocol["expected_image_count"], "protocol.expected_image_count") != SUPPORTED_EXPECTED_IMAGE_COUNT:
        raise Voc2012DatasetIdentityError(f"protocol.expected_image_count must be exactly {SUPPORTED_EXPECTED_IMAGE_COUNT}")

    root_structure = identity["expected_root_structure"]
    _require_exact_string(root_structure["description"], "expected_root_structure.description")
    if root_structure["voc2012_relative_root"] != "VOC2012":
        raise Voc2012DatasetIdentityError("expected_root_structure.voc2012_relative_root must be exactly 'VOC2012'")
    required_subpaths = _require_exact_string_list(root_structure["required_subpaths"], "expected_root_structure.required_subpaths")
    if set(required_subpaths) != {"JPEGImages", "SegmentationClass", "ImageSets/Segmentation/train.txt", "ImageSets/Segmentation/val.txt", "ImageSets/Segmentation/trainval.txt"}:
        raise Voc2012DatasetIdentityError("expected_root_structure.required_subpaths must be exactly the canonical VOC2012 subpath set")

    source_block = identity["source"]
    if source_block["image_relative_root"] != "JPEGImages":
        raise Voc2012DatasetIdentityError("source.image_relative_root must be exactly 'JPEGImages'")
    if source_block["annotation_relative_root"] != "SegmentationClass":
        raise Voc2012DatasetIdentityError("source.annotation_relative_root must be exactly 'SegmentationClass'")
    if source_block["split_relative_path"] != "ImageSets/Segmentation/val.txt":
        raise Voc2012DatasetIdentityError("source.split_relative_path must be exactly 'ImageSets/Segmentation/val.txt'")
    if source_block["image_suffix"] != SUPPORTED_IMAGE_SUFFIX:
        raise Voc2012DatasetIdentityError(f"source.image_suffix must be exactly {SUPPORTED_IMAGE_SUFFIX!r}")
    if source_block["annotation_suffix"] != SUPPORTED_ANNOTATION_SUFFIX:
        raise Voc2012DatasetIdentityError(f"source.annotation_suffix must be exactly {SUPPORTED_ANNOTATION_SUFFIX!r}")
    _require_exact_string(source_block["split_order_authority"], "source.split_order_authority")

    loader = identity["dataset_loader"]
    _require_relative_path(loader["v21_dataset_config_relative_path"], "dataset_loader.v21_dataset_config_relative_path")
    _require_sha256(loader["v21_dataset_config_sha256"], "dataset_loader.v21_dataset_config_sha256")
    _require_relative_path(loader["v20_dataset_config_relative_path"], "dataset_loader.v20_dataset_config_relative_path")
    _require_sha256(loader["v20_dataset_config_sha256"], "dataset_loader.v20_dataset_config_sha256")
    _require_relative_path(loader["v20_dataset_class_relative_path"], "dataset_loader.v20_dataset_class_relative_path")
    _require_sha256(loader["v20_dataset_class_sha256"], "dataset_loader.v20_dataset_class_sha256")
    if loader["v20_dataset_class_name"] != SUPPORTED_V20_DATASET_CLASS_NAME:
        raise Voc2012DatasetIdentityError(f"dataset_loader.v20_dataset_class_name must be exactly {SUPPORTED_V20_DATASET_CLASS_NAME!r}")
    if loader["v21_dataset_class_name"] != SUPPORTED_V21_DATASET_CLASS_NAME:
        raise Voc2012DatasetIdentityError(f"dataset_loader.v21_dataset_class_name must be exactly {SUPPORTED_V21_DATASET_CLASS_NAME!r}")
    _require_exact_string(loader["v21_dataset_class_source"], "dataset_loader.v21_dataset_class_source")
    if loader["mmsegmentation_required_version"] != SUPPORTED_MMSEGMENTATION_VERSION:
        raise Voc2012DatasetIdentityError(f"dataset_loader.mmsegmentation_required_version must be exactly {SUPPORTED_MMSEGMENTATION_VERSION!r}")
    if loader["data_root_relative_path"] != "data/VOCdevkit/VOC2012":
        raise Voc2012DatasetIdentityError("dataset_loader.data_root_relative_path must be exactly 'data/VOCdevkit/VOC2012'")

    class_contract = identity["class_contract"]
    if _require_exact_int(class_contract["v21_class_count"], "class_contract.v21_class_count") != SUPPORTED_V21_CLASS_COUNT:
        raise Voc2012DatasetIdentityError(f"class_contract.v21_class_count must be exactly {SUPPORTED_V21_CLASS_COUNT}")
    if _require_exact_int(class_contract["v20_class_count"], "class_contract.v20_class_count") != SUPPORTED_V20_CLASS_COUNT:
        raise Voc2012DatasetIdentityError(f"class_contract.v20_class_count must be exactly {SUPPORTED_V20_CLASS_COUNT}")
    if class_contract["v21_class_count"] != class_contract["v20_class_count"] + 1:
        raise Voc2012DatasetIdentityError("class_contract.v21_class_count must equal v20_class_count + 1")
    if _require_exact_int(class_contract["background_class_index"], "class_contract.background_class_index") != SUPPORTED_BACKGROUND_CLASS_INDEX:
        raise Voc2012DatasetIdentityError("class_contract.background_class_index must be exactly 0")
    _require_sha256(class_contract["v21_class_names_digest"], "class_contract.v21_class_names_digest")
    _require_sha256(class_contract["v20_class_names_digest"], "class_contract.v20_class_names_digest")
    if class_contract["class_names_digest_algorithm"] != SUPPORTED_CLASS_NAMES_DIGEST_ALGORITHM:
        raise Voc2012DatasetIdentityError(f"class_contract.class_names_digest_algorithm must be exactly {SUPPORTED_CLASS_NAMES_DIGEST_ALGORITHM!r}")
    _require_exact_string(class_contract["class_names_relationship"], "class_contract.class_names_relationship")

    label_policy = identity["label_policy"]
    if _require_exact_int(label_policy["raw_ignore_value"], "label_policy.raw_ignore_value") != SUPPORTED_RAW_IGNORE_VALUE:
        raise Voc2012DatasetIdentityError(f"label_policy.raw_ignore_value must be exactly {SUPPORTED_RAW_IGNORE_VALUE}")
    if _require_exact_int(label_policy["ignore_index"], "label_policy.ignore_index") != SUPPORTED_IGNORE_INDEX:
        raise Voc2012DatasetIdentityError(f"label_policy.ignore_index must be exactly {SUPPORTED_IGNORE_INDEX}")
    if _require_exact_bool(label_policy["v21_uses_raw_labels_directly"], "label_policy.v21_uses_raw_labels_directly") is not True:
        raise Voc2012DatasetIdentityError("label_policy.v21_uses_raw_labels_directly must be true")
    if _require_exact_bool(label_policy["v21_reduce_zero_label"], "label_policy.v21_reduce_zero_label") is not False:
        raise Voc2012DatasetIdentityError("label_policy.v21_reduce_zero_label must be false")
    if _require_exact_bool(label_policy["v20_reduce_zero_label"], "label_policy.v20_reduce_zero_label") is not True:
        raise Voc2012DatasetIdentityError("label_policy.v20_reduce_zero_label must be true")
    _require_exact_string(label_policy["v20_transform_description"], "label_policy.v20_transform_description")
    _require_exact_string(label_policy["v21_transform_description"], "label_policy.v21_transform_description")

    pipeline = identity["evaluation_pipeline"]
    _require_exact_string(pipeline["description"], "evaluation_pipeline.description")
    if pipeline["mode"] != SUPPORTED_MODE:
        raise Voc2012DatasetIdentityError(f"evaluation_pipeline.mode must be exactly {SUPPORTED_MODE!r}")
    if _require_exact_int_pair(pipeline["crop_size"], "evaluation_pipeline.crop_size") != SUPPORTED_CROP_SIZE:
        raise Voc2012DatasetIdentityError(f"evaluation_pipeline.crop_size must be exactly {list(SUPPORTED_CROP_SIZE)}")
    if _require_exact_int_pair(pipeline["stride"], "evaluation_pipeline.stride") != SUPPORTED_STRIDE:
        raise Voc2012DatasetIdentityError(f"evaluation_pipeline.stride must be exactly {list(SUPPORTED_STRIDE)}")

    provenance = identity["provenance_policy"]
    if _require_exact_bool(provenance["v20_v21_share_same_source_files"], "provenance_policy.v20_v21_share_same_source_files") is not True:
        raise Voc2012DatasetIdentityError("provenance_policy.v20_v21_share_same_source_files must be true")
    _require_exact_bool(provenance["conversion_required"], "provenance_policy.conversion_required")
    if _require_exact_bool(provenance["no_bridge_training_or_feature_extraction_allowed"], "provenance_policy.no_bridge_training_or_feature_extraction_allowed") is not True:
        raise Voc2012DatasetIdentityError("provenance_policy.no_bridge_training_or_feature_extraction_allowed must be true")
    _require_exact_string(provenance["statement"], "provenance_policy.statement")

    manifest_block = identity["manifest"]
    if manifest_block["schema_name"] != SUPPORTED_MANIFEST_SCHEMA_NAME:
        raise Voc2012DatasetIdentityError(f"manifest.schema_name must be exactly {SUPPORTED_MANIFEST_SCHEMA_NAME!r}")

    hashing = identity["hashing"]
    if hashing["algorithm"] != SUPPORTED_HASH_ALGORITHM:
        raise Voc2012DatasetIdentityError(f"hashing.algorithm must be exactly {SUPPORTED_HASH_ALGORITHM!r}")

    validation = identity["validation"]
    for flag in (
        "require_decoded_pixel_hash", "require_encoded_png_hash", "require_ordered_image_digest",
        "require_dimension_reconciliation", "require_label_set_reconciliation",
        "overwrite_incomplete_requires_explicit_flag",
    ):
        if _require_exact_bool(validation[flag], f"validation.{flag}") is not True:
            raise Voc2012DatasetIdentityError(f"validation.{flag} must be true")

    _require_exact_string_list(identity["prohibited"]["list"], "prohibited.list")

    return identity


def _check_git_ancestry(root: Path, commit: str, *, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"{label} is not an ancestor of HEAD"
        raise Voc2012DatasetIdentityError(f"Git ancestry check failed for {label} ({commit}): {detail}")


def _validate_loader_provenance(root: Path, identity: Mapping[str, Any]) -> None:
    import hashlib

    loader = identity["dataset_loader"]
    for field, sha_field in (
        ("v21_dataset_config_relative_path", "v21_dataset_config_sha256"),
        ("v20_dataset_config_relative_path", "v20_dataset_config_sha256"),
        ("v20_dataset_class_relative_path", "v20_dataset_class_sha256"),
    ):
        path = root / loader[field]
        try:
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise Voc2012DatasetIdentityError(f"cannot read dataset-loader provenance file {path}: {error}") from error
        if observed != loader[sha_field]:
            raise Voc2012DatasetIdentityError(
                f"{field} SHA256 mismatch: identity declares {loader[sha_field]}, observed {observed} at {path}"
            )


def validate_static_configuration(
    *, repo_root: Path | None = None, identity_path: Path | None = None, check_git: bool = True
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    if check_git:
        _check_git_ancestry(root, identity["identity"]["required_ancestor_commit"], label="identity.required_ancestor_commit")
    _validate_loader_provenance(root, identity)

    return {
        "identity_name": identity["identity"]["name"],
        "split": identity["protocol"]["split"],
        "expected_image_count": identity["protocol"]["expected_image_count"],
        "required_ancestor_commit": identity["identity"]["required_ancestor_commit"],
    }


def validate_v21_class_contract_against_installed_mmseg(identity: Mapping[str, Any]) -> tuple[str, ...]:
    """Live-import mmseg (CPU-only; no CUDA/model/dataset construction)
    and require its stock PascalVOCDataset.CLASSES to exactly equal the
    21-tuple this identity's v21_class_names_digest was computed from.
    mmsegmentation is an installed package, not vendored in this
    repository, so it cannot be git-blob-hash-pinned like the V20
    dataset-loader files; this direct value comparison is the
    authoritative equivalent."""
    import hashlib
    import json

    try:
        import mmseg
        from mmseg.datasets import PascalVOCDataset
    except ImportError as error:
        raise Voc2012DatasetIdentityError(f"cannot import mmseg.datasets.PascalVOCDataset: {error}") from error

    installed_version = getattr(mmseg, "__version__", None)
    required_version = identity["dataset_loader"]["mmsegmentation_required_version"]
    if installed_version != required_version:
        raise Voc2012DatasetIdentityError(
            f"installed mmsegmentation version {installed_version!r} does not match "
            f"dataset_loader.mmsegmentation_required_version {required_version!r}"
        )

    classes = PascalVOCDataset.CLASSES
    if type(classes) is not tuple or any(type(name) is not str for name in classes):
        raise Voc2012DatasetIdentityError("mmseg.datasets.PascalVOCDataset.CLASSES must be an exact tuple of exact strings")
    digest = hashlib.sha256(json.dumps(list(classes), ensure_ascii=True).encode("utf-8")).hexdigest()
    if digest != identity["class_contract"]["v21_class_names_digest"]:
        raise Voc2012DatasetIdentityError(
            f"live mmseg.datasets.PascalVOCDataset.CLASSES digest {digest!r} disagrees with "
            f"class_contract.v21_class_names_digest {identity['class_contract']['v21_class_names_digest']!r}"
        )
    return classes


def validate_v20_class_contract_against_source(root: Path, identity: Mapping[str, Any]) -> tuple[str, ...]:
    """Independently discover PascalVOCDataset20.CLASSES via AST over the
    already hash-pinned dataset-class source file -- never by importing
    the implementation's own class (which would require the full mmseg
    registry machinery) and never by re-declaring the tuple as a second
    Python literal in this module."""
    import ast
    import hashlib
    import json

    path = root / identity["dataset_loader"]["v20_dataset_class_relative_path"]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise Voc2012DatasetIdentityError(f"cannot read V20 dataset class source {path}: {error}") from error

    class_name = identity["dataset_loader"]["v20_dataset_class_name"]
    class_def = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if class_def is None:
        raise Voc2012DatasetIdentityError(f"class {class_name!r} not found in {path}")
    assign = next(
        (
            n for n in class_def.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CLASSES" for t in n.targets)
        ),
        None,
    )
    if assign is None:
        raise Voc2012DatasetIdentityError(f"{class_name}.CLASSES assignment not found in {path}")
    classes = ast.literal_eval(assign.value)
    if type(classes) is not tuple or any(type(name) is not str for name in classes):
        raise Voc2012DatasetIdentityError(f"{class_name}.CLASSES must be an exact tuple of exact strings")

    digest = hashlib.sha256(json.dumps(list(classes), ensure_ascii=True).encode("utf-8")).hexdigest()
    if digest != identity["class_contract"]["v20_class_names_digest"]:
        raise Voc2012DatasetIdentityError(
            f"source-discovered {class_name}.CLASSES digest {digest!r} disagrees with "
            f"class_contract.v20_class_names_digest {identity['class_contract']['v20_class_names_digest']!r}"
        )
    return classes


__all__ = [
    "IDENTITY_RELATIVE_PATH",
    "Voc2012DatasetIdentityError",
    "SUPPORTED_ANNOTATION_SUFFIX",
    "SUPPORTED_BACKGROUND_CLASS_INDEX",
    "SUPPORTED_CROP_SIZE",
    "SUPPORTED_EXPECTED_IMAGE_COUNT",
    "SUPPORTED_HASH_ALGORITHM",
    "SUPPORTED_IGNORE_INDEX",
    "SUPPORTED_IMAGE_SUFFIX",
    "SUPPORTED_MANIFEST_SCHEMA_NAME",
    "SUPPORTED_MMSEGMENTATION_VERSION",
    "SUPPORTED_MODE",
    "SUPPORTED_RAW_IGNORE_VALUE",
    "SUPPORTED_SPLIT",
    "SUPPORTED_STRIDE",
    "SUPPORTED_V20_CLASS_COUNT",
    "SUPPORTED_V20_DATASET_CLASS_NAME",
    "SUPPORTED_V21_CLASS_COUNT",
    "SUPPORTED_V21_DATASET_CLASS_NAME",
    "load_identity",
    "repository_root",
    "validate_static_configuration",
    "validate_v20_class_contract_against_source",
    "validate_v21_class_contract_against_installed_mmseg",
]
