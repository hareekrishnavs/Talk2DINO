"""Static identity loading/validation for the shared ADE20K
(ADEChallengeData2016) validation source contract.

Data-source identity, not a scientific-result identity: never loads
CUDA, the model, or a live dataset. Records and exact-type validates the
sole authority for the archive layout, the validation split, the C150
class contract, and the raw-label transform -- reusing the
already-committed ade20k.py dataset-config file and the installed
mmsegmentation ADE20KDataset class as the ground truth, never a
redesigned or re-derived mapping.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e12_ade20k_dataset_source.toml")

SUPPORTED_SPLIT = "validation"
SUPPORTED_EXPECTED_IMAGE_COUNT = 2000
SUPPORTED_CLASS_COUNT = 150
SUPPORTED_IGNORE_INDEX = 255
SUPPORTED_RAW_IGNORE_VALUE = 0
SUPPORTED_IMAGE_SUFFIX = ".jpg"
SUPPORTED_ANNOTATION_SUFFIX = ".png"
SUPPORTED_DATASET_CLASS_NAME = "ADE20KDataset"
SUPPORTED_MMSEGMENTATION_VERSION = "0.30.0"
SUPPORTED_MANIFEST_SCHEMA_NAME = "talk2dino-ade20k-dataset-manifest-v1"
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
    "expected_root_structure": frozenset({"description", "ade_relative_root", "required_subpaths"}),
    "source": frozenset(
        {
            "image_relative_root", "annotation_relative_root", "training_image_relative_root",
            "training_annotation_relative_root", "image_suffix", "annotation_suffix", "split_order_authority",
        }
    ),
    "dataset_loader": frozenset(
        {
            "dataset_config_relative_path", "dataset_config_sha256", "dataset_class_name",
            "dataset_class_source", "mmsegmentation_required_version", "data_root_relative_path",
        }
    ),
    "class_contract": frozenset({"class_count", "class_names_digest", "class_names_digest_algorithm"}),
    "label_policy": frozenset(
        {"raw_ignore_value", "ignore_index", "reduce_zero_label", "transform_description", "background_class_evaluated"}
    ),
    "evaluation_pipeline": frozenset({"description", "mode", "crop_size", "stride"}),
    "provenance_policy": frozenset({"conversion_required", "no_bridge_training_or_feature_extraction_allowed", "statement"}),
    "manifest": frozenset({"schema_name"}),
    "hashing": frozenset({"algorithm"}),
    "validation": frozenset(
        {
            "require_decoded_pixel_hash", "require_encoded_png_hash", "require_ordered_image_digest",
            "require_dimension_reconciliation", "require_label_set_reconciliation",
            "require_train_val_disjointness_when_training_present", "overwrite_incomplete_requires_explicit_flag",
        }
    ),
    "prohibited": frozenset({"list"}),
}


class Ade20kDatasetIdentityError(ValueError):
    """Raised when the ADE20K dataset-source identity, or the live
    dataset/loader it governs, fails any exact-type/schema/relational
    check. Always fail closed."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value.strip()):
        raise Ade20kDatasetIdentityError(f"{label} must be an exact non-empty string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise Ade20kDatasetIdentityError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise Ade20kDatasetIdentityError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise Ade20kDatasetIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise Ade20kDatasetIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", token) is None:
        raise Ade20kDatasetIdentityError(f"{label} must be a full Git identity")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token:
        raise Ade20kDatasetIdentityError(f"{label} must be a safe repository-relative path")
    return token


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise Ade20kDatasetIdentityError(f"{label} has an unexpected schema")
    return value


def _require_exact_string_list(value: Any, label: str, *, nonempty: bool = True) -> tuple[str, ...]:
    if type(value) is not list or (nonempty and not value):
        raise Ade20kDatasetIdentityError(f"{label} must be a non-empty exact list")
    if any(type(item) is not str for item in value):
        raise Ade20kDatasetIdentityError(f"{label} elements must be exact strings")
    return tuple(value)


def _require_exact_int_pair(value: Any, label: str) -> tuple[int, int]:
    if type(value) is not list or len(value) != 2:
        raise Ade20kDatasetIdentityError(f"{label} must be an exact two-element list")
    if any(type(item) is not int for item in value):
        raise Ade20kDatasetIdentityError(f"{label} elements must be exact integers")
    return (value[0], value[1])


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise Ade20kDatasetIdentityError(f"cannot load ade20k dataset-source identity {source}: {error}") from error

    if set(identity) != IDENTITY_TOP_KEYS:
        raise Ade20kDatasetIdentityError("ade20k dataset-source identity has an unexpected top-level schema")
    _require_exact_string(identity["format_version"], "format_version")
    if identity["format_version"] != "talk2dino-ade20k-dataset-source-identity-v1":
        raise Ade20kDatasetIdentityError("unsupported ade20k dataset-source identity format_version")
    for section, keys in IDENTITY_SECTION_KEYS.items():
        _require_closed_mapping(identity.get(section), keys, f"identity.{section}")

    block = identity["identity"]
    _require_exact_string(block["name"], "identity.name")
    _require_exact_string(block["schema_version"], "identity.schema_version")
    if block["schema_version"] != identity["format_version"]:
        raise Ade20kDatasetIdentityError("identity.schema_version disagrees with format_version")
    _require_exact_string(block["description"], "identity.description")
    _require_git_identity(block["required_ancestor_commit"], "identity.required_ancestor_commit")

    upstream = identity["upstream"]
    if upstream["dataset_name"] != "ADE20K (ADEChallengeData2016)":
        raise Ade20kDatasetIdentityError("upstream.dataset_name must be exactly 'ADE20K (ADEChallengeData2016)'")
    _require_exact_string(upstream["dataset_version"], "upstream.dataset_version")
    _require_exact_string(upstream["source_url"], "upstream.source_url")
    _require_exact_string(upstream["source_reference"], "upstream.source_reference")
    if upstream["archive_top_level_directory"] != "ADEChallengeData2016":
        raise Ade20kDatasetIdentityError("upstream.archive_top_level_directory must be exactly 'ADEChallengeData2016'")

    protocol = identity["protocol"]
    if protocol["label"] != "ADE20K-C150":
        raise Ade20kDatasetIdentityError("protocol.label must be exactly 'ADE20K-C150'")
    _require_exact_string(protocol["statement"], "protocol.statement")
    if protocol["split"] != SUPPORTED_SPLIT:
        raise Ade20kDatasetIdentityError(f"protocol.split must be exactly {SUPPORTED_SPLIT!r}")
    if _require_exact_int(protocol["expected_image_count"], "protocol.expected_image_count") != SUPPORTED_EXPECTED_IMAGE_COUNT:
        raise Ade20kDatasetIdentityError(f"protocol.expected_image_count must be exactly {SUPPORTED_EXPECTED_IMAGE_COUNT}")

    root_structure = identity["expected_root_structure"]
    _require_exact_string(root_structure["description"], "expected_root_structure.description")
    if root_structure["ade_relative_root"] != "ADEChallengeData2016":
        raise Ade20kDatasetIdentityError("expected_root_structure.ade_relative_root must be exactly 'ADEChallengeData2016'")
    required_subpaths = _require_exact_string_list(root_structure["required_subpaths"], "expected_root_structure.required_subpaths")
    if set(required_subpaths) != {"images/training", "images/validation", "annotations/training", "annotations/validation"}:
        raise Ade20kDatasetIdentityError("expected_root_structure.required_subpaths must be exactly the canonical ADE20K subpath set")

    source_block = identity["source"]
    if source_block["image_relative_root"] != "images/validation":
        raise Ade20kDatasetIdentityError("source.image_relative_root must be exactly 'images/validation'")
    if source_block["annotation_relative_root"] != "annotations/validation":
        raise Ade20kDatasetIdentityError("source.annotation_relative_root must be exactly 'annotations/validation'")
    if source_block["training_image_relative_root"] != "images/training":
        raise Ade20kDatasetIdentityError("source.training_image_relative_root must be exactly 'images/training'")
    if source_block["training_annotation_relative_root"] != "annotations/training":
        raise Ade20kDatasetIdentityError("source.training_annotation_relative_root must be exactly 'annotations/training'")
    if source_block["image_suffix"] != SUPPORTED_IMAGE_SUFFIX:
        raise Ade20kDatasetIdentityError(f"source.image_suffix must be exactly {SUPPORTED_IMAGE_SUFFIX!r}")
    if source_block["annotation_suffix"] != SUPPORTED_ANNOTATION_SUFFIX:
        raise Ade20kDatasetIdentityError(f"source.annotation_suffix must be exactly {SUPPORTED_ANNOTATION_SUFFIX!r}")
    _require_exact_string(source_block["split_order_authority"], "source.split_order_authority")

    loader = identity["dataset_loader"]
    _require_relative_path(loader["dataset_config_relative_path"], "dataset_loader.dataset_config_relative_path")
    _require_sha256(loader["dataset_config_sha256"], "dataset_loader.dataset_config_sha256")
    if loader["dataset_class_name"] != SUPPORTED_DATASET_CLASS_NAME:
        raise Ade20kDatasetIdentityError(f"dataset_loader.dataset_class_name must be exactly {SUPPORTED_DATASET_CLASS_NAME!r}")
    _require_exact_string(loader["dataset_class_source"], "dataset_loader.dataset_class_source")
    if loader["mmsegmentation_required_version"] != SUPPORTED_MMSEGMENTATION_VERSION:
        raise Ade20kDatasetIdentityError(f"dataset_loader.mmsegmentation_required_version must be exactly {SUPPORTED_MMSEGMENTATION_VERSION!r}")
    if loader["data_root_relative_path"] != "data/ade":
        raise Ade20kDatasetIdentityError("dataset_loader.data_root_relative_path must be exactly 'data/ade'")

    class_contract = identity["class_contract"]
    if _require_exact_int(class_contract["class_count"], "class_contract.class_count") != SUPPORTED_CLASS_COUNT:
        raise Ade20kDatasetIdentityError(f"class_contract.class_count must be exactly {SUPPORTED_CLASS_COUNT}")
    _require_sha256(class_contract["class_names_digest"], "class_contract.class_names_digest")
    if class_contract["class_names_digest_algorithm"] != SUPPORTED_CLASS_NAMES_DIGEST_ALGORITHM:
        raise Ade20kDatasetIdentityError(f"class_contract.class_names_digest_algorithm must be exactly {SUPPORTED_CLASS_NAMES_DIGEST_ALGORITHM!r}")

    label_policy = identity["label_policy"]
    if _require_exact_int(label_policy["raw_ignore_value"], "label_policy.raw_ignore_value") != SUPPORTED_RAW_IGNORE_VALUE:
        raise Ade20kDatasetIdentityError(f"label_policy.raw_ignore_value must be exactly {SUPPORTED_RAW_IGNORE_VALUE}")
    if _require_exact_int(label_policy["ignore_index"], "label_policy.ignore_index") != SUPPORTED_IGNORE_INDEX:
        raise Ade20kDatasetIdentityError(f"label_policy.ignore_index must be exactly {SUPPORTED_IGNORE_INDEX}")
    if _require_exact_bool(label_policy["reduce_zero_label"], "label_policy.reduce_zero_label") is not True:
        raise Ade20kDatasetIdentityError("label_policy.reduce_zero_label must be true")
    _require_exact_string(label_policy["transform_description"], "label_policy.transform_description")
    if _require_exact_bool(label_policy["background_class_evaluated"], "label_policy.background_class_evaluated") is not False:
        raise Ade20kDatasetIdentityError("label_policy.background_class_evaluated must be false")

    pipeline = identity["evaluation_pipeline"]
    _require_exact_string(pipeline["description"], "evaluation_pipeline.description")
    if pipeline["mode"] != SUPPORTED_MODE:
        raise Ade20kDatasetIdentityError(f"evaluation_pipeline.mode must be exactly {SUPPORTED_MODE!r}")
    if _require_exact_int_pair(pipeline["crop_size"], "evaluation_pipeline.crop_size") != SUPPORTED_CROP_SIZE:
        raise Ade20kDatasetIdentityError(f"evaluation_pipeline.crop_size must be exactly {list(SUPPORTED_CROP_SIZE)}")
    if _require_exact_int_pair(pipeline["stride"], "evaluation_pipeline.stride") != SUPPORTED_STRIDE:
        raise Ade20kDatasetIdentityError(f"evaluation_pipeline.stride must be exactly {list(SUPPORTED_STRIDE)}")

    provenance = identity["provenance_policy"]
    _require_exact_bool(provenance["conversion_required"], "provenance_policy.conversion_required")
    if provenance["conversion_required"] is not False:
        raise Ade20kDatasetIdentityError("provenance_policy.conversion_required must be false")
    if _require_exact_bool(provenance["no_bridge_training_or_feature_extraction_allowed"], "provenance_policy.no_bridge_training_or_feature_extraction_allowed") is not True:
        raise Ade20kDatasetIdentityError("provenance_policy.no_bridge_training_or_feature_extraction_allowed must be true")
    _require_exact_string(provenance["statement"], "provenance_policy.statement")

    manifest_block = identity["manifest"]
    if manifest_block["schema_name"] != SUPPORTED_MANIFEST_SCHEMA_NAME:
        raise Ade20kDatasetIdentityError(f"manifest.schema_name must be exactly {SUPPORTED_MANIFEST_SCHEMA_NAME!r}")

    hashing = identity["hashing"]
    if hashing["algorithm"] != SUPPORTED_HASH_ALGORITHM:
        raise Ade20kDatasetIdentityError(f"hashing.algorithm must be exactly {SUPPORTED_HASH_ALGORITHM!r}")

    validation = identity["validation"]
    for flag in (
        "require_decoded_pixel_hash", "require_encoded_png_hash", "require_ordered_image_digest",
        "require_dimension_reconciliation", "require_label_set_reconciliation",
        "require_train_val_disjointness_when_training_present", "overwrite_incomplete_requires_explicit_flag",
    ):
        if _require_exact_bool(validation[flag], f"validation.{flag}") is not True:
            raise Ade20kDatasetIdentityError(f"validation.{flag} must be true")

    _require_exact_string_list(identity["prohibited"]["list"], "prohibited.list")

    return identity


def _check_git_ancestry(root: Path, commit: str, *, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"{label} is not an ancestor of HEAD"
        raise Ade20kDatasetIdentityError(f"Git ancestry check failed for {label} ({commit}): {detail}")


def _validate_loader_provenance(root: Path, identity: Mapping[str, Any]) -> None:
    import hashlib

    loader = identity["dataset_loader"]
    path = root / loader["dataset_config_relative_path"]
    try:
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise Ade20kDatasetIdentityError(f"cannot read dataset-loader provenance file {path}: {error}") from error
    if observed != loader["dataset_config_sha256"]:
        raise Ade20kDatasetIdentityError(
            f"dataset_config_relative_path SHA256 mismatch: identity declares {loader['dataset_config_sha256']}, observed {observed} at {path}"
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


def validate_class_contract_against_installed_mmseg(identity: Mapping[str, Any]) -> tuple[str, ...]:
    """Live-import mmseg (CPU-only; no CUDA/model/dataset construction)
    and require its stock ADE20KDataset.CLASSES to exactly equal the
    150-tuple this identity's class_names_digest was computed from, and
    that reduce_zero_label is fixed True (never config-overridable)."""
    import hashlib
    import inspect
    import json

    try:
        import mmseg
        from mmseg.datasets import ADE20KDataset
    except ImportError as error:
        raise Ade20kDatasetIdentityError(f"cannot import mmseg.datasets.ADE20KDataset: {error}") from error

    installed_version = getattr(mmseg, "__version__", None)
    required_version = identity["dataset_loader"]["mmsegmentation_required_version"]
    if installed_version != required_version:
        raise Ade20kDatasetIdentityError(
            f"installed mmsegmentation version {installed_version!r} does not match "
            f"dataset_loader.mmsegmentation_required_version {required_version!r}"
        )

    classes = ADE20KDataset.CLASSES
    if type(classes) is not tuple or any(type(name) is not str for name in classes):
        raise Ade20kDatasetIdentityError("mmseg.datasets.ADE20KDataset.CLASSES must be an exact tuple of exact strings")
    digest = hashlib.sha256(json.dumps(list(classes), ensure_ascii=True).encode("utf-8")).hexdigest()
    if digest != identity["class_contract"]["class_names_digest"]:
        raise Ade20kDatasetIdentityError(
            f"live mmseg.datasets.ADE20KDataset.CLASSES digest {digest!r} disagrees with "
            f"class_contract.class_names_digest {identity['class_contract']['class_names_digest']!r}"
        )

    source = inspect.getsource(ADE20KDataset.__init__)
    if "reduce_zero_label=True" not in source.replace(" ", ""):
        raise Ade20kDatasetIdentityError(
            "live ADE20KDataset.__init__ no longer hardcodes reduce_zero_label=True -- "
            "the label-policy contract this identity records may no longer hold"
        )
    return classes


__all__ = [
    "IDENTITY_RELATIVE_PATH",
    "Ade20kDatasetIdentityError",
    "SUPPORTED_ANNOTATION_SUFFIX",
    "SUPPORTED_CLASS_COUNT",
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
    "SUPPORTED_DATASET_CLASS_NAME",
    "load_identity",
    "repository_root",
    "validate_static_configuration",
    "validate_class_contract_against_installed_mmseg",
]
