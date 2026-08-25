"""Tests for validate_live_class_order: the live-dataset CLASSES
binding against the identity's declared class count/background
position/digest. Uses lightweight fake dataset objects (no mmseg/CUDA
required) exposing the exact attributes the real COCOObjectDataset/mmseg
wrapper chain exposes, plus one end-to-end check against the real
dataset built from the verified materialized COCO-Object data."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.coco_object_protocol_confirmation_identity import (  # noqa: E402
    CocoObjectProtocolConfirmationIdentityError,
    load_identity,
    validate_live_class_order,
)

IDENTITY_PATH = ROOT / "evaluation_identities/e12_coco_object_protocol_confirmation.toml"
DATASET_CLASS_PATH = ROOT / "src/open_vocabulary_segmentation/segmentation/datasets/coco_object.py"
pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the protocol-confirmation identity")


@pytest.fixture(scope="module")
def identity():
    return load_identity(repo_root=ROOT)


@pytest.fixture(scope="module")
def canonical_classes():
    """The real COCOObjectDataset.CLASSES tuple, discovered via AST --
    never re-imports the implementation's own dataset module."""
    tree = ast.parse(DATASET_CLASS_PATH.read_text())
    class_def = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "COCOObjectDataset"
    )
    assign = next(
        node for node in class_def.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CLASSES" for t in node.targets)
    )
    return ast.literal_eval(assign.value)


def _fake_dataset(classes_value, *, instance_override=None):
    cls = type("COCOObjectDataset", (), {"CLASSES": classes_value})
    instance = cls()
    if instance_override is not None:
        instance.CLASSES = instance_override
    return instance


class _Wrapper:
    def __init__(self, inner):
        self.dataset = inner


def test_canonical_live_list_passes(identity, canonical_classes):
    digest = validate_live_class_order(_fake_dataset(canonical_classes), identity)
    assert digest == identity["dataset"]["class_names_digest"]


def test_same_count_reordered_list_fails(identity, canonical_classes):
    reordered = (canonical_classes[0],) + tuple(reversed(canonical_classes[1:]))
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_live_class_order(_fake_dataset(reordered), identity)


def test_renamed_class_fails(identity, canonical_classes):
    renamed = list(canonical_classes)
    renamed[5] = "not_a_real_class"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_live_class_order(_fake_dataset(tuple(renamed)), identity)


def test_missing_class_fails(identity, canonical_classes):
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_live_class_order(_fake_dataset(canonical_classes[:-1]), identity)


def test_duplicate_class_fails(identity, canonical_classes):
    duplicated = list(canonical_classes)
    duplicated[10] = duplicated[9]
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_live_class_order(_fake_dataset(tuple(duplicated)), identity)


def test_background_moved_from_index_0_fails(identity, canonical_classes):
    moved = canonical_classes[1:3] + (canonical_classes[0],) + canonical_classes[3:]
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_live_class_order(_fake_dataset(moved), identity)


def test_list_container_type_rejected(identity, canonical_classes):
    """A list carries the identical class content but violates the
    declared tuple contract -- must be rejected, not silently accepted."""
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_live_class_order(
            _fake_dataset(canonical_classes, instance_override=list(canonical_classes)), identity
        )


def test_wrapper_with_consistent_list_passes(identity, canonical_classes):
    digest = validate_live_class_order(_Wrapper(_fake_dataset(canonical_classes)), identity)
    assert digest == identity["dataset"]["class_names_digest"]


def test_wrapper_with_inconsistent_metadata_fails(identity, canonical_classes):
    overridden = list(canonical_classes)
    overridden[0] = "bg"
    inner = _fake_dataset(canonical_classes, instance_override=tuple(overridden))
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_live_class_order(_Wrapper(inner), identity)


def test_unresolvable_dataset_fails(identity):
    class NoClasses:
        pass

    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_live_class_order(NoClasses(), identity)


def test_input_class_sequence_remains_immutable(identity, canonical_classes):
    dataset = _fake_dataset(canonical_classes)
    before = dataset.CLASSES
    validate_live_class_order(dataset, identity)
    assert dataset.CLASSES is before
    assert dataset.CLASSES == canonical_classes


def test_non_string_element_rejected(identity, canonical_classes):
    tampered = list(canonical_classes)
    tampered[3] = 42
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_live_class_order(_fake_dataset(tuple(tampered)), identity)


REAL_MANIFEST = Path("/scratch/haree/coco_object_protocol/manifests/manifest-20443250.json")
REAL_DATA_ROOT = Path("/scratch/haree/coco_object_protocol")


@pytest.mark.skipif(
    not (REAL_MANIFEST.exists() and REAL_DATA_ROOT.exists()),
    reason="requires the real materialized COCO-Object data",
)
def test_real_built_dataset_passes(identity):
    sys.path.insert(0, str(ROOT / "src/open_vocabulary_segmentation"))
    from mmcv import Config as MMCVConfig
    from mmseg.datasets import build_dataset

    import main  # noqa: F401  -- registers the custom FloatImage transform

    dataset_config_path = ROOT / identity["dataset"]["dataset_config_relative_path"]
    dataset_cfg = MMCVConfig.fromfile(str(dataset_config_path))
    dataset_cfg.data.test.data_root = str(REAL_DATA_ROOT)
    dataset = build_dataset(dataset_cfg.data.test)
    digest = validate_live_class_order(dataset, identity)
    assert digest == identity["dataset"]["class_names_digest"]
