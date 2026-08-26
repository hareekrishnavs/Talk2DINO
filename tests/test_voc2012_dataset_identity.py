"""Identity tests for the shared VOC2012 (V20/V21) dataset-source
contract. Every expected value is read from the real, committed
identity/source files -- never a duplicated scientific literal."""

from __future__ import annotations

import copy
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.voc2012_dataset_identity import (  # noqa: E402
    Voc2012DatasetIdentityError,
    load_identity,
    validate_static_configuration,
    validate_v20_class_contract_against_source,
    validate_v21_class_contract_against_installed_mmseg,
)

IDENTITY_PATH = ROOT / "evaluation_identities/e12_voc2012_dataset_source.toml"
pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the voc2012 dataset-source identity")


def _load_raw() -> dict:
    with IDENTITY_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _manual_toml_dump(document: dict) -> str:
    def emit(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return repr(value)
        if isinstance(value, str):
            return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(value, list):
            return "[" + ", ".join(emit(v) for v in value) + "]"
        raise TypeError(f"unsupported TOML value type: {type(value)}")

    lines = [f"format_version = {emit(document['format_version'])}"]
    for section, block in document.items():
        if section == "format_version":
            continue
        lines.append(f"\n[{section}]")
        for key, value in block.items():
            lines.append(f"{key} = {emit(value)}")
    return "\n".join(lines) + "\n"


def _write_and_load(tmp_path, document: dict):
    path = tmp_path / "identity.toml"
    path.write_text(_manual_toml_dump(document))
    return load_identity(path, repo_root=ROOT)


def test_valid_identity_loads():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    assert identity["identity"]["name"] == "e12-voc2012-dataset-source"
    assert identity["protocol"]["split"] == "val"
    assert identity["protocol"]["expected_image_count"] == 1449
    assert identity["class_contract"]["v21_class_count"] == 21
    assert identity["class_contract"]["v20_class_count"] == 20


def test_validate_static_configuration_against_real_repo():
    result = validate_static_configuration(repo_root=ROOT, check_git=True)
    assert result["identity_name"] == "e12-voc2012-dataset-source"
    assert result["expected_image_count"] == 1449


def test_v20_class_contract_matches_real_source():
    identity = load_identity(repo_root=ROOT)
    classes = validate_v20_class_contract_against_source(ROOT, identity)
    assert len(classes) == 20
    assert "background" not in classes


def test_v21_class_contract_matches_installed_mmseg():
    identity = load_identity(repo_root=ROOT)
    classes = validate_v21_class_contract_against_installed_mmseg(identity)
    assert len(classes) == 21
    assert classes[0] == "background"


def test_v21_is_background_plus_v20_in_order():
    identity = load_identity(repo_root=ROOT)
    v20 = validate_v20_class_contract_against_source(ROOT, identity)
    v21 = validate_v21_class_contract_against_installed_mmseg(identity)
    assert v21 == ("background",) + v20


def test_unknown_top_level_key_rejected(tmp_path):
    document = _load_raw()
    document["extra_injected_section"] = {"x": 1}
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_missing_top_level_key_rejected(tmp_path):
    document = _load_raw()
    del document["label_policy"]
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_unknown_nested_key_rejected(tmp_path):
    document = _load_raw()
    document["protocol"]["extra_field"] = "y"
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_missing_nested_key_rejected(tmp_path):
    document = _load_raw()
    del document["protocol"]["split"]
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_exact_type_int_as_string_rejected(tmp_path):
    document = _load_raw()
    document["protocol"]["expected_image_count"] = "1449"
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_exact_type_bool_as_int_rejected(tmp_path):
    document = _load_raw()
    document["label_policy"]["v20_reduce_zero_label"] = 1
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_exact_type_int_as_bool_rejected(tmp_path):
    document = _load_raw()
    document["class_contract"]["v21_class_count"] = True  # bool is an int subclass but must still be rejected
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_unsafe_absolute_path_rejected(tmp_path):
    document = _load_raw()
    document["dataset_loader"]["v21_dataset_config_relative_path"] = "/etc/passwd"
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_unsafe_traversal_path_rejected(tmp_path):
    document = _load_raw()
    document["dataset_loader"]["v20_dataset_class_relative_path"] = "../../../../etc/passwd"
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_split_rejected(tmp_path):
    document = _load_raw()
    document["protocol"]["split"] = "train"
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_expected_image_count_rejected(tmp_path):
    document = _load_raw()
    document["protocol"]["expected_image_count"] = 1450
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_v21_class_count_disagreeing_with_v20_plus_one_rejected(tmp_path):
    document = _load_raw()
    document["class_contract"]["v21_class_count"] = 22
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_background_class_index_not_zero_rejected(tmp_path):
    document = _load_raw()
    document["class_contract"]["background_class_index"] = 1
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_v21_reduce_zero_label_true_rejected(tmp_path):
    document = _load_raw()
    document["label_policy"]["v21_reduce_zero_label"] = True
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_v20_reduce_zero_label_false_rejected(tmp_path):
    document = _load_raw()
    document["label_policy"]["v20_reduce_zero_label"] = False
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_ignore_index_wrong_value_rejected(tmp_path):
    document = _load_raw()
    document["label_policy"]["ignore_index"] = 0
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_crop_size_wrong_rejected(tmp_path):
    document = _load_raw()
    document["evaluation_pipeline"]["crop_size"] = [512, 512]
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_stride_wrong_rejected(tmp_path):
    document = _load_raw()
    document["evaluation_pipeline"]["stride"] = [256, 256]
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_v20_v21_shared_source_false_rejected(tmp_path):
    document = _load_raw()
    document["provenance_policy"]["v20_v21_share_same_source_files"] = False
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_feature_extraction_prohibition_false_rejected(tmp_path):
    document = _load_raw()
    document["provenance_policy"]["no_bridge_training_or_feature_extraction_allowed"] = False
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_v20_dataset_config_sha256_mismatch_rejected(tmp_path):
    document = _load_raw()
    document["dataset_loader"]["v20_dataset_config_sha256"] = "0" * 64
    path = tmp_path / "identity.toml"
    path.write_text(_manual_toml_dump(document))
    with pytest.raises(Voc2012DatasetIdentityError):
        validate_static_configuration(repo_root=ROOT, identity_path=path, check_git=False)


def test_v21_dataset_config_sha256_mismatch_rejected(tmp_path):
    document = _load_raw()
    document["dataset_loader"]["v21_dataset_config_sha256"] = "0" * 64
    path = tmp_path / "identity.toml"
    path.write_text(_manual_toml_dump(document))
    with pytest.raises(Voc2012DatasetIdentityError):
        validate_static_configuration(repo_root=ROOT, identity_path=path, check_git=False)


def test_v20_class_names_digest_mismatch_rejected():
    identity = copy.deepcopy(load_identity(repo_root=ROOT))
    identity["class_contract"] = dict(identity["class_contract"])
    identity["class_contract"]["v20_class_names_digest"] = "0" * 64
    with pytest.raises(Voc2012DatasetIdentityError):
        validate_v20_class_contract_against_source(ROOT, identity)


def test_v21_class_names_digest_mismatch_rejected():
    identity = copy.deepcopy(load_identity(repo_root=ROOT))
    identity["class_contract"] = dict(identity["class_contract"])
    identity["class_contract"]["v21_class_names_digest"] = "0" * 64
    with pytest.raises(Voc2012DatasetIdentityError):
        validate_v21_class_contract_against_installed_mmseg(identity)


def test_mmsegmentation_version_mismatch_rejected():
    identity = copy.deepcopy(load_identity(repo_root=ROOT))
    identity["dataset_loader"] = dict(identity["dataset_loader"])
    identity["dataset_loader"]["mmsegmentation_required_version"] = "9.9.9"
    with pytest.raises(Voc2012DatasetIdentityError):
        validate_v21_class_contract_against_installed_mmseg(identity)


def test_missing_identity_file_fails_closed(tmp_path):
    with pytest.raises(Voc2012DatasetIdentityError):
        load_identity(tmp_path / "does-not-exist.toml", repo_root=ROOT)


def test_malformed_toml_fails_closed(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("this is not [valid toml")
    with pytest.raises(Voc2012DatasetIdentityError):
        load_identity(path, repo_root=ROOT)


# Every field validated through the generic nonempty-exact-string path
# (src.voc2012_dataset_identity._require_exact_string, default
# nonempty=True), enumerated by (top-level-section-or-None, key).
NONEMPTY_STRING_FIELD_PATHS = [
    (None, "format_version"),
    ("identity", "name"),
    ("identity", "schema_version"),
    ("identity", "description"),
    ("upstream", "dataset_version"),
    ("upstream", "source_url"),
    ("upstream", "source_reference"),
    ("protocol", "statement"),
    ("expected_root_structure", "description"),
    ("source", "split_order_authority"),
    ("dataset_loader", "v21_dataset_class_source"),
    ("class_contract", "class_names_relationship"),
    ("label_policy", "v20_transform_description"),
    ("label_policy", "v21_transform_description"),
    ("evaluation_pipeline", "description"),
    ("provenance_policy", "statement"),
]


@pytest.mark.parametrize("section,key", NONEMPTY_STRING_FIELD_PATHS, ids=[f"{s or 'top'}.{k}" for s, k in NONEMPTY_STRING_FIELD_PATHS])
def test_whitespace_only_string_rejected_for_every_nonempty_string_field(tmp_path, section, key):
    document = _load_raw()
    if section is None:
        document[key] = "   "
    else:
        document[section][key] = "   \t  "
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


@pytest.mark.parametrize("section,key", NONEMPTY_STRING_FIELD_PATHS, ids=[f"{s or 'top'}.{k}" for s, k in NONEMPTY_STRING_FIELD_PATHS])
def test_empty_string_rejected_for_every_nonempty_string_field(tmp_path, section, key):
    document = _load_raw()
    if section is None:
        document[key] = ""
    else:
        document[section][key] = ""
    with pytest.raises(Voc2012DatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_nonempty_string_with_surrounding_whitespace_is_accepted_unmodified(tmp_path):
    """The whitespace-only rejection must not strip/normalize an
    otherwise-legitimate string's content -- it only rejects strings
    that are ENTIRELY whitespace."""
    document = _load_raw()
    padded = "  a description that is not blank  "
    document["identity"]["description"] = padded
    identity = _write_and_load(tmp_path, document)
    assert identity["identity"]["description"] == padded


def test_identity_input_toml_file_immutable():
    before = IDENTITY_PATH.read_bytes()
    load_identity(repo_root=ROOT)
    validate_static_configuration(repo_root=ROOT, check_git=True)
    after = IDENTITY_PATH.read_bytes()
    assert before == after
