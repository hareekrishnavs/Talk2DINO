"""Identity tests for the COCO-Object val2017 mask-materialization stage.
Every expected value here is read from the real, committed identity/
converter files -- never a duplicated scientific literal."""

from __future__ import annotations

import copy
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.coco_object_val_materialization_identity import (  # noqa: E402
    CocoObjectValMaterializationIdentityError,
    load_identity,
    validate_static_configuration,
)

IDENTITY_PATH = ROOT / "evaluation_identities/e12_coco_object_val_materialization.toml"
pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the materialization identity")


def _load_raw() -> dict:
    with IDENTITY_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _write_and_load(tmp_path, document: dict):
    path = tmp_path / "identity.toml"
    try:
        import tomli_w  # not guaranteed installed; fall back to manual TOML writer if unavailable
        with path.open("wb") as handle:
            tomli_w.dump(document, handle)
    except ImportError:
        path.write_text(_manual_toml_dump(document))
    return load_identity(path, repo_root=ROOT)


def _manual_toml_dump(document: dict) -> str:
    """Dependency-free TOML writer sufficient for this identity's flat
    two-level (table-of-scalars/lists) shape -- never imports the
    production loader's own writer, if one existed."""
    lines: list[str] = []

    def emit_value(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, list):
            return "[" + ", ".join(emit_value(v) for v in value) + "]"
        raise TypeError(f"unsupported TOML value type: {type(value)}")

    for key, value in document.items():
        if key == "format_version":
            lines.append(f"{key} = {emit_value(value)}")
    for section, block in document.items():
        if section == "format_version":
            continue
        lines.append(f"\n[{section}]")
        for key, value in block.items():
            lines.append(f"{key} = {emit_value(value)}")
    return "\n".join(lines) + "\n"


def test_valid_identity_loads(tmp_path):
    document = _load_raw()
    identity = _write_and_load(tmp_path, document)
    assert identity["identity"]["name"] == "e12-coco-object-val-materialization"
    assert identity["protocol"]["split"] == "val2017"
    assert identity["protocol"]["expected_image_count"] == 5000


def test_validate_static_configuration_against_real_repo():
    result = validate_static_configuration(repo_root=ROOT, check_git=True)
    assert result["split"] == "val2017"
    assert result["expected_image_count"] == 5000


def test_wrong_protocol_label_rejected(tmp_path):
    document = _load_raw()
    document["protocol"]["label"] = "COCO-Stuff"
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_split_rejected(tmp_path):
    document = _load_raw()
    document["protocol"]["split"] = "train2017"
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_expected_image_count_rejected(tmp_path):
    document = _load_raw()
    document["protocol"]["expected_image_count"] = 4999
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_class_count_rejected(tmp_path):
    document = _load_raw()
    document["class_contract"]["class_count"] = 80
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_background_class_index_rejected(tmp_path):
    document = _load_raw()
    document["class_contract"]["background_class_index"] = 1
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_output_mask_suffix_rejected(tmp_path):
    document = _load_raw()
    document["converter"]["output_mask_suffix"] = "_labelTrainIds.png"
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_converter_sha256_mismatch_rejected(tmp_path):
    # SHA256 *syntax* is checked by load_identity() alone; actual file-hash
    # *consistency* is checked only by validate_static_configuration's
    # provenance check -- the same two-tier pattern used throughout E12.
    document = _load_raw()
    document["converter"]["canonical_converter_sha256"] = "0" * 64
    path = tmp_path / "identity.toml"
    path.write_text(_manual_toml_dump(document))
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        validate_static_configuration(repo_root=ROOT, identity_path=path, check_git=False)


def test_mapping_table_sha256_wrong_length_rejected(tmp_path):
    document = _load_raw()
    document["label_mapping"]["mapping_table_sha256"] = "abc123"
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_raw_255_folds_to_background_must_be_true(tmp_path):
    document = _load_raw()
    document["label_mapping"]["raw_255_folds_to_background"] = False
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_train_conversion_prohibited_must_be_true(tmp_path):
    document = _load_raw()
    document["policy"]["train_conversion_prohibited"] = False
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_cuda_required_must_be_false(tmp_path):
    document = _load_raw()
    document["policy"]["cuda_required"] = True
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_unknown_top_level_key_rejected(tmp_path):
    document = _load_raw()
    document["unexpected_section"] = {"a": 1}
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_unknown_key_inside_section_rejected(tmp_path):
    document = _load_raw()
    document["protocol"]["unexpected_field"] = "x"
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_missing_required_field_rejected(tmp_path):
    document = _load_raw()
    del document["protocol"]["split"]
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_exact_type_int_as_string_rejected(tmp_path):
    document = _load_raw()
    document["protocol"]["expected_image_count"] = "5000"
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_exact_type_bool_as_int_rejected(tmp_path):
    document = _load_raw()
    document["policy"]["cuda_required"] = 0
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_train_val_mask_counts_must_sum_to_coco_len(tmp_path):
    document = _load_raw()
    document["source"]["train_mask_count_expected"] = 118286
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_unsafe_relative_path_rejected(tmp_path):
    document = _load_raw()
    document["converter"]["canonical_converter_relative_path"] = "../outside/converter.py"
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_absolute_path_rejected(tmp_path):
    document = _load_raw()
    document["converter"]["canonical_converter_relative_path"] = "/etc/passwd"
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _write_and_load(tmp_path, document)


def test_missing_identity_file_fails_closed(tmp_path):
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        load_identity(tmp_path / "does-not-exist.toml", repo_root=ROOT)


def test_malformed_toml_fails_closed(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("this is not [valid toml")
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        load_identity(path, repo_root=ROOT)


def test_git_ancestry_check_rejects_unknown_commit(tmp_path):
    document = _load_raw()
    document["identity"]["required_ancestor_commit"] = "f" * 40
    path = tmp_path / "identity.toml"
    path.write_text(_manual_toml_dump(document))
    from src.coco_object_val_materialization_identity import _check_git_ancestry
    with pytest.raises(CocoObjectValMaterializationIdentityError):
        _check_git_ancestry(ROOT, "f" * 40, label="test")
