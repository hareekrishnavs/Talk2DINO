"""Static/mutation tests for the ADE20K dataset-source identity. CPU-only,
no torch import needed for the identity-mutation tests; the live-mmseg
class-contract test needs mmseg but never CUDA/a model/a dataset."""

from __future__ import annotations

import copy
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.ade20k_dataset_identity import (  # noqa: E402
    Ade20kDatasetIdentityError,
    load_identity,
    validate_static_configuration,
)

IDENTITY_PATH = ROOT / "evaluation_identities/e12_ade20k_dataset_source.toml"


def _load_raw() -> dict:
    with IDENTITY_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _dump_toml(document: dict) -> str:
    """Minimal, non-general TOML writer sufficient for this identity's
    flat two-level {section: {key: value}} shape (no nested tables under
    a section, matching this file's own structure)."""
    lines: list[str] = []

    def _fmt(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, list):
            return "[" + ", ".join(_fmt(v) for v in value) + "]"
        raise TypeError(f"unsupported TOML value type: {type(value)}")

    lines.append(f'format_version = {_fmt(document["format_version"])}')
    for section, body in document.items():
        if section == "format_version":
            continue
        lines.append("")
        lines.append(f"[{section}]")
        for key, value in body.items():
            lines.append(f"{key} = {_fmt(value)}")
    return "\n".join(lines) + "\n"


def _write_and_load(tmp_path: Path, document: dict):
    path = tmp_path / "identity.toml"
    path.write_text(_dump_toml(document), encoding="utf-8")
    return load_identity(path, repo_root=ROOT)


# ---------------------------------------------------------------------------
# Baseline: the real, committed identity must load and statically validate.
# ---------------------------------------------------------------------------


def test_real_identity_loads():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    assert identity["identity"]["name"] == "e12-ade20k-dataset-source"
    assert identity["protocol"]["expected_image_count"] == 2000
    assert identity["class_contract"]["class_count"] == 150


def test_static_configuration_validates_against_real_repo():
    result = validate_static_configuration(repo_root=ROOT, check_git=True)
    assert result["expected_image_count"] == 2000


def test_no_private_path_in_identity_file():
    text = IDENTITY_PATH.read_text(encoding="utf-8")
    for fragment in ("/scratch/", "/project/", "/home/"):
        assert fragment not in text


# ---------------------------------------------------------------------------
# Mutation tests: every material field must fail closed when tampered.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("protocol", "expected_image_count", 2001),
        ("protocol", "split", "test"),
        ("class_contract", "class_count", 151),
        ("label_policy", "reduce_zero_label", False),
        ("label_policy", "background_class_evaluated", True),
        ("label_policy", "raw_ignore_value", 255),
        ("label_policy", "ignore_index", 254),
        ("evaluation_pipeline", "mode", "whole"),
        ("evaluation_pipeline", "crop_size", [512, 512]),
        ("evaluation_pipeline", "stride", [256, 256]),
        ("source", "image_suffix", ".png"),
        ("source", "annotation_suffix", ".jpg"),
        ("dataset_loader", "mmsegmentation_required_version", "0.31.0"),
        ("dataset_loader", "data_root_relative_path", "data/ade20k"),
        ("provenance_policy", "conversion_required", True),
        ("provenance_policy", "no_bridge_training_or_feature_extraction_allowed", False),
    ],
)
def test_mutated_field_rejected(tmp_path, section, key, value):
    document = _load_raw()
    document[section][key] = value
    with pytest.raises(Ade20kDatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_unknown_top_level_key_rejected(tmp_path):
    document = _load_raw()
    document["unexpected_section"] = {"x": 1}
    with pytest.raises(Ade20kDatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_missing_section_rejected(tmp_path):
    document = _load_raw()
    del document["label_policy"]
    with pytest.raises(Ade20kDatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_format_version_rejected(tmp_path):
    document = _load_raw()
    document["format_version"] = "talk2dino-ade20k-dataset-source-identity-v2"
    document["identity"]["schema_version"] = "talk2dino-ade20k-dataset-source-identity-v2"
    with pytest.raises(Ade20kDatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_bad_ancestor_commit_format_rejected(tmp_path):
    document = _load_raw()
    document["identity"]["required_ancestor_commit"] = "not-a-commit"
    with pytest.raises(Ade20kDatasetIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_class_names_digest_rejected_against_live_mmseg():
    """class_names_digest's correctness is checked against the live,
    installed mmseg ADE20KDataset.CLASSES, not by load_identity alone --
    matches the VOC2012 identity's split between static schema
    validation (load_identity) and live-value binding
    (validate_class_contract_against_installed_mmseg)."""
    from src.ade20k_dataset_identity import validate_class_contract_against_installed_mmseg

    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    tampered = copy.deepcopy(identity)
    tampered["class_contract"]["class_names_digest"] = "f" * 64
    with pytest.raises(Ade20kDatasetIdentityError):
        validate_class_contract_against_installed_mmseg(tampered)


def test_wrong_dataset_config_hash_rejected():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    tampered = copy.deepcopy(identity)
    tampered["dataset_loader"]["dataset_config_sha256"] = "f" * 64
    from src.ade20k_dataset_identity import _validate_loader_provenance

    with pytest.raises(Ade20kDatasetIdentityError):
        _validate_loader_provenance(ROOT, tampered)
