"""Strict adversarial tests for the additive Boundary IoU authority
supplement (evaluation_identities/e13_dfp_phase1a_boundary_iou.toml,
src/dfp_phase1a_boundary_iou_identity.py). CPU-only, no CUDA, no network,
no dataset access. Loads the committed identity and mutates only the
field under test -- never maintains a duplicated second authority.
"""

from __future__ import annotations

import copy
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.dfp_phase1a_boundary_iou_identity import (  # noqa: E402
    DfpPhase1aBoundaryIouIdentityError,
    load_identity,
    validate_static_configuration,
)

IDENTITY_PATH = ROOT / "evaluation_identities/e13_dfp_phase1a_boundary_iou.toml"


def _load_raw() -> dict:
    with IDENTITY_PATH.open("rb") as f:
        return tomllib.load(f)


def _fmt(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    raise TypeError(f"unsupported TOML value type: {type(value)}")


def _dump_toml(document: dict) -> str:
    lines = [f'format_version = {_fmt(document["format_version"])}']
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
# Positive loading
# ---------------------------------------------------------------------------


def test_real_identity_loads():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    assert identity["identity"]["name"] == "e13-dfp-phase1a-boundary-iou"


def test_static_configuration_validates_against_real_repo():
    result = validate_static_configuration(repo_root=ROOT)
    assert result["dilation_ratio_primary"] == 0.02
    assert result["fixed_sensitivity_widths"] == (1, 2, 4)


def test_load_identity_does_not_mutate_source_file():
    before = IDENTITY_PATH.read_bytes()
    load_identity(IDENTITY_PATH, repo_root=ROOT)
    after = IDENTITY_PATH.read_bytes()
    assert before == after


def test_no_private_path_in_identity_file():
    text = IDENTITY_PATH.read_text(encoding="utf-8")
    for fragment in ("/scratch/", "/project/", "/home/"):
        assert fragment not in text


# ---------------------------------------------------------------------------
# Structural adversarial matrix
# ---------------------------------------------------------------------------


def test_duplicate_toml_key_rejected(tmp_path):
    text = IDENTITY_PATH.read_text(encoding="utf-8")
    injected = text.replace(
        'name = "e13-dfp-phase1a-boundary-iou"',
        'name = "e13-dfp-phase1a-boundary-iou"\nname = "duplicate"',
        1,
    )
    path = tmp_path / "identity.toml"
    path.write_text(injected, encoding="utf-8")
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        load_identity(path, repo_root=ROOT)


def test_unknown_top_level_key_rejected(tmp_path):
    document = _load_raw()
    document["unexpected_section"] = {"x": 1}
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_missing_top_level_section_rejected(tmp_path):
    document = _load_raw()
    del document["paper"]
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_unknown_field_within_section_rejected(tmp_path):
    document = _load_raw()
    document["paper"]["extra_field"] = "x"
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_missing_field_within_section_rejected(tmp_path):
    document = _load_raw()
    del document["paper"]["doi"]
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Exact-type adversarial matrix
# ---------------------------------------------------------------------------


def test_bool_as_int_rejected(tmp_path):
    document = _load_raw()
    document["primitive"]["padding_value"] = True
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="exact"):
        _write_and_load(tmp_path, document)


def test_int_as_float_rejected(tmp_path):
    document = _load_raw()
    document["primitive"]["padding_width"] = 1.0
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="exact"):
        _write_and_load(tmp_path, document)


def test_numeric_string_rejected(tmp_path):
    document = _load_raw()
    document["primitive"]["dilation_ratio_primary"] = "0.02"
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="exact"):
        _write_and_load(tmp_path, document)


def test_nan_rejected(tmp_path):
    document = _load_raw()
    document["primitive"]["dilation_ratio_primary"] = float("nan")
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_infinity_rejected(tmp_path):
    document = _load_raw()
    document["primitive"]["dilation_ratio_primary"] = float("inf")
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Hash format adversarial matrix
# ---------------------------------------------------------------------------


def test_malformed_source_hash_rejected(tmp_path):
    document = _load_raw()
    document["official_source"]["source_sha256"] = document["official_source"]["source_sha256"][:-1]
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="lowercase SHA256"):
        _write_and_load(tmp_path, document)


def test_uppercase_source_hash_rejected(tmp_path):
    document = _load_raw()
    document["official_source"]["source_sha256"] = document["official_source"]["source_sha256"].upper()
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="lowercase SHA256"):
        _write_and_load(tmp_path, document)


def test_malformed_license_hash_rejected(tmp_path):
    document = _load_raw()
    document["official_source"]["license_sha256"] = "not-a-hash"
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="lowercase SHA256"):
        _write_and_load(tmp_path, document)


def test_wrong_external_commit_rejected(tmp_path):
    document = _load_raw()
    document["official_source"]["repository_commit"] = "0" * 40
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_source_hash_rejected(tmp_path):
    document = _load_raw()
    document["official_source"]["source_sha256"] = "f" * 64
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_license_hash_rejected(tmp_path):
    document = _load_raw()
    document["official_source"]["license_sha256"] = "f" * 64
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_malformed_commit_wrong_length_rejected(tmp_path):
    document = _load_raw()
    document["official_source"]["repository_commit"] = document["official_source"]["repository_commit"][:-5]
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Scientific-content adversarial matrix
# ---------------------------------------------------------------------------


def test_ratio_changed_rejected(tmp_path):
    document = _load_raw()
    document["primitive"]["dilation_ratio_primary"] = 0.05
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="0.02"):
        _write_and_load(tmp_path, document)


def test_morphology_kernel_dtype_changed_rejected(tmp_path):
    document = _load_raw()
    document["primitive"]["kernel_dtype"] = "int32"
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="disagrees"):
        _write_and_load(tmp_path, document)


def test_morphology_erosion_backend_changed_rejected(tmp_path):
    document = _load_raw()
    document["primitive"]["erosion_backend"] = "scipy.ndimage.binary_erosion"
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="disagrees"):
        _write_and_load(tmp_path, document)


def test_morphology_round_convention_changed_rejected(tmp_path):
    document = _load_raw()
    document["primitive"]["round_convention"] = "numpy_round_half_away_from_zero"
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="disagrees"):
        _write_and_load(tmp_path, document)


def test_boundary_formula_changed_rejected(tmp_path):
    document = _load_raw()
    document["primitive"]["boundary_formula"] = "boundary = eroded_mask - mask"
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="disagrees"):
        _write_and_load(tmp_path, document)


def test_fixed_sensitivity_widths_changed_rejected(tmp_path):
    document = _load_raw()
    document["primitive"]["fixed_sensitivity_widths"] = [1, 2, 3]
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_ignore_policy_changed_rejected(tmp_path):
    document = _load_raw()
    document["semantic_extension"]["ignore_policy"] = "a different ignore policy"
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="disagrees"):
        _write_and_load(tmp_path, document)


def test_zero_union_policy_changed_rejected(tmp_path):
    document = _load_raw()
    document["semantic_extension"]["zero_union_policy"] = "class bIoU falls back to zero when union is zero"
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="disagrees"):
        _write_and_load(tmp_path, document)


def test_is_official_semantic_definition_flipped_rejected(tmp_path):
    document = _load_raw()
    document["semantic_extension"]["is_official_semantic_definition"] = True
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_protocol_path_changed_rejected(tmp_path):
    document = _load_raw()
    document["protocol_binding"]["protocol_identity_relative_path"] = "evaluation_identities/some_other.toml"
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_protocol_hash_changed_rejected(tmp_path):
    document = _load_raw()
    document["protocol_binding"]["protocol_identity_sha256"] = "f" * 64
    identity = _write_and_load(tmp_path, document)
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="SHA256 mismatch"):
        validate_static_configuration(repo_root=ROOT, identity_path=tmp_path / "identity.toml")


def test_required_deferred_status_changed_rejected(tmp_path):
    document = _load_raw()
    document["protocol_binding"]["required_deferred_status"] = "bound"
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Whitespace / case / punctuation adversarial matrix on frozen strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section,field",
    [
        ("paper", "title"),
        ("paper", "authors"),
        ("official_source", "repository_url"),
        ("primitive", "boundary_formula"),
        ("semantic_extension", "statement"),
        ("semantic_extension", "ignore_policy"),
    ],
)
def test_whitespace_padding_rejected(tmp_path, section, field):
    document = _load_raw()
    document[section][field] = f"  {document[section][field]}  "
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="disagrees"):
        _write_and_load(tmp_path, document)


@pytest.mark.parametrize(
    "section,field",
    [
        ("paper", "title"),
        ("paper", "authors"),
        ("primitive", "erosion_backend"),
        ("semantic_extension", "label"),
    ],
)
def test_case_change_rejected(tmp_path, section, field):
    document = _load_raw()
    document[section][field] = document[section][field].upper()
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="disagrees"):
        _write_and_load(tmp_path, document)


def test_punctuation_change_rejected(tmp_path):
    document = _load_raw()
    document["primitive"]["boundary_formula"] = document["primitive"]["boundary_formula"].replace("=", ":=")
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError, match="disagrees"):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Unsafe paths
# ---------------------------------------------------------------------------


def test_unsafe_absolute_license_notice_path_rejected(tmp_path):
    document = _load_raw()
    document["local_reproduction"]["license_notice_relative_path"] = "/etc/passwd"
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_unsafe_traversal_license_notice_path_rejected(tmp_path):
    document = _load_raw()
    document["local_reproduction"]["license_notice_relative_path"] = "../../etc/passwd"
    with pytest.raises(DfpPhase1aBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_load_identity_input_immutability_across_mutations(tmp_path):
    document = _load_raw()
    before = copy.deepcopy(document)
    _write_and_load(tmp_path, document)
    assert document == before


def test_validate_static_configuration_does_not_mutate_source_file():
    before = IDENTITY_PATH.read_bytes()
    validate_static_configuration(repo_root=ROOT)
    after = IDENTITY_PATH.read_bytes()
    assert before == after
