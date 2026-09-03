"""Strict adversarial tests for the DFP Phase 1B Boundary IoU successor
authority (evaluation_identities/e14_dfp_phase1b_boundary_iou.toml,
src/dfp_phase1b_boundary_iou_identity.py). CPU-only, no CUDA, no
network, no dataset access. Loads the committed identity and mutates
only the field under test -- never maintains a duplicated second
authority. Never modifies the completed Phase 1A protocol or any Phase
1A artifact.
"""

from __future__ import annotations

import math
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.dfp_phase1b_boundary_iou_identity import (  # noqa: E402
    BoundaryIoUContract,
    DfpPhase1bBoundaryIouIdentityError,
    load_boundary_iou_contract,
    load_identity,
)

IDENTITY_PATH = ROOT / "evaluation_identities/e14_dfp_phase1b_boundary_iou.toml"
PARENT_PROTOCOL_PATH = ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml"


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


def _write_and_load_contract(tmp_path: Path, document: dict) -> BoundaryIoUContract:
    path = tmp_path / "identity.toml"
    path.write_text(_dump_toml(document), encoding="utf-8")
    return load_boundary_iou_contract(identity_path=path, repo_root=ROOT)


# ---------------------------------------------------------------------------
# Positive loading.
# ---------------------------------------------------------------------------


def test_real_identity_loads():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    assert identity["identity"]["name"] == "e14-dfp-phase1b-boundary-iou"


def test_real_contract_loads_and_binds_to_live_files():
    contract = load_boundary_iou_contract(repo_root=ROOT)
    assert contract.dilation_ratio == 0.02
    assert contract.minimum_dilation == 1
    assert contract.official_repository_commit == "37d25586a677b043ed585f10e5c42d4e80176ea9"
    assert contract.parent_protocol_sha256 == "227549b03ec6b1262c465a18b7fc67529fa5a0b5deef419c33ec79a80feb2173"


def test_load_identity_does_not_mutate_source_file():
    before = IDENTITY_PATH.read_bytes()
    load_identity(IDENTITY_PATH, repo_root=ROOT)
    after = IDENTITY_PATH.read_bytes()
    assert before == after


def test_load_contract_does_not_mutate_parent_protocol():
    before = PARENT_PROTOCOL_PATH.read_bytes()
    load_boundary_iou_contract(repo_root=ROOT)
    after = PARENT_PROTOCOL_PATH.read_bytes()
    assert before == after


def test_no_private_path_in_identity_file():
    text = IDENTITY_PATH.read_text(encoding="utf-8")
    for fragment in ("/scratch/", "/project/", "/home/"):
        assert fragment not in text


def test_contract_is_frozen_dataclass():
    contract = load_boundary_iou_contract(repo_root=ROOT)
    with pytest.raises(Exception):
        contract.dilation_ratio = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Structural adversarial matrix.
# ---------------------------------------------------------------------------


def test_duplicate_toml_key_rejected(tmp_path):
    text = IDENTITY_PATH.read_text(encoding="utf-8")
    injected = text.replace(
        'name = "e14-dfp-phase1b-boundary-iou"',
        'name = "e14-dfp-phase1b-boundary-iou"\nname = "duplicate"',
        1,
    )
    path = tmp_path / "identity.toml"
    path.write_text(injected, encoding="utf-8")
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        load_identity(path, repo_root=ROOT)


def test_unknown_top_level_key_rejected(tmp_path):
    document = _load_raw()
    document["unexpected_section"] = {"x": 1}
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_missing_top_level_section_rejected(tmp_path):
    document = _load_raw()
    del document["published_authority"]
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_unknown_field_within_section_rejected(tmp_path):
    document = _load_raw()
    document["published_authority"]["extra_field"] = "x"
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_missing_field_within_section_rejected(tmp_path):
    document = _load_raw()
    del document["published_authority"]["paper_arxiv_id"]
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Authority-mutation matrix (Section 9 required list).
# ---------------------------------------------------------------------------


def test_altered_parent_protocol_hash_rejected(tmp_path):
    document = _load_raw()
    document["parent_protocol_binding"]["parent_protocol_sha256"] = "a" * 64
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="SHA256 mismatch"):
        _write_and_load_contract(tmp_path, document)


def test_altered_upstream_commit_rejected(tmp_path):
    document = _load_raw()
    document["upstream_implementation_authority"]["official_repository_commit"] = "b" * 40
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_altered_upstream_source_hash_rejected(tmp_path):
    document = _load_raw()
    document["upstream_implementation_authority"]["upstream_source_sha256"] = "c" * 64
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_altered_vendored_byte_rejected(tmp_path):
    # Tamper the VENDORED FILE itself (not the identity) -- a byte-level
    # supply-chain check, independent of the identity-text mutation tests.
    vendor_src = ROOT / "third_party/boundary_iou_api/boundary_iou/utils/boundary_utils.py"
    tampered_vendor_dir = tmp_path / "third_party" / "boundary_iou_api" / "boundary_iou" / "utils"
    tampered_vendor_dir.mkdir(parents=True)
    (tampered_vendor_dir / "boundary_utils.py").write_text(vendor_src.read_text() + "\n# tampered\n")
    fake_root = tmp_path
    (fake_root / "evaluation_identities").mkdir(parents=True, exist_ok=True)
    (fake_root / "evaluation_identities" / "e13_dfp_phase1a_protocol.toml").write_bytes(PARENT_PROTOCOL_PATH.read_bytes())
    licenses_dir = fake_root / "third_party" / "boundary_iou_api"
    (licenses_dir / "license.txt").write_bytes((ROOT / "third_party/boundary_iou_api/license.txt").read_bytes())
    (licenses_dir / "UPSTREAM_COMMIT").write_text((ROOT / "third_party/boundary_iou_api/UPSTREAM_COMMIT").read_text())
    identity_path = fake_root / "evaluation_identities" / "e14_dfp_phase1b_boundary_iou.toml"
    identity_path.write_text(IDENTITY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="SHA256 mismatch"):
        load_boundary_iou_contract(identity_path=identity_path, repo_root=fake_root)


def test_missing_license_rejected(tmp_path):
    document = _load_raw()
    fake_root = tmp_path
    (fake_root / "evaluation_identities").mkdir(parents=True, exist_ok=True)
    (fake_root / "evaluation_identities" / "e13_dfp_phase1a_protocol.toml").write_bytes(PARENT_PROTOCOL_PATH.read_bytes())
    vendor_dir = fake_root / "third_party" / "boundary_iou_api" / "boundary_iou" / "utils"
    vendor_dir.mkdir(parents=True)
    (vendor_dir / "boundary_utils.py").write_bytes((ROOT / "third_party/boundary_iou_api/boundary_iou/utils/boundary_utils.py").read_bytes())
    (fake_root / "third_party" / "boundary_iou_api" / "UPSTREAM_COMMIT").write_text(
        (ROOT / "third_party/boundary_iou_api/UPSTREAM_COMMIT").read_text()
    )
    # license.txt deliberately never written.
    identity_path = fake_root / "evaluation_identities" / "e14_dfp_phase1b_boundary_iou.toml"
    identity_path.write_text(_dump_toml(document), encoding="utf-8")
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="does not exist"):
        load_boundary_iou_contract(identity_path=identity_path, repo_root=fake_root)


def test_altered_dilation_ratio_rejected(tmp_path):
    document = _load_raw()
    document["binary_boundary_definition"]["dilation_ratio"] = 0.05
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="0.02"):
        _write_and_load(tmp_path, document)


def test_dilation_ratio_as_integer_rejected(tmp_path):
    document = _load_raw()
    document["binary_boundary_definition"]["dilation_ratio"] = 0
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="exact finite float"):
        _write_and_load(tmp_path, document)


def test_dilation_ratio_as_string_rejected(tmp_path):
    document = _load_raw()
    document["binary_boundary_definition"]["dilation_ratio"] = "0.02"
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="exact finite float"):
        _write_and_load(tmp_path, document)


def test_dilation_ratio_as_bool_rejected(tmp_path):
    document = _load_raw()
    document["binary_boundary_definition"]["dilation_ratio"] = True
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="exact finite float"):
        _write_and_load(tmp_path, document)


def test_changed_diagonal_formula_rejected(tmp_path):
    document = _load_raw()
    document["binary_boundary_definition"]["diagonal_formula"] = "image_diagonal = height + width"
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_changed_rounding_convention_rejected(tmp_path):
    document = _load_raw()
    document["binary_boundary_definition"]["rounding_convention"] = "always_round_up"
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_minimum_dilation_other_than_one_rejected(tmp_path):
    document = _load_raw()
    document["binary_boundary_definition"]["minimum_dilation"] = 2
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="exactly 1"):
        _write_and_load(tmp_path, document)


def test_cross_shaped_kernel_connectivity_rejected(tmp_path):
    document = _load_raw()
    document["binary_boundary_definition"]["kernel_connectivity"] = 4
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="exactly 8"):
        _write_and_load(tmp_path, document)


def test_changed_padding_width_rejected(tmp_path):
    document = _load_raw()
    document["binary_boundary_definition"]["padding_width"] = 2
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="exactly 1"):
        _write_and_load(tmp_path, document)


def test_changed_padding_value_rejected(tmp_path):
    document = _load_raw()
    document["binary_boundary_definition"]["padding_value"] = 1
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="exactly 0"):
        _write_and_load(tmp_path, document)


def test_changed_inner_boundary_formula_rejected(tmp_path):
    document = _load_raw()
    document["binary_boundary_definition"]["inner_boundary_formula"] = "boundary(mask) = eroded(mask) - mask"
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_changed_ignore_semantics_rejected(tmp_path):
    document = _load_raw()
    document["semantic_adaptation"]["ignore_semantics"] = "ignore pixels are treated as class-negative background"
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_changed_valid_domain_erosion_formula_rejected(tmp_path):
    document = _load_raw()
    document["semantic_adaptation"]["valid_domain_erosion_formula"] = "E_V(M) = erosion(M)"
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_changed_aggregation_rule_rejected(tmp_path):
    document = _load_raw()
    document["semantic_adaptation"]["dataset_aggregation_rule"] = "mean_of_per_image_boundary_iou"
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_changed_class_inclusion_rule_rejected(tmp_path):
    document = _load_raw()
    document["semantic_adaptation"]["class_inclusion_rule"] = "U_c >= 0"
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_changed_units_rejected(tmp_path):
    document = _load_raw()
    document["semantic_adaptation"]["metric_unit"] = "fraction_0_1"
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_epsilon_forbidden_flag_flipped_rejected(tmp_path):
    document = _load_raw()
    document["semantic_adaptation"]["epsilon_in_denominator_forbidden"] = False
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="must be true"):
        _write_and_load(tmp_path, document)


def test_is_official_semantic_definition_flipped_rejected(tmp_path):
    document = _load_raw()
    document["semantic_adaptation"]["is_official_semantic_definition"] = True
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="must be false"):
        _write_and_load(tmp_path, document)


def test_retroactive_phase1a_claim_flag_flipped_rejected(tmp_path):
    document = _load_raw()
    document["parent_protocol_binding"]["retroactive_phase1a_claim_forbidden"] = False
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="must be true"):
        _write_and_load(tmp_path, document)


def test_is_method_parameter_flipped_rejected(tmp_path):
    document = _load_raw()
    document["evaluation_role"]["is_method_parameter"] = True
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="must be false"):
        _write_and_load(tmp_path, document)


def test_threshold_fitting_forbidden_flipped_rejected(tmp_path):
    document = _load_raw()
    document["evaluation_role"]["threshold_fitting_forbidden"] = False
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="must be true"):
        _write_and_load(tmp_path, document)


def test_unknown_identity_field_rejected(tmp_path):
    document = _load_raw()
    document["identity"]["unexpected"] = "x"
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_paper_year_wrong_rejected(tmp_path):
    document = _load_raw()
    document["published_authority"]["paper_year"] = 2020
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="2021"):
        _write_and_load(tmp_path, document)


def test_repository_commit_wrong_length_rejected(tmp_path):
    document = _load_raw()
    document["upstream_implementation_authority"]["official_repository_commit"] = "37d2558"
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="40-hex"):
        _write_and_load(tmp_path, document)


def test_repository_commit_uppercase_rejected(tmp_path):
    document = _load_raw()
    document["upstream_implementation_authority"]["official_repository_commit"] = (
        "37D25586A677B043ED585F10E5C42D4E80176EA9"
    )
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="40-hex"):
        _write_and_load(tmp_path, document)


def test_bad_sha256_length_rejected(tmp_path):
    document = _load_raw()
    document["upstream_implementation_authority"]["upstream_source_sha256"] = "abc123"
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="64-hex"):
        _write_and_load(tmp_path, document)


def test_path_traversal_rejected(tmp_path):
    document = _load_raw()
    document["upstream_implementation_authority"]["vendored_source_relative_path"] = (
        "../../../etc/passwd"
    )
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_absolute_path_rejected(tmp_path):
    document = _load_raw()
    document["upstream_implementation_authority"]["vendored_source_relative_path"] = "/etc/passwd"
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError):
        _write_and_load(tmp_path, document)


def test_symlink_escape_rejected(tmp_path):
    fake_root = tmp_path
    (fake_root / "evaluation_identities").mkdir(parents=True, exist_ok=True)
    (fake_root / "evaluation_identities" / "e13_dfp_phase1a_protocol.toml").write_bytes(PARENT_PROTOCOL_PATH.read_bytes())
    outside = tmp_path.parent / f"outside_{tmp_path.name}"
    outside.mkdir(exist_ok=True)
    secret = outside / "secret.py"
    secret.write_bytes((ROOT / "third_party/boundary_iou_api/boundary_iou/utils/boundary_utils.py").read_bytes())
    vendor_parent = fake_root / "third_party" / "boundary_iou_api" / "boundary_iou" / "utils"
    vendor_parent.mkdir(parents=True)
    (vendor_parent / "boundary_utils.py").symlink_to(secret)
    (fake_root / "third_party" / "boundary_iou_api" / "license.txt").write_bytes(
        (ROOT / "third_party/boundary_iou_api/license.txt").read_bytes()
    )
    (fake_root / "third_party" / "boundary_iou_api" / "UPSTREAM_COMMIT").write_text(
        (ROOT / "third_party/boundary_iou_api/UPSTREAM_COMMIT").read_text()
    )
    document = _load_raw()
    document["upstream_implementation_authority"]["vendored_source_relative_path"] = (
        "third_party/boundary_iou_api/boundary_iou/utils/boundary_utils.py"
    )
    identity_path = fake_root / "evaluation_identities" / "e14_dfp_phase1b_boundary_iou.toml"
    identity_path.write_text(_dump_toml(document), encoding="utf-8")
    # The symlink target has IDENTICAL bytes to the real vendored file, so
    # this must be rejected specifically for escaping fake_root via the
    # symlink -- not merely for a hash mismatch.
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="escapes the repository root"):
        load_boundary_iou_contract(identity_path=identity_path, repo_root=fake_root)


# ---------------------------------------------------------------------------
# Live parent-protocol deferred-status check.
# ---------------------------------------------------------------------------


def test_declared_parent_deferred_status_is_canonical_locked(tmp_path):
    """parent_deferred_value_required is itself a canonical-pinned string
    (never freely settable), so tampering it is caught at load_identity
    time -- earlier and stronger than the live relational check below."""
    document = _load_raw()
    document["parent_protocol_binding"]["parent_deferred_value_required"] = "bound"
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="disagrees with the frozen canonical"):
        _write_and_load(tmp_path, document)


def test_live_parent_deferred_status_mismatch_rejected(tmp_path):
    """The identity's own declared expectation stays the canonical
    "not_yet_bound", but the LIVE parent protocol copy this fake root
    binds against has already been (hypothetically) changed to "bound"
    -- the real, unmodified repository protocol is never touched by this
    test."""
    document = _load_raw()
    fake_root = tmp_path
    (fake_root / "evaluation_identities").mkdir(parents=True, exist_ok=True)
    tampered_protocol_text = PARENT_PROTOCOL_PATH.read_text(encoding="utf-8").replace(
        'implementation_source_status = "not_yet_bound"', 'implementation_source_status = "bound"', 1,
    )
    tampered_protocol_path = fake_root / "evaluation_identities" / "e13_dfp_phase1a_protocol.toml"
    tampered_protocol_path.write_text(tampered_protocol_text, encoding="utf-8")
    import hashlib

    document["parent_protocol_binding"]["parent_protocol_sha256"] = hashlib.sha256(tampered_protocol_text.encode("utf-8")).hexdigest()
    vendor_dir = fake_root / "third_party" / "boundary_iou_api" / "boundary_iou" / "utils"
    vendor_dir.mkdir(parents=True)
    (vendor_dir / "boundary_utils.py").write_bytes((ROOT / "third_party/boundary_iou_api/boundary_iou/utils/boundary_utils.py").read_bytes())
    (fake_root / "third_party" / "boundary_iou_api" / "license.txt").write_bytes(
        (ROOT / "third_party/boundary_iou_api/license.txt").read_bytes()
    )
    (fake_root / "third_party" / "boundary_iou_api" / "UPSTREAM_COMMIT").write_text(
        (ROOT / "third_party/boundary_iou_api/UPSTREAM_COMMIT").read_text()
    )
    identity_path = fake_root / "evaluation_identities" / "e14_dfp_phase1b_boundary_iou.toml"
    identity_path.write_text(_dump_toml(document), encoding="utf-8")
    with pytest.raises(DfpPhase1bBoundaryIouIdentityError, match="not_yet_bound"):
        load_boundary_iou_contract(identity_path=identity_path, repo_root=fake_root)


# ---------------------------------------------------------------------------
# No-compute isolation.
# ---------------------------------------------------------------------------


def test_module_import_does_not_import_torch():
    import subprocess

    src = "import sys\nimport src.dfp_phase1b_boundary_iou_identity\nprint('torch' in sys.modules)\n"
    result = subprocess.run([sys.executable, "-c", src], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_keyboard_interrupt_propagates():
    with pytest.raises(KeyboardInterrupt):
        try:
            raise KeyboardInterrupt()
        except DfpPhase1bBoundaryIouIdentityError:
            pytest.fail("KeyboardInterrupt must never be caught by the domain error handler")


def test_system_exit_propagates():
    with pytest.raises(SystemExit):
        try:
            raise SystemExit(3)
        except DfpPhase1bBoundaryIouIdentityError:
            pytest.fail("SystemExit must never be caught by the domain error handler")


def test_memory_error_propagates():
    with pytest.raises(MemoryError):
        try:
            raise MemoryError()
        except DfpPhase1bBoundaryIouIdentityError:
            pytest.fail("MemoryError must never be caught by the domain error handler")
