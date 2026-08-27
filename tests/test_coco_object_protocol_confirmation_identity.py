"""Identity tests for the COCO-Object protocol-confirmation evaluator.
Every expected value is read from the real, committed identity/parent-
identity files -- never a duplicated scientific literal."""

from __future__ import annotations

import copy
import sys
import tomllib
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.coco_object_protocol_confirmation_identity import (  # noqa: E402
    CocoObjectProtocolConfirmationIdentityError,
    load_identity,
    validate_bridge_checkpoint_binding,
    validate_e3_configuration_binding,
    validate_static_configuration,
)

IDENTITY_PATH = ROOT / "evaluation_identities/e12_coco_object_protocol_confirmation.toml"
pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the protocol-confirmation identity")


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
        if section == "e3_config":
            lines.append("\n[e3_config]")
            for key, value in block.items():
                if key == "resolved_configuration":
                    continue
                lines.append(f"{key} = {emit(value)}")
            lines.append("\n[e3_config.resolved_configuration]")
            for key, value in block["resolved_configuration"].items():
                lines.append(f"{key} = {emit(value)}")
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
    document = _load_raw()
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    assert identity["identity"]["name"] == "e12-coco-object-protocol-confirmation"
    assert identity["protocol"]["split"] == "val2017"
    assert identity["protocol"]["expected_image_count"] == 5000
    assert identity["dataset"]["class_count"] == 81


def test_validate_static_configuration_against_real_repo():
    result = validate_static_configuration(repo_root=ROOT, check_git=True)
    assert result["matched_identity_name"] == "e12-matched-k11-k12-t320"
    assert result["materialization_identity_name"] == "e12-coco-object-val-materialization"


def test_wrong_protocol_label_rejected(tmp_path):
    document = _load_raw()
    document["protocol"]["label"] = "COCO-Stuff"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_split_rejected(tmp_path):
    document = _load_raw()
    document["protocol"]["split"] = "train2017"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_class_count_rejected(tmp_path):
    document = _load_raw()
    document["dataset"]["class_count"] = 80
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_background_class_index_rejected(tmp_path):
    document = _load_raw()
    document["dataset"]["background_class_index"] = 1
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_ignore_index_rejected(tmp_path):
    document = _load_raw()
    document["dataset"]["ignore_index"] = 0
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_annotation_suffix_rejected(tmp_path):
    document = _load_raw()
    document["dataset"]["annotation_suffix"] = "_labelTrainIds.png"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_dataset_type_rejected(tmp_path):
    document = _load_raw()
    document["dataset"]["dataset_type"] = "COCOStuffDataset"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_canonical_configured_root_rejected(tmp_path):
    document = _load_raw()
    document["dataset_root_override"]["canonical_configured_root"] = "/scratch/haree/coco_object_protocol"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_override_field_rejected(tmp_path):
    document = _load_raw()
    document["dataset_root_override"]["override_field"] = "img_dir"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_background_mechanism_alias_rejected(tmp_path):
    document = _load_raw()
    document["background_protocol"]["mechanism"] = "adaptive_threshold"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_background_thresh_wrong_type_rejected(tmp_path):
    document = _load_raw()
    document["background_protocol"]["bg_thresh"] = "0.55"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_pamr_true_rejected(tmp_path):
    document = _load_raw()
    document["e3_config"]["pamr"] = True
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_variants_wrong_order_rejected(tmp_path):
    document = _load_raw()
    document["comparisons"]["variants"] = ["k11", "E3", "k12"]
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_primary_delta_alias_rejected(tmp_path):
    document = _load_raw()
    document["comparisons"]["primary_delta_definition"] = "mIoU_k12_minus_mIoU_k11"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_run_mode_image_count_rejected(tmp_path):
    document = _load_raw()
    document["run_modes"]["pilot20_images"] = 25
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_require_verify_output_before_cuda_flipped_rejected(tmp_path):
    document = _load_raw()
    document["verification"]["require_verify_output_before_cuda"] = False
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_unknown_top_level_key_rejected(tmp_path):
    document = _load_raw()
    document["extra_injected_section"] = {"x": 1}
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_unknown_nested_key_rejected(tmp_path):
    document = _load_raw()
    document["protocol"]["extra_field"] = "y"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_exact_type_int_as_string_rejected(tmp_path):
    document = _load_raw()
    document["protocol"]["expected_image_count"] = "5000"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_wrong_exact_type_bool_as_int_rejected(tmp_path):
    document = _load_raw()
    document["e3_config"]["pamr"] = 0
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_matched_identity_sha256_mismatch_rejected(tmp_path):
    document = _load_raw()
    document["parent_identities"]["matched_identity_sha256"] = "0" * 64
    path = tmp_path / "identity.toml"
    path.write_text(_manual_toml_dump(document))
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_static_configuration(repo_root=ROOT, identity_path=path, check_git=False)


def test_materialization_identity_sha256_mismatch_rejected(tmp_path):
    document = _load_raw()
    document["parent_identities"]["materialization_identity_sha256"] = "0" * 64
    path = tmp_path / "identity.toml"
    path.write_text(_manual_toml_dump(document))
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_static_configuration(repo_root=ROOT, identity_path=path, check_git=False)


def test_missing_identity_file_fails_closed(tmp_path):
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        load_identity(tmp_path / "does-not-exist.toml", repo_root=ROOT)


def test_malformed_toml_fails_closed(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("this is not [valid toml")
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        load_identity(path, repo_root=ROOT)


def test_class_names_digest_algorithm_alias_rejected(tmp_path):
    document = _load_raw()
    document["dataset"]["class_names_digest_algorithm"] = "sha256_hex_joined"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_constructor_class_wrong_rejected(tmp_path):
    document = _load_raw()
    document["e3_config"]["constructor_class"] = "NotDINOText"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_matched_identity_name_mismatch_rejected(tmp_path):
    document = _load_raw()
    document["parent_identities"]["matched_identity_name"] = "e12-matched-k11-k12-t320-wrong"
    path = tmp_path / "identity.toml"
    path.write_text(_manual_toml_dump(document))
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_static_configuration(repo_root=ROOT, identity_path=path, check_git=False)


def test_template_disagrees_with_resolved_runtime_config_rejected(tmp_path):
    document = _load_raw()
    document["e3_config"]["template"] = "openai_imagenet_template"
    path = tmp_path / "identity.toml"
    path.write_text(_manual_toml_dump(document))
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_static_configuration(repo_root=ROOT, identity_path=path, check_git=False)


def test_template_whitespace_alias_rejected(tmp_path):
    document = _load_raw()
    document["e3_config"]["template"] = "sub_imagenet_template "
    path = tmp_path / "identity.toml"
    path.write_text(_manual_toml_dump(document))
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_static_configuration(repo_root=ROOT, identity_path=path, check_git=False)


def test_with_bg_clean_flipped_disagrees_with_resolved_runtime_config_rejected(tmp_path):
    document = _load_raw()
    document["e3_config"]["with_bg_clean"] = False
    path = tmp_path / "identity.toml"
    path.write_text(_manual_toml_dump(document))
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_static_configuration(repo_root=ROOT, identity_path=path, check_git=False)


def test_with_bg_clean_int_1_rejected(tmp_path):
    document = _load_raw()
    document["e3_config"]["with_bg_clean"] = 1
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


# --- validate_e3_configuration_binding self-sufficiency: exercised by
# calling it directly on a hand-built identity mapping that never passed
# through load_identity's own _require_exact_bool gate. Reproduces the
# audit's exact finding: Python's `True == 1` let an int slip through the
# identity-side of the with_bg_clean comparison. ---


@pytest.fixture(scope="module")
def real_identity():
    return load_identity(repo_root=ROOT)


@pytest.mark.parametrize("value", [1, 0, 1.0, 0.0, "true", "false", None])
def test_e3_binding_identity_side_with_bg_clean_wrong_type_rejected(real_identity, value):
    """A. Identity-side with_bg_clean=1 (and B. other wrong identity-side
    values), calling validate_e3_configuration_binding directly and
    bypassing load_identity entirely."""
    mutated = copy.deepcopy(real_identity)
    mutated["e3_config"]["with_bg_clean"] = value
    before_e3_config = copy.deepcopy(mutated["e3_config"])
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError) as excinfo:
        validate_e3_configuration_binding(ROOT, mutated)
    assert "with_bg_clean" in str(excinfo.value)
    # inputs must remain unchanged
    assert mutated["e3_config"] == before_e3_config


def test_e3_binding_identity_side_with_bg_clean_int_1_specifically_rejected(real_identity):
    """The exact scenario from the audit, isolated as its own named test."""
    mutated = copy.deepcopy(real_identity)
    mutated["e3_config"]["with_bg_clean"] = 1
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError, match="with_bg_clean"):
        validate_e3_configuration_binding(ROOT, mutated)


def _synthetic_resolved(real_identity, *, runtime_with_bg_clean):
    return {
        "complete": {"runtime": {"model": {"with_bg_clean": runtime_with_bg_clean}}},
        "dataset": {},
        "dataset_pipeline": [],
        "evaluation": {"template": real_identity["e3_config"]["template"]},
        "model_projection": {},
        "sources": [],
    }


def _run_binding_with_forced_hash_match(real_identity, *, runtime_with_bg_clean, identity_with_bg_clean=None):
    """Force every resolved_configuration hash comparison to pass (via a
    mocked typed_configuration_sha256 returning the expected value in
    call order) and substitute a synthetic resolved-config dict, so the
    with_bg_clean field-level check can be exercised in isolation without
    needing to fabricate a real, hash-consistent resolved configuration."""
    from src.e3_evaluation_identity import resolve_complete_e3_configuration  # noqa: F401 -- import path used by mock.patch below
    from src.typed_configuration import typed_configuration_sha256  # noqa: F401

    identity = copy.deepcopy(real_identity)
    if identity_with_bg_clean is not None:
        identity["e3_config"]["with_bg_clean"] = identity_with_bg_clean
    expected = identity["e3_config"]["resolved_configuration"]
    hash_sequence = iter(
        [
            expected["full_sha256"], expected["dataset_sha256"], expected["dataset_pipeline_sha256"],
            expected["evaluation_sha256"], expected["model_projection_sha256"],
        ]
    )
    resolved = _synthetic_resolved(identity, runtime_with_bg_clean=runtime_with_bg_clean)
    with mock.patch(
        "src.e3_evaluation_identity.resolve_complete_e3_configuration", return_value=resolved
    ), mock.patch(
        "src.typed_configuration.typed_configuration_sha256", side_effect=lambda *a, **k: next(hash_sequence)
    ):
        return validate_e3_configuration_binding(ROOT, identity)


@pytest.mark.parametrize("value", [0, 1, "true"])
def test_e3_binding_runtime_side_with_bg_clean_wrong_type_rejected(real_identity, value):
    """C. Runtime-side wrong exact types: the RESOLVED (production
    configuration-merge) value itself is wrong-typed, independent of the
    identity side."""
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError, match="with_bg_clean"):
        _run_binding_with_forced_hash_match(real_identity, runtime_with_bg_clean=value)


def test_e3_binding_canonical_true_true_passes(real_identity):
    """D. Canonical positive control: runtime True / identity True."""
    resolved = _run_binding_with_forced_hash_match(real_identity, runtime_with_bg_clean=True)
    assert resolved["complete"]["runtime"]["model"]["with_bg_clean"] is True


def test_e3_binding_real_identity_against_real_repo_passes(real_identity):
    """D. Canonical positive control against the real, unmodified
    identity and the real, unmocked configuration-resolution path."""
    resolved = validate_e3_configuration_binding(ROOT, real_identity)
    assert resolved["complete"]["runtime"]["model"]["with_bg_clean"] is True


def test_with_bg_clean_omission_resolves_to_real_constructor_default():
    """D. Canonical omission/default path: when the model YAML omits
    with_bg_clean entirely, production resolution (resolve_complete_e3_configuration,
    via _load_constructor_defaults' AST-derived DINOText.__init__ default)
    must supply a real, exact bool -- never a placeholder or None.

    resolve_complete_e3_configuration requires every configuration source
    to resolve under repo_root (a path-safety invariant), so this uses a
    repo-local scratch directory rather than the OS tmp_path fixture, and
    removes it unconditionally afterwards. No real config file is modified."""
    import shutil

    from src.e3_evaluation_identity import resolve_complete_e3_configuration

    real_identity = load_identity(repo_root=ROOT)
    e3_config = real_identity["e3_config"]
    leaf_path = ROOT / e3_config["eval_config_relative_path"]
    leaf_dir = leaf_path.parent
    leaf_source = leaf_path.read_text()
    assert "with_bg_clean:" in leaf_source, "test fixture assumption: the real leaf yml declares with_bg_clean"
    omitted_leaf = "\n".join(
        line for line in leaf_source.splitlines() if "with_bg_clean:" not in line
    ) + "\n"

    scratch = ROOT / "tests" / "_scratch_with_bg_clean_omission"
    try:
        scratch.mkdir(exist_ok=False)
        tmp_leaf = scratch / "dinotext_coco_object_omitted_with_bg_clean.yml"
        tmp_leaf.write_text(omitted_leaf)
        # Replicate the leaf's own `_base_` chain (default.yml -> eval.yml)
        # alongside the omitted copy so the relative includes resolve.
        (scratch / "default.yml").write_text((leaf_dir / "default.yml").read_text())
        (scratch / "eval.yml").write_text((leaf_dir / "eval.yml").read_text())

        adapter_identity = {
            "evaluation": {
                "config_path": str(tmp_leaf.relative_to(ROOT)),
                "base_config_path": e3_config["eval_base_config_relative_path"],
            },
            "model": {
                "constructor_path": e3_config["model_constructor_relative_path"],
                "constructor_class": e3_config["constructor_class"],
            },
            "dataset": {"config_path": real_identity["dataset"]["dataset_config_relative_path"]},
            "projection": {"config_path": e3_config["projection_config_relative_path"]},
        }
        resolved = resolve_complete_e3_configuration(
            repo_root=ROOT,
            identity=adapter_identity,
            eval_config=tmp_leaf,
            eval_base_config=ROOT / e3_config["eval_base_config_relative_path"],
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    default_value = resolved["complete"]["runtime"]["model"]["with_bg_clean"]
    assert type(default_value) is bool
    assert default_value is False  # DINOText.__init__'s own declared default
    # the real leaf file must remain byte-identical
    assert leaf_path.read_text() == leaf_source


def test_resolved_configuration_full_sha256_tampered_rejected(tmp_path):
    document = _load_raw()
    document["e3_config"]["resolved_configuration"]["full_sha256"] = "0" * 64
    path = tmp_path / "identity.toml"
    path.write_text(_manual_toml_dump(document))
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_static_configuration(repo_root=ROOT, identity_path=path, check_git=False)


def test_materialization_manifest_schema_name_alias_rejected(tmp_path):
    document = _load_raw()
    document["verification"]["materialization_manifest_schema_name"] = (
        "talk2dino-coco-object-val-materialization-manifest-v2"
    )
    path = tmp_path / "identity.toml"
    path.write_text(_manual_toml_dump(document))
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_static_configuration(repo_root=ROOT, identity_path=path, check_git=False)


def test_materialization_identity_name_alias_rejected(tmp_path):
    document = _load_raw()
    document["parent_identities"]["materialization_identity_name"] = "e12-coco-object-val-materialization-v2"
    path = tmp_path / "identity.toml"
    path.write_text(_manual_toml_dump(document))
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_static_configuration(repo_root=ROOT, identity_path=path, check_git=False)


# ---------------------------------------------------------------------
# Bridge/projection checkpoint byte-level hash pin (validate_bridge_
# checkpoint_binding): closes the gap where the pre-existing typed-
# configuration hash chain only pins the checkpoint's PATH/config
# references, never its actual file content.
# ---------------------------------------------------------------------


def test_projection_checkpoint_sha256_field_present_and_well_formed():
    document = _load_raw()
    value = document["e3_config"]["projection_checkpoint_sha256"]
    assert isinstance(value, str)
    assert len(value) == 64
    assert all(c in "0123456789abcdef" for c in value)


def test_projection_checkpoint_sha256_missing_field_rejected(tmp_path):
    document = _load_raw()
    del document["e3_config"]["projection_checkpoint_sha256"]
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_projection_checkpoint_sha256_wrong_format_rejected(tmp_path):
    document = _load_raw()
    document["e3_config"]["projection_checkpoint_sha256"] = "not-a-valid-sha256"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_projection_checkpoint_sha256_uppercase_rejected(tmp_path):
    document = _load_raw()
    document["e3_config"]["projection_checkpoint_sha256"] = document["e3_config"]["projection_checkpoint_sha256"].upper()
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _write_and_load(tmp_path, document)


def test_validate_bridge_checkpoint_binding_accepts_real_checkpoint():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    observed = validate_bridge_checkpoint_binding(ROOT, identity)
    assert observed == identity["e3_config"]["projection_checkpoint_sha256"]


def test_validate_bridge_checkpoint_binding_rejects_wrong_hash():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    tampered = copy.deepcopy(identity)
    tampered["e3_config"]["projection_checkpoint_sha256"] = "f" * 64
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError, match="SHA256 mismatch"):
        validate_bridge_checkpoint_binding(ROOT, tampered)


def test_validate_bridge_checkpoint_binding_rejects_missing_file(tmp_path):
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError, match="cannot read"):
        validate_bridge_checkpoint_binding(tmp_path, identity)


def test_validate_static_configuration_calls_bridge_checkpoint_binding(tmp_path):
    """A checkpoint-content mismatch (identity hash tampered, everything
    else genuine) must fail validate_static_configuration end-to-end --
    not just the standalone validator function."""
    document = _load_raw()
    document["e3_config"]["projection_checkpoint_sha256"] = "e" * 64
    path = tmp_path / "identity.toml"
    path.write_text(_manual_toml_dump(document))
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError, match="SHA256 mismatch"):
        validate_static_configuration(repo_root=ROOT, identity_path=path, check_git=False)
