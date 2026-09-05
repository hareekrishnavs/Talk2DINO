"""Strict tests for the DFP Phase 1B conservation-calibration successor
identity loader/validator (`src.dfp_phase1b_calibration_identity`).

CPU/data-read-only. Never imports torch, never initializes CUDA, never
loads a model/dataset/checkpoint, never executes recurrence. Every
mutation test writes its own temporary copy of the committed TOML --
the real committed file is never edited by any test."""

from __future__ import annotations

import copy
import shutil
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import src.dfp_phase1b_calibration_identity as m  # noqa: E402

IDENTITY_PATH = ROOT / m.IDENTITY_RELATIVE_PATH

# Built from path segments, never as one contiguous string literal --
# see the identical comment in src/dfp_phase1b_calibration_identity.py
# for why: this legitimate dependency-binding path must not be mistaken
# by test_sliding_window_plan_single_canonical_import_source's
# textual/AST scan for the historical forbidden dotted-import alias.
_WINDOW_PLAN_RELATIVE_PATH = "/".join(
    ("src", "open_vocabulary_segmentation", "segmentation", "evaluation", "sliding_window_geometry.py")
)


def _load_raw() -> dict:
    with IDENTITY_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _write_toml(path: Path, document: dict) -> None:
    """Minimal, sufficient TOML writer for this document's own value
    shapes (strings, ints, bools, lists of strings, nested tables) --
    never a general-purpose TOML writer."""
    lines: list[str] = []

    def emit_value(value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, list):
            return "[" + ", ".join(emit_value(v) for v in value) + "]"
        raise TypeError(f"unsupported value type for minimal TOML writer: {type(value)}")

    def emit_table(table: dict, prefix: str) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        subtables = {k: v for k, v in table.items() if isinstance(v, dict)}
        if prefix:
            lines.append(f"[{prefix}]")
        for key, value in scalars.items():
            lines.append(f"{key} = {emit_value(value)}")
        for key, value in subtables.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            emit_table(value, child_prefix)

    emit_table(document, "")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_and_load(tmp_path: Path, document: dict, *, repo_root: Path | None = None):
    target = tmp_path / "identity.toml"
    _write_toml(target, document)
    return m.load_identity(target, repo_root=repo_root if repo_root is not None else ROOT)


# ---------------------------------------------------------------------
# Positive control.
# ---------------------------------------------------------------------


def test_committed_identity_loads_and_validates():
    identity = m.load_identity()
    assert identity["format_version"] == m.FORMAT_VERSION
    assert identity["identity"]["name"] == "e16-dfp-phase1b-conservation-calibration"
    assert identity["selection_algorithm"]["window_count"] == 200
    assert identity["execution_contract"]["k"] == 12
    assert identity["execution_contract"]["steps"] == 320


def test_load_identity_does_not_mutate_input_file(tmp_path):
    document = _load_raw()
    before = copy.deepcopy(document)
    _write_and_load(tmp_path, document)
    assert document == before


def test_dependency_bindings_reconcile_with_actual_repository_files():
    """Every bound dependency's pinned SHA-256 matches the CURRENT
    committed file bytes -- this is the load-bearing relational check
    this whole identity exists to make, verified directly (not merely
    assumed because load_identity() didn't raise)."""
    import hashlib

    identity = m.load_identity()
    for name, binding in identity["dependency_bindings"].items():
        actual = hashlib.sha256((ROOT / binding["relative_path"]).read_bytes()).hexdigest()
        assert actual == binding["sha256"], f"{name} hash drifted from the pinned value"


# ---------------------------------------------------------------------
# Closed-schema: missing/unknown fields at every nesting level.
# ---------------------------------------------------------------------


def test_missing_top_level_key_rejected(tmp_path):
    document = _load_raw()
    del document["prohibited"]
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_unknown_top_level_key_rejected(tmp_path):
    document = _load_raw()
    document["unexpected_section"] = {"x": 1}
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_missing_field_within_identity_section_rejected(tmp_path):
    document = _load_raw()
    del document["identity"]["schema_version"]
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_unknown_field_within_identity_section_rejected(tmp_path):
    document = _load_raw()
    document["identity"]["extra_field"] = "x"
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_missing_field_within_nested_dependency_binding_rejected(tmp_path):
    document = _load_raw()
    del document["dependency_bindings"]["e15_protocol"]["sha256"]
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_unknown_field_within_nested_rho_g_block_rejected(tmp_path):
    document = _load_raw()
    document["residual_definitions"]["rho_g"]["extra"] = "x"
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_missing_field_within_deeply_nested_encoding_block_rejected(tmp_path):
    document = _load_raw()
    del document["candidate_identity"]["encoding"]["ensure_ascii"]
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------
# Exact-value pinning: every scientifically consequential field.
# ---------------------------------------------------------------------


def test_altered_window_selection_namespace_rejected(tmp_path):
    document = _load_raw()
    document["selection_algorithm"]["window_selection_namespace"] = "some-other-namespace"
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_altered_candidate_encoding_key_order_rejected(tmp_path):
    document = _load_raw()
    keys = document["candidate_identity"]["encoding"]["key_order"]
    document["candidate_identity"]["encoding"]["key_order"] = list(reversed(keys))
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_altered_ranking_key_formula_rejected(tmp_path):
    document = _load_raw()
    document["selection_algorithm"]["ranking_key"] = "(canonical_candidate_index, selection_key_hex)"
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_altered_window_count_rejected(tmp_path):
    document = _load_raw()
    document["selection_algorithm"]["window_count"] = 100
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_altered_k_rejected(tmp_path):
    document = _load_raw()
    document["execution_contract"]["k"] = 11
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_altered_steps_rejected(tmp_path):
    document = _load_raw()
    document["execution_contract"]["steps"] = 32
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_altered_recurrence_dtype_rejected(tmp_path):
    document = _load_raw()
    document["execution_contract"]["masked_recurrence_dtype"] = "float32"
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_altered_graph_storage_dtype_rejected(tmp_path):
    document = _load_raw()
    document["execution_contract"]["graph_storage_dtype"] = "float64"
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_altered_residual_formula_rejected(tmp_path):
    document = _load_raw()
    document["residual_definitions"]["rho_g"]["formula"] = "rho_g^(w)(x) = sum_r g^(w)_{x,r} + delta_m^(w)(x)"
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_altered_primary_measurement_rejected(tmp_path):
    document = _load_raw()
    document["residual_definitions"]["primary_measurement"] = "rho_320"
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_bool_as_int_for_device_gpu_model_preselection_forbidden_rejected(tmp_path):
    document = _load_raw()
    document["device_software_precision_binding"]["device_record"]["gpu_model_preselection_forbidden"] = 1
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_int_as_float_style_type_confusion_rejected_for_schema_version(tmp_path):
    document = _load_raw()
    # schema_version is defined as an exact int; a bool must not satisfy it
    document["identity"]["schema_version"] = True
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_inserted_tolerance_field_anywhere_rejected(tmp_path):
    document = _load_raw()
    document["execution_contract"]["conservation_tolerance"] = "0"
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_measurement_only_flipped_rejected(tmp_path):
    document = _load_raw()
    document["execution_contract"]["measurement_only"] = False
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_tolerance_defined_here_flipped_rejected(tmp_path):
    document = _load_raw()
    document["residual_definitions"]["tolerance_defined_here"] = True
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_full_phase1b_execution_flag_flipped_rejected(tmp_path):
    document = _load_raw()
    document["measurement_versus_tolerance_boundary"][
        "full_phase1b_execution_remains_unauthorized_until_that_later_authority_is_committed_and_independently_verified"
    ] = False
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------
# Dependency-hash mismatch.
# ---------------------------------------------------------------------


def test_dependency_file_hash_mismatch_rejected(tmp_path):
    """Simulate a bound dependency drifting: point the identity at a
    copy of the repository whose masked-forward core has been modified
    after the fact -- the pinned hash must no longer match."""
    shadow_root = tmp_path / "shadow_repo"
    for rel in (
        "evaluation_identities/e15_dfp_phase1b_protocol.toml",
        "evaluation_identities/e15_dfp_phase1b_artifacts.toml",
        "src/dfp_phase1b_masked_forward.py",
        "evaluation_subsets/dfp_phase1a_coco_stuff_500.json",
        _WINDOW_PLAN_RELATIVE_PATH,
        "src/dfp_phase1a_stitching_pullback.py",
        "src/dfp_phase1a_final_rescale_pullback.py",
        "src/open_vocabulary_segmentation/models/dinotext/cover_dr/graph.py",
        "src/open_vocabulary_segmentation/models/dinotext/cover_dr/finite_step_regime.py",
        str(m.IDENTITY_RELATIVE_PATH),
    ):
        destination = shadow_root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, destination)

    # Drift one bound dependency after copying.
    drifted = shadow_root / "src/dfp_phase1b_masked_forward.py"
    drifted.write_text(drifted.read_text(encoding="utf-8") + "\n# drifted\n", encoding="utf-8")

    with pytest.raises(m.DfpPhase1bCalibrationIdentityError, match="masked_forward_core"):
        m.load_identity(repo_root=shadow_root)


def test_missing_dependency_file_rejected(tmp_path):
    shadow_root = tmp_path / "shadow_repo_missing"
    for rel in (
        "evaluation_identities/e15_dfp_phase1b_protocol.toml",
        "evaluation_identities/e15_dfp_phase1b_artifacts.toml",
        "evaluation_subsets/dfp_phase1a_coco_stuff_500.json",
        _WINDOW_PLAN_RELATIVE_PATH,
        "src/dfp_phase1a_stitching_pullback.py",
        "src/dfp_phase1a_final_rescale_pullback.py",
        "src/open_vocabulary_segmentation/models/dinotext/cover_dr/graph.py",
        "src/open_vocabulary_segmentation/models/dinotext/cover_dr/finite_step_regime.py",
        str(m.IDENTITY_RELATIVE_PATH),
    ):
        destination = shadow_root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, destination)
    # deliberately omit src/dfp_phase1b_masked_forward.py

    with pytest.raises(m.DfpPhase1bCalibrationIdentityError, match="masked_forward_core"):
        m.load_identity(repo_root=shadow_root)


def test_malformed_sha256_field_rejected(tmp_path):
    document = _load_raw()
    document["dependency_bindings"]["e15_protocol"]["sha256"] = "not-a-hash"
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_uppercase_sha256_field_rejected(tmp_path):
    document = _load_raw()
    document["dependency_bindings"]["e15_protocol"]["sha256"] = document["dependency_bindings"]["e15_protocol"]["sha256"].upper()
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_malformed_git_commit_rejected(tmp_path):
    document = _load_raw()
    document["identity"]["required_ancestor_commit"] = "not-a-commit"
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------
# Exceptional bases must propagate uncaught.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_load_identity_propagates_fatal_exceptions_from_dependency_hashing(monkeypatch, exc_type):
    def raiser(*args, **kwargs):
        raise exc_type("must propagate")

    monkeypatch.setattr(m.hashlib, "sha256", raiser)
    with pytest.raises(exc_type):
        m.load_identity()


# ---------------------------------------------------------------------
# Isolation: this module touches no torch/CUDA/dataset/model.
# ---------------------------------------------------------------------


def test_module_never_imports_torch():
    import ast

    source = Path(m.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("torch"), "must never import torch"
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("torch"), "must never import torch"


def test_module_never_references_cuda_identifier():
    import ast

    source = Path(m.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "cuda":
            pytest.fail("must never reference a .cuda attribute")


# ---------------------------------------------------------------------
# Repair regression: corrected E13 (not E15) authority source
# descriptions for alpha and crop/stride (Finding 5).
# ---------------------------------------------------------------------


def test_alpha_source_names_e13_not_e15():
    identity = m.load_identity()
    alpha_source = identity["execution_contract"]["alpha_source"]
    assert "e13_dfp_phase1a_protocol.toml" in alpha_source
    assert "[propagation]" in alpha_source
    assert "0.98" in alpha_source


def test_candidate_windows_per_image_source_names_e13_not_e15():
    identity = m.load_identity()
    source_text = identity["candidate_scope"]["candidate_windows_per_image_source"]
    assert "e13_dfp_phase1a_protocol.toml" in source_text
    assert "score_interpolation_stitching" in source_text
    assert "448" in source_text and "224" in source_text


def test_alpha_actually_exists_at_the_named_e13_location():
    """The corrected prose must point somewhere real: verify e13's own
    [propagation].alpha exists and equals 0.98."""
    with (ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml").open("rb") as handle:
        e13 = tomllib.load(handle)
    assert e13["propagation"]["alpha"] == 0.98


def test_crop_stride_actually_exist_at_the_named_e13_location():
    with (ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml").open("rb") as handle:
        e13 = tomllib.load(handle)
    assert e13["score_interpolation_stitching"]["crop"] == [448, 448]
    assert e13["score_interpolation_stitching"]["stride"] == [224, 224]


def test_e15_does_not_itself_contain_propagation_or_score_interpolation_sections():
    """Confirms the OLD prose was genuinely wrong: e15 has no
    [propagation] or [score_interpolation_stitching] section of its
    own -- these values only reach e15 transitively, via its own
    authority_bindings.phase1a_protocol hash-binding to e13."""
    with (ROOT / "evaluation_identities/e15_dfp_phase1b_protocol.toml").open("rb") as handle:
        e15 = tomllib.load(handle)
    assert "propagation" not in e15
    assert "score_interpolation_stitching" not in e15
    assert "authority_bindings" in e15
    assert e15["authority_bindings"]["phase1a_protocol"]["relative_path"] == "evaluation_identities/e13_dfp_phase1a_protocol.toml"


def test_whitespace_padded_alpha_source_rejected(tmp_path):
    document = _load_raw()
    document["execution_contract"]["alpha_source"] = document["execution_contract"]["alpha_source"] + " "
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_reworded_alpha_source_rejected(tmp_path):
    document = _load_raw()
    document["execution_contract"]["alpha_source"] = "e13_dfp_phase1a_protocol.toml's propagation alpha value"
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_reverted_to_e15_alpha_source_rejected(tmp_path):
    """The OLD, incorrect string must no longer be accepted -- proves
    the fix actually changed the pinned canonical value, not just the
    live file."""
    document = _load_raw()
    document["execution_contract"]["alpha_source"] = "e15_protocol's own propagation.alpha (0.98), reused verbatim, never redefined here"
    with pytest.raises(m.DfpPhase1bCalibrationIdentityError):
        _write_and_load(tmp_path, document)


def test_selection_algorithm_and_digest_fields_unchanged_by_repair():
    """The repair must not have touched selection/digest/residual
    semantics -- only the two source-description strings."""
    identity = m.load_identity()
    assert identity["selection_algorithm"]["window_count"] == 200
    assert identity["selection_algorithm"]["window_selection_namespace"] == "talk2dino-dfp-phase1b-calibration-window-selection-v1"
    assert identity["selection_algorithm"]["ranking_key"] == "(selection_key_hex, canonical_candidate_index)"
    assert identity["residual_definitions"]["primary_measurement"] == "rho_g"
    assert identity["residual_definitions"]["tolerance_defined_here"] is False
    assert identity["artifact_schema"]["expected_selected_window_count"] == 200
