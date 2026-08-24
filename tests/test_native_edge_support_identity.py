"""CPU-only coverage for src.native_edge_support_identity: valid identity
loading, missing/unknown-field rejection, exact-type rejection, altered
native-alignment/support-definition/ranking/undefined-reason rejection, and
parent-identity mismatch detection. Never initializes CUDA, never loads the
model or dataset."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.native_edge_support_identity import (  # noqa: E402
    NativeEdgeSupportAuditIdentityError,
    load_identity,
    validate_static_configuration,
)

IDENTITY_PATH = ROOT / "evaluation_identities/e12_native_edge_support_audit.toml"

pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the native-edge-support-audit identity")


def _raw() -> dict:
    with IDENTITY_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _write_toml_manually(path: Path, data: dict) -> None:
    """Minimal, dependency-free TOML writer (see
    tests/test_stitching_control_identity.py for the identical rationale):
    sufficient for this identity's shape, used only to re-serialize a
    mutated copy of the real identity dict for negative-path tests."""
    lines: list[str] = []

    def fmt_value(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, list):
            return "[" + ", ".join(fmt_value(x) for x in v) + "]"
        raise TypeError(f"unsupported TOML value type: {type(v)}")

    def write_table(prefix, table):
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        subtables = {k: v for k, v in table.items() if isinstance(v, dict)}
        if prefix:
            lines.append(f"[{prefix}]")
        for k, v in scalars.items():
            lines.append(f"{k} = {fmt_value(v)}")
        lines.append("")
        for k, v in subtables.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            write_table(new_prefix, v)

    write_table("", data)
    path.write_text("\n".join(lines))


def _mutated_load(tmp_path, mutate):
    data = _raw()
    mutate(data)
    path = Path(tmp_path) / "identity.toml"
    _write_toml_manually(path, data)
    return load_identity(path, repo_root=ROOT)


# ---------------------------------------------------------------------------
# Valid identity
# ---------------------------------------------------------------------------


def test_valid_identity_loads():
    identity = load_identity(repo_root=ROOT)
    assert identity["identity"]["name"] == "e12-native-edge-support-audit"
    assert identity["propagation"]["k"] == 12
    assert identity["geometry"]["patch_size"] == [14, 14]
    assert identity["geometry"]["grid_size"] == [32, 32]


def test_static_configuration_validates_full_parent_chain():
    result = validate_static_configuration(repo_root=ROOT, check_git=True)
    assert result["identity_name"] == "e12-native-edge-support-audit"
    assert result["stitching_control_identity"] == "e12-stitching-control-suite"
    assert result["power_evaluation_identity"] == "e12-k11-k12-power-evaluation"
    assert result["matched_identity"] == "e12-matched-k11-k12-t320"
    assert result["mechanics20_image_count"] == 20


# ---------------------------------------------------------------------------
# Missing/unknown fields
# ---------------------------------------------------------------------------


def test_missing_top_level_section_rejected(tmp_path):
    def mutate(d):
        del d["ranking"]

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_unknown_top_level_field_rejected(tmp_path):
    def mutate(d):
        d["unexpected_section"] = {"x": 1}

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_unknown_field_in_known_section_rejected(tmp_path):
    def mutate(d):
        d["support_definition"]["extra_field"] = True

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_missing_field_in_known_section_rejected(tmp_path):
    def mutate(d):
        del d["clamped_window_policy"]["tolerance_radius_patches"]

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Exact-type rejection
# ---------------------------------------------------------------------------


def test_alpha_as_int_rejected(tmp_path):
    def mutate(d):
        d["propagation"]["alpha"] = 1

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_k_as_bool_rejected(tmp_path):
    def mutate(d):
        d["propagation"]["k"] = True

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_align_corners_as_int_rejected(tmp_path):
    def mutate(d):
        d["geometry"]["align_corners"] = 1

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_class_count_as_string_rejected(tmp_path):
    def mutate(d):
        d["dataset"]["classes"] = "171"

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_directed_as_int_rejected(tmp_path):
    def mutate(d):
        d["propagation"]["directed"] = 1

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Alias/whitespace/case-change rejection
# ---------------------------------------------------------------------------


def test_directed_false_rejected(tmp_path):
    def mutate(d):
        d["propagation"]["directed"] = False

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_graph_mode_case_change_rejected(tmp_path):
    def mutate(d):
        d["propagation"]["graph_mode"] = "Directed_TopK"

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_affinity_function_alias_rejected(tmp_path):
    def mutate(d):
        d["propagation"]["affinity_function"] = "relu_cosine_power "

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Native-alignment / clamped-window policy
# ---------------------------------------------------------------------------


def test_tolerance_radius_nonzero_rejected(tmp_path):
    def mutate(d):
        d["clamped_window_policy"]["tolerance_radius_patches"] = 1

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_never_round_or_interpolate_false_rejected(tmp_path):
    def mutate(d):
        d["clamped_window_policy"]["never_round_or_interpolate"] = False

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_canonical_aligned_stride_offset_wrong_value_rejected(tmp_path):
    def mutate(d):
        d["native_alignment"]["canonical_aligned_stride_offset_patches"] = 8

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_stride_patch_size_inconsistent_with_offset_rejected(tmp_path):
    def mutate(d):
        d["geometry"]["stride"] = [112, 112]
        d["native_alignment"]["canonical_aligned_stride_offset_patches"] = 16

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_crop_not_multiple_of_patch_times_grid_rejected(tmp_path):
    def mutate(d):
        d["geometry"]["crop"] = [449, 448]

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Support definition -- topology-only
# ---------------------------------------------------------------------------


def test_weight_by_graph_weight_true_rejected(tmp_path):
    def mutate(d):
        d["support_definition"]["weight_by_graph_weight"] = True

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_reverse_edge_counts_as_support_true_rejected(tmp_path):
    def mutate(d):
        d["support_definition"]["reverse_edge_counts_as_support"] = True

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_source_window_excluded_false_rejected(tmp_path):
    def mutate(d):
        d["support_definition"]["source_window_excluded_as_observer"] = False

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_weighting_not_topology_only_rejected(tmp_path):
    def mutate(d):
        d["support_definition"]["weighting"] = "confidence_weighted"

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Undefined-reason enum / ranking
# ---------------------------------------------------------------------------


def test_undefined_reason_enum_reordered_rejected(tmp_path):
    def mutate(d):
        d["undefined_reason"]["enum"] = list(reversed(d["undefined_reason"]["enum"]))

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_undefined_reason_unknown_value_rejected(tmp_path):
    def mutate(d):
        d["undefined_reason"]["enum"] = ["single_window_image", "nearest_neighbor_fallback"]

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_ranking_applies_edit_true_rejected(tmp_path):
    def mutate(d):
        d["ranking"]["applies_edit"] = True

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_ranking_criteria_reordered_rejected(tmp_path):
    def mutate(d):
        criteria = list(d["ranking"]["criteria_in_order"])
        criteria[0], criteria[1] = criteria[1], criteria[0]
        d["ranking"]["criteria_in_order"] = criteria

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_ranking_criteria_missing_entry_rejected(tmp_path):
    def mutate(d):
        d["ranking"]["criteria_in_order"] = list(d["ranking"]["criteria_in_order"])[:-1]

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# GT diagnostics -- must never influence support/ranking
# ---------------------------------------------------------------------------


def test_gt_influences_support_true_rejected(tmp_path):
    def mutate(d):
        d["gt_diagnostics"]["influences_support_or_ranking"] = True

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_gt_computed_after_support_frozen_false_rejected(tmp_path):
    def mutate(d):
        d["gt_diagnostics"]["computed_after_support_frozen"] = False

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


def test_gt_ignore_index_false_rejected(tmp_path):
    def mutate(d):
        d["gt_diagnostics"]["respects_ignore_index"] = False

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Parent-identity hash consistency
# ---------------------------------------------------------------------------


def test_wrong_stitching_control_identity_sha256_rejected(tmp_path):
    def mutate(d):
        d["parent_identity"]["stitching_control_identity_sha256"] = "0" * 64

    identity = _mutated_load(tmp_path, mutate)
    from src.native_edge_support_identity import _validate_parent_identity

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _validate_parent_identity(ROOT, identity)


def test_wrong_power_evaluation_identity_sha256_rejected(tmp_path):
    def mutate(d):
        d["parent_identity"]["power_evaluation_identity_sha256"] = "1" * 64

    identity = _mutated_load(tmp_path, mutate)
    from src.native_edge_support_identity import _validate_parent_identity

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _validate_parent_identity(ROOT, identity)


def test_wrong_matched_identity_sha256_rejected(tmp_path):
    def mutate(d):
        d["parent_identity"]["matched_identity_sha256"] = "2" * 64

    identity = _mutated_load(tmp_path, mutate)
    from src.native_edge_support_identity import _validate_parent_identity

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _validate_parent_identity(ROOT, identity)


def test_wrong_e3_identity_sha256_rejected(tmp_path):
    def mutate(d):
        d["parent_identity"]["e3_identity_sha256"] = "3" * 64

    identity = _mutated_load(tmp_path, mutate)
    from src.native_edge_support_identity import _validate_parent_identity

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        _validate_parent_identity(ROOT, identity)


def test_wrong_required_ancestor_commit_fails_git_ancestry_check(tmp_path):
    def mutate(d):
        d["identity"]["required_ancestor_commit"] = "f" * 40

    with pytest.raises(NativeEdgeSupportAuditIdentityError):
        identity = _mutated_load(tmp_path, mutate)
        from src.native_edge_support_identity import _check_git_ancestry

        _check_git_ancestry(ROOT, identity["identity"]["required_ancestor_commit"], label="test")


# ---------------------------------------------------------------------------
# Scientific constants come from this identity/validated parents, never
# duplicated Python literals
# ---------------------------------------------------------------------------


def test_propagation_values_are_registered_not_hardcoded():
    identity = load_identity(repo_root=ROOT)
    from src.matched_k11_k12_identity import load_identity as load_matched_identity

    matched = load_matched_identity(ROOT / identity["parent_identity"]["matched_identity_path"], repo_root=ROOT)
    assert identity["propagation"]["alpha"] == matched["propagation"]["alpha"]
    assert identity["propagation"]["steps"] == matched["propagation"]["steps"]
    assert identity["propagation"]["affinity_power"] == matched["graph"]["affinity_power"]
    assert identity["geometry"]["patch_size"] == matched["geometry"]["patch_size"]
    assert identity["geometry"]["grid_size"] == matched["geometry"]["patch_grid"]
    assert identity["geometry"]["crop"] == matched["geometry"]["crop"]
    assert identity["geometry"]["stride"] == matched["geometry"]["stride"]
