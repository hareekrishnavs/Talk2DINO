"""CPU-only coverage for src.native_edge_support_reachability_gate_identity:
valid identity loading, missing/unknown-field rejection, exact-type
rejection, changed decision-mapping/downstream-stage rejection, altered
parent hash rejection, and alias/case/whitespace rejection. Never
initializes CUDA, never loads the model or dataset."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.native_edge_support_reachability_gate_identity import (  # noqa: E402
    NativeEdgeSupportReachabilityGateIdentityError,
    load_identity,
    validate_static_configuration,
)

IDENTITY_PATH = ROOT / "evaluation_identities/e12_native_edge_support_reachability_gate.toml"

pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the reachability-gate identity")


def _raw() -> dict:
    with IDENTITY_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _write_toml_manually(path: Path, data: dict) -> None:
    """Minimal, dependency-free TOML writer (same rationale as
    tests/test_stitching_control_identity.py and
    tests/test_native_edge_support_identity.py)."""
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
    assert identity["identity"]["name"] == "e12-native-edge-support-reachability-gate"
    assert identity["parent_contract"]["required_run_mode"] == "mechanics20"
    assert identity["parent_contract"]["required_image_count"] == 20


def test_static_configuration_validates_parent_chain():
    result = validate_static_configuration(repo_root=ROOT, check_git=True)
    assert result["identity_name"] == "e12-native-edge-support-reachability-gate"
    assert result["native_audit_identity"] == "e12-native-edge-support-audit"
    assert result["decision_outcomes"] == ["REACHABLE", "STRUCTURALLY_UNREACHABLE", "ALIGNMENT_LIMITED", "INCONCLUSIVE"]


# ---------------------------------------------------------------------------
# Missing/unknown fields
# ---------------------------------------------------------------------------


def test_missing_top_level_section_rejected(tmp_path):
    def mutate(d):
        del d["decision_mapping"]

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_unknown_top_level_field_rejected(tmp_path):
    def mutate(d):
        d["extra_section"] = {"x": 1}

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_unknown_field_in_known_section_rejected(tmp_path):
    def mutate(d):
        d["parent_contract"]["extra_field"] = True

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_missing_field_in_known_section_rejected(tmp_path):
    def mutate(d):
        del d["policy"]["cuda_required"]

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Exact-type rejection
# ---------------------------------------------------------------------------


def test_required_image_count_as_float_rejected(tmp_path):
    def mutate(d):
        d["parent_contract"]["required_image_count"] = 20.0

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_required_complete_as_int_rejected(tmp_path):
    def mutate(d):
        d["parent_contract"]["required_complete"] = 1

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_cuda_required_as_int_rejected(tmp_path):
    def mutate(d):
        d["policy"]["cuda_required"] = 0

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Changed decision mapping / downstream stage names / thresholds
# ---------------------------------------------------------------------------


def test_reachable_mapping_changed_rejected(tmp_path):
    def mutate(d):
        d["decision_mapping"]["REACHABLE"] = "SOMETHING_ELSE"

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_alignment_limited_mapping_changed_rejected(tmp_path):
    def mutate(d):
        d["decision_mapping"]["ALIGNMENT_LIMITED"] = "PROCEED_TO_ELIGIBILITY_MATCHED_PRUNING"

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_structural_stage_name_changed_rejected(tmp_path):
    def mutate(d):
        stages = list(d["roadmap"]["structural_stages_gated_on_reachable"])
        stages[0] = "feat: add something else entirely"
        d["roadmap"]["structural_stages_gated_on_reachable"] = stages

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_structural_stages_reordered_rejected(tmp_path):
    def mutate(d):
        d["roadmap"]["structural_stages_gated_on_reachable"] = list(reversed(d["roadmap"]["structural_stages_gated_on_reachable"]))

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_next_stage_if_reachable_changed_rejected(tmp_path):
    def mutate(d):
        d["roadmap"]["next_stage_if_reachable"] = "eval: add COCO-Object protocol confirmation"

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_next_stage_if_not_reachable_changed_rejected(tmp_path):
    def mutate(d):
        d["roadmap"]["next_stage_if_not_reachable"] = "feat: add eligibility-matched one-edge pruning variants"

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_no_threshold_duplication_rejected_if_relaxation_flag_flipped(tmp_path):
    def mutate(d):
        d["policy"]["no_threshold_relaxation"] = False

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_no_approximate_alignment_flag_flipped_rejected(tmp_path):
    def mutate(d):
        d["policy"]["no_approximate_alignment_introduced"] = False

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_no_pruning_flag_flipped_rejected(tmp_path):
    def mutate(d):
        d["policy"]["no_pruning_implemented"] = False

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Unsupported alignment mode / cause decomposition policy
# ---------------------------------------------------------------------------


def test_infer_from_pair_counts_true_rejected(tmp_path):
    def mutate(d):
        d["cause_decomposition"]["infer_from_pair_counts"] = True

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_fabricate_unavailable_subcategories_true_rejected(tmp_path):
    def mutate(d):
        d["cause_decomposition"]["fabricate_unavailable_subcategories"] = True

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_rerun_audit_to_fill_gaps_true_rejected(tmp_path):
    def mutate(d):
        d["cause_decomposition"]["rerun_audit_to_fill_gaps"] = True

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_desired_categories_reordered_rejected(tmp_path):
    def mutate(d):
        d["cause_decomposition"]["desired_categories"] = list(reversed(d["cause_decomposition"]["desired_categories"]))

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Altered parent hash
# ---------------------------------------------------------------------------


def test_wrong_native_audit_identity_sha256_rejected(tmp_path):
    def mutate(d):
        d["parent_identity"]["native_audit_identity_sha256"] = "0" * 64

    identity = _mutated_load(tmp_path, mutate)
    from src.native_edge_support_reachability_gate_identity import _validate_parent_identity

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _validate_parent_identity(ROOT, identity)


def test_malformed_native_audit_sha256_rejected(tmp_path):
    def mutate(d):
        d["parent_identity"]["native_audit_identity_sha256"] = "not-a-hash"

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_wrong_required_ancestor_commit_fails_git_ancestry_check(tmp_path):
    def mutate(d):
        d["identity"]["required_ancestor_commit"] = "f" * 40

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        identity = _mutated_load(tmp_path, mutate)
        from src.native_edge_support_reachability_gate_identity import _check_git_ancestry

        _check_git_ancestry(ROOT, identity["identity"]["required_ancestor_commit"], label="test")


# ---------------------------------------------------------------------------
# Alias/case/whitespace changes
# ---------------------------------------------------------------------------


def test_run_mode_case_change_rejected(tmp_path):
    def mutate(d):
        d["parent_contract"]["required_run_mode"] = "Mechanics20"

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_decision_outcome_trailing_space_rejected(tmp_path):
    def mutate(d):
        d["decision_vocabulary"]["outcomes"] = ["REACHABLE ", "STRUCTURALLY_UNREACHABLE", "ALIGNMENT_LIMITED", "INCONCLUSIVE"]

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


def test_required_final_true_rejected(tmp_path):
    # This audit's own schema never produces final=true for mechanics20;
    # requiring true here would reject every valid parent result.
    def mutate(d):
        d["parent_contract"]["required_final"] = True

    with pytest.raises(NativeEdgeSupportReachabilityGateIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Scientific/authority values come from identities, never duplicated
# ---------------------------------------------------------------------------


def test_decision_mapping_matches_python_constant():
    from src.native_edge_support_reachability_gate_identity import SUPPORTED_DECISION_MAPPING

    identity = load_identity(repo_root=ROOT)
    assert dict(identity["decision_mapping"]) == SUPPORTED_DECISION_MAPPING


def test_native_audit_identity_sha256_matches_real_file():
    identity = load_identity(repo_root=ROOT)
    import hashlib

    real_bytes = (ROOT / "evaluation_identities/e12_native_edge_support_audit.toml").read_bytes()
    assert hashlib.sha256(real_bytes).hexdigest() == identity["parent_identity"]["native_audit_identity_sha256"]
