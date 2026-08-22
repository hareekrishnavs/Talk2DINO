"""Integration tests for the k11/k12 stability gate identity, result and
checkpoint schema validation, and CLI failure contract. CPU-only; canonical
values are always obtained through ``load_identity()``, never duplicated."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

from src.k11_k12_stability_gate_identity import (
    K11K12StabilityGateError,
    load_identity,
    validate_static_configuration,
)
from src.k11_k12_stability_report import (
    CONDITION_DIAGNOSTICS_KEYS,
    DETERMINISM_DIAGNOSTICS_KEYS,
    GRAPH_DIAGNOSTICS_KEYS,
    MATCHED_DELTA_DIAGNOSTICS_KEYS,
    PHASE_RUNTIME_KEYS,
    PROVENANCE_KEYS,
    RECURRENCE_DIAGNOSTICS_KEYS,
    REFERENCE_DIAGNOSTICS_KEYS,
    RESUMABILITY_KEYS,
    SNAPSHOT_COMPARISON_KEYS,
    classify_regime,
    resume_window_start,
    verify_checkpoint_record,
    verify_record,
)

ROOT = Path(__file__).parents[1]
IDENTITY_PATH = ROOT / "evaluation_identities/e12_k11_k12_stability_gate.toml"
CLI = ROOT / "verify_k11_k12_stability.py"
PY = "/scratch/haree/venv/talk2dino-a100/bin/python"


def _identity():
    return load_identity(repo_root=ROOT)


def _identity_sha256() -> str:
    return hashlib.sha256(IDENTITY_PATH.read_bytes()).hexdigest()


def _snapshot_comparison(value: float = 0.0) -> dict:
    return {
        "max_absolute_error": Decimal(f"{value:.12f}"),
        "mean_absolute_error": Decimal(f"{value:.12f}"),
        "relative_frobenius_error": Decimal(f"{value:.12f}"),
        "argmax_disagreement_count": 0,
        "argmax_disagreement_rate": Decimal("0.000000000000"),
    }


def build_valid_record(identity, *, d320_relative_norm=0.005, t320_t640_relative_change=0.0):
    classification = classify_regime(
        numerical_validity_passed=True,
        d320_relative_norm=d320_relative_norm,
        t320_t640_relative_change=t320_t640_relative_change,
        thresholds=identity["gate_thresholds"],
    )
    return {
        "schema": "talk2dino-k11-k12-stability-gate-v1",
        "identity": identity["identity"]["name"],
        "identity_sha256": _identity_sha256(),
        "matched_identity_sha256": identity["parent_identity"]["matched_identity_sha256"],
        "git_commit": "a" * 40,
        "checkpoint_sha256": None,
        "complete": True,
        "final": True,
        "device": "cpu",
        "gpu_model": "synthetic",
        "torch_version": "2.0",
        "cuda_version": "none",
        "manifest_digest": "b" * 64,
        "windows_expected": identity["sample_selection"]["canonical_window_count"],
        "windows_processed": identity["sample_selection"]["canonical_window_count"],
        "graph_diagnostics": {
            "prefix_mismatch_count": 0,
            "fallback_row_count_k12": 3,
            "fallback_row_count_k11": 3,
            "fallback_row_mismatch_count": 0,
            "tie_row_count": 5,
            "row_sum_max_error_k11": Decimal("0.000000100000"),
            "row_sum_max_error_k12": Decimal("0.000000100000"),
            "negative_weight_count_k11": 0,
            "negative_weight_count_k12": 0,
            "non_fallback_self_edge_count_k11": 0,
            "non_fallback_self_edge_count_k12": 0,
            "directed_asymmetry_fraction_k12": Decimal("0.015000000000"),
        },
        "recurrence_diagnostics": {
            "steps_completed_k11": 640,
            "steps_completed_k12": 640,
            "snapshot_steps": [160, 320, 640],
            "early_termination": False,
            "solver_fallback_used": False,
            "cgls_call_count": 0,
            "dense_solve_call_count_in_production_path": 0,
            "t160_t320_k11": _snapshot_comparison(),
            "t160_t320_k12": _snapshot_comparison(),
            "t320_t640_k11": _snapshot_comparison(t320_t640_relative_change),
            "t320_t640_k12": _snapshot_comparison(t320_t640_relative_change),
            "t320_t640_relative_frobenius_change_mean": Decimal(f"{t320_t640_relative_change:.12f}"),
        },
        "reference_diagnostics": {
            "fp32_fp64_t160_k11": _snapshot_comparison(0.0001),
            "fp32_fp64_t160_k12": _snapshot_comparison(0.0001),
            "fp32_fp64_t320_k11": _snapshot_comparison(0.0001),
            "fp32_fp64_t320_k12": _snapshot_comparison(0.0001),
            "fp32_fp64_t640_k11": _snapshot_comparison(0.0001),
            "fp32_fp64_t640_k12": _snapshot_comparison(0.0001),
            "dense_equilibrium_t320_k11": _snapshot_comparison(0.01),
            "dense_equilibrium_t320_k12": _snapshot_comparison(0.01),
            "dense_equilibrium_t640_k11": _snapshot_comparison(0.001),
            "dense_equilibrium_t640_k12": _snapshot_comparison(0.001),
            "dense_residual_relative_k11": Decimal("0.000000000001"),
            "dense_residual_relative_k12": Decimal("0.000000000001"),
            "dense_backward_error_k11": Decimal("0.000000000001"),
            "dense_backward_error_k12": Decimal("0.000000000001"),
        },
        "condition_diagnostics": {
            "window_count": identity["reference_windows"]["condition_number_window_count"],
            "sigma_min_min": Decimal("0.010000000000"),
            "sigma_min_max": Decimal("0.050000000000"),
            "kappa_2_min": Decimal("20.000000000000"),
            "kappa_2_max": Decimal("100.000000000000"),
            "kappa_2_mean": Decimal("50.000000000000"),
            "departure_from_normality_illustrative_mean": Decimal("0.300000000000"),
        },
        "matched_delta_diagnostics": {
            "d160_norm_mean": Decimal("0.100000000000"),
            "d320_norm_mean": Decimal("0.120000000000"),
            "d640_norm_mean": Decimal("0.125000000000"),
            "d320_relative_norm_mean": Decimal(f"{d320_relative_norm:.12f}"),
            "d160_d320_stability_error_mean": Decimal("0.010000000000"),
            "d320_d640_stability_error_mean": Decimal("0.005000000000"),
            "argmax_disagreement_rate_k11_k12_t320_mean": Decimal("0.002000000000"),
            "label_sensitivity_argmax_disagreement_rate_mean": Decimal("0.002000000000"),
        },
        "determinism_diagnostics": {
            "replay_count": identity["tolerances"]["determinism_replay_count"],
            "manifest_matches": True,
            "graph_indices_match": True,
            "graph_weights_match": True,
            "p160_match": True,
            "p320_match": True,
            "p640_match": True,
            "argmax_digest_match": True,
            "drift_detected": False,
            "drift_description": "",
        },
        "phase_runtime_seconds": {name: Decimal("1.000000000000") for name in PHASE_RUNTIME_KEYS},
        "peak_gpu_memory_bytes": 1_000_000,
        "gate_classification": classification,
        "numerical_validity_passed": True,
        "failure_reason": None,
        "resumability": {"resumed_from_checkpoint": False, "resumed_window_count": 0, "checkpoint_path": None},
        "provenance": {
            "source_git_branch": "e12-connectivity-analysis",
            "source_git_dirty": False,
            "elapsed_seconds_total": Decimal("120.000000000000"),
        },
    }


def build_valid_checkpoint(identity, *, window_count=None, complete=False):
    total = identity["sample_selection"]["canonical_window_count"]
    count = total if complete else (window_count if window_count is not None else 37)
    return {
        "schema": "talk2dino-k11-k12-stability-checkpoint-v1",
        "identity": identity["identity"]["name"],
        "identity_sha256": _identity_sha256(),
        "manifest_digest": "b" * 64,
        "windows_expected": total,
        "complete": complete,
        "windows": [
            {"window_index": i, "image_id": f"img{i:012d}", "sha256": "c" * 64} for i in range(count)
        ],
        "created_at_utc": "2026-08-21T00:00:00Z",
        "updated_at_utc": "2026-08-21T00:05:00Z",
    }


# ---------------------------------------------------------------------------
# Identity loading and preflight
# ---------------------------------------------------------------------------


def test_identity_loads():
    identity = _identity()
    assert identity["identity"]["name"] == "e12-k11-k12-stability-gate"
    assert identity["snapshots"]["steps"] == [160, 320, 640]


def test_preflight_passes_against_the_real_repository():
    result = validate_static_configuration(repo_root=ROOT, check_git=True)
    assert result["identity_name"] == "e12-k11-k12-stability-gate"
    assert result["matched_identity"] == "e12-matched-k11-k12-t320"


def test_cli_preflight_passes():
    result = subprocess.run([PY, str(CLI), "preflight", "--repo-root", str(ROOT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "K11/K12 STABILITY GATE PREFLIGHT PASS" in result.stdout


def test_cli_help():
    result = subprocess.run([PY, str(CLI), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "preflight" in result.stdout
    assert "verify-result" in result.stdout
    assert "verify-checkpoint" in result.stdout


def test_gate_alpha_matches_matched_identity_alpha():
    identity = _identity()
    from src.matched_k11_k12_identity import load_identity as load_matched

    matched = load_matched(ROOT / identity["parent_identity"]["matched_identity_path"], repo_root=ROOT)
    assert identity["propagation"]["alpha"] == matched["propagation"]["alpha"]


# ---------------------------------------------------------------------------
# Gate-count closure: canonical_window_count and the reference-window counts
# are the implemented bounded harness protocol, not arbitrary positive
# integers -- every registered count must equal its SUPPORTED_* constant
# (Finding 2). Canonical values are always obtained through load_identity()
# / the SUPPORTED_* constants, never duplicated as a bare literal here.
# ---------------------------------------------------------------------------


def _mutated_identity_path(tmp_path, old: str, new: str) -> Path:
    text = IDENTITY_PATH.read_text()
    assert old in text, f"anchor not found in real identity TOML: {old!r}"
    mutated = tmp_path / "mutated_identity.toml"
    mutated.write_text(text.replace(old, new, 1))
    return mutated


def test_real_canonical_window_count_matches_supported_protocol():
    from src.k11_k12_stability_gate_identity import SUPPORTED_CANONICAL_WINDOW_COUNT

    identity = _identity()
    assert identity["sample_selection"]["canonical_window_count"] == SUPPORTED_CANONICAL_WINDOW_COUNT


@pytest.mark.parametrize(
    "mutated_toml_value", ["50", "99", "101", "0", "-1", "100.0", "true"]
)
def test_canonical_window_count_closed_to_supported_protocol(tmp_path, mutated_toml_value):
    path = _mutated_identity_path(
        tmp_path, "canonical_window_count = 100", f"canonical_window_count = {mutated_toml_value}"
    )
    with pytest.raises(K11K12StabilityGateError):
        load_identity(path, repo_root=ROOT)


def test_canonical_window_count_as_string_rejected(tmp_path):
    path = _mutated_identity_path(tmp_path, "canonical_window_count = 100", 'canonical_window_count = "100"')
    with pytest.raises(K11K12StabilityGateError):
        load_identity(path, repo_root=ROOT)


def test_canonical_window_count_missing_rejected(tmp_path):
    text = IDENTITY_PATH.read_text()
    assert "canonical_window_count = 100\n" in text
    path = tmp_path / "missing_count.toml"
    path.write_text(text.replace("canonical_window_count = 100\n", "", 1))
    with pytest.raises(K11K12StabilityGateError):
        load_identity(path, repo_root=ROOT)


def test_canonical_window_count_error_identifies_field_expected_and_observed(tmp_path):
    path = _mutated_identity_path(tmp_path, "canonical_window_count = 100", "canonical_window_count = 50")
    with pytest.raises(K11K12StabilityGateError, match=r"canonical_window_count.*100.*50"):
        load_identity(path, repo_root=ROOT)


def test_cli_preflight_wrong_window_count_exits_2_without_traceback(tmp_path):
    path = _mutated_identity_path(tmp_path, "canonical_window_count = 100", "canonical_window_count = 50")
    result = subprocess.run(
        [PY, str(CLI), "--identity", str(path), "preflight", "--repo-root", str(ROOT)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "toml_field,supported_constant_name,mutated_value",
    [
        ("fp64_finite_step_reference_window_count = 20", "SUPPORTED_FP64_FINITE_STEP_REFERENCE_WINDOW_COUNT", 10),
        ("dense_equilibrium_reference_window_count = 20", "SUPPORTED_DENSE_EQUILIBRIUM_REFERENCE_WINDOW_COUNT", 10),
        ("condition_number_window_count = 10", "SUPPORTED_CONDITION_NUMBER_WINDOW_COUNT", 5),
    ],
)
def test_reference_window_counts_closed_to_supported_protocol(tmp_path, toml_field, supported_constant_name, mutated_value):
    import src.k11_k12_stability_gate_identity as identity_module

    supported = getattr(identity_module, supported_constant_name)
    field_name, canonical_value = toml_field.split(" = ")
    assert int(canonical_value) == supported
    path = _mutated_identity_path(tmp_path, toml_field, f"{field_name} = {mutated_value}")
    with pytest.raises(K11K12StabilityGateError):
        load_identity(path, repo_root=ROOT)


# ---------------------------------------------------------------------------
# Threshold relation: the near-noise/clear-signal gap must exist (Finding 1)
# ---------------------------------------------------------------------------


def test_equal_gate_thresholds_rejected(tmp_path):
    path = _mutated_identity_path(
        tmp_path,
        "clear_signal_relative_delta_norm_min = 0.05",
        "clear_signal_relative_delta_norm_min = 0.01",
    )
    with pytest.raises(K11K12StabilityGateError, match="strictly less than"):
        load_identity(path, repo_root=ROOT)


def test_reversed_gate_thresholds_rejected(tmp_path):
    path = _mutated_identity_path(
        tmp_path,
        "clear_signal_relative_delta_norm_min = 0.05",
        "clear_signal_relative_delta_norm_min = 0.005",
    )
    with pytest.raises(K11K12StabilityGateError, match="strictly less than"):
        load_identity(path, repo_root=ROOT)


# ---------------------------------------------------------------------------
# Classification boundaries
# ---------------------------------------------------------------------------


def test_classify_regime_invalid_when_numerical_validity_fails():
    identity = _identity()
    classification = classify_regime(
        numerical_validity_passed=False, d320_relative_norm=0.5,
        t320_t640_relative_change=0.5, thresholds=identity["gate_thresholds"],
    )
    assert classification == "INVALID"


def test_classify_regime_truncation_sensitive_takes_priority():
    identity = _identity()
    thresholds = identity["gate_thresholds"]
    classification = classify_regime(
        numerical_validity_passed=True,
        d320_relative_norm=thresholds["clear_signal_relative_delta_norm_min"] + 1.0,
        t320_t640_relative_change=thresholds["truncation_sensitive_relative_change_min"],
        thresholds=thresholds,
    )
    assert classification == "TRUNCATION_SENSITIVE"


def test_classify_regime_near_noise_at_or_below_threshold():
    identity = _identity()
    thresholds = identity["gate_thresholds"]
    classification = classify_regime(
        numerical_validity_passed=True,
        d320_relative_norm=thresholds["effect_near_noise_relative_delta_norm_max"],
        t320_t640_relative_change=0.0,
        thresholds=thresholds,
    )
    assert classification == "NUMERICALLY_STABLE_BUT_EFFECT_NEAR_NOISE"


def test_classify_regime_clear_signal_at_or_above_threshold():
    identity = _identity()
    thresholds = identity["gate_thresholds"]
    classification = classify_regime(
        numerical_validity_passed=True,
        d320_relative_norm=thresholds["clear_signal_relative_delta_norm_min"],
        t320_t640_relative_change=0.0,
        thresholds=thresholds,
    )
    assert classification == "CLEAR_MATCHED_SIGNAL"


def test_classify_regime_between_thresholds_is_honestly_inconclusive():
    # A value strictly between the near-noise and clear-signal thresholds
    # is neither near noise nor clear signal -- it must not be mislabeled
    # as either (this was a confirmed defect: the classifier used to fall
    # through to NUMERICALLY_STABLE_BUT_EFFECT_NEAR_NOISE for this case).
    identity = _identity()
    thresholds = identity["gate_thresholds"]
    midpoint = (
        thresholds["effect_near_noise_relative_delta_norm_max"]
        + thresholds["clear_signal_relative_delta_norm_min"]
    ) / 2
    classification = classify_regime(
        numerical_validity_passed=True, d320_relative_norm=midpoint,
        t320_t640_relative_change=0.0, thresholds=thresholds,
    )
    assert classification == "NUMERICALLY_STABLE_BUT_REGIME_INCONCLUSIVE"


def test_classify_regime_never_claims_efficacy():
    # Structural guarantee: the function BODY's return values are drawn
    # only from the closed classification vocabulary -- never "k11_better"
    # or "k11_worse" or any mIoU-flavored label. The docstring's own
    # negation prose ("does not claim k11 helps or hurts mIoU") is
    # deliberately excluded from this scan.
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(classify_regime))
    tree = ast.parse(source)
    function_node = tree.body[0]
    body_without_docstring = ast.Module(body=function_node.body[1:], type_ignores=[])
    body_source = ast.unparse(body_without_docstring)
    for forbidden in ("better", "worse", "helps", "hurts", "miou"):
        assert forbidden not in body_source.lower()


@pytest.mark.parametrize(
    "d320_relative_norm,t320_t640_relative_change",
    [
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (float("-inf"), 0.0),
        (0.005, float("nan")),
    ],
)
def test_classify_regime_non_finite_diagnostics_yield_invalid(d320_relative_norm, t320_t640_relative_change):
    identity = _identity()
    classification = classify_regime(
        numerical_validity_passed=True,
        d320_relative_norm=d320_relative_norm,
        t320_t640_relative_change=t320_t640_relative_change,
        thresholds=identity["gate_thresholds"],
    )
    assert classification == "INVALID"


def test_classify_regime_just_above_near_noise_threshold_is_inconclusive():
    identity = _identity()
    thresholds = identity["gate_thresholds"]
    classification = classify_regime(
        numerical_validity_passed=True,
        d320_relative_norm=thresholds["effect_near_noise_relative_delta_norm_max"] + 1e-9,
        t320_t640_relative_change=0.0,
        thresholds=thresholds,
    )
    assert classification == "NUMERICALLY_STABLE_BUT_REGIME_INCONCLUSIVE"


def test_classify_regime_just_below_clear_signal_threshold_is_inconclusive():
    identity = _identity()
    thresholds = identity["gate_thresholds"]
    classification = classify_regime(
        numerical_validity_passed=True,
        d320_relative_norm=thresholds["clear_signal_relative_delta_norm_min"] - 1e-9,
        t320_t640_relative_change=0.0,
        thresholds=thresholds,
    )
    assert classification == "NUMERICALLY_STABLE_BUT_REGIME_INCONCLUSIVE"


def test_classify_regime_truncation_sensitive_still_takes_priority_over_inconclusive():
    identity = _identity()
    thresholds = identity["gate_thresholds"]
    midpoint = (
        thresholds["effect_near_noise_relative_delta_norm_max"]
        + thresholds["clear_signal_relative_delta_norm_min"]
    ) / 2
    classification = classify_regime(
        numerical_validity_passed=True,
        d320_relative_norm=midpoint,
        t320_t640_relative_change=thresholds["truncation_sensitive_relative_change_min"],
        thresholds=thresholds,
    )
    assert classification == "TRUNCATION_SENSITIVE"


def test_classify_regime_rejects_non_finite_threshold():
    identity = _identity()
    thresholds = dict(identity["gate_thresholds"])
    thresholds["clear_signal_relative_delta_norm_min"] = float("nan")
    with pytest.raises(K11K12StabilityGateError):
        classify_regime(
            numerical_validity_passed=True, d320_relative_norm=0.005,
            t320_t640_relative_change=0.0, thresholds=thresholds,
        )


# ---------------------------------------------------------------------------
# Result record schema: valid records
# ---------------------------------------------------------------------------


def test_valid_record_passes():
    identity = _identity()
    record = build_valid_record(identity)
    message = verify_record(record, identity, identity_sha256=_identity_sha256())
    assert "RESULT PASS" in message


def test_valid_record_clear_matched_signal_passes():
    identity = _identity()
    thresholds = identity["gate_thresholds"]
    record = build_valid_record(identity, d320_relative_norm=thresholds["clear_signal_relative_delta_norm_min"] + 0.05)
    verify_record(record, identity, identity_sha256=_identity_sha256())


def test_valid_record_near_noise_passes():
    identity = _identity()
    thresholds = identity["gate_thresholds"]
    record = build_valid_record(identity, d320_relative_norm=thresholds["effect_near_noise_relative_delta_norm_max"] / 2)
    verify_record(record, identity, identity_sha256=_identity_sha256())


def test_valid_record_inconclusive_passes():
    identity = _identity()
    thresholds = identity["gate_thresholds"]
    midpoint = (
        thresholds["effect_near_noise_relative_delta_norm_max"]
        + thresholds["clear_signal_relative_delta_norm_min"]
    ) / 2
    record = build_valid_record(identity, d320_relative_norm=midpoint)
    assert record["gate_classification"] == "NUMERICALLY_STABLE_BUT_REGIME_INCONCLUSIVE"
    message = verify_record(record, identity, identity_sha256=_identity_sha256())
    assert "RESULT PASS" in message


def test_record_does_not_mutate_the_input_mapping():
    identity = _identity()
    record = build_valid_record(identity)
    before = copy.deepcopy(record)
    verify_record(record, identity, identity_sha256=_identity_sha256())
    assert record == before


# ---------------------------------------------------------------------------
# Result record schema: rejections
# ---------------------------------------------------------------------------


def test_prefix_mismatch_nonzero_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["graph_diagnostics"]["prefix_mismatch_count"] = 1
    with pytest.raises(K11K12StabilityGateError, match="prefix_mismatch_count"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_fallback_row_mismatch_nonzero_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["graph_diagnostics"]["fallback_row_mismatch_count"] = 1
    with pytest.raises(K11K12StabilityGateError, match="fallback_row_mismatch_count"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_early_termination_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["recurrence_diagnostics"]["early_termination"] = True
    with pytest.raises(K11K12StabilityGateError, match="early_termination"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_solver_fallback_used_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["recurrence_diagnostics"]["solver_fallback_used"] = True
    with pytest.raises(K11K12StabilityGateError, match="solver_fallback_used"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_cgls_call_count_nonzero_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["recurrence_diagnostics"]["cgls_call_count"] = 1
    with pytest.raises(K11K12StabilityGateError, match="cgls_call_count"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_dense_solve_in_production_path_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["recurrence_diagnostics"]["dense_solve_call_count_in_production_path"] = 1
    with pytest.raises(K11K12StabilityGateError, match="dense_solve_call_count_in_production_path"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_wrong_step_count_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["recurrence_diagnostics"]["steps_completed_k11"] = 320
    with pytest.raises(K11K12StabilityGateError, match="steps_completed_k11"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_wrong_snapshot_steps_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["recurrence_diagnostics"]["snapshot_steps"] = [160, 320]
    with pytest.raises(K11K12StabilityGateError, match="snapshot_steps"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_fp32_fp64_tolerance_exceeded_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    tolerance = identity["tolerances"]["fp32_fp64_relative_frobenius_error_max"]
    record["reference_diagnostics"]["fp32_fp64_t320_k12"]["relative_frobenius_error"] = Decimal(
        f"{tolerance * 10:.12f}"
    )
    with pytest.raises(K11K12StabilityGateError, match="exceeds the registered tolerance"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_dense_residual_tolerance_exceeded_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    tolerance = identity["tolerances"]["dense_equilibrium_residual_relative_max"]
    record["reference_diagnostics"]["dense_residual_relative_k12"] = Decimal(f"{tolerance * 10:.12f}")
    with pytest.raises(K11K12StabilityGateError, match="exceeds the registered tolerance"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_gate_classification_inconsistent_with_thresholds_rejected():
    identity = _identity()
    record = build_valid_record(identity, d320_relative_norm=0.005)
    record["gate_classification"] = "TRUNCATION_SENSITIVE"
    with pytest.raises(K11K12StabilityGateError, match="inconsistent with TOML-defined thresholds"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_unknown_gate_classification_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["gate_classification"] = "K11_IS_BETTER"
    with pytest.raises(K11K12StabilityGateError, match="gate_classification"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_incomplete_result_marked_complete_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["windows_processed"] = identity["sample_selection"]["canonical_window_count"] - 1
    with pytest.raises(K11K12StabilityGateError, match="windows_processed"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_wrong_identity_sha256_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["identity_sha256"] = "0" * 64
    with pytest.raises(K11K12StabilityGateError, match="identity_sha256"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_wrong_matched_identity_sha256_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["matched_identity_sha256"] = "0" * 64
    with pytest.raises(K11K12StabilityGateError, match="matched_identity_sha256"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_non_finite_value_rejected_at_parse_time(tmp_path):
    path = tmp_path / "nonfinite.json"
    path.write_text('{"schema": "x", "value": NaN}')
    with pytest.raises(K11K12StabilityGateError, match="non-finite"):
        from src.k11_k12_stability_gate_identity import parse_structured_document
        parse_structured_document(path, label="structured result")


def test_unknown_top_level_key_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["bogus_extra_field"] = True
    with pytest.raises(K11K12StabilityGateError, match="unexpected schema"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_negative_row_sum_error_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["graph_diagnostics"]["row_sum_max_error_k11"] = Decimal("-0.000001000000")
    with pytest.raises(K11K12StabilityGateError):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_argmax_disagreement_rate_out_of_range_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["matched_delta_diagnostics"]["argmax_disagreement_rate_k11_k12_t320_mean"] = Decimal("1.500000000000")
    with pytest.raises(K11K12StabilityGateError, match=r"\[0, 1\]"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


# ---------------------------------------------------------------------------
# Checkpoint schema and resume safety
# ---------------------------------------------------------------------------


def test_valid_resumable_checkpoint_passes():
    identity = _identity()
    checkpoint = build_valid_checkpoint(identity, window_count=37)
    message = verify_checkpoint_record(checkpoint, identity, identity_sha256=_identity_sha256())
    assert "resumable at window 37" in message


def test_valid_complete_checkpoint_passes():
    identity = _identity()
    checkpoint = build_valid_checkpoint(identity, complete=True)
    message = verify_checkpoint_record(checkpoint, identity, identity_sha256=_identity_sha256())
    assert "status=complete" in message


def test_complete_checkpoint_with_partial_windows_rejected():
    identity = _identity()
    # build_valid_checkpoint always fills every window when complete=True,
    # so this specific "complete but partial" malformation is constructed
    # directly here rather than through the fixture helper.
    checkpoint = build_valid_checkpoint(identity, window_count=5, complete=False)
    checkpoint["complete"] = True
    with pytest.raises(K11K12StabilityGateError, match="exactly windows_expected"):
        verify_checkpoint_record(checkpoint, identity, identity_sha256=_identity_sha256())


def test_duplicate_window_index_rejected():
    identity = _identity()
    checkpoint = build_valid_checkpoint(identity, window_count=5)
    checkpoint["windows"][3]["window_index"] = 2
    with pytest.raises(K11K12StabilityGateError, match="duplicate window_index"):
        verify_checkpoint_record(checkpoint, identity, identity_sha256=_identity_sha256())


def test_non_contiguous_window_index_rejected():
    identity = _identity()
    checkpoint = build_valid_checkpoint(identity, window_count=5)
    checkpoint["windows"][4]["window_index"] = 9
    with pytest.raises(K11K12StabilityGateError, match="contiguous row-major order"):
        verify_checkpoint_record(checkpoint, identity, identity_sha256=_identity_sha256())


def test_out_of_order_window_index_rejected():
    identity = _identity()
    checkpoint = build_valid_checkpoint(identity, window_count=5)
    checkpoint["windows"] = list(reversed(checkpoint["windows"]))
    with pytest.raises(K11K12StabilityGateError, match="contiguous row-major order"):
        verify_checkpoint_record(checkpoint, identity, identity_sha256=_identity_sha256())


def test_malformed_checkpoint_unknown_key_rejected():
    identity = _identity()
    checkpoint = build_valid_checkpoint(identity, window_count=5)
    checkpoint["bogus"] = True
    with pytest.raises(K11K12StabilityGateError, match="unexpected schema"):
        verify_checkpoint_record(checkpoint, identity, identity_sha256=_identity_sha256())


def test_malformed_checkpoint_window_entry_unknown_key_rejected():
    identity = _identity()
    checkpoint = build_valid_checkpoint(identity, window_count=5)
    checkpoint["windows"][0]["bogus"] = 1
    with pytest.raises(K11K12StabilityGateError, match="unexpected schema"):
        verify_checkpoint_record(checkpoint, identity, identity_sha256=_identity_sha256())


def test_windows_exceeding_expected_rejected():
    identity = _identity()
    checkpoint = build_valid_checkpoint(identity, window_count=identity["sample_selection"]["canonical_window_count"] + 5)
    with pytest.raises(K11K12StabilityGateError, match="exceeds windows_expected"):
        verify_checkpoint_record(checkpoint, identity, identity_sha256=_identity_sha256())


def test_resume_window_start_returns_next_index():
    identity = _identity()
    checkpoint = build_valid_checkpoint(identity, window_count=37)
    assert resume_window_start(checkpoint) == 37


def test_completed_checkpoint_cannot_be_resumed():
    identity = _identity()
    checkpoint = build_valid_checkpoint(identity, complete=True)
    with pytest.raises(K11K12StabilityGateError, match="already complete"):
        resume_window_start(checkpoint)


def test_checkpoint_does_not_mutate_the_input_mapping():
    identity = _identity()
    checkpoint = build_valid_checkpoint(identity, window_count=10)
    before = copy.deepcopy(checkpoint)
    verify_checkpoint_record(checkpoint, identity, identity_sha256=_identity_sha256())
    assert checkpoint == before


# ---------------------------------------------------------------------------
# Deterministic serialization / atomic checkpoint write
# ---------------------------------------------------------------------------


def test_write_checkpoint_atomically_produces_sorted_deterministic_json(tmp_path):
    from src.k11_k12_stability_report import write_checkpoint_atomically

    identity = _identity()
    checkpoint = build_valid_checkpoint(identity, window_count=3)
    # Decimal isn't JSON-serializable directly here; use plain floats for
    # this structural/determinism check (schema validation already covers
    # Decimal-precision elsewhere).
    path = tmp_path / "checkpoint.json"
    write_checkpoint_atomically(path, checkpoint)
    text = path.read_text()
    assert text.endswith("\n")
    parsed_keys = list(json.loads(text).keys())
    assert parsed_keys == sorted(parsed_keys)
    # Re-writing the identical record produces byte-identical output.
    write_checkpoint_atomically(path, checkpoint)
    assert path.read_text() == text


def test_write_checkpoint_atomically_leaves_no_temp_file_on_success(tmp_path):
    from src.k11_k12_stability_report import write_checkpoint_atomically

    identity = _identity()
    checkpoint = build_valid_checkpoint(identity, window_count=3)
    path = tmp_path / "checkpoint.json"
    write_checkpoint_atomically(path, checkpoint)
    assert not path.with_suffix(path.suffix + ".tmp").exists()


# ---------------------------------------------------------------------------
# CLI failure contract
# ---------------------------------------------------------------------------


def test_cli_missing_result_file_clean_failure(tmp_path):
    missing = tmp_path / "does-not-exist.json"
    result = subprocess.run(
        [PY, str(CLI), "verify-result", "--result", str(missing), "--repo-root", str(ROOT)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert result.stderr.startswith("K11/K12 STABILITY GATE VERIFICATION FAIL:")
    assert "Traceback" not in result.stderr


def test_cli_malformed_result_clean_failure(tmp_path):
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not valid json")
    result = subprocess.run(
        [PY, str(CLI), "verify-result", "--result", str(malformed), "--repo-root", str(ROOT)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_cli_non_finite_numeric_result_clean_failure(tmp_path):
    malformed = tmp_path / "nonfinite_result.json"
    malformed.write_text('{"schema": "talk2dino-k11-k12-stability-gate-v1", "d320_relative_norm_mean": NaN}')
    result = subprocess.run(
        [PY, str(CLI), "verify-result", "--result", str(malformed), "--repo-root", str(ROOT)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_cli_missing_checkpoint_file_clean_failure(tmp_path):
    missing = tmp_path / "does-not-exist.json"
    result = subprocess.run(
        [PY, str(CLI), "verify-checkpoint", "--checkpoint", str(missing), "--repo-root", str(ROOT)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_cli_missing_identity_clean_failure(tmp_path):
    missing = tmp_path / "does-not-exist.toml"
    result = subprocess.run(
        [PY, str(CLI), "--identity", str(missing), "preflight", "--repo-root", str(ROOT)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def _decimal_to_float(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: _decimal_to_float(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decimal_to_float(item) for item in value]
    return value


def test_cli_valid_result_end_to_end(tmp_path):
    identity = _identity()
    record = build_valid_record(identity)
    # Decimal fields carry full precision for the in-memory schema checks;
    # for a real JSON file, plain floats round-trip through json.dumps as
    # ordinary (unquoted) numeric literals, which is what the CLI's strict
    # JSON parser expects.
    text = json.dumps(_decimal_to_float(record), indent=2)
    path = tmp_path / "result.json"
    path.write_text(text)
    result = subprocess.run(
        [PY, str(CLI), "verify-result", "--result", str(path), "--repo-root", str(ROOT)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "RESULT PASS" in result.stdout


def test_cli_valid_checkpoint_end_to_end(tmp_path):
    identity = _identity()
    checkpoint = build_valid_checkpoint(identity, window_count=10)
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(checkpoint, indent=2))
    result = subprocess.run(
        [PY, str(CLI), "verify-checkpoint", "--checkpoint", str(path), "--repo-root", str(ROOT)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "CHECKPOINT PASS" in result.stdout


def test_real_toml_unchanged_after_all_probes():
    before = IDENTITY_PATH.read_bytes()
    identity = _identity()
    build_valid_record(identity)
    build_valid_checkpoint(identity)
    after = IDENTITY_PATH.read_bytes()
    assert before == after
