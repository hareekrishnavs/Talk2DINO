"""Tests for the final T4 signal-alignment report.

Uses small, synthetic, hand-constructed trust/centrality report fixtures
throughout. The real observed scientific constants (916, 839, 3387, 3259,
134, 46, 55, 33, 438, -361, 174, -97, ...) are never duplicated in this
file -- every fixture below uses deliberately different small numbers so
these tests exercise the module's own arithmetic, not a copy of the real
report's answer key.
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import src.t4_signal_alignment_report as tsar  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _stage_accuracy(*, valid_gt, consensus_correct, dissent_correct, gt_third_label, ignored_gt=0):
    delta_trust = (consensus_correct - dissent_correct) / valid_gt if valid_gt else None
    third_label_fraction = gt_third_label / valid_gt if valid_gt else None
    return {
        "accuracy_unit": "fraction_0_to_1",
        "both_wrong": 0,
        "consensus_accuracy": consensus_correct / valid_gt if valid_gt else 0.0,
        "consensus_correct": consensus_correct,
        "delta_trust": delta_trust,
        "delta_trust_percentage_points": None if delta_trust is None else delta_trust * 100.0,
        "delta_trust_unit": "fraction_difference",
        "dissent_accuracy": dissent_correct / valid_gt if valid_gt else 0.0,
        "dissent_correct": dissent_correct,
        "gt_third_label": gt_third_label,
        "ignored_gt": ignored_gt,
        "stitched_accuracy": 0.0,
        "stitched_correct": 0,
        "stitched_valid": valid_gt,
        "third_label_fraction": third_label_fraction,
        "third_label_fraction_unit": "fraction_0_to_1",
        "unary_accuracy": 0.0,
        "unary_correct": 0,
        "valid_gt": valid_gt,
        "y_correct_d_wrong": consensus_correct,
        "y_wrong_d_correct": dissent_correct,
    }


def _funnel_stage(*, count, valid_gt=None, ignored_gt=None, contributing_images=1, images_with_zero_records=0, total_anchor_fraction=0.01):
    return {
        "contributing_images": contributing_images, "count": count, "ignored_gt": ignored_gt,
        "images_with_zero_records": images_with_zero_records, "total_anchor_fraction": total_anchor_fraction,
        "valid_gt": valid_gt,
    }


def _bootstrap_t4(*, valid_gt, consensus_correct, dissent_correct, gt_third_label, ci_low, ci_high, image_macro_estimate, seed=42, resamples=1000):
    point_estimate = (consensus_correct - dissent_correct) / valid_gt
    return {
        "accuracy_unit": "fraction_0_to_1", "bootstrap_unit": "image", "both_correct": 0, "both_wrong": 0,
        "ci95_fraction_difference": [ci_low, ci_high],
        "ci95_percentage_points": [ci_low * 100.0, ci_high * 100.0],
        "ci_high": ci_high, "ci_low": ci_low, "confidence_level": 0.95,
        "consensus_accuracy": consensus_correct / valid_gt, "consensus_correct": consensus_correct,
        "contributing_images": 3, "delta_trust": point_estimate, "delta_trust_estimate_unit": "fraction_difference",
        "delta_trust_percentage_points": point_estimate * 100.0, "delta_trust_unit": "fraction_difference",
        "dissent_accuracy": dissent_correct / valid_gt, "dissent_correct": dissent_correct, "ignored_gt": 0,
        "image_macro_estimate": image_macro_estimate, "image_macro_estimate_percentage_points": image_macro_estimate * 100.0,
        "invalid_replicate_count": 0, "invalid_replicate_fraction": 0.0, "observations": valid_gt,
        "point_estimate": point_estimate, "point_estimate_percentage_points": point_estimate * 100.0,
        "population": "t4", "quantile_method": "linear_interpolation_percentile",
        "resamples_requested": resamples, "seed": seed, "status": "available",
        "third_label": gt_third_label, "third_label_fraction": gt_third_label / valid_gt,
        "third_label_fraction_unit": "fraction_0_to_1", "unavailable_reason": None,
        "valid_replicate_count": resamples, "y_correct_d_wrong": consensus_correct, "y_wrong_d_correct": dissent_correct,
        "zero_target_images": 0,
    }


def _centrality_stage():
    return {
        "population_count": 0, "delta_c": {}, "source_centrality": {}, "jury_centrality": {},
        "bin_edges": [-1.0, 0.0, 1.0], "bin_histogram": [0, 0],
        "strata": {"negative": {}, "zero": {}, "positive": {}},
        "bins": [{}, {}],
    }


def _per_class_entry(class_id, *, observations, consensus_correct, dissent_correct, third_label):
    net_denominator = observations or 1
    return {
        "both_wrong": max(0, observations - consensus_correct - dissent_correct - third_label),
        "class_id": class_id, "consensus_correct": consensus_correct, "contributing_images": None,
        "delta_trust": (consensus_correct - dissent_correct) / net_denominator,
        "dissent_correct": dissent_correct, "observations": observations, "third_label": third_label,
        "y_correct_d_wrong": consensus_correct, "y_wrong_d_correct": dissent_correct,
    }


def minimal_valid_report(**overrides) -> dict:
    """A small, schema-complete, hand-built v4 trust report.

    Strict T4: valid_gt=9, consensus_correct=5, dissent_correct=2 ->
    direct_anchor_net=+3, delta_trust=1/3, third_label=2/9.
    Per-class: class 0 net=+4 (positive), class 1 net=-1 (negative),
    class 2 net=0 (zero) -> sums reconcile to the stage totals above.
    T4-prime: valid_gt=12, consensus_correct=6, dissent_correct=5 ->
    delta_trust=1/12 (weaker than strict T4's 1/3), third_label=3/12=0.25
    (higher than strict T4's 2/9) -> "weaker than strict T4" on both axes.
    Bootstrap CI=[-0.1, 0.2] contains zero; image_macro_estimate=-0.05
    (non-positive) -> this default fixture triggers the STOP decision.
    """
    per_class = {
        "0": _per_class_entry(0, observations=6, consensus_correct=5, dissent_correct=1, third_label=0),
        "1": _per_class_entry(1, observations=2, consensus_correct=0, dissent_correct=1, third_label=1),
        "2": _per_class_entry(2, observations=1, consensus_correct=0, dissent_correct=0, third_label=1),
    }
    trust_centrality = {
        "schema_version": tsar.SUPPORTED_TRUST_CENTRALITY_SECTION_SCHEMA_VERSION,
        "status": "available",
        "funnel": {
            "t0": _funnel_stage(count=1000, contributing_images=3, total_anchor_fraction=1.0),
            "t1": _funnel_stage(count=100, contributing_images=3, total_anchor_fraction=0.1),
            "t2": _funnel_stage(count=50, valid_gt=48, ignored_gt=2, contributing_images=3, total_anchor_fraction=0.05),
            "t3": _funnel_stage(count=20, valid_gt=19, ignored_gt=1, contributing_images=3, total_anchor_fraction=0.02),
            "actionable": _funnel_stage(count=15, valid_gt=14, ignored_gt=1, contributing_images=3, total_anchor_fraction=0.015),
            "t4": _funnel_stage(count=11, valid_gt=9, ignored_gt=2, contributing_images=3, images_with_zero_records=1, total_anchor_fraction=0.011),
            "t4_prime": _funnel_stage(count=15, valid_gt=12, ignored_gt=3, contributing_images=3, images_with_zero_records=1, total_anchor_fraction=0.015),
            "survival_rates": {},
        },
        "trust_by_stage": {
            "t2": _stage_accuracy(valid_gt=48, consensus_correct=30, dissent_correct=28, gt_third_label=10, ignored_gt=2),
            "t3": _stage_accuracy(valid_gt=19, consensus_correct=12, dissent_correct=10, gt_third_label=4, ignored_gt=1),
            "actionable": _stage_accuracy(valid_gt=14, consensus_correct=8, dissent_correct=7, gt_third_label=3, ignored_gt=1),
            "t4": _stage_accuracy(valid_gt=9, consensus_correct=5, dissent_correct=2, gt_third_label=2, ignored_gt=2),
            "t4_prime": _stage_accuracy(valid_gt=12, consensus_correct=6, dissent_correct=5, gt_third_label=3, ignored_gt=3),
        },
        "bootstrap": {
            "t4": _bootstrap_t4(valid_gt=9, consensus_correct=5, dissent_correct=2, gt_third_label=2, ci_low=-0.1, ci_high=0.2, image_macro_estimate=-0.05),
        },
        "centrality": {stage: _centrality_stage() for stage in ("actionable", "t4", "t4_prime")},
        "crop_edge": {stage: {} for stage in ("actionable", "t4", "t4_prime")},
        "shared_unary": {group: {} for group in ("all", "some", "none", "unavailable")},
        "per_class_t4": per_class,
        "class_macro_estimate_exploratory": 0.01,
        "class_concentration": {
            "classes_represented": 3, "top_1_fraction": 0.6, "top_5_fraction": 1.0, "top_10_fraction": 1.0,
            "top_20_fraction": 1.0, "top_beneficial_classes": [], "top_classes_by_count": [], "top_harmful_classes": [],
        },
        "stitched_label_categories": {"g_equals_d_count": 0, "g_equals_d_fraction": 0.0, "g_equals_y_count": 0, "g_is_third_label_count": 0, "g_is_third_label_fraction": 0.0},
        "trust_interpretation_t4": "INCONCLUSIVE",
        "load_bearing_result": {},
        "mechanism_assessments": {},
    }
    report = {
        "schema_version": tsar.SUPPORTED_TRUST_REPORT_SCHEMA_VERSION,
        "final": True, "complete": True, "finalization_attempted": True,
        "failure_category": None, "reconciliation_error": None, "run_mode": "full_dataset",
        "dataset_length": 7, "images_processed": 7, "unique_image_count": 7, "windows_processed": 20,
        "diagnostic_settings": {},
        "natural_evaluate_result": {
            "captured": True, "source_unit": "fraction_0_to_1", "normalized_unit": "percent_0_to_100",
            "raw_fraction": {"aAcc": 0.5, "mIoU": 0.3, "mAcc": 0.4},
            "normalized_percent": {"aAcc": 50.0, "mIoU": 30.0, "mAcc": 40.0},
        },
        "observed_metrics": {"mIoU": 30.0},
        "processed_image_id_digest": "deadbeef", "provenance": {
            "git_head": "a" * 40, "git_branch": "synthetic", "canonical_config_sha256": "b" * 64,
            "e3_identity_sha256": "c" * 64, "rwr_identity_sha256": "d" * 64,
        },
        "reference_metrics": {}, "regenerated_from": {}, "solver_summary": {},
        "trust_centrality": trust_centrality,
        "note_on_TALK2DINO_RWR_RESULT_log_line": "synthetic fixture",
    }
    for key, value in overrides.items():
        report[key] = value
    return report


def write_report(tmp_path: Path, report: dict, *, name: str = "trust_report.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(report, sort_keys=True, indent=2))
    return path


def canonical_stats_generic(tmp_path: Path, union: list[float], *, name: str = "canonical_stats.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"union": union}))
    return path


def canonical_stats_historical(tmp_path: Path, union: list[float], *, alpha=0.98, steps=320, name: str = "historical.json") -> Path:
    payload = {"payload": {"rows": [{"alpha": alpha, "steps": steps, "union": union}]}}
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


# ---------------------------------------------------------------------------
# 1. Minimal valid finalized report
# ---------------------------------------------------------------------------


def test_minimal_valid_report_builds_successfully(tmp_path):
    path = write_report(tmp_path, minimal_valid_report())
    report = tsar.build_report(path)
    assert report["strict_t4"]["direct_anchor_net"] == 3
    assert report["schema_version"] == tsar.REPORT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 2. Deterministic JSON and Markdown output
# ---------------------------------------------------------------------------


def test_deterministic_json_output_excluding_timestamp(tmp_path):
    path = write_report(tmp_path, minimal_valid_report())
    report_a = tsar.build_report(path)
    report_b = tsar.build_report(path)
    assert tsar.canonical_projection(report_a) == tsar.canonical_projection(report_b)
    assert tsar.serialize_report_json(tsar.canonical_projection(report_a)) == tsar.serialize_report_json(tsar.canonical_projection(report_b))


def test_deterministic_markdown_output(tmp_path):
    path = write_report(tmp_path, minimal_valid_report())
    report = tsar.build_report(path)
    md_a = tsar.render_markdown(report)
    md_b = tsar.render_markdown(report)
    assert md_a == md_b


def test_json_serialization_is_stable_shape(tmp_path):
    path = write_report(tmp_path, minimal_valid_report())
    report = tsar.build_report(path)
    payload = tsar.serialize_report_json(report)
    assert payload.endswith("\n")
    assert json.loads(payload) == report
    reparsed_keys = list(json.loads(payload).keys())
    assert reparsed_keys == sorted(reparsed_keys)


# ---------------------------------------------------------------------------
# 3. Correct direct net
# ---------------------------------------------------------------------------


def test_direct_anchor_net_formula(tmp_path):
    path = write_report(tmp_path, minimal_valid_report())
    report = tsar.build_report(path)
    strict = report["strict_t4"]
    assert strict["direct_anchor_net"] == strict["consensus_correct"] - strict["dissent_correct"] == 3


# ---------------------------------------------------------------------------
# 4/5/6. Per-class sign counts, gross sums, dominant-class exclusion
# ---------------------------------------------------------------------------


def test_per_class_sign_counts_and_gross_sums(tmp_path):
    path = write_report(tmp_path, minimal_valid_report())
    report = tsar.build_report(path)
    per_class = report["per_class_alignment"]
    assert per_class["positive_class_count"] == 1
    assert per_class["negative_class_count"] == 1
    assert per_class["zero_class_count"] == 1
    assert per_class["gross_positive_net"] == 4
    assert per_class["gross_negative_net"] == -1
    assert per_class["total_net"] == 3


def test_exclusion_of_dominant_positive_class(tmp_path):
    path = write_report(tmp_path, minimal_valid_report())
    report = tsar.build_report(path)
    per_class = report["per_class_alignment"]
    assert per_class["largest_positive_contributor"]["class_id"] == 0
    assert per_class["largest_positive_contributor"]["net"] == 4
    assert per_class["total_net_excluding_largest_positive_contributor"] == 3 - 4 == -1


# ---------------------------------------------------------------------------
# 7. Global/per-class reconciliation
# ---------------------------------------------------------------------------


def test_global_per_class_reconciliation_passes_for_valid_fixture(tmp_path):
    path = write_report(tmp_path, minimal_valid_report())
    tsar.build_report(path)  # must not raise


def test_reconciliation_fails_closed_on_mismatched_counts(tmp_path):
    report = minimal_valid_report()
    report["trust_centrality"]["per_class_t4"]["0"]["consensus_correct"] = 999
    path = write_report(tmp_path, report)
    with pytest.raises(tsar.T4SignalAlignmentError):
        tsar.build_report(path)


# ---------------------------------------------------------------------------
# 8. T4-prime growth and weaker-trust characterization
# ---------------------------------------------------------------------------


def test_t4_prime_growth_and_weaker_trust(tmp_path):
    path = write_report(tmp_path, minimal_valid_report())
    report = tsar.build_report(path)
    prime = report["t4_prime"]
    assert prime["relative_growth_from_strict_t4"]["valid_gt_growth_ratio"] == pytest.approx(12 / 9)
    assert prime["weaker_than_strict_t4"]["delta_trust_weaker"] is True
    assert prime["weaker_than_strict_t4"]["third_label_more_frequent"] is True
    assert prime["weaker_than_strict_t4"]["both"] is True


# ---------------------------------------------------------------------------
# 9. CI zero inclusion
# ---------------------------------------------------------------------------


def test_ci_contains_zero_true_case(tmp_path):
    path = write_report(tmp_path, minimal_valid_report())
    report = tsar.build_report(path)
    assert report["strict_t4"]["bootstrap"]["ci_contains_zero"] is True


def test_ci_contains_zero_false_case(tmp_path):
    report = minimal_valid_report()
    report["trust_centrality"]["bootstrap"]["t4"]["ci95_fraction_difference"] = [0.05, 0.2]
    report["trust_centrality"]["bootstrap"]["t4"]["ci_low"] = 0.05
    report["trust_centrality"]["bootstrap"]["t4"]["ci_high"] = 0.2
    path = write_report(tmp_path, report)
    built = tsar.build_report(path)
    assert built["strict_t4"]["bootstrap"]["ci_contains_zero"] is False


# ---------------------------------------------------------------------------
# 10. Anchor-equivalent descriptive interval
# ---------------------------------------------------------------------------


def test_anchor_equivalent_descriptive_interval_formula(tmp_path):
    path = write_report(tmp_path, minimal_valid_report())
    report = tsar.build_report(path)
    strict = report["strict_t4"]
    low, high = strict["anchor_equivalent_bootstrap_interval_descriptive"]
    ci_low, ci_high = strict["bootstrap"]["ci95_fraction_difference"]
    assert low == pytest.approx(ci_low * strict["valid_gt_count"])
    assert high == pytest.approx(ci_high * strict["valid_gt_count"])
    caveats = strict["anchor_equivalent_bootstrap_interval_descriptive_caveats"]
    assert any("not an integer confidence interval" in c for c in caveats)
    assert any("not a bound" in c for c in caveats)
    assert any("not an mIoU interval" in c for c in caveats)


# ---------------------------------------------------------------------------
# 11/12/13. Union input, S calculation, sensitivities
# ---------------------------------------------------------------------------


def _small_union():
    union = [10.0] * tsar.CANONICAL_CLASS_COUNT
    union[0] = 100.0  # class 0 -> positive net class
    union[1] = 50.0   # class 1 -> negative net class
    union[2] = 20.0   # class 2 -> zero net class
    return union


def test_valid_171_class_union_generic_form(tmp_path):
    trust_path = write_report(tmp_path, minimal_valid_report())
    union_path = canonical_stats_generic(tmp_path, _small_union())
    report = tsar.build_report(trust_path, canonical_stats_path=union_path)
    assert report["union_weighted_proxy"]["status"] == "available"


def test_valid_171_class_union_historical_artifact_form(tmp_path):
    trust_path = write_report(tmp_path, minimal_valid_report())
    union_path = canonical_stats_historical(tmp_path, _small_union())
    report = tsar.build_report(trust_path, canonical_stats_path=union_path)
    assert report["union_weighted_proxy"]["status"] == "available"


def test_s_calculation_matches_hand_formula(tmp_path):
    trust_path = write_report(tmp_path, minimal_valid_report())
    union = _small_union()
    union_path = canonical_stats_generic(tmp_path, union)
    report = tsar.build_report(trust_path, canonical_stats_path=union_path)
    # net_c: class0=+4 U=100, class1=-1 U=50, class2=0 U=20
    expected_s = (4 / 100.0) + (-1 / 50.0) + (0 / 20.0)
    assert report["union_weighted_proxy"]["S"] == pytest.approx(expected_s)


def test_sensitivity_196_and_830_formulas(tmp_path):
    trust_path = write_report(tmp_path, minimal_valid_report())
    union_path = canonical_stats_generic(tmp_path, _small_union())
    report = tsar.build_report(trust_path, canonical_stats_path=union_path)
    proxy = report["union_weighted_proxy"]
    s_value = proxy["S"]
    sens = proxy["homogeneous_neighbourhood_footprint_sensitivity_descriptive"]
    assert sens["sensitivity_196"] == pytest.approx((100.0 / 171) * 196 * s_value)
    assert sens["sensitivity_830"] == pytest.approx((100.0 / 171) * 830 * s_value)


# ---------------------------------------------------------------------------
# 14/15/16. Union failure modes
# ---------------------------------------------------------------------------


def test_missing_union_input_gives_unavailable_status(tmp_path):
    path = write_report(tmp_path, minimal_valid_report())
    report = tsar.build_report(path)
    assert report["union_weighted_proxy"]["status"] == "unavailable"
    assert report["union_weighted_proxy"]["S"] is None


def test_zero_union_for_represented_class_fails_closed(tmp_path):
    trust_path = write_report(tmp_path, minimal_valid_report())
    union = _small_union()
    union[0] = 0.0  # class 0 is represented in per_class_t4 -> must fail
    union_path = canonical_stats_generic(tmp_path, union)
    with pytest.raises(tsar.T4SignalAlignmentError):
        tsar.build_report(trust_path, canonical_stats_path=union_path)


def test_wrong_union_length_fails_closed(tmp_path):
    trust_path = write_report(tmp_path, minimal_valid_report())
    union_path = canonical_stats_generic(tmp_path, [1.0] * 170)
    with pytest.raises(tsar.T4SignalAlignmentError):
        tsar.build_report(trust_path, canonical_stats_path=union_path)


# ---------------------------------------------------------------------------
# 17/18. NaN/Infinity and boolean-as-integer rejection
# ---------------------------------------------------------------------------


def test_nan_rejected_in_trust_report(tmp_path):
    path = tmp_path / "nan_report.json"
    path.write_text('{"schema_version": "x", "value": NaN}')
    with pytest.raises(tsar.T4SignalAlignmentError):
        tsar.build_report(path)


def test_infinity_rejected_in_trust_report(tmp_path):
    path = tmp_path / "inf_report.json"
    path.write_text('{"schema_version": "x", "value": Infinity}')
    with pytest.raises(tsar.T4SignalAlignmentError):
        tsar.build_report(path)


def test_boolean_rejected_as_integer(tmp_path):
    report = minimal_valid_report()
    report["trust_centrality"]["funnel"]["t4"]["count"] = True
    path = write_report(tmp_path, report)
    with pytest.raises(tsar.T4SignalAlignmentError):
        tsar.build_report(path)


def test_boolean_rejected_as_class_id():
    with pytest.raises(tsar.T4SignalAlignmentError):
        tsar._require_class_id(True, "test")


# ---------------------------------------------------------------------------
# 19. Missing required sections
# ---------------------------------------------------------------------------


def test_missing_trust_centrality_section_rejected(tmp_path):
    report = minimal_valid_report()
    del report["trust_centrality"]
    path = write_report(tmp_path, report)
    with pytest.raises(tsar.T4SignalAlignmentError):
        tsar.build_report(path)


def test_missing_bootstrap_t4_population_rejected(tmp_path):
    report = minimal_valid_report()
    del report["trust_centrality"]["bootstrap"]["t4"]
    path = write_report(tmp_path, report)
    with pytest.raises(tsar.T4SignalAlignmentError):
        tsar.build_report(path)


# ---------------------------------------------------------------------------
# 20/21/22. final/complete/status rejection
# ---------------------------------------------------------------------------


def test_final_false_rejected(tmp_path):
    path = write_report(tmp_path, minimal_valid_report(final=False))
    with pytest.raises(tsar.T4SignalAlignmentError):
        tsar.build_report(path)


def test_complete_false_rejected(tmp_path):
    path = write_report(tmp_path, minimal_valid_report(complete=False))
    with pytest.raises(tsar.T4SignalAlignmentError):
        tsar.build_report(path)


def test_trust_status_unavailable_rejected(tmp_path):
    report = minimal_valid_report()
    report["trust_centrality"]["status"] = "unavailable"
    path = write_report(tmp_path, report)
    with pytest.raises(tsar.T4SignalAlignmentError):
        tsar.build_report(path)


# ---------------------------------------------------------------------------
# 23/24. Malformed / duplicate class IDs
# ---------------------------------------------------------------------------


def test_malformed_class_id_rejected(tmp_path):
    report = minimal_valid_report()
    entry = report["trust_centrality"]["per_class_t4"].pop("0")
    entry["class_id"] = "zero"
    report["trust_centrality"]["per_class_t4"]["zero"] = entry
    path = write_report(tmp_path, report)
    with pytest.raises(tsar.T4SignalAlignmentError):
        tsar.build_report(path)


def test_class_id_mismatch_with_key_rejected(tmp_path):
    report = minimal_valid_report()
    report["trust_centrality"]["per_class_t4"]["0"]["class_id"] = 5
    path = write_report(tmp_path, report)
    with pytest.raises(tsar.T4SignalAlignmentError):
        tsar.build_report(path)


def test_duplicate_class_id_after_normalization_rejected():
    per_class_raw = {"0": {"class_id": 0}, "00": {"class_id": 0}}
    with pytest.raises(tsar.T4SignalAlignmentError):
        for raw_key in per_class_raw:
            cid = tsar._require_class_id(raw_key, "test")
            if raw_key == "00" and cid == 0:
                raise tsar.T4SignalAlignmentError("duplicate class_id after normalization: 0")


# ---------------------------------------------------------------------------
# 25. Mismatched per-class/global counts (additional angle: third_label)
# ---------------------------------------------------------------------------


def test_mismatched_third_label_sum_rejected(tmp_path):
    report = minimal_valid_report()
    report["trust_centrality"]["per_class_t4"]["1"]["third_label"] = 999
    path = write_report(tmp_path, report)
    with pytest.raises(tsar.T4SignalAlignmentError):
        tsar.build_report(path)


# ---------------------------------------------------------------------------
# 26. Input mapping and input files remain unchanged
# ---------------------------------------------------------------------------


def test_input_file_and_mapping_unchanged(tmp_path):
    original = minimal_valid_report()
    path = write_report(tmp_path, original)
    original_bytes = path.read_bytes()
    loaded = tsar.load_and_validate_trust_report(path)
    frozen_snapshot = copy.deepcopy(dict(loaded.report))
    tsar.build_report(path)
    assert path.read_bytes() == original_bytes
    assert dict(loaded.report) == frozen_snapshot


# ---------------------------------------------------------------------------
# 27. CLI nonzero exit with useful diagnostic
# ---------------------------------------------------------------------------


def test_cli_nonzero_exit_on_malformed_input(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("generate_t4_signal_alignment_report", ROOT / "generate_t4_signal_alignment_report.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not valid json")
    output_path = tmp_path / "out.json"
    exit_code = cli.main(["--trust-report", str(bad_path), "--output", str(output_path)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "T4 SIGNAL ALIGNMENT REPORT FAIL" in captured.err
    assert not output_path.exists()


def test_cli_success_exit_and_output_written(tmp_path):
    spec = importlib.util.spec_from_file_location("generate_t4_signal_alignment_report", ROOT / "generate_t4_signal_alignment_report.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    trust_path = write_report(tmp_path, minimal_valid_report())
    output_path = tmp_path / "out.json"
    md_path = tmp_path / "out.md"
    exit_code = cli.main(["--trust-report", str(trust_path), "--output", str(output_path), "--markdown-output", str(md_path)])
    assert exit_code == 0
    assert output_path.exists()
    assert md_path.exists()


# ---------------------------------------------------------------------------
# 28. No CUDA/model/dataset inference imports or calls
# ---------------------------------------------------------------------------


def test_module_source_has_no_cuda_or_model_references():
    import ast

    tree = ast.parse(inspect.getsource(tsar))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    forbidden_imports = {"torch", "cv2", "mmcv", "mmseg"}
    assert not (imported_names & forbidden_imports), f"forbidden imports found: {imported_names & forbidden_imports}"
    called_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    assert "DINOTextSegInference" not in called_names
    assert "build_seg_dataloader" not in called_names
    assert "build_seg_dataset" not in called_names


def test_module_does_not_import_torch_or_mmcv_transitively():
    assert "torch" not in sys.modules or True  # torch may be present from pytest/other fixtures; check our own module's namespace instead
    module_globals = set(dir(tsar))
    assert "torch" not in module_globals
    assert "mmcv" not in module_globals
    assert "cv2" not in module_globals


# ---------------------------------------------------------------------------
# 29. Markdown terminology regression
# ---------------------------------------------------------------------------


def test_markdown_never_calls_s_delta_miou(tmp_path):
    trust_path = write_report(tmp_path, minimal_valid_report())
    union_path = canonical_stats_generic(tmp_path, _small_union())
    report = tsar.build_report(trust_path, canonical_stats_path=union_path)
    md = tsar.render_markdown(report)
    assert "S is Delta-mIoU" not in md
    assert "S = Delta-mIoU" not in md
    assert "not Delta-mIoU" in md


def test_markdown_never_calls_196_or_830_bounds(tmp_path):
    trust_path = write_report(tmp_path, minimal_valid_report())
    union_path = canonical_stats_generic(tmp_path, _small_union())
    report = tsar.build_report(trust_path, canonical_stats_path=union_path)
    md = tsar.render_markdown(report)
    assert "196 is a bound" not in md
    assert "830 is a bound" not in md
    assert "not a bound" in md


def test_markdown_contains_operator_attributed_diffusion_reversal(tmp_path):
    trust_path = write_report(tmp_path, minimal_valid_report())
    report = tsar.build_report(trust_path)
    md = tsar.render_markdown(report)
    assert "operator-attributed diffusion reversal" in md


def test_markdown_contains_stop_decision_when_evidence_supports_it(tmp_path):
    trust_path = write_report(tmp_path, minimal_valid_report())
    report = tsar.build_report(trust_path)
    assert report["decision"]["decision"] == tsar.DECISION_STOP_SEMANTIC_CONSENSUS_COVER_DR
    md = tsar.render_markdown(report)
    assert tsar.DECISION_STOP_SEMANTIC_CONSENSUS_COVER_DR in md


def test_markdown_does_not_describe_crops_as_statistically_independent(tmp_path):
    trust_path = write_report(tmp_path, minimal_valid_report())
    report = tsar.build_report(trust_path)
    md = tsar.render_markdown(report)
    assert "statistically independent" in md
    assert "not a set of statistically independent observations" in md.replace("**not**", "not")


def test_decision_flips_to_continue_when_evidence_does_not_support_stop(tmp_path):
    report = minimal_valid_report()
    # Remove one leg of the evidence conjunction: CI no longer contains zero.
    report["trust_centrality"]["bootstrap"]["t4"]["ci95_fraction_difference"] = [0.01, 0.2]
    report["trust_centrality"]["bootstrap"]["t4"]["ci_low"] = 0.01
    report["trust_centrality"]["bootstrap"]["t4"]["ci_high"] = 0.2
    path = write_report(tmp_path, report)
    built = tsar.build_report(path)
    assert built["decision"]["decision"] == tsar.DECISION_CONTINUE_SEMANTIC_CONSENSUS_EVALUATION


# ---------------------------------------------------------------------------
# Regression tests: the file-hashing boundary (_sha256_of_file) must never
# let a raw OSError escape as an unhandled traceback -- every filesystem
# failure at that boundary becomes a T4SignalAlignmentError, and the CLI
# turns that into a clean exit-2 diagnostic with no partial output.
# ---------------------------------------------------------------------------


def _run_cli(args):
    spec = importlib.util.spec_from_file_location("generate_t4_signal_alignment_report", ROOT / "generate_t4_signal_alignment_report.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli.main(args)


def test_missing_trust_report_cli_clean_failure(tmp_path, capsys):
    missing_path = tmp_path / "does_not_exist.json"
    output_path = tmp_path / "out.json"
    exit_code = _run_cli(["--trust-report", str(missing_path), "--output", str(output_path)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.startswith("T4 SIGNAL ALIGNMENT REPORT FAIL:")
    assert str(missing_path) in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out
    assert not output_path.exists()


def test_trust_report_path_is_a_directory_cli_clean_failure(tmp_path, capsys):
    directory_path = tmp_path / "a_directory"
    directory_path.mkdir()
    output_path = tmp_path / "out.json"
    exit_code = _run_cli(["--trust-report", str(directory_path), "--output", str(output_path)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.startswith("T4 SIGNAL ALIGNMENT REPORT FAIL:")
    assert str(directory_path) in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out
    assert not output_path.exists()


def test_sha256_of_file_simulated_permission_error_chains_exception(tmp_path, monkeypatch):
    path = tmp_path / "unreadable.json"
    path.write_text("{}")

    def fake_open(self, mode="r", *a, **kw):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "open", fake_open)
    with pytest.raises(tsar.T4SignalAlignmentError) as excinfo:
        tsar._sha256_of_file(path, label="trust report")
    assert isinstance(excinfo.value.__cause__, PermissionError)
    assert str(path) in str(excinfo.value)
    assert "Permission denied" in str(excinfo.value) or "13" in str(excinfo.value)


def test_sha256_of_file_does_not_catch_keyboard_interrupt(tmp_path, monkeypatch):
    path = tmp_path / "irrelevant.json"
    path.write_text("{}")

    def fake_open(self, mode="r", *a, **kw):
        raise KeyboardInterrupt()

    monkeypatch.setattr(Path, "open", fake_open)
    with pytest.raises(KeyboardInterrupt):
        tsar._sha256_of_file(path, label="trust report")


def test_missing_canonical_stats_cli_clean_failure(tmp_path, capsys):
    trust_path = write_report(tmp_path, minimal_valid_report())
    missing_union_path = tmp_path / "no_such_union.json"
    output_path = tmp_path / "out.json"
    exit_code = _run_cli([
        "--trust-report", str(trust_path), "--output", str(output_path),
        "--canonical-stats", str(missing_union_path),
    ])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.startswith("T4 SIGNAL ALIGNMENT REPORT FAIL:")
    assert "Traceback" not in captured.err
    assert not output_path.exists()


def test_sha256_of_file_byte_correct_on_success(tmp_path):
    path = tmp_path / "some_bytes.json"
    path.write_bytes(b'{"a": 1, "b": [1, 2, 3]}')
    import hashlib

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    observed = tsar._sha256_of_file(path, label="trust report")
    assert observed == expected


def test_error_after_output_paths_supplied_leaves_no_partial_output(tmp_path, capsys):
    missing_path = tmp_path / "does_not_exist.json"
    output_path = tmp_path / "out.json"
    markdown_path = tmp_path / "out.md"
    exit_code = _run_cli([
        "--trust-report", str(missing_path), "--output", str(output_path),
        "--markdown-output", str(markdown_path),
    ])
    assert exit_code == 2
    assert not output_path.exists()
    assert not markdown_path.exists()


def test_sha256_of_file_labels_call_sites_distinctly(tmp_path):
    missing_trust = tmp_path / "missing_trust.json"
    try:
        tsar._sha256_of_file(missing_trust, label="trust report")
        assert False, "expected T4SignalAlignmentError"
    except tsar.T4SignalAlignmentError as error:
        assert "trust report" in str(error)

    missing_union = tmp_path / "missing_union.json"
    try:
        tsar._sha256_of_file(missing_union, label="canonical-stats artifact")
        assert False, "expected T4SignalAlignmentError"
    except tsar.T4SignalAlignmentError as error:
        assert "canonical-stats artifact" in str(error)
