"""CPU-only coverage for src.native_edge_support_reachability_gate: parent
binding/validation, independent aggregate reconciliation, undefined-cause
decomposition, decision matrix, and roadmap authorization. Uses the REAL
mechanics20 result where available (ALIGNMENT_LIMITED case) plus small,
fully hand-verified synthetic mechanics20-schema fixtures for the other
three decision outcomes and negative-path testing. Never initializes CUDA,
never loads the model or dataset."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.native_edge_support_reachability_gate_identity import load_identity as load_gate_identity  # noqa: E402
from src.native_edge_support_reachability_gate import (  # noqa: E402
    NativeEdgeSupportReachabilityGateError,
    build_gate_report,
    decompose_undefined_causes,
    map_roadmap_authorization,
    reconcile_aggregate_ratios,
    reproduce_parent_decision,
    select_and_validate_parent_artifact,
    sha256_file,
)
from src.native_edge_support_identity import load_identity as load_native_audit_identity  # noqa: E402
from src.native_edge_support_report import verify_record as verify_native_audit_record  # noqa: E402

GATE_IDENTITY_PATH = ROOT / "evaluation_identities/e12_native_edge_support_reachability_gate.toml"
REAL_PARENT_RESULT = Path("/scratch/haree/e12_native_edge_support_audit/result-mechanics20-20420785.json")

pytestmark = pytest.mark.skipif(not GATE_IDENTITY_PATH.exists(), reason="requires the reachability-gate identity")


@pytest.fixture(scope="module")
def gate_identity():
    return load_gate_identity(repo_root=ROOT)


@pytest.fixture(scope="module")
def native_audit_identity():
    return load_native_audit_identity(repo_root=ROOT)


# ---------------------------------------------------------------------------
# Synthetic mechanics20-schema fixture builder -- small numbers, every
# derived field computed and self-checked against the REAL native-audit
# verifier before use, so gate tests never rely on an internally
# inconsistent fixture.
# ---------------------------------------------------------------------------

_VALID_SHA = "a" * 64
_VALID_COMMIT = "b" * 40
_SUPPORT_COUNT_KEYS = tuple(str(n) for n in range(13))
_SUPPORT_FRACTION_KEYS = (
    "0.0", "(0.0,0.1]", "(0.1,0.2]", "(0.2,0.3]", "(0.3,0.4]", "(0.4,0.5]",
    "(0.5,0.6]", "(0.6,0.7]", "(0.7,0.8]", "(0.8,0.9]", "(0.9,1.0]",
)
_BAND_KEYS = ("0", "1", "2", "3-4", "5-7", ">=8")
_RANK_KEYS = tuple(str(r) for r in range(12))


def _make_synthetic_record(
    identity, *, windows=2, edges_with_observer, misclassified_rows, misclassified_rows_any_defined,
    correct_rows_any_defined, ignored_gt_count=0, single_window_edges=0,
):
    graph_rows = windows * 1024
    directed_edges = graph_rows * 12
    undefined_edges = directed_edges - edges_with_observer
    assert 0 <= single_window_edges <= undefined_edges
    remaining_undefined = undefined_edges - single_window_edges

    rows_any_defined = misclassified_rows_any_defined + correct_rows_any_defined
    assert misclassified_rows_any_defined <= misclassified_rows <= graph_rows - ignored_gt_count
    assert rows_any_defined <= graph_rows

    counter0 = {k: 0 for k in _SUPPORT_COUNT_KEYS}
    counter0["0"] = edges_with_observer  # arbitrary valid distribution: sum must equal edges_with_observer
    fraction0 = {k: 0 for k in _SUPPORT_FRACTION_KEYS}
    fraction0["0.0"] = edges_with_observer
    band_graph_rows = {k: 0 for k in _BAND_KEYS}
    band_graph_rows[">=8"] = graph_rows
    band_directed_edges = {k: 0 for k in _BAND_KEYS}
    band_directed_edges[">=8"] = directed_edges
    rank_hist = {k: 0 for k in _RANK_KEYS}
    rank_hist["0"] = directed_edges

    correct_rows_total = graph_rows - ignored_gt_count - misclassified_rows
    cross_tab_defined_correct = correct_rows_any_defined
    cross_tab_defined_misclassified = misclassified_rows_any_defined
    cross_tab_undefined_correct = correct_rows_total - correct_rows_any_defined
    cross_tab_undefined_misclassified = misclassified_rows - misclassified_rows_any_defined
    assert cross_tab_undefined_correct >= 0 and cross_tab_undefined_misclassified >= 0
    cross_tab = {
        "defined_and_correct": cross_tab_defined_correct, "defined_and_misclassified": cross_tab_defined_misclassified,
        "undefined_and_correct": cross_tab_undefined_correct, "undefined_and_misclassified": cross_tab_undefined_misclassified,
    }

    rows_all_defined = 0  # keep simple/valid: never claim all-defined in this synthetic fixture
    rows_any_defined_zero_support = 0
    rows_with_unique_least_support = rows_any_defined
    rows_tied_for_least_support = 0

    funnel = {
        "images": 20, "windows": windows, "graph_rows": graph_rows, "directed_edges": directed_edges,
        "rows_with_other_crop": graph_rows if windows > 1 else 0,
        "rows_with_aligned_observer": rows_any_defined,
        "edges_with_observer": edges_with_observer,
        "rows_any_defined": rows_any_defined, "rows_all_defined": rows_all_defined,
        "rows_any_defined_zero_support": rows_any_defined_zero_support,
        "misclassified_rows": misclassified_rows, "misclassified_rows_any_defined": misclassified_rows_any_defined,
        "misclassified_rows_all_defined": 0,
        "aligned_window_pairs": 1 if windows > 1 else 0,
        "unaligned_window_pairs": max(windows * (windows - 1) // 2 - (1 if windows > 1 else 0), 0),
        "clamped_windows": 0, "non_clamped_windows": windows,
        "unanimous_support_edges": 0, "direction_reversal_only_cases": 0,
        "rows_with_unique_least_support": rows_with_unique_least_support,
        "rows_tied_for_least_support": rows_tied_for_least_support,
        "correct_rows_any_defined": correct_rows_any_defined,
    }

    record = {
        "schema": identity["run_modes"]["mechanics20_schema_name"], "run_mode": "mechanics20",
        "identity": identity["identity"]["name"],
        "identity_sha256": hashlib.sha256((ROOT / "evaluation_identities/e12_native_edge_support_audit.toml").read_bytes()).hexdigest(),
        "stitching_control_identity_sha256": identity["parent_identity"]["stitching_control_identity_sha256"],
        "power_evaluation_identity_sha256": identity["parent_identity"]["power_evaluation_identity_sha256"],
        "matched_identity_sha256": identity["parent_identity"]["matched_identity_sha256"],
        "e3_identity_sha256": identity["parent_identity"]["e3_identity_sha256"],
        "git_commit": _VALID_COMMIT, "complete": True, "final": False,
        "device": "cpu", "gpu_model": "cpu", "torch_version": "0.0.0", "cuda_version": "none",
        "image_count_expected": 20, "image_count_processed": 20,
        "image_order_digest": _VALID_SHA, "windows_processed_total": windows,
        "rows_processed_total": graph_rows, "edges_processed_total": directed_edges,
        "class_count": identity["dataset"]["classes"], "funnel": funnel,
        "undefined_reason_counts": {"single_window_image": single_window_edges, "no_exactly_aligned_observer_covering_both_endpoints": remaining_undefined},
        "support_count_histogram": counter0, "support_fraction_histogram": fraction0,
        "crop_edge_band_histogram": band_graph_rows, "image_edge_band_histogram": band_graph_rows,
        "displacement_band_histogram": band_directed_edges, "affinity_rank_histogram": rank_hist,
        "correctness_cross_tabs": {name: dict(cross_tab) for name in (
            "support_defined_vs_source_correct", "all_defined_vs_source_correct",
            "support_defined_vs_stitched_correct", "all_defined_vs_stitched_correct",
        )},
        "ignored_gt_count": ignored_gt_count,
        "ranking_definition": " -> ".join(identity["ranking"]["criteria_in_order"]),
        "per_image_stats_manifest_path": "stats.json", "per_image_stats_manifest_sha256": _VALID_SHA,
        "operation_telemetry": {
            "sample_pulls": 1, "window_enumerations": windows, "backbone_snapshot_calls": windows,
            "graph_builds": windows, "propagation_calls": windows, "probability_interpolation_calls": windows,
            "cross_view_comparisons_total": 0,
        },
        "phase_runtime_seconds": {"total": 1.0, "shared": 1.0},
        "peak_gpu_memory_bytes": 0, "resumed_from_checkpoint": False, "source_git_branch": "x", "failure_reason": None,
    }
    from src.native_edge_support_report import classify_reachability

    decision_output, decision_rationale = classify_reachability(funnel, identity, ignored_gt_count=ignored_gt_count)
    record["decision_output"] = decision_output
    record["decision_rationale"] = decision_rationale
    return record


@pytest.fixture(scope="module")
def synthetic_records(native_audit_identity):
    """Four synthetic records, one per decision outcome, each self-checked
    against the REAL native-audit verify_record before being handed to gate
    tests -- so a bug in the fixture builder fails loudly here, never
    silently inside a gate test."""
    minimum_rows = 20  # matches the real identity's decision.minimum_misclassified_rows_for_conclusive

    records = {}
    # INCONCLUSIVE: too few misclassified rows
    records["INCONCLUSIVE"] = _make_synthetic_record(
        native_audit_identity, edges_with_observer=100, misclassified_rows=minimum_rows - 1,
        misclassified_rows_any_defined=0, correct_rows_any_defined=0,
    )
    # ALIGNMENT_LIMITED: undefined fraction >= 0.5
    records["ALIGNMENT_LIMITED"] = _make_synthetic_record(
        native_audit_identity, edges_with_observer=1000, misclassified_rows=minimum_rows + 10,
        misclassified_rows_any_defined=5, correct_rows_any_defined=5, single_window_edges=100,
    )
    # REACHABLE: low undefined fraction, high wrong-row-defined vs correct-row-defined ratio
    records["REACHABLE"] = _make_synthetic_record(
        native_audit_identity, edges_with_observer=20000, misclassified_rows=minimum_rows + 30,
        misclassified_rows_any_defined=minimum_rows + 28, correct_rows_any_defined=2,
    )
    # STRUCTURALLY_UNREACHABLE: low undefined fraction, support concentrated at correct rows
    records["STRUCTURALLY_UNREACHABLE"] = _make_synthetic_record(
        native_audit_identity, edges_with_observer=20000, misclassified_rows=minimum_rows + 30,
        misclassified_rows_any_defined=2, correct_rows_any_defined=500,
    )

    identity_sha256 = hashlib.sha256((ROOT / "evaluation_identities/e12_native_edge_support_audit.toml").read_bytes()).hexdigest()
    for outcome, record in records.items():
        msg = verify_native_audit_record(record, native_audit_identity, identity_sha256=identity_sha256)
        assert record["decision_output"] == outcome, f"fixture builder produced {record['decision_output']!r}, expected {outcome!r}: {msg}"
    return records


# ---------------------------------------------------------------------------
# Ratio reconstruction
# ---------------------------------------------------------------------------


def test_undefined_defined_fractions_sum_to_one(synthetic_records):
    record = synthetic_records["ALIGNMENT_LIMITED"]
    ratios = reconcile_aggregate_ratios(record)
    total = ratios["undefined_edge_fraction"]["value"] + ratios["defined_edge_fraction"]["value"]
    assert abs(total - 1.0) < 1e-12


def test_wrong_row_reachability_matches_hand_computation(synthetic_records):
    record = synthetic_records["ALIGNMENT_LIMITED"]
    ratios = reconcile_aggregate_ratios(record)
    expected = record["funnel"]["misclassified_rows_any_defined"] / record["funnel"]["misclassified_rows"]
    assert ratios["wrong_row_reachability"]["value"] == pytest.approx(expected)


def test_aligned_pair_fraction_uses_exact_parent_denominator(synthetic_records):
    record = synthetic_records["ALIGNMENT_LIMITED"]
    ratios = reconcile_aggregate_ratios(record)
    funnel = record["funnel"]
    expected_denominator = funnel["aligned_window_pairs"] + funnel["unaligned_window_pairs"]
    assert ratios["aligned_pair_fraction_denominator"]["value"] == expected_denominator
    if expected_denominator > 0:
        assert ratios["aligned_pair_fraction"]["value"] == pytest.approx(funnel["aligned_window_pairs"] / expected_denominator)


def test_zero_denominator_yields_none_not_exception(synthetic_records):
    record = copy.deepcopy(synthetic_records["ALIGNMENT_LIMITED"])
    record["funnel"] = dict(record["funnel"])
    record["funnel"]["aligned_window_pairs"] = 0
    record["funnel"]["unaligned_window_pairs"] = 0
    ratios = reconcile_aggregate_ratios(record)
    assert ratios["aligned_pair_fraction"]["value"] is None
    assert ratios["aligned_pair_fraction_denominator"]["value"] == 0


def test_exact_count_reconciliation_defined_plus_undefined(synthetic_records):
    record = synthetic_records["ALIGNMENT_LIMITED"]
    ratios = reconcile_aggregate_ratios(record)
    assert ratios["support_defined_edges"]["value"] + ratios["undefined_edges"]["value"] == ratios["total_directed_edges"]["value"]


def test_mismatched_undefined_reason_sum_rejected(synthetic_records):
    record = copy.deepcopy(synthetic_records["ALIGNMENT_LIMITED"])
    record["undefined_reason_counts"] = dict(record["undefined_reason_counts"])
    record["undefined_reason_counts"]["single_window_image"] += 1  # break the sum invariant
    with pytest.raises(NativeEdgeSupportReachabilityGateError):
        reconcile_aggregate_ratios(record)


def test_cross_tab_mismatch_rejected(synthetic_records):
    record = copy.deepcopy(synthetic_records["ALIGNMENT_LIMITED"])
    record["correctness_cross_tabs"] = {k: dict(v) for k, v in record["correctness_cross_tabs"].items()}
    record["correctness_cross_tabs"]["support_defined_vs_source_correct"]["defined_and_correct"] += 1
    with pytest.raises(NativeEdgeSupportReachabilityGateError):
        reconcile_aggregate_ratios(record)


def test_negative_derived_count_rejected(synthetic_records):
    record = copy.deepcopy(synthetic_records["ALIGNMENT_LIMITED"])
    record["funnel"] = dict(record["funnel"])
    record["funnel"]["edges_with_observer"] = record["funnel"]["directed_edges"] + 1000  # forces undefined_edges negative
    with pytest.raises(NativeEdgeSupportReachabilityGateError):
        reconcile_aggregate_ratios(record)


def test_row_invariant_violation_rejected(synthetic_records):
    record = copy.deepcopy(synthetic_records["ALIGNMENT_LIMITED"])
    record["funnel"] = dict(record["funnel"])
    record["funnel"]["rows_with_unique_least_support"] += 1  # breaks unique+tied == rows_any_defined
    with pytest.raises(NativeEdgeSupportReachabilityGateError):
        reconcile_aggregate_ratios(record)


# ---------------------------------------------------------------------------
# Cause decomposition
# ---------------------------------------------------------------------------


def test_cause_categories_mutually_exclusive_and_exhaustive(synthetic_records):
    record = synthetic_records["ALIGNMENT_LIMITED"]
    causes = decompose_undefined_causes(record)
    assert causes["mutually_exclusive"] is True
    total = sum(
        entry["count"] for entry in causes["categories"].values() if entry["available"]
    )
    assert total == record["funnel"]["directed_edges"]


def test_combined_unknown_category_used_when_only_combined_reason_available(synthetic_records):
    record = synthetic_records["ALIGNMENT_LIMITED"]
    causes = decompose_undefined_causes(record)
    combined = causes["categories"]["no_native_observer_cause_not_further_identifiable"]
    assert combined["available"] is True
    assert combined["count"] == record["undefined_reason_counts"]["no_exactly_aligned_observer_covering_both_endpoints"]


def test_finer_subcategories_marked_unavailable_not_fabricated(synthetic_records):
    record = synthetic_records["ALIGNMENT_LIMITED"]
    causes = decompose_undefined_causes(record)
    for name in (
        "multi_window_no_overlapping_alternative", "candidate_observer_origin_unaligned",
        "source_maps_but_destination_outside_shared_overlap", "other_exact_geometry_failure",
    ):
        entry = causes["categories"][name]
        assert entry["available"] is False
        assert entry["count"] is None
    assert set(causes["unavailable_categories"]) == {
        "multi_window_no_overlapping_alternative", "candidate_observer_origin_unaligned",
        "source_maps_but_destination_outside_shared_overlap", "other_exact_geometry_failure",
    }


def test_cause_totals_not_matching_undefined_count_rejected(synthetic_records):
    record = copy.deepcopy(synthetic_records["ALIGNMENT_LIMITED"])
    record["funnel"] = dict(record["funnel"])
    record["funnel"]["edges_with_observer"] += 1  # breaks the exact partition (support_defined no longer matches)
    with pytest.raises(NativeEdgeSupportReachabilityGateError):
        decompose_undefined_causes(record)


def test_single_window_no_alternative_view_exact(synthetic_records):
    record = synthetic_records["ALIGNMENT_LIMITED"]
    causes = decompose_undefined_causes(record)
    assert causes["categories"]["single_window_no_alternative_view"]["count"] == record["undefined_reason_counts"]["single_window_image"]


def test_no_inference_from_pair_level_counts(synthetic_records):
    # Two records with identical undefined_reason_counts but DIFFERENT
    # aligned/unaligned pair counts must produce IDENTICAL cause
    # decomposition -- proving pair counts are never consulted.
    record_a = copy.deepcopy(synthetic_records["ALIGNMENT_LIMITED"])
    record_b = copy.deepcopy(synthetic_records["ALIGNMENT_LIMITED"])
    record_b["funnel"] = dict(record_b["funnel"])
    record_b["funnel"]["aligned_window_pairs"] = 999
    record_b["funnel"]["unaligned_window_pairs"] = 0
    causes_a = decompose_undefined_causes(record_a)
    causes_b = decompose_undefined_causes(record_b)
    assert causes_a["categories"] == causes_b["categories"]


# ---------------------------------------------------------------------------
# Decision matrix / roadmap authorization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome,expected_action,pruning_authorized", [
    ("REACHABLE", "PROCEED_TO_ELIGIBILITY_MATCHED_PRUNING", True),
    ("STRUCTURALLY_UNREACHABLE", "STOP_STRUCTURAL_CONNECTIVITY_BRANCH", False),
    ("ALIGNMENT_LIMITED", "STOP_NATIVE_SUPPORT_ALIGNMENT_LIMITED", False),
    ("INCONCLUSIVE", "DO_NOT_PROCEED_INCONCLUSIVE", False),
])
def test_decision_matrix(gate_identity, synthetic_records, outcome, expected_action, pruning_authorized):
    record = synthetic_records[outcome]
    assert record["decision_output"] == outcome
    roadmap = map_roadmap_authorization(record["decision_output"], gate_identity)
    assert roadmap["roadmap_action"] == expected_action
    assert all(stage["authorized"] == pruning_authorized for stage in roadmap["stages"])


def test_unknown_decision_rejected(gate_identity):
    with pytest.raises(NativeEdgeSupportReachabilityGateError):
        map_roadmap_authorization("SOMETHING_UNKNOWN", gate_identity)


def test_only_reachable_authorizes_pruning(gate_identity, synthetic_records):
    for outcome, record in synthetic_records.items():
        roadmap = map_roadmap_authorization(record["decision_output"], gate_identity)
        any_authorized = any(stage["authorized"] for stage in roadmap["stages"])
        assert any_authorized == (outcome == "REACHABLE")


def test_four_structural_stages_skipped_for_alignment_limited(gate_identity, synthetic_records):
    roadmap = map_roadmap_authorization("ALIGNMENT_LIMITED", gate_identity)
    assert len(roadmap["stages"]) == 4
    assert all(not s["authorized"] for s in roadmap["stages"])
    assert all("preregistered mechanics gate" in s["reason"] for s in roadmap["stages"])


def test_coco_object_next_for_alignment_limited(gate_identity):
    roadmap = map_roadmap_authorization("ALIGNMENT_LIMITED", gate_identity)
    assert roadmap["next_authorized_stage"] == "eval: add COCO-Object protocol confirmation"


def test_skip_reason_never_claims_efficacy_measured(gate_identity):
    roadmap = map_roadmap_authorization("ALIGNMENT_LIMITED", gate_identity)
    for stage in roadmap["stages"]:
        assert "efficacy" not in stage["reason"].lower() or "not because" in stage["reason"].lower()


# ---------------------------------------------------------------------------
# Parent decision reproduction
# ---------------------------------------------------------------------------


def test_reproduce_parent_decision_matches_for_all_synthetic_outcomes(native_audit_identity, synthetic_records):
    for outcome, record in synthetic_records.items():
        repro = reproduce_parent_decision(record, native_audit_identity)
        assert repro["match"] is True
        assert repro["gate_reproduced_decision"] == outcome


def test_reproduce_parent_decision_rejects_tampered_decision_output(native_audit_identity, synthetic_records):
    record = copy.deepcopy(synthetic_records["ALIGNMENT_LIMITED"])
    record["decision_output"] = "REACHABLE"  # tampered, disagrees with what funnel actually implies
    with pytest.raises(NativeEdgeSupportReachabilityGateError):
        reproduce_parent_decision(record, native_audit_identity)


# ---------------------------------------------------------------------------
# Real parent artifact (ALIGNMENT_LIMITED case) -- end-to-end
# ---------------------------------------------------------------------------

real_pytestmark = pytest.mark.skipif(not REAL_PARENT_RESULT.exists(), reason="requires the real mechanics20 GPU result on this machine")


@real_pytestmark
def test_real_parent_artifact_selects_and_validates(gate_identity):
    parent = select_and_validate_parent_artifact(REAL_PARENT_RESULT, repo_root=ROOT, gate_identity=gate_identity)
    assert parent["record"]["decision_output"] == "ALIGNMENT_LIMITED"
    assert parent["record"]["image_count_processed"] == 20


@real_pytestmark
def test_real_parent_artifact_never_modified(gate_identity):
    before_bytes = REAL_PARENT_RESULT.read_bytes()
    before_mtime = REAL_PARENT_RESULT.stat().st_mtime_ns
    select_and_validate_parent_artifact(REAL_PARENT_RESULT, repo_root=ROOT, gate_identity=gate_identity)
    after_bytes = REAL_PARENT_RESULT.read_bytes()
    after_mtime = REAL_PARENT_RESULT.stat().st_mtime_ns
    assert before_bytes == after_bytes
    assert before_mtime == after_mtime


@real_pytestmark
def test_real_parent_full_report_matches_documented_numbers(gate_identity, native_audit_identity):
    import datetime

    gate_identity_sha256 = sha256_file(GATE_IDENTITY_PATH)
    parent = select_and_validate_parent_artifact(REAL_PARENT_RESULT, repo_root=ROOT, gate_identity=gate_identity)
    report = build_gate_report(
        gate_identity=gate_identity, gate_identity_sha256=gate_identity_sha256, parent=parent,
        audit_result_path_reference=str(REAL_PARENT_RESULT),
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    ratios = report["reconstructed_counts_ratios"]
    assert ratios["undefined_edge_fraction"]["value"] == pytest.approx(0.7927, abs=1e-3)
    assert ratios["wrong_row_reachability"]["value"] == pytest.approx(0.2165, abs=1e-3)
    assert report["stop_proceed_decision"] == "STOP_NATIVE_SUPPORT_ALIGNMENT_LIMITED"
    assert report["roadmap_authorization"]["next_authorized_stage"] == "eval: add COCO-Object protocol confirmation"


# ---------------------------------------------------------------------------
# Parent binding negative paths (synthetic, no real GPU artifact needed)
# ---------------------------------------------------------------------------


def test_wrong_image_count_rejected(gate_identity, native_audit_identity, tmp_path):
    record = copy.deepcopy(_valid_minimal_record(native_audit_identity))
    record["image_count_processed"] = 5
    record["image_count_expected"] = 5
    path = tmp_path / "wrong_count.json"
    path.write_text(json.dumps(record))
    with pytest.raises(NativeEdgeSupportReachabilityGateError):
        select_and_validate_parent_artifact(path, repo_root=ROOT, gate_identity=gate_identity)


def test_incomplete_result_rejected(gate_identity, native_audit_identity, tmp_path):
    record = copy.deepcopy(_valid_minimal_record(native_audit_identity))
    record["complete"] = False
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(record))
    with pytest.raises(Exception):
        select_and_validate_parent_artifact(path, repo_root=ROOT, gate_identity=gate_identity)


def test_wrong_run_mode_rejected(gate_identity, native_audit_identity, tmp_path):
    record = copy.deepcopy(_valid_minimal_record(native_audit_identity))
    record["run_mode"] = "pilot100"
    record["schema"] = "talk2dino-native-edge-support-audit-pilot100-result-v1"
    path = tmp_path / "wrong_mode.json"
    path.write_text(json.dumps(record))
    with pytest.raises(Exception):
        select_and_validate_parent_artifact(path, repo_root=ROOT, gate_identity=gate_identity)


def test_malformed_duplicate_key_json_rejected(gate_identity, tmp_path):
    path = tmp_path / "malformed.json"
    path.write_text('{"a": 1, "a": 2}')
    with pytest.raises(Exception):
        select_and_validate_parent_artifact(path, repo_root=ROOT, gate_identity=gate_identity)


def test_nonexistent_path_rejected(gate_identity):
    with pytest.raises(NativeEdgeSupportReachabilityGateError):
        select_and_validate_parent_artifact(Path("/tmp/does-not-exist-12345.json"), repo_root=ROOT, gate_identity=gate_identity)


def _valid_minimal_record(native_audit_identity):
    return _make_synthetic_record(
        native_audit_identity, edges_with_observer=1000, misclassified_rows=25,
        misclassified_rows_any_defined=5, correct_rows_any_defined=5, single_window_edges=100,
    )
