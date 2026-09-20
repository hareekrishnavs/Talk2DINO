"""Tests for the E25 T64-early-exit-benchmark identities
(`src.dfp_phase1b_t64_early_exit_benchmark_identity`) -- the scientific/
operational contract (`e25_dfp_phase1b_t64_early_exit_benchmark.toml`)
and the closed artifact schemas
(`e25_dfp_phase1b_t64_early_exit_benchmark_artifacts.toml`).

No torch/CUDA/model/dataset access anywhere in this module. No
GPU job, no inference, no real benchmark execution -- pure TOML/
filesystem/hash validation and pure arithmetic validation-helper
tests only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

import pytest

import src.dfp_phase1b_t64_early_exit_benchmark_identity as e25

REPO_ROOT = Path(__file__).resolve().parent.parent
E = e25.DfpPhase1bT64EarlyExitBenchmarkIdentityError
RAW_E25_TEXT = (REPO_ROOT / e25.IDENTITY_RELATIVE_PATH).read_text()
RAW_ARTIFACTS_TEXT = (REPO_ROOT / e25.ARTIFACTS_IDENTITY_RELATIVE_PATH).read_text()


# ---------------------------------------------------------------------
# Baseline: the real committed identities load cleanly.
# ---------------------------------------------------------------------


def test_e25_identity_loads_and_verifies():
    doc = e25.load_identity()
    assert doc["benchmark_design"]["variants"] == ["T64_early_exit", "T320_reference"]
    assert doc["population_binding"]["measured_window_count"] == 24
    assert doc["decision_rule"]["end_to_end_speedup_threshold"] == 1.25
    assert doc["scientific_motivation"]["t64_minus_t320_development_miou_pp"] == -0.074828045106592


def test_e25_artifacts_identity_loads_and_verifies():
    doc = e25.load_artifacts_identity()
    assert doc["trial_schema"]["schema_name"] == "talk2dino-dfp-phase1b-t64-early-exit-benchmark-trial-v1"
    assert doc["result_schema"]["schema_name"] == "talk2dino-dfp-phase1b-t64-early-exit-benchmark-result-v1"
    assert doc["decision_schema"]["schema_name"] == "talk2dino-dfp-phase1b-t64-early-exit-decision-v1"


def test_e25_repository_gate_accepts_exact_and_descendant_commit():
    e25.verify_repository_gate(REPO_ROOT, e25.REQUIRED_ANCESTOR_COMMIT)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    e25.verify_repository_gate(REPO_ROOT, head)


def test_e25_repository_gate_rejects_unrelated_commit():
    with pytest.raises(E):
        e25.verify_repository_gate(REPO_ROOT, "0" * 40)


def test_e25_toml_parses_with_tomllib_directly_and_matches_loader():
    raw = tomllib.load(open(REPO_ROOT / e25.IDENTITY_RELATIVE_PATH, "rb"))
    loaded = e25.load_identity()
    assert raw == loaded


def test_no_gt_annotation_field_anywhere_in_e25():
    doc = e25.load_identity()
    text = str(doc)
    for needle in ("gt_seg_map", "ground_truth_label", "annotation_mask"):
        assert needle not in text


def test_no_adaptive_threshold_field_anywhere_in_e25():
    doc = e25.load_identity()
    assert doc["decision_rule"]["adaptive_threshold_authorized"] is False
    assert "adaptive" not in str(doc["decision_rule"]["end_to_end_speedup_threshold"])


def test_no_exact_masked_forward_dependency_bound():
    doc = e25.load_identity()
    for binding in doc["dependency_bindings"].values():
        if isinstance(binding, dict) and "relative_path" in binding:
            assert "masked_forward" not in binding["relative_path"]


# ---------------------------------------------------------------------
# Dependency-hash recomputation (independent of the loader's own
# internal recomputation -- re-derives every hash from scratch here).
# ---------------------------------------------------------------------


def test_dependency_hashes_recompute_independently():
    import hashlib

    doc = e25.load_identity()
    db = doc["dependency_bindings"]
    for name in (
        "e13_protocol", "matched_evaluator", "subset_manifest", "e17_identity", "e17_identity_loader_module",
        "e22_preservation", "e22_artifacts", "e23", "e24", "graph_module", "finite_step_regime_module",
        "recurrence_capture_module", "horizon_stitching_module", "stitching_pullback_module", "final_rescale_module",
        "identity_module", "production_provider_module", "artifact_schema_module",
    ):
        binding = db[name]
        actual = hashlib.sha256((REPO_ROOT / binding["relative_path"]).read_bytes()).hexdigest()
        assert actual == binding["sha256"], f"{name} hash mismatch"

    manifest_binding = db["e17_manifest"]
    actual_manifest = hashlib.sha256((REPO_ROOT / manifest_binding["relative_path"]).read_bytes()).hexdigest()
    assert actual_manifest == manifest_binding["sha256"]
    assert manifest_binding["window_count"] == 24


def test_artifacts_identity_protocol_binding_recomputes():
    import hashlib

    adoc = e25.load_artifacts_identity()
    actual = hashlib.sha256((REPO_ROOT / e25.IDENTITY_RELATIVE_PATH).read_bytes()).hexdigest()
    assert adoc["protocol_binding"]["protocol_identity_sha256"] == actual


# ---------------------------------------------------------------------
# Variant / horizon exactness.
# ---------------------------------------------------------------------


def test_exact_variant_set():
    doc = e25.load_identity()
    assert doc["benchmark_design"]["variants"] == ["T64_early_exit", "T320_reference"]
    assert doc["benchmark_design"]["variant_count"] == 2


def test_exact_horizons_64_and_320():
    doc = e25.load_identity()
    assert doc["benchmark_design"]["t64_early_exit"]["maximum_recurrence_step"] == 64
    assert doc["benchmark_design"]["t64_early_exit"]["output_step"] == 64
    assert doc["benchmark_design"]["t320_reference"]["maximum_recurrence_step"] == 320
    assert doc["benchmark_design"]["t320_reference"]["output_step"] == 320


def test_t64_forbidden_steps_65_to_320_declared():
    doc = e25.load_identity()
    assert "65" in doc["benchmark_design"]["t64_early_exit"]["forbidden_steps"]


def test_recurrence_mechanism_forbids_full_320_step_snapshot_for_t64():
    doc = e25.load_identity()
    contract = doc["recurrence_execution_contract"]
    assert contract["t64_call_steps"] == 64
    assert contract["t320_call_steps"] == 320
    assert contract["capture_dfp_phase1a_trajectory_usable_for_t64"] is False
    assert contract["capture_dfp_phase1a_trajectory_usable_for_t320"] is True
    assert contract["snapshot_of_a_320_step_run_forbidden_for_t64"] is True


def test_finite_step_regime_module_genuinely_supports_early_termination():
    """Independent, direct confirmation (not trusting E25's own prose)
    that the bound finite_step_regime_module's own committed source
    genuinely stops its recurrence loop at the requested `steps` value
    -- never running past it regardless of `snapshot_steps`."""
    text = (REPO_ROOT / "src/open_vocabulary_segmentation/models/dinotext/cover_dr/finite_step_regime.py").read_text()
    assert "for step in range(1, steps + 1)" in text
    assert "steps = _require_exact_int(steps" in text


def test_recurrence_capture_module_always_runs_full_320_never_usable_for_t64():
    """Independent, direct confirmation that capture_dfp_phase1a_trajectory
    cannot produce a genuine T64 early exit: it always reads
    `propagation.steps` (frozen at 320) and rejects any
    `requested_horizons` not exactly equal to the frozen [0,16,32,64,320]
    list -- it has no shortened-steps code path at all."""
    text = (REPO_ROOT / "src/dfp_phase1a_recurrence_capture.py").read_text()
    assert "requested_tuple != contract.horizons" in text
    assert "dfp_recurrence_contract_from_protocol" in text


# ---------------------------------------------------------------------
# Fixed population binding.
# ---------------------------------------------------------------------


def test_fixed_population_binding_exact():
    doc = e25.load_identity()
    pop = doc["population_binding"]
    assert pop["measured_window_count"] == 24
    assert pop["warmup_record_count"] == 1
    assert pop["gt_annotation_field_present_in_manifest"] is False
    assert pop["manifest_sha256"] == e25.MANIFEST_SHA256


def test_manifest_independently_self_verifies_via_committed_chunk_benchmark_verifier():
    import json

    import src.dfp_phase1b_chunk_benchmark as cb

    manifest_doc = json.loads((REPO_ROOT / "evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json").read_bytes())
    cb.verify_benchmark_manifest(manifest_doc)  # must not raise
    assert len(manifest_doc["benchmark_records"]) == 24
    for record in manifest_doc["benchmark_records"]:
        assert not any("gt" in k.lower() or "label" in k.lower() or "annotation" in k.lower() for k in record.keys())


# ---------------------------------------------------------------------
# Execution schedule.
# ---------------------------------------------------------------------


def test_exact_schedule_counterbalanced_order():
    doc = e25.load_identity()
    ordered = doc["execution_schedule"]["counterbalanced_order"]["ordered_trial_list"]
    expected = [
        {"trial_index": 0, "variant": "T320_reference", "repeat": 0},
        {"trial_index": 1, "variant": "T64_early_exit", "repeat": 0},
        {"trial_index": 2, "variant": "T64_early_exit", "repeat": 1},
        {"trial_index": 3, "variant": "T320_reference", "repeat": 1},
        {"trial_index": 4, "variant": "T320_reference", "repeat": 2},
        {"trial_index": 5, "variant": "T64_early_exit", "repeat": 2},
    ]
    assert ordered == expected


def test_exact_repeat_count_and_warmup_count():
    doc = e25.load_identity()
    assert doc["execution_schedule"]["warmup"]["t64_early_exit"] == 1
    assert doc["execution_schedule"]["warmup"]["t320_reference"] == 1
    assert doc["execution_schedule"]["measured_repeats"]["t64_early_exit"] == 3
    assert doc["execution_schedule"]["measured_repeats"]["t320_reference"] == 3
    assert doc["execution_schedule"]["measured_repeats"]["total_measured_trials"] == 6


def test_warmups_excluded_from_summary_statistics_flag():
    doc = e25.load_identity()
    assert doc["execution_schedule"]["warmup"]["excluded_from_summary_statistics"] is True
    assert doc["scientific_output_contract"]["warmup_excluded_from_performance_summaries"] is True


def test_gpu_resource_ceiling_exact():
    doc = e25.load_identity()
    sched = doc["execution_schedule"]
    assert sched["required_device_type"] == "cuda"
    assert sched["required_gpu_model"] == "h100_80gb"
    assert sched["gpu_count"] == 1
    assert sched["cpus_per_task"] == 8
    assert sched["host_memory"] == "80G"
    assert sched["max_wall_time"] == "02:00:00"


# ---------------------------------------------------------------------
# Authorization boundary / decision rule closed set.
# ---------------------------------------------------------------------


def test_authorization_boundary_all_false():
    doc = e25.load_identity()
    boundary = doc["authorization_boundary"]
    assert all(value is False for value in boundary.values())
    assert len(boundary) == 10


def test_no_confirmation_or_full_5000_authorization():
    doc = e25.load_identity()
    assert doc["authorization_boundary"]["confirmation_evaluation_authorized"] is False
    assert doc["authorization_boundary"]["full_5000_image_evaluation_authorized"] is False
    assert doc["decision_rule"]["authorizes_confirmation_evaluation_directly"] is False
    assert doc["decision_rule"]["authorizes_full_5000_image_evaluation_directly"] is False


def test_decision_outcomes_exact_two_element_closed_set():
    doc = e25.load_identity()
    assert doc["decision_rule"]["outcomes"] == ["PROCEED_TO_FREEZE_T64_CONFIRMATION_PROTOCOL", "STOP_T64_EARLY_EXIT_TRACK"]
    assert doc["decision_rule"]["outcome_set_closed"] is True


def test_speedup_threshold_exact_1_25():
    doc = e25.load_identity()
    assert doc["decision_rule"]["end_to_end_speedup_threshold"] == 1.25
    assert doc["decision_rule"]["threshold_comparison_operator"] == "greater_than_or_equal"


def test_five_thousand_projection_forbidden_as_measured_fact():
    doc = e25.load_identity()
    assert doc["summary_statistics"]["five_thousand_image_runtime_projection_from_24_window_benchmark_as_measured_fact_forbidden"] is True
    assert doc["summary_statistics"]["projection_rules"]["historical_approximately_1h18m_baseline_usable_as_e25_performance_baseline"] is False


# ---------------------------------------------------------------------
# Pure validation-helper functions: timing accounting identity.
# ---------------------------------------------------------------------


def _valid_timing():
    return {
        "process_total_wall_ns": 1000, "preflight_wall_ns": 100, "dataset_construction_wall_ns": 100,
        "model_construction_and_checkpoint_loading_wall_ns": 200, "measured_window_loop_wall_ns": 500,
        "artifact_validation_and_write_wall_ns": 50, "unattributed_overhead_ns": 50,
    }


def test_verify_timing_accounting_identity_accepts_valid():
    e25.verify_timing_accounting_identity(_valid_timing())  # must not raise


def test_verify_timing_accounting_identity_rejects_wrong_total():
    bad = dict(_valid_timing())
    bad["process_total_wall_ns"] = 999
    with pytest.raises(E, match="accounting identity"):
        e25.verify_timing_accounting_identity(bad)


def test_verify_timing_accounting_identity_rejects_bool():
    bad = dict(_valid_timing())
    bad["preflight_wall_ns"] = True
    with pytest.raises(E, match="exact int"):
        e25.verify_timing_accounting_identity(bad)


def test_verify_timing_accounting_identity_rejects_negative():
    bad = dict(_valid_timing())
    bad["preflight_wall_ns"] = -1
    with pytest.raises(E, match="non-negative"):
        e25.verify_timing_accounting_identity(bad)


def test_verify_timing_accounting_identity_rejects_missing_field():
    bad = dict(_valid_timing())
    del bad["unattributed_overhead_ns"]
    with pytest.raises(E, match="missing"):
        e25.verify_timing_accounting_identity(bad)


# ---------------------------------------------------------------------
# Pure validation-helper functions: count relations.
# ---------------------------------------------------------------------


def test_verify_count_relations_t64_exact():
    e25.verify_count_relations({"window_count": 24, "recurrence_steps_executed": 1536, "model_forward_count": 24, "graph_construction_count": 24}, variant_id="T64_early_exit")


def test_verify_count_relations_t320_exact():
    e25.verify_count_relations({"window_count": 24, "recurrence_steps_executed": 7680, "model_forward_count": 24, "graph_construction_count": 24}, variant_id="T320_reference")


def test_verify_count_relations_rejects_wrong_recurrence_steps():
    with pytest.raises(E, match="recurrence_steps_executed"):
        e25.verify_count_relations({"window_count": 24, "recurrence_steps_executed": 7680, "model_forward_count": 24, "graph_construction_count": 24}, variant_id="T64_early_exit")


def test_verify_count_relations_rejects_wrong_forward_count():
    with pytest.raises(E, match="model_forward_count"):
        e25.verify_count_relations({"window_count": 24, "recurrence_steps_executed": 1536, "model_forward_count": 23, "graph_construction_count": 24}, variant_id="T64_early_exit")


def test_verify_count_relations_rejects_wrong_graph_count():
    with pytest.raises(E, match="graph_construction_count"):
        e25.verify_count_relations({"window_count": 24, "recurrence_steps_executed": 1536, "model_forward_count": 24, "graph_construction_count": 25}, variant_id="T64_early_exit")


def test_verify_count_relations_rejects_wrong_window_count():
    with pytest.raises(E, match="window_count"):
        e25.verify_count_relations({"window_count": 23, "recurrence_steps_executed": 1472, "model_forward_count": 23, "graph_construction_count": 23}, variant_id="T64_early_exit")


def test_verify_count_relations_rejects_unknown_variant():
    with pytest.raises(E, match="variant_id"):
        e25.verify_count_relations({"window_count": 24, "recurrence_steps_executed": 1536, "model_forward_count": 24, "graph_construction_count": 24}, variant_id="T128_something")


# ---------------------------------------------------------------------
# Pure validation-helper functions: memory relations.
# ---------------------------------------------------------------------


def _valid_memory():
    return {"host_peak_rss_bytes": 1000, "gpu_peak_memory_allocated_bytes": 500, "gpu_peak_memory_reserved_bytes": 800, "gpu_total_memory_bytes": 80_000_000_000}


def test_verify_memory_relations_accepts_valid():
    e25.verify_memory_relations(_valid_memory())


def test_verify_memory_relations_rejects_allocated_exceeds_reserved():
    bad = dict(_valid_memory())
    bad["gpu_peak_memory_allocated_bytes"] = 900
    with pytest.raises(E, match="allocated"):
        e25.verify_memory_relations(bad)


def test_verify_memory_relations_rejects_reserved_exceeds_total():
    bad = dict(_valid_memory())
    bad["gpu_peak_memory_reserved_bytes"] = 90_000_000_000
    with pytest.raises(E, match="reserved"):
        e25.verify_memory_relations(bad)


def test_verify_memory_relations_rejects_negative():
    bad = dict(_valid_memory())
    bad["host_peak_rss_bytes"] = -5
    with pytest.raises(E, match="non-negative"):
        e25.verify_memory_relations(bad)


def test_verify_memory_relations_rejects_bool():
    bad = dict(_valid_memory())
    bad["host_peak_rss_bytes"] = True
    with pytest.raises(E, match="exact int"):
        e25.verify_memory_relations(bad)


# ---------------------------------------------------------------------
# Pure validation-helper functions: scientific-output finiteness /
# within-variant digest equality.
# ---------------------------------------------------------------------


def test_verify_scientific_output_finiteness_accepts_valid():
    e25.verify_scientific_output_finiteness({"finite_value_count": 100, "non_finite_value_count": 0})


def test_verify_scientific_output_finiteness_rejects_nonzero_nonfinite():
    with pytest.raises(E, match="non_finite_value_count"):
        e25.verify_scientific_output_finiteness({"finite_value_count": 100, "non_finite_value_count": 1})


def test_verify_within_variant_digest_equality_accepts_identical():
    d = "a" * 64
    e25.verify_within_variant_digest_equality([d, d, d], variant_id="T64_early_exit")


def test_verify_within_variant_digest_equality_rejects_mismatch():
    with pytest.raises(E, match="not all identical"):
        e25.verify_within_variant_digest_equality(["a" * 64, "a" * 64, "b" * 64], variant_id="T64_early_exit")


def test_verify_within_variant_digest_equality_cross_variant_not_required():
    """T64 and T320 are NOT required to share a digest -- this is a
    per-variant check only, never a cross-variant one."""
    t64_digest, t320_digest = "a" * 64, "b" * 64
    e25.verify_within_variant_digest_equality([t64_digest] * 3, variant_id="T64_early_exit")
    e25.verify_within_variant_digest_equality([t320_digest] * 3, variant_id="T320_reference")


def test_verify_within_variant_digest_equality_rejects_wrong_count():
    with pytest.raises(E, match="expected exactly"):
        e25.verify_within_variant_digest_equality(["a" * 64, "a" * 64], variant_id="T64_early_exit")


# ---------------------------------------------------------------------
# Pure validation-helper functions: exact median / exact speedup.
# ---------------------------------------------------------------------


def test_exact_median_of_three_middle_value():
    assert e25.exact_median_of_three([3.0, 1.0, 2.0]) == 2.0
    assert e25.exact_median_of_three([10, 10, 10]) == 10


def test_exact_median_of_three_rejects_wrong_length():
    with pytest.raises(E, match="exactly 3"):
        e25.exact_median_of_three([1.0, 2.0])


def test_exact_median_of_three_rejects_bool():
    with pytest.raises(E, match="not bool"):
        e25.exact_median_of_three([1.0, True, 2.0])


def test_verify_declared_median_accepts_exact():
    e25.verify_declared_median([3.0, 1.0, 2.0], 2.0)


def test_verify_declared_median_rejects_wrong_value():
    with pytest.raises(E, match="disagrees"):
        e25.verify_declared_median([3.0, 1.0, 2.0], 1.5)


def test_verify_declared_speedup_exact():
    e25.verify_declared_speedup(t320_median_seconds=10.0, t64_median_seconds=5.0, declared_speedup=2.0)


def test_verify_declared_speedup_rejects_wrong_value():
    with pytest.raises(E, match="disagrees"):
        e25.verify_declared_speedup(t320_median_seconds=10.0, t64_median_seconds=5.0, declared_speedup=1.9)


def test_verify_declared_speedup_rejects_non_positive_input():
    with pytest.raises(E, match="positive"):
        e25.verify_declared_speedup(t320_median_seconds=0.0, t64_median_seconds=5.0, declared_speedup=0.0)


# ---------------------------------------------------------------------
# Pure decision-rule application: threshold immediately below/at/above
# 1.25x, and every other required gate.
# ---------------------------------------------------------------------


def _decision_kwargs(**overrides):
    base = dict(
        all_trials_complete=True, all_artifacts_verified=True, within_variant_digests_identical=True,
        zero_non_finite_outputs=True, exact_counts_verified=True, no_oom=True, memory_relations_pass=True,
        t64_median_seconds=5.0, t320_median_seconds=10.0, end_to_end_speedup=2.0,
        development_accuracy_disclosure_pp=-0.074828045106592,
    )
    base.update(overrides)
    return base


def test_decision_rule_proceed_when_all_gates_pass():
    assert e25.apply_decision_rule(**_decision_kwargs()) == "PROCEED_TO_FREEZE_T64_CONFIRMATION_PROTOCOL"


def test_decision_rule_threshold_immediately_below_1_25_stops():
    kwargs = _decision_kwargs(end_to_end_speedup=1.24999999)
    assert e25.apply_decision_rule(**kwargs) == "STOP_T64_EARLY_EXIT_TRACK"


def test_decision_rule_threshold_exactly_1_25_proceeds():
    kwargs = _decision_kwargs(end_to_end_speedup=1.25)
    assert e25.apply_decision_rule(**kwargs) == "PROCEED_TO_FREEZE_T64_CONFIRMATION_PROTOCOL"


def test_decision_rule_threshold_immediately_above_1_25_proceeds():
    kwargs = _decision_kwargs(end_to_end_speedup=1.25000001)
    assert e25.apply_decision_rule(**kwargs) == "PROCEED_TO_FREEZE_T64_CONFIRMATION_PROTOCOL"


@pytest.mark.parametrize(
    "override,label",
    [
        ({"all_trials_complete": False}, "incomplete trials"),
        ({"all_artifacts_verified": False}, "unverified artifacts"),
        ({"within_variant_digests_identical": False}, "digest mismatch"),
        ({"zero_non_finite_outputs": False}, "non-finite outputs"),
        ({"exact_counts_verified": False}, "count mismatch"),
        ({"no_oom": False}, "OOM occurred"),
        ({"memory_relations_pass": False}, "memory relation failure"),
        ({"t64_median_seconds": 12.0}, "T64 not lower than T320"),
    ],
)
def test_decision_rule_stops_when_any_single_gate_fails(override, label):
    kwargs = _decision_kwargs(**override)
    assert e25.apply_decision_rule(**kwargs) == "STOP_T64_EARLY_EXIT_TRACK", label


def test_decision_rule_rejects_tampered_development_accuracy_disclosure():
    with pytest.raises(E, match="development_accuracy_disclosure_pp"):
        e25.apply_decision_rule(**_decision_kwargs(development_accuracy_disclosure_pp=0.5))


def test_decision_rule_rejects_non_bool_gate():
    with pytest.raises(E, match="must be a bool"):
        e25.apply_decision_rule(**_decision_kwargs(all_trials_complete=1))


def test_decision_rule_exact_two_outcome_closed_set():
    for outcome in (
        e25.apply_decision_rule(**_decision_kwargs()),
        e25.apply_decision_rule(**_decision_kwargs(no_oom=False)),
    ):
        assert outcome in e25.DECISION_OUTCOMES
    assert len(e25.DECISION_OUTCOMES) == 2


# ---------------------------------------------------------------------
# Adversarial mutation battery against the real E25 TOML (line-anchored
# string replacement -- never colliding with narrative comment text).
# ---------------------------------------------------------------------


def _mutate_and_expect_reject(old: str, new: str, label: str, *, artifacts: bool = False):
    raw = RAW_ARTIFACTS_TEXT if artifacts else RAW_E25_TEXT
    assert old in raw, f"construction error: {old!r} not found in {'artifacts' if artifacts else 'main'} text"
    mutated = raw.replace(old, new, 1)
    assert mutated != raw, f"construction error (no-op replace): {label}"
    tmp = Path(tempfile.mkdtemp(prefix="v_e25_adv_")) / "mutated.toml"
    tmp.write_text(mutated)
    try:
        if artifacts:
            with pytest.raises(E):
                e25.load_artifacts_identity(path=tmp, repo_root=REPO_ROOT)
        else:
            with pytest.raises(E):
                e25.load_identity(path=tmp, repo_root=REPO_ROOT)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def test_reject_unknown_top_level_key():
    _mutate_and_expect_reject("\n[prohibited]\n", "\n[unexpected_extra_table]\nfoo = 1\n\n[prohibited]\n", "unknown top-level key")


def test_reject_missing_required_key():
    _mutate_and_expect_reject('\nmutates_or_version_bumps_earlier_identity = false\n', "\n", "missing identity key")


def test_reject_wrong_variant_set():
    _mutate_and_expect_reject('variants = ["T64_early_exit", "T320_reference"]', 'variants = ["T64_early_exit", "T320_reference", "T128_extra"]', "extra variant")


def test_reject_wrong_t64_horizon():
    _mutate_and_expect_reject(
        "[benchmark_design.t64_early_exit]\nrecurrence_start = \"canonical S0, identical to T320_reference\"\ngraph = \"canonical k12 directed top-k graph, identical to T320_reference\"\nk = 12\nalpha = 0.98\naffinity_function = \"relu_cosine_power\"\naffinity_power = 3.0\nmaximum_recurrence_step = 64\noutput_step = 64",
        "[benchmark_design.t64_early_exit]\nrecurrence_start = \"canonical S0, identical to T320_reference\"\ngraph = \"canonical k12 directed top-k graph, identical to T320_reference\"\nk = 12\nalpha = 0.98\naffinity_function = \"relu_cosine_power\"\naffinity_power = 3.0\nmaximum_recurrence_step = 96\noutput_step = 96",
        "wrong T64 horizon (96 instead of 64)",
    )


def test_reject_wrong_schedule_order():
    _mutate_and_expect_reject(
        '{ trial_index = 0, variant = "T320_reference", repeat = 0 },',
        '{ trial_index = 0, variant = "T64_early_exit", repeat = 0 },',
        "wrong first-trial variant in counterbalanced order",
    )


def test_reject_wrong_repeat_count():
    _mutate_and_expect_reject("\nt64_early_exit = 3\nt320_reference = 3\ntotal_measured_trials = 6\n", "\nt64_early_exit = 4\nt320_reference = 3\ntotal_measured_trials = 6\n", "wrong T64 repeat count")


def test_reject_wrong_manifest_hash():
    _mutate_and_expect_reject(
        'manifest_sha256 = "be2c77466a2c1d64f0f39f17f4d073ce1de92dc526a59e47ae8c3dabb8c644ab"',
        'manifest_sha256 = "0000000000000000000000000000000000000000000000000000000000000a"',
        "wrong manifest hash",
    )


def test_reject_wrong_dependency_hash():
    _mutate_and_expect_reject(
        'sha256 = "227549b03ec6b1262c465a18b7fc67529fa5a0b5deef419c33ec79a80feb2173"',
        'sha256 = "1111111111111111111111111111111111111111111111111111111111111a"',
        "wrong e13_protocol dependency hash",
    )


def test_reject_relaxed_speedup_threshold():
    _mutate_and_expect_reject("\nend_to_end_speedup_threshold = 1.25\n", "\nend_to_end_speedup_threshold = 1.0\n", "relaxed speedup threshold")


def test_reject_adaptive_threshold_flag():
    _mutate_and_expect_reject("\nadaptive_threshold_authorized = false\n", "\nadaptive_threshold_authorized = true\n", "adaptive threshold authorized")


def test_reject_unauthorized_confirmation_field():
    _mutate_and_expect_reject("\nconfirmation_evaluation_authorized = false\n", "\nconfirmation_evaluation_authorized = true\n", "confirmation authorization injection")


def test_reject_unauthorized_full_5000_field():
    _mutate_and_expect_reject("\nfull_5000_image_evaluation_authorized = false\n", "\nfull_5000_image_evaluation_authorized = true\n", "full-5000 authorization injection")


def test_reject_unauthorized_runner_implementation_field():
    _mutate_and_expect_reject("\nrunner_implementation_authorized_by_e25 = false\n", "\nrunner_implementation_authorized_by_e25 = true\n", "runner-implementation authorization injection")


def test_reject_unauthorized_gt_access_field():
    _mutate_and_expect_reject("\ngt_access_authorized = false\n", "\ngt_access_authorized = true\n", "GT access authorization injection")


def test_reject_unauthorized_masked_forward_field():
    _mutate_and_expect_reject("\nexact_masked_forward_provenance_authorized = false\n", "\nexact_masked_forward_provenance_authorized = true\n", "masked-forward authorization injection")


def test_reject_unauthorized_balp_candidate_evaluation_field():
    _mutate_and_expect_reject("\nbalp_candidate_evaluation_authorized = false\n", "\nbalp_candidate_evaluation_authorized = true\n", "BALP candidate evaluation authorization injection")


def test_reject_bool_as_int_schema_version():
    _mutate_and_expect_reject("\nschema_version = 1\n", "\nschema_version = true\n", "bool-as-int schema_version")


def test_reject_wrong_ancestor_commit():
    _mutate_and_expect_reject(
        'required_ancestor_commit = "453b2d2dec90dcf0623ac94eb177aa5be64db40e"\nworking_branch = "dfp-phase1a-diagnostics"\nsuccessor_of = "e24-dfp-phase1b-balp-development-closure"',
        'required_ancestor_commit = "1111111111111111111111111111111111111a"\nworking_branch = "dfp-phase1a-diagnostics"\nsuccessor_of = "e24-dfp-phase1b-balp-development-closure"',
        "wrong required_ancestor_commit",
    )


def test_reject_removed_prohibited_entry():
    _mutate_and_expect_reject('    "candidate_specific_recurrence",\n', "", "removed candidate_specific_recurrence from prohibited list")


def test_reject_wrong_development_accuracy_disclosure():
    _mutate_and_expect_reject(
        "t64_minus_t320_development_miou_pp = -0.074828045106592",
        "t64_minus_t320_development_miou_pp = 0.5",
        "tampered development accuracy disclosure",
    )


def test_reject_wrong_future_order():
    _mutate_and_expect_reject('"independently_verify_e25",', '"commit_e25",', "reordered future_order first entry")


# --- artifacts-identity mutations ---


def test_artifacts_reject_unknown_schema_key():
    _mutate_and_expect_reject('\n[artifact_write_policy]\n', "\n[unexpected_extra]\nfoo = 1\n\n[artifact_write_policy]\n", "unknown top-level key in artifacts", artifacts=True)


def test_artifacts_reject_wrong_trial_schema_name():
    _mutate_and_expect_reject(
        'schema_name = "talk2dino-dfp-phase1b-t64-early-exit-benchmark-trial-v1"',
        'schema_name = "talk2dino-dfp-phase1b-t64-early-exit-benchmark-trial-v2"',
        "wrong trial schema name",
        artifacts=True,
    )


def test_artifacts_reject_overwrite_authorized_true():
    _mutate_and_expect_reject("\noverwrite_authorized = false\n", "\noverwrite_authorized = true\n", "overwrite authorization injection", artifacts=True)


def test_artifacts_reject_force_mode_authorized_true():
    _mutate_and_expect_reject("\nforce_mode_authorized = false\n", "\nforce_mode_authorized = true\n", "force-mode authorization injection", artifacts=True)


def test_artifacts_reject_tampered_protocol_hash():
    _mutate_and_expect_reject(
        'protocol_identity_sha256 = "8857cf5495b3f15c4ffb0ee772bddac663497663348cf8bd798da7ade7c99b7f"',
        'protocol_identity_sha256 = "0000000000000000000000000000000000000000000000000000000000000a"',
        "tampered protocol_identity_sha256",
        artifacts=True,
    )


def test_artifacts_reject_wrong_decision_outcome_closed_set():
    _mutate_and_expect_reject(
        'closed_set = ["PROCEED_TO_FREEZE_T64_CONFIRMATION_PROTOCOL", "STOP_T64_EARLY_EXIT_TRACK"]',
        'closed_set = ["PROCEED_TO_FREEZE_T64_CONFIRMATION_PROTOCOL", "STOP_T64_EARLY_EXIT_TRACK", "MAYBE_LATER"]',
        "extra decision outcome value",
        artifacts=True,
    )


# ---------------------------------------------------------------------
# Fatal-exception propagation.
# ---------------------------------------------------------------------


def test_fatal_exception_propagates_from_dependency_hash_check(monkeypatch):
    def _boom(*args, **kwargs):
        raise KeyboardInterrupt("simulated fatal")

    monkeypatch.setattr("hashlib.sha256", _boom)
    with pytest.raises(KeyboardInterrupt):
        e25.load_identity()


# ---------------------------------------------------------------------
# No overwrite semantics are exercised at the artifact-write-policy
# schema level (this identity implements no writer of its own).
# ---------------------------------------------------------------------


def test_artifact_write_policy_no_overwrite_no_force():
    doc = e25.load_identity()  # sanity: main identity still loads after prior mutation tests used fresh temp copies only
    adoc = e25.load_artifacts_identity()
    policy = adoc["artifact_write_policy"]
    assert policy["overwrite_authorized"] is False
    assert policy["force_mode_authorized"] is False
    assert policy["atomic_write_required"] is True
    assert policy["reopen_and_public_post_install_reverification_required"] is True


# =======================================================================
# REPAIR_E25_MANIFEST_DIGEST_BINDING regression tests.
#
# Defect: `dependency_bindings.e17_manifest.canonical_manifest_digest`/
# `.benchmark_record_digest` were validated for hex-format only --
# never cross-checked against the manifest's own content or against
# `population_binding`'s copy. Repro (pre-repair): a well-formed but
# wrong value in EITHER field, with `population_binding` left correct,
# was silently accepted.
#
# Repair: `_recompute_e17_manifest_digests` independently recomputes
# both digests from the real manifest's own canonical
# records/warmup/positions using the committed
# `src.dfp_phase1b_chunk_benchmark.canonical_encode`/`digest_over_records`
# primitives (never trusting any stored digest string as an input),
# then checks: manifest-internal fields == recomputed; recomputed ==
# E25's frozen pinned constants (catches coherent tampering);
# dependency_bindings' copy == recomputed; population_binding's copy
# == recomputed. `_verify_dependency_bindings` now returns the
# recomputed pair, threaded by `load_identity` into
# `_verify_population_binding`.
# =======================================================================

_REAL_MANIFEST_REL = "evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json"
_MANIFEST_FIXTURE_FILES = (
    e25.IDENTITY_RELATIVE_PATH.as_posix(),
    "evaluation_identities/e13_dfp_phase1a_protocol.toml",
    "evaluation_identities/e12_matched_k11_k12_t320.toml",
    "evaluation_subsets/dfp_phase1a_coco_stuff_500.json",
    "evaluation_identities/e17_dfp_phase1b_chunk_benchmark.toml",
    "src/dfp_phase1b_chunk_benchmark_identity.py",
    "evaluation_identities/e22_dfp_phase1b_lightweight_boundary_preservation.toml",
    "evaluation_identities/e22_dfp_phase1b_lightweight_boundary_artifacts.toml",
    "evaluation_identities/e23_dfp_phase1b_balp_development_execution.toml",
    "evaluation_identities/e24_dfp_phase1b_balp_development_closure.toml",
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/graph.py",
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/finite_step_regime.py",
    "src/dfp_phase1a_recurrence_capture.py",
    "src/dfp_phase1a_horizon_stitching.py",
    "src/dfp_phase1a_stitching_pullback.py",
    "src/dfp_phase1a_final_rescale_pullback.py",
    "src/dfp_phase1a_identity.py",
    "src/dfp_phase1b_calibration_production.py",
    "src/dfp_phase1a_artifact_schema.py",
    _REAL_MANIFEST_REL,
)

_REAL_CANONICAL_DIGEST = "4b962001ccab863b4c7a156cdbde5f55b66b3ff26a9d429a8b6f148550551847"
_REAL_RECORD_DIGEST = "8193ef03f5d1dee843098ded6ce896f0f5fe072d49f37d2657ad9b6507fec142"
_REAL_MANIFEST_SHA256 = "be2c77466a2c1d64f0f39f17f4d073ce1de92dc526a59e47ae8c3dabb8c644ab"
_WRONG64 = "f" * 64


def _make_manifest_fixture():
    """A minimal REAL-file (never symlinked) fixture: only the exact
    files E25's dependency_bindings reference are copied, so on-disk
    hashes/content can be independently mutated per test without
    touching the real repository, and without the slow full-repo-copy
    cost a naive `shutil.copytree` would incur."""
    fixture = Path(tempfile.mkdtemp(prefix="e25_manifest_fixture_"))
    for rel in _MANIFEST_FIXTURE_FILES:
        dest = fixture / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO_ROOT / rel).read_bytes())
    return fixture


def _fixture_toml_path(fixture: Path) -> Path:
    return fixture / e25.IDENTITY_RELATIVE_PATH


def _fixture_manifest_path(fixture: Path) -> Path:
    return fixture / _REAL_MANIFEST_REL


def _sync_fixture_manifest_file_hash(fixture: Path) -> None:
    """After mutating the fixture's manifest FILE content, updates the
    fixture TOML's `dependency_bindings.e17_manifest.sha256` to match
    the new bytes -- isolates a test to the RELATIONAL digest checks
    (canonical_manifest_digest/benchmark_record_digest) rather than
    tripping the earlier, unrelated file-byte-hash check."""
    new_hash = hashlib.sha256(_fixture_manifest_path(fixture).read_bytes()).hexdigest()
    toml_path = _fixture_toml_path(fixture)
    text = toml_path.read_text()
    updated = text.replace(f'sha256 = "{_REAL_MANIFEST_SHA256}"', f'sha256 = "{new_hash}"', 1)
    assert updated != text, "construction error: manifest sha256 anchor not found in fixture TOML"
    toml_path.write_text(updated)


# --- 1. valid real E25 identity still loads (repair introduces no regression) ---


def test_repair_real_identity_still_loads_cleanly():
    doc = e25.load_identity()
    assert doc["dependency_bindings"]["e17_manifest"]["canonical_manifest_digest"] == _REAL_CANONICAL_DIGEST
    assert doc["dependency_bindings"]["e17_manifest"]["benchmark_record_digest"] == _REAL_RECORD_DIGEST
    assert doc["population_binding"]["canonical_manifest_digest"] == _REAL_CANONICAL_DIGEST
    assert doc["population_binding"]["benchmark_record_digest"] == _REAL_RECORD_DIGEST


# --- 2/3. wrong dependency digest, population correct -> rejected (the original defect) ---


def test_repair_wrong_dependency_canonical_digest_with_correct_population_rejected():
    _mutate_and_expect_reject(
        f'canonical_manifest_digest = "{_REAL_CANONICAL_DIGEST}"',
        f'canonical_manifest_digest = "{_WRONG64}"',
        "wrong dependency_bindings.e17_manifest.canonical_manifest_digest, population_binding left correct",
    )


def test_repair_wrong_dependency_record_digest_with_correct_population_rejected():
    _mutate_and_expect_reject(
        f'benchmark_record_digest = "{_REAL_RECORD_DIGEST}"',
        f'benchmark_record_digest = "{_WRONG64}"',
        "wrong dependency_bindings.e17_manifest.benchmark_record_digest, population_binding left correct",
    )


# --- 4/5. wrong population digest, dependency correct -> rejected ---


def test_repair_wrong_population_canonical_digest_with_correct_dependency_rejected():
    raw = RAW_E25_TEXT
    anchor = (
        f'canonical_manifest_digest = "{_REAL_CANONICAL_DIGEST}"\n'
        f'benchmark_record_digest = "{_REAL_RECORD_DIGEST}"\n'
        'deterministic_order = "benchmark_slot ascending (0..23), identical for every trial and every variant"'
    )
    assert anchor in raw, "construction error: population_binding anchor not found"
    mutated = raw.replace(
        anchor,
        (
            f'canonical_manifest_digest = "{_WRONG64}"\n'
            f'benchmark_record_digest = "{_REAL_RECORD_DIGEST}"\n'
            'deterministic_order = "benchmark_slot ascending (0..23), identical for every trial and every variant"'
        ),
        1,
    )
    assert mutated != raw
    tmp = Path(tempfile.mkdtemp(prefix="e25_pop_digest_")) / "mutated.toml"
    tmp.write_text(mutated)
    try:
        with pytest.raises(E, match="population_binding.canonical_manifest_digest"):
            e25.load_identity(path=tmp, repo_root=REPO_ROOT)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def test_repair_wrong_population_record_digest_with_correct_dependency_rejected():
    raw = RAW_E25_TEXT
    anchor = (
        f'canonical_manifest_digest = "{_REAL_CANONICAL_DIGEST}"\n'
        f'benchmark_record_digest = "{_REAL_RECORD_DIGEST}"\n'
        'deterministic_order = "benchmark_slot ascending (0..23), identical for every trial and every variant"'
    )
    assert anchor in raw
    mutated = raw.replace(
        anchor,
        (
            f'canonical_manifest_digest = "{_REAL_CANONICAL_DIGEST}"\n'
            f'benchmark_record_digest = "{_WRONG64}"\n'
            'deterministic_order = "benchmark_slot ascending (0..23), identical for every trial and every variant"'
        ),
        1,
    )
    assert mutated != raw
    tmp = Path(tempfile.mkdtemp(prefix="e25_pop_digest2_")) / "mutated.toml"
    tmp.write_text(mutated)
    try:
        with pytest.raises(E, match="population_binding.benchmark_record_digest"):
            e25.load_identity(path=tmp, repo_root=REPO_ROOT)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


# --- 6/7. BOTH copies changed to the SAME wrong digest -> still rejected
# (via the frozen-constant pin, since dependency_bindings and
# population_binding would then agree with EACH OTHER but not with the
# real manifest content) ---


def test_repair_both_copies_changed_to_same_wrong_canonical_digest_rejected():
    mutated = RAW_E25_TEXT.replace(_REAL_CANONICAL_DIGEST, _WRONG64)  # replaces ALL occurrences
    assert mutated.count(_WRONG64) >= 2
    tmp = Path(tempfile.mkdtemp(prefix="e25_both_canonical_")) / "mutated.toml"
    tmp.write_text(mutated)
    try:
        with pytest.raises(E):
            e25.load_identity(path=tmp, repo_root=REPO_ROOT)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def test_repair_both_copies_changed_to_same_wrong_record_digest_rejected():
    mutated = RAW_E25_TEXT.replace(_REAL_RECORD_DIGEST, _WRONG64)  # replaces ALL occurrences
    assert mutated.count(_WRONG64) >= 2
    tmp = Path(tempfile.mkdtemp(prefix="e25_both_record_")) / "mutated.toml"
    tmp.write_text(mutated)
    try:
        with pytest.raises(E):
            e25.load_identity(path=tmp, repo_root=REPO_ROOT)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


# --- 8/9. manifest's OWN internal digest field changed, identity copies
# left correct -> rejected (requires a mutated manifest FILE, not just
# a mutated TOML) ---


def test_repair_manifest_internal_canonical_digest_changed_identity_correct_rejected():
    fixture = _make_manifest_fixture()
    try:
        manifest_path = _fixture_manifest_path(fixture)
        doc = json.loads(manifest_path.read_bytes())
        doc["canonical_manifest_digest"] = _WRONG64
        manifest_path.write_text(json.dumps(doc, indent=2, sort_keys=True))
        _sync_fixture_manifest_file_hash(fixture)
        with pytest.raises(E, match="manifest's own internal canonical_manifest_digest"):
            e25.load_identity(repo_root=fixture)
    finally:
        shutil.rmtree(fixture, ignore_errors=True)


def test_repair_manifest_internal_record_digest_changed_identity_correct_rejected():
    fixture = _make_manifest_fixture()
    try:
        manifest_path = _fixture_manifest_path(fixture)
        doc = json.loads(manifest_path.read_bytes())
        doc["benchmark_record_digest"] = _WRONG64
        manifest_path.write_text(json.dumps(doc, indent=2, sort_keys=True))
        _sync_fixture_manifest_file_hash(fixture)
        with pytest.raises(E, match="manifest's own internal benchmark_record_digest"):
            e25.load_identity(repo_root=fixture)
    finally:
        shutil.rmtree(fixture, ignore_errors=True)


# --- 10. manifest CONTENT changed, all stored digest strings retain
# their old (now-stale) values -> rejected ---


def test_repair_manifest_content_changed_stale_digests_retained_rejected():
    fixture = _make_manifest_fixture()
    try:
        manifest_path = _fixture_manifest_path(fixture)
        doc = json.loads(manifest_path.read_bytes())
        doc["benchmark_records"][0] = dict(doc["benchmark_records"][0], query_class_count=doc["benchmark_records"][0]["query_class_count"] + 1)
        # digest fields deliberately left untouched (now stale).
        manifest_path.write_text(json.dumps(doc, indent=2, sort_keys=True))
        _sync_fixture_manifest_file_hash(fixture)
        with pytest.raises(E, match="disagrees with the independently recomputed value"):
            e25.load_identity(repo_root=fixture)
    finally:
        shutil.rmtree(fixture, ignore_errors=True)


# --- 11. manifest content changed AND all stored digest strings
# coherently updated to match the NEW (fabricated) content -> still
# rejected, because the recomputed value now disagrees with E25's
# frozen pinned constant (never merely self-consistency-checked) ---


def test_repair_manifest_content_and_digests_coherently_fabricated_rejected():
    import src.dfp_phase1b_chunk_benchmark as cb

    fixture = _make_manifest_fixture()
    try:
        manifest_path = _fixture_manifest_path(fixture)
        doc = json.loads(manifest_path.read_bytes())
        doc["benchmark_records"][0] = dict(doc["benchmark_records"][0], query_class_count=doc["benchmark_records"][0]["query_class_count"] + 1)
        encoded = [cb.canonical_encode(r) for r in doc["benchmark_records"]]
        doc["benchmark_record_digest"] = cb.digest_over_records(encoded)
        doc["canonical_manifest_digest"] = cb.digest_over_records(
            encoded + [cb.canonical_encode(doc["warmup_record"])] + [cb.canonical_encode({"positions": doc["positions"]})]
        )
        # self-consistent now (manifest-internal recomputation would agree with the
        # tampered content) -- but must still be rejected against E25's frozen pin.
        assert doc["benchmark_record_digest"] != _REAL_RECORD_DIGEST
        assert doc["canonical_manifest_digest"] != _REAL_CANONICAL_DIGEST
        manifest_path.write_text(json.dumps(doc, indent=2, sort_keys=True))
        _sync_fixture_manifest_file_hash(fixture)
        with pytest.raises(E, match="frozen pinned value"):
            e25.load_identity(repo_root=fixture)
    finally:
        shutil.rmtree(fixture, ignore_errors=True)


# --- 12. valid file SHA-256 does not excuse a relational digest mismatch ---


def test_repair_valid_file_hash_does_not_excuse_relational_mismatch():
    """The manifest FILE bytes are completely untouched (file sha256
    still matches) in this mutation -- only the TOML's declared content
    digest is wrong. Must still be rejected."""
    doc = e25.load_identity()
    actual_file_hash = hashlib.sha256((REPO_ROOT / _REAL_MANIFEST_REL).read_bytes()).hexdigest()
    assert actual_file_hash == _REAL_MANIFEST_SHA256, "construction sanity: real manifest file hash unchanged"
    _mutate_and_expect_reject(
        f'canonical_manifest_digest = "{_REAL_CANONICAL_DIGEST}"',
        f'canonical_manifest_digest = "{_WRONG64}"',
        "file hash correct, content digest wrong -- must still be rejected",
    )


# --- 13/14. malformed digest string format (wrong length / non-hex) ---


def test_repair_digest_field_wrong_length_rejected():
    _mutate_and_expect_reject(
        f'canonical_manifest_digest = "{_REAL_CANONICAL_DIGEST}"',
        'canonical_manifest_digest = "abc123"',
        "wrong-length digest string",
    )


def test_repair_digest_field_non_hex_characters_rejected():
    bad = "g" * 64  # 'g' is not a valid hex character
    _mutate_and_expect_reject(
        f'canonical_manifest_digest = "{_REAL_CANONICAL_DIGEST}"',
        f'canonical_manifest_digest = "{bad}"',
        "non-hex-character digest string",
    )


# --- 15. uppercase/mixed-case follows the existing frozen (lowercase-only) convention ---


def test_repair_uppercase_digest_rejected_per_frozen_lowercase_convention():
    upper = _REAL_CANONICAL_DIGEST.upper()
    assert upper != _REAL_CANONICAL_DIGEST
    _mutate_and_expect_reject(
        f'canonical_manifest_digest = "{_REAL_CANONICAL_DIGEST}"',
        f'canonical_manifest_digest = "{upper}"',
        "uppercase-hex digest rejected (lowercase-only frozen convention)",
    )


# --- 16. missing digest field rejected ---


def test_repair_missing_dependency_canonical_digest_field_rejected():
    _mutate_and_expect_reject(
        f'\ncanonical_manifest_digest = "{_REAL_CANONICAL_DIGEST}"\nbenchmark_record_digest = "{_REAL_RECORD_DIGEST}"\nbinds = "the fixed 24 measured windows',
        f'\nbenchmark_record_digest = "{_REAL_RECORD_DIGEST}"\nbinds = "the fixed 24 measured windows',
        "missing dependency_bindings.e17_manifest.canonical_manifest_digest field",
    )


# --- 17. bool / int / null / list / mapping values rejected for a digest field ---


@pytest.mark.parametrize(
    "toml_value,label",
    [
        ("true", "bool"),
        ("12345", "int"),
        ("[1, 2, 3]", "list"),
        ("{ x = 1 }", "inline mapping"),
    ],
)
def test_repair_digest_field_wrong_type_rejected(toml_value, label):
    _mutate_and_expect_reject(
        f'canonical_manifest_digest = "{_REAL_CANONICAL_DIGEST}"',
        f'canonical_manifest_digest = {toml_value}',
        f"{label}-typed value for canonical_manifest_digest rejected",
    )


# --- 18. unknown field in the e17_manifest binding remains rejected ---


def test_repair_unknown_field_in_e17_manifest_binding_rejected():
    _mutate_and_expect_reject(
        f'benchmark_record_digest = "{_REAL_RECORD_DIGEST}"\nbinds = "the fixed 24 measured windows',
        f'benchmark_record_digest = "{_REAL_RECORD_DIGEST}"\nunexpected_extra_field = "x"\nbinds = "the fixed 24 measured windows',
        "unknown field injected into dependency_bindings.e17_manifest",
    )


# --- 19. path traversal / symlink escape remain rejected (structural, re-verified after repair) ---


def test_repair_path_traversal_in_manifest_binding_rejected():
    _mutate_and_expect_reject(
        'relative_path = "evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json"',
        'relative_path = "../../../../etc/passwd"',
        "path traversal in e17_manifest relative_path",
    )


def test_repair_symlink_escape_for_manifest_rejected():
    fixture = _make_manifest_fixture()
    outside_dir = None
    try:
        manifest_link = _fixture_manifest_path(fixture)
        outside_dir = Path(tempfile.mkdtemp(prefix="e25_outside_"))
        outside_file = outside_dir / "escaped.json"
        outside_file.write_bytes((REPO_ROOT / _REAL_MANIFEST_REL).read_bytes())
        manifest_link.unlink()
        manifest_link.symlink_to(outside_file)
        with pytest.raises(E, match="escapes the repository root"):
            e25.load_identity(repo_root=fixture)
    finally:
        shutil.rmtree(fixture, ignore_errors=True)
        if outside_dir:
            shutil.rmtree(outside_dir, ignore_errors=True)


# --- 20. domain exception used for ordinary failures (never a raw
# cb.DfpPhase1bChunkBenchmarkError or other module's exception leaks) ---


def test_repair_ordinary_failure_raises_domain_exception_not_raw_cb_error():
    import src.dfp_phase1b_chunk_benchmark as cb

    fixture = _make_manifest_fixture()
    try:
        manifest_path = _fixture_manifest_path(fixture)
        doc = json.loads(manifest_path.read_bytes())
        doc["benchmark_records"][0] = {"only_one_field": "not a valid canonical record shape at all, but still JSON-serializable"}
        manifest_path.write_text(json.dumps(doc, indent=2, sort_keys=True))
        try:
            e25.load_identity(repo_root=fixture)
            pytest.fail("malformed record content was accepted (bug)")
        except cb.DfpPhase1bChunkBenchmarkError:
            pytest.fail("raw src.dfp_phase1b_chunk_benchmark exception leaked -- must be wrapped in the E25 domain exception")
        except E:
            pass  # correct: wrapped in the E25 domain exception
    finally:
        shutil.rmtree(fixture, ignore_errors=True)


# --- 21. fatal exceptions propagate through the new recomputation path ---


def test_repair_fatal_exception_propagates_through_manifest_recomputation(monkeypatch):
    def _boom(*args, **kwargs):
        raise KeyboardInterrupt("simulated fatal during manifest JSON decode")

    monkeypatch.setattr("json.loads", _boom)
    with pytest.raises(KeyboardInterrupt):
        e25.load_identity()
