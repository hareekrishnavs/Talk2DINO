"""Tests for the E26 T64-early-exit-benchmark IMPLEMENTATION authority
(`src.dfp_phase1b_t64_early_exit_benchmark_implementation_identity`).

E26 authorizes implementation only -- no runner, no CLI, no launcher
exists yet. No torch/CUDA/model/dataset access anywhere in this
module. No GPU job, no inference, no real benchmark execution -- pure
TOML/filesystem/hash validation tests only.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

import pytest

import src.dfp_phase1b_t64_early_exit_benchmark_implementation_identity as e26

REPO_ROOT = Path(__file__).resolve().parent.parent
E = e26.DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError
RAW_TEXT = (REPO_ROOT / e26.IDENTITY_RELATIVE_PATH).read_text()


# ---------------------------------------------------------------------
# Baseline: the real committed identity loads cleanly.
# ---------------------------------------------------------------------


def test_e26_identity_loads_and_verifies():
    doc = e26.load_identity()
    assert doc["authorized_future_files"]["list"] == list(e26.AUTHORIZED_FUTURE_FILES)
    assert doc["runner_subcommands"]["list"] == list(e26.RUNNER_SUBCOMMANDS)
    assert doc["runner_population_contract"]["window_count"] == 24


def test_e26_toml_parses_with_tomllib_directly_and_matches_loader():
    raw = tomllib.load(open(REPO_ROOT / e26.IDENTITY_RELATIVE_PATH, "rb"))
    loaded = e26.load_identity()
    assert raw == loaded


def test_e26_repository_gate_accepts_exact_and_descendant_commit():
    e26.verify_repository_gate(REPO_ROOT, e26.REQUIRED_ANCESTOR_COMMIT)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    e26.verify_repository_gate(REPO_ROOT, head)


def test_e26_repository_gate_rejects_unrelated_commit():
    with pytest.raises(E):
        e26.verify_repository_gate(REPO_ROOT, "0" * 40)


def test_e26_successor_of_e25():
    doc = e26.load_identity()
    assert doc["identity"]["successor_of"] == "e25-dfp-phase1b-t64-early-exit-benchmark"


def test_e26_required_ancestor_is_the_committed_e25_commit():
    doc = e26.load_identity()
    assert doc["identity"]["required_ancestor_commit"] == "99da1be639932b0e34a96f2f32d23ea72e696f0a"


# ---------------------------------------------------------------------
# Dependency-hash recomputation (independent of the loader's own
# internal recomputation).
# ---------------------------------------------------------------------


def test_dependency_hashes_recompute_independently():
    doc = e26.load_identity()
    db = doc["dependency_bindings"]
    assert set(db.keys()) == {
        "e25_identity", "e25_artifacts", "e25_loader", "e25_docs", "e17_manifest", "e13_protocol",
        "matched_evaluator", "graph_module", "finite_step_regime_module", "horizon_stitching_module",
        "stitching_pullback_module", "final_rescale_module", "production_provider_module", "artifact_schema_module",
    }
    assert len(db) == 14
    for name, binding in db.items():
        actual = hashlib.sha256((REPO_ROOT / binding["relative_path"]).read_bytes()).hexdigest()
        assert actual == binding["sha256"], f"{name} hash mismatch"


def test_dependency_paths_are_canonical_and_contained():
    doc = e26.load_identity()
    for name, binding in doc["dependency_bindings"].items():
        rel = binding["relative_path"]
        assert ".." not in Path(rel).parts, f"{name}: relative_path contains '..'"
        resolved = (REPO_ROOT / rel).resolve()
        resolved.relative_to(REPO_ROOT.resolve())


def test_no_circular_self_hash_binding():
    doc = e26.load_identity()
    for name, binding in doc["dependency_bindings"].items():
        assert "e26" not in Path(binding["relative_path"]).name.lower()


def test_e25_loader_binding_matches_e25s_own_committed_bytes():
    import src.dfp_phase1b_t64_early_exit_benchmark_identity as e25

    doc = e26.load_identity()
    actual = hashlib.sha256((REPO_ROOT / "src/dfp_phase1b_t64_early_exit_benchmark_identity.py").read_bytes()).hexdigest()
    assert doc["dependency_bindings"]["e25_loader"]["sha256"] == actual
    # cross-check: loading the real E25 identity succeeds using that exact bound module.
    e25.load_identity()


# ---------------------------------------------------------------------
# Exact authorized future file set / subcommands / variants / horizons.
# ---------------------------------------------------------------------


def test_exact_authorized_future_file_list():
    doc = e26.load_identity()
    assert doc["authorized_future_files"]["list"] == [
        "src/dfp_phase1b_t64_early_exit_benchmark.py",
        "diagnostics/run_dfp_phase1b_t64_early_exit_benchmark.py",
        "tests/test_dfp_phase1b_t64_early_exit_benchmark.py",
        "docs/dfp_phase1b_t64_early_exit_benchmark_run.md",
        "dfp_phase1b_t64_early_exit_benchmark_h100.sbatch",
    ]
    assert doc["authorized_future_files"]["no_other_new_production_file_authorized"] is True
    assert doc["authorized_future_files"]["implemented_in_this_task"] is False


def test_none_of_the_five_authorized_future_files_exist_yet():
    doc = e26.load_identity()
    for rel in doc["authorized_future_files"]["list"]:
        assert not (REPO_ROOT / rel).exists(), f"{rel} must not exist yet -- E26 authorizes implementation, not this task performing it"


def test_exact_runner_subcommands():
    doc = e26.load_identity()
    assert doc["runner_subcommands"]["list"] == [
        "verify-protocol", "preflight", "run-trial", "verify-trial", "merge", "verify-result", "verify-decision",
    ]
    assert doc["runner_subcommands"]["verify_protocol_torch_free_required"] is True
    assert doc["runner_subcommands"]["preflight_torch_free_required"] is True
    assert doc["runner_subcommands"]["verify_protocol_cuda_free_required"] is True
    assert doc["runner_subcommands"]["preflight_cuda_free_required"] is True
    assert doc["runner_subcommands"]["run_trial_invoked_by_this_task"] is False


def test_exact_variants_and_horizons_via_construction_contract():
    doc = e26.load_identity()
    c = doc["runner_construction_contract"]
    assert c["t64_recurrence_steps"] == 64
    assert c["t320_recurrence_steps"] == 320
    assert c["model_forward_count_per_window"] == 1
    assert c["graph_construction_count_per_window"] == 1
    assert c["t64_via_320_step_trajectory_then_slice_forbidden"] is True
    assert c["candidate_specific_recurrence_forbidden"] is True
    assert c["exact_masked_forward_component_provenance_call_forbidden"] is True


def test_exact_population():
    doc = e26.load_identity()
    p = doc["runner_population_contract"]
    assert p["window_count"] == 24
    assert p["manifest_sha256"] == "be2c77466a2c1d64f0f39f17f4d073ce1de92dc526a59e47ae8c3dabb8c644ab"
    assert p["alternate_manifest_rejected"] is True
    assert p["gt_bearing_input_rejected"] is True


def test_exact_schedule():
    doc = e26.load_identity()
    ordered = doc["process_isolation"]["measured_order"]["ordered_trial_list"]
    assert ordered == [
        {"trial_index": 0, "variant": "T320_reference", "repeat": 0},
        {"trial_index": 1, "variant": "T64_early_exit", "repeat": 0},
        {"trial_index": 2, "variant": "T64_early_exit", "repeat": 1},
        {"trial_index": 3, "variant": "T320_reference", "repeat": 1},
        {"trial_index": 4, "variant": "T320_reference", "repeat": 2},
        {"trial_index": 5, "variant": "T64_early_exit", "repeat": 2},
    ]
    assert doc["process_isolation"]["warmup"]["list"] == ["T64_early_exit", "T320_reference"]
    assert doc["process_isolation"]["fresh_process_per_trial_required"] is True
    assert doc["process_isolation"]["no_partial_merged_result_authorized"] is True


def test_exact_resource_contract():
    doc = e26.load_identity()
    r = doc["launcher_resource_contract"]
    assert r["gpu_count"] == 1
    assert r["gpu_model"] == "h100_80gb"
    assert r["cpus_per_task"] == 8
    assert r["host_memory"] == "80G"
    assert r["max_wall_time"] == "02:00:00"
    assert r["account"] == "rrg-yangw"
    assert r["self_submission_forbidden"] is True
    assert r["literal_sbatch_invocation_forbidden"] is True
    assert r["overwrite_authorized"] is False
    assert r["force_mode_authorized"] is False
    assert r["submitted_by_this_task"] is False
    assert set(r["required_environment_variables"]) == {"repository_root", "venv_activate_path", "dataset_root", "output_root"}


def test_exact_timing_and_memory_requirements():
    doc = e26.load_identity()
    t = doc["timing_implementation_contract"]
    assert t["primary_timing_field"] == "measured_window_loop_wall_ns"
    assert t["fields"] == [
        "process_total_wall_ns", "preflight_wall_ns", "dataset_construction_wall_ns",
        "model_construction_and_checkpoint_loading_wall_ns", "measured_window_loop_wall_ns",
        "artifact_validation_and_write_wall_ns", "unattributed_overhead_ns",
    ]
    assert t["accounting_identity_required"] is True
    assert t["stage_totals_must_not_replace_primary_wall_time"] is True

    m = doc["telemetry_implementation_contract"]
    assert m["fields"] == ["host_peak_rss_bytes", "gpu_peak_memory_allocated_bytes", "gpu_peak_memory_reserved_bytes", "gpu_total_memory_bytes"]
    assert m["allocated_le_reserved_le_total_required"] is True
    assert m["missing_or_malformed_cuda_telemetry_fails_closed"] is True


def test_exact_count_relations():
    doc = e26.load_identity()
    c = doc["output_count_verification_contract"]
    assert c["model_forward_count_required_value"] == 24
    assert c["graph_construction_count_required_value"] == 24
    assert c["t64_recurrence_steps_executed_required_value"] == 1536
    assert c["t320_recurrence_steps_executed_required_value"] == 7680
    assert c["cross_variant_digest_equality_required"] is False
    assert c["all_outputs_finite_required"] is True


# ---------------------------------------------------------------------
# Authorization boundary exactness.
# ---------------------------------------------------------------------


def test_authorization_boundary_exact_true_and_false_sets():
    doc = e26.load_identity()
    boundary = doc["authorization_boundary"]
    assert len(boundary) == 16
    true_keys = {"runner_implementation_authorized", "cli_implementation_authorized", "launcher_implementation_authorized", "cpu_only_synthetic_testing_authorized"}
    for key in true_keys:
        assert boundary[key] is True, key
    for key, value in boundary.items():
        if key not in true_keys:
            assert value is False, key


def test_e26_rejects_runner_execution_authorization():
    """E26 authorizes IMPLEMENTATION, never a real GPU trial -- confirm
    this specific flag is false and any attempt to flip it is rejected
    (adversarial mutation, see below)."""
    doc = e26.load_identity()
    assert doc["authorization_boundary"]["real_gpu_trial_authorized"] is False


def test_e26_rejects_confirmation_and_full_5000_authorization():
    doc = e26.load_identity()
    assert doc["authorization_boundary"]["confirmation_execution_authorized"] is False
    assert doc["authorization_boundary"]["full_5000_execution_authorized"] is False


def test_e26_has_no_gt_or_masked_forward_dependency():
    doc = e26.load_identity()
    assert doc["authorization_boundary"]["gt_access_authorized"] is False
    assert doc["authorization_boundary"]["exact_masked_forward_provenance_authorized"] is False
    for binding in doc["dependency_bindings"].values():
        assert "masked_forward" not in binding["relative_path"]


def test_e26_does_not_authorize_e25_modification():
    doc = e26.load_identity()
    assert doc["authorization_boundary"]["e25_modification_authorized"] is False
    assert doc["authorization_boundary"]["scientific_kernel_modification_authorized"] is False


# ---------------------------------------------------------------------
# Adversarial mutation battery.
# ---------------------------------------------------------------------


def _mutate_and_expect_reject(old: str, new: str, label: str):
    assert old in RAW_TEXT, f"construction error: {old!r} not found"
    mutated = RAW_TEXT.replace(old, new, 1)
    assert mutated != RAW_TEXT, f"construction error (no-op): {label}"
    tmp = Path(tempfile.mkdtemp(prefix="e26_adv_")) / "mutated.toml"
    tmp.write_text(mutated)
    try:
        with pytest.raises(E):
            e26.load_identity(path=tmp, repo_root=REPO_ROOT)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def test_reject_unknown_top_level_key():
    _mutate_and_expect_reject("\n[prohibited]\n", "\n[unexpected_extra_table]\nfoo = 1\n\n[prohibited]\n", "unknown top-level key")


def test_reject_missing_identity_key():
    _mutate_and_expect_reject("\nmutates_or_version_bumps_earlier_identity = false\n", "\n", "missing identity key")


def test_reject_wrong_dependency_hash():
    wrong64 = "1" * 64
    _mutate_and_expect_reject(
        'sha256 = "227549b03ec6b1262c465a18b7fc67529fa5a0b5deef419c33ec79a80feb2173"',
        f'sha256 = "{wrong64}"',
        "wrong e13_protocol dependency hash",
    )


def test_reject_wrong_authorized_future_file():
    _mutate_and_expect_reject(
        '"src/dfp_phase1b_t64_early_exit_benchmark.py",',
        '"src/dfp_phase1b_t64_early_exit_benchmark_v2.py",',
        "wrong authorized runner filename",
    )


def test_reject_extra_authorized_future_file():
    _mutate_and_expect_reject(
        '"dfp_phase1b_t64_early_exit_benchmark_h100.sbatch",\n]',
        '"dfp_phase1b_t64_early_exit_benchmark_h100.sbatch",\n    "src/extra_unauthorized_file.py",\n]',
        "extra unauthorized future file injected",
    )


def test_reject_wrong_subcommand_list():
    _mutate_and_expect_reject(
        'list = ["verify-protocol", "preflight", "run-trial", "verify-trial", "merge", "verify-result", "verify-decision"]',
        'list = ["verify-protocol", "preflight", "execute", "run-trial", "verify-trial", "merge", "verify-result", "verify-decision"]',
        "extra/wrong subcommand injected",
    )


def test_reject_wrong_t64_steps():
    _mutate_and_expect_reject("\nt64_recurrence_steps = 64\n", "\nt64_recurrence_steps = 96\n", "wrong T64 step count")


def test_reject_wrong_t320_steps():
    _mutate_and_expect_reject("\nt320_recurrence_steps = 320\n", "\nt320_recurrence_steps = 200\n", "wrong T320 step count")


def test_reject_wrong_window_count():
    _mutate_and_expect_reject("\nwindow_count = 24\n", "\nwindow_count = 23\n", "wrong window count")


def test_reject_wrong_schedule_order():
    _mutate_and_expect_reject(
        '{ trial_index = 0, variant = "T320_reference", repeat = 0 },',
        '{ trial_index = 0, variant = "T64_early_exit", repeat = 0 },',
        "wrong first-trial variant in schedule",
    )


def test_reject_wrong_gpu_model():
    _mutate_and_expect_reject('gpu_model = "h100_80gb"', 'gpu_model = "a100_80gb"', "wrong GPU model")


def test_reject_wrong_wall_time():
    _mutate_and_expect_reject('max_wall_time = "02:00:00"', 'max_wall_time = "05:00:00"', "wrong wall-time limit")


def test_reject_relaxed_overwrite_flag():
    _mutate_and_expect_reject("\noverwrite_authorized = false\n", "\noverwrite_authorized = true\n", "overwrite authorization injection")


def test_reject_relaxed_force_mode_flag():
    _mutate_and_expect_reject("\nforce_mode_authorized = false\n", "\nforce_mode_authorized = true\n", "force-mode authorization injection")


def test_reject_self_submission_flag_flip():
    _mutate_and_expect_reject("\nself_submission_forbidden = true\n", "\nself_submission_forbidden = false\n", "self-submission-forbidden flag flipped")


def test_reject_runner_execution_authorization_injection():
    _mutate_and_expect_reject("\nreal_gpu_trial_authorized = false\n", "\nreal_gpu_trial_authorized = true\n", "real GPU trial authorization injection")


def test_reject_confirmation_authorization_injection():
    _mutate_and_expect_reject("\nconfirmation_execution_authorized = false\n", "\nconfirmation_execution_authorized = true\n", "confirmation authorization injection")


def test_reject_full_5000_authorization_injection():
    _mutate_and_expect_reject("\nfull_5000_execution_authorized = false\n", "\nfull_5000_execution_authorized = true\n", "full-5000 authorization injection")


def test_reject_gt_access_authorization_injection():
    _mutate_and_expect_reject("\ngt_access_authorized = false\n", "\ngt_access_authorized = true\n", "GT access authorization injection")


def test_reject_masked_forward_authorization_injection():
    _mutate_and_expect_reject("\nexact_masked_forward_provenance_authorized = false\n", "\nexact_masked_forward_provenance_authorized = true\n", "masked-forward authorization injection")


def test_reject_e25_modification_authorization_injection():
    _mutate_and_expect_reject("\ne25_modification_authorized = false\n", "\ne25_modification_authorized = true\n", "E25 modification authorization injection")


def test_reject_scientific_kernel_modification_authorization_injection():
    _mutate_and_expect_reject("\nscientific_kernel_modification_authorized = false\n", "\nscientific_kernel_modification_authorized = true\n", "scientific-kernel modification authorization injection")


def test_reject_removed_prohibited_entry():
    _mutate_and_expect_reject('    "candidate_specific_recurrence",\n', "", "removed candidate_specific_recurrence from prohibited list")


def test_reject_bool_as_int_schema_version():
    _mutate_and_expect_reject("\nschema_version = 1\n", "\nschema_version = true\n", "bool-as-int schema_version")


def test_reject_wrong_required_ancestor_commit():
    _mutate_and_expect_reject(
        'required_ancestor_commit = "99da1be639932b0e34a96f2f32d23ea72e696f0a"',
        'required_ancestor_commit = "1111111111111111111111111111111111111a"',
        "wrong required_ancestor_commit",
    )


def test_reject_nan_in_a_numeric_field():
    _mutate_and_expect_reject("\ncpus_per_task = 8\n", "\ncpus_per_task = nan\n", "NaN injected into cpus_per_task")


def test_reject_infinity_in_a_numeric_field():
    _mutate_and_expect_reject("\nmax_wall_time = \"02:00:00\"\naccount = \"rrg-yangw\"\n", "\nmax_wall_time = \"02:00:00\"\naccount = \"rrg-yangw\"\ngpu_count_extra = inf\n", "infinity injected via extra field")


def test_reject_wrong_type_for_bool_field():
    _mutate_and_expect_reject("\nself_submission_forbidden = true\n", "\nself_submission_forbidden = 1\n", "int-as-bool for self_submission_forbidden")


def test_reject_missing_authorization_boundary_field():
    _mutate_and_expect_reject("\ngt_access_authorized = false\n", "\n", "missing gt_access_authorized field")


# ---------------------------------------------------------------------
# Fatal-exception propagation.
# ---------------------------------------------------------------------


def test_fatal_exception_propagates_from_dependency_hash_check(monkeypatch):
    def _boom(*args, **kwargs):
        raise KeyboardInterrupt("simulated fatal")

    monkeypatch.setattr("hashlib.sha256", _boom)
    with pytest.raises(KeyboardInterrupt):
        e26.load_identity()


def test_system_exit_propagates(monkeypatch):
    def _boom(*args, **kwargs):
        raise SystemExit(1)

    monkeypatch.setattr("hashlib.sha256", _boom)
    with pytest.raises(SystemExit):
        e26.load_identity()


def test_memory_error_propagates(monkeypatch):
    def _boom(*args, **kwargs):
        raise MemoryError("simulated")

    monkeypatch.setattr("hashlib.sha256", _boom)
    with pytest.raises(MemoryError):
        e26.load_identity()
