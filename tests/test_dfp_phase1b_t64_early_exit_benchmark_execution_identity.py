"""Tests for the E27 T64-early-exit-benchmark EXECUTION authority
(`src.dfp_phase1b_t64_early_exit_benchmark_execution_identity`).

E27 authorizes future execution only, conditional on a chain of
activation prerequisites this task itself never satisfies. No torch/
CUDA/model/dataset access anywhere in this module -- pure TOML/
filesystem/hash/git validation tests only. Dirty-tree and staged-
change scenarios use a freshly-created, minimal, temporary git
repository -- never the real repository's own working tree or history.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

import pytest

import src.dfp_phase1b_t64_early_exit_benchmark_execution_identity as e27
import src.dfp_phase1b_chunk_benchmark_identity as generic

REPO_ROOT = Path(__file__).resolve().parent.parent
E = e27.DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError
RAW_TEXT = (REPO_ROOT / e27.IDENTITY_RELATIVE_PATH).read_text()


# ---------------------------------------------------------------------
# Baseline: the real committed (well, real on-disk) identity loads
# cleanly. E27 itself is not yet committed at authoring time -- this
# module only ever performs read-only loading/verification.
# ---------------------------------------------------------------------


def test_e27_identity_loads_and_verifies():
    doc = e27.load_identity()
    assert doc["execution_scope"]["population"]["measured_window_count"] == 24
    assert doc["decision_binding"]["end_to_end_speedup_threshold"] == 1.25


def test_e27_toml_parses_with_tomllib_directly_and_matches_loader():
    raw = tomllib.load(open(REPO_ROOT / e27.IDENTITY_RELATIVE_PATH, "rb"))
    loaded = e27.load_identity()
    assert raw == loaded


def test_ordinary_public_loading_never_claims_activation():
    """load_identity() alone -- even called on an uncommitted E27 -- must
    never itself claim execution is activated. It is a pure, read-only
    schema/hash check, strictly distinct from verify_execution_readiness."""
    doc = e27.load_identity()
    assert "activated" not in str(doc).lower()
    # Structural distinction: load_identity's own signature/behavior
    # never touches git state at all.
    import inspect
    sig = inspect.signature(e27.load_identity)
    assert "repository_commit" not in sig.parameters


def test_verify_execution_readiness_is_strictly_separate_and_checks_git_state():
    """Confirms the real, currently-dirty working tree (E27's own 4 new
    files are uncommitted at authoring time) causes
    verify_execution_readiness to fail closed -- proving it performs a
    STRICTER check than mere ordinary loading. This is expected,
    correct behavior, not a defect: it demonstrates the readiness gate
    actually functions."""
    e27.load_identity()  # ordinary loading succeeds regardless
    with pytest.raises(E):
        e27.verify_execution_readiness()


def test_verify_execution_readiness_is_torch_free():
    import sys
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; sys.path.insert(0, '.')\n"
            "import src.dfp_phase1b_t64_early_exit_benchmark_execution_identity as e27\n"
            "try:\n"
            "    e27.verify_execution_readiness()\n"
            "except e27.DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError:\n"
            "    pass\n"
            "print('torch' in sys.modules)"
        )],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_module_import_is_torch_free():
    import sys
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, '.'); import src.dfp_phase1b_t64_early_exit_benchmark_execution_identity; print('torch' in sys.modules)"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


# ---------------------------------------------------------------------
# Identity block.
# ---------------------------------------------------------------------


def test_e27_successor_of_e26():
    doc = e27.load_identity()
    assert doc["identity"]["successor_of"] == "e26-dfp-phase1b-t64-early-exit-benchmark-implementation"


def test_e27_required_ancestor_is_the_committed_implementation_commit():
    doc = e27.load_identity()
    assert doc["identity"]["required_ancestor_commit"] == "af92d4acc64254674d2156a13b55bf4f5d368d3d"
    assert doc["identity"]["required_ancestor_commit"] == e27.REQUIRED_ANCESTOR_COMMIT


def test_e27_narrowly_supersedes_execution_restriction_only():
    doc = e27.load_identity()
    assert doc["identity"]["supersedes_predecessor_execution_restriction"] is True
    assert doc["identity"]["mutates_or_version_bumps_earlier_identity"] is False


def test_e27_does_not_pin_its_own_future_commit_no_circular_self_hash():
    doc = e27.load_identity()
    for name, binding in doc["dependency_bindings"].items():
        assert "e27" not in Path(binding["relative_path"]).name.lower(), f"circular self-reference in {name}"


# ---------------------------------------------------------------------
# Dependency bindings: independent recompute, canonical/contained
# paths, tampered bytes, path traversal, symlink escape.
# ---------------------------------------------------------------------


def test_dependency_hashes_recompute_independently():
    import hashlib
    doc = e27.load_identity()
    for name, binding in doc["dependency_bindings"].items():
        target = REPO_ROOT / binding["relative_path"]
        assert hashlib.sha256(target.read_bytes()).hexdigest() == binding["sha256"], f"{name} hash mismatch"


def test_dependency_paths_are_canonical_and_contained():
    doc = e27.load_identity()
    for name, binding in doc["dependency_bindings"].items():
        rel = binding["relative_path"]
        assert not rel.startswith("/"), f"{name}: absolute path"
        assert ".." not in Path(rel).parts, f"{name}: contains '..'"
        target = (REPO_ROOT / rel).resolve()
        target.relative_to(REPO_ROOT.resolve())  # raises ValueError if escaped


def test_dependency_binding_count_and_names():
    doc = e27.load_identity()
    assert frozenset(doc["dependency_bindings"].keys()) == e27._ALL_BINDING_NAMES
    assert len(doc["dependency_bindings"]) == 21


def test_implementation_runner_cli_launcher_bound_to_actual_committed_bytes():
    import hashlib
    doc = e27.load_identity()
    for name, path in (
        ("implementation_runner", "src/dfp_phase1b_t64_early_exit_benchmark.py"),
        ("implementation_cli", "diagnostics/run_dfp_phase1b_t64_early_exit_benchmark.py"),
        ("implementation_launcher", "dfp_phase1b_t64_early_exit_benchmark_h100.sbatch"),
    ):
        assert doc["dependency_bindings"][name]["relative_path"] == path
        observed = hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        assert observed == doc["dependency_bindings"][name]["sha256"]


def test_e17_manifest_digests_match_independently_established_values():
    doc = e27.load_identity()
    binding = doc["dependency_bindings"]["e17_manifest"]
    assert binding["sha256"] == "be2c77466a2c1d64f0f39f17f4d073ce1de92dc526a59e47ae8c3dabb8c644ab"
    assert binding["canonical_manifest_digest"] == "4b962001ccab863b4c7a156cdbde5f55b66b3ff26a9d429a8b6f148550551847"
    assert binding["benchmark_record_digest"] == "8193ef03f5d1dee843098ded6ce896f0f5fe072d49f37d2657ad9b6507fec142"


def _mutate_and_expect_reject(old: str, new: str, label: str, match: str | None = None):
    assert old in RAW_TEXT, f"construction error: anchor not found for {label}"
    mutated = RAW_TEXT.replace(old, new, 1)
    assert mutated != RAW_TEXT, f"construction error (no-op) for {label}"
    tmp_dir = Path(tempfile.mkdtemp(prefix="e27_mut_"))
    try:
        tmp_path = tmp_dir / "mutated.toml"
        tmp_path.write_text(mutated)
        with pytest.raises(E, match=match) if match else pytest.raises(E):
            e27.load_identity(path=tmp_path, repo_root=REPO_ROOT)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_reject_wrong_dependency_hash():
    _mutate_and_expect_reject(
        'sha256 = "806b6ea9a20413c8aac714f88d4de18078974525e1d912b29d5b7f446663381f"',
        'sha256 = "' + "f" * 64 + '"',
        "implementation_runner hash",
    )


def test_reject_path_traversal_in_dependency_binding():
    _mutate_and_expect_reject(
        'relative_path = "src/dfp_phase1b_t64_early_exit_benchmark.py"',
        'relative_path = "../../../../etc/passwd"',
        "path traversal",
    )


def test_reject_symlink_escape_in_dependency_binding(tmp_path):
    """Builds a fixture repo_root whose implementation_runner binding
    resolves through a symlink to a file OUTSIDE that fixture root --
    _resolve_repo_relative must reject this via .resolve() containment,
    identical to E25/E26's own established convention."""
    fixture_root = tmp_path / "fixture_repo"
    fixture_root.mkdir()
    # Copy every OTHER bound file so only the runner binding is exercised via symlink.
    doc = tomllib.load(open(REPO_ROOT / e27.IDENTITY_RELATIVE_PATH, "rb"))
    for name, binding in doc["dependency_bindings"].items():
        rel = binding["relative_path"]
        dest = fixture_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if name == "implementation_runner":
            outside_dir = tmp_path / "outside"
            outside_dir.mkdir(exist_ok=True)
            outside_file = outside_dir / "runner.py"
            outside_file.write_bytes((REPO_ROOT / rel).read_bytes())
            dest.symlink_to(outside_file)
        else:
            dest.write_bytes((REPO_ROOT / rel).read_bytes())
    identity_dest = fixture_root / e27.IDENTITY_RELATIVE_PATH
    identity_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_ROOT / e27.IDENTITY_RELATIVE_PATH, identity_dest)
    with pytest.raises(E):
        e27.load_identity(repo_root=fixture_root)


def test_reject_tampered_dependency_bytes_at_correct_path(tmp_path):
    fixture_root = tmp_path / "fixture_repo"
    fixture_root.mkdir()
    doc = tomllib.load(open(REPO_ROOT / e27.IDENTITY_RELATIVE_PATH, "rb"))
    for name, binding in doc["dependency_bindings"].items():
        rel = binding["relative_path"]
        dest = fixture_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = (REPO_ROOT / rel).read_bytes()
        if name == "e13_protocol":
            content += b"\n# tampered\n"
        dest.write_bytes(content)
    identity_dest = fixture_root / e27.IDENTITY_RELATIVE_PATH
    identity_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_ROOT / e27.IDENTITY_RELATIVE_PATH, identity_dest)
    with pytest.raises(E):
        e27.load_identity(repo_root=fixture_root)


# ---------------------------------------------------------------------
# Exact execution scope: resources, variants, population, schedule.
# ---------------------------------------------------------------------


def test_exact_resources():
    doc = e27.load_identity()
    r = doc["execution_scope"]["resources"]
    assert r["gpu_count"] == 1
    assert r["gpu_model"] == "h100_80gb"
    assert r["gpu_model_string_match"] == "NVIDIA H100 80GB HBM3"
    assert r["cpus_per_task"] == 8
    assert r["host_memory"] == "80G"
    assert r["max_wall_time"] == "02:00:00"
    assert r["account"] == "rrg-yangw"


def test_exact_variants_and_horizons():
    doc = e27.load_identity()
    v = doc["execution_scope"]["variants"]
    assert v["list"] == ["T64_early_exit", "T320_reference"]
    assert v["t64_early_exit"]["recurrence_steps"] == 64
    assert v["t320_reference"]["recurrence_steps"] == 320


def test_exact_population():
    doc = e27.load_identity()
    p = doc["execution_scope"]["population"]
    assert p["measured_window_count"] == 24
    assert p["manifest_auxiliary_warmup_record_count"] == 1
    assert p["warmup_trial_window_count"] == 24
    assert p["measured_trial_window_count"] == 24
    assert p["warmup_trials_process_canonical_benchmark_records"] is True
    assert p["manifest_override_authorized"] is False
    assert p["alternate_population_authorized"] is False


def test_exact_schedule():
    doc = e27.load_identity()
    s = doc["execution_scope"]["schedule"]
    assert s["warmup"]["ordered_list"] == ["T64_early_exit", "T320_reference"]
    assert s["measured"]["ordered_trial_list"] == [dict(e) for e in e27.COUNTERBALANCED_ORDER]


def test_reject_wrong_gpu_model():
    _mutate_and_expect_reject('gpu_model = "h100_80gb"', 'gpu_model = "a100_80gb"', "gpu model")


def test_reject_wrong_wall_time():
    _mutate_and_expect_reject('max_wall_time = "02:00:00"', 'max_wall_time = "12:00:00"', "wall time")


def test_reject_wrong_window_count():
    _mutate_and_expect_reject("measured_window_count = 24", "measured_window_count = 23", "window count")


def test_reject_wrong_t64_steps():
    _mutate_and_expect_reject(
        '[execution_scope.variants.t64_early_exit]\nrecurrence_steps = 64',
        '[execution_scope.variants.t64_early_exit]\nrecurrence_steps = 63',
        "t64 steps",
    )


def test_reject_wrong_schedule_order():
    _mutate_and_expect_reject(
        '{ trial_index = 0, variant = "T320_reference", repeat = 0 },',
        '{ trial_index = 0, variant = "T64_early_exit", repeat = 0 },',
        "schedule order",
    )


def test_reject_relaxed_manifest_override():
    _mutate_and_expect_reject("manifest_override_authorized = false", "manifest_override_authorized = true", "manifest override")


def test_reject_relaxed_adaptive_horizons():
    _mutate_and_expect_reject("adaptive_horizons_authorized = false", "adaptive_horizons_authorized = true", "adaptive horizons")


def test_reject_relaxed_extra_repeats():
    _mutate_and_expect_reject("extra_repeats_authorized = false", "extra_repeats_authorized = true", "extra repeats")


# ---------------------------------------------------------------------
# REPAIR_E27_WARMUP_SCOPE_DISCLOSURE regression tests (20): corrected
# warm-up-scope semantics (execution_scope.population), the new
# relationally-verified execution_scope.campaign_workload table, and
# the independently-reconstructed dataset_access_scope population
# overlap disclosure (processed_ids / resolution_only_ids /
# provider_resolution_ids).
# ---------------------------------------------------------------------


def test_repaired_population_schema_loads_with_new_field_names():
    doc = e27.load_identity()
    p = doc["execution_scope"]["population"]
    assert set(p.keys()) == e27._POPULATION_KEYS
    assert "warmup_record_count" not in p


def test_reject_old_warmup_record_count_key_name_reintroduced():
    """The pre-repair field name must never silently reappear alongside
    the new schema -- reintroducing it makes the population table fail
    the closed-key-set check (unknown key)."""
    _mutate_and_expect_reject(
        "measured_window_count = 24\nmanifest_auxiliary_warmup_record_count = 1",
        "measured_window_count = 24\nmanifest_auxiliary_warmup_record_count = 1\nwarmup_record_count = 1",
        "reintroduced warmup_record_count key",
    )


def test_warmup_and_measured_trial_window_counts_equal_measured_window_count():
    doc = e27.load_identity()
    p = doc["execution_scope"]["population"]
    assert p["warmup_trial_window_count"] == p["measured_window_count"]
    assert p["measured_trial_window_count"] == p["measured_window_count"]
    assert p["warmup_trial_window_count"] == p["measured_trial_window_count"] == e27.WINDOW_COUNT


def test_manifest_auxiliary_warmup_record_count_is_distinct_from_window_counts():
    """Guards against re-conflating the manifest's own single auxiliary
    record (an ID-resolution count) with a per-trial window count --
    the original defect this repair fixes."""
    doc = e27.load_identity()
    p = doc["execution_scope"]["population"]
    assert p["manifest_auxiliary_warmup_record_count"] == 1
    assert p["manifest_auxiliary_warmup_record_count"] != p["warmup_trial_window_count"]


def test_reject_warmup_trial_window_count_reduced_to_one():
    """Directly re-tests the original defect: a warm-up trial processing
    only a single window is exactly the mischaracterization this repair
    corrects, and must be rejected if ever reintroduced."""
    _mutate_and_expect_reject("warmup_trial_window_count = 24", "warmup_trial_window_count = 1", "warmup trial window count reduced to one")


def test_reject_measured_trial_window_count_mismatch():
    _mutate_and_expect_reject("measured_trial_window_count = 24", "measured_trial_window_count = 23", "measured trial window count mismatch")


def test_reject_warmup_trials_do_not_process_canonical_population():
    _mutate_and_expect_reject(
        "warmup_trials_process_canonical_benchmark_records = true",
        "warmup_trials_process_canonical_benchmark_records = false",
        "warmup trials do not process canonical population",
    )


def test_campaign_workload_totals_and_subtotals_exact():
    doc = e27.load_identity()
    c = doc["execution_scope"]["campaign_workload"]
    assert c["warmup_window_executions"] == 48
    assert c["measured_window_executions"] == 144
    assert c["total_window_executions"] == 192
    assert c["warmup_window_executions"] + c["measured_window_executions"] == c["total_window_executions"]
    assert c["counts_are_scheduled_workload_not_evidence_of_execution"] is True


def test_reject_wrong_total_window_executions():
    _mutate_and_expect_reject("total_window_executions = 192", "total_window_executions = 191", "wrong total window executions")


def test_campaign_recurrence_steps_derivation_exact():
    doc = e27.load_identity()
    c = doc["execution_scope"]["campaign_workload"]
    assert c["t64_campaign_recurrence_steps"] == c["trials_per_variant"] * e27.WINDOW_COUNT * e27.T64_STEPS == 6144
    assert c["t320_campaign_recurrence_steps"] == c["trials_per_variant"] * e27.WINDOW_COUNT * e27.T320_STEPS == 30720
    assert c["total_campaign_recurrence_steps"] == c["t64_campaign_recurrence_steps"] + c["t320_campaign_recurrence_steps"] == 36864


def test_reject_wrong_t64_campaign_recurrence_steps():
    _mutate_and_expect_reject("t64_campaign_recurrence_steps = 6144", "t64_campaign_recurrence_steps = 6143", "wrong t64 campaign recurrence steps")


def test_reject_wrong_trials_per_variant():
    """4 trials per variant (1 warm-up + 3 measured) is itself derived
    relationally from warmup_trial_count/measured_trial_count divided
    across the 2 variants -- an inconsistent value must be rejected."""
    _mutate_and_expect_reject("trials_per_variant = 4", "trials_per_variant = 5", "wrong trials per variant")


def test_independent_id_reconstruction_matches_declared_dataset_access_scope():
    """Reconstructs processed_ids/resolution_only_ids/provider_resolution_ids
    from source -- the E17 manifest JSON read directly (never through
    e27's own loader) plus the committed BALP split loader -- replicating
    the committed CLI's own exact set-construction rule
    (`sorted({warmup_record["image_id"]} | {r["image_id"] for r in
    benchmark_records})`), and cross-checks the result against E27's own
    declared, loaded figures."""
    import json

    import src.dfp_phase1b_lightweight_boundary_identity as balp

    doc = e27.load_identity()
    e17_relative_path = doc["dependency_bindings"]["e17_manifest"]["relative_path"]
    manifest_doc = json.loads((REPO_ROOT / e17_relative_path).read_bytes())
    processed_ids = {r["image_id"] for r in manifest_doc["benchmark_records"]}
    resolution_only_ids = {manifest_doc["warmup_record"]["image_id"]}
    provider_resolution_ids = processed_ids | resolution_only_ids
    assert provider_resolution_ids == set(sorted(processed_ids | resolution_only_ids))

    split = balp.load_and_verify_split(repo_root=REPO_ROOT)
    development_ids = set(split["development_image_ids"])
    confirmation_ids = set(split["confirmation_image_ids"])

    overlap = doc["dataset_access_scope"]["population_overlap_disclosure"]
    assert overlap["processed_ids"]["count"] == len(processed_ids)
    assert overlap["processed_ids"]["development_overlap_count"] == len(processed_ids & development_ids)
    assert overlap["processed_ids"]["confirmation_overlap_count"] == len(processed_ids & confirmation_ids)
    assert overlap["resolution_only_ids"]["count"] == len(resolution_only_ids)
    assert overlap["resolution_only_ids"]["development_overlap_count"] == len(resolution_only_ids & development_ids)
    assert overlap["provider_resolution_ids"]["count"] == len(provider_resolution_ids)
    assert overlap["provider_resolution_ids"]["development_overlap_count"] == len(provider_resolution_ids & development_ids)
    assert overlap["provider_resolution_ids"]["confirmation_overlap_count"] == len(provider_resolution_ids & confirmation_ids)


def test_reject_processed_ids_count_replaced_with_provider_resolution_count():
    """The original defect: substituting the unique-image
    provider_resolution_ids count (25) for the actually-decoded
    processed_ids count (24) must be rejected by the relational
    recomputation, not merely accepted as an internally-consistent
    TOML value."""
    _mutate_and_expect_reject(
        "[dataset_access_scope.population_overlap_disclosure.processed_ids]\ncount = 24",
        "[dataset_access_scope.population_overlap_disclosure.processed_ids]\ncount = 25",
        "processed_ids count substituted with provider_resolution_ids count",
    )


def test_reject_off_by_one_development_overlap_count():
    _mutate_and_expect_reject(
        "[dataset_access_scope.population_overlap_disclosure.processed_ids]\ncount = 24\ndevelopment_overlap_count = 18",
        "[dataset_access_scope.population_overlap_disclosure.processed_ids]\ncount = 24\ndevelopment_overlap_count = 17",
        "off-by-one development overlap count",
    )


def test_reject_fabricated_confirmation_overlap_count():
    _mutate_and_expect_reject(
        "[dataset_access_scope.population_overlap_disclosure.provider_resolution_ids]\ncount = 25\ndevelopment_overlap_count = 19\nconfirmation_overlap_count = 6",
        "[dataset_access_scope.population_overlap_disclosure.provider_resolution_ids]\ncount = 25\ndevelopment_overlap_count = 19\nconfirmation_overlap_count = 7",
        "fabricated confirmation overlap count",
    )


def test_reject_unknown_key_in_campaign_workload_table():
    _mutate_and_expect_reject(
        "counts_are_scheduled_workload_not_evidence_of_execution = true\n",
        "counts_are_scheduled_workload_not_evidence_of_execution = true\nextra_unauthorized_field = 1\n",
        "unknown key in campaign_workload",
    )


def test_reject_unknown_key_in_processed_ids_subtable():
    _mutate_and_expect_reject(
        "[dataset_access_scope.population_overlap_disclosure.processed_ids]\ncount = 24",
        "[dataset_access_scope.population_overlap_disclosure.processed_ids]\ncount = 24\nextra_unauthorized_field = 1",
        "unknown key in processed_ids subtable",
    )


def test_reject_bool_as_int_for_warmup_trials_process_canonical_flag():
    _mutate_and_expect_reject(
        "warmup_trials_process_canonical_benchmark_records = true",
        "warmup_trials_process_canonical_benchmark_records = 1",
        "int in place of bool for warmup_trials_process_canonical_benchmark_records",
    )


def test_dataset_access_scope_reconstruction_uses_real_balp_split_not_trusted_toml(monkeypatch):
    """Confirms the new dataset_access_scope verification genuinely calls
    out to the BALP split loader (and therefore fails closed if that
    independent source disagrees) rather than merely re-checking the
    TOML's own internal arithmetic. A domain error from the BALP loader
    must propagate as E27's own domain exception; a fatal exception
    (KeyboardInterrupt) raised from within must propagate unmodified.

    (REPAIR_E27_SPLIT_DEPENDENCY_BINDINGS: the BALP loader module is now
    loaded by exact verified path via `_import_verified_module`, never
    via a bare `import` statement -- so monkeypatching the globally
    cached `src.dfp_phase1b_lightweight_boundary_identity` module no
    longer has any effect on it. This test patches `_import_verified_
    module` itself, the actual call point, instead.)"""
    real_import_verified_module = e27._import_verified_module

    class _FakeDomainErrorType(Exception):
        pass

    class _FakeBalpModuleDomain:
        DfpPhase1bLightweightBoundaryIdentityError = _FakeDomainErrorType

        @staticmethod
        def load_and_verify_split(*a, **k):
            raise _FakeDomainErrorType("synthetic split failure")

    def fake_import_domain(path, *, module_name, label):
        return _FakeBalpModuleDomain

    monkeypatch.setattr(e27, "_import_verified_module", fake_import_domain)
    with pytest.raises(E):
        e27.load_identity()
    monkeypatch.setattr(e27, "_import_verified_module", real_import_verified_module)

    class _FakeBalpModuleFatal:
        DfpPhase1bLightweightBoundaryIdentityError = _FakeDomainErrorType

        @staticmethod
        def load_and_verify_split(*a, **k):
            raise KeyboardInterrupt

    def fake_import_fatal(path, *, module_name, label):
        return _FakeBalpModuleFatal

    monkeypatch.setattr(e27, "_import_verified_module", fake_import_fatal)
    with pytest.raises(KeyboardInterrupt):
        e27.load_identity()


# ---------------------------------------------------------------------
# Dataset-access scope and population-overlap disclosure.
# ---------------------------------------------------------------------


def test_dataset_access_scope_forbids_gt_and_confirmation():
    doc = e27.load_identity()
    d = doc["dataset_access_scope"]
    assert d["gt_access_authorized"] is False
    assert d["gt_label_use_authorized"] is False
    assert d["confirmation_accuracy_evaluation_authorized"] is False
    assert d["decoding_or_inference_on_unrelated_images_authorized"] is False


def test_population_overlap_disclosure_exact():
    """REPAIR (REPAIR_E27_WARMUP_SCOPE_DISCLOSURE): the correct
    processed-image population is 24/18/6/0 -- the previously-reported
    25/19/6/0 figures describe provider_resolution_ids (the union of
    processed_ids and the single metadata-only resolution_only_ids),
    never the actually-decoded/inferred processed_ids population."""
    doc = e27.load_identity()
    o = doc["dataset_access_scope"]["population_overlap_disclosure"]
    assert o["development_split_size"] == 350
    assert o["confirmation_split_size"] == 150

    processed = o["processed_ids"]
    assert processed["count"] == 24
    assert processed["development_overlap_count"] == 18
    assert processed["confirmation_overlap_count"] == 6
    assert processed["outside_frozen_population_count"] == 0

    resolution_only = o["resolution_only_ids"]
    assert resolution_only["count"] == 1
    assert resolution_only["development_overlap_count"] == 1
    assert resolution_only["confirmation_overlap_count"] == 0
    assert resolution_only["outside_frozen_population_count"] == 0
    assert resolution_only["disjoint_from_processed_ids"] is True

    provider = o["provider_resolution_ids"]
    assert provider["count"] == 25
    assert provider["development_overlap_count"] == 19
    assert provider["confirmation_overlap_count"] == 6
    assert provider["outside_frozen_population_count"] == 0


def test_reject_gt_access_authorization_injection():
    _mutate_and_expect_reject("gt_access_authorized = false\ngt_label_use_authorized = false", "gt_access_authorized = true\ngt_label_use_authorized = false", "gt access")


def test_reject_confirmation_accuracy_evaluation_injection():
    _mutate_and_expect_reject(
        "confirmation_accuracy_evaluation_authorized = false\ndecoding_or_inference_on_unrelated_images_authorized = false",
        "confirmation_accuracy_evaluation_authorized = true\ndecoding_or_inference_on_unrelated_images_authorized = false",
        "confirmation accuracy",
    )


# ---------------------------------------------------------------------
# Output lifecycle contract.
# ---------------------------------------------------------------------


def test_output_lifecycle_env_vars_match_existing_launcher():
    doc = e27.load_identity()
    o = doc["output_lifecycle_contract"]
    assert set(o["launcher_required_environment_variables"]) == {"DFP_REPO_ROOT", "DFP_VENV_ACTIVATE_PATH", "COCO_STUFF_REAL_DATA_ROOT", "TALK2DINO_WEIGHT_DIR", "DFP_BENCHMARK_OUTPUT_ROOT"}
    assert set(o["launcher_optional_environment_variables"]) == {"DFP_OPENCV_MODULE"}
    assert o["result_bundle_directory_name"] == "result"
    assert o["result_bundle_contents"] == ["merged_result.json", "decision.json"]


def test_reject_relaxed_overwrite_flag():
    _mutate_and_expect_reject(
        "\noverwrite_authorized = false\nforce_mode_authorized = false\nunique_campaign_destination_required = true",
        "\noverwrite_authorized = true\nforce_mode_authorized = false\nunique_campaign_destination_required = true",
        "overwrite authorized",
    )


def test_reject_relaxed_force_mode_flag():
    _mutate_and_expect_reject(
        "\noverwrite_authorized = false\nforce_mode_authorized = false\nunique_campaign_destination_required = true",
        "\noverwrite_authorized = false\nforce_mode_authorized = true\nunique_campaign_destination_required = true",
        "force mode",
    )


def test_reject_relaxed_success_before_verification():
    _mutate_and_expect_reject("success_before_verification_authorized = false", "success_before_verification_authorized = true", "success before verification")


def test_environment_variable_names_match_actual_launcher_source():
    """Cross-check against the ACTUAL committed launcher source, not
    only E27's own declared set -- catches drift if the launcher is
    ever amended without updating E27."""
    launcher_text = (REPO_ROOT / "dfp_phase1b_t64_early_exit_benchmark_h100.sbatch").read_text()
    doc = e27.load_identity()
    for var in doc["output_lifecycle_contract"]["launcher_required_environment_variables"]:
        assert var in launcher_text, f"{var} not found in the actual launcher source"


# ---------------------------------------------------------------------
# Measurement bindings and decision binding.
# ---------------------------------------------------------------------


def test_measurement_bindings_exact_counts():
    doc = e27.load_identity()
    c = doc["measurement_bindings"]["expected_counts_per_measured_trial"]
    assert c["window_count"] == 24
    assert c["model_forward_count"] == 24
    assert c["graph_construction_count"] == 24
    assert c["t64_recurrence_steps_executed"] == 1536
    assert c["t320_recurrence_steps_executed"] == 7680


def test_window_throughput_disclosure_never_claims_image_throughput():
    doc = e27.load_identity()
    w = doc["measurement_bindings"]["window_throughput_disclosure"]
    assert w["represents_image_throughput"] is False
    assert w["represents_full_5000_image_runtime"] is False


def test_decision_binding_exact_threshold_and_outcomes():
    doc = e27.load_identity()
    d = doc["decision_binding"]
    assert d["end_to_end_speedup_threshold"] == 1.25
    assert d["threshold_comparison"] == ">="
    assert d["outcomes"] == ["PROCEED_TO_FREEZE_T64_CONFIRMATION_PROTOCOL", "STOP_T64_EARLY_EXIT_TRACK"]
    assert d["proceed_outcome_authorizes_confirmation_execution"] is False
    assert d["proceed_outcome_authorizes_full_5000_evaluation"] is False


def test_reject_altered_speedup_threshold():
    _mutate_and_expect_reject("end_to_end_speedup_threshold = 1.25", "end_to_end_speedup_threshold = 1.1", "speedup threshold")


def test_reject_broadened_proceed_authorizes_confirmation():
    _mutate_and_expect_reject(
        "proceed_outcome_authorizes_confirmation_execution = false",
        "proceed_outcome_authorizes_confirmation_execution = true",
        "proceed authorizes confirmation",
    )


def test_reject_broadened_proceed_authorizes_full_5000():
    _mutate_and_expect_reject(
        "proceed_outcome_authorizes_full_5000_evaluation = false",
        "proceed_outcome_authorizes_full_5000_evaluation = true",
        "proceed authorizes full 5000",
    )


# ---------------------------------------------------------------------
# Authorization boundary: exact true/false sets, individually toggled.
# ---------------------------------------------------------------------


def test_authorization_boundary_exact_true_and_false_sets():
    doc = e27.load_identity()
    ab = doc["authorization_boundary"]
    assert len(ab) == 21
    for key in e27._AUTHORIZATION_BOUNDARY_TRUE_KEYS:
        assert ab[key] is True, key
    for key in e27._AUTHORIZATION_BOUNDARY_FALSE_KEYS:
        assert ab[key] is False, key


@pytest.mark.parametrize("key", list(e27._AUTHORIZATION_BOUNDARY_FALSE_KEYS))
def test_reject_broadened_authorization_flag(key):
    _mutate_and_expect_reject(f"\n{key} = false", f"\n{key} = true", f"broadened {key}")


@pytest.mark.parametrize("key", list(e27._AUTHORIZATION_BOUNDARY_TRUE_KEYS))
def test_reject_narrowed_required_true_flag(key):
    _mutate_and_expect_reject(f"\n{key} = true", f"\n{key} = false", f"narrowed {key}")


def test_reject_removed_prohibited_entry():
    _mutate_and_expect_reject('"overwrite_or_force",\n', "", "removed prohibited entry")


# ---------------------------------------------------------------------
# Activation prerequisites: successor authorization scoping.
# ---------------------------------------------------------------------


def test_activation_prerequisites_exact_list():
    doc = e27.load_identity()
    assert tuple(doc["activation_prerequisites"]["list"]) == e27.ACTIVATION_PREREQUISITES
    assert doc["activation_prerequisites"]["all_required"] is True
    assert doc["activation_prerequisites"]["this_task_satisfies_none_of_these_by_itself"] is True


def test_reject_removed_activation_prerequisite():
    _mutate_and_expect_reject(
        '    "verified_implementation_and_dependency_hashes",\n',
        "",
        "removed activation prerequisite",
    )


def test_reject_relaxed_all_required_flag():
    _mutate_and_expect_reject("all_required = true", "all_required = false", "all_required relaxed")


def test_execution_readiness_contract_never_claims_launcher_enforcement():
    doc = e27.load_identity()
    c = doc["execution_readiness_contract"]
    assert c["launcher_does_not_import_this_module"] is True
    assert c["operator_must_run_this_separately_before_submission"] is True
    assert c["never_checks_gpu_dataset_model_or_cuda"] is True


# ---------------------------------------------------------------------
# Deferred next steps / prohibited.
# ---------------------------------------------------------------------


def test_deferred_next_steps_exact_sequence():
    doc = e27.load_identity()
    assert tuple(doc["deferred_next_steps"]["sequence"]) == (
        "independent_e27_verification", "commit_e27", "committed_head_production_readiness",
        "operator_manually_submits_the_verified_launcher", "independent_benchmark_result_verification",
    )


def test_prohibited_list_exact():
    doc = e27.load_identity()
    assert tuple(doc["prohibited"]["list"]) == e27._REQUIRED_PROHIBITED


# ---------------------------------------------------------------------
# Malformed types / wrong / missing / extra keys.
# ---------------------------------------------------------------------


def test_reject_unknown_top_level_key():
    _mutate_and_expect_reject("\n[prohibited]\n", "\n[unexpected_extra]\nfoo=1\n\n[prohibited]\n", "unknown top-level key")


def test_reject_missing_identity_key():
    _mutate_and_expect_reject("\nmutates_or_version_bumps_earlier_identity = false\n", "\n", "missing identity key")


def test_reject_bool_as_int_schema_version():
    _mutate_and_expect_reject("\nschema_version = 1\n", "\nschema_version = true\n", "bool schema_version")


def test_reject_nan_in_numeric_field():
    _mutate_and_expect_reject("cpus_per_task = 8", "cpus_per_task = nan", "NaN")


def test_reject_infinity_in_numeric_field():
    _mutate_and_expect_reject("cpus_per_task = 8", "cpus_per_task = inf", "infinity")


def test_reject_wrong_type_for_bool_field():
    _mutate_and_expect_reject("\nsingle_node_required = true\n", "\nsingle_node_required = 1\n", "int for bool")


def test_reject_missing_authorization_boundary_field():
    _mutate_and_expect_reject("\ngt_access_authorized = false\n", "\n", "missing authorization field")


def test_reject_extra_dependency_binding():
    extra = '\n[dependency_bindings.extra_unauthorized]\nrelative_path = "evaluation_identities/e13_dfp_phase1a_protocol.toml"\nsha256 = "227549b03ec6b1262c465a18b7fc67529fa5a0b5deef419c33ec79a80feb2173"\nbinds = "extra"\n'
    _mutate_and_expect_reject("\n[execution_scope.resources]", extra + "\n[execution_scope.resources]", "extra dependency binding")


# ---------------------------------------------------------------------
# Repository gate: minimal, freshly-created temporary git fixtures --
# never the real repository's own working tree or history.
# ---------------------------------------------------------------------


def _make_minimal_git_repo(tmp_path: Path, *, seed: str = "v1") -> Path:
    repo = tmp_path / "minimal_repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    # `seed` guarantees a genuinely distinct tree (and therefore commit
    # hash) between separately-constructed fixture repos even if they
    # happen to be created within the same one-second git timestamp
    # resolution window -- an identical tree + identical committer
    # identity + identical message + identical second would otherwise
    # produce IDENTICAL commit hashes across two "unrelated" repos.
    (repo / "placeholder.txt").write_text(f"{seed}\n")
    subprocess.run(["git", "add", "placeholder.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", f"initial-{seed}"], cwd=repo, check=True)
    return repo


def test_repository_gate_accepts_exact_and_descendant_commit(tmp_path):
    repo = _make_minimal_git_repo(tmp_path)
    base_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    e27.verify_repository_gate.__wrapped__ if False else None  # no-op; keep direct call below

    # exact match
    import src.dfp_phase1b_t64_early_exit_benchmark_execution_identity as mod
    orig_required = mod.REQUIRED_ANCESTOR_COMMIT
    mod.REQUIRED_ANCESTOR_COMMIT = base_commit
    try:
        mod.verify_repository_gate(repo, base_commit)  # exact match: no raise

        # descendant commit
        (repo / "placeholder.txt").write_text("v2\n")
        subprocess.run(["git", "add", "placeholder.txt"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "second"], cwd=repo, check=True)
        descendant_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        mod.verify_repository_gate(repo, descendant_commit)  # descendant: no raise
    finally:
        mod.REQUIRED_ANCESTOR_COMMIT = orig_required


def test_repository_gate_rejects_unrelated_commit(tmp_path):
    repo_a = _make_minimal_git_repo(tmp_path / "a", seed="repo-a-unique-content")
    repo_b = _make_minimal_git_repo(tmp_path / "b", seed="repo-b-different-content")
    commit_a = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_a, capture_output=True, text=True, check=True).stdout.strip()
    commit_b = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_b, capture_output=True, text=True, check=True).stdout.strip()

    import src.dfp_phase1b_t64_early_exit_benchmark_execution_identity as mod
    orig_required = mod.REQUIRED_ANCESTOR_COMMIT
    mod.REQUIRED_ANCESTOR_COMMIT = commit_a
    try:
        with pytest.raises(E):
            mod.verify_repository_gate(repo_b, commit_b)
    finally:
        mod.REQUIRED_ANCESTOR_COMMIT = orig_required


def test_dirty_tree_rejected_via_minimal_fixture(tmp_path):
    """require_clean_worktree (the generic, already-committed helper
    E27 reuses unmodified) correctly rejects an untracked file in a
    freshly-created, minimal temporary git repository."""
    repo = _make_minimal_git_repo(tmp_path)
    require_clean_worktree = generic.require_clean_worktree
    require_clean_worktree(repo)  # clean: no raise
    (repo / "untracked.txt").write_text("dirty")
    with pytest.raises(generic.DfpPhase1bChunkBenchmarkIdentityError):
        require_clean_worktree(repo)


def test_staged_change_rejected_via_minimal_fixture(tmp_path):
    repo = _make_minimal_git_repo(tmp_path)
    (repo / "staged.txt").write_text("staged")
    subprocess.run(["git", "add", "staged.txt"], cwd=repo, check=True)
    with pytest.raises(generic.DfpPhase1bChunkBenchmarkIdentityError):
        generic.require_clean_worktree(repo)


def test_working_tree_committed_byte_divergence_detected(tmp_path):
    """A dependency file whose WORKING-TREE bytes diverge from its
    COMMITTED bytes (uncommitted local edit) must be rejected -- builds
    a minimal fixture repo with the real E27 dependency set, commits it,
    then edits one bound file without committing."""
    fixture_root = tmp_path / "fixture_repo"
    fixture_root.mkdir()
    doc = tomllib.load(open(REPO_ROOT / e27.IDENTITY_RELATIVE_PATH, "rb"))
    for name, binding in doc["dependency_bindings"].items():
        rel = binding["relative_path"]
        dest = fixture_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO_ROOT / rel).read_bytes())
    identity_dest = fixture_root / e27.IDENTITY_RELATIVE_PATH
    identity_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_ROOT / e27.IDENTITY_RELATIVE_PATH, identity_dest)

    # (REPAIR_E27_SPLIT_DEPENDENCY_BINDINGS: balp_split_loader/
    # balp_split_artifact/balp_subset_manifest are now ordinary
    # dependency_bindings entries, so the loop above already copies
    # them -- no separate copy step is needed any more.)

    # baseline loads cleanly against the fixture (bytes match declared hashes)
    e27.load_identity(repo_root=fixture_root)

    # now diverge one dependency's working-tree bytes from its declared hash
    tampered_path = fixture_root / "src/dfp_phase1a_final_rescale_pullback.py"
    tampered_path.write_bytes(tampered_path.read_bytes() + b"\n# local edit\n")
    with pytest.raises(E):
        e27.load_identity(repo_root=fixture_root)


# ---------------------------------------------------------------------
# Domain-error behavior and fatal-exception propagation.
# ---------------------------------------------------------------------


def test_single_domain_exception_type_for_all_ordinary_failures():
    with pytest.raises(e27.DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError):
        e27.load_identity(path=Path("/nonexistent/e27.toml"))


def test_fatal_exception_propagates_from_dependency_hash_check(monkeypatch):
    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr("hashlib.sha256", boom)
    with pytest.raises(KeyboardInterrupt):
        e27.load_identity()


def test_system_exit_propagates(monkeypatch):
    def boom(*a, **k):
        raise SystemExit(3)

    monkeypatch.setattr("hashlib.sha256", boom)
    with pytest.raises(SystemExit):
        e27.load_identity()


def test_memory_error_propagates(monkeypatch):
    def boom(*a, **k):
        raise MemoryError

    monkeypatch.setattr("hashlib.sha256", boom)
    with pytest.raises(MemoryError):
        e27.load_identity()


def test_fatal_exception_propagates_through_verify_execution_readiness(monkeypatch):
    def boom(*a, **k):
        raise MemoryError

    monkeypatch.setattr(generic, "require_clean_worktree", boom)
    with pytest.raises(MemoryError):
        e27.verify_execution_readiness()


# =======================================================================
# REPAIR_E27_SPLIT_DEPENDENCY_BINDINGS regression tests (19): the three
# new direct bindings (balp_split_loader / balp_split_artifact /
# balp_subset_manifest) that close the previously-unpinned BALP
# split-loader dependency gap, plus the exact-verified-path import
# mechanism (_import_verified_module) that guarantees the file whose
# bytes are hash-checked is always the same file that is executed.
# =======================================================================

import hashlib
import sys


def _make_full_fixture(tmp_path: Path, *, subdir: str = "fixture_repo") -> Path:
    """Builds a minimal fixture repo containing byte-identical copies of
    every currently-declared E27 dependency (all 21, including the 3
    new BALP bindings), plus the identity TOML itself."""
    fixture_root = tmp_path / subdir
    fixture_root.mkdir(parents=True)
    doc = tomllib.load(open(REPO_ROOT / e27.IDENTITY_RELATIVE_PATH, "rb"))
    for name, binding in doc["dependency_bindings"].items():
        rel = binding["relative_path"]
        dest = fixture_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO_ROOT / rel).read_bytes())
    identity_dest = fixture_root / e27.IDENTITY_RELATIVE_PATH
    identity_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_ROOT / e27.IDENTITY_RELATIVE_PATH, identity_dest)
    return fixture_root


# --- 1. valid real E27 loads with the repaired binding set ---


def test_1_repaired_e27_loads_with_full_binding_set():
    doc = e27.load_identity()
    assert len(doc["dependency_bindings"]) == 21
    for name in ("balp_split_loader", "balp_split_artifact", "balp_subset_manifest"):
        assert name in doc["dependency_bindings"]
    # unaffected by this repair
    o = doc["dataset_access_scope"]["population_overlap_disclosure"]
    assert (o["processed_ids"]["count"], o["processed_ids"]["development_overlap_count"], o["processed_ids"]["confirmation_overlap_count"]) == (24, 18, 6)


# --- 2. new loader binding matches independently computed committed bytes ---


def test_2_balp_split_loader_binding_matches_independently_computed_bytes():
    doc = e27.load_identity()
    binding = doc["dependency_bindings"]["balp_split_loader"]
    assert binding["relative_path"] == "src/dfp_phase1b_lightweight_boundary_identity.py"
    independently_computed = hashlib.sha256((REPO_ROOT / binding["relative_path"]).read_bytes()).hexdigest()
    assert binding["sha256"] == independently_computed
    committed_bytes = subprocess.run(["git", "show", f"HEAD:{binding['relative_path']}"], cwd=REPO_ROOT, capture_output=True).stdout
    assert hashlib.sha256(committed_bytes).hexdigest() == independently_computed


# --- 3. working-tree dependency divergence is rejected ---


@pytest.mark.parametrize("binding_name,rel_path", [
    ("balp_split_loader", "src/dfp_phase1b_lightweight_boundary_identity.py"),
    ("balp_split_artifact", "evaluation_subsets/dfp_phase1b_lightweight_boundary_500_split.json"),
    ("balp_subset_manifest", "evaluation_subsets/dfp_phase1a_coco_stuff_500.json"),
])
def test_3_reject_working_tree_divergence_for_new_balp_bindings(tmp_path, binding_name, rel_path):
    fixture_root = _make_full_fixture(tmp_path)
    e27.load_identity(repo_root=fixture_root)  # baseline: clean
    diverged = fixture_root / rel_path
    diverged.write_bytes(diverged.read_bytes() + b"\n")
    with pytest.raises(E, match=binding_name):
        e27.load_identity(repo_root=fixture_root)


# --- 4. missing loader binding is rejected ---


def test_4_reject_missing_balp_split_loader_binding():
    old = RAW_TEXT[RAW_TEXT.index("[dependency_bindings.balp_split_loader]"):RAW_TEXT.index("[dependency_bindings.balp_split_artifact]")]
    _mutate_and_expect_reject(old, "", "missing balp_split_loader binding table")


# --- 5. extra/unknown binding is rejected ---


def test_5_reject_extra_unknown_binding_name():
    extra = '\n[dependency_bindings.balp_split_loader_extra_unauthorized]\nrelative_path = "evaluation_identities/e13_dfp_phase1a_protocol.toml"\nsha256 = "227549b03ec6b1262c465a18b7fc67529fa5a0b5deef419c33ec79a80feb2173"\nbinds = "extra"\n'
    _mutate_and_expect_reject("\n[dependency_bindings.e13_protocol]", extra + "\n[dependency_bindings.e13_protocol]", "extra unauthorized binding")


# --- 6. wrong declared loader SHA-256 is rejected ---


def test_6_reject_wrong_balp_split_loader_sha256():
    doc = tomllib.load(open(REPO_ROOT / e27.IDENTITY_RELATIVE_PATH, "rb"))
    real_hash = doc["dependency_bindings"]["balp_split_loader"]["sha256"]
    _mutate_and_expect_reject(f'sha256 = "{real_hash}"', 'sha256 = "' + "f" * 64 + '"', "wrong balp_split_loader sha256")


# --- 7. altered loader bytes with unchanged declared SHA-256 are rejected ---


def test_7_reject_altered_balp_split_loader_bytes_with_unchanged_declared_hash(tmp_path):
    fixture_root = _make_full_fixture(tmp_path)
    e27.load_identity(repo_root=fixture_root)  # baseline: clean
    loader_path = fixture_root / "src/dfp_phase1b_lightweight_boundary_identity.py"
    text = loader_path.read_text()
    loader_path.write_text(text.replace('"""', '"""(silently altered docstring)', 1))
    with pytest.raises(E, match="balp_split_loader"):
        e27.load_identity(repo_root=fixture_root)


# --- 8. substituted loader path is rejected, even if copied bytes match ---


def test_8_reject_substituted_balp_loader_path_even_with_matching_bytes(tmp_path):
    fixture_root = _make_full_fixture(tmp_path)
    real_loader_bytes = (REPO_ROOT / "src/dfp_phase1b_lightweight_boundary_identity.py").read_bytes()
    substituted_rel = "src/_substituted_balp_copy.py"
    (fixture_root / substituted_rel).write_bytes(real_loader_bytes)
    assert hashlib.sha256(real_loader_bytes).hexdigest() == hashlib.sha256((fixture_root / substituted_rel).read_bytes()).hexdigest()

    identity_text = (REPO_ROOT / e27.IDENTITY_RELATIVE_PATH).read_text()
    mutated = identity_text.replace(
        'relative_path = "src/dfp_phase1b_lightweight_boundary_identity.py"',
        f'relative_path = "{substituted_rel}"',
        1,
    )
    assert mutated != identity_text
    (fixture_root / e27.IDENTITY_RELATIVE_PATH).write_text(mutated)
    with pytest.raises(E, match="canonical path"):
        e27.load_identity(repo_root=fixture_root)


# --- 9. lexical traversal and symlink escape are rejected ---


def test_9a_reject_path_traversal_for_balp_split_loader():
    _mutate_and_expect_reject(
        'relative_path = "src/dfp_phase1b_lightweight_boundary_identity.py"',
        'relative_path = "../../../../etc/passwd"',
        "path traversal in balp_split_loader",
    )


def test_9b_reject_symlink_escape_for_balp_split_loader(tmp_path):
    fixture_root = _make_full_fixture(tmp_path)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "evil_loader.py"
    outside_file.write_bytes((REPO_ROOT / "src/dfp_phase1b_lightweight_boundary_identity.py").read_bytes())
    loader_path = fixture_root / "src/dfp_phase1b_lightweight_boundary_identity.py"
    loader_path.unlink()
    loader_path.symlink_to(outside_file)
    with pytest.raises(E):
        e27.load_identity(repo_root=fixture_root)


# --- 10. each newly bound split/subset input rejects altered bytes ---


@pytest.mark.parametrize("binding_name,rel_path", [
    ("balp_split_artifact", "evaluation_subsets/dfp_phase1b_lightweight_boundary_500_split.json"),
    ("balp_subset_manifest", "evaluation_subsets/dfp_phase1a_coco_stuff_500.json"),
])
def test_10_split_and_subset_bindings_reject_altered_bytes(tmp_path, binding_name, rel_path):
    fixture_root = _make_full_fixture(tmp_path)
    target = fixture_root / rel_path
    import json as _json
    doc = _json.loads(target.read_bytes())
    doc["__tamper_marker__"] = True
    target.write_text(_json.dumps(doc))
    with pytest.raises(E, match=binding_name):
        e27.load_identity(repo_root=fixture_root)


# --- 11. transitive/relational reconstruction check actually executes (beyond the direct hash pins) ---


def test_11_transitive_reconstruction_check_still_executes_beyond_direct_hash_pins(tmp_path):
    """Self-consistent tamper: the split JSON is altered AND its
    declared sha256 is updated to match (so the direct hash-pin check
    alone would pass) -- but the subset manifest is left untouched, so
    load_and_verify_split's OWN internal reconstruction must still
    catch the cross-file disagreement. Proves that layer genuinely
    executes rather than being dead code shadowed by the new direct
    bindings."""
    import json as _json

    fixture_root = _make_full_fixture(tmp_path)
    split_path = fixture_root / "evaluation_subsets/dfp_phase1b_lightweight_boundary_500_split.json"
    split_doc = _json.loads(split_path.read_bytes())
    split_doc["development_image_ids"][0], split_doc["confirmation_image_ids"][0] = (
        split_doc["confirmation_image_ids"][0], split_doc["development_image_ids"][0],
    )
    new_bytes = _json.dumps(split_doc).encode()
    split_path.write_bytes(new_bytes)
    new_hash = hashlib.sha256(new_bytes).hexdigest()

    identity_path = fixture_root / e27.IDENTITY_RELATIVE_PATH
    real_doc = tomllib.load(open(REPO_ROOT / e27.IDENTITY_RELATIVE_PATH, "rb"))
    old_hash = real_doc["dependency_bindings"]["balp_split_artifact"]["sha256"]
    text = identity_path.read_text()
    mutated = text.replace(f'sha256 = "{old_hash}"', f'sha256 = "{new_hash}"', 1)
    assert mutated != text
    identity_path.write_text(mutated)

    # the direct hash-pin for balp_split_artifact now matches (self-consistent) --
    # confirm the failure still comes from the reconstruction layer, not a hash mismatch.
    with pytest.raises(E, match="independent BALP split reverification failed"):
        e27.load_identity(repo_root=fixture_root)


# --- 12. a failed dependency check prevents the split loader from being called ---


def test_12_failed_dependency_check_prevents_split_loader_from_being_called(monkeypatch):
    calls = []
    original = e27._import_verified_module

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(e27, "_import_verified_module", spy)
    _mutate_and_expect_reject(
        'sha256 = "1f2f6d9555755dcf94590b92e23dcee2616e39bda8c8d90aa74bc6640bc59efd"',
        'sha256 = "' + "e" * 64 + '"',
        "wrong balp_split_loader hash prevents loader import",
    )
    assert calls == [], "the BALP split loader must never be imported when its own dependency binding fails to verify"


# --- 13. fresh subprocess: loader-source tampering rejected before import-time code executes ---


def test_13_fresh_subprocess_loader_source_tampering_rejected_before_import(tmp_path):
    fixture_root = _make_full_fixture(tmp_path)
    loader_path = fixture_root / "src/dfp_phase1b_lightweight_boundary_identity.py"
    # a syntactically INVALID mutation: if this were somehow imported
    # without prior hash verification, it would raise SyntaxError, not
    # our own domain exception -- proves verification precedes import.
    loader_path.write_text(loader_path.read_text() + "\ndef this is not valid python(((\n")

    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "import src.dfp_phase1b_t64_early_exit_benchmark_execution_identity as e27\n"
        "try:\n"
        "    e27.load_identity(repo_root=%r)\n"
        "    print('UNEXPECTED_SUCCESS')\n"
        "except e27.DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError as exc:\n"
        "    print('DOMAIN_ERROR_REJECTED')\n"
        "except SyntaxError:\n"
        "    print('SYNTAX_ERROR_LEAKED -- verification did not precede import')\n"
    ) % (str(REPO_ROOT), str(fixture_root))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "DOMAIN_ERROR_REJECTED", (result.stdout, result.stderr)


# --- 14. the imported module resolves to the verified repository source ---


def test_14_imported_module_resolves_to_verified_repository_source(tmp_path):
    fixture_root = _make_full_fixture(tmp_path)
    loader_rel = "src/dfp_phase1b_lightweight_boundary_identity.py"
    loader_path = fixture_root / loader_rel
    sentinel_source = loader_path.read_text() + "\nE27_TEST_SENTINEL = 'fixture-copy'\n"
    loader_path.write_text(sentinel_source)
    new_hash = hashlib.sha256(sentinel_source.encode()).hexdigest()

    identity_path = fixture_root / e27.IDENTITY_RELATIVE_PATH
    real_doc = tomllib.load(open(REPO_ROOT / e27.IDENTITY_RELATIVE_PATH, "rb"))
    old_hash = real_doc["dependency_bindings"]["balp_split_loader"]["sha256"]
    text = identity_path.read_text()
    identity_path.write_text(text.replace(f'sha256 = "{old_hash}"', f'sha256 = "{new_hash}"', 1))

    doc = e27.load_identity(repo_root=fixture_root)
    balp_rel = doc["dependency_bindings"]["balp_split_loader"]["relative_path"]
    resolved_path = e27._resolve_repo_relative(fixture_root, balp_rel, label="test")
    loaded_module = e27._import_verified_module(resolved_path, module_name="_test_e27_sentinel_module", label="test")

    assert getattr(loaded_module, "E27_TEST_SENTINEL", None) == "fixture-copy"
    import src.dfp_phase1b_lightweight_boundary_identity as real_module
    assert not hasattr(real_module, "E27_TEST_SENTINEL"), "the real, already-imported repo module must never be mutated by a fixture load"


# --- 15. declared counts still match independent manifest/split reconstruction ---


def test_15_declared_population_counts_match_fresh_independent_reconstruction():
    import json as _json

    doc = e27.load_identity()
    manifest_rel = doc["dependency_bindings"]["e17_manifest"]["relative_path"]
    manifest_doc = _json.loads((REPO_ROOT / manifest_rel).read_bytes())
    processed_ids = {r["image_id"] for r in manifest_doc["benchmark_records"]}
    resolution_only_ids = {manifest_doc["warmup_record"]["image_id"]}
    provider_resolution_ids = processed_ids | resolution_only_ids

    split_rel = doc["dependency_bindings"]["balp_split_artifact"]["relative_path"]
    split_doc = _json.loads((REPO_ROOT / split_rel).read_bytes())
    development_ids = set(split_doc["development_image_ids"])
    confirmation_ids = set(split_doc["confirmation_image_ids"])

    o = doc["dataset_access_scope"]["population_overlap_disclosure"]
    assert o["processed_ids"]["count"] == len(processed_ids) == 24
    assert o["processed_ids"]["development_overlap_count"] == len(processed_ids & development_ids) == 18
    assert o["processed_ids"]["confirmation_overlap_count"] == len(processed_ids & confirmation_ids) == 6
    assert o["provider_resolution_ids"]["count"] == len(provider_resolution_ids) == 25
    assert o["provider_resolution_ids"]["development_overlap_count"] == len(provider_resolution_ids & development_ids) == 19
    assert o["provider_resolution_ids"]["confirmation_overlap_count"] == len(provider_resolution_ids & confirmation_ids) == 6
    assert o["resolution_only_ids"]["count"] == len(resolution_only_ids) == 1
    assert o["resolution_only_ids"]["development_overlap_count"] == len(resolution_only_ids & development_ids) == 1
    assert o["resolution_only_ids"]["confirmation_overlap_count"] == len(resolution_only_ids & confirmation_ids) == 0

    cw = doc["execution_scope"]["campaign_workload"]
    assert (cw["warmup_trial_count"], cw["measured_trial_count"]) == (2, 6)
    assert (cw["warmup_window_executions"], cw["measured_window_executions"], cw["total_window_executions"]) == (48, 144, 192)
    assert (cw["t64_campaign_recurrence_steps"], cw["t320_campaign_recurrence_steps"], cw["total_campaign_recurrence_steps"]) == (6144, 30720, 36864)


# --- 16. ordinary failures raise the E27 domain exception ---


@pytest.mark.parametrize("old,new", [
    ('sha256 = "1f2f6d9555755dcf94590b92e23dcee2616e39bda8c8d90aa74bc6640bc59efd"', 'sha256 = "' + "0" * 64 + '"'),
    ('sha256 = "5c72634e2b27213162f547bcc9c15fe135056c84bcdc1174dc8e8b77a4451976"', 'sha256 = "' + "1" * 64 + '"'),
    ('sha256 = "75373aca8ef6d434ae02a7e90b6dcc9ba13fd8a31d1e87d858a2f542c19ef482"', 'sha256 = "' + "2" * 64 + '"'),
])
def test_16_ordinary_failures_raise_e27_domain_exception_not_a_bare_python_error(old, new):
    assert old in RAW_TEXT
    mutated = RAW_TEXT.replace(old, new, 1)
    tmp_dir = Path(tempfile.mkdtemp(prefix="e27_domain_"))
    try:
        tmp_path = tmp_dir / "mutated.toml"
        tmp_path.write_text(mutated)
        try:
            e27.load_identity(path=tmp_path, repo_root=REPO_ROOT)
            assert False, "expected rejection"
        except E:
            pass
        except Exception as exc:  # pragma: no cover -- failure path
            assert False, f"wrong exception type: {type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --- 17. fatal exceptions propagate through the new BALP verification path ---


def test_17a_keyboard_interrupt_propagates_through_import_verified_module(monkeypatch):
    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr("importlib.util.spec_from_file_location", boom)
    with pytest.raises(KeyboardInterrupt):
        e27.load_identity()


def test_17b_memory_error_propagates_through_load_and_verify_split(monkeypatch):
    """The BALP loader module is loaded by exact verified path
    (_import_verified_module), not via a bare `import` statement, so
    this patches the actual call point rather than the globally cached
    module (which _verify_dataset_access_scope no longer consults)."""

    class _FakeBalpModule:
        class DfpPhase1bLightweightBoundaryIdentityError(Exception):
            pass

        @staticmethod
        def load_and_verify_split(*a, **k):
            raise MemoryError

    monkeypatch.setattr(e27, "_import_verified_module", lambda path, *, module_name, label: _FakeBalpModule)
    with pytest.raises(MemoryError):
        e27.load_identity()


def test_17c_system_exit_propagates_through_import_verified_module(monkeypatch):
    def boom(*a, **k):
        raise SystemExit(7)

    monkeypatch.setattr("importlib.util.spec_from_file_location", boom)
    with pytest.raises(SystemExit):
        e27.load_identity()


# --- 18. E27 inspection remains torch-free and CUDA-free ---


def test_18_e27_inspection_remains_torch_free_and_cuda_free():
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; sys.path.insert(0, '.')\n"
            "import src.dfp_phase1b_t64_early_exit_benchmark_execution_identity as e27\n"
            "e27.load_identity()\n"
            "try:\n"
            "    e27.verify_execution_readiness()\n"
            "except e27.DfpPhase1bT64EarlyExitBenchmarkExecutionIdentityError:\n"
            "    pass\n"
            "print('torch' in sys.modules, 'cuda' in ' '.join(sys.modules).lower())"
        )],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False"


# --- 19. execution readiness still rejects the actual uncommitted tree ---


def test_19_execution_readiness_still_rejects_the_actual_uncommitted_tree_post_repair():
    e27.load_identity()  # ordinary loading succeeds regardless
    with pytest.raises(E):
        e27.verify_execution_readiness()
