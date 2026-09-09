"""CPU-only synthetic + real-artifact tests for the DFP Phase 1B Task 4
deterministic chunk-size selection (E18 identity, selector module,
analysis CLI). No real model, no real GPU, no torch. Every synthetic
trial is hand-built via the real, committed
``src.dfp_phase1b_chunk_benchmark.build_trial_document``/``merge_trials``
so this test file never reimplements Task 3's own schema/merge logic
-- it exercises Task 4's selection logic against both real committed
production output and adversarially-tampered fixtures."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess as sp
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
for _path in (ROOT, ROOT / "src/open_vocabulary_segmentation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import src.dfp_phase1b_chunk_benchmark as bench  # noqa: E402
import src.dfp_phase1b_chunk_benchmark_identity as ident  # noqa: E402
import src.dfp_phase1b_chunk_selection as sel  # noqa: E402

CLI_PATH = ROOT / "diagnostics/analyze_dfp_phase1b_chunk_benchmark.py"
IDENTITY_PATH = ROOT / sel.IDENTITY_RELATIVE_PATH

REAL_OUTPUT_DIR = Path("/scratch/haree/dfp_phase1b_chunk_benchmark/21331697")
_REAL_OUTPUT_AVAILABLE = REAL_OUTPUT_DIR.exists() and (REAL_OUTPUT_DIR / "merged_result.json").exists()
requires_real_output = pytest.mark.skipif(not _REAL_OUTPUT_AVAILABLE, reason="real completed Task 3 benchmark output not present on this host")


# ---------------------------------------------------------------------
# Section: synthetic six-trial fixture builder. Every trial is built
# through the real bench.build_trial_document constructor (never a
# hand-rolled dict), then optionally mutated afterward (dict(...) with
# overrides) to simulate a tampered on-disk trial file -- exactly the
# threat model this module's file-loading code must defend against.
# ---------------------------------------------------------------------

COMMON_DIGEST = "d" * 64
GPU_MODEL = sel.REQUIRED_GPU_MODEL
DEVICE_TYPE = sel.REQUIRED_DEVICE_TYPE
_HOST_SAFE = 10_000_000_000
_GPU_SAFE = 10_000_000_000
_HOST_UNSAFE = 80_000_000_000  # exceeds 0.90*80GiB
_GPU_UNSAFE = 80_000_000_000


def _real_head() -> str:
    result = sp.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _device_record(gpu_model=GPU_MODEL, device_type=DEVICE_TYPE):
    return {
        "device_type": device_type, "gpu_model_name": gpu_model, "cuda_compute_capability": "9.0",
        "cuda_driver_version": "1.0.0", "cuda_runtime_version": "1.0", "total_device_memory_bytes": 85899345920,
    }


def _software_record():
    return {
        "autocast_enabled": False, "cuda_version": "1.0", "cudnn_version": "1", "deterministic_algorithms_enabled": False,
        "numpy_version": "1.0", "python_version": "3.11.5", "repository_commit": _real_head(), "tf32_enabled": False,
        "torch_version": "0.0.0",
    }


def _windows(n=24):
    return [{"slot": i} for i in range(n)]


def _digests(n=24, digest=COMMON_DIGEST):
    return {"collection_digest": digest, "per_window_digests": {str(i): "e" * 64 for i in range(n)}}


def _memory(*, host_peak=_HOST_SAFE, gpu_reserved_peak=_GPU_SAFE):
    gpu_alloc_peak = gpu_reserved_peak // 2
    return {
        "gpu_baseline_allocated_bytes": 0, "gpu_baseline_reserved_bytes": 0,
        "gpu_peak_allocated_bytes": gpu_alloc_peak, "gpu_peak_reserved_bytes": gpu_reserved_peak,
        "gpu_end_allocated_bytes": gpu_alloc_peak, "gpu_end_reserved_bytes": gpu_reserved_peak,
        "host_process_lifetime_peak_rss_bytes": host_peak, "host_process_lifetime_vmhwm_bytes": host_peak,
        "host_current_rss_bytes": host_peak // 2,
    }


def _code_provenance(*, commit=None, ancestor=None):
    head = _real_head()
    return bench.build_code_provenance(
        identity_loader_sha256="1" * 64, runner_sha256="2" * 64, cli_sha256="3" * 64, launcher_sha256="4" * 64,
        repository_commit=commit or head, required_ancestor_commit=ancestor or head, repository_worktree_clean=True,
    )


def _trial(trial_index, repeat_index, chunk_size, elapsed_ns, *, host_peak=_HOST_SAFE, gpu_reserved_peak=_GPU_SAFE,
           digest=COMMON_DIGEST, gpu_model=GPU_MODEL, device_type=DEVICE_TYPE, complete=True, final=True,
           aborted=False, window_count=24, software_record=None, code_provenance=None,
           authority_sha256="a" * 64, manifest_sha256="b" * 64):
    return bench.build_trial_document(
        authority_sha256=authority_sha256, manifest_sha256=manifest_sha256,
        trial_index=trial_index, repeat_index=repeat_index, chunk_size=chunk_size,
        device_record=_device_record(gpu_model, device_type), software_record=software_record or _software_record(),
        window_identity=_windows(window_count), primary_elapsed_ns=elapsed_ns, secondary_elapsed_ns=None,
        per_record_timing=[{} for _ in range(window_count)], workload_records=[{} for _ in range(window_count)],
        memory=_memory(host_peak=host_peak, gpu_reserved_peak=gpu_reserved_peak),
        scientific_output_digests=_digests(window_count, digest),
        complete=complete, final=final, code_provenance=code_provenance or _code_provenance(),
        abort_status={"aborted": True, "reason": "guard"} if aborted else None,
    )


def _six_trials(elapsed_map, *, host_peak_map=None, gpu_peak_map=None, digest=COMMON_DIGEST, **kw):
    """``elapsed_map``: {chunk_size: (repeat0_ns, repeat1_ns)}.
    ``host_peak_map``/``gpu_peak_map``: optional {chunk_size: (repeat0, repeat1)} overrides."""
    host_peak_map = host_peak_map or {}
    gpu_peak_map = gpu_peak_map or {}
    trials = []
    for ti, ri, cs in bench.TRIAL_SCHEDULE:
        hp = host_peak_map[cs][ri] if cs in host_peak_map else _HOST_SAFE
        gp = gpu_peak_map[cs][ri] if cs in gpu_peak_map else _GPU_SAFE
        trials.append(_trial(ti, ri, cs, elapsed_map[cs][ri], host_peak=hp, gpu_reserved_peak=gp, digest=digest, **kw))
    return trials


_DEFAULT_ELAPSED = {16: (500, 400), 32: (600, 550), 64: (700, 650)}  # chunk 16 wins


def _write_output(tmp_path, trials, *, merged_override=None, name="output"):
    """Writes each trial to its content-determined filename (so file
    order never matters) and builds merged_result.json from the
    CANONICAL trial_index-sorted order -- exactly what the real
    production launcher does (it merges ``--trial`` flags in schedule
    order). This keeps the on-disk merged artifact's ``raw_trials``
    order independent of whatever order this fixture's caller happened
    to construct/pass the in-memory ``trials`` list in."""
    out = tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    for t in trials:
        p = out / f"trial_{t['trial_index']}_repeat{t['repeat_index']}_chunk{t['chunk_size']}.json"
        p.write_text(json.dumps(t))
    if merged_override is not None:
        merged = merged_override
    else:
        canonical_order = sorted(trials, key=lambda t: t["trial_index"])
        merged = bench.merge_trials(canonical_order, verify_repo_root=ROOT)
    (out / "merged_result.json").write_text(json.dumps(merged))
    return out


# ---------------------------------------------------------------------
# Section: E18 identity -- happy path + closed-schema/value mutations.
# ---------------------------------------------------------------------


def test_e18_identity_loads_and_verifies():
    doc = sel.load_e18_identity(ROOT)
    assert doc["identity"]["successor_of"] == "e17-dfp-phase1b-chunk-benchmark"
    assert doc["operational_scope"]["operational_chunk_size_selected"] is True
    assert doc["operational_scope"]["conservation_tolerance_defined"] is False


def test_e18_dependency_bindings_verify_against_real_files():
    doc = sel.load_e18_identity(ROOT)
    sel.verify_e18_dependency_bindings(ROOT, doc, merged_result_path=REAL_OUTPUT_DIR / "merged_result.json") \
        if _REAL_OUTPUT_AVAILABLE else pytest.skip("no real output")


def test_e18_cross_checked_against_live_e17():
    doc = sel.load_e18_identity(ROOT)
    e17 = sel.verify_e18_against_e17(ROOT, doc)
    assert e17["identity"]["name"] == "e17-dfp-phase1b-chunk-benchmark"


@pytest.mark.parametrize("find,replace", [
    # --- operational_scope booleans (unauthorized flags) ---
    ('conservation_tolerance_defined = false', 'conservation_tolerance_defined = true'),
    ('full_phase1b_execution_authorized = false', 'full_phase1b_execution_authorized = true'),
    ('five_thousand_image_evaluation_authorized = false', 'five_thousand_image_evaluation_authorized = true'),
    ('gpu_execution_performed_by_this_step = false', 'gpu_execution_performed_by_this_step = true'),
    ('scientific_metric = false', 'scientific_metric = true'),
    ('statistical_significance_claimed = false', 'statistical_significance_claimed = true'),
    # --- authorization_boundary (second occurrence of the same flags) ---
    ('[authorization_boundary]\noperational_chunk_size_selected = true\nconservation_tolerance_defined = false',
     '[authorization_boundary]\noperational_chunk_size_selected = true\nconservation_tolerance_defined = true'),
    ('five_thousand_image_execution_authorized = false', 'five_thousand_image_execution_authorized = true'),
    # --- selection contract (changed candidate list / repeat / window count / statistic) ---
    ('candidate_chunk_sizes = [16, 32, 64]', 'candidate_chunk_sizes = [16, 32]'),
    ('required_repeats_per_candidate = 2', 'required_repeats_per_candidate = 3'),
    ('required_measured_windows_per_trial = 24', 'required_measured_windows_per_trial = 12'),
    ('elapsed_time_field = "timing.primary_elapsed_ns"', 'elapsed_time_field = "timing.secondary_elapsed_ns"'),
    # --- tie-break order ---
    ('lower_max_host_process_lifetime_peak_rss_bytes",\n    "lower_max_gpu_peak_reserved_bytes',
     'lower_max_gpu_peak_reserved_bytes",\n    "lower_max_host_process_lifetime_peak_rss_bytes'),
    # --- safety contract ---
    ('comparison_operator = "strictly_greater_than"', 'comparison_operator = "greater_than_or_equal"'),
    ('host_rss_abort_fraction_of_allocation = 0.90', 'host_rss_abort_fraction_of_allocation = 0.50'),
    # --- disclosure booleans ---
    ('two_repeats_do_not_support_statistical_significance = true', 'two_repeats_do_not_support_statistical_significance = false'),
    ('chunk_16_not_claimed_scientifically_superior = true', 'chunk_16_not_claimed_scientifically_superior = false'),
    # --- schema identity ---
    ('schema_name = "talk2dino-dfp-phase1b-chunk-selection-identity-v1"', 'schema_name = "talk2dino-dfp-phase1b-chunk-selection-identity-v2"'),
    ('successor_of = "e17-dfp-phase1b-chunk-benchmark"', 'successor_of = "e16-dfp-phase1b-conservation-calibration"'),
])
def test_e18_identity_rejects_every_frozen_value_mutation(tmp_path, find, replace):
    text = IDENTITY_PATH.read_text()
    assert find in text, f"fixture assumption stale: {find!r} not found in the committed E18 identity"
    mutated = text.replace(find, replace, 1)
    bad_path = tmp_path / "e18_mutated.toml"
    bad_path.write_text(mutated)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_e18_identity(ROOT, path=bad_path)


def test_e18_identity_rejects_unknown_top_level_key(tmp_path):
    text = IDENTITY_PATH.read_text()
    mutated = text + '\n[unexpected_new_section]\nfoo = 1\n'
    bad_path = tmp_path / "e18_extra.toml"
    bad_path.write_text(mutated)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_e18_identity(ROOT, path=bad_path)


def test_e18_identity_rejects_missing_top_level_key(tmp_path):
    text = IDENTITY_PATH.read_text()
    # Remove the entire [disclosure] table and everything up to the next top-level table.
    start = text.index("[disclosure]")
    end = text.index("[prohibited]")
    mutated = text[:start] + text[end:]
    bad_path = tmp_path / "e18_missing.toml"
    bad_path.write_text(mutated)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_e18_identity(ROOT, path=bad_path)


def test_e18_identity_rejects_bool_as_int(tmp_path):
    text = IDENTITY_PATH.read_text().replace("scientific_metric = false", "scientific_metric = 0")
    bad_path = tmp_path / "e18_boolint.toml"
    bad_path.write_text(text)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_e18_identity(ROOT, path=bad_path)


def test_e18_identity_rejects_malformed_sha(tmp_path):
    text = IDENTITY_PATH.read_text()
    real_sha = "cf022800d64b70f08f6f1421c82a278e50ce378c3fe3df0e59384e15814a1876"
    text = text.replace(f'sha256 = "{real_sha}"', 'sha256 = "not-a-real-hash"')
    bad_path = tmp_path / "e18_badsha.toml"
    bad_path.write_text(text)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_e18_identity(ROOT, path=bad_path)


def test_e18_identity_rejects_absolute_dependency_path(tmp_path):
    text = IDENTITY_PATH.read_text().replace(
        'relative_path = "evaluation_identities/e17_dfp_phase1b_chunk_benchmark.toml"',
        'relative_path = "/etc/passwd"',
    )
    bad_path = tmp_path / "e18_abs.toml"
    bad_path.write_text(text)
    doc = sel.load_e18_identity(ROOT, path=bad_path)  # schema-level load does not resolve paths
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.verify_e18_dependency_bindings(ROOT, doc, merged_result_path=IDENTITY_PATH)


def test_e18_identity_rejects_lexical_traversal_dependency_path(tmp_path):
    text = IDENTITY_PATH.read_text().replace(
        'relative_path = "evaluation_identities/e17_dfp_phase1b_chunk_benchmark.toml"',
        'relative_path = "../../../etc/passwd"',
    )
    bad_path = tmp_path / "e18_traversal.toml"
    bad_path.write_text(text)
    doc = sel.load_e18_identity(ROOT, path=bad_path)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.verify_e18_dependency_bindings(ROOT, doc, merged_result_path=IDENTITY_PATH)


def test_e18_identity_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside.toml"
    outside.write_text("not the real E17")
    escape_dir = ROOT / "evaluation_identities" / "_test_symlink_escape_e18"
    try:
        escape_dir.symlink_to(outside)
        text = IDENTITY_PATH.read_text().replace(
            'relative_path = "evaluation_identities/e17_dfp_phase1b_chunk_benchmark.toml"',
            'relative_path = "evaluation_identities/_test_symlink_escape_e18"',
        )
        bad_path = tmp_path / "e18_symlink.toml"
        bad_path.write_text(text)
        doc = sel.load_e18_identity(ROOT, path=bad_path)
        with pytest.raises(sel.DfpPhase1bChunkSelectionError):
            sel.verify_e18_dependency_bindings(ROOT, doc, merged_result_path=IDENTITY_PATH)
    finally:
        if escape_dir.is_symlink() or escape_dir.exists():
            escape_dir.unlink()


def test_e18_identity_rejects_stale_dependency_hash(tmp_path):
    fake = tmp_path / "fake_e17.toml"
    fake.write_text("format_version = 1\n")
    doc = sel.load_e18_identity(ROOT)
    doc = copy.deepcopy(doc)
    doc["dependency_bindings"] = dict(doc["dependency_bindings"])
    # Point at a real, resolvable repo file but with the WRONG pinned hash.
    doc["dependency_bindings"]["e17_identity"] = dict(doc["dependency_bindings"]["e17_identity"], sha256="0" * 64)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.verify_e18_dependency_bindings(ROOT, doc, merged_result_path=IDENTITY_PATH)


def test_e18_identity_rejects_wrong_e17_binding_relationally():
    doc = copy.deepcopy(sel.load_e18_identity(ROOT))
    doc["safety_contract"] = dict(doc["safety_contract"], host_rss_abort_fraction_of_allocation=0.10)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.verify_e18_against_e17(ROOT, doc)


# ---------------------------------------------------------------------
# Section: benchmark-output loading integrity (independent
# re-verification of all six raw trials + merged-result reconstruction).
# ---------------------------------------------------------------------


@requires_real_output
def test_real_benchmark_output_loads_and_verifies():
    loaded = sel.load_and_verify_benchmark_output(repo_root=ROOT, output_dir=REAL_OUTPUT_DIR)
    assert len(loaded["trials"]) == 6
    assert loaded["merged"]["scientific_output_digest"] == "0ab0dcf6db4fca0774825b70767d64de5a7d416576071e82faaa1133fabf7bbf"


@requires_real_output
def test_real_benchmark_analysis_selects_chunk_16():
    analysis = sel.analyze_benchmark(repo_root=ROOT, output_dir=REAL_OUTPUT_DIR)
    assert analysis["winner"] == 16
    assert analysis["tie_break_path"] == "unique_minimum_no_tie_break_needed"
    assert set(analysis["eligible"]) == {16, 32, 64}
    means = {cs: m["mean_elapsed_ns"] for cs, m in analysis["measurements"].items()}
    assert means[16] < means[32] < means[64]


@requires_real_output
def test_real_verify_selection_matches_recorded_winner():
    analysis = sel.verify_selection(repo_root=ROOT, output_dir=REAL_OUTPUT_DIR)
    assert analysis["winner"] == 16


def test_synthetic_six_trials_round_trip(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED)
    out = _write_output(tmp_path, trials)
    loaded = sel.load_and_verify_benchmark_output(repo_root=ROOT, output_dir=out)
    assert len(loaded["trials"]) == 6
    analysis = sel.analyze_benchmark(repo_root=ROOT, output_dir=out)
    assert analysis["winner"] == 16


def test_missing_trial_rejected(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED)[:-1]  # drop the last trial
    out = tmp_path / "output"
    out.mkdir()
    for t in trials:
        (out / f"trial_{t['trial_index']}_repeat{t['repeat_index']}_chunk{t['chunk_size']}.json").write_text(json.dumps(t))
    (out / "merged_result.json").write_text(json.dumps({}))
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_and_verify_benchmark_output(repo_root=ROOT, output_dir=out)


def test_extra_trial_file_rejected(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED)
    out = _write_output(tmp_path, trials)
    (out / "trial_99_repeat0_chunk16.json").write_text(json.dumps(trials[0]))
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_and_verify_benchmark_output(repo_root=ROOT, output_dir=out)


def test_duplicate_trial_rejected(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED)
    dup = dict(trials[0])
    trials_with_dup = trials[:-1] + [dup]  # two copies of trial 0, missing trial 5
    out = tmp_path / "output"
    out.mkdir()
    for i, t in enumerate(trials_with_dup):
        (out / f"trial_dup{i}.json").write_text(json.dumps(t))
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_and_verify_benchmark_output(repo_root=ROOT, output_dir=out)


def test_wrong_trial_index_rejected(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED)
    trials[0] = dict(trials[0], trial_index=99)
    out = _write_output(tmp_path, trials, merged_override={"raw_trials": trials})
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_and_verify_benchmark_output(repo_root=ROOT, output_dir=out)


def test_wrong_repeat_index_rejected(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED)
    trials[0] = dict(trials[0], repeat_index=1)
    out = _write_output(tmp_path, trials, merged_override={"raw_trials": trials})
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_and_verify_benchmark_output(repo_root=ROOT, output_dir=out)


def test_wrong_chunk_size_field_rejected(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED)
    trials[0] = dict(trials[0], chunk_size=128)
    out = _write_output(tmp_path, trials, merged_override={"raw_trials": trials})
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_and_verify_benchmark_output(repo_root=ROOT, output_dir=out)


def test_incomplete_trial_rejected_by_merge(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED, complete=False)
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(trials, verify_repo_root=ROOT)


def test_aborted_trial_marked_complete_final_is_rejected_at_verification():
    doc = _trial(0, 0, 16, 500, complete=True, final=True, aborted=False)
    tampered = dict(doc, abort_status={"aborted": True, "reason": "guard"})
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.verify_trial_document(tampered, verify_repo_root=ROOT)


def test_non_finite_elapsed_time_rejected(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED)
    trials[0] = dict(trials[0])
    trials[0]["timing"] = dict(trials[0]["timing"], primary_elapsed_ns=math.inf)
    out = tmp_path / "output"
    out.mkdir()
    for t in trials:
        # json.dumps(allow_nan=True) emits the non-standard "Infinity"
        # token; json.loads parses it straight back to float("inf") --
        # exactly the on-disk tampering scenario this module must reject.
        (out / f"trial_{t['trial_index']}_repeat{t['repeat_index']}_chunk{t['chunk_size']}.json").write_text(json.dumps(t))
    (out / "merged_result.json").write_text(json.dumps({"raw_trials": trials}))
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_and_verify_benchmark_output(repo_root=ROOT, output_dir=out)


def test_negative_memory_rejected():
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        _trial(0, 0, 16, 500, host_peak=-1)


def test_non_finite_memory_rejected_on_verify(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED)
    tampered = dict(trials[0])
    tampered["memory"] = dict(tampered["memory"], host_process_lifetime_peak_rss_bytes="NaN")
    out = _write_output(tmp_path, [tampered] + trials[1:], merged_override={"raw_trials": [tampered] + trials[1:]})
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_and_verify_benchmark_output(repo_root=ROOT, output_dir=out)


def test_wrong_gpu_model_makes_candidate_unsafe(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED, gpu_model="NVIDIA A100")
    out = _write_output(tmp_path, trials)
    analysis = sel.analyze_benchmark(repo_root=ROOT, output_dir=out)
    for cs in sel.CANDIDATE_CHUNK_SIZES:
        assert analysis["safety"][cs]["safe"] is False
    assert analysis["winner"] is None


def test_wrong_software_record_rejected_by_reconstruction(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED)
    trials[0] = dict(trials[0], software_record=dict(trials[0]["software_record"], torch_version="9.9.9"))
    out = _write_output(tmp_path, trials, merged_override={"raw_trials": trials})
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_and_verify_benchmark_output(repo_root=ROOT, output_dir=out)


def test_wrong_repository_commit_rejected_by_reconstruction(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED)
    trials[0] = dict(trials[0], code_provenance=dict(trials[0]["code_provenance"], repository_commit="f" * 40))
    out = _write_output(tmp_path, trials, merged_override={"raw_trials": trials})
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_and_verify_benchmark_output(repo_root=ROOT, output_dir=out)


def test_wrong_authority_hash_rejected_by_reconstruction(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED)
    trials[0] = dict(trials[0], authority_sha256="9" * 64)
    out = _write_output(tmp_path, trials, merged_override={"raw_trials": trials})
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_and_verify_benchmark_output(repo_root=ROOT, output_dir=out)


def test_changed_scientific_digest_rejected(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED)
    trials[0] = dict(trials[0], scientific_output_digests={"collection_digest": "f" * 64, "per_window_digests": {}})
    out = _write_output(tmp_path, trials, merged_override={"raw_trials": trials})
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_and_verify_benchmark_output(repo_root=ROOT, output_dir=out)


def test_wrong_measured_window_count_rejected_at_reconstruction(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED)
    # Rebuild trial 0 with only 12 windows. Task 3's own merge_trials
    # requires window_identity to be identical across all six trials
    # (one fixed manifest) and rejects this outright; this IS the
    # correct, stronger rejection point for a wrong measured-window
    # count -- it never reaches Task 4's own eligibility check.
    trials[0] = _trial(0, 0, 16, _DEFAULT_ELAPSED[16][0], window_count=12)
    out = tmp_path / "output"
    out.mkdir()
    for t in trials:
        (out / f"trial_{t['trial_index']}_repeat{t['repeat_index']}_chunk{t['chunk_size']}.json").write_text(json.dumps(t))
    (out / "merged_result.json").write_text(json.dumps({"raw_trials": trials}))
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_and_verify_benchmark_output(repo_root=ROOT, output_dir=out)


def test_wrong_measured_window_count_marks_candidate_unsafe_when_schema_permits_it():
    # Directly exercises Task 4's OWN eligibility condition (not Task
    # 3's merge-time equality check) using compute_candidate_measurements/
    # evaluate_candidate_safety directly with a hand-built trial list
    # that never passes through merge_trials.
    trials = _six_trials(_DEFAULT_ELAPSED)
    trials[0] = _trial(0, 0, 16, _DEFAULT_ELAPSED[16][0], window_count=12)
    measurements = sel.compute_candidate_measurements(trials)
    safety = sel.evaluate_candidate_safety(
        trials, measurements, host_threshold_bytes=0.90 * 80 * (1024 ** 3), gpu_threshold_bytes=0.90 * 80 * (1024 ** 3),
        common_digest=COMMON_DIGEST,
    )
    assert safety[16]["safe"] is False
    assert any("window_count" in r for r in safety[16]["reasons"])
    winner, _, _ = sel.select_winner(measurements, safety)
    assert winner in (32, 64)


def test_tampered_stored_summary_with_untouched_raw_trials_detected(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED)
    real_merged = bench.merge_trials(trials, verify_repo_root=ROOT)
    tampered_merged = copy.deepcopy(real_merged)
    tampered_merged["descriptive_summaries_by_chunk_size"]["16"]["min_ns"] = 1
    out = _write_output(tmp_path, trials, merged_override=tampered_merged)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_and_verify_benchmark_output(repo_root=ROOT, output_dir=out)


def test_host_memory_exceeding_e17_threshold_marks_candidate_unsafe(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED, host_peak_map={16: (_HOST_UNSAFE, _HOST_UNSAFE)})
    out = _write_output(tmp_path, trials)
    analysis = sel.analyze_benchmark(repo_root=ROOT, output_dir=out)
    assert analysis["safety"][16]["safe"] is False
    assert "host_memory_exceeds_safety_threshold" in analysis["safety"][16]["reasons"]
    assert analysis["winner"] in (32, 64)


def test_gpu_memory_exceeding_e17_threshold_marks_candidate_unsafe(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED, gpu_peak_map={16: (_GPU_UNSAFE, _GPU_UNSAFE)})
    out = _write_output(tmp_path, trials)
    analysis = sel.analyze_benchmark(repo_root=ROOT, output_dir=out)
    assert analysis["safety"][16]["safe"] is False
    assert "gpu_memory_exceeds_safety_threshold" in analysis["safety"][16]["reasons"]


# ---------------------------------------------------------------------
# Section: selection logic (winner/tie-break/order-independence).
# ---------------------------------------------------------------------


def test_chunk_16_fastest_and_all_safe(tmp_path):
    trials = _six_trials({16: (100, 100), 32: (200, 200), 64: (300, 300)})
    out = _write_output(tmp_path, trials)
    analysis = sel.analyze_benchmark(repo_root=ROOT, output_dir=out)
    assert analysis["winner"] == 16


def test_chunk_32_fastest_and_all_safe(tmp_path):
    trials = _six_trials({16: (300, 300), 32: (100, 100), 64: (200, 200)})
    out = _write_output(tmp_path, trials)
    analysis = sel.analyze_benchmark(repo_root=ROOT, output_dir=out)
    assert analysis["winner"] == 32


def test_chunk_64_fastest_and_all_safe(tmp_path):
    trials = _six_trials({16: (300, 300), 32: (200, 200), 64: (100, 100)})
    out = _write_output(tmp_path, trials)
    analysis = sel.analyze_benchmark(repo_root=ROOT, output_dir=out)
    assert analysis["winner"] == 64


def test_fastest_candidate_unsafe_next_fastest_selected(tmp_path):
    # chunk 16 is fastest but exceeds the host-memory safety threshold.
    trials = _six_trials({16: (50, 50), 32: (200, 200), 64: (300, 300)}, host_peak_map={16: (_HOST_UNSAFE, _HOST_UNSAFE)})
    out = _write_output(tmp_path, trials)
    analysis = sel.analyze_benchmark(repo_root=ROOT, output_dir=out)
    assert analysis["safety"][16]["safe"] is False
    assert analysis["winner"] == 32


def test_only_one_safe_candidate(tmp_path):
    trials = _six_trials(
        _DEFAULT_ELAPSED,
        host_peak_map={32: (_HOST_UNSAFE, _HOST_UNSAFE), 64: (_HOST_UNSAFE, _HOST_UNSAFE)},
    )
    out = _write_output(tmp_path, trials)
    analysis = sel.analyze_benchmark(repo_root=ROOT, output_dir=out)
    assert analysis["eligible"] == [16]
    assert analysis["winner"] == 16


def test_no_safe_candidates(tmp_path):
    trials = _six_trials(
        _DEFAULT_ELAPSED,
        host_peak_map={16: (_HOST_UNSAFE, _HOST_UNSAFE), 32: (_HOST_UNSAFE, _HOST_UNSAFE), 64: (_HOST_UNSAFE, _HOST_UNSAFE)},
    )
    out = _write_output(tmp_path, trials)
    analysis = sel.analyze_benchmark(repo_root=ROOT, output_dir=out)
    assert analysis["eligible"] == []
    assert analysis["winner"] is None
    assert analysis["tie_break_path"] == "no_safe_candidate"


def test_exact_elapsed_time_tie_falls_through_to_host_memory():
    measurements = {
        16: {"mean_elapsed_ns": 1000.0, "max_host_process_lifetime_peak_rss_bytes": 500, "max_gpu_peak_reserved_bytes": 10},
        32: {"mean_elapsed_ns": 1000.0, "max_host_process_lifetime_peak_rss_bytes": 300, "max_gpu_peak_reserved_bytes": 10},
        64: {"mean_elapsed_ns": 2000.0, "max_host_process_lifetime_peak_rss_bytes": 100, "max_gpu_peak_reserved_bytes": 10},
    }
    safety = {cs: {"safe": True, "reasons": []} for cs in sel.CANDIDATE_CHUNK_SIZES}
    winner, path, eligible = sel.select_winner(measurements, safety)
    assert winner == 32
    assert path == "resolved_by_host_memory_tie_break"


def test_host_memory_tie_falls_through_to_gpu_memory():
    measurements = {
        16: {"mean_elapsed_ns": 1000.0, "max_host_process_lifetime_peak_rss_bytes": 500, "max_gpu_peak_reserved_bytes": 300},
        32: {"mean_elapsed_ns": 1000.0, "max_host_process_lifetime_peak_rss_bytes": 500, "max_gpu_peak_reserved_bytes": 100},
        64: {"mean_elapsed_ns": 2000.0, "max_host_process_lifetime_peak_rss_bytes": 100, "max_gpu_peak_reserved_bytes": 10},
    }
    safety = {cs: {"safe": True, "reasons": []} for cs in sel.CANDIDATE_CHUNK_SIZES}
    winner, path, eligible = sel.select_winner(measurements, safety)
    assert winner == 32
    assert path == "resolved_by_gpu_memory_tie_break"


def test_full_tie_falls_through_to_smaller_chunk_size():
    measurements = {
        16: {"mean_elapsed_ns": 1000.0, "max_host_process_lifetime_peak_rss_bytes": 500, "max_gpu_peak_reserved_bytes": 300},
        32: {"mean_elapsed_ns": 1000.0, "max_host_process_lifetime_peak_rss_bytes": 500, "max_gpu_peak_reserved_bytes": 300},
        64: {"mean_elapsed_ns": 2000.0, "max_host_process_lifetime_peak_rss_bytes": 100, "max_gpu_peak_reserved_bytes": 10},
    }
    safety = {cs: {"safe": True, "reasons": []} for cs in sel.CANDIDATE_CHUNK_SIZES}
    winner, path, eligible = sel.select_winner(measurements, safety)
    assert winner == 16
    assert path == "resolved_by_smaller_chunk_size_tie_break"


def test_no_rounding_before_selection_subtle_inequality():
    # 1/3 vs a value one float ULP below -- must not compare equal by
    # accidental rounding; the smaller one must win outright, no tie path.
    measurements = {
        16: {"mean_elapsed_ns": 1000.0000000001, "max_host_process_lifetime_peak_rss_bytes": 999, "max_gpu_peak_reserved_bytes": 999},
        32: {"mean_elapsed_ns": 1000.0, "max_host_process_lifetime_peak_rss_bytes": 1, "max_gpu_peak_reserved_bytes": 1},
        64: {"mean_elapsed_ns": 2000.0, "max_host_process_lifetime_peak_rss_bytes": 1, "max_gpu_peak_reserved_bytes": 1},
    }
    safety = {cs: {"safe": True, "reasons": []} for cs in sel.CANDIDATE_CHUNK_SIZES}
    winner, path, eligible = sel.select_winner(measurements, safety)
    assert winner == 32
    assert path == "unique_minimum_no_tie_break_needed"


@pytest.mark.parametrize("order_mode", ["schedule", "reversed", "shuffled"])
@pytest.mark.parametrize("hashseed", ["0", "1", "42"])
def test_selection_invariant_to_trial_order_and_hashseed(tmp_path, order_mode, hashseed, monkeypatch):
    trials = _six_trials(_DEFAULT_ELAPSED)
    if order_mode == "reversed":
        trials = list(reversed(trials))
    elif order_mode == "shuffled":
        trials = [trials[3], trials[0], trials[5], trials[1], trials[4], trials[2]]
    out = _write_output(tmp_path / order_mode, trials)

    script = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from pathlib import Path
import src.dfp_phase1b_chunk_selection as sel
analysis = sel.analyze_benchmark(repo_root=Path({str(ROOT)!r}), output_dir=Path({str(out)!r}))
print(analysis["winner"])
"""
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    result = sp.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True)
    assert result.stdout.strip() == "16"


def test_dict_key_order_independence():
    # measurements/safety built with keys inserted in a different order
    # must not change the outcome (CANDIDATE_CHUNK_SIZES iteration is
    # explicit, never relies on dict insertion order).
    m1 = {16: {"mean_elapsed_ns": 1.0, "max_host_process_lifetime_peak_rss_bytes": 1, "max_gpu_peak_reserved_bytes": 1},
          32: {"mean_elapsed_ns": 2.0, "max_host_process_lifetime_peak_rss_bytes": 1, "max_gpu_peak_reserved_bytes": 1},
          64: {"mean_elapsed_ns": 3.0, "max_host_process_lifetime_peak_rss_bytes": 1, "max_gpu_peak_reserved_bytes": 1}}
    m2 = {64: m1[64], 16: m1[16], 32: m1[32]}
    s = {cs: {"safe": True, "reasons": []} for cs in sel.CANDIDATE_CHUNK_SIZES}
    assert sel.select_winner(m1, s) == sel.select_winner(m2, s)


# ---------------------------------------------------------------------
# Section: independent test-side oracle for the selection rule --
# never calls sel.select_winner/compute_candidate_measurements. Built
# fresh from the six raw trial dicts, mirroring the spec's own prose
# rather than the production implementation.
# ---------------------------------------------------------------------


def _oracle_select(trials):
    by_chunk = {}
    for t in trials:
        by_chunk.setdefault(t["chunk_size"], []).append(t)
    means = {}
    host_max = {}
    gpu_max = {}
    safe = {}
    for cs, group in by_chunk.items():
        if len(group) != 2:
            safe[cs] = False
            continue
        by_repeat = sorted(group, key=lambda t: t["repeat_index"])
        e = [t["timing"]["primary_elapsed_ns"] for t in by_repeat]
        means[cs] = (e[0] + e[1]) / 2
        host_max[cs] = max(t["memory"]["host_process_lifetime_peak_rss_bytes"] for t in by_repeat)
        gpu_max[cs] = max(t["memory"]["gpu_peak_reserved_bytes"] for t in by_repeat)
        threshold = 0.90 * 80 * (1024 ** 3)
        ok = (
            all(t["complete"] and t["final"] and not t["abort_status"]["aborted"] for t in by_repeat)
            and all(len(t["window_identity"]) == 24 for t in by_repeat)
            and host_max[cs] <= threshold and gpu_max[cs] <= threshold
            and len({t["scientific_output_digests"]["collection_digest"] for t in by_repeat}) == 1
        )
        safe[cs] = ok
    eligible = [cs for cs in (16, 32, 64) if safe.get(cs)]
    if not eligible:
        return None
    best = min(eligible, key=lambda cs: (means[cs], host_max[cs], gpu_max[cs], cs))
    return best


@pytest.mark.parametrize("elapsed", [
    {16: (100, 100), 32: (200, 200), 64: (300, 300)},
    {16: (300, 300), 32: (100, 100), 64: (200, 200)},
    {16: (300, 300), 32: (200, 200), 64: (100, 100)},
    {16: (490611161126, 490611161127), 32: (522828581988, 522828581989), 64: (524774010415, 524774010416)},
])
def test_independent_oracle_matches_production_selection(tmp_path, elapsed):
    trials = _six_trials(elapsed)
    oracle_winner = _oracle_select(trials)
    measurements = sel.compute_candidate_measurements(trials)
    common_digest = trials[0]["scientific_output_digests"]["collection_digest"]
    safety = sel.evaluate_candidate_safety(
        trials, measurements, host_threshold_bytes=0.90 * 80 * (1024 ** 3), gpu_threshold_bytes=0.90 * 80 * (1024 ** 3),
        common_digest=common_digest,
    )
    production_winner, _, _ = sel.select_winner(measurements, safety)
    assert production_winner == oracle_winner


# ---------------------------------------------------------------------
# Section: artifact behavior.
# ---------------------------------------------------------------------


def _real_analysis():
    if not _REAL_OUTPUT_AVAILABLE:
        pytest.skip("no real output")
    return sel.analyze_benchmark(repo_root=ROOT, output_dir=REAL_OUTPUT_DIR)


@requires_real_output
def test_selection_artifact_is_deterministic():
    analysis = _real_analysis()
    a1 = sel.build_selection_artifact(analysis, e18_sha256="a" * 64, e17_sha256="y", manifest_sha256="z", repository_commit="c" * 40, analysis_generation_commit="d" * 40, gpu_execution_performed_by_task4=False, command="cmd")
    a2 = sel.build_selection_artifact(analysis, e18_sha256="a" * 64, e17_sha256="y", manifest_sha256="z", repository_commit="c" * 40, analysis_generation_commit="d" * 40, gpu_execution_performed_by_task4=False, command="cmd")
    assert json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True)


@requires_real_output
def test_markdown_report_is_deterministic():
    analysis = _real_analysis()
    artifact = sel.build_selection_artifact(analysis, e18_sha256="a" * 64, e17_sha256="y", manifest_sha256="z", repository_commit="c" * 40, analysis_generation_commit="d" * 40, gpu_execution_performed_by_task4=False, command="cmd")
    md1 = sel.render_markdown_report(artifact)
    md2 = sel.render_markdown_report(artifact)
    assert md1 == md2
    assert "16" in md1 and "Selected chunk size" in md1


@requires_real_output
def test_write_selection_artifact_no_overwrite_by_default(tmp_path):
    analysis = _real_analysis()
    artifact = sel.build_selection_artifact(analysis, e18_sha256="a" * 64, e17_sha256="y", manifest_sha256="z", repository_commit="c" * 40, analysis_generation_commit="d" * 40, gpu_execution_performed_by_task4=False, command="cmd")
    sel.write_selection_artifact_atomically(artifact, output_dir=tmp_path, basename="phase1b_chunk_selection.json")
    dest = tmp_path / "phase1b_chunk_selection.json"
    original_mtime = dest.stat().st_mtime_ns
    original_text = dest.read_text()
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.write_selection_artifact_atomically(dict(artifact, extra="x"), output_dir=tmp_path, basename="phase1b_chunk_selection.json")
    assert dest.stat().st_mtime_ns == original_mtime
    assert dest.read_text() == original_text  # existing destination preserved on failure


@requires_real_output
def test_write_selection_artifact_atomic_no_partial_output_on_construction_failure(tmp_path):
    with pytest.raises(TypeError):
        sel.write_selection_artifact_atomically({"bad": {1, 2, 3}}, output_dir=tmp_path, basename="phase1b_chunk_selection.json")  # a set() is not JSON-serializable
    assert not (tmp_path / "phase1b_chunk_selection.json").exists()
    leftover_tmp = list(tmp_path.glob(".*.tmp"))
    assert leftover_tmp == []


@requires_real_output
def test_write_selection_artifact_allow_overwrite(tmp_path):
    analysis = _real_analysis()
    artifact = sel.build_selection_artifact(analysis, e18_sha256="a" * 64, e17_sha256="y", manifest_sha256="z", repository_commit="c" * 40, analysis_generation_commit="d" * 40, gpu_execution_performed_by_task4=False, command="cmd")
    sel.write_selection_artifact_atomically(artifact, output_dir=tmp_path, basename="phase1b_chunk_selection.json")
    sel.write_selection_artifact_atomically(dict(artifact, note="updated"), output_dir=tmp_path, basename="phase1b_chunk_selection.json", allow_overwrite=True)
    assert json.loads((tmp_path / "phase1b_chunk_selection.json").read_text())["note"] == "updated"


@requires_real_output
def test_analyze_does_not_modify_source_files(tmp_path):
    trials = _six_trials(_DEFAULT_ELAPSED)
    out = _write_output(tmp_path, trials)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    sel.analyze_benchmark(repo_root=ROOT, output_dir=out)
    after = {p.name: p.read_bytes() for p in out.iterdir()}
    assert before == after


# ---------------------------------------------------------------------
# Section: CLI.
# ---------------------------------------------------------------------


def _run_cli(*args, env=None):
    cmd = [sys.executable, str(CLI_PATH), *args]
    return sp.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)


def _copy_real_output_to(tmp_path):
    """Copies the real completed Task 3 benchmark output (six trials +
    merged_result.json) into a fresh tmp_path directory -- so tests
    that WRITE a preview artifact never touch the real
    /scratch production directory."""
    import shutil
    dest = tmp_path / "output_copy"
    dest.mkdir(exist_ok=True)
    for name in ("merged_result.json", "trial_0_repeat0_chunk16.json", "trial_1_repeat0_chunk32.json",
                 "trial_2_repeat0_chunk64.json", "trial_3_repeat1_chunk64.json", "trial_4_repeat1_chunk32.json",
                 "trial_5_repeat1_chunk16.json"):
        shutil.copy2(REAL_OUTPUT_DIR / name, dest / name)
    return dest


def test_cli_help_exits_0():
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "verify-protocol" in result.stdout


def test_cli_verify_protocol_help_exits_0():
    result = _run_cli("verify-protocol", "--help")
    assert result.returncode == 0


def test_cli_verify_protocol_imports_no_torch():
    script = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import runpy
sys.argv = ["analyze_dfp_phase1b_chunk_benchmark.py", "verify-protocol"]
try:
    runpy.run_path({str(CLI_PATH)!r}, run_name="__main__")
except SystemExit as exc:
    assert exc.code in (0, None), f"verify-protocol failed: exit code {{exc.code}}"
assert "torch" not in sys.modules, "torch was imported by verify-protocol"
print("NO_TORCH_OK")
"""
    result = sp.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert "NO_TORCH_OK" in result.stdout, result.stderr + result.stdout


@requires_real_output
def test_cli_analyze_runs_no_inference_and_no_torch(tmp_path):
    script = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import runpy
sys.argv = ["analyze_dfp_phase1b_chunk_benchmark.py", "analyze", "--output-dir", {str(REAL_OUTPUT_DIR)!r}]
try:
    runpy.run_path({str(CLI_PATH)!r}, run_name="__main__")
except SystemExit as exc:
    assert exc.code in (0, None), f"analyze failed: exit code {{exc.code}}"
assert "torch" not in sys.modules
print("NO_TORCH_OK")
"""
    result = sp.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert "NO_TORCH_OK" in result.stdout, result.stderr + result.stdout


@requires_real_output
def test_cli_analyze_selects_chunk_16():
    result = _run_cli("analyze", "--output-dir", str(REAL_OUTPUT_DIR))
    assert result.returncode == 0
    assert "selected_chunk_size=16" in result.stdout


@requires_real_output
def test_cli_verify_selection_passes():
    result = _run_cli("verify-selection", "--output-dir", str(REAL_OUTPUT_DIR))
    assert result.returncode == 0
    assert "PASSES independent verification" in result.stdout


def test_cli_expected_failure_exits_2_no_traceback(tmp_path):
    missing = tmp_path / "does_not_exist"
    result = _run_cli("analyze", "--output-dir", str(missing))
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "DFP PHASE 1B CHUNK SELECTION FAIL" in result.stderr


def test_cli_no_overwrite_by_default(tmp_path):
    if not _REAL_OUTPUT_AVAILABLE:
        pytest.skip("no real output")
    out_dir = _copy_real_output_to(tmp_path)
    dest = out_dir / "phase1b_chunk_selection.json"
    dest.write_text('{"pre-existing": true}')
    result = _run_cli("analyze", "--output-dir", str(out_dir), "--output", "phase1b_chunk_selection.json")
    assert result.returncode == 2
    assert json.loads(dest.read_text()) == {"pre-existing": True}


def test_cli_no_gpu_flag_exists():
    result = _run_cli("analyze", "--help")
    assert "--device" not in result.stdout
    assert "--gpu" not in result.stdout
    result2 = _run_cli("--help")
    assert "5000" not in result2.stdout and "5,000" not in result2.stdout


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_fatal_exceptions_propagate_through_cli_dispatch(monkeypatch, exc_type):
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import importlib

    cli_mod = importlib.import_module("analyze_dfp_phase1b_chunk_benchmark")

    def _boom(_args):
        raise exc_type("boom")

    monkeypatch.setitem(cli_mod._DISPATCH, "analyze", _boom)
    if exc_type is SystemExit:
        with pytest.raises(SystemExit) as excinfo:
            cli_mod.main(["analyze", "--output-dir", "/nonexistent"])
        assert not isinstance(excinfo.value.code, int) or excinfo.value.code not in (0, 2)
    else:
        with pytest.raises(exc_type):
            cli_mod.main(["analyze", "--output-dir", "/nonexistent"])


# ---------------------------------------------------------------------
# Section: Findings-repair -- canonical relational-comparison matrix.
# Every mutation below is independently authored against the REAL
# committed E18 identity text (never the production selector's own
# fixtures), covering every persisted source-derived field for every
# candidate, plus cross-cutting top-level fields. Each must be
# rejected by verify_selection via build_expected_selection_result's
# recomputation -- not merely by formatting/schema checks.
# ---------------------------------------------------------------------


def _e18_text():
    return IDENTITY_PATH.read_text()


def _assert_mutation_rejected(tmp_path, name, find, replace, *, n=1):
    text = _e18_text()
    assert find in text, f"fixture assumption stale for {name!r}: {find!r} not found in committed E18"
    mutated = text.replace(find, replace, n)
    assert mutated != text
    p = tmp_path / f"{name}.toml"
    p.write_text(mutated)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.verify_selection(repo_root=ROOT, e18_path=p, output_dir=REAL_OUTPUT_DIR)


@requires_real_output
@pytest.mark.parametrize("chunk,find,replace", [
    (16, "repeat0_primary_elapsed_ns = 532762109651", "repeat0_primary_elapsed_ns = 1"),
    (16, "repeat1_primary_elapsed_ns = 448460212602", "repeat1_primary_elapsed_ns = 1"),
    (16, "max_host_process_lifetime_peak_rss_bytes = 15295516672", "max_host_process_lifetime_peak_rss_bytes = 1"),
    (16, "max_gpu_peak_reserved_bytes = 8006926336", "max_gpu_peak_reserved_bytes = 1"),
    (32, "repeat0_primary_elapsed_ns = 609678129698", "repeat0_primary_elapsed_ns = 1"),
    (32, "repeat1_primary_elapsed_ns = 435979034279", "repeat1_primary_elapsed_ns = 1"),
    (32, "mean_elapsed_ns = 522828581988.5", "mean_elapsed_ns = 1.0"),
    (32, "max_host_process_lifetime_peak_rss_bytes = 27382038528", "max_host_process_lifetime_peak_rss_bytes = 1"),
    (64, "repeat0_primary_elapsed_ns = 584577861113", "repeat0_primary_elapsed_ns = 1"),
    (64, "repeat1_primary_elapsed_ns = 464970159718", "repeat1_primary_elapsed_ns = 1"),
    (64, "mean_elapsed_ns = 524774010415.5", "mean_elapsed_ns = 1.0"),
    (64, "max_host_process_lifetime_peak_rss_bytes = 51819261952", "max_host_process_lifetime_peak_rss_bytes = 1"),
])
def test_supporting_evidence_mutation_rejected_winner_unaffected(tmp_path, chunk, find, replace):
    """These mutations never make the mutated candidate faster than
    chunk 16 -- the TRUE winner would still be 16 even if the mutated
    value were trusted. This is supporting-EVIDENCE integrity, not
    merely winner integrity: the mutation must still be rejected."""
    _assert_mutation_rejected(tmp_path, f"c{chunk}_{find[:12]}", find, replace)


@requires_real_output
def test_c16_mean_mutation_rejected_would_have_changed_winner(tmp_path):
    # chunk 16's own mean, if trusted, is already the winner's own
    # value -- mutating it directly probes the top candidate record.
    find = "mean_elapsed_ns = 490611161126.5"
    text = _e18_text()
    # first occurrence is the top-level selected_mean_elapsed_ns; second is chunk_16's candidate mean.
    idx = text.index(find)
    idx = text.index(find, idx + 1)
    mutated = text[:idx] + "mean_elapsed_ns = 1.0" + text[idx + len(find):]
    p = tmp_path / "c16_mean.toml"
    p.write_text(mutated)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.verify_selection(repo_root=ROOT, e18_path=p, output_dir=REAL_OUTPUT_DIR)


@requires_real_output
@pytest.mark.parametrize("find,replace", [
    ("selected_chunk_size = 16", "selected_chunk_size = 32"),
    ("selected_mean_elapsed_ns = 490611161126.5", "selected_mean_elapsed_ns = 1.0"),
    ('scientific_output_digest = "0ab0dcf6db4fca0774825b70767d64de5a7d416576071e82faaa1133fabf7bbf"',
     'scientific_output_digest = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"'),
    ('''tie_break_sequence = [
    "lower_max_host_process_lifetime_peak_rss_bytes",
    "lower_max_gpu_peak_reserved_bytes",
    "smaller_numerical_chunk_size",
]''', '''tie_break_sequence = [
    "smaller_numerical_chunk_size",
    "lower_max_host_process_lifetime_peak_rss_bytes",
    "lower_max_gpu_peak_reserved_bytes",
]'''),
])
def test_top_level_decision_mutation_rejected(tmp_path, find, replace):
    _assert_mutation_rejected(tmp_path, "top_level", find, replace)


@requires_real_output
def test_safety_flag_mutation_rejected_for_chunk64():
    text = _e18_text()
    block = ("[selection_result.candidates.chunk_64]\nchunk_size = 64\nrepeat0_trial_index = 2\nrepeat1_trial_index = 3\n"
             "repeat0_primary_elapsed_ns = 584577861113\nrepeat1_primary_elapsed_ns = 464970159718\n"
             "mean_elapsed_ns = 524774010415.5\nmax_host_process_lifetime_peak_rss_bytes = 51819261952\n"
             "max_gpu_peak_reserved_bytes = 7998537728\nsafe = true")
    assert block in text
    mutated = text.replace(block, block.replace("safe = true", "safe = false"), 1)
    p = Path("/tmp") / "c64_safe_probe.toml"
    p.write_text(mutated)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.verify_selection(repo_root=ROOT, e18_path=p, output_dir=REAL_OUTPUT_DIR)
    p.unlink()


@requires_real_output
def test_safety_violations_mutation_rejected():
    _assert_mutation_rejected(Path("/tmp"), "safety_violations_probe", "safety_violations = []", "safety_violations = [\"fake\"]", n=1)


@requires_real_output
def test_error_message_names_the_offending_field(tmp_path):
    p = tmp_path / "named_field.toml"
    p.write_text(_e18_text().replace("max_gpu_peak_reserved_bytes = 7998537728\nsafe = true\nsafety_violations = []\n\n[selection_result.candidates.chunk_64]",
                                      "max_gpu_peak_reserved_bytes = 1\nsafe = true\nsafety_violations = []\n\n[selection_result.candidates.chunk_64]", 1))
    with pytest.raises(sel.DfpPhase1bChunkSelectionError, match=r"chunk_32\.max_gpu_peak_reserved_bytes"):
        sel.verify_selection(repo_root=ROOT, e18_path=p, output_dir=REAL_OUTPUT_DIR)


# ---------------------------------------------------------------------
# Section: E18 candidate-field schema-level type checks (new in the
# findings repair -- these fields previously had no type validation at
# all).
# ---------------------------------------------------------------------


@requires_real_output
@pytest.mark.parametrize("find,replace,label", [
    ("repeat0_primary_elapsed_ns = 532762109651", "repeat0_primary_elapsed_ns = 532762109651.0", "float-not-int"),
    ("repeat0_primary_elapsed_ns = 532762109651", "repeat0_primary_elapsed_ns = -1", "negative"),
    ("max_host_process_lifetime_peak_rss_bytes = 15295516672", "max_host_process_lifetime_peak_rss_bytes = true", "bool-as-int"),
    ("mean_elapsed_ns = 490611161126.5", "mean_elapsed_ns = nan", "NaN"),
    ("mean_elapsed_ns = 490611161126.5", "mean_elapsed_ns = inf", "Infinity"),
    ("safe = true\nsafety_violations = []\n\n[selection_result.candidates.chunk_32]",
     "safe = 1\nsafety_violations = []\n\n[selection_result.candidates.chunk_32]", "safe-bool-as-int"),
])
def test_e18_candidate_field_type_violations_rejected(tmp_path, find, replace, label):
    text = _e18_text()
    assert find in text, f"stale fixture for {label}"
    mutated = text.replace(find, replace, 1)
    p = tmp_path / f"type_{label}.toml"
    p.write_text(mutated)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_e18_identity(ROOT, path=p)


def test_e18_candidate_missing_field_rejected(tmp_path):
    text = _e18_text()
    find = "repeat0_primary_elapsed_ns = 532762109651\n"
    assert find in text
    mutated = text.replace(find, "", 1)
    p = tmp_path / "missing_field.toml"
    p.write_text(mutated)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_e18_identity(ROOT, path=p)


def test_e18_candidate_unknown_field_rejected(tmp_path):
    text = _e18_text()
    find = "[selection_result.candidates.chunk_16]\nchunk_size = 16"
    assert find in text
    mutated = text.replace(find, find + "\nunknown_extra_field = 1", 1)
    p = tmp_path / "unknown_field.toml"
    p.write_text(mutated)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.load_e18_identity(ROOT, path=p)


# ---------------------------------------------------------------------
# Section: Findings-repair -- output-path safety matrix (direct API).
# ---------------------------------------------------------------------


UNSAFE_BASENAMES = [
    "../escape.json", "../../escape.json", "subdir/escape.json", "/absolute/escape.json",
    "C:\\escape.json", "\\\\server\\share\\escape.json", "..\\escape.json",
    ".", "..", "", "   ", "evil\x00name.json",
]


@pytest.mark.parametrize("basename", UNSAFE_BASENAMES)
def test_resolve_safe_output_basename_rejects_unsafe(tmp_path, basename):
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.resolve_safe_output_basename(basename, output_dir=tmp_path, label="probe", required_suffix=".json")


def test_resolve_safe_output_basename_rejects_directory_target(tmp_path):
    (tmp_path / "adir.json").mkdir()
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.resolve_safe_output_basename("adir.json", output_dir=tmp_path, label="probe", required_suffix=".json")


def test_resolve_safe_output_basename_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / f"outside_{tmp_path.name}.json"
    outside.write_text("x")
    try:
        (tmp_path / "escape.json").symlink_to(outside)
        with pytest.raises(sel.DfpPhase1bChunkSelectionError):
            sel.resolve_safe_output_basename("escape.json", output_dir=tmp_path, label="probe", required_suffix=".json")
    finally:
        outside.unlink(missing_ok=True)


@pytest.mark.parametrize("basename", ["phase1b_chunk_selection.json", "run..v2.json", "a.b.c.json"])
def test_resolve_safe_output_basename_accepts_valid_names(tmp_path, basename):
    p = sel.resolve_safe_output_basename(basename, output_dir=tmp_path, label="probe", required_suffix=".json")
    assert p == tmp_path / basename
    assert p.parent == tmp_path


def test_resolve_safe_output_basename_rejects_wrong_suffix(tmp_path):
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.resolve_safe_output_basename("report.md", output_dir=tmp_path, label="probe", required_suffix=".json")


def test_double_dot_substring_not_confused_with_traversal_component(tmp_path):
    # "run..v2.json" contains the substring ".." but is NOT the exact
    # path component ".." -- must be accepted.
    p = sel.resolve_safe_output_basename("run..v2.json", output_dir=tmp_path, label="probe", required_suffix=".json")
    assert p.name == "run..v2.json"


@pytest.mark.parametrize("bad_output_dir_suffix", ["/../escape", "/a/../../escape"])
def test_resolve_output_dir_rejects_lexical_traversal(tmp_path, bad_output_dir_suffix):
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.resolve_output_dir(str(tmp_path) + bad_output_dir_suffix)


def test_resolve_output_dir_rejects_nul():
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.resolve_output_dir("/tmp/evil\x00dir")


def test_resolve_output_dir_rejects_nonexistent():
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.resolve_output_dir("/tmp/definitely_does_not_exist_xyz123")


def test_resolve_output_dir_rejects_file_not_directory(tmp_path):
    f = tmp_path / "a_file"
    f.write_text("x")
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.resolve_output_dir(f)


def test_resolve_output_dir_accepts_valid_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = sel.resolve_output_dir(".")
    assert r == tmp_path.resolve()


def test_install_bundle_rejects_identical_destinations(tmp_path):
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.install_selection_bundle({"a": 1}, "md", output_dir=tmp_path, json_basename="same.json", markdown_basename="same.json")


def test_install_bundle_rejects_wrong_suffix_for_markdown(tmp_path):
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.install_selection_bundle({"a": 1}, "md", output_dir=tmp_path, json_basename="a.json", markdown_basename="a.json")


def test_direct_writer_traversal_rejected_without_cli(tmp_path):
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.write_selection_artifact_atomically({"a": 1}, output_dir=tmp_path, basename="../escape.json")
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.write_markdown_report("x", output_dir=tmp_path, basename="../escape.md")


def test_unsafe_destination_creates_no_directory_or_file(tmp_path):
    before = set(tmp_path.iterdir())
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.write_selection_artifact_atomically({"a": 1}, output_dir=tmp_path, basename="../escape.json")
    after = set(tmp_path.iterdir())
    assert before == after
    escaped = tmp_path.parent / "escape.json"
    assert not escaped.exists()


# ---------------------------------------------------------------------
# Section: atomic two-file installation.
# ---------------------------------------------------------------------


def test_install_bundle_atomic_pair(tmp_path):
    j, m = sel.install_selection_bundle({"a": 1}, "md text", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md")
    assert j.exists() and m.exists()
    assert j.read_text().strip().startswith("{")
    assert m.read_text() == "md text"


def test_install_bundle_no_overwrite_leaves_both_untouched(tmp_path):
    sel.install_selection_bundle({"a": 1}, "md1", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md")
    j_before = (tmp_path / "x.json").read_bytes()
    m_before = (tmp_path / "x.md").read_bytes()
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.install_selection_bundle({"a": 2}, "md2", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md")
    assert (tmp_path / "x.json").read_bytes() == j_before
    assert (tmp_path / "x.md").read_bytes() == m_before


def test_install_bundle_failure_on_json_generation_installs_neither(tmp_path):
    with pytest.raises(TypeError):
        sel.install_selection_bundle({"bad": {1, 2, 3}}, "md text", output_dir=tmp_path, json_basename="y.json", markdown_basename="y.md")
    assert not (tmp_path / "y.json").exists()
    assert not (tmp_path / "y.md").exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_install_bundle_failure_one_destination_unsafe_installs_neither(tmp_path):
    before = set(tmp_path.iterdir())
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.install_selection_bundle({"a": 1}, "md text", output_dir=tmp_path, json_basename="z.json", markdown_basename="../escape.md")
    after = set(tmp_path.iterdir())
    assert before == after


def test_install_bundle_no_tmp_residue_after_success(tmp_path):
    sel.install_selection_bundle({"a": 1}, "md text", output_dir=tmp_path, json_basename="w.json", markdown_basename="w.md")
    assert list(tmp_path.glob(".*.tmp")) == []


# ---------------------------------------------------------------------
# Section: CLI output-path adversarial matrix.
# ---------------------------------------------------------------------


@requires_real_output
@pytest.mark.parametrize("bad_basename", [
    "../escape.json", "../../escape.json", "subdir/escape.json", "/absolute/escape.json",
    "C:\\escape.json", "..\\escape.json", ".", "..",
])
def test_cli_output_rejects_unsafe_basename(tmp_path, bad_basename):
    out_dir = _copy_real_output_to(tmp_path)
    result = _run_cli("analyze", "--output-dir", str(out_dir), "--output", bad_basename)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "DFP PHASE 1B CHUNK SELECTION FAIL" in result.stderr


@requires_real_output
def test_cli_output_empty_basename_rejected(tmp_path):
    out_dir = _copy_real_output_to(tmp_path)
    result = _run_cli("analyze", "--output-dir", str(out_dir), "--output", "")
    assert result.returncode == 2


@requires_real_output
def test_cli_markdown_output_traversal_rejected(tmp_path):
    out_dir = _copy_real_output_to(tmp_path)
    result = _run_cli("analyze", "--output-dir", str(out_dir), "--output", "ok.json", "--markdown-output", "../escape.md")
    assert result.returncode == 2
    assert not (out_dir.parent / "escape.md").exists()
    assert not (out_dir / "ok.json").exists()  # bundle install rejected before either write


@requires_real_output
def test_cli_valid_basenames_accepted(fixture_clone, tmp_path):
    # analyze --output now requires a clean, committed worktree
    # (Findings-repair round 2, Finding 2) to resolve
    # analysis_generation_commit -- the REAL repository is legitimately
    # dirty during active development, so artifact-GENERATION CLI tests
    # use the isolated, disposable committed clone instead (same
    # pattern as the E18-hash/generation-commit tests above).
    clone_root, commit = fixture_clone
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    import shutil as _shutil
    for f in REAL_OUTPUT_DIR.glob("trial_*.json"):
        _shutil.copy2(f, out_dir / f.name)
    _shutil.copy2(REAL_OUTPUT_DIR / "merged_result.json", out_dir / "merged_result.json")
    cli_path = clone_root / "diagnostics/analyze_dfp_phase1b_chunk_benchmark.py"
    result = sp.run(
        [sys.executable, str(cli_path), "--repo-root", str(clone_root), "analyze", "--output-dir", str(out_dir),
         "--output", "phase1b_chunk_selection.json", "--markdown-output", "phase1b_chunk_selection_report.md"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "phase1b_chunk_selection.json").exists()
    assert (out_dir / "phase1b_chunk_selection_report.md").exists()


@requires_real_output
def test_cli_double_dot_substring_basename_accepted(fixture_clone, tmp_path):
    clone_root, commit = fixture_clone
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    import shutil as _shutil
    for f in REAL_OUTPUT_DIR.glob("trial_*.json"):
        _shutil.copy2(f, out_dir / f.name)
    _shutil.copy2(REAL_OUTPUT_DIR / "merged_result.json", out_dir / "merged_result.json")
    cli_path = clone_root / "diagnostics/analyze_dfp_phase1b_chunk_benchmark.py"
    result = sp.run(
        [sys.executable, str(cli_path), "--repo-root", str(clone_root), "analyze", "--output-dir", str(out_dir), "--output", "run..v2.json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "run..v2.json").exists()


# =======================================================================
# Findings-repair round 2: E18-identity-hash / analysis_generation_commit
# / no-GPU-disclosure adversarial tests.
#
# All generation tests use an ISOLATED, disposable local git clone
# (git objects are content-addressed, so a local clone touches nothing
# in the real repository) with the CURRENT in-progress repaired files
# copied in and committed as one synthetic descendant commit, and a
# fixture-local E18 with ONLY its selector/analysis-CLI hashes patched
# to match the fixture's own file bytes -- never the real repository's
# E18. This is exactly the "temporary committed clone" strategy the
# repair task itself requires: the durable /scratch artifacts are
# never touched, and generation never runs against a dirty tree.
# =======================================================================

import contextlib as _contextlib


def _clean_committed_clone(dest_dir):
    """Clones ROOT, overlays the CURRENT (possibly in-progress/dirty)
    Task 4 implementation files, patches a fixture-local copy of E18's
    selector/CLI dependency hashes to match those overlaid bytes, and
    commits everything as one disposable synthetic commit. Returns
    (clone_root, commit_sha)."""
    clone_dir = dest_dir / "clone"
    result = sp.run(["git", "clone", "-q", str(ROOT), str(clone_dir)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"git clone failed: {result.stderr}"
    for rel in (
        "src/dfp_phase1b_chunk_selection.py",
        "diagnostics/analyze_dfp_phase1b_chunk_benchmark.py",
        "tests/test_dfp_phase1b_chunk_selection.py",
        "docs/dfp_phase1b_chunk_selection.md",
        "evaluation_identities/e18_dfp_phase1b_chunk_selection.toml",
    ):
        dest = clone_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel).read_bytes())

    import tomllib as _tomllib
    e18_path = clone_dir / "evaluation_identities/e18_dfp_phase1b_chunk_selection.toml"
    doc = _tomllib.load(open(e18_path, "rb"))
    old_sel = doc["dependency_bindings"]["selector_module"]["sha256"]
    old_cli = doc["dependency_bindings"]["analysis_cli_module"]["sha256"]
    new_sel = hashlib.sha256((clone_dir / "src/dfp_phase1b_chunk_selection.py").read_bytes()).hexdigest()
    new_cli = hashlib.sha256((clone_dir / "diagnostics/analyze_dfp_phase1b_chunk_benchmark.py").read_bytes()).hexdigest()
    text = e18_path.read_text()
    assert text.count(old_sel) == 1 and text.count(old_cli) == 1
    e18_path.write_text(text.replace(old_sel, new_sel, 1).replace(old_cli, new_cli, 1))

    sp.run(["git", "-C", str(clone_dir), "config", "user.email", "dfp-test-fixture@example.invalid"], check=True)
    sp.run(["git", "-C", str(clone_dir), "config", "user.name", "DFP Test Fixture"], check=True)
    sp.run(["git", "-C", str(clone_dir), "add", "-A"], check=True)
    commit_result = sp.run(["git", "-C", str(clone_dir), "commit", "-q", "-m", "fixture: disposable synthetic Task4 repair commit"],
                            capture_output=True, text=True)
    assert commit_result.returncode == 0, f"fixture commit failed: {commit_result.stderr}"
    commit = sp.run(["git", "-C", str(clone_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    status = sp.run(["git", "-C", str(clone_dir), "status", "--porcelain"], capture_output=True, text=True, check=True)
    assert status.stdout.strip() == ""
    return clone_dir, commit


@pytest.fixture(scope="module")
def fixture_clone(tmp_path_factory):
    dest = tmp_path_factory.mktemp("dfp_task4_fixture_clone")
    clone_dir, commit = _clean_committed_clone(dest)
    return clone_dir, commit


@_contextlib.contextmanager
def _fixture_dirty(clone_root, mutate_fn):
    mutate_fn(clone_root)
    try:
        yield
    finally:
        sp.run(["git", "-C", str(clone_root), "reset", "-q", "--hard", "HEAD"], check=True)
        sp.run(["git", "-C", str(clone_root), "clean", "-q", "-fd"], check=True)


def _import_fixture_sel(clone_root):
    """Imports the fixture clone's OWN copy of src.dfp_phase1b_chunk_selection
    as a fresh module (never the real repo's already-imported module),
    so tests exercise the exact code committed in the fixture."""
    import importlib
    import sys as _sys
    clone_root = str(clone_root)
    if clone_root not in _sys.path:
        _sys.path.insert(0, clone_root)
    # Force a fresh import bound to this clone's sys.path entry.
    for mod_name in ("src.dfp_phase1b_chunk_selection",):
        _sys.modules.pop(mod_name, None)
    return importlib.import_module("src.dfp_phase1b_chunk_selection")


# ---------------------------------------------------------------------
# Section: E18 identity hash (Findings repair round 2, Finding 1).
# ---------------------------------------------------------------------


def test_e18_hash_exact_real_value_accepted(fixture_clone):
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)
    real_hash = hashlib.sha256((clone_root / fsel.IDENTITY_RELATIVE_PATH).read_bytes()).hexdigest()
    analysis = fsel.analyze_benchmark(repo_root=clone_root, output_dir=REAL_OUTPUT_DIR)
    e18 = fsel.load_e18_identity(clone_root)
    artifact = fsel.build_selection_artifact(
        analysis, e18_sha256=real_hash, e17_sha256=e18["dependency_bindings"]["e17_identity"]["sha256"],
        manifest_sha256=e18["dependency_bindings"]["benchmark_manifest"]["sha256"],
        repository_commit=analysis["merged"]["code_provenance"]["repository_commit"],
        analysis_generation_commit=commit, gpu_execution_performed_by_task4=False, command="test",
    )
    assert artifact["e18_identity_sha256"] == real_hash


@pytest.mark.parametrize("bad_hash", ["", "not-a-hash", "F" * 64, "a" * 63, "a" * 65])
def test_e18_hash_build_rejects_malformed_values(fixture_clone, bad_hash):
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)
    analysis = fsel.analyze_benchmark(repo_root=clone_root, output_dir=REAL_OUTPUT_DIR)
    e18 = fsel.load_e18_identity(clone_root)
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
        fsel.build_selection_artifact(
            analysis, e18_sha256=bad_hash, e17_sha256=e18["dependency_bindings"]["e17_identity"]["sha256"],
            manifest_sha256=e18["dependency_bindings"]["benchmark_manifest"]["sha256"],
            repository_commit=analysis["merged"]["code_provenance"]["repository_commit"],
            analysis_generation_commit=commit, gpu_execution_performed_by_task4=False, command="test",
        )


@pytest.mark.parametrize("wrong_hash_source", ["0" * 64, "wrong_file"])
def test_e18_hash_well_formed_but_incorrect_rejected_by_verifier(fixture_clone, tmp_path, wrong_hash_source):
    """A syntactically well-formed 64-hex value that is NOT the live
    E18 file's actual hash (either an all-zero placeholder or the real
    hash of a different, unrelated repository file) passes build-time
    FORMAT validation (build only checks shape) but must be rejected
    by the RELATIONAL verifier, which checks it against live bytes."""
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    if wrong_hash_source == "0" * 64:
        wrong_hash = "0" * 64
    else:
        wrong_hash = hashlib.sha256((clone_root / "src/dfp_phase1b_chunk_benchmark.py").read_bytes()).hexdigest()
    artifact["e18_identity_sha256"] = wrong_hash
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
        fsel.verify_selection_artifact(p, repo_root=clone_root)


def test_e18_bytes_tampered_stale_hash_rejected_by_verifier(fixture_clone, tmp_path):
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)
    real_hash = hashlib.sha256((clone_root / fsel.IDENTITY_RELATIVE_PATH).read_bytes()).hexdigest()
    analysis = fsel.analyze_benchmark(repo_root=clone_root, output_dir=REAL_OUTPUT_DIR)
    e18 = fsel.load_e18_identity(clone_root)
    artifact = fsel.build_selection_artifact(
        analysis, e18_sha256=real_hash, e17_sha256=e18["dependency_bindings"]["e17_identity"]["sha256"],
        manifest_sha256=e18["dependency_bindings"]["benchmark_manifest"]["sha256"],
        repository_commit=analysis["merged"]["code_provenance"]["repository_commit"],
        analysis_generation_commit=commit, gpu_execution_performed_by_task4=False, command="test",
    )
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact))
    # E18 bytes on disk change (tamper), but the artifact's recorded
    # hash still reflects the OLD bytes -- must be rejected.
    e18_path = clone_root / fsel.IDENTITY_RELATIVE_PATH
    original = e18_path.read_bytes()
    try:
        e18_path.write_bytes(original + b"\n# tampered\n")
        with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
            fsel.verify_selection_artifact(artifact_path, repo_root=clone_root)
    finally:
        e18_path.write_bytes(original)


def test_e18_stored_hash_tampered_bytes_unchanged_rejected(fixture_clone, tmp_path):
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)
    real_hash = hashlib.sha256((clone_root / fsel.IDENTITY_RELATIVE_PATH).read_bytes()).hexdigest()
    analysis = fsel.analyze_benchmark(repo_root=clone_root, output_dir=REAL_OUTPUT_DIR)
    e18 = fsel.load_e18_identity(clone_root)
    artifact = fsel.build_selection_artifact(
        analysis, e18_sha256=real_hash, e17_sha256=e18["dependency_bindings"]["e17_identity"]["sha256"],
        manifest_sha256=e18["dependency_bindings"]["benchmark_manifest"]["sha256"],
        repository_commit=analysis["merged"]["code_provenance"]["repository_commit"],
        analysis_generation_commit=commit, gpu_execution_performed_by_task4=False, command="test",
    )
    artifact["e18_identity_sha256"] = "f" * 64  # tamper the STORED hash; E18 bytes untouched
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
        fsel.verify_selection_artifact(artifact_path, repo_root=clone_root)


def test_e18_loader_failure_translated_through_domain_boundary(fixture_clone, tmp_path):
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)
    bad_e18 = tmp_path / "corrupt.toml"
    bad_e18.write_text("{not valid toml [[[")
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
        fsel.load_e18_identity(clone_root, path=bad_e18)


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_e18_load_fatal_exceptions_propagate(fixture_clone, exc_type, monkeypatch):
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)

    def poison_open(*a, **k):
        raise exc_type("probe")

    monkeypatch.setattr("builtins.open", poison_open)
    with pytest.raises(exc_type):
        fsel.load_e18_identity(clone_root)


def test_cli_e18_verify_protocol_from_arbitrary_cwd_sanitized_pythonpath(fixture_clone, tmp_path):
    clone_root, commit = fixture_clone
    cli_path = clone_root / "diagnostics/analyze_dfp_phase1b_chunk_benchmark.py"
    env = dict(PATH=__import__("os").environ["PATH"], HOME=__import__("os").environ.get("HOME", ""))
    result = sp.run(
        [sys.executable, str(cli_path), "--repo-root", str(clone_root), "verify-protocol"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "torch" not in result.stdout.lower() or "Traceback" not in result.stderr


# ---------------------------------------------------------------------
# Section: analysis_generation_commit (Findings repair round 2, Finding 2).
# ---------------------------------------------------------------------


def test_generation_commit_full_hex_accepted(fixture_clone):
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)
    e18 = fsel.load_e18_identity(clone_root)
    resolved = fsel.resolve_analysis_generation_commit(clone_root, e18_document=e18)
    assert resolved == commit
    assert len(resolved) == 40 and resolved == resolved.lower()


@pytest.mark.parametrize("bad_commit", [
    "a" * 39, "a" * 41, "A" * 40, "not-hex-not-hex-not-hex-not-hex-nothx12", "", "g" * 40,
])
def test_generation_commit_malformed_rejected_at_build(fixture_clone, bad_commit):
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)
    analysis = fsel.analyze_benchmark(repo_root=clone_root, output_dir=REAL_OUTPUT_DIR)
    e18 = fsel.load_e18_identity(clone_root)
    real_hash = hashlib.sha256((clone_root / fsel.IDENTITY_RELATIVE_PATH).read_bytes()).hexdigest()
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
        fsel.build_selection_artifact(
            analysis, e18_sha256=real_hash, e17_sha256=e18["dependency_bindings"]["e17_identity"]["sha256"],
            manifest_sha256=e18["dependency_bindings"]["benchmark_manifest"]["sha256"],
            repository_commit=analysis["merged"]["code_provenance"]["repository_commit"],
            analysis_generation_commit=bad_commit, gpu_execution_performed_by_task4=False, command="test",
        )


def test_generation_commit_nonexistent_rejected(fixture_clone, tmp_path):
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    artifact["analysis_generation_commit"] = "f" * 40  # well-formed, but not a real commit object
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
        fsel.verify_selection_artifact(p, repo_root=clone_root)


def test_generation_commit_wrong_ancestor_rejected(fixture_clone, tmp_path):
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)
    # Build a genuinely unrelated orphan commit in the SAME repo.
    orphan_dir = tmp_path / "orphan_check"
    sp.run(["git", "clone", "-q", str(clone_root), str(orphan_dir)], check=True)
    sp.run(["git", "-C", str(orphan_dir), "config", "user.email", "x@example.invalid"], check=True)
    sp.run(["git", "-C", str(orphan_dir), "config", "user.name", "X"], check=True)
    sp.run(["git", "-C", str(orphan_dir), "checkout", "-q", "--orphan", "unrelated"], check=True)
    sp.run(["git", "-C", str(orphan_dir), "rm", "-rf", "-q", "."], check=True)
    (orphan_dir / "z.txt").write_text("z")
    sp.run(["git", "-C", str(orphan_dir), "add", "z.txt"], check=True)
    sp.run(["git", "-C", str(orphan_dir), "commit", "-q", "-m", "unrelated root"], check=True)
    orphan_commit = sp.run(["git", "-C", str(orphan_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    # Fetch that orphan commit into clone_root so it's a real, resolvable object there too.
    sp.run(["git", "-C", str(clone_root), "fetch", "-q", str(orphan_dir), f"{orphan_commit}:refs/heads/orphan_probe"], check=True)
    e18 = fsel.load_e18_identity(clone_root)
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        fsel.verify_ancestor_relation(clone_root, required_ancestor_commit=e18["identity"]["required_ancestor_commit"], repository_commit=orphan_commit)


def test_benchmark_source_and_analysis_commits_swapped_rejected(fixture_clone, tmp_path):
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)
    analysis = fsel.analyze_benchmark(repo_root=clone_root, output_dir=REAL_OUTPUT_DIR)
    e18 = fsel.load_e18_identity(clone_root)
    real_hash = hashlib.sha256((clone_root / fsel.IDENTITY_RELATIVE_PATH).read_bytes()).hexdigest()
    artifact = fsel.build_selection_artifact(
        analysis, e18_sha256=real_hash, e17_sha256=e18["dependency_bindings"]["e17_identity"]["sha256"],
        manifest_sha256=e18["dependency_bindings"]["benchmark_manifest"]["sha256"],
        repository_commit=analysis["merged"]["code_provenance"]["repository_commit"],
        analysis_generation_commit=commit, gpu_execution_performed_by_task4=False, command="test",
    )
    artifact["repository_commit"], artifact["analysis_generation_commit"] = artifact["analysis_generation_commit"], artifact["repository_commit"]
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
        fsel.verify_selection_artifact(artifact_path, repo_root=clone_root)


def test_generation_rejected_when_repository_dirty(fixture_clone):
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)
    e18 = fsel.load_e18_identity(clone_root)
    with _fixture_dirty(clone_root, lambda root: (root / "dirty_sentinel_gen.txt").write_text("x")):
        with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
            fsel.resolve_analysis_generation_commit(clone_root, e18_document=e18)


def test_generation_rejected_when_repository_staged(fixture_clone):
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)
    e18 = fsel.load_e18_identity(clone_root)

    def stage(root):
        (root / "docs/dfp_phase1b_chunk_selection.md").write_text("dirty staged\n")
        sp.run(["git", "-C", str(root), "add", "docs/dfp_phase1b_chunk_selection.md"], check=True)

    with _fixture_dirty(clone_root, stage):
        with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
            fsel.resolve_analysis_generation_commit(clone_root, e18_document=e18)


def test_clean_committed_clone_generation_succeeds_and_verifies(fixture_clone, tmp_path):
    clone_root, commit = fixture_clone
    cli_path = clone_root / "diagnostics/analyze_dfp_phase1b_chunk_benchmark.py"
    preview = tmp_path / "preview"
    preview.mkdir()
    for name in ("merged_result.json",) + tuple(f"trial_{i}.json" for i in ()):
        pass
    import shutil as _shutil
    for f in REAL_OUTPUT_DIR.glob("trial_*.json"):
        _shutil.copy2(f, preview / f.name)
    _shutil.copy2(REAL_OUTPUT_DIR / "merged_result.json", preview / "merged_result.json")

    result = sp.run(
        [sys.executable, str(cli_path), "--repo-root", str(clone_root), "analyze",
         "--output-dir", str(preview), "--output", "phase1b_chunk_selection.json",
         "--markdown-output", "phase1b_chunk_selection_report.md"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    artifact = json.loads((preview / "phase1b_chunk_selection.json").read_text())
    assert artifact["analysis_generation_commit"] == commit
    assert artifact["selected_chunk_size"] == 16
    assert artifact["repository_commit"] != artifact["analysis_generation_commit"]

    verify_result = sp.run(
        [sys.executable, str(cli_path), "--repo-root", str(clone_root), "verify-selection", "--output-dir", str(preview)],
        capture_output=True, text=True,
    )
    assert verify_result.returncode == 0, verify_result.stderr


# =======================================================================
# Section: optional expected-generation-commit freshness check for
# verify_selection_artifact (Finding 2 from the independent verification
# of the installed pair -- validator hardening only, never touches
# production). All fixtures below are disposable tmp_path/fixture_clone
# state -- never the real production directory.
# =======================================================================


def _clone_parent_commit(clone_root):
    """The commit fixture_clone's clone was cloned FROM, before its own
    synthetic overlay commit -- i.e. a real, valid, ancestor-satisfying
    but now-superseded ("stale parent") commit, exactly mirroring the
    real e872da51.../1bf47e5d... relationship structurally without
    hardcoding real repo hashes."""
    return sp.run(["git", "-C", str(clone_root), "rev-parse", "HEAD^"], capture_output=True, text=True, check=True).stdout.strip()


def _make_dangling_descendant_commit(clone_root, parent_commit):
    """Creates a real, resolvable commit object descending from
    parent_commit via `git commit-tree` -- NEVER touches clone_root's
    HEAD, index, or working tree (no ref points at it; it is simply a
    valid, existing commit object usable as a substitution target)."""
    tree = sp.run(["git", "-C", str(clone_root), "rev-parse", f"{parent_commit}^{{tree}}"], capture_output=True, text=True, check=True).stdout.strip()
    result = sp.run(
        ["git", "-C", str(clone_root), "commit-tree", tree, "-p", parent_commit, "-m", "fixture: dangling descendant probe"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


# --- positive tests ---

def _real_installed_artifact_copy_with_live_e18_hash(tmp_path, name="installed_copy.json"):
    """Copies the REAL installed production artifact into a disposable
    tmp_path file, patching ONLY `e18_identity_sha256` to the actual
    current live E18 file hash. This is necessary because E18's own
    hash legitimately changes independently of when the durable
    production artifact is next regenerated (exactly the case in this
    repair task itself, which explicitly must not regenerate
    production) -- these freshness tests target
    `analysis_generation_commit` behavior specifically and must not be
    coupled to that separate, already out-of-scope staleness. The real
    production file is only ever read, never written."""
    real_artifact_path = REAL_OUTPUT_DIR / "phase1b_chunk_selection.json"
    artifact = json.loads(real_artifact_path.read_text())
    artifact["e18_identity_sha256"] = sel.sha256_file(ROOT / sel.IDENTITY_RELATIVE_PATH)
    p = tmp_path / name
    p.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return p, real_artifact_path


def test_freshness_real_installed_artifact_matches_actual_head(tmp_path):
    """Requirement 1/2: the REAL installed production artifact (read
    via a disposable tmp_path COPY, never mutated) plus its own actual
    analysis_generation_commit as the expected value passes."""
    if not _REAL_OUTPUT_AVAILABLE:
        pytest.skip("no real output")
    real_artifact_path = REAL_OUTPUT_DIR / "phase1b_chunk_selection.json"
    if not real_artifact_path.exists():
        pytest.skip("no installed production artifact present")
    production_hash_before = hashlib.sha256(real_artifact_path.read_bytes()).hexdigest()
    copy_path, _ = _real_installed_artifact_copy_with_live_e18_hash(tmp_path)
    real_commit = json.loads(copy_path.read_text())["analysis_generation_commit"]
    result = sel.verify_selection_artifact(copy_path, repo_root=ROOT, expected_analysis_generation_commit=real_commit)
    assert result["analysis_generation_commit"] == real_commit
    # production untouched by this read-only test -- only the disposable copy was read.
    assert hashlib.sha256(real_artifact_path.read_bytes()).hexdigest() == production_hash_before


def test_freshness_freshly_built_artifact_matches_its_own_commit(fixture_clone, tmp_path):
    """Requirement 2: a freshly constructed valid artifact plus its
    exact generation commit passes."""
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    result = fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit=artifact["analysis_generation_commit"])
    assert result["analysis_generation_commit"] == artifact["analysis_generation_commit"]


def test_freshness_omitted_argument_preserves_historical_behavior(fixture_clone, tmp_path):
    """Requirement 3: omitting the optional argument preserves existing
    historical-artifact behavior -- an artifact generated at the
    clone's real (stale-but-valid) PARENT commit still verifies when no
    expectation is supplied."""
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)
    parent_commit = _clone_parent_commit(clone_root)
    analysis = fsel.analyze_benchmark(repo_root=clone_root, output_dir=REAL_OUTPUT_DIR)
    e18 = fsel.load_e18_identity(clone_root)
    real_hash = hashlib.sha256((clone_root / fsel.IDENTITY_RELATIVE_PATH).read_bytes()).hexdigest()
    artifact = fsel.build_selection_artifact(
        analysis, e18_sha256=real_hash, e17_sha256=e18["dependency_bindings"]["e17_identity"]["sha256"],
        manifest_sha256=e18["dependency_bindings"]["benchmark_manifest"]["sha256"],
        repository_commit=analysis["merged"]["code_provenance"]["repository_commit"],
        analysis_generation_commit=parent_commit, gpu_execution_performed_by_task4=False, command="test",
    )
    p = tmp_path / "historical.json"
    p.write_text(json.dumps(artifact))
    result = fsel.verify_selection_artifact(p, repo_root=clone_root)  # no expected_ argument at all
    assert result["analysis_generation_commit"] == parent_commit


def test_freshness_explicit_none_matches_omission(fixture_clone, tmp_path):
    """Requirement 4: explicit None behaves the same as omission."""
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)
    parent_commit = _clone_parent_commit(clone_root)
    analysis = fsel.analyze_benchmark(repo_root=clone_root, output_dir=REAL_OUTPUT_DIR)
    e18 = fsel.load_e18_identity(clone_root)
    real_hash = hashlib.sha256((clone_root / fsel.IDENTITY_RELATIVE_PATH).read_bytes()).hexdigest()
    artifact = fsel.build_selection_artifact(
        analysis, e18_sha256=real_hash, e17_sha256=e18["dependency_bindings"]["e17_identity"]["sha256"],
        manifest_sha256=e18["dependency_bindings"]["benchmark_manifest"]["sha256"],
        repository_commit=analysis["merged"]["code_provenance"]["repository_commit"],
        analysis_generation_commit=parent_commit, gpu_execution_performed_by_task4=False, command="test",
    )
    p = tmp_path / "historical_explicit_none.json"
    p.write_text(json.dumps(artifact))
    result = fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit=None)
    assert result["analysis_generation_commit"] == parent_commit


def test_freshness_expected_value_not_derived_from_artifact(fixture_clone, tmp_path):
    """Requirement 5: the expected value is never read from the
    artifact -- supplying a DIFFERENT (but itself valid) commit than
    the artifact's own must be rejected, proving the comparison uses
    the caller's value, not something silently pulled from the
    artifact under test."""
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    parent_commit = _clone_parent_commit(clone_root)
    assert parent_commit != artifact["analysis_generation_commit"]
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError, match="does not exactly match"):
        fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit=parent_commit)


def test_freshness_other_checks_still_execute_when_freshness_satisfied(fixture_clone, tmp_path):
    """Requirement 6: existing structural/dependency/selection/
    provenance checks still execute even when the (matching) freshness
    check would otherwise pass -- a corrupted digest is still caught."""
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    artifact["common_scientific_output_digest"] = "1" * 64
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
        fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit=artifact["analysis_generation_commit"])


# --- negative tests ---

def test_freshness_exact_former_adversarial_case_now_rejected(tmp_path):
    """The exact case from the independent verification finding: the
    REAL installed artifact's analysis_generation_commit replaced with
    its valid parent commit, verified with the ACTUAL expected commit
    supplied -- must change from accepted (pre-fix) to rejected."""
    if not _REAL_OUTPUT_AVAILABLE:
        pytest.skip("no real output")
    real_artifact_path = REAL_OUTPUT_DIR / "phase1b_chunk_selection.json"
    if not real_artifact_path.exists():
        pytest.skip("no installed production artifact present")
    production_hash_before = hashlib.sha256(real_artifact_path.read_bytes()).hexdigest()
    artifact = json.loads(real_artifact_path.read_text())
    # See _real_installed_artifact_copy_with_live_e18_hash: E18's own
    # hash can legitimately differ from what the not-yet-regenerated
    # production artifact recorded -- irrelevant to this specific
    # freshness-check test, so pinned to the current live value.
    artifact["e18_identity_sha256"] = sel.sha256_file(ROOT / sel.IDENTITY_RELATIVE_PATH)
    real_commit = artifact["analysis_generation_commit"]
    parent_result = sp.run(["git", "-C", str(ROOT), "rev-parse", f"{real_commit}^"], capture_output=True, text=True)
    if parent_result.returncode != 0:
        pytest.skip("cannot resolve real HEAD's parent in this environment")
    parent_commit = parent_result.stdout.strip()
    artifact["analysis_generation_commit"] = parent_commit
    p = tmp_path / "stale_parent.json"
    p.write_text(json.dumps(artifact))

    # Without the expected argument: still accepted (unchanged default behavior).
    sel.verify_selection_artifact(p, repo_root=ROOT)

    # With the actual expected commit supplied: now rejected.
    with pytest.raises(sel.DfpPhase1bChunkSelectionError, match="does not exactly match"):
        sel.verify_selection_artifact(p, repo_root=ROOT, expected_analysis_generation_commit=real_commit)

    # Production itself was never touched by this test -- only the
    # disposable tmp_path copy was ever mutated.
    assert hashlib.sha256(real_artifact_path.read_bytes()).hexdigest() == production_hash_before


def test_freshness_artifact_stale_expected_fresh_rejected(fixture_clone, tmp_path):
    """Negative case 1: artifact says the stale parent; expected is the
    fresh (child) commit."""
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)
    parent_commit = _clone_parent_commit(clone_root)
    analysis = fsel.analyze_benchmark(repo_root=clone_root, output_dir=REAL_OUTPUT_DIR)
    e18 = fsel.load_e18_identity(clone_root)
    real_hash = hashlib.sha256((clone_root / fsel.IDENTITY_RELATIVE_PATH).read_bytes()).hexdigest()
    artifact = fsel.build_selection_artifact(
        analysis, e18_sha256=real_hash, e17_sha256=e18["dependency_bindings"]["e17_identity"]["sha256"],
        manifest_sha256=e18["dependency_bindings"]["benchmark_manifest"]["sha256"],
        repository_commit=analysis["merged"]["code_provenance"]["repository_commit"],
        analysis_generation_commit=parent_commit, gpu_execution_performed_by_task4=False, command="test",
    )
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError, match="does not exactly match"):
        fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit=commit)


def test_freshness_artifact_fresh_expected_stale_rejected(fixture_clone, tmp_path):
    """Negative case 2: artifact says the fresh commit; expected is the
    stale parent."""
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    parent_commit = _clone_parent_commit(clone_root)
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError, match="does not exactly match"):
        fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit=parent_commit)


def test_freshness_artifact_replaced_with_another_valid_ancestor_rejected(fixture_clone, tmp_path):
    """Negative case 3: artifact commit is replaced with another valid
    ancestor (the clone's real parent, distinct from both the clone's
    own commit and the true expected commit)."""
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    original_commit = artifact["analysis_generation_commit"]
    parent_commit = _clone_parent_commit(clone_root)
    artifact["analysis_generation_commit"] = parent_commit
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError, match="does not exactly match"):
        fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit=original_commit)


def test_freshness_artifact_replaced_with_valid_descendant_rejected(fixture_clone, tmp_path):
    """Negative case 4: artifact commit is replaced with a valid
    descendant (constructed via `git commit-tree`, never touching the
    fixture clone's HEAD/working tree) -- ancestor-relationship
    validity must never substitute for exact equality."""
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    original_commit = artifact["analysis_generation_commit"]
    descendant = _make_dangling_descendant_commit(clone_root, original_commit)
    assert descendant != original_commit
    artifact["analysis_generation_commit"] = descendant
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError, match="does not exactly match"):
        fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit=original_commit)


def test_freshness_one_character_commit_mutation_rejected(fixture_clone, tmp_path):
    """Negative case 5: a single-character mutation of an otherwise
    well-formed 40-hex commit id."""
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    original_commit = artifact["analysis_generation_commit"]
    flipped_char = "0" if original_commit[0] != "0" else "1"
    mutated = flipped_char + original_commit[1:]
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
        fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit=mutated)


def test_freshness_correct_prefix_incomplete_hash_rejected(fixture_clone, tmp_path):
    """Negative case 6: correct prefix but incomplete (abbreviated)
    hash -- must be rejected by format, not accepted as a prefix
    match."""
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    original_commit = artifact["analysis_generation_commit"]
    abbreviated = original_commit[:7]
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError, match="40-lowercase-hex"):
        fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit=abbreviated)


@pytest.mark.parametrize("bad_expected", [
    True, False, 12345, 1.5, [], {},
    "", "   ", "\t\n",
    "a" * 39, "a" * 41,
])
def test_freshness_invalid_expected_types_and_formats_rejected(fixture_clone, tmp_path, bad_expected):
    """Negative case 7 (part 1): invalid expected types/lengths -- bool,
    non-string, empty, whitespace-only, wrong length. `None` itself
    means "omitted" and is excluded from this parametrization (covered
    separately by test_freshness_explicit_none_matches_omission)."""
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
        fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit=bad_expected)


@pytest.mark.parametrize("bad_expected", [
    "A" * 40,  # uppercase
    " " + "a" * 39,  # leading whitespace, still 40 chars total but not all hex
    "a" * 39 + " ",  # trailing whitespace
    "g" * 40,  # non-hex character
    "not-hex-not-hex-not-hex-not-hex-nothx12",  # non-hex, right length
    "a" * 20 + "\x00" + "a" * 19,  # NUL-containing, 40 chars
])
def test_freshness_invalid_expected_hex_formats_rejected(fixture_clone, tmp_path, bad_expected):
    """Negative case 7 (part 2): uppercase, leading/trailing whitespace,
    non-hex characters, and NUL-containing values -- all rejected by
    the same strict 40-lowercase-hex format rule used for the
    artifact's own field."""
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError, match="40-lowercase-hex"):
        fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit=bad_expected)


def test_freshness_expected_syntactically_valid_but_absent_commit_rejected(fixture_clone, tmp_path):
    """Negative case 8: expected commit is syntactically valid (40
    lowercase hex) but does not exist as a real commit object."""
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError, match="does not exist as a commit object"):
        fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit="f" * 40)


def test_freshness_expected_valid_but_wrong_object_type_rejected(fixture_clone, tmp_path):
    """Negative case 7 (part 3): expected value is a syntactically
    valid 40-hex SHA-1 that resolves to a real object, but NOT a commit
    (a blob), exercising the same `^{commit}` dereference check used
    for the artifact's own field."""
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    blob_hash = sp.run(
        ["git", "-C", str(clone_root), "hash-object", "-w", "--stdin"],
        input="not a commit\n", capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert len(blob_hash) == 40
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError, match="does not exist as a commit object"):
        fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit=blob_hash)


def test_freshness_swapped_benchmark_source_and_generation_commits_still_rejected(fixture_clone, tmp_path):
    """Negative case 9: benchmark-source and analysis-generation
    commits swapped -- must still be rejected by the PRE-EXISTING
    provenance check (independent of the new freshness argument), and
    remains rejected even when a (necessarily mismatched, since the
    swap changed analysis_generation_commit) expected value is
    supplied."""
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    original_generation_commit = artifact["analysis_generation_commit"]
    artifact["repository_commit"], artifact["analysis_generation_commit"] = (
        artifact["analysis_generation_commit"], artifact["repository_commit"],
    )
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
        fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit=original_generation_commit)


# --- domain-exception-contract / fatal-exception tests ---

def test_freshness_no_raw_python_exceptions_leak(fixture_clone, tmp_path):
    """Malformed expected values (wrong Python type entirely) must be
    translated into the domain exception, never a raw TypeError."""
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    for bad in (object(), 3.14, [1, 2], {"a": 1}):
        with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
            fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit=bad)


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_freshness_fatal_exceptions_propagate_unchanged(fixture_clone, tmp_path, exc_type, monkeypatch):
    """A fatal exception raised while resolving the expected-commit's
    existence must propagate with its exact original type, matching
    the established convention elsewhere in this module."""
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))

    orig_run = fsel._run_git_analysis_commit_check
    def fatal_run(root, *args):
        if args and args[-1].startswith("f" * 40):
            raise exc_type("fatal during expected-commit existence check")
        return orig_run(root, *args)
    monkeypatch.setattr(fsel, "_run_git_analysis_commit_check", fatal_run)
    with pytest.raises(exc_type):
        fsel.verify_selection_artifact(p, repo_root=clone_root, expected_analysis_generation_commit="f" * 40)


# ---------------------------------------------------------------------
# Section: no-GPU disclosure (Findings repair round 2, Finding 3).
# ---------------------------------------------------------------------


def _artifact_and_clone(fixture_clone):
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)
    analysis = fsel.analyze_benchmark(repo_root=clone_root, output_dir=REAL_OUTPUT_DIR)
    e18 = fsel.load_e18_identity(clone_root)
    real_hash = hashlib.sha256((clone_root / fsel.IDENTITY_RELATIVE_PATH).read_bytes()).hexdigest()
    artifact = fsel.build_selection_artifact(
        analysis, e18_sha256=real_hash, e17_sha256=e18["dependency_bindings"]["e17_identity"]["sha256"],
        manifest_sha256=e18["dependency_bindings"]["benchmark_manifest"]["sha256"],
        repository_commit=analysis["merged"]["code_provenance"]["repository_commit"],
        analysis_generation_commit=commit, gpu_execution_performed_by_task4=False, command="test",
    )
    return fsel, clone_root, artifact


def test_no_gpu_disclosure_false_accepted(fixture_clone):
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    assert artifact["limitations"]["gpu_execution_performed_by_task4"] is False


@pytest.mark.parametrize("bad_value", [True, 0, "false", None])
def test_no_gpu_disclosure_bad_values_rejected_at_build(fixture_clone, bad_value):
    clone_root, commit = fixture_clone
    fsel = _import_fixture_sel(clone_root)
    analysis = fsel.analyze_benchmark(repo_root=clone_root, output_dir=REAL_OUTPUT_DIR)
    e18 = fsel.load_e18_identity(clone_root)
    real_hash = hashlib.sha256((clone_root / fsel.IDENTITY_RELATIVE_PATH).read_bytes()).hexdigest()
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
        fsel.build_selection_artifact(
            analysis, e18_sha256=real_hash, e17_sha256=e18["dependency_bindings"]["e17_identity"]["sha256"],
            manifest_sha256=e18["dependency_bindings"]["benchmark_manifest"]["sha256"],
            repository_commit=analysis["merged"]["code_provenance"]["repository_commit"],
            analysis_generation_commit=commit, gpu_execution_performed_by_task4=bad_value, command="test",
        )


@pytest.mark.parametrize("bad_value", [True, 0, "false", None])
def test_no_gpu_disclosure_bad_values_rejected_by_verifier(fixture_clone, bad_value, tmp_path):
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    artifact["limitations"]["gpu_execution_performed_by_task4"] = bad_value
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
        fsel.verify_selection_artifact(p, repo_root=clone_root)


def test_no_gpu_disclosure_missing_key_rejected(fixture_clone, tmp_path):
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    del artifact["limitations"]["gpu_execution_performed_by_task4"]
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
        fsel.verify_selection_artifact(p, repo_root=clone_root)


def test_no_gpu_disclosure_unknown_replacement_key_rejected(fixture_clone, tmp_path):
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    del artifact["limitations"]["gpu_execution_performed_by_task4"]
    artifact["limitations"]["task4_no_gpu_execution"] = True
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(fsel.DfpPhase1bChunkSelectionError):
        fsel.verify_selection_artifact(p, repo_root=clone_root)


def test_no_gpu_disclosure_json_markdown_consistent(fixture_clone):
    fsel, clone_root, artifact = _artifact_and_clone(fixture_clone)
    md = fsel.render_markdown_report(artifact)
    assert "GPU execution performed by Task 4" in md
    assert "**False**" in md
    assert "H100" in md


# =======================================================================
# Findings-repair round 3: rollback-safe overwrite transaction for
# install_selection_bundle. All fixtures below are disposable
# tmp_path directories -- never the real production directory.
# =======================================================================


def _install_bundle_flaky_replace(fail_on_call, *, raise_type=OSError, message="injected failure"):
    """Returns a context-manager-free monkeypatch pair: apply
    `Path.replace = patched` before the operation, and always restore
    the original afterward via the caller's own try/finally."""
    orig_replace = Path.replace
    calls = {"n": 0}

    def patched(self, target):
        calls["n"] += 1
        if calls["n"] == fail_on_call:
            raise raise_type(message)
        return orig_replace(self, target)

    return orig_replace, patched, calls


@pytest.fixture
def old_pair(tmp_path):
    j = tmp_path / "x.json"
    m = tmp_path / "x.md"
    j.write_text("OLD_JSON_CONTENT")
    m.write_text("OLD_MD_CONTENT")
    return j, m


def test_overwrite_second_replacement_regression_exact(tmp_path, old_pair):
    """The exact reproduced defect: both destinations pre-exist, the
    first replacement succeeds, the second raises -- both originals
    must survive byte-identical, with no residue."""
    j, m = old_pair
    j_hash, m_hash = sel.sha256_file(j), sel.sha256_file(m)
    orig_replace, patched, calls = _install_bundle_flaky_replace(2)
    Path.replace = patched
    try:
        with pytest.raises(OSError):
            sel.install_selection_bundle({"a": 1}, "NEW_MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    finally:
        Path.replace = orig_replace
    assert j.exists() and m.exists()
    assert j.read_text() == "OLD_JSON_CONTENT"
    assert m.read_text() == "OLD_MD_CONTENT"
    assert sel.sha256_file(j) == j_hash
    assert sel.sha256_file(m) == m_hash
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json", "x.md"]


def test_overwrite_success_path(tmp_path, old_pair):
    j, m = old_pair
    (tmp_path / "unrelated.txt").write_text("unrelated")
    j2, m2 = sel.install_selection_bundle({"a": 1}, "NEW_MD_CONTENT", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    assert json.loads(j2.read_text()) == {"a": 1}
    assert m2.read_text() == "NEW_MD_CONTENT"
    assert "OLD_JSON_CONTENT" not in j2.read_text()
    assert "OLD_MD_CONTENT" not in m2.read_text()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["unrelated.txt", "x.json", "x.md"]
    assert (tmp_path / "unrelated.txt").read_text() == "unrelated"


def test_fresh_install_regression_unaffected(tmp_path):
    j, m = sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="f.json", markdown_basename="f.md")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f.json", "f.md"]


def test_fresh_install_failure_removes_only_created_files(tmp_path):
    (tmp_path / "sibling.txt").write_text("keep me")
    orig_replace, patched, calls = _install_bundle_flaky_replace(2)
    Path.replace = patched
    try:
        with pytest.raises(OSError):
            sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="g.json", markdown_basename="g.md")
    finally:
        Path.replace = orig_replace
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sibling.txt"]
    assert (tmp_path / "sibling.txt").read_text() == "keep me"


# --- failure-injection matrix ---

def test_inject_first_backup_creation_failure(tmp_path, old_pair, monkeypatch):
    j, m = old_pair
    orig = sel._make_sibling_backup
    calls = {"n": 0}

    def flaky(dest):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sel.DfpPhase1bChunkSelectionError("injected first-backup failure")
        return orig(dest)
    monkeypatch.setattr(sel, "_make_sibling_backup", flaky)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    assert j.read_text() == "OLD_JSON_CONTENT"
    assert m.read_text() == "OLD_MD_CONTENT"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json", "x.md"]


def test_inject_second_backup_creation_failure(tmp_path, old_pair, monkeypatch):
    j, m = old_pair
    orig = sel._make_sibling_backup
    calls = {"n": 0}

    def flaky(dest):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sel.DfpPhase1bChunkSelectionError("injected second-backup failure")
        return orig(dest)
    monkeypatch.setattr(sel, "_make_sibling_backup", flaky)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    # First backup must have been cleaned up too -- no residue.
    assert j.read_text() == "OLD_JSON_CONTENT"
    assert m.read_text() == "OLD_MD_CONTENT"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json", "x.md"]


def test_inject_backup_verification_failure(tmp_path, old_pair, monkeypatch):
    j, m = old_pair
    orig_sha = sel.sha256_file
    calls = {"n": 0}

    def flaky_sha(path):
        calls["n"] += 1
        # Corrupt the verification of the backup (not the original)
        # the second time sha256_file runs against a backup-shaped path.
        if calls["n"] == 2 and ".rollback-" in str(path):
            return "0" * 64
        return orig_sha(path)
    monkeypatch.setattr(sel, "sha256_file", flaky_sha)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    assert j.read_text() == "OLD_JSON_CONTENT"
    assert m.read_text() == "OLD_MD_CONTENT"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json", "x.md"]


def test_inject_staged_json_creation_failure(tmp_path, old_pair):
    j, m = old_pair
    orig_write_text = Path.write_text
    calls = {"n": 0}

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1 and self.name.endswith(".json.tmp"):
            raise OSError("injected staged-json failure")
        return orig_write_text(self, *a, **k)
    Path.write_text = flaky
    try:
        with pytest.raises(OSError):
            sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    finally:
        Path.write_text = orig_write_text
    assert j.read_text() == "OLD_JSON_CONTENT"
    assert m.read_text() == "OLD_MD_CONTENT"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json", "x.md"]


def test_inject_staged_markdown_creation_failure(tmp_path, old_pair):
    j, m = old_pair
    orig_write_text = Path.write_text

    def flaky(self, *a, **k):
        if self.name.endswith(".md.tmp"):
            raise OSError("injected staged-markdown failure")
        return orig_write_text(self, *a, **k)
    Path.write_text = flaky
    try:
        with pytest.raises(OSError):
            sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    finally:
        Path.write_text = orig_write_text
    assert j.read_text() == "OLD_JSON_CONTENT"
    assert m.read_text() == "OLD_MD_CONTENT"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json", "x.md"]


def test_inject_staged_pair_verification_failure(tmp_path, old_pair, monkeypatch):
    j, m = old_pair
    orig_sha = sel.sha256_file

    def flaky(path):
        if str(path).endswith(".json.tmp"):
            return "1" * 64
        return orig_sha(path)
    monkeypatch.setattr(sel, "sha256_file", flaky)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    assert j.read_text() == "OLD_JSON_CONTENT"
    assert m.read_text() == "OLD_MD_CONTENT"


def test_inject_first_destination_replacement_failure(tmp_path, old_pair):
    j, m = old_pair
    orig_replace, patched, calls = _install_bundle_flaky_replace(1)
    Path.replace = patched
    try:
        with pytest.raises(OSError):
            sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    finally:
        Path.replace = orig_replace
    assert j.read_text() == "OLD_JSON_CONTENT"
    assert m.read_text() == "OLD_MD_CONTENT"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json", "x.md"]


def test_inject_second_destination_replacement_failure(tmp_path, old_pair):
    j, m = old_pair
    orig_replace, patched, calls = _install_bundle_flaky_replace(2)
    Path.replace = patched
    try:
        with pytest.raises(OSError):
            sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    finally:
        Path.replace = orig_replace
    assert j.read_text() == "OLD_JSON_CONTENT"
    assert m.read_text() == "OLD_MD_CONTENT"


def test_inject_final_json_verification_failure(tmp_path, old_pair, monkeypatch):
    j, m = old_pair
    orig_sha = sel.sha256_file
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        # Let backup-phase + staging-phase hashing succeed; corrupt
        # only the POST-INSTALL verification read of json_dest.
        if str(path).endswith("x.json") and calls["n"] > 2:
            return "2" * 64
        return orig_sha(path)
    monkeypatch.setattr(sel, "sha256_file", flaky)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    assert j.read_text() == "OLD_JSON_CONTENT"
    assert m.read_text() == "OLD_MD_CONTENT"


def test_inject_rollback_first_restoration_failure(tmp_path, old_pair):
    j, m = old_pair
    orig_replace = Path.replace
    calls = {"n": 0}

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("second install replace fails")
        if calls["n"] == 3:
            raise OSError("rollback restore of json fails")
        return orig_replace(self, target)
    Path.replace = flaky
    try:
        with pytest.raises(sel.DfpPhase1bChunkSelectionError, match="rollback could not fully restore"):
            sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    finally:
        Path.replace = orig_replace
    # Backups must NOT have been deleted -- recovery evidence preserved.
    backups = list(tmp_path.glob("*.rollback-*.bak"))
    assert len(backups) >= 1


def test_inject_rollback_verification_failure(tmp_path, old_pair, monkeypatch):
    """Rollback's replace() succeeds, but the POST-restore hash check
    fails -- must still be reported as a rollback failure, not a
    silent success."""
    j, m = old_pair
    orig_replace = Path.replace
    replace_calls = {"n": 0}

    def flaky_replace(self, target):
        replace_calls["n"] += 1
        if replace_calls["n"] == 2:
            raise OSError("second install replace fails")
        return orig_replace(self, target)
    Path.replace = flaky_replace

    orig_sha = sel.sha256_file
    sha_calls = {"n": 0}

    def flaky_sha(path):
        sha_calls["n"] += 1
        real = orig_sha(path)
        return real
    # Simpler: corrupt sha256_file only for the restored json_dest read
    # (post-rollback verification), by intercepting AFTER real replace calls settle.
    def flaky_sha2(path):
        val = orig_sha(path)
        if str(path).endswith("x.json") and flaky_replace_state["restoring"]:
            return "3" * 64
        return val
    flaky_replace_state = {"restoring": False}

    def flaky_replace2(self, target):
        replace_calls["n"] += 1
        if replace_calls["n"] == 2:
            raise OSError("second install replace fails")
        if replace_calls["n"] == 3:
            flaky_replace_state["restoring"] = True
        return orig_replace(self, target)

    Path.replace = flaky_replace2
    monkeypatch.setattr(sel, "sha256_file", flaky_sha2)
    try:
        with pytest.raises(sel.DfpPhase1bChunkSelectionError, match="rollback could not fully restore"):
            sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    finally:
        Path.replace = orig_replace
    backups = list(tmp_path.glob("*.rollback-*.bak"))
    assert len(backups) >= 1, "backups must be preserved when rollback verification fails"


def test_inject_backup_cleanup_failure_after_success_does_not_roll_back(tmp_path, old_pair, monkeypatch):
    j, m = old_pair
    orig_unlink = Path.unlink

    def flaky_unlink(self, *a, **k):
        if ".rollback-" in self.name:
            raise OSError("injected cleanup failure")
        return orig_unlink(self, *a, **k)
    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    # Must NOT raise -- successful install/verify is never rolled back
    # merely because best-effort post-commit cleanup fails.
    j2, m2 = sel.install_selection_bundle({"a": 1}, "NEW_MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    assert json.loads(j2.read_text()) == {"a": 1}
    assert m2.read_text() == "NEW_MD"


# --- state and path adversaries ---

def test_only_one_destination_exists(tmp_path):
    (tmp_path / "x.json").write_text("OLD_JSON")
    # x.md does not exist.
    j, m = sel.install_selection_bundle({"a": 1}, "NEW_MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    assert json.loads(j.read_text()) == {"a": 1}
    assert m.read_text() == "NEW_MD"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json", "x.md"]


def test_destination_is_symlink_rejected(tmp_path):
    real = tmp_path / "real.json"
    real.write_text("REAL")
    link = tmp_path / "x.json"
    link.symlink_to(real)
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)


def test_destination_is_directory_rejected(tmp_path):
    (tmp_path / "x.json").mkdir()
    with pytest.raises(sel.DfpPhase1bChunkSelectionError):
        sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)


def test_backup_name_collision_rejected(tmp_path, old_pair, monkeypatch):
    j, m = old_pair
    # Force uuid4 to a fixed value, then pre-create a file at the exact
    # path _make_sibling_backup would compute for json_dest's backup --
    # the function must detect and reject this collision rather than
    # silently overwrite an unrelated pre-existing file.
    import uuid as uuid_mod
    fixed = uuid_mod.UUID("12345678123456781234567812345678")
    monkeypatch.setattr(sel.uuid, "uuid4", lambda: fixed)
    colliding_path = tmp_path / f".x.json.rollback-{fixed.hex}.bak"
    colliding_path.write_text("PRE-EXISTING, UNRELATED CONTENT")
    with pytest.raises(sel.DfpPhase1bChunkSelectionError, match="collision"):
        sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    assert j.read_text() == "OLD_JSON_CONTENT"
    assert m.read_text() == "OLD_MD_CONTENT"
    assert colliding_path.read_text() == "PRE-EXISTING, UNRELATED CONTENT"


def test_caller_input_document_unmodified(tmp_path, old_pair):
    j, m = old_pair
    artifact = {"a": 1, "nested": {"b": 2}}
    snapshot = copy.deepcopy(artifact)
    sel.install_selection_bundle(artifact, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    assert artifact == snapshot


def test_unrelated_directory_contents_unchanged_on_failure(tmp_path, old_pair):
    j, m = old_pair
    (tmp_path / "other_file.txt").write_text("other")
    (tmp_path / "subdir").mkdir()
    orig_replace, patched, calls = _install_bundle_flaky_replace(2)
    Path.replace = patched
    try:
        with pytest.raises(OSError):
            sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    finally:
        Path.replace = orig_replace
    assert (tmp_path / "other_file.txt").read_text() == "other"
    assert (tmp_path / "subdir").is_dir()


# --- exception-domain tests ---

@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_fatal_exception_propagates_after_rollback(tmp_path, old_pair, exc_type):
    j, m = old_pair
    orig_replace = Path.replace
    calls = {"n": 0}

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] == 2:
            raise exc_type("fatal during second replace")
        return orig_replace(self, target)
    Path.replace = flaky
    try:
        with pytest.raises(exc_type):
            sel.install_selection_bundle({"a": 1}, "MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    finally:
        Path.replace = orig_replace
    # Rollback must still have occurred despite the fatal exception.
    assert j.read_text() == "OLD_JSON_CONTENT"
    assert m.read_text() == "OLD_MD_CONTENT"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json", "x.md"]


def test_cli_normal_installation_failure_exits_2_no_traceback(tmp_path):
    """Through the real CLI: an overwrite attempt without --force must
    exit 2 with a concise diagnostic, no traceback."""
    real_output = REAL_OUTPUT_DIR if _REAL_OUTPUT_AVAILABLE else None
    if real_output is None:
        pytest.skip("no real output")
    out_dir = _copy_real_output_to(tmp_path)
    (out_dir / "phase1b_chunk_selection.json").write_text('{"pre-existing": true}')
    result = _run_cli("analyze", "--output-dir", str(out_dir), "--output", "phase1b_chunk_selection.json")
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert json.loads((out_dir / "phase1b_chunk_selection.json").read_text()) == {"pre-existing": True}


# =======================================================================
# Findings-repair round 4: symmetric staged-file disk verification.
# Both the JSON and the Markdown ``.tmp`` staging files must be
# independently re-read from disk and hash-compared against their
# expected encoded payload BEFORE either destination is ever touched
# (i.e. before the first ``Path.replace`` call of this transaction).
# Prior to this round, only the JSON staged file was verified this way
# -- the Markdown staged hash was computed purely from the in-memory
# string, so a staged-Markdown corruption was only ever caught by the
# POST-install verification, after both destinations had already been
# mutated and required a rollback. All fixtures below are disposable
# tmp_path directories -- never the real production directory.
# =======================================================================


def _track_replace_calls():
    """Monkeypatch-free call-count helper: apply the returned `patched`
    as `Path.replace`, then inspect `calls` -- used to prove a
    destination was (or, critically, was NOT) ever touched."""
    orig_replace = Path.replace
    calls = []

    def patched(self, target):
        calls.append(self.name)
        return orig_replace(self, target)

    return orig_replace, patched, calls


def test_staged_json_disk_hash_actually_verified(tmp_path):
    """Positive path: the staged JSON's on-disk SHA-256 is genuinely
    read back and matches the expected encoded-payload hash, and
    installation succeeds normally."""
    artifact = {"marker": "STAGED_JSON_OK", "value": 7}
    j, m = sel.install_selection_bundle(artifact, "# report\n", output_dir=tmp_path, json_basename="s.json", markdown_basename="s.md")
    expected = hashlib.sha256((json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    assert sel.sha256_file(j) == expected
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.json", "s.md"]


def test_staged_markdown_disk_hash_actually_verified(tmp_path):
    """Positive path: the staged Markdown's on-disk SHA-256 is
    genuinely read back (not merely computed from the in-memory
    string) and matches the expected encoded-payload hash, and
    installation succeeds normally."""
    md_text = "# STAGED_MD_OK report\n\nsome content\n"
    j, m = sel.install_selection_bundle({"marker": "X"}, md_text, output_dir=tmp_path, json_basename="s.json", markdown_basename="s.md")
    expected = hashlib.sha256(md_text.encode("utf-8")).hexdigest()
    assert sel.sha256_file(m) == expected
    assert m.read_text() == md_text
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.json", "s.md"]


def test_staged_markdown_disk_hash_verification_survives_overwrite_and_rollback(tmp_path, old_pair):
    """The new staged-Markdown disk verification must not interfere
    with the already-verified overwrite/rollback transaction: a normal
    overwrite still succeeds, and the exact second-replacement
    rollback regression is still exercised correctly with this check
    in place."""
    j, m = old_pair
    j2, m2 = sel.install_selection_bundle({"a": 1}, "NEW_MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    assert j2.read_text() == '{\n  "a": 1\n}\n'
    assert m2.read_text() == "NEW_MD"


def test_staged_markdown_corruption_rejected_before_any_destination_mutated(tmp_path, old_pair):
    """The central regression for this round. Corrupts the ACTUAL
    BYTES written to the staged Markdown ``.tmp`` file (not merely the
    hash-check result) after it is written but before it would be
    hashed, then requires: DfpPhase1bChunkSelectionError naming the
    staged Markdown/hash mismatch; ZERO calls to Path.replace (i.e.
    neither destination is EVER touched, not touched-then-rolled-back
    -- this is what distinguishes the fix from the pre-fix behavior,
    which mutated both destinations before its later post-install
    check caught the same corruption); both destinations byte-identical
    to their originals; no mixed pair; no `.tmp`/`.bak` residue; an
    unrelated sibling file untouched.

    This test would FAIL against the pre-fix implementation: replayed
    directly against the pre-fix `install_selection_bundle` (see the
    verification report), this exact corruption still raises
    DfpPhase1bChunkSelectionError, but only via the POST-install check,
    by which point Path.replace has already been called for BOTH
    destinations (and, for the pre-existing pair here, two more times
    for rollback) -- i.e. `replace_calls` would be non-empty, failing
    the assertion below.
    """
    j, m = old_pair
    j_hash, m_hash = sel.sha256_file(j), sel.sha256_file(m)
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("do not touch\n")
    unrelated_hash = sel.sha256_file(unrelated)

    orig_write_text = Path.write_text

    def corrupt_md_write(self, data, *a, **k):
        if self.name.endswith(".md.tmp"):
            # Genuinely write DIFFERENT bytes than what was requested --
            # not a monkeypatched hash result.
            return orig_write_text(self, data + "CORRUPTED_ON_DISK_BYTES", *a, **k)
        return orig_write_text(self, data, *a, **k)

    orig_replace, patched_replace, replace_calls = _track_replace_calls()
    Path.write_text = corrupt_md_write
    Path.replace = patched_replace
    try:
        with pytest.raises(sel.DfpPhase1bChunkSelectionError, match=r"[Ss]taged [Mm]arkdown"):
            sel.install_selection_bundle({"a": 1}, "NEW_MD_CONTENT", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    finally:
        Path.write_text = orig_write_text
        Path.replace = orig_replace

    assert replace_calls == [], f"a destination was mutated before staged verification caught the corruption: {replace_calls}"
    assert j.exists() and m.exists()
    assert j.read_text() == "OLD_JSON_CONTENT"
    assert m.read_text() == "OLD_MD_CONTENT"
    assert sel.sha256_file(j) == j_hash
    assert sel.sha256_file(m) == m_hash
    assert sorted(p.name for p in tmp_path.iterdir()) == ["unrelated.txt", "x.json", "x.md"]
    assert sel.sha256_file(unrelated) == unrelated_hash


def test_staged_json_corruption_rejected_before_any_destination_mutated(tmp_path, old_pair):
    """Symmetric counterpart for the JSON side (already covered less
    directly by test_inject_staged_pair_verification_failure, which
    monkeypatches the hash function rather than corrupting real bytes
    -- this test corrupts the actual staged JSON bytes on disk)."""
    j, m = old_pair
    j_hash, m_hash = sel.sha256_file(j), sel.sha256_file(m)

    orig_write_text = Path.write_text

    def corrupt_json_write(self, data, *a, **k):
        if self.name.endswith(".json.tmp"):
            return orig_write_text(self, data + "CORRUPTED", *a, **k)
        return orig_write_text(self, data, *a, **k)

    orig_replace, patched_replace, replace_calls = _track_replace_calls()
    Path.write_text = corrupt_json_write
    Path.replace = patched_replace
    try:
        with pytest.raises(sel.DfpPhase1bChunkSelectionError, match=r"[Ss]taged JSON"):
            sel.install_selection_bundle({"a": 1}, "NEW_MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    finally:
        Path.write_text = orig_write_text
        Path.replace = orig_replace

    assert replace_calls == []
    assert j.read_text() == "OLD_JSON_CONTENT"
    assert m.read_text() == "OLD_MD_CONTENT"
    assert sel.sha256_file(j) == j_hash
    assert sel.sha256_file(m) == m_hash
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json", "x.md"]


def test_staged_markdown_hash_read_failure_before_mutation(tmp_path, old_pair, monkeypatch):
    """A genuine failure while hashing the staged Markdown file (e.g.
    a transient read error) must be raised before either destination
    is touched, and must leave both originals unchanged."""
    j, m = old_pair
    orig_sha = sel.sha256_file

    def flaky_sha(path):
        if str(path).endswith(".md.tmp"):
            raise OSError("injected staged-markdown hash-read failure")
        return orig_sha(path)
    monkeypatch.setattr(sel, "sha256_file", flaky_sha)

    orig_replace, patched_replace, replace_calls = _track_replace_calls()
    Path.replace = patched_replace
    try:
        with pytest.raises(OSError):
            sel.install_selection_bundle({"a": 1}, "NEW_MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    finally:
        Path.replace = orig_replace

    assert replace_calls == []
    assert j.read_text() == "OLD_JSON_CONTENT"
    assert m.read_text() == "OLD_MD_CONTENT"


def test_staged_markdown_hash_read_failure_via_cli_exits_2_no_traceback(tmp_path, monkeypatch, capsys):
    """The established domain/exit contract still applies: a genuine
    staged-Markdown hash-read failure, exercised through the real CLI
    boundary (`diagnostics.analyze_dfp_phase1b_chunk_benchmark.main`),
    must exit 2 with no traceback -- exactly like every other
    OSError-domain failure already covered for this module
    (`_KNOWN_ERRORS` includes OSError at the CLI boundary).

    Exercised in-process (not via subprocess) so the injected failure
    can be pinpointed to exactly the staged-Markdown hash read, rather
    than any earlier CLI-only precondition (E18 binding check,
    clean-worktree check) that a subprocess invocation cannot isolate
    from mid-repair, uncommitted state. The clean-worktree gate is
    bypassed here ONLY because it is unrelated to this repair and
    already covered by its own dedicated tests elsewhere -- nothing
    about install_selection_bundle or its staged-verification is
    patched.

    The patch target is `cli.install_selection_bundle.__globals__`
    itself (not this test file's own top-level `sel` alias) because
    other tests in this same session (the E18 fixture-clone tests) can
    repoint `sys.modules["src.dfp_phase1b_chunk_selection"]` at a
    freshly-imported clone copy; patching by module *name* would then
    silently miss whichever module object the CLI's already-bound
    `install_selection_bundle` reference actually executes against.
    Patching its own `__globals__` mapping directly is unambiguous
    regardless of test order or any other test's module-cache
    manipulation."""
    if not _REAL_OUTPUT_AVAILABLE:
        pytest.skip("no real output")
    import importlib
    cli = importlib.import_module("diagnostics.analyze_dfp_phase1b_chunk_benchmark")
    out_dir = _copy_real_output_to(tmp_path)

    monkeypatch.setattr(cli, "resolve_analysis_generation_commit", lambda repo_root, e18_document=None: "0" * 40)

    target_globals = cli.install_selection_bundle.__globals__
    orig_sha = target_globals["sha256_file"]

    def flaky_sha(path):
        if str(path).endswith(".md.tmp"):
            raise OSError("injected staged-markdown hash-read failure")
        return orig_sha(path)
    monkeypatch.setitem(target_globals, "sha256_file", flaky_sha)

    argv = [
        "--repo-root", str(ROOT), "analyze", "--output-dir", str(out_dir),
        "--output", "phase1b_chunk_selection.json", "--markdown-output", "phase1b_chunk_selection_report.md",
    ]
    rc = cli.main(argv)
    captured = capsys.readouterr()
    assert rc == 2
    assert "Traceback" not in captured.err
    assert "injected staged-markdown hash-read failure" in captured.err
    assert not (out_dir / "phase1b_chunk_selection.json").exists()
    assert not (out_dir / "phase1b_chunk_selection_report.md").exists()


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_fatal_exception_during_staged_markdown_hash_read_propagates(tmp_path, old_pair, exc_type, monkeypatch):
    """A fatal exception raised while hashing the staged Markdown file
    (before any destination mutation) must propagate with its exact
    original type, exactly like every other fatal-exception injection
    point already covered for this transaction."""
    j, m = old_pair
    orig_sha = sel.sha256_file

    def fatal_sha(path):
        if str(path).endswith(".md.tmp"):
            raise exc_type("fatal during staged markdown hash read")
        return orig_sha(path)
    monkeypatch.setattr(sel, "sha256_file", fatal_sha)

    orig_replace, patched_replace, replace_calls = _track_replace_calls()
    Path.replace = patched_replace
    try:
        with pytest.raises(exc_type):
            sel.install_selection_bundle({"a": 1}, "NEW_MD", output_dir=tmp_path, json_basename="x.json", markdown_basename="x.md", allow_overwrite=True)
    finally:
        Path.replace = orig_replace

    assert replace_calls == []
    assert j.read_text() == "OLD_JSON_CONTENT"
    assert m.read_text() == "OLD_MD_CONTENT"
