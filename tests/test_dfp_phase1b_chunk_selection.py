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
    a1 = sel.build_selection_artifact(analysis, e18_sha256="x", e17_sha256="y", manifest_sha256="z", repository_commit="c", command="cmd")
    a2 = sel.build_selection_artifact(analysis, e18_sha256="x", e17_sha256="y", manifest_sha256="z", repository_commit="c", command="cmd")
    assert json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True)


@requires_real_output
def test_markdown_report_is_deterministic():
    analysis = _real_analysis()
    artifact = sel.build_selection_artifact(analysis, e18_sha256="x", e17_sha256="y", manifest_sha256="z", repository_commit="c", command="cmd")
    md1 = sel.render_markdown_report(artifact)
    md2 = sel.render_markdown_report(artifact)
    assert md1 == md2
    assert "16" in md1 and "Selected chunk size" in md1


@requires_real_output
def test_write_selection_artifact_no_overwrite_by_default(tmp_path):
    analysis = _real_analysis()
    artifact = sel.build_selection_artifact(analysis, e18_sha256="x", e17_sha256="y", manifest_sha256="z", repository_commit="c", command="cmd")
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
    artifact = sel.build_selection_artifact(analysis, e18_sha256="x", e17_sha256="y", manifest_sha256="z", repository_commit="c", command="cmd")
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
def test_cli_valid_basenames_accepted(tmp_path):
    out_dir = _copy_real_output_to(tmp_path)
    result = _run_cli("analyze", "--output-dir", str(out_dir), "--output", "phase1b_chunk_selection.json",
                       "--markdown-output", "phase1b_chunk_selection_report.md")
    assert result.returncode == 0
    assert (out_dir / "phase1b_chunk_selection.json").exists()
    assert (out_dir / "phase1b_chunk_selection_report.md").exists()


@requires_real_output
def test_cli_double_dot_substring_basename_accepted(tmp_path):
    out_dir = _copy_real_output_to(tmp_path)
    result = _run_cli("analyze", "--output-dir", str(out_dir), "--output", "run..v2.json")
    assert result.returncode == 0
    assert (out_dir / "run..v2.json").exists()
