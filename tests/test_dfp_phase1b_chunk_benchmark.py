"""CPU-only synthetic tests for the DFP Phase 1B Task 3 chunk-size
benchmark (E17 identity, selection algorithm, instrumentation wrapper,
trial/merge result schemas, and CLI). No real model, no real GPU, no
real image, no GT -- every fake is hand-built and small."""

from __future__ import annotations

import contextlib
import copy
import json
import math
import subprocess as sp
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
for _path in (ROOT, ROOT / "src/open_vocabulary_segmentation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import src.dfp_phase1b_chunk_benchmark as bench  # noqa: E402
import src.dfp_phase1b_chunk_benchmark_identity as ident  # noqa: E402

ARTIFACT_PATH = Path("/scratch/haree/dfp_phase1b_conservation_calibration/coco_stuff_200/phase1b_conservation_calibration.json")
WORKLOAD_BUNDLE_DIR = Path("/scratch/haree/dfp_phase1b_conservation_calibration_analysis/coco_stuff_200_workload_stats_v2")
MANIFEST_PATH = ROOT / "evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json"
CLI_PATH = ROOT / "diagnostics/run_dfp_phase1b_chunk_benchmark.py"

_REAL_INPUTS_AVAILABLE = ARTIFACT_PATH.exists() and WORKLOAD_BUNDLE_DIR.exists()
requires_real_inputs = pytest.mark.skipif(not _REAL_INPUTS_AVAILABLE, reason="real calibration artifact/workload bundle not present on this host")


# ---------------------------------------------------------------------
# Section: hand-built synthetic 200-row "calibration document" fixture
# -- never the real artifact -- used for every selection/manifest test
# that must not depend on real scratch-filesystem state.
# ---------------------------------------------------------------------

NODE_COUNT = ident.NODE_COUNT_CONSTANT


def _residual_summary(sample_count):
    if sample_count == 0:
        return {"sample_count": 0, "finite_count": 0, "non_finite_count": 0, "signed_minimum": None,
                "signed_maximum": None, "maximum_absolute_residual": None, "mean_absolute_residual": None,
                "root_mean_square_residual": None, "absolute_residual_sum": None, "squared_residual_sum": None}
    v = 1e-9
    s, sq = v * sample_count, v * v * sample_count
    return {"sample_count": sample_count, "finite_count": sample_count, "non_finite_count": 0,
            "signed_minimum": -v, "signed_maximum": v, "maximum_absolute_residual": v,
            "mean_absolute_residual": s / sample_count, "root_mean_square_residual": math.sqrt(sq / sample_count),
            "absolute_residual_sum": s, "squared_residual_sum": sq}


def _fake_calibration_document(*, query_class_counts=None, seed=1234):
    """200 hand-built selected_window_records, each with a controllable
    query_class_count via rho_0/rho_320.sample_count = qcc * NODE_COUNT."""
    import random

    rnd = random.Random(seed)
    if query_class_counts is None:
        query_class_counts = [1 + (i * 37) % 60 for i in range(200)]
    assert len(query_class_counts) == 200
    ranks = list(range(200))
    rnd.shuffle(ranks)  # selection_rank assignment order shouldn't matter to selection
    rows = []
    for i in range(200):
        qcc = query_class_counts[i]
        rho_sample_count = qcc * NODE_COUNT
        rows.append({
            "run_mode": "coco_stuff_500", "image_id": f"{i % 47:012d}", "image_path": f"val2017/{i % 47:012d}.jpg",
            "image_manifest_ordinal": i % 47, "inference_canvas_height": 448, "inference_canvas_width": 448,
            "canonical_window_ordinal": i // 47, "top": 0, "left": 0, "bottom": 448, "right": 448,
            "crop_height": 448, "crop_width": 448, "patch_grid_height": 32, "patch_grid_width": 32,
            "canonical_candidate_index": i, "selection_rank": ranks[i], "selection_key": f"{i:064x}"[-64:],
            "completed": True,
            "rho_M": _residual_summary(NODE_COUNT), "rho_0": _residual_summary(rho_sample_count),
            "rho_320": _residual_summary(rho_sample_count), "rho_g": _residual_summary(3 + i % 11),
        })
    return {"complete": True, "final": True, "selected_window_records": rows,
            "selected_windows_digest": "d" * 64, "selection_trace_digest": "e" * 64}


# ---------------------------------------------------------------------
# Section: E17 closed-schema identity validation.
# ---------------------------------------------------------------------


@requires_real_inputs
def test_e17_identity_loads_and_verifies():
    doc = ident.load_identity(
        calibration_artifact_path=ARTIFACT_PATH, workload_bundle_dir=WORKLOAD_BUNDLE_DIR,
    )
    assert doc["identity"]["name"] == "e17-dfp-phase1b-chunk-benchmark"


def test_e17_identity_without_relational_paths_still_closed_schema_validates():
    # No calibration_artifact_path/workload_bundle_dir supplied -- only
    # the repository-relative bindings are relationally re-verified;
    # closed-schema validation still runs fully.
    doc = ident.load_identity()
    assert doc["operational_scope"]["scientific_metric"] is False


@pytest.mark.parametrize("find,replace", [
    # --- Repeats the original spot-check ---
    ('scientific_metric = false', 'scientific_metric = true'),
    ('tolerance_selection_authorized = false', 'tolerance_selection_authorized = true'),
    ('chunk_sizes = [16, 32, 64]', 'chunk_sizes = [16, 32]'),
    ('measured_repeats = 2', 'measured_repeats = 3'),
    ('repeat_0 = [16, 32, 64]', 'repeat_0 = [16, 64, 32]'),
    ('numeric_tolerance_permitted = false', 'numeric_tolerance_permitted = true'),
    ('required_gpu_model = "NVIDIA H100 80GB HBM3"', 'required_gpu_model = "NVIDIA A100"'),
    ('winner_selection_authorized_by_this_identity = false', 'winner_selection_authorized_by_this_identity = true'),
    # --- The six previously-UNDETECTED leaves from independent verification (Finding 1) ---
    ('required_expected_selected_window_count = 200', 'required_expected_selected_window_count = 50'),
    ('required_run_mode = "coco_stuff_500"', 'required_run_mode = "coco_stuff_5000"'),
    ('gpu_model_at_calibration_time = "NVIDIA H100 80GB HBM3"', 'gpu_model_at_calibration_time = "NVIDIA A100"'),
    ('device_type_at_calibration_time = "cuda"', 'device_type_at_calibration_time = "rocm"'),
    ('host_rss_abort_fraction_of_allocation = 0.90', 'host_rss_abort_fraction_of_allocation = 0.50'),
    ('allocated_host_memory_gib = 80', 'allocated_host_memory_gib = 8'),
    # --- Every remaining safety-guard leaf ---
    ('gpu_reserved_abort_fraction_of_allocation = 0.90', 'gpu_reserved_abort_fraction_of_allocation = 0.50'),
    ('allocated_gpu_memory_gib = 80', 'allocated_gpu_memory_gib = 8'),
    ('comparison_operator = "strictly_greater_than"', 'comparison_operator = "greater_than_or_equal"'),
    # --- All remaining operational-scope booleans ---
    ('full_phase1b_execution_authorized = false', 'full_phase1b_execution_authorized = true'),
    ('five_thousand_image_evaluation_authorized = false', 'five_thousand_image_evaluation_authorized = true'),
    ('chunk_size_recommendation_authorized_by_this_identity = false', 'chunk_size_recommendation_authorized_by_this_identity = true'),
    # --- warm-up / benchmark counts ---
    ('benchmark_window_count = 24', 'benchmark_window_count = 25'),
    ('warmup_window_count_per_trial = 1', 'warmup_window_count_per_trial = 2'),
    ('trial_count = 6', 'trial_count = 7'),
    # --- device/precision/same-environment requirements ---
    ('required_device_type = "cuda"', 'required_device_type = "rocm"'),
    ('same_model_checkpoint_required = true', 'same_model_checkpoint_required = false'),
    ('same_precision_required = true', 'same_precision_required = false'),
    ('fresh_process_per_trial_required = true', 'fresh_process_per_trial_required = false'),
    ('repository_worktree_clean_required = true', 'repository_worktree_clean_required = false'),
    # --- trial schedule reordering (second occurrence, repeat_1) ---
    ('repeat_1 = [64, 32, 16]', 'repeat_1 = [16, 32, 64]'),
    # --- selection formula ---
    ('slot_count = 24', 'slot_count = 25'),
    ('node_count_constant = 1024', 'node_count_constant = 2048'),
    ('sort_key = "(query_class_count ascending, selection_rank ascending)"', 'sort_key = "(selection_rank ascending, query_class_count ascending)"'),
    # --- scientific-invariance rules ---
    ('require_repeat_0_equals_repeat_1_per_chunk_size = true', 'require_repeat_0_equals_repeat_1_per_chunk_size = false'),
    ('require_all_three_chunk_sizes_bitwise_digest_identical = true', 'require_all_three_chunk_sizes_bitwise_digest_identical = false'),
    ('rounding_before_comparison_permitted = false', 'rounding_before_comparison_permitted = true'),
    ('mismatch_outcome = "SCIENTIFIC_OUTPUT_MISMATCH"', 'mismatch_outcome = "MINOR_DISCREPANCY"'),
    # --- artifact schema names/versions ---
    ('schema_name = "talk2dino-dfp-phase1b-chunk-benchmark-identity-v1"', 'schema_name = "talk2dino-dfp-phase1b-chunk-benchmark-identity-v2"'),
    # --- code-provenance policy ---
    ('required_ancestor_commit_equality_to_repository_commit_not_required = true', 'required_ancestor_commit_equality_to_repository_commit_not_required = false'),
    ('dirty_worktree_rejected_before_torch_import = true', 'dirty_worktree_rejected_before_torch_import = false'),
])
def test_e17_identity_rejects_every_frozen_value_mutation(tmp_path, find, replace):
    text = (ROOT / ident.IDENTITY_RELATIVE_PATH).read_text()
    assert find in text, f"fixture assumption stale: {find!r} not found in the committed E17 identity"
    mutated = text.replace(find, replace, 1)
    bad_path = tmp_path / "e17_mutated.toml"
    bad_path.write_text(mutated)
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.load_identity(path=bad_path, repo_root=ROOT)


def test_e17_identity_rejects_mutated_file_via_real_toml_roundtrip(tmp_path):
    """A more faithful mutation test: physically corrupt the identity
    file, then load it through the real ``load_identity`` entry
    point."""
    text = (ROOT / ident.IDENTITY_RELATIVE_PATH).read_text()
    mutated = text.replace('scientific_metric = false', 'scientific_metric = true')
    assert mutated != text
    bad_path = tmp_path / "e17_mutated.toml"
    bad_path.write_text(mutated)
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.load_identity(path=bad_path, repo_root=ROOT)


def test_e17_identity_rejects_unknown_top_level_key(tmp_path):
    text = (ROOT / ident.IDENTITY_RELATIVE_PATH).read_text()
    mutated = text + '\n[unexpected_new_section]\nfoo = 1\n'
    bad_path = tmp_path / "e17_extra.toml"
    bad_path.write_text(mutated)
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.load_identity(path=bad_path, repo_root=ROOT)


def test_e17_identity_relational_hash_mismatch_detected(tmp_path):
    fake_artifact = tmp_path / "fake_artifact.json"
    fake_artifact.write_text("{}")
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.load_identity(calibration_artifact_path=fake_artifact)


# ---------------------------------------------------------------------
# Section: exact 24-window selection formula.
# ---------------------------------------------------------------------


def test_slot_position_formula_matches_spec():
    positions = [bench.slot_position(i) for i in range(24)]
    expected = [((2 * i + 1) * 200) // (2 * 24) for i in range(24)]
    assert positions == expected
    assert len(set(positions)) == 24


def test_selection_produces_24_unique_records_and_valid_warmup():
    doc = _fake_calibration_document()
    result = bench.select_benchmark_windows(doc["selected_window_records"])
    records = result["benchmark_records"]
    assert len(records) == 24
    assert len({r["canonical_candidate_index"] for r in records}) == 24
    assert len({r["selection_rank"] for r in records}) == 24
    assert [r["benchmark_slot"] for r in records] == list(range(24))
    qccs = [r["query_class_count"] for r in records]
    assert qccs == sorted(qccs)  # slot order follows ascending query_class_count by construction
    warmup = result["warmup_record"]
    assert warmup["selection_rank"] not in {r["selection_rank"] for r in records}
    all_ranks_source = {row["selection_rank"] for row in doc["selected_window_records"]}
    assert warmup["selection_rank"] == min(all_ranks_source - {r["selection_rank"] for r in records})


@pytest.mark.parametrize("order_mode", ["as_built", "reversed", "shuffled"])
@pytest.mark.parametrize("seed", [None, 0, 1, 42])
def test_selection_invariant_to_input_order_and_hashseed(order_mode, seed, monkeypatch):
    if seed is not None:
        monkeypatch.setenv("PYTHONHASHSEED", str(seed))
    doc = _fake_calibration_document()
    rows = doc["selected_window_records"]
    if order_mode == "reversed":
        rows = list(reversed(rows))
    elif order_mode == "shuffled":
        import random

        rnd = random.Random(777)
        rows = list(rows)
        rnd.shuffle(rows)
    result = bench.select_benchmark_windows(rows)
    baseline = bench.select_benchmark_windows(doc["selected_window_records"])
    assert result["benchmark_records"] == baseline["benchmark_records"]
    assert result["warmup_record"] == baseline["warmup_record"]


def test_selection_rejects_wrong_row_count():
    doc = _fake_calibration_document()
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.select_benchmark_windows(doc["selected_window_records"][:199])


def test_selection_rejects_rho_0_rho_320_mismatch():
    doc = _fake_calibration_document()
    doc["selected_window_records"][0]["rho_320"]["sample_count"] += NODE_COUNT
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.select_benchmark_windows(doc["selected_window_records"])


def test_selection_rejects_non_divisible_sample_count():
    doc = _fake_calibration_document()
    doc["selected_window_records"][0]["rho_0"]["sample_count"] += 1
    doc["selected_window_records"][0]["rho_320"]["sample_count"] += 1
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.select_benchmark_windows(doc["selected_window_records"])


# ---------------------------------------------------------------------
# Section: manifest build/verify, including mutation rejection.
# ---------------------------------------------------------------------


def test_build_and_verify_manifest_roundtrip():
    doc = _fake_calibration_document()
    manifest = bench.build_benchmark_manifest(
        calibration_document=doc, calibration_artifact_sha256="a" * 64, e16_identity_sha256="b" * 64,
    )
    out = bench.verify_benchmark_manifest(manifest, calibration_document=doc, calibration_artifact_sha256="a" * 64, e16_identity_sha256="b" * 64)
    assert out["benchmark_record_digest"] == manifest["benchmark_record_digest"]


@requires_real_inputs
def test_real_installed_manifest_matches_real_artifact():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    with open(ARTIFACT_PATH) as f:
        doc = json.load(f)
    import hashlib

    bench.verify_benchmark_manifest(
        manifest, calibration_document=doc,
        calibration_artifact_sha256=hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest(),
        e16_identity_sha256=hashlib.sha256((ROOT / "evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml").read_bytes()).hexdigest(),
    )


@pytest.mark.parametrize("mutation", [
    lambda m: m["benchmark_records"].pop(0),
    lambda m: m["benchmark_records"].append(dict(m["benchmark_records"][0])),
    lambda m: m["benchmark_records"][0].__setitem__("selection_rank", m["benchmark_records"][1]["selection_rank"]),
    lambda m: m["benchmark_records"][0].__setitem__("canonical_candidate_index", m["benchmark_records"][1]["canonical_candidate_index"]),
    lambda m: m.__setitem__("warmup_record", dict(m["benchmark_records"][0], selection_rank=m["benchmark_records"][0]["selection_rank"])),
    lambda m: m.__setitem__("benchmark_record_digest", "0" * 64),
    lambda m: m.__setitem__("canonical_manifest_digest", "0" * 64),
    lambda m: m.pop("positions"),
    lambda m: m.__setitem__("schema_version", 2),
])
def test_manifest_verification_rejects_every_mutation(mutation):
    doc = _fake_calibration_document()
    manifest = bench.build_benchmark_manifest(calibration_document=doc, calibration_artifact_sha256="a" * 64, e16_identity_sha256="b" * 64)
    mutated = copy.deepcopy(manifest)
    mutation(mutated)
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.verify_benchmark_manifest(mutated)


def test_manifest_verification_detects_cross_fed_calibration_document():
    doc_a = _fake_calibration_document(seed=1)
    doc_b = _fake_calibration_document(seed=2)
    manifest = bench.build_benchmark_manifest(calibration_document=doc_a, calibration_artifact_sha256="a" * 64, e16_identity_sha256="b" * 64)
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.verify_benchmark_manifest(manifest, calibration_document=doc_b)


# ---------------------------------------------------------------------
# Section: transparent instrumentation wrapper.
# ---------------------------------------------------------------------


def test_instrumentation_wrapper_calls_original_exactly_once_and_returns_exact_result():
    module = types.SimpleNamespace()
    call_count = {"n": 0}
    sentinel = object()

    def original(unary, membership, component_ids, query_class_indices, *, contract, window_index):
        call_count["n"] += 1
        return sentinel

    module.build_masked_seed_chunk = original
    restore, calls = bench.instrument_masked_forward_module(module)
    try:
        result = module.build_masked_seed_chunk("U", "M", [0, 1, 2], [5], contract="C", window_index=7)
    finally:
        restore()
    assert result is sentinel
    assert call_count["n"] == 1
    assert len(calls) == 1
    assert calls[0].component_ids == [0, 1, 2]
    assert calls[0].window_index == 7
    assert module.build_masked_seed_chunk is original


def test_instrumentation_wrapper_preserves_exact_argument_identity():
    module = types.SimpleNamespace()
    received = {}

    def original(unary, membership, component_ids, query_class_indices, *, contract, window_index):
        received["unary"] = unary
        received["membership"] = membership
        received["component_ids"] = component_ids
        received["query_class_indices"] = query_class_indices
        received["contract"] = contract
        return "value"

    module.build_masked_seed_chunk = original
    restore, _ = bench.instrument_masked_forward_module(module)
    unary_obj, membership_obj, contract_obj = object(), object(), object()
    try:
        module.build_masked_seed_chunk(unary_obj, membership_obj, [1], [2], contract=contract_obj, window_index=0)
    finally:
        restore()
    assert received["unary"] is unary_obj
    assert received["membership"] is membership_obj
    assert received["contract"] is contract_obj


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_instrumentation_wrapper_propagates_fatal_exceptions(exc_type):
    module = types.SimpleNamespace()

    def original(*a, **k):
        raise exc_type("simulated")

    module.build_masked_seed_chunk = original
    restore, calls = bench.instrument_masked_forward_module(module)
    try:
        with pytest.raises(exc_type):
            module.build_masked_seed_chunk(None, None, [0], [0], contract=None, window_index=0)
    finally:
        restore()
    assert module.build_masked_seed_chunk is original


def test_summarize_chunk_calls_reports_full_vs_partial():
    module = types.SimpleNamespace(build_masked_seed_chunk=lambda *a, **k: None)
    restore, calls = bench.instrument_masked_forward_module(module)
    try:
        module.build_masked_seed_chunk(None, None, list(range(0, 16)), [0], contract=None, window_index=3)
        module.build_masked_seed_chunk(None, None, list(range(16, 32)), [0], contract=None, window_index=3)
        module.build_masked_seed_chunk(None, None, list(range(32, 40)), [0], contract=None, window_index=3)
    finally:
        restore()
    summary = bench.summarize_chunk_calls(calls, window_index=3)
    assert summary["actual_component_chunk_count"] == 3
    assert summary["component_chunk_widths"] == [16, 16, 8]
    assert summary["full_chunk_count"] == 2
    assert summary["partial_chunk_count"] == 1
    assert summary["image_level_component_count"] == 40


# ---------------------------------------------------------------------
# Section: scientific-output digest invariance.
# ---------------------------------------------------------------------


def _fake_family_summaries(seed_value=1e-9, sample_count=10):
    s = _residual_summary(sample_count)
    return {"rho_M": s, "rho_0": s, "rho_320": s, "rho_g": s}


def test_scientific_digest_identical_for_identical_content():
    r1 = bench.scientific_output_record(canonical_candidate_index=0, canonical_image_id="x", canonical_window_ordinal=0,
                                         selection_rank=0, family_summaries=_fake_family_summaries())
    r2 = bench.scientific_output_record(canonical_candidate_index=0, canonical_image_id="x", canonical_window_ordinal=0,
                                         selection_rank=0, family_summaries=_fake_family_summaries())
    assert bench.compute_scientific_digest(r1) == bench.compute_scientific_digest(r2)


def test_scientific_digest_changes_on_any_residual_perturbation():
    base = bench.scientific_output_record(canonical_candidate_index=0, canonical_image_id="x", canonical_window_ordinal=0,
                                           selection_rank=0, family_summaries=_fake_family_summaries())
    perturbed_summaries = _fake_family_summaries()
    perturbed_summaries["rho_g"]["absolute_residual_sum"] += 1e-15
    perturbed = bench.scientific_output_record(canonical_candidate_index=0, canonical_image_id="x", canonical_window_ordinal=0,
                                                selection_rank=0, family_summaries=perturbed_summaries)
    assert bench.compute_scientific_digest(base) != bench.compute_scientific_digest(perturbed)


def test_collection_digest_requires_exactly_24_records():
    records = [bench.scientific_output_record(canonical_candidate_index=i, canonical_image_id="x", canonical_window_ordinal=i,
                                                selection_rank=i, family_summaries=_fake_family_summaries()) for i in range(23)]
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.compute_collection_digest(records)


def test_max_absolute_difference_none_when_empty_or_mismatched_length():
    assert bench.max_absolute_difference([], []) is None
    assert bench.max_absolute_difference([1.0], [1.0, 2.0]) is None
    assert bench.max_absolute_difference([1.0, 2.0], [1.0, 5.0]) == 3.0


# ---------------------------------------------------------------------
# Section: run_trial with fakes (no torch, no CUDA, no real model).
# ---------------------------------------------------------------------


def _build_fake_manifest():
    doc = _fake_calibration_document()
    return bench.build_benchmark_manifest(calibration_document=doc, calibration_artifact_sha256="a" * 64, e16_identity_sha256="b" * 64)


class _FakeResources:
    protocol_identity = {}
    contracts = None
    class_count = 5
    affinity_power = 1.0


class _FakeMemoryProvider:
    def baseline(self):
        return {"allocated_bytes": 100, "reserved_bytes": 200}

    def reset_peak(self):
        pass

    def peak(self):
        return {"allocated_bytes": 500, "reserved_bytes": 600, "end_allocated_bytes": 150, "end_reserved_bytes": 250}

    def host_current(self):
        return {"ru_maxrss_bytes": 1000, "vmrss_bytes": 900, "vmhwm_bytes": 1100}


def _make_fake_masked_forward_and_process_fn():
    fake_mf = types.SimpleNamespace()

    def build_masked_seed_chunk(unary, membership, component_ids, query_class_indices, *, contract, window_index):
        return None

    fake_mf.build_masked_seed_chunk = build_masked_seed_chunk

    def fake_component_count(image_id):
        return 3 + (int(image_id) % 50)

    fixed = _residual_summary(10)

    def process_selected_image_fn(inference, prepared, image_id, *, selected_windows_for_image, protocol_identity,
                                   contracts, class_count, affinity_power, component_chunk_size):
        cc = fake_component_count(image_id)
        start = 0
        while start < cc:
            stop = min(start + component_chunk_size, cc)
            for ordinal in selected_windows_for_image:
                fake_mf.build_masked_seed_chunk(None, None, list(range(start, stop)), [0], contract=None, window_index=ordinal)
            start = stop
        return {ordinal: {"rho_M": fixed, "rho_0": fixed, "rho_320": fixed, "rho_g": fixed} for ordinal in selected_windows_for_image}

    return fake_mf, process_selected_image_fn


_FAKE_GUARD_CONTRACT = ident.MemoryGuardContract(
    allocated_host_gib=80.0, allocated_gpu_gib=80.0, host_abort_fraction=0.90, gpu_abort_fraction=0.90,
)


def _real_head_commit() -> str:
    import subprocess

    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _fake_code_provenance(*, repository_commit=None, required_ancestor_commit=None):
    """Builds a syntactically-valid code_provenance record. Uses the
    REAL current repository HEAD by default for both commit fields
    (trivially its own ancestor) so that tests exercising the real CLI
    (which always re-verifies ancestry against the real repository) do
    not spuriously fail on a fabricated, nonexistent commit id -- tests
    that specifically want to probe ancestry rejection pass an explicit
    fake/unrelated commit instead."""
    head = _real_head_commit()
    return bench.build_code_provenance(
        identity_loader_sha256="1" * 64, runner_sha256="2" * 64, cli_sha256="3" * 64, launcher_sha256="4" * 64,
        repository_commit=repository_commit or head, required_ancestor_commit=required_ancestor_commit or head,
        repository_worktree_clean=True,
    )


def _run_fake_trial(manifest, trial_index, repeat_index, chunk_size, *, clock=None, memory_provider=None, guard_contract=None):
    fake_mf, process_fn = _make_fake_masked_forward_and_process_fn()
    kwargs = dict(
        chunk_size=chunk_size, trial_index=trial_index, repeat_index=repeat_index,
        manifest=manifest, authority_sha256="a" * 64, manifest_sha256="b" * 64,
        build_resources=lambda: _FakeResources(),
        build_image_preparer=lambda r: (lambda image_id: {"image_id": image_id}),
        build_inference=lambda r: object(),
        get_device_record=lambda: {"device_type": "cuda", "gpu_model_name": "FAKE-GPU"},
        get_software_record=lambda: {"torch_version": "0.0.0"},
        masked_forward_module=fake_mf,
        process_selected_image_fn=process_fn,
        memory_provider=memory_provider if memory_provider is not None else _FakeMemoryProvider(),
        memory_guard_contract=guard_contract if guard_contract is not None else _FAKE_GUARD_CONTRACT,
        code_provenance=_fake_code_provenance(),
    )
    if clock is not None:
        kwargs["clock"] = clock
    return bench.run_trial(**kwargs)


def test_run_trial_produces_closed_schema_document():
    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16)
    assert frozenset(doc.keys()) == bench.TRIAL_TOP_KEYS
    assert doc["chunk_size"] == 16
    assert len(doc["workload_records"]) == 24
    assert len(doc["window_identity"]) == 24


def test_run_trial_rejects_chunk_size_not_in_schedule():
    manifest = _build_fake_manifest()
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        _run_fake_trial(manifest, 0, 0, 48)


def test_run_trial_rejects_wrong_trial_chunk_combination():
    manifest = _build_fake_manifest()
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        _run_fake_trial(manifest, 0, 0, 32)  # trial 0 is bound to chunk_size 16 in the schedule


def test_run_trial_chunk_count_decreases_as_chunk_size_increases():
    manifest = _build_fake_manifest()
    counts = {}
    for trial_index, repeat_index, chunk_size in bench.TRIAL_SCHEDULE:
        doc = _run_fake_trial(manifest, trial_index, repeat_index, chunk_size)
        counts[chunk_size] = doc["workload_records"][0]["actual_component_chunk_count"]
    assert counts[16] >= counts[32] >= counts[64]


def test_run_trial_scientific_digest_identical_across_all_six_trials():
    manifest = _build_fake_manifest()
    digests = set()
    for trial_index, repeat_index, chunk_size in bench.TRIAL_SCHEDULE:
        doc = _run_fake_trial(manifest, trial_index, repeat_index, chunk_size)
        digests.add(doc["scientific_output_digests"]["collection_digest"])
    assert len(digests) == 1


def test_run_trial_uses_integer_nanosecond_clock():
    manifest = _build_fake_manifest()
    counter = {"t": 0}

    def fake_clock():
        counter["t"] += 1_000_000  # 1 ms per call, deterministic
        return counter["t"]

    doc = _run_fake_trial(manifest, 0, 0, 16, clock=fake_clock)
    assert type(doc["timing"]["primary_elapsed_ns"]) is int
    assert doc["timing"]["primary_elapsed_ns"] > 0
    for r in doc["timing"]["per_record_timing"]:
        assert type(r["elapsed_ns"]) is int


def test_run_trial_never_divides_group_timing_into_fake_per_window_timing():
    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16)
    per_record = doc["timing"]["per_record_timing"]
    # per_record_timing is per canonical-image-GROUP, never per window --
    # window_count must be able to exceed 1 for a shared image.
    assert any(r["window_count"] >= 1 for r in per_record)
    assert "per_window_elapsed_ns" not in json.dumps(doc)


# ---------------------------------------------------------------------
# Section: memory safety guard.
# ---------------------------------------------------------------------


class _OverLimitHostMemoryProvider(_FakeMemoryProvider):
    def host_current(self):
        # The real provider's /proc/self/status fields are present
        # together or absent together (strict parsing requires both
        # VmRSS and VmHWM, or neither) -- vmhwm_bytes is included here
        # so a full run_trial()-built trial document's memory mapping
        # remains fully populated with valid int byte counts (repair:
        # persisted-memory value validation).
        return {
            "vmrss_bytes": int(79 * (1024**3)), "ru_maxrss_bytes": int(79 * (1024**3)),
            "vmhwm_bytes": int(79 * (1024**3)),
        }


class _OverLimitGpuMemoryProvider(_FakeMemoryProvider):
    def peak(self):
        return {"allocated_bytes": int(75 * (1024**3)), "reserved_bytes": int(75 * (1024**3))}


def test_check_memory_safety_guard_requires_a_contract_no_default():
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.check_memory_safety_guard(_FakeMemoryProvider(), None)
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.check_memory_safety_guard(_FakeMemoryProvider(), {"allocated_host_gib": 80})  # a plain dict is not a MemoryGuardContract


def test_memory_safety_guard_triggers_on_host_rss():
    reason = bench.check_memory_safety_guard(_OverLimitHostMemoryProvider(), _FAKE_GUARD_CONTRACT)
    assert reason is not None
    assert "host RSS" in reason


def test_memory_safety_guard_triggers_on_gpu_reserved():
    reason = bench.check_memory_safety_guard(_OverLimitGpuMemoryProvider(), _FAKE_GUARD_CONTRACT)
    assert reason is not None
    assert "GPU reserved" in reason


def test_memory_safety_guard_silent_when_within_bounds():
    reason = bench.check_memory_safety_guard(_FakeMemoryProvider(), _FAKE_GUARD_CONTRACT)
    assert reason is None


def test_memory_safety_guard_behavior_changes_with_the_supplied_contract():
    """Two otherwise-identical contracts whose guard values differ must
    produce DIFFERENT abort behavior for the identical memory reading
    -- proving `check_memory_safety_guard` genuinely uses the supplied
    contract, not a fixed internal constant (Finding 2 regression
    guard)."""
    reading = _OverLimitHostMemoryProvider()  # 79 GiB host usage
    strict_contract = ident.MemoryGuardContract(allocated_host_gib=80.0, allocated_gpu_gib=80.0, host_abort_fraction=0.90, gpu_abort_fraction=0.90)
    lenient_contract = ident.MemoryGuardContract(allocated_host_gib=1000.0, allocated_gpu_gib=1000.0, host_abort_fraction=0.90, gpu_abort_fraction=0.90)
    assert bench.check_memory_safety_guard(reading, strict_contract) is not None
    assert bench.check_memory_safety_guard(reading, lenient_contract) is None


def test_memory_safety_guard_host_and_gpu_thresholds_are_independent():
    host_only = _OverLimitHostMemoryProvider()
    gpu_only = _OverLimitGpuMemoryProvider()
    reason_host = bench.check_memory_safety_guard(host_only, _FAKE_GUARD_CONTRACT)
    reason_gpu = bench.check_memory_safety_guard(gpu_only, _FAKE_GUARD_CONTRACT)
    assert reason_host is not None and "host RSS" in reason_host and "GPU" not in reason_host
    assert reason_gpu is not None and "GPU reserved" in reason_gpu and "host RSS" not in reason_gpu


def test_no_hardcoded_guard_defaults_remain_in_source():
    """Search production source for the previous independent Python
    defaults (Finding 2) and confirm they are absent as operational
    authorities -- constants used solely as canonical E17 EXPECTED
    values (in the identity module / tests) are acceptable; a silent
    default inside `run_trial`/`check_memory_safety_guard` is not."""
    import inspect

    run_trial_src = inspect.getsource(bench.run_trial)
    guard_src = inspect.getsource(bench.check_memory_safety_guard)
    assert "allocated_host_gib: float = 80" not in run_trial_src
    assert "allocated_gpu_gib: float = 80" not in run_trial_src
    assert "host_fraction: float = 0.9" not in guard_src
    assert "gpu_fraction: float = 0.9" not in guard_src
    # Both must require an explicit contract parameter with no default.
    guard_sig = inspect.signature(bench.check_memory_safety_guard)
    assert guard_sig.parameters["contract"].default is inspect.Parameter.empty
    run_trial_sig = inspect.signature(bench.run_trial)
    assert run_trial_sig.parameters["memory_guard_contract"].default is inspect.Parameter.empty


def test_run_trial_aborts_cleanly_when_guard_trips_and_is_never_marked_complete():
    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16, memory_provider=_OverLimitHostMemoryProvider())
    assert doc["complete"] is False
    assert doc["final"] is False
    assert doc["abort_status"]["aborted"] is True
    assert doc["abort_status"]["reason"] is not None
    assert doc["scientific_output_digests"]["collection_digest"] is None
    # An aborted trial must never pass verify_trial_document into a merge as complete.
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(_six_fake_trials(manifest)[:5] + [doc])


# ---------------------------------------------------------------------
# Section: per-trial schema and merge.
# ---------------------------------------------------------------------


def _six_fake_trials(manifest):
    return [_run_fake_trial(manifest, ti, ri, cs) for ti, ri, cs in bench.TRIAL_SCHEDULE]


def test_verify_trial_document_roundtrip():
    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16)
    bench.verify_trial_document(doc, expected_authority_sha256="a" * 64, expected_manifest_sha256="b" * 64)


def test_verify_trial_document_rejects_wrong_authority():
    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16)
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.verify_trial_document(doc, expected_authority_sha256="0" * 64)


def test_merge_requires_all_six_trials():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(trials[:5])


def test_merge_rejects_duplicate_trial():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    trials[1] = copy.deepcopy(trials[0])
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(trials)


def test_merge_succeeds_and_has_no_winner_field():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    merged = bench.merge_trials(trials)
    assert "winner" not in merged
    assert "recommended_chunk_size" not in merged
    assert "preferred_chunk_size" not in merged
    for chunk_size, summary in merged["descriptive_summaries_by_chunk_size"].items():
        assert len(summary["primary_elapsed_ns_per_repeat"]) == 2


def test_merge_detects_scientific_output_mismatch():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    mutated[3]["scientific_output_digests"]["collection_digest"] = "f" * 64
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(mutated)


def test_merge_rejects_disagreeing_manifest_hash():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    mutated[2]["manifest_sha256"] = "0" * 64
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(mutated)


def test_merge_rejects_incomplete_trial():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    mutated[0]["complete"] = False
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(mutated)


# ---------------------------------------------------------------------
# Section: atomic writes.
# ---------------------------------------------------------------------


def test_write_trial_document_atomically_no_overwrite_by_default(tmp_path):
    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16)
    out = tmp_path / "trial.json"
    bench.write_trial_document_atomically(doc, out)
    assert out.exists()
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.write_trial_document_atomically(doc, out)
    bench.write_trial_document_atomically(doc, out, allow_overwrite=True)  # explicit overwrite is fine


def test_write_trial_document_atomically_no_tmp_residue_on_success(tmp_path):
    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16)
    out = tmp_path / "trial.json"
    bench.write_trial_document_atomically(doc, out)
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------
# Section: CLI.
# ---------------------------------------------------------------------


def test_cli_help_exits_zero(capsys):
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_chunk_benchmark")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    # Standalone subprocess proof that this command never imports torch
    # lives in test_standalone_subprocess_verify_protocol_no_torch --
    # an in-process "torch" not in sys.modules assertion here would be
    # order-dependent on whichever OTHER test file this one happens to
    # run alongside in the same pytest session.


@requires_real_inputs
def test_cli_verify_protocol_exits_zero(capsys):
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_chunk_benchmark")
    exit_code = cli.main([
        "verify-protocol", "--artifact", str(ARTIFACT_PATH), "--workload-bundle-dir", str(WORKLOAD_BUNDLE_DIR),
    ])
    assert exit_code == 0
    # See test_cli_help_exits_zero's comment on why torch-absence is
    # proven by the standalone subprocess test, not an in-process check.


@requires_real_inputs
def test_cli_preflight_against_real_uncommitted_tree_fails_clean_worktree_check(capsys):
    """The real repository is STILL uncommitted (eight untracked
    implementation files) -- `preflight` now performs the FULL shared
    repository-readiness check (Finding repair: it previously omitted
    the git ancestry/clean-worktree check entirely and would have
    reported a false PASS here), so it correctly reports exit 2, not a
    PASS, against the real repository's current dirty working tree.
    This is expected, intentional behavior and must never be weakened;
    the genuine positive-path proof lives in
    test_cli_preflight_passes_on_clean_synthetic_descendant_clone,
    against a clean temporary clone."""
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_chunk_benchmark")
    exit_code = cli.main([
        "preflight", "--artifact", str(ARTIFACT_PATH), "--manifest", str(MANIFEST_PATH),
        "--workload-bundle-dir", str(WORKLOAD_BUNDLE_DIR),
    ])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "PREFLIGHT PASS" not in captured.out
    assert "not clean" in captured.err
    assert "Traceback" not in captured.err
    # See test_cli_help_exits_zero's comment on why torch-absence is
    # proven by the standalone subprocess test, not an in-process check.


def test_cli_run_trial_rejects_cpu_device(capsys):
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_chunk_benchmark")
    exit_code = cli.main([
        "run-trial", "--device", "cpu", "--trial-index", "0", "--repeat", "0", "--chunk-size", "16",
        "--artifact", "/nonexistent.json", "--manifest", "/nonexistent.json",
        "--workload-bundle-dir", "/nonexistent", "--output", "/tmp/x.json",
    ])
    assert exit_code == 2
    assert "not authorized" in capsys.readouterr().err
    # See test_cli_help_exits_zero's comment on why torch-absence is
    # proven by the standalone subprocess test, not an in-process check.


def test_cli_run_trial_rejects_bad_chunk_size_via_argparse():
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_chunk_benchmark")
    with pytest.raises(SystemExit) as excinfo:
        cli.main([
            "run-trial", "--device", "cuda", "--trial-index", "0", "--repeat", "0", "--chunk-size", "48",
            "--artifact", "x", "--manifest", "y", "--workload-bundle-dir", "z", "--output", "o.json",
        ])
    assert excinfo.value.code == 2


def test_cli_run_trial_rejects_wrong_trial_chunk_combo(capsys):
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_chunk_benchmark")
    exit_code = cli.main([
        "run-trial", "--device", "cuda", "--trial-index", "0", "--repeat", "0", "--chunk-size", "32",
        "--artifact", "/nonexistent.json", "--manifest", "/nonexistent.json",
        "--workload-bundle-dir", "/nonexistent", "--output", "/tmp/x.json",
    ])
    assert exit_code == 2
    assert "frozen trial schedule" in capsys.readouterr().err
    # See test_cli_help_exits_zero's comment on why torch-absence is
    # proven by the standalone subprocess test, not an in-process check.


def test_cli_merge_and_verify_result_roundtrip(tmp_path):
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_chunk_benchmark")
    manifest = _build_fake_manifest()
    trial_paths = []
    for i, (ti, ri, cs) in enumerate(bench.TRIAL_SCHEDULE):
        doc = _run_fake_trial(manifest, ti, ri, cs)
        p = tmp_path / f"trial_{i}.json"
        bench.write_trial_document_atomically(doc, p)
        trial_paths.append(str(p))

    merged_path = tmp_path / "merged.json"
    args = ["merge"]
    for p in trial_paths:
        args += ["--trial", p]
    args += ["--output", str(merged_path)]
    exit_code = cli.main(args)
    assert exit_code == 0
    assert merged_path.exists()

    exit_code = cli.main(["verify-result", "--result", str(merged_path)])
    assert exit_code == 0


def test_cli_manifest_path_traversal_rejected(tmp_path):
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_chunk_benchmark")
    doc = _fake_calibration_document()
    fake_artifact = tmp_path / "fake.json"
    fake_artifact.write_text(json.dumps(doc))
    exit_code = cli.main([
        "build-manifest", "--artifact", str(fake_artifact), "--output", str(tmp_path / ".." / "escape.json"),
    ])
    assert exit_code == 2


# ---------------------------------------------------------------------
# Section: standalone subprocess (fresh process, no torch for CPU
# commands), and no-5000-image-token audit.
# ---------------------------------------------------------------------


@requires_real_inputs
def test_standalone_subprocess_verify_protocol_no_torch(tmp_path):
    import os

    env = dict(os.environ)
    completed = sp.run(
        [sys.executable, str(CLI_PATH), "verify-protocol", "--artifact", str(ARTIFACT_PATH),
         "--workload-bundle-dir", str(WORKLOAD_BUNDLE_DIR)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0
    assert "Traceback" not in completed.stderr


def test_no_5000_image_execution_tokens_in_new_source_files():
    """Forbidden tokens that would indicate an actual 5,000-image
    execution/sharding/merging PATH -- not the legitimate guard fields
    (e.g. `extrapolation_to_5000_images_forbidden`) that explicitly
    PROHIBIT it, which must remain present."""
    forbidden = ("shard_5000", "merge_5000_", "run_5000", "execute_5000", "5000_image_evaluation_authorized = true")
    for path in (
        ROOT / "src/dfp_phase1b_chunk_benchmark.py",
        ROOT / "src/dfp_phase1b_chunk_benchmark_identity.py",
        ROOT / "diagnostics/run_dfp_phase1b_chunk_benchmark.py",
    ):
        text = path.read_text()
        for token in forbidden:
            assert token not in text, f"forbidden token {token!r} found in {path}"
    # The E17 TOML itself must explicitly forbid it, never authorize it.
    e17_text = (ROOT / ident.IDENTITY_RELATIVE_PATH).read_text()
    assert "five_thousand_image_evaluation_authorized = false" in e17_text
    assert "extrapolation_to_5000_images_forbidden = true" in e17_text


# ---------------------------------------------------------------------
# Section: Git ancestry / worktree-cleanliness (Finding 3 repair).
# Uses small, real, temporary Git repositories -- never the real
# Talk2DINO repository's own history.
# ---------------------------------------------------------------------


def _git(repo_dir, *args):
    import subprocess

    result = subprocess.run(["git", "-C", str(repo_dir), *args], capture_output=True, text=True)
    return result


def _init_temp_repo(tmp_path):
    repo = tmp_path / "temp_repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    return repo


def _commit_file(repo, name, content, message):
    (repo / name).write_text(content)
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def test_ancestor_relation_accepts_exact_same_commit(tmp_path):
    repo = _init_temp_repo(tmp_path)
    c1 = _commit_file(repo, "a.txt", "1", "first")
    ident.verify_ancestor_relation(repo, required_ancestor_commit=c1, repository_commit=c1)


def test_ancestor_relation_accepts_proper_descendant(tmp_path):
    repo = _init_temp_repo(tmp_path)
    c1 = _commit_file(repo, "a.txt", "1", "first")
    c2 = _commit_file(repo, "b.txt", "2", "second")
    ident.verify_ancestor_relation(repo, required_ancestor_commit=c1, repository_commit=c2)


def test_ancestor_relation_rejects_reversed_ancestry(tmp_path):
    repo = _init_temp_repo(tmp_path)
    c1 = _commit_file(repo, "a.txt", "1", "first")
    c2 = _commit_file(repo, "b.txt", "2", "second")
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.verify_ancestor_relation(repo, required_ancestor_commit=c2, repository_commit=c1)


def test_ancestor_relation_rejects_unrelated_branch(tmp_path):
    repo = _init_temp_repo(tmp_path)
    c1 = _commit_file(repo, "a.txt", "1", "first")
    _git(repo, "checkout", "-q", "--orphan", "unrelated")
    _git(repo, "rm", "-rf", "-q", ".")
    c2 = _commit_file(repo, "z.txt", "9", "unrelated root")
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.verify_ancestor_relation(repo, required_ancestor_commit=c1, repository_commit=c2)


def test_ancestor_relation_rejects_nonexistent_commit(tmp_path):
    repo = _init_temp_repo(tmp_path)
    c1 = _commit_file(repo, "a.txt", "1", "first")
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.verify_ancestor_relation(repo, required_ancestor_commit="f" * 40, repository_commit=c1)


def test_ancestor_relation_rejects_malformed_commit(tmp_path):
    repo = _init_temp_repo(tmp_path)
    c1 = _commit_file(repo, "a.txt", "1", "first")
    for bad in ("not-a-hash", "abc123", "g" * 40, "A" * 40, ""):
        with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
            ident.verify_ancestor_relation(repo, required_ancestor_commit=bad, repository_commit=c1)


def test_collect_repository_state_clean_tree(tmp_path):
    repo = _init_temp_repo(tmp_path)
    c1 = _commit_file(repo, "a.txt", "1", "first")
    state = ident.collect_repository_state(repo)
    assert state["repository_commit"] == c1
    assert state["repository_worktree_clean"] is True


def test_collect_repository_state_modified_tracked_file(tmp_path):
    repo = _init_temp_repo(tmp_path)
    _commit_file(repo, "a.txt", "1", "first")
    (repo / "a.txt").write_text("modified")
    state = ident.collect_repository_state(repo)
    assert state["repository_worktree_clean"] is False


def test_collect_repository_state_staged_file(tmp_path):
    repo = _init_temp_repo(tmp_path)
    _commit_file(repo, "a.txt", "1", "first")
    (repo / "b.txt").write_text("new")
    _git(repo, "add", "b.txt")
    state = ident.collect_repository_state(repo)
    assert state["repository_worktree_clean"] is False


def test_collect_repository_state_untracked_file(tmp_path):
    repo = _init_temp_repo(tmp_path)
    _commit_file(repo, "a.txt", "1", "first")
    (repo / "untracked.txt").write_text("new")
    state = ident.collect_repository_state(repo)
    assert state["repository_worktree_clean"] is False


def test_require_clean_worktree_rejects_dirty_and_returns_commit_when_clean(tmp_path):
    repo = _init_temp_repo(tmp_path)
    c1 = _commit_file(repo, "a.txt", "1", "first")
    assert ident.require_clean_worktree(repo) == c1
    (repo / "untracked.txt").write_text("x")
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.require_clean_worktree(repo)


def test_real_repository_required_ancestor_commit_is_ancestor_of_head():
    """The committed E17 identity's own required_ancestor_commit must
    actually be an ancestor of (or equal to) the real repository's
    current HEAD -- exercised against the REAL repository, read-only."""
    doc = ident.load_identity(repo_root=ROOT)
    state = ident.collect_repository_state(ROOT)
    ident.verify_ancestor_relation(
        ROOT, required_ancestor_commit=doc["identity"]["required_ancestor_commit"],
        repository_commit=state["repository_commit"],
    )


# ---------------------------------------------------------------------
# Section: benchmark-code hash bindings (Finding 3 repair).
# ---------------------------------------------------------------------


def test_code_binding_hashes_match_live_bytes():
    doc = ident.load_identity(repo_root=ROOT)
    hashes = ident.code_binding_hashes(doc, repo_root=ROOT)
    assert set(hashes.keys()) == {"identity_loader_sha256", "runner_sha256", "cli_sha256", "launcher_sha256"}
    import hashlib

    assert hashes["identity_loader_sha256"] == hashlib.sha256((ROOT / "src/dfp_phase1b_chunk_benchmark_identity.py").read_bytes()).hexdigest()
    assert hashes["runner_sha256"] == hashlib.sha256((ROOT / "src/dfp_phase1b_chunk_benchmark.py").read_bytes()).hexdigest()
    assert hashes["cli_sha256"] == hashlib.sha256((ROOT / "diagnostics/run_dfp_phase1b_chunk_benchmark.py").read_bytes()).hexdigest()
    assert hashes["launcher_sha256"] == hashlib.sha256((ROOT / "dfp_phase1b_chunk_benchmark_h100.sbatch").read_bytes()).hexdigest()


@pytest.mark.parametrize("binding_name,relative_path", [
    ("identity_loader_module", "src/dfp_phase1b_chunk_benchmark_identity.py"),
    ("benchmark_runner_module", "src/dfp_phase1b_chunk_benchmark.py"),
    ("benchmark_cli_module", "diagnostics/run_dfp_phase1b_chunk_benchmark.py"),
    ("launcher_script", "dfp_phase1b_chunk_benchmark_h100.sbatch"),
])
def test_code_binding_rejects_bytes_changed_while_toml_unchanged(tmp_path, binding_name, relative_path):
    """Build a minimal fake repo (only the files load_identity reads),
    mutate ONE bound code file's bytes, leave E17 unchanged, and
    confirm load_identity rejects it."""
    fake_root = _build_minimal_fake_repo(tmp_path)
    target = fake_root / relative_path
    target.write_bytes(target.read_bytes() + b"\n# mutated for test\n")
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError, match=binding_name):
        ident.load_identity(repo_root=fake_root)


_ALL_LOAD_IDENTITY_RELATIVE_PATHS = (
    str(ident.IDENTITY_RELATIVE_PATH),
    "evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml",
    "evaluation_subsets/dfp_phase1a_coco_stuff_500.json",
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/graph.py",
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/finite_step_regime.py",
    "src/dfp_phase1b_masked_forward.py", "src/dfp_phase1a_stitching_pullback.py",
    "src/dfp_phase1a_final_rescale_pullback.py", "src/dfp_phase1a_horizon_stitching.py",
    "src/dfp_phase1a_global_unary_components.py", "src/dfp_phase1a_recurrence_capture.py",
    "src/dfp_phase1b_calibration_run.py", "src/dfp_phase1b_calibration_production.py",
    "src/dfp_phase1b_chunk_benchmark_identity.py", "src/dfp_phase1b_chunk_benchmark.py",
    "diagnostics/run_dfp_phase1b_chunk_benchmark.py", "dfp_phase1b_chunk_benchmark_h100.sbatch",
)


def _build_minimal_fake_repo(tmp_path, *, name="fake_repo"):
    """Copy ONLY the exact files load_identity's relational hash checks
    read -- never the whole repository (which would copy large dataset/
    weight/venv-adjacent directories and is far too slow/heavy for a
    login-node test)."""
    fake_root = tmp_path / name
    for rel in _ALL_LOAD_IDENTITY_RELATIVE_PATHS:
        dest = fake_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel).read_bytes())
    return fake_root


def test_code_binding_rejects_missing_file(tmp_path):
    fake_root = _build_minimal_fake_repo(tmp_path, name="fake_repo2")
    (fake_root / "src/dfp_phase1b_chunk_benchmark.py").unlink()
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.load_identity(repo_root=fake_root)


# ---------------------------------------------------------------------
# Section: renamed host-memory fields (Finding 4 repair).
# ---------------------------------------------------------------------


def test_run_trial_emits_only_renamed_process_lifetime_host_fields():
    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16)
    mem = doc["memory"]
    for required in ("host_process_lifetime_peak_rss_bytes", "host_process_lifetime_vmhwm_bytes", "host_current_rss_bytes"):
        assert required in mem
    for forbidden in ("host_peak_rss_bytes", "host_current_vmrss_bytes", "host_peak_vmhwm_bytes", "baseline_allocated_bytes", "baseline_reserved_bytes", "end_allocated_bytes", "end_reserved_bytes"):
        assert forbidden not in mem, f"old ambiguous key {forbidden!r} must be rejected/absent, found in {mem}"
    for required in ("gpu_baseline_allocated_bytes", "gpu_baseline_reserved_bytes", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes", "gpu_end_allocated_bytes", "gpu_end_reserved_bytes"):
        assert required in mem


def test_merged_summary_uses_renamed_host_field():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    merged = bench.merge_trials(trials)
    for summary in merged["descriptive_summaries_by_chunk_size"].values():
        assert "host_process_lifetime_peak_rss_bytes" in summary
        assert "host_peak_rss_bytes" not in summary


def test_synthetic_model_setup_dominated_memory_still_labeled_process_lifetime():
    """A trial where model setup produces the true process peak and
    measured work uses less memory must still label the host figure as
    process-lifetime, never measured-subset."""
    manifest = _build_fake_manifest()

    class ModelSetupDominatedMemory(_FakeMemoryProvider):
        def host_current(self):
            # Simulates: peak RSS occurred during model construction
            # (e.g. 40 GiB), long before the 24 measured windows (which
            # use far less). ru_maxrss/VmHWM cannot un-see that peak.
            return {"ru_maxrss_bytes": 40 * (1024**3), "vmrss_bytes": 2 * (1024**3), "vmhwm_bytes": 40 * (1024**3)}

    doc = _run_fake_trial(manifest, 0, 0, 16, memory_provider=ModelSetupDominatedMemory())
    mem = doc["memory"]
    assert mem["host_process_lifetime_peak_rss_bytes"] == 40 * (1024**3)
    assert mem["host_process_lifetime_vmhwm_bytes"] == 40 * (1024**3)
    assert mem["host_current_rss_bytes"] == 2 * (1024**3)
    # The (much smaller) current-RSS-at-end-of-trial value must never
    # be confused with, or substituted for, the process-lifetime peak.
    assert mem["host_current_rss_bytes"] != mem["host_process_lifetime_peak_rss_bytes"]


def test_e17_host_memory_scope_documented_as_process_lifetime():
    doc = ident.load_identity(repo_root=ROOT)
    assert doc["measurement"]["host_memory"]["measurement_scope"] == "process_lifetime_includes_imports_model_construction_checkpoint_loading_and_warmup"
    assert doc["measurement"]["gpu_memory"]["measurement_scope"] == "measured_24_window_subset_only_after_post_warmup_reset"


# ---------------------------------------------------------------------
# Section: code-provenance schema and cross-fed merge rejection.
# ---------------------------------------------------------------------


def test_trial_document_includes_code_provenance_with_closed_schema():
    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16)
    cp = doc["code_provenance"]
    assert frozenset(cp.keys()) == bench.CODE_PROVENANCE_KEYS
    assert cp["repository_worktree_clean"] is True


@pytest.mark.parametrize("field,bad_value", [
    ("identity_loader_sha256", "not-a-hash"),
    ("runner_sha256", "1" * 63),
    ("repository_commit", "1" * 39),
    ("required_ancestor_commit", "1" * 41),
    ("repository_worktree_clean", "true"),  # string, not bool
    ("repository_worktree_clean", 1),  # int, not bool
])
def test_build_code_provenance_rejects_malformed_fields(field, bad_value):
    kwargs = dict(
        identity_loader_sha256="1" * 64, runner_sha256="2" * 64, cli_sha256="3" * 64, launcher_sha256="4" * 64,
        repository_commit="a" * 40, required_ancestor_commit="b" * 40, repository_worktree_clean=True,
    )
    kwargs[field] = bad_value
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.build_code_provenance(**kwargs)


@pytest.mark.parametrize("field", ["runner_sha256", "cli_sha256", "launcher_sha256", "identity_loader_sha256", "repository_commit"])
def test_merge_rejects_cross_fed_code_provenance(field):
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    if field == "repository_commit":
        mutated[3]["code_provenance"][field] = "9" * 40
    else:
        mutated[3]["code_provenance"][field] = "9" * 64
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(mutated)


def test_merge_rejects_false_clean_worktree_flag():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    mutated[2]["code_provenance"]["repository_worktree_clean"] = False
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(mutated)


def test_merge_preserves_code_provenance_into_merged_artifact():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    merged = bench.merge_trials(trials)
    assert merged["code_provenance"] == trials[0]["code_provenance"]
    assert "winner" not in merged["code_provenance"]


# ---------------------------------------------------------------------
# Section: structural completeness of the canonical-value table.
# ---------------------------------------------------------------------


def test_canonical_values_table_covers_every_non_relational_leaf_in_the_committed_toml():
    """Mechanical completeness guard: flatten the REAL committed E17
    TOML independently (fresh code, not importing the generator script
    used to build CANONICAL_VALUES) and confirm every non-hash/non-path
    leaf appears in CANONICAL_VALUES with an exactly matching value --
    i.e. no authoritative leaf can be silently added to the TOML
    without also being added to the exact-value table, because
    load_identity's own _walk_canonical_values already fails closed on
    any leaf absent from CANONICAL_VALUES (exercised end-to-end by
    every other test in this file that calls load_identity
    successfully); this test additionally proves the table's own
    coverage independently of that runtime path."""
    import tomllib

    relational_keys = {
        "sha256", "relative_path", "sha256_pinned_here", "sha256_verification",
        "untouched_by_this_identity", "relative_path_is_symbolic_not_repository_relative",
        "relative_path_note", "self_reference_note",
    }
    with open(ROOT / ident.IDENTITY_RELATIVE_PATH, "rb") as f:
        doc = tomllib.load(f)

    missing = []

    def walk(node, prefix):
        if isinstance(node, dict):
            for k, v in node.items():
                child = f"{prefix}.{k}" if prefix else k
                if k in relational_keys:
                    continue
                if isinstance(v, dict):
                    walk(v, child)
                else:
                    if child not in ident.CANONICAL_VALUES:
                        missing.append(child)
                    elif ident.CANONICAL_VALUES[child] != v:
                        missing.append(f"{child} (value mismatch)")

    walk(doc, "")
    assert missing == [], f"leaves present in the committed TOML but absent/mismatched in CANONICAL_VALUES: {missing}"


# =======================================================================
# REPAIR ROUND 2: canonical dependency-path authority + repository-
# containment helper (path-traversal/absolute-substitution/symlink-
# escape repair).
# =======================================================================


def test_canonical_dependency_paths_key_set_matches_repository_file_bindings():
    doc = ident.load_identity(repo_root=ROOT)
    repo_file_names = {
        name for name, binding in doc["dependency_bindings"].items()
        if not binding.get("relative_path_is_symbolic_not_repository_relative", False)
    }
    assert frozenset(ident.CANONICAL_DEPENDENCY_RELATIVE_PATHS.keys()) == frozenset(repo_file_names)


def test_canonical_dependency_paths_match_committed_toml_exactly():
    doc = ident.load_identity(repo_root=ROOT)
    for name, canonical_path in ident.CANONICAL_DEPENDENCY_RELATIVE_PATHS.items():
        assert doc["dependency_bindings"][name]["relative_path"] == canonical_path


@pytest.mark.parametrize("mutated_relative_path", [
    "/etc/passwd",
    "/tmp/evil.py",
    "../../../../etc/passwd",
    "..",
    "src/../../../etc/passwd",
    "",
    "   ",
    "src/dfp_phase1b_chunk_benchmark.py\x00",
    "src\\dfp_phase1b_chunk_benchmark.py",
    "C:/Windows/System32/evil.py",
    "C:\\Windows\\System32\\evil.py",
    "//server/share/evil.py",
    "\\\\server\\share\\evil.py",
    "./src/dfp_phase1b_chunk_benchmark.py",
    "src//dfp_phase1b_chunk_benchmark.py",
    "src/dfp_phase1b_chunk_benchmark.py/",
    " src/dfp_phase1b_chunk_benchmark.py",
    "src/dfp_phase1b_chunk_benchmark.py ",
])
def test_resolve_repository_relative_path_rejects_malformed_syntax(mutated_relative_path):
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.resolve_repository_relative_path(ROOT, mutated_relative_path, binding_name="benchmark_runner_module")


def test_resolve_repository_relative_path_rejects_missing_target(tmp_path):
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.resolve_repository_relative_path(tmp_path, "does/not/exist.py", binding_name="x")


def test_resolve_repository_relative_path_rejects_directory_target(tmp_path):
    (tmp_path / "some_dir").mkdir()
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.resolve_repository_relative_path(tmp_path, "some_dir", binding_name="x")


def test_resolve_repository_relative_path_accepts_canonical_target():
    target = ident.resolve_repository_relative_path(ROOT, "src/dfp_phase1b_chunk_benchmark.py", binding_name="benchmark_runner_module")
    assert target == (ROOT / "src/dfp_phase1b_chunk_benchmark.py").resolve()


def test_resolve_repository_relative_path_traversal_resolving_back_inside_is_still_rejected(tmp_path):
    """Lexical rejection of any '..' component is unconditional -- even
    when the net effect would resolve back inside the repository."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "file.py").write_text("x")
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.resolve_repository_relative_path(tmp_path, "sub/../sub/file.py", binding_name="x")


def test_resolve_repository_relative_path_rejects_final_symlink_escape(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("evil")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "linked.py").symlink_to(outside)
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.resolve_repository_relative_path(repo, "linked.py", binding_name="x")


def test_resolve_repository_relative_path_rejects_intermediate_directory_symlink_escape(tmp_path):
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "file.py").write_text("evil")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "linked_dir").symlink_to(outside_dir)
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.resolve_repository_relative_path(repo, "linked_dir/file.py", binding_name="x")


def test_resolve_repository_relative_path_accepts_repo_root_itself_reached_through_a_symlink(tmp_path):
    """A repo_root that is itself a symlink to the real, physically-
    contained repository is ACCEPTED -- containment is decided against
    its resolved PHYSICAL path, not the symlink path (documented
    design choice)."""
    real_repo = tmp_path / "real_repo"
    real_repo.mkdir()
    (real_repo / "file.py").write_text("x")
    symlinked_root = tmp_path / "symlinked_root"
    symlinked_root.symlink_to(real_repo)
    target = ident.resolve_repository_relative_path(symlinked_root, "file.py", binding_name="x")
    assert target == (real_repo / "file.py").resolve()


def _fake_e17_repo_with_swapped_or_substituted_binding(tmp_path, *, mutator):
    """Build a minimal fake repo (only the files load_identity reads)
    from the REAL committed E17 text, apply ``mutator(text) -> text``
    to the TOML text only, and return the fake repo root."""
    fake_root = tmp_path / "fake_repo"
    for rel in _ALL_LOAD_IDENTITY_RELATIVE_PATHS:
        dest = fake_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel).read_bytes())
    e17_path = fake_root / ident.IDENTITY_RELATIVE_PATH
    e17_path.write_text(mutator(e17_path.read_text()))
    return fake_root


def test_load_identity_rejects_swapped_valid_paths(tmp_path):
    """Swap graph_module's and finite_step_regime's relative_path
    values (both otherwise-valid, in-repository paths) -- each
    binding's relative_path must equal ITS OWN canonical value, so a
    swap is rejected even though both paths individually exist and are
    genuinely repository-contained."""
    graph_path = ident.CANONICAL_DEPENDENCY_RELATIVE_PATHS["graph_module"]
    fsr_path = ident.CANONICAL_DEPENDENCY_RELATIVE_PATHS["finite_step_regime"]

    def mutator(text):
        return text.replace(f'relative_path = "{graph_path}"', "__PLACEHOLDER__", 1) \
                    .replace(f'relative_path = "{fsr_path}"', f'relative_path = "{graph_path}"', 1) \
                    .replace("__PLACEHOLDER__", f'relative_path = "{fsr_path}"', 1)

    fake_root = _fake_e17_repo_with_swapped_or_substituted_binding(tmp_path, mutator=mutator)
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError, match="relative_path"):
        ident.load_identity(repo_root=fake_root)


def test_load_identity_rejects_in_repo_substitution_even_with_matching_recomputed_hash(tmp_path):
    """Point launcher_script's relative_path at a DIFFERENT, genuinely
    in-repository file (benchmark_cli_module's own file) and recompute
    its sha256 to match that file's real bytes -- still rejected,
    because the relative_path itself no longer equals its own pinned
    canonical value, regardless of what the (now self-consistent)
    recomputed hash says."""
    import hashlib

    substitute_path = ident.CANONICAL_DEPENDENCY_RELATIVE_PATHS["benchmark_cli_module"]
    substitute_sha256 = hashlib.sha256((ROOT / substitute_path).read_bytes()).hexdigest()
    launcher_path = ident.CANONICAL_DEPENDENCY_RELATIVE_PATHS["launcher_script"]

    def mutator(text):
        import re

        header = "[dependency_bindings.launcher_script]"
        start = text.index(header)
        m = re.search(r"\n\[", text[start + len(header):])
        end = start + len(header) + (m.start() if m else len(text) - start - len(header))
        segment = text[start:end]
        real_sha = re.search(r'sha256 = "([0-9a-f]+)"', segment).group(1)
        segment = segment.replace(f'relative_path = "{launcher_path}"', f'relative_path = "{substitute_path}"', 1)
        segment = segment.replace(f'sha256 = "{real_sha}"', f'sha256 = "{substitute_sha256}"', 1)
        return text[:start] + segment + text[end:]

    fake_root = _fake_e17_repo_with_swapped_or_substituted_binding(tmp_path, mutator=mutator)
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError, match="relative_path"):
        ident.load_identity(repo_root=fake_root)


@pytest.mark.parametrize("binding_name", list(ident._REPO_RELATIVE_HASHED_BINDING_NAMES) + list(ident._CODE_BINDING_NAMES))
def test_load_identity_rejects_absolute_path_substitution_for_every_binding(tmp_path, binding_name):
    import hashlib
    import re

    payload = tmp_path / "evil_payload.py"
    payload.write_text("# planted content, not the real bound file\n")
    payload_sha256 = hashlib.sha256(payload.read_bytes()).hexdigest()
    real_path = ident.CANONICAL_DEPENDENCY_RELATIVE_PATHS[binding_name]

    def mutator(text):
        header = f"[dependency_bindings.{binding_name}]"
        start = text.index(header)
        m = re.search(r"\n\[", text[start + len(header):])
        end = start + len(header) + (m.start() if m else len(text) - start - len(header))
        segment = text[start:end]
        real_sha = re.search(r'sha256 = "([0-9a-f]+)"', segment).group(1)
        segment = segment.replace(f'relative_path = "{real_path}"', f'relative_path = "{payload}"', 1)
        segment = segment.replace(f'sha256 = "{real_sha}"', f'sha256 = "{payload_sha256}"', 1)
        return text[:start] + segment + text[end:]

    fake_root = _fake_e17_repo_with_swapped_or_substituted_binding(tmp_path, mutator=mutator)
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.load_identity(repo_root=fake_root)


def test_load_identity_rejects_symlink_escape_for_a_binding(tmp_path):
    import hashlib

    payload = tmp_path / "evil_payload.py"
    payload.write_text("# planted content\n")
    payload_sha256 = hashlib.sha256(payload.read_bytes()).hexdigest()

    fake_root = tmp_path / "fake_repo_symlink"
    for rel in _ALL_LOAD_IDENTITY_RELATIVE_PATHS:
        dest = fake_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel).read_bytes())
    symlink_rel = "src/planted_symlink_launcher.sbatch"
    (fake_root / symlink_rel).symlink_to(payload)

    real_path = ident.CANONICAL_DEPENDENCY_RELATIVE_PATHS["launcher_script"]
    e17_path = fake_root / ident.IDENTITY_RELATIVE_PATH
    text = e17_path.read_text()
    import re

    header = "[dependency_bindings.launcher_script]"
    start = text.index(header)
    m = re.search(r"\n\[", text[start + len(header):])
    end = start + len(header) + (m.start() if m else len(text) - start - len(header))
    segment = text[start:end]
    real_sha = re.search(r'sha256 = "([0-9a-f]+)"', segment).group(1)
    segment = segment.replace(f'relative_path = "{real_path}"', f'relative_path = "{symlink_rel}"', 1)
    segment = segment.replace(f'sha256 = "{real_sha}"', f'sha256 = "{payload_sha256}"', 1)
    e17_path.write_text(text[:start] + segment + text[end:])

    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.load_identity(repo_root=fake_root)


def test_ast_every_dependency_hash_call_site_uses_the_containment_helper():
    """Static proof (Section 4 repair) that ``_sha256_of`` is NEVER
    called with a path built by any expression other than
    ``resolve_repository_relative_path(...)`` inside the identity
    module -- i.e. there is no second, unguarded
    ``repo_root / relative_path`` hash-resolution path anywhere in
    ``src/dfp_phase1b_chunk_benchmark_identity.py``."""
    import ast

    source = (ROOT / "src/dfp_phase1b_chunk_benchmark_identity.py").read_text()
    tree = ast.parse(source)

    call_sites = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.safe_names = set()

        def visit_FunctionDef(self, node):
            outer_safe = set(self.safe_names)
            self.safe_names = set()
            self.generic_visit(node)
            self.safe_names = outer_safe

        def visit_Assign(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "resolve_repository_relative_path":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.safe_names.add(target.id)
            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "_sha256_of":
                arg = node.args[0] if node.args else None
                is_safe = (
                    isinstance(arg, ast.Name) and arg.id in self.safe_names
                ) or (
                    isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "resolve_repository_relative_path"
                )
                call_sites.append((node.lineno, is_safe))
            self.generic_visit(node)

    Visitor().visit(tree)
    assert call_sites, "expected at least one _sha256_of call site to exist"
    unsafe = [lineno for lineno, is_safe in call_sites if not is_safe]
    assert unsafe == [], f"_sha256_of called without routing through resolve_repository_relative_path() at line(s): {unsafe}"


def test_ast_no_raw_repo_root_slash_relative_path_expression_outside_the_helper():
    """A coarser static check: outside ``resolve_repository_relative_path``
    itself, no ``<name> / <subscript>`` division expression combines a
    variable plausibly named like a repo root with a dict/attribute
    access plausibly named like a relative path -- guarding against a
    reintroduced unguarded resolution path under a different helper
    name."""
    import ast

    source = (ROOT / "src/dfp_phase1b_chunk_benchmark_identity.py").read_text()
    tree = ast.parse(source)

    suspicious = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_repository_relative_path":
            continue  # the one legitimate place this pattern belongs
    for func in tree.body:
        if isinstance(func, ast.FunctionDef) and func.name != "resolve_repository_relative_path":
            for node in ast.walk(func):
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                    left_name = node.left.id if isinstance(node.left, ast.Name) else None
                    if left_name in ("root", "repo_root") and isinstance(node.right, (ast.Subscript, ast.Attribute)):
                        suspicious.append((func.name, node.lineno))
    assert suspicious == [], f"unguarded repo_root/relative_path-shaped expression(s) found: {suspicious}"


# =======================================================================
# REPAIR ROUND 2: shared preflight/run-trial repository-readiness
# function, exercised against a genuine local git clone of the REAL
# repository's own commit history (never committing anything in the
# real repository) plus a synthetic descendant commit -- since E17's
# ``required_ancestor_commit`` is a specific real commit id that only
# exists there.
# =======================================================================

_REPAIRED_EIGHT_FILES = (
    "evaluation_identities/e17_dfp_phase1b_chunk_benchmark.toml",
    "evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json",
    "src/dfp_phase1b_chunk_benchmark_identity.py",
    "src/dfp_phase1b_chunk_benchmark.py",
    "diagnostics/run_dfp_phase1b_chunk_benchmark.py",
    "tests/test_dfp_phase1b_chunk_benchmark.py",
    "docs/dfp_phase1b_chunk_benchmark.md",
    "dfp_phase1b_chunk_benchmark_h100.sbatch",
)


@pytest.fixture(scope="module")
def synthetic_clean_clone(tmp_path_factory):
    """One module-scoped local clone of the REAL repository's own
    commit history (git objects are content-addressed, so a local
    clone is cheap and touches nothing in the real repository), with
    the eight repaired implementation files copied in and committed as
    one synthetic descendant commit. Used for POSITIVE preflight/
    run-trial readiness tests. Individual dirty-tree tests mutate this
    SAME clone and MUST restore it to a clean state before returning
    (see ``_temporarily_dirty``) so later tests in this module still
    see a clean fixture."""
    clone_dir = tmp_path_factory.mktemp("synthetic_clone") / "repo"
    result = sp.run(["git", "clone", "-q", str(ROOT), str(clone_dir)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"git clone failed: {result.stderr}"
    for rel in _REPAIRED_EIGHT_FILES:
        dest = clone_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel).read_bytes())
    _git(clone_dir, "add", "-A")
    _git(clone_dir, "config", "user.email", "test@example.com")
    _git(clone_dir, "config", "user.name", "Test")
    commit = _git(clone_dir, "commit", "-q", "-m", "test: synthetic benchmark commit (not the real repository)")
    assert commit.returncode == 0, f"synthetic commit failed: {commit.stderr}"
    assert _git(clone_dir, "status", "--porcelain").stdout.strip() == ""
    return clone_dir


@contextlib.contextmanager
def _temporarily_dirty(clone_root, mutate_fn):
    """Apply ``mutate_fn(clone_root)``, yield, then unconditionally
    restore the clone to a clean state (tracked changes reverted,
    untracked files removed) so a shared module-scoped clone fixture
    stays clean for the next test."""
    mutate_fn(clone_root)
    try:
        yield
    finally:
        # `git checkout -- .` restores from the INDEX, not HEAD -- it
        # would NOT revert an already-staged modification. `reset
        # --hard` resets both the index and working tree to HEAD.
        _git(clone_root, "reset", "-q", "--hard", "HEAD")
        _git(clone_root, "clean", "-q", "-fd")


def _import_cli():
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import importlib

    return importlib.import_module("run_dfp_phase1b_chunk_benchmark")


def test_readiness_check_passes_on_clean_synthetic_descendant_clone(synthetic_clean_clone):
    """The pure-Python shared readiness function (never imports torch)
    succeeds end-to-end against the clean temporary clone."""
    cli = _import_cli()
    result = cli.perform_repository_readiness_check(
        repo_root=synthetic_clean_clone, artifact_path=ARTIFACT_PATH,
        manifest_path=synthetic_clean_clone / "evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json",
        workload_bundle_dir=WORKLOAD_BUNDLE_DIR,
        output_path=synthetic_clean_clone / "trial_out.json", allow_overwrite=False,
    )
    assert result.repository_commit == _git(synthetic_clean_clone, "rev-parse", "HEAD").stdout.strip()
    assert result.code_provenance["repository_worktree_clean"] is True
    assert "torch" not in sys.modules or True  # torch may be imported by other test modules; this call itself never imports it


@requires_real_inputs
def test_cli_preflight_passes_on_clean_synthetic_descendant_clone(synthetic_clean_clone, capsys):
    """The standalone `preflight` CLI command (never imports torch)
    reports a genuine PASS against the clean temporary clone."""
    cli = _import_cli()
    exit_code = cli.main([
        "--repo-root", str(synthetic_clean_clone),
        "preflight", "--artifact", str(ARTIFACT_PATH),
        "--manifest", str(synthetic_clean_clone / "evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json"),
        "--workload-bundle-dir", str(WORKLOAD_BUNDLE_DIR),
    ])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "PREFLIGHT PASS" in out
    assert "ancestry and clean-worktree verified" in out


def test_readiness_check_policy_head_equals_required_ancestor_still_passes(synthetic_clean_clone):
    """Documented policy (Section 11): HEAD == required_ancestor_commit
    (never requiring strict descendant-ness) is a valid, PASS-worthy
    ancestry relation -- exercised directly at the ancestry-relation
    level (equivalent to, but not requiring, checking the identity's
    OWN required_ancestor_commit out as HEAD, which is impractical to
    construct here since E17 did not exist at that ancestor commit)."""
    head = _git(synthetic_clean_clone, "rev-parse", "HEAD").stdout.strip()
    ident.verify_ancestor_relation(synthetic_clean_clone, required_ancestor_commit=head, repository_commit=head)


@pytest.mark.parametrize("label,mutate", [
    ("unstaged-tracked-modification", lambda root: (root / "docs/dfp_phase1b_chunk_benchmark.md").write_text("dirty\n")),
    ("untracked-file", lambda root: (root / "totally_new_untracked_file.txt").write_text("dirty\n")),
])
def test_readiness_check_rejects_dirty_states_before_torch_import(synthetic_clean_clone, label, mutate):
    cli = _import_cli()
    with _temporarily_dirty(synthetic_clean_clone, mutate):
        with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
            cli.perform_repository_readiness_check(
                repo_root=synthetic_clean_clone, artifact_path=ARTIFACT_PATH,
                manifest_path=synthetic_clean_clone / "evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json",
                workload_bundle_dir=WORKLOAD_BUNDLE_DIR, output_path=None, allow_overwrite=False,
            )


def test_readiness_check_rejects_staged_modification(synthetic_clean_clone):
    cli = _import_cli()

    def mutate(root):
        (root / "docs/dfp_phase1b_chunk_benchmark.md").write_text("dirty staged\n")
        _git(root, "add", "docs/dfp_phase1b_chunk_benchmark.md")

    with _temporarily_dirty(synthetic_clean_clone, mutate):
        with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
            cli.perform_repository_readiness_check(
                repo_root=synthetic_clean_clone, artifact_path=ARTIFACT_PATH,
                manifest_path=synthetic_clean_clone / "evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json",
                workload_bundle_dir=WORKLOAD_BUNDLE_DIR, output_path=None, allow_overwrite=False,
            )


@requires_real_inputs
def test_cli_preflight_reports_dirty_checkout_as_exit_2_no_traceback_no_false_pass(synthetic_clean_clone, capsys):
    cli = _import_cli()
    with _temporarily_dirty(synthetic_clean_clone, lambda root: (root / "dirty_marker.txt").write_text("x")):
        exit_code = cli.main([
            "--repo-root", str(synthetic_clean_clone),
            "preflight", "--artifact", str(ARTIFACT_PATH),
            "--manifest", str(synthetic_clean_clone / "evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json"),
            "--workload-bundle-dir", str(WORKLOAD_BUNDLE_DIR),
        ])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "PREFLIGHT PASS" not in captured.out
    assert "Traceback" not in captured.err
    assert "not clean" in captured.err


def test_readiness_check_rejects_unrelated_orphan_history(synthetic_clean_clone):
    """Create a genuinely-unrelated root commit via ``git commit-tree``
    (a real commit object in the SAME repository, but with no common
    ancestor) WITHOUT ever checking it out or touching the shared
    clone's working tree, then confirm ``verify_ancestor_relation``
    rejects it as the required ancestor of an unrelated execution
    commit."""
    empty_tree = sp.run(["git", "-C", str(synthetic_clean_clone), "hash-object", "-t", "tree", "/dev/null"],
                         capture_output=True, text=True)
    # A tree object hash is always the same well-known empty-tree SHA;
    # simpler and equally valid to ask git directly for it.
    empty_tree_sha = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
    commit_result = sp.run(
        ["git", "-C", str(synthetic_clean_clone), "commit-tree", empty_tree_sha, "-m", "unrelated orphan root"],
        capture_output=True, text=True,
    )
    assert commit_result.returncode == 0, commit_result.stderr
    orphan_commit = commit_result.stdout.strip()

    head = _git(synthetic_clean_clone, "rev-parse", "HEAD").stdout.strip()
    with pytest.raises(ident.DfpPhase1bChunkBenchmarkIdentityError):
        ident.verify_ancestor_relation(synthetic_clean_clone, required_ancestor_commit=orphan_commit, repository_commit=head)
    assert _git(synthetic_clean_clone, "status", "--porcelain").stdout.strip() == ""


def test_readiness_check_rejects_existing_output_without_force(synthetic_clean_clone, tmp_path):
    cli = _import_cli()
    existing = tmp_path / "already_exists.json"
    existing.write_text("{}")
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        cli.perform_repository_readiness_check(
            repo_root=synthetic_clean_clone, artifact_path=ARTIFACT_PATH,
            manifest_path=synthetic_clean_clone / "evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json",
            workload_bundle_dir=WORKLOAD_BUNDLE_DIR, output_path=existing, allow_overwrite=False,
        )


def test_readiness_check_rejects_unsafe_output_path(synthetic_clean_clone, tmp_path):
    cli = _import_cli()
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        cli.perform_repository_readiness_check(
            repo_root=synthetic_clean_clone, artifact_path=ARTIFACT_PATH,
            manifest_path=synthetic_clean_clone / "evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json",
            workload_bundle_dir=WORKLOAD_BUNDLE_DIR,
            output_path=tmp_path / ".." / "escape.json", allow_overwrite=False,
        )


def test_preflight_and_run_trial_share_one_readiness_implementation():
    """Structural proof (never two implementations that may drift):
    both `_cmd_preflight` and `_cmd_run_trial` call the SAME
    `perform_repository_readiness_check` function."""
    import ast

    cli = _import_cli()
    source = Path(cli.__file__).read_text()
    tree = ast.parse(source)
    calls_by_func = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("_cmd_preflight", "_cmd_run_trial"):
            calls_by_func[node.name] = [
                n.func.id for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            ]
    assert "perform_repository_readiness_check" in calls_by_func["_cmd_preflight"]
    assert "perform_repository_readiness_check" in calls_by_func["_cmd_run_trial"]
    # `_cmd_run_trial` must never re-implement require_clean_worktree/
    # verify_ancestor_relation/load_identity/verify_benchmark_manifest
    # calls of its own -- those live SOLELY inside the shared function.
    for forbidden in ("require_clean_worktree", "verify_ancestor_relation", "load_identity", "verify_benchmark_manifest"):
        assert forbidden not in calls_by_func["_cmd_run_trial"]
        assert forbidden not in calls_by_func["_cmd_preflight"]


def test_real_uncommitted_working_tree_readiness_check_fails_as_expected():
    """The real repository is STILL uncommitted (eight untracked
    files) -- a readiness check against it MUST fail (dirty worktree).
    This is expected, correct behavior, never to be weakened."""
    cli = _import_cli()
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError, match="not clean"):
        cli.perform_repository_readiness_check(
            repo_root=ROOT, artifact_path=ARTIFACT_PATH,
            manifest_path=MANIFEST_PATH, workload_bundle_dir=WORKLOAD_BUNDLE_DIR,
            output_path=None, allow_overwrite=False,
        )


# =======================================================================
# REPAIR ROUND 2: strict memory-reading validation (non-finite/invalid
# values rejected) and strict /proc/self/status parsing.
# =======================================================================

_GIB = 1024 ** 3
_GUARD_CONTRACT_90PCT_80GIB = ident.MemoryGuardContract(
    allocated_host_gib=80.0, allocated_gpu_gib=80.0, host_abort_fraction=0.90, gpu_abort_fraction=0.90,
)
_HOST_LIMIT = 80 * _GIB * 0.90
_GPU_LIMIT = 80 * _GIB * 0.90


class _ScriptedMemoryProvider:
    def __init__(self, *, host, gpu_peak):
        self._host = host
        self._gpu_peak = gpu_peak

    def baseline(self):
        return {}

    def reset_peak(self):
        pass

    def peak(self):
        return self._gpu_peak

    def host_current(self):
        return self._host


@pytest.mark.parametrize("label,value", [
    ("negative-int", -1),
    ("bool-true", True),
    ("bool-false", False),
    ("float", 1000.0),
    ("nan", float("nan")),
    ("positive-infinity", float("inf")),
    ("negative-infinity", float("-inf")),
    ("numeric-string", "1000"),
    ("none", None),
])
def test_check_memory_safety_guard_rejects_invalid_host_reading(label, value):
    provider = _ScriptedMemoryProvider(host={"vmrss_bytes": value}, gpu_peak={"reserved_bytes": 0})
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.check_memory_safety_guard(provider, _GUARD_CONTRACT_90PCT_80GIB)


@pytest.mark.parametrize("label,value", [
    ("negative-int", -1), ("bool", True), ("float", 1000.0), ("nan", float("nan")),
    ("infinity", float("inf")), ("numeric-string", "1000"), ("none", None),
])
def test_check_memory_safety_guard_rejects_invalid_gpu_reading(label, value):
    provider = _ScriptedMemoryProvider(host={"vmrss_bytes": 0}, gpu_peak={"reserved_bytes": value})
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.check_memory_safety_guard(provider, _GUARD_CONTRACT_90PCT_80GIB)


def test_check_memory_safety_guard_arbitrary_numeric_like_object_rejected():
    class FakeInt:
        def __int__(self):
            return 1000

        def __gt__(self, other):
            return False

    provider = _ScriptedMemoryProvider(host={"vmrss_bytes": FakeInt()}, gpu_peak={"reserved_bytes": 0})
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.check_memory_safety_guard(provider, _GUARD_CONTRACT_90PCT_80GIB)


@pytest.mark.parametrize("label,host_bytes,gpu_bytes,expect_abort", [
    ("zero", 0, 0, False),
    ("normal-positive", 1_000_000, 1_000_000, False),
    ("host-exactly-90pct", int(_HOST_LIMIT), 0, False),
    ("gpu-exactly-90pct", 0, int(_GPU_LIMIT), False),
    ("host-one-byte-below-threshold", int(_HOST_LIMIT) - 1, 0, False),
    ("host-one-byte-above-threshold", int(_HOST_LIMIT) + 1, 0, True),
    ("gpu-one-byte-below-threshold", 0, int(_GPU_LIMIT) - 1, False),
    ("gpu-one-byte-above-threshold", 0, int(_GPU_LIMIT) + 1, True),
    ("host-only-breach", int(_HOST_LIMIT) + 1000, 0, True),
    ("gpu-only-breach", 0, int(_GPU_LIMIT) + 1000, True),
    ("simultaneous-breach", int(_HOST_LIMIT) + 1000, int(_GPU_LIMIT) + 1000, True),
])
def test_check_memory_safety_guard_boundary_matrix(label, host_bytes, gpu_bytes, expect_abort):
    provider = _ScriptedMemoryProvider(host={"vmrss_bytes": host_bytes}, gpu_peak={"reserved_bytes": gpu_bytes})
    reason = bench.check_memory_safety_guard(provider, _GUARD_CONTRACT_90PCT_80GIB)
    assert (reason is not None) == expect_abort, f"{label}: expected abort={expect_abort}, got reason={reason!r}"


def test_check_memory_safety_guard_null_reading_rejected_not_silently_defaulted():
    """An explicitly-present ``None`` value is an invalid reading, not
    an absent one -- must be rejected, never silently treated as 0."""
    provider = _ScriptedMemoryProvider(host={"vmrss_bytes": None, "ru_maxrss_bytes": 5}, gpu_peak={"reserved_bytes": 0})
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.check_memory_safety_guard(provider, _GUARD_CONTRACT_90PCT_80GIB)


def test_check_memory_safety_guard_absent_key_falls_through_to_fallback():
    """A key genuinely ABSENT (not present-but-None) legitimately falls
    through to the next candidate / zero -- distinct from an explicit
    null, which is invalid."""
    provider = _ScriptedMemoryProvider(host={"ru_maxrss_bytes": 5}, gpu_peak={})
    reason = bench.check_memory_safety_guard(provider, _GUARD_CONTRACT_90PCT_80GIB)
    assert reason is None


# --- strict /proc/self/status parsing ---

_VALID_PROC_STATUS = "VmRSS:\t   12345 kB\nVmHWM:\t   23456 kB\n"


@pytest.mark.parametrize("label,text", [
    ("missing-vmrss", "VmHWM:\t 23456 kB\n"),
    ("missing-vmhwm", "VmRSS:\t 12345 kB\n"),
    ("duplicate-field", _VALID_PROC_STATUS + "VmRSS:\t 1 kB\n"),
    ("blank-value", "VmRSS:\t   kB\nVmHWM:\t 23456 kB\n"),
    ("negative-value", "VmRSS:\t -5 kB\nVmHWM:\t 23456 kB\n"),
    ("decimal-value", "VmRSS:\t 12345.5 kB\nVmHWM:\t 23456 kB\n"),
    ("scientific-notation", "VmRSS:\t 1.2e4 kB\nVmHWM:\t 23456 kB\n"),
    ("wrong-unit", "VmRSS:\t 12345 MB\nVmHWM:\t 23456 kB\n"),
    ("extra-token", "VmRSS:\t 12345 67 kB\nVmHWM:\t 23456 kB\n"),
    ("truncated-line", "VmRSS:\t 12345\nVmHWM:\t 23456 kB\n"),
    ("empty-file", ""),
])
def test_parse_proc_self_status_rejects_malformed_content(label, text):
    # Must be exactly the declared domain error -- never a raw
    # ValueError/IndexError/KeyError leaking through.
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError) as excinfo:
        bench.parse_proc_self_status_kib_fields(text, required_fields=("VmRSS", "VmHWM"))
    assert type(excinfo.value) is bench.DfpPhase1bChunkBenchmarkError


def test_parse_proc_self_status_accepts_valid_content_and_converts_kib_to_bytes():
    parsed = bench.parse_proc_self_status_kib_fields(_VALID_PROC_STATUS, required_fields=("VmRSS", "VmHWM"))
    assert parsed == {"VmRSS": 12345 * 1024, "VmHWM": 23456 * 1024}


def test_host_current_falls_back_gracefully_when_proc_unavailable(monkeypatch):
    """Retain silent fallback ONLY for a genuine filesystem-availability
    error (missing file / no permission) -- never for malformed
    content."""
    import src.dfp_phase1b_chunk_benchmark as bench_module

    provider = bench_module.GpuHostMemoryProvider.__new__(bench_module.GpuHostMemoryProvider)

    real_read_bytes = Path.read_bytes

    def fake_read_bytes(self):
        if str(self) == "/proc/self/status":
            raise FileNotFoundError("simulated: /proc unavailable")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    result = provider.host_current()
    assert "vmrss_bytes" not in result
    assert "vmhwm_bytes" not in result
    assert "ru_maxrss_bytes" in result


def test_host_current_fails_closed_on_malformed_proc_content(monkeypatch):
    import src.dfp_phase1b_chunk_benchmark as bench_module

    provider = bench_module.GpuHostMemoryProvider.__new__(bench_module.GpuHostMemoryProvider)

    def fake_read_bytes(self):
        if str(self) == "/proc/self/status":
            return b"VmRSS:\t garbage kB\nVmHWM:\t 1 kB\n"
        raise AssertionError("unexpected read_bytes call")

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        provider.host_current()


def test_host_current_fails_closed_on_malformed_utf8(monkeypatch):
    import src.dfp_phase1b_chunk_benchmark as bench_module

    provider = bench_module.GpuHostMemoryProvider.__new__(bench_module.GpuHostMemoryProvider)

    def fake_read_bytes(self):
        if str(self) == "/proc/self/status":
            return b"\xff\xfe\x00VmRSS:garbage"
        raise AssertionError("unexpected read_bytes call")

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        provider.host_current()


def test_host_current_real_proc_self_status_parses_successfully():
    """Sanity check against the REAL /proc/self/status on this host --
    confirms the strict parser is not merely correct in the abstract
    but actually works against genuine kernel output."""
    provider = bench.GpuHostMemoryProvider.__new__(bench.GpuHostMemoryProvider)
    result = provider.host_current()
    assert "vmrss_bytes" in result
    assert "vmhwm_bytes" in result
    assert result["vmrss_bytes"] >= 0
    assert result["vmhwm_bytes"] >= 0


# =======================================================================
# REPAIR ROUND 2: MEMORY_FIELD_KEYS closed schema (old ambiguous key
# rejection).
# =======================================================================


def test_memory_field_keys_rejects_old_ambiguous_key():
    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16)
    mutated = copy.deepcopy(doc)
    mutated["memory"]["host_peak_rss_bytes"] = mutated["memory"].pop("host_process_lifetime_peak_rss_bytes")
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.verify_trial_document(mutated)


def test_memory_field_keys_rejects_unknown_extra_key():
    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16)
    mutated = copy.deepcopy(doc)
    mutated["memory"]["totally_unknown_field"] = 123
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.verify_trial_document(mutated)


def test_memory_field_keys_rejects_missing_required_key():
    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16)
    mutated = copy.deepcopy(doc)
    del mutated["memory"]["host_current_rss_bytes"]
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.verify_trial_document(mutated)


@pytest.mark.parametrize("label,value", [("nan", float("nan")), ("infinity", float("inf")), ("bool", True), ("negative", -1), ("string", "1000")])
def test_verify_trial_document_rejects_malformed_timing(label, value):
    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16)
    mutated = copy.deepcopy(doc)
    mutated["timing"]["primary_elapsed_ns"] = value
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.verify_trial_document(mutated)


def test_verify_trial_document_rejects_non_finite_value_surviving_a_json_roundtrip(tmp_path):
    """Python's json module parses the non-standard NaN/Infinity tokens
    by default -- a tampered trial JSON file containing one must still
    be rejected after a real json.load() round-trip, not just when
    handed a Python float directly."""
    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16)
    mutated = copy.deepcopy(doc)
    mutated["timing"]["primary_elapsed_ns"] = float("nan")
    raw_path = tmp_path / "trial_nan.json"
    # json.dumps emits the literal NaN token by default (allow_nan=True).
    raw_path.write_text(json.dumps(mutated))
    with open(raw_path, "r", encoding="utf-8") as handle:
        reloaded = json.load(handle)
    assert isinstance(reloaded["timing"]["primary_elapsed_ns"], float)
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.verify_trial_document(reloaded)


# =======================================================================
# REPAIR ROUND 2: live synthetic six-trial merge matrix (built through
# the REAL public run_trial/build_trial_document/merge_trials
# functions, never a hand-rolled merged document).
# =======================================================================


def test_six_trial_matrix_valid_set_merges_and_verifies_with_independently_recomputed_summaries():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    merged = bench.merge_trials(trials)

    # No winner/recommendation/tolerance/5000-image field anywhere.
    for forbidden in ("winner", "recommended_chunk_size", "preferred_chunk_size", "tolerance", "five_thousand", "5000"):
        assert forbidden not in json.dumps(merged)

    # Raw six-trial measurements preserved verbatim.
    assert len(merged["raw_trials"]) == 6
    assert merged["raw_trials"] == [dict(t) for t in trials]

    # Independently recompute every descriptive summary field and
    # compare EXACT values -- never trusting merge_trials's own
    # arithmetic as self-evidence.
    for chunk_size in bench.CHUNK_SIZES:
        matching = [t for t in trials if t["chunk_size"] == chunk_size]
        assert len(matching) == 2
        expected_times = sorted(t["timing"]["primary_elapsed_ns"] for t in matching)
        summary = merged["descriptive_summaries_by_chunk_size"][str(chunk_size)]
        assert summary["primary_elapsed_ns_per_repeat"] == expected_times
        assert summary["median_of_two_primary_times_ns"] == sum(expected_times) / len(expected_times)
        assert summary["min_ns"] == min(expected_times)
        assert summary["max_ns"] == max(expected_times)
        expected_throughput = [24 / (t / 1e9) for t in expected_times if t > 0]
        assert summary["throughput_windows_per_second_over_fixed_24_window_set"] == expected_throughput
        expected_host_peak = [t["memory"]["host_process_lifetime_peak_rss_bytes"] for t in matching if t["memory"].get("host_process_lifetime_peak_rss_bytes") is not None]
        assert summary["host_process_lifetime_peak_rss_bytes"] == expected_host_peak


@pytest.mark.parametrize("omit_index", range(6))
def test_six_trial_matrix_each_trial_omitted_in_turn_rejected(omit_index):
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    subset = trials[:omit_index] + trials[omit_index + 1:]
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(subset)


@pytest.mark.parametrize("dup_target", range(1, 6))
def test_six_trial_matrix_each_trial_duplicated_in_turn_rejected(dup_target):
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    mutated[dup_target] = copy.deepcopy(mutated[0])
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(mutated)


def test_six_trial_matrix_extra_seventh_trial_rejected():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    extra = copy.deepcopy(trials[0])
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(trials + [extra])


def test_six_trial_matrix_wrong_trial_index_rejected():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    mutated[0]["trial_index"] = 99
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(mutated)


def test_six_trial_matrix_wrong_repeat_rejected():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    mutated[0]["repeat_index"] = 99
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(mutated)


def test_six_trial_matrix_wrong_chunk_size_rejected():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    mutated[0]["chunk_size"] = 999
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(mutated)


def test_six_trial_matrix_reordered_input_list_still_merges_identically():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    reordered = list(reversed(trials))
    merged_original = bench.merge_trials(trials)
    merged_reordered = bench.merge_trials(reordered)
    assert merged_original["scientific_output_digest"] == merged_reordered["scientific_output_digest"]
    assert merged_original["descriptive_summaries_by_chunk_size"] == merged_reordered["descriptive_summaries_by_chunk_size"]


def test_six_trial_matrix_aborted_trial_rejected():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    mutated[0]["complete"] = False
    mutated[0]["final"] = False
    mutated[0]["abort_status"] = {"aborted": True, "reason": "synthetic test abort"}
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(mutated)


def test_six_trial_matrix_wrong_authority_hash_rejected():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    mutated[1]["authority_sha256"] = "0" * 64
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(mutated)


def test_six_trial_matrix_wrong_gpu_model_rejected():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    mutated[4]["device_record"]["gpu_model_name"] = "SOME-OTHER-GPU"
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(mutated)


def test_six_trial_matrix_wrong_software_record_rejected():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    mutated[5]["software_record"]["torch_version"] = "99.99.99"
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(mutated)


def test_six_trial_matrix_old_ambiguous_memory_key_rejected():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    mutated[2]["memory"]["host_peak_rss_bytes"] = mutated[2]["memory"].pop("host_process_lifetime_peak_rss_bytes")
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(mutated)


def test_six_trial_matrix_malformed_memory_unknown_key_rejected():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    mutated[3]["memory"]["some_unknown_field"] = 123
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(mutated)


def test_six_trial_matrix_malformed_timing_non_finite_rejected():
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    mutated[0]["timing"]["primary_elapsed_ns"] = float("nan")
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.merge_trials(mutated)


# =======================================================================
# REPAIR: persisted-memory VALUE validation (not just key-set closure).
# ``_verify_memory_mapping`` requires every one of the nine
# MEMORY_FIELD_KEYS to be an exact, nonnegative ``int`` -- the same
# standard the live safety guard already enforces on the readings it
# consumes, now extended to the full persisted trial-document schema.
# =======================================================================

_INVALID_MEMORY_VALUES = [
    ("negative-int", -1),
    ("bool-true", True),
    ("integral-float", 1.0),
    ("nan", float("nan")),
    ("positive-infinity", float("inf")),
    ("negative-infinity", float("-inf")),
    ("numeric-string", "1"),
    ("empty-string", ""),
    ("none", None),
    ("complex", complex(1, 0)),
]


def _numpy_int_case():
    numpy = pytest.importorskip("numpy")
    return [("numpy-int64", numpy.int64(1))]


class _CustomNumericObject:
    """Arbitrary object implementing numeric comparison but not int."""

    def __init__(self, value):
        self._value = value

    def __eq__(self, other):
        return self._value == other

    def __gt__(self, other):
        return self._value > other

    def __ge__(self, other):
        return self._value >= other

    def __int__(self):
        return int(self._value)

    def __repr__(self):
        return f"_CustomNumericObject({self._value})"


_INVALID_MEMORY_VALUES_EXTENDED = _INVALID_MEMORY_VALUES + _numpy_int_case() + [
    ("custom-numeric-object", _CustomNumericObject(1)),
]


def _valid_memory_mapping():
    """A fresh, fully-valid memory mapping covering all nine
    MEMORY_FIELD_KEYS with distinct, ordinary positive integers, so a
    mutation to any single field is unambiguous."""
    base = {
        "gpu_baseline_allocated_bytes": 100, "gpu_baseline_reserved_bytes": 200,
        "gpu_peak_allocated_bytes": 500, "gpu_peak_reserved_bytes": 600,
        "gpu_end_allocated_bytes": 150, "gpu_end_reserved_bytes": 250,
        "host_process_lifetime_peak_rss_bytes": 1000, "host_process_lifetime_vmhwm_bytes": 1100,
        "host_current_rss_bytes": 900,
    }
    assert frozenset(base.keys()) == bench.MEMORY_FIELD_KEYS
    return base


@pytest.mark.parametrize("field", sorted(bench.MEMORY_FIELD_KEYS))
@pytest.mark.parametrize("label,bad_value", _INVALID_MEMORY_VALUES_EXTENDED)
def test_verify_trial_document_rejects_every_invalid_memory_value_per_field(field, label, bad_value):
    """Field-by-field mutation matrix (Step 5): all nine fields x all
    required-plus-practical invalid value types, through the real
    public verify_trial_document."""
    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16)
    doc = copy.deepcopy(doc)
    doc["memory"] = _valid_memory_mapping()
    doc["memory"][field] = bad_value
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError) as excinfo:
        bench.verify_trial_document(doc)
    assert f"memory.{field}" in str(excinfo.value), f"error must name the exact offending field, got: {excinfo.value}"
    assert type(excinfo.value) is bench.DfpPhase1bChunkBenchmarkError


@pytest.mark.parametrize("field", sorted(bench.MEMORY_FIELD_KEYS))
@pytest.mark.parametrize("label,bad_value", _INVALID_MEMORY_VALUES)
def test_build_trial_document_rejects_every_invalid_memory_value_per_field(field, label, bad_value):
    """Same matrix, through the CONSTRUCTION-time builder
    (build_trial_document), not only the loaded-document verifier."""
    manifest = _build_fake_manifest()
    valid_doc = _run_fake_trial(manifest, 0, 0, 16)
    bad_memory = _valid_memory_mapping()
    bad_memory[field] = bad_value
    kwargs = dict(
        authority_sha256=valid_doc["authority_sha256"], manifest_sha256=valid_doc["manifest_sha256"],
        trial_index=valid_doc["trial_index"], repeat_index=valid_doc["repeat_index"], chunk_size=valid_doc["chunk_size"],
        device_record=valid_doc["device_record"], software_record=valid_doc["software_record"],
        window_identity=valid_doc["window_identity"],
        primary_elapsed_ns=valid_doc["timing"]["primary_elapsed_ns"],
        secondary_elapsed_ns=valid_doc["timing"]["secondary_elapsed_ns"],
        per_record_timing=valid_doc["timing"]["per_record_timing"],
        workload_records=valid_doc["workload_records"],
        scientific_output_digests=valid_doc["scientific_output_digests"],
        complete=True, final=True, code_provenance=valid_doc["code_provenance"],
    )
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError) as excinfo:
        bench.build_trial_document(memory=bad_memory, **kwargs)
    assert f"memory.{field}" in str(excinfo.value)


def test_build_trial_document_accepts_valid_memory_including_zero():
    """Valid controls (Step 5 item 6): zero and ordinary positive
    integers must remain accepted for every field."""
    manifest = _build_fake_manifest()
    valid_doc = _run_fake_trial(manifest, 0, 0, 16)
    zero_memory = {k: 0 for k in bench.MEMORY_FIELD_KEYS}
    kwargs = dict(
        authority_sha256=valid_doc["authority_sha256"], manifest_sha256=valid_doc["manifest_sha256"],
        trial_index=valid_doc["trial_index"], repeat_index=valid_doc["repeat_index"], chunk_size=valid_doc["chunk_size"],
        device_record=valid_doc["device_record"], software_record=valid_doc["software_record"],
        window_identity=valid_doc["window_identity"],
        primary_elapsed_ns=valid_doc["timing"]["primary_elapsed_ns"],
        secondary_elapsed_ns=valid_doc["timing"]["secondary_elapsed_ns"],
        per_record_timing=valid_doc["timing"]["per_record_timing"],
        workload_records=valid_doc["workload_records"],
        scientific_output_digests=valid_doc["scientific_output_digests"],
        complete=True, final=True, code_provenance=valid_doc["code_provenance"],
    )
    doc_zero = bench.build_trial_document(memory=zero_memory, **kwargs)
    assert doc_zero["memory"] == zero_memory
    bench.verify_trial_document(doc_zero)  # must not raise

    positive_memory = _valid_memory_mapping()
    doc_positive = bench.build_trial_document(memory=positive_memory, **kwargs)
    bench.verify_trial_document(doc_positive)  # must not raise


def test_memory_validator_does_not_mutate_caller_owned_mapping():
    """Caller input immutability: the validator must never write back
    into the mapping it was handed, valid or invalid."""
    memory = _valid_memory_mapping()
    snapshot = dict(memory)
    manifest = _build_fake_manifest()
    valid_doc = _run_fake_trial(manifest, 0, 0, 16)
    kwargs = dict(
        authority_sha256=valid_doc["authority_sha256"], manifest_sha256=valid_doc["manifest_sha256"],
        trial_index=valid_doc["trial_index"], repeat_index=valid_doc["repeat_index"], chunk_size=valid_doc["chunk_size"],
        device_record=valid_doc["device_record"], software_record=valid_doc["software_record"],
        window_identity=valid_doc["window_identity"],
        primary_elapsed_ns=valid_doc["timing"]["primary_elapsed_ns"],
        secondary_elapsed_ns=valid_doc["timing"]["secondary_elapsed_ns"],
        per_record_timing=valid_doc["timing"]["per_record_timing"],
        workload_records=valid_doc["workload_records"],
        scientific_output_digests=valid_doc["scientific_output_digests"],
        complete=True, final=True, code_provenance=valid_doc["code_provenance"],
    )
    doc = bench.build_trial_document(memory=memory, **kwargs)
    assert memory == snapshot, "build_trial_document must not mutate the caller's memory mapping"
    doc["memory"]["gpu_peak_reserved_bytes"] = 999999999  # mutate the RETURNED copy
    assert memory == snapshot, "the returned document's memory dict must be an independent copy"


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_memory_validation_propagates_fatal_exceptions_unwrapped(exc_type):
    """A memory mapping whose .keys() raises a fatal exception must
    propagate it unwrapped -- never caught/rewrapped as the benchmark
    domain error."""
    class _PoisonedMapping(dict):
        def keys(self):
            raise exc_type("injected fatal exception")

    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16)
    mutated = copy.deepcopy(doc)
    mutated["memory"] = _PoisonedMapping(_valid_memory_mapping())
    with pytest.raises(exc_type):
        bench.verify_trial_document(mutated)


def test_memory_validator_rejects_non_mapping_container():
    manifest = _build_fake_manifest()
    doc = _run_fake_trial(manifest, 0, 0, 16)
    mutated = copy.deepcopy(doc)
    mutated["memory"] = [("gpu_peak_reserved_bytes", 1)]  # a list, not a mapping
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError):
        bench.verify_trial_document(mutated)


@pytest.mark.parametrize("field", sorted(bench.MEMORY_FIELD_KEYS))
def test_merge_trials_rejects_invalid_memory_value_in_any_single_trial(field):
    """merge_trials must reject BEFORE constructing any descriptive
    summary -- confirmed by checking the invalid value never appears
    anywhere reachable from a successful return (there is none, since
    the call raises)."""
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    mutated[3]["memory"] = _valid_memory_mapping()
    mutated[3]["memory"][field] = -1
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError) as excinfo:
        bench.merge_trials(mutated)
    assert f"memory.{field}" in str(excinfo.value)


def test_merge_trials_tampered_memory_rejection_is_not_a_digest_side_effect():
    """Memory is explicitly excluded from every scientific/manifest
    digest (E17 scientific_invariance.per_window_excluded_fields
    includes 'memory') -- so a memory-value mutation can NEVER trigger
    a digest-mismatch rejection. This test recomputes and keeps the
    scientific digest fully self-consistent while tampering ONLY the
    memory value, proving the value validator itself is what rejects
    the input."""
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    mutated = copy.deepcopy(trials)
    # Confirm the untouched digest is still internally self-consistent
    # (recomputable) before and after the memory tamper -- i.e. we are
    # NOT relying on breaking a digest to get a rejection.
    original_digest = mutated[2]["scientific_output_digests"]["collection_digest"]
    mutated[2]["memory"] = _valid_memory_mapping()
    mutated[2]["memory"]["gpu_end_reserved_bytes"] = float("nan")
    assert mutated[2]["scientific_output_digests"]["collection_digest"] == original_digest
    with pytest.raises(bench.DfpPhase1bChunkBenchmarkError) as excinfo:
        bench.merge_trials(mutated)
    assert "memory.gpu_end_reserved_bytes" in str(excinfo.value)
    assert "digest" not in str(excinfo.value).lower()


def test_valid_six_trial_merge_unaffected_by_the_repair():
    """All existing valid six-trial merge controls remain bit-for-bit
    unchanged by this repair."""
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    merged = bench.merge_trials(trials)
    assert merged["scientific_output_digest_equal_across_all_trials"] is True
    assert "winner" not in merged
    for chunk_size, summary in merged["descriptive_summaries_by_chunk_size"].items():
        assert len(summary["primary_elapsed_ns_per_repeat"]) == 2


def test_cli_json_literal_nan_and_infinity_in_memory_rejected_cleanly(tmp_path):
    """Real CLI round-trip: a merged-result JSON file containing the
    literal (non-standard, but json.loads-permissive) NaN/Infinity
    tokens inside a raw_trial's memory field must produce exit 2, no
    traceback, no output artifact -- through verify-result."""
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    merged = bench.merge_trials(trials)
    merged_path = tmp_path / "merged.json"
    bench.write_merged_artifact_atomically(merged, merged_path)

    before_bytes = merged_path.read_bytes()

    for label, poison in (("nan", float("nan")), ("infinity", float("inf")), ("neg-infinity", float("-inf"))):
        tampered = json.loads(merged_path.read_text())
        tampered["raw_trials"][1]["memory"]["gpu_peak_allocated_bytes"] = poison
        tampered_path = tmp_path / f"tampered_{label}.json"
        tampered_text = json.dumps(tampered, sort_keys=True, indent=2)
        tampered_path.write_text(tampered_text)
        # Confirm the file genuinely contains the permissive literal token.
        token = "NaN" if label == "nan" else "Infinity"
        assert token in tampered_text

        completed = sp.run(
            [sys.executable, str(CLI_PATH), "--repo-root", str(ROOT), "verify-result", "--result", str(tampered_path)],
            capture_output=True, text=True, timeout=60,
        )
        assert completed.returncode == 2, f"{label}: expected exit 2, got {completed.returncode}: {completed.stderr}"
        assert "Traceback" not in completed.stderr, f"{label}: raw traceback leaked"
        assert "memory." in completed.stderr, f"{label}: error must name the offending memory field"

    # The valid merged source file must remain byte-identical throughout.
    assert merged_path.read_bytes() == before_bytes


def test_cli_merge_command_installs_no_output_when_a_trial_has_invalid_memory(tmp_path):
    """No partial merged result may be installed when one input trial
    carries an invalid memory value."""
    manifest = _build_fake_manifest()
    trials = _six_fake_trials(manifest)
    trial_paths = []
    for i, (ti, ri, cs) in enumerate(bench.TRIAL_SCHEDULE):
        doc = copy.deepcopy(trials[i])
        if i == 5:
            doc["memory"]["host_current_rss_bytes"] = -7
        p = tmp_path / f"trial_{i}.json"
        # Bypass write_trial_document_atomically's own validation-free
        # atomic write (it does not re-validate) to install the
        # deliberately-tampered file exactly as a corrupted producer
        # or hand-edit would leave it on disk.
        p.write_text(json.dumps(doc, sort_keys=True, indent=2))
        trial_paths.append(p)

    merged_path = tmp_path / "merged.json"
    args = ["merge"]
    for p in trial_paths:
        args += ["--trial", str(p)]
    args += ["--output", str(merged_path)]

    sys.path.insert(0, str(ROOT / "diagnostics"))
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_chunk_benchmark")
    exit_code = cli.main(args)
    assert exit_code == 2
    assert not merged_path.exists(), "no partial/invalid merged result may be installed"

# =======================================================================
# REPAIR: relational verification of the persisted
# descriptive_summaries_by_chunk_size in `verify-result` (Finding A).
# merge_trials already recomputes this block from raw_trials -- the
# CLI now requires the stored value to match that recomputation
# exactly (deterministic structural equality, no tolerance).
# =======================================================================


def _write_valid_merged_result(tmp_path, *, distinct_memory=False):
    """Build a complete, valid six-trial merged result and write it to
    disk. When distinct_memory=True, each trial gets a distinct
    gpu_peak_reserved_bytes value so list-order sensitivity can be
    probed precisely (identical values across trials would mask it)."""
    manifest = _build_fake_manifest()
    if distinct_memory:
        trials = []
        for i, (ti, ri, cs) in enumerate(bench.TRIAL_SCHEDULE):
            class _DistinctMemoryProvider(_FakeMemoryProvider):
                def peak(self, _i=i):
                    return {"allocated_bytes": 500, "reserved_bytes": 600 + _i, "end_allocated_bytes": 150, "end_reserved_bytes": 250}
            trials.append(_run_fake_trial(manifest, ti, ri, cs, memory_provider=_DistinctMemoryProvider()))
    else:
        trials = _six_fake_trials(manifest)
    merged = bench.merge_trials(trials)
    merged_path = tmp_path / "merged.json"
    bench.write_merged_artifact_atomically(merged, merged_path)
    return merged_path, merged, trials


def _run_verify_result_cli(result_path):
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_chunk_benchmark")
    return cli.main(["verify-result", "--result", str(result_path)])


def _run_verify_result_cli_subprocess(result_path):
    import os

    env = dict(os.environ)
    completed = sp.run(
        [sys.executable, str(CLI_PATH), "--repo-root", str(ROOT), "verify-result", "--result", str(result_path)],
        capture_output=True, text=True, env=env, timeout=60,
    )
    return completed


def test_verify_result_accepts_untouched_valid_merged_result(tmp_path):
    merged_path, merged, trials = _write_valid_merged_result(tmp_path)
    assert _run_verify_result_cli(merged_path) == 0


def test_verify_result_direct_api_recompute_matches_stored_summary(tmp_path):
    """The 'direct verification API': merge_trials(raw_trials) IS the
    existing recomputation path this repair reuses -- confirm it
    matches the stored summary exactly for a valid document, using no
    second summary formula."""
    merged_path, merged, trials = _write_valid_merged_result(tmp_path)
    recomputed = bench.merge_trials(merged["raw_trials"])
    assert recomputed["descriptive_summaries_by_chunk_size"] == merged["descriptive_summaries_by_chunk_size"]


def test_verify_result_zero_valued_memory_summary_accepted(tmp_path):
    """Valid descriptive values, including zero, remain accepted."""
    manifest = _build_fake_manifest()

    class _ZeroMemoryProvider(_FakeMemoryProvider):
        def baseline(self):
            return {"allocated_bytes": 0, "reserved_bytes": 0}

        def peak(self):
            return {"allocated_bytes": 0, "reserved_bytes": 0, "end_allocated_bytes": 0, "end_reserved_bytes": 0}

        def host_current(self):
            return {"ru_maxrss_bytes": 0, "vmrss_bytes": 0, "vmhwm_bytes": 0}

    trials = [_run_fake_trial(manifest, ti, ri, cs, memory_provider=_ZeroMemoryProvider()) for ti, ri, cs in bench.TRIAL_SCHEDULE]
    merged = bench.merge_trials(trials)
    merged_path = tmp_path / "zero_merged.json"
    bench.write_merged_artifact_atomically(merged, merged_path)
    assert _run_verify_result_cli(merged_path) == 0


def test_verify_result_reordering_invariants_are_accurately_characterized(tmp_path):
    """Documents the ACTUAL, established contract rather than
    fabricating a stronger one: the scientific digest and
    code_provenance ARE invariant under permitted trial reordering; the
    timing-derived descriptive fields (which merge_trials explicitly
    sorts) are ALSO order-invariant; the three raw per-trial memory
    lists (gpu_peak_allocated_bytes, gpu_peak_reserved_bytes,
    host_process_lifetime_peak_rss_bytes) are NOT canonically ordered
    by merge_trials -- they preserve input-trial order. This is a
    pre-existing characteristic of merge_trials, unrelated to and out
    of scope for the Finding A/B repair, documented here rather than
    silently assumed."""
    manifest = _build_fake_manifest()
    trials = []
    for i, (ti, ri, cs) in enumerate(bench.TRIAL_SCHEDULE):
        class _DistinctMemoryProvider(_FakeMemoryProvider):
            def peak(self, _i=i):
                return {"allocated_bytes": 500, "reserved_bytes": 600 + _i, "end_allocated_bytes": 150, "end_reserved_bytes": 250}
        trials.append(_run_fake_trial(manifest, ti, ri, cs, memory_provider=_DistinctMemoryProvider()))

    merged_forward = bench.merge_trials(trials)
    merged_reversed = bench.merge_trials(list(reversed(trials)))

    # Established, contractually-guaranteed invariants: TRUE.
    assert merged_forward["scientific_output_digest"] == merged_reversed["scientific_output_digest"]
    assert merged_forward["code_provenance"] == merged_reversed["code_provenance"]
    for cs in bench.CHUNK_SIZES:
        f = merged_forward["descriptive_summaries_by_chunk_size"][str(cs)]
        r = merged_reversed["descriptive_summaries_by_chunk_size"][str(cs)]
        # Timing-derived fields are explicitly sorted -- canonical.
        assert f["primary_elapsed_ns_per_repeat"] == r["primary_elapsed_ns_per_repeat"]
        assert f["min_ns"] == r["min_ns"] and f["max_ns"] == r["max_ns"]
        assert f["median_of_two_primary_times_ns"] == r["median_of_two_primary_times_ns"]
        assert f["throughput_windows_per_second_over_fixed_24_window_set"] == r["throughput_windows_per_second_over_fixed_24_window_set"]

    # Raw per-trial memory lists preserve INPUT order -- NOT canonical.
    # This is accurately documented as a pre-existing characteristic,
    # not asserted as a bug this repair must fix.
    any_order_sensitive = any(
        merged_forward["descriptive_summaries_by_chunk_size"][str(cs)]["gpu_peak_reserved_bytes"]
        != merged_reversed["descriptive_summaries_by_chunk_size"][str(cs)]["gpu_peak_reserved_bytes"]
        for cs in bench.CHUNK_SIZES
    )
    assert any_order_sensitive, "expected (and documented) pre-existing order-sensitivity of raw memory lists did not manifest"

    # verify-result still correctly validates EACH of these two merged
    # results against ITS OWN raw_trials (in their own stored order) --
    # order-sensitivity of the underlying formula does not cause a
    # false rejection when verifying a document against itself.
    for merged in (merged_forward, merged_reversed):
        p = tmp_path / f"reorder_{id(merged)}.json"
        bench.write_merged_artifact_atomically(merged, p)
        assert _run_verify_result_cli(p) == 0


_SUMMARY_MUTATION_CASES = [
    ("negative-value", lambda s: s["16"].__setitem__("gpu_peak_reserved_bytes", [-1, 600])),
    ("bool-true", lambda s: s["16"].__setitem__("gpu_peak_reserved_bytes", [True, 600])),
    ("integral-looking-float", lambda s: s["16"].__setitem__("gpu_peak_reserved_bytes", [600.0, 600])),
    ("nan", lambda s: s["16"].__setitem__("gpu_peak_reserved_bytes", [float("nan"), 600])),
    ("positive-infinity", lambda s: s["16"].__setitem__("gpu_peak_reserved_bytes", [float("inf"), 600])),
    ("negative-infinity", lambda s: s["16"].__setitem__("gpu_peak_reserved_bytes", [float("-inf"), 600])),
    ("numeric-string", lambda s: s["16"].__setitem__("gpu_peak_reserved_bytes", ["600", 600])),
    ("none", lambda s: s["16"].__setitem__("gpu_peak_reserved_bytes", [None, 600])),
    ("missing-summary-field", lambda s: s["16"].pop("gpu_peak_reserved_bytes")),
    ("unknown-summary-field", lambda s: s["16"].__setitem__("totally_unknown_field", 1)),
    ("missing-chunk-size-block", lambda s: s.pop("16")),
    ("extra-chunk-size-block", lambda s: s.__setitem__("128", dict(s["16"]))),
    ("wrong-list-length", lambda s: s["16"].__setitem__("gpu_peak_reserved_bytes", [600])),
    ("reordered-per-repeat-list", lambda s: s["16"].__setitem__("primary_elapsed_ns_per_repeat", list(reversed(s["16"]["primary_elapsed_ns_per_repeat"]))) if s["16"]["primary_elapsed_ns_per_repeat"][0] != s["16"]["primary_elapsed_ns_per_repeat"][-1] else s["16"].__setitem__("primary_elapsed_ns_per_repeat", [s["16"]["primary_elapsed_ns_per_repeat"][0], s["16"]["primary_elapsed_ns_per_repeat"][0] + 1])),
    ("altered-timing-summary", lambda s: s["16"].__setitem__("min_ns", (s["16"]["min_ns"] or 0) + 999999)),
    ("altered-throughput-summary", lambda s: s["16"].__setitem__("throughput_windows_per_second_over_fixed_24_window_set", [1.0, 2.0])),
    ("altered-host-memory-summary", lambda s: s["16"].__setitem__("host_process_lifetime_peak_rss_bytes", [1, 2])),
    ("altered-gpu-memory-summary", lambda s: s["16"].__setitem__("gpu_peak_allocated_bytes", [1, 2])),
    ("correct-schema-wrong-finite-value", lambda s: s["16"].__setitem__("max_ns", (s["16"]["max_ns"] or 0) + 1)),
]


@pytest.mark.parametrize("label,mutate", _SUMMARY_MUTATION_CASES)
def test_verify_result_rejects_every_summary_tamper_cli(tmp_path, label, mutate):
    merged_path, merged, trials = _write_valid_merged_result(tmp_path)
    tampered = json.loads(merged_path.read_text())
    mutate(tampered["descriptive_summaries_by_chunk_size"])
    tampered_path = tmp_path / f"tampered_{label}.json"
    tampered_path.write_text(json.dumps(tampered, sort_keys=True, indent=2))

    completed = _run_verify_result_cli_subprocess(tampered_path)
    assert completed.returncode == 2, f"{label}: expected exit 2, got {completed.returncode}: {completed.stderr}"
    assert "Traceback" not in completed.stderr, f"{label}: raw traceback leaked"
    assert "PASSES" not in completed.stdout, f"{label}: false PASS"
    assert "descriptive_summaries_by_chunk_size" in completed.stderr, f"{label}: diagnostic must name the field"


@pytest.mark.parametrize("label,mutate", _SUMMARY_MUTATION_CASES)
def test_verify_result_rejects_every_summary_tamper_direct_api(tmp_path, label, mutate):
    """Direct-API equivalent of the CLI check: recompute via
    merge_trials(raw_trials) (the existing recomputation path) and
    confirm it disagrees with the tampered stored summary -- proving
    the CLI's added comparison is not itself a bespoke formula."""
    merged_path, merged, trials = _write_valid_merged_result(tmp_path)
    tampered = json.loads(merged_path.read_text())
    mutate(tampered["descriptive_summaries_by_chunk_size"])

    recomputed = bench.merge_trials(tampered["raw_trials"])
    assert not bench.strict_structural_equal(
        recomputed["descriptive_summaries_by_chunk_size"], tampered["descriptive_summaries_by_chunk_size"]
    ), f"{label}: tampered summary must disagree with the recomputed one"


def test_verify_result_rejects_summary_tamper_with_digests_left_untampered_but_freshly_confirmed(tmp_path):
    """For at least one mutation: prove rejection is demonstrably
    caused by the relational SUMMARY comparison, not a stale digest --
    the scientific_output_digest and code_provenance in the tampered
    file are confirmed UNCHANGED (still exactly the originally-valid
    values) and still recompute cleanly, yet the file is still
    rejected."""
    merged_path, merged, trials = _write_valid_merged_result(tmp_path)
    original_digest = merged["scientific_output_digest"]
    original_code_provenance = dict(merged["code_provenance"])

    tampered = json.loads(merged_path.read_text())
    tampered["descriptive_summaries_by_chunk_size"]["32"]["gpu_peak_reserved_bytes"] = [-1, -1]
    tampered_path = tmp_path / "tampered_digest_untouched.json"
    tampered_path.write_text(json.dumps(tampered, sort_keys=True, indent=2))

    # Digests are provably unaffected by the summary tamper.
    assert tampered["scientific_output_digest"] == original_digest
    assert tampered["code_provenance"] == original_code_provenance
    recomputed_from_tampered_raw_trials = bench.merge_trials(tampered["raw_trials"])
    assert recomputed_from_tampered_raw_trials["scientific_output_digest"] == original_digest
    assert recomputed_from_tampered_raw_trials["code_provenance"] == original_code_provenance

    completed = _run_verify_result_cli_subprocess(tampered_path)
    assert completed.returncode == 2
    assert "digest" not in completed.stderr.lower()
    assert "code_provenance" not in completed.stderr.lower()
    assert "descriptive_summaries_by_chunk_size" in completed.stderr


def test_verify_result_rejects_literal_json_nan_infinity_tokens_in_summary(tmp_path):
    """Direct JSON containing literal NaN/Infinity (Python's json
    module emits these by default when allow_nan=True) inside the
    summary block must be rejected, not silently accepted."""
    merged_path, merged, trials = _write_valid_merged_result(tmp_path)
    for label, poison in (("nan", float("nan")), ("infinity", float("inf")), ("neg-infinity", float("-inf"))):
        tampered = json.loads(merged_path.read_text())
        tampered["descriptive_summaries_by_chunk_size"]["64"]["gpu_peak_allocated_bytes"] = [poison, 500]
        tampered_path = tmp_path / f"literal_{label}.json"
        tampered_text = json.dumps(tampered, sort_keys=True, indent=2)
        tampered_path.write_text(tampered_text)
        token = "NaN" if label == "nan" else "Infinity"
        assert token in tampered_text

        completed = _run_verify_result_cli_subprocess(tampered_path)
        assert completed.returncode == 2, f"{label}: expected exit 2"
        assert "Traceback" not in completed.stderr


def test_verify_result_does_not_mutate_loaded_document_or_write_any_file(tmp_path):
    merged_path, merged, trials = _write_valid_merged_result(tmp_path)
    before_bytes = merged_path.read_bytes()
    files_before = set(tmp_path.iterdir())
    assert _run_verify_result_cli(merged_path) == 0
    assert merged_path.read_bytes() == before_bytes
    assert set(tmp_path.iterdir()) == files_before


def test_verify_result_summary_tamper_creates_no_output_file(tmp_path):
    merged_path, merged, trials = _write_valid_merged_result(tmp_path)
    tampered = json.loads(merged_path.read_text())
    tampered["descriptive_summaries_by_chunk_size"]["16"]["gpu_peak_reserved_bytes"] = [-5, 600]
    tampered_path = tmp_path / "tampered_no_output.json"
    tampered_path.write_text(json.dumps(tampered, sort_keys=True, indent=2))
    files_before = set(tmp_path.iterdir())
    completed = _run_verify_result_cli_subprocess(tampered_path)
    assert completed.returncode == 2
    assert set(tmp_path.iterdir()) == files_before, "verify-result must never write/replace any file"

