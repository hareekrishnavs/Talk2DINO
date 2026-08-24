"""End-to-end CPU-only tests for the real reachability-gate CLI
(``evaluate_native_edge_support_reachability_gate.py``). Never initializes
CUDA, never imports torch/mmcv/mmseg -- this whole stage is offline
analysis over an already-produced JSON file."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import evaluate_native_edge_support_reachability_gate as cli  # noqa: E402
from src.native_edge_support_identity import load_identity as load_native_audit_identity  # noqa: E402
from src.native_edge_support_report import classify_reachability, verify_record as verify_native_audit_record  # noqa: E402

GATE_IDENTITY_PATH = ROOT / "evaluation_identities/e12_native_edge_support_reachability_gate.toml"
pytestmark = pytest.mark.skipif(not GATE_IDENTITY_PATH.exists(), reason="requires the reachability-gate identity")

_SUPPORT_COUNT_KEYS = tuple(str(n) for n in range(13))
_SUPPORT_FRACTION_KEYS = (
    "0.0", "(0.0,0.1]", "(0.1,0.2]", "(0.2,0.3]", "(0.3,0.4]", "(0.4,0.5]",
    "(0.5,0.6]", "(0.6,0.7]", "(0.7,0.8]", "(0.8,0.9]", "(0.9,1.0]",
)
_BAND_KEYS = ("0", "1", "2", "3-4", "5-7", ">=8")
_RANK_KEYS = tuple(str(r) for r in range(12))


def _make_synthetic_record(identity, *, windows=2, edges_with_observer, misclassified_rows, misclassified_rows_any_defined, correct_rows_any_defined, ignored_gt_count=0, single_window_edges=0):
    graph_rows = windows * 1024
    directed_edges = graph_rows * 12
    undefined_edges = directed_edges - edges_with_observer
    remaining_undefined = undefined_edges - single_window_edges
    rows_any_defined = misclassified_rows_any_defined + correct_rows_any_defined

    counter0 = {k: 0 for k in _SUPPORT_COUNT_KEYS}
    counter0["0"] = edges_with_observer
    fraction0 = {k: 0 for k in _SUPPORT_FRACTION_KEYS}
    fraction0["0.0"] = edges_with_observer
    band_graph_rows = {k: 0 for k in _BAND_KEYS}
    band_graph_rows[">=8"] = graph_rows
    band_directed_edges = {k: 0 for k in _BAND_KEYS}
    band_directed_edges[">=8"] = directed_edges
    rank_hist = {k: 0 for k in _RANK_KEYS}
    rank_hist["0"] = directed_edges

    correct_rows_total = graph_rows - ignored_gt_count - misclassified_rows
    cross_tab = {
        "defined_and_correct": correct_rows_any_defined, "defined_and_misclassified": misclassified_rows_any_defined,
        "undefined_and_correct": correct_rows_total - correct_rows_any_defined,
        "undefined_and_misclassified": misclassified_rows - misclassified_rows_any_defined,
    }

    funnel = {
        "images": 20, "windows": windows, "graph_rows": graph_rows, "directed_edges": directed_edges,
        "rows_with_other_crop": graph_rows if windows > 1 else 0,
        "rows_with_aligned_observer": rows_any_defined,
        "edges_with_observer": edges_with_observer,
        "rows_any_defined": rows_any_defined, "rows_all_defined": 0,
        "rows_any_defined_zero_support": 0,
        "misclassified_rows": misclassified_rows, "misclassified_rows_any_defined": misclassified_rows_any_defined,
        "misclassified_rows_all_defined": 0,
        "aligned_window_pairs": 1 if windows > 1 else 0,
        "unaligned_window_pairs": max(windows * (windows - 1) // 2 - (1 if windows > 1 else 0), 0),
        "clamped_windows": 0, "non_clamped_windows": windows,
        "unanimous_support_edges": 0, "direction_reversal_only_cases": 0,
        "rows_with_unique_least_support": rows_any_defined, "rows_tied_for_least_support": 0,
        "correct_rows_any_defined": correct_rows_any_defined,
    }
    identity_sha256 = hashlib.sha256((ROOT / "evaluation_identities/e12_native_edge_support_audit.toml").read_bytes()).hexdigest()
    record = {
        "schema": identity["run_modes"]["mechanics20_schema_name"], "run_mode": "mechanics20",
        "identity": identity["identity"]["name"], "identity_sha256": identity_sha256,
        "stitching_control_identity_sha256": identity["parent_identity"]["stitching_control_identity_sha256"],
        "power_evaluation_identity_sha256": identity["parent_identity"]["power_evaluation_identity_sha256"],
        "matched_identity_sha256": identity["parent_identity"]["matched_identity_sha256"],
        "e3_identity_sha256": identity["parent_identity"]["e3_identity_sha256"],
        "git_commit": "b" * 40, "complete": True, "final": False,
        "device": "cpu", "gpu_model": "cpu", "torch_version": "0.0.0", "cuda_version": "none",
        "image_count_expected": 20, "image_count_processed": 20,
        "image_order_digest": "a" * 64, "windows_processed_total": windows,
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
        "per_image_stats_manifest_path": "stats.json", "per_image_stats_manifest_sha256": "a" * 64,
        "operation_telemetry": {
            "sample_pulls": 1, "window_enumerations": windows, "backbone_snapshot_calls": windows,
            "graph_builds": windows, "propagation_calls": windows, "probability_interpolation_calls": windows,
            "cross_view_comparisons_total": 0,
        },
        "phase_runtime_seconds": {"total": 1.0, "shared": 1.0},
        "peak_gpu_memory_bytes": 0, "resumed_from_checkpoint": False, "source_git_branch": "x", "failure_reason": None,
    }
    decision_output, decision_rationale = classify_reachability(funnel, identity, ignored_gt_count=ignored_gt_count)
    record["decision_output"] = decision_output
    record["decision_rationale"] = decision_rationale
    return record


@pytest.fixture
def scratch(tmp_path):
    return tmp_path


@pytest.fixture
def alignment_limited_parent(scratch):
    identity = load_native_audit_identity(repo_root=ROOT)
    record = _make_synthetic_record(
        identity, edges_with_observer=1000, misclassified_rows=30,
        misclassified_rows_any_defined=5, correct_rows_any_defined=5, single_window_edges=100,
    )
    identity_sha256 = hashlib.sha256((ROOT / "evaluation_identities/e12_native_edge_support_audit.toml").read_bytes()).hexdigest()
    verify_native_audit_record(record, identity, identity_sha256=identity_sha256)  # self-check the fixture
    assert record["decision_output"] == "ALIGNMENT_LIMITED"
    path = scratch / "parent-result.json"
    path.write_text(json.dumps(record))
    return path


# ---------------------------------------------------------------------------
# Baseline end-to-end run
# ---------------------------------------------------------------------------


def test_full_run_produces_valid_deterministic_result(scratch, alignment_limited_parent):
    result_path = scratch / "gate-result.json"
    exit_code = cli.main(["--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent), "--result", str(result_path)])
    assert exit_code == 0
    assert result_path.exists()

    record = json.loads(result_path.read_text())
    assert record["stop_proceed_decision"] == "STOP_NATIVE_SUPPORT_ALIGNMENT_LIMITED"
    assert record["roadmap_authorization"]["next_authorized_stage"] == "eval: add COCO-Object protocol confirmation"
    assert len(record["content_digest"]) == 64


def test_deterministic_output_same_input_same_digest(scratch, alignment_limited_parent):
    result_path_a = scratch / "gate-result-a.json"
    result_path_b = scratch / "gate-result-b.json"
    cli.main(["--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent), "--result", str(result_path_a)])
    cli.main(["--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent), "--result", str(result_path_b)])
    record_a = json.loads(result_path_a.read_text())
    record_b = json.loads(result_path_b.read_text())
    assert record_a["content_digest"] == record_b["content_digest"]
    # only the timestamp should legitimately differ between two independent runs
    diff_keys = {k for k in record_a if record_a[k] != record_b.get(k)}
    assert diff_keys <= {"created_at_utc"}


def test_no_torch_or_cuda_imported_by_cli_module():
    # Static (AST-based) check on the CLI module's OWN source, never global
    # sys.modules: this test file runs alongside sibling test files that DO
    # import torch for unrelated reasons, so a sys.modules check would be
    # contaminated by process-wide state that has nothing to do with what
    # THIS module itself imports.
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(cli))
    forbidden = {"torch", "cuda", "mmcv", "mmseg"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden, f"forbidden top-level import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert node.module.split(".")[0] not in forbidden, f"forbidden top-level import: {node.module}"


# ---------------------------------------------------------------------------
# Failure safety
# ---------------------------------------------------------------------------


def test_existing_result_preserved_without_overwrite(scratch, alignment_limited_parent):
    result_path = scratch / "gate-result.json"
    exit0 = cli.main(["--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent), "--result", str(result_path)])
    assert exit0 == 0
    original = result_path.read_text()

    exit1 = cli.main(["--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent), "--result", str(result_path)])
    assert exit1 == 2
    assert result_path.read_text() == original


def test_overwrite_flag_allows_rerun(scratch, alignment_limited_parent):
    result_path = scratch / "gate-result.json"
    cli.main(["--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent), "--result", str(result_path)])
    exit_code = cli.main(["--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent), "--result", str(result_path), "--overwrite"])
    assert exit_code == 0


def test_malformed_parent_json_exit_2_no_traceback(scratch):
    bad_parent = scratch / "bad.json"
    bad_parent.write_text('{"a": 1, "a": 2}')
    result_path = scratch / "gate-result.json"
    exit_code = cli.main(["--repo-root", str(ROOT), "--audit-result", str(bad_parent), "--result", str(result_path)])
    assert exit_code == 2
    assert not result_path.exists()


def test_nonexistent_parent_path_exit_2(scratch):
    result_path = scratch / "gate-result.json"
    exit_code = cli.main([
        "--repo-root", str(ROOT), "--audit-result", str(scratch / "does-not-exist.json"), "--result", str(result_path),
    ])
    assert exit_code == 2
    assert not result_path.exists()


def test_no_partial_output_on_failure(scratch):
    bad_parent = scratch / "bad.json"
    bad_parent.write_text('{"not": "a valid mechanics20 result"}')
    result_path = scratch / "gate-result.json"
    exit_code = cli.main(["--repo-root", str(ROOT), "--audit-result", str(bad_parent), "--result", str(result_path)])
    assert exit_code == 2
    leftover_tmp = list(scratch.glob("*.tmp"))
    assert leftover_tmp == []
    assert not result_path.exists()


def test_keyboard_interrupt_not_swallowed(scratch, alignment_limited_parent):
    real_build = cli.build_gate_report

    def raising(*a, **k):
        raise KeyboardInterrupt()

    cli.build_gate_report = raising
    try:
        result_path = scratch / "gate-result.json"
        with pytest.raises(KeyboardInterrupt):
            cli.main(["--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent), "--result", str(result_path)])
    finally:
        cli.build_gate_report = real_build


def test_system_exit_not_swallowed(scratch, alignment_limited_parent):
    real_build = cli.build_gate_report

    def raising(*a, **k):
        raise SystemExit(77)

    cli.build_gate_report = raising
    try:
        result_path = scratch / "gate-result.json"
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent), "--result", str(result_path)])
        assert excinfo.value.code == 77
    finally:
        cli.build_gate_report = real_build


def test_parent_artifact_never_modified_by_cli(scratch, alignment_limited_parent):
    before_bytes = alignment_limited_parent.read_bytes()
    before_mtime = alignment_limited_parent.stat().st_mtime_ns
    result_path = scratch / "gate-result.json"
    cli.main(["--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent), "--result", str(result_path)])
    assert alignment_limited_parent.read_bytes() == before_bytes
    assert alignment_limited_parent.stat().st_mtime_ns == before_mtime


# ---------------------------------------------------------------------------
# --list-candidates: informational only, never auto-selects
# ---------------------------------------------------------------------------


def test_list_candidates_never_selects_a_file(scratch, alignment_limited_parent, capsys):
    exit_code = cli.main(["--list-candidates", str(scratch)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "informational only" in captured.out or "no candidate" in captured.out


# ---------------------------------------------------------------------------
# OSError exception boundary: main() must convert every expected filesystem
# OSError into a clean gate-domain failure (exit 2, standard failure prefix,
# no traceback, no output/temp-file mutation) -- never a raw Python
# traceback (Python's default uncaught-exception behaviour, exit 1).
# ---------------------------------------------------------------------------


def test_output_parent_is_regular_file_exit_2_no_traceback(scratch, alignment_limited_parent):
    blocker = scratch / "blocker"
    blocker.write_text("not a directory")
    result_path = blocker / "subdir" / "result.json"

    proc = subprocess.run(
        [
            sys.executable, str(ROOT / "evaluate_native_edge_support_reachability_gate.py"),
            "--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent), "--result", str(result_path),
        ],
        capture_output=True, text=True,
    )

    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "NATIVE EDGE SUPPORT REACHABILITY GATE FAIL" in proc.stderr
    assert str(blocker) in proc.stderr
    assert "Traceback" not in proc.stderr
    assert not result_path.exists()
    assert list(scratch.glob("**/*.tmp")) == []


def test_mkdir_permission_error_exit_2_no_traceback(scratch, alignment_limited_parent, monkeypatch, capsys):
    def raising_mkdir(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "mkdir", raising_mkdir)
    result_path = scratch / "subdir" / "gate-result.json"

    exit_code = cli.main(["--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent), "--result", str(result_path)])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "NATIVE EDGE SUPPORT REACHABILITY GATE FAIL" in captured.err
    assert "Traceback" not in captured.err
    assert not result_path.exists()


def test_atomic_write_failure_exit_2_preserves_existing_result(scratch, alignment_limited_parent, capsys):
    result_path = scratch / "gate-result.json"
    exit0 = cli.main(["--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent), "--result", str(result_path)])
    assert exit0 == 0
    before_bytes = result_path.read_bytes()
    before_sha256 = hashlib.sha256(before_bytes).hexdigest()
    before_mtime = result_path.stat().st_mtime_ns

    real_write_checkpoint_atomically = cli.write_checkpoint_atomically

    def raising(*args, **kwargs):
        raise OSError("atomic write probe")

    cli.write_checkpoint_atomically = raising
    try:
        exit1 = cli.main([
            "--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent),
            "--result", str(result_path), "--overwrite",
        ])
    finally:
        cli.write_checkpoint_atomically = real_write_checkpoint_atomically

    assert exit1 == 2
    captured = capsys.readouterr()
    assert "NATIVE EDGE SUPPORT REACHABILITY GATE FAIL" in captured.err
    assert "Traceback" not in captured.err
    assert result_path.read_bytes() == before_bytes
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == before_sha256
    assert result_path.stat().st_mtime_ns == before_mtime
    assert list(scratch.glob("*.tmp")) == []


def test_atomic_write_failure_exit_2_no_partial_output_without_existing_result(scratch, alignment_limited_parent, capsys):
    result_path = scratch / "gate-result.json"
    real_write_checkpoint_atomically = cli.write_checkpoint_atomically

    def raising(*args, **kwargs):
        raise OSError("atomic write probe")

    cli.write_checkpoint_atomically = raising
    try:
        exit_code = cli.main(["--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent), "--result", str(result_path)])
    finally:
        cli.write_checkpoint_atomically = real_write_checkpoint_atomically

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "NATIVE EDGE SUPPORT REACHABILITY GATE FAIL" in captured.err
    assert "Traceback" not in captured.err
    assert not result_path.exists()
    assert list(scratch.glob("*.tmp")) == []


def test_parent_artifact_read_failure_exit_2_no_output_mutation(scratch, alignment_limited_parent, capsys):
    real_select_and_validate_parent_artifact = cli.select_and_validate_parent_artifact

    def raising(*args, **kwargs):
        raise OSError("parent artifact read probe")

    cli.select_and_validate_parent_artifact = raising
    try:
        result_path = scratch / "gate-result.json"
        exit_code = cli.main(["--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent), "--result", str(result_path)])
    finally:
        cli.select_and_validate_parent_artifact = real_select_and_validate_parent_artifact

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "NATIVE EDGE SUPPORT REACHABILITY GATE FAIL" in captured.err
    assert "Traceback" not in captured.err
    assert not result_path.exists()
    assert list(scratch.glob("*.tmp")) == []


def test_memory_error_not_reclassified_as_validation_failure(scratch, alignment_limited_parent):
    real_build_gate_report = cli.build_gate_report

    def raising(*args, **kwargs):
        raise MemoryError()

    cli.build_gate_report = raising
    try:
        result_path = scratch / "gate-result.json"
        with pytest.raises(MemoryError):
            cli.main(["--repo-root", str(ROOT), "--audit-result", str(alignment_limited_parent), "--result", str(result_path)])
    finally:
        cli.build_gate_report = real_build_gate_report
