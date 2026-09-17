"""Tests for the E24 BALP development-closure authority
(`src.dfp_phase1b_balp_development_closure_identity`,
`src.dfp_phase1b_balp_development_closure`) and its CPU-only analysis
CLI (`diagnostics/run_dfp_phase1b_balp_development_closure.py`).

Most tests use a small SYNTHETIC checkpoint fixture (never the real
1.7GB checkpoint arrays) to exercise eligibility-rule edge cases
deterministically and fast. A few tests exercise the REAL, already-
completed, immutable source output root at
`/scratch/haree/dfp_phase1b_balp_development/coco_stuff_350` --
strictly READ-ONLY, and always via a temporary, fresh
`--destination-root` this suite creates and tears down itself, never
the frozen analysis destination the closure task installs into.

No CUDA, no torch, no model, no inference anywhere in this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import src.dfp_phase1b_balp_development_closure as closure
import src.dfp_phase1b_balp_development_closure_identity as e24
import src.dfp_phase1b_lightweight_boundary_identity as e22

ROOT = Path(__file__).resolve().parent.parent
REAL_SOURCE_ROOT = Path("/scratch/haree/dfp_phase1b_balp_development/coco_stuff_350")
CLI = ROOT / "diagnostics" / "run_dfp_phase1b_balp_development_closure.py"
E = closure.DfpPhase1bBalpDevelopmentClosureError

# ---------------------------------------------------------------------
# Shared, session-lifetime test doubles for the provenance parameters
# `build_closure_document` requires. Never used by the real `generate`
# CLI path (which always injects `verify_clean_head_generation_gate`'s
# real result) -- these exist purely so the many ELIGIBILITY/VALIDATION
# tests below (which build documents directly, not through the CLI)
# don't each need to construct a real git-backed source/repo fixture.
# ---------------------------------------------------------------------

_FAKE_SOURCE_ROOT = Path(tempfile.mkdtemp(prefix="balp_closure_fake_source_"))
(_FAKE_SOURCE_ROOT / "checkpoint_generations" / "generation_1").mkdir(parents=True)
(_FAKE_SOURCE_ROOT / "current_checkpoint.json").write_text(json.dumps({
    "format": "dfp-phase1b-lightweight-boundary-current-checkpoint-pointer-v1",
    "current_generation": 1,
    "generation_directory": "generation_1",
    "checkpoint_json_sha256": "0" * 64,
    "checkpoint_arrays_sha256": "0" * 64,
}))

_FAKE_CLOSURE_GENERATION = {
    "repository_commit": "1" * 40,
    "implementation_sha256": "2" * 64,
    "cli_sha256": "3" * 64,
}


def _build(checkpoint, *, source_output_root=None, closure_generation=None, repo_root=None):
    """Shorthand for `build_closure_document` with the fake provenance
    test doubles above, for tests concerned with eligibility/validation
    logic rather than provenance/generation-gating itself."""
    return closure.build_closure_document(
        checkpoint,
        repo_root=repo_root or ROOT,
        source_output_root=source_output_root or _FAKE_SOURCE_ROOT,
        closure_generation=closure_generation or _FAKE_CLOSURE_GENERATION,
    )


# ---------------------------------------------------------------------
# E24 identity itself.
# ---------------------------------------------------------------------


def test_e24_identity_loads_and_verifies():
    doc = e24.load_identity(repo_root=ROOT)
    assert doc["frozen_decision"]["eligible_candidate_count"] == 0
    assert doc["frozen_decision"]["development_outcome"] == "NO_BALP_CANDIDATE_PROCEEDS"
    assert doc["frozen_decision"]["profiling_status"] == "NOT_RECORDED"
    assert doc["population"]["development_images"] == 350
    assert doc["population"]["development_windows"] == 769
    assert doc["population"]["evaluated_pixels"] == 91454742


def test_e24_repository_gate_accepts_exact_and_descendant_commit():
    e24.verify_repository_gate(ROOT, e24.REQUIRED_ANCESTOR_COMMIT)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    e24.verify_repository_gate(ROOT, head)


def test_e24_repository_gate_rejects_unrelated_commit():
    unrelated = "0" * 40
    with pytest.raises(e24.DfpPhase1bBalpDevelopmentClosureIdentityError):
        e24.verify_repository_gate(ROOT, unrelated)


# ---------------------------------------------------------------------
# Synthetic checkpoint fixture: two active classes (0 and 1), symmetric
# confusion so mIoU = I / (I + FN + FP) exactly, in percentage points
# -- no floating-point rounding ambiguity for the eligibility-boundary
# tests below.
# ---------------------------------------------------------------------


def _make_cm(intersection: int, fn: int, fp: int) -> list:
    cm = np.zeros((closure.CLASS_COUNT, closure.CLASS_COUNT), dtype=np.int64)
    cm[0, 0] = intersection
    cm[1, 1] = intersection
    cm[0, 1] = fn
    cm[1, 0] = fp
    return cm.tolist()


def _synthetic_checkpoint(*, t320=(80, 10, 10), candidate_overrides: dict | None = None):
    """All 5 controls default to T320's own (I,FN,FP); every candidate
    defaults to a clearly-below-T320 value (70,15,15 -> 70.0pp), except
    entries named in `candidate_overrides`."""
    candidate_overrides = candidate_overrides or {}
    agg = {}
    for control in e22.CANONICAL_CONTROLS:
        agg[control] = _make_cm(*t320)
    per_image = {}
    for cid in e22.reconstruct_candidate_ids():
        i, fn, fp = candidate_overrides.get(cid, (70, 15, 15))
        agg[cid] = _make_cm(i, fn, fp)
        per_image[cid] = {"candidate_versus_t320_changed_pixel_count": [0], "preservation_mask_pixel_count": [0]}
    return {"aggregate_confusion": agg, "per_image_reconstruction_statistics": per_image}


def _t320_miou(t320=(80, 10, 10)):
    i, fn, fp = t320
    return i / (i + fn + fp) * 100.0


def test_synthetic_checkpoint_zero_eligible_baseline():
    checkpoint = _synthetic_checkpoint()
    doc = _build(checkpoint)
    assert doc["decision"]["eligible_candidate_count"] == 0
    assert doc["decision"]["outcome"] == "NO_BALP_CANDIDATE_PROCEEDS"
    assert doc["decision"]["eligible_candidates"] == []
    assert abs(doc["controls"]["T320"]["miou"] - _t320_miou()) < 1e-9


def test_equality_to_t320_remains_ineligible():
    """A candidate whose mIoU is EXACTLY equal to T320's must remain
    ineligible under E22's strict `>` rule."""
    some_id = e22.reconstruct_candidate_ids()[0]
    checkpoint = _synthetic_checkpoint(candidate_overrides={some_id: (80, 10, 10)})  # identical to T320
    doc = _build(checkpoint)
    assert doc["candidates"][some_id]["miou"] == doc["controls"]["T320"]["miou"]
    assert doc["candidates"][some_id]["eligible"] is False
    assert doc["decision"]["eligible_candidate_count"] == 0


def test_one_candidate_strictly_above_t320_becomes_eligible():
    some_id = e22.reconstruct_candidate_ids()[0]
    checkpoint = _synthetic_checkpoint(candidate_overrides={some_id: (81, 9, 10)})  # 81.0pp > 80.0pp
    doc = _build(checkpoint)
    assert doc["candidates"][some_id]["miou"] > doc["controls"]["T320"]["miou"]
    assert doc["candidates"][some_id]["eligible"] is True
    assert doc["decision"]["eligible_candidate_count"] == 1
    assert doc["decision"]["eligible_candidates"] == [some_id]
    # A single eligible candidate under a MUTATED fixture must NOT be
    # forced to the frozen NO_BALP_CANDIDATE_PROCEEDS outcome string --
    # validate_closure_document itself rejects that combination.
    assert doc["decision"]["outcome"] != "NO_BALP_CANDIDATE_PROCEEDS"


def test_validate_closure_document_rejects_outcome_disagreeing_with_eligible_count():
    checkpoint = _synthetic_checkpoint()
    doc = _build(checkpoint)
    mutated = copy.deepcopy(doc)
    some_id = next(iter(mutated["candidates"]))
    mutated["candidates"][some_id]["eligible"] = True
    mutated["decision"]["eligible_candidate_count"] = 1
    mutated["decision"]["eligible_candidates"] = [some_id]
    # outcome deliberately left stale at the frozen zero-eligible string.
    with pytest.raises(E, match="outcome"):
        closure.validate_closure_document(mutated)


def test_candidate_order_and_completeness():
    checkpoint = _synthetic_checkpoint()
    doc = _build(checkpoint)
    assert set(doc["candidates"].keys()) == set(e22.reconstruct_candidate_ids())
    assert len(doc["candidates"]) == 16
    assert set(doc["controls"].keys()) == set(e22.CANONICAL_CONTROLS)


def test_validate_closure_document_rejects_missing_candidate():
    checkpoint = _synthetic_checkpoint()
    doc = _build(checkpoint)
    mutated = copy.deepcopy(doc)
    del mutated["candidates"][next(iter(mutated["candidates"]))]
    with pytest.raises(E, match="16-candidate family"):
        closure.validate_closure_document(mutated)


def test_validate_closure_document_rejects_extra_candidate():
    checkpoint = _synthetic_checkpoint()
    doc = _build(checkpoint)
    mutated = copy.deepcopy(doc)
    extra = dict(next(iter(mutated["candidates"].values())))
    extra["candidate_id"] = "balp_s999_r99"
    mutated["candidates"]["balp_s999_r99"] = extra
    with pytest.raises(E, match="16-candidate family"):
        closure.validate_closure_document(mutated)


def test_validate_closure_document_rejects_unknown_top_level_key():
    checkpoint = _synthetic_checkpoint()
    doc = _build(checkpoint)
    mutated = dict(doc)
    mutated["extra_field"] = 1
    with pytest.raises(E, match="wrong key set"):
        closure.validate_closure_document(mutated)


def test_validate_closure_document_rejects_fabricated_profiling():
    checkpoint = _synthetic_checkpoint()
    doc = _build(checkpoint)
    mutated = copy.deepcopy(doc)
    mutated["profiling_disclosure"]["profiling_artifact_present"] = True
    with pytest.raises(E, match="fabrication is forbidden"):
        closure.validate_closure_document(mutated)

    mutated2 = copy.deepcopy(doc)
    mutated2["profiling_disclosure"]["profiling_status"] = "RECORDED"
    with pytest.raises(E, match="fabrication is forbidden"):
        closure.validate_closure_document(mutated2)


def test_validate_closure_document_rejects_fabricated_telemetry():
    checkpoint = _synthetic_checkpoint()
    doc = _build(checkpoint)
    for field in ("stage_times_available", "host_rss_available", "gpu_vram_available"):
        mutated = copy.deepcopy(doc)
        mutated["profiling_disclosure"][field] = True
        with pytest.raises(E, match="unavailable telemetry"):
            closure.validate_closure_document(mutated)


def test_validate_closure_document_rejects_bottleneck_claim():
    checkpoint = _synthetic_checkpoint()
    doc = _build(checkpoint)
    mutated = copy.deepcopy(doc)
    mutated["profiling_disclosure"]["bottleneck_identified"] = True
    with pytest.raises(E, match="bottleneck_identified"):
        closure.validate_closure_document(mutated)


def test_validate_closure_document_rejects_confirmation_or_full_eval_flags():
    checkpoint = _synthetic_checkpoint()
    doc = _build(checkpoint)
    mutated = copy.deepcopy(doc)
    mutated["decision"]["confirmation_must_not_run"] = False
    with pytest.raises(E, match="confirmation_must_not_run"):
        closure.validate_closure_document(mutated)

    mutated2 = copy.deepcopy(doc)
    mutated2["decision"]["candidate_selected"] = True
    with pytest.raises(E, match="candidate_selected"):
        closure.validate_closure_document(mutated2)

    mutated3 = copy.deepcopy(doc)
    mutated3["interpretation_boundary"]["confirmation_claim_made"] = True
    with pytest.raises(E, match="confirmation, full-validation, or a bottleneck"):
        closure.validate_closure_document(mutated3)


def test_validate_closure_document_rejects_tampered_linear_extrapolation_label():
    checkpoint = _synthetic_checkpoint()
    doc = _build(checkpoint)
    mutated = copy.deepcopy(doc)
    mutated["linear_extrapolation"]["label"] = "PROFILED MEASUREMENT"
    with pytest.raises(E, match="disagrees with the frozen required disclosure text"):
        closure.validate_closure_document(mutated)


def test_validate_closure_document_rejects_tampered_projection_arithmetic():
    checkpoint = _synthetic_checkpoint()
    doc = _build(checkpoint)
    mutated = copy.deepcopy(doc)
    mutated["linear_extrapolation"]["projected_5000_image_hours"] = 0.0
    with pytest.raises(E, match="fresh recomputation"):
        closure.validate_closure_document(mutated)


# ---------------------------------------------------------------------
# Atomic install: no overwrite, roundtrip, determinism across two
# independent fresh destinations.
# ---------------------------------------------------------------------


def test_install_closure_roundtrip_and_no_overwrite():
    checkpoint = _synthetic_checkpoint()
    doc = _build(checkpoint)
    md = closure.build_closure_markdown(doc)
    dest = Path(tempfile.mkdtemp(prefix="balp_closure_")) / "closure_dest"
    try:
        installed = closure.install_closure(dest, json_doc=doc, markdown_text=md)
        assert installed == dest
        assert (dest / "closure.json").is_file()
        assert (dest / "closure.md").is_file()
        reloaded = closure.load_closure(dest)
        assert reloaded == doc

        with pytest.raises(E, match="already exists"):
            closure.install_closure(dest, json_doc=doc, markdown_text=md)
        assert closure.load_closure(dest) == doc  # untouched
    finally:
        shutil.rmtree(dest.parent, ignore_errors=True)


def test_install_closure_rejects_non_fresh_destination_that_is_a_plain_file():
    checkpoint = _synthetic_checkpoint()
    doc = _build(checkpoint)
    md = closure.build_closure_markdown(doc)
    parent = Path(tempfile.mkdtemp(prefix="balp_closure_nonfresh_"))
    dest = parent / "already_here"
    try:
        dest.write_text("pre-existing, not a closure directory")
        with pytest.raises(E, match="already exists"):
            closure.install_closure(dest, json_doc=doc, markdown_text=md)
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def test_generation_twice_into_separate_fresh_destinations_is_byte_identical():
    """Determinism: the closure document (all scientific/decision
    content -- this schema carries no timestamp field at all) must be
    byte-identical across two independent generations from the same
    source checkpoint."""
    checkpoint = _synthetic_checkpoint()
    doc_a = _build(checkpoint)
    doc_b = _build(checkpoint)
    assert doc_a == doc_b

    md_a = closure.build_closure_markdown(doc_a)
    md_b = closure.build_closure_markdown(doc_b)
    assert md_a == md_b

    parent = Path(tempfile.mkdtemp(prefix="balp_closure_determinism_"))
    try:
        dest_a = parent / "dest_a"
        dest_b = parent / "dest_b"
        closure.install_closure(dest_a, json_doc=doc_a, markdown_text=md_a)
        closure.install_closure(dest_b, json_doc=doc_b, markdown_text=md_b)
        assert (dest_a / "closure.json").read_bytes() == (dest_b / "closure.json").read_bytes()
        assert (dest_a / "closure.md").read_bytes() == (dest_b / "closure.md").read_bytes()
    finally:
        shutil.rmtree(parent, ignore_errors=True)


# ---------------------------------------------------------------------
# Real (read-only) source verification -- rejects a wrong digest / wrong
# result hash, and confirms the real source artifact is byte-unchanged
# by this suite's own read-only access.
# ---------------------------------------------------------------------


@pytest.mark.slow
def test_real_source_verification_passes_and_is_read_only():
    before = (REAL_SOURCE_ROOT / "development_result" / "result.json").read_bytes()
    checkpoint = closure.verify_source_result(REAL_SOURCE_ROOT, repo_root=ROOT)
    assert checkpoint["complete"] is True and checkpoint["final"] is True
    after = (REAL_SOURCE_ROOT / "development_result" / "result.json").read_bytes()
    assert before == after, "reading the source result must never modify it"

    doc = _build(checkpoint, source_output_root=REAL_SOURCE_ROOT)
    assert doc["provenance"]["source_run"]["scientific_output_digest"] == e24.SCIENTIFIC_OUTPUT_DIGEST
    assert doc["provenance"]["source_run"]["repository_commit"] == e24.REQUIRED_ANCESTOR_COMMIT
    assert doc["provenance"]["closure_generation"]["repository_commit"] == _FAKE_CLOSURE_GENERATION["repository_commit"]
    assert doc["decision"]["eligible_candidate_count"] == 0
    assert doc["decision"]["outcome"] == "NO_BALP_CANDIDATE_PROCEEDS"


def test_verify_source_result_rejects_wrong_scientific_digest(tmp_path):
    """A source whose result.json scientific_output_digest disagrees
    with E24's pinned value must be rejected -- simulated by pointing
    at a corrupted COPY of the real source root's result.json only.
    The (1.7GB) checkpoint_generations/ and current_checkpoint.json are
    SYMLINKED, never copied -- this test only ever reads them."""
    import json

    fake_root = tmp_path / "poisoned_source"
    fake_root.mkdir()
    (fake_root / "checkpoint_generations").symlink_to(REAL_SOURCE_ROOT / "checkpoint_generations")
    (fake_root / "current_checkpoint.json").symlink_to(REAL_SOURCE_ROOT / "current_checkpoint.json")
    (fake_root / "development_result").mkdir()

    real_result = json.loads((REAL_SOURCE_ROOT / "development_result" / "result.json").read_bytes())
    real_result["scientific_output_digest"] = "f" * 64
    (fake_root / "development_result" / "result.json").write_text(json.dumps(real_result, indent=2, sort_keys=True) + "\n")

    with pytest.raises(E, match="sha256 drift|scientific_output_digest"):
        closure.verify_source_result(fake_root, repo_root=ROOT)


# ---------------------------------------------------------------------
# CLI subprocess tests.
# ---------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CLI), *args], capture_output=True, text=True, cwd=ROOT)


def test_cli_verify_authority():
    result = _run_cli("verify-authority")
    assert result.returncode == 0, result.stderr
    assert "eligible_candidate_count=0" in result.stdout


@pytest.mark.slow
def test_cli_verify_source_real_root():
    result = _run_cli("verify-source", "--source-output-root", str(REAL_SOURCE_ROOT))
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout


@pytest.mark.slow
def test_cli_generate_against_the_real_uncommitted_repo_is_rejected_and_creates_no_output():
    """This repair is deliberately left uncommitted (Section H). The
    real CLI's `generate`, against the real repo root, must therefore
    fail closed right now -- and must create no destination at all.
    This test's PASS condition depends on the repository staying dirty,
    which is exactly the state this task's own instructions require."""
    dest = Path(tempfile.mkdtemp(prefix="balp_closure_cli_dirty_")) / "closure"
    try:
        result = _run_cli("generate", "--source-output-root", str(REAL_SOURCE_ROOT), "--destination-root", str(dest))
        assert result.returncode != 0
        assert "not clean" in result.stderr or "clean" in result.stderr
        assert not dest.exists(), "a rejected generate must never create the destination"
    finally:
        shutil.rmtree(dest.parent, ignore_errors=True)


# ---------------------------------------------------------------------
# Controlled temporary fixture repository: a throwaway clone of this
# repo's REAL committed history (so E24's REQUIRED_ANCESTOR_COMMIT
# really is an ancestor), with the closure files added and committed
# ON TOP as one ordinary test-fixture commit. This is explicitly NOT
# claimed to be any real production commit -- its hash is whatever git
# assigns it, never fabricated or asserted to equal a specific value.
# Exercises "clean committed-HEAD generation" without ever touching or
# depending on the real, currently-uncommitted Talk2DINO working tree.
# ---------------------------------------------------------------------

_FIXTURE_CLOSURE_FILES = (
    "evaluation_identities/e24_dfp_phase1b_balp_development_closure.toml",
    "src/dfp_phase1b_balp_development_closure.py",
    "src/dfp_phase1b_balp_development_closure_identity.py",
    "diagnostics/run_dfp_phase1b_balp_development_closure.py",
)


def _make_clean_fixture_repo() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="balp_closure_fixture_repo_"))
    repo = tmp / "repo"
    subprocess.run(["git", "clone", "--quiet", str(ROOT), str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", e24.REQUIRED_ANCESTOR_COMMIT], check=True, capture_output=True)
    for rel in _FIXTURE_CLOSURE_FILES:
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        [
            "git", "-C", str(repo), "-c", "user.email=test@example.invalid", "-c", "user.name=balp-closure-tests",
            "commit", "--quiet", "-m", "test fixture only: add closure files on top of the real ancestor commit",
        ],
        check=True, capture_output=True,
    )
    return repo


@pytest.fixture(scope="module")
def clean_fixture_repo():
    repo = _make_clean_fixture_repo()
    try:
        yield repo
    finally:
        shutil.rmtree(repo.parent, ignore_errors=True)


def test_fixture_repo_ancestor_commit_is_a_real_ancestor(clean_fixture_repo):
    """Sanity check on the fixture itself: its HEAD must genuinely
    descend from E24's REQUIRED_ANCESTOR_COMMIT (not an unrelated,
    disconnected history) -- otherwise a "clean generation succeeds"
    test against it would prove nothing."""
    e24.verify_repository_gate(clean_fixture_repo, e24.REQUIRED_ANCESTOR_COMMIT)
    head = subprocess.run(["git", "-C", str(clean_fixture_repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert head != e24.REQUIRED_ANCESTOR_COMMIT, "the fixture must add a NEW commit on top, not reuse the source-run commit itself"


def test_generation_gate_succeeds_on_clean_fixture_repo(clean_fixture_repo):
    result = closure.verify_clean_head_generation_gate(clean_fixture_repo)
    head = subprocess.run(["git", "-C", str(clean_fixture_repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert result["repository_commit"] == head
    committed_impl = subprocess.run(
        ["git", "-C", str(clean_fixture_repo), "show", f"{head}:{closure.IMPLEMENTATION_RELATIVE_PATH.as_posix()}"],
        capture_output=True, check=True,
    ).stdout
    assert result["implementation_sha256"] == hashlib.sha256(committed_impl).hexdigest()
    committed_cli = subprocess.run(
        ["git", "-C", str(clean_fixture_repo), "show", f"{head}:{closure.CLI_RELATIVE_PATH.as_posix()}"],
        capture_output=True, check=True,
    ).stdout
    assert result["cli_sha256"] == hashlib.sha256(committed_cli).hexdigest()
    # Never fabricates/claims the fixture's throwaway commit is the
    # real, frozen source-run commit.
    assert result["repository_commit"] != e24.REQUIRED_ANCESTOR_COMMIT


def test_generation_gate_rejects_staged_changes(clean_fixture_repo):
    (clean_fixture_repo / "README_TEST_STAGE_ONLY.txt").write_text("staged-only test file, never committed\n")
    subprocess.run(["git", "-C", str(clean_fixture_repo), "add", "README_TEST_STAGE_ONLY.txt"], check=True, capture_output=True)
    try:
        with pytest.raises(E, match="staged"):
            closure.verify_clean_head_generation_gate(clean_fixture_repo)
    finally:
        subprocess.run(["git", "-C", str(clean_fixture_repo), "reset", "--quiet", "HEAD", "--", "README_TEST_STAGE_ONLY.txt"], capture_output=True)
        (clean_fixture_repo / "README_TEST_STAGE_ONLY.txt").unlink(missing_ok=True)


def test_generation_gate_rejects_unstaged_dirty_changes(clean_fixture_repo):
    target = clean_fixture_repo / closure.IMPLEMENTATION_RELATIVE_PATH
    original = target.read_bytes()
    target.write_bytes(original + b"\n# unstaged tamper, never committed\n")
    try:
        with pytest.raises(E, match="not clean"):
            closure.verify_clean_head_generation_gate(clean_fixture_repo)
    finally:
        target.write_bytes(original)


def test_generation_gate_rejects_when_committed_bytes_differ_from_on_disk(clean_fixture_repo, monkeypatch):
    """Defense-in-depth, isolated from the porcelain-cleanliness check:
    even if `git status`/`git diff --cached` somehow report clean, a
    committed-vs-on-disk byte mismatch for the closure implementation
    must still be caught and rejected."""
    real_run = subprocess.run

    def fake_run(cmd, *a, **kw):
        if len(cmd) >= 4 and cmd[0] == "git" and cmd[3] == "show" and closure.IMPLEMENTATION_RELATIVE_PATH.as_posix() in cmd[-1]:
            result = subprocess.CompletedProcess(cmd, 0, stdout=b"# tampered committed bytes, never actually on disk\n", stderr=b"")
            return result
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(closure.subprocess, "run", fake_run)
    with pytest.raises(E, match="differ from committed HEAD"):
        closure.verify_clean_head_generation_gate(clean_fixture_repo)


@pytest.mark.slow
def test_cli_generate_on_fixture_repo_then_verify_then_reject_second_install(clean_fixture_repo):
    dest = Path(tempfile.mkdtemp(prefix="balp_closure_cli_fixture_")) / "closure"
    try:
        result = _run_cli(
            "--repo-root", str(clean_fixture_repo), "generate",
            "--source-output-root", str(REAL_SOURCE_ROOT), "--destination-root", str(dest),
        )
        assert result.returncode == 0, result.stderr
        assert "clean-HEAD generation gate: PASS" in result.stdout
        assert "CLOSURE INSTALLED AND VERIFIED" in result.stdout
        assert (dest / "closure.json").is_file()
        assert (dest / "closure.md").is_file()

        doc = json.loads((dest / "closure.json").read_bytes())
        head = subprocess.run(["git", "-C", str(clean_fixture_repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        assert doc["provenance"]["source_run"]["repository_commit"] == e24.REQUIRED_ANCESTOR_COMMIT
        assert doc["provenance"]["closure_generation"]["repository_commit"] == head
        assert doc["provenance"]["source_run"]["repository_commit"] != doc["provenance"]["closure_generation"]["repository_commit"]

        verify_result = _run_cli("verify-closure", "--destination-root", str(dest))
        assert verify_result.returncode == 0, verify_result.stderr
        assert "NO_BALP_CANDIDATE_PROCEEDS" in verify_result.stdout

        second = _run_cli(
            "--repo-root", str(clean_fixture_repo), "generate",
            "--source-output-root", str(REAL_SOURCE_ROOT), "--destination-root", str(dest),
        )
        assert second.returncode != 0
        assert "already exists" in second.stderr
    finally:
        shutil.rmtree(dest.parent, ignore_errors=True)


# ---------------------------------------------------------------------
# Provenance: source_run vs. closure_generation must never be one
# ambiguous field, and swapping/conflating/malforming either is
# rejected at the document-validation level (independent of the gate).
# ---------------------------------------------------------------------


def test_provenance_has_two_separate_commit_fields():
    doc = _build(_synthetic_checkpoint())
    sr = doc["provenance"]["source_run"]
    cg = doc["provenance"]["closure_generation"]
    assert sr["repository_commit"] == e24.REQUIRED_ANCESTOR_COMMIT
    assert cg["repository_commit"] == _FAKE_CLOSURE_GENERATION["repository_commit"]
    assert sr["repository_commit"] != cg["repository_commit"]
    assert sr["job_id"] == e24.SLURM_JOB_ID


def test_validate_closure_document_rejects_conflated_commits():
    doc = _build(_synthetic_checkpoint())
    mutated = copy.deepcopy(doc)
    mutated["provenance"]["closure_generation"]["repository_commit"] = mutated["provenance"]["source_run"]["repository_commit"]
    with pytest.raises(E, match="conflate"):
        closure.validate_closure_document(mutated)


def test_validate_closure_document_rejects_wrong_source_run_commit():
    doc = _build(_synthetic_checkpoint())
    mutated = copy.deepcopy(doc)
    mutated["provenance"]["source_run"]["repository_commit"] = "a" * 40
    with pytest.raises(E, match="frozen source-run commit"):
        closure.validate_closure_document(mutated)


def test_validate_closure_document_rejects_malformed_closure_generation_commit():
    doc = _build(_synthetic_checkpoint())
    mutated = copy.deepcopy(doc)
    mutated["provenance"]["closure_generation"]["repository_commit"] = "not-a-commit"
    with pytest.raises(E, match="40-hex"):
        closure.validate_closure_document(mutated)


def test_validate_closure_document_rejects_missing_source_run_key():
    doc = _build(_synthetic_checkpoint())
    mutated = copy.deepcopy(doc)
    del mutated["provenance"]["source_run"]["job_id"]
    with pytest.raises(E, match="wrong key set"):
        closure.validate_closure_document(mutated)


def test_validate_closure_document_rejects_missing_closure_generation_key():
    doc = _build(_synthetic_checkpoint())
    mutated = copy.deepcopy(doc)
    del mutated["provenance"]["closure_generation"]["cli_sha256"]
    with pytest.raises(E, match="wrong key set"):
        closure.validate_closure_document(mutated)


def test_validate_closure_document_rejects_wrong_source_hashes():
    doc = _build(_synthetic_checkpoint())
    for field in ("result_sha256", "checkpoint_sha256", "checkpoint_arrays_sha256", "scientific_output_digest"):
        mutated = copy.deepcopy(doc)
        mutated["provenance"]["source_run"][field] = "f" * 64
        with pytest.raises(E, match="disagrees with E24"):
            closure.validate_closure_document(mutated)


# ---------------------------------------------------------------------
# Closure JSON and Markdown must agree on provenance (Section G #20).
# ---------------------------------------------------------------------


def test_closure_json_and_markdown_agree_on_provenance():
    doc = _build(_synthetic_checkpoint())
    md = closure.build_closure_markdown(doc)
    sr = doc["provenance"]["source_run"]
    cg = doc["provenance"]["closure_generation"]
    assert sr["repository_commit"] in md
    assert cg["repository_commit"] in md
    assert sr["scientific_output_digest"] in md
    assert sr["job_id"] in md
    assert doc["decision"]["outcome"] in md
    assert str(doc["decision"]["eligible_candidate_count"]) in md


# ---------------------------------------------------------------------
# No raw-profiling schema/authority anywhere in this repository
# (Section C's required post-repair search) or in E24 itself.
# ---------------------------------------------------------------------


_FORBIDDEN_RAW_PROFILING_STRINGS = (
    "dfp-phase1b-balp-development-raw-profiling-v1",
    "install_raw_profiling",
    "verify_raw_profiling",
    "raw_profiling",
)


def test_no_raw_profiling_schema_or_authority_anywhere_in_repo():
    # This test file itself is deliberately excluded: it legitimately
    # names the forbidden strings as literal search patterns (this
    # function) and in historical-context comments -- a "historical
    # statement in a repair report/test is acceptable" (Section C);
    # only executable code, schema authority, and active documentation
    # elsewhere in the repository are checked.
    self_path = Path(__file__).resolve()
    hits: list[str] = []
    for pattern in ("*.py", "*.md", "*.toml"):
        for path in ROOT.rglob(pattern):
            if "/.git/" in str(path) or "__pycache__" in str(path) or path.resolve() == self_path:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in _FORBIDDEN_RAW_PROFILING_STRINGS:
                if needle in text:
                    hits.append(f"{path.relative_to(ROOT)}: {needle!r}")
    assert hits == [], f"unauthorized raw-profiling references remain: {hits}"


def test_e24_identity_has_no_raw_profiling_authorization():
    doc = e24.load_identity(repo_root=ROOT)
    text = json.dumps(doc)
    for needle in _FORBIDDEN_RAW_PROFILING_STRINGS:
        assert needle not in text
    assert doc["frozen_decision"]["profiling_status"] == "NOT_RECORDED"
    assert "fabricated_profiling" in doc["prohibited"]["list"]


def test_committed_e22_e23_are_byte_identical_to_head():
    """E22/E23 must never be amended by this repair (Section H)."""
    for rel in (
        "evaluation_identities/e22_dfp_phase1b_lightweight_boundary_artifacts.toml",
        "evaluation_identities/e22_dfp_phase1b_lightweight_boundary_preservation.toml",
        "evaluation_identities/e23_dfp_phase1b_balp_development_execution.toml",
    ):
        committed = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"], capture_output=True, check=True).stdout
        on_disk = (ROOT / rel).read_bytes()
        assert committed == on_disk, f"{rel} has drifted from committed HEAD"


def test_committed_balp_kernel_and_execution_are_byte_identical_to_head():
    """The BALP scientific kernel/execution/identity/artifact modules
    must never be amended by this repair (Section H)."""
    for rel in (
        "src/dfp_phase1b_lightweight_boundary.py",
        "src/dfp_phase1b_lightweight_boundary_execute.py",
        "src/dfp_phase1b_lightweight_boundary_identity.py",
        "src/dfp_phase1b_lightweight_boundary_artifact.py",
        "src/dfp_phase1b_balp_development_execution_identity.py",
        "src/dfp_phase1b_balp_development_production.py",
        "diagnostics/run_dfp_phase1b_balp_development.py",
    ):
        committed = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"], capture_output=True, check=True).stdout
        on_disk = (ROOT / rel).read_bytes()
        assert committed == on_disk, f"{rel} has drifted from committed HEAD (unauthorized-profiling repair incomplete?)"


# ---------------------------------------------------------------------
# Fatal exceptions always propagate -- never caught by this module's
# broad-looking error handling anywhere in the closure pipeline.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_fatal_exceptions_propagate_through_validate_closure_document(exc_type):
    doc = _build(_synthetic_checkpoint())

    class _Boom(dict):
        def keys(self):
            raise exc_type("fatal, must propagate")

    with pytest.raises(exc_type):
        closure.validate_closure_document(_Boom(doc))


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_fatal_exceptions_propagate_through_generation_gate(clean_fixture_repo, exc_type, monkeypatch):
    def _boom(*a, **kw):
        raise exc_type("fatal, must propagate")

    monkeypatch.setattr(closure.subprocess, "run", _boom)
    with pytest.raises(exc_type):
        closure.verify_clean_head_generation_gate(clean_fixture_repo)


# ---------------------------------------------------------------------
# No torch/CUDA/dataset/job-submission access anywhere in this module
# or the identity loader (Section H / test requirement #25).
# ---------------------------------------------------------------------


def test_closure_modules_import_no_torch_and_touch_no_cuda():
    import sys as _sys

    for modname in ("src.dfp_phase1b_balp_development_closure", "src.dfp_phase1b_balp_development_closure_identity"):
        mod = _sys.modules[modname]
        src_text = Path(mod.__file__).read_text(encoding="utf-8")
        assert "import torch" not in src_text
        assert "sbatch" not in src_text and "srun" not in src_text and "salloc" not in src_text
    assert "torch" not in _sys.modules or not any(
        "dfp_phase1b_balp_development_closure" in m for m in _sys.modules if "torch" in m
    )
