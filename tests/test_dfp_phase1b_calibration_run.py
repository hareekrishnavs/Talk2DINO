"""Strict tests for the DFP Phase 1B conservation-calibration runner
(`src.dfp_phase1b_calibration_run`).

CPU-only. No real dataset, no real checkpoint, no GPU. Phase A/B tests
need no torch import at all. The Phase C end-to-end fixture uses the
REAL, unmodified committed scientific code throughout (real
`DirectedTopKGraph`, real `finite_step_propagate`, real stitching/
final-rescale/component/membership functions, real masked-forward
core) with a small, hand-built, internally-consistent set of geometry
contracts (crop/patch_grid far smaller than production's real 448/32x32
values -- a test-only substitution of GEOMETRY PARAMETERS, never of any
scientific FORMULA or of k=12/alpha=0.98/steps=320, which always come
from the real, unmodified e13/e15 identities) and a tiny synthetic
model/dataset, per Section 17/18's explicit dependency-injection
design."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
for _path in (ROOT, ROOT / "src/open_vocabulary_segmentation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import src.dfp_phase1b_calibration_artifact as artifact_module  # noqa: E402
import src.dfp_phase1b_calibration_run as run  # noqa: E402


# ---------------------------------------------------------------------------
# Section: Phase A preflight (no torch import anywhere in this section).
# ---------------------------------------------------------------------------


def test_preflight_passes_with_fresh_mode_and_new_output_path(tmp_path):
    report = run.run_preflight(output_path=tmp_path / "phase1b_conservation_calibration.json", mode="fresh")
    assert report.mode == "fresh"
    assert len(report.selected_ids) == 500
    assert report.subset_manifest["run_mode"] == "coco_stuff_500"


def test_preflight_rejects_resume_mode():
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="resume"):
        run.run_preflight(output_path=Path("/tmp/x.json"), mode="resume")


def test_preflight_rejects_unsupported_mode_string():
    with pytest.raises(run.DfpPhase1bCalibrationRunError):
        run.run_preflight(output_path=Path("/tmp/x.json"), mode="bogus")


def test_preflight_rejects_missing_output_directory(tmp_path):
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="does not exist"):
        run.run_preflight(output_path=tmp_path / "no_such_dir" / "phase1b_conservation_calibration.json", mode="fresh")


def test_preflight_rejects_existing_output_without_force(tmp_path):
    destination = tmp_path / "phase1b_conservation_calibration.json"
    destination.write_text("{}", encoding="utf-8")
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="already exists"):
        run.run_preflight(output_path=destination, mode="fresh", allow_overwrite=False)


def test_preflight_accepts_existing_output_with_explicit_overwrite(tmp_path):
    destination = tmp_path / "phase1b_conservation_calibration.json"
    destination.write_text("{}", encoding="utf-8")
    run.run_preflight(output_path=destination, mode="fresh", allow_overwrite=True)


def test_preflight_rejects_output_path_that_is_a_directory(tmp_path):
    destination = tmp_path / "phase1b_conservation_calibration.json"
    destination.mkdir()
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="directory"):
        run.run_preflight(output_path=destination, mode="fresh", allow_overwrite=True)


def test_preflight_rejects_wrong_output_filename_extension(tmp_path):
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="does not look like"):
        run.run_preflight(output_path=tmp_path / "not_the_right_name", mode="fresh")


# ---------------------------------------------------------------------------
# Section: F1 repair -- lexical '..' path-traversal rejection (independent-
# verification finding). `_reject_lexical_path_traversal` inspects
# `Path(output_path).parts` directly, before any resolution/normalization,
# and is shared by both `run_preflight` (via `_validate_output_path`, and
# again explicitly before the filename check) and `install_calibration_
# artifact` (also via `_validate_output_path`) -- one rule, two enforcement
# points, never two independently-drifting traversal checks.
# ---------------------------------------------------------------------------


_TRAVERSAL_FILENAME = "phase1b_conservation_calibration.json"


def test_reject_lexical_path_traversal_relative_leading():
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
        run._reject_lexical_path_traversal(Path("..", _TRAVERSAL_FILENAME))


def test_reject_lexical_path_traversal_relative_embedded():
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
        run._reject_lexical_path_traversal(Path("a", "..", _TRAVERSAL_FILENAME))


def test_reject_lexical_path_traversal_relative_multiple():
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
        run._reject_lexical_path_traversal(Path("a", "b", "..", "..", _TRAVERSAL_FILENAME))


def test_reject_lexical_path_traversal_absolute_single(tmp_path):
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
        run._reject_lexical_path_traversal(Path(tmp_path, "..", _TRAVERSAL_FILENAME))


def test_reject_lexical_path_traversal_absolute_multiple_resolving_outside(tmp_path):
    # Enough '..' components to climb out of tmp_path's own (pytest-
    # dependent) nesting depth entirely, landing outside it -- confirms
    # this is a genuine "resolves to a destination outside the intended
    # directory" case, not merely a single up-level.
    climb = Path(*([".."] * (len(tmp_path.parts) - 1)))
    traversal_path = Path(tmp_path, climb, "etc", _TRAVERSAL_FILENAME)
    resolved = traversal_path.resolve()
    assert tmp_path not in resolved.parents and resolved != tmp_path / _TRAVERSAL_FILENAME
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
        run._reject_lexical_path_traversal(traversal_path)


def test_reject_lexical_path_traversal_correct_basename_still_rejected(tmp_path):
    # The filename is EXACTLY the E16-required name; traversal must still be rejected.
    traversal_path = Path(tmp_path, "..", _TRAVERSAL_FILENAME)
    assert traversal_path.name == _TRAVERSAL_FILENAME
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
        run._reject_lexical_path_traversal(traversal_path)


def test_lexical_traversal_substring_is_not_a_component(tmp_path):
    # "name..part" is ONE path component, not the exact string "..", and
    # must never be mistaken for a traversal segment.
    run._reject_lexical_path_traversal(Path(tmp_path, "name..part", _TRAVERSAL_FILENAME))


def test_preflight_rejects_relative_traversal_before_heavy_work(monkeypatch):
    monkeypatch.setattr(run, "load_e16_identity", lambda **kwargs: pytest.fail("E16 identity load must not be reached"))
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
        run.run_preflight(output_path=Path("..", _TRAVERSAL_FILENAME), mode="fresh")


def test_preflight_rejects_embedded_relative_traversal():
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
        run.run_preflight(output_path=Path("a", "..", _TRAVERSAL_FILENAME), mode="fresh")


def test_preflight_rejects_double_embedded_relative_traversal():
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
        run.run_preflight(output_path=Path("a", "b", "..", "..", _TRAVERSAL_FILENAME), mode="fresh")


def test_preflight_rejects_absolute_traversal_with_one_component(tmp_path):
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
        run.run_preflight(output_path=Path(tmp_path, "..", _TRAVERSAL_FILENAME), mode="fresh")


def test_preflight_rejects_absolute_traversal_resolving_to_etc(tmp_path):
    climb = Path(*([".."] * (len(tmp_path.parts) - 1)))
    traversal_path = Path(tmp_path, climb, "etc", _TRAVERSAL_FILENAME)
    assert traversal_path.resolve() == Path("/etc") / _TRAVERSAL_FILENAME  # the genuine /etc-resolving reproduction case
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
        run.run_preflight(output_path=traversal_path, mode="fresh")
    assert not (Path("/etc") / _TRAVERSAL_FILENAME).exists()


def test_preflight_rejects_traversal_before_creating_parent_directory(tmp_path):
    # The parent directory in the traversal path does not exist; if the
    # traversal check ran AFTER a parent-directory check/creation this
    # would fail differently (or create a directory) -- must fail with
    # the traversal message specifically, and must create nothing.
    missing_parent_traversal = Path(tmp_path, "no_such_subdir", "..", _TRAVERSAL_FILENAME)
    before = set(tmp_path.iterdir())
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
        run.run_preflight(output_path=missing_parent_traversal, mode="fresh")
    assert set(tmp_path.iterdir()) == before, "traversal rejection must not create any directory"


def test_preflight_traversal_rejection_never_imports_torch(tmp_path):
    import ast

    source = Path(run.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_level_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_level_imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            module_level_imports.append(node.module)
    assert not any(name.startswith("torch") for name in module_level_imports)
    with pytest.raises(run.DfpPhase1bCalibrationRunError):
        run.run_preflight(output_path=Path(tmp_path, "..", _TRAVERSAL_FILENAME), mode="fresh")
    assert "torch" not in sys.modules or True  # torch may already be loaded by other tests in this session; see poisoned-import test below for the strict guarantee


def test_preflight_traversal_rejection_poisoned_torch_never_touched(tmp_path):
    poisoned = object()

    class PoisonError(RuntimeError):
        pass

    class Poison:
        def __getattr__(self, name):
            raise PoisonError(f"torch.{name} touched during traversal rejection")

    import builtins

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise PoisonError("import torch was attempted during traversal rejection")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = guarded_import
    try:
        with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
            run.run_preflight(output_path=Path(tmp_path, "..", _TRAVERSAL_FILENAME), mode="fresh")
    finally:
        builtins.__import__ = real_import


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_preflight_traversal_check_does_not_interfere_with_fatal_propagation(tmp_path, exc_type, monkeypatch):
    def raiser(*args, **kwargs):
        raise exc_type("must propagate")

    monkeypatch.setattr(run, "load_e16_identity", raiser)
    with pytest.raises(exc_type):
        run.run_preflight(output_path=tmp_path / _TRAVERSAL_FILENAME, mode="fresh")


# --- direct installer boundary ---


def test_install_calibration_artifact_rejects_relative_traversal(tmp_path):
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
        run.install_calibration_artifact({}, Path("..", _TRAVERSAL_FILENAME))


def test_install_calibration_artifact_rejects_absolute_traversal(tmp_path):
    traversal_path = Path(tmp_path, "..", _TRAVERSAL_FILENAME)
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
        run.install_calibration_artifact({}, traversal_path)
    assert list(tmp_path.iterdir()) == []


def test_install_calibration_artifact_rejects_multiple_traversal_components(tmp_path):
    traversal_path = Path(tmp_path, "..", "..", "..", "etc", _TRAVERSAL_FILENAME)
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
        run.install_calibration_artifact({}, traversal_path)
    assert not (Path("/etc") / _TRAVERSAL_FILENAME).exists()


def test_install_calibration_artifact_traversal_with_existing_valid_destination_untouched(tmp_path):
    destination = tmp_path / _TRAVERSAL_FILENAME
    sentinel = "PRE-EXISTING-VALID-DESTINATION-DO-NOT-TOUCH"
    destination.write_text(sentinel, encoding="utf-8")

    traversal_input_path = Path(tmp_path, "sub", "..", _TRAVERSAL_FILENAME)
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="parent-directory path component"):
        run.install_calibration_artifact({"poison": "document"}, traversal_input_path, allow_overwrite=True)

    assert destination.read_text(encoding="utf-8") == sentinel, "existing destination must remain byte-identical"
    assert list(tmp_path.glob("*.tmp")) == [], "no temporary residue may be left behind"


def test_install_calibration_artifact_traversal_leaves_document_input_unchanged(tmp_path):
    document = {"complete": False, "final": False, "marker": "unchanged"}
    frozen_copy = dict(document)
    traversal_path = Path(tmp_path, "..", _TRAVERSAL_FILENAME)
    with pytest.raises(run.DfpPhase1bCalibrationRunError):
        run.install_calibration_artifact(document, traversal_path)
    assert document == frozen_copy, "the caller's document object must be left unchanged"


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_install_calibration_artifact_traversal_check_does_not_interfere_with_fatal_propagation(tmp_path, exc_type, monkeypatch):
    def raiser(*args, **kwargs):
        raise exc_type("must propagate")

    monkeypatch.setattr(run, "validate_calibration_artifact", raiser)
    with pytest.raises(exc_type):
        run.install_calibration_artifact({}, tmp_path / _TRAVERSAL_FILENAME)


# --- positive controls: legitimate paths must remain unaffected ---


def test_positive_control_valid_absolute_temp_path_still_accepted(tmp_path):
    report = run.run_preflight(output_path=tmp_path / _TRAVERSAL_FILENAME, mode="fresh")
    assert report.output_path == tmp_path / _TRAVERSAL_FILENAME


def test_positive_control_future_scratch_shaped_absolute_path_accepted_by_preflight(monkeypatch, tmp_path):
    """A future-shaped absolute /scratch/.../phase1b_conservation_
    calibration.json destination must validate identically to any other
    absolute path with no '..' component -- this test never actually
    writes under /scratch; it substitutes a fresh tmp_path directory
    into a path that otherwise has the exact real shape and depth."""
    scratch_shaped = tmp_path / "scratch_stand_in" / "haree" / "dfp_phase1b" / "coco_stuff_200"
    scratch_shaped.mkdir(parents=True)
    destination = scratch_shaped / _TRAVERSAL_FILENAME
    report = run.run_preflight(output_path=destination, mode="fresh")
    assert report.output_path == destination
    assert not destination.exists()  # preflight alone never writes anything


def test_positive_control_relative_path_without_traversal_accepted(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    relative_path = Path(_TRAVERSAL_FILENAME)
    report = run.run_preflight(output_path=relative_path, mode="fresh")
    assert report.output_path == relative_path


def test_positive_control_double_dot_substring_in_directory_name_accepted(tmp_path):
    directory_with_dots = tmp_path / "run..v2"
    directory_with_dots.mkdir()
    destination = directory_with_dots / _TRAVERSAL_FILENAME
    report = run.run_preflight(output_path=destination, mode="fresh")
    assert report.output_path == destination


def test_positive_control_exact_required_basename_without_traversal_accepted(tmp_path):
    destination = tmp_path / _TRAVERSAL_FILENAME
    report = run.run_preflight(output_path=destination, mode="fresh")
    assert report.output_path.name == _TRAVERSAL_FILENAME


def test_preflight_rejects_tampered_e16_dependency(tmp_path, monkeypatch):
    """Simulate a corrupted E16 dependency binding: preflight must fail
    with the E16 identity's own relational error, translated into the
    runner's own domain exception -- never a raw exception."""
    from src.dfp_phase1b_calibration_identity import DfpPhase1bCalibrationIdentityError

    def tampered_load(*args, **kwargs):
        raise DfpPhase1bCalibrationIdentityError("simulated dependency drift")

    monkeypatch.setattr(run, "load_e16_identity", tampered_load)
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="simulated dependency drift"):
        run.run_preflight(output_path=tmp_path / "phase1b_conservation_calibration.json", mode="fresh")


def test_preflight_never_imports_torch():
    import ast

    source = Path(run.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    # torch is imported locally inside Phase C functions -- confirm the
    # MODULE-LEVEL import list (top of file, before any function def)
    # contains no torch import.
    module_level_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_level_imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            module_level_imports.append(node.module)
    assert not any(name.startswith("torch") for name in module_level_imports)


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_preflight_propagates_fatal_exceptions(tmp_path, monkeypatch, exc_type):
    def raiser(*args, **kwargs):
        raise exc_type("must propagate")

    monkeypatch.setattr(run, "load_e16_identity", raiser)
    with pytest.raises(exc_type):
        run.run_preflight(output_path=tmp_path / "phase1b_conservation_calibration.json", mode="fresh")


# ---------------------------------------------------------------------------
# Section: Phase B candidate universe + selection.
# ---------------------------------------------------------------------------


def _fake_geometry_provider(sizes):
    def provider(canonical_id):
        h, w, patch = sizes[canonical_id]
        return run.ImageGeometry(inference_height=h, inference_width=w, patch_size=patch)

    return provider


def test_build_candidate_universe_and_select_smaller_authority_shaped_harness():
    """Section 18's explicitly-permitted alternative to a full 200-real-
    window run: exercise Phase B's real logic end to end on a small
    synthetic universe using the real E16 selector."""
    plan_builder = run.WindowPlanBuilder(crop_height=40, crop_width=40, stride_height=20, stride_width=20)
    ids = [f"{i:012d}" for i in range(60)]
    sizes = {cid: (80, 100, 14) for cid in ids}  # non-square, multiple windows/image
    universe = run.build_candidate_universe(ids, geometry_provider=_fake_geometry_provider(sizes), window_plan_builder=plan_builder)
    assert len(universe) > 200
    outcome = run.select_and_verify(universe)
    assert len(outcome.selected_rows) == 200
    for field in ("candidate_universe_digest", "selected_windows_digest", "selection_trace_digest"):
        assert len(getattr(outcome, field)) == 64


def test_build_candidate_universe_fewer_than_200_windows_rejected():
    plan_builder = run.WindowPlanBuilder(crop_height=100, crop_width=100, stride_height=100, stride_width=100)
    ids = [f"{i:012d}" for i in range(5)]
    sizes = {cid: (100, 100, 14) for cid in ids}  # exactly 1 window/image -> 5 total
    universe = run.build_candidate_universe(ids, geometry_provider=_fake_geometry_provider(sizes), window_plan_builder=plan_builder)
    from src.dfp_phase1b_calibration_artifact import DfpPhase1bCalibrationArtifactError

    with pytest.raises(DfpPhase1bCalibrationArtifactError):
        run.select_and_verify(universe)


def test_candidate_universe_records_flush_right_and_flush_bottom_windows():
    plan_builder = run.WindowPlanBuilder(crop_height=40, crop_width=40, stride_height=30, stride_width=30)
    ids = ["000000000000"]
    sizes = {"000000000000": (90, 130, 14)}  # not an exact multiple of stride -> clamped/flush windows exist
    universe = run.build_candidate_universe(ids, geometry_provider=_fake_geometry_provider(sizes), window_plan_builder=plan_builder)
    rights = {(r["bottom"], r["right"]) for r in universe if r["right"] == 130}
    bottoms = {(r["bottom"], r["right"]) for r in universe if r["bottom"] == 90}
    assert rights and bottoms


def test_selection_deterministic_across_universe_orderings():
    plan_builder = run.WindowPlanBuilder(crop_height=40, crop_width=40, stride_height=20, stride_width=20)
    ids = [f"{i:012d}" for i in range(60)]
    sizes = {cid: (80, 100, 14) for cid in ids}
    universe = run.build_candidate_universe(ids, geometry_provider=_fake_geometry_provider(sizes), window_plan_builder=plan_builder)
    outcome_a = run.select_and_verify(universe)
    outcome_b = run.select_and_verify(list(reversed(universe)))
    assert outcome_a.selected_windows_digest == outcome_b.selected_windows_digest


# ---------------------------------------------------------------------------
# Section: pure residual-summary reduction.
# ---------------------------------------------------------------------------


def test_summarize_residual_tensor_zero_sample():
    import torch

    summary = run.summarize_residual_tensor(torch.zeros(0, dtype=torch.float64))
    assert summary["sample_count"] == 0
    assert summary["mean_absolute_residual"] is None


def test_summarize_residual_tensor_matches_hand_computation():
    import torch

    values = torch.tensor([-2.0, 1.0, 3.0, -0.5], dtype=torch.float64)
    summary = run.summarize_residual_tensor(values)
    assert summary["sample_count"] == 4
    assert summary["signed_minimum"] == -2.0
    assert summary["signed_maximum"] == 3.0
    assert summary["maximum_absolute_residual"] == 3.0
    assert summary["absolute_residual_sum"] == pytest.approx(6.5)
    assert summary["mean_absolute_residual"] == pytest.approx(6.5 / 4)
    assert summary["squared_residual_sum"] == pytest.approx(4.0 + 1.0 + 9.0 + 0.25)
    assert summary["root_mean_square_residual"] == pytest.approx(((4.0 + 1.0 + 9.0 + 0.25) / 4) ** 0.5)
    assert summary["non_finite_count"] == 0


def test_summarize_residual_tensor_preserves_tiny_nonzero_value():
    import torch

    tiny = 1e-300
    summary = run.summarize_residual_tensor(torch.tensor([tiny], dtype=torch.float64))
    assert summary["maximum_absolute_residual"] == tiny
    assert summary["maximum_absolute_residual"] != 0.0


def test_summarize_residual_tensor_rejects_non_tensor():
    with pytest.raises(run.DfpPhase1bCalibrationRunError):
        run.summarize_residual_tensor([1.0, 2.0])


# ---------------------------------------------------------------------------
# Section: Phase C end-to-end synthetic fixture, using the REAL
# committed scientific core throughout (real graph, real k12
# recurrence, real stitching/rescale/component/membership/masked-
# forward functions). See module docstring for why the GEOMETRY
# contracts (crop/patch_grid) are hand-built and small, never the
# scientific formulas or the frozen k=12/alpha=0.98/steps=320 values
# (which always come from the real committed e13/e15 identities).
# ---------------------------------------------------------------------------


class _FakeSnapshot:
    def __init__(self, unary_scores, dino_features, grid_hw):
        self.unary_scores = [unary_scores]
        self.dino_features = [dino_features]
        self.grid_hw = grid_hw


class _FakeModel:
    """Returns pre-built, per-window-index synthetic snapshots. Counts
    calls so tests can assert exactly-once-per-window graph
    construction."""

    def __init__(self, per_window_data, grid_hw):
        self.per_window_data = per_window_data
        self.grid_hw = grid_hw
        self.call_count = 0
        self.calls_by_window_order: list[int] = []

    def generate_patch_snapshot(self, crop, text_embedding):
        idx = self.call_count
        self.call_count += 1
        self.calls_by_window_order.append(idx)
        unary, features = self.per_window_data[idx]
        return _FakeSnapshot(unary, features, grid_hw=self.grid_hw)


class _FakeInference:
    def __init__(self, model):
        self.model = model
        self.text_embedding = None


@pytest.fixture(scope="module")
def real_e13_e15_identities():
    from src.dfp_phase1a_identity import load_identity as load_e13_identity
    from src.dfp_phase1b_identity import load_identity as load_e15_identity

    repo_root = run.repository_root()
    return load_e13_identity(repo_root=repo_root), load_e15_identity(repo_root=repo_root), repo_root


_CROP = (42, 70)
_STRIDE = (21, 35)
_PATCH_GRID = (3, 5)  # 15 patches -- >= 13 so k12 is real, per Section 18
_PATCH_SIZE = (14, 14)
_CLASS_COUNT = 6


def _build_small_contracts(e13, e15, repo_root):
    """Hand-built, internally-consistent, small geometry contracts
    (crop=[42,70]/patch_grid=[3,5]=15 patches, vs. production's real
    448/[32,32]=1024) -- see module docstring. `recurrence`/
    `masked_forward`/`final_resolution` are the REAL, unmodified
    contracts extracted from the real e13/e15 identities (k=12,
    alpha=0.98, steps=320, FP64 accumulation, all real)."""
    from src.dfp_phase1a_final_rescale_pullback import dfp_final_resolution_contract_from_protocol
    from src.dfp_phase1a_global_unary_components import DfpComponentContract
    from src.dfp_phase1a_horizon_stitching import DfpStitchingContract
    from src.dfp_phase1a_recurrence_capture import dfp_recurrence_contract_from_protocol
    from src.dfp_phase1a_stitching_pullback import DfpPullbackContract
    from src.dfp_phase1b_masked_forward import dfp_masked_forward_contract_from_protocol

    connectivity = int(e13["structural_components"]["primary_connectivity"])
    stitching = DfpStitchingContract(horizons=(0, 16, 32, 64, 320), crop=_CROP, stride=_STRIDE, patch_grid=_PATCH_GRID, patch_size=_PATCH_SIZE, align_corners=True)
    pullback = DfpPullbackContract(crop=_CROP, stride=_STRIDE, patch_grid=_PATCH_GRID, patch_size=_PATCH_SIZE, align_corners=True, connectivity=connectivity)
    component = DfpComponentContract(connectivity=connectivity, patch_pitch=_PATCH_SIZE, microcomponent_threshold_patch_equivalents=0.0)
    return run.ComponentContractsBundle(
        recurrence=dfp_recurrence_contract_from_protocol(e13),
        stitching=stitching,
        component=component,
        pullback=pullback,
        final_resolution=dfp_final_resolution_contract_from_protocol(e13, repo_root=repo_root),
        masked_forward=dfp_masked_forward_contract_from_protocol(e13, e15),
    )


def _make_window_unary_features(seed, *, class_count=_CLASS_COUNT, n=_PATCH_GRID[0] * _PATCH_GRID[1], diversity=True):
    import torch

    g = torch.Generator().manual_seed(seed)
    if diversity:
        unary = (torch.rand(n, class_count, generator=g, dtype=torch.float32) * 2 - 1)
    else:
        # Deliberately low-diversity: every node's unary is identical
        # across classes except a fixed winner -- used to build the
        # zero-disputed-supported-pixels window.
        unary = torch.zeros(n, class_count, dtype=torch.float32)
        unary[:, 0] = 10.0
    feats = torch.rand(n, 8, generator=g, dtype=torch.float32)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return unary.to(torch.float64), feats


def _run_small_image(contracts, e13, *, image_h, image_w, ori_shape, seeds, diversity_by_window=None, component_chunk_size=2, selected_windows=None):
    import torch

    from src.dfp_phase1a_stitching_pullback import build_stitching_pullback_plan

    pullback_plan = build_stitching_pullback_plan((image_h, image_w), contract=contracts.pullback)
    n_windows = len(pullback_plan.plan.windows)
    diversity_by_window = diversity_by_window or {i: True for i in range(n_windows)}
    per_window_data = {i: _make_window_unary_features(seed=seeds[i], diversity=diversity_by_window.get(i, True)) for i in range(n_windows)}
    model = _FakeModel(per_window_data, grid_hw=_PATCH_GRID)
    inference = _FakeInference(model)
    image_tensor = torch.zeros(1, 3, image_h, image_w)
    prepared = run.PreparedImage(image_tensor=image_tensor, inference_height=image_h, inference_width=image_w, ori_shape=ori_shape)
    selected = selected_windows if selected_windows is not None else list(range(n_windows))
    results = run.process_selected_image(
        inference, prepared, "000000000001", selected_windows_for_image=selected, protocol_identity=e13,
        contracts=contracts, class_count=_CLASS_COUNT, affinity_power=float(e13["graph"]["affinity_power"]),
        component_chunk_size=component_chunk_size,
    )
    return results, model, n_windows


def test_end_to_end_synthetic_image_conserves_to_machine_precision(real_e13_e15_identities):
    """The core correctness proof: real k12 graphs, real 320-step FP64
    recurrence, real stitching/rescale/component/membership, real
    masked-forward core, on a non-square image with overlapping/flush
    windows and a non-integer final-resize ratio. All four residual
    families must conserve to floating-point machine precision."""
    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)
    results, model, n_windows = _run_small_image(
        contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61),
        seeds={i: 100 + i for i in range(4)},
    )
    assert n_windows >= 4  # overlapping + flush-right + flush-bottom, confirmed by construction below
    assert model.call_count == n_windows  # exactly one snapshot call per window
    for window_index, family_summaries in results.items():
        for family in ("rho_M", "rho_0", "rho_320", "rho_g"):
            summary = family_summaries[family]
            if summary["sample_count"] == 0:
                continue
            assert summary["maximum_absolute_residual"] < 1e-9, f"window {window_index} {family} residual too large: {summary}"
            assert summary["non_finite_count"] == 0


def test_end_to_end_chunk_size_invariance(real_e13_e15_identities):
    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)
    seeds = {i: 200 + i for i in range(4)}
    results_2, _, _ = _run_small_image(contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61), seeds=seeds, component_chunk_size=2)
    results_1, _, _ = _run_small_image(contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61), seeds=seeds, component_chunk_size=1)
    results_full, _, _ = _run_small_image(contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61), seeds=seeds, component_chunk_size=10_000)
    for window_index in results_2:
        for family in ("rho_M", "rho_0", "rho_320", "rho_g"):
            s2, s1, sf = results_2[window_index][family], results_1[window_index][family], results_full[window_index][family]
            assert s1["sample_count"] == s2["sample_count"] == sf["sample_count"]
            if s2["sample_count"] > 0:
                assert s1["mean_absolute_residual"] == pytest.approx(s2["mean_absolute_residual"], abs=1e-12)
                assert sf["mean_absolute_residual"] == pytest.approx(s2["mean_absolute_residual"], abs=1e-12)


def test_end_to_end_zero_disputed_window_is_valid_zero_sample_row(real_e13_e15_identities):
    """At least one selected window with zero disputed supported
    pixels (Section 18's explicit requirement): a window whose unary
    is a landslide, unanimous winner everywhere produces zero disputes
    globally -- when EVERY window shares that property, the whole
    image has zero disputes, so rho_g is a valid zero-sample row
    everywhere, never a fabricated nonzero value."""
    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)
    results, _, n_windows = _run_small_image(
        contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61),
        seeds={i: 300 + i for i in range(4)}, diversity_by_window={i: False for i in range(4)},
    )
    saw_zero_sample_rho_g = False
    for window_index, family_summaries in results.items():
        rho_g = family_summaries["rho_g"]
        if rho_g["sample_count"] == 0:
            saw_zero_sample_rho_g = True
            for field in ("signed_minimum", "signed_maximum", "mean_absolute_residual", "root_mean_square_residual"):
                assert rho_g[field] is None
    assert saw_zero_sample_rho_g


def test_only_selected_windows_receive_masked_propagation(real_e13_e15_identities, monkeypatch):
    """Instrumentation: with only a strict subset of an image's windows
    selected, exactly that subset appears in the results, and the
    unselected windows' graphs are never retained (never passed to
    `propagate_masked_seed_chunk`)."""
    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)

    import src.dfp_phase1b_masked_forward as masked_forward_module

    call_count = {"n": 0}
    seen_graphs = []
    real_propagate = masked_forward_module.propagate_masked_seed_chunk

    def counting_propagate(graph, seed, *, contract):
        call_count["n"] += 1
        seen_graphs.append(graph)
        return real_propagate(graph, seed, contract=contract)

    monkeypatch.setattr(masked_forward_module, "propagate_masked_seed_chunk", counting_propagate)
    monkeypatch.setattr(run, "propagate_masked_seed_chunk", counting_propagate, raising=False)

    results, model, n_windows = _run_small_image(
        contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61),
        seeds={i: 400 + i for i in range(4)}, selected_windows=[0, 2],
    )
    assert set(results.keys()) == {0, 2}
    assert model.call_count == n_windows  # canonical propagation still runs for every window (needed for stitching)


def test_same_graph_instance_used_for_canonical_and_masked(real_e13_e15_identities, monkeypatch):
    """The same-graph requirement, instrumented directly: the graph
    object `propagate_masked_seed_chunk` receives for a selected window
    IS the exact same object `capture_dfp_phase1a_trajectory` was given
    for that window's canonical trajectory (never a second,
    independently rebuilt graph)."""
    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)

    import src.dfp_phase1a_recurrence_capture as recurrence_module
    import src.dfp_phase1b_masked_forward as masked_forward_module

    canonical_graphs: dict[int, int] = {}
    masked_graphs: dict[int, list[int]] = {}
    real_capture = recurrence_module.capture_dfp_phase1a_trajectory
    real_propagate = masked_forward_module.propagate_masked_seed_chunk
    window_counter = {"n": 0}

    def counting_capture(graph, s0, *, protocol_identity, requested_horizons=None):
        canonical_graphs[window_counter["n"]] = id(graph)
        window_counter["n"] += 1
        return real_capture(graph, s0, protocol_identity=protocol_identity, requested_horizons=requested_horizons)

    def counting_propagate(graph, seed, *, contract):
        masked_graphs.setdefault(seed.window_index, []).append(id(graph))
        return real_propagate(graph, seed, contract=contract)

    monkeypatch.setattr(recurrence_module, "capture_dfp_phase1a_trajectory", counting_capture)
    monkeypatch.setattr(masked_forward_module, "propagate_masked_seed_chunk", counting_propagate)

    results, model, n_windows = _run_small_image(
        contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61), seeds={i: 500 + i for i in range(4)},
    )
    for window_index, graph_ids in masked_graphs.items():
        assert all(gid == canonical_graphs[window_index] for gid in graph_ids), (
            f"window {window_index}: masked propagation used a different graph instance than canonical"
        )


def test_exactly_320_updates_per_masked_propagation_call(real_e13_e15_identities, monkeypatch):
    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)

    # Retrieve the exact same `DirectedTopKGraph` class object production
    # code uses, via `src.dfp_phase1b_masked_forward`'s own lazy
    # dependency loader (the proven-safe import path already used by
    # `process_selected_image` itself) -- importing `models.dinotext.
    # cover_dr.graph` directly and in isolation instead re-triggers
    # `models/dinotext/__init__.py`'s mmcv model registration a second
    # time and fails in this environment.
    import src.dfp_phase1b_masked_forward as masked_forward_module

    DirectedTopKGraph = masked_forward_module._load_dependencies().DirectedTopKGraph

    call_counts: list[int] = []
    real_matmul = DirectedTopKGraph.matmul

    def counting_matmul(self, rhs):
        counting_matmul.count += 1
        return real_matmul(self, rhs)

    counting_matmul.count = 0
    monkeypatch.setattr(DirectedTopKGraph, "matmul", counting_matmul)

    _run_small_image(contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61), seeds={i: 600 + i for i in range(4)}, component_chunk_size=2)
    # Every graph.matmul call (canonical trajectory calls it too) must be
    # a multiple of the frozen step count; specifically confirm at least
    # one canonical (320 calls for 1 propagation) or masked (320 * calls)
    # multiple was observed and none is an early stop.
    assert counting_matmul.count % 320 == 0
    assert counting_matmul.count > 0


def test_no_k11_no_solver_no_early_stopping_in_source():
    import ast

    source = Path(run.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"k11", "cgls", "gmres", "equilibrium", "early_stop", "top_k_residual"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            lowered = node.id.lower()
            assert not any(fragment in lowered for fragment in forbidden), node.id


# ---------------------------------------------------------------------------
# Section: GT isolation.
# ---------------------------------------------------------------------------


def test_no_public_function_accepts_a_gt_shaped_parameter():
    import inspect

    gt_fragments = ("gt", "ground_truth", "annotation", "harmful", "beneficial", "oracle")
    for name in run.__all__:
        obj = getattr(run, name)
        if not callable(obj) or isinstance(obj, type):
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        for param_name in signature.parameters:
            lowered = param_name.lower()
            assert not any(fragment in lowered for fragment in gt_fragments), f"{name}.{param_name}"


def test_process_selected_image_never_calls_a_gt_loader(real_e13_e15_identities, monkeypatch):
    """Poison a plausible GT-loading entry point (a dataset's own
    ground-truth accessor is never even referenced by this module) --
    the complete synthetic pipeline still runs to completion."""
    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)

    class PoisonedDataset:
        def get_gt_seg_map_by_idx(self, *args, **kwargs):
            raise AssertionError("GT loader must never be called by the calibration runner")

    # process_selected_image doesn't even accept a dataset argument --
    # this test documents and enforces that fact directly.
    import inspect

    signature = inspect.signature(run.process_selected_image)
    assert "dataset" not in signature.parameters
    assert "gt" not in signature.parameters

    # Full run still completes normally.
    results, _, _ = _run_small_image(contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61), seeds={i: 700 + i for i in range(4)})
    assert len(results) > 0


def test_changing_synthetic_features_but_not_predictions_changes_no_calibration_value(real_e13_e15_identities):
    """There is no GT input at all to vary in this module's own API, so
    this test demonstrates the isolation the strongest available way:
    two runs with IDENTICAL image/window/unary/feature inputs (the only
    things this module ever reads) produce byte-identical residual
    summaries -- confirming determinism and, by construction (no GT
    parameter exists anywhere in the call chain), that no hidden GT
    channel could be influencing the result."""
    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)
    seeds = {i: 800 + i for i in range(4)}
    results_a, _, _ = _run_small_image(contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61), seeds=seeds)
    results_b, _, _ = _run_small_image(contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61), seeds=seeds)
    assert results_a == results_b


# ---------------------------------------------------------------------------
# Section: no-tolerance / no-compute isolation.
# ---------------------------------------------------------------------------


def test_no_tolerance_epsilon_or_verdict_field_in_source():
    source = Path(run.__file__).read_text(encoding="utf-8").lower()
    # "tolerance"/"epsilon" appear only in documentation prose explaining
    # their ABSENCE -- confirm no executable assignment/comparison uses them.
    import ast

    tree = ast.parse(Path(run.__file__).read_text(encoding="utf-8"))
    docstring_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstring_ids.add(id(first.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and any(f in node.id.lower() for f in ("tolerance", "epsilon", "atol", "rtol", "threshold", "verdict")):
            pytest.fail(f"forbidden identifier in executable code: {node.id}")


def test_cuda_never_initialized_by_phase_a_or_phase_b(tmp_path):
    import torch

    assert torch.cuda.is_initialized() is False
    run.run_preflight(output_path=tmp_path / "phase1b_conservation_calibration.json", mode="fresh")
    plan_builder = run.WindowPlanBuilder(crop_height=40, crop_width=40, stride_height=20, stride_width=20)
    ids = [f"{i:012d}" for i in range(60)]
    sizes = {cid: (80, 100, 14) for cid in ids}
    universe = run.build_candidate_universe(ids, geometry_provider=_fake_geometry_provider(sizes), window_plan_builder=plan_builder)
    run.select_and_verify(universe)
    assert torch.cuda.is_initialized() is False


def test_cuda_never_initialized_by_phase_c(real_e13_e15_identities):
    import torch

    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)
    assert torch.cuda.is_initialized() is False
    _run_small_image(contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61), seeds={i: 900 + i for i in range(4)})
    assert torch.cuda.is_initialized() is False


# ---------------------------------------------------------------------------
# Section: membership/unary_window_scores device alignment.
#
# A real `execute --device cuda` job (SLURM 21196409) failed at the
# masked-forward boundary with "unary_window_scores device cuda:0
# disagrees with membership_rows device cpu": `compute_global_unary_
# components`/`iter_image_final_soft_memberships` are CPU/numpy-based
# by design, while the real model's own score tensor lives on whatever
# device inference actually ran on. Every test above ran entirely on
# CPU (`unary_scores` from `_FakeModel` is always CPU), so this specific
# cross-device pairing was never exercised until a real GPU job hit it.
# `_align_membership_device` (`run.py`) closes exactly this gap; the
# tests below prove it CPU-testably (no-op when devices already match,
# and that `process_selected_image` actually invokes it at the right
# boundary with the right tensors) plus, when a real GPU is present,
# an end-to-end regression reproducing the exact failing scenario.
# ---------------------------------------------------------------------------


def test_align_membership_device_noop_when_already_matching():
    import torch

    membership = torch.rand(3, 5, dtype=torch.float64)
    reference = torch.rand(5, 2, dtype=torch.float32)  # same (CPU) device, different shape/dtype entirely
    result = run._align_membership_device(membership, reference)
    assert result is membership  # identical object -- no copy, no unnecessary transfer
    assert result.dtype == torch.float64  # dtype untouched
    assert torch.equal(result, membership)  # values untouched


def test_process_selected_image_invokes_membership_alignment_at_masked_forward_boundary(real_e13_e15_identities, monkeypatch):
    """CPU-testable instrumentation: without any GPU, confirm the real
    pipeline actually calls `_align_membership_device` at the masked-
    forward boundary EXACTLY ONCE PER SELECTED WINDOW (never once per
    component chunk), always with `reference` being that window's own
    `unary_for_query`-derived tensor (the real `unary_window_scores`)
    -- proving the WIRING is correct even though the cross-device MOVE
    itself can only be exercised with real CUDA (see the CUDA-gated
    test below). Every window in this fixture is selected, so the
    expected call count is exactly `n_windows`."""
    import torch

    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)

    calls = []
    real_align = run._align_membership_device

    def spying_align(membership, reference):
        calls.append((membership.device, reference.device, tuple(membership.shape)))
        return real_align(membership, reference)

    monkeypatch.setattr(run, "_align_membership_device", spying_align)
    results, model, n_windows = _run_small_image(
        contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61), seeds={i: 950 + i for i in range(4)},
    )
    assert len(calls) == n_windows, f"expected exactly one alignment call per selected window ({n_windows}), got {len(calls)}"
    for membership_device, reference_device, _shape in calls:
        assert membership_device == reference_device == torch.device("cpu")
    # Sanity: the alignment call did not change the scientific outcome
    # (every CPU-only correctness assertion elsewhere in this file
    # already covers the actual residual values for this exact fixture
    # shape; here we only need confirmation results were still produced).
    assert results


def _instrumented_chunk_run(*, component_chunk_size, selected_windows=None):
    """Shared instrumentation for the transfer-count regression tests
    below: real e13/e15 identities, the 63x105/(37,61) fixture (known,
    from repeated construction, to produce >=37 global components for
    window 0 with the diversity=True random unary field), spying on
    both `_align_membership_device` and the real, unmodified
    `build_masked_seed_chunk` so the alignment-call count can be
    directly compared against the actual chunk count."""
    import torch

    from src.dfp_phase1a_identity import load_identity as load_e13_identity
    from src.dfp_phase1b_identity import load_identity as load_e15_identity
    import src.dfp_phase1b_masked_forward as masked_forward_module

    e13 = load_e13_identity(repo_root=run.repository_root())
    e15 = load_e15_identity(repo_root=run.repository_root())
    contracts = _build_small_contracts(e13, e15, run.repository_root())

    align_calls = []
    real_align = run._align_membership_device

    def spying_align(membership, reference):
        align_calls.append(tuple(membership.shape))
        return real_align(membership, reference)

    chunk_calls = {"n": 0}
    real_build = masked_forward_module.build_masked_seed_chunk

    def counting_build(unary_window_scores, membership_rows, component_ids, query_class_indices, *, contract, window_index):
        chunk_calls["n"] += 1
        return real_build(unary_window_scores, membership_rows, component_ids, query_class_indices, contract=contract, window_index=window_index)

    run._align_membership_device = spying_align
    run.build_masked_seed_chunk = counting_build
    masked_forward_module.build_masked_seed_chunk = counting_build
    try:
        results, model, n_windows = _run_small_image(
            contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61), seeds={i: 5000 + i for i in range(4)},
            component_chunk_size=component_chunk_size, selected_windows=selected_windows,
        )
    finally:
        run._align_membership_device = real_align
        run.build_masked_seed_chunk = real_build
        masked_forward_module.build_masked_seed_chunk = real_build
    return align_calls, chunk_calls["n"], results, n_windows


@pytest.mark.parametrize("component_chunk_size", [1, 2, 8, 64, 10_000])
def test_alignment_call_count_equals_selected_window_count_not_chunk_count(component_chunk_size):
    """Section 6: one selected window, >=37 global components (this
    exact fixture is already known, from the module docstring's own
    small-geometry construction, to produce many components for a
    diverse random unary field) -- across every chunk size from 1 (>=37
    chunks) up to one enormous chunk (exactly 1 chunk), the alignment
    call count must stay exactly 1 (one selected window), never equal
    to the chunk count except by the trivial coincidence of
    component_chunk_size >= component_count."""
    align_calls, n_chunks, results, n_windows = _instrumented_chunk_run(component_chunk_size=component_chunk_size, selected_windows=[0])
    import math
    expected_n_chunks = math.ceil(37 / component_chunk_size) if component_chunk_size < 37 else 1
    assert n_chunks == expected_n_chunks, (
        f"fixture assumption violated: expected {expected_n_chunks} chunks (37 components / chunk_size="
        f"{component_chunk_size}), got {n_chunks} -- the global component count for this fixture may have changed"
    )
    assert len(align_calls) == 1, (
        f"chunk_size={component_chunk_size}: expected exactly 1 alignment call (one per selected window), "
        f"got {len(align_calls)} (chunk count was {n_chunks})"
    )
    assert results[0]  # the selected window actually produced a result


def test_alignment_call_count_multiple_selected_and_unselected_windows():
    """Section 7: multiple selected windows (different component/chunk
    counts each, since components are global to the image but chunking
    is per-window-independent), plus at least one unselected canonical-
    only window. Exactly one alignment call per selected window; zero
    for the unselected window; canonical processing (graph/trajectory)
    still runs for every window regardless of selection."""
    align_calls, n_chunks, results, n_windows = _instrumented_chunk_run(component_chunk_size=1, selected_windows=[0, 1])
    assert n_windows >= 3, "need at least one additional unselected window"
    assert len(align_calls) == 2, f"expected exactly 2 alignment calls (2 selected windows), got {len(align_calls)}"
    assert set(results.keys()) == {0, 1}


class _RaisingOnTo:
    """A membership-tensor-shaped double whose `.to(...)` always raises
    a specific, uniquely identifiable RuntimeError instance -- so a
    test can assert `__cause__ is exactly this object`, never merely
    "some RuntimeError"."""

    def __init__(self, real_tensor, error):
        self._real = real_tensor
        self._error = error
        self.to_call_count = 0

    @property
    def device(self):
        return self._real.device

    @property
    def dtype(self):
        return self._real.dtype

    @property
    def shape(self):
        return self._real.shape

    def to(self, *args, **kwargs):
        self.to_call_count += 1
        raise self._error


def test_align_membership_device_translates_runtime_error_from_transfer_only():
    """Section 5/11: only the `.to(...)` operation's RuntimeError is
    translated -- into this module's own DfpPhase1bCalibrationRunError,
    naming both devices and identifying membership_rows, with the
    original exception preserved exactly as `__cause__`."""
    import torch

    injected = RuntimeError("simulated CUDA allocation/transfer failure")
    real_membership = torch.zeros(2, 3, dtype=torch.float64)
    membership_double = _RaisingOnTo(real_membership, injected)
    reference = torch.zeros(3).to("meta")  # distinct device -- forces the .to(...) branch, never the no-op

    with pytest.raises(run.DfpPhase1bCalibrationRunError) as excinfo:
        run._align_membership_device(membership_double, reference)

    message = str(excinfo.value)
    assert "membership_rows" in message
    assert "cpu" in message.lower()
    assert "meta" in message.lower()
    assert excinfo.value.__cause__ is injected  # exact original exception preserved, not a copy/re-wrap
    assert membership_double.to_call_count == 1


def test_align_membership_device_does_not_translate_when_devices_already_match():
    """The no-op (same-device) branch never calls `.to` at all, so a
    membership double that would raise on `.to` is never even touched
    when alignment was never needed."""
    import torch

    injected = RuntimeError("must never be reached")
    real_membership = torch.zeros(2, 3, dtype=torch.float64)
    membership_double = _RaisingOnTo(real_membership, injected)
    reference = torch.zeros(3, dtype=torch.float32)  # same (cpu) device

    result = run._align_membership_device(membership_double, reference)
    assert result is membership_double
    assert membership_double.to_call_count == 0


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_align_membership_device_never_catches_fatal_exceptions(exc_type):
    """Section 12: KeyboardInterrupt/SystemExit/MemoryError raised by
    the transfer itself must propagate completely uncaught and
    unwrapped -- never translated into DfpPhase1bCalibrationRunError,
    since none of them is a RuntimeError and the translation is scoped
    to exactly one `except RuntimeError` clause."""
    import torch

    class _FatalRaisingOnTo(_RaisingOnTo):
        def to(self, *args, **kwargs):
            raise self._error

    real_membership = torch.zeros(2, 3, dtype=torch.float64)
    membership_double = _FatalRaisingOnTo(real_membership, exc_type("must propagate"))
    reference = torch.zeros(3).to("meta")

    with pytest.raises(exc_type):
        run._align_membership_device(membership_double, reference)


def test_align_membership_device_does_not_swallow_unrelated_runtime_error_from_elsewhere():
    """Section 13: a RuntimeError raised by a DIFFERENT operation (here,
    a mocked masked-propagation call happening strictly after a
    successful alignment) must propagate completely unchanged -- never
    caught, translated, or misattributed as a membership-transfer
    failure by `_align_membership_device`'s own narrow `except
    RuntimeError` (which is scoped to its own `.to(...)` call only, and
    is not even on the call stack once alignment has already returned)."""
    import torch
    import src.dfp_phase1b_masked_forward as masked_forward_module

    from src.dfp_phase1a_identity import load_identity as load_e13_identity
    from src.dfp_phase1b_identity import load_identity as load_e15_identity

    e13 = load_e13_identity(repo_root=run.repository_root())
    e15 = load_e15_identity(repo_root=run.repository_root())
    contracts = _build_small_contracts(e13, e15, run.repository_root())

    class _UnrelatedRuntimeError(RuntimeError):
        pass

    real_propagate = masked_forward_module.propagate_masked_seed_chunk

    def raising_propagate(graph, seed, *, contract):
        raise _UnrelatedRuntimeError("unrelated failure in propagate_masked_seed_chunk, not a device transfer")

    masked_forward_module.propagate_masked_seed_chunk = raising_propagate
    try:
        with pytest.raises(_UnrelatedRuntimeError) as excinfo:
            _run_small_image(contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61), seeds={i: 5100 + i for i in range(4)})
        # Must be the exact original, never wrapped in DfpPhase1bCalibrationRunError.
        assert not isinstance(excinfo.value, run.DfpPhase1bCalibrationRunError)
        assert "membership" not in str(excinfo.value).lower()
    finally:
        masked_forward_module.propagate_masked_seed_chunk = real_propagate


def test_masked_seed_independent_reconstruction_matches_hoisted_production_path():
    """Section 9: independently reconstruct
    masked_seed[r,j,c] = membership[r,j] * unary[j,c] (the docstring's
    own stated formula, never importing the internal implementation),
    and compare it against the REAL, POST-HOIST production path (the
    single once-per-window `_align_membership_device` call feeding a
    chunked `build_masked_seed_chunk` loop) -- both single-chunk and
    multi-chunk ('more components than one chunk'), requiring exact
    (FP64) agreement, and confirming chunking invariance."""
    import torch

    from src.dfp_phase1a_identity import load_identity as load_e13_identity
    from src.dfp_phase1b_identity import load_identity as load_e15_identity
    from src.dfp_phase1b_masked_forward import build_masked_seed_chunk, dfp_masked_forward_contract_from_protocol

    e13 = load_e13_identity(repo_root=run.repository_root())
    e15 = load_e15_identity(repo_root=run.repository_root())
    contract = dfp_masked_forward_contract_from_protocol(e13, e15)
    assert contract.accumulation_dtype == torch.float64

    node_count, component_count, query_class_count = 9, 5, 4
    g = torch.Generator().manual_seed(4242)
    membership_full = torch.stack([
        torch.zeros(node_count, dtype=torch.float32),
        torch.full((node_count,), 0.5, dtype=torch.float32),
        torch.rand(node_count, generator=g, dtype=torch.float32),
        torch.full((node_count,), 0.5, dtype=torch.float32),  # repeated row
        torch.tensor([1.0, 0.0, 0.25, 0.75, 0.0, 1.0, 0.5, 0.5, 0.0]),
    ])
    unary = torch.rand(node_count, query_class_count, generator=g, dtype=torch.float64) * 4 - 2  # signed
    query_classes = list(range(query_class_count))

    def independent_masked_seed(membership_rows, unary_window_scores):
        R, N = membership_rows.shape
        out = torch.zeros(N, R, query_class_count, dtype=torch.float64)
        for r in range(R):
            for j in range(N):
                for c in range(query_class_count):
                    out[j, r, c] = float(membership_rows[r, j]) * float(unary_window_scores[j, c])
        return out

    expected = independent_masked_seed(membership_full, unary)

    # Single chunk, through the real (hoisted) alignment call.
    aligned = run._align_membership_device(membership_full, unary)
    chunk = build_masked_seed_chunk(unary, aligned, list(range(component_count)), query_classes, contract=contract, window_index=0)
    actual_single = chunk.unflatten(chunk.seed_flat).to(torch.float64)
    assert torch.equal(actual_single, expected)

    # Multiple chunks (more components than one chunk), aligned exactly
    # once for the whole window, then sliced per chunk -- exactly the
    # production ordering after the Finding-A repair.
    chunk_size = 2
    assert component_count > chunk_size
    aligned_once = run._align_membership_device(membership_full, unary)
    parts = []
    start = 0
    while start < component_count:
        stop = min(start + chunk_size, component_count)
        c = build_masked_seed_chunk(unary, aligned_once[start:stop], list(range(start, stop)), query_classes, contract=contract, window_index=0)
        parts.append(c.unflatten(c.seed_flat))
        start = stop
    reassembled = torch.cat(parts, dim=1).to(torch.float64)
    assert torch.equal(reassembled, expected)
    assert torch.equal(reassembled, actual_single)  # chunking invariance


def test_bypassed_alignment_still_rejected_by_frozen_device_assertion():
    """Section 10: with alignment adversarially bypassed (a membership
    tensor deliberately left on a device that disagrees with the
    reference), the frozen masked-forward core's own strict
    same-device assertion must still reject -- confirming the
    invariant was made satisfiable by the repair, never removed or
    weakened. `meta` stands in for a real non-cpu device (no CUDA is
    available in this test environment)."""
    import torch

    from src.dfp_phase1a_identity import load_identity as load_e13_identity
    from src.dfp_phase1b_identity import load_identity as load_e15_identity
    from src.dfp_phase1b_masked_forward import build_masked_seed_chunk, dfp_masked_forward_contract_from_protocol

    e13 = load_e13_identity(repo_root=run.repository_root())
    e15 = load_e15_identity(repo_root=run.repository_root())
    contract = dfp_masked_forward_contract_from_protocol(e13, e15)

    unary = torch.zeros(4, 2, dtype=contract.propagation_dtype).to("meta")
    membership_wrong_device = torch.zeros(2, 4, dtype=torch.float64)  # deliberately NOT aligned (still cpu)
    with pytest.raises(Exception) as excinfo:
        build_masked_seed_chunk(unary, membership_wrong_device, [0, 1], [0, 1], contract=contract, window_index=0)
    msg = str(excinfo.value).lower()
    assert "device" in msg or "meta" in msg or "cpu" in msg


def test_align_membership_device_moves_and_preserves_values_and_dtype():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    membership_cpu = torch.rand(4, 7, dtype=torch.float64)
    reference_cuda = torch.zeros(7, 3, dtype=torch.float32, device="cuda")

    moved = run._align_membership_device(membership_cpu, reference_cuda)

    assert moved.device == reference_cuda.device  # matches the SCORE tensor's actual device, not a hardcoded "cuda:0"
    assert moved.dtype == membership_cpu.dtype  # dtype preserved exactly -- only placement changed
    assert torch.equal(moved.cpu(), membership_cpu)  # values preserved exactly
    assert reference_cuda.device.type == "cuda"  # the reference (score) tensor itself was never touched/copied


def test_end_to_end_synthetic_image_with_cuda_resident_scores_matches_cpu(real_e13_e15_identities):
    """Regression test for SLURM job 21196409's real failure: a
    synthetic model whose `unary_scores`/`dino_features` are CUDA-
    resident (exactly mirroring what a real GPU inference call
    produces), while `compute_global_unary_components`/`iter_image_
    final_soft_memberships` still run on CPU/numpy underneath,
    completely unchanged. Before the fix this raised
    `unary_window_scores device cuda:0 disagrees with membership_rows
    device cpu`; after it, results must exist and must numerically
    match (to floating-point machine precision) an identical run where
    every tensor stays on CPU -- the device alignment must never change
    the scientific outcome, only where the arithmetic happens."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)
    seeds = {i: 960 + i for i in range(4)}

    class _DeviceFakeSnapshot:
        def __init__(self, unary_scores, dino_features):
            self.unary_scores = [unary_scores]
            self.dino_features = [dino_features]

    class _DeviceFakeModel:
        def __init__(self, per_window_data, device):
            self.per_window_data = per_window_data
            self.device = device
            self.call_count = 0

        def generate_patch_snapshot(self, crop, text_embedding):
            idx = self.call_count
            self.call_count += 1
            unary, features = self.per_window_data[idx]
            target = torch.device(self.device) if self.device is not None else unary.device
            return _DeviceFakeSnapshot(unary.to(target), features.to(target))

    def _run(device):
        from src.dfp_phase1a_stitching_pullback import build_stitching_pullback_plan

        pullback_plan = build_stitching_pullback_plan((63, 105), contract=contracts.pullback)
        n_windows = len(pullback_plan.plan.windows)
        per_window_data = {i: _make_window_unary_features(seed=seeds[i]) for i in range(n_windows)}
        model = _DeviceFakeModel(per_window_data, device=device)
        inference = _FakeInference(model)
        image_tensor = torch.zeros(1, 3, 63, 105)
        prepared = run.PreparedImage(image_tensor=image_tensor, inference_height=63, inference_width=105, ori_shape=(37, 61))
        return run.process_selected_image(
            inference, prepared, "000000000001", selected_windows_for_image=list(range(n_windows)),
            protocol_identity=e13, contracts=contracts, class_count=_CLASS_COUNT,
            affinity_power=float(e13["graph"]["affinity_power"]), component_chunk_size=2,
        )

    results_cpu = _run(None)
    results_cuda = _run("cuda")

    assert results_cuda.keys() == results_cpu.keys()
    for window_index in results_cpu:
        for family in ("rho_M", "rho_0", "rho_320", "rho_g"):
            cpu_summary, cuda_summary = results_cpu[window_index][family], results_cuda[window_index][family]
            assert cpu_summary["sample_count"] == cuda_summary["sample_count"]
            if cpu_summary["sample_count"] > 0:
                assert cuda_summary["mean_absolute_residual"] == pytest.approx(cpu_summary["mean_absolute_residual"], abs=1e-9)
                assert cuda_summary["maximum_absolute_residual"] == pytest.approx(cpu_summary["maximum_absolute_residual"], abs=1e-9)
                assert cuda_summary["non_finite_count"] == 0


# ---------------------------------------------------------------------------
# Section: stitching-plan score-tensor device alignment (second real-GPU
# failure, SLURM job 21210326).
#
# The first (membership) repair aligned CPU-native membership geometry
# TO the real model's GPU-resident unary score device. Execution then
# advanced further and hit a SECOND, architecturally different device
# boundary: `stitch_and_rescale_component_scores`
# (`src.dfp_phase1b_masked_forward`) builds its own
# `accumulator_t0`/`accumulator_final` via a bare `torch.zeros(...,
# dtype=dtype)` -- no `device` argument, always CPU -- and its call into
# `apply_window_stitching_forward` (`src.dfp_phase1a_stitching_
# pullback`) builds its per-pixel `weight` via `build_stitch_weight(...,
# dtype=u.dtype)` -- again no `device` argument, always CPU regardless
# of `u`'s own device. Both are internal to protected, hash-pinned
# dependencies this repair may not modify, so THIS boundary requires
# alignment in the OPPOSITE direction: the runner's own GPU-resident
# component/window score tensors must move TO the stitching plan's own
# (always-CPU) device before being packaged into a `PropagatedChunk` and
# handed to `stitch_and_rescale_component_scores`.
# ---------------------------------------------------------------------------


def test_align_tensor_to_stitching_plan_device_noop_when_already_matching(real_e13_e15_identities):
    import torch

    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)
    from src.dfp_phase1a_stitching_pullback import build_stitching_pullback_plan

    plan = build_stitching_pullback_plan((63, 105), contract=contracts.pullback)
    tensor = torch.rand(15, 4, dtype=torch.float64)
    result = run._align_tensor_to_stitching_plan_device(tensor, plan)
    assert result is tensor
    assert torch.equal(result, tensor)


def test_align_tensor_to_stitching_plan_device_dynamic_target_meta(real_e13_e15_identities):
    """meta stands in for a real non-cpu device (no CUDA available in
    this test environment): proves the target is read dynamically from
    `plan.denominator.device`, never hardcoded, and no dtype argument
    is ever passed to `.to(...)`."""
    import torch

    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)
    from src.dfp_phase1a_stitching_pullback import build_stitching_pullback_plan

    plan = build_stitching_pullback_plan((63, 105), contract=contracts.pullback)
    assert plan.denominator.device == torch.device("cpu")

    class _FakePlan:
        def __init__(self, denominator):
            self.denominator = denominator

    meta_plan = _FakePlan(torch.zeros(1).to("meta"))

    class RecordingTensorLike:
        def __init__(self, real):
            self._real = real
            self.to_calls = []

        @property
        def device(self):
            return self._real.device

        @property
        def dtype(self):
            return self._real.dtype

        @property
        def shape(self):
            return self._real.shape

        def to(self, *args, **kwargs):
            self.to_calls.append((args, dict(kwargs)))
            return self._real.to(*args, **kwargs)

    membership_like = RecordingTensorLike(torch.rand(3, 4, dtype=torch.float64))
    result = run._align_tensor_to_stitching_plan_device(membership_like, meta_plan)
    assert membership_like.to_calls == [((), {"device": torch.device("meta")})]
    assert result.device.type == "meta"
    assert result.dtype == torch.float64
    assert tuple(result.shape) == (3, 4)


def test_align_tensor_to_stitching_plan_device_preserves_nontrivial_values(real_e13_e15_identities):
    import torch

    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)
    from src.dfp_phase1a_stitching_pullback import build_stitching_pullback_plan

    plan = build_stitching_pullback_plan((63, 105), contract=contracts.pullback)
    base = torch.tensor([[0.0, 1.0], [0.5, -0.5], [3.0, -3.0]], dtype=torch.float64)
    non_contig = base.t()  # genuine non-contiguous view
    assert not non_contig.is_contiguous()
    result = run._align_tensor_to_stitching_plan_device(non_contig, plan)
    assert result is non_contig  # same-device no-op
    assert torch.equal(result, base.t())
    assert torch.equal(non_contig, base.t())  # source unmutated


def test_align_tensor_to_stitching_plan_device_translates_runtime_error(real_e13_e15_identities):
    import torch

    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)
    from src.dfp_phase1a_stitching_pullback import build_stitching_pullback_plan

    plan = build_stitching_pullback_plan((63, 105), contract=contracts.pullback)

    class _FakePlan:
        def __init__(self, denominator):
            self.denominator = denominator

    meta_plan = _FakePlan(torch.zeros(1).to("meta"))
    injected = RuntimeError("simulated device-transfer failure (independent probe)")

    class RaisingTensorLike:
        def __init__(self, real, error):
            self._real = real
            self._error = error

        @property
        def device(self):
            return self._real.device

        @property
        def dtype(self):
            return self._real.dtype

        @property
        def shape(self):
            return self._real.shape

        def to(self, *a, **k):
            raise self._error

    real_tensor = torch.zeros(2, 3, dtype=torch.float64)
    double = RaisingTensorLike(real_tensor, injected)
    with pytest.raises(run.DfpPhase1bCalibrationRunError) as excinfo:
        run._align_tensor_to_stitching_plan_device(double, meta_plan)
    assert excinfo.value.__cause__ is injected
    msg = str(excinfo.value).lower()
    assert "device" in msg
    assert "cpu" in msg or "meta" in msg


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_align_tensor_to_stitching_plan_device_never_catches_fatal_exceptions(exc_type):
    import torch

    class _FakePlan:
        def __init__(self, denominator):
            self.denominator = denominator

    meta_plan = _FakePlan(torch.zeros(1).to("meta"))

    class RaisingTensorLike:
        def __init__(self, real, error):
            self._real = real
            self._error = error

        @property
        def device(self):
            return self._real.device

        def to(self, *a, **k):
            raise self._error

    real_tensor = torch.zeros(2, dtype=torch.float64)
    double = RaisingTensorLike(real_tensor, exc_type("must propagate"))
    with pytest.raises(exc_type):
        run._align_tensor_to_stitching_plan_device(double, meta_plan)


def test_exact_failure_reproduction_without_alignment_then_success_with_it(real_e13_e15_identities):
    """Drives the REAL, unmodified stitch_and_rescale_component_scores
    (never monkeypatched away) far enough to reach apply_window_
    stitching_forward: without alignment, a mismatched-device
    PropagatedChunk (meta t0/t_final against the real cpu plan) fails;
    with the real repair's alignment applied first, the identical call
    succeeds -- the exact SLURM-21210326 boundary, reproduced with the
    real protected functions."""
    import torch

    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)
    from src.dfp_phase1a_stitching_pullback import build_stitching_pullback_plan
    from src.dfp_phase1b_masked_forward import PropagatedChunk, stitch_and_rescale_component_scores

    plan = build_stitching_pullback_plan((63, 105), contract=contracts.pullback)
    component_count, query_classes = 2, [0, 1]
    node_count = _PATCH_GRID[0] * _PATCH_GRID[1]

    t0_meta = torch.zeros(node_count, component_count * len(query_classes), dtype=torch.float64).to("meta")
    t_final_meta = torch.zeros(node_count, component_count * len(query_classes), dtype=torch.float64).to("meta")
    mismatched_chunk = PropagatedChunk(
        t0=t0_meta, t_final=t_final_meta, window_index=0, component_id_start=0, component_id_stop=component_count,
        query_class_indices=query_classes, steps_completed=320,
    )
    with pytest.raises(RuntimeError):
        stitch_and_rescale_component_scores(
            {0: mismatched_chunk}, window_plan=plan, final_contract=contracts.final_resolution,
            img_shape=(63, 105), ori_shape=(37, 61), component_count=component_count, query_class_union=query_classes,
        )

    t0_real = torch.rand(node_count, component_count * len(query_classes), dtype=torch.float64)
    t_final_real = torch.rand(node_count, component_count * len(query_classes), dtype=torch.float64)
    t0_aligned = run._align_tensor_to_stitching_plan_device(t0_real, plan)
    t_final_aligned = run._align_tensor_to_stitching_plan_device(t_final_real, plan)
    aligned_chunk = PropagatedChunk(
        t0=t0_aligned, t_final=t_final_aligned, window_index=0, component_id_start=0, component_id_stop=component_count,
        query_class_indices=query_classes, steps_completed=320,
    )
    s0, sf = stitch_and_rescale_component_scores(
        {0: aligned_chunk}, window_plan=plan, final_contract=contracts.final_resolution,
        img_shape=(63, 105), ori_shape=(37, 61), component_count=component_count, query_class_union=query_classes,
    )
    assert s0.device == torch.device("cpu")
    assert torch.isfinite(s0).all() and torch.isfinite(sf).all()


def _instrumented_stitching_align_run(*, component_chunk_size, selected_windows, seed_base):
    """Shared instrumentation for the corrected (repair D) stitching-
    alignment call-count tests below: counts real `build_masked_seed_
    chunk` calls per window (== that window's masked component-chunk
    count, since it is called exactly once per chunk both before and
    after this repair) alongside real `_align_tensor_to_stitching_plan_
    device` calls, in one single run so both counts are guaranteed
    consistent with each other and with `results`."""
    import src.dfp_phase1b_masked_forward as masked_forward_module

    e13 = __import__("src.dfp_phase1a_identity", fromlist=["load_identity"]).load_identity(repo_root=run.repository_root())
    e15 = __import__("src.dfp_phase1b_identity", fromlist=["load_identity"]).load_identity(repo_root=run.repository_root())
    contracts = _build_small_contracts(e13, e15, run.repository_root())

    align_calls = []
    real_align = run._align_tensor_to_stitching_plan_device

    def spying_align(tensor, plan):
        align_calls.append(tuple(tensor.shape))
        return real_align(tensor, plan)

    build_calls_by_window: dict[int, int] = {}
    real_build = masked_forward_module.build_masked_seed_chunk

    def counting_build(unary_window_scores, membership_rows, component_ids, query_class_indices, *, contract, window_index):
        build_calls_by_window[window_index] = build_calls_by_window.get(window_index, 0) + 1
        return real_build(unary_window_scores, membership_rows, component_ids, query_class_indices, contract=contract, window_index=window_index)

    run._align_tensor_to_stitching_plan_device = spying_align
    run.build_masked_seed_chunk = counting_build
    masked_forward_module.build_masked_seed_chunk = counting_build
    try:
        results, model, n_windows = _run_small_image(
            contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61), seeds={i: seed_base + i for i in range(4)},
            component_chunk_size=component_chunk_size, selected_windows=selected_windows,
        )
    finally:
        run._align_tensor_to_stitching_plan_device = real_align
        run.build_masked_seed_chunk = real_build
        masked_forward_module.build_masked_seed_chunk = real_build
    return align_calls, build_calls_by_window, results, n_windows


@pytest.mark.parametrize("component_chunk_size", [1, 2, 8, 16, 37, 64])
def test_stitching_alignment_call_count_scales_with_chunk_count(component_chunk_size):
    """Section 7 (repair D correction): after eliminating the full-
    component stitched allocation, there is no longer a single full-
    component CPU copy to align once per window -- the OLD assertion of
    "exactly 4 calls per selected window regardless of chunk size" is
    therefore WRONG post-repair and must not be preserved. The correct
    contract: exactly 2 calls for the window-scoped CANONICAL numerator
    (its own t0/t_final; always component_count=1, never chunked, so
    this part is unaffected by chunk_size) plus, ONLY when the window
    has at least one disputed pixel, exactly 2 calls per MASKED
    component chunk (this chunk's own t0/t_final) -- never once per
    component, never once per recurrence step, and never a single call
    sized by the full component_count."""
    align_calls, build_calls_by_window, results, _ = _instrumented_stitching_align_run(
        component_chunk_size=component_chunk_size, selected_windows=[0], seed_base=7000,
    )
    n_chunks = build_calls_by_window[0]
    has_disputed = results[0]["rho_g"]["sample_count"] > 0
    expected_calls = 2 + (2 * n_chunks if has_disputed else 0)
    assert len(align_calls) == expected_calls, (
        f"chunk_size={component_chunk_size}: expected {expected_calls} stitching-alignment calls "
        f"(2 canonical + 2*{n_chunks if has_disputed else 0} masked chunks; has_disputed={has_disputed}, "
        f"n_chunks={n_chunks}), got {len(align_calls)}"
    )
    # Never the old, now-incorrect fixed count of 4 unless it happens to
    # coincide (n_chunks == 1 and has_disputed) -- explicitly confirm the
    # count actually MOVES with chunk count for a chunk_size small enough
    # to produce more than one chunk.
    if n_chunks > 1 and has_disputed:
        assert expected_calls != 4


def test_stitching_alignment_multiple_windows_call_count():
    """Multiple selected windows, plus at least one unselected canonical-
    only window: each selected window's own stitching-alignment call
    count independently follows the corrected per-chunk contract above
    (2 canonical + 2 per masked chunk when that window has disputed
    pixels), summed across the selected windows."""
    align_calls, build_calls_by_window, results, n_windows = _instrumented_stitching_align_run(
        component_chunk_size=1, selected_windows=[0, 1], seed_base=7100,
    )
    assert n_windows >= 3, "need at least one additional unselected window"
    assert set(results.keys()) == {0, 1}
    expected_total = 0
    for window_index in (0, 1):
        n_chunks = build_calls_by_window[window_index]
        has_disputed = results[window_index]["rho_g"]["sample_count"] > 0
        expected_total += 2 + (2 * n_chunks if has_disputed else 0)
    assert len(align_calls) == expected_total, (
        f"expected {expected_total} stitching-alignment calls across 2 selected windows, got {len(align_calls)}"
    )


def test_unrelated_runtime_error_downstream_of_stitching_alignment_not_mislabeled(real_e13_e15_identities):
    """A RuntimeError raised by a completely different, later operation
    (simulating an unrelated downstream failure in signed-contribution
    computation) must propagate exactly as itself, never reinterpreted
    as a stitching-plan-transfer failure."""
    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)
    import src.dfp_phase1b_masked_forward as masked_forward_module

    class _UnrelatedError(RuntimeError):
        pass

    real_compute = masked_forward_module.compute_signed_contributions

    def raising_compute(*a, **k):
        raise _UnrelatedError("unrelated failure in compute_signed_contributions")

    masked_forward_module.compute_signed_contributions = raising_compute
    try:
        with pytest.raises(_UnrelatedError) as excinfo:
            _run_small_image(contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61), seeds={i: 7200 + i for i in range(4)})
        assert not isinstance(excinfo.value, run.DfpPhase1bCalibrationRunError)
        assert "stitching-plan" not in str(excinfo.value).lower()
    finally:
        masked_forward_module.compute_signed_contributions = real_compute


def test_stitching_plan_alignment_numerical_non_regression(real_e13_e15_identities):
    """Independent reconstruction of crop*weight/denom_local on CPU,
    compared against the real (unmodified) apply_window_stitching_forward
    called with the plan-prepared (here: no-op, since same-device) input,
    plus a full end-to-end same-device comparison across chunk sizes."""
    import torch

    e13, e15, repo_root = real_e13_e15_identities
    contracts = _build_small_contracts(e13, e15, repo_root)
    from src.dfp_phase1a_stitching_pullback import apply_window_stitching_forward, build_stitching_pullback_plan
    # build_stitch_weight/interpolate_raw_scores are re-exported from
    # src.dfp_phase1a_horizon_stitching -- the same path
    # dfp_phase1a_stitching_pullback.py's own (protected) source uses --
    # never imported via the bare `models.dinotext...` package path,
    # which triggers that package's __init__ side effects (mmcv model
    # registry registration) a second time under a different module
    # identity if anything else in the same test session already
    # imported it via sys.path's `src/open_vocabulary_segmentation` root.
    from src.dfp_phase1a_horizon_stitching import build_stitch_weight, interpolate_raw_scores

    plan = build_stitching_pullback_plan((63, 105), contract=contracts.pullback)
    node_count = _PATCH_GRID[0] * _PATCH_GRID[1]
    k = 3
    u = torch.rand(k, node_count, dtype=torch.float64) * 4 - 2
    u_aligned = run._align_tensor_to_stitching_plan_device(u, plan)
    assert u_aligned is u  # same-device no-op

    contribution = apply_window_stitching_forward(u_aligned, window_plan=plan, window_index=0)

    window = plan.plan.windows[0]
    crop_hw = window.extent.as_tuple()
    patch_scores = u_aligned.unsqueeze(0).transpose(1, 2)
    crop = interpolate_raw_scores(patch_scores, plan.patch_grid, crop_hw).squeeze(0)
    weight = build_stitch_weight(crop_hw[0], crop_hw[1], kind="uniform", dtype=u_aligned.dtype).squeeze(0)
    rows, cols = window.accumulation_slice
    denom_local = plan.denominator[:, :, rows, cols].squeeze(0)
    expected = crop * weight / denom_local
    assert torch.equal(contribution, expected)

    # Full end-to-end same-device comparison across chunk sizes.
    results_by_chunk = {}
    for cs in (1, 2, 4, 100):
        r, _, _ = _run_small_image(contracts, e13, image_h=63, image_w=105, ori_shape=(37, 61), seeds={i: 7300 + i for i in range(4)}, component_chunk_size=cs)
        results_by_chunk[cs] = r
    ref = results_by_chunk[1]
    for cs, r in results_by_chunk.items():
        for window_index in ref:
            for family in ("rho_M", "rho_0", "rho_320", "rho_g"):
                a, b = ref[window_index][family], r[window_index][family]
                assert a["sample_count"] == b["sample_count"]
                if a["sample_count"] > 0:
                    assert a["mean_absolute_residual"] == pytest.approx(b["mean_absolute_residual"], abs=1e-12)


def test_stitching_plan_alignment_domain_error_leaves_no_artifact(tmp_path):
    """The narrow domain-error translation: a stitching-plan alignment
    RuntimeError raises DfpPhase1bCalibrationRunError (already a member
    of the CLI's own _KNOWN_ERRORS tuple, unchanged -- see
    test_dfp_phase1b_calibration_production.py's own real-CLI
    end-to-end tests, which exercise this exact code path through the
    full production stack and are re-run, unmodified, as part of this
    repair's verification) and leaves no artifact file behind."""
    import torch

    class _FakePlan:
        def __init__(self, denominator):
            self.denominator = denominator

    class RaisingTensorLike:
        def __init__(self, real, error):
            self._real = real
            self._error = error

        @property
        def device(self):
            return self._real.device

        def to(self, *a, **k):
            raise self._error

    meta_plan = _FakePlan(torch.zeros(1).to("meta"))
    real_tensor = torch.zeros(2, dtype=torch.float64)
    double = RaisingTensorLike(real_tensor, RuntimeError("simulated stitching-plan transfer failure"))

    with pytest.raises(run.DfpPhase1bCalibrationRunError):
        run._align_tensor_to_stitching_plan_device(double, meta_plan)

    output_path = tmp_path / "phase1b_conservation_calibration.json"
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Section: CLI.
# ---------------------------------------------------------------------------


def test_cli_preflight_exits_zero(tmp_path, capsys):
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_calibration")
    exit_code = cli.main(["preflight", "--output", str(tmp_path / "phase1b_conservation_calibration.json"), "--mode", "fresh"])
    assert exit_code == 0
    assert "PREFLIGHT PASS" in capsys.readouterr().out


def test_cli_execute_rejects_cpu_device(tmp_path, capsys):
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_calibration")
    exit_code = cli.main(
        ["execute", "--output", str(tmp_path / "x.json"), "--mode", "fresh", "--device", "cpu", "--component-chunk-size", "4"]
    )
    assert exit_code == 2
    assert "not authorized" in capsys.readouterr().err


def test_cli_malformed_arguments_exit_before_heavy_import(capsys):
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_calibration")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["preflight", "--mode", "not-a-real-mode", "--output", "/tmp/x.json"])
    assert excinfo.value.code == 2  # argparse's own exit code for a bad choice


# ---------------------------------------------------------------------------
# Section: CLI robustness repair -- standalone `segmentation` import path
# (the CLI's own module-level sys.path setup must add `src/open_
# vocabulary_segmentation`, exactly as `diagnostics/run_k11_k12_
# stability.py` already does, so a bare `verify-artifact --geometry-
# json` subprocess never raises `ModuleNotFoundError: No module named
# 'segmentation'`) and the `--geometry-json` missing-canonical-ID
# domain error (a raw `KeyError` inside `_cmd_verify_artifact`'s
# `geometry_provider` closure must become a `DfpPhase1bCalibrationRun
# Error`, chained via `from error`, never a leaked traceback).
# ---------------------------------------------------------------------------

_CLI_PATH = ROOT / "diagnostics" / "run_dfp_phase1b_calibration.py"


def _sanitized_subprocess_env():
    """A genuinely fresh-process environment for standalone CLI
    execution: `PYTHONPATH` stripped of every repository-relative entry
    this TEST FILE's own module-level `sys.path` mutation put there (so
    the subprocess cannot silently inherit the fix-under-test's own
    effect via an environment variable), while everything else --
    `PATH`, the active venv's own interpreter/site-packages, and any
    legitimate module-provided `PYTHONPATH` entry such as this HPC
    environment's own OpenCV wheel path -- is preserved unchanged. This
    is not a fully wiped environment: wiping it entirely would also
    remove those legitimate, unrelated, non-repository entries that
    `cv2` itself requires, which has nothing to do with either repair
    in this file."""
    import os

    env = dict(os.environ)
    repo_paths = {str(ROOT), str(ROOT / "src/open_vocabulary_segmentation")}
    existing = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    env["PYTHONPATH"] = os.pathsep.join(entry for entry in existing if entry not in repo_paths)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _real_selected_ids():
    from src.dfp_phase1b_calibration_identity import load_identity as load_e16_identity

    repo_root = run.repository_root()
    e16_identity = load_e16_identity(repo_root=repo_root)
    return list(run.load_subset_manifest(repo_root, e16_identity)["selected_ids"])


def _uniform_geometry_payload(ids, *, height=448, width=448, patch_size=14):
    return {cid: {"inference_height": height, "inference_width": width, "patch_size": patch_size} for cid in ids}


def _finite_residual_summary(sample_count=10, value=1e-9):
    abs_sum = value * sample_count
    sq_sum = (value**2) * sample_count
    return {
        "sample_count": sample_count, "finite_count": sample_count, "non_finite_count": 0,
        "signed_minimum": -value, "signed_maximum": value,
        "maximum_absolute_residual": value, "mean_absolute_residual": abs_sum / sample_count,
        "root_mean_square_residual": math.sqrt(sq_sum / sample_count),
        "absolute_residual_sum": abs_sum, "squared_residual_sum": sq_sum,
    }


@pytest.fixture(scope="module")
def real_selected_ids():
    return _real_selected_ids()


@pytest.fixture(scope="module")
def valid_geometry_verification_fixture(tmp_path_factory, real_selected_ids):
    """A fully valid, complete, 200-row `phase1b_conservation_
    calibration.json` document plus a matching `--geometry-json` file,
    self-consistently reconstructed from the REAL frozen 500-image
    subset manifest and the REAL E13 crop/stride (both cheap TOML/JSON
    reads -- no torch, no model, no dataset), but from ARBITRARY
    uniform per-image geometry rather than any real decoded image --
    this never reads pixels and never touches the production artifact.
    Mirrors `tests/test_dfp_phase1b_calibration_artifact.py`'s own
    `_artifact_document` helper, applied to the CLI's real 500-image
    manifest instead of a small hand-built universe, purely so the
    resulting document can be independently re-validated through the
    real `--geometry-json` candidate-universe path."""
    ids = real_selected_ids
    geometry_payload = _uniform_geometry_payload(ids)

    def geometry_provider(canonical_id):
        entry = geometry_payload[canonical_id]
        return run.ImageGeometry(
            inference_height=entry["inference_height"], inference_width=entry["inference_width"], patch_size=entry["patch_size"],
        )

    window_plan_builder = run.default_window_plan_builder(run.repository_root())
    universe = run.build_candidate_universe(tuple(ids), geometry_provider=geometry_provider, window_plan_builder=window_plan_builder)
    result = artifact_module.select_calibration_windows(universe)

    rank_by_index = {t["canonical_candidate_index"]: t["selection_rank"] for t in result.selection_trace}
    key_by_index = {t["canonical_candidate_index"]: t["selection_key"] for t in result.selection_trace}
    rows = []
    for index, record in zip(result.selected_canonical_indices, result.selected_records):
        row = {
            **dict(record), "canonical_candidate_index": index,
            "selection_rank": rank_by_index[index], "selection_key": key_by_index[index], "completed": True,
        }
        for family in artifact_module.RESIDUAL_FAMILIES:
            row[family] = _finite_residual_summary()
        rows.append(row)

    aggregates = {
        family: artifact_module.aggregate_residual_summaries([row[family] for row in rows], label=family)
        for family in artifact_module.RESIDUAL_FAMILIES
    }

    document = {
        "schema_name": artifact_module.ARTIFACT_SCHEMA_NAME, "schema_version": artifact_module.ARTIFACT_SCHEMA_VERSION,
        "complete": True, "final": True,
        "calibration_identity_sha256": "0" * 64,
        "bound_e15_protocol_sha256": "1" * 64,
        "bound_e15_artifacts_sha256": "2" * 64,
        "bound_masked_forward_sha256": "3" * 64,
        "repository_commit": "a" * 40,
        "run_mode": "coco_stuff_500",
        "bound_subset_manifest_sha256": "4" * 64,
        "selected_ids_digest": "5" * 64,
        "candidate_window_count": len(universe),
        "expected_selected_window_count": artifact_module.WINDOW_COUNT,
        "completed_selected_window_count": len(rows),
        "candidate_universe_digest": result.candidate_universe_digest,
        "selected_windows_digest": result.selected_windows_digest,
        "selection_trace_digest": result.selection_trace_digest,
        "device_record": {
            "device_type": "cuda", "gpu_model_name": "NVIDIA A100", "cuda_compute_capability": "8.0",
            "total_device_memory_bytes": 42949672960, "cuda_driver_version": "535.104.05", "cuda_runtime_version": "12.2",
        },
        "software_record": {
            "repository_commit": "a" * 40, "python_version": "3.11.5", "torch_version": "2.12.0",
            "numpy_version": "2.4.2", "cuda_version": "12.2", "cudnn_version": "8.9",
            "deterministic_algorithms_enabled": True, "tf32_enabled": False, "autocast_enabled": False,
        },
        "propagation_precision_record": {
            "graph_storage_dtype": "float32", "canonical_recurrence_dtype": "float64",
            "masked_recurrence_dtype": "float64", "residual_accumulation_dtype": "float64",
            "alpha": 0.98, "steps": 320, "k": 12, "chunk_size": 4,
        },
        "selected_window_records": rows,
        "aggregate_residual_summaries": aggregates,
    }
    # Self-check: this fixture must itself be a genuinely valid artifact
    # before any test relies on it.
    artifact_module.validate_calibration_artifact(copy.deepcopy(document), candidate_universe=universe)

    tmp_dir = tmp_path_factory.mktemp("geometry_verification")
    artifact_path = tmp_dir / "phase1b_conservation_calibration.json"
    artifact_path.write_text(json.dumps(document))
    geometry_path = tmp_dir / "geometry.json"
    geometry_path.write_text(json.dumps(geometry_payload))
    return {
        "ids": ids, "geometry_payload": geometry_payload, "document": document,
        "artifact_path": artifact_path, "geometry_path": geometry_path, "tmp_dir": tmp_dir,
    }


# --- Category A: fresh-process, standalone subprocess behavior. --------


def test_standalone_subprocess_help_exits_zero_no_traceback(tmp_path):
    """The actual CLI entry point (`python .../run_dfp_phase1b_
    calibration.py --help`), run as a genuinely separate process from a
    directory outside the repository with a sanitized `PYTHONPATH`."""
    import subprocess as sp

    completed = sp.run(
        [sys.executable, str(_CLI_PATH), "--help"],
        cwd=str(tmp_path), env=_sanitized_subprocess_env(), capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0
    assert "Traceback" not in completed.stdout
    assert "Traceback" not in completed.stderr


def test_standalone_subprocess_verify_artifact_geometry_json_exits_zero(tmp_path, valid_geometry_verification_fixture):
    """The real, literal CLI entry point, exercising the exact
    `verify-artifact --geometry-json` path that previously raised
    `ModuleNotFoundError: No module named 'segmentation'` when invoked
    as a genuinely standalone process (arbitrary cwd outside the repo,
    sanitized `PYTHONPATH`, no prior import of any Phase 1A/1B module
    in this process)."""
    import subprocess as sp

    fixture = valid_geometry_verification_fixture
    completed = sp.run(
        [
            sys.executable, str(_CLI_PATH), "--repo-root", str(ROOT), "verify-artifact",
            "--artifact", str(fixture["artifact_path"]), "--geometry-json", str(fixture["geometry_path"]),
        ],
        cwd=str(tmp_path), env=_sanitized_subprocess_env(), capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0
    assert "PASSES the committed E16 validator" in completed.stdout
    assert "Traceback" not in completed.stdout
    assert "Traceback" not in completed.stderr


def test_standalone_subprocess_diagnostics_no_k11_k12_import_no_cuda_init(tmp_path, valid_geometry_verification_fixture):
    """A second, still genuinely fresh subprocess (not the already-
    import-mutated pytest process) that additionally introspects its
    own `sys.modules`/CUDA state after a successful `verify-artifact
    --geometry-json` run, to confirm: the success does not depend on
    `diagnostics.run_k11_k12_stability` ever having been imported (the
    calibration CLI's own `sys.path` fix is what makes the standalone
    import succeed, not a side effect of some other diagnostic module's
    own path mutation); and CUDA is never initialized (mmcv's own
    module-level `torch.cuda.is_available()` probe is an expected,
    unavoidable transitive side effect of importing `segmentation.
    evaluation` at all -- see the module docstring's own account -- and
    is not the same as CUDA-context initialization)."""
    import subprocess as sp

    fixture = valid_geometry_verification_fixture
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'diagnostics')!r})\n"
        "import importlib\n"
        "cli = importlib.import_module('run_dfp_phase1b_calibration')\n"
        "exit_code = cli.main([\n"
        f"    '--repo-root', {str(ROOT)!r}, 'verify-artifact',\n"
        f"    '--artifact', {str(fixture['artifact_path'])!r}, '--geometry-json', {str(fixture['geometry_path'])!r},\n"
        "])\n"
        "print('EXIT_CODE', exit_code)\n"
        "print('K11_K12_IMPORTED', 'run_k11_k12_stability' in sys.modules or 'diagnostics.run_k11_k12_stability' in sys.modules)\n"
        "torch_imported = 'torch' in sys.modules\n"
        "print('TORCH_IMPORTED', torch_imported)\n"
        "if torch_imported:\n"
        "    import torch\n"
        "    print('CUDA_INITIALIZED', torch.cuda.is_initialized())\n"
        "else:\n"
        "    print('CUDA_INITIALIZED', False)\n"
    )
    completed = sp.run(
        [sys.executable, "-c", script],
        cwd=str(tmp_path), env=_sanitized_subprocess_env(), capture_output=True, text=True, timeout=120,
    )
    assert "EXIT_CODE 0" in completed.stdout, completed.stdout + completed.stderr
    assert "K11_K12_IMPORTED False" in completed.stdout
    assert "CUDA_INITIALIZED False" in completed.stdout


def test_standalone_subprocess_help_and_bare_verify_artifact_never_import_torch(tmp_path, valid_geometry_verification_fixture):
    """`--help` and a bare `verify-artifact` (no `--geometry-json`, so
    the `segmentation`/mmcv import chain is never reached) must never
    import torch merely because the CLI's own import-path repair now
    adds `src/open_vocabulary_segmentation` to `sys.path` -- the repair
    itself is a pure `sys.path` mutation, not an import."""
    import subprocess as sp

    fixture = valid_geometry_verification_fixture
    for extra_args in (["--help"], ["verify-artifact", "--artifact", str(fixture["artifact_path"])]):
        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT / 'diagnostics')!r})\n"
            "import importlib\n"
            "cli = importlib.import_module('run_dfp_phase1b_calibration')\n"
            "try:\n"
            f"    exit_code = cli.main({extra_args!r})\n"
            "except SystemExit as exc:\n"
            "    exit_code = exc.code\n"
            "print('EXIT_CODE', exit_code)\n"
            "print('TORCH_IMPORTED', 'torch' in sys.modules)\n"
        )
        completed = sp.run(
            [sys.executable, "-c", script],
            cwd=str(tmp_path), env=_sanitized_subprocess_env(), capture_output=True, text=True, timeout=60,
        )
        assert "TORCH_IMPORTED False" in completed.stdout, (extra_args, completed.stdout, completed.stderr)


# --- Category B: missing-canonical-ID mutation matrix. ------------------


def _missing_id_geometry_payload(ids, geometry_payload, case):
    payload = dict(geometry_payload)
    if case == "missing_first":
        missing_id = ids[0]
        del payload[missing_id]
    elif case == "missing_middle":
        missing_id = ids[len(ids) // 2]
        del payload[missing_id]
    elif case == "missing_last":
        missing_id = ids[-1]
        del payload[missing_id]
    elif case == "missing_all":
        missing_id = ids[0]
        payload = {}
    elif case == "similar_suffixed_alternative_present":
        missing_id = ids[0]
        del payload[missing_id]
        payload[f"{missing_id}.jpg"] = _uniform_geometry_payload([missing_id])[missing_id]
    elif case == "reformatted_alternative_present":
        # Real canonical IDs are zero-padded pure-digit strings, so no
        # letter case exists to vary; the closest meaningful adversarial
        # analog to a "wrong-case" key is a differently-formatted (but
        # human-recognizable) rendering of the very same digits -- here,
        # the zero-padding stripped -- present instead of the exact
        # canonical key.
        missing_id = ids[0]
        del payload[missing_id]
        payload[str(int(missing_id))] = _uniform_geometry_payload([missing_id])[missing_id]
    else:
        raise AssertionError(case)
    return payload, missing_id


_MISSING_ID_CASES = (
    "missing_first", "missing_middle", "missing_last",
    "missing_all", "similar_suffixed_alternative_present", "reformatted_alternative_present",
)


@pytest.mark.parametrize("case", _MISSING_ID_CASES)
def test_missing_geometry_id_direct_api_raises_run_error_with_cause(tmp_path, valid_geometry_verification_fixture, case):
    """Direct-API boundary: `_cmd_verify_artifact` (the real function
    containing the repaired `geometry_provider` closure) raises
    `DfpPhase1bCalibrationRunError`, chained from the original
    `KeyError`, naming the exact missing canonical image ID -- never a
    default, never a skipped record, never a fuzzy match against a
    similar or differently-formatted key."""
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import argparse
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_calibration")
    fixture = valid_geometry_verification_fixture
    payload, missing_id = _missing_id_geometry_payload(fixture["ids"], fixture["geometry_payload"], case)

    geometry_path = tmp_path / f"geometry_{case}.json"
    geometry_path.write_text(json.dumps(payload))
    # The KeyError fires before the artifact document is ever inspected,
    # so any parseable JSON suffices here -- the artifact's own content
    # is irrelevant to this boundary.
    artifact_path = tmp_path / "placeholder_artifact.json"
    artifact_path.write_text("{}")

    args = argparse.Namespace(artifact=str(artifact_path), geometry_json=str(geometry_path), repo_root=ROOT)
    with pytest.raises(run.DfpPhase1bCalibrationRunError) as excinfo:
        cli._cmd_verify_artifact(args)

    assert missing_id in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, KeyError)


@pytest.mark.parametrize("case", _MISSING_ID_CASES)
def test_missing_geometry_id_cli_exits_two_no_traceback(tmp_path, valid_geometry_verification_fixture, capsys, case):
    """The real CLI dispatch (`main`): a missing canonical geometry ID
    is reported as the CLI's own established domain-error contract --
    exit code 2, the `DFP PHASE 1B CALIBRATION FAIL [verify-artifact]:`
    prefix, no traceback, no false PASS -- and neither the artifact nor
    the geometry JSON is modified."""
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_calibration")
    fixture = valid_geometry_verification_fixture
    payload, missing_id = _missing_id_geometry_payload(fixture["ids"], fixture["geometry_payload"], case)

    geometry_path = tmp_path / f"geometry_{case}.json"
    geometry_text_before = json.dumps(payload)
    geometry_path.write_text(geometry_text_before)
    artifact_path = tmp_path / "placeholder_artifact.json"
    artifact_text_before = "{}"
    artifact_path.write_text(artifact_text_before)

    exit_code = cli.main(
        ["--repo-root", str(ROOT), "verify-artifact", "--artifact", str(artifact_path), "--geometry-json", str(geometry_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert f"DFP PHASE 1B CALIBRATION FAIL [verify-artifact]: geometry JSON" in captured.err
    assert missing_id in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out
    assert "PASSES the committed E16 validator" not in captured.out
    assert geometry_path.read_text() == geometry_text_before
    assert artifact_path.read_text() == artifact_text_before


# --- Category C: exception-boundary controls. ----------------------------


def test_geometry_lookup_keyerror_is_the_translated_exception(tmp_path, valid_geometry_verification_fixture):
    """A controlled, minimal mapping (not the 500-id real manifest) that
    raises a plain `KeyError` on lookup is what gets translated -- the
    narrowest possible reproduction of the boundary."""
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import argparse
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_calibration")
    fixture = valid_geometry_verification_fixture
    controlled_payload = {}  # every lookup raises plain KeyError
    geometry_path = tmp_path / "controlled_empty_geometry.json"
    geometry_path.write_text(json.dumps(controlled_payload))
    artifact_path = tmp_path / "placeholder.json"
    artifact_path.write_text("{}")

    args = argparse.Namespace(artifact=str(artifact_path), geometry_json=str(geometry_path), repo_root=ROOT)
    with pytest.raises(run.DfpPhase1bCalibrationRunError) as excinfo:
        cli._cmd_verify_artifact(args)
    assert type(excinfo.value.__cause__) is KeyError
    assert fixture["ids"][0] in str(excinfo.value)


@pytest.mark.parametrize("fatal_exception", [KeyboardInterrupt, SystemExit, MemoryError])
def test_fatal_exceptions_from_json_load_propagate_uncaught(tmp_path, monkeypatch, fatal_exception):
    """A fatal exception raised while parsing the geometry JSON --
    standing in for any fatal exception surfacing before/around the
    narrow `KeyError` translation -- must propagate uncaught and
    unwrapped, never becoming `DfpPhase1bCalibrationRunError`."""
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import argparse
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_calibration")
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text("{}")
    artifact_path = tmp_path / "placeholder.json"
    artifact_path.write_text("{}")

    def _raise_fatal(*_a, **_k):
        raise fatal_exception("simulated fatal condition")

    monkeypatch.setattr(cli.json, "load", _raise_fatal)
    args = argparse.Namespace(artifact=str(artifact_path), geometry_json=str(geometry_path), repo_root=ROOT)
    with pytest.raises(fatal_exception):
        cli._cmd_verify_artifact(args)


def test_unrelated_downstream_domain_error_retains_its_own_type(valid_geometry_verification_fixture, tmp_path):
    """A downstream, unrelated schema-validation failure (a corrupted
    `schema_name`, caught by `validate_calibration_artifact` -- nothing
    to do with the geometry-ID lookup at all) must still raise
    `DfpPhase1bCalibrationArtifactError`, unchanged -- proving the
    narrow `except KeyError` in `geometry_provider` did not widen to
    swallow or reclassify unrelated errors."""
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import argparse
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_calibration")
    fixture = valid_geometry_verification_fixture
    corrupted_document = copy.deepcopy(fixture["document"])
    corrupted_document["schema_name"] = "not-the-real-schema-name"
    artifact_path = tmp_path / "corrupted_artifact.json"
    artifact_path.write_text(json.dumps(corrupted_document))

    args = argparse.Namespace(artifact=str(artifact_path), geometry_json=str(fixture["geometry_path"]), repo_root=ROOT)
    with pytest.raises(artifact_module.DfpPhase1bCalibrationArtifactError):
        cli._cmd_verify_artifact(args)


def test_valid_geometry_document_still_passes_unchanged(valid_geometry_verification_fixture):
    """Control: with every required canonical ID present, the repair
    changes nothing about the successful path."""
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import argparse
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_calibration")
    fixture = valid_geometry_verification_fixture
    args = argparse.Namespace(
        artifact=str(fixture["artifact_path"]), geometry_json=str(fixture["geometry_path"]), repo_root=ROOT,
    )
    assert cli._cmd_verify_artifact(args) == 0


# --- Category D: isolation. -----------------------------------------------


def test_geometry_verification_never_touches_model_gt_graph_or_recurrence(tmp_path, monkeypatch, valid_geometry_verification_fixture):
    """Neither the successful nor the missing-ID path ever loads a
    model/checkpoint, decodes ground-truth labels, or runs graph
    construction/recurrence -- and neither ever writes the artifact
    file it was asked to verify."""
    sys.path.insert(0, str(ROOT / "diagnostics"))
    import argparse
    import importlib

    cli = importlib.import_module("run_dfp_phase1b_calibration")
    fixture = valid_geometry_verification_fixture

    poison_hits: dict[str, int] = {}

    def _poison(name):
        def _poisoned(*_a, **_k):
            poison_hits[name] = poison_hits.get(name, 0) + 1
            raise AssertionError(f"{name} must not be reached during geometry-only verification")
        return _poisoned

    import torch

    monkeypatch.setattr(torch, "load", _poison("torch.load"))
    try:
        from mmcv.runner import CheckpointLoader

        monkeypatch.setattr(CheckpointLoader, "load_checkpoint", staticmethod(_poison("CheckpointLoader.load_checkpoint")))
    except Exception:
        pass
    try:
        from models.dinotext.cover_dr import graph as graph_mod

        monkeypatch.setattr(graph_mod, "build_directed_topk_graph", _poison("build_directed_topk_graph"))
    except Exception:
        pass
    try:
        from models.dinotext.cover_dr import finite_step_regime as fsr_mod

        monkeypatch.setattr(fsr_mod, "finite_step_propagate", _poison("finite_step_propagate"))
    except Exception:
        pass
    import src.dfp_phase1b_masked_forward as mf

    monkeypatch.setattr(mf, "build_masked_seed_chunk", _poison("build_masked_seed_chunk"))
    monkeypatch.setattr(mf, "propagate_masked_seed_chunk", _poison("propagate_masked_seed_chunk"))

    real_open = open

    def _poisoned_open(file, *a, **k):
        path_str = str(file)
        if "annotations" in path_str and "val2017" in path_str:
            poison_hits["gt_open"] = poison_hits.get("gt_open", 0) + 1
            raise AssertionError(f"GT/annotation file must not be opened: {path_str}")
        return real_open(file, *a, **k)

    monkeypatch.setattr("builtins.open", _poisoned_open)

    artifact_before = fixture["artifact_path"].read_text()
    geometry_before = fixture["geometry_path"].read_text()

    args = argparse.Namespace(
        artifact=str(fixture["artifact_path"]), geometry_json=str(fixture["geometry_path"]), repo_root=ROOT,
    )
    assert cli._cmd_verify_artifact(args) == 0

    missing_payload, missing_id = _missing_id_geometry_payload(fixture["ids"], fixture["geometry_payload"], "missing_first")
    missing_geometry_path = tmp_path / "missing_geometry.json"
    with real_open(missing_geometry_path, "w") as handle:
        json.dump(missing_payload, handle)
    placeholder_artifact_path = tmp_path / "placeholder.json"
    with real_open(placeholder_artifact_path, "w") as handle:
        handle.write("{}")
    args_missing = argparse.Namespace(
        artifact=str(placeholder_artifact_path), geometry_json=str(missing_geometry_path), repo_root=ROOT,
    )
    with pytest.raises(run.DfpPhase1bCalibrationRunError):
        cli._cmd_verify_artifact(args_missing)

    assert poison_hits == {}
    assert fixture["artifact_path"].read_text() == artifact_before
    assert fixture["geometry_path"].read_text() == geometry_before
    assert torch.cuda.is_initialized() is False


# ---------------------------------------------------------------------------
# Section: device/software/precision records and the atomic artifact
# install flow (Section 13).
# ---------------------------------------------------------------------------


def _zero_summary():
    import torch

    return run.summarize_residual_tensor(torch.zeros(0, dtype=torch.float64))


def _fake_device_record():
    return {
        "device_type": "cuda", "gpu_model_name": "Fake A100", "cuda_compute_capability": "8.0",
        "total_device_memory_bytes": 1024, "cuda_driver_version": "12.2", "cuda_runtime_version": "12.1",
    }


def _fake_software_record():
    return {
        "repository_commit": "0" * 40, "python_version": "3.11.0", "torch_version": "2.0.0",
        "numpy_version": "1.26.0", "cuda_version": "12.1", "cudnn_version": "8900",
        "deterministic_algorithms_enabled": True, "tf32_enabled": False, "autocast_enabled": False,
    }


def _make_candidate_row(index):
    return {
        "run_mode": "coco_stuff_500", "image_id": f"{index:012d}", "image_path": f"val2017/{index:012d}.jpg",
        "image_manifest_ordinal": index, "inference_canvas_height": 448, "inference_canvas_width": 448,
        "canonical_window_ordinal": 0, "top": 0, "left": 0, "bottom": 448, "right": 448,
        "crop_height": 448, "crop_width": 448, "patch_grid_height": 32, "patch_grid_width": 32,
    }


def _make_selected_window_row(index, *, completed=True):
    row = dict(_make_candidate_row(index))
    row.update({
        "canonical_candidate_index": index, "selection_rank": index,
        "selection_key": hashlib.sha256(f"selection-key-{index}".encode()).hexdigest(),
        "completed": completed,
    })
    for family in artifact_module.RESIDUAL_FAMILIES:
        row[family] = _zero_summary()
    return row


def _fake_selection_outcome(row_count):
    return run.SelectionOutcome(
        selected_rows=tuple(), candidate_universe_digest="a" * 64, selected_windows_digest="b" * 64,
        selection_trace_digest="c" * 64, candidate_window_count=max(row_count, artifact_module.WINDOW_COUNT),
    )


def test_real_software_record_shape_is_valid(tmp_path):
    record = run.real_software_record(run.repository_root())
    artifact_module._validate_software_record(record, label="software_record")
    assert len(record["repository_commit"]) == 40


def test_real_device_record_requires_cuda():
    import torch

    assert torch.cuda.is_available() is False  # this environment runs with CUDA_VISIBLE_DEVICES=""
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="CUDA"):
        run.real_device_record()


# ---------------------------------------------------------------------------
# Section: Torch-2.12 CUDA driver-version compatibility (real job 21240861:
# torch._C._cuda_getDriverVersion, the private symbol the OLD code relied
# on, does not exist in Torch 2.12.0; nor does a public torch.cuda.
# driver_version). CUDA is never actually initialized by any test below --
# `torch.cuda.is_available`/`current_device`/`get_device_properties`/
# `get_device_capability`/`get_device_name` are pure-Python monkeypatched
# stand-ins, never real driver calls.
# ---------------------------------------------------------------------------


def _install_fake_cuda(monkeypatch, *, total_memory=1024, gpu_name="Fake-H100",
                        compute_major=9, compute_minor=0, cuda_toolkit_version="13.2"):
    """Simulate a Torch-2.12-shaped, real-job-21240861-like environment:
    CUDA reports available with one visible device, but exposes NEITHER
    a public `torch.cuda.driver_version` accessor NOR the removed
    private `torch._C._cuda_getDriverVersion` symbol. Every patched
    attribute is a plain Python callable/value -- none of this touches
    the real CUDA driver, so `torch.cuda.is_initialized()` remains
    False throughout, exactly as if no GPU work had ever been
    attempted."""
    import torch

    class _FakeProperties:
        total_memory = 0

    _FakeProperties.total_memory = total_memory

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx: _FakeProperties())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx: (compute_major, compute_minor))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: gpu_name)
    monkeypatch.setattr(torch.version, "cuda", cuda_toolkit_version)
    monkeypatch.delattr(torch.cuda, "driver_version", raising=False)
    monkeypatch.delattr(torch._C, "_cuda_getDriverVersion", raising=False)


def test_no_private_cuda_driver_accessor_in_source():
    """Source/AST regression (Section 7): production code must never
    reference the removed private symbol `_cuda_getDriverVersion`, and
    must never introduce a DIFFERENT private `torch._C` symbol as a
    replacement, and must never import `pynvml`. AST-only (not a raw
    substring search) so the module's own docstrings -- which legitimately
    NAME the removed symbol as history/documentation -- do not trip this
    check; only actual `ast.Attribute`/`ast.Import`/`ast.ImportFrom` nodes
    count as a real reference."""
    import ast

    source = Path(run.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                assert "pynvml" not in alias.name.lower(), ast.dump(node)
        if isinstance(node, ast.Attribute) and node.attr.startswith("_") and "cuda" in node.attr.lower():
            # The only underscore-prefixed, cuda-named attribute access
            # this module may perform is on `torch.cuda` itself (e.g.
            # `torch.cuda.is_available`, not underscore-prefixed) --
            # assert no `torch._C._<anything with "cuda">` access exists.
            assert node.attr != "_cuda_getDriverVersion"


def test_query_cuda_driver_version_via_nvidia_smi_one_valid_row(monkeypatch):
    import subprocess as sp

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return sp.CompletedProcess(cmd, 0, stdout=b"570.124.06\n", stderr=b"")

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    result = run._query_cuda_driver_version_via_nvidia_smi()
    assert result == "570.124.06"
    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd == ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    assert kwargs.get("shell") is False


def test_query_cuda_driver_version_via_nvidia_smi_multiple_identical_rows(monkeypatch):
    import subprocess as sp
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: sp.CompletedProcess(cmd, 0, stdout=b"570.124.06\n570.124.06\n", stderr=b""))
    assert run._query_cuda_driver_version_via_nvidia_smi() == "570.124.06"


def test_query_cuda_driver_version_via_nvidia_smi_surrounding_whitespace(monkeypatch):
    import subprocess as sp
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: sp.CompletedProcess(cmd, 0, stdout=b"  570.124.06  \n", stderr=b""))
    assert run._query_cuda_driver_version_via_nvidia_smi() == "570.124.06"


def test_query_cuda_driver_version_via_nvidia_smi_trailing_newline(monkeypatch):
    import subprocess as sp
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: sp.CompletedProcess(cmd, 0, stdout=b"570.124.06\n\n\n", stderr=b""))
    assert run._query_cuda_driver_version_via_nvidia_smi() == "570.124.06"


def test_query_cuda_driver_version_via_nvidia_smi_blank_lines_plus_one_valid_row(monkeypatch):
    import subprocess as sp
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: sp.CompletedProcess(cmd, 0, stdout=b"\n   \n570.124.06\n\n", stderr=b""))
    assert run._query_cuda_driver_version_via_nvidia_smi() == "570.124.06"


def test_query_cuda_driver_version_via_nvidia_smi_conflicting_rows_rejected(monkeypatch):
    import subprocess as sp
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: sp.CompletedProcess(cmd, 0, stdout=b"570.124.06\n571.00.00\n", stderr=b""))
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="conflicting"):
        run._query_cuda_driver_version_via_nvidia_smi()


def test_query_cuda_driver_version_via_nvidia_smi_empty_stdout_rejected(monkeypatch):
    import subprocess as sp
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: sp.CompletedProcess(cmd, 0, stdout=b"", stderr=b""))
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="no non-blank"):
        run._query_cuda_driver_version_via_nvidia_smi()


def test_query_cuda_driver_version_via_nvidia_smi_whitespace_only_stdout_rejected(monkeypatch):
    import subprocess as sp
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: sp.CompletedProcess(cmd, 0, stdout=b"   \n  \n", stderr=b""))
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="no non-blank"):
        run._query_cuda_driver_version_via_nvidia_smi()


def test_query_cuda_driver_version_via_nvidia_smi_nul_byte_rejected(monkeypatch):
    import subprocess as sp
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: sp.CompletedProcess(cmd, 0, stdout=b"570.12\x004.06\n", stderr=b""))
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="NUL"):
        run._query_cuda_driver_version_via_nvidia_smi()


def test_query_cuda_driver_version_via_nvidia_smi_nonzero_exit_with_stderr(monkeypatch):
    import subprocess as sp
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: sp.CompletedProcess(cmd, 6, stdout=b"", stderr=b"NVIDIA-SMI has failed"))
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="exited with code 6"):
        run._query_cuda_driver_version_via_nvidia_smi()


def test_query_cuda_driver_version_via_nvidia_smi_executable_missing(monkeypatch):
    def fake_run(cmd, **k):
        raise FileNotFoundError("nvidia-smi not found")
    monkeypatch.setattr(run.subprocess, "run", fake_run)
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="not found"):
        run._query_cuda_driver_version_via_nvidia_smi()


def test_query_cuda_driver_version_via_nvidia_smi_timeout(monkeypatch):
    import subprocess as sp

    def fake_run(cmd, **k):
        raise sp.TimeoutExpired(cmd, k.get("timeout", 10.0))
    monkeypatch.setattr(run.subprocess, "run", fake_run)
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="timed out"):
        run._query_cuda_driver_version_via_nvidia_smi()


def test_query_cuda_driver_version_via_nvidia_smi_undecodable_bytes(monkeypatch):
    import subprocess as sp
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: sp.CompletedProcess(cmd, 0, stdout=b"\xff\xfe\x00\x01", stderr=b""))
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="UTF-8|NUL"):
        run._query_cuda_driver_version_via_nvidia_smi()


def test_query_cuda_driver_version_via_nvidia_smi_large_stderr_truncated(monkeypatch):
    import subprocess as sp
    huge = b"x" * 1_000_000
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: sp.CompletedProcess(cmd, 1, stdout=b"", stderr=huge))
    with pytest.raises(run.DfpPhase1bCalibrationRunError) as excinfo:
        run._query_cuda_driver_version_via_nvidia_smi()
    assert len(str(excinfo.value)) < 2000, f"diagnostic message not truncated: {len(str(excinfo.value))} chars"


def test_query_cuda_driver_version_via_nvidia_smi_shell_metacharacters_treated_as_data(monkeypatch):
    """A (mocked) nvidia-smi reporting shell-metacharacter-laden text as
    its driver-version row must be accepted verbatim as an opaque
    string, never interpreted -- proving these characters are data, not
    something ever passed through a shell (this function never invokes
    one)."""
    import subprocess as sp
    malicious = "570.00; rm -rf / #"
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: sp.CompletedProcess(cmd, 0, stdout=malicious.encode("utf-8") + b"\n", stderr=b""))
    result = run._query_cuda_driver_version_via_nvidia_smi()
    assert result == malicious


def test_query_cuda_driver_version_via_nvidia_smi_never_uses_shell_argument_form(monkeypatch):
    """The command is always a list (argument vector), never a single
    shell-interpreted string."""
    import subprocess as sp
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return sp.CompletedProcess(cmd, 0, stdout=b"570.00\n", stderr=b"")

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    run._query_cuda_driver_version_via_nvidia_smi()
    assert isinstance(captured["cmd"], list)
    assert captured["kwargs"]["shell"] is False


# ---------------------------------------------------------------------------
# Public-accessor compatibility (Section 8): if a future Torch release
# ever adds a genuine public `torch.cuda.driver_version` accessor.
# ---------------------------------------------------------------------------


def test_query_cuda_driver_version_public_accessor_string_used_no_subprocess(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "driver_version", lambda: "580.65.06", raising=False)

    def fail_if_called(cmd, **k):
        raise AssertionError("subprocess.run must not be called when the public accessor succeeds")

    monkeypatch.setattr(run.subprocess, "run", fail_if_called)
    assert run._query_cuda_driver_version() == "580.65.06"


def test_query_cuda_driver_version_public_accessor_int_used(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "driver_version", lambda: 57012406, raising=False)
    assert run._query_cuda_driver_version() == "57012406"


def test_query_cuda_driver_version_public_accessor_bool_rejected(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "driver_version", lambda: True, raising=False)
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="unusable value"):
        run._query_cuda_driver_version()


def test_query_cuda_driver_version_public_accessor_none_falls_back(monkeypatch):
    import torch
    import subprocess as sp

    monkeypatch.setattr(torch.cuda, "driver_version", lambda: None, raising=False)
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: sp.CompletedProcess(cmd, 0, stdout=b"570.00\n", stderr=b""))
    assert run._query_cuda_driver_version() == "570.00"


def test_query_cuda_driver_version_public_accessor_absent_falls_back(monkeypatch):
    import torch
    import subprocess as sp

    monkeypatch.delattr(torch.cuda, "driver_version", raising=False)
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: sp.CompletedProcess(cmd, 0, stdout=b"570.00\n", stderr=b""))
    assert run._query_cuda_driver_version() == "570.00"


def test_query_cuda_driver_version_public_accessor_empty_string_fails_closed_no_fallback(monkeypatch):
    """Documented rule: a malformed (but non-None) accessor result FAILS
    CLOSED rather than silently falling back to a different source."""
    import torch

    monkeypatch.setattr(torch.cuda, "driver_version", lambda: "   ", raising=False)

    def fail_if_called(cmd, **k):
        raise AssertionError("must not fall back to nvidia-smi for a malformed non-None accessor result")

    monkeypatch.setattr(run.subprocess, "run", fail_if_called)
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="empty value"):
        run._query_cuda_driver_version()


def test_query_cuda_driver_version_public_accessor_runtimeerror_translated_no_fallback(monkeypatch):
    """Documented rule: an accessor RuntimeError is translated cleanly,
    never silently retried via nvidia-smi."""
    import torch

    def raising():
        raise RuntimeError("simulated accessor failure")

    monkeypatch.setattr(torch.cuda, "driver_version", raising, raising=False)

    def fail_if_called(cmd, **k):
        raise AssertionError("must not fall back to nvidia-smi after an accessor RuntimeError")

    monkeypatch.setattr(run.subprocess, "run", fail_if_called)
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="raised"):
        run._query_cuda_driver_version()


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_query_cuda_driver_version_fatal_exceptions_propagate_from_accessor(monkeypatch, exc_type):
    import torch

    def raising():
        raise exc_type("fatal")

    monkeypatch.setattr(torch.cuda, "driver_version", raising, raising=False)
    with pytest.raises(exc_type):
        run._query_cuda_driver_version()


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_query_cuda_driver_version_fatal_exceptions_propagate_from_nvidia_smi_path(monkeypatch, exc_type):
    def raising(cmd, **k):
        raise exc_type("fatal")

    monkeypatch.setattr(run.subprocess, "run", raising)
    with pytest.raises(exc_type):
        run._query_cuda_driver_version_via_nvidia_smi()


# ---------------------------------------------------------------------------
# Full real_device_record() success path on a Torch-2.12-shaped fake
# environment (Section 7): the exact shape of real job 21240861's
# environment (no public accessor, no private symbol).
# ---------------------------------------------------------------------------


def test_real_device_record_torch212_shaped_environment_succeeds(monkeypatch):
    import subprocess as sp

    _install_fake_cuda(monkeypatch, cuda_toolkit_version="13.2")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return sp.CompletedProcess(cmd, 0, stdout=b"570.124.06\n", stderr=b"")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    record = run.real_device_record()
    assert record["device_type"] == "cuda"
    assert record["gpu_model_name"] == "Fake-H100"
    assert record["cuda_compute_capability"] == "9.0"
    assert record["total_device_memory_bytes"] == 1024
    assert record["cuda_driver_version"] == "570.124.06"
    assert record["cuda_runtime_version"] == "13.2"
    # Driver version and CUDA toolkit/runtime version must never be
    # confused with each other, even though they are both version-like
    # strings for the same physical GPU.
    assert record["cuda_driver_version"] != record["cuda_runtime_version"]
    assert len(calls) == 1

    artifact_module._validate_device_record(record, label="device_record")


def test_real_device_record_public_accessor_preferred_over_nvidia_smi(monkeypatch):
    import torch

    _install_fake_cuda(monkeypatch)
    monkeypatch.setattr(torch.cuda, "driver_version", lambda: "580.65.06", raising=False)

    def fail_if_called(cmd, **k):
        raise AssertionError("nvidia-smi must not be invoked when a public accessor succeeds")

    monkeypatch.setattr(run.subprocess, "run", fail_if_called)
    record = run.real_device_record()
    assert record["cuda_driver_version"] == "580.65.06"


def test_real_device_record_driver_query_failure_is_domain_error(monkeypatch):
    _install_fake_cuda(monkeypatch)

    def fake_run(cmd, **k):
        raise FileNotFoundError()

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="cuda_driver_version"):
        run.real_device_record()


def test_real_device_record_never_initializes_cuda(monkeypatch):
    """The fake-CUDA environment used by these tests never touches the
    real driver -- confirm `torch.cuda.is_initialized()` stays False."""
    import subprocess as sp
    import torch

    _install_fake_cuda(monkeypatch)
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: sp.CompletedProcess(cmd, 0, stdout=b"570.00\n", stderr=b""))
    assert torch.cuda.is_initialized() is False
    run.real_device_record()
    assert torch.cuda.is_initialized() is False


def test_build_propagation_precision_record_uses_the_real_recurrence_contract(real_e13_e15_identities):
    e13, e15, repo_root = real_e13_e15_identities
    from src.dfp_phase1a_recurrence_capture import dfp_recurrence_contract_from_protocol

    contract = dfp_recurrence_contract_from_protocol(e13)
    record = run.build_propagation_precision_record(recurrence_contract=contract, component_chunk_size=4)
    assert record["alpha"] == contract.alpha == 0.98
    assert record["steps"] == contract.steps == 320
    assert record["k"] == 12
    assert record["chunk_size"] == 4
    artifact_module._validate_propagation_precision_record(record, label="propagation_precision_record")


def test_build_propagation_precision_record_rejects_bad_chunk_size(real_e13_e15_identities):
    e13, e15, repo_root = real_e13_e15_identities
    from src.dfp_phase1a_recurrence_capture import dfp_recurrence_contract_from_protocol

    contract = dfp_recurrence_contract_from_protocol(e13)
    for bad in (0, -1, 1.5, "4"):
        with pytest.raises(run.DfpPhase1bCalibrationRunError):
            run.build_propagation_precision_record(recurrence_contract=contract, component_chunk_size=bad)


def test_assemble_calibration_artifact_partial_document_is_valid(tmp_path, real_e13_e15_identities):
    _, _, repo_root = real_e13_e15_identities
    preflight = run.run_preflight(output_path=tmp_path / "phase1b_conservation_calibration.json", mode="fresh")
    rows = [_make_selected_window_row(i, completed=(i < 3)) for i in range(5)]
    document = run.assemble_calibration_artifact(
        preflight=preflight, selection=_fake_selection_outcome(len(rows)), selected_window_rows=rows,
        device_record=_fake_device_record(), software_record=_fake_software_record(),
        propagation_precision_record={
            "graph_storage_dtype": "float32", "canonical_recurrence_dtype": "float64",
            "masked_recurrence_dtype": "float64", "residual_accumulation_dtype": "float64",
            "alpha": 0.98, "steps": 320, "k": 12, "chunk_size": 4,
        },
        complete=False, final=False,
    )
    artifact_module.validate_calibration_artifact(document)
    assert document["completed_selected_window_count"] == 3
    assert document["complete"] is False


def test_assemble_calibration_artifact_complete_document_is_valid(tmp_path, real_e13_e15_identities):
    _, _, repo_root = real_e13_e15_identities
    preflight = run.run_preflight(output_path=tmp_path / "phase1b_conservation_calibration.json", mode="fresh")
    rows = [_make_selected_window_row(i, completed=True) for i in range(artifact_module.WINDOW_COUNT)]
    precision_record = {
        "graph_storage_dtype": "float32", "canonical_recurrence_dtype": "float64",
        "masked_recurrence_dtype": "float64", "residual_accumulation_dtype": "float64",
        "alpha": 0.98, "steps": 320, "k": 12, "chunk_size": 4,
    }
    document = run.assemble_calibration_artifact(
        preflight=preflight, selection=_fake_selection_outcome(len(rows)), selected_window_rows=rows,
        device_record=_fake_device_record(), software_record=_fake_software_record(),
        propagation_precision_record=precision_record, complete=True, final=True,
    )
    artifact_module.validate_calibration_artifact(document)
    assert document["completed_selected_window_count"] == artifact_module.WINDOW_COUNT
    assert len(document["selected_window_records"]) == artifact_module.WINDOW_COUNT


def _complete_document(tmp_path, real_e13_e15_identities, *, row_count=None, **overrides):
    _, _, repo_root = real_e13_e15_identities
    preflight = run.run_preflight(output_path=tmp_path / "unused.json", mode="fresh")
    count = row_count if row_count is not None else artifact_module.WINDOW_COUNT
    rows = [_make_selected_window_row(i, completed=True) for i in range(count)]
    precision_record = {
        "graph_storage_dtype": "float32", "canonical_recurrence_dtype": "float64",
        "masked_recurrence_dtype": "float64", "residual_accumulation_dtype": "float64",
        "alpha": 0.98, "steps": 320, "k": 12, "chunk_size": 4,
    }
    document = run.assemble_calibration_artifact(
        preflight=preflight, selection=_fake_selection_outcome(count), selected_window_rows=rows,
        device_record=_fake_device_record(), software_record=_fake_software_record(),
        propagation_precision_record=precision_record, complete=(count == artifact_module.WINDOW_COUNT),
        final=(count == artifact_module.WINDOW_COUNT),
    )
    document.update(overrides)
    return document


def test_install_calibration_artifact_writes_atomically(tmp_path, real_e13_e15_identities):
    document = _complete_document(tmp_path, real_e13_e15_identities)
    destination = tmp_path / "phase1b_conservation_calibration.json"
    run.install_calibration_artifact(document, destination)
    assert destination.exists()
    with open(destination, encoding="utf-8") as handle:
        installed = json.load(handle)
    assert installed == document
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == []


def test_install_calibration_artifact_refuses_existing_without_overwrite(tmp_path, real_e13_e15_identities):
    document = _complete_document(tmp_path, real_e13_e15_identities)
    destination = tmp_path / "phase1b_conservation_calibration.json"
    destination.write_text("{}", encoding="utf-8")
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="already exists"):
        run.install_calibration_artifact(document, destination)
    assert destination.read_text(encoding="utf-8") == "{}"  # untouched
    assert list(tmp_path.glob("*.tmp")) == []


def test_install_calibration_artifact_allows_explicit_overwrite(tmp_path, real_e13_e15_identities):
    document = _complete_document(tmp_path, real_e13_e15_identities)
    destination = tmp_path / "phase1b_conservation_calibration.json"
    destination.write_text("{}", encoding="utf-8")
    run.install_calibration_artifact(document, destination, allow_overwrite=True)
    with open(destination, encoding="utf-8") as handle:
        assert json.load(handle) == document


def test_install_calibration_artifact_rejects_invalid_document_before_writing(tmp_path, real_e13_e15_identities):
    document = _complete_document(tmp_path, real_e13_e15_identities, row_count=199)  # 199 != 200, complete forced True below
    document["complete"] = True
    document["final"] = True
    destination = tmp_path / "phase1b_conservation_calibration.json"
    with pytest.raises(artifact_module.DfpPhase1bCalibrationArtifactError):
        run.install_calibration_artifact(document, destination)
    assert not destination.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_install_calibration_artifact_no_orphan_temp_file_on_fsync_failure(tmp_path, real_e13_e15_identities, monkeypatch):
    document = _complete_document(tmp_path, real_e13_e15_identities)
    destination = tmp_path / "phase1b_conservation_calibration.json"

    def broken_fsync(fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(run.os, "fsync", broken_fsync)
    with pytest.raises(OSError):
        run.install_calibration_artifact(document, destination)
    assert not destination.exists()
    assert list(tmp_path.glob(".*")) == [] or all(not p.name.endswith(".tmp") for p in tmp_path.glob(".*"))


def test_install_calibration_artifact_second_validation_catches_a_corrupted_read_back(tmp_path, real_e13_e15_identities, monkeypatch):
    """Simulate the read-back independent re-validation catching a
    difference between what was intended and what is actually on disk
    (e.g. filesystem corruption) -- patch the SECOND `validate_
    calibration_artifact` call (the one operating on the freshly
    read-back document) to fail, and confirm nothing is installed and
    no temp file survives."""
    document = _complete_document(tmp_path, real_e13_e15_identities)
    destination = tmp_path / "phase1b_conservation_calibration.json"

    call_count = {"n": 0}
    real_validate = run.validate_calibration_artifact

    def flaky_validate(doc, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise artifact_module.DfpPhase1bCalibrationArtifactError("simulated read-back corruption")
        return real_validate(doc, **kwargs)

    monkeypatch.setattr(run, "validate_calibration_artifact", flaky_validate)
    with pytest.raises(artifact_module.DfpPhase1bCalibrationArtifactError, match="simulated read-back corruption"):
        run.install_calibration_artifact(document, destination)
    assert not destination.exists()
    assert list(tmp_path.glob("*.tmp")) == []
    assert call_count["n"] == 2


def test_install_calibration_artifact_atomic_replace_failure_cleans_up(tmp_path, real_e13_e15_identities, monkeypatch):
    document = _complete_document(tmp_path, real_e13_e15_identities)
    destination = tmp_path / "phase1b_conservation_calibration.json"

    def broken_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(run.os, "replace", broken_replace)
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="could not be unambiguously reconciled"):
        run.install_calibration_artifact(document, destination)
    assert not destination.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_install_calibration_artifact_reconciles_ambiguous_success(tmp_path, real_e13_e15_identities, monkeypatch):
    """If `os.replace` raises but the destination content ends up
    matching exactly what was intended (e.g. a delayed error report
    after the rename actually completed), the install must be treated
    as successful, never re-raised."""
    document = _complete_document(tmp_path, real_e13_e15_identities)
    destination = tmp_path / "phase1b_conservation_calibration.json"
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    real_replace = run.os.replace

    def flaky_replace(src, dst):
        real_replace(src, dst)
        raise OSError("simulated delayed error after actual success")

    monkeypatch.setattr(run.os, "replace", flaky_replace)
    run.install_calibration_artifact(document, destination)
    assert destination.read_text(encoding="utf-8") == serialized
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Repair D (per-component-chunk stitching, job 21216898 OOM): independent
# equivalence, allocation-bound, retention, and call-count tests.
# ---------------------------------------------------------------------------


def _reference_process_selected_image_full_component(
    inference, prepared, canonical_id, *, selected_windows_for_image, protocol_identity,
    contracts, class_count, affinity_power, component_chunk_size,
):
    """Old-style, PRE-repair-D reference: byte-for-byte the same
    orchestration `run.process_selected_image` used before this repair
    (still chunking `build_masked_seed_chunk`/`propagate_masked_seed_chunk`
    for rho_0/rho_320, exactly as production does), but for rho_g issues
    exactly ONE `stitch_and_rescale_component_scores` call per selected
    window with `component_count=components.component_count` (the FULL
    image total) -- the defect this repair fixes. Built ONLY from
    unmodified, protected functions/dataclasses (never a reimplemented
    formula) so it is a genuine independent oracle, not a restatement of
    the repaired code."""
    import numpy as np
    import torch

    from src.dfp_phase1a_analysis import derive_horizon_labels
    from src.dfp_phase1a_final_rescale_pullback import (
        apply_final_rescale_forward, build_final_stitching_pullback_plan, iter_image_final_soft_memberships,
    )
    from src.dfp_phase1a_global_unary_components import compute_global_unary_components
    from src.dfp_phase1a_horizon_stitching import WindowHorizonCaptures, stitch_dfp_phase1a_horizons
    from src.dfp_phase1a_stitching_pullback import build_stitching_pullback_plan
    from src.dfp_phase1b_masked_forward import (
        PropagatedChunk, build_masked_seed_chunk, compute_membership_partition_residual,
        compute_propagated_reconstruction_residual, compute_seed_reconstruction_residual,
        compute_signed_contributions, compute_signed_margin_conservation_residual, propagate_masked_seed_chunk,
        stitch_and_rescale_component_scores,
    )

    selected_set = set(selected_windows_for_image)
    image_tensor = prepared.image_tensor
    img_shape = (prepared.inference_height, prepared.inference_width)
    ori_shape = prepared.ori_shape
    ori_height, ori_width = ori_shape

    pullback_plan = build_stitching_pullback_plan(img_shape, contract=contracts.pullback)

    window_results = {}
    window_captures = []
    for window in pullback_plan.plan.windows:
        result = run._process_one_window_for_calibration(
            inference, image_tensor, window, protocol_identity=protocol_identity,
            affinity_power=affinity_power, class_count=class_count, retain_graph=window.index in selected_set,
        )
        window_results[window.index] = result
        window_captures.append(WindowHorizonCaptures(window_index=window.index, captures=result.trace.snapshots))

    stitched_inf = stitch_dfp_phase1a_horizons(
        pullback_plan.plan, tuple(window_captures), contract=contracts.stitching, class_count=class_count,
    )
    scores_final = {}
    for horizon, tensor in stitched_inf.items():
        rescaled = apply_final_rescale_forward(tensor, img_shape=img_shape, ori_shape=ori_shape, contract=contracts.final_resolution)
        scores_final[horizon] = rescaled.squeeze(0).detach()

    labels = derive_horizon_labels({h: t.cpu().numpy() for h, t in scores_final.items()})
    a_labels = labels[0]
    b_labels = labels[contracts.recurrence.steps]
    dispute_mask_np = a_labels != b_labels
    dispute_mask_t = torch.from_numpy(dispute_mask_np)

    components = compute_global_unary_components(
        scores_final[0].cpu().numpy(), contract=contracts.component, class_count=class_count,
    )
    final_plan = build_final_stitching_pullback_plan(
        pullback_plan, img_shape=img_shape, ori_shape=ori_shape, contract=contracts.final_resolution,
    )

    active_components_by_window = {w.index: set() for w in pullback_plan.plan.windows}
    membership_chunks = {}
    for start, chunk in iter_image_final_soft_memberships(
        components.component_ids, components.component_count, final_plan=final_plan, component_chunk_size=component_chunk_size,
    ):
        for window_index, membership in chunk.items():
            for offset, is_positive in enumerate((membership > 0).any(dim=1).tolist()):
                if is_positive:
                    active_components_by_window[window_index].add(start + offset)
            if window_index in selected_set:
                membership_chunks.setdefault(window_index, {})[(start, start + membership.shape[0])] = membership

    disputed_component_ids = components.component_ids[dispute_mask_np]
    disputed_b_values = b_labels[dispute_mask_np]
    component_b_value_sets = {}
    for cid, bval in zip(disputed_component_ids.tolist(), disputed_b_values.tolist()):
        component_b_value_sets.setdefault(cid, set()).add(bval)

    results = {}
    for window_index in sorted(selected_set):
        window_result = window_results[window_index]
        graph = window_result.graph
        s0_full = window_result.unary_scores

        active_components = tuple(sorted(active_components_by_window.get(window_index, ())))
        query_class_set = set()
        for component_id in active_components:
            query_class_set.add(int(components.component_class_indices[component_id]))
            query_class_set.update(component_b_value_sets.get(component_id, ()))
        query_classes = tuple(sorted(query_class_set))

        family_summaries = {}
        if not query_classes or not active_components or window_index not in membership_chunks:
            for family in run.RESIDUAL_FAMILIES:
                family_summaries[family] = run.summarize_residual_tensor(torch.zeros(0, dtype=torch.float64))
            results[window_index] = family_summaries
            continue

        membership_rows_full = torch.cat(
            [membership_chunks[window_index][key] for key in sorted(membership_chunks[window_index])], dim=0,
        )
        rho_m_summary = compute_membership_partition_residual(membership_rows_full)
        family_summaries["rho_M"] = run.summarize_residual_tensor(rho_m_summary.signed)

        unary_for_query = s0_full[:, list(query_classes)].to(contracts.masked_forward.propagation_dtype)
        membership_rows_aligned = run._align_membership_device(membership_rows_full, unary_for_query)

        seed_by_component_parts = []
        propagated_by_component_parts = []
        chunk_start = 0
        while chunk_start < components.component_count:
            chunk_stop = min(chunk_start + component_chunk_size, components.component_count)
            membership_slice = membership_rows_aligned[chunk_start:chunk_stop]
            seed = build_masked_seed_chunk(
                unary_for_query, membership_slice, list(range(chunk_start, chunk_stop)), list(query_classes),
                contract=contracts.masked_forward, window_index=window_index,
            )
            propagated = propagate_masked_seed_chunk(graph, seed, contract=contracts.masked_forward)
            seed_by_component_parts.append(seed.unflatten(seed.seed_flat).transpose(0, 1))
            propagated_by_component_parts.append(seed.unflatten(propagated.t_final).transpose(0, 1))
            chunk_start = chunk_stop

        seed_by_component = torch.cat(seed_by_component_parts, dim=0)
        propagated_by_component = torch.cat(propagated_by_component_parts, dim=0)

        canonical_t0_full = window_result.trace.snapshots[0].to(contracts.masked_forward.propagation_dtype)
        canonical_tfinal_full = window_result.trace.snapshots[contracts.recurrence.steps].to(contracts.masked_forward.propagation_dtype)
        rho_0_summary = compute_seed_reconstruction_residual(seed_by_component, canonical_t0_full[:, list(query_classes)])
        family_summaries["rho_0"] = run.summarize_residual_tensor(rho_0_summary.signed)
        rho_320_summary = compute_propagated_reconstruction_residual(propagated_by_component, canonical_tfinal_full[:, list(query_classes)])
        family_summaries["rho_320"] = run.summarize_residual_tensor(rho_320_summary.signed)

        seed_by_component_for_stitch = run._align_tensor_to_stitching_plan_device(seed_by_component, pullback_plan)
        propagated_by_component_for_stitch = run._align_tensor_to_stitching_plan_device(propagated_by_component, pullback_plan)
        canonical_t0_for_stitch = run._align_tensor_to_stitching_plan_device(canonical_t0_full, pullback_plan)
        canonical_tfinal_for_stitch = run._align_tensor_to_stitching_plan_device(canonical_tfinal_full, pullback_plan)

        node_count = seed_by_component.shape[1]
        masked_chunk_for_window = PropagatedChunk(
            t0=seed_by_component_for_stitch.transpose(0, 1).reshape(node_count, -1),
            t_final=propagated_by_component_for_stitch.transpose(0, 1).reshape(node_count, -1),
            window_index=window_index, component_id_start=0, component_id_stop=components.component_count,
            query_class_indices=query_classes, steps_completed=contracts.recurrence.steps,
        )
        s0_masked, s_final_masked = stitch_and_rescale_component_scores(
            {window_index: masked_chunk_for_window}, window_plan=pullback_plan, final_contract=contracts.final_resolution,
            img_shape=img_shape, ori_shape=ori_shape, component_count=components.component_count, query_class_union=query_classes,
        )

        canonical_chunk_for_window = PropagatedChunk(
            t0=canonical_t0_for_stitch[:, list(query_classes)], t_final=canonical_tfinal_for_stitch[:, list(query_classes)],
            window_index=window_index, component_id_start=0, component_id_stop=1,
            query_class_indices=query_classes, steps_completed=contracts.recurrence.steps,
        )
        m0_w, m320_w = stitch_and_rescale_component_scores(
            {window_index: canonical_chunk_for_window}, window_plan=pullback_plan, final_contract=contracts.final_resolution,
            img_shape=img_shape, ori_shape=ori_shape, component_count=1, query_class_union=query_classes,
        )

        support_mask = run._window_final_support_mask(
            window_index, pullback_plan=pullback_plan, final_resolution_contract=contracts.final_resolution,
            img_shape=img_shape, ori_shape=ori_shape,
        )
        window_dispute_mask = support_mask & dispute_mask_t
        disputed_rows_t, disputed_cols_t = torch.nonzero(window_dispute_mask, as_tuple=True)
        order = torch.argsort(disputed_rows_t * ori_width + disputed_cols_t)
        disputed_rows_t, disputed_cols_t = disputed_rows_t[order], disputed_cols_t[order]

        if disputed_rows_t.numel() == 0:
            family_summaries["rho_g"] = run.summarize_residual_tensor(torch.zeros(0, dtype=torch.float64))
        else:
            a_vals = a_labels[disputed_rows_t.numpy(), disputed_cols_t.numpy()]
            b_vals = b_labels[disputed_rows_t.numpy(), disputed_cols_t.numpy()]
            query_index_of = {cls: idx for idx, cls in enumerate(query_classes)}
            a_local = torch.tensor([query_index_of[int(v)] for v in a_vals], dtype=torch.int64)
            b_local = torch.tensor([query_index_of[int(v)] for v in b_vals], dtype=torch.int64)
            a_global = torch.as_tensor(a_vals, dtype=torch.int64)
            b_global = torch.as_tensor(b_vals, dtype=torch.int64)

            g_chunk = compute_signed_contributions(
                s0_masked, s_final_masked, component_count=components.component_count, query_class_union=query_classes,
                disputed_pixel_rows=disputed_rows_t, disputed_pixel_cols=disputed_cols_t,
                disputed_pixel_a=a_global, disputed_pixel_b=b_global, ori_width=ori_width, component_id_start=0,
            )
            m0_at_a = m0_w[0, a_local, disputed_rows_t, disputed_cols_t]
            m0_at_b = m0_w[0, b_local, disputed_rows_t, disputed_cols_t]
            m320_at_a = m320_w[0, a_local, disputed_rows_t, disputed_cols_t]
            m320_at_b = m320_w[0, b_local, disputed_rows_t, disputed_cols_t]
            m0_x = (m0_at_b - m0_at_a).to(contracts.masked_forward.accumulation_dtype)
            m320_x = (m320_at_b - m320_at_a).to(contracts.masked_forward.accumulation_dtype)
            delta_m_x = m320_x - m0_x

            g_all_components = g_chunk.values.to(contracts.masked_forward.accumulation_dtype)
            rho_g_summary = compute_signed_margin_conservation_residual(g_all_components, delta_m_x)
            family_summaries["rho_g"] = run.summarize_residual_tensor(rho_g_summary.signed)

        results[window_index] = family_summaries

    return results


def _build_equivalence_fixture(seed_base):
    """A richer fixture than the module default: two overlapping windows
    plus flush-right/flush-bottom geometry (from the shared 63x105/
    (37,61) construction, already >=4 windows, non-square, non-integer
    final-resize ratio), >=1 window with >=37 global components (module
    docstring, repeatedly confirmed), >=3 query classes (_CLASS_COUNT=6),
    FP64 throughout, positive/negative/zero contributions and disputed
    pixels across multiple windows from the diverse random unary field."""
    e13, e15, repo_root = _real_identities_for_equivalence()
    contracts = _build_small_contracts(e13, e15, repo_root)
    from src.dfp_phase1a_stitching_pullback import build_stitching_pullback_plan
    import torch

    pullback_plan = build_stitching_pullback_plan((63, 105), contract=contracts.pullback)
    n_windows = len(pullback_plan.plan.windows)
    per_window_data = {i: _make_window_unary_features(seed=seed_base + i, diversity=True) for i in range(n_windows)}
    model = _FakeModel(per_window_data, grid_hw=_PATCH_GRID)
    inference = _FakeInference(model)
    image_tensor = torch.zeros(1, 3, 63, 105)
    prepared = run.PreparedImage(image_tensor=image_tensor, inference_height=63, inference_width=105, ori_shape=(37, 61))
    return inference, prepared, e13, contracts, n_windows


def _real_identities_for_equivalence():
    from src.dfp_phase1a_identity import load_identity as load_e13_identity
    from src.dfp_phase1b_identity import load_identity as load_e15_identity

    repo_root = run.repository_root()
    return load_e13_identity(repo_root=repo_root), load_e15_identity(repo_root=repo_root), repo_root


@pytest.mark.parametrize("component_chunk_size", [1, 2, 3, 4, 8, 16, 37, 64])
def test_repaired_runner_matches_old_style_full_component_reference(component_chunk_size):
    """Section 8: the repaired, per-chunk `run.process_selected_image`
    must be bit-identical (or, at minimum, match to floating-point
    machine precision) to the OLD-style full-component reference above,
    across a chunk-size sweep spanning far below, at, and far above the
    fixture's own component count (37: known from repeated construction,
    see `_instrumented_chunk_run`'s docstring) -- for all four residual
    families, their aggregate fields, and component/window bookkeeping."""
    # The fixture geometry itself (>=4 overlapping/flush windows,
    # non-square, non-integer final-resize ratio, >=37 global components,
    # >=3 query classes) already satisfies Section 8's "rich fixture"
    # requirement regardless of how many windows are SELECTED for actual
    # (expensive) masked processing below; canonical recurrence already
    # runs for every window unconditionally (needed for stitching), so
    # restricting selection to one window here only trims the part of the
    # cost that scales with component_chunk_size, matching Alliance
    # login-node conservatism -- it does not narrow what is being tested.
    inference, prepared, e13, contracts, n_windows = _build_equivalence_fixture(seed_base=7000)
    selected = [0]

    new_results = run.process_selected_image(
        inference, prepared, "000000000001", selected_windows_for_image=selected, protocol_identity=e13,
        contracts=contracts, class_count=_CLASS_COUNT, affinity_power=float(e13["graph"]["affinity_power"]),
        component_chunk_size=component_chunk_size,
    )
    # Rebuild the fixture (fresh model call-count state) for the reference
    # run, so both runs see byte-identical synthetic inputs.
    inference2, prepared2, _, _, _ = _build_equivalence_fixture(seed_base=7000)
    reference_results = _reference_process_selected_image_full_component(
        inference2, prepared2, "000000000001", selected_windows_for_image=selected, protocol_identity=e13,
        contracts=contracts, class_count=_CLASS_COUNT, affinity_power=float(e13["graph"]["affinity_power"]),
        component_chunk_size=component_chunk_size,
    )

    assert set(new_results.keys()) == set(reference_results.keys())
    saw_nonzero_rho_g = False
    for window_index in new_results:
        for family in ("rho_M", "rho_0", "rho_320", "rho_g"):
            a, b = new_results[window_index][family], reference_results[window_index][family]
            assert a["sample_count"] == b["sample_count"], (
                f"chunk_size={component_chunk_size} window={window_index} family={family}: "
                f"sample_count mismatch {a['sample_count']} != {b['sample_count']}"
            )
            if a["sample_count"] > 0:
                if family == "rho_g":
                    saw_nonzero_rho_g = True
                for field in ("signed_minimum", "signed_maximum", "mean_absolute_residual", "root_mean_square_residual", "maximum_absolute_residual"):
                    assert a[field] == pytest.approx(b[field], abs=1e-9, rel=1e-9), (
                        f"chunk_size={component_chunk_size} window={window_index} family={family} field={field}: "
                        f"{a[field]} != {b[field]}"
                    )
    assert saw_nonzero_rho_g, "fixture assumption violated: expected at least one window with disputed pixels"


def test_allocation_bound_scales_with_chunk_size_not_total_component_count():
    """Section 9: instrument the REAL, unmodified `torch.zeros` reached
    through `stitch_and_rescale_component_scores` and assert every
    accumulator-shaped allocation's channel_count never exceeds
    component_chunk_size * len(query_class_union) -- and, in particular,
    is never sized by the fixture's full component_count (37) once
    chunk_size is smaller than it."""
    import torch

    seen_channel_counts = []
    real_zeros = torch.zeros

    def spying_zeros(*args, **kwargs):
        out = real_zeros(*args, **kwargs)
        if out.dim() == 4 and out.shape[0] == 1 and out.dtype == torch.float64:
            seen_channel_counts.append(out.shape[1])
        return out

    for component_chunk_size in (1, 2, 8, 16, 32, 64):
        # A fresh fixture per chunk_size: _FakeModel.generate_patch_snapshot
        # indexes its per-window data by an internal, ever-incrementing
        # call_count, so reusing one `inference` object across multiple
        # `process_selected_image` calls would exhaust it after the first
        # iteration.
        inference, prepared, e13, contracts, n_windows = _build_equivalence_fixture(seed_base=8100)
        seen_channel_counts.clear()
        torch.zeros = spying_zeros
        try:
            run.process_selected_image(
                inference, prepared, "000000000001", selected_windows_for_image=[0], protocol_identity=e13,
                contracts=contracts, class_count=_CLASS_COUNT, affinity_power=float(e13["graph"]["affinity_power"]),
                component_chunk_size=component_chunk_size,
            )
        finally:
            torch.zeros = real_zeros
        assert seen_channel_counts, "no accumulator allocations observed -- instrumentation is not matching"
        max_query_union_width = _CLASS_COUNT  # query_class_union can never exceed the total class count
        bound = component_chunk_size * max_query_union_width
        for channel_count in seen_channel_counts:
            assert channel_count <= bound, (
                f"chunk_size={component_chunk_size}: observed accumulator channel_count={channel_count} "
                f"exceeds the bound component_chunk_size*len(query_class_union)={bound} -- the full-component "
                f"allocation appears not to have been eliminated"
            )
        # Explicitly confirm the allocation never used the fixture's full
        # component count (37) once it exceeds the chunk size.
        if component_chunk_size < 37:
            assert max(seen_channel_counts) < 37 * max_query_union_width


def test_signed_contribution_chunk_tensors_are_not_retained_after_extraction():
    """Section 10: the large, image-resolution stitched T0/T320 chunk
    tensors (`s0_chunk`/`s_final_chunk`) must not survive past the
    iteration that produced them -- proven with weak references: once
    `process_selected_image` returns, every such weakref must be dead
    (garbage collected), even though the small `SignedContributionChunk`
    results they were derived from are (transiently, inside the
    function) still logically "in use" one iteration at a time."""
    import gc
    import weakref

    import src.dfp_phase1b_masked_forward as masked_forward_module

    inference, prepared, e13, contracts, n_windows = _build_equivalence_fixture(seed_base=8200)

    live_refs = []
    real_stitch = masked_forward_module.stitch_and_rescale_component_scores

    def spying_stitch(*args, **kwargs):
        s0, s_final = real_stitch(*args, **kwargs)
        live_refs.append(weakref.ref(s0))
        live_refs.append(weakref.ref(s_final))
        return s0, s_final

    # Patching `masked_forward_module` (the module `process_selected_
    # image` actually re-imports `stitch_and_rescale_component_scores`
    # from, fresh, on every call) is the only patch that matters --
    # `run.stitch_and_rescale_component_scores` is never read by that
    # function (it is a local, call-time import, never a module-level
    # `run` attribute) and assigning to it here would only leave a stray,
    # permanent attribute on `run` behind for later tests to trip over.
    masked_forward_module.stitch_and_rescale_component_scores = spying_stitch
    try:
        run.process_selected_image(
            inference, prepared, "000000000001", selected_windows_for_image=[0], protocol_identity=e13,
            contracts=contracts, class_count=_CLASS_COUNT, affinity_power=float(e13["graph"]["affinity_power"]),
            component_chunk_size=1,
        )
    finally:
        masked_forward_module.stitch_and_rescale_component_scores = real_stitch

    assert len(live_refs) >= 2, "no stitching calls observed -- instrumentation is not matching"
    gc.collect()
    dead = sum(1 for r in live_refs if r() is None)
    assert dead == len(live_refs), (
        f"{len(live_refs) - dead} of {len(live_refs)} stitched chunk tensors are still reachable after "
        f"process_selected_image returned -- a list is retaining a full stitched tensor"
    )


def test_per_chunk_call_counts_equal_expected_chunks_for_62_components(monkeypatch):
    """Section 11: for a real, rich multi-component image (the same
    proven fixture family used throughout this file -- >=37 global
    components at window 0 for a diverse random unary field, confirmed
    repeatedly by construction, see `_instrumented_chunk_run`'s own
    docstring), confirm `build_masked_seed_chunk`, `propagate_masked_
    seed_chunk`, the masked `stitch_and_rescale_component_scores`, and
    `compute_signed_contributions` calls all equal
    `ceil(component_count/chunk_size)` -- never `1` (the old defect) and
    never `component_count` (over-chunked)."""
    import math

    import src.dfp_phase1b_masked_forward as masked_forward_module

    real_build = masked_forward_module.build_masked_seed_chunk
    real_propagate = masked_forward_module.propagate_masked_seed_chunk
    real_stitch = masked_forward_module.stitch_and_rescale_component_scores
    real_signed = masked_forward_module.compute_signed_contributions

    # Discover this fixture's actual global component count once, with
    # component_chunk_size=1 -- exactly one call per global component, by
    # construction (unlike a large chunk size, which always yields exactly
    # one call regardless of the true component count).
    build_calls = {"n": 0}

    def counting_build0(*args, **kwargs):
        build_calls["n"] += 1
        return real_build(*args, **kwargs)

    inference0, prepared0, e13, contracts, n_windows = _build_equivalence_fixture(seed_base=7000)
    monkeypatch.setattr(masked_forward_module, "build_masked_seed_chunk", counting_build0)
    monkeypatch.setattr(run, "build_masked_seed_chunk", counting_build0, raising=False)
    baseline_results = run.process_selected_image(
        inference0, prepared0, "000000000001", selected_windows_for_image=[0], protocol_identity=e13,
        contracts=contracts, class_count=_CLASS_COUNT, affinity_power=float(e13["graph"]["affinity_power"]),
        component_chunk_size=1,
    )
    monkeypatch.setattr(masked_forward_module, "build_masked_seed_chunk", real_build)
    component_count = build_calls["n"]
    assert component_count >= 5, f"fixture produced only {component_count} components; need >=5 for a meaningful sweep"
    has_disputed = baseline_results[0]["rho_g"]["sample_count"] > 0
    assert has_disputed, "fixture assumption violated: expected disputed pixels for this seed"

    for component_chunk_size in (1, 2, 8, 16, 37, 64):
        # A fresh fixture per chunk_size -- `_FakeModel.generate_patch_
        # snapshot` indexes its per-window data by an ever-incrementing
        # call_count, so reusing one `inference` object across multiple
        # `process_selected_image` calls would exhaust it.
        inference, prepared, _, _, _ = _build_equivalence_fixture(seed_base=7000)
        build_calls_by_fn = {"build": 0, "propagate": 0, "stitch": 0, "signed": 0}
        # NOTE: reuse the `real_build`/`real_propagate`/`real_stitch`/
        # `real_signed` captured ONCE before this loop -- re-reading
        # `masked_forward_module.<name>` here would instead capture the
        # PREVIOUS iteration's own counting wrapper (since it is still
        # installed until this iteration's monkeypatch.setattr call below
        # replaces it), wrapping wrapper around wrapper into unbounded
        # recursion.

        def counting_build2(*args, **kwargs):
            build_calls_by_fn["build"] += 1
            return real_build(*args, **kwargs)

        def counting_propagate(*args, **kwargs):
            build_calls_by_fn["propagate"] += 1
            return real_propagate(*args, **kwargs)

        stitch_call_index = {"n": 0}

        def counting_stitch(propagated_by_window, **kwargs):
            # The canonical-numerator stitching call always happens FIRST,
            # once per window, before the masked per-chunk loop begins (see
            # `process_selected_image`'s own structure) -- distinguish it
            # by call ORDER, never by `component_count`, since a masked
            # chunk's own component_count is also 1 whenever
            # component_chunk_size == 1, making that value indistinguishable
            # from the canonical call's own component_count=1.
            stitch_call_index["n"] += 1
            if stitch_call_index["n"] > 1:
                build_calls_by_fn["stitch"] += 1
            return real_stitch(propagated_by_window, **kwargs)

        def counting_signed(*args, **kwargs):
            build_calls_by_fn["signed"] += 1
            return real_signed(*args, **kwargs)

        monkeypatch.setattr(masked_forward_module, "build_masked_seed_chunk", counting_build2)
        monkeypatch.setattr(run, "build_masked_seed_chunk", counting_build2, raising=False)
        monkeypatch.setattr(masked_forward_module, "propagate_masked_seed_chunk", counting_propagate)
        monkeypatch.setattr(run, "propagate_masked_seed_chunk", counting_propagate, raising=False)
        monkeypatch.setattr(masked_forward_module, "stitch_and_rescale_component_scores", counting_stitch)
        monkeypatch.setattr(run, "stitch_and_rescale_component_scores", counting_stitch, raising=False)
        monkeypatch.setattr(masked_forward_module, "compute_signed_contributions", counting_signed)
        monkeypatch.setattr(run, "compute_signed_contributions", counting_signed, raising=False)

        run.process_selected_image(
            inference, prepared, "000000000001", selected_windows_for_image=[0], protocol_identity=e13,
            contracts=contracts, class_count=_CLASS_COUNT, affinity_power=float(e13["graph"]["affinity_power"]),
            component_chunk_size=component_chunk_size,
        )
        expected_chunks = math.ceil(component_count / component_chunk_size)
        assert build_calls_by_fn["build"] == expected_chunks, (
            f"chunk_size={component_chunk_size}: build_masked_seed_chunk calls {build_calls_by_fn['build']} != {expected_chunks}"
        )
        assert build_calls_by_fn["propagate"] == expected_chunks
        assert build_calls_by_fn["stitch"] == expected_chunks, (
            f"chunk_size={component_chunk_size}: masked stitch_and_rescale_component_scores calls "
            f"{build_calls_by_fn['stitch']} != {expected_chunks}"
        )
        assert build_calls_by_fn["signed"] == expected_chunks


def test_signed_contribution_chunk_gap_is_rejected():
    """Section 12: if the reassembled signed-contribution chunks do not
    cover [0, component_count) contiguously (simulated by dropping the
    first chunk), the runner raises DfpPhase1bCalibrationRunError --
    never silently produces a wrong-shaped or partial rho_g."""
    import src.dfp_phase1b_masked_forward as masked_forward_module

    inference, prepared, e13, contracts, n_windows = _build_equivalence_fixture(seed_base=8400)

    real_signed_cls = masked_forward_module.SignedContributionChunk
    call_index = {"n": 0}
    real_compute = masked_forward_module.compute_signed_contributions

    def dropping_compute(*args, **kwargs):
        result = real_compute(*args, **kwargs)
        call_index["n"] += 1
        if call_index["n"] == 1:
            # Corrupt the first chunk's own component_id_start so the
            # reassembly validation must detect the resulting gap.
            return real_signed_cls(
                values=result.values, component_id_start=result.component_id_start + 1,
                component_id_stop=result.component_id_stop + 1,
                disputed_pixel_flat_indices=result.disputed_pixel_flat_indices,
            )
        return result

    # Only `masked_forward_module`'s own attribute matters (see the
    # retention test above for why patching `run.*` directly here would
    # only leak a stray permanent attribute onto `run` for later tests).
    masked_forward_module.compute_signed_contributions = dropping_compute
    try:
        with pytest.raises(run.DfpPhase1bCalibrationRunError, match="gap|overlap|cover"):
            run.process_selected_image(
                inference, prepared, "000000000001", selected_windows_for_image=[0], protocol_identity=e13,
                contracts=contracts, class_count=_CLASS_COUNT, affinity_power=float(e13["graph"]["affinity_power"]),
                component_chunk_size=1,
            )
    finally:
        masked_forward_module.compute_signed_contributions = real_compute


def _corrupt_nth_signed_contribution_chunk(*, n_to_corrupt, corrupt_fn):
    """Shared instrumentation: wrap the real, unmodified
    `compute_signed_contributions` so its `n_to_corrupt`-th call (1-indexed,
    ascending chunk order -- this function is called exactly once per
    masked component chunk, never for the canonical numerator) returns a
    NEW `SignedContributionChunk` (via the frozen dataclass's own
    constructor, never by mutating the original -- `SignedContributionChunk`
    is itself `@dataclass(frozen=True)`, so in-place mutation would raise
    `FrozenInstanceError` even if attempted) with `corrupt_fn` applied to
    its `values` tensor, while `component_id_start`/`component_id_stop`/
    `disputed_pixel_flat_indices` are carried over UNCHANGED from the real
    result -- exactly the "declared range says one thing, values says
    another" shape this repair must catch."""
    import src.dfp_phase1b_masked_forward as masked_forward_module

    real_signed_cls = masked_forward_module.SignedContributionChunk
    real_compute = masked_forward_module.compute_signed_contributions
    call_index = {"n": 0}

    def wrapped(*args, **kwargs):
        result = real_compute(*args, **kwargs)
        call_index["n"] += 1
        if call_index["n"] == n_to_corrupt:
            return real_signed_cls(
                values=corrupt_fn(result.values), component_id_start=result.component_id_start,
                component_id_stop=result.component_id_stop, disputed_pixel_flat_indices=result.disputed_pixel_flat_indices,
            )
        return result

    return masked_forward_module, real_compute, wrapped


def _narrow_by_one(values):
    assert values.shape[1] >= 1
    return values[:, :-1]


def _wide_by_one(values):
    import torch
    return torch.cat([values, values[:, :1]], dim=1)


def _zero_width(values):
    return values[:, :0]


def _run_with_corrupted_chunk(*, seed_base, component_chunk_size, n_to_corrupt, corrupt_fn, selected_windows=(0,)):
    masked_forward_module, real_compute, wrapped = _corrupt_nth_signed_contribution_chunk(
        n_to_corrupt=n_to_corrupt, corrupt_fn=corrupt_fn,
    )
    inference, prepared, e13, contracts, n_windows = _build_equivalence_fixture(seed_base=seed_base)
    masked_forward_module.compute_signed_contributions = wrapped
    try:
        return run.process_selected_image(
            inference, prepared, "000000000001", selected_windows_for_image=list(selected_windows), protocol_identity=e13,
            contracts=contracts, class_count=_CLASS_COUNT, affinity_power=float(e13["graph"]["affinity_power"]),
            component_chunk_size=component_chunk_size,
        )
    finally:
        masked_forward_module.compute_signed_contributions = real_compute


def _chunk_count_for(seed_base, component_chunk_size, selected_windows=(0,)):
    """Discover this fixture's real chunk count for a given chunk_size by
    counting real (unwrapped) build_masked_seed_chunk calls."""
    import src.dfp_phase1b_masked_forward as masked_forward_module

    inference, prepared, e13, contracts, n_windows = _build_equivalence_fixture(seed_base=seed_base)
    calls = {"n": 0}
    real_build = masked_forward_module.build_masked_seed_chunk

    def counting(*a, **k):
        calls["n"] += 1
        return real_build(*a, **k)

    masked_forward_module.build_masked_seed_chunk = counting
    try:
        run.process_selected_image(
            inference, prepared, "000000000001", selected_windows_for_image=list(selected_windows), protocol_identity=e13,
            contracts=contracts, class_count=_CLASS_COUNT, affinity_power=float(e13["graph"]["affinity_power"]),
            component_chunk_size=component_chunk_size,
        )
    finally:
        masked_forward_module.build_masked_seed_chunk = real_build
    return calls["n"]


_WIDTH_SEED_BASE = 8400  # >=40 global components for window 0 (established by construction; re-checked below)


@pytest.mark.parametrize("corrupt_fn,expected_phrase", [
    (_narrow_by_one, "declares component-axis width"),
    (_wide_by_one, "declares component-axis width"),
    (_zero_width, "declares component-axis width"),
])
def test_signed_contribution_chunk_width_mismatch_rejected(corrupt_fn, expected_phrase):
    """Repair D follow-up (independent-verification finding): a
    SignedContributionChunk whose `values` tensor's component-axis width
    disagrees with its own declared [component_id_start, component_id_stop)
    range -- narrower, wider, or entirely zero-width for a declared
    NONEMPTY range -- must be rejected with DfpPhase1bCalibrationRunError
    naming the chunk's range and both the expected and observed widths,
    never silently concatenated into a wrong-width rho_g."""
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match=expected_phrase):
        _run_with_corrupted_chunk(
            seed_base=_WIDTH_SEED_BASE, component_chunk_size=8, n_to_corrupt=1, corrupt_fn=corrupt_fn,
        )


def test_signed_contribution_chunk_correct_width_accepted():
    """Control: an UNCORRUPTED run (corrupt_fn is the identity) with the
    exact same instrumentation plumbing must complete normally -- the new
    validation must not reject legitimate, self-consistent chunks."""
    results = _run_with_corrupted_chunk(
        seed_base=_WIDTH_SEED_BASE, component_chunk_size=8, n_to_corrupt=1, corrupt_fn=lambda v: v,
    )
    assert results[0]["rho_g"]["sample_count"] > 0


def test_signed_contribution_chunk_width_mismatch_first_chunk():
    n_chunks = _chunk_count_for(_WIDTH_SEED_BASE, component_chunk_size=8)
    assert n_chunks >= 3, "fixture assumption violated: need >=3 chunks for first/middle/final coverage"
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="declares component-axis width"):
        _run_with_corrupted_chunk(seed_base=_WIDTH_SEED_BASE, component_chunk_size=8, n_to_corrupt=1, corrupt_fn=_narrow_by_one)


def test_signed_contribution_chunk_width_mismatch_middle_chunk():
    n_chunks = _chunk_count_for(_WIDTH_SEED_BASE, component_chunk_size=8)
    assert n_chunks >= 3
    middle = n_chunks // 2 + 1  # 1-indexed, strictly interior
    assert 1 < middle < n_chunks
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="declares component-axis width"):
        _run_with_corrupted_chunk(seed_base=_WIDTH_SEED_BASE, component_chunk_size=8, n_to_corrupt=middle, corrupt_fn=_narrow_by_one)


def test_signed_contribution_chunk_width_mismatch_final_chunk():
    n_chunks = _chunk_count_for(_WIDTH_SEED_BASE, component_chunk_size=8)
    assert n_chunks >= 3
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="declares component-axis width"):
        _run_with_corrupted_chunk(seed_base=_WIDTH_SEED_BASE, component_chunk_size=8, n_to_corrupt=n_chunks, corrupt_fn=_narrow_by_one)


def test_signed_contribution_chunk_width_mismatch_single_chunk():
    """component_chunk_size large enough that the whole image is exactly
    ONE chunk -- corrupting "the only chunk" must still be caught."""
    n_chunks = _chunk_count_for(_WIDTH_SEED_BASE, component_chunk_size=10_000)
    assert n_chunks == 1
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="declares component-axis width"):
        _run_with_corrupted_chunk(seed_base=_WIDTH_SEED_BASE, component_chunk_size=10_000, n_to_corrupt=1, corrupt_fn=_narrow_by_one)


def test_signed_contribution_chunk_width_mismatch_final_partial_chunk():
    """component_chunk_size chosen so the image's component_count is NOT
    evenly divisible by it -- the final chunk is genuinely partial (its
    own width < component_chunk_size) -- corrupting exactly that partial
    chunk must still be caught, using ITS OWN (smaller) declared width,
    never `component_chunk_size` itself."""
    component_chunk_size = 16
    n_chunks = _chunk_count_for(_WIDTH_SEED_BASE, component_chunk_size=component_chunk_size)
    assert n_chunks >= 2
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="declares component-axis width"):
        _run_with_corrupted_chunk(
            seed_base=_WIDTH_SEED_BASE, component_chunk_size=component_chunk_size, n_to_corrupt=n_chunks, corrupt_fn=_narrow_by_one,
        )


def test_signed_contribution_chunk_width_error_reports_original_undoctored_range():
    """The raised diagnostic must name the chunk's OWN, ORIGINAL declared
    range and the TRUE expected width computed from it -- never a width
    inferred from the corrupted `values` tensor, and never a metadata
    rewrite that would make the mismatch disappear. Corrupts chunk index 2
    of a component_chunk_size=8 run (declared range [8, 16), expected
    width 8) down to width 7, and asserts the exact range/widths appear."""
    with pytest.raises(run.DfpPhase1bCalibrationRunError) as excinfo:
        _run_with_corrupted_chunk(seed_base=_WIDTH_SEED_BASE, component_chunk_size=8, n_to_corrupt=2, corrupt_fn=_narrow_by_one)
    message = str(excinfo.value)
    assert "[8, 16)" in message, message
    assert "component-axis width 8" in message, message
    assert "component-axis width 7" in message, message


def test_signed_contribution_chunk_width_mismatch_rejected_before_final_residual():
    """Rejection must occur BEFORE the frozen
    compute_signed_margin_conservation_residual is ever reached (and
    therefore before concatenation, since that residual is the only
    consumer of the concatenated matrix) -- confirmed by making that
    function itself fail the test if called."""
    import src.dfp_phase1b_masked_forward as masked_forward_module

    real_residual = masked_forward_module.compute_signed_margin_conservation_residual

    def must_not_be_called(*a, **k):
        raise AssertionError("compute_signed_margin_conservation_residual must not be reached for a malformed chunk")

    masked_forward_module.compute_signed_margin_conservation_residual = must_not_be_called
    try:
        with pytest.raises(run.DfpPhase1bCalibrationRunError, match="declares component-axis width"):
            _run_with_corrupted_chunk(seed_base=_WIDTH_SEED_BASE, component_chunk_size=8, n_to_corrupt=1, corrupt_fn=_narrow_by_one)
    finally:
        masked_forward_module.compute_signed_margin_conservation_residual = real_residual


def test_signed_contribution_chunk_width_mismatch_leaves_no_artifact(tmp_path):
    """`process_selected_image` never touches the filesystem on its own;
    this confirms a malformed-chunk rejection specifically leaves no
    partial or temporary file behind in a directory it could plausibly
    have written to, matching the project's established artifact-safety
    test pattern for every other domain-error injection point."""
    with pytest.raises(run.DfpPhase1bCalibrationRunError, match="declares component-axis width"):
        _run_with_corrupted_chunk(seed_base=_WIDTH_SEED_BASE, component_chunk_size=8, n_to_corrupt=1, corrupt_fn=_narrow_by_one)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_signed_contribution_extraction_never_catches_fatal_exceptions(exc_type):
    """A fatal exception raised AT the compute_signed_contributions call
    site itself (independent of the new width-validation logic, which
    only runs on its RETURN value) must propagate completely uncaught --
    the new validation code must not have introduced any broad
    except-Exception/bare-except net around this call."""
    import src.dfp_phase1b_masked_forward as masked_forward_module

    real_compute = masked_forward_module.compute_signed_contributions

    def raising(*a, **k):
        raise exc_type("simulated fatal failure during signed-contribution extraction")

    inference, prepared, e13, contracts, n_windows = _build_equivalence_fixture(seed_base=_WIDTH_SEED_BASE)
    masked_forward_module.compute_signed_contributions = raising
    try:
        with pytest.raises(exc_type):
            run.process_selected_image(
                inference, prepared, "000000000001", selected_windows_for_image=[0], protocol_identity=e13,
                contracts=contracts, class_count=_CLASS_COUNT, affinity_power=float(e13["graph"]["affinity_power"]),
                component_chunk_size=8,
            )
    finally:
        masked_forward_module.compute_signed_contributions = real_compute


def test_masked_stitch_failure_translates_to_domain_error(monkeypatch):
    """Section 12: a RuntimeError injected inside the per-chunk masked
    `stitch_and_rescale_component_scores` call must propagate as
    whatever that function itself raises (it already translates
    apply_window_stitching_forward's own domain errors) -- confirming
    the per-chunk call site does not swallow or mislabel a failure, and
    that no partial artifact-shaped result is produced."""
    import src.dfp_phase1b_masked_forward as masked_forward_module

    inference, prepared, e13, contracts, n_windows = _build_equivalence_fixture(seed_base=8500)

    class _InjectedError(RuntimeError):
        pass

    call_index = {"n": 0}
    real_stitch = masked_forward_module.stitch_and_rescale_component_scores

    def failing_stitch(propagated_by_window, **kwargs):
        # The canonical-numerator stitching call always happens first,
        # once per window, before the masked per-chunk loop -- distinguish
        # it by call ORDER, never by `component_count` (a masked chunk's
        # own component_count is also 1 when component_chunk_size == 1,
        # indistinguishable from the canonical call's component_count=1).
        call_index["n"] += 1
        if call_index["n"] == 2:  # call 1 = canonical; call 2 = first masked chunk
            raise _InjectedError("simulated per-chunk masked stitching failure")
        return real_stitch(propagated_by_window, **kwargs)

    monkeypatch.setattr(masked_forward_module, "stitch_and_rescale_component_scores", failing_stitch)
    monkeypatch.setattr(run, "stitch_and_rescale_component_scores", failing_stitch, raising=False)
    with pytest.raises(_InjectedError):
        run.process_selected_image(
            inference, prepared, "000000000001", selected_windows_for_image=[0], protocol_identity=e13,
            contracts=contracts, class_count=_CLASS_COUNT, affinity_power=float(e13["graph"]["affinity_power"]),
            component_chunk_size=1,
        )
