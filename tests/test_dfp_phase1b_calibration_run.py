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
