"""Tests for `src.dfp_phase1b_calibration_production` -- the
production provider wiring for the DFP Phase 1B calibration runner's
`execute` path.

No real dataset access, no real checkpoint, no CUDA. The production
dataset/model-construction entry points
(`_build_dataset_only`/`_build_inference`/`_extract_prepared_image`,
all from `diagnostics.run_k11_k12_stability`) are monkeypatched with
small synthetic stand-ins; everything else -- E13/E15/E16/E3/matched-
geometry/artifact-schema identity loading, the REAL manifest
execution-view adapter (`build_manifest_execution_view`/
`verify_position_identity`), candidate-universe construction, the
exact-200-window selection algorithm, `process_selected_image`,
artifact assembly/install/validation -- runs for real, unmodified,
against the real committed frozen 500-image subset manifest. The
end-to-end test drives the actual `_cmd_execute` CLI function after a
real preflight, never bypassing it by constructing a runner context
directly."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
for _path in (ROOT, ROOT / "src/open_vocabulary_segmentation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import src.dfp_phase1b_calibration_artifact as art  # noqa: E402
import src.dfp_phase1b_calibration_production as prod  # noqa: E402
import src.dfp_phase1b_calibration_run as run  # noqa: E402

# `diagnostics.run_k11_k12_stability` is imported LOCALLY, inside each
# test function that needs it (never at this file's own module/
# collection scope) -- exactly matching `tests/test_dfp_phase1a_
# execute.py`'s own established convention for the identical module.
# Importing it at collection time was tried and found to cause
# cross-file interference: with it imported here, OTHER already-
# passing tests in `test_dfp_phase1a_execute.py` started failing (their
# own `monkeypatch.setattr(rk, "_extract_prepared_image", ...)` calls
# stopped taking effect) when both files were collected in the same
# session, even though each file passes cleanly alone -- a genuine
# test-isolation hazard specific to importing this particular module
# at collection time, not a defect in the production code itself.


# ---------------------------------------------------------------------------
# Deferred-import boundary.
# ---------------------------------------------------------------------------


def test_production_module_never_imports_torch_at_module_scope():
    source = Path(prod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_level_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_level_imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            module_level_imports.append(node.module)
    heavy = ("torch", "mmcv", "mmseg", "models", "segmentation", "diagnostics.run_k11_k12_stability")
    assert not any(name.startswith(heavy) for name in module_level_imports)


def test_production_module_importable_with_torch_poisoned():
    class Poison:
        def __getattr__(self, name):
            raise RuntimeError(f"POISONED: torch.{name} touched by importing the production module")

    real_torch = sys.modules.get("torch")
    sys.modules["torch"] = Poison()
    try:
        import importlib

        import src.dfp_phase1b_calibration_production as reloaded
        importlib.reload(reloaded)
    finally:
        if real_torch is not None:
            sys.modules["torch"] = real_torch
        else:
            sys.modules.pop("torch", None)


# ---------------------------------------------------------------------------
# verify_backbone_weights_present -- existence-only (no frozen digest
# exists for these two files anywhere in the committed authorities).
# ---------------------------------------------------------------------------


def test_verify_backbone_weights_present_missing(tmp_path, monkeypatch):
    import src.local_weights as local_weights

    monkeypatch.setattr(local_weights, "DEFAULT_WEIGHT_DIR", str(tmp_path))
    fake_e3 = {"model": {"backbone_checkpoint_name": "missing_backbone.pth", "clip_checkpoint_name": "missing_clip.pt"}}
    with pytest.raises(prod.DfpPhase1bCalibrationProductionError, match="not found"):
        prod.verify_backbone_weights_present(fake_e3)


def test_verify_backbone_weights_present_ok(tmp_path, monkeypatch):
    import src.local_weights as local_weights

    (tmp_path / "backbone.pth").write_bytes(b"fake-backbone")
    (tmp_path / "clip.pt").write_bytes(b"fake-clip")
    monkeypatch.setattr(local_weights, "DEFAULT_WEIGHT_DIR", str(tmp_path))
    fake_e3 = {"model": {"backbone_checkpoint_name": "backbone.pth", "clip_checkpoint_name": "clip.pt"}}
    backbone_path, clip_path = prod.verify_backbone_weights_present(fake_e3)
    assert backbone_path == tmp_path / "backbone.pth"
    assert clip_path == tmp_path / "clip.pt"


def test_verify_backbone_weights_present_never_touches_torch(tmp_path, monkeypatch):
    import src.local_weights as local_weights

    monkeypatch.setattr(local_weights, "DEFAULT_WEIGHT_DIR", str(tmp_path))
    fake_e3 = {"model": {"backbone_checkpoint_name": "x.pth", "clip_checkpoint_name": "y.pt"}}

    class Poison:
        def __getattr__(self, name):
            raise RuntimeError(f"POISONED: torch.{name}")

    real_torch = sys.modules.get("torch")
    sys.modules["torch"] = Poison()
    try:
        with pytest.raises(prod.DfpPhase1bCalibrationProductionError):
            prod.verify_backbone_weights_present(fake_e3)
    finally:
        if real_torch is not None:
            sys.modules["torch"] = real_torch
        else:
            sys.modules.pop("torch", None)


# ---------------------------------------------------------------------------
# Fake dataset / _extract_prepared_image stand-ins -- exercised through
# the REAL build_manifest_execution_view/verify_position_identity
# (never monkeypatched: these are pure, torch-free functions, and using
# the real ones is a genuine interface-compatibility proof).
# ---------------------------------------------------------------------------


class FakeDataset:
    """Mimics exactly the attributes `build_native_image_index` reads:
    `img_infos` (native on-disk order, DELIBERATELY different from
    manifest order), `img_suffix`, `img_dir`. Exposes no GT accessor at
    all -- `get_gt_seg_map_by_idx` does not exist on this object."""

    def __init__(self, native_order_ids):
        self.img_infos = [{"filename": f"{cid}.jpg"} for cid in native_order_ids]
        self.img_suffix = ".jpg"
        self.img_dir = "images/val2017"


def _make_fake_prepared_image(canonical_id: str, *, height: int, width: int, class_count: int):
    import torch

    tensor = torch.zeros(3, height, width, dtype=torch.float32)
    img_metas = {
        "ori_shape": (height, width, 3), "img_shape": (height, width, 3),
        "filename": f"{canonical_id}.jpg",
    }
    return SimpleNamespace(
        dataset_index=-1, image_id=canonical_id, image_tensor=tensor, img_metas=img_metas,
        inference_height=height, inference_width=width, source_shape_provenance="fake",
    )


# ---------------------------------------------------------------------------
# GeometryProvider / ImagePreparer unit tests against fake resources.
# ---------------------------------------------------------------------------


def _fake_resources(tmp_path, native_order_ids, geometry_by_id, *, patch_size=14, class_count=4, manifest_order_ids=None):
    from src.dfp_phase1a_dataset_view import build_manifest_execution_view

    dataset = FakeDataset(native_order_ids)
    selected_ids = manifest_order_ids if manifest_order_ids is not None else native_order_ids
    execution_view = build_manifest_execution_view(dataset, selected_ids)
    return prod.ProductionCalibrationResources(
        repo_root=ROOT, protocol_identity={"graph": {"affinity_power": 3.0}}, phase1b_identity={},
        e3_identity={}, contracts=None, class_count=class_count, affinity_power=3.0, patch_size=patch_size,
        dataset=dataset, execution_view=execution_view,
    ), geometry_by_id


def test_geometry_provider_and_image_preparer_against_fake_dataset(monkeypatch):
    ids = [f"{i:012d}" for i in range(6)]
    native_order = [ids[3], ids[0], ids[5], ids[1], ids[4], ids[2]]  # native order != manifest order
    geometry_by_id = {ids[0]: (448, 448), ids[1]: (448, 672), ids[2]: (448, 448), ids[3]: (448, 448), ids[4]: (448, 448), ids[5]: (448, 448)}
    resources, geometry_by_id = _fake_resources(None, native_order, geometry_by_id, manifest_order_ids=ids)

    # Confirm manifest position 0 maps to a NONZERO native index (the
    # exact regression this module's own docstring documents).
    position0 = resources.execution_view[0]
    assert position0.canonical_id == ids[0]
    assert position0.native_index != 0
    assert position0.relative_path == f"{ids[0]}.jpg"

    def fake_extract(dataset, native_index, *, canonical_image_id=None, image_root=None):
        assert dataset is resources.dataset
        cid = canonical_image_id.rsplit(".", 1)[0]
        h, w = geometry_by_id[cid]
        return _make_fake_prepared_image(cid, height=h, width=w, class_count=4)

    import diagnostics.run_k11_k12_stability as k11stab

    monkeypatch.setattr(k11stab, "_extract_prepared_image", fake_extract)

    geometry_provider = prod.build_production_geometry_provider(resources)
    geom0 = geometry_provider(ids[0])
    assert (geom0.inference_height, geom0.inference_width, geom0.patch_size) == (448, 448, 14)
    geom1 = geometry_provider(ids[1])
    assert (geom1.inference_height, geom1.inference_width) == (448, 672)

    image_preparer = prod.build_production_image_preparer(resources)
    prepared = image_preparer(ids[0])
    assert prepared.inference_height == 448 and prepared.inference_width == 448
    assert prepared.ori_shape == (448, 448)
    assert tuple(prepared.image_tensor.shape) == (1, 3, 448, 448)


def test_geometry_provider_unknown_id_rejected():
    resources, _ = _fake_resources(None, [f"{i:012d}" for i in range(3)], {})
    geometry_provider = prod.build_production_geometry_provider(resources)
    with pytest.raises(prod.DfpPhase1bCalibrationProductionError, match="no resolved native-dataset position"):
        geometry_provider("999999999999")


def test_geometry_and_image_preparer_never_read_gt(monkeypatch):
    """FakeDataset exposes no get_gt_seg_map_by_idx at all; prove the
    providers complete without ever needing one."""
    ids = [f"{i:012d}" for i in range(3)]
    resources, _ = _fake_resources(None, ids, {})
    assert not hasattr(resources.dataset, "get_gt_seg_map_by_idx")

    def fake_extract(dataset, native_index, *, canonical_image_id=None, image_root=None):
        cid = canonical_image_id.rsplit(".", 1)[0]
        return _make_fake_prepared_image(cid, height=448, width=448, class_count=4)

    import diagnostics.run_k11_k12_stability as k11stab

    monkeypatch.setattr(k11stab, "_extract_prepared_image", fake_extract)
    geometry_provider = prod.build_production_geometry_provider(resources)
    image_preparer = prod.build_production_image_preparer(resources)
    for cid in ids:
        geometry_provider(cid)
        image_preparer(cid)


# ---------------------------------------------------------------------------
# Full end-to-end synthetic execute() test, driven through the actual
# CLI _cmd_execute path after a REAL preflight against the real,
# committed 500-image frozen manifest. Only dataset/model construction
# entry points are faked; the exact-200 selection is exercised for
# real, then -- as an explicit, documented, TEST-ONLY authority
# override (never touching the production default) -- the number of
# windows actually carried through scientific execution is reduced via
# monkeypatching `select_calibration_windows`'s own bound default, to
# keep this test's real k12/320-step computation small and appropriate
# for a shared login node (never weakening k=12/T=320/FP64/alpha, only
# the WINDOW COUNT, and only inside this isolated test).
# ---------------------------------------------------------------------------


_TEST_WINDOW_COUNT = 1
"""Reduced to the minimum meaningful value (1): each selected window
at real production scale is already a genuine 1024-node (32x32 patch
grid, frozen), 171-class, k=12, 320-step FP64 recurrence -- exactly
the real, unweakened scientific scale -- so even a single one is a
real, non-trivial computation appropriate for a shared login node.
Proving the wiring at N=1 is exactly as valid evidence of correct
PRODUCTION ADAPTER WIRING (object identity, data flow, artifact
assembly) as N=200 would be; the scientific formulas themselves are
independently, exhaustively verified elsewhere (`tests/test_dfp_
phase1b_calibration_run.py`) at this exact same real scale."""


@pytest.fixture
def small_window_count(monkeypatch):
    """Test-only authority override: temporarily reduce the number of
    windows `select_calibration_windows` actually selects, WITHOUT
    touching the production WINDOW_COUNT constant anywhere -- restored
    automatically at the end of the test. `namespace`/`window_count`
    are KEYWORD-ONLY parameters, so their bound defaults live in the
    function's `__kwdefaults__` dict, never `__defaults__` (which holds
    only positional-parameter defaults, and is `None` here since
    `select_calibration_windows` has no purely-positional default at
    all) -- assigning `__defaults__` instead would silently do nothing
    and leave the real window_count=200 default in effect."""
    original_kwdefaults = dict(art.select_calibration_windows.__kwdefaults__)
    original_verify_kwdefaults = dict(art.verify_selection_against_universe.__kwdefaults__)
    art.select_calibration_windows.__kwdefaults__["window_count"] = _TEST_WINDOW_COUNT
    # `verify_selection_against_universe` is a SEPARATE function object
    # with its OWN independently-bound `window_count=WINDOW_COUNT`
    # keyword-only default -- patching `select_calibration_windows`'s
    # own default does not touch this one.
    art.verify_selection_against_universe.__kwdefaults__["window_count"] = _TEST_WINDOW_COUNT
    # `art.validate_calibration_artifact`'s own `expected_selected_
    # window_count != WINDOW_COUNT` check, `run.select_and_verify`'s
    # own internal consistency check, `run.assemble_calibration_
    # artifact`'s `expected_selected_window_count` field write, and
    # this module's `execute_production_calibration`'s post-check EACH
    # independently reference their OWN already-imported `WINDOW_COUNT`
    # name (a separate name binding per module, bound once at each
    # module's own import time -- patching one does not retroactively
    # change any other, since ints are immutable) -- all four must be
    # patched together for internal consistency.
    monkeypatch.setattr(art, "WINDOW_COUNT", _TEST_WINDOW_COUNT)
    monkeypatch.setattr(run, "WINDOW_COUNT", _TEST_WINDOW_COUNT)
    monkeypatch.setattr(prod, "WINDOW_COUNT", _TEST_WINDOW_COUNT)
    try:
        yield _TEST_WINDOW_COUNT
    finally:
        art.select_calibration_windows.__kwdefaults__.update(original_kwdefaults)
        art.verify_selection_against_universe.__kwdefaults__.update(original_verify_kwdefaults)


def test_end_to_end_execute_through_real_cli(tmp_path, monkeypatch, small_window_count):
    import torch

    from models.dinotext.cover_dr.graph import build_directed_topk_graph

    # 1. Real preflight to obtain the real 500-image frozen manifest order.
    preflight = run.run_preflight(output_path=tmp_path / "phase1b_conservation_calibration.json", mode="fresh")
    real_selected_ids = preflight.selected_ids
    assert len(real_selected_ids) == 500

    # 2. A fake native dataset covering every one of the 500 real IDs,
    #    in a DIFFERENT (reversed) native order.
    native_order = list(reversed(real_selected_ids))
    fake_dataset = FakeDataset(native_order)

    # 3. Deterministic per-image geometry: every image is a single
    #    flush 448x448 window (patch_grid is frozen at 32x32 regardless
    #    of canvas size). Non-square/overlapping/flush-edge/multi-window
    #    candidate geometry is already exhaustively exercised, cheaply,
    #    by `test_geometry_provider_and_image_preparer_against_fake_
    #    dataset` above and by the lower-level scientific test suite
    #    (`tests/test_dfp_phase1b_calibration_run.py`); THIS test's job
    #    is to prove the production ADAPTER WIRING (object identity,
    #    data flow, artifact assembly) at the real 1024-node/171-class
    #    scale for as few real windows as still meaningfully exercises
    #    it -- deliberately kept to exactly one selected window so this
    #    real (unweakened k=12/T=320/171-class/1024-node) computation
    #    stays appropriate for a shared login node.
    def geometry_for(cid, ordinal):
        return 448, 448

    class_count = prod._load_class_count(ROOT)  # the real frozen COCO-Stuff class count (171) -- never hardcoded here
    patch_grid = (32, 32)  # frozen production value
    n_patches = patch_grid[0] * patch_grid[1]

    def fake_extract(dataset, native_index, *, canonical_image_id=None, image_root=None):
        cid = canonical_image_id.rsplit(".", 1)[0]
        ordinal = real_selected_ids.index(cid)
        h, w = geometry_for(cid, ordinal)
        return _make_fake_prepared_image(cid, height=h, width=w, class_count=class_count)

    call_log = {"dataset_only": 0, "inference": 0, "extract": 0}

    def fake_build_dataset_only(repo_root, e3_identity):
        call_log["dataset_only"] += 1
        return object(), fake_dataset, "fake-config-path"

    def make_generate_patch_snapshot(seed_base):
        state = {"calls": 0}

        def generate_patch_snapshot(crop, text_embedding):
            call_log["extract"] += 1  # reused as a general "model touched" counter here
            idx = state["calls"]
            state["calls"] += 1
            # Two clean, noise-free block regions (top half class 0,
            # bottom half class 1) -- exactly TWO global connected
            # components, keeping masked-forward chunking trivial while
            # still producing a genuine T0/T320 dispute after real k12
            # recurrence smooths the sharp boundary. CRITICALLY, the
            # graph-construction features must be SPATIALLY CORRELATED
            # with this same block structure (each half gets its own
            # base direction plus small noise, not pure iid randomness)
            # -- k12's top-k neighbors are chosen by feature-cosine
            # similarity, never by patch position, so spatially
            # UNCORRELATED random features would make each patch's
            # neighbors an essentially random scatter across the whole
            # image; after 320 real recurrence steps that destroys all
            # spatial coherence and fragments the final argmax field
            # into hundreds of tiny connected components -- exactly the
            # scale explosion this repository's own prior verification
            # rounds already diagnosed and avoided the same way.
            base = torch.zeros(*patch_grid, class_count, dtype=torch.float64)
            base[: patch_grid[0] // 2, :, 0] = 3.0
            base[patch_grid[0] // 2 :, :, 1] = 3.0
            unary = base.reshape(n_patches, class_count)

            g = torch.Generator().manual_seed(seed_base + idx)
            direction_a = torch.rand(8, generator=g, dtype=torch.float32)
            direction_b = torch.rand(8, generator=g, dtype=torch.float32)
            feats = torch.zeros(patch_grid[0], patch_grid[1], 8, dtype=torch.float32)
            feats[: patch_grid[0] // 2, :, :] = direction_a
            feats[patch_grid[0] // 2 :, :, :] = direction_b
            feats = feats.reshape(n_patches, 8) + torch.rand(n_patches, 8, generator=g, dtype=torch.float32) * 0.02
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return SimpleNamespace(unary_scores=[unary], dino_features=[feats])

        return generate_patch_snapshot

    class FakeModel:
        def __init__(self, seed_base):
            self.generate_patch_snapshot = make_generate_patch_snapshot(seed_base)

    class FakeInference:
        def __init__(self, seed_base):
            self.model = FakeModel(seed_base)
            self.text_embedding = None

    def fake_build_inference(repo_root, e3_identity, device, *, log_dir):
        call_log["inference"] += 1
        assert device == "cuda"
        return FakeInference(seed_base=12345), fake_dataset

    import diagnostics.run_k11_k12_stability as k11stab

    monkeypatch.setattr(k11stab, "_build_dataset_only", fake_build_dataset_only)
    monkeypatch.setattr(k11stab, "_build_inference", fake_build_inference)
    monkeypatch.setattr(k11stab, "_extract_prepared_image", fake_extract)
    monkeypatch.setattr(prod, "verify_backbone_weights_present", lambda e3_identity: (Path("/fake/backbone"), Path("/fake/clip")))

    # `real_device_record` requires torch.cuda.is_available(); this
    # test never touches CUDA, so it is monkeypatched to a fixed fake
    # record -- exactly the injectable seam `src.dfp_phase1b_
    # calibration_run` already documents for this exact purpose.
    fake_device_record = {
        "device_type": "cuda", "gpu_model_name": "Fake-Test-GPU", "cuda_compute_capability": "9.0",
        "total_device_memory_bytes": 1024, "cuda_driver_version": "0.0", "cuda_runtime_version": "0.0",
    }
    # `prod.py` imported `real_device_record` by name at ITS OWN module
    # load time (`from src.dfp_phase1b_calibration_run import ...
    # real_device_record ...`); patching `run.real_device_record`
    # rebinds only `run`'s own namespace entry, never `prod`'s already-
    # bound reference to the same original function object -- patch the
    # name actually called from (`prod.real_device_record`).
    monkeypatch.setattr(prod, "real_device_record", lambda: fake_device_record)

    output_path = tmp_path / "prod_output" / "phase1b_conservation_calibration.json"
    output_path.parent.mkdir(parents=True)

    import diagnostics.run_dfp_phase1b_calibration as cli

    # A wide chunk size (rather than a small one) is used deliberately:
    # it consolidates every global component into as few masked-forward
    # chunks as possible (each chunk still runs the real, unweakened
    # 320-step k12 recurrence once) -- exact conservation is invariant
    # to chunk size (independently verified in `tests/test_dfp_
    # phase1b_calibration_run.py::test_end_to_end_chunk_size_
    # invariance`), so this keeps this real-scale, real-recurrence test
    # appropriate for a shared login node without weakening anything
    # scientific.
    exit_code = cli.main(["execute", "--output", str(output_path), "--mode", "fresh", "--device", "cuda", "--component-chunk-size", "64"])

    assert exit_code == 0, "the actual _cmd_execute CLI path did not report success"
    assert call_log["dataset_only"] == 1, "dataset must be built exactly once"
    assert call_log["inference"] == 1, "model/inference must be built exactly once"
    assert output_path.exists()

    with open(output_path, encoding="utf-8") as handle:
        document = json.load(handle)
    assert document["complete"] is True
    assert document["final"] is True
    assert len(document["selected_window_records"]) == _TEST_WINDOW_COUNT
    art.validate_calibration_artifact(document)
    print(f"end-to-end production adapter test: {call_log}")


def test_end_to_end_execute_rejects_when_output_not_fresh(tmp_path, monkeypatch, small_window_count):
    output_path = tmp_path / "phase1b_conservation_calibration.json"
    output_path.write_text("{}", encoding="utf-8")
    import diagnostics.run_dfp_phase1b_calibration as cli

    exit_code = cli.main(["execute", "--output", str(output_path), "--mode", "fresh", "--device", "cuda", "--component-chunk-size", "2"])
    assert exit_code == 2
    assert output_path.read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_execute_production_calibration_propagates_fatal_exceptions(tmp_path, monkeypatch, exc_type):
    def raiser(*a, **k):
        raise exc_type("must propagate")

    monkeypatch.setattr(prod, "build_production_dataset_resources", raiser)
    with pytest.raises(exc_type):
        prod.execute_production_calibration(
            output_path=tmp_path / "phase1b_conservation_calibration.json", device="cuda", component_chunk_size=1,
        )
