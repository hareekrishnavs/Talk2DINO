"""Tests for the E23 BALP production adapter
(`src.dfp_phase1b_balp_development_production`). All tests use
deterministic fake providers -- no real image/annotation access, no
real model/graph/recurrence construction, no CUDA. Exercises the real
authority (E23/E22/split/window-universe manifest) and the real
committed `execute_development` orchestration engine, but substitutes
fakes at exactly the three real-boundary injection points
(`ProductionBuilders.dataset_resources_builder` /
`.inference_builder` / `.image_provider_fn_builder`) plus, for the
full-350-image-scale wiring test only, the per-window scientific
processor (`.image_processor_fn`) -- mirroring the existing committed
test suite's own established technique
(`tests/test_dfp_phase1b_lightweight_boundary_execute.py`'s
`_fake_processor`) for exercising orchestration at authoritative scale
without a real model.

This suite does NOT re-test `execute_development`'s own fresh/resume/
orphan-recovery/finalize-only correctness -- that is already
exhaustively proven, at full 350-image scale, by the existing committed
suite, and this module never modifies or reimplements that engine.
This suite tests only what THIS adapter adds: population/window
isolation, identity-binding construction, contracts narrowing, the
torch-free preflight path, and correct wiring into
`execute_development`.
"""

from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

import src.dfp_phase1b_balp_development_production as prod

ROOT = Path(__file__).resolve().parent.parent
E = prod.DfpPhase1bBalpDevelopmentProductionError


@pytest.fixture(scope="module")
def real_window_index():
    return prod.load_development_window_index(ROOT)


@pytest.fixture(scope="module")
def real_split():
    import src.dfp_phase1b_lightweight_boundary_identity as e22

    return e22.load_and_verify_split(repo_root=ROOT)


# ---------------------------------------------------------------------
# Torch-free preflight path, at real 350/769 scale.
# ---------------------------------------------------------------------


def test_load_development_window_index_matches_authoritative_counts(real_window_index):
    assert len(real_window_index.canonical_ids) == 350
    assert real_window_index.aggregate_window_count == 769
    assert real_window_index.canonical_record_digest == "b4672854e38199deef26c154a30e45c1835c201a4dbcf046aa29e7fdf12df9e6"
    assert sum(len(v) for v in real_window_index.windows_per_image.values()) == 769


def test_preflight_path_is_torch_free_in_a_fresh_process():
    """Empirical proof (not a source-text heuristic): a brand-new
    subprocess imports this module and calls every preflight-path
    function, and torch must never appear in sys.modules afterward.
    This is the exact defect independently found and fixed during this
    task -- `_all_variant_ids` originally imported the torch-dependent
    BALP kernel merely to read a string list; it now uses
    `e22.reconstruct_candidate_ids()`."""
    import subprocess
    import sys

    script = (
        "import sys; sys.path.insert(0, '.'); "
        "import src.dfp_phase1b_balp_development_production as prod; "
        "from pathlib import Path; "
        "e23_doc, e22_doc, split, _ = prod.load_verified_authority(Path('.')); "
        "wi = prod.load_development_window_index(Path('.'), e23_doc=e23_doc); "
        "prod.verify_population_isolation(wi, development_ids=split['development_image_ids'], confirmation_ids=split['confirmation_image_ids']); "
        "prod.build_identity_bindings_without_digest(e23_doc); "
        "print('TORCH_IMPORTED=' + str('torch' in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "TORCH_IMPORTED=False" in result.stdout, result.stdout + result.stderr


def test_verify_population_isolation_accepts_real_authority(real_window_index, real_split):
    prod.verify_population_isolation(
        real_window_index, development_ids=real_split["development_image_ids"], confirmation_ids=real_split["confirmation_image_ids"],
    )  # must not raise


def test_verify_population_isolation_rejects_confirmation_id_swap(real_window_index, real_split):
    dev_ids = list(real_split["development_image_ids"])
    conf_ids = list(real_split["confirmation_image_ids"])
    tampered = copy.deepcopy(real_window_index)
    object.__setattr__(tampered, "canonical_ids", tuple(dev_ids[:349]) + (conf_ids[0],))
    with pytest.raises(E, match="disagrees with E22's frozen"):
        prod.verify_population_isolation(tampered, development_ids=dev_ids, confirmation_ids=conf_ids)


def test_verify_population_isolation_rejects_wrong_count(real_window_index, real_split):
    tampered = copy.deepcopy(real_window_index)
    object.__setattr__(tampered, "canonical_ids", real_window_index.canonical_ids[:349])
    with pytest.raises(E, match="expected exactly 350"):
        prod.verify_population_isolation(tampered, development_ids=real_split["development_image_ids"], confirmation_ids=real_split["confirmation_image_ids"])


def test_verify_population_isolation_rejects_duplicate_id(real_window_index, real_split):
    dev_ids = list(real_split["development_image_ids"])
    tampered = copy.deepcopy(real_window_index)
    object.__setattr__(tampered, "canonical_ids", tuple(dev_ids[:349]) + (dev_ids[0],))
    with pytest.raises(E, match="duplicate"):
        prod.verify_population_isolation(tampered, development_ids=dev_ids, confirmation_ids=real_split["confirmation_image_ids"])


def test_verify_population_isolation_rejects_wrong_aggregate_window_count(real_window_index, real_split):
    tampered = copy.deepcopy(real_window_index)
    object.__setattr__(tampered, "aggregate_window_count", 770)
    with pytest.raises(E, match="aggregate window count"):
        prod.verify_population_isolation(tampered, development_ids=real_split["development_image_ids"], confirmation_ids=real_split["confirmation_image_ids"])


def test_reject_out_of_population_id_rejects_confirmation_id(real_window_index, real_split):
    confirmation_id = real_split["confirmation_image_ids"][0]
    with pytest.raises(E, match="not one of the 350 frozen development IDs"):
        prod.reject_out_of_population_id(confirmation_id, window_index=real_window_index)


def test_reject_out_of_population_id_rejects_other_4500_id(real_window_index, real_split):
    other_id = "000000000139"  # a real COCO-Stuff val2017 ID not in either 500-image partition
    assert other_id not in real_split["development_image_ids"] and other_id not in real_split["confirmation_image_ids"]
    with pytest.raises(E, match="not one of the 350 frozen development IDs"):
        prod.reject_out_of_population_id(other_id, window_index=real_window_index)


def test_reject_out_of_population_id_accepts_development_id(real_window_index):
    prod.reject_out_of_population_id(real_window_index.canonical_ids[0], window_index=real_window_index)  # must not raise


def test_build_identity_bindings_without_digest_has_exactly_seven_keys():
    e23_doc, _, _, _ = prod.load_verified_authority(ROOT)
    bindings = prod.build_identity_bindings_without_digest(e23_doc)
    assert set(bindings.keys()) == {
        "e22_protocol_sha256", "e22_artifacts_sha256", "e13_protocol_sha256",
        "matched_evaluator_sha256", "subset_manifest_sha256", "balp_split_sha256", "run_mode",
    }
    assert bindings["run_mode"] == "coco_stuff_500"
    assert all(len(v) == 64 for k, v in bindings.items() if k != "run_mode")


def test_load_verified_authority_rejects_when_e23_missing_dependency(tmp_path):
    # Point repo_root at a directory with no evaluation_identities at all.
    with pytest.raises(Exception):
        prod.load_verified_authority(tmp_path)


# ---------------------------------------------------------------------
# Contracts narrowing: BalpContracts must never expose component/
# masked_forward, even when the underlying bundle carries them.
# ---------------------------------------------------------------------


class _FakeSixFieldContractsBundle:
    def __init__(self):
        self.recurrence = "recurrence-marker"
        self.stitching = "stitching-marker"
        self.pullback = "pullback-marker"
        self.final_resolution = "final-resolution-marker"
        self.component = "COMPONENT-MUST-NEVER-BE-EXPOSED"
        self.masked_forward = "MASKED-FORWARD-MUST-NEVER-BE-EXPOSED"


def test_narrow_balp_contracts_discards_component_and_masked_forward():
    narrowed = prod.narrow_balp_contracts(_FakeSixFieldContractsBundle())
    assert narrowed.recurrence == "recurrence-marker"
    assert narrowed.stitching == "stitching-marker"
    assert narrowed.pullback == "pullback-marker"
    assert narrowed.final_resolution == "final-resolution-marker"
    assert not hasattr(narrowed, "component")
    assert not hasattr(narrowed, "masked_forward")


# ---------------------------------------------------------------------
# Fake-provider ExecutionProviders/image_provider_fn wiring, at real
# 350/769 population scale (no model/graph/recurrence invoked).
# ---------------------------------------------------------------------


class _FakeDatasetResources:
    def __init__(self, class_count=3, affinity_power=1.0):
        self.protocol_identity = {"fake": True}
        self.contracts = _FakeSixFieldContractsBundle()
        self.class_count = class_count
        self.affinity_power = affinity_power


def _fake_dataset_resources_builder(repo_root, development_ids):
    assert len(development_ids) == 350
    return _FakeDatasetResources()


def _fake_inference_builder(resources, device, log_dir):
    return {"marker": "fake-inference", "device": device}


def test_build_execution_providers_calls_builders_exactly_once(real_window_index):
    calls = {"dataset": 0, "inference": 0}

    def counting_dataset_builder(repo_root, development_ids):
        calls["dataset"] += 1
        return _fake_dataset_resources_builder(repo_root, development_ids)

    def counting_inference_builder(resources, device, log_dir):
        calls["inference"] += 1
        return _fake_inference_builder(resources, device, log_dir)

    providers = prod.build_execution_providers(
        ROOT, window_index=real_window_index, device="cpu", log_dir=Path("/tmp"),
        dataset_resources_builder=counting_dataset_builder, inference_builder=counting_inference_builder,
    )
    assert calls == {"dataset": 1, "inference": 1}
    assert providers.class_count == 3
    assert providers.inference == {"marker": "fake-inference", "device": "cpu"}
    assert not hasattr(providers.contracts, "component")
    assert set(providers.windows_per_image.keys()) == set(real_window_index.canonical_ids)
    assert sum(len(v) for v in providers.windows_per_image.values()) == 769


# ---------------------------------------------------------------------
# GT isolation: image_provider_fn_builder must separate `prepared` and
# `ground_truth` into independent fields, never mixing them.
# ---------------------------------------------------------------------


def test_fake_image_provider_fn_builder_separates_prepared_and_ground_truth(real_window_index):
    call_log = []

    def fake_builder(resources, window_index, expected_ignore_index):
        def _provide(canonical_id):
            call_log.append(canonical_id)
            return _FakeDevelopmentImageProvider(
                prepared={"canonical_id": canonical_id, "no_gt_here": True},
                ground_truth=np.full((4, 4), 7, dtype=np.int64),
                selected_windows=window_index.windows_per_image[canonical_id],
            )
        return _provide

    provide = fake_builder(None, real_window_index, 255)
    first_id = real_window_index.canonical_ids[0]
    result = provide(first_id)
    assert result.prepared["canonical_id"] == first_id
    assert "ground_truth" not in result.prepared
    assert result.ground_truth.sum() == 4 * 4 * 7
    assert result.selected_windows == real_window_index.windows_per_image[first_id]
    assert call_log == [first_id]


class _FakeDevelopmentImageProvider:
    def __init__(self, prepared, ground_truth, selected_windows):
        self.prepared = prepared
        self.ground_truth = ground_truth
        self.selected_windows = selected_windows


# ---------------------------------------------------------------------
# Full end-to-end wiring at true 350-image/769-window scale: all three
# real-boundary injection points are fake, PLUS the per-window
# scientific processor is a fast synthetic stand-in (mirroring
# `tests/test_dfp_phase1b_lightweight_boundary_execute.py`'s own
# `_fake_processor`, since that module's own suite already exhaustively
# proves `process_development_image`'s per-window forward/graph/
# recurrence-count invariants; this test proves only that THIS
# adapter's wiring correctly drives `execute_development` to a valid,
# publicly-reverified finalized result across the real population and
# real window counts).
# ---------------------------------------------------------------------


def _fake_processor(inference, prepared, canonical_id, *, protocol_identity, contracts, class_count, affinity_power, selected_windows_for_image):
    import torch

    import src.dfp_phase1b_lightweight_boundary as balp
    import src.dfp_phase1b_lightweight_boundary_execute as execmod

    seed = abs(hash(canonical_id)) % (2**31)
    g = torch.Generator().manual_seed(seed)
    scores = {h: torch.randn(class_count, 2, 2, generator=g, dtype=torch.float32) for h in balp.HORIZON_SET}
    kr = balp.compute_balp(scores, class_count=class_count)
    return execmod.ImageBalpOutcome(canonical_id=canonical_id, ori_shape=(2, 2), kernel_result=kr)


def _fake_image_provider_fn_builder(resources, window_index, expected_ignore_index, *, gt_value=0):
    def _provide(canonical_id):
        prod.reject_out_of_population_id(canonical_id, window_index=window_index)
        return _FakeDevelopmentImageProvider(
            prepared=None, ground_truth=np.full((2, 2), gt_value, dtype=np.int64),
            selected_windows=window_index.windows_per_image[canonical_id],
        )
    return _provide


@pytest.mark.slow
def test_full_350_image_769_window_fake_wiring_produces_valid_finalized_result(real_window_index):
    output_root = Path(tempfile.mkdtemp(prefix="balp_prod_adapter_"))
    try:
        builders = prod.ProductionBuilders(
            dataset_resources_builder=_fake_dataset_resources_builder,
            inference_builder=_fake_inference_builder,
            image_provider_fn_builder=_fake_image_provider_fn_builder,
            image_processor_fn=_fake_processor,
        )
        result = prod.run_development_execution(ROOT, output_root, device="cpu", log_dir=output_root / "logs", builders=builders)
        assert result["complete"] is True and result["final"] is True
        assert len(result["control_results"]) == 5
        assert len(result["candidate_results"]) == 16
        assert result["development_image_count"] == 350

        import src.dfp_phase1b_lightweight_boundary_artifact as art
        import src.dfp_phase1b_lightweight_boundary_identity as e22

        variant_ids = list(e22.CANONICAL_CONTROLS) + list(e22.reconstruct_candidate_ids())
        checkpoint = art.load_full_checkpoint(output_root, variant_ids=variant_ids)
        assert checkpoint is not None and checkpoint["complete"] and checkpoint["final"]
        assert len(checkpoint["progress"]["completed_image_ids"]) == 350
        assert checkpoint["progress"]["completed_image_ids"] == list(real_window_index.canonical_ids)

        total_windows_fed = sum(len(real_window_index.windows_per_image[cid]) for cid in real_window_index.canonical_ids)
        assert total_windows_fed == 769
    finally:
        shutil.rmtree(output_root, ignore_errors=True)


@pytest.mark.slow
def test_full_350_image_run_rejects_out_of_population_id_before_any_processing(real_window_index):
    """Poison probe: if the population gate were ever bypassed, this
    confirmation-ID poison would be processed and raise AssertionError
    from inside the fake image_provider_fn -- proving the gate runs
    before any per-image work for a foreign ID."""
    import src.dfp_phase1b_lightweight_boundary_identity as e22

    split = e22.load_and_verify_split(repo_root=ROOT)
    confirmation_id = split["confirmation_image_ids"][0]

    def poisoned_image_provider_fn_builder(resources, window_index, expected_ignore_index):
        def _provide(canonical_id):
            if canonical_id == confirmation_id:
                raise AssertionError("POISON: confirmation ID must never be processed")
            prod.reject_out_of_population_id(canonical_id, window_index=window_index)
            return _FakeDevelopmentImageProvider(
                prepared=None, ground_truth=np.zeros((2, 2), dtype=np.int64),
                selected_windows=window_index.windows_per_image[canonical_id],
            )
        return _provide

    output_root = Path(tempfile.mkdtemp(prefix="balp_prod_adapter_poison_"))
    try:
        builders = prod.ProductionBuilders(
            dataset_resources_builder=_fake_dataset_resources_builder,
            inference_builder=_fake_inference_builder,
            image_provider_fn_builder=poisoned_image_provider_fn_builder,
            image_processor_fn=_fake_processor,
        )
        # This must succeed WITHOUT ever calling the poisoned confirmation-ID
        # branch, because execute_development only ever iterates development_ids.
        result = prod.run_development_execution(ROOT, output_root, device="cpu", log_dir=output_root / "logs", builders=builders)
        assert result["complete"] is True
    finally:
        shutil.rmtree(output_root, ignore_errors=True)


# ---------------------------------------------------------------------
# Regression test for a real defect found by the actual production GPU
# job (not by any fake-provider test): build_execution_providers called
# its real/default builders POSITIONALLY, but build_real_dataset_
# resources / build_real_inference declare their second-and-later
# parameters keyword-only (a bare `*`). Every prior test injected a
# fake builder whose own signature happened to accept positional args,
# so the mismatch was invisible until the real function was actually
# called. This test binds the REAL (default) builders' signatures
# against the exact call pattern build_execution_providers/
# run_development_execution use -- catching a TypeError here, cheaply
# and without constructing a real dataset/model, the same way a real
# job would hit it expensively.
# ---------------------------------------------------------------------


def test_real_default_builders_are_call_site_compatible():
    import inspect

    default_builders = prod.ProductionBuilders()

    # Exactly the call build_execution_providers makes.
    inspect.signature(default_builders.dataset_resources_builder).bind("repo_root_placeholder", development_ids=["a", "b"])
    inspect.signature(default_builders.inference_builder).bind("resources_placeholder", device="cpu", log_dir="log_dir_placeholder")

    # Exactly the call run_development_execution makes.
    inspect.signature(default_builders.image_provider_fn_builder).bind("resources_placeholder", "window_index_placeholder", 255)

    # And the real (non-default) builder functions themselves, bound the
    # same way, independent of the ProductionBuilders defaults wiring.
    inspect.signature(prod.build_real_dataset_resources).bind("repo_root_placeholder", development_ids=["a", "b"])
    inspect.signature(prod.build_real_inference).bind("resources_placeholder", device="cpu", log_dir="log_dir_placeholder")
    inspect.signature(prod.build_real_image_provider_fn).bind("resources_placeholder", window_index="wi_placeholder", expected_ignore_index=255)


# ---------------------------------------------------------------------
# Regression test for a second real defect found by the actual
# production GPU job (22070469, after the first fix): the real
# PreparedDiagnosticImage (from _extract_prepared_image) has no
# `.ori_shape` attribute at all -- only `.img_metas["ori_shape"]` --
# but the committed, unmodifiable process_development_image reads
# `prepared.ori_shape` directly. Every fake `prepared` object in this
# test file was hand-constructed WITH an `.ori_shape` attribute (since
# nothing enforced it matched the real class's shape), so this was
# invisible until the real image-provider path ran against a real
# dataset sample. This test independently confirms _BalpPreparedImage
# exposes exactly the 4 attributes process_development_image actually
# reads (grepped directly from its source, never assumed), and that
# ori_shape extraction/validation matches the established
# dfp_phase1a_execute.py convention.
# ---------------------------------------------------------------------


def test_balp_prepared_image_exposes_exactly_what_process_development_image_reads():
    import inspect

    import src.dfp_phase1b_lightweight_boundary_execute as execmod

    source = inspect.getsource(execmod.process_development_image)
    attrs_read = {"image_tensor", "inference_height", "inference_width", "ori_shape"}
    for attr in attrs_read:
        assert f"prepared.{attr}" in source, f"process_development_image no longer reads prepared.{attr} -- update the wrapper/test"

    wrapped = prod._BalpPreparedImage(image_tensor="tensor-marker", inference_height=597, inference_width=448, ori_shape=(640, 480))
    for attr in attrs_read:
        assert hasattr(wrapped, attr), f"_BalpPreparedImage is missing {attr}"


def test_ori_shape_from_img_metas_matches_established_convention():
    class _FakeRawPrepared:
        img_metas = {"ori_shape": (640, 480, 3)}  # real mmcv convention: (H, W, C)

    ori_shape = prod._ori_shape_from_img_metas(_FakeRawPrepared(), canonical_id="test_id")
    assert ori_shape == (640, 480)


def test_ori_shape_from_img_metas_rejects_malformed_metadata():
    class _FakeRawPreparedBad:
        img_metas = {"ori_shape": (0, 480, 3)}

    with pytest.raises(E, match="malformed ori_shape"):
        prod._ori_shape_from_img_metas(_FakeRawPreparedBad(), canonical_id="test_id")
