"""Direct CPU coverage of the E3 dataset-task resolution used by the
bounded k11/k12 stability gate's real-inference construction
(``diagnostics.run_k11_k12_stability._build_inference``).

Regression coverage for the GPU-only integration defect where the dataset
config lookup key was hardcoded as the literal ``"stuff"`` instead of read
from the E3 identity's own ``dataset.task`` field (the real key is
``"coco_stuff"``), causing ``build_seg_dataset(None)`` -> a ``TypeError``
deep inside ``mmcv.Config.fromfile``. Every test here uses a *synthetic*
task name distinct from both ``"stuff"`` and ``"coco_stuff"`` wherever
possible, so passing proves genuine dynamic resolution rather than merely
happening to accept the canonical value.

Importing ``diagnostics.run_k11_k12_stability`` itself requires no mmcv,
mmseg, or CUDA (its heavy dependencies are all lazy, function-local
imports inside ``_build_inference``); only the ``_build_inference``
orchestration tests below need those entry points, and they use fully
synthetic stand-ins rather than the real mmcv/model packages.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PY = "/scratch/haree/venv/talk2dino-a100/bin/python"

spec = importlib.util.spec_from_file_location(
    "diagnostics.run_k11_k12_stability", ROOT / "diagnostics/run_k11_k12_stability.py"
)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

from src.k11_k12_stability_gate_identity import K11K12StabilityGateError


class FakeEvaluate(dict):
    """Stand-in for an OmegaConf ``cfg.evaluate`` DictConfig: supports
    ``in`` and ``.get`` exactly like the real object does."""


class FakeCfg:
    def __init__(self, evaluate):
        self.evaluate = evaluate
        self.model = object()


def e3_identity(task="synthetic_task_xyz", config_path="configs/synthetic_dataset.py"):
    return {"dataset": {"task": task, "config_path": config_path}}


# ---------------------------------------------------------------------------
# resolve_e3_dataset_config_path: pure resolver, direct CPU tests
# ---------------------------------------------------------------------------


def test_synthetic_task_resolves_dynamically():
    identity = e3_identity(task="a_totally_made_up_task_name", config_path="configs/made_up.py")
    cfg = FakeCfg(FakeEvaluate({"a_totally_made_up_task_name": "configs/made_up.py"}))
    resolved = runner.resolve_e3_dataset_config_path(identity, cfg)
    assert resolved == "configs/made_up.py"


def test_canonical_e3_task_resolves():
    identity = e3_identity(task="coco_stuff", config_path="src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/stuff.py")
    cfg = FakeCfg(FakeEvaluate({"coco_stuff": "src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/stuff.py"}))
    resolved = runner.resolve_e3_dataset_config_path(identity, cfg)
    assert resolved == "src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/stuff.py"


def test_no_literal_stuff_assumption():
    # The literal "stuff" key is deliberately ABSENT from cfg.evaluate here
    # -- if the resolver still hardcoded "stuff" anywhere, this would
    # raise (key not found) even though the correct key is present.
    identity = e3_identity(task="totally_different_key", config_path="configs/x.py")
    cfg = FakeCfg(FakeEvaluate({"totally_different_key": "configs/x.py"}))
    resolved = runner.resolve_e3_dataset_config_path(identity, cfg)
    assert resolved == "configs/x.py"
    assert "stuff" not in cfg.evaluate


def test_missing_dataset_task_key_fails():
    identity = {"dataset": {"config_path": "configs/x.py"}}
    cfg = FakeCfg(FakeEvaluate({"anything": "configs/x.py"}))
    with pytest.raises(KeyError):
        runner.resolve_e3_dataset_config_path(identity, cfg)


def test_empty_task_fails():
    identity = e3_identity(task="", config_path="configs/x.py")
    cfg = FakeCfg(FakeEvaluate({"": "configs/x.py"}))
    with pytest.raises(K11K12StabilityGateError, match="non-empty string"):
        runner.resolve_e3_dataset_config_path(identity, cfg)


@pytest.mark.parametrize("bad_task", [True, False, 123, 1.5, None, ["task"]])
def test_non_string_task_fails(bad_task):
    identity = e3_identity(task=bad_task, config_path="configs/x.py")
    cfg = FakeCfg(FakeEvaluate({}))
    with pytest.raises(K11K12StabilityGateError, match="non-empty string"):
        runner.resolve_e3_dataset_config_path(identity, cfg)


def test_task_absent_from_cfg_evaluate_fails():
    identity = e3_identity(task="present_in_identity_only", config_path="configs/x.py")
    cfg = FakeCfg(FakeEvaluate({"some_other_task": "configs/y.py"}))
    with pytest.raises(K11K12StabilityGateError, match="not present"):
        runner.resolve_e3_dataset_config_path(identity, cfg)


def test_resolved_value_none_fails():
    identity = e3_identity(task="task_with_none_value", config_path="configs/x.py")
    cfg = FakeCfg(FakeEvaluate({"task_with_none_value": None}))
    with pytest.raises(K11K12StabilityGateError, match="non-string/empty"):
        runner.resolve_e3_dataset_config_path(identity, cfg)


@pytest.mark.parametrize("bad_value", [123, 1.5, True, ["configs/x.py"], {}])
def test_resolved_value_non_string_fails(bad_value):
    identity = e3_identity(task="task_with_bad_value", config_path="configs/x.py")
    cfg = FakeCfg(FakeEvaluate({"task_with_bad_value": bad_value}))
    with pytest.raises(K11K12StabilityGateError, match="non-string/empty"):
        runner.resolve_e3_dataset_config_path(identity, cfg)


def test_resolved_path_disagreeing_with_identity_fails():
    identity = e3_identity(task="mismatched_task", config_path="configs/expected.py")
    cfg = FakeCfg(FakeEvaluate({"mismatched_task": "configs/DIFFERENT.py"}))
    with pytest.raises(K11K12StabilityGateError, match="disagrees with the registered E3 identity") as excinfo:
        runner.resolve_e3_dataset_config_path(identity, cfg)
    assert "configs/expected.py" in str(excinfo.value)
    assert "configs/DIFFERENT.py" in str(excinfo.value)


def test_e3_identity_and_cfg_not_mutated():
    identity = e3_identity(task="stable_task", config_path="configs/stable.py")
    cfg = FakeCfg(FakeEvaluate({"stable_task": "configs/stable.py"}))
    import copy
    identity_before = copy.deepcopy(identity)
    evaluate_before = dict(cfg.evaluate)
    runner.resolve_e3_dataset_config_path(identity, cfg)
    assert identity == identity_before
    assert dict(cfg.evaluate) == evaluate_before


# ---------------------------------------------------------------------------
# _build_inference orchestration: both consumers get the identical resolved
# path, and resolution failure happens before model/checkpoint construction.
# Uses fully synthetic stand-ins for utils/models/segmentation.evaluation/
# mmcv -- never the real heavy packages.
# ---------------------------------------------------------------------------


def _install_inference_stubs(*, evaluate_map, model_calls, dataset_calls, inference_calls, checkpoint_calls):
    saved = {name: sys.modules.get(name) for name in (
        "utils", "utils.config", "models", "segmentation", "segmentation.evaluation", "mmcv", "mmcv.runner",
    )}

    utils_mod = types.ModuleType("utils")
    utils_mod.__path__ = []
    utils_config_mod = types.ModuleType("utils.config")

    def fake_load_config(path):
        return FakeCfg(FakeEvaluate(evaluate_map))

    utils_config_mod.load_config = fake_load_config
    sys.modules["utils"] = utils_mod
    sys.modules["utils.config"] = utils_config_mod

    models_mod = types.ModuleType("models")
    models_mod.__path__ = []

    class FakeModel:
        def load_state_dict(self, state_dict, strict=False):
            pass

        def cuda(self):
            pass

        def eval(self):
            pass

    def fake_build_model(model_cfg):
        model_calls.append(model_cfg)
        return FakeModel()

    models_mod.build_model = fake_build_model
    sys.modules["models"] = models_mod

    segmentation_mod = types.ModuleType("segmentation")
    segmentation_mod.__path__ = []
    segmentation_eval_mod = types.ModuleType("segmentation.evaluation")

    class FakeDataset:
        pass

    def fake_build_seg_dataset(path):
        dataset_calls.append(path)
        return FakeDataset()

    class FakeInference:
        def reset_evaluation_state(self):
            pass

    def fake_build_dinotext_seg_inference(model, dataset, cfg, seg_config):
        inference_calls.append(seg_config)
        return FakeInference()

    segmentation_eval_mod.build_seg_dataset = fake_build_seg_dataset
    segmentation_eval_mod.build_dinotext_seg_inference = fake_build_dinotext_seg_inference
    sys.modules["segmentation"] = segmentation_mod
    sys.modules["segmentation.evaluation"] = segmentation_eval_mod

    mmcv_mod = types.ModuleType("mmcv")
    mmcv_mod.__path__ = []
    mmcv_runner_mod = types.ModuleType("mmcv.runner")

    class FakeCheckpointLoader:
        @staticmethod
        def load_checkpoint(path, map_location=None):
            checkpoint_calls.append(path)
            return {"state_dict": {}}

    mmcv_runner_mod.CheckpointLoader = FakeCheckpointLoader
    sys.modules["mmcv"] = mmcv_mod
    sys.modules["mmcv.runner"] = mmcv_runner_mod

    return saved


def _restore_stubs(saved):
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


@pytest.fixture
def fake_checkpoint_file(tmp_path):
    path = tmp_path / "fake_checkpoint.pt"
    path.write_bytes(b"not a real checkpoint, just needs to exist for sha256")
    return path


def test_build_seg_dataset_receives_resolved_path_exactly(tmp_path, fake_checkpoint_file):
    identity = {
        "evaluation": {"config_path": "configs/eval.yml"},
        "dataset": {"task": "synthetic_orchestration_task", "config_path": "configs/orchestration_dataset.py"},
        "projection": {"checkpoint_path": str(fake_checkpoint_file.relative_to(tmp_path))},
    }
    model_calls, dataset_calls, inference_calls, checkpoint_calls = [], [], [], []
    saved = _install_inference_stubs(
        evaluate_map={"synthetic_orchestration_task": "configs/orchestration_dataset.py"},
        model_calls=model_calls, dataset_calls=dataset_calls,
        inference_calls=inference_calls, checkpoint_calls=checkpoint_calls,
    )
    try:
        inference, dataset = runner._build_inference(tmp_path, identity, "cpu")
    finally:
        _restore_stubs(saved)
    assert dataset_calls == ["configs/orchestration_dataset.py"]


def test_build_dinotext_seg_inference_receives_same_resolved_path(tmp_path, fake_checkpoint_file):
    identity = {
        "evaluation": {"config_path": "configs/eval.yml"},
        "dataset": {"task": "synthetic_orchestration_task", "config_path": "configs/orchestration_dataset.py"},
        "projection": {"checkpoint_path": str(fake_checkpoint_file.relative_to(tmp_path))},
    }
    model_calls, dataset_calls, inference_calls, checkpoint_calls = [], [], [], []
    saved = _install_inference_stubs(
        evaluate_map={"synthetic_orchestration_task": "configs/orchestration_dataset.py"},
        model_calls=model_calls, dataset_calls=dataset_calls,
        inference_calls=inference_calls, checkpoint_calls=checkpoint_calls,
    )
    try:
        runner._build_inference(tmp_path, identity, "cpu")
    finally:
        _restore_stubs(saved)
    assert inference_calls == ["configs/orchestration_dataset.py"]
    # both consumers received the IDENTICAL resolved path
    assert dataset_calls == inference_calls


def test_resolution_failure_occurs_before_model_and_checkpoint_construction(tmp_path, fake_checkpoint_file):
    # dataset task deliberately absent from cfg.evaluate -> resolver must
    # raise before build_model / CheckpointLoader are ever invoked.
    identity = {
        "evaluation": {"config_path": "configs/eval.yml"},
        "dataset": {"task": "task_not_in_cfg", "config_path": "configs/orchestration_dataset.py"},
        "projection": {"checkpoint_path": str(fake_checkpoint_file.relative_to(tmp_path))},
    }
    model_calls, dataset_calls, inference_calls, checkpoint_calls = [], [], [], []
    saved = _install_inference_stubs(
        evaluate_map={"some_other_task": "configs/orchestration_dataset.py"},
        model_calls=model_calls, dataset_calls=dataset_calls,
        inference_calls=inference_calls, checkpoint_calls=checkpoint_calls,
    )
    try:
        with pytest.raises(K11K12StabilityGateError, match="not present"):
            runner._build_inference(tmp_path, identity, "cpu")
    finally:
        _restore_stubs(saved)
    assert model_calls == []
    assert checkpoint_calls == []
    assert dataset_calls == []


# ---------------------------------------------------------------------------
# CLI-level failure contract: concise, nonzero, no traceback
# ---------------------------------------------------------------------------


def test_cli_dry_run_manifest_still_works_after_the_fix():
    # sanity: the fix must not have broken argument parsing.
    result = subprocess.run(
        [PY, str(ROOT / "diagnostics/run_k11_k12_stability.py"), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Traceback" not in result.stderr


def test_dataset_task_resolution_error_is_a_K11K12StabilityGateError():
    # The specific exception type resolve_e3_dataset_config_path raises on
    # failure is exactly the type main()'s top-level handler catches and
    # converts to exit code 2 with a concise stderr message (verified
    # structurally below) -- never an uncaught traceback.
    identity = e3_identity(task="task_not_in_cfg", config_path="configs/x.py")
    cfg = FakeCfg(FakeEvaluate({"some_other_task": "configs/orchestration_dataset.py"}))
    with pytest.raises(K11K12StabilityGateError):
        runner.resolve_e3_dataset_config_path(identity, cfg)


def test_main_catches_K11K12StabilityGateError_from_run_gate_as_exit_2_no_traceback(capsys):
    # Directly exercises main()'s real exception-handling contract: ANY
    # K11K12StabilityGateError raised inside run_gate() (which is exactly
    # what a dataset-task resolution failure raises) becomes exit code 2
    # with a concise stderr message, never an uncaught traceback.
    orig_run_gate = runner.run_gate

    def failing_run_gate(args):
        raise K11K12StabilityGateError("synthetic dataset-task resolution failure for this test")

    runner.run_gate = failing_run_gate
    try:
        exit_code = runner.main([
            "--repo-root", str(ROOT), "--output", "/tmp/does-not-matter.json",
            "--checkpoint", "/tmp/does-not-matter-checkpoint.json", "--device", "cpu",
        ])
    finally:
        runner.run_gate = orig_run_gate

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Traceback" not in captured.err
    assert "K11/K12 STABILITY GATE FAIL" in captured.err
    assert "synthetic dataset-task resolution failure" in captured.err
