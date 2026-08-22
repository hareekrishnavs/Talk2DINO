"""Direct CPU coverage of the E3 dataset-task resolution used by the
bounded k11/k12 stability gate's real-inference construction
(``diagnostics.run_k11_k12_stability._build_inference``).

Regression coverage for two defects:

1. The original GPU-only integration defect where the dataset config
   lookup key was hardcoded as the literal ``"stuff"`` instead of read
   from the E3 identity's own ``dataset.task`` field (the real key is
   ``"coco_stuff"``), causing ``build_seg_dataset(None)`` -> a
   ``TypeError`` deep inside ``mmcv.Config.fromfile``.
2. Two follow-up validation gaps found by independent re-verification: a
   whitespace-only/padded task string was silently accepted, and an
   unsafe (absolute, path-traversal, or symlink-escaping) dataset config
   path was silently accepted whenever the E3 identity's own recorded
   path happened to agree with it.

Every test uses a *synthetic* task name distinct from both ``"stuff"``
and ``"coco_stuff"`` wherever possible, so passing proves genuine dynamic
resolution rather than merely happening to accept the canonical value.
Path-safety tests use temporary repository roots and temporary files
under ``/tmp`` -- never a real system path such as ``/etc/passwd``, whose
string form is only ever used as an *invalid input value*, never
accessed.

Importing ``diagnostics.run_k11_k12_stability`` itself requires no mmcv,
mmseg, or CUDA (its heavy dependencies are all lazy, function-local
imports inside ``_build_inference``); only the ``_build_inference``
orchestration tests below need those entry points, and they use fully
synthetic stand-ins rather than the real mmcv/model packages.
"""

from __future__ import annotations

import copy
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


@pytest.fixture
def synth_repo(tmp_path):
    """A temporary repository root containing one real, valid dataset
    config file at ``configs/valid.py``, plus a real subdirectory (for the
    'directory rejected' case)."""
    (tmp_path / "configs").mkdir()
    valid_file = tmp_path / "configs" / "valid.py"
    valid_file.write_text("# synthetic dataset config\n")
    (tmp_path / "configs" / "a_directory.py").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# Task validation
# ---------------------------------------------------------------------------


def test_task_clean_synthetic_value_passes():
    assert runner._validate_task_key("clean_synthetic_task", "label") == "clean_synthetic_task"


def test_task_clean_canonical_value_passes():
    assert runner._validate_task_key("coco_stuff", "label") == "coco_stuff"


@pytest.mark.parametrize("bad_task", ["", " ", "   ", "\t", "\n"])
def test_task_empty_or_whitespace_only_rejected(bad_task):
    with pytest.raises(K11K12StabilityGateError):
        runner._validate_task_key(bad_task, "e3_identity.dataset.task")


@pytest.mark.parametrize("bad_task", [" coco_stuff", "coco_stuff ", "\tcoco_stuff", "coco_stuff\n"])
def test_task_leading_or_trailing_whitespace_rejected(bad_task):
    with pytest.raises(K11K12StabilityGateError, match="whitespace"):
        runner._validate_task_key(bad_task, "e3_identity.dataset.task")


@pytest.mark.parametrize("bad_task", [True, False, 123, 1.5, None, ["task"]])
def test_task_wrong_exact_types_rejected(bad_task):
    with pytest.raises(K11K12StabilityGateError, match="non-empty string"):
        runner._validate_task_key(bad_task, "e3_identity.dataset.task")


def test_task_not_silently_stripped():
    # A padded value must FAIL, never be normalized/stripped into a
    # passing, different string.
    with pytest.raises(K11K12StabilityGateError):
        runner._validate_task_key(" coco_stuff ", "e3_identity.dataset.task")


def test_task_error_names_the_field():
    with pytest.raises(K11K12StabilityGateError, match=r"e3_identity\.dataset\.task"):
        runner._validate_task_key("", "e3_identity.dataset.task")


def test_task_error_does_not_leak_raw_multiline_value_unescaped():
    # A malicious/corrupted, whitespace-padded multiline task is rejected
    # (trailing whitespace), and must appear escaped via repr() in the
    # diagnostic -- never inserted as raw, uncontrolled newlines that
    # could forge fake log lines in captured output.
    malicious = "coco_stuff\nFAKE LOG LINE: PREFLIGHT PASS\n"
    with pytest.raises(K11K12StabilityGateError) as excinfo:
        runner._validate_task_key(malicious, "e3_identity.dataset.task")
    message = str(excinfo.value)
    assert "\n" not in message  # no literal embedded newline
    assert "\\n" in message  # present only in its escaped (repr) form


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def test_path_valid_synthetic_repository_file_passes(synth_repo):
    result = runner._validate_safe_repo_relative_file_path("configs/valid.py", "label", repo_root=synth_repo)
    assert result == "configs/valid.py"


def test_path_canonical_e3_path_passes():
    canonical = "src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/stuff.py"
    result = runner._validate_safe_repo_relative_file_path(canonical, "label", repo_root=ROOT)
    assert result == canonical


def test_path_absolute_rejected(synth_repo):
    with pytest.raises(K11K12StabilityGateError):
        runner._validate_safe_repo_relative_file_path("/etc/passwd", "label", repo_root=synth_repo)


def test_path_leading_traversal_rejected(synth_repo):
    with pytest.raises(K11K12StabilityGateError):
        runner._validate_safe_repo_relative_file_path("../../../../etc/passwd", "label", repo_root=synth_repo)


def test_path_embedded_traversal_rejected(synth_repo):
    with pytest.raises(K11K12StabilityGateError):
        runner._validate_safe_repo_relative_file_path("configs/../../etc/passwd", "label", repo_root=synth_repo)


def test_path_leading_traversal_single_rejected(synth_repo):
    with pytest.raises(K11K12StabilityGateError):
        runner._validate_safe_repo_relative_file_path("../stuff.py", "label", repo_root=synth_repo)


def test_path_backslash_traversal_rejected(synth_repo):
    with pytest.raises(K11K12StabilityGateError):
        runner._validate_safe_repo_relative_file_path("configs\\..\\..\\etc\\passwd", "label", repo_root=synth_repo)


@pytest.mark.parametrize("bad_path", [" configs/valid.py", "configs/valid.py "])
def test_path_leading_or_trailing_whitespace_rejected(bad_path, synth_repo):
    with pytest.raises(K11K12StabilityGateError, match="whitespace"):
        runner._validate_safe_repo_relative_file_path(bad_path, "label", repo_root=synth_repo)


def test_path_nul_byte_rejected(synth_repo):
    with pytest.raises(K11K12StabilityGateError):
        runner._validate_safe_repo_relative_file_path("configs/valid.py\x00.txt", "label", repo_root=synth_repo)


def test_path_missing_file_rejected(synth_repo):
    with pytest.raises(K11K12StabilityGateError):
        runner._validate_safe_repo_relative_file_path("configs/does_not_exist.py", "label", repo_root=synth_repo)


def test_path_directory_rejected(synth_repo):
    with pytest.raises(K11K12StabilityGateError):
        runner._validate_safe_repo_relative_file_path("configs/a_directory.py", "label", repo_root=synth_repo)


def test_path_external_symlink_rejected(tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "secret.py"
    outside_file.write_text("# outside the repository\n")
    symlink_path = repo_root / "configs" / "escape.py"
    symlink_path.symlink_to(outside_file)
    with pytest.raises(K11K12StabilityGateError, match="outside the repository"):
        runner._validate_safe_repo_relative_file_path("configs/escape.py", "label", repo_root=repo_root)


@pytest.mark.parametrize("bad_path", [True, False, 123, 1.5, None, ["configs/valid.py"], {}])
def test_path_wrong_exact_types_rejected(bad_path, synth_repo):
    with pytest.raises(K11K12StabilityGateError, match="non-empty string"):
        runner._validate_safe_repo_relative_file_path(bad_path, "label", repo_root=synth_repo)


def test_path_empty_string_rejected(synth_repo):
    with pytest.raises(K11K12StabilityGateError):
        runner._validate_safe_repo_relative_file_path("", "label", repo_root=synth_repo)


def test_path_whitespace_only_rejected(synth_repo):
    with pytest.raises(K11K12StabilityGateError):
        runner._validate_safe_repo_relative_file_path("   ", "label", repo_root=synth_repo)


# ---------------------------------------------------------------------------
# resolve_e3_dataset_config_path: composition of task + path validation
# ---------------------------------------------------------------------------


def test_resolver_synthetic_task_resolves_dynamically(synth_repo):
    identity = e3_identity(task="a_totally_made_up_task_name", config_path="configs/valid.py")
    cfg = FakeCfg(FakeEvaluate({"a_totally_made_up_task_name": "configs/valid.py"}))
    resolved = runner.resolve_e3_dataset_config_path(identity, cfg, repo_root=synth_repo)
    assert resolved == "configs/valid.py"


def test_resolver_no_literal_stuff_assumption(synth_repo):
    identity = e3_identity(task="totally_different_key", config_path="configs/valid.py")
    cfg = FakeCfg(FakeEvaluate({"totally_different_key": "configs/valid.py"}))
    resolved = runner.resolve_e3_dataset_config_path(identity, cfg, repo_root=synth_repo)
    assert resolved == "configs/valid.py"
    assert "stuff" not in cfg.evaluate


def test_resolver_missing_dataset_task_key_fails(synth_repo):
    identity = {"dataset": {"config_path": "configs/valid.py"}}
    cfg = FakeCfg(FakeEvaluate({"anything": "configs/valid.py"}))
    with pytest.raises(KeyError):
        runner.resolve_e3_dataset_config_path(identity, cfg, repo_root=synth_repo)


def test_resolver_task_absent_from_cfg_evaluate_fails(synth_repo):
    identity = e3_identity(task="present_in_identity_only", config_path="configs/valid.py")
    cfg = FakeCfg(FakeEvaluate({"some_other_task": "configs/valid.py"}))
    with pytest.raises(K11K12StabilityGateError, match="not present"):
        runner.resolve_e3_dataset_config_path(identity, cfg, repo_root=synth_repo)


def test_resolver_expected_unsafe_path_rejected_even_if_observed_matches(synth_repo):
    # both the identity's own config_path AND the resolved value are the
    # SAME unsafe traversal string -- a naive equality-only check would
    # accept this; both sides must be independently validated first.
    unsafe = "../../../../etc/passwd"
    identity = e3_identity(task="trav_task", config_path=unsafe)
    cfg = FakeCfg(FakeEvaluate({"trav_task": unsafe}))
    with pytest.raises(K11K12StabilityGateError):
        runner.resolve_e3_dataset_config_path(identity, cfg, repo_root=synth_repo)


def test_resolver_observed_unsafe_path_rejected_independently(synth_repo):
    # identity's config_path is safe and valid; the RESOLVED value from
    # cfg.evaluate is the unsafe one -- must still be rejected even though
    # it would fail the equality check anyway (this proves it's rejected
    # for being unsafe, not just for disagreeing).
    identity = e3_identity(task="trav_task", config_path="configs/valid.py")
    cfg = FakeCfg(FakeEvaluate({"trav_task": "../../../../etc/passwd"}))
    with pytest.raises(K11K12StabilityGateError):
        runner.resolve_e3_dataset_config_path(identity, cfg, repo_root=synth_repo)


def test_resolver_safe_but_mismatched_paths_rejected(synth_repo):
    (synth_repo / "configs" / "other.py").write_text("# a second, different, safe file\n")
    identity = e3_identity(task="mismatched_task", config_path="configs/valid.py")
    cfg = FakeCfg(FakeEvaluate({"mismatched_task": "configs/other.py"}))
    with pytest.raises(K11K12StabilityGateError, match="disagrees with the registered E3 identity") as excinfo:
        runner.resolve_e3_dataset_config_path(identity, cfg, repo_root=synth_repo)
    assert "configs/valid.py" in str(excinfo.value)
    assert "configs/other.py" in str(excinfo.value)


def test_resolver_e3_identity_and_cfg_not_mutated(synth_repo):
    identity = e3_identity(task="stable_task", config_path="configs/valid.py")
    cfg = FakeCfg(FakeEvaluate({"stable_task": "configs/valid.py"}))
    identity_before = copy.deepcopy(identity)
    evaluate_before = dict(cfg.evaluate)
    runner.resolve_e3_dataset_config_path(identity, cfg, repo_root=synth_repo)
    assert identity == identity_before
    assert dict(cfg.evaluate) == evaluate_before


def test_resolver_whitespace_only_task_rejected_end_to_end(synth_repo):
    identity = e3_identity(task="   ", config_path="configs/valid.py")
    cfg = FakeCfg(FakeEvaluate({"   ": "configs/valid.py"}))
    with pytest.raises(K11K12StabilityGateError):
        runner.resolve_e3_dataset_config_path(identity, cfg, repo_root=synth_repo)


def test_resolver_canonical_real_identity_and_config(tmp_path):
    # Loads the REAL E3 identity and REAL evaluation config (via a
    # file-path import of the real load_config, which avoids triggering
    # utils/__init__.py's mmcv-eager chain -- cv2 is not installed on this
    # login node). Never builds the real model or touches CUDA.
    from src.e3_evaluation_identity import load_identity as load_e3_identity
    from src.matched_k11_k12_identity import load_identity as load_matched_identity
    from src.k11_k12_stability_gate_identity import load_identity as load_gate_identity

    gate_identity = load_gate_identity(repo_root=ROOT)
    matched_identity = load_matched_identity(ROOT / gate_identity["parent_identity"]["matched_identity_path"], repo_root=ROOT)
    e3_id = load_e3_identity(ROOT / matched_identity["parent_identities"]["e3_identity_path"], repo_root=ROOT)

    config_spec = importlib.util.spec_from_file_location(
        "utils.config", ROOT / "src/open_vocabulary_segmentation/utils/config.py"
    )
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    cfg = config_module.load_config(str(ROOT / e3_id["evaluation"]["config_path"]))

    resolved = runner.resolve_e3_dataset_config_path(e3_id, cfg, repo_root=ROOT)
    assert resolved == e3_id["dataset"]["config_path"]
    assert resolved == "src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/stuff.py"


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
            model_calls.append("load_state_dict")

        def cuda(self):
            model_calls.append("cuda")

        def eval(self):
            model_calls.append("eval")

    def fake_build_model(model_cfg):
        model_calls.append("build_model")
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
def fake_checkpoint_file(synth_repo):
    path = synth_repo / "fake_checkpoint.pt"
    path.write_bytes(b"not a real checkpoint, just needs to exist for sha256")
    return path


def test_build_seg_dataset_receives_resolved_path_exactly(synth_repo, fake_checkpoint_file):
    identity = {
        "evaluation": {"config_path": "configs/eval.yml"},
        "dataset": {"task": "synthetic_orchestration_task", "config_path": "configs/valid.py"},
        "projection": {"checkpoint_path": str(fake_checkpoint_file.relative_to(synth_repo))},
    }
    model_calls, dataset_calls, inference_calls, checkpoint_calls = [], [], [], []
    saved = _install_inference_stubs(
        evaluate_map={"synthetic_orchestration_task": "configs/valid.py"},
        model_calls=model_calls, dataset_calls=dataset_calls,
        inference_calls=inference_calls, checkpoint_calls=checkpoint_calls,
    )
    try:
        runner._build_inference(synth_repo, identity, "cpu")
    finally:
        _restore_stubs(saved)
    assert dataset_calls == ["configs/valid.py"]


def test_build_dinotext_seg_inference_receives_same_resolved_path(synth_repo, fake_checkpoint_file):
    identity = {
        "evaluation": {"config_path": "configs/eval.yml"},
        "dataset": {"task": "synthetic_orchestration_task", "config_path": "configs/valid.py"},
        "projection": {"checkpoint_path": str(fake_checkpoint_file.relative_to(synth_repo))},
    }
    model_calls, dataset_calls, inference_calls, checkpoint_calls = [], [], [], []
    saved = _install_inference_stubs(
        evaluate_map={"synthetic_orchestration_task": "configs/valid.py"},
        model_calls=model_calls, dataset_calls=dataset_calls,
        inference_calls=inference_calls, checkpoint_calls=checkpoint_calls,
    )
    try:
        runner._build_inference(synth_repo, identity, "cpu")
    finally:
        _restore_stubs(saved)
    assert inference_calls == ["configs/valid.py"]
    # both consumers received the IDENTICAL resolved path (same original
    # safe repository-relative string, not a resolved absolute path)
    assert dataset_calls == inference_calls == ["configs/valid.py"]


@pytest.mark.parametrize(
    "bad_task,expected_evaluate_map,match",
    [
        ("task_not_in_cfg", {"some_other_task": "configs/valid.py"}, "not present"),
        ("   ", {"   ": "configs/valid.py"}, None),
    ],
)
def test_resolution_failure_occurs_before_all_downstream_operations(synth_repo, fake_checkpoint_file, bad_task, expected_evaluate_map, match):
    identity = {
        "evaluation": {"config_path": "configs/eval.yml"},
        "dataset": {"task": bad_task, "config_path": "configs/valid.py"},
        "projection": {"checkpoint_path": str(fake_checkpoint_file.relative_to(synth_repo))},
    }
    model_calls, dataset_calls, inference_calls, checkpoint_calls = [], [], [], []
    saved = _install_inference_stubs(
        evaluate_map=expected_evaluate_map,
        model_calls=model_calls, dataset_calls=dataset_calls,
        inference_calls=inference_calls, checkpoint_calls=checkpoint_calls,
    )
    try:
        with pytest.raises(K11K12StabilityGateError, match=match) if match else pytest.raises(K11K12StabilityGateError):
            runner._build_inference(synth_repo, identity, "cpu")
    finally:
        _restore_stubs(saved)
    assert model_calls == []
    assert checkpoint_calls == []
    assert dataset_calls == []
    assert inference_calls == []


def test_resolution_failure_for_unsafe_path_occurs_before_all_downstream_operations(synth_repo, fake_checkpoint_file):
    identity = {
        "evaluation": {"config_path": "configs/eval.yml"},
        "dataset": {"task": "trav_task", "config_path": "../../../../etc/passwd"},
        "projection": {"checkpoint_path": str(fake_checkpoint_file.relative_to(synth_repo))},
    }
    model_calls, dataset_calls, inference_calls, checkpoint_calls = [], [], [], []
    saved = _install_inference_stubs(
        evaluate_map={"trav_task": "../../../../etc/passwd"},
        model_calls=model_calls, dataset_calls=dataset_calls,
        inference_calls=inference_calls, checkpoint_calls=checkpoint_calls,
    )
    try:
        with pytest.raises(K11K12StabilityGateError):
            runner._build_inference(synth_repo, identity, "cpu")
    finally:
        _restore_stubs(saved)
    assert model_calls == []
    assert checkpoint_calls == []
    assert dataset_calls == []
    assert inference_calls == []


# ---------------------------------------------------------------------------
# CLI-level failure contract: concise, nonzero, no traceback
# ---------------------------------------------------------------------------


def test_cli_help_still_works_after_the_fix():
    result = subprocess.run(
        [PY, str(ROOT / "diagnostics/run_k11_k12_stability.py"), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Traceback" not in result.stderr


def test_dataset_task_resolution_error_is_a_K11K12StabilityGateError(synth_repo):
    identity = e3_identity(task="task_not_in_cfg", config_path="configs/valid.py")
    cfg = FakeCfg(FakeEvaluate({"some_other_task": "configs/valid.py"}))
    with pytest.raises(K11K12StabilityGateError):
        runner.resolve_e3_dataset_config_path(identity, cfg, repo_root=synth_repo)


def test_path_safety_error_is_a_K11K12StabilityGateError(synth_repo):
    identity = e3_identity(task="trav_task", config_path="../../../../etc/passwd")
    cfg = FakeCfg(FakeEvaluate({"trav_task": "../../../../etc/passwd"}))
    with pytest.raises(K11K12StabilityGateError):
        runner.resolve_e3_dataset_config_path(identity, cfg, repo_root=synth_repo)


def test_main_catches_K11K12StabilityGateError_from_run_gate_as_exit_2_no_traceback(capsys):
    # Directly exercises main()'s real exception-handling contract: ANY
    # K11K12StabilityGateError raised inside run_gate() (which is exactly
    # what a dataset-task/path resolution failure raises) becomes exit
    # code 2 with a concise stderr message, never an uncaught traceback.
    orig_run_gate = runner.run_gate

    def failing_run_gate(args):
        raise K11K12StabilityGateError("synthetic dataset-task/path resolution failure for this test")

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
    assert "synthetic dataset-task/path resolution failure" in captured.err
