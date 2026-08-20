import importlib.util
import inspect
import copy
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rwr_reproduction_identity import load_identity


ROOT = Path(__file__).parents[1]
PACKAGE_PATH = ROOT / "src/open_vocabulary_segmentation/models/dinotext/cover_dr"


def _load_package():
    spec = importlib.util.spec_from_file_location(
        "cover_dr_integration_under_test",
        PACKAGE_PATH / "__init__.py",
        submodule_search_locations=[str(PACKAGE_PATH)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cover_dr = _load_package()
inference_module = sys.modules[f"{cover_dr.__name__}.inference"]


def _config(**overrides):
    identity = load_identity(repo_root=ROOT)
    values = {
        "enabled": identity["rwr"]["enabled"],
        "identity_path": "evaluation_identities/e3_canonical_directed_rwr.toml",
        "graph_mode": identity["rwr"]["graph_mode"],
        "alpha": 0.8,
        "top_k": 2,
        "affinity_power": identity["rwr"]["affinity_power"],
        "solver": identity["solver"]["method"],
        "solver_rtol": identity["solver"]["rtol"],
        "solver_atol": identity["solver"]["atol"],
        "solver_max_iterations": identity["solver"]["max_iterations"],
        "expected_class_count": 3,
        "config_path": identity["canonical_config_path"],
    }
    values.update(overrides)
    config = cover_dr.RWRInferenceConfig(**values)
    config.validate()
    return config


def _snapshot(dtype=torch.float32):
    features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.2, -0.1, 0.3],
                [0.6, 0.7, 0.1, -0.2],
                [-0.2, 0.8, 0.5, 0.1],
                [0.3, -0.1, 0.9, 0.6],
            ],
            dtype=dtype,
        ),
        dim=-1,
    )[None]
    scores = torch.tensor(
        [
            [0.2, -0.4, 1.1],
            [1.3, 0.1, -0.7],
            [-0.8, 0.9, 0.4],
            [0.5, -1.2, 0.3],
        ],
        dtype=dtype,
    )[None]
    return types.SimpleNamespace(
        unary_scores=scores,
        dino_features=features,
        grid_hw=(2, 2),
    )


def _canonical_config():
    identity = load_identity(repo_root=ROOT)
    config = cover_dr.RWRInferenceConfig(
        enabled=identity["rwr"]["enabled"],
        identity_path="evaluation_identities/e3_canonical_directed_rwr.toml",
        graph_mode=identity["rwr"]["graph_mode"],
        alpha=identity["rwr"]["alpha"],
        top_k=identity["rwr"]["top_k"],
        affinity_power=identity["rwr"]["affinity_power"],
        solver=identity["solver"]["method"],
        solver_rtol=identity["solver"]["rtol"],
        solver_atol=identity["solver"]["atol"],
        solver_max_iterations=identity["solver"]["max_iterations"],
        expected_class_count=identity["dataset"]["classes"],
        config_path=identity["canonical_config_path"],
    )
    config.validate()
    return config


def _record_builder_inputs():
    identity = load_identity(repo_root=ROOT)
    return {
        "config": _canonical_config(),
        "canonical_identity": identity,
        "metrics": {
            name: identity["expected_metrics"][identity_name] / 100.0
            for name, identity_name in (
                ("aAcc", "rwr_aAcc"),
                ("mIoU", "rwr_mIoU"),
                ("mAcc", "rwr_mAcc"),
            )
        },
        "image_count": identity["dataset"]["images"],
        "class_count": identity["dataset"]["classes"],
        "crop": tuple(identity["evaluation"]["crop"]),
        "stride": tuple(identity["evaluation"]["stride"]),
        "pamr": identity["evaluation"]["pamr"],
        "background_class": identity["dataset"]["background_class"],
        "checkpoint_path": identity["checkpoint"]["path"],
        "source_git_commit": identity["source_e10_commit"],
        "source_git_branch": "synthetic-branch",
        "source_git_dirty": False,
        "gpu_model": "synthetic GPU",
        "torch_version": "synthetic torch",
        "cuda_version": "synthetic CUDA",
        "elapsed_seconds": 1.0,
        "solver_summary": {
            "window_count": 1,
            "converged_window_count": 1,
            "total_iterations": 1,
            "minimum_iterations": 1,
            "maximum_iterations": 1,
            "total_restarts": 0,
            "nonzero_restart_windows": 0,
            "total_residual_replacements": 0,
            "total_fallback_rows": 0,
            "maximum_scaled_residual": 0.0,
        },
    }


def test_runtime_record_builder_accepts_exact_canonical_types():
    inputs = _record_builder_inputs()
    record = cover_dr.build_rwr_structured_record(**inputs)
    assert type(record["image_count"]) is int
    assert type(record["rwr_enabled"]) is bool
    assert type(record["alpha"]) is float
    assert type(record["crop"]) is list


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("image_count", 1.0, "image_count.*integer"),
        ("class_count", True, "class_count.*integer"),
        ("pamr", 0, "pamr.*boolean"),
        ("background_class", 0, "background_class.*boolean"),
        ("crop", [448, 448], "crop.*tuple"),
        ("crop", (448.0, 448), "crop.*integers"),
        ("source_git_dirty", 0, "source_git_dirty.*boolean"),
        ("elapsed_seconds", 1, "elapsed_seconds.*float"),
        ("elapsed_seconds", float("nan"), "elapsed_seconds.*float"),
        ("checkpoint_path", 1, "checkpoint_path.*string"),
    ],
)
def test_runtime_record_builder_rejects_coercible_wrong_types(field, value, match):
    inputs = _record_builder_inputs()
    inputs[field] = value
    with pytest.raises(ValueError, match=match):
        cover_dr.build_rwr_structured_record(**inputs)


@pytest.mark.parametrize("value", [1, float("nan"), float("inf")])
def test_runtime_record_builder_rejects_nonexact_metric_float(value):
    inputs = _record_builder_inputs()
    inputs["metrics"] = dict(inputs["metrics"])
    inputs["metrics"]["mIoU"] = value
    with pytest.raises(ValueError, match="metric mIoU.*float"):
        cover_dr.build_rwr_structured_record(**inputs)


def test_runtime_record_builder_rejects_solver_summary_alias_or_schema_coercion():
    inputs = _record_builder_inputs()
    inputs["solver_summary"] = dict(inputs["solver_summary"])
    inputs["solver_summary"]["window_count"] = 1.0
    with pytest.raises(ValueError, match="window_count.*integer"):
        cover_dr.build_rwr_structured_record(**inputs)


def test_disabled_path_is_owned_identity_and_never_builds_or_solves(monkeypatch):
    snapshot = _snapshot()
    monkeypatch.setattr(
        inference_module,
        "build_directed_topk_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disabled RWR built a graph")
        ),
    )
    monkeypatch.setattr(
        inference_module,
        "solve_rwr_cgls",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disabled RWR invoked a solver")
        ),
    )
    result = cover_dr.apply_rwr_to_e3_snapshot(
        snapshot, cover_dr.RWRInferenceConfig(enabled=False)
    )
    assert torch.equal(result.patch_scores, snapshot.unary_scores)
    assert result.patch_scores.untyped_storage().data_ptr() != (
        snapshot.unary_scores.untyped_storage().data_ptr()
    )
    assert result.windows == ()


def test_current_rwr_inference_has_no_historical_cache_dependency():
    source = inspect.getsource(inference_module.apply_rwr_to_e3_snapshot)
    for forbidden in ("torch.load", "open(", "manifest", "cache_path"):
        assert forbidden not in source


def test_alpha_zero_is_exact_and_avoids_graph_and_solver(monkeypatch):
    snapshot = _snapshot()
    monkeypatch.setattr(
        inference_module,
        "build_directed_topk_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )
    monkeypatch.setattr(
        inference_module,
        "solve_rwr_cgls",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )
    result = cover_dr.apply_rwr_to_e3_snapshot(snapshot, _config(alpha=0.0))
    assert torch.equal(result.patch_scores, snapshot.unary_scores)


def test_enabled_integration_equals_independent_composition_and_orientation():
    snapshot = _snapshot(torch.float64)
    config = _config()
    result = cover_dr.apply_rwr_to_e3_snapshot(snapshot, config)
    graph = cover_dr.build_directed_topk_graph(
        snapshot.dino_features[0], k=2, affinity_power=3.0
    )
    expected = cover_dr.solve_rwr_cgls(
        graph, snapshot.unary_scores[0], alpha=0.8
    )
    assert result.patch_scores.shape == (1, 4, 3)
    assert snapshot.dino_features.shape == (1, 4, 4)
    torch.testing.assert_close(result.patch_scores[0], expected.scores)
    assert result.windows[0].iterations == expected.iterations


def test_graph_is_built_exactly_once_for_each_crop(monkeypatch):
    snapshot = _snapshot()
    calls = 0
    original = inference_module.build_directed_topk_graph

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(inference_module, "build_directed_topk_graph", counted)
    cover_dr.apply_rwr_to_e3_snapshot(snapshot, _config())
    assert calls == 1


def test_solver_tolerances_are_passed_explicitly_even_if_defaults_change(monkeypatch):
    observed = {}

    def solver(graph, scores, **kwargs):
        observed.update(kwargs)
        return types.SimpleNamespace(
            scores=scores,
            iterations=1,
            work_count=1,
            total_restart_count=0,
            total_residual_replacement_count=0,
            maximum_scaled_residual=0.0,
            working_maximum_scaled_residual=0.0,
            certified_maximum_scaled_residual=0.0,
            certificate_dtype="float64_quantized_fp32_system",
            fp64_certificate_checks=1,
            fp64_certified_rhs=scores.shape[1],
            fp64_certificate_rejections=0,
            fp64_certificate_restart_count=0,
            fp64_certificate_work=1,
        )

    monkeypatch.setattr(inference_module, "solve_rwr_cgls", solver)
    config = _config(
        solver_rtol=2e-5,
        solver_atol=3e-7,
        solver_max_iterations=4321,
    )
    cover_dr.apply_rwr_to_e3_snapshot(_snapshot(), config)
    assert observed == {
        "alpha": config.alpha,
        "rtol": config.solver_rtol,
        "atol": config.solver_atol,
        "max_iter": config.solver_max_iterations,
    }


def test_inputs_are_unmodified_and_output_is_owned():
    snapshot = _snapshot()
    scores_before = snapshot.unary_scores.clone()
    features_before = snapshot.dino_features.clone()
    result = cover_dr.apply_rwr_to_e3_snapshot(snapshot, _config())
    assert torch.equal(snapshot.unary_scores, scores_before)
    assert torch.equal(snapshot.dino_features, features_before)
    assert result.patch_scores.is_contiguous()
    assert not result.patch_scores.requires_grad
    assert result.patch_scores.untyped_storage().data_ptr() != (
        snapshot.unary_scores.untyped_storage().data_ptr()
    )


def test_downstream_transform_is_one_sigmoid_then_e3_upsampling(monkeypatch):
    scores = _snapshot().unary_scores
    calls = 0
    original = torch.sigmoid

    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(torch, "sigmoid", counted)
    masks = cover_dr.patch_scores_to_masks(scores, (2, 2), (4, 4))
    expected = F.interpolate(
        original(scores.reshape(1, 2, 2, 3).permute(0, 3, 1, 2)),
        (4, 4),
        mode="bilinear",
        align_corners=True,
    )
    assert calls == 1
    torch.testing.assert_close(masks, expected)


def test_solver_failure_propagates_without_e3_fallback(monkeypatch):
    def fail(*_args, **_kwargs):
        raise cover_dr.RWRNonConvergenceError("synthetic failure")

    monkeypatch.setattr(inference_module, "solve_rwr_cgls", fail)
    with pytest.raises(cover_dr.RWRNonConvergenceError, match="synthetic failure"):
        cover_dr.apply_rwr_to_e3_snapshot(_snapshot(), _config())


@pytest.mark.parametrize(
    "override",
    [
        {"alpha": float("nan")},
        {"alpha": 0},
        {"alpha": 1.0},
        {"top_k": 0},
        {"top_k": 2.0},
        {"affinity_power": 3},
        {"affinity_power": float("inf")},
        {"graph_mode": "symmetric"},
        {"solver": "fixed_point"},
        {"solver_rtol": float("nan")},
        {"solver_rtol": -1.0},
        {"solver_atol": True},
        {"solver_max_iterations": 0},
        {"solver_max_iterations": 5.0},
        {"solver_max_iterations": True},
        {"expected_class_count": True},
    ],
)
def test_invalid_enabled_configuration_fails_closed(override):
    with pytest.raises(cover_dr.RWRInferenceConfigError):
        _config(**override)


@pytest.mark.parametrize(
    ("field", "observed", "expected"),
    [
        ("graph_mode", "undirected_topk", load_identity(repo_root=ROOT)["rwr"]["graph_mode"]),
        ("solver", "gmres", load_identity(repo_root=ROOT)["solver"]["method"]),
    ],
)
def test_runtime_validation_reuses_identity_capabilities(field, observed, expected):
    with pytest.raises(cover_dr.RWRInferenceConfigError) as error:
        _config(**{field: observed})
    message = str(error.value)
    assert repr(expected) in message
    assert repr(observed) in message


def test_canonical_enabled_configuration_loads_science_from_identity():
    identity = load_identity(repo_root=ROOT)
    mapping = {
        "enabled": identity["rwr"]["enabled"],
        "identity_path": "evaluation_identities/e3_canonical_directed_rwr.toml",
    }
    before = copy.deepcopy(mapping)
    config = cover_dr.RWRInferenceConfig.from_mapping(mapping)
    assert mapping == before
    assert config.alpha == identity["rwr"]["alpha"]
    assert config.top_k == identity["rwr"]["top_k"]
    assert config.affinity_power == identity["rwr"]["affinity_power"]
    assert config.solver_rtol == identity["solver"]["rtol"]
    assert config.solver_atol == identity["solver"]["atol"]
    assert config.solver_max_iterations == identity["solver"]["max_iterations"]
    assert config.graph_mode == identity["rwr"]["graph_mode"]
    assert config.solver == identity["solver"]["method"]


@pytest.mark.parametrize(
    "mapping",
    [
        {"enabled": True},
        {"enabled": 1, "identity_path": "identity.toml"},
        {"enabled": True, "identity_path": 1},
        {"enabled": True, "identity_path": "identity.toml", "alpha": 0.5},
    ],
)
def test_enabled_mapping_requires_only_exact_identity_reference(mapping):
    with pytest.raises(cover_dr.RWRInferenceConfigError):
        cover_dr.RWRInferenceConfig.from_mapping(mapping)


@pytest.mark.parametrize("field", ["unary_scores", "dino_features"])
def test_nonfinite_snapshot_fails_closed(field):
    snapshot = _snapshot()
    getattr(snapshot, field)[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        cover_dr.apply_rwr_to_e3_snapshot(snapshot, _config())


def test_wrong_class_count_and_patch_alignment_fail_closed():
    with pytest.raises(ValueError, match="class-count"):
        cover_dr.apply_rwr_to_e3_snapshot(
            _snapshot(), _config(expected_class_count=4)
        )
    snapshot = _snapshot()
    snapshot.dino_features = snapshot.dino_features[:, :3]
    with pytest.raises(ValueError, match="patch count"):
        cover_dr.apply_rwr_to_e3_snapshot(snapshot, _config())


def test_deterministic_replay_and_fixed_point_agreement():
    snapshot = _snapshot(torch.float64)
    config = _config()
    first = cover_dr.apply_rwr_to_e3_snapshot(snapshot, config)
    second = cover_dr.apply_rwr_to_e3_snapshot(snapshot, config)
    assert torch.equal(first.patch_scores, second.patch_scores)
    assert first.windows == second.windows
    graph = cover_dr.build_directed_topk_graph(
        snapshot.dino_features[0], k=2, affinity_power=3.0
    )
    fixed = cover_dr.solve_rwr_fixed_point(
        graph, snapshot.unary_scores[0], alpha=0.8
    )
    torch.testing.assert_close(
        first.patch_scores[0], fixed.scores, rtol=2e-9, atol=2e-10
    )


def _load_segmentation_module():
    mmcv = types.ModuleType("mmcv")
    class Config(dict):
        __getattr__ = dict.__getitem__

    mmcv.Config = Config
    utils = types.ModuleType("utils")
    utils.get_logger = lambda: types.SimpleNamespace(info=lambda *_args: None)
    models = types.ModuleType("models")
    models.__path__ = []
    dinotext = types.ModuleType("models.dinotext")
    dinotext.__path__ = []
    # dinotext_seg.py imports its sliding-window geometry helper with a
    # relative import, so it needs a real package context (pointing at the
    # actual directory on disk) to resolve "from .sliding_window_geometry
    # import ...".
    path = ROOT / "src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_seg.py"
    segmentation = types.ModuleType("segmentation")
    segmentation.__path__ = [str(path.parent.parent)]
    evaluation = types.ModuleType("segmentation.evaluation")
    evaluation.__path__ = [str(path.parent)]
    module_names = (
        "mmcv", "utils", "models", "models.dinotext", "models.dinotext.cover_dr",
        "segmentation", "segmentation.evaluation",
        "segmentation.evaluation.sliding_window_geometry",
    )
    previous = {name: sys.modules.get(name) for name in module_names}
    sys.modules.update(
        {
            "mmcv": mmcv,
            "utils": utils,
            "models": models,
            "models.dinotext": dinotext,
            "models.dinotext.cover_dr": cover_dr,
            "segmentation": segmentation,
            "segmentation.evaluation": evaluation,
        }
    )
    sys.modules.pop("segmentation.evaluation.sliding_window_geometry", None)
    try:
        spec = importlib.util.spec_from_file_location(
            "segmentation.evaluation.dinotext_seg", path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("segmentation.evaluation.dinotext_seg", None)
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_real_segmentation_disabled_path_never_requests_snapshot():
    module = _load_segmentation_module()

    class Model(nn.Module):
        def generate_masks(self, image, text, apply_pamr=False):
            del text, apply_pamr
            return torch.full((1, 3, *image.shape[-2:]), 0.25), torch.empty(0)

        def generate_patch_snapshot(self, *_args, **_kwargs):
            raise AssertionError("disabled production path requested a snapshot")

    model = Model()
    inference = module.DINOTextSegInference(
        model,
        torch.randn(3, 4),
        ["a", "b", "c"],
        with_bg=False,
    )
    result = inference.encode_decode(torch.randn(1, 3, 4, 4), None)
    assert torch.equal(result, torch.full((1, 3, 4, 4), 0.25))


def test_real_enabled_path_orders_snapshot_rwr_then_downstream(monkeypatch):
    module = _load_segmentation_module()
    snapshot = _snapshot()
    events = []

    class Model(nn.Module):
        def generate_masks(self, *_args, **_kwargs):
            raise AssertionError("enabled production path called baseline API")

        def generate_patch_snapshot(self, image, text):
            del text
            events.append("snapshot")
            return snapshot

        def masks_from_patch_scores(self, scores, grid_hw, output_hw):
            return downstream(scores, grid_hw, output_hw)

    expected_scores = snapshot.unary_scores + 7

    def apply(observed_snapshot, config):
        assert observed_snapshot is snapshot
        assert config.alpha == load_identity(repo_root=ROOT)["rwr"]["alpha"]
        events.append("rwr")
        return types.SimpleNamespace(patch_scores=expected_scores, windows=())

    def downstream(scores, grid_hw, output_hw):
        assert torch.equal(scores, expected_scores)
        assert grid_hw == (2, 2)
        assert output_hw == (4, 4)
        events.append("downstream")
        return torch.full((1, 3, 4, 4), 0.75)

    monkeypatch.setattr(module, "apply_rwr_to_e3_snapshot", apply)
    inference = module.DINOTextSegInference(
        Model(),
        torch.randn(3, 4),
        ["a", "b", "c"],
        with_bg=False,
    )
    inference.rwr_config = _config(
        alpha=load_identity(repo_root=ROOT)["rwr"]["alpha"]
    )
    result = inference.encode_decode(torch.randn(1, 3, 4, 4), None)
    assert events == ["snapshot", "rwr", "downstream"]
    assert torch.equal(result, torch.full((1, 3, 4, 4), 0.75))


def test_full_enabled_production_path_has_one_sigmoid_and_one_interpolation(monkeypatch):
    module = _load_segmentation_module()
    snapshot = _snapshot()
    calls = {"sigmoid": 0, "interpolate": 0}
    original_sigmoid = torch.sigmoid
    original_interpolate = F.interpolate

    def counted_sigmoid(value):
        calls["sigmoid"] += 1
        return original_sigmoid(value)

    def counted_interpolate(*args, **kwargs):
        calls["interpolate"] += 1
        return original_interpolate(*args, **kwargs)

    monkeypatch.setattr(torch, "sigmoid", counted_sigmoid)
    monkeypatch.setattr(F, "interpolate", counted_interpolate)

    class Model(nn.Module):
        def generate_patch_snapshot(self, _image, _text):
            return snapshot

        def masks_from_patch_scores(self, scores, grid_hw, output_hw):
            logits = scores.reshape(1, *grid_hw, 3).permute(0, 3, 1, 2)
            return F.interpolate(
                torch.sigmoid(logits), output_hw,
                mode="bilinear", align_corners=True,
            )

    monkeypatch.setattr(
        module,
        "apply_rwr_to_e3_snapshot",
        lambda observed, _config: types.SimpleNamespace(
            patch_scores=observed.unary_scores, windows=()
        ),
    )
    inference = module.DINOTextSegInference(
        Model(), torch.randn(3, 4), ["a", "b", "c"], with_bg=False,
    )
    inference.rwr_config = _config()
    inference.encode_decode(torch.randn(1, 3, 4, 4), None)
    assert calls == {"sigmoid": 1, "interpolate": 1}


# ---------------------------------------------------------------------------
# Evaluation-owned state reset lifecycle
# ---------------------------------------------------------------------------


def _plain_inference():
    module = _load_segmentation_module()

    class Model(nn.Module):
        def generate_masks(self, image, text, apply_pamr=False):
            del text, apply_pamr
            return torch.full((1, 3, *image.shape[-2:]), 0.25), torch.empty(0)

    return module.DINOTextSegInference(Model(), torch.randn(3, 4), ["a", "b", "c"], with_bg=False)


def test_reset_evaluation_state_is_the_fresh_construction_default():
    inference = _plain_inference()
    assert inference.natural_evaluation_result is None
    assert inference.primary_solver_summary_override is None
    assert inference.parity_solver_summary_override is None
    assert inference.rwr_runtime.window_count == 0


def test_reset_evaluation_state_clears_a_completed_evaluations_state():
    inference = _plain_inference()
    inference.natural_evaluation_result = {"captured": True, "mIoU": 0.3}
    inference.primary_solver_summary_override = {"window_count": 5}
    inference.parity_solver_summary_override = {"window_count": 1}
    inference.rwr_runtime.window_count = 5  # simulate accumulated telemetry

    inference.reset_evaluation_state()

    assert inference.natural_evaluation_result is None
    assert inference.primary_solver_summary_override is None
    assert inference.parity_solver_summary_override is None
    assert inference.rwr_runtime.window_count == 0


def test_reset_evaluation_state_does_not_touch_constructor_validated_config():
    inference = _plain_inference()
    original_rwr_config = inference.rwr_config
    original_test_cfg = inference.test_cfg
    original_classnames = inference.classnames
    inference.reset_evaluation_state()
    assert inference.rwr_config is original_rwr_config
    assert inference.test_cfg is original_test_cfg
    assert inference.classnames is original_classnames


def test_successive_evaluations_on_the_same_instance_do_not_leak_state():
    # Simulate two sequential dataset evaluations sharing one model
    # instance -- the pattern reset_evaluation_state() exists to protect,
    # even though the current main.py call site always builds a fresh
    # instance per evaluation instead.
    inference = _plain_inference()

    inference.reset_evaluation_state()  # evaluation 1 begins
    inference.natural_evaluation_result = {"captured": True, "mIoU": 0.10}
    inference.primary_solver_summary_override = {"window_count": 3}
    first_result = inference.natural_evaluation_result

    inference.reset_evaluation_state()  # evaluation 2 begins
    assert inference.natural_evaluation_result is None
    assert inference.primary_solver_summary_override is None
    inference.natural_evaluation_result = {"captured": True, "mIoU": 0.20}
    second_result = inference.natural_evaluation_result

    assert first_result != second_result
    assert inference.natural_evaluation_result["mIoU"] == 0.20


def test_inference_failure_leaves_no_stale_natural_result_after_reset():
    inference = _plain_inference()
    inference.reset_evaluation_state()
    try:
        raise RuntimeError("simulated inference failure before natural evaluation ran")
    except RuntimeError:
        pass
    # natural_evaluation_result was never set on this (failed) evaluation --
    # it must still read as None, never a stale prior value.
    assert inference.natural_evaluation_result is None


def test_two_separate_instances_never_share_evaluation_state():
    a = _plain_inference()
    b = _plain_inference()
    a.natural_evaluation_result = {"captured": True, "mIoU": 0.5}
    assert b.natural_evaluation_result is None
