import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = (
    ROOT
    / "src/open_vocabulary_segmentation/segmentation/evaluation/builder.py"
)
MAIN_PATH = ROOT / "src/open_vocabulary_segmentation/main.py"


def _load_builder(monkeypatch):
    calls = []

    fake_mmcv = types.ModuleType("mmcv")
    fake_mmcv.Config = object
    fake_mmseg = types.ModuleType("mmseg")
    fake_mmseg_datasets = types.ModuleType("mmseg.datasets")

    def build_dataloader(dataset, **kwargs):
        calls.append((dataset, kwargs))
        return "loader"

    fake_mmseg_datasets.build_dataloader = build_dataloader
    fake_mmseg_datasets.build_dataset = lambda config: config
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.get_template = lambda name: name

    monkeypatch.setitem(sys.modules, "mmcv", fake_mmcv)
    monkeypatch.setitem(sys.modules, "mmseg", fake_mmseg)
    monkeypatch.setitem(sys.modules, "mmseg.datasets", fake_mmseg_datasets)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    module_name = "_test_segmentation_evaluation_builder"
    spec = importlib.util.spec_from_file_location(module_name, BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, calls


@pytest.mark.parametrize(
    ("num_workers", "persistent_workers"),
    [(0, False), (3, True)],
)
def test_segmentation_loader_uses_requested_worker_count(
    monkeypatch, num_workers, persistent_workers
):
    builder, calls = _load_builder(monkeypatch)
    dataset = object()

    assert builder.build_seg_dataloader(dataset, num_workers=num_workers) == "loader"
    assert calls == [
        (
            dataset,
            {
                "samples_per_gpu": 1,
                "workers_per_gpu": num_workers,
                "dist": True,
                "shuffle": False,
                "persistent_workers": persistent_workers,
                "pin_memory": False,
            },
        )
    ]


def test_segmentation_loader_preserves_legacy_direct_caller_default(monkeypatch):
    builder, calls = _load_builder(monkeypatch)

    builder.build_seg_dataloader(object())

    assert calls[0][1]["workers_per_gpu"] == 1
    assert calls[0][1]["persistent_workers"] is True


@pytest.mark.parametrize("num_workers", [-1, 1.0, True, None])
def test_segmentation_loader_rejects_invalid_worker_count(monkeypatch, num_workers):
    builder, calls = _load_builder(monkeypatch)

    with pytest.raises(ValueError, match="non-negative integer"):
        builder.build_seg_dataloader(object(), num_workers=num_workers)

    assert calls == []


def test_evaluation_entrypoint_passes_configured_worker_count():
    tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_seg_dataloader"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    value = keywords["num_workers"]
    assert isinstance(value, ast.Attribute)
    assert value.attr == "num_workers"
    assert isinstance(value.value, ast.Attribute)
    assert value.value.attr == "data"
    assert isinstance(value.value.value, ast.Name)
    assert value.value.value.id == "cfg"
