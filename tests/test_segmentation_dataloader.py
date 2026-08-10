import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_builder(monkeypatch, build_dataloader):
    mmcv = ModuleType("mmcv")
    mmcv.Config = SimpleNamespace(fromfile=lambda path: path)

    mmseg = ModuleType("mmseg")
    mmseg_datasets = ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataloader = build_dataloader
    mmseg_datasets.build_dataset = lambda config: config
    mmseg.datasets = mmseg_datasets

    datasets = ModuleType("datasets")
    datasets.get_template = lambda name: []

    monkeypatch.setitem(sys.modules, "mmcv", mmcv)
    monkeypatch.setitem(sys.modules, "mmseg", mmseg)
    monkeypatch.setitem(sys.modules, "mmseg.datasets", mmseg_datasets)
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    path = (
        Path(__file__).parents[1]
        / "src/open_vocabulary_segmentation/segmentation/evaluation/builder.py"
    )
    spec = importlib.util.spec_from_file_location("segmentation_loader_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standard_evaluation_retains_persistent_worker(monkeypatch):
    calls = []
    builder = _load_builder(
        monkeypatch,
        lambda dataset, **kwargs: calls.append((dataset, kwargs)) or "loader",
    )

    assert builder.build_seg_dataloader("dataset") == "loader"
    assert calls == [
        (
            "dataset",
            {
                "samples_per_gpu": 1,
                "workers_per_gpu": 1,
                "dist": True,
                "shuffle": False,
                "persistent_workers": True,
                "pin_memory": False,
            },
        )
    ]


def test_affinity_oracle_uses_rank_process_without_persistent_worker(monkeypatch):
    calls = []
    builder = _load_builder(
        monkeypatch,
        lambda dataset, **kwargs: calls.append((dataset, kwargs)) or "loader",
    )

    assert (
        builder.build_seg_dataloader("dataset", affinity_oracle_enabled=True)
        == "loader"
    )
    assert calls[0][1]["workers_per_gpu"] == 0
    assert calls[0][1]["persistent_workers"] is False
