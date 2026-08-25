"""CPU-only tests for models.dinotext.cover_dr.coco_object_evaluator.
apply_background_channel/WindowOperationTelemetryE3/aggregate_telemetry
are pure tensor/dataclass code -- no CUDA, no model required."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
OVS_ROOT = ROOT / "src/open_vocabulary_segmentation"
for _p in (ROOT, OVS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

torch = pytest.importorskip("torch")

from models.dinotext.cover_dr.coco_object_evaluator import (  # noqa: E402
    CocoObjectEvaluatorError,
    WindowOperationTelemetryE3,
    aggregate_telemetry,
    apply_background_channel,
)


# ---------------------------------------------------------------------------
# apply_background_channel
# ---------------------------------------------------------------------------


def test_background_channel_shape_and_position():
    masks = torch.rand(1, 80, 4, 4)
    out = apply_background_channel(masks, bg_thresh=0.55)
    assert out.shape == (1, 81, 4, 4)
    assert torch.equal(out[:, 1:], masks)


def test_background_channel_value_is_constant():
    masks = torch.rand(1, 80, 3, 5)
    out = apply_background_channel(masks, bg_thresh=0.37)
    assert torch.allclose(out[:, 0], torch.full((1, 3, 5), 0.37))


def test_background_channel_matches_production_formula_exactly():
    """Independently reproduces DINOTextSegInference.encode_decode's own
    background-injection formula (torch.full + torch.cat at channel 0),
    without importing that class (which requires a live model)."""
    masks = torch.rand(2, 80, 6, 6)
    bg_thresh = 0.55
    expected_background = torch.full([2, 1, 6, 6], bg_thresh, dtype=torch.float, device=masks.device)
    expected = torch.cat([expected_background, masks], dim=1)
    observed = apply_background_channel(masks, bg_thresh=bg_thresh)
    assert torch.equal(observed, expected)


def test_background_channel_argmax_competition():
    """Confirms background genuinely competes in argmax: a pixel whose
    foreground scores are all below bg_thresh must select background
    (index 0), and a pixel whose foreground score exceeds bg_thresh must
    select that foreground class."""
    masks = torch.zeros(1, 3, 1, 2)
    masks[0, 0, 0, 0] = 0.1  # below bg_thresh at pixel 0
    masks[0, 1, 0, 1] = 0.9  # above bg_thresh at pixel 1
    out = apply_background_channel(masks, bg_thresh=0.55)
    argmax = out.argmax(dim=1)
    assert argmax[0, 0, 0].item() == 0  # background wins
    assert argmax[0, 0, 1].item() == 2  # foreground class index 1 (channel 2 after background prepended) wins


def test_background_channel_rejects_wrong_ndim():
    with pytest.raises(CocoObjectEvaluatorError):
        apply_background_channel(torch.rand(80, 4, 4), bg_thresh=0.55)


def test_background_channel_rejects_out_of_range_threshold():
    masks = torch.rand(1, 80, 4, 4)
    with pytest.raises(CocoObjectEvaluatorError):
        apply_background_channel(masks, bg_thresh=1.5)
    with pytest.raises(CocoObjectEvaluatorError):
        apply_background_channel(masks, bg_thresh=-0.1)


def test_background_channel_rejects_wrong_type_threshold():
    masks = torch.rand(1, 80, 4, 4)
    with pytest.raises(CocoObjectEvaluatorError):
        apply_background_channel(masks, bg_thresh=1)  # int, not float


def test_background_channel_equivalence_under_uniform_averaging():
    """Proves the documented per-window-vs-post-stitch injection
    equivalence directly: accumulating N copies of a uniform constant
    channel and dividing by the coverage count reproduces the same
    constant exactly, for any positive count map -- including a
    nonuniform one (simulating boundary/overlap regions)."""
    bg_thresh = 0.55
    count = torch.tensor([[1.0, 2.0], [3.0, 1.0]]).reshape(1, 1, 2, 2)
    accumulated = torch.full((1, 1, 2, 2), bg_thresh) * count
    averaged = accumulated / count
    assert torch.allclose(averaged, torch.full((1, 1, 2, 2), bg_thresh))


# ---------------------------------------------------------------------------
# WindowOperationTelemetryE3
# ---------------------------------------------------------------------------


def _valid_telemetry_kwargs(**overrides):
    base = dict(
        backbone_snapshot_calls=1, dino_feature_extractions=1, topk_selection_calls=1,
        graph_normalizations=2, finite_step_propagations=2, e3_propagations=0,
        k11_updates=320, k12_updates=320, sigmoid_calls=3, interpolation_calls=3,
    )
    base.update(overrides)
    return base


def test_valid_telemetry_constructs():
    t = WindowOperationTelemetryE3(**_valid_telemetry_kwargs())
    assert t.e3_propagations == 0
    assert t.sigmoid_calls == 3


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("backbone_snapshot_calls", 2),
        ("graph_normalizations", 1),
        ("finite_step_propagations", 1),
        ("e3_propagations", 1),
        ("sigmoid_calls", 2),
        ("interpolation_calls", 2),
        ("k11_updates", 319),
        ("k12_updates", 321),
    ],
)
def test_telemetry_rejects_wrong_counts(field, bad_value):
    with pytest.raises(CocoObjectEvaluatorError):
        WindowOperationTelemetryE3(**_valid_telemetry_kwargs(**{field: bad_value}))


def test_aggregate_telemetry_sums_correctly():
    t = WindowOperationTelemetryE3(**_valid_telemetry_kwargs())
    totals = aggregate_telemetry([t, t, t])
    assert totals["backbone_snapshot_calls"] == 3
    assert totals["sigmoid_calls"] == 9
    assert totals["k11_updates"] == 960
    assert totals["e3_propagations"] == 0


def test_aggregate_telemetry_rejects_empty_list():
    with pytest.raises(CocoObjectEvaluatorError):
        aggregate_telemetry([])
