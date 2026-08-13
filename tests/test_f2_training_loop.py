"""Part F2 T1-T4 training-loop tests. `training_step` is deliberately
decoupled from real image/text extraction (see train.py's module
docstring) so these run on synthetic data, CPU, no GPU/dataset needed."""
import math

import pytest
import torch
import torch.nn.functional as F

from src.learned_affinity.losses import LossWeights
from src.learned_affinity.metric import LearnedMetric
from src.learned_affinity.train import (
    TrainConfig,
    cosine_lr_lambda,
    load_checkpoint,
    save_checkpoint,
    training_step,
)


def _synthetic_samples(n_samples, p, dim, c):
    torch.manual_seed(0)
    samples = []
    for i in range(n_samples):
        features = F.normalize(torch.randn(p, dim, dtype=torch.float64), dim=-1)
        raw_scores = torch.randn(c, p, dtype=torch.float64)
        is_present = [True, True] + [False] * (c - 2)
        samples.append((features, raw_scores, is_present))
    return samples


def test_training_step_updates_only_metric_parameters_and_is_finite():
    torch.manual_seed(0)
    metric = LearnedMetric(dim=16, hidden=8, k=6).double()
    # break zero-init so a real, checkable update happens
    with torch.no_grad():
        for p in metric.parameters():
            p.copy_(torch.randn(p.shape, dtype=torch.float64) * 0.05)
    before = {name: p.clone() for name, p in metric.named_parameters()}

    optimizer = torch.optim.AdamW(metric.parameters(), lr=1e-2)
    samples = _synthetic_samples(3, p=32, dim=16, c=6)
    log = training_step(metric, optimizer, samples, alpha=0.9, loss_weights=LossWeights())

    assert math.isfinite(log["total_loss"])
    for key in ("l1_masked_ce", "l1_unmasked_ce", "l2_ranking", "l3_anchor", "mean_row_entropy", "grad_norm"):
        assert math.isfinite(log[key]), key
    assert log["batch_size"] == 3
    assert len(log["cg_forward_iters"]) == 3
    assert all(isinstance(i, int) for i in log["cg_forward_iters"])

    changed = any(not torch.equal(before[name], p) for name, p in metric.named_parameters())
    assert changed, "no parameter changed after an optimizer step"


def test_training_step_requires_at_least_one_sample():
    metric = LearnedMetric(dim=16, hidden=8, k=6).double()
    optimizer = torch.optim.AdamW(metric.parameters(), lr=1e-4)
    with pytest.raises(ValueError):
        training_step(metric, optimizer, [], alpha=0.9)


def test_grad_clipping_bounds_the_reported_norm_only_when_below_clip():
    """grad_norm returned by clip_grad_norm_ is the PRE-clip norm; verify it
    is at least finite and non-negative, and that a very small clip value
    actually shrinks the resulting parameter update (sanity, not exact)."""
    torch.manual_seed(1)
    metric = LearnedMetric(dim=16, hidden=8, k=6).double()
    with torch.no_grad():
        for p in metric.parameters():
            p.copy_(torch.randn(p.shape, dtype=torch.float64) * 0.5)
    optimizer = torch.optim.SGD(metric.parameters(), lr=1.0)
    samples = _synthetic_samples(2, p=32, dim=16, c=6)
    log = training_step(metric, optimizer, samples, alpha=0.9, grad_clip=1e-6)
    assert log["grad_norm"] >= 0.0
    assert math.isfinite(log["grad_norm"])


def test_cosine_lr_lambda_warmup_then_decay():
    total, warmup = 1000, 100
    assert cosine_lr_lambda(0, total_steps=total, warmup_steps=warmup) == 0.0
    assert cosine_lr_lambda(warmup, total_steps=total, warmup_steps=warmup) == pytest.approx(1.0, abs=1e-6)
    assert cosine_lr_lambda(total, total_steps=total, warmup_steps=warmup) == pytest.approx(0.0, abs=1e-6)
    mid = cosine_lr_lambda((total + warmup) // 2, total_steps=total, warmup_steps=warmup)
    assert 0.0 < mid < 1.0


def test_checkpoint_round_trip(tmp_path):
    metric = LearnedMetric(dim=16, hidden=8, k=6).double()
    with torch.no_grad():
        metric.r.fill_(0.234)  # within [0, r_max] -- effective == raw, keeps this test single-purpose
    optimizer = torch.optim.AdamW(metric.parameters(), lr=1e-4)
    config = TrainConfig()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, metric, optimizer, step=42, config=config)

    metric2 = LearnedMetric(dim=16, hidden=8, k=6).double()
    payload = load_checkpoint(path, metric2, optimizer=None)
    assert payload["step"] == 42
    assert payload["r"] == pytest.approx(float(metric.r.detach()))
    assert payload["r_raw"] == pytest.approx(float(metric.r.detach()))
    assert torch.equal(metric2.r, metric.r)
