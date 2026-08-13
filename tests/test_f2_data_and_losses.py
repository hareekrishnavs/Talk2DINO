"""Part F2 infrastructure tests: coco_captions (D3/D4/D5), crop_dataset (D1),
text_vocab (D3), losses (L1-L4). CPU-only, synthetic data -- none of these
need the real COCO dataset, spaCy model download, or a live model/GPU."""
import random

import pytest
import torch
import torch.nn.functional as F

from src.learned_affinity.coco_captions import (
    build_noun_vocabulary,
    disjoint_by_image_split,
    extract_nouns,
    sample_step_vocabulary,
)
from src.learned_affinity.losses import (
    LossWeights,
    anchor_loss,
    caption_noun_ranking_loss,
    compute_total_loss,
    masked_reconstruction_loss,
    mean_row_entropy,
    row_entropy_floor_loss,
    sample_patch_mask,
)
from src.learned_affinity.text_vocab import VocabularyEmbeddings

spacy = pytest.importorskip("spacy", reason="spaCy not installed in this environment")


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


# --- D3/D4/D5: coco_captions -------------------------------------------------

def test_extract_nouns_basic(nlp):
    nouns = extract_nouns("A bicycle replica with a clock as the front wheel.", nlp)
    assert "bicycle" in nouns
    assert "clock" in nouns
    assert "wheel" in nouns
    # pronouns/adjectives/verbs excluded
    assert "front" not in nouns


def test_build_noun_vocabulary_filters_by_min_count(nlp):
    images = {
        1: {"captions": ["a dog runs in the park"]},
        2: {"captions": ["a dog sleeps on the mat"]},
        3: {"captions": ["a cat sits on a chair"]},  # "cat", "chair" appear once each
    }
    vocabulary, per_image = build_noun_vocabulary(images, nlp, min_count=2)
    assert "dog" in vocabulary  # appears in images 1 and 2
    assert "cat" not in vocabulary  # appears only once
    assert set(per_image[1]).issubset(set(vocabulary))
    assert "cat" not in per_image[3]  # filtered out of the per-image list too


def test_disjoint_by_image_split_is_disjoint_and_covers_all():
    ids = list(range(1000))
    train_ids, val_ids = disjoint_by_image_split(ids, val_fraction=0.1, seed=0)
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids) | set(val_ids) == set(ids)
    assert 90 <= len(val_ids) <= 110  # ~10%


def test_sample_step_vocabulary_present_first_then_distractors():
    vocabulary = [f"noun{i}" for i in range(100)]
    present = ["noun3", "noun7"]
    class_list, is_present = sample_step_vocabulary(present, vocabulary, n_distractor=10, seed=0)
    assert class_list[:2] == present
    assert is_present[:2] == [True, True]
    assert sum(is_present) == 2
    assert len(class_list) == 12
    assert set(class_list[2:]).isdisjoint(present)  # distractors never include present nouns


def test_sample_step_vocabulary_requires_at_least_one_present():
    with pytest.raises(ValueError):
        sample_step_vocabulary([], ["a", "b"], n_distractor=5)


# --- text_vocab: VocabularyEmbeddings ---------------------------------------

def test_vocabulary_embeddings_gather():
    vocabulary = ["cat", "dog", "car"]
    embeddings = F.normalize(torch.randn(3, 8), dim=-1)
    ve = VocabularyEmbeddings(vocabulary, embeddings)
    gathered = ve.gather(["dog", "cat", "dog"])
    assert torch.equal(gathered[0], embeddings[1])
    assert torch.equal(gathered[1], embeddings[0])
    assert torch.equal(gathered[2], embeddings[1])


def test_vocabulary_embeddings_gather_missing_noun_raises():
    ve = VocabularyEmbeddings(["cat"], torch.randn(1, 4))
    with pytest.raises(KeyError):
        ve.gather(["dog"])


# --- losses: L1-L4 -----------------------------------------------------------

def _synthetic_graph(n, k, seed=0):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.zeros(n, k, dtype=torch.int64)
    for p in range(n):
        candidates = [i for i in range(n) if i != p]
        chosen = torch.tensor(candidates)[torch.randperm(len(candidates), generator=generator)[:k]]
        indices[p] = chosen
    raw = torch.rand(n, k, generator=generator, dtype=torch.float64) + 0.1
    weights = (raw / raw.sum(-1, keepdim=True)).clone().requires_grad_(True)
    return indices, weights


def test_sample_patch_mask_covers_requested_fraction():
    mask = sample_patch_mask(1000, 0.3, device="cpu", generator=torch.Generator().manual_seed(0))
    assert mask.dtype == torch.bool
    assert 290 <= int(mask.sum()) <= 310


def test_masked_reconstruction_loss_gradient_flows_and_is_finite():
    n, c, k, alpha = 64, 10, 6, 0.9
    indices, weights = _synthetic_graph(n, k)
    raw_scores = torch.randn(c, n, dtype=torch.float64)
    result = masked_reconstruction_loss(
        raw_scores, indices, weights, alpha,
        mask_fraction=0.3, tau=0.1, generator=torch.Generator().manual_seed(1),
    )
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["masked_ce"])
    assert torch.isfinite(result["unmasked_ce"])
    result["loss"].backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()
    assert weights.grad.abs().sum() > 0  # not a trivially-zero gradient


def test_masked_reconstruction_unmasked_loss_smaller_than_masked():
    """Sanity: reconstructing UNMASKED patches (which keep their true S_0 in
    the propagation's own (1-alpha)*S0 term) should be easier than
    reconstructing MASKED ones (pure diffusion from neighbours)."""
    n, c, k, alpha = 256, 10, 12, 0.9
    indices, weights = _synthetic_graph(n, k, seed=2)
    raw_scores = torch.randn(c, n, dtype=torch.float64) * 3  # sharper distribution
    with torch.no_grad():
        result = masked_reconstruction_loss(
            raw_scores, indices, weights, alpha,
            mask_fraction=0.3, tau=0.1, generator=torch.Generator().manual_seed(3),
        )
    assert result["unmasked_ce"] < result["masked_ce"]


def test_caption_noun_ranking_loss_zero_when_present_dominates():
    n, k, alpha = 64, 6, 0.9
    indices, weights = _synthetic_graph(n, k, seed=4)
    is_present = [True, True, False, False, False]
    raw_scores = torch.zeros(5, n, dtype=torch.float64)
    raw_scores[0] = 10.0  # present noun 0 dominates every patch
    raw_scores[1] = 9.0   # present noun 1 also dominates
    loss = caption_noun_ranking_loss(raw_scores, indices, weights, alpha, is_present, margin=0.2)
    assert loss.item() < 1e-3  # present already outranks distractors by a wide margin


def test_caption_noun_ranking_loss_requires_both_present_and_distractor():
    n, k, alpha = 32, 4, 0.9
    indices, weights = _synthetic_graph(n, k, seed=5)
    raw_scores = torch.randn(4, n, dtype=torch.float64)
    with pytest.raises(ValueError):
        caption_noun_ranking_loss(raw_scores, indices, weights, alpha, [True] * 4, margin=0.2)
    with pytest.raises(ValueError):
        caption_noun_ranking_loss(raw_scores, indices, weights, alpha, [False] * 4, margin=0.2)


def test_anchor_loss_zero_when_g_equals_f():
    f = F.normalize(torch.randn(100, 32), dim=-1)
    assert anchor_loss(f, f).item() == pytest.approx(0.0, abs=1e-6)


def test_anchor_loss_positive_when_different():
    torch.manual_seed(0)
    f = F.normalize(torch.randn(100, 32), dim=-1)
    g = F.normalize(torch.randn(100, 32), dim=-1)
    assert anchor_loss(g, f).item() > 0.5  # random unit vectors are mostly far apart


def test_mean_row_entropy_matches_uniform_case():
    k = 12
    weights = torch.full((10, k), 1.0 / k, dtype=torch.float64)
    import math
    expected = math.log(k)  # entropy of a uniform distribution over k outcomes
    assert mean_row_entropy(weights).item() == pytest.approx(expected, abs=1e-6)


def test_mean_row_entropy_low_when_concentrated():
    weights = torch.zeros(10, 12, dtype=torch.float64)
    weights[:, 0] = 1.0  # fully concentrated on one neighbour -> zero entropy
    assert mean_row_entropy(weights).item() == pytest.approx(0.0, abs=1e-6)


def test_row_entropy_floor_loss_only_penalises_below_floor():
    k = 12
    uniform = torch.full((10, k), 1.0 / k, dtype=torch.float64)
    import math
    high_floor_loss = row_entropy_floor_loss(uniform, floor=math.log(k) + 1.0)
    low_floor_loss = row_entropy_floor_loss(uniform, floor=0.0)
    assert high_floor_loss.item() > 0.0  # entropy is below this (unreachable) floor
    assert low_floor_loss.item() == pytest.approx(0.0, abs=1e-9)  # entropy already exceeds floor 0


def test_compute_total_loss_end_to_end_gradient_flows_to_weights_and_g():
    n, c, k, alpha, dim = 64, 8, 6, 0.9, 16
    indices, weights = _synthetic_graph(n, k, seed=6)
    f = F.normalize(torch.randn(n, dim, dtype=torch.float64), dim=-1)
    g = F.normalize(f + 0.1 * torch.randn(n, dim, dtype=torch.float64), dim=-1).requires_grad_(True)
    raw_scores = torch.randn(c, n, dtype=torch.float64)
    is_present = [True, True, False, False, False, False, False, False]

    result = compute_total_loss(
        f=f, g=g, indices=indices, weights=weights, raw_scores=raw_scores,
        is_present=is_present, alpha=alpha,
        loss_weights=LossWeights(l4=0.1, l4_entropy_floor=5.0),  # nonzero l4 to exercise that path too
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.isfinite(result["total"])
    for key in ("l1_masked_ce", "l1_unmasked_ce", "l2_ranking", "l3_anchor", "l4_entropy_floor", "mean_row_entropy"):
        assert torch.isfinite(result[key]), key

    result["total"].backward()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()
    assert g.grad is not None and torch.isfinite(g.grad).all()
    assert weights.grad.abs().sum() > 0
    assert g.grad.abs().sum() > 0
