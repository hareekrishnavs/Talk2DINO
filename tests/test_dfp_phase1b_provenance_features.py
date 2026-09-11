"""Exhaustive independent-oracle + adversarial tests for the DFP Phase
1B Stage-A provenance-feature scientific kernel
(``src.dfp_phase1b_provenance_features``) and its E21 successor
identity (``src.dfp_phase1b_provenance_features_identity``).

Every correctness assertion in the "oracle" sections below is checked
against ``_oracle_features`` -- a plain, pure-Python-loop
re-implementation of the eight E15 Stage-A component-evidence formulas,
built independently from the frozen E15 TOML text, using Python's own
``sorted``/``sum``/``max``/``min`` -- never the production kernel
itself, never a "reasonable range" check, and never torch. The one
end-to-end section additionally exercises the REAL, unmodified
masked-forward provenance core (a real k=12 recurrence graph, not a
toy k=1/k=2 fixture) and the REAL, unmodified Stage-A relational
validator, to prove the kernel's output is actually consumable by
already-committed authority, not merely self-consistent.

CPU-only. Never initializes CUDA merely to run (a CUDA path is
exercised only if a GPU is actually present, via
``torch.cuda.is_available()`` guards). Every adversarial identity
mutation writes its own temporary copy -- the committed TOML is never
edited by any test."""

from __future__ import annotations

import copy
import inspect
import math
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import src.dfp_phase1b_provenance_features as k  # noqa: E402
import src.dfp_phase1b_provenance_features_identity as idmod  # noqa: E402

K = k.DfpPhase1bProvenanceFeaturesError
IDENTITY_PATH = ROOT / idmod.IDENTITY_RELATIVE_PATH
CLI_PATH = ROOT / "diagnostics/run_dfp_phase1b_provenance_features.py"


# =====================================================================
# Section 0: independent, pure-Python-loop oracle.
# =====================================================================


def _oracle_row(row: list[float], m320: float, component_ids: list[int]) -> dict:
    """One disputed pixel's eight fields, computed by plain Python
    loops from the frozen E15 formulas -- independently of the kernel
    and of torch."""
    pos_sum = math.fsum(max(g, 0.0) for g in row)
    neg_sum = math.fsum(min(g, 0.0) for g in row)
    n_pos = sum(1 for g in row if g > 0.0)
    ranked = sorted(zip(row, component_ids), key=lambda t: (-t[0], t[1]))
    positive_ranked = [(v, cid) for v, cid in ranked if v > 0.0]
    assert len(positive_ranked) == n_pos
    top_value, top_id = positive_ranked[0]
    top1_share = top_value / pos_sum
    cum = 0.0
    k_star = None
    cum_at_k = None
    for idx, (v, _cid) in enumerate(positive_ranked, start=1):
        cum += v
        if cum >= m320:
            k_star = idx
            cum_at_k = cum
            break
    assert k_star is not None, "oracle: no k satisfies the crossing condition -- fixture is invalid"
    phi = cum_at_k / pos_sum
    return dict(
        k_star=k_star, phi=phi, top1_positive_share=top1_share,
        positive_evidence_sum=pos_sum, negative_evidence_sum=neg_sum,
        n_positive_components=n_pos, top_positive_component_id=top_id,
        top_positive_evidence=top_value,
    )


def _oracle_rows(G_rows: list[list[float]], m320_vals: list[float], component_ids: list[int]) -> list[dict]:
    return [_oracle_row(row, m320, component_ids) for row, m320 in zip(G_rows, m320_vals)]


def _to_tensors(G_rows: list[list[float]], m320_vals: list[float], component_ids: list[int]):
    G = torch.tensor(G_rows, dtype=torch.float64)
    m320 = torch.tensor(m320_vals, dtype=torch.float64)
    ids = torch.tensor(component_ids, dtype=torch.int64)
    return G, m320, ids


def _assert_matches_oracle(feats: "k.ComponentEvidenceFeatures", oracle_rows: list[dict]) -> None:
    for i, expected in enumerate(oracle_rows):
        assert feats.k_star[i].item() == expected["k_star"], i
        assert feats.n_positive_components[i].item() == expected["n_positive_components"], i
        assert feats.top_positive_component_id[i].item() == expected["top_positive_component_id"], i
        assert math.isclose(feats.phi[i].item(), expected["phi"], rel_tol=1e-12, abs_tol=1e-12), i
        assert math.isclose(feats.top1_positive_share[i].item(), expected["top1_positive_share"], rel_tol=1e-12, abs_tol=1e-12), i
        assert math.isclose(feats.positive_evidence_sum[i].item(), expected["positive_evidence_sum"], rel_tol=1e-12, abs_tol=1e-12), i
        assert math.isclose(feats.negative_evidence_sum[i].item(), expected["negative_evidence_sum"], rel_tol=1e-12, abs_tol=1e-12), i
        assert math.isclose(feats.top_positive_evidence[i].item(), expected["top_positive_evidence"], rel_tol=1e-12, abs_tol=1e-12), i


# =====================================================================
# Section 1: hand-verified fixtures.
# =====================================================================


def test_hand_fixture_basic():
    G_rows = [[3.0, -1.0, 2.0]]
    m320_vals = [4.0]
    ids = [0, 1, 2]
    G, m320, cid = _to_tensors(G_rows, m320_vals, ids)
    feats = k.compute_component_evidence_features(G, m320, component_ids=cid)
    oracle = _oracle_rows(G_rows, m320_vals, ids)
    _assert_matches_oracle(feats, oracle)
    assert feats.k_star.tolist() == [2]
    assert feats.phi.tolist() == [1.0]
    assert feats.top1_positive_share.tolist() == [0.6]
    assert feats.positive_evidence_sum.tolist() == [5.0]
    assert feats.negative_evidence_sum.tolist() == [-1.0]
    assert feats.n_positive_components.tolist() == [2]
    assert feats.top_positive_component_id.tolist() == [0]
    assert feats.top_positive_evidence.tolist() == [3.0]


def test_hand_fixture_k_star_one():
    G_rows = [[5.0, -2.0]]
    m320_vals = [1.0]
    ids = [10, 20]
    G, m320, cid = _to_tensors(G_rows, m320_vals, ids)
    feats = k.compute_component_evidence_features(G, m320, component_ids=cid)
    _assert_matches_oracle(feats, _oracle_rows(G_rows, m320_vals, ids))
    assert feats.k_star.tolist() == [1]
    assert feats.phi.tolist() == [1.0]


def test_hand_fixture_tie_break_ascending_component_id():
    G_rows = [[2.0, 2.0]]
    m320_vals = [1.0]
    ids = [7, 3]  # lower id (3) is at column index 1
    G, m320, cid = _to_tensors(G_rows, m320_vals, ids)
    feats = k.compute_component_evidence_features(G, m320, component_ids=cid)
    _assert_matches_oracle(feats, _oracle_rows(G_rows, m320_vals, ids))
    assert feats.top_positive_component_id.tolist() == [3]


def test_hand_fixture_three_way_tie_break():
    G_rows = [[5.0, 5.0, 5.0, 1.0]]
    m320_vals = [12.0]
    ids = [40, 10, 25, 1]
    G, m320, cid = _to_tensors(G_rows, m320_vals, ids)
    feats = k.compute_component_evidence_features(G, m320, component_ids=cid)
    oracle = _oracle_rows(G_rows, m320_vals, ids)
    _assert_matches_oracle(feats, oracle)
    # ranked order by (value desc, id asc): id10(5.0), id25(5.0), id40(5.0), id1(1.0)
    assert feats.top_positive_component_id.tolist() == [10]
    assert feats.k_star.tolist() == [3]  # 5+5+5=15>=12, 5+5=10<12


def test_hand_fixture_zero_valued_component_never_counted_positive():
    G_rows = [[3.0, 0.0, -1.0]]
    m320_vals = [1.0]
    ids = [0, 1, 2]
    G, m320, cid = _to_tensors(G_rows, m320_vals, ids)
    feats = k.compute_component_evidence_features(G, m320, component_ids=cid)
    _assert_matches_oracle(feats, _oracle_rows(G_rows, m320_vals, ids))
    assert feats.n_positive_components.tolist() == [1]
    assert feats.negative_evidence_sum.tolist() == [-1.0]  # min(0,0)=0 contributes nothing


# =====================================================================
# Section 2: randomized property tests against the independent oracle.
# =====================================================================


def _random_valid_row(rng: random.Random, component_count: int) -> tuple[list[float], float]:
    """A random row guaranteed to satisfy the kernel's own required
    invariants (m320 > 0, positive_evidence_sum > 0): at least one
    strictly positive value, and m320 chosen <= positive_evidence_sum
    so a valid k_star exists."""
    row = [rng.uniform(-5.0, 5.0) for _ in range(component_count)]
    # force at least one strictly positive component
    row[rng.randrange(component_count)] = rng.uniform(0.01, 5.0)
    pos_sum = sum(v for v in row if v > 0.0)
    m320 = rng.uniform(1e-9, pos_sum)
    return row, m320


@pytest.mark.parametrize("trial", range(40))
def test_randomized_property_matches_oracle(trial):
    rng = random.Random(1000 + trial)
    component_count = rng.randint(1, 20)
    row_count = rng.randint(1, 6)
    ids = rng.sample(range(1000), component_count)
    G_rows, m320_vals = [], []
    for _ in range(row_count):
        row, m320 = _random_valid_row(rng, component_count)
        G_rows.append(row)
        m320_vals.append(m320)
    G, m320, cid = _to_tensors(G_rows, m320_vals, ids)
    feats = k.compute_component_evidence_features(G, m320, component_ids=cid)
    _assert_matches_oracle(feats, _oracle_rows(G_rows, m320_vals, ids))


@pytest.mark.parametrize("trial", range(10))
def test_randomized_property_with_exact_ties(trial):
    """Every component shares the same value -- exercises the
    ascending-component-id tie-break exhaustively."""
    rng = random.Random(2000 + trial)
    component_count = rng.randint(2, 12)
    tie_value = rng.uniform(0.1, 3.0)
    row = [tie_value] * component_count
    m320 = rng.uniform(0.01, tie_value * component_count)
    ids = rng.sample(range(1000), component_count)
    G, m320_t, cid = _to_tensors([row], [m320], ids)
    feats = k.compute_component_evidence_features(G, m320_t, component_ids=cid)
    oracle = _oracle_rows([row], [m320], ids)
    _assert_matches_oracle(feats, oracle)
    assert feats.top_positive_component_id.item() == min(ids)


# =====================================================================
# Section 3: permutation invariance.
# =====================================================================


@pytest.mark.parametrize("trial", range(15))
def test_column_permutation_invariance(trial):
    rng = random.Random(3000 + trial)
    component_count = rng.randint(2, 10)
    row, m320 = _random_valid_row(rng, component_count)
    ids = rng.sample(range(1000), component_count)
    G, m320_t, cid = _to_tensors([row], [m320], ids)
    feats_original = k.compute_component_evidence_features(G, m320_t, component_ids=cid)

    perm = list(range(component_count))
    rng.shuffle(perm)
    G_perm = G[:, perm]
    cid_perm = cid[perm]
    feats_perm = k.compute_component_evidence_features(G_perm, m320_t, component_ids=cid_perm)

    assert feats_perm.k_star.tolist() == feats_original.k_star.tolist()
    assert feats_perm.n_positive_components.tolist() == feats_original.n_positive_components.tolist()
    assert feats_perm.top_positive_component_id.tolist() == feats_original.top_positive_component_id.tolist()
    assert math.isclose(feats_perm.phi.item(), feats_original.phi.item(), rel_tol=1e-12)
    assert math.isclose(feats_perm.positive_evidence_sum.item(), feats_original.positive_evidence_sum.item(), rel_tol=1e-12)
    assert math.isclose(feats_perm.negative_evidence_sum.item(), feats_original.negative_evidence_sum.item(), rel_tol=1e-12)


# =====================================================================
# Section 4: shape / dtype / device validation.
# =====================================================================


def test_rejects_non_tensor_G():
    with pytest.raises(K):
        k.compute_component_evidence_features([[1.0]], torch.tensor([1.0], dtype=torch.float64), component_ids=torch.tensor([0], dtype=torch.int64))


def test_rejects_wrong_dtype_G_float32():
    G = torch.tensor([[1.0]], dtype=torch.float32)
    m320 = torch.tensor([1.0], dtype=torch.float64)
    cid = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_boolean_dtype_G():
    G = torch.zeros((1, 1), dtype=torch.bool)
    m320 = torch.tensor([1.0], dtype=torch.float64)
    cid = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_wrong_ndim_G_1d():
    G = torch.tensor([1.0, 2.0], dtype=torch.float64)
    m320 = torch.tensor([1.0], dtype=torch.float64)
    cid = torch.tensor([0, 1], dtype=torch.int64)
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_wrong_ndim_G_3d():
    G = torch.zeros((1, 1, 1), dtype=torch.float64)
    m320 = torch.tensor([1.0], dtype=torch.float64)
    cid = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_wrong_ndim_m320_2d():
    G = torch.tensor([[1.0]], dtype=torch.float64)
    m320 = torch.tensor([[1.0]], dtype=torch.float64)
    cid = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_m320_dtype_float32():
    G = torch.tensor([[1.0]], dtype=torch.float64)
    m320 = torch.tensor([1.0], dtype=torch.float32)
    cid = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_component_ids_wrong_dtype_float64():
    G = torch.tensor([[1.0]], dtype=torch.float64)
    m320 = torch.tensor([1.0], dtype=torch.float64)
    cid = torch.tensor([0.0], dtype=torch.float64)
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_component_ids_wrong_ndim():
    G = torch.tensor([[1.0]], dtype=torch.float64)
    m320 = torch.tensor([1.0], dtype=torch.float64)
    cid = torch.tensor([[0]], dtype=torch.int64)
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_m320_row_count_mismatch():
    G = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    m320 = torch.tensor([1.0], dtype=torch.float64)
    cid = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_component_ids_length_mismatch():
    G = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    m320 = torch.tensor([1.0], dtype=torch.float64)
    cid = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_zero_component_count():
    G = torch.zeros((1, 0), dtype=torch.float64)
    m320 = torch.tensor([1.0], dtype=torch.float64)
    cid = torch.zeros((0,), dtype=torch.int64)
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_accepts_zero_disputed_pixel_rows():
    G = torch.zeros((0, 3), dtype=torch.float64)
    m320 = torch.zeros((0,), dtype=torch.float64)
    cid = torch.tensor([0, 1, 2], dtype=torch.int64)
    feats = k.compute_component_evidence_features(G, m320, component_ids=cid)
    assert feats.k_star.shape == (0,)
    assert feats.k_star.dtype == torch.int64
    assert feats.phi.shape == (0,)
    assert feats.phi.dtype == torch.float64


def test_rejects_nan_in_G():
    G = torch.tensor([[float("nan"), 1.0]], dtype=torch.float64)
    m320 = torch.tensor([1.0], dtype=torch.float64)
    cid = torch.tensor([0, 1], dtype=torch.int64)
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_inf_in_G():
    G = torch.tensor([[float("inf"), 1.0]], dtype=torch.float64)
    m320 = torch.tensor([1.0], dtype=torch.float64)
    cid = torch.tensor([0, 1], dtype=torch.int64)
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_nan_in_m320():
    G = torch.tensor([[1.0]], dtype=torch.float64)
    m320 = torch.tensor([float("nan")], dtype=torch.float64)
    cid = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_negative_component_id():
    G = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    m320 = torch.tensor([1.0], dtype=torch.float64)
    cid = torch.tensor([-1, 0], dtype=torch.int64)
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_duplicate_component_id():
    G = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    m320 = torch.tensor([1.0], dtype=torch.float64)
    cid = torch.tensor([5, 5], dtype=torch.int64)
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real second device to exercise a genuine mismatch")
def test_rejects_device_mismatch_G_cuda_m320_cpu():
    G = torch.tensor([[1.0]], dtype=torch.float64, device="cuda")
    m320 = torch.tensor([1.0], dtype=torch.float64, device="cpu")
    cid = torch.tensor([0], dtype=torch.int64, device="cuda")
    with pytest.raises(K):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real GPU to prove no silent CPU transfer")
def test_output_stays_on_cuda_device_never_silently_moved():
    G = torch.tensor([[3.0, -1.0]], dtype=torch.float64, device="cuda")
    m320 = torch.tensor([1.0], dtype=torch.float64, device="cuda")
    cid = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    feats = k.compute_component_evidence_features(G, m320, component_ids=cid)
    for field in ("k_star", "phi", "top1_positive_share", "positive_evidence_sum",
                  "negative_evidence_sum", "n_positive_components", "top_positive_component_id",
                  "top_positive_evidence"):
        assert getattr(feats, field).device.type == "cuda", field


def test_output_dtypes_exact():
    G, m320, cid = _to_tensors([[3.0, -1.0, 2.0]], [4.0], [0, 1, 2])
    feats = k.compute_component_evidence_features(G, m320, component_ids=cid)
    assert feats.k_star.dtype == torch.int64
    assert feats.n_positive_components.dtype == torch.int64
    assert feats.top_positive_component_id.dtype == torch.int64
    assert feats.phi.dtype == torch.float64
    assert feats.top1_positive_share.dtype == torch.float64
    assert feats.positive_evidence_sum.dtype == torch.float64
    assert feats.negative_evidence_sum.dtype == torch.float64
    assert feats.top_positive_evidence.dtype == torch.float64


def test_does_not_mutate_inputs():
    G, m320, cid = _to_tensors([[3.0, -1.0, 2.0]], [4.0], [0, 1, 2])
    G_before, m320_before, cid_before = G.clone(), m320.clone(), cid.clone()
    k.compute_component_evidence_features(G, m320, component_ids=cid)
    assert torch.equal(G, G_before)
    assert torch.equal(m320, m320_before)
    assert torch.equal(cid, cid_before)


def test_output_is_not_a_view_into_G():
    G, m320, cid = _to_tensors([[3.0, -1.0, 2.0]], [4.0], [0, 1, 2])
    feats = k.compute_component_evidence_features(G, m320, component_ids=cid)
    feats.top_positive_evidence[0] = 999.0
    assert G[0, 0].item() == 3.0  # unaffected


def test_result_type_is_frozen_dataclass():
    G, m320, cid = _to_tensors([[3.0, -1.0, 2.0]], [4.0], [0, 1, 2])
    feats = k.compute_component_evidence_features(G, m320, component_ids=cid)
    with pytest.raises(Exception):
        feats.k_star = torch.zeros(1, dtype=torch.int64)  # frozen dataclass rejects assignment


# =====================================================================
# Section 5: domain-invariant violations (fail closed, never fabricate).
# =====================================================================


def test_rejects_m320_zero():
    G = torch.tensor([[1.0]], dtype=torch.float64)
    m320 = torch.tensor([0.0], dtype=torch.float64)
    cid = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(K, match="strictly positive"):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_m320_negative():
    G = torch.tensor([[1.0]], dtype=torch.float64)
    m320 = torch.tensor([-1.0], dtype=torch.float64)
    cid = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(K, match="strictly positive"):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_all_non_positive_components():
    G = torch.tensor([[-1.0, -2.0, 0.0]], dtype=torch.float64)
    m320 = torch.tensor([1.0], dtype=torch.float64)
    cid = torch.tensor([0, 1, 2], dtype=torch.int64)
    with pytest.raises(K, match="positive_evidence_sum"):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


def test_rejects_insufficient_positive_evidence_no_crossing():
    """positive_evidence_sum < m320 -- no k satisfies the crossing
    condition -- must reject rather than invent an infinite/undefined
    k_star."""
    G = torch.tensor([[1.0]], dtype=torch.float64)
    m320 = torch.tensor([100.0], dtype=torch.float64)
    cid = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(K, match="crossing condition"):
        k.compute_component_evidence_features(G, m320, component_ids=cid)


# =====================================================================
# Section 6: metamorphic / relational tests.
# =====================================================================


@pytest.mark.parametrize("scale", [0.5, 2.0, 100.0, 1e-3])
def test_metamorphic_positive_scale_invariance(scale):
    G_rows = [[3.0, -1.0, 2.0]]
    m320_vals = [4.0]
    ids = [0, 1, 2]
    G, m320, cid = _to_tensors(G_rows, m320_vals, ids)
    base = k.compute_component_evidence_features(G, m320, component_ids=cid)

    G_scaled = G * scale
    m320_scaled = m320 * scale
    scaled = k.compute_component_evidence_features(G_scaled, m320_scaled, component_ids=cid)

    assert scaled.k_star.tolist() == base.k_star.tolist()
    assert scaled.n_positive_components.tolist() == base.n_positive_components.tolist()
    assert scaled.top_positive_component_id.tolist() == base.top_positive_component_id.tolist()
    assert math.isclose(scaled.phi.item(), base.phi.item(), rel_tol=1e-9)
    assert math.isclose(scaled.top1_positive_share.item(), base.top1_positive_share.item(), rel_tol=1e-9)
    assert math.isclose(scaled.positive_evidence_sum.item(), base.positive_evidence_sum.item() * scale, rel_tol=1e-9)
    assert math.isclose(scaled.negative_evidence_sum.item(), base.negative_evidence_sum.item() * scale, rel_tol=1e-9)
    assert math.isclose(scaled.top_positive_evidence.item(), base.top_positive_evidence.item() * scale, rel_tol=1e-9)


def test_metamorphic_appending_zero_component_is_a_no_op():
    G_rows = [[3.0, -1.0, 2.0]]
    m320_vals = [4.0]
    ids = [0, 1, 2]
    G, m320, cid = _to_tensors(G_rows, m320_vals, ids)
    base = k.compute_component_evidence_features(G, m320, component_ids=cid)

    G_extra_rows = [[3.0, -1.0, 2.0, 0.0]]
    ids_extra = [0, 1, 2, 99]
    G2, m320_2, cid2 = _to_tensors(G_extra_rows, m320_vals, ids_extra)
    extended = k.compute_component_evidence_features(G2, m320_2, component_ids=cid2)

    assert extended.k_star.tolist() == base.k_star.tolist()
    assert extended.n_positive_components.tolist() == base.n_positive_components.tolist()
    assert extended.top_positive_component_id.tolist() == base.top_positive_component_id.tolist()
    assert math.isclose(extended.phi.item(), base.phi.item(), rel_tol=1e-12)
    assert math.isclose(extended.positive_evidence_sum.item(), base.positive_evidence_sum.item(), rel_tol=1e-12)
    assert math.isclose(extended.negative_evidence_sum.item(), base.negative_evidence_sum.item(), rel_tol=1e-12)


# =====================================================================
# Section 7: conservation-tolerance separation.
# =====================================================================


def test_kernel_signature_has_no_tolerance_or_epsilon_parameter():
    sig = inspect.signature(k.compute_component_evidence_features)
    forbidden_fragments = ("tolerance", "epsilon", "eps", "atol", "rtol")
    for name in sig.parameters:
        lowered = name.lower()
        assert not any(fragment in lowered for fragment in forbidden_fragments), name


def test_kernel_source_never_references_e19_tolerance():
    source = inspect.getsource(sys.modules["src.dfp_phase1b_provenance_features"])
    assert "9.5367431640625e-07" not in source
    assert "conservation_tolerance" not in source
    assert "dfp_phase1b_conservation_tolerance" not in source


def test_k_star_crossing_uses_exact_boundary_no_epsilon():
    # cumsum exactly equals m320 at k=1 -- must cross (>=), not require a margin.
    G, m320, cid = _to_tensors([[3.0, 1.5]], [3.0], [0, 1])
    feats = k.compute_component_evidence_features(G, m320, component_ids=cid)
    assert feats.k_star.item() == 1

    # one ULP below m320 at k=1 -- must NOT cross yet, needs k=2.
    just_below = math.nextafter(3.0, 0.0)
    G2, m320_2, cid2 = _to_tensors([[just_below, 1.5]], [3.0], [0, 1])
    feats2 = k.compute_component_evidence_features(G2, m320_2, component_ids=cid2)
    assert feats2.k_star.item() == 2


# =====================================================================
# Section 8: fatal exceptions propagate unchanged.
# =====================================================================


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_fatal_exceptions_not_swallowed_in_kernel(monkeypatch, exc_type):
    real_sort = torch.sort

    def boom(*a, **kw):
        raise exc_type("fatal during sort")

    monkeypatch.setattr(torch, "sort", boom)
    try:
        G, m320, cid = _to_tensors([[3.0, -1.0, 2.0]], [4.0], [0, 1, 2])
        with pytest.raises(exc_type):
            k.compute_component_evidence_features(G, m320, component_ids=cid)
    finally:
        monkeypatch.setattr(torch, "sort", real_sort)


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_fatal_exceptions_not_swallowed_in_load_identity(monkeypatch, exc_type):
    import tomllib as _tomllib

    def boom(*a, **kw):
        raise exc_type("fatal during load")

    monkeypatch.setattr(_tomllib, "load", boom)
    with pytest.raises(exc_type):
        idmod.load_identity()


# =====================================================================
# Section 9: end-to-end Stage-A row test using the REAL masked-forward
# core on a small, real k=12 synthetic graph, validated by the REAL,
# unmodified Stage-A relational validator.
# =====================================================================


def _recurrence_environment_broken() -> str | None:
    """Narrow, exact-signature environment-gap detector -- the same
    two known gaps ``tests/test_dfp_phase1b_masked_forward.py`` already
    recognizes, replicated here (not imported from that test file) so
    this file stays self-contained. Any OTHER ImportError/
    ModuleNotFoundError is re-raised, never silently skipped."""
    import src.dfp_phase1b_masked_forward as mf

    try:
        mf._load_dependencies()
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            return f"cv2 is not importable in this environment: {exc}"
        raise
    except ImportError as exc:
        if "numpy.core.multiarray failed to import" in str(exc) or "_ARRAY_API" in str(exc):
            return f"cv2 is importable but ABI-incompatible with installed NumPy: {exc}"
        raise
    return None


@pytest.fixture(scope="module")
def _real_stage_a_row_fixture():
    """Build a small, REAL k=12 recurrence graph, run the real masked-
    forward core (build_masked_seed_chunk -> propagate_masked_seed_
    chunk -> compute_signed_contributions) to obtain a genuine G[x,r],
    filter to the disputed pixels the real computation actually made
    valid (m320 > 0), and return (G, m320, component_ids) for those
    rows. Deterministic: no RNG anywhere in this fixture."""
    reason = _recurrence_environment_broken()
    if reason is not None:
        pytest.skip(reason)

    import src.dfp_phase1b_masked_forward as mf

    deps = mf._load_dependencies()
    N, K12 = 15, 12
    node_ids = torch.arange(N, dtype=torch.int64).unsqueeze(1)
    offsets = torch.arange(1, K12 + 1, dtype=torch.int64).unsqueeze(0)
    neighbor_indices = (node_ids + offsets) % N

    i = torch.arange(N, dtype=torch.float64).unsqueeze(1)
    s = torch.arange(K12, dtype=torch.float64).unsqueeze(0)
    raw = 1.0 + torch.remainder(i * 7.0 + s * 3.0, 11.0)
    weights32 = (raw / raw.sum(dim=1, keepdim=True)).to(torch.float32)
    graph = deps.DirectedTopKGraph(
        neighbor_indices=neighbor_indices, transition_weights=weights32, edge_affinities=weights32.clone(),
        self_loop_fallback=torch.zeros(N, dtype=torch.bool), num_nodes=N, k=K12, affinity_power=1.0,
    )
    graph.validate()  # the real committed validator accepts this k=12 graph on its own terms

    R, Q = 3, 2
    contract = mf.MaskedForwardContract(alpha=0.6, steps=8, propagation_dtype=torch.float64, accumulation_dtype=torch.float64)

    unary = torch.zeros(N, Q, dtype=torch.float64)
    unary[:, 0] = 1.0
    bvals = torch.tensor(
        [3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, -2.0, -2.0, -2.0, -2.0, -2.0], dtype=torch.float64,
    )
    unary[:, 1] = bvals

    membership = torch.zeros(R, N, dtype=torch.float64)
    membership[0, 2] = 1.0
    membership[1, 9] = 1.0
    membership[2, 12] = 1.0

    seed = mf.build_masked_seed_chunk(unary, membership, [0, 1, 2], [0, 1], contract=contract, window_index=0)
    prop = mf.propagate_masked_seed_chunk(graph, seed, contract=contract)

    s0_layout = seed.seed_flat.transpose(0, 1).reshape(1, R * Q, 1, N)
    sfinal_layout = prop.t_final.transpose(0, 1).reshape(1, R * Q, 1, N)

    rows = torch.zeros(N, dtype=torch.int64)
    cols = torch.arange(N, dtype=torch.int64)
    a = torch.zeros(N, dtype=torch.int64)
    b = torch.ones(N, dtype=torch.int64)

    chunk = mf.compute_signed_contributions(
        s0_layout, sfinal_layout, component_count=R, query_class_union=[0, 1],
        disputed_pixel_rows=rows, disputed_pixel_cols=cols, disputed_pixel_a=a, disputed_pixel_b=b,
        ori_width=N, component_id_start=0,
    )
    G_all = chunk.values  # real, unmodified [15, 3] signed component-evidence matrix
    row_sums = G_all.sum(dim=1)

    # m0 = 0.0 for every row (frozen m0(x) <= 0 permits equality); with
    # m0 = 0, margin_change = m320 = row_sum exactly, so conservation
    # is exact by construction from REAL computed values -- never a
    # fabricated number. Keep only rows the real computation actually
    # made valid (m320 > 0), exactly as a real Stage-A row assembler
    # would restrict to genuinely disputed pixels.
    valid_mask = row_sums > 0.0
    assert bool(valid_mask.any()), "fixture invariant: at least one real disputed pixel must have m320 > 0"
    valid_cols = cols[valid_mask].tolist()
    G = G_all[valid_mask].clone()
    m320 = row_sums[valid_mask].clone()
    component_ids = torch.tensor([0, 1, 2], dtype=torch.int64)
    return G, m320, component_ids, valid_cols


def test_end_to_end_real_k12_graph_signed_contributions_produce_valid_stage_a_row(_real_stage_a_row_fixture):
    G, m320, component_ids, valid_cols = _real_stage_a_row_fixture
    row_count = G.shape[0]
    assert row_count >= 1

    feats = k.compute_component_evidence_features(G, m320, component_ids=component_ids)

    # Cross-check the kernel's output against the independent Python
    # oracle on these REAL (not hand-typed) G/m320 values.
    G_rows = G.tolist()
    m320_vals = m320.tolist()
    oracle = _oracle_rows(G_rows, m320_vals, component_ids.tolist())
    _assert_matches_oracle(feats, oracle)

    # Assemble a full Stage-A flat-array row set and validate it with
    # the REAL, unmodified relational validator.
    import src.dfp_phase1b_artifact_schema as schema
    import src.dfp_phase1b_conservation_tolerance as tol19

    m0 = np.zeros(row_count, dtype=np.float64)
    m320_np = m320.numpy().astype(np.float64)
    margin_change = m320_np - m0
    eta_net = m320_np / margin_change
    sum_component_evidence = G.sum(dim=1).numpy().astype(np.float64)
    conservation_residual = np.abs(sum_component_evidence - margin_change)

    arrays = {
        "image_row_index": np.zeros(row_count, dtype=np.int64),
        "final_pixel_linear_index": np.array(valid_cols, dtype=np.int64),
        "unary_class_a": np.zeros(row_count, dtype=np.int64),
        "propagated_class_b": np.ones(row_count, dtype=np.int64),
        "t16_label": np.zeros(row_count, dtype=np.int64),
        "t32_label": np.zeros(row_count, dtype=np.int64),
        "t64_label": np.zeros(row_count, dtype=np.int64),
        "m0": m0,
        "m320": m320_np,
        "margin_change": margin_change,
        "eta_net": eta_net,
        "text_cosine_ab": np.full(row_count, 0.5, dtype=np.float64),
        "positive_evidence_sum": feats.positive_evidence_sum.numpy().astype(np.float64),
        "negative_evidence_sum": feats.negative_evidence_sum.numpy().astype(np.float64),
        "sum_component_evidence": sum_component_evidence,
        "conservation_residual": conservation_residual,
        "n_positive_components": feats.n_positive_components.numpy().astype(np.int64),
        "k_star": feats.k_star.numpy().astype(np.int64),
        "phi": feats.phi.numpy().astype(np.float64),
        "top_positive_component_id": feats.top_positive_component_id.numpy().astype(np.int64),
        "top_positive_evidence": feats.top_positive_evidence.numpy().astype(np.float64),
        "top1_positive_share": feats.top1_positive_share.numpy().astype(np.float64),
    }

    conservation_tolerance = tol19.load_identity()["tolerance"]["conservation_tolerance"]
    schema.validate_stage_a_flat_arrays(arrays, row_count=row_count, conservation_tolerance=conservation_tolerance)
    # must not raise -- proves the real unmodified Stage-A validator accepts this kernel's output


def test_end_to_end_fixture_exercises_multi_component_k_star(_real_stage_a_row_fixture):
    """Sanity check that this real fixture is not a trivial all-
    single-component case -- at least one disputed pixel needs more
    than one ranked component to cross m320, genuinely exercising the
    k_star loop, not just its k=1 shortcut."""
    G, m320, component_ids, _valid_cols = _real_stage_a_row_fixture
    feats = k.compute_component_evidence_features(G, m320, component_ids=component_ids)
    assert bool((feats.k_star > 1).any()), feats.k_star.tolist()


# =====================================================================
# Section 10: E21 identity adversarial mutation matrix (temp copies
# only -- the committed E21 TOML is never edited by any test).
# =====================================================================


def _identity_text() -> str:
    return IDENTITY_PATH.read_text()


def _write(tmp_path: Path, text: str, name: str = "e21.toml") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def _mutate_line(text: str, old_line: str, new_line: str) -> str:
    assert old_line in text, f"anchor not found: {old_line!r}"
    return text.replace(old_line, new_line, 1)


def test_committed_e21_identity_loads_clean():
    identity = idmod.load_identity()
    assert identity["identity"]["name"] == "e21-dfp-phase1b-provenance-features"
    assert identity["identity"]["successor_of"] == "e20-dfp-phase1b-diagnostic-execution"
    assert identity["operational_scope"]["scientific_kernel_only"] is True
    assert identity["operational_scope"]["runner_implemented_by_this_task"] is False
    assert identity["kernel_binding"]["module"] == "src.dfp_phase1b_provenance_features"
    assert identity["kernel_binding"]["function"] == "compute_component_evidence_features"


def test_e21_kernel_module_hash_binding_matches_live_bytes():
    import hashlib
    identity = idmod.load_identity()
    live_hash = hashlib.sha256((ROOT / "src/dfp_phase1b_provenance_features.py").read_bytes()).hexdigest()
    assert identity["dependency_bindings"]["provenance_features_module"]["sha256"] == live_hash


def test_e21_ancestor_commit_is_e20_commit():
    identity = idmod.load_identity()
    assert identity["identity"]["required_ancestor_commit"] == "38df8087abd3e0fbc163523c471c2cdd01e7d96c"
    assert idmod.REQUIRED_ANCESTOR_COMMIT == "38df8087abd3e0fbc163523c471c2cdd01e7d96c"


def test_e21_authorization_boundary_all_false_where_required():
    identity = idmod.load_identity()
    ab = identity["authorization_boundary"]
    for key in (
        "runner_implemented_by_this_task", "profiler_implemented_by_this_task",
        "checkpoint_writer_implemented_by_this_task", "result_writer_implemented_by_this_task",
        "cli_execution_command_implemented_by_this_task", "slurm_launcher_implemented_by_this_task",
        "real_inference_executed", "gpu_execution_performed", "coco_stuff_dataset_accessed",
        "phase1b_500_image_execution_authorized", "phase1b_5000_image_exact_provenance_authorized",
    ):
        assert ab[key] is False, key
    assert ab["authority_and_kernel_only_task"] is True
    assert ab["scientific_kernel_only"] is True


@pytest.mark.parametrize(
    "old,new",
    [
        ("scientific_kernel_only = true", "scientific_kernel_only = false"),
        ("runner_implemented_by_this_task = false", "runner_implemented_by_this_task = true"),
        ("real_inference_executed = false", "real_inference_executed = true"),
        ("gpu_execution_performed = false", "gpu_execution_performed = true"),
        ("phase1b_500_image_execution_authorized = false", "phase1b_500_image_execution_authorized = true"),
        ("phase1b_5000_image_exact_provenance_authorized = false", "phase1b_5000_image_exact_provenance_authorized = true"),
        ("new_scientific_convention_invented_by_this_task = false", "new_scientific_convention_invented_by_this_task = true"),
        ('function = "compute_component_evidence_features"', 'function = "compute_features"'),
        ('result_type = "ComponentEvidenceFeatures"', 'result_type = "Features"'),
        ("formulas_re_derived_by_this_kernel = false", "formulas_re_derived_by_this_kernel = true"),
        ("tolerance_or_epsilon_used_in_feature_construction = false", "tolerance_or_epsilon_used_in_feature_construction = true"),
        ('accumulation_dtype = "float64"', 'accumulation_dtype = "float32"'),
        ("k_star_dtype = \"int64\"", "k_star_dtype = \"float64\""),
        ("no_new_scientific_formula_invented = true", "no_new_scientific_formula_invented = false"),
    ],
)
def test_e21_relational_mutation_matrix_rejected(tmp_path, old, new):
    text = _mutate_line(_identity_text(), old, new)
    p = _write(tmp_path, text)
    with pytest.raises(idmod.DfpPhase1bProvenanceFeaturesIdentityError):
        idmod.load_identity(path=p, repo_root=ROOT)


def test_e21_missing_field_rejected(tmp_path):
    text = _mutate_line(_identity_text(), "new_scientific_convention_invented_by_this_task = false\n", "")
    p = _write(tmp_path, text)
    with pytest.raises(idmod.DfpPhase1bProvenanceFeaturesIdentityError):
        idmod.load_identity(path=p, repo_root=ROOT)


def test_e21_extra_field_rejected(tmp_path):
    text = _mutate_line(
        _identity_text(), "scientific_kernel_only = true", "scientific_kernel_only = true\nsurprise_field = 1",
    )
    p = _write(tmp_path, text)
    with pytest.raises(idmod.DfpPhase1bProvenanceFeaturesIdentityError):
        idmod.load_identity(path=p, repo_root=ROOT)


def test_e21_missing_dependency_binding_rejected(tmp_path):
    text = _mutate_line(
        _identity_text(),
        '[dependency_bindings.masked_forward_core]\nrelative_path = "src/dfp_phase1b_masked_forward.py"\nsha256 = "72f9b7093379cdbde14691e0dcc97c54eb2563d8bc8690da027409d6c81b56b3"\nuntouched_by_this_identity = true\n',
        '',
    )
    p = _write(tmp_path, text)
    with pytest.raises(idmod.DfpPhase1bProvenanceFeaturesIdentityError):
        idmod.load_identity(path=p, repo_root=ROOT)


@pytest.mark.parametrize(
    "old,new",
    [
        ('relative_path = "evaluation_identities/e15_dfp_phase1b_protocol.toml"',
         'relative_path = "../evaluation_identities/e15_dfp_phase1b_protocol.toml"'),
        ('relative_path = "evaluation_identities/e15_dfp_phase1b_protocol.toml"',
         'relative_path = "/etc/passwd"'),
        ('relative_path = "src/dfp_phase1b_masked_forward.py"',
         'relative_path = "evaluation_identities/e15_dfp_phase1b_protocol.toml"'),
    ],
)
def test_e21_path_traversal_and_substitution_rejected(tmp_path, old, new):
    text = _mutate_line(_identity_text(), old, new)
    p = _write(tmp_path, text)
    with pytest.raises(idmod.DfpPhase1bProvenanceFeaturesIdentityError):
        idmod.load_identity(path=p, repo_root=ROOT)


def test_e21_wrong_dependency_hash_rejected(tmp_path):
    text = _mutate_line(
        _identity_text(),
        '[dependency_bindings.provenance_features_module]\nrelative_path = "src/dfp_phase1b_provenance_features.py"\nsha256 = "ace237fe454834a7a0999f16aaf23f01554759d6630a8abb01418800f3becd24"',
        '[dependency_bindings.provenance_features_module]\nrelative_path = "src/dfp_phase1b_provenance_features.py"\nsha256 = "0000000000000000000000000000000000000000000000000000000000000000"',
    )
    p = _write(tmp_path, text)
    with pytest.raises(idmod.DfpPhase1bProvenanceFeaturesIdentityError):
        idmod.load_identity(path=p, repo_root=ROOT)


def test_e21_not_toml_wrapped_in_domain_exception(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("{{{ not toml")
    with pytest.raises(idmod.DfpPhase1bProvenanceFeaturesIdentityError):
        idmod.load_identity(path=p, repo_root=ROOT)


def test_e21_untampered_temp_copy_still_loads_clean(tmp_path):
    """Harness sanity check: an EXACT (unmutated) copy of the committed
    TOML, loaded via the same path+repo_root machinery the mutation
    tests use, must still load cleanly -- proving rejection above comes
    from the injected mutation, not from the temp-copy mechanism
    itself."""
    p = _write(tmp_path, _identity_text())
    identity = idmod.load_identity(path=p, repo_root=ROOT)
    assert identity["identity"]["name"] == "e21-dfp-phase1b-provenance-features"


# =====================================================================
# Section 11: CPU-only CLI smoke tests.
# =====================================================================


def test_cli_verify_protocol_exit_0():
    r = subprocess.run([sys.executable, str(CLI_PATH), "verify-protocol"], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    assert "E21 identity verified" in r.stdout


def test_cli_show_contract_exit_0():
    r = subprocess.run([sys.executable, str(CLI_PATH), "show-contract"], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    assert "SCIENTIFIC KERNEL ONLY" in r.stdout
    assert "compute_component_evidence_features" in r.stdout


def test_cli_has_no_execution_run_resume_or_submit_command():
    """Check the actual argparse SUBCOMMAND set, not the whole --help
    text (which legitimately contains the substring "run" in the
    program's own filename, run_dfp_phase1b_provenance_features.py)."""
    r = subprocess.run([sys.executable, str(CLI_PATH), "--help"], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    match = re.search(r"\{([^}]*)\}", r.stdout)
    assert match is not None, r.stdout
    subcommands = {s.strip() for s in match.group(1).split(",")}
    assert subcommands == {"verify-protocol", "show-contract"}, subcommands


def test_cli_never_imports_torch(tmp_path):
    """Hermetic subprocess check: 'verify-protocol' and 'show-contract'
    must complete without ever importing torch -- the CLI only hash-
    verifies the E21 authority, it never loads the numerical kernel."""
    child = tmp_path / "iv_hermetic.py"
    child.write_text(
        "import sys\n"
        "assert 'torch' not in sys.modules, 'torch imported before CLI ran'\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        f"sys.argv = [{str(CLI_PATH)!r}, 'verify-protocol']\n"
        "import runpy\n"
        "try:\n"
        f"    runpy.run_path({str(CLI_PATH)!r}, run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 0, f'CLI exited with {exc.code}'\n"
        "assert 'torch' not in sys.modules, 'torch was imported by the CLI'\n"
        "print('HERMETIC_OK')\n"
    )
    r = subprocess.run([sys.executable, "-I", str(child)], capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "HERMETIC_OK" in r.stdout


def test_cli_ordinary_failure_exit_2_no_traceback(tmp_path):
    """Pointing --repo-root at a directory with no E21 identity file at
    all must fail cleanly (exit 2, domain-exception message) -- never
    an uncaught Python traceback."""
    empty_repo_root = tmp_path / "empty_repo"
    empty_repo_root.mkdir()
    r = subprocess.run(
        [sys.executable, str(CLI_PATH), "--repo-root", str(empty_repo_root), "verify-protocol"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert r.returncode == 2, r.stdout + r.stderr
    assert "Traceback (most recent call last)" not in r.stderr
    assert "DFP PHASE 1B PROVENANCE FEATURES FAIL" in r.stderr
