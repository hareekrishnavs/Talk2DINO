import dataclasses
import importlib.util
import inspect
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).parents[1]
GRAPH_PATH = (
    ROOT
    / "src/open_vocabulary_segmentation/models/dinotext/cover_dr/graph.py"
)


def _load_graph_module():
    spec = importlib.util.spec_from_file_location(
        "directed_topk_graph_under_test", GRAPH_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


graph_module = _load_graph_module()
DirectedTopKGraph = graph_module.DirectedTopKGraph
build_directed_topk_graph = graph_module.build_directed_topk_graph


def _normalize(value, dtype=torch.float32):
    return F.normalize(torch.as_tensor(value, dtype=dtype), dim=-1)


def _reference(features, k, power):
    features = features.detach().to(dtype=torch.float32, device="cpu")
    affinity = (features @ features.T).clamp_min(0).pow(power)
    num_nodes = features.shape[0]
    indices = torch.empty(num_nodes, k, dtype=torch.int64)
    edges = torch.empty(num_nodes, k, dtype=torch.float32)
    weights = torch.zeros(num_nodes, k, dtype=torch.float32)
    fallback = torch.zeros(num_nodes, dtype=torch.bool)
    for source in range(num_nodes):
        destinations = [value for value in range(num_nodes) if value != source]
        destinations.sort(key=lambda destination: (-float(affinity[source, destination]), destination))
        selected = destinations[:k]
        indices[source] = torch.tensor(selected)
        edges[source] = affinity[source, selected]
        total = edges[source].sum()
        if total > 0:
            weights[source] = edges[source] / total
        else:
            fallback[source] = True
            indices[source, 0] = source
            weights[source, 0] = 1
    dense = torch.zeros(num_nodes, num_nodes)
    dense.scatter_add_(1, indices, weights)
    return indices, edges, weights, fallback, dense


def _non_tied_features(dtype=torch.float32, device="cpu"):
    value = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.2, 0.7, 0.685565],
            [-0.5, 0.1, 0.860233],
            [-0.8, -0.5, 0.331662],
        ],
        dtype=dtype,
        device=device,
    )
    return F.normalize(value, dim=-1)


def test_canonical_defaults():
    signature = inspect.signature(build_directed_topk_graph)
    assert signature.parameters["k"].default == 12
    assert signature.parameters["affinity_power"].default == 3.0


def test_exact_affinity_formula_and_rowwise_topk_match_reference():
    features = _non_tied_features()
    actual = build_directed_topk_graph(features, k=3, affinity_power=3.0)
    indices, edges, weights, fallback, _dense = _reference(features, 3, 3.0)
    assert torch.equal(actual.neighbor_indices.cpu(), indices)
    torch.testing.assert_close(actual.edge_affinities.cpu(), edges)
    torch.testing.assert_close(actual.transition_weights.cpu(), weights)
    assert torch.equal(actual.self_loop_fallback.cpu(), fallback)


def test_relu_removes_negative_cosines():
    features = _normalize([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    graph = build_directed_topk_graph(features, k=2)
    assert torch.count_nonzero(graph.edge_affinities) == 0
    assert graph.self_loop_fallback.all()


def test_power_is_applied_before_normalization():
    features = _normalize([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8]])
    graph = build_directed_topk_graph(features, k=2, affinity_power=3)
    source = 0
    destinations = graph.neighbor_indices[source]
    cosine = features[source] @ features[destinations].T
    expected_edges = cosine.clamp_min(0).pow(3)
    assert torch.equal(graph.edge_affinities[source], expected_edges)
    torch.testing.assert_close(
        graph.transition_weights[source], expected_edges / expected_edges.sum()
    )


def test_self_is_excluded_before_selection():
    graph = build_directed_topk_graph(_non_tied_features(), k=4)
    rows = torch.arange(graph.num_nodes)[:, None]
    assert not torch.any(graph.neighbor_indices == rows)


def test_directed_asymmetric_fixture_has_nonreciprocal_edge():
    angles = torch.deg2rad(torch.tensor([0.0, 10.0, 11.0, 150.0]))
    features = torch.stack((angles.cos(), angles.sin()), dim=-1)
    graph = build_directed_topk_graph(features, k=1)
    dense = graph.to_dense()
    assert graph.neighbor_indices[0, 0] == 1
    assert graph.neighbor_indices[1, 0] == 2
    assert dense[0, 1] > 0
    assert dense[1, 0] == 0
    assert not torch.equal(dense, dense.T)


@pytest.mark.parametrize("operation", ["average", "maximum", "sum", "mutual"])
def test_anti_symmetrization_fixture(operation):
    angles = torch.deg2rad(torch.tensor([0.0, 10.0, 11.0, 150.0]))
    features = torch.stack((angles.cos(), angles.sin()), dim=-1)
    directed = build_directed_topk_graph(features, k=1).to_dense()
    if operation == "average":
        wrong = (directed + directed.T) / 2
    elif operation == "maximum":
        wrong = torch.maximum(directed, directed.T)
    elif operation == "sum":
        wrong = directed + directed.T
    else:
        wrong = directed * (directed.T > 0)
    assert not torch.equal(directed, wrong)


def test_positive_rows_sum_to_one_and_zero_slots_stay_zero():
    features = _normalize([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    graph = build_directed_topk_graph(features, k=2)
    torch.testing.assert_close(
        graph.transition_weights.sum(dim=1), torch.ones(3)
    )
    ordinary = ~graph.self_loop_fallback
    zero_slots = graph.edge_affinities[ordinary] == 0
    assert torch.count_nonzero(zero_slots) > 0
    assert torch.count_nonzero(graph.transition_weights[ordinary][zero_slots]) == 0


def test_zero_rows_use_only_documented_self_loop_fallback():
    features = torch.eye(4)
    graph = build_directed_topk_graph(features, k=2)
    assert graph.self_loop_fallback.all()
    assert torch.equal(graph.neighbor_indices[:, 0], torch.arange(4))
    assert torch.equal(graph.transition_weights[:, 0], torch.ones(4))
    assert torch.count_nonzero(graph.transition_weights[:, 1:]) == 0
    assert torch.count_nonzero(graph.edge_affinities) == 0


def test_ties_use_lower_destination_index_without_perturbing_affinity():
    features = torch.eye(5)
    graph = build_directed_topk_graph(features, k=3)
    assert torch.equal(graph.neighbor_indices[4], torch.tensor([4, 1, 2]))
    assert torch.count_nonzero(graph.edge_affinities[4]) == 0


def test_tie_at_kth_boundary_is_deterministic():
    features = _normalize(
        [[1.0, 0.0], [0.5, math.sqrt(0.75)], [0.5, -math.sqrt(0.75)], [-1.0, 0.0]]
    )
    first = build_directed_topk_graph(features, k=1)
    second = build_directed_topk_graph(features, k=1)
    assert first.neighbor_indices[0, 0] == 1
    assert torch.equal(first.neighbor_indices, second.neighbor_indices)
    assert torch.equal(first.edge_affinities, second.edge_affinities)
    assert torch.equal(first.transition_weights, second.transition_weights)


def test_input_is_not_mutated_and_outputs_are_detached_owned_storage():
    source = _non_tied_features().requires_grad_(True)
    original = source.detach().clone()
    graph = build_directed_topk_graph(source, k=2)
    assert torch.equal(source.detach(), original)
    for value in (
        graph.neighbor_indices,
        graph.transition_weights,
        graph.edge_affinities,
        graph.self_loop_fallback,
    ):
        assert value.grad_fn is None
        assert value.requires_grad is False
        assert value.is_contiguous()
        assert value.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


def test_output_shapes_dtypes_and_device():
    features = _non_tied_features(dtype=torch.float64)
    graph = build_directed_topk_graph(features, k=2)
    assert graph.neighbor_indices.shape == (5, 2)
    assert graph.transition_weights.shape == (5, 2)
    assert graph.edge_affinities.shape == (5, 2)
    assert graph.self_loop_fallback.shape == (5,)
    assert graph.neighbor_indices.dtype == torch.int64
    assert graph.transition_weights.dtype == torch.float32
    assert graph.edge_affinities.dtype == torch.float32
    assert graph.self_loop_fallback.dtype == torch.bool
    assert all(value.device == features.device for value in graph._tensor_fields())


@pytest.mark.parametrize("k", [1, 4])
def test_boundary_valid_k_values(k):
    graph = build_directed_topk_graph(_non_tied_features(), k=k)
    assert graph.neighbor_indices.shape == (5, k)


@pytest.mark.parametrize("k", [0, -1, 5, 6, True])
def test_invalid_k_fails(k):
    with pytest.raises(ValueError, match=r"0 < k < N"):
        build_directed_topk_graph(_non_tied_features(), k=k)


@pytest.mark.parametrize(
    "power", [0.0, -1.0, float("nan"), float("inf"), -float("inf"), True]
)
def test_invalid_affinity_power_fails(power):
    with pytest.raises(ValueError, match="affinity_power"):
        build_directed_topk_graph(
            _non_tied_features(), k=2, affinity_power=power
        )


@pytest.mark.parametrize("shape", [(4,), (1, 2, 3)])
def test_rank_mismatch_fails(shape):
    with pytest.raises(ValueError, match=r"\[N, D\]"):
        build_directed_topk_graph(torch.ones(shape), k=1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_features_fail(value):
    features = _non_tied_features()
    features[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        build_directed_topk_graph(features, k=2)


def test_zero_norm_features_fail():
    features = _non_tied_features()
    features[0].zero_()
    with pytest.raises(ValueError, match="non-zero"):
        build_directed_topk_graph(features, k=2)


def test_clearly_unnormalized_features_fail():
    with pytest.raises(ValueError, match="already be L2-normalized"):
        build_directed_topk_graph(_non_tied_features() * 2, k=2)


def test_dtype_aware_norm_tolerance_accepts_normalized_half_features():
    features = F.normalize(torch.randn(20, 8), dim=-1).to(torch.float16)
    graph = build_directed_topk_graph(features, k=3)
    assert graph.transition_weights.dtype == torch.float32


def test_sparse_vector_matmul_matches_independent_dense_reference():
    features = _non_tied_features()
    graph = build_directed_topk_graph(features, k=2)
    rhs = torch.randn(5, dtype=torch.float64)
    _indices, _edges, _weights, _fallback, dense = _reference(features, 2, 3)
    actual = graph.matmul(rhs)
    expected = dense.to(torch.float64) @ rhs
    assert actual.dtype == rhs.dtype
    torch.testing.assert_close(actual, expected)


def test_sparse_matrix_matmul_matches_independent_dense_reference():
    features = _non_tied_features()
    graph = build_directed_topk_graph(features, k=3)
    rhs = torch.randn(5, 7)
    _indices, _edges, _weights, _fallback, dense = _reference(features, 3, 3)
    torch.testing.assert_close(graph.matmul(rhs), dense @ rhs)


def test_matmul_validates_rhs():
    graph = build_directed_topk_graph(_non_tied_features(), k=2)
    with pytest.raises(ValueError, match="node mismatch"):
        graph.matmul(torch.randn(4))
    with pytest.raises(ValueError, match=r"\[N\].*\[N, R\]"):
        graph.matmul(torch.randn(5, 2, 1))
    with pytest.raises(TypeError, match="floating point"):
        graph.matmul(torch.ones(5, dtype=torch.int64))


def test_to_dense_matches_reference_and_preserves_orientation():
    features = _non_tied_features()
    graph = build_directed_topk_graph(features, k=2)
    _indices, _edges, _weights, _fallback, expected = _reference(features, 2, 3)
    dense = graph.to_dense()
    torch.testing.assert_close(dense, expected)
    rhs = torch.arange(5, dtype=torch.float32)
    torch.testing.assert_close(graph.matmul(rhs), dense @ rhs)


def test_container_is_frozen_and_owns_constructor_inputs():
    features = _non_tied_features()
    graph = build_directed_topk_graph(features, k=2)
    source_indices = graph.neighbor_indices.clone()
    source_weights = graph.transition_weights.clone()
    source_affinities = graph.edge_affinities.clone()
    source_fallback = graph.self_loop_fallback.clone()
    copied = DirectedTopKGraph(
        source_indices,
        source_weights,
        source_affinities,
        source_fallback,
        graph.num_nodes,
        graph.k,
        graph.affinity_power,
    )
    source_indices.zero_()
    source_weights.zero_()
    source_affinities.zero_()
    source_fallback.logical_not_()
    assert torch.equal(copied.neighbor_indices, graph.neighbor_indices)
    assert torch.equal(copied.transition_weights, graph.transition_weights)
    with pytest.raises(dataclasses.FrozenInstanceError):
        copied.k = 1


def test_canonical_storage_is_sparse_and_does_not_retain_dense_tensor():
    torch.manual_seed(91)
    features = F.normalize(torch.randn(1024, 768), dim=-1)
    graph = build_directed_topk_graph(features)
    assert graph.neighbor_indices.shape == (1024, 12)
    assert graph.transition_weights.shape == (1024, 12)
    assert graph.edge_affinities.shape == (1024, 12)
    assert all(tuple(value.shape) != (1024, 1024) for value in graph._tensor_fields())
    persistent_bytes = sum(
        value.numel() * value.element_size() for value in graph._tensor_fields()
    )
    assert persistent_bytes == 197632


@dataclass(frozen=True)
class _SnapshotFixture:
    dino_features: torch.Tensor


def test_snapshot_features_pass_without_mutation_or_storage_aliasing():
    snapshot = _SnapshotFixture(_non_tied_features().clone())
    source = snapshot.dino_features.clone()
    graph = build_directed_topk_graph(snapshot.dino_features, k=2)
    assert torch.equal(snapshot.dino_features, source)
    for value in graph._tensor_fields():
        assert value.untyped_storage().data_ptr() != snapshot.dino_features.untyped_storage().data_ptr()
    graph.transition_weights.zero_()
    graph.edge_affinities.zero_()
    assert torch.equal(snapshot.dino_features, source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_and_cuda_match_on_non_tied_fixture():
    features = _non_tied_features()
    cpu = build_directed_topk_graph(features, k=2)
    cuda = build_directed_topk_graph(features.cuda(), k=2)
    assert torch.equal(cpu.neighbor_indices, cuda.neighbor_indices.cpu())
    torch.testing.assert_close(cpu.edge_affinities, cuda.edge_affinities.cpu())
    torch.testing.assert_close(cpu.transition_weights, cuda.transition_weights.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_and_cuda_ties_use_same_lower_index_rule():
    features = torch.eye(8)
    cpu = build_directed_topk_graph(features, k=3)
    cuda = build_directed_topk_graph(features.cuda(), k=3)
    assert torch.equal(cpu.neighbor_indices, cuda.neighbor_indices.cpu())
