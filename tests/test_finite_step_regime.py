"""CPU-only synthetic tests for the reusable finite-step propagation
kernel, matched k11-from-k12 graph construction, and FP64
reference/diagnostic utilities. Never requires CUDA; loaded via file-path
import with lightweight package stand-ins, mirroring
``tests/test_image_window_cache.py``."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
COVER_DR_PACKAGE_PATH = ROOT / "src/open_vocabulary_segmentation/models/dinotext/cover_dr"


def _install_models_stub() -> None:
    if "models" not in sys.modules:
        stub = types.ModuleType("models")
        stub.__path__ = []
        sys.modules["models"] = stub
    if "models.dinotext" not in sys.modules:
        stub = types.ModuleType("models.dinotext")
        stub.__path__ = []
        sys.modules["models.dinotext"] = stub


def _load_cover_dr_package():
    _install_models_stub()
    spec = importlib.util.spec_from_file_location(
        "models.dinotext.cover_dr",
        COVER_DR_PACKAGE_PATH / "__init__.py",
        submodule_search_locations=[str(COVER_DR_PACKAGE_PATH)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cover_dr = _load_cover_dr_package()

FiniteStepRegimeError = cover_dr.FiniteStepRegimeError
build_matched_k11_from_k12 = cover_dr.build_matched_k11_from_k12
compute_matched_graph_diagnostics = cover_dr.compute_matched_graph_diagnostics
finite_step_propagate = cover_dr.finite_step_propagate
dense_fp64_equilibrium_reference = cover_dr.dense_fp64_equilibrium_reference
compute_condition_diagnostics = cover_dr.compute_condition_diagnostics
compare_snapshots = cover_dr.compare_snapshots
compute_matched_delta = cover_dr.compute_matched_delta
delta_stability_error = cover_dr.delta_stability_error
diagnostic_pixel_argmax_map = cover_dr.diagnostic_pixel_argmax_map
compare_label_sensitivity = cover_dr.compare_label_sensitivity
build_directed_topk_graph = cover_dr.build_directed_topk_graph
DirectedTopKGraph = cover_dr.DirectedTopKGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _random_graph(num_nodes=30, dim=12, k=12, affinity_power=3.0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    features = torch.nn.functional.normalize(torch.randn(num_nodes, dim, generator=generator), dim=-1)
    return build_directed_topk_graph(features, k=k, affinity_power=affinity_power)


def _graph_with_forced_fallback_rows(num_nodes=15, fallback_rows=(0, 3), extra_dim=8, seed=1):
    """Construct a graph where specific rows have exactly zero cosine
    similarity to every other node (fallback and non-fallback alike),
    forcing the canonical zero-affinity self-loop fallback for exactly
    those rows -- and only those rows.

    Each row gets a unique one-hot "identity" component (dimension ``row``
    of a ``num_nodes``-wide block), so two distinct rows' identity
    components never overlap and contribute exactly 0 to their cosine
    similarity. Fallback rows are pure identity vectors (zero everywhere
    else), which drives their cosine similarity to literally everyone
    (fallback or not) to exactly 0. Non-fallback rows additionally carry a
    shared random component in extra dimensions, giving them nonzero
    (positive for at least one other row, with high probability) affinity
    among themselves.
    """
    dim = num_nodes + extra_dim
    generator = torch.Generator().manual_seed(seed)
    features = torch.zeros(num_nodes, dim)
    for row in range(num_nodes):
        features[row, row] = 1.0
    non_fallback = [row for row in range(num_nodes) if row not in fallback_rows]
    if non_fallback:
        shared = torch.randn(len(non_fallback), extra_dim, generator=generator)
        for position, row in enumerate(non_fallback):
            features[row, num_nodes:] = shared[position]
    features = torch.nn.functional.normalize(features, dim=-1)
    return build_directed_topk_graph(features, k=12, affinity_power=3.0)


# ---------------------------------------------------------------------------
# 1-4: recurrence mathematics, exactly one (1-alpha) factor, snapshot counts,
# continuous-run equivalence to separate runs
# ---------------------------------------------------------------------------


def test_recurrence_matches_hand_computed_closed_form_for_two_steps():
    graph = _random_graph(num_nodes=10, dim=6, k=3)
    s0 = torch.randn(10, 4)
    alpha = 0.5
    trace = finite_step_propagate(graph, s0, alpha=alpha, steps=2, snapshot_steps=(0, 1, 2))
    p1_expected = alpha * graph.matmul(s0) + (1 - alpha) * s0
    p2_expected = alpha * graph.matmul(p1_expected) + (1 - alpha) * s0
    assert torch.allclose(trace.snapshots[0], s0)
    assert torch.allclose(trace.snapshots[1], p1_expected)
    assert torch.allclose(trace.snapshots[2], p2_expected)


def test_exactly_one_one_minus_alpha_factor_per_step():
    # A wrong implementation applying (1-alpha) twice per step would diverge
    # sharply from the hand-computed single-factor closed form after a few
    # steps; confirm the real kernel matches the single-factor recurrence
    # exactly (within floating-point tolerance) over 5 steps.
    graph = _random_graph(num_nodes=8, dim=5, k=3)
    s0 = torch.randn(8, 3)
    alpha = 0.9
    trace = finite_step_propagate(graph, s0, alpha=alpha, steps=5, snapshot_steps=(5,))
    p = s0.clone()
    for _ in range(5):
        p = alpha * graph.matmul(p) + (1 - alpha) * s0
    assert torch.allclose(trace.snapshots[5], p, atol=1e-6)


def test_snapshot_counts_and_keys_match_registered_steps():
    graph = _random_graph(num_nodes=10, dim=6, k=3)
    s0 = torch.randn(10, 4)
    trace = finite_step_propagate(graph, s0, alpha=0.98, steps=640, snapshot_steps=(0, 160, 320, 640))
    assert set(trace.snapshots.keys()) == {0, 160, 320, 640}
    assert trace.steps_completed == 640


def test_continuous_run_equivalent_to_separate_restart_runs():
    graph = _random_graph(num_nodes=12, dim=6, k=4)
    s0 = torch.randn(12, 5)
    alpha = 0.98
    continuous = finite_step_propagate(graph, s0, alpha=alpha, steps=640, snapshot_steps=(160, 320, 640))
    for steps in (160, 320, 640):
        restarted = finite_step_propagate(graph, s0, alpha=alpha, steps=steps, snapshot_steps=(steps,))
        assert torch.equal(continuous.snapshots[steps], restarted.snapshots[steps])


def test_snapshot_step_zero_returns_s0():
    graph = _random_graph(num_nodes=10, dim=6, k=3)
    s0 = torch.randn(10, 4)
    trace = finite_step_propagate(graph, s0, alpha=0.9, steps=3, snapshot_steps=(0,))
    assert torch.equal(trace.snapshots[0], s0)


def test_snapshot_step_greater_than_total_steps_rejected():
    graph = _random_graph(num_nodes=10, dim=6, k=3)
    s0 = torch.randn(10, 4)
    with pytest.raises(FiniteStepRegimeError, match="exceeds"):
        finite_step_propagate(graph, s0, alpha=0.9, steps=10, snapshot_steps=(20,))


# ---------------------------------------------------------------------------
# 5. Asymmetric directed graphs
# ---------------------------------------------------------------------------


def test_graph_is_directed_and_not_symmetrized():
    graph = _random_graph(num_nodes=20, dim=8, k=12, seed=3)
    dense = graph.to_dense()
    assert not torch.equal(dense, dense.T)


def test_finite_step_propagate_never_symmetrizes_the_graph():
    graph = _random_graph(num_nodes=20, dim=8, k=12, seed=3)
    dense_before = graph.to_dense().clone()
    s0 = torch.randn(20, 4)
    finite_step_propagate(graph, s0, alpha=0.9, steps=5, snapshot_steps=(5,))
    assert torch.equal(graph.to_dense(), dense_before)


# ---------------------------------------------------------------------------
# 6-9: fallback rows, k11 prefix construction, row renormalization,
# prefix mismatch detection
# ---------------------------------------------------------------------------


def test_fallback_rows_preserved_and_k_invariant():
    graph12 = _graph_with_forced_fallback_rows(num_nodes=15, fallback_rows=(0, 3))
    graph11 = build_matched_k11_from_k12(graph12)
    diagnostics = compute_matched_graph_diagnostics(graph12, graph11)
    assert diagnostics.fallback_row_count_k12 >= 2
    assert diagnostics.fallback_row_count_k11 == diagnostics.fallback_row_count_k12
    assert diagnostics.fallback_row_mismatch_count == 0
    assert graph11.self_loop_fallback[0]
    assert graph11.self_loop_fallback[3]
    assert graph11.neighbor_indices[0, 0].item() == 0
    assert graph11.transition_weights[0, 0].item() == 1.0


def test_k11_neighbor_indices_are_exact_prefix_of_k12():
    graph12 = _random_graph(num_nodes=25, dim=10, k=12, seed=4)
    graph11 = build_matched_k11_from_k12(graph12)
    assert torch.equal(graph11.neighbor_indices, graph12.neighbor_indices[:, :11])


def test_k11_row_weights_are_renormalized_not_truncated_k12_weights():
    graph12 = _random_graph(num_nodes=25, dim=10, k=12, seed=4)
    graph11 = build_matched_k11_from_k12(graph12)
    row_sums = graph11.transition_weights.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    # For an ordinary row, the retained weights must equal
    # affinity / sum(retained 11 affinities), not affinity / sum(12 affinities).
    ordinary = (~graph11.self_loop_fallback).nonzero(as_tuple=True)[0][0].item()
    affinities_11 = graph12.edge_affinities[ordinary, :11]
    expected = affinities_11 / affinities_11.sum()
    assert torch.allclose(graph11.transition_weights[ordinary], expected, atol=1e-6)


def test_build_matched_k11_never_calls_topk_independently():
    # Structural guarantee: build_matched_k11_from_k12 takes an
    # already-built k=12 graph and never re-derives neighbor selection from
    # raw features -- confirmed by requiring only a DirectedTopKGraph input
    # (no dino_features parameter exists on this function at all).
    import inspect

    signature = inspect.signature(build_matched_k11_from_k12)
    assert list(signature.parameters) == ["graph12"]


def test_prefix_mismatch_detected_when_k11_indices_are_tampered():
    graph12 = _random_graph(num_nodes=20, dim=8, k=12, seed=5)
    graph11 = build_matched_k11_from_k12(graph12)
    tampered_indices = graph11.neighbor_indices.clone()
    # Swap two ordinary-row entries to break the prefix relationship.
    ordinary_rows = (~graph11.self_loop_fallback).nonzero(as_tuple=True)[0]
    row = ordinary_rows[0].item()
    tampered_indices[row, 0], tampered_indices[row, 1] = (
        tampered_indices[row, 1].clone(),
        tampered_indices[row, 0].clone(),
    )
    tampered = DirectedTopKGraph(
        neighbor_indices=tampered_indices,
        transition_weights=graph11.transition_weights.clone(),
        edge_affinities=graph11.edge_affinities.clone(),
        self_loop_fallback=graph11.self_loop_fallback.clone(),
        num_nodes=graph11.num_nodes,
        k=11,
        affinity_power=graph11.affinity_power,
    )
    with pytest.raises(FiniteStepRegimeError, match="not an exact prefix"):
        compute_matched_graph_diagnostics(graph12, tampered)


def test_wrong_parent_k_rejected():
    graph_k8 = _random_graph(num_nodes=15, dim=6, k=8, seed=6)
    with pytest.raises(FiniteStepRegimeError, match="k=12 parent graph"):
        build_matched_k11_from_k12(graph_k8)


def test_row_sum_and_negative_weight_diagnostics_are_zero_for_valid_graphs():
    graph12 = _random_graph(num_nodes=20, dim=8, k=12, seed=7)
    graph11 = build_matched_k11_from_k12(graph12)
    diagnostics = compute_matched_graph_diagnostics(graph12, graph11)
    assert diagnostics.negative_weight_count_k11 == 0
    assert diagnostics.negative_weight_count_k12 == 0
    assert diagnostics.non_fallback_self_edge_count_k11 == 0
    assert diagnostics.non_fallback_self_edge_count_k12 == 0
    assert diagnostics.row_sum_max_error_k11 < 1e-4
    assert diagnostics.row_sum_max_error_k12 < 1e-4


# ---------------------------------------------------------------------------
# 10. Input nonmutation
# ---------------------------------------------------------------------------


def test_finite_step_propagate_does_not_mutate_s0():
    graph = _random_graph(num_nodes=10, dim=6, k=3)
    s0 = torch.randn(10, 4)
    before = s0.clone()
    finite_step_propagate(graph, s0, alpha=0.9, steps=20, snapshot_steps=(20,))
    assert torch.equal(s0, before)


def test_build_matched_k11_does_not_mutate_graph12():
    graph12 = _random_graph(num_nodes=20, dim=8, k=12, seed=8)
    indices_before = graph12.neighbor_indices.clone()
    weights_before = graph12.transition_weights.clone()
    build_matched_k11_from_k12(graph12)
    assert torch.equal(graph12.neighbor_indices, indices_before)
    assert torch.equal(graph12.transition_weights, weights_before)


def test_dense_equilibrium_reference_does_not_mutate_inputs():
    graph = _random_graph(num_nodes=15, dim=6, k=6, seed=9)
    s0 = torch.randn(15, 3)
    s0_before = s0.clone()
    dense_before = graph.to_dense().clone()
    dense_fp64_equilibrium_reference(graph, s0, alpha=0.9)
    assert torch.equal(s0, s0_before)
    assert torch.equal(graph.to_dense(), dense_before)


# ---------------------------------------------------------------------------
# 11. FP32/FP64 preservation
# ---------------------------------------------------------------------------


def test_finite_step_propagate_preserves_fp32_dtype():
    graph = _random_graph(num_nodes=10, dim=6, k=3)
    s0 = torch.randn(10, 4, dtype=torch.float32)
    trace = finite_step_propagate(graph, s0, alpha=0.9, steps=5, snapshot_steps=(5,))
    assert trace.snapshots[5].dtype == torch.float32


def test_finite_step_propagate_preserves_fp64_dtype():
    graph = _random_graph(num_nodes=10, dim=6, k=3)
    s0 = torch.randn(10, 4, dtype=torch.float64)
    trace = finite_step_propagate(graph, s0, alpha=0.9, steps=5, snapshot_steps=(5,))
    assert trace.snapshots[5].dtype == torch.float64


def test_fp64_promotion_uses_the_same_fp32_constructed_graph_weights():
    # The FP64 finite-step reference promotes the ALREADY fp32-constructed
    # weights, not a freshly-recomputed fp64 row-normalization -- confirmed
    # by checking the promoted weight equals float64(float32_weight) exactly.
    graph = _random_graph(num_nodes=10, dim=6, k=3)
    s0 = torch.randn(10, 4, dtype=torch.float64)
    trace = finite_step_propagate(graph, s0, alpha=0.9, steps=1, snapshot_steps=(1,))
    expected_weights_64 = graph.transition_weights.to(torch.float64)
    p1 = 0.9 * (expected_weights_64.unsqueeze(-1) * s0[graph.neighbor_indices]).sum(dim=1) + 0.1 * s0
    assert torch.allclose(trace.snapshots[1], p1, atol=1e-12)


# ---------------------------------------------------------------------------
# 12-13. Dense solve reference, true residual
# ---------------------------------------------------------------------------


def test_dense_equilibrium_residual_is_near_machine_epsilon():
    graph = _random_graph(num_nodes=15, dim=6, k=6, seed=10)
    s0 = torch.randn(15, 3)
    reference = dense_fp64_equilibrium_reference(graph, s0, alpha=0.9)
    assert reference.residual_relative_frobenius_norm < 1e-10
    assert reference.backward_error < 1e-10


def test_dense_equilibrium_residual_computed_independently_matches():
    graph = _random_graph(num_nodes=12, dim=6, k=5, seed=11)
    s0 = torch.randn(12, 3)
    alpha = 0.8
    reference = dense_fp64_equilibrium_reference(graph, s0, alpha=alpha)
    dense = graph.to_dense().to(torch.float64)
    identity = torch.eye(12, dtype=torch.float64)
    k_matrix = identity - alpha * dense
    b_vector = (1 - alpha) * s0.to(torch.float64)
    residual_independent = b_vector - k_matrix @ reference.p_equilibrium
    assert torch.allclose(residual_independent, reference.residual, atol=1e-14)


def test_dense_equilibrium_uses_solve_not_inverse():
    import inspect

    source = inspect.getsource(dense_fp64_equilibrium_reference)
    assert "linalg.inv" not in source
    assert "linalg.solve" in source


def test_dense_equilibrium_result_matches_long_finite_step_run():
    # Sanity cross-check: a long finite-step run should approach the dense
    # equilibrium (this is NOT asserted as "exact" anywhere in the module).
    graph = _random_graph(num_nodes=12, dim=6, k=5, seed=12)
    s0 = torch.randn(12, 3)
    alpha = 0.9
    reference = dense_fp64_equilibrium_reference(graph, s0, alpha=alpha)
    trace = finite_step_propagate(graph, s0.double(), alpha=alpha, steps=2000, snapshot_steps=(2000,))
    comparison = compare_snapshots(trace.snapshots[2000], reference.p_equilibrium)
    assert comparison.relative_frobenius_error < 1e-3


def test_condition_diagnostics_uses_svd_of_k_not_of_ktk():
    import inspect

    source = inspect.getsource(compute_condition_diagnostics)
    # sigma_min/sigma_max come directly from svdvals(K); K^T @ K only
    # appears afterward, inside the separately-labeled departure-from-
    # normality computation, never as the input to the singular-value call.
    assert "torch.linalg.svdvals(k_matrix)" in source
    svd_call_index = source.index("torch.linalg.svdvals(k_matrix)")
    ktk_index = source.index("k_transpose @ k_matrix")
    assert svd_call_index < ktk_index


def test_condition_diagnostics_sigma_values_are_positive_and_ordered():
    graph = _random_graph(num_nodes=15, dim=6, k=6, seed=13)
    diagnostics = compute_condition_diagnostics(graph, alpha=0.9)
    assert diagnostics.sigma_min > 0
    assert diagnostics.sigma_max >= diagnostics.sigma_min
    assert diagnostics.kappa_2 == pytest.approx(diagnostics.sigma_max / diagnostics.sigma_min)


# ---------------------------------------------------------------------------
# 14. Zero-union / constant-field cases
# ---------------------------------------------------------------------------


def test_constant_s0_field_propagates_to_the_same_constant():
    # If S0 is spatially constant, P(t) must remain that same constant at
    # every step (since A is row-stochastic, A @ constant = constant).
    graph = _random_graph(num_nodes=10, dim=6, k=3)
    constant_value = torch.full((10, 4), 2.5)
    trace = finite_step_propagate(graph, constant_value, alpha=0.95, steps=50, snapshot_steps=(50,))
    assert torch.allclose(trace.snapshots[50], constant_value, atol=1e-5)


def test_all_zero_s0_propagates_to_zero():
    graph = _random_graph(num_nodes=10, dim=6, k=3)
    zeros = torch.zeros(10, 4)
    trace = finite_step_propagate(graph, zeros, alpha=0.9, steps=50, snapshot_steps=(50,))
    assert torch.allclose(trace.snapshots[50], zeros, atol=1e-8)


def test_all_fallback_graph_leaves_s0_unchanged():
    # A graph where every row is a fallback self-loop is the identity
    # matrix; P(t+1) = alpha*P(t) + (1-alpha)*S0 converges toward S0 for
    # any alpha, and stays exactly S0 if it starts there.
    graph12 = _graph_with_forced_fallback_rows(num_nodes=15, fallback_rows=tuple(range(15)))
    s0 = torch.randn(15, 3)
    trace = finite_step_propagate(graph12, s0, alpha=0.9, steps=10, snapshot_steps=(10,))
    assert torch.allclose(trace.snapshots[10], s0, atol=1e-6)


# ---------------------------------------------------------------------------
# 15. Matched-delta diagnostics
# ---------------------------------------------------------------------------


def test_matched_delta_zero_when_k11_equals_k12():
    p = torch.randn(10, 4)
    delta = compute_matched_delta(p, p)
    assert delta.frobenius_norm == 0.0
    assert delta.argmax_disagreement_count == 0


def test_matched_delta_nonzero_and_argmax_disagreement_detected():
    p_k12 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    p_k11 = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    delta = compute_matched_delta(p_k11, p_k12)
    assert delta.frobenius_norm > 0
    assert delta.argmax_disagreement_count == 1
    assert delta.argmax_disagreement_rate == 0.5


def test_delta_stability_error_zero_when_deltas_are_identical():
    d = torch.randn(10, 4)
    assert delta_stability_error(d, d) == 0.0


def test_delta_stability_error_uses_epsilon_floor_for_near_zero_denominator():
    d_a = torch.full((5, 2), 1e-9)
    d_b = torch.zeros(5, 2)
    error = delta_stability_error(d_a, d_b, epsilon=1e-6)
    assert error == pytest.approx(torch.linalg.matrix_norm(d_a.double()).item() / 1e-6, rel=1e-6)


# ---------------------------------------------------------------------------
# 16. Exact types / fail-closed input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_alpha", [1, "0.9", True, float("nan"), 1.0, -0.1])
def test_finite_step_propagate_rejects_bad_alpha(bad_alpha):
    graph = _random_graph(num_nodes=10, dim=6, k=3)
    s0 = torch.randn(10, 4)
    with pytest.raises(FiniteStepRegimeError):
        finite_step_propagate(graph, s0, alpha=bad_alpha, steps=5, snapshot_steps=(5,))


@pytest.mark.parametrize("bad_steps", [0, -1, 1.5, "5", True])
def test_finite_step_propagate_rejects_bad_steps(bad_steps):
    graph = _random_graph(num_nodes=10, dim=6, k=3)
    s0 = torch.randn(10, 4)
    with pytest.raises(FiniteStepRegimeError):
        finite_step_propagate(graph, s0, alpha=0.9, steps=bad_steps, snapshot_steps=(1,))


def test_finite_step_propagate_rejects_non_finite_s0():
    graph = _random_graph(num_nodes=10, dim=6, k=3)
    s0 = torch.randn(10, 4)
    s0[0, 0] = float("nan")
    with pytest.raises(FiniteStepRegimeError, match="finite"):
        finite_step_propagate(graph, s0, alpha=0.9, steps=5, snapshot_steps=(5,))


def test_finite_step_propagate_rejects_wrong_node_count():
    graph = _random_graph(num_nodes=10, dim=6, k=3)
    s0 = torch.randn(9, 4)
    with pytest.raises(FiniteStepRegimeError, match="node count"):
        finite_step_propagate(graph, s0, alpha=0.9, steps=5, snapshot_steps=(5,))


def test_finite_step_propagate_rejects_non_floating_s0():
    graph = _random_graph(num_nodes=10, dim=6, k=3)
    s0 = torch.randint(0, 5, (10, 4))
    with pytest.raises(FiniteStepRegimeError, match="floating point"):
        finite_step_propagate(graph, s0, alpha=0.9, steps=5, snapshot_steps=(5,))


def test_build_matched_k11_rejects_non_graph_input():
    with pytest.raises(FiniteStepRegimeError, match="DirectedTopKGraph"):
        build_matched_k11_from_k12("not a graph")


def test_condition_diagnostics_rejects_bad_alpha():
    graph = _random_graph(num_nodes=10, dim=6, k=3)
    with pytest.raises(FiniteStepRegimeError):
        compute_condition_diagnostics(graph, alpha=1.5)


# ---------------------------------------------------------------------------
# 17. No forbidden solver calls anywhere in this module
# ---------------------------------------------------------------------------


def test_module_source_contains_no_forbidden_solver_calls():
    # AST-based: only flags actual Call nodes referencing a forbidden
    # solver name, so prose in docstrings/comments explaining what this
    # module deliberately does NOT do (e.g. "no CGLS, no GMRES") is never
    # a false positive.
    import ast

    source = (COVER_DR_PACKAGE_PATH / "finite_step_regime.py").read_text()
    tree = ast.parse(source)
    forbidden_call_names = {"solve_rwr_cgls", "solve_rwr", "solve_rwr_fixed_point", "cgls", "gmres"}
    offending_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name and name.lower() in {n.lower() for n in forbidden_call_names}:
                offending_calls.append(name)
    assert offending_calls == [], f"forbidden solver calls found: {offending_calls}"


def test_module_does_not_import_rwr_or_inference_solver_machinery():
    import ast

    tree = ast.parse((COVER_DR_PACKAGE_PATH / "finite_step_regime.py").read_text())
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    assert not any("rwr" in name.lower() for name in imported_modules)


# ---------------------------------------------------------------------------
# Diagnostic-only label sensitivity (never stitched, never mIoU)
# ---------------------------------------------------------------------------


def test_diagnostic_pixel_argmax_map_shape_and_no_stitching_claim():
    patch_scores = torch.randn(16, 5)  # 4x4 grid, 5 classes
    argmax_map = diagnostic_pixel_argmax_map(patch_scores, grid_hw=(4, 4), crop_hw=(32, 32))
    assert tuple(argmax_map.shape) == (32, 32)


def test_compare_label_sensitivity_reports_disagreement_rate():
    map_a = torch.zeros(4, 4, dtype=torch.int64)
    map_b = torch.zeros(4, 4, dtype=torch.int64)
    map_b[0, 0] = 1
    sensitivity = compare_label_sensitivity(map_a, map_b)
    assert sensitivity.argmax_disagreement_count == 1
    assert sensitivity.pixel_count == 16
    assert sensitivity.argmax_disagreement_rate == pytest.approx(1 / 16)


def test_snapshot_comparison_identical_tensors_zero_error():
    p = torch.randn(10, 4)
    comparison = compare_snapshots(p, p)
    assert comparison.max_absolute_error == 0.0
    assert comparison.relative_frobenius_error == 0.0
    assert comparison.argmax_disagreement_count == 0
