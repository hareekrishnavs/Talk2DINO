import importlib.util
import inspect
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).parents[1]
PACKAGE_PATH = (
    ROOT / "src/open_vocabulary_segmentation/models/dinotext/cover_dr"
)


def _load_package():
    spec = importlib.util.spec_from_file_location(
        "cover_dr_rwr_under_test",
        PACKAGE_PATH / "__init__.py",
        submodule_search_locations=[str(PACKAGE_PATH)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cover_dr = _load_package()
rwr_module = sys.modules[f"{cover_dr.__name__}.rwr"]
DirectedTopKGraph = cover_dr.DirectedTopKGraph
RWRInputError = cover_dr.RWRInputError
RWRNonConvergenceError = cover_dr.RWRNonConvergenceError
RWRNumericalBreakdownError = cover_dr.RWRNumericalBreakdownError
SparseRWROperator = cover_dr.SparseRWROperator
build_directed_topk_graph = cover_dr.build_directed_topk_graph
solve_rwr = cover_dr.solve_rwr
solve_rwr_cgls = cover_dr.solve_rwr_cgls
solve_rwr_fixed_point = cover_dr.solve_rwr_fixed_point


def _asymmetric_graph(device="cpu"):
    return DirectedTopKGraph(
        neighbor_indices=torch.tensor(
            [[1, 2], [2, 0], [0, 3], [3, 0]],
            dtype=torch.int64,
            device=device,
        ),
        transition_weights=torch.tensor(
            [[0.8, 0.2], [0.75, 0.25], [0.6, 0.4], [1.0, 0.0]],
            dtype=torch.float32,
            device=device,
        ),
        edge_affinities=torch.tensor(
            [[0.8, 0.2], [0.75, 0.25], [0.6, 0.4], [0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        ),
        self_loop_fallback=torch.tensor(
            [False, False, False, True], dtype=torch.bool, device=device
        ),
        num_nodes=4,
        k=2,
        affinity_power=3.0,
    )


def _two_component_graph():
    return DirectedTopKGraph(
        neighbor_indices=torch.tensor([[1], [0], [3], [4], [2]]),
        transition_weights=torch.ones(5, 1),
        edge_affinities=torch.ones(5, 1),
        self_loop_fallback=torch.zeros(5, dtype=torch.bool),
        num_nodes=5,
        k=1,
        affinity_power=3.0,
    )


def _identity_fallback_graph(num_nodes, k, device="cpu"):
    indices = torch.empty(num_nodes, k, dtype=torch.int64, device=device)
    rows = torch.arange(num_nodes, device=device)
    indices[:, 0] = rows
    for slot in range(1, k):
        indices[:, slot] = (rows + slot) % num_nodes
    weights = torch.zeros(num_nodes, k, dtype=torch.float32, device=device)
    weights[:, 0] = 1
    return DirectedTopKGraph(
        neighbor_indices=indices,
        transition_weights=weights,
        edge_affinities=torch.zeros_like(weights),
        self_loop_fallback=torch.ones(num_nodes, dtype=torch.bool, device=device),
        num_nodes=num_nodes,
        k=k,
        affinity_power=3.0,
    )


def _independent_dense(graph, dtype=torch.float64):
    dense = torch.zeros(
        graph.num_nodes,
        graph.num_nodes,
        dtype=dtype,
        device=graph.transition_weights.device,
    )
    for source in range(graph.num_nodes):
        for slot in range(graph.k):
            destination = int(graph.neighbor_indices[source, slot])
            dense[source, destination] += graph.transition_weights[
                source, slot
            ].to(dtype)
    return dense


def _dense_solution(graph, scores, alpha):
    dtype = scores.dtype
    adjacency = _independent_dense(graph, dtype=dtype)
    system = torch.eye(graph.num_nodes, dtype=dtype, device=scores.device)
    system = system - alpha * adjacency
    return torch.linalg.solve(system, (1 - alpha) * scores)


def _historical_cgls(graph, scores, alpha, *, tolerance=1e-5, max_iter=5000):
    """Test-only recurrence copied from fd5d615 implicit_solve.py."""
    operator = SparseRWROperator(graph, alpha)
    right_hand_side = (1 - alpha) * scores
    epsilon = torch.finfo(scores.dtype).tiny
    solution = right_hand_side.clone()
    residual = right_hand_side - operator.matmul(solution)
    normal = operator.transpose_matmul(residual)
    direction = normal.clone()
    gamma = (normal * normal).sum(dim=0)
    rhs_norm = right_hand_side.norm(dim=0).clamp_min(epsilon)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        forward = operator.matmul(direction)
        denominator = (forward * forward).sum(dim=0).clamp_min(epsilon)
        step = gamma / denominator
        solution = solution + step[None, :] * direction
        residual = residual - step[None, :] * forward
        if torch.all(residual.norm(dim=0) / rhs_norm < tolerance):
            converged = True
            break
        next_normal = operator.transpose_matmul(residual)
        next_gamma = (next_normal * next_normal).sum(dim=0)
        direction = next_normal + direction * (next_gamma / gamma.clamp_min(epsilon))[None, :]
        gamma = next_gamma
    return solution, iterations, converged


def _scores(dtype=torch.float64, device="cpu"):
    return torch.tensor(
        [[0.2, -0.7, 1.1], [1.3, 0.1, -0.2], [-0.4, 0.8, 0.6], [0.9, -0.3, 0.4]],
        dtype=dtype,
        device=device,
    )


def _difficult_directed_fixture(dtype=torch.float32, device="cpu"):
    """Pinned finite-precision fixture that defeated unrestarted CGLS.

    The earlier four-node graph and 1024-node identity-fallback smoke do not
    exercise finite-precision loss of Krylov conjugacy. This valid asymmetric
    graph has ``cond(I - 0.98*A) ~= 130`` and no fallback rows. Its first RHS
    stagnated or overflowed under the former FP32 recurrence.
    """
    indices = torch.tensor(
        [
            [8, 6, 4],
            [4, 7, 6],
            [10, 0, 1],
            [8, 5, 0],
            [0, 6, 10],
            [9, 3, 8],
            [0, 4, 8],
            [8, 1, 3],
            [0, 3, 6],
            [5, 6, 8],
            [4, 2, 0],
        ],
        dtype=torch.int64,
        device=device,
    )
    weights = torch.tensor(
        [
            [0.3975238800048828, 0.3766304552555084, 0.225845605134964],
            [0.38048094511032104, 0.3519677221775055, 0.2675513029098511],
            [1.0, 0.0, 0.0],
            [0.5562735795974731, 0.2585965394973755, 0.18512992560863495],
            [0.42243754863739014, 0.2968692183494568, 0.2806931734085083],
            [0.6323502659797668, 0.2486851066350937, 0.11896459758281708],
            [0.5758843421936035, 0.24268008768558502, 0.18143554031848907],
            [0.7478132247924805, 0.14306816458702087, 0.10911871492862701],
            [0.5185107588768005, 0.3267156183719635, 0.15477365255355835],
            [0.8706234693527222, 0.09371259063482285, 0.03566388040781021],
            [0.8167767524719238, 0.18285968899726868, 0.00036350375739857554],
        ],
        dtype=torch.float32,
        device=device,
    )
    graph = DirectedTopKGraph(
        neighbor_indices=indices,
        transition_weights=weights,
        edge_affinities=weights,
        self_loop_fallback=torch.zeros(11, dtype=torch.bool, device=device),
        num_nodes=11,
        k=3,
        affinity_power=3.0,
    )
    scores = torch.tensor(
        [
            [0.14281052350997925, -0.5702493786811829, -1.2664172649383545, 0.683472752571106, 1.8808677196502686, 0.8643179535865784, 2.3457889556884766],
            [-0.198085755109787, 1.226643681526184, -0.163796529173851, 0.008618825115263462, 1.9698442220687866, -0.7432278394699097, -1.439994215965271],
            [-0.44994568824768066, -0.8670238852500916, 0.8926357626914978, 0.1777360886335373, 0.28586819767951965, -2.3443167209625244, -0.08263545483350754],
            [0.46694380044937134, 0.31593888998031616, -0.6927435398101807, -1.057605266571045, -1.3688819408416748, 0.9688977003097534, -0.48614028096199036],
            [-1.3146231174468994, 0.2009158432483673, 0.10126891732215881, -1.5017646551132202, -0.4377370774745941, -1.0598247051239014, 0.16515405476093292],
            [0.3926621377468109, -0.7501369714736938, -1.0037225484848022, -0.1210615262389183, -0.3894941210746765, -1.056300163269043, -1.3400498628616333],
            [-1.9004313945770264, -0.9864697456359863, 0.14951403439044952, 1.3155663013458252, 0.6826961636543274, 0.19818814098834991, 0.45870107412338257],
            [-2.0181777477264404, -0.6632566452026367, 1.70564866065979, 0.49005335569381714, -1.0976284742355347, -0.6243201494216919, -0.015243764035403728],
            [-0.5167803168296814, 0.5059662461280823, 1.1159874200820923, 0.30032721161842346, -0.3798520267009735, 0.4682076871395111, 0.8531948924064636],
            [-0.0650317594408989, 0.1610269695520401, -0.0715838372707367, 1.4806512594223022, -0.5126020312309265, -0.5465044975280762, 1.0864946842193604],
            [-1.1003519296646118, 0.3492905795574188, 0.3647814989089966, 1.0125114917755127, 1.3895926475524902, -0.4687526524066925, -0.45624402165412903],
        ],
        dtype=dtype,
        device=device,
    )
    return graph, scores


def test_defaults_are_canonical():
    for function in (solve_rwr, solve_rwr_cgls, solve_rwr_fixed_point):
        assert inspect.signature(function).parameters["alpha"].default == 0.98
        assert inspect.signature(function).parameters["rtol"].default is None
        assert inspect.signature(function).parameters["atol"].default is None
    rwr_module = sys.modules[f"{cover_dr.__name__}.rwr"]
    assert rwr_module._default_tolerances(torch.float32) == (1e-5, 1e-7)
    assert rwr_module._default_tolerances(torch.float64) == (1e-10, 1e-12)


@pytest.mark.parametrize("method", ["cgls", "fixed_point"])
@pytest.mark.parametrize("vector", [False, True])
def test_alpha_zero_is_exact_owned_identity(method, vector):
    source = _scores(torch.float64)
    if vector:
        source = source[:, 0]
    result = solve_rwr(_asymmetric_graph(), source, alpha=0, method=method)
    assert torch.equal(result.scores, source)
    assert result.iterations == 0
    assert result.absolute_residual_inf == 0
    assert result.fixed_point_delta_inf == 0
    assert result.scores.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


def test_directed_operator_is_nonsymmetric_and_not_repaired():
    graph = _asymmetric_graph()
    adjacency = _independent_dense(graph)
    system = torch.eye(4, dtype=torch.float64) - 0.8 * adjacency
    assert not torch.equal(adjacency, adjacency.T)
    assert not torch.equal(system, system.T)
    operator = SparseRWROperator(graph, 0.8)
    probe = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    torch.testing.assert_close(operator.matmul(probe), system @ probe)
    assert not torch.allclose(operator.matmul(probe), system.T @ probe)


@pytest.mark.parametrize("matrix", [False, True])
def test_sparse_graph_transpose_matmul_matches_independent_dense(matrix):
    graph = _asymmetric_graph()
    value = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    if not matrix:
        value = value[:, 0]
    expected = _independent_dense(graph).T @ value
    torch.testing.assert_close(graph.transpose_matmul(value), expected)


def test_sparse_operator_transpose_matches_dense_reference():
    graph = _asymmetric_graph()
    operator = SparseRWROperator(graph, 0.73)
    value = _scores(torch.float64)
    system = torch.eye(4, dtype=torch.float64) - 0.73 * _independent_dense(graph)
    torch.testing.assert_close(operator.transpose_matmul(value), system.T @ value)


def test_operator_validates_shape_nodes_and_dtype_and_promotes_half():
    operator = SparseRWROperator(_asymmetric_graph(), 0.8)
    with pytest.raises(RWRInputError, match=r"\[N\].*\[N, R\]"):
        operator.matmul(torch.randn(4, 2, 1))
    with pytest.raises(RWRInputError, match="node mismatch"):
        operator.transpose_matmul(torch.randn(3, 2))
    with pytest.raises(RWRInputError, match="floating point"):
        operator.matmul(torch.ones(4, dtype=torch.int64))
    assert operator.matmul(torch.randn(4, 2).half()).dtype == torch.float32


def test_adjoint_inner_product_identity():
    graph = _asymmetric_graph()
    left = _scores(torch.float64)
    right = torch.flip(left, dims=(0,))
    lhs = (graph.matmul(left) * right).sum()
    rhs = (left * graph.transpose_matmul(right)).sum()
    torch.testing.assert_close(lhs, rhs, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("solver", [solve_rwr_fixed_point, solve_rwr_cgls])
def test_solvers_match_independent_dense_float64(solver):
    graph = _asymmetric_graph()
    scores = _scores(torch.float64)
    result = solver(graph, scores, alpha=0.8, rtol=1e-11, atol=1e-13)
    expected = _dense_solution(graph, scores, 0.8)
    torch.testing.assert_close(result.scores, expected, rtol=2e-10, atol=2e-11)
    assert result.converged
    assert result.maximum_scaled_residual <= 1


def test_cgls_matches_fixed_point_at_canonical_alpha():
    graph = _asymmetric_graph()
    scores = _scores(torch.float64)
    fixed = solve_rwr_fixed_point(
        graph, scores, alpha=0.98, rtol=2e-10, atol=1e-12
    )
    cgls = solve_rwr_cgls(
        graph, scores, alpha=0.98, rtol=2e-10, atol=1e-12
    )
    torch.testing.assert_close(cgls.scores, fixed.scores, rtol=3e-8, atol=3e-9)


@pytest.mark.parametrize("solver", [solve_rwr_fixed_point, solve_rwr_cgls])
def test_returned_scores_satisfy_fixed_point_identity(solver):
    graph = _asymmetric_graph()
    scores = _scores(torch.float64)
    result = solver(graph, scores, alpha=0.75, rtol=1e-11, atol=1e-13)
    update = 0.25 * scores + 0.75 * graph.matmul(result.scores)
    torch.testing.assert_close(result.scores, update, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("solver", [solve_rwr_fixed_point, solve_rwr_cgls])
def test_single_rhs_shape_and_dense_equivalence(solver):
    graph = _asymmetric_graph()
    scores = _scores(torch.float64)[:, 0]
    result = solver(graph, scores, alpha=0.7, rtol=1e-11, atol=1e-13)
    assert result.scores.shape == (4,)
    torch.testing.assert_close(
        result.scores,
        _dense_solution(graph, scores, 0.7),
        rtol=2e-10,
        atol=2e-11,
    )


def test_per_rhs_convergence_and_zero_rhs_are_independent():
    graph = _asymmetric_graph()
    scores = _scores(torch.float64)
    scores[:, 1] = 0
    result = solve_rwr_cgls(
        graph, scores, alpha=0.9, rtol=1e-11, atol=1e-13
    )
    assert torch.count_nonzero(result.scores[:, 1]) == 0
    expected = _dense_solution(graph, scores, 0.9)
    torch.testing.assert_close(result.scores, expected, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("solver", [solve_rwr_fixed_point, solve_rwr_cgls])
def test_all_zero_unary_converges_immediately(solver):
    result = solver(_asymmetric_graph(), torch.zeros(4, 3), alpha=0.98)
    assert torch.count_nonzero(result.scores) == 0
    assert result.iterations == 0
    assert result.absolute_residual_inf == 0


@pytest.mark.parametrize("solver", [solve_rwr_fixed_point, solve_rwr_cgls])
def test_constant_field_is_preserved(solver):
    constant = torch.tensor([0.4, -0.2, 1.7], dtype=torch.float64)
    scores = constant.expand(4, -1).clone()
    result = solver(_asymmetric_graph(), scores, alpha=0.98)
    # Stored graph weights are float32, so their row sums carry float32 error
    # even when solve arithmetic is float64.
    torch.testing.assert_close(result.scores, scores, rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("solver", [solve_rwr_fixed_point, solve_rwr_cgls])
def test_fallback_self_loop_row_preserves_its_unary(solver):
    scores = _scores(torch.float64)
    result = solver(_asymmetric_graph(), scores, alpha=0.9)
    torch.testing.assert_close(result.scores[3], scores[3], rtol=2e-9, atol=2e-9)


def test_disconnected_directed_components_match_dense_reference():
    graph = _two_component_graph()
    scores = torch.randn(5, 4, dtype=torch.float64)
    result = solve_rwr_cgls(graph, scores, alpha=0.92, rtol=1e-11, atol=1e-13)
    torch.testing.assert_close(
        result.scores,
        _dense_solution(graph, scores, 0.92),
        rtol=2e-9,
        atol=2e-10,
    )


@pytest.mark.parametrize("solver", [solve_rwr_fixed_point, solve_rwr_cgls])
def test_convex_range_property(solver):
    graph = _asymmetric_graph()
    scores = _scores(torch.float64)
    result = solver(graph, scores, alpha=0.85, rtol=1e-11, atol=1e-13)
    assert torch.all(result.scores >= scores.amin(dim=0) - 2e-10)
    assert torch.all(result.scores <= scores.amax(dim=0) + 2e-10)


def test_neumann_series_matches_cgls():
    graph = _asymmetric_graph()
    scores = _scores(torch.float64)
    alpha = 0.7
    adjacency = _independent_dense(graph)
    term = scores.clone()
    series = torch.zeros_like(scores)
    for _ in range(200):
        series += (1 - alpha) * term
        term = alpha * (adjacency @ term)
    result = solve_rwr_cgls(graph, scores, alpha=alpha, rtol=1e-11, atol=1e-13)
    torch.testing.assert_close(result.scores, series, rtol=2e-10, atol=2e-11)


def test_true_residual_diagnostics_match_independent_recomputation():
    graph = _asymmetric_graph()
    scores = _scores(torch.float64)
    alpha = 0.94
    result = solve_rwr_cgls(graph, scores, alpha=alpha)
    system = torch.eye(4, dtype=torch.float64) - alpha * _independent_dense(graph)
    residual = system @ result.scores - (1 - alpha) * scores
    assert result.absolute_residual_inf == pytest.approx(
        residual.abs().max().item(), rel=0, abs=2e-15
    )


@pytest.mark.parametrize("solver", [solve_rwr_fixed_point, solve_rwr_cgls])
def test_nonconvergence_raises_instead_of_returning_scores(solver):
    with pytest.raises(RWRNonConvergenceError):
        solver(
            _asymmetric_graph(),
            _scores(torch.float64),
            alpha=0.98,
            rtol=0,
            atol=0,
            max_iter=1,
        )


_FAILURE_FIELDS = (
    "method",
    "iteration",
    "abs_primal_residual_inf",
    "max_scaled_primal_residual",
    "rtol",
    "atol",
    "reason",
)


def _assert_failure_contract(error, *, method, iteration, reason, rtol, atol):
    message = str(error)
    for field in _FAILURE_FIELDS:
        assert f"{field}=" in message
        assert hasattr(error, field)
    assert error.method == method
    assert error.iteration == iteration
    assert error.reason == reason
    assert error.rtol == rtol
    assert error.atol == atol


@pytest.mark.parametrize(
    "solver,method",
    [
        (solve_rwr_fixed_point, "fixed_point"),
        (solve_rwr_cgls, "cgls"),
    ],
)
def test_exhaustion_failure_has_true_residual_diagnostics(solver, method):
    graph = _asymmetric_graph()
    scores = _scores(torch.float64)
    rtol = 0.0
    atol = 0.0
    with pytest.raises(RWRNonConvergenceError) as raised:
        solver(
            graph,
            scores,
            alpha=0.98,
            rtol=rtol,
            atol=atol,
            max_iter=1,
        )
    error = raised.value
    _assert_failure_contract(
        error,
        method=method,
        iteration=1,
        reason="max_iterations_exhausted",
        rtol=rtol,
        atol=atol,
    )
    adjacency = _independent_dense(graph)
    system = torch.eye(4, dtype=torch.float64) - 0.98 * adjacency
    expected_residual = 0.02 * scores - system @ error.current_scores
    torch.testing.assert_close(error.primal_residual, expected_residual)
    assert error.abs_primal_residual_inf == pytest.approx(
        expected_residual.abs().max().item(), rel=0, abs=2e-15
    )
    assert math.isinf(error.max_scaled_primal_residual)
    assert error.failing_rhs


def test_cgls_failure_residual_uses_only_active_rhs_columns():
    graph = _asymmetric_graph()
    scores = _scores(torch.float64)[:, :2]
    scores[:, 0] = 0
    with pytest.raises(RWRNonConvergenceError) as raised:
        solve_rwr_cgls(
            graph,
            scores,
            alpha=0.98,
            rtol=0,
            atol=0,
            max_iter=1,
        )
    error = raised.value
    assert error.active_rhs == (1,)
    assert error.failing_rhs == (1,)
    adjacency = _independent_dense(graph)
    system = torch.eye(4, dtype=torch.float64) - 0.98 * adjacency
    residual = 0.02 * scores - system @ error.current_scores
    assert error.abs_primal_residual_inf == pytest.approx(
        residual[:, 1].abs().max().item(), rel=0, abs=2e-15
    )


def test_cgls_initial_breakdown_has_iteration_zero_and_failing_rhs():
    scores = torch.tensor(
        [[1e-30, 1e-30], [1e-30, 2e-30], [1e-30, 3e-30], [1e-30, 4e-30]],
        dtype=torch.float32,
    )
    with pytest.raises(RWRNumericalBreakdownError) as raised:
        solve_rwr_cgls(
            _asymmetric_graph(), scores, alpha=0.5, rtol=0, atol=0
        )
    error = raised.value
    _assert_failure_contract(
        error,
        method="cgls",
        iteration=0,
        reason="initial_normal_residual_vanished",
        rtol=0,
        atol=0,
    )
    assert error.failing_rhs
    assert "failing_rhs=" in str(error)
    assert error.stage == "initialization"


def test_cgls_one_time_in_loop_breakdown_restarts_and_converges(monkeypatch):
    original = SparseRWROperator._transpose_matmul_prepared
    calls = 0

    def vanish_after_first_update(self, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            return torch.zeros_like(value)
        return original(self, value)

    monkeypatch.setattr(
        SparseRWROperator,
        "_transpose_matmul_prepared",
        vanish_after_first_update,
    )
    result = solve_rwr_cgls(
        _asymmetric_graph(),
        _scores(torch.float64),
        alpha=0.8,
    )
    assert result.converged
    assert result.total_restart_count >= 1
    assert result.work_count > result.iterations
    assert result.work_count <= result.iterations + result.total_restart_count
    assert dict(result.restart_reason_counts)[
        "updated_normal_residual_vanished"
    ] >= 1


@pytest.mark.parametrize(
    "solver,method",
    [
        (solve_rwr_fixed_point, "fixed_point"),
        (solve_rwr_cgls, "cgls"),
    ],
)
def test_nonfinite_input_has_complete_iteration_zero_diagnostic(solver, method):
    scores = _scores(torch.float32)
    scores[0, 0] = float("nan")
    with pytest.raises(cover_dr.RWRNonFiniteError) as raised:
        solver(
            _asymmetric_graph(),
            scores,
            rtol=3e-4,
            atol=2e-5,
        )
    error = raised.value
    _assert_failure_contract(
        error,
        method=method,
        iteration=0,
        reason="non_finite_input",
        rtol=3e-4,
        atol=2e-5,
    )
    assert math.isinf(error.abs_primal_residual_inf)
    assert math.isinf(error.max_scaled_primal_residual)
    assert error.tensor == "unary_scores"


def test_one_time_nonfinite_iterative_arithmetic_restarts_cleanly(monkeypatch):
    original = SparseRWROperator._matmul_prepared
    calls = 0

    def nonfinite_search_operator(self, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            return torch.full_like(value, torch.inf)
        return original(self, value)

    monkeypatch.setattr(
        SparseRWROperator,
        "_matmul_prepared",
        nonfinite_search_operator,
    )
    result = solve_rwr_cgls(
        _asymmetric_graph(),
        _scores(torch.float64),
        alpha=0.8,
        rtol=1e-8,
        atol=1e-10,
    )
    assert result.converged
    assert dict(result.restart_reason_counts)[
        "non_finite_search_direction_operator_output"
    ] == result.scores.shape[1]


def test_max_iter_counts_completed_updates_not_restart_work(monkeypatch):
    original = SparseRWROperator._matmul_prepared
    calls = 0

    def one_time_nonfinite_forward(self, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            return torch.full_like(value, torch.inf)
        return original(self, value)

    monkeypatch.setattr(
        SparseRWROperator,
        "_matmul_prepared",
        one_time_nonfinite_forward,
    )
    with pytest.raises(RWRNonConvergenceError) as raised:
        solve_rwr_cgls(
            _asymmetric_graph(),
            _scores(torch.float64),
            alpha=0.8,
            rtol=0,
            atol=0,
            max_iter=1,
        )

    error = raised.value
    assert error.iteration == 1
    assert error.work_count == 2
    assert error.total_restart_count == _scores().shape[1]
    assert "completed_iteration_limit=1" in error.detail


def test_repeated_recoverable_breakdown_raises_complete_diagnostic(monkeypatch):
    original = SparseRWROperator._matmul_prepared
    calls = 0

    def fail_each_search_direction(self, value):
        nonlocal calls
        calls += 1
        if calls % 2 == 0:
            return torch.full_like(value, torch.inf)
        return original(self, value)

    monkeypatch.setattr(
        SparseRWROperator,
        "_matmul_prepared",
        fail_each_search_direction,
    )
    with pytest.raises(cover_dr.RWRNonFiniteError) as raised:
        solve_rwr_cgls(
            _asymmetric_graph(),
            _scores(torch.float64),
            alpha=0.8,
        )
    error = raised.value
    _assert_failure_contract(
        error,
        method="cgls",
        iteration=0,
        reason="repeated_recovery_failure",
        rtol=1e-10,
        atol=1e-12,
    )
    assert error.total_restart_count == 3 * _scores().shape[1]
    assert error.max_restarts_per_rhs == 3
    assert error.latest_restart_reason == (
        "non_finite_search_direction_operator_output"
    )
    assert error.failing_rhs == (0, 1, 2)


def test_numerical_breakdown_is_detected():
    tiny_scores = torch.full((4, 2), 1e-30, dtype=torch.float32)
    tiny_scores[:, 1] *= torch.tensor([1, 2, 3, 4])
    with pytest.raises(RWRNumericalBreakdownError):
        solve_rwr_cgls(
            _asymmetric_graph(),
            tiny_scores,
            alpha=0.5,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_unary_is_rejected(value):
    scores = _scores(torch.float32)
    scores[0, 0] = value
    with pytest.raises(RWRNumericalBreakdownError, match="non-finite"):
        solve_rwr_cgls(_asymmetric_graph(), scores)


@pytest.mark.parametrize(
    "alpha", [-0.1, 1.0, 1.1, float("nan"), float("inf"), True]
)
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(RWRInputError, match="alpha"):
        solve_rwr_cgls(_asymmetric_graph(), _scores(), alpha=alpha)


@pytest.mark.parametrize(
    "name,value", [
        ("rtol", -1),
        ("rtol", float("nan")),
        ("rtol", float("inf")),
        ("atol", -1),
        ("atol", float("nan")),
        ("atol", float("inf")),
    ]
)
def test_invalid_tolerance_is_rejected(name, value):
    with pytest.raises(RWRInputError, match=name):
        solve_rwr_cgls(_asymmetric_graph(), _scores(), **{name: value})


@pytest.mark.parametrize("max_iter", [0, -1, 1.5, True])
def test_invalid_max_iter_is_rejected(max_iter):
    with pytest.raises(RWRInputError, match="max_iter"):
        solve_rwr_cgls(_asymmetric_graph(), _scores(), max_iter=max_iter)


def test_node_mismatch_and_empty_scores_are_rejected():
    graph = _asymmetric_graph()
    with pytest.raises(RWRInputError, match="node mismatch"):
        solve_rwr_cgls(graph, torch.randn(3, 2))
    with pytest.raises(RWRInputError, match="nonempty"):
        solve_rwr_cgls(graph, torch.empty(4, 0))


def test_unary_rank_and_nonfloating_dtype_are_rejected():
    graph = _asymmetric_graph()
    with pytest.raises(RWRInputError, match=r"\[N\].*\[N, C\]"):
        solve_rwr_cgls(graph, torch.randn(4, 2, 1))
    with pytest.raises(RWRInputError, match="floating point"):
        solve_rwr_cgls(graph, torch.ones(4, 2, dtype=torch.int64))


@pytest.mark.parametrize(
    "input_dtype,output_dtype",
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_dtype_policy(input_dtype, output_dtype):
    result = solve_rwr_cgls(
        _asymmetric_graph(), _scores(torch.float32).to(input_dtype), alpha=0.8
    )
    assert result.scores.dtype == output_dtype
    assert result.scores.device.type == "cpu"


def test_inputs_and_graph_are_not_mutated():
    graph = _asymmetric_graph()
    before_graph = tuple(value.clone() for value in graph._tensor_fields())
    scores = _scores(torch.float64)
    before_scores = scores.clone()
    solve_rwr_cgls(graph, scores, alpha=0.8)
    assert torch.equal(scores, before_scores)
    for before, after in zip(before_graph, graph._tensor_fields()):
        assert torch.equal(before, after)


def test_output_is_owned_contiguous_and_detached():
    scores = _scores(torch.float64).requires_grad_(True)
    result = solve_rwr_cgls(_asymmetric_graph(), scores, alpha=0.8)
    assert result.scores.is_contiguous()
    assert not result.scores.requires_grad
    assert result.scores.grad_fn is None
    assert result.scores.untyped_storage().data_ptr() != scores.untyped_storage().data_ptr()


def test_repeated_solves_are_deterministic():
    graph = _asymmetric_graph()
    scores = _scores(torch.float64)
    first = solve_rwr_cgls(graph, scores, alpha=0.91)
    second = solve_rwr_cgls(graph, scores, alpha=0.91)
    assert torch.equal(first.scores, second.scores)
    assert first.method == second.method
    assert first.iterations == second.iterations
    assert first.absolute_residual_inf == second.absolute_residual_inf
    assert first.maximum_scaled_residual == second.maximum_scaled_residual
    assert first.fixed_point_delta_inf == second.fixed_point_delta_inf
    assert first.work_count == second.work_count


def test_larger_safety_limit_does_not_change_converged_solution():
    graph = _asymmetric_graph()
    scores = _scores(torch.float64)
    first = solve_rwr_cgls(graph, scores, alpha=0.9, max_iter=20)
    second = solve_rwr_cgls(graph, scores, alpha=0.9, max_iter=200)
    assert torch.equal(first.scores, second.scores)
    assert first.iterations == second.iterations


def test_fixed_point_does_not_accept_small_iterate_delta_without_residual():
    scores = _scores(torch.float64) * 1e-12
    with pytest.raises(RWRNonConvergenceError):
        solve_rwr_fixed_point(
            _asymmetric_graph(),
            scores,
            alpha=0.98,
            rtol=0,
            atol=0,
            max_iter=1,
        )


def test_plain_cg_assumptions_fail_but_verified_solvers_match_dense():
    graph = _asymmetric_graph()
    adjacency = _independent_dense(graph)
    system = torch.eye(4, dtype=torch.float64) - 0.98 * adjacency
    assert not torch.allclose(system, system.T)
    scores = _scores(torch.float64)
    right_hand_side = 0.02 * scores
    plain_cg_solution = scores.clone()
    residual = right_hand_side - system @ plain_cg_solution
    direction = residual.clone()
    residual_square = (residual * residual).sum(dim=0)
    for _ in range(100):
        forward = system @ direction
        step = residual_square / (direction * forward).sum(dim=0)
        plain_cg_solution += direction * step
        residual -= forward * step
        next_square = (residual * residual).sum(dim=0)
        direction = residual + direction * (next_square / residual_square)
        residual_square = next_square
    assert (system @ plain_cg_solution - right_hand_side).abs().max() > 0.1
    expected = _dense_solution(graph, scores, 0.98)
    for solver in (solve_rwr_cgls, solve_rwr_fixed_point):
        result = solver(graph, scores, alpha=0.98)
        torch.testing.assert_close(result.scores, expected, rtol=3e-8, atol=3e-9)


def test_solver_does_not_materialize_dense_operators(monkeypatch):
    monkeypatch.setattr(
        DirectedTopKGraph,
        "to_dense",
        lambda self: (_ for _ in ()).throw(AssertionError("dense graph used")),
    )
    monkeypatch.setattr(
        torch.linalg,
        "solve",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense solve used")
        ),
    )
    result = solve_rwr_cgls(_asymmetric_graph(), _scores(), alpha=0.8)
    assert result.converged
    assert all(
        not (isinstance(value, torch.Tensor) and value.shape == (4, 4))
        for value in vars(result).values()
    )


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16])
def test_reliable_cgls_converges_on_pinned_difficult_fixture(input_dtype):
    graph, scores = _difficult_directed_fixture(input_dtype)
    promoted_scores = scores.to(torch.float32)
    expected = _dense_solution(graph, promoted_scores.to(torch.float64), 0.98)

    result = solve_rwr_cgls(graph, scores, alpha=0.98)

    assert result.scores.dtype == torch.float32
    assert result.method == "cgls"
    assert result.maximum_scaled_residual <= 1
    # The repaired recurrence preserves conjugacy and reaches this compact
    # fixture before a reliable replacement is needed.
    assert result.total_restart_count == 0
    assert result.max_restarts_per_rhs == 0
    assert result.total_residual_replacement_count == result.total_restart_count
    assert result.restarts_per_rhs[0] == 0
    assert result.restarts_per_rhs[1] == 0
    assert result.iterations > result.max_restarts_per_rhs
    assert result.iterations <= 5000
    assert result.work_count >= result.iterations
    assert result.work_count <= result.iterations + result.total_restart_count
    assert result.restart_reason_counts == ()
    torch.testing.assert_close(
        result.scores.to(torch.float64), expected, rtol=3e-5, atol=3e-5
    )

    operator = SparseRWROperator(graph, 0.98)
    true_residual = (1 - 0.98) * promoted_scores - operator.matmul(result.scores)
    residual_norm = torch.linalg.vector_norm(true_residual, dim=0)
    rhs_norm = torch.linalg.vector_norm((1 - 0.98) * promoted_scores, dim=0)
    threshold = 1e-7 * math.sqrt(graph.num_nodes) + 1e-5 * rhs_norm
    assert torch.all(residual_norm <= threshold)


def test_pinned_fixture_exposes_old_unrestarted_fp32_recurrence():
    graph, scores = _difficult_directed_fixture(torch.float32)
    operator = SparseRWROperator(graph, 0.98)
    right_hand_side = (1 - 0.98) * scores
    solution = scores.clone()
    residual = right_hand_side - operator.matmul(solution)
    direction = operator.transpose_matmul(residual)
    gamma = (direction * direction).sum(dim=0)

    # Test-only reproduction of the removed recurrence. It deliberately has
    # no reliable replacement, restart, compensated update, or best iterate.
    for _ in range(2000):
        forward = operator.matmul(direction)
        denominator = (forward * forward).sum(dim=0)
        if not bool(torch.isfinite(denominator).all()):
            break
        step = gamma / denominator
        solution = solution + direction * step[None, :]
        residual = right_hand_side - operator.matmul(solution)
        normal_residual = operator.transpose_matmul(residual)
        next_gamma = (normal_residual * normal_residual).sum(dim=0)
        if not bool(torch.isfinite(next_gamma).all()):
            break
        direction = normal_residual + direction * (next_gamma / gamma)[None, :]
        gamma = next_gamma

    residual_norm = torch.linalg.vector_norm(
        right_hand_side - operator.matmul(solution), dim=0
    )
    rhs_norm = torch.linalg.vector_norm(right_hand_side, dim=0)
    threshold = 1e-7 * math.sqrt(graph.num_nodes) + 1e-5 * rhs_norm
    assert bool(torch.any(~torch.isfinite(residual_norm) | (residual_norm > threshold)))


def test_repaired_recurrence_and_historical_cgls_pass_true_residual():
    graph, scores = _difficult_directed_fixture(torch.float32)
    repaired = solve_rwr_cgls(graph, scores, alpha=0.98)
    historical, historical_iterations, historical_converged = _historical_cgls(
        graph, scores, 0.98
    )
    expected = _dense_solution(graph, scores.to(torch.float64), 0.98)

    assert historical_converged
    assert historical_iterations <= 5000
    for value in (repaired.scores, historical):
        residual = 0.02 * scores - SparseRWROperator(graph, 0.98).matmul(value)
        residual_norm = torch.linalg.vector_norm(residual, dim=0)
        rhs_norm = torch.linalg.vector_norm(0.02 * scores, dim=0)
        threshold = 1e-7 * math.sqrt(graph.num_nodes) + 1e-5 * rhs_norm
        assert torch.all(residual_norm <= threshold)
        torch.testing.assert_close(
            value.to(torch.float64), expected, rtol=3e-5, atol=3e-5
        )


def test_difficult_fixture_fp64_and_fixed_point_remain_accurate():
    graph, scores = _difficult_directed_fixture(torch.float64)
    expected = _dense_solution(graph, scores, 0.98)
    system = torch.eye(11, dtype=torch.float64) - 0.98 * _independent_dense(graph)
    assert torch.linalg.cond(system) == pytest.approx(130.19, rel=2e-3)
    assert not torch.allclose(system, system.T)
    assert not bool(torch.any(graph.self_loop_fallback))
    assert torch.all((graph.transition_weights > 0).sum(dim=1) >= 1)
    assert torch.count_nonzero((graph.transition_weights > 0).sum(dim=1) > 1) == 10

    cgls = solve_rwr_cgls(graph, scores, alpha=0.98)
    fixed = solve_rwr_fixed_point(graph, scores, alpha=0.98)

    assert cgls.total_restart_count == 0
    torch.testing.assert_close(cgls.scores, expected, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(fixed.scores, expected, rtol=2e-7, atol=2e-6)


def test_difficult_per_rhs_convergence_preserves_zero_rhs():
    graph, scores = _difficult_directed_fixture(torch.float32)
    scores[:, -1] = 0
    first = solve_rwr_cgls(graph, scores, alpha=0.98)
    second = solve_rwr_cgls(graph, scores, alpha=0.98)

    assert first.restarts_per_rhs[0] == 0
    assert first.restarts_per_rhs[1] == 0
    assert first.restarts_per_rhs[-1] == 0
    assert torch.count_nonzero(first.scores[:, -1]) == 0
    assert torch.equal(first.scores, second.scores)
    assert first.iterations == second.iterations
    assert first.work_count == second.work_count
    assert first.restarts_per_rhs == second.restarts_per_rhs
    assert first.residual_replacements_per_rhs == (
        second.residual_replacements_per_rhs
    )
    assert first.restart_reason_counts == second.restart_reason_counts


def test_default_dispatcher_uses_stabilized_cgls():
    graph, scores = _difficult_directed_fixture(torch.float32)
    direct = solve_rwr_cgls(graph, scores, alpha=0.98)
    dispatched = solve_rwr(graph, scores, alpha=0.98)
    assert dispatched.method == "cgls"
    assert torch.equal(dispatched.scores, direct.scores)
    assert dispatched.restarts_per_rhs == direct.restarts_per_rhs
    assert dispatched.restart_reason_counts == direct.restart_reason_counts


def test_canonical_1024_by_171_sparse_smoke():
    graph = _identity_fallback_graph(1024, 12)
    scores = torch.randn(1024, 171)
    result = solve_rwr_cgls(graph, scores, alpha=0.98)
    assert result.scores.shape == (1024, 171)
    assert result.scores.dtype == torch.float32
    torch.testing.assert_close(result.scores, scores, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_nontrivial_canonical_directed_restart_smoke():
    num_nodes, classes, k = 1024, 171, 12
    rows = torch.arange(num_nodes, device="cuda")[:, None]
    offsets = torch.arange(1, k + 1, device="cuda")[None, :]
    indices = (rows + offsets) % num_nodes
    weights = torch.arange(1, k + 1, dtype=torch.float32, device="cuda")
    weights = (weights / weights.sum()).expand(num_nodes, -1).contiguous()
    graph = DirectedTopKGraph(
        neighbor_indices=indices,
        transition_weights=weights,
        edge_affinities=weights,
        self_loop_fallback=torch.zeros(num_nodes, dtype=torch.bool, device="cuda"),
        num_nodes=num_nodes,
        k=k,
        affinity_power=3.0,
    )
    generator = torch.Generator(device="cuda").manual_seed(4404)
    scores = torch.randn(
        num_nodes, classes, device="cuda", generator=generator
    )

    result = solve_rwr_cgls(graph, scores, alpha=0.98)
    replay = solve_rwr_cgls(graph, scores, alpha=0.98)

    assert result.scores.shape == (num_nodes, classes)
    assert result.scores.device.type == "cuda"
    assert result.maximum_scaled_residual <= 1
    assert result.certificate_dtype == "float64_quantized_fp32_system"
    assert result.fp64_certificate_checks >= 2
    assert torch.equal(result.scores, replay.scores)
    assert result.fp64_certificate_checks == replay.fp64_certificate_checks
    assert not torch.equal(graph.to_dense(), graph.to_dense().T)


def test_extreme_valid_alpha_matches_dense_reference():
    graph = _asymmetric_graph()
    scores = _scores(torch.float64)
    result = solve_rwr_cgls(
        graph, scores, alpha=0.999, rtol=1e-9, atol=1e-12, max_iter=100
    )
    torch.testing.assert_close(
        result.scores,
        _dense_solution(graph, scores, 0.999),
        rtol=2e-7,
        atol=2e-8,
    )


def test_dispatcher_rejects_unknown_method():
    with pytest.raises(RWRInputError, match="method"):
        solve_rwr(_asymmetric_graph(), _scores(), method="cg")


def test_library_solver_does_not_print(capsys):
    solve_rwr_cgls(_asymmetric_graph(), _scores(), alpha=0.8)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_diagnostics_are_finite_and_nonnegative():
    result = solve_rwr_cgls(_asymmetric_graph(), _scores(), alpha=0.8)
    for value in (
        result.absolute_residual_inf,
        result.maximum_scaled_residual,
        result.fixed_point_delta_inf,
    ):
        assert math.isfinite(value)
        assert value >= 0


def test_sparse_fp64_certificate_detects_fp32_rounding_boundary():
    graph = _asymmetric_graph()
    solution = torch.tensor(
        [-2.3779537677764893, 12.662543296813965,
         5.293134689331055, -4.801339626312256],
        dtype=torch.float32,
    )[:, None]
    rhs = torch.tensor(
        [-9.213384628295898, 5.232193470001221,
         -7.72400426864624, 1.721980094909668],
        dtype=torch.float32,
    )[:, None]
    working = rhs - SparseRWROperator(graph, 0.98).matmul(solution)
    working_norm = torch.linalg.vector_norm(working, dim=0)
    certificate = rwr_module._sparse_fp64_residual_certificate(
        graph,
        0.98,
        solution,
        rhs,
        rtol=0,
        atol=8.70288799,
    )

    assert float(working_norm.item() / (2 * 8.70288799)) <= 1
    assert certificate.scaled_residual.item() > 1
    assert certificate.residual.dtype == torch.float64
    assert certificate.residual.device == solution.device


def test_sparse_fp64_certificate_matches_independent_dense_quantized_system():
    graph = _asymmetric_graph()
    scores = _scores(torch.float32)
    result = solve_rwr_cgls(graph, scores, alpha=0.98)
    rhs = (1 - 0.98) * scores
    certificate = rwr_module._sparse_fp64_residual_certificate(
        graph,
        0.98,
        result.scores,
        rhs,
        columns=torch.tensor([2, 0], dtype=torch.int64),
        rtol=1e-5,
        atol=1e-7,
    )
    adjacency64 = _independent_dense(graph, torch.float64)
    system64 = torch.eye(graph.num_nodes, dtype=torch.float64) - 0.98 * adjacency64
    expected = rhs[:, [2, 0]].to(torch.float64) - system64 @ result.scores[
        :, [2, 0]
    ].to(torch.float64)
    expected_norm = torch.linalg.vector_norm(expected, dim=0)
    expected_threshold = 1e-7 * math.sqrt(graph.num_nodes) + 1e-5 * torch.linalg.vector_norm(
        rhs[:, [2, 0]].to(torch.float64), dim=0
    )

    torch.testing.assert_close(certificate.residual, expected, rtol=0, atol=1e-15)
    torch.testing.assert_close(
        certificate.scaled_residual,
        expected_norm / expected_threshold,
        rtol=1e-8,
        atol=1e-12,
    )


def test_fp64_rejected_candidate_restarts_only_rejected_rhs(monkeypatch):
    original = rwr_module._sparse_fp64_residual_certificate
    candidate_calls = []
    injected = False

    def reject_one_candidate(*args, **kwargs):
        nonlocal injected
        certificate = original(*args, **kwargs)
        columns = certificate.columns.tolist()
        candidate_calls.append(columns)
        if kwargs.get("columns") is not None and not injected and columns:
            injected = True
            scaled = certificate.scaled_residual.clone()
            scaled[0] = 1.01
            residual = certificate.residual.clone()
            residual[:, 0] *= 1.02
            return rwr_module._FP64ResidualCertificate(
                certificate.columns,
                residual,
                certificate.residual_norm,
                certificate.threshold,
                scaled,
            )
        return certificate

    monkeypatch.setattr(
        rwr_module, "_sparse_fp64_residual_certificate", reject_one_candidate
    )
    result = solve_rwr_cgls(
        _asymmetric_graph(), _scores(torch.float32), alpha=0.8
    )

    assert injected
    assert result.converged
    assert result.fp64_certificate_rejections == 1
    assert result.fp64_certificate_restart_count == 1
    assert result.restarts_per_rhs[0] == 1
    assert result.restarts_per_rhs[1:] == (0, 0)
    assert dict(result.restart_reason_counts)["fp64_certificate_rejection"] == 1
    assert result.fp64_certificate_work == result.fp64_certificate_checks
    assert result.fp64_certificate_checks == len(candidate_calls)
    assert candidate_calls[-1] == [0, 1, 2]
    for certified_column in (1, 2):
        assert sum(certified_column in call for call in candidate_calls[:-1]) == 1


def test_certificate_retries_are_bounded_by_updates_not_breakdown_streak(monkeypatch):
    original = rwr_module._sparse_fp64_residual_certificate
    rejections_remaining = 4

    def reject_column_zero_four_times(*args, **kwargs):
        nonlocal rejections_remaining
        certificate = original(*args, **kwargs)
        if (
            kwargs.get("columns") is not None
            and 0 in certificate.columns.tolist()
            and rejections_remaining
        ):
            local_index = certificate.columns.tolist().index(0)
            scaled = certificate.scaled_residual.clone()
            scaled[local_index] = 1.01
            residual = certificate.residual.clone()
            residual[:, local_index] *= 1.02
            rejections_remaining -= 1
            return rwr_module._FP64ResidualCertificate(
                certificate.columns,
                residual,
                certificate.residual_norm,
                certificate.threshold,
                scaled,
            )
        return certificate

    monkeypatch.setattr(
        rwr_module,
        "_sparse_fp64_residual_certificate",
        reject_column_zero_four_times,
    )
    result = solve_rwr_cgls(
        _asymmetric_graph(), _scores(torch.float32), alpha=0.8, max_iter=5000
    )

    assert rejections_remaining == 0
    assert result.converged
    assert result.fp64_certificate_rejections == 4
    assert result.fp64_certificate_restart_count == 4
    assert result.restarts_per_rhs[0] == 4
    assert result.iterations <= 5000


def test_final_certificate_and_promoted_dtype_telemetry():
    for input_dtype in (torch.float32, torch.float16):
        scores = _scores(torch.float32).to(input_dtype)
        result = solve_rwr_cgls(_asymmetric_graph(), scores, alpha=0.8)
        assert result.scores.dtype == torch.float32
        assert result.scores.device == scores.device
        assert result.certificate_dtype == "float64_quantized_fp32_system"
        assert result.fp64_certificate_checks >= 2
        assert result.fp64_certificate_work == result.fp64_certificate_checks
        assert result.fp64_certified_rhs == scores.shape[1]
        assert result.certified_maximum_scaled_residual <= 1
        assert result.maximum_scaled_residual == (
            result.certified_maximum_scaled_residual
        )


def test_native_fp64_uses_native_rhs_and_no_quantized_certificate(monkeypatch):
    monkeypatch.setattr(
        rwr_module,
        "_sparse_fp64_residual_certificate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("quantized FP32 certificate used by native FP64 solve")
        ),
    )
    graph = _asymmetric_graph()
    scores = _scores(torch.float64)
    result = solve_rwr_cgls(graph, scores, alpha=0.8)
    expected = _dense_solution(graph, scores, 0.8)
    torch.testing.assert_close(result.scores, expected, rtol=2e-10, atol=2e-11)
    assert result.certificate_dtype == "float64_native_solver"
    assert result.fp64_certificate_checks == 0
    assert result.fp64_certificate_work == 0


def test_certificate_exhaustion_has_complete_fp32_and_fp64_diagnostics(monkeypatch):
    original = rwr_module._sparse_fp64_residual_certificate

    def reject_every_candidate(*args, **kwargs):
        certificate = original(*args, **kwargs)
        return rwr_module._FP64ResidualCertificate(
            certificate.columns,
            certificate.residual,
            certificate.residual_norm,
            certificate.threshold,
            torch.full_like(certificate.scaled_residual, 2.0),
        )

    monkeypatch.setattr(
        rwr_module, "_sparse_fp64_residual_certificate", reject_every_candidate
    )
    with pytest.raises(RWRNonConvergenceError) as raised:
        solve_rwr_cgls(
            _asymmetric_graph(),
            _scores(torch.float32),
            alpha=0.8,
            max_iter=2,
        )
    error = raised.value
    assert error.iteration == 2
    assert error.work_count >= error.iteration
    assert error.fp64_certificate_checks >= 1
    assert error.fp64_certificate_rejections >= 1
    assert error.certificate_dtype == "float64_quantized_fp32_system"
    assert error.working_primal_residual is not None
    assert error.certified_primal_residual is not None
    assert error.certified_max_scaled_primal_residual == 2.0
    assert error.rtol == 1e-5
    assert error.atol == 1e-7


def test_certificate_path_contains_no_dense_solve_or_cpu_transfer():
    source = inspect.getsource(rwr_module._sparse_fp64_residual_certificate)
    assert "to_dense" not in source
    assert "linalg.solve" not in source
    assert ".cpu(" not in source
    assert "_transpose_matmul" not in source
    assert inspect.signature(solve_rwr_cgls).parameters["max_iter"].default == 5000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_device_is_preserved_and_cpu_agrees():
    cpu_graph = _asymmetric_graph()
    gpu_graph = _asymmetric_graph("cuda")
    scores = _scores(torch.float64)
    cpu = solve_rwr_cgls(cpu_graph, scores, alpha=0.8)
    gpu = solve_rwr_cgls(gpu_graph, scores.cuda(), alpha=0.8)
    gpu_replay = solve_rwr_cgls(gpu_graph, scores.cuda(), alpha=0.8)
    assert gpu.scores.device.type == "cuda"
    torch.testing.assert_close(gpu.scores.cpu(), cpu.scores, rtol=2e-10, atol=2e-11)
    assert torch.equal(gpu.scores, gpu_replay.scores)
    assert gpu.iterations == gpu_replay.iterations
    assert gpu.fp64_certificate_checks == gpu_replay.fp64_certificate_checks
    assert gpu.certified_maximum_scaled_residual == (
        gpu_replay.certified_maximum_scaled_residual
    )
