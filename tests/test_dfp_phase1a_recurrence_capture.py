"""Tests for src/dfp_phase1a_recurrence_capture.py: the DFP-specific
multi-horizon capture binding layer over the existing, unmodified
finite_step_propagate recurrence kernel. CPU-only, no CUDA, no dataset
access. Independently unrolls the recurrence in this file (never
importing the new capture helper for that part) to prove the binding
layer adds no scientific behavior of its own.
"""

from __future__ import annotations

import copy
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
finite_step_propagate = cover_dr.finite_step_propagate
build_directed_topk_graph = cover_dr.build_directed_topk_graph
DirectedTopKGraph = cover_dr.DirectedTopKGraph

import src.dfp_phase1a_recurrence_capture as rc  # noqa: E402
from src.dfp_phase1a_identity import load_identity  # noqa: E402

PROTOCOL_PATH = ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml"


def _real_protocol():
    return load_identity(PROTOCOL_PATH, repo_root=ROOT)


def _random_graph(num_nodes=20, dim=10, k=12, affinity_power=3.0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    features = torch.nn.functional.normalize(torch.randn(num_nodes, dim, generator=generator), dim=-1)
    return build_directed_topk_graph(features, k=k, affinity_power=affinity_power)


def _hand_unroll(graph, s0, *, alpha, steps):
    """Independent, from-scratch reference unroll -- never imports or
    calls the capture helper being tested."""
    p = s0.clone()
    trace = {0: p.clone()}
    for t in range(1, steps + 1):
        p = alpha * graph.matmul(p) + (1 - alpha) * s0
        trace[t] = p.clone()
    return trace


# ---------------------------------------------------------------------------
# Contract extraction from the live protocol identity
# ---------------------------------------------------------------------------


def test_contract_from_real_protocol():
    protocol = _real_protocol()
    contract = rc.dfp_recurrence_contract_from_protocol(protocol)
    assert contract.alpha == 0.98
    assert contract.steps == 320
    assert contract.horizons == (0, 16, 32, 64, 320)


def test_contract_does_not_hardcode_horizons_module_constant():
    # There must be no actual code-level [0, 16, 32, 64, 320] /
    # (0, 16, 32, 64, 320) literal serving as a second scientific
    # authority in this module. The module's own docstrings legitimately
    # *mention* that exact text while explaining this anti-pattern, so
    # strip every docstring (via ast) before searching, rather than a
    # bare substring match over the raw file text.
    import ast

    source = Path(rc.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc_node = node.body[0] if node.body else None
            if (
                isinstance(doc_node, ast.Expr)
                and isinstance(doc_node.value, ast.Constant)
                and isinstance(doc_node.value.value, str)
            ):
                docstring_lines.update(range(doc_node.lineno, (doc_node.end_lineno or doc_node.lineno) + 1))

    for lineno, line in enumerate(source.splitlines(), start=1):
        if lineno in docstring_lines:
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "[0, 16, 32, 64, 320]" not in line, f"found a live horizons literal at line {lineno}: {line!r}"
        assert "(0, 16, 32, 64, 320)" not in line, f"found a live horizons literal at line {lineno}: {line!r}"


# ---------------------------------------------------------------------------
# Horizons validation adversarial matrix
# ---------------------------------------------------------------------------


def _mutated_protocol(mutate):
    protocol = copy.deepcopy(_real_protocol())
    protocol["propagation"] = dict(protocol["propagation"])
    mutate(protocol["propagation"])
    return protocol


def _mutated_graph_block(mutate):
    protocol = copy.deepcopy(_real_protocol())
    protocol["graph"] = dict(protocol["graph"])
    mutate(protocol["graph"])
    return protocol


def test_missing_horizon_rejected():
    protocol = _mutated_protocol(lambda p: p.__setitem__("horizons", [0, 16, 32, 64]))
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match="final entry"):
        rc.dfp_recurrence_contract_from_protocol(protocol)


def test_extra_horizon_rejected():
    protocol = _mutated_protocol(lambda p: p.__setitem__("horizons", [0, 16, 32, 64, 160, 320]))
    # [0, 16, 32, 64, 160, 320] is structurally valid in isolation
    # (strictly ascending, in-bounds, final entry == steps) -- but the
    # canonical propagation authority (src.dfp_phase1a_identity.
    # validate_propagation_block) pins the exact horizons set, so this
    # extra T=160 horizon must still be rejected before a recurrence
    # contract is ever constructed, not merely by the separate
    # requested_horizons exact-equality gate below.
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match="horizons must be exactly"):
        rc.dfp_recurrence_contract_from_protocol(protocol)
    graph = _random_graph()
    s0 = torch.randn(20, 4)
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError):
        rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol, requested_horizons=(0, 16, 32, 64, 320))


def test_reordered_horizons_rejected():
    protocol = _mutated_protocol(lambda p: p.__setitem__("horizons", [0, 32, 16, 64, 320]))
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match="ascending"):
        rc.dfp_recurrence_contract_from_protocol(protocol)


def test_duplicate_horizons_rejected():
    protocol = _mutated_protocol(lambda p: p.__setitem__("horizons", [0, 16, 16, 32, 64, 320]))
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match="ascending"):
        rc.dfp_recurrence_contract_from_protocol(protocol)


def test_negative_horizon_rejected():
    protocol = _mutated_protocol(lambda p: p.__setitem__("horizons", [-1, 16, 32, 64, 320]))
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match="non-negative"):
        rc.dfp_recurrence_contract_from_protocol(protocol)


def test_horizon_above_total_updates_rejected():
    protocol = _mutated_protocol(lambda p: p.__setitem__("horizons", [0, 16, 32, 64, 999]))
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match="exceeds total_steps"):
        rc.dfp_recurrence_contract_from_protocol(protocol)


def test_bool_as_int_horizon_rejected():
    protocol = _mutated_protocol(lambda p: p.__setitem__("horizons", [0, 16, 32, 64, True]))
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match="exact non-boolean integer"):
        rc.dfp_recurrence_contract_from_protocol(protocol)


def test_int_as_float_horizon_rejected():
    protocol = _mutated_protocol(lambda p: p.__setitem__("horizons", [0, 16, 32, 64.0, 320]))
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match="exact non-boolean integer"):
        rc.dfp_recurrence_contract_from_protocol(protocol)


def test_numeric_string_horizon_rejected():
    protocol = _mutated_protocol(lambda p: p.__setitem__("horizons", [0, 16, 32, "64", 320]))
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match="exact non-boolean integer"):
        rc.dfp_recurrence_contract_from_protocol(protocol)


def test_non_list_container_horizons_rejected():
    protocol = _mutated_protocol(lambda p: p.__setitem__("horizons", (0, 16, 32, 64, 320)))
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match="exact list"):
        rc.dfp_recurrence_contract_from_protocol(protocol)


def test_steps_inconsistent_with_horizon_320_rejected():
    protocol = _mutated_protocol(lambda p: p.__setitem__("steps", 640))
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match="final entry"):
        rc.dfp_recurrence_contract_from_protocol(protocol)


def test_bool_as_int_steps_rejected():
    protocol = _mutated_protocol(lambda p: p.__setitem__("steps", True))
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match="exact non-boolean integer"):
        rc.dfp_recurrence_contract_from_protocol(protocol)


def test_int_as_float_alpha_rejected():
    protocol = _mutated_protocol(lambda p: p.__setitem__("alpha", 1))
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match="exact finite float"):
        rc.dfp_recurrence_contract_from_protocol(protocol)


def test_numeric_string_alpha_rejected():
    protocol = _mutated_protocol(lambda p: p.__setitem__("alpha", "0.98"))
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match="exact finite float"):
        rc.dfp_recurrence_contract_from_protocol(protocol)


def test_nan_alpha_rejected():
    protocol = _mutated_protocol(lambda p: p.__setitem__("alpha", float("nan")))
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match="finite"):
        rc.dfp_recurrence_contract_from_protocol(protocol)


def test_validation_does_not_mutate_protocol_identity():
    protocol = _real_protocol()
    before = copy.deepcopy(protocol)
    rc.dfp_recurrence_contract_from_protocol(protocol)
    assert protocol == before


def test_requested_horizons_matching_frozen_identity_accepted():
    protocol = _real_protocol()
    graph = _random_graph()
    s0 = torch.randn(20, 4)
    trace = rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol, requested_horizons=(0, 16, 32, 64, 320))
    assert set(trace.snapshots) == {0, 16, 32, 64, 320}


def test_requested_horizons_list_accepted():
    # list is explicitly supported alongside tuple.
    protocol = _real_protocol()
    graph = _random_graph()
    s0 = torch.randn(20, 4)
    trace = rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol, requested_horizons=[0, 16, 32, 64, 320])
    assert set(trace.snapshots) == {0, 16, 32, 64, 320}


# ---------------------------------------------------------------------------
# requested_horizons container-type rejection (an arbitrary iterable must
# never be silently coerced via tuple(); a dict whose KEYS happen to equal
# the canonical horizons must still be rejected on container type alone)
# ---------------------------------------------------------------------------


def _assert_requested_horizons_rejected(value, *, match="list or tuple"):
    protocol = _real_protocol()
    graph = _random_graph()
    call_count = {"n": 0}
    real_matmul = graph.matmul

    def _counting_matmul(x):
        call_count["n"] += 1
        return real_matmul(x)

    object.__setattr__(graph, "matmul", _counting_matmul)
    s0 = torch.randn(20, 4)
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match=match):
        rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol, requested_horizons=value)
    assert call_count["n"] == 0


def test_requested_horizons_dict_with_matching_keys_rejected():
    # The exact reproduction from the audit: tuple(dict) would iterate
    # this dict's keys as (0, 16, 32, 64, 320), which happens to equal
    # the canonical horizons -- it must still be rejected on container
    # type alone, never silently accepted.
    _assert_requested_horizons_rejected({0: 1, 16: 1, 32: 1, 64: 1, 320: 1})


def test_requested_horizons_set_rejected():
    _assert_requested_horizons_rejected({0, 16, 32, 64, 320})


def test_requested_horizons_string_rejected():
    _assert_requested_horizons_rejected("0,16,32,64,320")


def test_requested_horizons_generator_rejected():
    _assert_requested_horizons_rejected(h for h in (0, 16, 32, 64, 320))


def test_requested_horizons_numpy_array_rejected():
    numpy = pytest.importorskip("numpy")
    _assert_requested_horizons_rejected(numpy.array([0, 16, 32, 64, 320]))


# ---------------------------------------------------------------------------
# Canonical propagation authority re-validation (structural checks above
# accept some values -- e.g. alpha=0.99, or a horizons list missing T=0
# while still ending at steps -- that are scientifically wrong; these
# tests prove dfp_recurrence_contract_from_protocol independently
# delegates to src.dfp_phase1a_identity.validate_graph_block /
# validate_propagation_block to still reject them, before the first
# graph-matmul call.)
# ---------------------------------------------------------------------------


def _assert_rejected_before_any_matmul(protocol, *, match=None):
    graph = _random_graph()
    call_count = {"n": 0}
    real_matmul = graph.matmul

    def _counting_matmul(x):
        call_count["n"] += 1
        return real_matmul(x)

    object.__setattr__(graph, "matmul", _counting_matmul)
    s0 = torch.randn(20, 4)
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match=match):
        rc.dfp_recurrence_contract_from_protocol(protocol)
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError, match=match):
        rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)
    assert call_count["n"] == 0


def test_altered_alpha_099_rejected_before_execution():
    protocol = _mutated_protocol(lambda p: p.__setitem__("alpha", 0.99))
    _assert_rejected_before_any_matmul(protocol, match="alpha must be exactly 0.98")


def test_altered_alpha_barely_off_rejected_before_execution():
    protocol = _mutated_protocol(lambda p: p.__setitem__("alpha", 0.9800001))
    _assert_rejected_before_any_matmul(protocol, match="alpha must be exactly 0.98")


def test_missing_t0_horizon_rejected_before_execution():
    # [16, 32, 64, 320] is structurally valid in isolation (ascending,
    # in-bounds, final entry == steps) -- the missing T=0 is only caught
    # by the canonical authority delegate, not by local structural checks.
    protocol = _mutated_protocol(lambda p: p.__setitem__("horizons", [16, 32, 64, 320]))
    _assert_rejected_before_any_matmul(protocol, match="horizons must be exactly")


def test_extra_t8_horizon_rejected_before_execution():
    protocol = _mutated_protocol(lambda p: p.__setitem__("horizons", [0, 8, 16, 32, 64, 320]))
    _assert_rejected_before_any_matmul(protocol, match="horizons must be exactly")


def test_reordered_horizons_rejected_before_execution():
    protocol = _mutated_protocol(lambda p: p.__setitem__("horizons", [0, 32, 16, 64, 320]))
    _assert_rejected_before_any_matmul(protocol, match="ascending")


def test_altered_steps_rejected_before_execution():
    # steps=321 with the real, unmutated horizons list: the horizons'
    # final entry (320) then disagrees with steps (321), caught by the
    # pre-existing local structural check before the canonical delegate
    # is even reached -- this proves the new delegate composes correctly
    # with that check rather than masking or bypassing it.
    protocol = _mutated_protocol(lambda p: p.__setitem__("steps", 321))
    _assert_rejected_before_any_matmul(protocol, match="final entry")


def test_altered_canonical_graph_variant_rejected_before_execution():
    protocol = _mutated_graph_block(lambda g: g.__setitem__("canonical_k_variant", 11))
    _assert_rejected_before_any_matmul(protocol, match="canonical_k_variant must be exactly 12")


def test_altered_early_stopping_rejected_before_execution():
    protocol = _mutated_protocol(lambda p: p.__setitem__("early_stopping", True))
    _assert_rejected_before_any_matmul(protocol, match="early_stopping must be false")


def test_altered_fallback_solver_rejected_before_execution():
    protocol = _mutated_protocol(lambda p: p.__setitem__("fallback_solver", "cgls"))
    _assert_rejected_before_any_matmul(protocol)


def test_canonical_authority_delegate_does_not_hardcode_pinned_values():
    # Same anti-duplication guarantee as
    # test_contract_does_not_hardcode_horizons_module_constant, but for
    # the module-level literals this repair could have introduced
    # (0.98, 320, 12) instead of delegating.
    import ast

    source = Path(rc.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc_node = node.body[0] if node.body else None
            if (
                isinstance(doc_node, ast.Expr)
                and isinstance(doc_node.value, ast.Constant)
                and isinstance(doc_node.value.value, str)
            ):
                docstring_lines.update(range(doc_node.lineno, (doc_node.end_lineno or doc_node.lineno) + 1))

    for lineno, line in enumerate(source.splitlines(), start=1):
        if lineno in docstring_lines:
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "0.98" not in line, f"found a live alpha literal at line {lineno}: {line!r}"
        assert "canonical_k_variant" not in line or "validate_graph_block" in line or "protocol_identity[\"graph\"]" in line, (
            f"found unexpected canonical_k_variant reference at line {lineno}: {line!r}"
        )


# ---------------------------------------------------------------------------
# Off-by-one distinguishing fixtures (section 6)
# ---------------------------------------------------------------------------


def test_scalar_one_state_hand_computable_sequence():
    # 1-node graph, self-loop weight 1 -> P(t+1) = alpha*P(t) + (1-alpha)*S0
    # closed form: P(t) = S0 (constant), since alpha*S0+(1-alpha)*S0 = S0.
    # Use a non-self-loop 2-node directed graph instead for a genuinely
    # time-varying scalar-like sequence.
    neighbor_indices = torch.tensor([[1], [0]], dtype=torch.int64)
    transition_weights = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    edge_affinities = transition_weights.clone()
    self_loop_fallback = torch.tensor([False, False])
    graph = DirectedTopKGraph(
        neighbor_indices=neighbor_indices, transition_weights=transition_weights,
        edge_affinities=edge_affinities, self_loop_fallback=self_loop_fallback,
        num_nodes=2, k=1, affinity_power=3.0,
    )
    s0 = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    alpha = 0.98
    hand = _hand_unroll(graph, s0, alpha=alpha, steps=20)
    trace = finite_step_propagate(graph, s0, alpha=alpha, steps=20, snapshot_steps=tuple(range(21)))
    for t in range(21):
        assert torch.equal(trace.snapshots[t], hand[t])
    # hand-verify a few values by direct scalar recursion: x0=1, y0=0,
    # x(t+1) = alpha*y(t) + (1-alpha)*1, y(t+1) = alpha*x(t) + (1-alpha)*0
    x, y = 1.0, 0.0
    for _ in range(5):
        x, y = alpha * y + (1 - alpha) * 1.0, alpha * x + (1 - alpha) * 0.0
    assert abs(trace.snapshots[5][0, 0].item() - x) < 1e-12
    assert abs(trace.snapshots[5][1, 0].item() - y) < 1e-12


def test_directed_asymmetric_transition_matrix_distinguishing_fixture():
    graph = _random_graph(num_nodes=15, dim=8, k=6, seed=11)
    s0 = torch.randn(15, 5, generator=torch.Generator().manual_seed(12))
    alpha = 0.98
    hand = _hand_unroll(graph, s0, alpha=alpha, steps=320)
    trace = finite_step_propagate(graph, s0, alpha=alpha, steps=320, snapshot_steps=(0, 15, 16, 17, 31, 32, 33, 63, 64, 65, 320))
    for t in (0, 15, 16, 17, 31, 32, 33, 63, 64, 65, 320):
        assert torch.equal(trace.snapshots[t], hand[t]), f"mismatch at t={t}"
    # off-by-one distinguishability: T=16 must differ from T=15 and T=17
    assert not torch.equal(trace.snapshots[16], trace.snapshots[15])
    assert not torch.equal(trace.snapshots[16], trace.snapshots[17])
    assert not torch.equal(trace.snapshots[32], trace.snapshots[31])
    assert not torch.equal(trace.snapshots[32], trace.snapshots[33])
    assert not torch.equal(trace.snapshots[64], trace.snapshots[63])
    assert not torch.equal(trace.snapshots[64], trace.snapshots[65])


def test_non_stationary_seed_fixture():
    graph = _random_graph(num_nodes=12, dim=6, k=5, seed=21)
    s0 = torch.zeros(12, 3)
    s0[0, 0] = 1.0  # a single "hot" node, non-uniform
    alpha = 0.98
    hand = _hand_unroll(graph, s0, alpha=alpha, steps=320)
    trace = finite_step_propagate(graph, s0, alpha=alpha, steps=320, snapshot_steps=(0, 16, 32, 64, 320))
    for t in (0, 16, 32, 64, 320):
        assert torch.equal(trace.snapshots[t], hand[t])


def test_fixed_point_fixture_still_executes_all_320_updates():
    # A graph + seed combination where P is ALREADY at its fixed point
    # (P = alpha*A@P + (1-alpha)*S0 satisfied exactly by P=S0 requires
    # A@S0 == S0; construct a uniform S0 with a row-stochastic graph so
    # A@S0 == S0 exactly for a constant-per-column S0).
    graph = _random_graph(num_nodes=10, dim=6, k=4, seed=31)
    s0 = torch.ones(10, 3) * 0.5  # constant per node -> A @ s0 == s0 (row-stochastic)
    alpha = 0.98
    assert torch.allclose(graph.matmul(s0), s0, atol=1e-6)
    call_count = {"n": 0}
    real_matmul = graph.matmul

    def _counting_matmul(x):
        call_count["n"] += 1
        return real_matmul(x)

    object.__setattr__(graph, "matmul", _counting_matmul)
    trace = finite_step_propagate(graph, s0, alpha=alpha, steps=320, snapshot_steps=(0, 16, 32, 64, 320))
    assert call_count["n"] == 320  # all 320 updates still executed despite the fixed point
    assert trace.steps_completed == 320
    for t in (0, 16, 32, 64, 320):
        assert torch.allclose(trace.snapshots[t], s0, atol=1e-6)


def test_float32_and_float64_cases():
    graph = _random_graph(num_nodes=10, dim=6, k=4, seed=41)
    s0_32 = torch.randn(10, 3, dtype=torch.float32, generator=torch.Generator().manual_seed(42))
    s0_64 = s0_32.to(torch.float64)
    trace32 = finite_step_propagate(graph, s0_32, alpha=0.98, steps=320, snapshot_steps=(0, 16, 32, 64, 320))
    trace64 = finite_step_propagate(graph, s0_64, alpha=0.98, steps=320, snapshot_steps=(0, 16, 32, 64, 320))
    assert trace32.dtype == torch.float32
    assert trace64.dtype == torch.float64
    for snap in trace32.snapshots.values():
        assert snap.dtype == torch.float32
    for snap in trace64.snapshots.values():
        assert snap.dtype == torch.float64
    # fp32 vs fp64 should agree closely but not necessarily bit-exactly
    assert torch.allclose(trace32.snapshots[320].double(), trace64.snapshots[320], atol=1e-4)


def test_multiclass_tensor_labels_differ_at_successive_horizons():
    graph = _random_graph(num_nodes=25, dim=10, k=8, seed=51)
    s0 = torch.randn(25, 6, generator=torch.Generator().manual_seed(52))
    trace = finite_step_propagate(graph, s0, alpha=0.98, steps=320, snapshot_steps=(0, 16, 32, 64, 320))
    labels = {t: trace.snapshots[t].argmax(dim=-1) for t in (0, 16, 32, 64, 320)}
    # at least one node's argmax label must differ between at least two
    # horizons on a real random fixture (proves the fixture is genuinely
    # distinguishing, not degenerate)
    all_same = all(torch.equal(labels[0], labels[t]) for t in (16, 32, 64, 320))
    assert not all_same


def test_completed_updates_exactly_320():
    protocol = _real_protocol()
    graph = _random_graph(seed=61)
    s0 = torch.randn(20, 4, generator=torch.Generator().manual_seed(62))
    trace = rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)
    assert trace.steps_completed == 320


# ---------------------------------------------------------------------------
# Existing-result invariance (section 7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [71, 72, 73])
def test_capture_t320_matches_existing_single_horizon_call(seed):
    protocol = _real_protocol()
    graph = _random_graph(seed=seed)
    s0 = torch.randn(20, 4, generator=torch.Generator().manual_seed(seed))
    existing = finite_step_propagate(graph, s0, alpha=0.98, steps=320, snapshot_steps=(320,))
    captured = rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)
    assert torch.equal(existing.snapshots[320], captured.snapshots[320])
    assert existing.steps_completed == captured.steps_completed
    assert existing.dtype == captured.dtype


def test_capture_matches_existing_on_fallback_row_graph():
    num_nodes = 15
    dim = num_nodes + 8
    generator = torch.Generator().manual_seed(81)
    features = torch.zeros(num_nodes, dim)
    for row in range(num_nodes):
        features[row, row] = 1.0  # every row is a pure identity vector -> all-fallback graph
    graph = build_directed_topk_graph(torch.nn.functional.normalize(features, dim=-1), k=12, affinity_power=3.0)
    assert bool(graph.self_loop_fallback.all())
    protocol = _real_protocol()
    s0 = torch.randn(num_nodes, 4, generator=generator)
    existing = finite_step_propagate(graph, s0, alpha=0.98, steps=320, snapshot_steps=(320,))
    captured = rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)
    assert torch.equal(existing.snapshots[320], captured.snapshots[320])


def test_capture_matches_existing_on_tie_fixture():
    # two neighbors with identical affinity (forced tie at rank boundary)
    num_nodes = 14
    dim = 6
    generator = torch.Generator().manual_seed(91)
    features = torch.nn.functional.normalize(torch.randn(num_nodes, dim, generator=generator), dim=-1)
    features[1] = features[0]  # force an exact tie between node 0 and node 1's similarity profile to others
    graph = build_directed_topk_graph(features, k=6, affinity_power=3.0)
    protocol = _real_protocol()
    s0 = torch.randn(num_nodes, 3, generator=generator)
    existing = finite_step_propagate(graph, s0, alpha=0.98, steps=320, snapshot_steps=(320,))
    captured = rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)
    assert torch.equal(existing.snapshots[320], captured.snapshots[320])


def test_capture_does_not_mutate_graph_or_s0():
    protocol = _real_protocol()
    graph = _random_graph(seed=101)
    neighbor_before = graph.neighbor_indices.clone()
    weights_before = graph.transition_weights.clone()
    s0 = torch.randn(20, 4, generator=torch.Generator().manual_seed(102))
    s0_before = s0.clone()
    rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)
    assert torch.equal(graph.neighbor_indices, neighbor_before)
    assert torch.equal(graph.transition_weights, weights_before)
    assert torch.equal(s0, s0_before)


def test_snapshots_are_non_aliased_and_mutation_isolated():
    protocol = _real_protocol()
    graph = _random_graph(seed=111)
    s0 = torch.randn(20, 4, generator=torch.Generator().manual_seed(112))
    trace = rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)
    snap0, snap320 = trace.snapshots[0], trace.snapshots[320]
    assert snap0.data_ptr() != snap320.data_ptr()
    assert snap0.data_ptr() != s0.data_ptr()
    before320 = snap320.clone()
    snap0 += 1000.0  # mutate one returned snapshot
    assert torch.equal(trace.snapshots[320], before320)  # unaffected
    assert not torch.equal(s0, snap0)  # s0 itself untouched by mutating the snapshot


def test_snapshots_do_not_retain_autograd_graph():
    protocol = _real_protocol()
    graph = _random_graph(seed=121)
    s0 = torch.randn(20, 4, generator=torch.Generator().manual_seed(122), requires_grad=True)
    trace = rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)
    for snap in trace.snapshots.values():
        assert not snap.requires_grad
        assert snap.grad_fn is None


def test_capture_preserves_dtype_and_does_not_move_device():
    protocol = _real_protocol()
    graph = _random_graph(seed=131)
    s0 = torch.randn(20, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(132))
    trace = rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)
    for snap in trace.snapshots.values():
        assert snap.dtype == torch.float64
        assert snap.device == s0.device


# ---------------------------------------------------------------------------
# Single-pass execution contract (section 5)
# ---------------------------------------------------------------------------


def test_single_pass_execution_contract_for_n_windows():
    protocol = _real_protocol()
    n_windows = 3
    graph_build_count = {"n": 0}
    matmul_call_count = {"n": 0}
    completed_updates_total = {"n": 0}

    real_build = build_directed_topk_graph

    def _counting_build(*args, **kwargs):
        graph_build_count["n"] += 1
        graph = real_build(*args, **kwargs)
        real_matmul = graph.matmul

        def _counting_matmul(x):
            matmul_call_count["n"] += 1
            completed_updates_total["n"] += 1
            return real_matmul(x)

        object.__setattr__(graph, "matmul", _counting_matmul)
        return graph

    captures_per_horizon = {0: 0, 16: 0, 32: 0, 64: 0, 320: 0}
    for w in range(n_windows):
        generator = torch.Generator().manual_seed(1000 + w)
        features = torch.nn.functional.normalize(torch.randn(20, 8, generator=generator), dim=-1)
        graph = _counting_build(features, k=12, affinity_power=3.0)
        s0 = torch.randn(20, 4, generator=generator)
        trace = rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)
        for horizon in captures_per_horizon:
            if horizon in trace.snapshots:
                captures_per_horizon[horizon] += 1

    assert graph_build_count["n"] == n_windows
    assert completed_updates_total["n"] == 320 * n_windows
    for horizon, count in captures_per_horizon.items():
        assert count == n_windows, f"horizon {horizon} captured {count} times, expected {n_windows}"


# ---------------------------------------------------------------------------
# Forbidden-path poisoning (section 8)
# ---------------------------------------------------------------------------


def test_forbidden_paths_never_touched(monkeypatch):
    def _poison(name):
        def _raise(*a, **kw):
            raise AssertionError(f"forbidden path {name} was called")
        return _raise

    monkeypatch.setattr(torch.linalg, "solve", _poison("torch.linalg.solve"))
    monkeypatch.setattr(torch.linalg, "inv", _poison("torch.linalg.inv"))
    monkeypatch.setattr(torch.linalg, "lstsq", _poison("torch.linalg.lstsq (CGLS-adjacent)"))
    monkeypatch.setattr(torch.linalg, "svdvals", _poison("torch.linalg.svdvals"))
    monkeypatch.setattr(torch.linalg, "eig", _poison("torch.linalg.eig"))
    monkeypatch.setattr(torch.linalg, "eigvals", _poison("torch.linalg.eigvals"))
    monkeypatch.setattr(cover_dr, "dense_fp64_equilibrium_reference", _poison("dense_fp64_equilibrium_reference"))
    monkeypatch.setattr(cover_dr, "compute_condition_diagnostics", _poison("compute_condition_diagnostics"))

    protocol = _real_protocol()
    graph = _random_graph(seed=141)
    s0 = torch.randn(20, 4, generator=torch.Generator().manual_seed(142))
    trace = rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)
    assert trace.steps_completed == 320
    assert set(trace.snapshots) == {0, 16, 32, 64, 320}


def test_exactly_320_updates_with_zero_scores():
    protocol = _real_protocol()
    graph = _random_graph(seed=151)
    s0 = torch.zeros(20, 4)
    call_count = {"n": 0}
    real_matmul = graph.matmul

    def _counting(x):
        call_count["n"] += 1
        return real_matmul(x)

    object.__setattr__(graph, "matmul", _counting)
    trace = rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)
    assert call_count["n"] == 320
    assert trace.steps_completed == 320
    for snap in trace.snapshots.values():
        assert torch.equal(snap, torch.zeros(20, 4))


def test_exactly_320_updates_with_all_fallback_rows():
    num_nodes = 13
    dim = num_nodes + 4
    generator = torch.Generator().manual_seed(161)
    features = torch.zeros(num_nodes, dim)
    for row in range(num_nodes):
        features[row, row] = 1.0
    graph = build_directed_topk_graph(torch.nn.functional.normalize(features, dim=-1), k=12, affinity_power=3.0)
    assert bool(graph.self_loop_fallback.all())
    protocol = _real_protocol()
    s0 = torch.randn(num_nodes, 3, generator=generator)
    call_count = {"n": 0}
    real_matmul = graph.matmul

    def _counting(x):
        call_count["n"] += 1
        return real_matmul(x)

    object.__setattr__(graph, "matmul", _counting)
    trace = rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)
    assert call_count["n"] == 320
    assert trace.steps_completed == 320


# ---------------------------------------------------------------------------
# Failure behavior: validation before propagation, exception propagation
# ---------------------------------------------------------------------------


def test_validation_failure_occurs_before_any_propagation_update():
    protocol = _mutated_protocol(lambda p: p.__setitem__("horizons", [0, 16, 32, 64, 999]))
    graph = _random_graph(seed=171)
    call_count = {"n": 0}
    real_matmul = graph.matmul

    def _counting(x):
        call_count["n"] += 1
        return real_matmul(x)

    object.__setattr__(graph, "matmul", _counting)
    s0 = torch.randn(20, 4, generator=torch.Generator().manual_seed(172))
    with pytest.raises(rc.DfpPhase1aRecurrenceCaptureError):
        rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)
    assert call_count["n"] == 0


def test_keyboard_interrupt_propagates(monkeypatch):
    protocol = _real_protocol()
    graph = _random_graph(seed=181)
    s0 = torch.randn(20, 4, generator=torch.Generator().manual_seed(182))

    def _boom(*a, **kw):
        raise KeyboardInterrupt()

    monkeypatch.setattr(rc, "finite_step_propagate", _boom)
    with pytest.raises(KeyboardInterrupt):
        rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)


def test_system_exit_propagates(monkeypatch):
    protocol = _real_protocol()
    graph = _random_graph(seed=191)
    s0 = torch.randn(20, 4, generator=torch.Generator().manual_seed(192))

    def _boom(*a, **kw):
        raise SystemExit(3)

    monkeypatch.setattr(rc, "finite_step_propagate", _boom)
    with pytest.raises(SystemExit):
        rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)


def test_memory_error_propagates(monkeypatch):
    protocol = _real_protocol()
    graph = _random_graph(seed=201)
    s0 = torch.randn(20, 4, generator=torch.Generator().manual_seed(202))

    def _boom(*a, **kw):
        raise MemoryError()

    monkeypatch.setattr(rc, "finite_step_propagate", _boom)
    with pytest.raises(MemoryError):
        rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)


def test_cuda_never_initialized():
    assert torch.cuda.is_initialized() is False
    protocol = _real_protocol()
    graph = _random_graph(seed=211)
    s0 = torch.randn(20, 4, generator=torch.Generator().manual_seed(212))
    rc.capture_dfp_phase1a_trajectory(graph, s0, protocol_identity=protocol)
    assert torch.cuda.is_initialized() is False
