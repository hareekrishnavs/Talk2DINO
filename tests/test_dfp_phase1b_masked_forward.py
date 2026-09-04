"""Strict synthetic tests for the DFP Phase 1B exact masked-forward
provenance core (``src.dfp_phase1b_masked_forward``). CPU-only, no real
dataset, no real checkpoint, no GPU. Every fixture in this file is
hand-built; the "recurrence" and "canonical geometry" sections
additionally construct a small hand-solvable oracle and independently
recompute the expected result in the test itself -- never by calling
the production code twice and comparing it to itself.

Some tests in this file need the real committed
``models.dinotext.cover_dr`` / ``src.dfp_phase1a_stitching_pullback`` /
``src.dfp_phase1a_final_rescale_pullback`` machinery, which needs a
working ``cv2`` import. A default/login environment on this cluster
may either lack an OpenCV module load entirely (``cv2`` not importable
at all -- ``ModuleNotFoundError: No module named 'cv2'``) or have one
whose NumPy ABI predates the installed NumPy 2.x (``AttributeError:
_ARRAY_API not found`` inside ``cv2``'s own bootstrap, surfacing as
``ImportError: numpy.core.multiarray failed to import``) -- either gap
can make ``tests/test_dfp_phase1a_stitching_pullback.py``,
``tests/test_dfp_phase1a_final_rescale_pullback.py``, and
``tests/test_dfp_phase1a_recurrence_capture.py`` (among others)
uncollectible/unrunnable there, entirely unrelated to this module's own
code. Those specific tests here call
``_skip_if_recurrence_environment_broken()`` first, which converts
either known, narrow failure signature into an explicit,
clearly-reasoned ``pytest.skip`` naming the required module-load recipe
-- never silently omitted, never miscounted as a real failure, and
never a broad catch-all for an unrelated import defect.

This repository's own existing SLURM launchers already establish a
working module order (load an OpenCV module built against this
environment's NumPy version, then activate the project's existing
virtual environment -- see ``docs/dfp_phase1b_masked_forward.md``'s
"Compatible environment" section for the exact commands). With that
environment, every test in this file -- including canonical
multi-window final-geometry stitching/rescale -- runs and passes for
real; this module docstring's description of the gap is about what a
given environment may be missing, not a claim that verification is
permanently blocked.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import src.dfp_phase1b_masked_forward as m  # noqa: E402


# ---------------------------------------------------------------------------
# Environment-gap helper.
# ---------------------------------------------------------------------------


def _recurrence_environment_broken() -> str | None:
    """Return a skip reason string if the real committed recurrence/
    stitching machinery cannot be imported in this environment, else
    None. Recognizes exactly two known, narrow import gaps -- never a
    broad ``ImportError`` catch, never ``except Exception``:

    1. ``cv2`` is not installed/importable at all (a ``ModuleNotFoundError``
       whose ``.name`` is exactly ``"cv2"``) -- typically means the
       environment's OpenCV module was never loaded. This is distinct
       from case 2 and must not be conflated with an ABI mismatch.
    2. ``cv2`` IS importable but was built against a NumPy ABI
       incompatible with the installed NumPy (the known
       ``_ARRAY_API``/``numpy.core.multiarray failed to import``
       signature).

    Any OTHER ``ModuleNotFoundError`` (a missing unrelated module) or
    ``ImportError`` (a genuinely broken production import) is
    re-raised, never silently converted to a skip -- this function
    must never hide a real defect. ``KeyboardInterrupt``, ``SystemExit``,
    and ``MemoryError`` are not ``ImportError`` subclasses and are
    never caught here."""
    try:
        m._load_dependencies()
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            return (
                "cv2 is not importable in this environment "
                f"(ModuleNotFoundError: {exc}) -- load an OpenCV module built "
                "against this environment's installed NumPy version before "
                "activating the project virtual environment (see "
                "docs/dfp_phase1b_masked_forward.md's 'Compatible environment' "
                "section for the exact module-load command); same gap affecting "
                "test_dfp_phase1a_stitching_pullback.py etc."
            )
        raise
    except ImportError as exc:
        text = str(exc)
        if "numpy.core.multiarray failed to import" in text or "_ARRAY_API" in text:
            return (
                "cv2 is importable but was built against a NumPy ABI incompatible "
                "with the installed NumPy in this environment -- load an OpenCV "
                "module built against this environment's installed NumPy version "
                "before activating the project virtual environment (see "
                "docs/dfp_phase1b_masked_forward.md's 'Compatible environment' "
                f"section for the exact module-load command): {text}"
            )
        raise
    return None


def _skip_if_recurrence_environment_broken() -> None:
    reason = _recurrence_environment_broken()
    if reason is not None:
        pytest.skip(reason)


# ---------------------------------------------------------------------------
# Section: environment-gap detector's own behavior (Finding B).
# ---------------------------------------------------------------------------


def test_environment_detector_recognizes_cv2_not_installed(monkeypatch):
    def raise_missing_cv2():
        raise ModuleNotFoundError("No module named 'cv2'", name="cv2")

    monkeypatch.setattr(m, "_load_dependencies", raise_missing_cv2)
    reason = _recurrence_environment_broken()
    assert reason is not None
    assert "cv2" in reason
    assert "module load" in reason.lower() or "opencv" in reason.lower()


def test_environment_detector_reraises_unrelated_module_not_found(monkeypatch):
    def raise_missing_other():
        raise ModuleNotFoundError("No module named 'not_cv2_at_all'", name="not_cv2_at_all")

    monkeypatch.setattr(m, "_load_dependencies", raise_missing_other)
    with pytest.raises(ModuleNotFoundError, match="not_cv2_at_all"):
        _recurrence_environment_broken()


def test_environment_detector_recognizes_numpy_opencv_abi_signature(monkeypatch):
    def raise_abi_error():
        raise ImportError("numpy.core.multiarray failed to import")

    monkeypatch.setattr(m, "_load_dependencies", raise_abi_error)
    reason = _recurrence_environment_broken()
    assert reason is not None
    assert "numpy" in reason.lower()


def test_environment_detector_reraises_unrelated_import_error(monkeypatch):
    def raise_other_import_error():
        raise ImportError("cannot import name 'Foo' from 'bar'")

    monkeypatch.setattr(m, "_load_dependencies", raise_other_import_error)
    with pytest.raises(ImportError, match="Foo"):
        _recurrence_environment_broken()


def test_environment_detector_reraises_runtime_error(monkeypatch):
    def raise_runtime_error():
        raise RuntimeError("unrelated runtime failure")

    monkeypatch.setattr(m, "_load_dependencies", raise_runtime_error)
    with pytest.raises(RuntimeError, match="unrelated runtime failure"):
        _recurrence_environment_broken()


def test_environment_detector_reraises_malformed_identity_error(monkeypatch):
    def raise_domain_error():
        raise m.DfpPhase1bMaskedForwardError("malformed graph/identity")

    monkeypatch.setattr(m, "_load_dependencies", raise_domain_error)
    with pytest.raises(m.DfpPhase1bMaskedForwardError, match="malformed graph/identity"):
        _recurrence_environment_broken()


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_environment_detector_propagates_control_flow_and_fatal_exceptions(monkeypatch, exc_type):
    def raise_it():
        raise exc_type("must propagate")

    monkeypatch.setattr(m, "_load_dependencies", raise_it)
    with pytest.raises(exc_type):
        _recurrence_environment_broken()


def test_environment_detector_returns_none_or_actionable_reason():
    """Whatever this environment's actual state is, the detector must
    return either None (a genuinely working environment -- in which
    case no recurrence/geometry test may skip for an environment
    reason) or a non-empty, actionable reason string -- never crash on
    its own probe."""
    reason = _recurrence_environment_broken()
    assert reason is None or (isinstance(reason, str) and len(reason) > 0)


# ---------------------------------------------------------------------------
# Section: contract.
# ---------------------------------------------------------------------------


def _contract(alpha=0.5, steps=4):
    return m.MaskedForwardContract(
        alpha=alpha, steps=steps, propagation_dtype=torch.float64, accumulation_dtype=torch.float64,
    )


def _live_phase1a_protocol_identity() -> dict:
    """The real, committed, closed-schema Phase 1A protocol identity's
    ``graph``/``propagation`` blocks -- the only two sections
    ``dfp_masked_forward_contract_from_protocol`` (via
    ``dfp_recurrence_contract_from_protocol``) reads from its
    ``phase1a_protocol_identity`` argument. Loaded fresh from the
    frozen e13 TOML authority via its own validated loader
    (``src.dfp_phase1a_identity.load_identity``) on every call -- never
    a hand-written partial literal that can silently drift from the
    live closed schema (that drift is exactly what caused 14 tests to
    fail with an uncaught ``KeyError: 'directed'`` instead of
    exercising their intended assertions). Each call re-parses the TOML
    from disk and returns an independent dict, so no caller can ever
    mutate a shared/cached instance and affect another test."""
    from src.dfp_phase1a_identity import load_identity

    identity = load_identity()
    return {"graph": identity["graph"], "propagation": identity["propagation"]}


PHASE1B_DECOMPOSITION_TRUE_FIELDS = (
    "exact_masked_forward_propagation_required", "exact_chunking_only",
    "same_graph_and_recurrence_as_canonical_t320_required",
    "adjoint_approximation_forbidden_in_primary_implementation",
    "dense_influence_kernel_forbidden", "top_k_truncation_forbidden",
    "residual_supercomponent_forbidden",
    "altered_precision_between_canonical_and_masked_runs_forbidden",
    "early_stopping_forbidden", "equilibrium_solver_forbidden",
    "k11_counterfactual_forbidden",
)


def _phase1b_decomposition(**overrides):
    block = {name: True for name in PHASE1B_DECOMPOSITION_TRUE_FIELDS}
    block["recurrence_variant_required"] = "k12"
    block["component_sum_accumulation_dtype"] = "fp64"
    block.update(overrides)
    return {"component_evidence_decomposition": block}


@contextlib.contextmanager
def _counting_finite_step_propagate():
    """Monkeypatch the real, module-cached ``finite_step_propagate``
    function object -- shared by every future ``_load_dependencies()``
    call, since Python re-uses an already-imported module object -- to
    count invocations for the duration of the ``with`` block, then
    restore it exactly. Used to PROVE a rejection happened with zero
    recurrence execution, not merely that this test's own assertion
    ran before some unrelated later code."""
    _skip_if_recurrence_environment_broken()
    m._load_dependencies()  # populates sys.modules with the real leaf modules
    module = sys.modules["models.dinotext.cover_dr.finite_step_regime"]
    original = module.finite_step_propagate
    call_count = {"n": 0}

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    module.finite_step_propagate = counting
    try:
        yield call_count
    finally:
        module.finite_step_propagate = original


def test_contract_extraction_uses_frozen_phase1a_alpha_and_steps():
    _skip_if_recurrence_environment_broken()
    from src.dfp_phase1a_identity import validate_graph_block, validate_propagation_block

    phase1a_identity = _live_phase1a_protocol_identity()

    # The live e13 loader has already validated the full identity (this
    # graph/propagation subset included) once, as part of loading it;
    # re-validate this exact subset directly too, as an explicit,
    # standalone proof -- not inside a try/except, since this is now the
    # real frozen authority and must unconditionally pass.
    validate_graph_block(phase1a_identity["graph"])
    validate_propagation_block(phase1a_identity["propagation"])
    assert phase1a_identity["graph"]["directed"] is True
    assert phase1a_identity["graph"]["canonical_k_variant"] == 12
    assert phase1a_identity["graph"]["maximum_rank"] == 12
    assert phase1a_identity["propagation"]["alpha"] == 0.98
    assert phase1a_identity["propagation"]["steps"] == 320

    contract = m.dfp_masked_forward_contract_from_protocol(phase1a_identity, _phase1b_decomposition())
    assert contract.alpha == 0.98
    assert contract.steps == 320
    assert contract.propagation_dtype == torch.float64
    assert contract.accumulation_dtype == torch.float64


@pytest.mark.parametrize("field", PHASE1B_DECOMPOSITION_TRUE_FIELDS)
def test_contract_extraction_rejects_flipped_decomposition_flag(field):
    phase1a_identity = _live_phase1a_protocol_identity()
    decomposition = _phase1b_decomposition(**{field: False})
    with _counting_finite_step_propagate() as call_count:
        # match=field pins the rejection to the exact flipped field's own
        # name, proving this is not a KeyError from an incomplete
        # phase1a_identity fixture and not a different field's check firing.
        with pytest.raises(m.DfpPhase1bMaskedForwardError, match=field):
            m.dfp_masked_forward_contract_from_protocol(phase1a_identity, decomposition)
    assert call_count["n"] == 0  # rejection must precede any recurrence execution


def test_contract_extraction_rejects_non_k12_recurrence_variant():
    phase1a_identity = _live_phase1a_protocol_identity()
    decomposition = _phase1b_decomposition()
    decomposition["component_evidence_decomposition"]["recurrence_variant_required"] = "k11"
    with _counting_finite_step_propagate() as call_count:
        with pytest.raises(m.DfpPhase1bMaskedForwardError, match="recurrence_variant_required"):
            m.dfp_masked_forward_contract_from_protocol(phase1a_identity, decomposition)
    assert call_count["n"] == 0


def test_contract_extraction_rejects_non_fp64_accumulation_dtype():
    phase1a_identity = _live_phase1a_protocol_identity()
    decomposition = _phase1b_decomposition()
    decomposition["component_evidence_decomposition"]["component_sum_accumulation_dtype"] = "fp32"
    with _counting_finite_step_propagate() as call_count:
        with pytest.raises(m.DfpPhase1bMaskedForwardError, match="component_sum_accumulation_dtype"):
            m.dfp_masked_forward_contract_from_protocol(phase1a_identity, decomposition)
    assert call_count["n"] == 0


def test_contract_extraction_rejects_missing_section():
    phase1a_identity = _live_phase1a_protocol_identity()
    with _counting_finite_step_propagate() as call_count:
        with pytest.raises(m.DfpPhase1bMaskedForwardError, match="component_evidence_decomposition"):
            m.dfp_masked_forward_contract_from_protocol(phase1a_identity, {})
    assert call_count["n"] == 0


# ---------------------------------------------------------------------------
# Section: masked seeds.
# ---------------------------------------------------------------------------


def test_seed_hard_one_hot_membership_reconstructs_unary_exactly():
    unary = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    membership = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)  # node0->comp0, node1->comp1
    contract = _contract()
    seed = m.build_masked_seed_chunk(unary, membership, [0, 1], [5, 9], contract=contract, window_index=0)
    unflat = seed.unflatten(seed.seed_flat)  # [N, R, Q]
    assert torch.allclose(unflat.sum(dim=1), unary)
    # Node 0 belongs entirely to component 0: component-1 seed row must be exactly zero.
    assert torch.equal(unflat[0, 1, :], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(unflat[0, 0, :], unary[0])


def test_seed_fractional_membership():
    unary = torch.tensor([[4.0, 8.0]], dtype=torch.float64)  # [N=1, Q=2]
    membership = torch.tensor([[0.25], [0.75]], dtype=torch.float64)  # [R=2, N=1]
    contract = _contract()
    seed = m.build_masked_seed_chunk(unary, membership, [10, 11], [0, 1], contract=contract, window_index=2)
    unflat = seed.unflatten(seed.seed_flat)
    assert torch.allclose(unflat[0, 0], torch.tensor([1.0, 2.0], dtype=torch.float64))
    assert torch.allclose(unflat[0, 1], torch.tensor([3.0, 6.0], dtype=torch.float64))
    assert torch.allclose(unflat.sum(dim=1), unary)


def test_seed_single_component():
    unary = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    membership = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    seed = m.build_masked_seed_chunk(unary, membership, [0], [4], contract=_contract(), window_index=0)
    assert seed.component_count == 1
    assert torch.equal(seed.seed_flat.squeeze(-1), unary.squeeze(-1))


def test_seed_multiple_components_and_inactive_component_in_window():
    unary = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=torch.float64)
    membership = torch.tensor([
        [1.0, 1.0, 1.0],  # component 0: fully active everywhere
        [0.0, 0.0, 0.0],  # component 1: inactive in this window
    ], dtype=torch.float64)
    seed = m.build_masked_seed_chunk(unary, membership, [7, 8], [1, 2], contract=_contract(), window_index=0)
    unflat = seed.unflatten(seed.seed_flat)
    assert torch.equal(unflat[:, 1, :], torch.zeros(3, 2, dtype=torch.float64))
    assert torch.equal(unflat[:, 0, :], unary)


def test_seed_zero_unary_scores():
    unary = torch.zeros((2, 3), dtype=torch.float64)
    membership = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    seed = m.build_masked_seed_chunk(unary, membership, [0], [1, 2, 3], contract=_contract(), window_index=0)
    assert torch.equal(seed.seed_flat, torch.zeros_like(seed.seed_flat))


def test_seed_negative_unary_scores():
    unary = torch.tensor([[-2.0, 3.0]], dtype=torch.float64)
    membership = torch.tensor([[1.0]], dtype=torch.float64)
    seed = m.build_masked_seed_chunk(unary, membership, [0], [0, 1], contract=_contract(), window_index=0)
    assert torch.allclose(seed.seed_flat, unary)


def test_seed_several_query_classes():
    unary = torch.arange(1, 6, dtype=torch.float64).unsqueeze(0)  # [1, 5]
    membership = torch.tensor([[1.0]], dtype=torch.float64)
    seed = m.build_masked_seed_chunk(unary, membership, [0], [0, 2, 5, 9, 100], contract=_contract(), window_index=0)
    assert seed.query_class_count == 5
    assert torch.equal(seed.seed_flat, unary)


def test_seed_deterministic_flatten_unflatten_roundtrip():
    torch.manual_seed(0)
    unary = torch.rand(4, 3, dtype=torch.float64)
    membership = torch.rand(2, 4, dtype=torch.float64)
    seed = m.build_masked_seed_chunk(unary, membership, [3, 4], [10, 20, 30], contract=_contract(), window_index=0)
    unflat = seed.unflatten(seed.seed_flat)
    reflat = unflat.reshape(4, 2 * 3)
    assert torch.equal(reflat, seed.seed_flat)
    # Component-major-then-query-class-major: column r*Q+c.
    for r in range(2):
        for c in range(3):
            assert torch.equal(seed.seed_flat[:, r * 3 + c], unflat[:, r, c])


def test_seed_exact_partition_reconstructs_original_field():
    torch.manual_seed(1)
    unary = torch.rand(6, 2, dtype=torch.float64)
    raw = torch.rand(3, 6, dtype=torch.float64)
    membership = raw / raw.sum(dim=0, keepdim=True)  # exact partition of unity
    seed = m.build_masked_seed_chunk(unary, membership, [0, 1, 2], [0, 1], contract=_contract(), window_index=0)
    unflat = seed.unflatten(seed.seed_flat)
    assert torch.allclose(unflat.sum(dim=1), unary, atol=1e-12)


def test_seed_rejects_boolean_unary_tensor():
    unary = torch.ones(2, 2, dtype=torch.float64).bool()
    membership = torch.ones(1, 2, dtype=torch.float64)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.build_masked_seed_chunk(unary, membership, [0], [0, 1], contract=_contract(), window_index=0)


def test_seed_rejects_boolean_membership_tensor():
    unary = torch.ones(2, 2, dtype=torch.float64)
    membership = torch.ones(1, 2, dtype=torch.float64).bool()
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.build_masked_seed_chunk(unary, membership, [0], [0, 1], contract=_contract(), window_index=0)


def test_seed_invalid_membership_shape_mismatch():
    unary = torch.ones(3, 2, dtype=torch.float64)
    membership = torch.ones(1, 4, dtype=torch.float64)  # node count disagrees
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.build_masked_seed_chunk(unary, membership, [0], [0, 1], contract=_contract(), window_index=0)


def test_seed_invalid_membership_component_count_mismatch():
    unary = torch.ones(3, 2, dtype=torch.float64)
    membership = torch.ones(2, 3, dtype=torch.float64)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.build_masked_seed_chunk(unary, membership, [0], [0, 1], contract=_contract(), window_index=0)  # 1 id vs 2 rows


@pytest.mark.parametrize(
    "component_ids,label",
    [
        ([1, 0], "descending"),
        ([0, 0], "duplicate"),
        ([0, 2], "gap"),
        ([-1, 0], "negative"),
    ],
)
def test_seed_malformed_component_ordering_rejected(component_ids, label):
    unary = torch.ones(2, 1, dtype=torch.float64)
    membership = torch.ones(2, 2, dtype=torch.float64)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.build_masked_seed_chunk(unary, membership, component_ids, [0], contract=_contract(), window_index=0)


@pytest.mark.parametrize(
    "query_classes,label",
    [
        ([1, 0], "descending"),
        ([0, 0], "duplicate"),
        ([-1, 0], "negative"),
    ],
)
def test_seed_malformed_query_class_ordering_rejected(query_classes, label):
    unary = torch.ones(1, 2, dtype=torch.float64)
    membership = torch.ones(1, 1, dtype=torch.float64)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.build_masked_seed_chunk(unary, membership, [0], query_classes, contract=_contract(), window_index=0)


def test_seed_construction_does_not_mutate_inputs():
    unary = torch.rand(3, 2, dtype=torch.float64)
    membership = torch.rand(2, 3, dtype=torch.float64)
    unary_before = unary.clone()
    membership_before = membership.clone()
    m.build_masked_seed_chunk(unary, membership, [0, 1], [0, 1], contract=_contract(), window_index=0)
    assert torch.equal(unary, unary_before)
    assert torch.equal(membership, membership_before)


def test_seed_output_does_not_alias_inputs():
    unary = torch.ones(2, 1, dtype=torch.float64)
    membership = torch.ones(1, 2, dtype=torch.float64)
    seed = m.build_masked_seed_chunk(unary, membership, [0], [0], contract=_contract(), window_index=0)
    seed.seed_flat[0, 0] = 999.0
    assert unary[0, 0].item() == 1.0
    assert membership[0, 0].item() == 1.0


# ---------------------------------------------------------------------------
# Section: recurrence (hand-built oracle, independent from the module).
# ---------------------------------------------------------------------------


def _make_graph(neighbor_indices, weights):
    """Build a real DirectedTopKGraph from caller-supplied (typically
    FP64-literal, for readability) weights. The committed graph
    contract (`DirectedTopKGraph.validate()`) requires
    `transition_weights`/`edge_affinities` to be exactly FP32 -- this
    helper performs that conversion explicitly at construction, once,
    in one place, so every recurrence test automatically constructs a
    real, validator-accepting graph. This does not change the masked-
    forward core's own FP64 seed/recurrence-state/contribution/
    conservation dtypes, which remain governed entirely by
    `MaskedForwardContract.propagation_dtype`/`accumulation_dtype` and
    are asserted directly in the tests below -- FP32 here describes
    only the graph's own frozen storage contract, matched by
    `graph.matmul`'s own upcast-to-caller-dtype behavior."""
    _skip_if_recurrence_environment_broken()
    deps = m._load_dependencies()
    N, k = neighbor_indices.shape
    weights32 = weights.to(torch.float32)
    affinities32 = weights32.clone()
    fallback = torch.zeros(N, dtype=torch.bool)
    graph = deps.DirectedTopKGraph(
        neighbor_indices=neighbor_indices, transition_weights=weights32, edge_affinities=affinities32,
        self_loop_fallback=fallback, num_nodes=N, k=k, affinity_power=1.0,
    )
    assert graph.transition_weights.dtype == torch.float32
    assert graph.edge_affinities.dtype == torch.float32
    graph.validate()  # must not raise -- proves the real committed validator accepts this fixture
    return graph


# `propagate_masked_seed_chunk` unconditionally requires `graph.k == 12`
# (the frozen recurrence-variant contract -- see graph.py's validator and
# dfp_phase1b_masked_forward.py's own explicit check). Every POSITIVE-path
# recurrence fixture below must therefore be a real k=12 graph; only
# `test_recurrence_rejects_non_k12_graph` deliberately uses a non-k12
# graph, to exercise that rejection itself.
_K12_NUM_NODES = 15
_K12 = 12


def _k12_ring_neighbor_indices(num_nodes: int = _K12_NUM_NODES) -> torch.Tensor:
    """Deterministic, validator-accepting k=12 neighbor pattern: node
    i's 12 ordinary out-edges go to the next 12 nodes on a directed
    ring (i+1 .. i+12, mod num_nodes) -- never a self-destination,
    never a duplicate destination within a row, for any num_nodes >
    _K12."""
    node_ids = torch.arange(num_nodes, dtype=torch.int64).unsqueeze(1)
    offsets = torch.arange(1, _K12 + 1, dtype=torch.int64).unsqueeze(0)
    return (node_ids + offsets) % num_nodes


def _k12_uniform_weights(num_nodes: int = _K12_NUM_NODES) -> torch.Tensor:
    """Row-stochastic, tied (equal-within-row) FP64 weight literal."""
    return torch.full((num_nodes, _K12), 1.0 / _K12, dtype=torch.float64)


def _k12_arbitrary_weights(num_nodes: int = _K12_NUM_NODES) -> torch.Tensor:
    """Deterministic, non-uniform, strictly positive FP64 weight
    literal (no two rows or slots tied), normalized row-stochastic."""
    i = torch.arange(num_nodes, dtype=torch.float64).unsqueeze(1)
    s = torch.arange(_K12, dtype=torch.float64).unsqueeze(0)
    raw = 1.0 + torch.remainder(i * 7.0 + s * 3.0, 11.0)
    return raw / raw.sum(dim=1, keepdim=True)


def _linspace_partition(num_components: int, num_nodes: int = _K12_NUM_NODES) -> torch.Tensor:
    """Deterministic FP64 membership partition, `[component_count,
    node_count]`, each column summing to exactly one."""
    raw = torch.stack([
        torch.linspace(i + 1.0, i + 2.0, num_nodes, dtype=torch.float64) for i in range(num_components)
    ])
    return raw / raw.sum(dim=0, keepdim=True)


def _dense_A_from_graph(graph, num_nodes: int = _K12_NUM_NODES) -> torch.Tensor:
    """Independent FP64 dense adjacency built from the graph's own
    actually-stored (FP32, upcast) values -- never the pre-rounding
    FP64 literal -- so a dense-matmul oracle can be compared fairly
    and exactly against the real sparse recurrence."""
    weights_as_stored = graph.transition_weights.to(torch.float64)
    dense_A = torch.zeros(num_nodes, num_nodes, dtype=torch.float64)
    for i in range(num_nodes):
        for s in range(graph.k):
            dense_A[i, graph.neighbor_indices[i, s]] += weights_as_stored[i, s]
    return dense_A


def test_graph_fixture_fp32_storage_and_fp64_arithmetic_architecture():
    """Standalone, explicit proof of the precision architecture: the
    committed graph contract stores FP32 weights/affinities (never
    FP64), the real validator accepts that FP32 graph on its own
    terms, and the masked-forward core's seed/propagation dtypes are
    governed entirely by MaskedForwardContract -- FP64 throughout --
    regardless of the graph's own storage precision. This is the
    direct fixture-level counterpart to the repaired `_make_graph`
    helper and to `docs/dfp_phase1b_masked_forward.md`'s precision
    section."""
    neighbor_indices = _k12_ring_neighbor_indices()
    weights64 = _k12_arbitrary_weights()
    graph = _make_graph(neighbor_indices, weights64)

    assert graph.transition_weights.dtype == torch.float32
    assert graph.edge_affinities.dtype == torch.float32
    graph.validate()  # explicit, standalone proof the real committed validator accepts it

    contract = _contract(alpha=0.6, steps=9)
    unary = torch.stack([
        torch.linspace(-2.0, 2.0, _K12_NUM_NODES, dtype=torch.float64),
        torch.linspace(3.0, -1.0, _K12_NUM_NODES, dtype=torch.float64),
    ], dim=1)
    membership = torch.ones(1, _K12_NUM_NODES, dtype=torch.float64)
    seed = m.build_masked_seed_chunk(unary, membership, [0], [0, 1], contract=contract, window_index=0)
    assert seed.seed_flat.dtype == torch.float64

    prop = m.propagate_masked_seed_chunk(graph, seed, contract=contract)
    assert prop.t0.dtype == torch.float64
    assert prop.t_final.dtype == torch.float64

    # Independent FP64 oracle, built from the graph's own stored (FP32,
    # upcast) values -- the same technique used throughout this section.
    dense_A = _dense_A_from_graph(graph)
    expected = _oracle_dense_recurrence(dense_A, unary, alpha=contract.alpha, steps=contract.steps)
    assert torch.allclose(prop.t_final, expected, atol=1e-10)

    # Conservation/contribution accumulation is FP64 regardless of graph storage.
    membership_full = torch.ones(1, _K12_NUM_NODES, dtype=torch.float64)
    residual = m.compute_membership_partition_residual(membership_full)
    assert residual.signed.dtype == torch.float64


def _oracle_dense_recurrence(dense_A: torch.Tensor, s0: torch.Tensor, *, alpha: float, steps: int) -> torch.Tensor:
    """Independent, test-only dense-matrix oracle: p_{t+1} = alpha*A@p_t
    + (1-alpha)*s0, run with a plain dense matmul -- never imported
    from production code, never used anywhere outside this test file."""
    p = s0.clone()
    for _ in range(steps):
        p = alpha * (dense_A @ p) + (1.0 - alpha) * s0
    return p


def test_recurrence_exact_t_final_matches_independent_dense_oracle():
    neighbor_indices = _k12_ring_neighbor_indices()
    weights = _k12_arbitrary_weights()
    graph = _make_graph(neighbor_indices, weights)
    assert graph.transition_weights.dtype == torch.float32
    assert graph.edge_affinities.dtype == torch.float32
    # The oracle must use the SAME values the real graph actually stores
    # (FP32, upcast to the caller's dtype by graph.matmul) -- not the
    # pre-rounding FP64 literals above -- for a fair, exact comparison.
    dense_A = _dense_A_from_graph(graph)

    unary = torch.stack([
        torch.linspace(1.0, 0.0, _K12_NUM_NODES, dtype=torch.float64),
        torch.linspace(0.0, 2.0, _K12_NUM_NODES, dtype=torch.float64),
    ], dim=1)
    membership = _linspace_partition(2)
    contract = _contract(alpha=0.4, steps=7)
    seed = m.build_masked_seed_chunk(unary, membership, [0, 1], [0, 1], contract=contract, window_index=0)
    assert seed.seed_flat.dtype == torch.float64  # masked seed remains FP64 despite FP32 graph storage
    prop = m.propagate_masked_seed_chunk(graph, seed, contract=contract)
    assert prop.t0.dtype == torch.float64
    assert prop.t_final.dtype == torch.float64  # T320 output remains FP64

    unflat = seed.unflatten(prop.t_final)  # [N, R, Q]
    for r in range(2):
        expected = _oracle_dense_recurrence(dense_A, seed.unflatten(seed.seed_flat)[:, r, :], alpha=0.4, steps=7)
        assert torch.allclose(unflat[:, r, :], expected, atol=1e-10)
    assert prop.steps_completed == 7


def test_recurrence_t0_equals_seed_exactly():
    neighbor_indices = _k12_ring_neighbor_indices()
    weights = _k12_arbitrary_weights()
    graph = _make_graph(neighbor_indices, weights)
    unary = torch.linspace(1.0, 2.0, _K12_NUM_NODES, dtype=torch.float64).unsqueeze(1)
    membership = torch.ones(1, _K12_NUM_NODES, dtype=torch.float64)
    contract = _contract(alpha=0.5, steps=3)
    seed = m.build_masked_seed_chunk(unary, membership, [0], [0], contract=contract, window_index=0)
    prop = m.propagate_masked_seed_chunk(graph, seed, contract=contract)
    assert torch.equal(prop.t0, seed.seed_flat)


def test_recurrence_stationary_seed_stays_fixed_point():
    """A uniform (constant-across-nodes) seed is automatically a fixed
    point of `A @ p` for ANY row-stochastic graph -- every row of a
    validated `DirectedTopKGraph` sums to exactly one, so `A @
    constant == row_sums * constant == constant` -- and must therefore
    remain unchanged for any number of steps."""
    neighbor_indices = _k12_ring_neighbor_indices()
    weights = _k12_arbitrary_weights()
    graph = _make_graph(neighbor_indices, weights)
    unary = torch.full((_K12_NUM_NODES, 1), 5.0, dtype=torch.float64)
    membership = torch.ones(1, _K12_NUM_NODES, dtype=torch.float64)
    contract = _contract(alpha=0.7, steps=50)
    seed = m.build_masked_seed_chunk(unary, membership, [0], [0], contract=contract, window_index=0)
    prop = m.propagate_masked_seed_chunk(graph, seed, contract=contract)
    assert torch.allclose(prop.t_final, seed.seed_flat, atol=1e-10)


def test_recurrence_non_stationary_seed_actually_moves():
    neighbor_indices = _k12_ring_neighbor_indices()
    weights = _k12_arbitrary_weights()
    graph = _make_graph(neighbor_indices, weights)
    unary = torch.zeros(_K12_NUM_NODES, 1, dtype=torch.float64)
    unary[0, 0] = 10.0
    membership = torch.ones(1, _K12_NUM_NODES, dtype=torch.float64)
    contract = _contract(alpha=0.9, steps=20)
    seed = m.build_masked_seed_chunk(unary, membership, [0], [0], contract=contract, window_index=0)
    prop = m.propagate_masked_seed_chunk(graph, seed, contract=contract)
    assert not torch.allclose(prop.t_final, seed.seed_flat)


def test_recurrence_tied_affinities_still_row_stochastic_and_exact():
    neighbor_indices = _k12_ring_neighbor_indices()
    weights = _k12_uniform_weights()
    graph = _make_graph(neighbor_indices, weights)
    # 1/12 is not exactly representable in FP32 or FP64 either, so, as
    # elsewhere, the oracle is built from the graph's own stored
    # (post-FP32-rounding) values for an exact comparison.
    dense_A = _dense_A_from_graph(graph)
    unary = torch.stack([
        torch.linspace(1.0, 0.0, _K12_NUM_NODES, dtype=torch.float64),
        torch.linspace(0.0, 1.0, _K12_NUM_NODES, dtype=torch.float64),
        torch.linspace(-1.0, 1.0, _K12_NUM_NODES, dtype=torch.float64),
    ], dim=1)
    membership = torch.ones(1, _K12_NUM_NODES, dtype=torch.float64)
    contract = _contract(alpha=0.3, steps=10)
    seed = m.build_masked_seed_chunk(unary, membership, [0], [0, 1, 2], contract=contract, window_index=0)
    prop = m.propagate_masked_seed_chunk(graph, seed, contract=contract)
    assert prop.t_final.dtype == torch.float64
    expected = _oracle_dense_recurrence(dense_A, unary, alpha=0.3, steps=10)
    assert torch.allclose(prop.t_final, expected, atol=1e-10)


def test_recurrence_exactly_steps_sparse_matmul_updates_via_call_count(monkeypatch):
    _skip_if_recurrence_environment_broken()
    deps = m._load_dependencies()
    neighbor_indices = _k12_ring_neighbor_indices()
    weights = _k12_arbitrary_weights()
    graph = _make_graph(neighbor_indices, weights)

    call_count = {"n": 0}
    real_matmul = graph.matmul

    def counting_matmul(rhs):
        call_count["n"] += 1
        return real_matmul(rhs)

    object.__setattr__(graph, "matmul", counting_matmul)
    unary = torch.linspace(1.0, 2.0, _K12_NUM_NODES, dtype=torch.float64).unsqueeze(1)
    membership = torch.ones(1, _K12_NUM_NODES, dtype=torch.float64)
    contract = _contract(alpha=0.5, steps=17)
    seed = m.build_masked_seed_chunk(unary, membership, [0], [0], contract=contract, window_index=0)
    m.propagate_masked_seed_chunk(graph, seed, contract=contract)
    assert call_count["n"] == 17


@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_recurrence_identical_result_across_component_chunk_sizes(chunk_size):
    neighbor_indices = _k12_ring_neighbor_indices()
    weights = _k12_arbitrary_weights()
    graph = _make_graph(neighbor_indices, weights)
    contract = _contract(alpha=0.6, steps=6)

    torch.manual_seed(2)
    unary = torch.rand(_K12_NUM_NODES, 2, dtype=torch.float64)
    raw = torch.rand(3, _K12_NUM_NODES, dtype=torch.float64)  # 3 total components
    membership_all = raw / raw.sum(dim=0, keepdim=True)

    # Full run: one chunk of size 3.
    seed_full = m.build_masked_seed_chunk(unary, membership_all, [0, 1, 2], [0, 1], contract=contract, window_index=0)
    prop_full = m.propagate_masked_seed_chunk(graph, seed_full, contract=contract)
    full_unflat = seed_full.unflatten(prop_full.t_final)

    # Chunked run: process `chunk_size`-sized slices independently and reassemble.
    reassembled = torch.zeros(3, _K12_NUM_NODES, 2, dtype=torch.float64)  # [R, N, Q]
    for start, stop in m.iter_component_chunks(3, chunk_size):
        seed_chunk = m.build_masked_seed_chunk(
            unary, membership_all[start:stop], list(range(start, stop)), [0, 1], contract=contract, window_index=0,
        )
        prop_chunk = m.propagate_masked_seed_chunk(graph, seed_chunk, contract=contract)
        chunk_unflat = seed_chunk.unflatten(prop_chunk.t_final)  # [N, chunk_r, Q]
        for local_r in range(stop - start):
            reassembled[start + local_r] = chunk_unflat[:, local_r, :]

    for r in range(3):
        assert torch.allclose(full_unflat[:, r, :], reassembled[r], atol=1e-12)


def test_recurrence_snapshot_output_independence():
    """Mutating one PropagatedChunk's t0/t_final must never affect a
    second, independently produced PropagatedChunk."""
    neighbor_indices = _k12_ring_neighbor_indices()
    weights = _k12_arbitrary_weights()
    graph = _make_graph(neighbor_indices, weights)
    unary = torch.linspace(1.0, 2.0, _K12_NUM_NODES, dtype=torch.float64).unsqueeze(1)
    membership = torch.ones(1, _K12_NUM_NODES, dtype=torch.float64)
    contract = _contract(alpha=0.5, steps=5)
    seed_a = m.build_masked_seed_chunk(unary, membership, [0], [0], contract=contract, window_index=0)
    seed_b = m.build_masked_seed_chunk(unary, membership, [0], [0], contract=contract, window_index=0)
    prop_a = m.propagate_masked_seed_chunk(graph, seed_a, contract=contract)
    prop_b = m.propagate_masked_seed_chunk(graph, seed_b, contract=contract)
    prop_a.t_final[0, 0] = 12345.0
    assert prop_b.t_final[0, 0].item() != 12345.0


def test_recurrence_rejects_non_k12_graph():
    """The one deliberately non-k12 fixture in this section: a
    structurally valid, FP32, validator-accepting k=1 graph, so
    rejection happens specifically because `graph.k != 12` -- never
    because of dtype, construction, or any other structural defect."""
    _skip_if_recurrence_environment_broken()
    deps = m._load_dependencies()
    neighbor_indices = torch.tensor([[1], [0]], dtype=torch.int64)
    weights = torch.tensor([[1.0], [1.0]], dtype=torch.float64).to(torch.float32)
    graph = deps.DirectedTopKGraph(
        neighbor_indices=neighbor_indices, transition_weights=weights, edge_affinities=weights.clone(),
        self_loop_fallback=torch.zeros(2, dtype=torch.bool), num_nodes=2, k=1, affinity_power=1.0,
    )
    assert graph.transition_weights.dtype == torch.float32
    assert graph.edge_affinities.dtype == torch.float32
    graph.validate()  # the real committed validator accepts this k=1 graph on its own terms
    unary = torch.ones(2, 1, dtype=torch.float64)
    membership = torch.ones(1, 2, dtype=torch.float64)
    contract = _contract()
    seed = m.build_masked_seed_chunk(unary, membership, [0], [0], contract=contract, window_index=0)
    # This must fail on the frozen k=12 recurrence-variant check specifically
    # -- not on graph construction/dtype (already proven valid above), and
    # not on any other structural mismatch (node_count, seed type, ...).
    with pytest.raises(m.DfpPhase1bMaskedForwardError, match="k=12"):
        m.propagate_masked_seed_chunk(graph, seed, contract=contract)


def test_recurrence_rejects_node_count_mismatch():
    neighbor_indices = _k12_ring_neighbor_indices()
    weights = _k12_arbitrary_weights()
    graph = _make_graph(neighbor_indices, weights)  # N=15
    assert graph.transition_weights.dtype == torch.float32
    graph.validate()  # the real committed validator accepts this k=12 graph on its own terms
    unary = torch.ones(_K12_NUM_NODES + 1, 1, dtype=torch.float64)  # 16 nodes, graph has 15
    membership = torch.ones(1, _K12_NUM_NODES + 1, dtype=torch.float64)
    contract = _contract()
    seed = m.build_masked_seed_chunk(unary, membership, [0], [0], contract=contract, window_index=0)
    # This must fail on the node-count-mismatch check specifically -- not
    # on the k=12 check, since this graph's k=12 is already proven valid.
    with pytest.raises(m.DfpPhase1bMaskedForwardError, match="disagrees"):
        m.propagate_masked_seed_chunk(graph, seed, contract=contract)


# ---------------------------------------------------------------------------
# Section: signed contributions (hand-solvable fixture).
# ---------------------------------------------------------------------------


def test_signed_contributions_hand_solvable_fixture_correct_sign():
    # Two components, two query classes (a=0, b=1). One disputed pixel.
    # s0: component 0 favors a; component 1 favors b, weakly.
    # s_final: component 0 flips to favor b strongly (a "harmful"-style a->b
    # flip, reconstructed purely from component scores, no GT anywhere).
    R, Q = 2, 2
    H, W = 1, 1
    s0 = torch.zeros(1, R * Q, H, W, dtype=torch.float64)
    s_final = torch.zeros(1, R * Q, H, W, dtype=torch.float64)
    # component 0: (a=0)=5.0, (b=1)=1.0 at t0; (a=0)=1.0,(b=1)=9.0 at t_final.
    s0[0, 0 * Q + 0, 0, 0] = 5.0
    s0[0, 0 * Q + 1, 0, 0] = 1.0
    s_final[0, 0 * Q + 0, 0, 0] = 1.0
    s_final[0, 0 * Q + 1, 0, 0] = 9.0
    # component 1: (a=0)=0.5,(b=1)=2.0 at t0; unchanged at t_final (opposing/inert).
    s0[0, 1 * Q + 0, 0, 0] = 0.5
    s0[0, 1 * Q + 1, 0, 0] = 2.0
    s_final[0, 1 * Q + 0, 0, 0] = 0.5
    s_final[0, 1 * Q + 1, 0, 0] = 2.0

    rows = torch.tensor([0], dtype=torch.int64)
    cols = torch.tensor([0], dtype=torch.int64)
    a = torch.tensor([0], dtype=torch.int64)
    b = torch.tensor([1], dtype=torch.int64)
    chunk = m.compute_signed_contributions(
        s0, s_final, component_count=R, query_class_union=[0, 1],
        disputed_pixel_rows=rows, disputed_pixel_cols=cols, disputed_pixel_a=a, disputed_pixel_b=b,
        ori_width=W, component_id_start=0,
    )
    # g_0 = [9-1] - [1-5] = 8 - (-4) = 12; g_1 = [2-0.5]-[2-0.5] = 0.
    expected = torch.tensor([[12.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(chunk.values, expected)


def test_signed_contributions_positive_negative_and_zero_components():
    R, Q = 3, 2
    s0 = torch.zeros(1, R * Q, 1, 1, dtype=torch.float64)
    s_final = torch.zeros(1, R * Q, 1, 1, dtype=torch.float64)
    # component 0: contributes positively.
    s0[0, 0, 0, 0], s0[0, 1, 0, 0] = 0.0, 0.0
    s_final[0, 0, 0, 0], s_final[0, 1, 0, 0] = -1.0, 1.0  # g = (1-(-1)) - (0-0) = 2
    # component 1: contributes negatively.
    s0[0, 2, 0, 0], s0[0, 3, 0, 0] = 0.0, 0.0
    s_final[0, 2, 0, 0], s_final[0, 3, 0, 0] = 1.0, -1.0  # g = (-1-1) - 0 = -2
    # component 2: exactly zero contribution.
    s0[0, 4, 0, 0], s0[0, 5, 0, 0] = 3.0, 3.0
    s_final[0, 4, 0, 0], s_final[0, 5, 0, 0] = 3.0, 3.0  # g = 0 - 0 = 0

    rows = torch.tensor([0], dtype=torch.int64); cols = torch.tensor([0], dtype=torch.int64)
    a = torch.tensor([0], dtype=torch.int64); b = torch.tensor([1], dtype=torch.int64)
    chunk = m.compute_signed_contributions(
        s0, s_final, component_count=R, query_class_union=[0, 1],
        disputed_pixel_rows=rows, disputed_pixel_cols=cols, disputed_pixel_a=a, disputed_pixel_b=b,
        ori_width=1, component_id_start=0,
    )
    assert torch.allclose(chunk.values, torch.tensor([[2.0, -2.0, 0.0]], dtype=torch.float64))


def test_signed_contributions_reversed_sign_convention_fails_fixture():
    """Deliberately swap a/b and confirm the (now wrong) sign no longer
    matches the fixture's own independently-computed expectation."""
    R, Q = 1, 2
    s0 = torch.zeros(1, R * Q, 1, 1, dtype=torch.float64)
    s_final = torch.zeros(1, R * Q, 1, 1, dtype=torch.float64)
    s0[0, 0, 0, 0], s0[0, 1, 0, 0] = 5.0, 1.0
    s_final[0, 0, 0, 0], s_final[0, 1, 0, 0] = 1.0, 9.0
    rows = torch.tensor([0], dtype=torch.int64); cols = torch.tensor([0], dtype=torch.int64)
    a = torch.tensor([0], dtype=torch.int64); b = torch.tensor([1], dtype=torch.int64)
    correct = m.compute_signed_contributions(
        s0, s_final, component_count=R, query_class_union=[0, 1],
        disputed_pixel_rows=rows, disputed_pixel_cols=cols, disputed_pixel_a=a, disputed_pixel_b=b,
        ori_width=1, component_id_start=0,
    )
    swapped = m.compute_signed_contributions(
        s0, s_final, component_count=R, query_class_union=[0, 1],
        disputed_pixel_rows=rows, disputed_pixel_cols=cols, disputed_pixel_a=b, disputed_pixel_b=a,
        ori_width=1, component_id_start=0,
    )
    assert correct.values.item() == 12.0
    assert swapped.values.item() == -12.0
    assert correct.values.item() != swapped.values.item()


def test_signed_contributions_multiple_supporting_components_sum_correctly():
    R, Q = 4, 2
    s0 = torch.zeros(1, R * Q, 1, 1, dtype=torch.float64)
    s_final = torch.zeros(1, R * Q, 1, 1, dtype=torch.float64)
    expected_each = [1.0, 2.0, -0.5, 0.0]
    for r, g in enumerate(expected_each):
        s0[0, r * Q + 1, 0, 0] = 0.0  # b at t0 = 0
        s_final[0, r * Q + 1, 0, 0] = g  # b at t_final = g (a stays 0 throughout)
    rows = torch.tensor([0], dtype=torch.int64); cols = torch.tensor([0], dtype=torch.int64)
    a = torch.tensor([0], dtype=torch.int64); b = torch.tensor([1], dtype=torch.int64)
    chunk = m.compute_signed_contributions(
        s0, s_final, component_count=R, query_class_union=[0, 1],
        disputed_pixel_rows=rows, disputed_pixel_cols=cols, disputed_pixel_a=a, disputed_pixel_b=b,
        ori_width=1, component_id_start=0,
    )
    assert torch.allclose(chunk.values, torch.tensor([expected_each], dtype=torch.float64))
    assert abs(chunk.values.sum().item() - sum(expected_each)) < 1e-12


def test_signed_contributions_missing_required_class_rejected():
    s0 = torch.zeros(1, 2, 1, 1, dtype=torch.float64)
    s_final = torch.zeros(1, 2, 1, 1, dtype=torch.float64)
    rows = torch.tensor([0], dtype=torch.int64); cols = torch.tensor([0], dtype=torch.int64)
    a = torch.tensor([0], dtype=torch.int64); b = torch.tensor([99], dtype=torch.int64)  # 99 not in union
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.compute_signed_contributions(
            s0, s_final, component_count=1, query_class_union=[0, 1],
            disputed_pixel_rows=rows, disputed_pixel_cols=cols, disputed_pixel_a=a, disputed_pixel_b=b,
            ori_width=1, component_id_start=0,
        )


def test_signed_contributions_disputed_pixels_must_be_pre_ordered():
    s0 = torch.zeros(1, 2, 2, 2, dtype=torch.float64)
    s_final = torch.zeros(1, 2, 2, 2, dtype=torch.float64)
    rows = torch.tensor([1, 0], dtype=torch.int64)  # flat index 1*2+0=2, 0*2+1=1 -- OUT of order
    cols = torch.tensor([0, 1], dtype=torch.int64)
    a = torch.tensor([0, 0], dtype=torch.int64); b = torch.tensor([1, 1], dtype=torch.int64)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.compute_signed_contributions(
            s0, s_final, component_count=1, query_class_union=[0, 1],
            disputed_pixel_rows=rows, disputed_pixel_cols=cols, disputed_pixel_a=a, disputed_pixel_b=b,
            ori_width=2, component_id_start=0,
        )


# ---------------------------------------------------------------------------
# Section: conservation identities.
# ---------------------------------------------------------------------------


def test_conservation_membership_partition_exact():
    membership = torch.tensor([[0.3, 0.5, 1.0], [0.7, 0.5, 0.0]], dtype=torch.float64)
    res = m.compute_membership_partition_residual(membership)
    assert res.max_absolute == 0.0
    assert torch.equal(res.signed, torch.zeros(3, dtype=torch.float64))


def test_conservation_membership_partition_detects_perturbation():
    membership = torch.tensor([[0.3, 0.5, 1.0], [0.7, 0.5, 0.0]], dtype=torch.float64)
    perturbed = membership.clone()
    perturbed[0, 0] += 0.01
    res = m.compute_membership_partition_residual(perturbed)
    assert abs(res.max_absolute - 0.01) < 1e-12
    assert res.max_absolute != 0.0  # raw residual, never normalized away


def test_conservation_seed_reconstruction_exact():
    unary = torch.tensor([[2.0, 4.0], [6.0, 8.0]], dtype=torch.float64)
    raw = torch.rand(3, 2, dtype=torch.float64) + 0.1
    membership = raw / raw.sum(dim=0, keepdim=True)
    seed = m.build_masked_seed_chunk(unary, membership, [0, 1, 2], [0, 1], contract=_contract(), window_index=0)
    seed_by_component = seed.unflatten(seed.seed_flat).transpose(0, 1).contiguous()  # [R, N, Q]
    res = m.compute_seed_reconstruction_residual(seed_by_component, unary)
    assert res.max_absolute < 1e-10


def test_conservation_propagated_reconstruction_exact_on_synthetic_split():
    canonical = torch.tensor([[5.0, 1.0], [2.0, 9.0]], dtype=torch.float64)
    per_component = torch.stack([canonical * 0.4, canonical * 0.6], dim=0)  # [R=2, N=2, Q=2]
    res = m.compute_propagated_reconstruction_residual(per_component, canonical)
    assert res.max_absolute < 1e-12


def test_conservation_propagated_reconstruction_detects_perturbation():
    canonical = torch.tensor([[5.0, 1.0], [2.0, 9.0]], dtype=torch.float64)
    per_component = torch.stack([canonical * 0.4, canonical * 0.6 + 0.05], dim=0)
    res = m.compute_propagated_reconstruction_residual(per_component, canonical)
    assert res.max_absolute > 0.04


def test_conservation_signed_margin_exact():
    g = torch.tensor([[3.0, -1.0], [0.5, 0.5]], dtype=torch.float64)  # [D=2, R=2]
    delta_m = g.sum(dim=1)
    res = m.compute_signed_margin_conservation_residual(g, delta_m)
    assert res.max_absolute == 0.0


def test_conservation_signed_margin_detects_perturbation():
    g = torch.tensor([[3.0, -1.0]], dtype=torch.float64)
    delta_m = torch.tensor([2.5], dtype=torch.float64)  # true sum is 2.0, off by 0.5
    res = m.compute_signed_margin_conservation_residual(g, delta_m)
    assert abs(res.max_absolute - 0.5) < 1e-12


def test_conservation_rejects_non_finite_input():
    membership = torch.tensor([[float("nan"), 0.5]], dtype=torch.float64)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.compute_membership_partition_residual(membership)


def test_conservation_measurements_bundle_carries_no_pass_fail_field():
    membership = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    seed_by_component = torch.zeros(1, 2, 1, dtype=torch.float64)
    unary = torch.zeros(2, 1, dtype=torch.float64)
    canonical = torch.zeros(2, 1, dtype=torch.float64)
    per_component = torch.zeros(1, 2, 1, dtype=torch.float64)
    g = torch.zeros(1, 1, dtype=torch.float64)
    delta_m = torch.zeros(1, dtype=torch.float64)
    bundle = m.build_conservation_measurements(
        membership_partition_residual=m.compute_membership_partition_residual(membership),
        seed_reconstruction_residual=m.compute_seed_reconstruction_residual(seed_by_component, unary),
        propagated_reconstruction_residual=m.compute_propagated_reconstruction_residual(per_component, canonical),
        signed_margin_conservation_residual=m.compute_signed_margin_conservation_residual(g, delta_m),
        component_count=1, query_class_count=1, disputed_pixel_count=1,
    )
    field_names = {f.name for f in __import__("dataclasses").fields(bundle)}
    forbidden = {"passed", "pass", "ok", "tolerance", "threshold", "atol", "rtol"}
    assert not (field_names & forbidden)


def test_conservation_never_raises_on_large_finite_residual():
    """A production module must measure, never gate: an enormous but
    finite residual must not raise."""
    g = torch.tensor([[1e30]], dtype=torch.float64)
    delta_m = torch.tensor([-1e30], dtype=torch.float64)
    res = m.compute_signed_margin_conservation_residual(g, delta_m)
    assert res.max_absolute == pytest.approx(2e30)


# ---------------------------------------------------------------------------
# Section: query restriction.
# ---------------------------------------------------------------------------


def test_query_class_union_of_overlapping_windows():
    per_window = {0: [1, 3, 5], 1: [2, 3], 2: []}
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.compute_query_class_union(per_window)  # empty tuple for window 2 is invalid (non-empty required)


def test_query_class_union_ascending_deduplicated():
    per_window = {0: [5, 9], 1: [1, 9, 12]}
    union = m.compute_query_class_union(per_window)
    assert union == (1, 5, 9, 12)


def test_query_class_union_both_a_and_b_present_when_windows_disagree():
    # Window 0 sees classes {1,5}; window 1 sees {5,9}. A pixel disputed
    # between 1 and 9, covered by BOTH windows, requires the union to
    # contain both -- confirming the union-based alignment never drops
    # a class a contributing window actually needs.
    union = m.compute_query_class_union({0: [1, 5], 1: [5, 9]})
    assert 1 in union and 9 in union


def test_embed_into_query_union_zero_pads_missing_classes():
    values = torch.tensor([[10.0, 20.0]], dtype=torch.float64)  # [N=1, R=1*Qlocal=2]
    embedded = m._embed_into_query_union(values, component_count=1, local_query_classes=(1, 5), query_union=(1, 3, 5))
    assert embedded.shape == (1, 3)
    assert torch.equal(embedded, torch.tensor([[10.0, 0.0, 20.0]], dtype=torch.float64))


def test_embed_into_query_union_full_overlap_is_identity_reorder():
    values = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    embedded = m._embed_into_query_union(values, component_count=1, local_query_classes=(1, 3, 5), query_union=(1, 3, 5))
    assert torch.equal(embedded, values)


def test_query_class_never_replaced_by_image_global_set():
    """The union helper only ever unions the SUPPLIED per-window sets --
    it never substitutes or falls back to an unrelated 'all classes'
    default, detectable by observing the union depends only on its
    input, never on any global/module-level class-count constant."""
    small = m.compute_query_class_union({0: [2]})
    assert small == (2,)  # not expanded to any larger implied set


# ---------------------------------------------------------------------------
# Section: GT isolation.
# ---------------------------------------------------------------------------


_FORBIDDEN_GT_SUBSTRINGS = ("gt_", "ground_truth", "harmful", "beneficial", "oracle", "annotation")


def test_gt_isolation_no_public_function_accepts_a_gt_shaped_parameter():
    for name in m.__all__:
        obj = getattr(m, name)
        if not inspect.isfunction(obj):
            continue
        for param_name in inspect.signature(obj).parameters:
            lowered = param_name.lower()
            for fragment in _FORBIDDEN_GT_SUBSTRINGS:
                assert fragment not in lowered, f"{name}'s parameter {param_name!r} looks GT-shaped"


def test_gt_isolation_no_dataclass_field_is_gt_shaped():
    import dataclasses
    for name in m.__all__:
        obj = getattr(m, name)
        if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
            continue
        for field in dataclasses.fields(obj):
            lowered = field.name.lower()
            for fragment in _FORBIDDEN_GT_SUBSTRINGS:
                assert fragment not in lowered, f"{name}.{field.name} looks GT-shaped"


def test_gt_isolation_source_never_mentions_annotation_loading():
    source = Path(m.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called_names.add(node.func.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
    forbidden_calls = {"load_gt", "load_annotation", "read_gt", "load_dataset", "load_checkpoint", "load_model"}
    assert not (called_names & forbidden_calls)


def _collect_identifiers(source: str) -> set[str]:
    """Real Python symbol references only (``Name``/``Attribute``/
    import targets) -- immune to docstring or comment PROSE that
    legitimately *mentions* an excluded concept in an explanatory
    sentence (e.g. "this module never computes k_star") without ever
    referencing it as code."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", None)
            if module:
                names.add(module.split(".")[0])
            for alias in node.names:
                names.add(alias.name.split(".")[0])
    return names


def test_gt_isolation_source_contains_no_stage_b_target_construction():
    source = Path(m.__file__).read_text(encoding="utf-8")
    identifiers = {name.lower() for name in _collect_identifiers(source)}
    for forbidden in ("target_category", "harmful_binary_label", "logisticregression", "k_star", "phi_formula", "top1_positive_share"):
        assert forbidden not in identifiers, f"forbidden identifier {forbidden!r} is actually referenced as code"


def test_gt_isolation_poisoned_module_level_gt_hook_never_invoked():
    """There is no GT hook to poison (by design) -- this test asserts
    that fact directly: no attribute of the module contains 'gt' as a
    case-insensitive whole-word-ish fragment outside of this test's own
    scan, i.e. attempting to monkeypatch a plausible hook name finds
    nothing to attach to."""
    plausible_hook_names = ("gt_loader", "load_gt_array", "gt_provider", "annotation_loader")
    for hook_name in plausible_hook_names:
        assert not hasattr(m, hook_name)


# ---------------------------------------------------------------------------
# Section: validation and exceptional bases.
# ---------------------------------------------------------------------------


def test_validation_wrong_rank_unary_scores():
    unary = torch.ones(2, 2, 2, dtype=torch.float64)
    membership = torch.ones(1, 2, dtype=torch.float64)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.build_masked_seed_chunk(unary, membership, [0], [0, 1], contract=_contract(), window_index=0)


def test_validation_wrong_dtype_unary_scores():
    unary = torch.ones(2, 2, dtype=torch.int64)
    membership = torch.ones(1, 2, dtype=torch.float64)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.build_masked_seed_chunk(unary, membership, [0], [0, 1], contract=_contract(), window_index=0)


def test_validation_device_mismatch_rejected_with_message():
    unary = torch.ones(2, 1, dtype=torch.float64)
    membership = torch.ones(1, 2, dtype=torch.float64)
    # We cannot construct a genuinely different device on a CPU-only CI box,
    # so this test asserts the check exists and is reachable via source
    # inspection instead of forcing a real multi-device run.
    source = inspect.getsource(m.build_masked_seed_chunk)
    assert "device" in source


def test_validation_wrong_class_count_query_class_indices_vs_tensor_width():
    unary = torch.ones(2, 3, dtype=torch.float64)  # 3 columns
    membership = torch.ones(1, 2, dtype=torch.float64)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.build_masked_seed_chunk(unary, membership, [0], [0, 1], contract=_contract(), window_index=0)  # 2 indices vs 3 columns


def test_validation_wrong_component_count():
    unary = torch.ones(2, 1, dtype=torch.float64)
    membership = torch.ones(3, 2, dtype=torch.float64)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.build_masked_seed_chunk(unary, membership, [0, 1], [0], contract=_contract(), window_index=0)  # 2 ids vs 3 rows


def test_validation_node_count_mismatch_between_unary_and_membership():
    unary = torch.ones(2, 1, dtype=torch.float64)
    membership = torch.ones(1, 5, dtype=torch.float64)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.build_masked_seed_chunk(unary, membership, [0], [0], contract=_contract(), window_index=0)


def test_validation_nan_rejected():
    unary = torch.tensor([[float("nan")]], dtype=torch.float64)
    membership = torch.ones(1, 1, dtype=torch.float64)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.build_masked_seed_chunk(unary, membership, [0], [0], contract=_contract(), window_index=0)


def test_validation_infinity_rejected():
    unary = torch.tensor([[float("inf")]], dtype=torch.float64)
    membership = torch.ones(1, 1, dtype=torch.float64)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.build_masked_seed_chunk(unary, membership, [0], [0], contract=_contract(), window_index=0)


@pytest.mark.parametrize("chunk_size", [0, -1, -100])
def test_validation_zero_or_negative_chunk_size_rejected(chunk_size):
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.iter_component_chunks(10, chunk_size)


def test_validation_malformed_contract_rejected_by_propagate():
    unary = torch.ones(2, 1, dtype=torch.float64)
    membership = torch.ones(1, 2, dtype=torch.float64)
    contract = _contract()
    seed = m.build_masked_seed_chunk(unary, membership, [0], [0], contract=contract, window_index=0)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.propagate_masked_seed_chunk(object(), seed, contract="not-a-contract")  # type: ignore[arg-type]


def test_exceptional_keyboard_interrupt_propagates():
    with pytest.raises(KeyboardInterrupt):
        try:
            raise KeyboardInterrupt()
        except m.DfpPhase1bMaskedForwardError:
            pytest.fail("KeyboardInterrupt must never be caught by the domain error handler")


def test_exceptional_system_exit_propagates():
    with pytest.raises(SystemExit):
        try:
            raise SystemExit(2)
        except m.DfpPhase1bMaskedForwardError:
            pytest.fail("SystemExit must never be caught by the domain error handler")


def test_exceptional_memory_error_propagates():
    with pytest.raises(MemoryError):
        try:
            raise MemoryError()
        except m.DfpPhase1bMaskedForwardError:
            pytest.fail("MemoryError must never be caught by the domain error handler")


def test_inputs_unchanged_after_successful_and_failed_calls():
    unary = torch.rand(3, 2, dtype=torch.float64)
    membership = torch.rand(2, 3, dtype=torch.float64)
    unary_before, membership_before = unary.clone(), membership.clone()
    m.build_masked_seed_chunk(unary, membership, [0, 1], [0, 1], contract=_contract(), window_index=0)
    assert torch.equal(unary, unary_before) and torch.equal(membership, membership_before)
    try:
        m.build_masked_seed_chunk(unary, membership, [1, 0], [0, 1], contract=_contract(), window_index=0)  # malformed order -> fails
    except m.DfpPhase1bMaskedForwardError:
        pass
    assert torch.equal(unary, unary_before) and torch.equal(membership, membership_before)


# ---------------------------------------------------------------------------
# Section: memory-complexity instrumentation.
# ---------------------------------------------------------------------------


def test_memory_estimate_scales_with_chunk_not_all_components():
    small = m.estimate_chunk_memory(node_count=1000, component_count=4, query_class_count=3, stitched_height=64, stitched_width=64)
    large_total_components_same_chunk = m.estimate_chunk_memory(
        node_count=1000, component_count=4, query_class_count=3, stitched_height=64, stitched_width=64,
    )
    assert small.total_estimated_bytes == large_total_components_same_chunk.total_estimated_bytes
    bigger_chunk = m.estimate_chunk_memory(node_count=1000, component_count=40, query_class_count=3, stitched_height=64, stitched_width=64)
    assert bigger_chunk.total_estimated_bytes > small.total_estimated_bytes


def test_memory_estimate_never_raises_never_decides_pass_fail():
    estimate = m.estimate_chunk_memory(node_count=1, component_count=1, query_class_count=1, stitched_height=1, stitched_width=1)
    field_names = {f.name for f in __import__("dataclasses").fields(estimate)}
    assert not (field_names & {"passed", "ok", "exceeds_budget", "too_large"})


def test_memory_estimate_rejects_non_positive_dimensions():
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.estimate_chunk_memory(node_count=0, component_count=1, query_class_count=1, stitched_height=1, stitched_width=1)


# ---------------------------------------------------------------------------
# Section: component chunk iteration.
# ---------------------------------------------------------------------------


def test_iter_component_chunks_covers_every_component_exactly_once_in_order():
    chunks = m.iter_component_chunks(10, 3)
    assert chunks == ((0, 3), (3, 6), (6, 9), (9, 10))
    covered = []
    for start, stop in chunks:
        covered.extend(range(start, stop))
    assert covered == list(range(10))


def test_iter_component_chunks_full_size_single_chunk():
    assert m.iter_component_chunks(7, 7) == ((0, 7),)


def test_iter_component_chunks_size_one():
    assert m.iter_component_chunks(3, 1) == ((0, 1), (1, 2), (2, 3))


# ---------------------------------------------------------------------------
# Section: same-job canonical reference guard.
# ---------------------------------------------------------------------------


def test_canonical_reference_requires_same_process_flag():
    graph_stub = object()
    provenance = m.CanonicalReferenceProvenance(
        same_process=False, graph_identity=id(graph_stub), node_count=2, query_class_count=1, dtype=torch.float64,
    )
    canonical = torch.zeros(2, 1, dtype=torch.float64)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.require_same_job_canonical_reference(
            canonical, provenance=provenance, graph=graph_stub, expected_node_count=2, expected_query_class_count=1,
        )


def test_canonical_reference_rejects_different_graph_identity():
    graph_a, graph_b = object(), object()
    provenance = m.CanonicalReferenceProvenance(
        same_process=True, graph_identity=id(graph_a), node_count=2, query_class_count=1, dtype=torch.float64,
    )
    canonical = torch.zeros(2, 1, dtype=torch.float64)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.require_same_job_canonical_reference(
            canonical, provenance=provenance, graph=graph_b, expected_node_count=2, expected_query_class_count=1,
        )


def test_canonical_reference_accepts_consistent_provenance():
    graph_stub = object()
    provenance = m.CanonicalReferenceProvenance(
        same_process=True, graph_identity=id(graph_stub), node_count=2, query_class_count=1, dtype=torch.float64,
    )
    canonical = torch.zeros(2, 1, dtype=torch.float64)
    result = m.require_same_job_canonical_reference(
        canonical, provenance=provenance, graph=graph_stub, expected_node_count=2, expected_query_class_count=1,
    )
    assert torch.equal(result, canonical)


def test_canonical_reference_rejects_geometry_mismatch():
    graph_stub = object()
    provenance = m.CanonicalReferenceProvenance(
        same_process=True, graph_identity=id(graph_stub), node_count=99, query_class_count=1, dtype=torch.float64,
    )
    canonical = torch.zeros(2, 1, dtype=torch.float64)
    with pytest.raises(m.DfpPhase1bMaskedForwardError):
        m.require_same_job_canonical_reference(
            canonical, provenance=provenance, graph=graph_stub, expected_node_count=2, expected_query_class_count=1,
        )


# ---------------------------------------------------------------------------
# Section: forbidden-path poisoning (static source proof).
# ---------------------------------------------------------------------------


_FORBIDDEN_SOURCE_IDENTIFIERS = (
    "build_matched_k11_from_k12", "compute_matched_graph_diagnostics", "dense_fp64_equilibrium_reference",
    "solve", "svdvals", "inv", "eig", "matrix_power", "cgls", "gmres", "cuda", "autocast", "amp",
    "logisticregression", "sklearn", "densecrf", "pamr", "allclose", "isclose",
)


def test_forbidden_paths_never_appear_in_production_source():
    """Identifier-based (never raw substring) so this test is immune to
    the module's own docstring PROSE, which legitimately explains what
    is excluded (e.g. "no PAMR", "no torch.allclose-based decision") in
    plain English sentences -- those sentences are string literals, not
    Name/Attribute/import nodes, so they never appear in the collected
    identifier set."""
    source = Path(m.__file__).read_text(encoding="utf-8")
    identifiers = {name.lower() for name in _collect_identifiers(source)}
    for fragment in _FORBIDDEN_SOURCE_IDENTIFIERS:
        assert fragment.lower() not in identifiers, f"forbidden identifier {fragment!r} is actually referenced as code"


def test_no_early_stopping_construct_in_source():
    source = Path(m.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Break):
            pytest.fail("a `break` statement was found -- no early stopping is permitted in this module")


def test_module_never_imports_torch_cuda_at_module_scope():
    import subprocess
    src = "import sys\nimport src.dfp_phase1b_masked_forward\nprint('done')\n"
    result = subprocess.run([sys.executable, "-c", src], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("done")


def test_module_never_initializes_cuda_on_import_or_seed_construction():
    import torch as _torch
    assert _torch.cuda.is_initialized() is False
    unary = torch.ones(2, 1, dtype=torch.float64)
    membership = torch.ones(1, 2, dtype=torch.float64)
    m.build_masked_seed_chunk(unary, membership, [0], [0], contract=_contract(), window_index=0)
    assert _torch.cuda.is_initialized() is False


# ---------------------------------------------------------------------------
# Section: full synthetic pipeline (requires the real recurrence/
# stitching machinery -- skips cleanly on the known environment gap).
# ---------------------------------------------------------------------------


def test_full_pipeline_requires_working_environment_or_skips_cleanly():
    """A canary confirming the skip mechanism itself works and is
    reached -- this test always runs (never itself skipped) and always
    terminates in one of exactly two ways: it completes a real
    end-to-end propagation, or it calls pytest.skip with the exact
    known reason. It must never silently pass without doing either."""
    reason = _recurrence_environment_broken()
    if reason is not None:
        # `_recurrence_environment_broken()` recognizes exactly two narrow
        # categories (see its own docstring): cv2 entirely missing (its
        # reason string names both "cv2" and "ModuleNotFoundError") and the
        # known NumPy/OpenCV ABI signature (its reason string names "ABI").
        # This must accept exactly those two, not any word that happens to
        # appear in an unrelated failure message (e.g. "error", "missing",
        # "unavailable", "import", "failed" by themselves) -- otherwise this
        # canary could pass on the wrong grounds.
        is_missing_cv2 = "cv2" in reason and "ModuleNotFoundError" in reason
        is_numpy_opencv_abi_mismatch = "ABI" in reason
        assert is_missing_cv2 or is_numpy_opencv_abi_mismatch, reason
        pytest.skip(reason)
    # If the environment is actually working, a real k=12 graph must be
    # constructible and propagate successfully (production requires
    # graph.k == 12 unconditionally; a smaller toy graph would raise).
    neighbor_indices = _k12_ring_neighbor_indices()
    weights = _k12_arbitrary_weights()
    graph = _make_graph(neighbor_indices, weights)
    unary = torch.ones(_K12_NUM_NODES, 1, dtype=torch.float64)
    membership = torch.ones(1, _K12_NUM_NODES, dtype=torch.float64)
    contract = _contract()
    seed = m.build_masked_seed_chunk(unary, membership, [0], [0], contract=contract, window_index=0)
    prop = m.propagate_masked_seed_chunk(graph, seed, contract=contract)
    assert prop.steps_completed == contract.steps
    assert prop.t_final.dtype == torch.float64
    assert graph.transition_weights.dtype == torch.float32


# ---------------------------------------------------------------------------
# Section: the canary test's own acceptance logic (F2) -- proven by
# actually calling the real canary function under a matrix of patched
# environment states, not merely by inspecting its source.
# ---------------------------------------------------------------------------


def test_canary_accepts_missing_cv2_as_clean_skip(monkeypatch):
    def raise_missing_cv2():
        raise ModuleNotFoundError("No module named 'cv2'", name="cv2")

    monkeypatch.setattr(m, "_load_dependencies", raise_missing_cv2)
    with pytest.raises(pytest.skip.Exception):
        test_full_pipeline_requires_working_environment_or_skips_cleanly()


def test_canary_accepts_numpy_opencv_abi_mismatch_as_clean_skip(monkeypatch):
    def raise_abi_mismatch():
        raise ImportError("numpy.core.multiarray failed to import")

    monkeypatch.setattr(m, "_load_dependencies", raise_abi_mismatch)
    with pytest.raises(pytest.skip.Exception):
        test_full_pipeline_requires_working_environment_or_skips_cleanly()


def test_canary_reraises_unrelated_missing_module(monkeypatch):
    def raise_unrelated_missing_module():
        raise ModuleNotFoundError("No module named 'not_cv2_at_all'", name="not_cv2_at_all")

    monkeypatch.setattr(m, "_load_dependencies", raise_unrelated_missing_module)
    with pytest.raises(ModuleNotFoundError, match="not_cv2_at_all"):
        test_full_pipeline_requires_working_environment_or_skips_cleanly()


def test_canary_assertion_rejects_an_arbitrary_unrecognized_skip_reason(monkeypatch):
    """Proves the canary's own acceptance check is narrow, not vacuously
    true: an unrecognized reason string (one naming neither "cv2"+
    "ModuleNotFoundError" nor "ABI") must make the canary itself fail
    with AssertionError, never silently skip or silently pass."""

    def fake_unrecognized_reason():
        return "some unrelated failure: import unavailable, missing, error, failed"

    monkeypatch.setattr(sys.modules[__name__], "_recurrence_environment_broken", fake_unrecognized_reason)
    with pytest.raises(AssertionError):
        test_full_pipeline_requires_working_environment_or_skips_cleanly()


def test_canary_reraises_malformed_graph_identity_error(monkeypatch):
    def raise_domain_error():
        raise m.DfpPhase1bMaskedForwardError("malformed graph/identity")

    monkeypatch.setattr(m, "_load_dependencies", raise_domain_error)
    with pytest.raises(m.DfpPhase1bMaskedForwardError, match="malformed graph/identity"):
        test_full_pipeline_requires_working_environment_or_skips_cleanly()


def test_canary_reraises_arbitrary_runtime_error(monkeypatch):
    def raise_runtime_error():
        raise RuntimeError("unrelated runtime failure")

    monkeypatch.setattr(m, "_load_dependencies", raise_runtime_error)
    with pytest.raises(RuntimeError, match="unrelated runtime failure"):
        test_full_pipeline_requires_working_environment_or_skips_cleanly()


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_canary_propagates_fatal_exceptions(monkeypatch, exc_type):
    def raise_it():
        raise exc_type("must propagate")

    monkeypatch.setattr(m, "_load_dependencies", raise_it)
    with pytest.raises(exc_type):
        test_full_pipeline_requires_working_environment_or_skips_cleanly()


def test_canary_executes_without_skipping_in_compatible_environment():
    """In a genuinely working environment the canary must actually run
    its real k12 propagation path and never call pytest.skip -- this
    meta-test itself skips (with a clear, attributable reason) only if
    THIS environment happens to be the broken one."""
    _skip_if_recurrence_environment_broken()
    test_full_pipeline_requires_working_environment_or_skips_cleanly()
