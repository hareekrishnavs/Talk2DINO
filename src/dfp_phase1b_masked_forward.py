"""DFP Phase 1B: exact GT-free masked-forward provenance core.

This module implements the component-masked forward propagation and
signed-evidence extraction frozen by
``evaluation_identities/e15_dfp_phase1b_protocol.toml``'s
``[component_evidence_decomposition]`` block, reusing -- never
duplicating -- the existing committed authorities:

* ``models.dinotext.cover_dr.finite_step_regime.finite_step_propagate``
  -- the SAME unmodified finite-step k12 recurrence kernel every other
  production caller (the matched k11/k12 evaluator, the COCO-Object
  evaluator, ``src.dfp_phase1a_recurrence_capture``) already uses.
  This module never copies its 320-step update loop, never changes
  ``alpha``, never adds a convergence test, and never calls a solver.
* ``src.dfp_phase1a_stitching_pullback.apply_window_stitching_forward``
  -- the canonical per-window forward stitching operator ``A_w``,
  already generalized to accept an arbitrary batch width, reused here
  unmodified with ``K = R_chunk * |Q_w|`` as that batch width.
* ``src.dfp_phase1a_final_rescale_pullback.apply_final_rescale_forward``
  -- the canonical final-rescale operator ``R``, reused unmodified.
* ``src.dfp_phase1b_compute_estimator.compute_window_query_classes`` --
  the single existing frozen ``Q_w`` derivation, reused unmodified
  (never a second, independent ``Q_w`` computation).

``A_final = R ∘ A_stitch`` composed with the frozen k12 masked
recurrence gives, for every global component ``r``:

    Y0^(r)[w,j,c]   = M[r,w,j] * Y0[w,j,c]                       (seed)
    Y_{t+1,w}^(r)   = alpha * P_w @ Y_{t,w}^(r) + (1-alpha) * Y0^(r)   (recurrence, t=0..steps-1)
    S_t^{(r)}       = A_final(Y_t^{(r)})                          (t in {0, steps})
    g_{x,r}         = [S_320^{(r)}(x,b) - S_320^{(r)}(x,a)]
                       - [S_0^{(r)}(x,b) - S_0^{(r)}(x,a)]

with the frozen conservation identity ``sum_r g_{x,r} = delta_m(x) =
m320(x) - m0(x)``.

**This module measures conservation residuals but does not decide
whether they pass. No Phase 1B provenance has been run on real data.
No Gate-2 result exists. No GPU execution is authorized by this
implementation.** See ``docs/dfp_phase1b_masked_forward.md`` for the
full authority/formula/exclusion account.

No masked propagation, real image, GT, PAMR, text-distance, ``k_star``/
``phi``, logistic regression, bootstrap, or Gate-2 decision logic is
implemented, computed, or run by this module.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OVS_ROOT = _REPO_ROOT / "src/open_vocabulary_segmentation"


def _ensure_ovs_on_sys_path() -> None:
    """Insert the repo root and the ``open_vocabulary_segmentation``
    tree onto ``sys.path`` -- matching ``src.dfp_phase1a_recurrence_
    capture``'s and ``src.dfp_phase1a_execute``'s own established
    convention exactly, never a new/different path convention."""
    for path in (_REPO_ROOT, _OVS_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


# Every frozen authority this module reuses -- the finite-step k12
# kernel, the graph type, the window stitching forward operator, the
# final-rescale forward operator, and the Phase 1A recurrence-contract
# extractor -- is imported LAZILY (function-local, inside
# ``_load_dependencies`` below), never at module import time. This
# matches ``src.dfp_phase1a_execute``'s own already-established pattern
# for the identical reason: ``models/__init__.py`` unconditionally
# imports the full model-building stack (``mmcv`` -> ``cv2``), which is
# broken by a pre-existing NumPy-2/OpenCV/mmcv ABI incompatibility in
# this Alliance-wheelhouse environment, unrelated to this module --
# ``src.dfp_phase1a_stitching_pullback``, ``src.dfp_phase1a_final_
# rescale_pullback``, and ``src.dfp_phase1a_recurrence_capture``
# themselves already fail to import at module scope here today, exactly
# as independently confirmed for ``models.dinotext.cover_dr.*``
# directly. Deferring every one of these imports lets every function in
# this module that does NOT need them (seed construction, residual
# arithmetic, memory estimation, component chunking) remain importable
# and testable even where that environment gap is present -- exactly as
# already true for ``dfp_phase1a_execute``'s own non-recurrence-touching
# code paths. This module never re-implements any of these operators to
# work around the gap; it always imports and calls the SAME unmodified
# functions, only deferred to call time.
if TYPE_CHECKING:  # pragma: no cover -- for static type checkers only, never executed
    from models.dinotext.cover_dr.graph import DirectedTopKGraph
    from src.dfp_phase1a_final_rescale_pullback import DfpFinalResolutionContract
    from src.dfp_phase1a_stitching_pullback import StitchingPullbackPlan


@dataclass(frozen=True)
class _LazyDependencies:
    """The full set of frozen-authority names this module needs,
    resolved once per call site via ``_load_dependencies()`` -- never
    duplicated, never reimplemented, always the real committed
    function/class objects."""

    FiniteStepRegimeError: type
    finite_step_propagate: Any
    DirectedTopKGraph: type
    DfpPhase1aRecurrenceCaptureError: type
    dfp_recurrence_contract_from_protocol: Any
    DfpPhase1aStitchingPullbackError: type
    StitchingPullbackPlan: type
    apply_window_stitching_forward: Any
    DfpPhase1aFinalRescalePullbackError: type
    DfpFinalResolutionContract: type
    apply_final_rescale_forward: Any
    dfp_final_resolution_contract_from_protocol: Any


def _load_dependencies() -> "_LazyDependencies":
    """Import every frozen authority this module reuses, exactly once
    per call, in the exact same way ``src.dfp_phase1a_execute`` already
    does for the identical dependencies. Any ``ImportError`` raised
    here (including the pre-existing NumPy-2/OpenCV/mmcv ABI gap) is
    never caught or wrapped -- it propagates unmodified, so a caller in
    a broken environment sees the real, unmodified upstream error."""
    _ensure_ovs_on_sys_path()
    from models.dinotext.cover_dr.finite_step_regime import (
        FiniteStepRegimeError,
        finite_step_propagate,
    )
    from models.dinotext.cover_dr.graph import DirectedTopKGraph
    from src.dfp_phase1a_final_rescale_pullback import (
        DfpFinalResolutionContract,
        DfpPhase1aFinalRescalePullbackError,
        apply_final_rescale_forward,
        dfp_final_resolution_contract_from_protocol,
    )
    from src.dfp_phase1a_recurrence_capture import (
        DfpPhase1aRecurrenceCaptureError,
        dfp_recurrence_contract_from_protocol,
    )
    from src.dfp_phase1a_stitching_pullback import (
        DfpPhase1aStitchingPullbackError,
        StitchingPullbackPlan,
        apply_window_stitching_forward,
    )

    return _LazyDependencies(
        FiniteStepRegimeError=FiniteStepRegimeError, finite_step_propagate=finite_step_propagate,
        DirectedTopKGraph=DirectedTopKGraph,
        DfpPhase1aRecurrenceCaptureError=DfpPhase1aRecurrenceCaptureError,
        dfp_recurrence_contract_from_protocol=dfp_recurrence_contract_from_protocol,
        DfpPhase1aStitchingPullbackError=DfpPhase1aStitchingPullbackError,
        StitchingPullbackPlan=StitchingPullbackPlan, apply_window_stitching_forward=apply_window_stitching_forward,
        DfpPhase1aFinalRescalePullbackError=DfpPhase1aFinalRescalePullbackError,
        DfpFinalResolutionContract=DfpFinalResolutionContract, apply_final_rescale_forward=apply_final_rescale_forward,
        dfp_final_resolution_contract_from_protocol=dfp_final_resolution_contract_from_protocol,
    )


class DfpPhase1bMaskedForwardError(ValueError):
    """Raised on any DFP Phase 1B masked-forward invariant violation.
    Always fail closed: never silently normalize, clamp, threshold,
    drop, merge, or approximate a malformed or scientifically-invalid
    input. Never raised for, and never catches, ``KeyboardInterrupt``,
    ``SystemExit``, or ``MemoryError`` -- those always propagate
    unmodified. Dependency exceptions (``FiniteStepRegimeError``,
    ``DfpPhase1aStitchingPullbackError``, ``DfpPhase1aFinalRescale
    PullbackError``, ``DfpPhase1aRecurrenceCaptureError``) are always
    re-raised as this module's own error type, chained via ``from
    exc`` so ``__cause__`` is preserved."""


ACCUMULATION_DTYPE = torch.float64
PROPAGATION_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------


def _require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise DfpPhase1bMaskedForwardError(f"{label} must be an exact non-boolean integer, observed {type(value).__name__}")
    return value


def _require_positive_exact_int(value: Any, label: str) -> int:
    value = _require_exact_int(value, label)
    if value <= 0:
        raise DfpPhase1bMaskedForwardError(f"{label} must be positive, observed {value}")
    return value


def _require_ascending_distinct_ints(values: Sequence[Any], label: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise DfpPhase1bMaskedForwardError(f"{label} must be a list or tuple, observed {type(values).__name__}")
    if len(values) == 0:
        raise DfpPhase1bMaskedForwardError(f"{label} must be non-empty")
    validated: list[int] = []
    for index, item in enumerate(values):
        if type(item) is not int:
            raise DfpPhase1bMaskedForwardError(f"{label}[{index}] must be an exact non-boolean integer, observed {type(item).__name__}")
        if item < 0:
            raise DfpPhase1bMaskedForwardError(f"{label}[{index}] must be non-negative, observed {item}")
        validated.append(item)
    for earlier, later in zip(validated, validated[1:]):
        if later <= earlier:
            raise DfpPhase1bMaskedForwardError(
                f"{label} must be strictly ascending with no duplicates or reordering, observed {validated}"
            )
    return tuple(validated)


def _require_ascending_contiguous_component_ids(values: Sequence[Any], label: str) -> tuple[int, ...]:
    ids = _require_ascending_distinct_ints(values, label)
    for earlier, later in zip(ids, ids[1:]):
        if later != earlier + 1:
            raise DfpPhase1bMaskedForwardError(
                f"{label} must be an exact contiguous ascending run of global component IDs (no gaps), observed {ids}"
            )
    return ids


def _require_floating_tensor(value: Any, label: str, *, ndim: int) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise DfpPhase1bMaskedForwardError(f"{label} must be a torch.Tensor")
    if value.dtype == torch.bool:
        raise DfpPhase1bMaskedForwardError(f"{label} must not be a boolean tensor (boolean-as-integer/float is forbidden)")
    if not value.is_floating_point():
        raise DfpPhase1bMaskedForwardError(f"{label} must be floating point, observed dtype {value.dtype}")
    if value.ndim != ndim:
        raise DfpPhase1bMaskedForwardError(f"{label} must be exactly {ndim}-D, observed {value.ndim}-D")
    if not torch.isfinite(value).all():
        raise DfpPhase1bMaskedForwardError(f"{label} contains a non-finite value (NaN or Infinity)")
    return value


# ---------------------------------------------------------------------------
# Frozen contract: alpha / steps / precision, extracted from the live
# Phase 1A and Phase 1B protocol identities -- never hardcoded here.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskedForwardContract:
    """Independently validated masked-forward parameters, extracted
    fresh from the live Phase 1A and Phase 1B protocol identities every
    call -- never a hardcoded module constant. ``alpha``/``steps`` are
    the SAME values every canonical (unmasked) T320 propagation uses
    (``same_graph_and_recurrence_as_canonical_t320_required`` in the
    Phase 1B identity), re-validated via
    ``src.dfp_phase1a_recurrence_capture.dfp_recurrence_contract_from_
    protocol`` -- the single existing canonical authority for those two
    values -- never a second, independently hardcoded copy."""

    alpha: float
    steps: int
    propagation_dtype: torch.dtype
    accumulation_dtype: torch.dtype


def dfp_masked_forward_contract_from_protocol(
    phase1a_protocol_identity: Mapping[str, Any], phase1b_identity: Mapping[str, Any],
) -> MaskedForwardContract:
    """Strictly extract and validate the masked-forward contract.

    ``alpha``/``steps`` come from ``phase1a_protocol_identity``'s own
    ``[propagation]`` block via the single existing canonical authority
    (never re-derived or hardcoded here); every scientifically-relevant
    boolean/string field of ``phase1b_identity``'s own
    ``[component_evidence_decomposition]`` block is independently
    re-checked against its frozen required value -- ``exact_masked_
    forward_propagation_required``, ``exact_chunking_only``, ``same_
    graph_and_recurrence_as_canonical_t320_required``, ``adjoint_
    approximation_forbidden_in_primary_implementation``, ``dense_
    influence_kernel_forbidden``, ``top_k_truncation_forbidden``,
    ``residual_supercomponent_forbidden``, ``altered_precision_between_
    canonical_and_masked_runs_forbidden``, ``component_sum_
    accumulation_dtype == "fp64"``, ``early_stopping_forbidden``,
    ``equilibrium_solver_forbidden``, ``recurrence_variant_required ==
    "k12"``, ``k11_counterfactual_forbidden`` -- so a caller that
    somehow supplies an altered in-memory identity dict (bypassing
    ``src.dfp_phase1b_identity.load_identity``'s own closed-schema
    pinning) is still rejected here, before any graph construction or
    recurrence update."""
    deps = _load_dependencies()
    try:
        recurrence_contract = deps.dfp_recurrence_contract_from_protocol(phase1a_protocol_identity)
    except deps.DfpPhase1aRecurrenceCaptureError as exc:
        raise DfpPhase1bMaskedForwardError(str(exc)) from exc

    try:
        decomposition = phase1b_identity["component_evidence_decomposition"]
    except KeyError as exc:
        raise DfpPhase1bMaskedForwardError(f"phase1b_identity is missing a required section: {exc}") from exc

    required_true = (
        "exact_masked_forward_propagation_required", "exact_chunking_only",
        "same_graph_and_recurrence_as_canonical_t320_required",
        "adjoint_approximation_forbidden_in_primary_implementation",
        "dense_influence_kernel_forbidden", "top_k_truncation_forbidden",
        "residual_supercomponent_forbidden",
        "altered_precision_between_canonical_and_masked_runs_forbidden",
        "early_stopping_forbidden", "equilibrium_solver_forbidden",
        "k11_counterfactual_forbidden",
    )
    for field_name in required_true:
        try:
            value = decomposition[field_name]
        except KeyError as exc:
            raise DfpPhase1bMaskedForwardError(f"component_evidence_decomposition is missing required field: {exc}") from exc
        if value is not True:
            raise DfpPhase1bMaskedForwardError(
                f"component_evidence_decomposition.{field_name} must be frozen true, observed {value!r}"
            )

    if decomposition.get("recurrence_variant_required") != "k12":
        raise DfpPhase1bMaskedForwardError(
            f"component_evidence_decomposition.recurrence_variant_required must be 'k12', observed {decomposition.get('recurrence_variant_required')!r}"
        )
    if decomposition.get("component_sum_accumulation_dtype") != "fp64":
        raise DfpPhase1bMaskedForwardError(
            f"component_evidence_decomposition.component_sum_accumulation_dtype must be 'fp64', observed {decomposition.get('component_sum_accumulation_dtype')!r}"
        )

    return MaskedForwardContract(
        alpha=recurrence_contract.alpha, steps=recurrence_contract.steps,
        propagation_dtype=PROPAGATION_DTYPE, accumulation_dtype=ACCUMULATION_DTYPE,
    )


# ---------------------------------------------------------------------------
# Section 5: component-masked seed construction.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskedSeedChunk:
    """One component chunk's masked unary seeds for one window.

    ``seed_flat`` has shape ``[node_count, component_count *
    query_class_count]``, flattened COMPONENT-MAJOR then QUERY-CLASS-
    MAJOR: column index ``r_local * query_class_count + c_local`` holds
    ``Y0^(r)[w, j, query_class_indices[c_local]]`` for the component
    whose global id is ``component_id_start + r_local``. This exact
    flattening order is required to invert (``unflatten``) without
    guessing. Immutable: constructed once, never mutated; does not
    alias ``unary_window_scores`` or ``membership_rows``."""

    seed_flat: torch.Tensor
    window_index: int
    node_count: int
    component_id_start: int
    component_id_stop: int
    query_class_indices: tuple[int, ...]

    @property
    def component_count(self) -> int:
        return self.component_id_stop - self.component_id_start

    @property
    def query_class_count(self) -> int:
        return len(self.query_class_indices)

    def unflatten(self, flat: torch.Tensor) -> torch.Tensor:
        """``[node_count, component_count * query_class_count] ->
        [node_count, component_count, query_class_count]``, inverting
        this chunk's own flattening convention exactly."""
        if flat.ndim != 2 or flat.shape[1] != self.component_count * self.query_class_count:
            raise DfpPhase1bMaskedForwardError(
                f"cannot unflatten tensor of shape {tuple(flat.shape)} against "
                f"component_count={self.component_count}, query_class_count={self.query_class_count}"
            )
        return flat.reshape(flat.shape[0], self.component_count, self.query_class_count)


def build_masked_seed_chunk(
    unary_window_scores: torch.Tensor, membership_rows: torch.Tensor,
    global_component_ids: Sequence[int], query_class_indices: Sequence[int],
    *, contract: MaskedForwardContract, window_index: int,
) -> MaskedSeedChunk:
    """``Y0^(r)[w,j,c] = M[r,w,j] * Y0[w,j,c]`` for one window and one
    contiguous chunk of global component ids -- a strict pure function:
    performs exact elementwise multiplication only, with no
    normalization, clamping, thresholding, binarization, rounding, or
    epsilon-adjustment of ``membership_rows``, no silent discarding of
    a zero-evidence component, no combining of small components, and
    no residual component. Never inspects GT: nothing about a target
    category, ``harmful``/``beneficial`` label, or oracle value is
    accepted by this function's signature.

    ``unary_window_scores``: ``[node_count, query_class_count]``,
    already restricted to ``query_class_indices`` in that exact order
    (this function never re-derives or reorders the restriction).
    ``membership_rows``: ``[component_count, node_count]``, exactly
    ``iter_image_final_soft_memberships``' own per-(window, chunk)
    output -- this function never recomputes membership with a
    different operator."""
    window_index = _require_exact_int(window_index, "window_index")
    if window_index < 0:
        raise DfpPhase1bMaskedForwardError("window_index must be non-negative")
    if not isinstance(contract, MaskedForwardContract):
        raise DfpPhase1bMaskedForwardError("contract must be a MaskedForwardContract")

    component_ids = _require_ascending_contiguous_component_ids(global_component_ids, "global_component_ids")
    query_classes = _require_ascending_distinct_ints(query_class_indices, "query_class_indices")

    unary_window_scores = _require_floating_tensor(unary_window_scores, "unary_window_scores", ndim=2)
    membership_rows = _require_floating_tensor(membership_rows, "membership_rows", ndim=2)

    if unary_window_scores.device != membership_rows.device:
        raise DfpPhase1bMaskedForwardError(
            f"unary_window_scores device {unary_window_scores.device} disagrees with membership_rows device {membership_rows.device}"
        )
    node_count = unary_window_scores.shape[0]
    if membership_rows.shape[1] != node_count:
        raise DfpPhase1bMaskedForwardError(
            f"membership_rows node dimension {membership_rows.shape[1]} disagrees with unary_window_scores node dimension {node_count}"
        )
    if unary_window_scores.shape[1] != len(query_classes):
        raise DfpPhase1bMaskedForwardError(
            f"unary_window_scores has {unary_window_scores.shape[1]} query-class columns, expected len(query_class_indices)={len(query_classes)}"
        )
    if membership_rows.shape[0] != len(component_ids):
        raise DfpPhase1bMaskedForwardError(
            f"membership_rows has {membership_rows.shape[0]} component rows, expected len(global_component_ids)={len(component_ids)}"
        )

    dtype = contract.accumulation_dtype
    unary64 = unary_window_scores.detach().to(dtype)
    membership64 = membership_rows.detach().to(dtype)

    # [R, N] -> [N, R, 1]; [N, Q] -> [N, 1, Q]; product -> [N, R, Q].
    seed = (membership64.transpose(0, 1).unsqueeze(-1) * unary64.unsqueeze(1)).reshape(
        node_count, len(component_ids) * len(query_classes)
    )
    seed = seed.clone().detach()
    if not torch.isfinite(seed).all():
        raise DfpPhase1bMaskedForwardError("constructed seed contains a non-finite value")

    return MaskedSeedChunk(
        seed_flat=seed, window_index=window_index, node_count=node_count,
        component_id_start=component_ids[0], component_id_stop=component_ids[-1] + 1,
        query_class_indices=query_classes,
    )


# ---------------------------------------------------------------------------
# Section 6: exact chunked recurrence -- direct reuse of
# finite_step_propagate, no second recurrence implementation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropagatedChunk:
    """``Y_0^(r)`` and ``Y_320^(r)`` (or, generally, ``Y_steps^(r)``)
    for one window and one component chunk. Both tensors are detached
    from autograd and own their storage (never aliasing the seed, the
    graph's weights, or the membership tensors that built the seed)."""

    t0: torch.Tensor
    t_final: torch.Tensor
    window_index: int
    component_id_start: int
    component_id_stop: int
    query_class_indices: tuple[int, ...]
    steps_completed: int

    @property
    def component_count(self) -> int:
        return self.component_id_stop - self.component_id_start


def propagate_masked_seed_chunk(
    graph: DirectedTopKGraph, seed: MaskedSeedChunk, *, contract: MaskedForwardContract,
) -> PropagatedChunk:
    """Run exactly ``contract.steps`` completed updates of the
    UNMODIFIED ``finite_step_propagate`` kernel over the frozen k12
    ``graph``, capturing only the two snapshots this evidence core
    needs (``t=0`` and ``t=contract.steps``) from the SAME one
    continuous pass -- never a second, separate propagation run, never
    an early stop, never a convergence check, never a solver call.
    ``graph`` must already be the frozen k12 transition operator for
    this window; this function never constructs, symmetrizes, or
    re-selects a graph of its own.

    Cheap, dependency-free structural validation (``seed``/``contract``
    type checks) runs BEFORE the frozen-authority import, so an
    obviously malformed call is rejected without paying for -- or
    requiring a working environment for -- the heavy import."""
    if not isinstance(seed, MaskedSeedChunk):
        raise DfpPhase1bMaskedForwardError("seed must be a MaskedSeedChunk")
    if not isinstance(contract, MaskedForwardContract):
        raise DfpPhase1bMaskedForwardError("contract must be a MaskedForwardContract")

    deps = _load_dependencies()
    if not isinstance(graph, deps.DirectedTopKGraph):
        raise DfpPhase1bMaskedForwardError("graph must be a DirectedTopKGraph")
    if graph.k != 12:
        raise DfpPhase1bMaskedForwardError(f"graph must be the frozen k=12 recurrence graph, observed k={graph.k}")
    if graph.num_nodes != seed.node_count:
        raise DfpPhase1bMaskedForwardError(
            f"graph.num_nodes ({graph.num_nodes}) disagrees with seed.node_count ({seed.node_count})"
        )

    seed_for_propagation = seed.seed_flat.detach().to(contract.propagation_dtype)
    try:
        trace = deps.finite_step_propagate(
            graph, seed_for_propagation, alpha=contract.alpha, steps=contract.steps,
            snapshot_steps=(0, contract.steps),
        )
    except deps.FiniteStepRegimeError as exc:
        raise DfpPhase1bMaskedForwardError(str(exc)) from exc

    t0 = trace.snapshots[0].detach().clone()
    t_final = trace.snapshots[contract.steps].detach().clone()
    if t0.data_ptr() == seed.seed_flat.data_ptr() or t_final.data_ptr() == seed.seed_flat.data_ptr():
        raise DfpPhase1bMaskedForwardError("propagated output must not alias the seed tensor's storage")

    return PropagatedChunk(
        t0=t0, t_final=t_final, window_index=seed.window_index,
        component_id_start=seed.component_id_start, component_id_stop=seed.component_id_stop,
        query_class_indices=seed.query_class_indices, steps_completed=trace.steps_completed,
    )


# ---------------------------------------------------------------------------
# Section 8/9: canonical forward stitching + final rescale, with
# per-window query-class alignment onto a shared chunk-level union.
# ---------------------------------------------------------------------------


def compute_query_class_union(per_window_query_classes: Mapping[int, Sequence[int]]) -> tuple[int, ...]:
    """The ascending, deduplicated union of every window's own frozen
    ``Q_w`` in this call -- used only to align heterogeneous per-window
    channel widths onto one shared stitching accumulator; never a
    substitute for any individual window's own ``Q_w`` (each window's
    seed is still built from, and only from, its own ``Q_w``)."""
    if not isinstance(per_window_query_classes, Mapping) or not per_window_query_classes:
        raise DfpPhase1bMaskedForwardError("per_window_query_classes must be a non-empty mapping")
    union: set[int] = set()
    for window_index, classes in per_window_query_classes.items():
        _require_exact_int(window_index, "per_window_query_classes key")
        union.update(_require_ascending_distinct_ints(tuple(classes), f"per_window_query_classes[{window_index}]"))
    return tuple(sorted(union))


def _embed_into_query_union(
    values: torch.Tensor, *, component_count: int, local_query_classes: tuple[int, ...],
    query_union: tuple[int, ...],
) -> torch.Tensor:
    """``[node_count, component_count * len(local_query_classes)] ->
    [node_count, component_count * len(query_union)]``, zero-padding
    every query-class column this window did not itself need. Exact
    (no interpolation, no approximation): every retained column is
    copied verbatim; every padded column is exactly zero, which is the
    mathematically correct contribution of a window whose own frozen
    ``Q_w`` never included that class (a window not needing a class
    contributes nothing to it anywhere)."""
    node_count = values.shape[0]
    reshaped = values.reshape(node_count, component_count, len(local_query_classes))
    union_index = {c: i for i, c in enumerate(query_union)}
    positions = torch.tensor([union_index[c] for c in local_query_classes], dtype=torch.long, device=values.device)
    embedded = torch.zeros((node_count, component_count, len(query_union)), dtype=values.dtype, device=values.device)
    embedded[:, :, positions] = reshaped
    return embedded.reshape(node_count, component_count * len(query_union))


def stitch_and_rescale_component_scores(
    propagated_by_window: Mapping[int, PropagatedChunk], *, window_plan: StitchingPullbackPlan,
    final_contract: DfpFinalResolutionContract, img_shape: tuple[int, int], ori_shape: tuple[int, int],
    component_count: int, query_class_union: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``S_0^{(r)} = A_final(Y_0^{(r)})`` and ``S_final^{(r)} =
    A_final(Y_final^{(r)})`` for one component chunk, composed as
    ``R ∘ A_stitch`` using only the existing, unmodified canonical
    operators (``apply_window_stitching_forward``,
    ``apply_final_rescale_forward``) -- no nearest-neighbor path, no
    alternative stitching convention, no dense final-pixels x
    patch-sources matrix. Returns ``(S0, S_final)``, each
    ``[1, component_count * len(query_class_union), ori_h, ori_w]``."""
    deps = _load_dependencies()
    if not isinstance(window_plan, deps.StitchingPullbackPlan):
        raise DfpPhase1bMaskedForwardError("window_plan must be a StitchingPullbackPlan")
    if not isinstance(final_contract, deps.DfpFinalResolutionContract):
        raise DfpPhase1bMaskedForwardError("final_contract must be a DfpFinalResolutionContract")
    component_count = _require_positive_exact_int(component_count, "component_count")
    query_union = _require_ascending_distinct_ints(tuple(query_class_union), "query_class_union")
    if not isinstance(propagated_by_window, Mapping) or not propagated_by_window:
        raise DfpPhase1bMaskedForwardError("propagated_by_window must be a non-empty mapping")

    height, width = window_plan.image_size
    channel_count = component_count * len(query_union)
    dtype = ACCUMULATION_DTYPE
    accumulator_t0 = torch.zeros((1, channel_count, height, width), dtype=dtype)
    accumulator_final = torch.zeros((1, channel_count, height, width), dtype=dtype)

    for window_index, chunk in propagated_by_window.items():
        if not isinstance(chunk, PropagatedChunk):
            raise DfpPhase1bMaskedForwardError(f"propagated_by_window[{window_index}] must be a PropagatedChunk")
        if chunk.window_index != window_index:
            raise DfpPhase1bMaskedForwardError(
                f"propagated_by_window key {window_index} disagrees with chunk.window_index {chunk.window_index}"
            )
        if chunk.component_count != component_count:
            raise DfpPhase1bMaskedForwardError(
                f"window {window_index}: component_count {chunk.component_count} disagrees with the requested {component_count}"
            )
        window = window_plan.plan.windows[window_index]
        rows, cols = window.accumulation_slice

        for snapshot, accumulator in ((chunk.t0, accumulator_t0), (chunk.t_final, accumulator_final)):
            embedded = _embed_into_query_union(
                snapshot.to(dtype), component_count=component_count,
                local_query_classes=chunk.query_class_indices, query_union=query_union,
            )
            u = embedded.transpose(0, 1)  # [K, N]
            try:
                contribution = deps.apply_window_stitching_forward(u, window_plan=window_plan, window_index=window_index)
            except deps.DfpPhase1aStitchingPullbackError as exc:
                raise DfpPhase1bMaskedForwardError(str(exc)) from exc
            accumulator[:, :, rows, cols] += contribution.unsqueeze(0).to(dtype)

    try:
        s0 = deps.apply_final_rescale_forward(accumulator_t0, img_shape=img_shape, ori_shape=ori_shape, contract=final_contract)
        s_final = deps.apply_final_rescale_forward(accumulator_final, img_shape=img_shape, ori_shape=ori_shape, contract=final_contract)
    except deps.DfpPhase1aFinalRescalePullbackError as exc:
        raise DfpPhase1bMaskedForwardError(str(exc)) from exc

    return s0.detach(), s_final.detach()


# ---------------------------------------------------------------------------
# Section 10: streaming signed contributions.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignedContributionChunk:
    """``g_{x,r}`` for one component chunk, over a fixed, deterministic
    set of disputed final-resolution pixels. ``values`` has shape
    ``[disputed_pixel_count, component_count]`` -- exact-pinned
    orientation, documented here: rows are disputed pixels ordered by
    increasing final-image flat index (``row * ori_width + col``),
    columns are global component ids in ascending order starting at
    ``component_id_start``. Signed values are retained verbatim,
    including zeros and negatives; never sorted by magnitude."""

    values: torch.Tensor
    component_id_start: int
    component_id_stop: int
    disputed_pixel_flat_indices: tuple[int, ...]

    @property
    def component_count(self) -> int:
        return self.component_id_stop - self.component_id_start


def compute_signed_contributions(
    s0: torch.Tensor, s_final: torch.Tensor, *, component_count: int, query_class_union: Sequence[int],
    disputed_pixel_rows: torch.Tensor, disputed_pixel_cols: torch.Tensor,
    disputed_pixel_a: torch.Tensor, disputed_pixel_b: torch.Tensor,
    ori_width: int, component_id_start: int,
) -> SignedContributionChunk:
    """``g_{x,r} = [S_final^{(r)}(x,b) - S_final^{(r)}(x,a)] -
    [S_0^{(r)}(x,b) - S_0^{(r)}(x,a)]`` for every disputed pixel in
    this call and every component in this chunk. ``disputed_pixel_a``/
    ``disputed_pixel_b`` are the GLOBAL class ids required present in
    ``query_class_union`` for every listed pixel (guaranteed by the
    frozen ``Q_w`` definition whenever the pixel's covering windows'
    ``Q_w``s were included in the union) -- a missing required class
    fails closed rather than silently substituting a nearby class."""
    query_union = _require_ascending_distinct_ints(tuple(query_class_union), "query_class_union")
    component_count = _require_positive_exact_int(component_count, "component_count")
    component_id_start = _require_exact_int(component_id_start, "component_id_start")
    ori_width = _require_positive_exact_int(ori_width, "ori_width")

    for name, tensor in (
        ("disputed_pixel_rows", disputed_pixel_rows), ("disputed_pixel_cols", disputed_pixel_cols),
        ("disputed_pixel_a", disputed_pixel_a), ("disputed_pixel_b", disputed_pixel_b),
    ):
        if not torch.is_tensor(tensor):
            raise DfpPhase1bMaskedForwardError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.int64:
            raise DfpPhase1bMaskedForwardError(f"{name} must have dtype exactly int64, observed {tensor.dtype}")
        if tensor.ndim != 1:
            raise DfpPhase1bMaskedForwardError(f"{name} must be exactly 1-D, observed {tensor.ndim}-D")
    pixel_count = disputed_pixel_rows.shape[0]
    for name, tensor in (
        ("disputed_pixel_cols", disputed_pixel_cols), ("disputed_pixel_a", disputed_pixel_a), ("disputed_pixel_b", disputed_pixel_b),
    ):
        if tensor.shape[0] != pixel_count:
            raise DfpPhase1bMaskedForwardError(f"{name} length {tensor.shape[0]} disagrees with disputed_pixel_rows length {pixel_count}")
    if pixel_count == 0:
        raise DfpPhase1bMaskedForwardError("at least one disputed pixel is required")

    union_index = {c: i for i, c in enumerate(query_union)}
    a_list = disputed_pixel_a.tolist()
    b_list = disputed_pixel_b.tolist()
    for name, values in (("disputed_pixel_a", a_list), ("disputed_pixel_b", b_list)):
        for value in values:
            if value not in union_index:
                raise DfpPhase1bMaskedForwardError(
                    f"{name} contains class {value}, which is not present in query_class_union {query_union} -- "
                    "a covering window's Q_w must include every disputed pixel's a(x)/b(x)"
                )
    a_local = torch.tensor([union_index[v] for v in a_list], dtype=torch.long)
    b_local = torch.tensor([union_index[v] for v in b_list], dtype=torch.long)

    for name, tensor in (("s0", s0), ("s_final", s_final)):
        if not torch.is_tensor(tensor) or tensor.ndim != 4 or tensor.shape[0] != 1:
            raise DfpPhase1bMaskedForwardError(f"{name} must be a batched [1, K, H, W] tensor")
        if tensor.shape[1] != component_count * len(query_union):
            raise DfpPhase1bMaskedForwardError(
                f"{name} channel dimension {tensor.shape[1]} disagrees with component_count * len(query_class_union) "
                f"= {component_count * len(query_union)}"
            )
        if not torch.isfinite(tensor).all():
            raise DfpPhase1bMaskedForwardError(f"{name} contains a non-finite value (NaN or Infinity)")

    s0v = s0.squeeze(0).to(ACCUMULATION_DTYPE).reshape(component_count, len(query_union), s0.shape[2], s0.shape[3])
    sfv = s_final.squeeze(0).to(ACCUMULATION_DTYPE).reshape(component_count, len(query_union), s_final.shape[2], s_final.shape[3])

    s0_a = s0v[:, a_local, disputed_pixel_rows, disputed_pixel_cols]  # [R, D]
    s0_b = s0v[:, b_local, disputed_pixel_rows, disputed_pixel_cols]
    sf_a = sfv[:, a_local, disputed_pixel_rows, disputed_pixel_cols]
    sf_b = sfv[:, b_local, disputed_pixel_rows, disputed_pixel_cols]

    g = (sf_b - sf_a) - (s0_b - s0_a)  # [R, D]
    g = g.transpose(0, 1).contiguous().detach()  # [D, R] per the pinned output orientation

    flat_indices = (disputed_pixel_rows * ori_width + disputed_pixel_cols).tolist()
    order = sorted(range(pixel_count), key=lambda i: flat_indices[i])
    if order != list(range(pixel_count)):
        raise DfpPhase1bMaskedForwardError(
            "disputed pixels must already be ordered by increasing final-image flat index"
        )

    return SignedContributionChunk(
        values=g, component_id_start=component_id_start, component_id_stop=component_id_start + component_count,
        disputed_pixel_flat_indices=tuple(flat_indices),
    )


# ---------------------------------------------------------------------------
# Section 11: raw conservation diagnostics -- measurement only, no
# tolerance, no pass/fail decision.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualSummary:
    """One raw residual measurement -- values only, never a pass/fail
    verdict. ``signed`` and ``absolute`` are the full residual arrays
    (never discarded); the scalar fields summarize them without
    hiding a nonzero residual behind a threshold."""

    signed: torch.Tensor
    absolute: torch.Tensor
    max_absolute: float
    mean_absolute: float
    finite_count: int
    non_finite_count: int
    compared_element_count: int


def _summarize_residual(residual: torch.Tensor) -> ResidualSummary:
    residual = residual.detach().to(ACCUMULATION_DTYPE)
    finite_mask = torch.isfinite(residual)
    finite_count = int(finite_mask.sum().item())
    non_finite_count = int((~finite_mask).sum().item())
    absolute = residual.abs()
    finite_absolute = absolute[finite_mask]
    max_absolute = float(finite_absolute.max().item()) if finite_absolute.numel() else float("nan")
    mean_absolute = float(finite_absolute.mean().item()) if finite_absolute.numel() else float("nan")
    return ResidualSummary(
        signed=residual.clone(), absolute=absolute.clone(), max_absolute=max_absolute, mean_absolute=mean_absolute,
        finite_count=finite_count, non_finite_count=non_finite_count, compared_element_count=residual.numel(),
    )


@dataclass(frozen=True)
class ConservationMeasurements:
    """Raw, un-gated conservation diagnostics for one component chunk.
    Every field is a MEASUREMENT: this dataclass contains no boolean
    pass/fail field, no threshold, and no tolerance anywhere.
    Constructing an instance never raises on a large or nonzero
    residual -- only on non-finite (NaN/Infinity) INPUT, which is a
    structural precondition failure, not a conservation decision. The
    future 200-window calibration step establishes the tolerance this
    module deliberately does not define."""

    membership_partition_residual: ResidualSummary
    seed_reconstruction_residual: ResidualSummary
    propagated_reconstruction_residual: ResidualSummary
    signed_margin_conservation_residual: ResidualSummary
    component_count: int
    query_class_count: int
    disputed_pixel_count: int
    dtype: torch.dtype


def compute_membership_partition_residual(membership_rows: torch.Tensor) -> ResidualSummary:
    """``rho_M(j) = sum_r M[r,w,j] - 1``. ``membership_rows`` is
    ``[component_count, node_count]`` for ONE window's FULL component
    set (every component with any membership at this window, not one
    arbitrary chunk) -- a partial component set would make this
    residual meaningless, so callers must supply the complete row set
    themselves; this function performs no completeness check of its
    own since it cannot know the full component count from the tensor
    alone."""
    membership_rows = _require_floating_tensor(membership_rows, "membership_rows", ndim=2)
    total = membership_rows.detach().to(ACCUMULATION_DTYPE).sum(dim=0)
    residual = total - 1.0
    return _summarize_residual(residual)


def compute_seed_reconstruction_residual(seed_by_component: torch.Tensor, unary_window_scores: torch.Tensor) -> ResidualSummary:
    """``rho_0(j,c) = sum_r Y0^(r)[w,j,c] - Y0[w,j,c]``.
    ``seed_by_component`` is ``[component_count, node_count,
    query_class_count]`` (every component contributing to this
    window); ``unary_window_scores`` is ``[node_count,
    query_class_count]``."""
    if not torch.is_tensor(seed_by_component) or seed_by_component.ndim != 3:
        raise DfpPhase1bMaskedForwardError("seed_by_component must be a 3-D [component_count, node_count, query_class_count] tensor")
    unary_window_scores = _require_floating_tensor(unary_window_scores, "unary_window_scores", ndim=2)
    total = seed_by_component.detach().to(ACCUMULATION_DTYPE).sum(dim=0)
    if total.shape != unary_window_scores.shape:
        raise DfpPhase1bMaskedForwardError(
            f"summed seed shape {tuple(total.shape)} disagrees with unary_window_scores shape {tuple(unary_window_scores.shape)}"
        )
    residual = total - unary_window_scores.detach().to(ACCUMULATION_DTYPE)
    return _summarize_residual(residual)


def compute_propagated_reconstruction_residual(
    propagated_by_component: torch.Tensor, canonical_propagated: torch.Tensor,
) -> ResidualSummary:
    """``rho_320(j,c) = sum_r Y_320^(r)[w,j,c] - Y_320[w,j,c]``.
    ``propagated_by_component`` is ``[component_count, node_count,
    query_class_count]`` (every component contributing to this
    window's coverage); ``canonical_propagated`` is ``[node_count,
    query_class_count]``, produced by the SAME-job canonical (unmasked)
    reference (see ``require_same_job_canonical_reference``)."""
    if not torch.is_tensor(propagated_by_component) or propagated_by_component.ndim != 3:
        raise DfpPhase1bMaskedForwardError("propagated_by_component must be a 3-D [component_count, node_count, query_class_count] tensor")
    canonical_propagated = _require_floating_tensor(canonical_propagated, "canonical_propagated", ndim=2)
    total = propagated_by_component.detach().to(ACCUMULATION_DTYPE).sum(dim=0)
    if total.shape != canonical_propagated.shape:
        raise DfpPhase1bMaskedForwardError(
            f"summed propagated shape {tuple(total.shape)} disagrees with canonical_propagated shape {tuple(canonical_propagated.shape)}"
        )
    residual = total - canonical_propagated.detach().to(ACCUMULATION_DTYPE)
    return _summarize_residual(residual)


def compute_signed_margin_conservation_residual(
    signed_contributions_all_components: torch.Tensor, delta_m: torch.Tensor,
) -> ResidualSummary:
    """``rho_g(x) = sum_r g_{x,r} - delta_m(x)``.
    ``signed_contributions_all_components`` is ``[disputed_pixel_count,
    total_component_count]`` -- EVERY component contributing evidence
    to any listed pixel, not one arbitrary chunk (a partial component
    sum would make this residual meaningless, exactly like the other
    three residuals above); ``delta_m`` is ``[disputed_pixel_count]``,
    the frozen ``m320(x) - m0(x)`` for the same pixels in the same
    order."""
    if not torch.is_tensor(signed_contributions_all_components) or signed_contributions_all_components.ndim != 2:
        raise DfpPhase1bMaskedForwardError("signed_contributions_all_components must be a 2-D [disputed_pixel_count, component_count] tensor")
    delta_m = _require_floating_tensor(delta_m, "delta_m", ndim=1)
    if signed_contributions_all_components.shape[0] != delta_m.shape[0]:
        raise DfpPhase1bMaskedForwardError(
            f"signed_contributions_all_components row count {signed_contributions_all_components.shape[0]} "
            f"disagrees with delta_m length {delta_m.shape[0]}"
        )
    total = signed_contributions_all_components.detach().to(ACCUMULATION_DTYPE).sum(dim=1)
    residual = total - delta_m.detach().to(ACCUMULATION_DTYPE)
    return _summarize_residual(residual)


def build_conservation_measurements(
    *, membership_partition_residual: ResidualSummary, seed_reconstruction_residual: ResidualSummary,
    propagated_reconstruction_residual: ResidualSummary, signed_margin_conservation_residual: ResidualSummary,
    component_count: int, query_class_count: int, disputed_pixel_count: int,
) -> ConservationMeasurements:
    """Bundle the four raw residual measurements into one immutable
    record. Performs no comparison, no threshold, and no pass/fail
    decision of any kind -- purely a structural container."""
    for name, value in (
        ("membership_partition_residual", membership_partition_residual),
        ("seed_reconstruction_residual", seed_reconstruction_residual),
        ("propagated_reconstruction_residual", propagated_reconstruction_residual),
        ("signed_margin_conservation_residual", signed_margin_conservation_residual),
    ):
        if not isinstance(value, ResidualSummary):
            raise DfpPhase1bMaskedForwardError(f"{name} must be a ResidualSummary")
    component_count = _require_positive_exact_int(component_count, "component_count")
    query_class_count = _require_positive_exact_int(query_class_count, "query_class_count")
    disputed_pixel_count = _require_positive_exact_int(disputed_pixel_count, "disputed_pixel_count")
    return ConservationMeasurements(
        membership_partition_residual=membership_partition_residual,
        seed_reconstruction_residual=seed_reconstruction_residual,
        propagated_reconstruction_residual=propagated_reconstruction_residual,
        signed_margin_conservation_residual=signed_margin_conservation_residual,
        component_count=component_count, query_class_count=query_class_count,
        disputed_pixel_count=disputed_pixel_count, dtype=ACCUMULATION_DTYPE,
    )


# ---------------------------------------------------------------------------
# Section 12: same-job canonical unmasked reference guard.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalReferenceProvenance:
    """Declares that a canonical (unmasked) reference tensor was
    produced in the SAME process/job as the masked run it will be
    compared against -- never a previously serialized tensor loaded
    from disk as the sole comparison authority. This module does not
    itself run the canonical reference (that is
    ``src.dfp_phase1a_recurrence_capture.capture_dfp_phase1a_
    trajectory``'s job, reused unmodified by the caller); it only
    validates that a supplied reference's provenance/shape/dtype/
    geometry are consistent with the masked run before it is used in
    any residual."""

    same_process: bool
    graph_identity: int
    node_count: int
    query_class_count: int
    dtype: torch.dtype


def require_same_job_canonical_reference(
    canonical_propagated: torch.Tensor, *, provenance: CanonicalReferenceProvenance,
    graph: DirectedTopKGraph, expected_node_count: int, expected_query_class_count: int,
) -> torch.Tensor:
    """Fail closed unless ``provenance`` declares this reference was
    produced same-process, against the SAME graph object identity, and
    with shape/dtype consistent with the masked run it will be
    compared against. Never accepts a bare tensor with no provenance
    record, and never itself loads a tensor from disk."""
    if not isinstance(provenance, CanonicalReferenceProvenance):
        raise DfpPhase1bMaskedForwardError("provenance must be a CanonicalReferenceProvenance")
    if provenance.same_process is not True:
        raise DfpPhase1bMaskedForwardError(
            "canonical reference provenance does not declare same_process=True -- "
            "a previously serialized tensor is never accepted as the sole comparison authority"
        )
    if provenance.graph_identity != id(graph):
        raise DfpPhase1bMaskedForwardError(
            "canonical reference was produced against a different graph object than the one supplied here -- "
            "refusing an accidental cross-run comparison"
        )
    if provenance.node_count != expected_node_count or provenance.query_class_count != expected_query_class_count:
        raise DfpPhase1bMaskedForwardError(
            f"canonical reference geometry (node_count={provenance.node_count}, "
            f"query_class_count={provenance.query_class_count}) disagrees with the masked run's own "
            f"(node_count={expected_node_count}, query_class_count={expected_query_class_count})"
        )
    canonical_propagated = _require_floating_tensor(canonical_propagated, "canonical_propagated", ndim=2)
    if canonical_propagated.dtype != provenance.dtype:
        raise DfpPhase1bMaskedForwardError(
            f"canonical_propagated dtype {canonical_propagated.dtype} disagrees with its own declared provenance dtype {provenance.dtype}"
        )
    if tuple(canonical_propagated.shape) != (expected_node_count, expected_query_class_count):
        raise DfpPhase1bMaskedForwardError(
            f"canonical_propagated shape {tuple(canonical_propagated.shape)} disagrees with expected "
            f"({expected_node_count}, {expected_query_class_count})"
        )
    return canonical_propagated


# ---------------------------------------------------------------------------
# Section 14: memory-complexity instrumentation -- reporting only, no
# pass/fail resource decision, no automatic approximation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkMemoryEstimate:
    """Reported tensor shapes and estimated peak bytes for one
    component-chunk propagation call. A measurement only: this
    dataclass never decides whether a chunk size is acceptable and
    never triggers an automatic smaller chunk size. If a requested
    chunk is too large, the caller must select a smaller explicit
    ``component_chunk_size`` -- this module never approximates."""

    seed_shape: tuple[int, int]
    propagated_shape: tuple[int, int]
    stitched_shape: tuple[int, int, int, int]
    element_bytes: int
    seed_bytes: int
    propagated_bytes: int
    stitched_bytes: int
    total_estimated_bytes: int


def estimate_chunk_memory(
    *, node_count: int, component_count: int, query_class_count: int, stitched_height: int, stitched_width: int,
    dtype: torch.dtype = ACCUMULATION_DTYPE,
) -> ChunkMemoryEstimate:
    """Pure arithmetic, no allocation. Peak masked recurrence state is
    ``O(node_count * component_count * query_class_count)`` (never
    ``N x N``, never per-full-image-component); stitched working
    memory is bounded by one component chunk's own channel width, not
    all components."""
    node_count = _require_positive_exact_int(node_count, "node_count")
    component_count = _require_positive_exact_int(component_count, "component_count")
    query_class_count = _require_positive_exact_int(query_class_count, "query_class_count")
    stitched_height = _require_positive_exact_int(stitched_height, "stitched_height")
    stitched_width = _require_positive_exact_int(stitched_width, "stitched_width")

    element_bytes = torch.tensor([], dtype=dtype).element_size()
    channels = component_count * query_class_count
    seed_shape = (node_count, channels)
    propagated_shape = (node_count, channels)
    stitched_shape = (1, channels, stitched_height, stitched_width)

    seed_bytes = node_count * channels * element_bytes
    propagated_bytes = 2 * node_count * channels * element_bytes  # t0 + t_final
    stitched_bytes = 2 * channels * stitched_height * stitched_width * element_bytes  # accumulator_t0 + accumulator_final

    return ChunkMemoryEstimate(
        seed_shape=seed_shape, propagated_shape=propagated_shape, stitched_shape=stitched_shape,
        element_bytes=element_bytes, seed_bytes=seed_bytes, propagated_bytes=propagated_bytes,
        stitched_bytes=stitched_bytes, total_estimated_bytes=seed_bytes + propagated_bytes + stitched_bytes,
    )


# ---------------------------------------------------------------------------
# Section 6: chunked component iteration -- deterministic, explicit
# chunk size, every component exactly once, order preserved.
# ---------------------------------------------------------------------------


def iter_component_chunks(component_count: int, component_chunk_size: int) -> Sequence[tuple[int, int]]:
    """Yield ``(start, stop)`` (stop exclusive) covering ``0 ..
    component_count-1`` in ascending, non-overlapping, gap-free order.
    ``component_chunk_size`` must be an explicit positive integer --
    never data-dependent, never GT-dependent, never a hidden default
    selected from observed evidence."""
    component_count = _require_positive_exact_int(component_count, "component_count")
    component_chunk_size = _require_positive_exact_int(component_chunk_size, "component_chunk_size")
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < component_count:
        stop = min(start + component_chunk_size, component_count)
        chunks.append((start, stop))
        start = stop
    return tuple(chunks)


__all__ = [
    "DfpPhase1bMaskedForwardError",
    "ACCUMULATION_DTYPE",
    "PROPAGATION_DTYPE",
    "MaskedForwardContract",
    "dfp_masked_forward_contract_from_protocol",
    "MaskedSeedChunk",
    "build_masked_seed_chunk",
    "PropagatedChunk",
    "propagate_masked_seed_chunk",
    "compute_query_class_union",
    "stitch_and_rescale_component_scores",
    "SignedContributionChunk",
    "compute_signed_contributions",
    "ResidualSummary",
    "ConservationMeasurements",
    "compute_membership_partition_residual",
    "compute_seed_reconstruction_residual",
    "compute_propagated_reconstruction_residual",
    "compute_signed_margin_conservation_residual",
    "build_conservation_measurements",
    "CanonicalReferenceProvenance",
    "require_same_job_canonical_reference",
    "ChunkMemoryEstimate",
    "estimate_chunk_memory",
    "iter_component_chunks",
]
