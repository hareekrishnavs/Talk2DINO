"""DFP Phase 1A Step 4: multi-horizon capture binding layer over the
existing, unmodified finite-step recurrence kernel
(``models.dinotext.cover_dr.finite_step_regime.finite_step_propagate``).

This module adds NOTHING to the recurrence itself: it never copies the
320-step update loop, never re-implements graph construction or row
normalization, never changes alpha, never changes the update count, never
adds a convergence check, and never calls a solver. The recurrence's own
``snapshot_steps`` mechanism (already present, unmodified, well before
this step) already captures every requested horizon from ONE continuous
pass -- this module is only the DFP-specific *binding* on top of it:
strictly extracting and validating ``alpha`` / ``steps`` / ``horizons``
from the protocol identity (never hardcoding ``[0, 16, 32, 64, 320]`` or
``320`` as an independent production constant anywhere in this module),
and then calling ``finite_step_propagate`` exactly as every other
production caller in this repository already does. This module never
assumes its ``protocol_identity`` argument already went through
``src.dfp_phase1a_identity.load_identity``: after its own structural/
exact-type checks, it independently re-delegates to that module's public
``validate_graph_block`` / ``validate_propagation_block`` -- the single
existing canonical authority -- so a value that is structurally
well-formed but scientifically wrong (``alpha=0.99``, a horizons list
missing ``T=0``, an extra horizon, an altered ``canonical_k_variant``,
``early_stopping=true``, a non-``"none"`` ``fallback_solver``, ...) is
still rejected before the recurrence contract is constructed and before
any graph construction or update.

No existing caller of ``finite_step_propagate`` (the matched k11/k12
evaluator, the COCO-Object evaluator, the stitching-control harness, the
native-edge-support audit) is touched by this module, and none of them
needs to be: they already pass their own ``snapshot_steps`` explicitly
and remain completely unaffected by this new, separate DFP-specific
entry point.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OVS_ROOT = _REPO_ROOT / "src/open_vocabulary_segmentation"
for _path in (_REPO_ROOT, _OVS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from models.dinotext.cover_dr.finite_step_regime import (  # noqa: E402
    FiniteStepRegimeError,
    FiniteStepTrace,
    finite_step_propagate,
)
from models.dinotext.cover_dr.graph import DirectedTopKGraph  # noqa: E402

from src.dfp_phase1a_identity import (  # noqa: E402
    DfpPhase1aIdentityError,
    validate_graph_block,
    validate_propagation_block,
)


class DfpPhase1aRecurrenceCaptureError(ValueError):
    """Raised on any DFP Phase 1A multi-horizon capture binding
    invariant violation. Always fail closed: never silently normalize,
    sort, deduplicate, or coerce a malformed horizons value."""


def _require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise DfpPhase1aRecurrenceCaptureError(f"{label} must be an exact non-boolean integer")
    return value


def _require_exact_float(value: Any, label: str) -> float:
    if type(value) is not float:
        raise DfpPhase1aRecurrenceCaptureError(f"{label} must be an exact finite float")
    if not math.isfinite(value):
        raise DfpPhase1aRecurrenceCaptureError(f"{label} must be finite")
    return value


def _require_exact_horizons_list(value: Any, *, total_steps: int, label: str) -> tuple[int, ...]:
    """Fail-closed structural validation, independent of whatever
    upstream validation `src.dfp_phase1a_identity.load_identity` already
    performed -- this binding layer never trusts an in-memory dict.
    Requires: a genuine `list` container (never a tuple/set/generator);
    every element an exact non-boolean int (never bool-as-int, float,
    or numeric string); non-negative; no value above `total_steps`;
    strictly ascending (this single check simultaneously rejects
    reordering AND duplication, since neither is possible in a strictly
    ascending sequence); the final entry exactly equal to `total_steps`
    (catching an identity horizon inconsistent with the total update
    count); non-empty. Never sorts, deduplicates, or otherwise coerces
    the input -- any violation is rejected outright."""
    if type(value) is not list:
        raise DfpPhase1aRecurrenceCaptureError(f"{label} must be an exact list (observed {type(value).__name__})")
    if not value:
        raise DfpPhase1aRecurrenceCaptureError(f"{label} must be non-empty")
    validated: list[int] = []
    for index, item in enumerate(value):
        if type(item) is not int:
            raise DfpPhase1aRecurrenceCaptureError(f"{label}[{index}] must be an exact non-boolean integer (observed {type(item).__name__})")
        if item < 0:
            raise DfpPhase1aRecurrenceCaptureError(f"{label}[{index}] must be non-negative")
        if item > total_steps:
            raise DfpPhase1aRecurrenceCaptureError(f"{label}[{index}] = {item} exceeds total_steps = {total_steps}")
        validated.append(item)
    for earlier, later in zip(validated, validated[1:]):
        if later <= earlier:
            raise DfpPhase1aRecurrenceCaptureError(
                f"{label} must be strictly ascending with no duplicates or reordering, observed {validated}"
            )
    if validated[-1] != total_steps:
        raise DfpPhase1aRecurrenceCaptureError(
            f"{label}'s final entry ({validated[-1]}) must equal total_steps ({total_steps})"
        )
    return tuple(validated)


@dataclass(frozen=True)
class DfpRecurrenceContract:
    """Independently validated alpha/steps/horizons extracted from the
    live protocol identity -- never a hardcoded module constant."""

    alpha: float
    steps: int
    horizons: tuple[int, ...]


def dfp_recurrence_contract_from_protocol(protocol_identity: Mapping[str, Any]) -> DfpRecurrenceContract:
    """Strictly extract and validate `propagation.alpha` /
    `propagation.steps` / `propagation.horizons` from an already-loaded
    protocol identity. Independent of, and does not rely on,
    `src.dfp_phase1a_identity.load_identity` having already validated
    these fields -- this function re-validates from scratch every time,
    so it remains correct even if called with a hand-constructed dict
    that bypassed that loader.

    Structural/exact-type validation (list-vs-tuple, exact int/float,
    non-negative, ascending, in-bounds, final entry equal to `steps`)
    happens first, exactly as before. That alone cannot catch a value
    that is structurally well-formed but scientifically wrong -- e.g.
    `alpha=0.99`, or a horizons list missing `T=0` while still correctly
    ending at `steps`. To close that gap without introducing a second,
    independently-hardcoded copy of the frozen constants (`0.98`,
    `[0, 16, 32, 64, 320]`, `canonical_k_variant=12`, ...), this function
    additionally delegates to the single existing canonical authority --
    `src.dfp_phase1a_identity.validate_graph_block` /
    `validate_propagation_block` -- which independently re-checks alpha,
    steps, horizons, early_stopping, fallback_solver, and the graph's
    `canonical_k_variant`, all before this function returns and therefore
    before any graph construction or recurrence update can occur."""
    propagation = protocol_identity["propagation"]
    alpha = _require_exact_float(propagation["alpha"], "propagation.alpha")
    steps = _require_exact_int(propagation["steps"], "propagation.steps")
    if steps < 1:
        raise DfpPhase1aRecurrenceCaptureError("propagation.steps must be at least 1")
    horizons = _require_exact_horizons_list(propagation["horizons"], total_steps=steps, label="propagation.horizons")

    try:
        validate_graph_block(protocol_identity["graph"])
        validate_propagation_block(propagation)
    except DfpPhase1aIdentityError as exc:
        raise DfpPhase1aRecurrenceCaptureError(str(exc)) from exc

    return DfpRecurrenceContract(alpha=alpha, steps=steps, horizons=horizons)


def capture_dfp_phase1a_trajectory(
    graph: DirectedTopKGraph, s0: torch.Tensor, *, protocol_identity: Mapping[str, Any], requested_horizons: Sequence[int] | None = None,
) -> FiniteStepTrace:
    """Run the DFP Phase 1A frozen trajectory: exactly `propagation.steps`
    completed updates of the UNMODIFIED `finite_step_propagate` kernel,
    capturing a snapshot at every horizon named in `propagation.horizons`
    -- both values read fresh from `protocol_identity` every call, never
    hardcoded here.

    If `requested_horizons` is given, it must be a genuine `list` or
    `tuple` -- never an arbitrary iterable (a `dict`, `set`, string,
    generator, or numpy array is always rejected, even one whose
    iteration order happens to agree with the frozen horizons) -- and
    must EXACTLY equal (as an ordered sequence) the identity's own frozen
    horizons. This is the DFP-specific binding layer's exact-equality
    gate for a generic capture primitive that could otherwise accept
    arbitrary capture steps; any conflict is rejected rather than
    silently resolved in either value's favor.

    Performs zero graph construction, zero row normalization, zero
    convergence checks, and zero solver calls -- all of that lives
    exclusively in `finite_step_propagate` and `DirectedTopKGraph`,
    reused here completely unmodified."""
    contract = dfp_recurrence_contract_from_protocol(protocol_identity)
    if requested_horizons is not None:
        if type(requested_horizons) not in (list, tuple):
            raise DfpPhase1aRecurrenceCaptureError(
                f"requested_horizons must be a list or tuple (observed {type(requested_horizons).__name__}); "
                "arbitrary iterables (dict, set, string, generator, numpy array, ...) are never accepted, "
                "even if their iteration order happens to agree with the frozen horizons"
            )
        requested_tuple = tuple(requested_horizons)
        if requested_tuple != contract.horizons:
            raise DfpPhase1aRecurrenceCaptureError(
                f"requested_horizons {requested_tuple} disagrees with the frozen protocol identity's horizons {contract.horizons}"
            )
    return finite_step_propagate(graph, s0, alpha=contract.alpha, steps=contract.steps, snapshot_steps=contract.horizons)


__all__ = [
    "DfpPhase1aRecurrenceCaptureError",
    "DfpRecurrenceContract",
    "dfp_recurrence_contract_from_protocol",
    "capture_dfp_phase1a_trajectory",
    "FiniteStepRegimeError",
    "FiniteStepTrace",
]
