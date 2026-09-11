"""DFP Phase 1B: E15 Stage-A component-evidence PROVENANCE-FEATURE kernel.

Computes, per disputed pixel, the eight E15 Stage-A
``[stage_a_schema.flat_disputed_pixel_members]`` fields that depend on
the per-component signed evidence matrix ``G[x,r]`` --

    n_positive_components, positive_evidence_sum, negative_evidence_sum,
    top_positive_component_id, top_positive_evidence, top1_positive_share,
    k_star, phi

-- exactly per ``evaluation_identities/e15_dfp_phase1b_protocol.toml``'s
``[provenance_features.component_evidence]`` block, reusing --  never
recomputing -- the already-frozen, already-committed
``G[x,r] = [S320^(r)(x,b) - S320^(r)(x,a)] - [S0^(r)(x,b) - S0^(r)(x,a)]``
tensor produced by ``src.dfp_phase1b_masked_forward.compute_signed_
contributions``.

Frozen relations (verbatim from committed E15 authority, never
re-derived or re-guessed here):

    G_plus                  = { G[x,r] : G[x,r] > 0 }                          (STRICT; e15 g_plus_definition)
    positive_evidence_sum   = sum_r max(G[x,r], 0)                             (e15 positive_evidence_sum_formula)
    negative_evidence_sum   = sum_r min(G[x,r], 0)                             (e15 negative_evidence_sum_formula; <= 0)
    sort_order               = descending G value, ties by ascending
                               component ID                                    (e15 sort_order)
    top_positive_evidence    = G_positive_sorted[1]
    top_positive_component_id = the component ID at that rank
    top1_positive_share      = top_positive_evidence / positive_evidence_sum   (e15 top1_positive_share_formula)
    k_star(x)                = minimum k such that
                                m320(x) - sum_{q=1..k} G_positive_sorted[q] <= 0 (e15 k_star_definition; exact <=0, no epsilon)
    phi(x)                   = sum_{q=1..k_star} G_positive_sorted[q]
                                / positive_evidence_sum                        (e15 phi_formula; phi_epsilon_forbidden=true)

Domain facts this module RELIES ON but never re-derives (each already
frozen by E15's own margins/schema authority, and each independently
verified true for every valid disputed pixel):

* ``m320(x) > 0`` (e15 [margins].m320_definition) -- so k_star is never 0.
* ``m0(x) <= 0`` and ``delta_m(x) = m320(x) - m0(x) > 0`` (e15 [margins]),
  together with the conservation identity ``sum_r G[x,r] = delta_m(x)``
  (e15 [component_evidence_decomposition].conservation_identity), force
  ``positive_evidence_sum >= delta_m(x) >= m320(x)`` for every valid
  disputed pixel -- so a k in [1, n_positive_components] satisfying the
  k_star crossing condition always exists, and ``n_positive_components``
  is never 0 (``positive_evidence_sum > 0`` is also directly required by
  ``e15-dfp-phase1b-artifacts``'s own [stage_a_schema.relational_
  validation]). If either invariant is observed violated at runtime, this
  is an invalid state and this module fails closed -- exactly the
  discipline e15's own ``k_star_non_finite_case_behavior`` ("reject an
  invalid non-finite case rather than inventing infinity") requires,
  generalized to every input-domain violation this module can detect.

This module performs no masked propagation, no recurrence, no model or
dataset access, no CUDA initialization merely on import, and reads no
GT. It operates on whatever tensors the caller supplies, on the
caller's own device, and never silently transfers a CUDA tensor to
CPU. `KeyboardInterrupt`, `SystemExit`, and `MemoryError` always
propagate unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class DfpPhase1bProvenanceFeaturesError(ValueError):
    """Raised on any provenance-feature kernel input/domain-invariant
    violation. Always fail closed. Never raised for, and never catches,
    KeyboardInterrupt / SystemExit / MemoryError -- those always
    propagate unmodified."""


@dataclass(frozen=True)
class ComponentEvidenceFeatures:
    """The eight E15 Stage-A component-evidence fields, one value per
    disputed pixel, in the caller's row order. Every tensor is detached,
    freshly allocated (never a view into an input tensor), and on the
    same device as the inputs."""

    k_star: torch.Tensor                    # int64   [P]
    phi: torch.Tensor                       # float64 [P]
    top1_positive_share: torch.Tensor       # float64 [P]
    positive_evidence_sum: torch.Tensor     # float64 [P]
    negative_evidence_sum: torch.Tensor     # float64 [P]
    n_positive_components: torch.Tensor     # int64   [P]
    top_positive_component_id: torch.Tensor  # int64   [P]
    top_positive_evidence: torch.Tensor     # float64 [P]


def _require_tensor(value: Any, label: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise DfpPhase1bProvenanceFeaturesError(f"{label} must be a torch.Tensor, observed {type(value).__name__}")
    return value


def _require_float64_matrix(value: Any, label: str, *, ndim: int) -> torch.Tensor:
    _require_tensor(value, label)
    if value.dtype == torch.bool:
        raise DfpPhase1bProvenanceFeaturesError(f"{label} must not be a boolean tensor")
    if value.dtype != torch.float64:
        raise DfpPhase1bProvenanceFeaturesError(
            f"{label} must be torch.float64 (component_sum_accumulation_dtype='fp64' per e15), observed {value.dtype}"
        )
    if value.ndim != ndim:
        raise DfpPhase1bProvenanceFeaturesError(f"{label} must be exactly {ndim}-D, observed {value.ndim}-D")
    if not torch.isfinite(value).all():
        raise DfpPhase1bProvenanceFeaturesError(f"{label} contains a non-finite value (NaN or Infinity)")
    return value


def _require_int64_vector(value: Any, label: str, *, length: int) -> torch.Tensor:
    _require_tensor(value, label)
    if value.dtype != torch.int64:
        raise DfpPhase1bProvenanceFeaturesError(f"{label} must be torch.int64, observed {value.dtype}")
    if value.ndim != 1:
        raise DfpPhase1bProvenanceFeaturesError(f"{label} must be exactly 1-D, observed {value.ndim}-D")
    if value.shape[0] != length:
        raise DfpPhase1bProvenanceFeaturesError(f"{label} has length {value.shape[0]}, expected {length}")
    return value


def compute_component_evidence_features(
    G: torch.Tensor, m320: torch.Tensor, *, component_ids: torch.Tensor,
) -> ComponentEvidenceFeatures:
    """Compute the eight E15 Stage-A component-evidence fields for every
    disputed pixel (row of ``G``).

    ``G``: ``[disputed_pixel_count, component_count]`` float64 -- the
        already-computed, already-frozen signed component-evidence
        matrix ``G[x,r]`` (e.g. concatenated
        ``SignedContributionChunk.values`` from
        ``src.dfp_phase1b_masked_forward.compute_signed_contributions``,
        covering every global component exactly once, in
        ``component_ids`` column order).
    ``m320``: ``[disputed_pixel_count]`` float64 -- the per-pixel
        propagated margin ``m320(x)`` (e15 ``propagated_margin =
        m320``); must be strictly positive for every row (frozen
        ``m320(x) > 0``).
    ``component_ids``: ``[component_count]`` int64 -- the CANONICAL
        component identity for each column of ``G``, independent of
        column position; used for the frozen ascending-ID tie-break.
        Must contain no duplicate and no negative value.

    Returns a :class:`ComponentEvidenceFeatures` with one value per row
    of ``G``, in the same row order, on the same device.

    Raises :class:`DfpPhase1bProvenanceFeaturesError` on any shape/
    dtype/device/domain-invariant violation (including a row for which
    no valid ``k_star`` exists -- which the frozen ``e15`` margin
    invariants guarantee cannot happen for a genuinely valid disputed
    pixel, so its occurrence here means the caller's input violated
    those invariants).
    """
    _require_float64_matrix(G, "G", ndim=2)
    disputed_pixel_count, component_count = G.shape
    _require_float64_matrix(m320, "m320", ndim=1)
    if m320.shape[0] != disputed_pixel_count:
        raise DfpPhase1bProvenanceFeaturesError(
            f"m320 has length {m320.shape[0]}, expected {disputed_pixel_count} (G's row count)"
        )
    _require_int64_vector(component_ids, "component_ids", length=component_count)
    if component_count == 0:
        raise DfpPhase1bProvenanceFeaturesError("G must have at least one component column")
    if disputed_pixel_count == 0:
        # A zero-row call is a legitimate degenerate case (e.g. an image
        # with zero disputed pixels in this chunk) -- return zero-length
        # tensors, never fabricate a row.
        pass
    if bool((component_ids < 0).any()):
        raise DfpPhase1bProvenanceFeaturesError("component_ids contains a negative component ID")
    if torch.unique(component_ids).numel() != component_ids.numel():
        raise DfpPhase1bProvenanceFeaturesError("component_ids contains a duplicate component ID")
    if G.device != m320.device or G.device != component_ids.device:
        raise DfpPhase1bProvenanceFeaturesError(
            f"G, m320, and component_ids must share the same device (observed G={G.device}, "
            f"m320={m320.device}, component_ids={component_ids.device})"
        )
    if bool((m320 <= 0.0).any()):
        raise DfpPhase1bProvenanceFeaturesError("m320 must be strictly positive for every disputed pixel (frozen m320(x) > 0)")

    device = G.device

    # positive_evidence_sum / negative_evidence_sum -- e15 formulas,
    # verbatim. Neither reads nor mutates G; each is a fresh elementwise
    # clamp followed by a row sum (memory: one extra [P, C] buffer each,
    # same order as G -- no combinatorial blowup).
    positive_part = torch.clamp(G, min=0.0)  # max(G, 0)
    negative_part = torch.clamp(G, max=0.0)  # min(G, 0)
    positive_evidence_sum = positive_part.sum(dim=1)
    negative_evidence_sum = negative_part.sum(dim=1)
    del positive_part, negative_part

    if disputed_pixel_count and bool((positive_evidence_sum <= 0.0).any()):
        raise DfpPhase1bProvenanceFeaturesError(
            "positive_evidence_sum must be strictly positive for every disputed pixel -- a valid disputed "
            "pixel always has at least one G[x,r] > 0 component under the frozen conservation identity "
            "(sum_r G[x,r] = delta_m(x) > 0) and e15-dfp-phase1b-artifacts' own stage_a_schema relational "
            "validation (positive_evidence_sum > 0); observing a non-positive sum means the input violates "
            "that already-frozen invariant"
        )

    is_positive = G > 0.0  # STRICT, per e15 g_plus_definition
    n_positive_components = is_positive.sum(dim=1)
    if disputed_pixel_count and bool((n_positive_components < 1).any()):
        raise DfpPhase1bProvenanceFeaturesError(
            "n_positive_components must be >= 1 for every disputed pixel (implied by positive_evidence_sum > 0)"
        )

    # --- deterministic descending-value / ascending-ID sort -----------
    # component_ids is the SAME for every row (one global component set),
    # so the ascending-ID column permutation is computed ONCE, not per
    # pixel. Reordering columns into ascending-ID order first, then
    # applying a STABLE descending sort by value, reproduces the frozen
    # tie rule exactly -- a stable sort preserves the relative order of
    # equal keys, so equal-value entries remain in ascending-ID order
    # after the value sort -- without depending on any single sort
    # primitive's own (unspecified, possibly device-dependent) tie
    # behavior for equal keys.
    id_order = torch.argsort(component_ids, stable=True)  # ascending component ID, once
    component_ids_by_id = component_ids[id_order]
    G_by_id = G.index_select(1, id_order)
    is_positive_by_id = is_positive.index_select(1, id_order)

    neg_inf = torch.finfo(torch.float64).min  # a finite, safely-sortable sentinel: always < any real G value
    key_value = torch.where(is_positive_by_id, G_by_id, torch.full_like(G_by_id, neg_inf))
    sorted_values, sorted_pos = torch.sort(key_value, dim=1, descending=True, stable=True)
    sorted_component_ids = component_ids_by_id.unsqueeze(0).expand(max(disputed_pixel_count, 1), component_count).gather(1, sorted_pos)

    top_positive_evidence = sorted_values[:, 0].clone()
    top_positive_component_id = sorted_component_ids[:, 0].clone()
    top1_positive_share = top_positive_evidence / positive_evidence_sum

    # cumulative sum of the ranked positive values only; positions beyond
    # n_positive_components hold the neg_inf sentinel and are replaced by
    # 0.0 for the cumulative sum (never read past n_positive_components
    # below, since k_star <= n_positive_components is already guaranteed).
    sorted_values_for_cumsum = torch.where(sorted_values > neg_inf, sorted_values, torch.zeros_like(sorted_values))
    cumsum = torch.cumsum(sorted_values_for_cumsum, dim=1)

    crossed = cumsum >= m320.unsqueeze(1)
    if disputed_pixel_count and bool((~crossed.any(dim=1)).any()):
        raise DfpPhase1bProvenanceFeaturesError(
            "no k in [1, n_positive_components] satisfies the frozen k_star crossing condition for at least "
            "one disputed pixel -- the frozen margin invariants (m0<=0, m320>0, conservation identity) "
            "guarantee this is always satisfiable for a valid disputed pixel; this indicates invalid input, "
            "not a legitimate no-solution case (k_star_non_finite_case_behavior: reject, never invent infinity)"
        )
    k_star_0idx = torch.argmax(crossed.to(torch.int64), dim=1)  # first True (argmax returns the first max)
    k_star = (k_star_0idx + 1).clone()

    phi = cumsum.gather(1, k_star_0idx.unsqueeze(1)).squeeze(1) / positive_evidence_sum

    return ComponentEvidenceFeatures(
        k_star=k_star.to(device=device),
        phi=phi.detach().to(device=device),
        top1_positive_share=top1_positive_share.detach().to(device=device),
        positive_evidence_sum=positive_evidence_sum.detach().to(device=device),
        negative_evidence_sum=negative_evidence_sum.detach().to(device=device),
        n_positive_components=n_positive_components.to(device=device),
        top_positive_component_id=top_positive_component_id.to(device=device),
        top_positive_evidence=top_positive_evidence.detach().to(device=device),
    )


__all__ = [
    "ComponentEvidenceFeatures",
    "DfpPhase1bProvenanceFeaturesError",
    "compute_component_evidence_features",
]
