# DFP Phase 1A — Step 4: multi-horizon capture

`src/dfp_phase1a_recurrence_capture.py` binds the frozen Phase 1A
horizons to the **existing, unmodified** finite-step recurrence kernel
(`models.dinotext.cover_dr.finite_step_regime.finite_step_propagate`).
It adds no new mathematics: the 320-step update loop, graph
construction, row normalization, and snapshot mechanism it uses were
already implemented and tested before this step, originally for the
k11/k12 stability-gate harness. This module is only the DFP-specific
binding on top of that unmodified kernel.

## Frozen horizons and completed-update semantics

`alpha=0.98`, `steps=320`, `horizons=[0, 16, 32, 64, 320]` are read
fresh from the committed `e13-dfp-phase1a-protocol` identity on every
call via `dfp_recurrence_contract_from_protocol` -- never hardcoded as
an independent module constant. `T=0` is the unpropagated unary/E3 seed
`Y0` itself, captured **before** any recurrence update. `T=n` (n > 0)
is captured immediately **after** the *n*th completed update. The
recurrence always completes exactly `steps` (320) updates in one
continuous pass, regardless of which horizons are requested or whether
the state has already reached a fixed point.

## One-pass capture, not four propagation runs

`finite_step_propagate`'s own `snapshot_steps` mechanism (already
present, unmodified) captures every requested horizon from a single
`for step in range(1, steps + 1)` loop -- there is no restart, no
separate propagation call per horizon, and no rebuilt graph per
horizon. `capture_dfp_phase1a_trajectory` calls this kernel exactly
once per window, with `snapshot_steps` equal to the identity's own
`propagation.horizons`.

## Exact relation to the existing T=320 recurrence

Every existing production caller of `finite_step_propagate` (the
matched k11/k12 evaluator, the COCO-Object evaluator, the
stitching-control harness, the native-edge-support audit) already calls
it with `snapshot_steps=(steps,)` -- i.e. T=320-only -- and is
completely untouched by this step. `capture_dfp_phase1a_trajectory`'s
`T=320` snapshot is bit-identical to what those existing callers already
produce for the same graph, seed, alpha, and step count: this is proven
directly by `tests/test_dfp_phase1a_recurrence_capture.py`'s
existing-result-invariance tests, run on directed-asymmetric,
all-fallback-row, and tied-affinity fixtures.

## Non-aliasing guarantees

Every captured snapshot is an owned, detached clone (`finite_step_
propagate`'s own contract, unchanged here): mutating one returned
snapshot cannot alter another snapshot, the input `s0`, or the graph.
No snapshot retains an autograd graph even when `s0.requires_grad` is
true. dtype and device are preserved exactly; nothing is moved to CPU
or silently cast inside the capture path.

## No graph rebuild per horizon, no early stopping, no solver

One `DirectedTopKGraph` is built once per window and reused for every
horizon capture. There is no convergence check, no tolerance-based early
stop, and no call anywhere in this path to CGLS, a dense linear solve,
a matrix inverse, GMRES, or an eigensolver -- those live only in
clearly-separate, never-called-from-here diagnostic functions
(`dense_fp64_equilibrium_reference`, `compute_condition_diagnostics`) in
the same kernel module.

## Identity-binding behavior

`dfp_recurrence_contract_from_protocol` independently, structurally
validates `propagation.alpha` / `propagation.steps` /
`propagation.horizons` from the live identity every call -- never
trusting that `src.dfp_phase1a_identity.load_identity` already checked
them. Rejects: a non-`list` horizons container, a non-exact-int entry
(bool-as-int, int-as-float, numeric string), a negative entry, an entry
above `steps`, any non-strictly-ascending ordering (which simultaneously
catches reordering and duplication), a final entry not equal to `steps`,
and a non-exact-int/float `steps`/`alpha`.

Structural validation alone cannot tell a scientifically wrong value
from a merely differently-shaped one -- `alpha=0.99` and a horizons list
missing `T=0` (while still correctly ending at `steps`) are both
structurally well-formed. To close that gap, this function additionally,
independently invokes the frozen propagation authority --
`src.dfp_phase1a_identity.validate_graph_block` and
`validate_propagation_block`, the same functions `load_identity` itself
uses -- so `alpha`, `steps`, `horizons`, the graph's
`canonical_k_variant`, `early_stopping`, and `fallback_solver` are all
rejected if altered, whether or not the caller's `protocol_identity` ever
passed through `load_identity`. This reuses the single existing
canonical authority rather than hardcoding a second copy of `0.98`,
`[0, 16, 32, 64, 320]`, or `12` in this module. Both the structural
checks and this delegated re-validation complete, and any rejection is
raised, strictly before `dfp_recurrence_contract_from_protocol` returns
-- i.e. before any graph construction or the first recurrence update.

If a caller also supplies `requested_horizons`, it must be a genuine
`list` or `tuple` -- never an arbitrary iterable (a `dict`, `set`,
string, generator, or numpy array is always rejected outright, even one
whose iteration order happens to agree with the frozen horizons) -- and
must exactly equal the identity's own frozen horizons, or the call is
rejected -- never silently resolved in either value's favor.

## Not implemented in this step

This step captures window-level score trajectories only. It does not
run sliding-window interpolation/stitching, does not compute pixel
components or soft source memberships, does not run oracle analysis or
select a final DFP predictor, and does not touch Boundary IoU. **Phase
1A analysis has not been run by this step** -- it only adds and verifies
the capacity to capture the frozen trajectory. The next step consumes
these captured per-window score fields as input to canonical
interpolation and stitching (already frozen and unmodified elsewhere in
this repository), which is out of this step's scope.
