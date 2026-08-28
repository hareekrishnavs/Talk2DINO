# DFP Phase 1A — Step 9: estimating Phase 1B compute requirements

`src/dfp_phase1b_compute_estimator.py` is a pure, static estimator for
the arithmetic width, operation count, and working-memory requirements
of the *proposed* Phase 1B exact chunked masked-forward propagation. It
never runs that propagation -- it consumes already-derived structural
metadata (Steps 5-8's stitched fields, components, pullback plan, and
soft memberships) and produces exact integer counts, a declared FLOP
convention, a transparent byte model, and hardware-independent
feasibility.

## Dispute set D

`a(x) = lowest_index_argmax(S0(x))`, `b(x) = lowest_index_argmax(S320(x))`
(reusing Step 3's own `derive_horizon_labels`, never a second
tie-break), `D = {x : a(x) != b(x)}`.

The frozen `[dispute_protocol].dispute_set_definition` additionally
restricts D to "non-ignore evaluation pixels" for the *metric*-
computation context, where ground truth is available. This estimator
has no ground-truth input by design (per this step's own explicit
instructions), so it computes D over the *complete* stitched-image
domain. Dropping the ignore-mask restriction can only ever add pixels
to D, never remove any -- so this estimator's D is a safe, conservative
superset of the true evaluation-restricted dispute set, exactly the
direction a capacity estimate should err in.

Only S0 and S320 are used -- never T16/T32/T64, never per-window or
per-patch argmax classes, never GT/ignore masks/oracle labels.

## Query classes Q_w

`Q_w = {a(x), b(x) : x in D and window w contributes to x}`. "Window w
contributes to x" is derived from Step 7's own validated window
geometry (`window.accumulation_slice`) *and* the actual positivity of
Step 7's own denominator field there (already an invariant Step 7
itself enforces when the plan is built) -- never an independently
assumed bounding box. `Q_w` is stored as an ascending, deduplicated
tuple of class indices; a window with no supported disputed pixel has
`Q_w = ()`.

## Active components R_w

`C_w = {r : exists j such that M[r,w,j] > 0}`, `R_w = |C_w|`, using
Step 8's raw, un-thresholded, un-epsilon'd soft memberships directly
(`iter_soft_membership_records`, reused unmodified). Both the exact
active component IDs and `R_w` are always reported, even when `Q_w` is
empty -- in that case the *projected width* is zero (`K_w = R_w * 0`),
never `R_w` itself.

## Projected width

`K_w = R_w * |Q_w|` -- the frozen `query_compute_projection.
projected_width_formula`. For a component chunk of size `b`:
`K_chunk,w = b * |Q_w|`.

## FLOP convention

- `spmm_flops(width) = 2 * E_w * width` (one multiply + one add per
  stored transition/class pair, per recurrence update).
- `elementwise_flops(width) = 3 * N_w * width`, traced literally from
  `finite_step_regime.finite_step_propagate`'s update line `p =
  alpha_value * graph.matmul(p) + one_minus_alpha * s0_owned` (two
  scalar multiplies, one scalar add, per node/column) -- never assumed
  without this trace.
- `recurrence_flops(width) = T * (spmm_flops(width) +
  elementwise_flops(width))`, linear in width, so `sum(recurrence_
  flops(chunk_width) for each chunk) == recurrence_flops(full_width)`
  exactly. Chunking changes peak width and invocation count, never the
  total mathematical work.
- Masking/seed-construction work is reported separately
  (`masking_flops`), as an explicitly-labeled minimal plausible
  estimate (one elementwise multiply per node/column) -- never a claim
  about the eventual Phase 1B implementation's exact cost.
- Every count is a Python arbitrary-precision `int`; nothing risks
  int32/int64 overflow at any aggregate scale.

## Byte-counting model

Graph storage, membership chunk, query-class source scores, working
recurrence state, capture buffers, and fixed geometry metadata are each
a separately-named, separately-computed term (see the module docstring
for the exact formula and derivation of each). Two headline numbers are
always reported distinctly:

- **arithmetic tensor lower bound**: the minimal live-tensor byte count
  assuming perfect in-place reuse.
- **planned live-tensor working-set estimate**: a conservative model
  counting four simultaneously-live width_chunk-sized tensors per
  recurrence step (current state, matmul output, next state, persistent
  masked seed), traced from the same update line as the FLOP model.

Neither is ever labeled "measured GPU peak memory." Allocator/workspace
overhead is unknown and excluded -- no percentage multiplier is
invented for it.

## Chunking

`chunks_w = ceil(R_w / B)`, computed via exact integer arithmetic
(`chunk_component_counts`), the same partition Step 8's own chunked API
already uses. A window with `Q_w` empty requires zero Phase 1B
recurrence invocations regardless of `R_w`.

## Aggregation and percentiles

`aggregate_window_estimates` sums exact integer totals across windows
and retains every per-window raw vector needed to reconstruct any
descriptive statistic. `descriptive_statistics` reports mean, median,
p90, p95, p99, and maximum using numpy's default linear-interpolation
percentile method (`PERCENTILE_METHOD`, explicitly documented) -- these
are always labeled descriptive, never a scientific gate, and the raw
input values are always retained alongside.

## Memory-budget scenarios

An optional `memory_budget_bytes` (an exact positive int) is a
*scenario*, never an identity field or a claim of actual hardware
availability. Feasibility, excess bytes, and the largest feasible
component chunk size under the declared memory model are computed by
exact integer search -- never estimated or approximated. No H100/A100
or private-cluster memory size is ever hardcoded.

## No fabricated runtime

`runtime_status` is always
`"not_estimated_without_measured_hardware_calibration"` unless a
caller supplies an explicit, separately-measured `HardwareCalibration`
record (hardware identifier, software environment, measured
throughput, calibration workload definition, artifact digest) --
never invented by this module. No estimate produced here contains
seconds, images/second, GPU-hours, or a completion date derived from
FLOPs alone.

## No real-subset estimates at this stage

Every fixture in this step's own test suite is synthetic. This module
verifies its own capability, not real Phase 1B feasibility -- no real
COCO-Stuff/ADE20K p95/p99/max value, graph nonzero count, or memory
distribution is fabricated here.

## Excluded from this step

No masked seed construction, no masked forward recurrence, no
component evidence, no `G_xr`, no supercohort propagation, no
k-star/phi, no eta-net, no Gate 1/2 analysis, no GPU microbenchmark, no
real-data driver, no result artifact, no checkpoint/resume, no SLURM
script, no final selector. **No real Phase 1B work is executed by this
step.**
