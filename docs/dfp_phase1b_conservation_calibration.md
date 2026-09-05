# DFP Phase 1B: Conservation-Calibration Authority (E16)

Freezes the scientific authority and closed artifact schema for the
DFP Phase 1B 200-window FP64 conservation-calibration step, per
`evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml`
(loaded/validated by `src/dfp_phase1b_calibration_identity.py`) and the
candidate/selection/artifact logic in
`src/dfp_phase1b_calibration_artifact.py`.

**This is an authority-and-schema task only. No calibration runner is
implemented here.** This runner records raw conservation residuals
only. It does not choose or apply a conservation tolerance. No real
Phase 1B provenance or calibration has been executed. No Gate-2 result
exists. No GPU execution is authorized by this implementation.

## Why this identity exists

A prior narrow attempt to implement the calibration runner correctly
stopped because the committed `e15-dfp-phase1b-protocol` authority
left five scientifically consequential choices unspecified: the exact
200-window selection algorithm, the canonical candidate order/identity,
the exact "stitched-field residual" formula, a closed schema for
`phase1b_conservation_calibration.json`, and the boundary between raw
measurement and later tolerance selection. E16 is a **new successor
leaf authority** that resolves all five, without modifying or
reinterpreting `e15-dfp-phase1b-protocol`, `e15-dfp-phase1b-artifacts`,
or any other frozen identity.

## Committed authority and dependency bindings

E16 binds, by exact repository-relative path and SHA-256 content
digest, the following already-committed files -- it never modifies or
re-derives any of them, only relationally re-verifies their hashes on
every load:

- `evaluation_identities/e15_dfp_phase1b_protocol.toml`
- `evaluation_identities/e15_dfp_phase1b_artifacts.toml`
- `src/dfp_phase1b_masked_forward.py` (the exact masked-forward
  provenance core)
- `evaluation_subsets/dfp_phase1a_coco_stuff_500.json` (the frozen
  COCO-Stuff-500 subset manifest)
- `src/open_vocabulary_segmentation/segmentation/evaluation/sliding_window_geometry.py`
  (`SlidingWindowPlan`, the canonical window-plan implementation)
- `src/dfp_phase1a_stitching_pullback.py` /
  `src/dfp_phase1a_final_rescale_pullback.py` (the composed
  `A_final = R ∘ A_stitch` operator)
- `src/open_vocabulary_segmentation/models/dinotext/cover_dr/graph.py`
  / `.../finite_step_regime.py` (the frozen k12 recurrence kernel)

## Candidate scope and canonical identity

Scope: `run_mode = "coco_stuff_500"`, exactly the 500 images named in
the bound subset manifest's own `selected_ids` field (verbatim, already
canonical lexicographic order -- never re-sorted here), with candidate
windows enumerated by the bound `SlidingWindowPlan.build(...)` per
image. No GT, annotation, prediction, dispute, or model output may
influence candidate existence, identity, or order.

Each candidate record has exactly 15 fields: `run_mode`, `image_id`,
`image_path`, `image_manifest_ordinal`, `inference_canvas_height`,
`inference_canvas_width`, `canonical_window_ordinal`, `top`, `left`,
`bottom`, `right`, `crop_height`, `crop_width`, `patch_grid_height`,
`patch_grid_width`. The frozen byte encoding is
`talk2dino-dfp-phase1b-calibration-candidate-record-v1`: UTF-8 JSON,
alphabetical key order, `separators=(',', ':')`, `ensure_ascii=True`,
no floats/bools/nulls, no trailing newline.

**Canonical candidate order** (used for storage, never for selection):
`image_manifest_ordinal` ascending, then `canonical_window_ordinal`
ascending. Coordinates are validation fields only, never an ordering
authority.

## Deterministic selection algorithm

1. Sort the complete candidate universe into canonical candidate order.
2. Encode each record with the frozen encoding.
3. `selection_key = SHA256(namespace_utf8 + 0x00 + encoded_record)`,
   where `namespace` is e15's own already-frozen
   `talk2dino-dfp-phase1b-calibration-window-selection-v1` string,
   reused verbatim.
4. Rank ascending by `(selection_key_hex, canonical_candidate_index)`.
5. Select the first exactly 200. Fail closed if the universe has fewer
   than 200 candidates.
6. Store the 200 selected records in **canonical candidate order** in
   the artifact, never selection-rank order.

Three digests are frozen, each `sha256(concat(len(record).to_bytes(8,
"big") + record for record in records))`:
`candidate_universe_digest` (the complete universe, canonical order),
`selected_windows_digest` (the 200 selected, canonical order), and
`selection_trace_digest` (200 `{canonical_candidate_index,
selection_key, selection_rank}` records, selection-rank order).

Selection is independent of filesystem/dict/`PYTHONHASHSEED` order,
worker count, predictions, GT, GPU availability, component/dispute
counts, and residual magnitude -- verified directly (including across
real `PYTHONHASHSEED`-varied subprocesses) in
`tests/test_dfp_phase1b_calibration_artifact.py`.

## Execution contract (frozen; not implemented here)

Exactly 200 selected windows; canonical and masked propagation in the
same process/job, from the same graph instance; frozen k=12, T=320
recurrence; graph storage FP32, canonical/masked recurrence and
residual accumulation FP64; no mixed precision, autocast, or TF32; no
early stopping, equilibrium solver, k11 path, dense influence kernel,
adjoint approximation, or Top-K residual approximation; reuses the
committed masked-forward APIs only (no duplicate seed construction,
recurrence, signed-contribution formula, conservation formula,
stitching, final rescale, or membership construction). Measurement
only -- issues no conservation pass/fail decision.

## Residual definitions

All four existing conservation quantities from
`src/dfp_phase1b_masked_forward.py` are recorded as diagnostics:
`rho_M` (`compute_membership_partition_residual`, over the window's
full component-membership rows), `rho_0`
(`compute_seed_reconstruction_residual`, seed vs. summed masked seeds,
raw per-window recurrence-state domain), `rho_320`
(`compute_propagated_reconstruction_residual`, same domain at T=320).

**`rho_g` is the primary calibration measurement**
(`compute_signed_margin_conservation_residual`), measured *after* the
composed final-resolution operator `A_final,w = R ∘ A_stitch,w`:
`rho_g^(w)(x) = sum_r g^(w)_{x,r} - delta_m^(w)(x)`, where `g^(w)_{x,r}`
and `delta_m^(w)(x)` are both computed from canonical final-resolution
T0/T320 predictions after applying the identical operator, over the
window's own disputed (`a(x) != b(x)`) final-resolution pixels only,
using the complete canonical window plan's stitch denominator. A
selected window with zero disputed pixels remains valid and records a
zero sample count -- never a fabricated value or an omitted row.

No new, incompatible mathematics is introduced anywhere: every
residual is bound directly to its existing production function.

## Raw measurement summary and artifact schema

Per selected window, per residual family, ten raw FP64 fields:
`sample_count`, `finite_count`, `non_finite_count`, `signed_minimum`,
`signed_maximum`, `maximum_absolute_residual`, `mean_absolute_residual`,
`root_mean_square_residual`, `absolute_residual_sum`,
`squared_residual_sum`. The last two exist specifically so
`aggregate_residual_summaries` can be reconstructed **exactly** from
per-window sums (never a separately-computed pooled statistic) -- see
`aggregate_residual_summaries()` in
`src/dfp_phase1b_calibration_artifact.py`. A zero-sample-count family
records `null` for every floating field; NaN/Infinity are never
serialized, and a completed artifact requires `non_finite_count == 0`
everywhere.

`phase1b_conservation_calibration.json` (schema
`talk2dino-dfp-phase1b-conservation-calibration-v1`, version 1) is a
closed-schema document: 23 top-level fields (identity/dependency
hashes, selection digests, device/software/precision records, the
ordered `selected_window_records`, and `aggregate_residual_summaries`)
and 23 fields per selected-window row (the 15 candidate fields plus
`canonical_candidate_index`/`selection_rank`/`selection_key`/
`completed` plus the four residual-family summaries). A complete/final
artifact requires exactly 200 rows, in canonical candidate order, with
every digest independently re-derivable from a supplied candidate
universe -- `validate_calibration_artifact(document,
candidate_universe=...)` performs exactly that re-derivation, never a
mere internal cross-check of hashes the document carries against
itself.

## No hard-coded tolerance, ever

Nowhere in this identity, this schema, or this code does a numeric
conservation tolerance, epsilon, default, safety multiplier,
percentile cutoff, or pass/fail/acceptable/within-tolerance/verdict
field exist. `raw_measurement_summary.forbidden_fields` in the TOML
enumerates the exact forbidden name fragments, and
`validate_calibration_artifact` asserts none of the schema's own field
names resemble one. A future, separate successor authority must bind
an independently-verified calibration artifact and freeze the numeric
tolerance; full Phase 1B execution remains unauthorized until that
authority is committed and independently verified.

## GT isolation

No candidate, selection, or residual-summary field, and no function
signature in either module, accepts a GT/annotation/oracle/harmful/
beneficial-shaped parameter. Selection depends only on candidate
geometry/order; residuals depend only on canonical/masked predictions
and the frozen disputed-pixel definition already enforced by
`compute_signed_contributions`.

## Exclusions

No calibration runner, CLI execution command, SLURM launcher, real
image/dataset/model/checkpoint access, CUDA initialization, PAMR
execution, Gate-2 model fitting, or Phase 1B execution driver is
implemented by this identity or this code. Every test in
`tests/test_dfp_phase1b_calibration_identity.py` and
`tests/test_dfp_phase1b_calibration_artifact.py` runs CPU-only against
hand-built synthetic fixtures.

## Next step

Implement the 200-window conservation-calibration runner against this
identity; generate `phase1b_conservation_calibration.json` on the
GPU/device class intended for Phase 1B; independently verify the
generated artifact; freeze a new successor authority binding the
verified artifact and the numeric conservation tolerance. Full Phase 1B
execution remains unauthorized until that later authority is committed
and independently verified.
