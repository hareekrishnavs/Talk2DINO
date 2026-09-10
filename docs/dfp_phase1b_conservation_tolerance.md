# DFP Phase 1B Task 6A — Conservation Tolerance (POST-CALIBRATION, PRE-EVALUATION TOLERANCE)

**Identity:** `evaluation_identities/e19_dfp_phase1b_conservation_tolerance.toml`
**Loader / validator:** `src/dfp_phase1b_conservation_tolerance.py`
**CPU-only CLI:** `diagnostics/run_dfp_phase1b_conservation_tolerance.py`
**Successor of:** `e16-dfp-phase1b-conservation-calibration` (additive only — E16 is never edited)

## Why calibration must precede tolerance binding

The numeric size of the FP64 conservation residual
`|sum_component_evidence − margin_change|` is an empirical property of
the exact masked-forward decomposition on this hardware/software/
precision stack. It cannot be known before running the calibration.
E16 therefore froze a deterministic 200-window **measurement** protocol
and closed artifact schema and *explicitly defined no tolerance*
(`e16…[residual_definitions].tolerance_defined_here = false`,
`e16…[measurement_versus_tolerance_boundary].later_successor_authority_required_to_bind_verified_artifact_and_freeze_tolerance = true`).
This document/identity is exactly the later step E16 deferred: it binds
the completed, independently-verified calibration artifact by content
hash and freezes the one scalar tolerance.

## What was examined, and when it was frozen

The tolerance was selected **after** examining the verified calibration
artifact `phase1b_conservation_calibration.json`
(SHA-256 `f065b143933cf7881e5a31dd663783719a684e6d430e9e7b2bb314ed7e792d22`,
200 rows, `complete = true`, `final = true`, official E16
`verify-artifact` PASS). It was frozen **before** the 500-image Phase 1B
provenance diagnostic and **before** any 5,000-image evaluation — no
such result exists or is authorized. This is a **POST-CALIBRATION,
PRE-EVALUATION TOLERANCE**.

## Governed residual family and units

There is **exactly one** governed family and **one scalar** tolerance.
This is read directly from committed authority, not chosen for
convenience:

- `e15-dfp-phase1b-artifacts` `[stage_a_schema]` stores
  `conservation_residual = "float64"` (one value per pixel) and its
  relational rule is *"`sum_component_evidence ~= margin_change`
  (within **the** separately bound calibration tolerance …)"* —
  singular.
- Its committed Python mirror
  `src/dfp_phase1b_artifact_schema.py::validate_stage_a_flat_arrays(*, conservation_tolerance: float)`
  takes **one scalar `float`** and compares it against
  `residual = |sum_component_evidence − margin_change|` — the `rho_g`
  residual — failing only when `residual > conservation_tolerance`.
- `e16…[residual_definitions].primary_measurement = "rho_g"`.

| family | formula | recorded | governed |
|---|---|---|---|
| `rho_g` | `sum_r g^(w)_{x,r} − delta_m^(w)(x)` | yes | **yes** |
| `rho_M` | `sum_r M[r,w,j] − 1` | yes | no |
| `rho_0` | `sum_r Y0^(r)[w,j,c] − Y0[w,j,c]` | yes | no |
| `rho_320` | `sum_r Y_320^(r)[w,j,c] − Y_320[w,j,c]` | yes | no |

**Units of the governed residual:** final-resolution class-score margin
units (the same units as `margin_change = m320 − m0`) — a real-valued,
dimensionless model-score scale, no physical unit. Domain:
final-resolution disputed pixels of the selected window's own support
where `a(x) ≠ b(x)`.

`rho_M`, `rho_0`, `rho_320` are recorded per E16 so the calibration run
is independently auditable at each decomposition stage; **no committed
authority applies a numeric tolerance to them and this identity freezes
none.** Their reconstructed maxima are recorded in E19's
`[derivation_inputs]` as provenance only.

## The exact deterministic rule

For the governed family, from the 200 verified rows' lowest-level
persisted maxima:

```
M = max over all 200 selected_window_records of record["rho_g"]["maximum_absolute_residual"]
```

reconstructed independently and cross-checked against the artifact's own
`aggregate_residual_summaries` (row-level and aggregate must agree
exactly). Then:

```
if M == 0:            tolerance = 0
elif M is 2**e:       tolerance = M                 # exact power of two: no doubling
else:                 tolerance = 2 ** ceil(log2(M))
```

implemented exactly via `math.frexp` / `math.ldexp` (no floating-point
boundary ambiguity; exact-power-of-two and immediately-adjacent
float inputs are unit-tested). No percentile, fitted distribution,
manual multiplier, confidence interval, decimal rounding, or manually
typed constant is used, and **no** 500-image or 5,000-image evaluation
information is used.

## Reconstructed calibration maxima and the resulting tolerance

| family | reconstructed max abs residual | samples |
|---|---|---|
| `rho_M` | `7.771561172376096e-15` | 204800 |
| `rho_0` | `3.4416913763379853e-15` | 4221952 |
| `rho_320` | `1.3007687971189341e-06` | 4221952 |
| **`rho_g` (governed)** | **`5.889683609150431e-07`** | 7887805 |

```
M            = 5.889683609150431e-07   (not an exact power of two; frexp exponent -20)
2**-21       = 4.76837158203125e-07    (previous power of two, strictly < M)
tolerance    = 2**-20 = 9.5367431640625e-07
M <= tolerance          ✓
tolerance / M ≈ 1.619   (< 2)
```

## Inclusive acceptance semantics

The frozen conservation acceptance relation is:

```
absolute_residual <= conservation_tolerance
```

**inclusive** — strictly identical to
`src/dfp_phase1b_artifact_schema.py`, which fails a pixel only when
`residual > conservation_tolerance`.

## What this is, and what it is not

This is an **operational numerical-conservation threshold**: it checks
that the exact component-evidence decomposition reproduces the margin
change to within a bound never smaller than the largest FP64 residual
the calibration actually observed. It is **not** a segmentation-quality
threshold, not an accuracy metric, and not a method-selection
criterion. **Passing conservation does not imply good mIoU, good
Boundary IoU, good provenance quality, or deployability.**

## Boundaries (authoritative — `[authorization_boundary]`)

- Task 6A performs **no inference**, loads no model weights,
  initializes no CUDA, accesses no COCO-Stuff image or annotation, and
  submits no SLURM job.
- Full Phase 1B execution, the **500-image Phase 1B diagnostic**, and
  the **5,000-image evaluation** all remain **unauthorized**.
- No conservation-tolerance revision may ever be made from a future
  evaluation result without creating a **separately disclosed
  successor protocol**.
