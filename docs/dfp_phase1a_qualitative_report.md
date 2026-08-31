# DFP Phase 1A Step 11 — deterministic qualitative reporting

Deterministic, predeclared qualitative-example selection and rendering
infrastructure for future DFP Phase 1A diagnostic runs. This step
produces no real COCO-Stuff/ADE20K conclusion: every fixture in its own
test suite and every CLI smoke path uses synthetic data only.

## Why selection is deterministic

Manual example curation after seeing predictions is cherry-picking.
Every example this step could ever select is instead determined purely
by: (1) a fixed category-eligibility test applied to already-verified
Step 3 analysis outputs, and (2) a SHA-256 hash of a frozen namespace,
dataset name, category name, and canonical image ID — never Python
`hash()`, never an RNG, never wall-clock time, never filesystem order,
never a GT-derived magnitude. Re-running selection on the same verified
inputs always reproduces the identical selected IDs.

## Reporting authority

`evaluation_identities/e13_dfp_phase1a_qualitative_report.toml` (loaded/
validated by `src/dfp_phase1a_qualitative_selection.py`) is the closed,
storage/selection/rendering-only authority. It binds, and re-verifies on
every load against the live files: the frozen protocol identity's
SHA-256, Step 10's artifact-schema identity SHA-256, the Boundary IoU
authority supplement's SHA-256, and both frozen 500-image subset
manifest SHA-256 values. It defines no scientific quantity — the
dispute-category formulas are pinned verbatim to the protocol's own
`[dispute_categories]` section and cross-checked against the live
protocol identity every time this identity loads.

## Category definitions (all reused from Step 3, never redefined)

| Category | Source |
|---|---|
| `harmful_washout` | `DisputeResult.harmful` |
| `beneficial_correction` | `DisputeResult.beneficial` |
| `third_label_transition` | `DisputeResult.third_label` |
| `disputed_boundary` / `disputed_interior` | the dispute mask split by Step 3's own pinned `_official_mask_to_boundary` |
| `oracle_reverted` | the harmful-reversion pixels `OracleMaps.o_p` reverts (`DisputeResult.harmful`) |
| `horizon_persistence_reverting` | `PersistenceMaps[primary_persistence_horizon]` reversion pixels |
| `revert_set_fragmentation` | `compute_reversion_coherence` over the primary-horizon reversion mask |

Eligibility is exact: `at_least_one_category_pixel_in_the_valid_region`
— no pixel-count threshold is ever introduced. `a` = stitched T0/unary
lowest-index-argmax winner, `b` = stitched T320 lowest-index-argmax
winner, matching the frozen protocol's own tie rule exactly.

## Selection formula

```
selection_key = SHA256(namespace + "\0" + dataset_name + "\0"
                        + category_name + "\0" + canonical_image_id)
```

Eligible IDs (traversed in the frozen subset manifest's own canonical
order) are sorted by `(selection_key, canonical_image_id)`; the first K
are selected. `K`, `namespace`, and the separator are all bound in the
reporting identity's `[selection_policy]`, never hardcoded ad hoc.

## Shortage and zero-coverage behavior

If fewer than K images are eligible for a category: all eligible images
are selected, never more, never filled from another category, never
duplicated, and the category's `shortage_status` is recorded as
`"shortage"` (not falsely `"complete"`). If zero images are eligible,
`shortage_status` is `"zero_coverage"` with an empty `selected_ids` list.
An image may legitimately appear in multiple categories — the frozen
policy (`allow_cross_category_duplicates = true`) permits this and
selection never silently enforces cross-category uniqueness.

## Manifest and bundle bindings

The **selection manifest** (`talk2dino-dfp-phase1a-qualitative-selection-v1`)
binds the reporting/protocol/artifact-schema/subset-manifest SHA-256
values, the source result hash, per-category requested/eligible/selected
counts and IDs, every selection key, category pixel counts (audit-only),
shortage status per category, and a `selection_trace_digest` sufficient
to detect any change to the outcome without re-running selection. The
**bundle manifest** (`talk2dino-dfp-phase1a-qualitative-bundle-v1`) binds
the reporting identity hash, the selection manifest's own hash, the
source result hash, the panel order, the palette digest, and one entry
per rendered panel (encoded PNG SHA-256 + decoded pixel-content SHA-256 +
dimensions). Both are strict JSON (duplicate-key rejection, no NaN/
Infinity, exact types, closed key sets, deterministic serialization) via
the same primitives Step 10 established.

## GT's analysis-only role

Ground truth flows into exactly one place: `analyze_phase1a_image` (Step
3) and the boundary/reversion-coherence helpers built directly on top of
its outputs, all for post-hoc category labeling and panel visualization
only. It never reaches S0/S320 prediction construction, the graph/
recurrence/component/membership machinery, or model input — enforced by
a poison test in `tests/test_dfp_phase1a_qualitative_report.py` that
monkeypatches every CUDA/model/checkpoint entry point and proves a
complete selection+analysis run never touches any of them, plus a
source-level forbidden-call audit.

## Canonical palette and panels

No authoritative COCO-Stuff-171/ADE20K-150-scale semantic palette exists
in this repository, so this step defines a deterministic reporting
palette: for class index `c`, `HSV(hue=(c * 137.507764°) mod 360°,
saturation=0.65, value=0.95)` converted to sRGB uint8 via `colorsys` —
lowest-class-index-first, no RNG, never adapted after viewing results.
Ignore/background colors and the 8 category overlay colors are fixed,
distinct RGB triples bound in `[palette_binding]`. Component-diagnostic
colors (if rendered) are a separate `SHA256(namespace + "\0component\0" +
component_id)`-derived color, independent of chunk size, dictionary
order, or `PYTHONHASHSEED`, and are never confused with a semantic class
color. The panel order (`rgb`, `gt`, `unary_t0`,
`persistence_primary_horizon`, `propagated_t320`, `dispute_categories`,
`oracle_reverted`, `reversion_coherence`) is frozen in `[panel_binding]`
and never added to or trimmed based on how an example looks.

## Rendering determinism

RGB panels resize with bilinear interpolation; every categorical/label
panel resizes with nearest-neighbor only (never antialiased — that would
invent fractional/blended class IDs). PNGs are written with explicit
compression settings and mode, no `pnginfo` (so no timestamp/tEXt/tIME
chunk is ever embedded), via write-temp-then-`os.replace`. Two distinct
digests are always recorded: the **encoded PNG SHA-256** (bytes on disk)
and the **decoded pixel-array SHA-256** (the raw RGB content after
decoding) — the decoded digest is what actually distinguishes scientific
rendering content from encoder/compression-level differences; both are
confirmed to match across repeated writes and across different
compression levels in the test suite. Captions are strict JSON/HTML
metadata only — no raster text is rendered, avoiding any font/
environment dependency.

## Atomic installation

`install_qualitative_bundle` builds the complete bundle (selection
manifest, bundle manifest, every panel PNG) in a fresh sibling staging
directory, fully self-verifies it (structural validation, cross-file
hash checks, decode-and-re-hash every panel, exact expected file set),
then atomically renames the staging directory into place. It rejects
cleanly if the destination already exists — no independently-hardened
overwrite policy exists for this bundle kind in this step, matching
Step 10's own atomicity discipline. On any failure the destination is
absent and no staging directory remains.

## Verifier commands

CPU-only; portable placeholders below. Every command takes explicit
input artifact paths or a synthetic fixture-bundle directory — none
resolves or scans a real dataset root.

```
python3 verify_dfp_phase1a_qualitative_report.py select --fixture-bundle <dir> --dataset coco_stuff --output <selection.json>
python3 verify_dfp_phase1a_qualitative_report.py verify-selection --selection-manifest <selection.json>
python3 verify_dfp_phase1a_qualitative_report.py render --fixture-bundle <dir> --selection-manifest <selection.json> --dataset coco_stuff --output-dir <bundle_dir>
python3 verify_dfp_phase1a_qualitative_report.py verify-report --bundle <bundle_dir>
```

Every domain error, `OSError`, and JSON `ValueError` at the CLI boundary
prints one concise line and exits `2` with no traceback;
`KeyboardInterrupt`/`SystemExit`/`MemoryError` are never caught.

## No real qualitative results produced yet

This step is CPU/synthetic reporting infrastructure only. It implements
no model or dataset execution, no prediction generation, no Phase 1B
masked propagation, no final DFP selector, no adaptive tuning, and no
real result analysis. No real COCO-Stuff/ADE20K image ID is committed
anywhere in this step's source, tests, or documentation.
