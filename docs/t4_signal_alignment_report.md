# Final T4 signal-alignment report

A CPU-only, read-only analysis of an already-finalized trust/centrality
report (`talk2dino-trust-centrality-report-v4`). It runs no model, no
CUDA, and no dataset inference, and re-derives no T4/trust/centrality
sufficient statistics — it only reads already-computed fields from the
finalized report (and, optionally, a canonical per-class pixel-union
artifact) and reports derived, explicitly-labeled quantities.

## Running the CLI

```sh
python generate_t4_signal_alignment_report.py \
    --trust-report trust_centrality_full_5000_rerun_v4.json \
    --output t4_signal_alignment_report.json \
    --markdown-output t4_signal_alignment_report.md
```

Run with whichever interpreter has this repository's dependencies
installed (e.g. `<python-environment>/bin/python`); no particular
environment path is required by the tool itself.

Add `--canonical-stats PATH` to enable the optional union-weighted proxy
(Section 8 below). The CLI exits `0` on success and `2` with a diagnostic
on `stderr` on any validation failure; it never partially writes an output
file on failure.

## Input contract

`--trust-report` must be a `talk2dino-trust-centrality-report-v4` JSON
document with `final: true`, `complete: true`, and a `trust_centrality`
section with `status: "available"` and schema
`talk2dino-trust-centrality-full-report-v3`. Validation reuses the
existing `validate_report_v4_unit_contract`, `require_full_run_complete`,
and `validate_trust_centrality_section_complete` functions from
`trust_centrality_harness.py` — this module never re-implements a weaker
parallel parser. JSON is loaded strictly: `NaN`/`Infinity` and duplicate
object keys are rejected outright, `bool` is never accepted where an
`int` is required, and every fraction is checked to lie in `[0, 1]`.

The input file and the parsed mapping are never mutated.

`--canonical-stats` (optional) accepts either the historical e10
affinity-oracle sweep artifact shape (`payload.rows`, selecting the row
with `alpha == 0.98` and `steps == 320` by exact typed-field equality —
never by array position) or a strict generic form: a top-level `union`
key holding exactly 171 finite, non-negative numbers indexed by class ID.
If a different canonical-union artifact schema is needed later, add a new
branch to `parse_canonical_stats()` keyed off an unambiguous, explicit
shape marker (never fuzzy detection) — do not fabricate a union when the
real artifact isn't available.

## Fractions versus percentage points

Every quantity in this report that is a probability/rate — `Delta_trust`,
`third_label_fraction`, accuracies, the bootstrap CI — is reported twice:
once as a `_fraction` field in `[-1, 1]` (or `[0, 1]` where non-negative),
and once as a `_percentage_points` display twin (`fraction * 100`). The
fraction is the canonical value; the percentage-point field exists purely
for display convenience and is never a separate measurement.

## Why anchor net is not mIoU

`direct_anchor_net = consensus_correct - dissent_correct` is a count of
**patch anchors**, not pixels. Converting it to a segmentation-metric
(mIoU) impact would require accounting for: how many pixels each anchor's
bilinear interpolation footprint actually touches; whether those pixels
were already correct under the *stitched* prediction (not just the raw
anchor label); each class's union denominator in the mIoU formula; how
close each affected pixel's argmax margin was to a decision boundary; and
nonlocal effects (repairing one anchor shifts the RWR equilibrium of
neighboring anchors through the graph). None of that is recoverable from
anchor counts alone — this report never claims otherwise.

## Why the union-weighted proxy is sign-only

The optional Section 8 quantity `S = sum_c(net_c / U_c)` divides a
patch-anchor count (`net_c`) by a pixel count (`U_c`). The result is not
dimensionally an mIoU-like quantity — it's explicitly labeled a **mixed-
unit sign/alignment proxy**. `S`'s sign and rough magnitude give a
directional signal (are net-positive classes also the classes with small
canonical unions, i.e. classes where a given anchor-count change would
matter proportionally more?) without claiming to predict an actual mIoU
delta.

## Why the two footprint sensitivities are descriptive only

`sensitivity_196` and `sensitivity_830` rescale `S` by two illustrative
per-anchor pixel-footprint sizes: 196 (= 14×14, the average bilinear
interpolation weight footprint of one patch node under DINOv2's patch
size) and ~830 (a maximal joint bilinear-support footprint across the
four neighboring nodes a boundary pixel can draw from). Neither is a
bound on anything, and neither predicts actual mIoU change: pixels within
a node's footprint may already be correct, may carry a different GT label
entirely, or may never cross an argmax decision boundary even if the
anchor's own label changes.

## Why this stops semantic COVER-DR while leaving structural connectivity analysis open

The report's decision section (`STOP_SEMANTIC_CONSENSUS_COVER_DR` when
the evidence supports it) is a conjunction of three measured signals: the
image-level bootstrap 95% CI for strict-T4 `Delta_trust` contains zero,
the image-macro estimate (which does not let a few heavily-sampled images
dominate) is non-positive, and broadening to T4-prime *weakens*
`Delta_trust` while *increasing* the third-label failure rate — refuting
the idea that strict T4 was merely "too strict." Together, this is not
strong evidence that cross-view semantic consensus is a trustworthy
enough signal to drive a repair mechanism.

This conclusion is deliberately narrow:

- it stops semantic-consensus-guided COVER-DR specifically;
- it does **not** claim all graph-topology methods are invalid;
- it does **not** claim cross-view structural edge support is invalid;
- it does **not** claim T4 reversals are necessarily semantically
  incorrect — T4 remains a useful **operator-attributed diffusion
  reversal** diagnostic (it proves the RWR operator changed the label
  relative to unanimous cross-view context, which is a different claim
  than "the change was wrong");
- it explicitly directs the next experiment toward a **matched k11/k12
  connectivity** comparison — isolating whether indiscriminate
  single-neighbor graph pruning helps or hurts under the canonical
  unary/affinity/solver settings, independent of any semantic-consensus
  repair mechanism.

Leave-one-window-out cross-view context is not a set of statistically
independent observations — overlapping crops share most of their pixels,
most of their DINO features, and the same frozen backbone and projection
head. This report is careful never to describe it as independent
replication.
