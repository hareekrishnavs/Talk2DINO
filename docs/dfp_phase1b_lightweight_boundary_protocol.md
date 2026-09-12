# DFP Phase 1B: Boundary-Aware Late-Flip Preservation (BALP) — E22 protocol

## 1. Why exact provenance was discontinued

The exact masked-forward provenance runner (E20/E21, `src/dfp_phase1b_diagnostic_execute.py`
and its companions) is **discontinued**. An independent diagnosis established that its
multi-window Stage-A semantics are ambiguous: it derives the disputed-pixel `a(x)`/`b(x)`
labels from an image-global, all-windows-blended canonical field, while it computes the
`m0(x)`/`m320(x)` margins used for conservation checking from a window-local-only
reconstruction of the same field. Neither E15 nor E16 states which domain is authoritative,
and resolving the ambiguity correctly would require a new calibration cycle. Rather than
attempt that, this identity (E22) freezes a replacement, lightweight method: BALP.

The five uncommitted runner files (`src/dfp_phase1b_diagnostic_execute.py`,
`diagnostics/run_dfp_phase1b_diagnostic.py`, `dfp_phase1b_diagnostic_coco_stuff_500.sbatch`,
`docs/dfp_phase1b_diagnostic_execute.md`, `tests/test_dfp_phase1b_diagnostic_execute.py`)
are **abandoned, deferred research artifacts**. They are not dependencies of E22 and were
left byte-identical throughout this identity's creation.

## 2. What motivated BALP

A completed 500-image Gate-1 analysis (`gate1_analysis.json`, content-hash bound in
`[motivating_evidence]`) found:

- `mIoU(T0) = 29.358`, `mIoU(T64) = 31.167`, `mIoU(T320) = 31.207` — T64 already recovers
  almost all of T320's gain over T0.
- `G_recover = +2.545pp`, `G_oracle|unary = +4.394pp` — real headroom exceeds what T320
  realizes.
- Harmful late-stage "washout" (pixels that regress between T64 and T320) is **97.65%**
  boundary-concentrated, far above the frozen 70% reframe threshold — the boundary-reframe
  trigger **FIRED**.

This motivates *selectively preserving* stable pre-T320 (specifically T64) predictions near
predicted semantic boundaries, instead of trusting T320 everywhere. It does **not** authorize
using oracle labels, exact provenance, component masks, or GT at inference — BALP uses only
the same canonical (non-oracle) horizon score fields DFP Phase 1A already produces.

## 3. BALP uses canonical horizon outputs only

BALP is a deterministic, inference-time **arbitration rule** over `S_h(q, x)`, the canonical
evaluator-bound class-score field at final `ori_shape` resolution for horizon
`h ∈ {0, 16, 32, 64, 320}` — produced by the same model forward, same graph, same single k12
320-step recurrence, existing horizon capture, existing stitching, and existing final
crop/bilinear-rescale operator. **Exactly one recurrence pass per image** exposes all five
horizons; BALP never re-runs the model, graph, or recurrence per horizon or per candidate.

`y_h(x) = argmax_q S_h(q,x)`, using the repository's existing lowest-class-index tie rule
(`exact_argmax_ties_select_lowest_class_index`, frozen and sourced from E13's own
`[dispute_protocol].tie_rule` — corrected from an earlier, nonexistent `[oracle_definitions]`
citation; the tie-rule *value* was never wrong, only its section citation, and the loader now
independently re-verifies the citation against E13's real bytes on every load).
PAMR is not part of the bound baseline: it is explicitly prohibited from DFP's own
graph/recurrence pipeline and, where it appears elsewhere, is a separate historical
comparator, never part of `S_h`.

## 4. Boundary, stability, late-flip, arbitration

Predicted (GT-independent), 8-connected boundary:

```
B(y) = {x : some in-image 8-connected neighbor x' has y(x') != y(x)}
B_union = B(y_64) ∪ B(y_320)
D_r(B_union) = {x : min_{z in B_union} ||x - z||_∞ <= r}       (Chebyshev dilation, radii [0,1,2,4])
```

Stability and late flip, over `S = [0, 16, 32, 64]`:

```
stable_s(x) = true iff y_h(x) identical for all h in suffix(s)
late_flip(x) = [y_64(x) != y_320(x)]
P_{s,r}(x) = stable_s(x) AND late_flip(x) AND [x in D_r(B_union)]
```

Arbitration (the preserved label is always the stable T64 label — never a blend):

```
S_BALP_{s,r}(:,x) = S_64(:,x)   if P_{s,r}(x)
                   = S_320(:,x) otherwise
y_BALP_{s,r}(x) = argmax_q S_BALP_{s,r}(q,x)
```

This is provably equivalent to selecting `y_64` on the mask and `y_320` elsewhere — the
identity loader independently re-verifies this relational equivalence rather than trusting
the argmax formula alone. No blending coefficient, confidence threshold, temperature,
learned weight, random choice, component feature, or adaptive iteration count exists
anywhere in BALP.

## 5. The exact 16-candidate family

`(s, r) ∈ {0, 16, 32, 64} × {0, 1, 2, 4}`, 16 candidates, IDs `balp_s{s:03d}_r{r:02d}`,
ordered row-major (`s` outer ascending, `r` inner ascending):

```
balp_s000_r00 balp_s000_r01 balp_s000_r02 balp_s000_r04
balp_s016_r00 balp_s016_r01 balp_s016_r02 balp_s016_r04
balp_s032_r00 balp_s032_r01 balp_s032_r02 balp_s032_r04
balp_s064_r00 balp_s064_r01 balp_s064_r02 balp_s064_r04
```

Plus 5 canonical controls (`T0, T16, T32, T64, T320`); `T320` is the deployment baseline.
No candidate may be added after observing results without a new successor authority.

## 6. Development / confirmation separation

The frozen `coco_stuff_500` subset (500 images) is deterministically split, from canonical
image IDs alone (namespace `talk2dino-dfp-phase1b-balp-development-split-v1`, SHA-256,
sorted by `(selection_key, canonical_image_id)`), into:

- **development**: first 350 (ranks 0–349) — used for candidate selection.
- **internal confirmation**: remaining 150 (ranks 350–499) — evaluated **exactly once**,
  after development-only selection is finalized, and never used to choose or re-rank a
  candidate.

**The confirmation split is an internal check, not an independent dataset or untouched
external benchmark.** The eventual 5,000-image COCO-Stuff evaluation will include these
same 500 images and must disclose that development/confirmation used a frozen subset of the
same validation universe. No other image (of the remaining 4,500) is accessed anywhere in
this identity.

## 7. Candidate selection and rejection

Eligibility (development images only): schema-valid complete output, no GT-isolation
violation, finite metrics, development mIoU strictly exceeds T320, no recurrence/model
duplication, candidate in the frozen 16-candidate family.

Primary ranking: highest full-precision development dataset-level mIoU → smallest
preservation fraction → smaller radius → longer stability requirement (`s=0` before `16`
before `32` before `64`) → lexicographically smallest candidate ID.

The one frozen winner (if any) is evaluated exactly once on confirmation. Acceptance
requires confirmation gain vs. T320 of **at least +0.10pp**, finite metrics, no GT-isolation
violation, and passing operational gates. Otherwise: `NO_BALP_CANDIDATE_PROCEEDS`. The grid,
radii, stability requirement, or candidate choice may never be altered after seeing
confirmation results without a new successor authority.

## 8. Operational limits and no GT at inference

The deployed candidate reuses one model forward, one graph, one 320-step recurrence, and
existing horizon snapshots; it performs no masked-forward propagation, component
construction, pullback/membership computation, iterative solve, or CPU/GPU synchronization
in the scientific path, and introduces no learned parameters. **GT is never available to any
part of prediction** (horizon scores, boundaries, stability, late-flip, preservation masks,
candidate score fields/predictions, or timing control flow) — only to post-hoc confusion
accumulation and metric computation. A poisoned-GT verification is required before any
implementation may be trusted.

Paired operational acceptance (same GPU/software/order/checkpoint/warm-up), with every
metric's exact formula, comparison operator, and threshold frozen (never left to
implementation convention):

```
total_wall_time_overhead_pct          = 100 * (candidate_total_wall_time_ns - baseline_total_wall_time_ns) / baseline_total_wall_time_ns          <= 15.0
mean_per_image_latency_overhead_pct   = 100 * (candidate_mean_per_image_latency_ns - baseline_mean_per_image_latency_ns) / baseline_mean_per_image_latency_ns  <= 15.0
throughput_reduction_pct              = 100 * (baseline_throughput_images_per_second - candidate_throughput_images_per_second) / baseline_throughput_images_per_second  <= 15.0
peak_allocated_vram_increase_pct      = 100 * (candidate_peak_allocated_vram_bytes - baseline_peak_allocated_vram_bytes) / baseline_peak_allocated_vram_bytes  <= 10.0
peak_reserved_vram_increase_pct       = 100 * (candidate_peak_reserved_vram_bytes - baseline_peak_reserved_vram_bytes) / baseline_peak_reserved_vram_bytes    <= 10.0
```

Every comparison is the **inclusive** `<=` (never `<`); every baseline denominator must be
strictly positive (a zero or negative baseline measurement is rejected, never silently
divided-by-zero or treated as a pass); every value must be a finite float64 (NaN, ±Infinity,
bool, string, and null are all rejected). **A negative result is a valid, meaningful
outcome** — it means the candidate was faster, had higher throughput, or used less memory
than the baseline, not an error. Display rounding never affects acceptance; the full-precision
value is compared against the threshold first. The same-job T320 control is the authoritative
denominator; the historical ~1h18m runtime is contextual only. Host RSS is recorded (absolute
difference always; percentage difference only when the baseline is strictly positive) with
**no pass/fail threshold** — E22 authorizes none.

## 9. Checkpoint safety and artifact write policy

The development checkpoint (350 images) freezes exact, mechanically checkable invariants —
not merely descriptive booleans. Every checkpoint document has an **exact closed top-level
key set** (`schema_name`, `schema_version`, `identity_bindings`, `progress`,
`aggregate_confusion`, `per_image_reconstruction_statistics`, `complete`, `final`,
`final_result_binding` — no more, no fewer), and every nested mapping reached within it
(`progress`, `identity_bindings`, each control/candidate's `per_image_reconstruction_statistics`
block, and `final_result_binding` when present) is closed the same way. **Profiling, timing,
debug, and telemetry fields can never appear inside a scientific checkpoint document** — an
extra key such as a leaked `profiling_leak`/`timing_ns`/`gpu_peak_vram_bytes`/`debug` field is
rejected exactly like a missing required one, never silently ignored. Within one checkpoint:
`completed_image_ids` must be exactly the frozen development split's first
`next_development_index` IDs (no reorder/skip/duplicate); candidate and control keys must
appear in exactly their frozen order; every aggregate confusion matrix must equal the
elementwise sum of that control/candidate's persisted per-image confusion matrices;
`final == true` requires `complete == true` and a valid `final_result_binding`; an incomplete
checkpoint can never carry one. Across generations: the older checkpoint's completed IDs must
be an exact, byte-identical prefix of the newer one's; identity bindings, candidate/control
ordering, and every already-persisted per-image value must be unchanged; the newer aggregate
must equal the older aggregate plus exactly the newly appended images' contributions. A fresh
run and a checkpoint-then-resumed run of the same population must produce byte-identical
final results.

Artifact writing (policy only — no writer is implemented by this authority) reuses the
already-proven E15/E20 staged-write pattern: every member of one logical generation is
written into a uniquely-named staging area on the *same filesystem* as its final destination,
flushed and closed, self-verified (schema + relational + sibling-digest checks) entirely
before installation, then installed by exactly **one atomic operation** — a directory rename
for multi-file artifacts (development result, selection artifact, profiling artifact, the
future full-evaluation result), or a same-directory `os.replace` for a single replaceable
pointer file. Unguarded sequential per-file final renames are explicitly forbidden.

**Pre-install and post-install verification are two separate, both-mandatory requirements —
neither replaces the other.** After the atomic rename/replace completes, the **installed**
destination path — never the (now-gone) staging path — must be reopened from disk, and the
same public artifact validator run against those freshly-read installed bytes; member hashes
and sibling-digest bindings are recomputed from the installed bytes, not carried over from
staging. Success is reported only once this post-install verification passes; a post-install
failure is reported as an installation failure and is never silently treated as success. No
downstream state may advance ahead of that verification — a current-generation pointer, a
checkpoint's `final`/`final_result_binding`, or a selection artifact's selected-candidate
binding must never be set until the artifact they point to has been reopened and reverified.
This mirrors E20's own `atomic_write_stage_a_npz`, which reloads and re-validates the
installed NPZ after `os.replace`; if that reload-verify fails, E20 performs **no automatic
cleanup** — the newly-installed (invalid) file is left on disk for forensic inspection, and
the exception propagates. E22's policy states this precisely rather than inventing a stricter
deletion/quarantine behavior E20 does not actually have.

**E22 grants no overwrite authority** — every multi-file artifact installs only into an absent
destination, immutable checkpoint generations are never overwritten, and post-install
verification itself grants no overwrite either. On any failure before the final atomic step,
the prior valid state is left untouched, staging residue is removed, and no partial generation
ever becomes current; a rollback-cleanup failure must itself be reported, never hidden.

## 10. No claims made here

This identity makes **no claim that BALP improves anything** — no candidate has been run,
selected, or measured. It makes **no claim that BALP explains, reproduces, or approximates
exact component-level provenance** — it is a different, simpler method built only on
canonical (unmasked) horizon fields. It **authorizes no inference, no GPU execution, no
candidate selection, and no full 5,000-image evaluation** — all of those require a
separately authorized, later task, after implementation, independent verification, a
completed 350-image development result, verified selection, a one-time 150-image
confirmation, and a passing operational benchmark.
