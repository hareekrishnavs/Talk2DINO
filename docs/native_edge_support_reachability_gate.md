# 20-image structural reachability gate

## Parent mechanics20 result

Job `20420785`, commit `b62d023a09b641d0e5cfbf0cbee5207c89021363`:
`decision_output = ALIGNMENT_LIMITED` — 79.27% of directed edges are
undefined (mostly because no other window natively covers both endpoints;
the rest are single-window images with no other view at all), and only
21.65% of currently misclassified rows have any support-defined edge.

## Audit vs. gate

The **audit** (`e12-native-edge-support-audit`) measures — it builds real
graphs on real images and produces a funnel, a set of histograms, and a
`decision_output` classified by its own `classify_reachability` function.
The **gate** (this stage) never measures anything new: it consumes one
already-produced, already-verified audit result and *formalizes* that
result into a roadmap authorization record. It is entirely offline —
no CUDA, no model, no dataset, not even a rerun of the audit's own
classifier logic with different inputs. It calls the audit's classifier
*again*, but only against the exact same funnel the audit already
produced, as a reproduction check, never as a second, independent
threshold authority.

## Exact decision mapping

Locked in `evaluation_identities/e12_native_edge_support_reachability_gate.toml`
`[decision_mapping]`, validated exact (no relaxation, no unknown decision
accepted):

| Parent decision | Roadmap action |
|---|---|
| `REACHABLE` | `PROCEED_TO_ELIGIBILITY_MATCHED_PRUNING` |
| `STRUCTURALLY_UNREACHABLE` | `STOP_STRUCTURAL_CONNECTIVITY_BRANCH` |
| `ALIGNMENT_LIMITED` | `STOP_NATIVE_SUPPORT_ALIGNMENT_LIMITED` |
| `INCONCLUSIVE` | `DO_NOT_PROCEED_INCONCLUSIVE` |

Only `REACHABLE` authorizes the four structural-pruning stages.

## Count/ratio formulas

Every ratio is reconstructed from raw counts, never trusted precomputed
(`src.native_edge_support_reachability_gate.reconcile_aggregate_ratios`):

```
undefined_edge_fraction        = undefined_edges / total_directed_edges
defined_edge_fraction          = support_defined_edges / total_directed_edges
wrong_row_reachability         = misclassified_rows_any_defined / misclassified_rows
support_defined_wrong_fraction = defined_and_misclassified / (defined_and_correct + defined_and_misclassified)
clamped_window_fraction        = clamped_windows / windows
aligned_pair_fraction          = aligned_window_pairs / (aligned_window_pairs + unaligned_window_pairs)
```

Required count invariants (all checked, all fail closed): defined +
undefined = total edges; `undefined_reason_counts` sums to undefined
edges; each GT cross-tab reconciles with `graph_rows - ignored_gt_count`;
`rows_with_unique_least_support + rows_tied_for_least_support =
rows_any_defined`; `clamped_windows + non_clamped_windows = windows`.

## Available cause decomposition

The parent audit's `undefined_reason_counts` distinguishes exactly two
reasons: `single_window_image` and a single combined
`no_exactly_aligned_observer_covering_both_endpoints`. The gate maps these
to the desired 6-category scheme as far as the evidence allows:

- `single_window_no_alternative_view` — available, exact
  (`= undefined_reason_counts.single_window_image`).
- `no_native_observer_cause_not_further_identifiable` — available, exact
  (`= undefined_reason_counts.no_exactly_aligned_observer_covering_both_endpoints`).
- `support_defined` — available, exact (`= funnel.edges_with_observer`).

These three counts sum to `directed_edges` exactly (checked).

## Unavailable cause detail

`multi_window_no_overlapping_alternative`, `candidate_observer_origin_unaligned`,
`source_maps_but_destination_outside_shared_overlap`, and
`other_exact_geometry_failure` are all marked `available: false`. The
audit does not separately track which of these applied to any given
undefined edge, and the gate **never infers this from aggregate window/pair
counts** (`aligned_window_pairs`/`unaligned_window_pairs` are per-image
window-pair statistics, not per-edge cause codes — attributing individual
edge failures to them without per-edge evidence would be a fabrication).
Obtaining finer decomposition would require re-instrumenting and rerunning
the GPU audit, which this offline gate does not do.

## Reachability interpretation

> Native exact-alignment cross-view edge support is geometrically
> unavailable for most directed edges in the mechanics20 sample, and only
> a minority of currently misclassified rows are reachable. Under the
> preregistered mechanics decision, native-support pruning is not
> authorized.

This is a structural/geometric reachability read, not a pruning-efficacy
result. See `limitations` in every gate result for the full required list.

## Why the four structural stages are skipped

`feat: add eligibility-matched one-edge pruning variants`,
`test: add matched-budget graph and replay suite`,
`eval: add paired bootstrap and sample-size lock`, and
`eval: run locked structural-connectivity efficacy pilot` are all marked
`authorized: false`, with the reason **"skipped because the preregistered
mechanics gate did not return REACHABLE, not because efficacy was measured
and found negative."** No pruning variant was implemented or evaluated at
any point in this stage.

## Why no approximate alignment is introduced

The gate never rounds, quantizes, or nearest-matches an unaligned window
pair, and never authorizes nearest-neighbour/bilinear/learned DINO-feature
transport as a substitute alignment strategy. Doing so post-hoc, after
observing that exact alignment produced `ALIGNMENT_LIMITED`, would be
exactly the kind of "rescue the method by changing rules after seeing the
result" this gate is designed to prevent
(`policy.no_approximate_alignment_introduced = true` in the identity).

## Next authorized stage

`eval: add COCO-Object protocol confirmation`.

## Reproduction command

```
module load opencv/4.14.0
source /scratch/haree/venv/talk2dino-a100/bin/activate
python evaluate_native_edge_support_reachability_gate.py \
    --repo-root . \
    --audit-result /scratch/haree/e12_native_edge_support_audit/result-mechanics20-20420785.json \
    --result /scratch/haree/e12_native_edge_support_audit/reachability-gate-mechanics20-20420785.json
```

## Artifact immutability

`select_and_validate_parent_artifact` reads the parent bytes and mtime
once at the start and re-checks both are unchanged immediately after
validation, before returning — the gate can never modify the artifact it
consumes. The parent path is recorded only as a non-deterministic
`audit_result_path_reference` field, kept outside the gate result's
deterministic `content_digest`; the deterministic `parent_artifact.label`
is derived purely from content (`git_commit`/`image_order_digest`), never
from the filesystem path.
