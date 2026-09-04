# DFP Phase 1B Artifact Schemas

Governed by `evaluation_identities/e15_dfp_phase1b_artifacts.toml`
(binds `evaluation_identities/e15_dfp_phase1b_protocol.toml` by exact
SHA-256) and loaded/validated by `src/dfp_phase1b_artifact_schema.py`.
This is a **distinct schema** from the completed Phase 1A v2
per-image-stats schema — it neither reuses nor mutates it, and no
cross-feeding between the two is ever accepted.

This task freezes schema and validation only. **No writer or execution
driver is implemented here** — see `docs/dfp_phase1b_protocol.md`'s
*Future Gate-3 work* for what comes next.

## Stage A: GT-free provenance feature artifact

Schema name `talk2dino-dfp-phase1b-stage-a-features-v1`, version 1.

**Manifest fields**: canonical image IDs, selected-IDs digest, image
count (500), dispute count per image, dispute offsets
(`image_count + 1`), component/window counts per image, total processed
windows, per-image max/mean conservation residual, per-image projected
compute fields, deterministic fold assignment + digest, checkpoint
prefix, `complete`.

**Flat disputed-pixel members** (native-endian, no object/structured
dtype): `image_row_index`, `final_pixel_linear_index`, `unary_class_a`,
`propagated_class_b`, `t16_label`, `t32_label`, `t64_label` (int64);
`m0`, `m320`, `margin_change`, `eta_net`, `text_cosine_ab`,
`positive_evidence_sum`, `negative_evidence_sum`,
`sum_component_evidence`, `conservation_residual`, `phi`,
`top_positive_evidence`, `top1_positive_share` (float64);
`n_positive_components`, `k_star`, `top_positive_component_id` (int64).
int64/float64 uniformly — the smallest-safe-dtype exception requires
proving exact bounds first, which this schema-freezing task does not
do.

**Relational validation** (`validate_stage_a_flat_arrays`,
`validate_stage_a_offsets`): offsets start at 0, nondecreasing, end at
the flat row count, and their diffs equal the per-image dispute counts;
`a != b`; `m0 <= 0`; `m320 > 0`; `margin_change = m320 - m0`;
`eta_net = m320 / margin_change`; `positive_evidence_sum > 0`;
`negative_evidence_sum <= 0`; `1 <= k_star <= n_positive_components`;
`0 < phi <= 1`; `0 < top1_positive_share <= 1`;
`sum_component_evidence ≈ margin_change` — checked against a
**caller-supplied** `conservation_tolerance` (no default; the function
raises `TypeError` if omitted), sourced from the separately bound
`phase1b_conservation_calibration.json`, never a guessed constant.

Stage A contains **no GT-derived array**. `validate_stage_a_member_names`
enforces the exact closed member set and additionally denylists any
member name containing a GT-shaped fragment (`gt`, `harmful`,
`beneficial`, `third_label`, `oracle`, `boundary_label`) as
defense-in-depth beyond exact-set closure.

## Stage B: evaluation-label artifact

Schema name `talk2dino-dfp-phase1b-stage-b-labels-v1`, version 1. Binds
the exact Stage-A digest (`validate_stage_b_digest_binding`).

Target-category codes (frozen exactly): `harmful=0`, `beneficial=1`,
`third_label=2`. Per-pixel members: `gt_class`, `target_category`,
`binary_eligible`, `harmful_binary_label` (`1`/`0`/`-1` sentinel for
not-binary-eligible — third-label pixels are always the sentinel, never
`0` or `1`), `boundary_interior_flag`, `fold_id`, `training_selection_flag`
(all int64). Diagnostic-prediction members
(`out_of_fold_cheap_control_risk`, `out_of_fold_provenance_augmented_risk`)
are **absent** until all five folds are complete — a Stage-B artifact
with fewer folds fitted must not include them at all
(`validate_stage_b_member_names(..., diagnostic_predictions_present=False)`).

Relational validation: `harmful + beneficial + third_label = disputes`;
`binary_eligible = harmful + beneficial`; third-label pixels never
carry a `0`/`1` binary label; `fold_id` must agree with the image-level
frozen fold assignment for the same image
(`validate_stage_b_per_pixel_arrays`).

## Fixed prediction-statistic member naming

12 frozen variants, in this exact order: the original ten — `t0`,
`t16`, `t32`, `t64`, `t320`, `oracle_o_p`, `persist_h32_full`,
`persist_h32_harmful_only`, `persist_h32_beneficial_only`,
`persist_h32_third_label_only` — followed by two appended PAMR
baseline variants, `t0_pamr` and `t320_pamr` (PAMR applied to DFP's own
T0/T320 score tensors, respectively, with identical parameters/RGB
input/output geometry/argmax convention for both — see
`docs/dfp_phase1b_protocol.md`'s *RGB/local-boundary baseline* section
for the full binding). No PAMR image tensor, raw RGB data, or raw
prediction map is stored by this schema — only int64 confusion-count
sufficient statistics, exactly as for every other variant.

Semantic: `semantic_{variant}_{stat}` for
`intersection`/`union`/`predicted_pixels`/`ground_truth_pixels`
(12 variants × 4 stats = **48 members total**). Boundary IoU:
`boundary_{variant}_{stat}` for
`boundary_intersection`/`boundary_union`/`predicted_boundary_pixels`/
`ground_truth_boundary_pixels` (12 variants × 4 stats = **48 members
total**), bound to the e14 authority. Both: shape
`[image_count, class_count]`, dtype int64.
`validate_semantic_confusion_arrays` / `validate_boundary_confusion_arrays`
require the member set to be **exactly** these 48 names each (rejecting
any missing member, e.g. an incomplete pre-repair 10-variant/40-member
set, and any unexpected extra member, e.g. a 13th unapproved variant),
check the union-relation exactly, and that every one of the 12 variants
shares identical per-class ground-truth (boundary) pixel counts —
including the two PAMR variants, which must report the same GT/GT-
boundary counts as every other variant since they evaluate against the
same ground truth.

## Checkpoint/resume schema (invariants only)

Append-only generation; source-manifest-prefix agreement; no
skip/duplicate/reorder; Stage A and Stage B carry distinct completion
states; Stage B cannot start until Stage A is complete and sealed;
Stage-B checkpoint binds the Stage-A final digest; exact
`complete`/`final` boolean combinations; zero-progress arrays valid
without `np.stack([])`; each installed generation self-verifies before
atomic pointer replacement; orphaned generations never survive a
handled failure; resume is byte-identical to uninterrupted execution;
final result installation is CPU-only and never rebuilds the model.
**No writer or execution driver implements these invariants yet** —
this document and `src/dfp_phase1b_artifact_schema.py` freeze the
contract only.

## Strict validation guarantees

Both TOML identities (`src/dfp_phase1b_identity.py`,
`src/dfp_phase1b_artifact_schema.py`) use closed-mapping, exact-type,
exact-value validation generated mechanically from the committed
files — every scalar/list leaf is pinned, not merely checked for
non-emptiness. Array validation rejects: wrong rank/shape/dtype,
non-native byte order, object/structured dtype, NaN/Infinity,
duplicate/traversal/directory ZIP member names, and any Phase-1A-v2-
shaped or GT-shaped member name appearing where it shouldn't. No
scientific computation occurs anywhere in either module.

## Limitations

- No Phase 1B result exists; no real Boundary IoU result exists.
- No PAMR result exists for `t0_pamr` or `t320_pamr` on the frozen
  500-image Phase 1B run; PAMR was bound read-only, not executed.
- No selector is frozen.
- No GPU run is authorized.
- The 5,000-image evaluation remains untouched.
- Phase 1B provenance has not yet been implemented.
