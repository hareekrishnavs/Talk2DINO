# DFP Phase 1A Step 10 — checkpoint/artifact infrastructure

Storage-only infrastructure for future DFP Phase 1A diagnostic runs on
the frozen COCO-Stuff-500 and ADE20K-500 subsets. This step implements no
diagnostic driver, no dataset access, no model construction, and no
scientific quantity: D, Q_w, R_w, horizons, recurrence, components,
memberships, compute formulas, oracle definitions, gates, and metrics all
remain exactly as already frozen by
`evaluation_identities/e13_dfp_phase1a_protocol.toml` and by Steps 3-9.

## Schema authority

`evaluation_identities/e13_dfp_phase1a_artifacts.toml` (loaded/validated by
`src/dfp_phase1a_artifact_schema.py`) is the closed, storage-only schema
authority. It binds:

- the frozen protocol identity's SHA-256;
- both frozen 500-image subset-manifest SHA-256 values;
- the matched-k11-k12, RWR, ADE20K-source, and ADE20K-evaluator parent
  identity SHA-256 values (re-checked against the *live* protocol
  identity's own bindings, not merely format-checked);
- the bridge checkpoint weights SHA-256;
- the checkpoint/per-image-stats/result schema names and versions;
- the exact NPZ member-name/dtype lists;
- path policy and atomic-write policy statements.

`validate_static_configuration()` re-hashes every one of these files
against the live repository content every time it runs -- a
syntactically valid but stale or wrong hash is always rejected.

## Hash/binding chain

Every checkpoint binds relationally (not merely format-checked) to:
artifact-schema identity SHA-256, protocol identity SHA-256, subset
manifest SHA-256, selected-IDs digest, bridge checkpoint SHA-256, and
`required_ancestor_commit`. Every per-image-stats manifest binds to its
owning checkpoint (run_mode, class_count, class_names_digest, image
count, and the exact completed-ID list) and to its own NPZ (filename +
SHA-256). A final result binds to its owning checkpoint's SHA-256 and
transitively to everything the checkpoint itself binds.

## Checkpoint state machine

`incomplete` (`next_dataset_index < image_count_expected`, `complete =
false`, `final_result_produced = false`) → `complete`
(`next_dataset_index == image_count_expected`, `complete = true`) →
`finalized` (`final_result_produced = true`, only settable once a final
result has installed and self-verified). A checkpoint with
`final_result_produced = true` and `complete = false` is always rejected
-- an incomplete run may never carry a final-result flag.

## Canonical-prefix resume rule

`next_dataset_index == len(completed_image_ids)` always. Resuming
requires `completed_image_ids == selected_ids[:next_dataset_index]`
against the frozen manifest's own canonical order -- any skip, reorder,
or duplicate is rejected before resume. `resume_dataset_index()` itself
refuses to resume an already-`complete` checkpoint.

## Zero-row checkpoint handling

A checkpoint at `next_dataset_index == 0` is valid. Its per-image-stats
NPZ has every member shaped `(0,)`, built via `build_zero_row_arrays()`
-- never via `np.stack([])`, which raises on an empty sequence.

## Per-image compute-projection statistics (NPZ exact-member policy)

The per-image-stats NPZ persists only the *aggregate* query/compute-
projection quantities the protocol's own `[query_compute_projection]`
section already requires: `dataset_indices`, dispute/window counts,
Step 9's aggregate FLOP/byte/chunk totals, and the six required `|Q_w|`
descriptive statistics (`mean`, `median`, `p90`, `p95`, `p99`,
`maximum`). Thirteen `int64` members and six `float64` members form a
closed set -- `np.load(..., allow_pickle=False)` is followed by an exact
`frozenset` member-name comparison, a raw-ZIP duplicate/directory/
traversal-name pre-check (numpy's own loader silently tolerates a
duplicate ZIP member), and an explicit dtype/rank/shape/finiteness check
on every member, so a missing member, an extra member, an object array, a
structured dtype, a wrong shape, or a non-finite float is always
rejected. Per-window arrays are deliberately never persisted: window
count varies per real image (sliding-window geometry depends on each
image's actual height/width), so no fixed-shape NPZ column could hold it,
and the frozen protocol requires only the aggregate statistics anyway.

## Per-image-stats schema v2 (versioned successor)

`talk2dino-dfp-phase1a-per-image-stats-v2` (`schema_version = 1`) is a
**versioned successor** to the v1 schema above -- never a version bump
under the v1 name, and never a rewrite of it. The v1 schema, its name,
version, and closed 19-member set are byte-for-byte unchanged and remain
independently loadable/verifiable (`load_per_image_stats_npz`,
`build_zero_row_arrays`); any artifact already generated under v1
continues to validate exactly as before. Schema identification is
exact-match on the `(schema, schema_version)` pair, via the single
dispatcher `src.dfp_phase1a_checkpoint.identify_per_image_stats_schema_
variant` -- an unrecognized pair (including a correct name paired with
the wrong version) is always rejected, never guessed.

v1 is deliberately scoped only to the aggregate Phase-1B compute-
projection quantities. v2 additively carries every v1 member forward
unchanged and adds the sufficient statistics needed to reconstruct every
preregistered Phase-1A Gate-1 measurement without rerunning inference,
in four member categories (122 members total, see
`src.dfp_phase1a_artifact_schema`'s own module-level v2 comment and
`evaluation_identities/e13_dfp_phase1a_artifacts.toml`'s
`[per_image_stats_schema_v2]` section for the authoritative name list):

- **scalar** (`shape (image_count,)`, 33 `int64` + 6 `float64`): every
  v1 member plus dispute-category/persistence-reverted/ignore pixel
  counts, global-unary/reversion-coherence component counts, and the
  harmful-washout x boundary-band JOINT counts
  `harmful_boundary_pixel_count`/`harmful_interior_pixel_count`
  (`harmful_boundary_pixel_count + harmful_interior_pixel_count ==
  harmful_pixel_count`, each bounded by the corresponding
  `disputed_boundary_pixel_count`/`disputed_interior_pixel_count`
  marginal) -- required because the protocol's own
  `[gate_expectations].boundary_mass_reframe_threshold` is defined
  specifically over harmful-washout mass within the boundary band, a
  quantity the marginal `harmful_pixel_count` and the ALL-category
  `disputed_boundary_pixel_count` cannot reconstruct on their own (their
  intersection is not determined by the marginals without assuming
  independence, which this schema never does). No equivalent joint is
  defined for beneficial/third-label: neither the protocol's
  `[gate_expectations]` nor the frozen qualitative-report identity's own
  `[category_binding]` defines a beneficial- or third-label-specific
  boundary/interior category;
- **per-class confusion-2D** (`shape (image_count, class_count)`,
  `int64`, 76 members): `intersection`/`union`/`predicted_pixels`/
  `ground_truth_pixels` -- exactly `[metric_contract].required_
  statistics` -- for 19 prediction variants: the five raw horizon label
  maps T=0/16/32/64/320, the harmful-only recovery oracle `O_P`, the
  Third-Label Reversion Delta `third_label_revert_to_t0` (begin with the
  T320 prediction; wherever GT is neither T0's a(x) nor T320's b(x),
  replace b(x) with a(x); leave every other pixel equal to T320 -- an
  exact reproduction of `src.dfp_phase1a_analysis.construct_oracle_
  maps`'s `third_label_diagnostic` map, per the protocol's own
  `[oracle_definitions].third_label_diagnostic_rule`), and all four
  horizon-persistence variants at each of the three persistence horizons
  16/32/64. Every one of these is a GT-conditioned diagnostic only,
  never a deployable prediction. Ground truth does not change across
  variants, so `ground_truth_pixels` is additionally enforced to be
  IDENTICAL, class by class, across every variant for a given image --
  not merely equal in row sum. Boundary IoU per-class statistics are
  deliberately excluded: the live protocol's `[boundary_iou].
  implementation_source_status` is still `"not_yet_bound"`, so no run
  this schema describes can legitimately compute one;
- **offset** (`shape (image_count + 1,)`, `int64`, monotonically
  non-decreasing, starting at 0, 2 members): a CSR-style per-image row
  index into a parallel flattened member -- `component_offsets` and
  `reversion_component_offsets`;
- **flattened variable-length** (`shape (offsets[-1],)`, 3 `int64` + 2
  `float64` members, each keyed to exactly one offset member): global
  unary component class indices/pixel areas/patch-equivalent areas/
  source mass, and reversion-coherence component sizes at the primary
  persistence horizon (32) -- never a fixed-shape/invented histogram
  bin, since component/reversion counts are genuinely data-dependent.

Every relational content invariant these categories imply (confusion
`union == predicted + ground_truth - intersection`; confusion row sums
equal `evaluated_pixel_count`; classwise `ground_truth_pixels` equality
across every variant; dispute/boundary/persistence count partitions,
including the harmful x boundary-band joint; offset-diff-vs-scalar-count
agreement; flat-length and per-image segment-sum agreement against the
offset arrays) is checked by `_validate_per_image_stats_v2_relations`,
run automatically by `load_per_image_stats_npz_v2`. Zero-progress
support is `build_zero_row_arrays_v2(class_count=...)` -- scalar/flat
members get shape `(0,)`, confusion-2D members get shape `(0,
class_count)`, offset members get the single-element array `[0]` --
never `np.stack([])`. `validate_resume_state` reads the on-disk
per-image-stats manifest's own declared variant and dispatches to the
matching loader; it never guesses.

## Atomic generation/install order

Each checkpoint update: (1) validates the candidate NPZ arrays and
checkpoint fields against the exact same validators a reader would use;
(2) writes the NPZ to an unpredictable, task-specific generation filename
(`..._{run_mode}_{next_index:06d}_{16-byte-hex-nonce}.npz`), never
overwriting a previous generation's file; (3) self-verifies by re-hashing
and fully re-loading the just-installed NPZ; (4) writes and self-verifies
the per-image-stats manifest referencing it; (5) only then atomically
installs the checkpoint document (`write_json_atomically`: temp file in
the destination directory, `fsync`, `os.replace`) referencing the new
generation. Any failure before step 5 cleans up the new generation's
orphaned files and leaves the previous valid checkpoint (and everything
it references) byte-identical.

## Result finalization

A result may be `complete=true`/`final=true` only if its bound
checkpoint is itself `complete` at the exact expected image count, its
`completed_image_ids` equals the complete frozen selected-ID list, and
every referenced artifact (checkpoint, per-image-stats manifest, NPZ)
hash-reconciles. `install_result_bundle()` builds the complete bundle
(`result.json`, a copy of `checkpoint.json`, the per-image-stats manifest
and NPZ) in a fresh sibling staging directory, fully self-verifies it,
then `os.rename`s the staging directory into place -- rejecting cleanly
if the destination already exists, never partially overwriting it. This
step's result document deliberately excludes `G_recover`, `G_oracle_unary`,
gate outcomes, and `aAcc`/`mAcc`/`mIoU`/`boundary_iou`: no Phase 1A
diagnostic driver exists yet to compute them.

## Verifier commands

CPU-only; portable placeholders below.

```
python3 verify_dfp_phase1a_artifacts.py verify-schema
python3 verify_dfp_phase1a_artifacts.py verify-checkpoint --checkpoint <path/to/checkpoint.json>
python3 verify_dfp_phase1a_artifacts.py verify-resume --checkpoints-dir <path/to/checkpoints> --checkpoint-filename checkpoint.json
python3 verify_dfp_phase1a_artifacts.py verify-result --bundle <path/to/result_bundle_dir>
```

None of these commands access a real dataset root, construct a dataset,
import/initialize CUDA, build a model, load model weights into torch, or
run graph/recurrence code. Every domain error, `OSError`, and JSON
`ValueError` at the CLI boundary prints one concise line and exits `2`
with no traceback; `KeyboardInterrupt`/`SystemExit`/`MemoryError` are
never caught and propagate uncaught through both the library functions
and `main()`.

## Excluded from this step

Real Phase 1A driver, dataset/model inference, multi-horizon GPU
execution, masked propagation, component evidence, Gate 1/2 analysis,
real result production, hardware calibration, SLURM scripts, GPU
submission. This step never runs the diagnostic.
