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
