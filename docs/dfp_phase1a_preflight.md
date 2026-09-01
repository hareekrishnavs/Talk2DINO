# DFP Phase 1A Step 12: unified CLI and ordered preflight

Step 12 adds one unified, CPU-only CLI (`diagnostics/run_dfp_phase1a.py`)
and a reusable Python preflight API (`src/dfp_phase1a_preflight.py`)
that prove every scientific, dataset, checkpoint, artifact, path, and
resume binding is valid **before any heavy execution begins**. It
implements no new scientific logic: every check is delegated to the
already-committed, already-verified Step 1-11 validators. Its own new
logic is limited to (a) the explicit ordered composition of those
validators into Phase A / B / C, (b) resolving the frozen dataset root
and reconciling the 500 selected canonical IDs against physical
image/annotation paths without ever guessing a path from an ID alone,
(c) output-root safety, and (d) the fresh/resume decision and the
immutable `PreflightContext` a future execution driver will consume.

**Step 12 performs no evaluation.** The CLI exposes no `run`, `evaluate`,
`pilot`, or `full` command, and nothing in `src/dfp_phase1a_preflight.py`
imports `torch`, constructs a model or dataset object, calls
`torch.load`, or queries CUDA.

## Subcommands

```
diagnostics/run_dfp_phase1a.py [--repo-root PATH] <command> ...
```

| Command                     | Purpose                                                                                          |
|------------------------------|---------------------------------------------------------------------------------------------------|
| `verify-protocol`            | Verify the frozen scientific protocol identity.                                                   |
| `verify-subset`               | Verify one frozen 500-image subset manifest's schema/relational bindings (no dataset access).     |
| `verify-source-binding`      | Like `verify-subset`, plus independent recomputation of the selection from the real dataset.       |
| `verify-model-binding`       | Verify the model/evaluator config-source-hash and bridge-checkpoint raw-byte binding for a dataset.|
| `verify-checkpoint`          | Verify one checkpoint document's structure and bindings (delegates to Step 10).                    |
| `verify-result`              | Verify an installed final-result bundle directory (delegates to Step 10).                          |
| `verify-qualitative-report`  | Verify an installed qualitative-report bundle directory (delegates to Step 11).                    |
| `preflight`                  | Run the full ordered Phase A/B/C preflight for one run; prints one `READY:` line on success.       |

Every subcommand's exact arguments are visible via `--help`; the two
most commonly used are:

```
diagnostics/run_dfp_phase1a.py verify-subset --dataset {coco_stuff,ade20k} [--manifest PATH] [--no-git-check]

diagnostics/run_dfp_phase1a.py preflight \
    --run {coco_stuff_500,ade20k_500} \
    --mode {fresh,resume} \
    --output-root PATH \
    [--data-root PATH] [--checkpoints-dir PATH] [--checkpoint-filename NAME] \
    [--allow-fresh-overwrite] [--probe-output-writeable] [--no-git-check]
```

`--run` accepts **only** the two exact frozen run names from the
protocol (`coco_stuff_500`, `ade20k_500`) -- no aliases, case changes,
abbreviations, or inferred names.

## Environment variables and precedence

Dataset roots are never embedded as a private default. Each dataset has
its own portable environment variable:

```
COCO_STUFF_REAL_DATA_ROOT
ADE20K_REAL_DATA_ROOT
```

Precedence (identical to the established Step 2 policy in
`src/dfp_phase1a_subset_manifest.require_env_root`, reused verbatim, not
reimplemented): an explicit `--data-root` CLI argument **always**
overrides the environment variable when both are given; if neither is
given, the command fails closed (`... is not set and no --*-root
override was given`). An empty or whitespace-only value is rejected the
same way as an absent one. `Path("")` is never used as a fallback.

## Ordered preflight: Phase A / B / C

The preflight runs three phases, always in this exact order, each
raising `DfpPhase1aPreflightError` (or a chained domain error from a
delegated validator) on its first violation. No phase partially
completes and proceeds to the next; a stage is never skipped or
reordered (`src.dfp_phase1a_preflight.PHASE_A_STAGES` /
`PHASE_B_STAGES` / `PHASE_C_STAGES` / `ALL_STAGES` are the frozen,
test-observable stage-name tuples; `PreflightTrace` is a production-
optional, test-only hook that records each stage as it runs).

**Phase A -- repository and schema (no dataset/model/CUDA):**
parse the exact run name; resolve/validate the repository root; load
and validate the scientific protocol (`dfp_phase1a_identity`); validate
the required Git ancestry policy (part of the protocol load, when
enabled); load and validate the artifact schema (`dfp_phase1a_artifact_
schema`, which also relationally hashes the bridge checkpoint's raw
bytes and every parent identity in the same pass); load and validate the
qualitative-report schema (`dfp_phase1a_qualitative_selection`); load and
verify the frozen subset manifest for the requested run (`dfp_phase1a_
subset_manifest.verify_manifest_document`, via the narrow
`verify_frozen_subset_manifest` adapter reused by both Phase A and the
standalone `verify-subset` command); cross-check every authority's own
observed protocol hash agrees; select the exact run-specific binding
(class count/digest, selected IDs, expected count).

**Phase B -- filesystem/source validation (still no model/CUDA):**
resolve the dataset root safely from the environment variable/override
(`require_env_root` + `resolve_coco_stuff_root` / the ADE20K
identity's own `resolve_dataset_root`); load the dataset source
identity; reconcile the 500 *selected* canonical IDs against physical
image/annotation files (see "Logical ID vs physical filename" below --
existence-only, never a full pixel decode, which is the separately
available and much heavier `verify-source-binding` recomputation);
verify the model/evaluator config-source-hash binding
(`matched_k11_k12_identity.validate_static_configuration` for
COCO-Stuff, `ade20k_matched_evaluator_identity.validate_static_
configuration` for ADE20K -- both already independently hash their
own config YAMLs and bridge/projection checkpoint bytes, no torch
import); the bridge checkpoint's raw-byte hash was already
independently proven in Phase A and is not re-hashed; validate
output-root safety.

**Phase C -- resume/result validation (still no model/CUDA):**
decide the exact fresh vs. resume mode (never inferred); in fresh mode,
reject an existing checkpoint or final result unless
`--allow-fresh-overwrite` is passed; in resume mode, require the
checkpoint to exist and delegate the *entire* ordered validation
(checkpoint JSON, canonical completed-ID prefix, per-image-stats
manifest, NPZ existence/hash/schema/shape/dtype, checkpoint/array
reconciliation) to `dfp_phase1a_checkpoint.validate_resume_state` --
called exactly once, never re-parsing the NPZ afterward; reject an
already-complete checkpoint under resume mode; reject resuming a
checkpoint bound to a different run mode than requested; if a final
result already exists, verify it (`dfp_phase1a_report.
verify_result_bundle`) and reject resuming an already-finalized run;
build the immutable `PreflightContext`.

## Fresh vs. resume

Mode is always explicit (`--mode fresh` / `--mode resume`) -- never
silently inferred from what happens to already exist on disk.

- **fresh**: rejects an existing checkpoint or an existing final result
  at the target checkpoints directory unless `--allow-fresh-overwrite`
  is explicitly passed.
- **resume**: requires the checkpoint (and its per-image-stats
  manifest/NPZ) to already exist and pass every Step 10 invariant;
  requires the checkpoint's own `run_mode` to match the requested run
  (refusing to resume across runs); requires the checkpoint to be
  *incomplete* (an already-complete checkpoint is rejected under
  resume -- redo it via fresh mode with an explicit overwrite
  authorization instead); if a final result is already installed and
  marked `final`, resume is refused.

## Logical ID vs. physical filename

A canonical dataset image ID is never the same string as its physical
filename. `src.dfp_phase1a_preflight.reconcile_selected_ids_to_physical_
paths` constructs every physical path as exactly
`dataset_root / <image|annotation>_relative_root / (canonical_id +
suffix)`, where the relative root and suffix always come from the
dataset source identity's own bound fields (COCO-Stuff:
`images/val2017` / `annotations/val2017`, `.jpg` /
`_labelTrainIds.png`, from `dfp_phase1a_subset_manifest`'s frozen
constants; ADE20K: `identity["source"]["image_relative_root"]` /
`annotation_relative_root` / `image_suffix` / `annotation_suffix`, from
`ade20k_dataset_identity`). It is never constructed as
`image_root / canonical_id` alone. `tests/test_dfp_phase1a_preflight.py`
carries an explicit VOC-style regression test: a canonical ID with no
extension whose physical file has had its suffix stripped fails
reconciliation cleanly, and a bare-canonical-ID-named file (no suffix at
all) never satisfies reconciliation.

## Model/checkpoint hash verification without torch

The bridge/projection checkpoint's binding is proven entirely by
**raw-byte SHA-256** comparison against the frozen identity value --
`hashlib.sha256(path.read_bytes())`, nothing more. This happens inside
`dfp_phase1a_artifact_schema.validate_static_configuration` (Phase A)
and, redundantly for its own dataset-specific config chain, inside
`matched_k11_k12_identity.validate_static_configuration` /
`ade20k_matched_evaluator_identity.validate_static_configuration`
(Phase B). None of these call `torch.load`, import the model, deserialize
a tensor, construct a backbone, or query CUDA. A corrupt or substituted
checkpoint file fails from this byte/hash binding alone, before any
future execution step could ever touch it.

## The `PreflightContext`

`run_preflight(...)` returns a frozen `PreflightContext` dataclass: run
name, expected count, class count/digest, the 500 selected canonical IDs
(tuple), every relevant identity/schema SHA-256, the resolved dataset
root and output root (private, in-memory only -- this object is never
itself serialized into a portable manifest), fresh/resume mode, the next
dataset index, completed IDs (tuple), and -- only in resume mode -- the
already-validated checkpoint arrays as a read-only (`MappingProxyType`
of non-writeable NumPy arrays) mapping, loaded exactly once by Phase C
and never re-parsed. `to_summary_dict()` returns a schema-safe,
non-secret `MappingProxyType` summary (no resolved path, no raw arrays)
suitable for a log line. No field is a model object, dataset object,
CUDA device, torch tensor, or mutable caller-owned dict/list.

## Success/failure output contract

A successful command prints exactly one unambiguous line --
`PASS: ...` for a `verify-*` command, `READY: ...` for `preflight`
(never printed before every phase has finished) -- with supplementary,
non-secret detail on stderr. A failure prints exactly one
`DFP PHASE 1A FAIL [<command>]: <diagnostic>` line to stderr and exits 2;
no traceback, no partial output, no temporary residue. The CLI boundary
catches only the established DFP domain error types plus expected
`OSError`/`ValueError`; it never catches `KeyboardInterrupt`,
`SystemExit` (beyond argparse's own ordinary behavior), or `MemoryError`
-- both propagate uncaught, as proven in
`tests/test_dfp_phase1a_preflight.py`.

## Example commands (portable, no private paths)

```bash
# Verify everything that's already committed (no dataset access):
diagnostics/run_dfp_phase1a.py verify-protocol
diagnostics/run_dfp_phase1a.py verify-subset --dataset coco_stuff
diagnostics/run_dfp_phase1a.py verify-subset --dataset ade20k
diagnostics/run_dfp_phase1a.py verify-model-binding --dataset coco_stuff
diagnostics/run_dfp_phase1a.py verify-model-binding --dataset ade20k

# Full ordered preflight for a fresh COCO-Stuff run:
export COCO_STUFF_REAL_DATA_ROOT=/path/to/coco_stuff_root
diagnostics/run_dfp_phase1a.py preflight \
    --run coco_stuff_500 --mode fresh \
    --output-root /path/to/output --probe-output-writeable

# Resuming an existing ADE20K run:
export ADE20K_REAL_DATA_ROOT=/path/to/ade20k_root
diagnostics/run_dfp_phase1a.py preflight \
    --run ade20k_500 --mode resume \
    --output-root /path/to/output
```

## Scope: no execution in Step 12

Step 12 implements no evaluation loop, no model/dataset construction for
execution, no snapshot extraction, no graph/recurrence execution, no
horizon-stitching execution, no Phase 1B masked propagation, no Gate 1/2
analysis, no result generation, no qualitative rendering from real data,
no GPU runtime calibration, and no SLURM scripts or job submission. A
future step adds the actual execution driver that consumes a
`PreflightContext` this module produces.
