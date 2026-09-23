# T320 Real Smoke Runner

Task: `IMPLEMENT_T320_REAL_SMOKE_RUNNER`, implementing the committed contract
`research_specs/semantic_evidence_t320_real_smoke_v1.toml`
(`src/semantic_evidence_t320_real_smoke_identity.py`).

This document explains how to run the two-image E3/T320 integration smoke
and what it does and does not prove. It is not the contract itself; where
this document and the committed authority disagree, the authority governs.

## What this proves, and what it does not

A `PASS` on this smoke proves that the committed, synthetically-verified
isolated adapter and T320 target generator work correctly against the real
E3 model, real external weights, real preprocessing, and real window
geometry, for exactly two discovery images. It does **not** prove, claim,
or estimate:

- accuracy, IoU, mIoU, mAcc, or pixel accuracy, on any image;
- full-dataset (5,000-image) runtime, from this smoke's two-image timings;
- that the +1.5 mIoU publication target is met;
- readiness for the 96-image feasibility study, confirmation evaluation, or
  interventions -- each remains a separate, later authorization.

## New files (this task)

- `src/semantic_evidence_t320_real_smoke.py` -- production module: torch-free
  preflight, the real (lazily-torch-importing) execution core (`run_smoke`),
  the CLI-facing orchestrator (`execute`), the closed result builder/schema
  validator, the atomic installer, and the torch-free public verifier.
- `diagnostics/run_semantic_evidence_t320_real_smoke.py` -- CLI with four
  subcommands: `verify-protocol`, `preflight`, `execute`, `verify-result`.
- `tests/test_semantic_evidence_t320_real_smoke.py` -- CPU-only tests using
  fake providers and synthetic tensors; never builds the real model or
  touches real data.
- `semantic_evidence_t320_real_smoke_h100.sbatch` -- the H100 launcher.
  Contains no `sbatch` invocation of itself; only a human operator submits
  it, and only after every authorization-boundary condition below holds.
- this document.

No existing file is modified. Nothing is staged or committed by this task,
and no real image, GT, checkpoint, model, or CUDA access occurs while
implementing or testing it.

## Architecture

Four layers, matching the committed contract's own structure:

1. **Torch-free preflight** (`preflight()`) -- re-runs the committed
   authority's own official verifier (`semantic_evidence_t320_real_smoke_identity.verify_all`),
   re-hashes the three external checkpoint *files* (raw bytes -> SHA-256,
   never `torch.load`) against the authority's declared bindings, confirms
   the two authorized image files resolve (never touching any annotation
   path), and confirms the output destination is fresh and safe. Runs
   without importing torch, initializing CUDA, or reading a pixel.
2. **Real execution path** (`run_smoke()`, wrapped by `execute()`) -- lazily
   imports torch and the committed `semantic_evidence_e3_adapter`/
   `semantic_evidence_t320` modules only here. Constructs the E3 model and
   loads its weights exactly once (the same event, since `DINOText.__init__`
   itself performs the checkpoint load). Processes the two images in frozen
   order, generates the canonical window universe per image from its real
   dimensions, captures exactly one E3 forward per window via the committed
   adapter, validates same-forward T0 preservation, and runs the pure
   target-generation twice from the same cached snapshots.
3. **Closed result builder / validator / atomic installer**
   (`build_result`, `validate_result_schema`, `install_result_atomic`) --
   builds the closed result dict, stages it in a sibling temporary
   directory, validates the staged copy after reopening it, and only then
   atomically renames the staging directory into place. Never overwrites an
   existing destination; a validation failure at any point removes the
   staging directory and leaves no partial artifact at the final path.
4. **Torch-free public result verifier** (`verify_installed_result()`) --
   reopens the installed `smoke_result.json` from disk (never trusts the
   in-memory value the installer just wrote), re-validates its schema, and
   independently recomputes every count relation from the recorded `N`,
   `M`, `R` before agreeing the artifact is `verified`.

`execute()`'s CLI-facing flow is: preflight -> `run_smoke()` with the real
default providers -> validate -> atomic install -> reopen -> public verify
-> print success only after that verification passes. Tests call
`run_smoke()` directly with fake providers, since real preflight requires
real checkpoint and image files this task's own restrictions forbid
touching.

## Window geometry

Window bounds are computed by `compute_canonical_windows()`, a pure
arithmetic mirror -- never a reimplementation of the inference logic -- of
`DINOTextSegInference.slide_inference`'s own loop
(`src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_seg.py:78-94`):
same `h_grids`/`w_grids` formula, same row-major order, same flush-right/
flush-bottom clamping. Crop `[448, 448]`, stride `[224, 224]`, patch
`[14, 14]`, all read from the committed authority, never hardcoded
independently. No debug, max-window, subsampling, or reorder path exists.

## Two-image whitelist enforcement

`_default_real_image_loader` independently enforces the exact two-ID
whitelist (`_AUTHORIZED_IMAGE_IDS`, an internal immutable tuple, cross-checked
by `preflight()` against the committed two-image manifest) before touching
the filesystem at all: type, whitespace, path-separator/traversal, exact
12-digit format, and exact membership are all checked, in that order, before
any `.jpg` path is even constructed. Unauthorized IDs raise `SmokeRunnerError`
with zero file opens -- confirmed by instrumenting `open()` in tests. No CLI
flag or public API parameter can override this whitelist. An authorized ID
whose file is missing, unreadable, a directory, or fails to decode as an
image also raises `SmokeRunnerError` with a concise message -- never a raw
`FileNotFoundError`/`PermissionError`/`OSError`/image-library exception.
This restriction applies only to the *filename the loader is permitted to
open*, not to `image_root`/`weight_dir` themselves: those read-only dataset
and weight directories may legitimately be symlinks into shared HPC storage,
and this loader does not restrict that.

## One-forward preservation and cached-snapshot repeats

For every window: `semantic_evidence_e3_adapter.extract_e3_window_snapshot`
performs exactly one E3 forward and returns both the ordinary sigmoid-domain
mask output and the immutable `WindowSnapshot` (features + raw S0). T0 is
reconstructed from that same snapshot via `internal_upsample(S0, ...)`, and
must equal the adapter's ordinary output **bitwise** -- any mismatch raises
`SmokeRunnerError` and the smoke fails closed, never a warning. After all of
an image's windows are captured, `generate_t0_t320_targets` (which is
**image-scoped**: one call = one image, never a mixed multi-image campaign)
is called twice from the identical cached snapshots, with zero additional
model forwards between the two calls. The two repeats' T0/T320 prediction
digests must match exactly.

## Event counting

Every count in the result artifact is derived from actually-instrumented
events (call-counting wrappers around `torch.sigmoid`,
`DirectedTopKGraph.matmul`, `build_t320_graph`, `stitch_windows`, and
`finalize_prediction`, installed only for the duration of each
`generate_t0_t320_targets` call and restored immediately after), never
assigned from the expected formula after the fact. The runner additionally
cross-checks the observed per-repeat totals against their structural
invariants (exactly `2*windows` sigmoid calls, exactly `windows` graph
constructions, exactly `windows*320` recurrence updates) before accepting
them, and finally compares the fully-accumulated actual counts against the
independently-derived expected values (`N`, `N*R`, `N*R*320` for
window-scoped quantities; `M*R` for the four image-scoped stitch/finalize
quantities) -- a mismatch raises `SmokeRunnerError`, not a silent pass.

The four image-scoped counts (`t0_stitch_calls`, `t0_finalize_calls`,
`t320_stitch_calls`, `t320_finalize_calls`) are independently re-derived at
*validation* time too, both when a result is first built and again on every
reopen: `M` is recounted from `len(selected_image_ids)` and `R` from the
actual structure of `t0_t320_repeat_digests`, never trusted from the
artifact's own `M`/`R`/`expected` fields, and `validate_result_schema`
requires each of the four counts to equal that independently-recomputed
`M*R` exactly, with a closed 7-key schema (`unit`, `M`, `R`, and the four
count fields) that rejects any missing, extra, merged, or ambiguous field.

## Result artifact, provenance, and installation

The result (`smoke_result.json`, scratch-only, never committed to Git) is
one of exactly two outcomes: `PASS_REAL_E3_T320_INTEGRATION_SMOKE` or
`FAIL_REAL_E3_T320_INTEGRATION_SMOKE`. `PASS` requires every mandatory
check (count match, prediction equality between repeats, bitwise
preservation on every window, zero non-finite values) -- there is no
warning-only bypass, enforced by `validate_result_schema`. Installation is
staged in a sibling temporary directory, reopened and re-validated there,
then atomically renamed into place; a failure at any point removes the
staging directory and leaves any pre-existing path at the destination
untouched (a fresh destination is required -- the installer refuses to
overwrite).

**Every final artifact cryptographically binds the exact committed bytes of
the code that produced it.** The existing `implementation_hashes` field
(already required by the committed authority's frozen 21-field result
schema -- no new top-level field was added) is extended with six
runtime-provenance records, each `{path, sha256, role,
must_be_committed_at_execution_head}`: the four mandatory roles
`smoke_runner` (`src/semantic_evidence_t320_real_smoke.py`), `smoke_cli`
(`diagnostics/run_semantic_evidence_t320_real_smoke.py`), `smoke_launcher`
(`semantic_evidence_t320_real_smoke_h100.sbatch`), and `smoke_authority`
(`research_specs/semantic_evidence_t320_real_smoke_v1.toml`), plus two
optional roles (`smoke_authority_identity_loader`,
`smoke_two_image_manifest`). Building any of these records requires the
corresponding file to be git-tracked, have a clean working tree, and have a
working-tree hash equal to its committed-HEAD blob hash -- this is what
makes real execution structurally impossible before this exact runner/CLI/
launcher/authority file set has actually been committed.

`verify-result` (and `verify_installed_result` whenever given a
`repo_root`, which the CLI always supplies) independently re-derives every
`implementation_hashes` entry's SHA-256 from git objects -- both at the
result's own recorded commit and at the repository's current HEAD -- and
requires the recorded commit to exist and to actually contain each
recorded path, before agreeing the artifact's provenance is verified. This
public reverification remains entirely torch-free.

## CLI

```
diagnostics/run_semantic_evidence_t320_real_smoke.py verify-protocol
diagnostics/run_semantic_evidence_t320_real_smoke.py preflight   --weight-dir DIR --image-root DIR --output-root DIR
diagnostics/run_semantic_evidence_t320_real_smoke.py execute     --weight-dir DIR --image-root DIR --output-root DIR [--slurm-job-id ID] [--node NAME]
diagnostics/run_semantic_evidence_t320_real_smoke.py verify-result --result-path PATH
```

`verify-protocol`, `preflight`, and `verify-result` never import torch
(confirmed by running them under a Python interpreter with no torch
installed at all). No subcommand accepts `--max-images`, an alternate image
ID, a debug/subset window override, a GT/annotation path, a scientific
parameter override, or overwrite/force flags -- none exist.

## H100 launcher

`semantic_evidence_t320_real_smoke_h100.sbatch` requests exactly 1×H100
80GB, 8 CPUs, 80GB host memory, `00:30:00`, account `rrg-yangw`. It contains
no `sbatch` invocation of itself and is never submitted by any automation in
this repository -- only a human operator, running `sbatch
semantic_evidence_t320_real_smoke_h100.sbatch` directly, after every
condition below holds. It fails closed on a pre-existing output destination
before doing anything else, sets `TALK2DINO_WEIGHT_DIR` from the operator's
own environment variable (never invents a default), and runs preflight,
then execute, then the torch-free public verifier as three explicit steps,
printing the authorization-boundary disclosure before and after.

## Authorization boundary

Implementing and CPU-testing this runner does **not** authorize real
execution. Per the committed authority's own `status` table, real execution
requires, in order:

1. this contract passed independent verification (done:
   `INDEPENDENT_VERIFY_T320_REAL_SMOKE_CONTRACT`, `PASS`);
2. it is committed (done: commit `6047823164f8b80d3b90913cc7214dff6c46cc74`);
3. the runner and launcher are implemented (this task);
4. the runner passes independent verification (separate, later task);
5. committed-HEAD readiness passes (separate, later task, after the runner
   itself is committed).

Only after all five does an operator submit the launcher manually. Nothing
in this task submits it, and this task did not run it against real data.

## Repair note (provenance, image-scoped counts, loader whitelist)

An independent adversarial verification of the original runner found three
mechanical defects, all repaired here: (1) result artifacts did not
cryptographically bind the runner/CLI/launcher/authority that produced them
-- see "Result artifact, provenance, and installation" above; (2)
`validate_result_schema` did not independently re-derive the four
image-scoped stitch/finalize counts against `M*R` -- see "Event counting"
above; (3) the default image loader relied entirely on its caller to never
pass an unauthorized ID and leaked a raw `FileNotFoundError` -- see
"Two-image whitelist enforcement" above. None of these repairs changed any
scientific behavior, window geometry, population selection, resource
limits, or authorization boundary; the same fake-provider fixture produces
bit-identical scientific digests (window digest, T0/T320 prediction
digests, all window- and image-scoped counts) before and after this repair.
This document still does not claim the real smoke has executed.
