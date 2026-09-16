# DFP Phase 1B: lightweight BALP 350-image development runner

This document describes the execution-layer implementation built on top of the
frozen E22 identity (`docs/dfp_phase1b_lightweight_boundary_protocol.md`,
`evaluation_identities/e22_dfp_phase1b_lightweight_boundary_preservation.toml`,
`evaluation_identities/e22_dfp_phase1b_lightweight_boundary_artifacts.toml`).
It implements the BALP scientific kernel, the checkpoint/result artifact layer,
the 350-image development-population orchestration, and a CPU-only,
non-executing verification CLI. **It authorizes no inference, no GPU
execution, no candidate selection, and no confirmation/full-population
access** — every one of those remains gated behind a later successor
authority.

## 1. New files

| File | Role |
|---|---|
| `src/dfp_phase1b_lightweight_boundary.py` | Pure BALP scientific kernel (torch tensors only; no identity/data/model import). |
| `src/dfp_phase1b_lightweight_boundary_artifact.py` | Checkpoint/development-result schema validators and atomic writers. |
| `src/dfp_phase1b_lightweight_boundary_execute.py` | Canonical-inference integration, GT-isolation boundary, 350-image fresh/resume/finalize orchestration. |
| `diagnostics/run_dfp_phase1b_lightweight_boundary.py` | Non-executing CLI: `verify-protocol`, `preflight`, `verify-checkpoint`, `verify-development-result`. |
| `tests/test_dfp_phase1b_lightweight_boundary.py` | Kernel tests against independent NumPy oracles. |
| `tests/test_dfp_phase1b_lightweight_boundary_artifact.py` | Artifact/atomicity/adversarial tests. |
| `tests/test_dfp_phase1b_lightweight_boundary_execute.py` | Real canonical-inference fixture, metric, population, GT-isolation, and full 350-image harness tests. |

## 2. Scientific kernel (`dfp_phase1b_lightweight_boundary.py`)

Operates on the 5 canonical final-resolution score fields `S_h`,
`h ∈ {0,16,32,64,320}`, each shape `[class_count, ori_H, ori_W]`. It:

1. Computes `y_h = lowest_index_argmax(S_h)` for each horizon (own tie-mask +
   argmin construction, independently cross-checked against
   `src.dfp_phase1a_analysis._lowest_index_argmax`; never relies on
   `torch.argmax`'s undocumented tie behavior).
2. Computes 8-connected boundaries of `y64` and `y320`, unions them.
3. Computes 4 Chebyshev dilations of the boundary union (`radius ∈
   {0,1,2,4}`) via `max_pool2d(kernel=2r+1, stride=1, padding=r) > 0.5`
   (`radius=0` is the identity mask).
4. Computes 4 stability masks (`start ∈ {0,16,32,64}`; `start=64` is always
   `True`) and `late_flip = (y64 != y320)`.
5. Builds all 16 preservation masks `P_{s,r} = stable_s AND late_flip AND
   D_r(boundary_union)` and all 16 candidate predictions `y_{s,r} =
   where(P_{s,r}, y64, y320)`, in exactly `CANDIDATE_IDS` order (`s` outer,
   `r` inner, matching E22's own frozen enumeration).

The kernel never materializes 16 full score tensors and never recomputes a
boundary or a horizon prediction per candidate — it reuses the same 5
predictions, 4 stability masks, 1 late-flip mask, and 4 dilated boundary
masks for every candidate. It preserves the input device throughout, uses no
`.item()`-driven control flow, and has no learned weight, confidence
threshold, GT parameter, component, pullback, or masked-forward dependency.

## 3. Canonical inference integration (`dfp_phase1b_lightweight_boundary_execute.py`)

`process_development_image` reuses the existing, already-verified Phase 1A
primitives — `build_stitching_pullback_plan`, `build_directed_topk_graph`
(k=12), `capture_dfp_phase1a_trajectory`, `stitch_dfp_phase1a_horizons`,
`apply_final_rescale_forward` — and runs **exactly one** 320-step k12
recurrence per selected window per image (proven by an explicit
`model.call_count` assertion in
`test_real_single_window_canonical_inference_produces_valid_balp_outcome`).
All 5 horizon captures for an image come from that single trajectory; BALP
never re-runs the model, graph, or recurrence per horizon or per candidate.
This function has **no `ground_truth` parameter at all** — a structural,
not merely behavioral, GT-isolation guarantee (asserted directly via
`inspect.signature` in
`test_real_canonical_inference_ground_truth_never_touched_before_finalization`).

Note on naming: `src.dfp_phase1a_stitching_pullback` and
`src.dfp_phase1a_final_rescale_pullback` are Phase 1A's *spatial coordinate*
pullback (window-to-full-image stitching and final crop/rescale) — an
existing, required, GT-free canonical primitive. This is unrelated to, and
must not be confused with, the *prohibited* semantic-component
membership/pullback machinery used by the discontinued exact-provenance
family (E15/E16/E19). The BALP kernel and runner import neither
`src.dfp_phase1b_masked_forward` nor any E19/E20/E21 module (verified by a
static grep audit of every new file).

`accumulate_image_metrics` is the **only** function that ever reads
`ground_truth`; it is called strictly after `process_development_image`
returns a fully finalized `ImageBalpOutcome`. Per image it produces, for all
21 variants (5 controls + 16 candidates), a confusion matrix, evaluated/
ignored pixel counts, changed-vs-`T320` pixel count, and (candidates only)
preservation-mask pixel counts and boundary/non-boundary preservation
counts — never a value derived by rounding, and never a per-image mIoU
(aggregation happens only via summed confusion matrices at finalize time,
proven not to equal a naive per-image mIoU average in
`test_metric_aggregation_is_confusion_sum_not_per_image_mean`).

## 4. Population enforcement

`execute_development` independently loads E22's own frozen split
(`e22.load_and_verify_split`) and rejects unless the caller's
`development_ids`/`confirmation_ids` are **byte-identical, in order**, to
that split — not merely self-consistent. This rejects a reordered,
shortened, resampled, or substituted-dataset population even where such a
population would otherwise pass every downstream (self-consistent-only)
relational check. `development_ids` must have exactly 350 entries; any ID
found in `confirmation_ids` is rejected before processing.

## 5. Checkpoint / development-result artifacts

E22's artifacts identity is explicitly `storage_representation_only =
True`: it governs the checkpoint *document's* closed logical schema
(exact top-level/nested keys, dtypes, relational rules), never a
physical file layout. `art.py`'s on-disk layout is documented here as
that storage-layer design choice, revised once (see §9 below) after an
independent verification pass found the original flat-file layout was
not crash-atomic:

```text
output_root/
  checkpoint_generations/
    generation_<N>/
      checkpoint.json
      checkpoint_arrays.npz
  current_checkpoint.json
  development_result/
    result.json
```

Each checkpoint generation is one **immutable** directory containing a
`checkpoint.json` metadata file and a sibling `checkpoint_arrays.npz`
bulk-numeric file, bound by a SHA-256 recorded in
`identity_bindings.development_digest`. A generation is written into a
private staging directory, self-verified there, and made visible under
its real name via exactly **one** atomic directory rename (`os.replace`)
— so a generation directory is either fully absent or fully complete;
there is no observable state with only one of its two member files. Only
after that installed generation has been reopened from disk and
re-verified does the small `current_checkpoint.json` pointer — itself
replaced via exactly one atomic single-file `os.replace` — get advanced
to reference it. A failure at any point before that final pointer
replace leaves the previous current generation completely untouched; a
generation directory is never overwritten or mutated in place. Every
completed development image installs its own new generation (generation
numbers are monotonic per install, not per resume-session), matching
E22's own `cross_generation_invariants` (an earlier generation is always
an exact prefix of a later one for the same run).

The final development result is a directory bundle
(`development_result/result.json`) installed once via the same
stage-validate-atomic-rename-reopen-reverify sequence into an *absent*
destination — a second install attempt is rejected outright. This
directory-level atomic-rename pattern was always used for the
development result; §9 explains why the checkpoint layer now uses it
too.

Every expected NPZ/JSON corruption or parsing failure (a truncated file,
random bytes, a valid zip missing a required member, a valid zip whose
internal `.npy` payload is malformed) is wrapped in
`DfpPhase1bLightweightBoundaryArtifactError` with the original exception
preserved as `__cause__` — never a raw `zipfile.BadZipFile` or other
exception, and never via a bare `except:`/`except Exception:`. Fatal
`BaseException` subclasses (`KeyboardInterrupt`, `SystemExit`,
`MemoryError`, ...) are never caught anywhere in this module.

## 6. Fresh / resume / finalize-only equivalence

`execute_development` accepts injectable `image_processor_fn`/
`metrics_fn` seams (mirroring the discontinued E20 runner's
`stage_a_processor` seam), enabling a full 350-real-development-ID,
no-torch/no-model/no-dataset control-plane harness. That harness (in
`tests/test_dfp_phase1b_lightweight_boundary_execute.py`) proves, using a
poisoned image-provider that raises on any already-completed ID:

- A clean fresh run over all 350 IDs completes with `complete=True,
  final=True`.
- Resuming after 0, 1, 175, and 349 completed images, and resuming a
  complete-but-unfinalized checkpoint (finalize-only recovery, with *every*
  development ID poisoned), all produce a **byte-identical**
  `scientific_output_digest` and full result document to the clean-fresh
  baseline, with zero reprocessing of any already-completed image.
- Calling `execute_development` again against an already-finalized
  checkpoint raises `DfpPhase1bLightweightBoundaryExecuteError`.

## 7. CLI (`diagnostics/run_dfp_phase1b_lightweight_boundary.py`)

`verify-protocol` and `preflight` never import torch or `cv2`, never
initialize CUDA, and never access a dataset image or model checkpoint
(verified directly by inspecting `sys.modules` after invocation).
`preflight` additionally checks the current branch against E22's own
`identity.working_branch`, the repository's clean/dirty state (`git status
--porcelain`; overridable via `--allow-dirty-repo` for local iteration),
output-root path safety (reusing
`src.dfp_phase1a_preflight.validate_output_root_safety`, which rejects a
regular-file target, a path overlapping `src/`, the evaluation identity/
subset/docs/tests/diagnostics directories, or `.git`, and a symlink
escaping its own containing directory), and the mode-specific checkpoint
state at a given `--output-root` (fresh / resume / complete-but-unfinalized
/ already-finalized). `verify-checkpoint` and `verify-development-result`
load and validate already-installed artifacts through the same real
validators the runner itself uses.

An `execute` subcommand is registered for future wiring but fails closed
immediately with an authorization error and exit code 2 — it never imports
`src.dfp_phase1b_lightweight_boundary_execute` (which would transitively
import torch through the kernel module), so invoking it cannot initialize
CUDA, construct a model, or access data under any circumstance.

## 8. What this runner does not do

No candidate is selected. No confirmation-split image is ever read (any ID
present in `confirmation_ids` is rejected). No image outside the frozen
350-image development population is processed. No operational (15%/10%)
profiling gate is applied — `ProfilingRecorder` records only raw wall time,
per-image latency, and host RSS, with an optional injectable GPU-telemetry
adapter that stays `None` in every CPU synthetic test; none of this affects
any scientific array or digest. No `sbatch`/`srun`/`salloc` invocation
exists anywhere in these files. No GT pixel is ever read before every
prediction for an image is finalized.

## 9. Defect repair (independent-verification follow-up)

An independent adversarial verification pass found four confirmed defects
in the original implementation; all four are fixed as of this section
and reverified. Nothing about the BALP science, the E22 identity, the
16-candidate definitions, the 350/150 split, the metric formulas, or the
authorization boundary changed.

1. **Severe — checkpoint installation was not crash-atomic.** The
   original layout replaced a single canonical `checkpoint_arrays.npz`
   and then a single canonical `checkpoint.json` sequentially. A failure
   between those two replacements (a crash, a disk-full condition, a
   kill signal) left the old JSON pointing at a `development_digest`
   that no longer matched the just-replaced NPZ bytes — a permanently
   unloadable, unresumable checkpoint, with no way to recover the
   previous valid state. §5 above describes the fix: immutable
   per-generation directories installed via one atomic directory
   rename, made current only via a separate atomic single-file pointer
   replace performed strictly after the new generation has been
   reopened from disk and reverified. The original failure mode (NPZ
   successfully installed, JSON write fails) is no longer representable
   as a corrupted current checkpoint: under the new layout that failure
   can only occur while still inside a private staging directory, before
   anything is ever renamed into a real, visible generation directory —
   so the previous current generation is untouched either way.
2. **Moderate — first-use import depended on an unrelated side
   effect.** `process_development_image` imported
   `models.dinotext.cover_dr.graph` before any of the
   `src.dfp_phase1a_*` modules that (as a necessary consequence of their
   own normal operation) make the top-level `models`/`segmentation`
   packages importable at all. Existing tests masked this because a
   fixture happened to import one of those modules first; called fresh
   in a new process, it raised `ModuleNotFoundError`. Fixed by
   reordering the function's own lazy imports so a genuinely-required
   `src.dfp_phase1a_*` import always precedes the `models.dinotext`
   import, making the resolution deterministic from the function's own
   source order alone — no new `sys.path` mutation was added anywhere.
   Verified hermetically: a fresh interpreter, a sanitized `PYTHONPATH`
   containing only the repository root, a fresh `/tmp` cwd, and
   `process_development_image` as the first-ever call in the process,
   repeated with several different prior-import orderings.
3. **Moderate — corrupted NPZ could leak a raw exception.** A
   truncated or corrupted `checkpoint_arrays.npz` could raise
   `zipfile.BadZipFile` (or, for some corruption shapes, a raw
   `AttributeError` from code that assumed a member was already a valid
   `numpy.ndarray`) straight through `load_full_checkpoint` and the
   CLI. Fixed by funneling every NPZ open/read through `_read_npz_flat`
   (wrapping `zipfile.BadZipFile`/`OSError`/`ValueError`/`EOFError`,
   with the original exception preserved as `__cause__`) and by
   `_require_npz_ndarray` (rejecting a member that parsed as something
   other than an array, for the corruption shapes numpy itself doesn't
   raise on). `KeyboardInterrupt`/`SystemExit`/`MemoryError` are never
   caught by either.
4. **Minor — GPU telemetry adapter output was unvalidated.**
   `ProfilingRecorder.gpu_snapshot()` returned whatever an injected
   telemetry adapter's `snapshot()` produced, unchecked. Fixed by
   `_validate_gpu_telemetry_snapshot`, using E22's own
   `profiling_schema.per_variant_record` field names
   (`gpu_peak_allocated_bytes`, `gpu_peak_reserved_bytes`): exact `int`
   only (bool and float — including NaN/Infinity — rejected), the
   closed two-key set, non-negative, and `allocated <= reserved`. This
   remains a raw-measurement validator only — E22's profiling schema
   does not record a device-identifier field to cross-check, and this
   task still implements no paired T320/BALP benchmark and applies no
   operational gate.

## 10. Orphan-successor recovery (second independent-verification follow-up)

A second, narrower independent verification pass found that §9's crash-atomic
checkpoint repair, while it eliminated data corruption, introduced a *permanent
resumability deadlock*: if a failure landed strictly between a new generation
directory being successfully installed (the atomic directory rename already
succeeded) and the current pointer successfully advancing to reference it, the
orphaned generation permanently occupied its generation number. Since the next
run always re-derives `current_pointer.generation + 1`, every subsequent retry
recomputed the same already-occupied number and failed forever with
`generation directory ... already exists`.

`art.reconcile_orphan_successor` repairs this. It runs unconditionally at the
very start of `execute_development`, immediately after loading the current
checkpoint (if any) and **before any dataset, model, graph, recurrence,
image-processor, or GT provider is ever touched**:

- **No successor present.** Nothing to do; execution proceeds normally.
- **Exactly one successor present, at the canonical slot `generation_<N+1>`**
  (or `generation_0` when there is no current checkpoint yet). It is treated
  as a possibly-recoverable interrupted commit and validated *completely*
  before ever being trusted:
  - directory-name/logical-generation-number agreement, path containment,
    and symlink-escape rejection (the same checks the pointer loader itself
    uses);
  - both member files present, NPZ structurally valid, sibling digest bound
    correctly;
  - `identity_bindings` (everything except the per-generation
    `development_digest`) match this run's own bindings exactly -- a
    successor from a foreign run is never adopted;
  - full closed-schema and E22 relational validation via the same
    `validate_full_checkpoint` every other installed generation goes
    through -- this alone rejects an invalid completed-ID prefix, an
    inserted confirmation ID, or an illegal complete/final state;
  - every `checkpoint_schema.cross_generation_invariants` rule against the
    current generation, via E22's own committed `validate_checkpoint_resume_pair`
    (exact-prefix `completed_image_ids`, byte-identical already-completed
    per-image rows, `aggregate_confusion` equal to the old aggregate plus
    exactly the newly appended per-image matrices) -- run with
    `development_digest` neutralized to a shared placeholder in both probe
    copies first, since that one field is intentionally different for every
    generation and is already checked separately;
  - the successor must extend the current checkpoint by **exactly one**
    image (or, with no current checkpoint, represent exactly the first
    image) -- never more, never fewer.

  Only if every check passes does adoption happen: the current pointer is
  atomically advanced to the existing, already-installed generation (its
  contents are never recomputed, rewritten, or touched), then reopened
  through the public loader and revalidated before continuing. The image
  the adopted generation already represents is never reprocessed.
- **A successor is present but fails any check.** Fails closed with the
  declared `DfpPhase1bLightweightBoundaryArtifactError`, naming what failed.
  The invalid successor is never overwritten, deleted, renamed, or skipped
  past, and the current valid generation is left untouched.
- **The filesystem state at the successor slot is ambiguous** -- more than
  one non-historical directory present, a generation gap, or a present
  directory whose name disagrees with the single canonical expected
  name -- fails closed the same way. Nothing is ever selected by mtime,
  directory listing order, or filename sort; only the one canonically-named
  slot is ever inspected.

This module implements no garbage collection and never deletes a directory
under `checkpoint_generations/` for any reason, valid or invalid, referenced
or orphaned -- an invalid or superseded generation is left in place for
forensic inspection, exactly as before this repair.
