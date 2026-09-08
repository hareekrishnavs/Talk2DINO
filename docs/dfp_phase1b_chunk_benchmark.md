# DFP Phase 1B Task 3 — Fixed Chunk-Size Performance Benchmark

```text
This benchmark measures operational performance only.
It does not change scientific outputs, select a conservation tolerance,
authorize full Phase 1B, or execute the 5,000-image evaluation.
```

Authority: `evaluation_identities/e17_dfp_phase1b_chunk_benchmark.toml`
("E17"). E17 is a new leaf authority: it binds, but never modifies or
reinterprets, E16, the completed and independently-verified E16
calibration artifact, the independently-verified Task 2 v2
workload-report bundle, the frozen COCO-Stuff-500 manifest, and every
scientific implementation module the calibration runner already
reuses (graph/recurrence, masked-forward, stitching, final-rescale).

## Operational purpose

Compares the wall-clock/memory cost of the already-committed
`component_chunk_size` parameter of
`src.dfp_phase1b_calibration_run.process_selected_image` at three
fixed values — 16, 32, 64 — on identical real hardware (one NVIDIA
H100 80GB HBM3), model checkpoint, dataset, window manifest, and
precision configuration. It selects no chunk size, freezes no
tolerance, and authorizes no full Phase 1B or 5,000-image execution.

## Exact 24-window selection formula

The benchmark subset is chosen **before any benchmark timing or
component count is ever observed**, from the already-completed and
independently-verified E16 calibration artifact's own 200
`selected_window_records` — never a fresh selection, never re-derived
from the candidate universe.

1. `node_count = (crop_height // patch_size) * (crop_width // patch_size)
   = (448 // 14) * (448 // 14) = 1024` (frozen protocol constants,
   identical for every window).
2. `query_class_count = rho_0.sample_count // node_count`, cross-checked
   against `rho_320.sample_count // node_count`, for each of the 200
   rows — exactly reconstructable from the artifact alone (this is the
   same derivation independently verified in the Task 2 v2 workload
   report).
3. Sort the 200 rows by `(query_class_count ascending, selection_rank
   ascending)`.
4. For benchmark slot `i` in `0..23`, choose the zero-based sorted
   position `floor(((2*i + 1) * 200) / (2 * 24))`.
5. All 24 positions and candidate records must be unique (verified).
6. The 24 records are preserved in slot order.
7. The **warm-up record** is the smallest original `selection_rank` not
   present among the 24 measured records.

This selection is fully invariant to dictionary/filesystem/hash-seed
order and to component count, runtime, memory, or residual magnitude —
see `tests/test_dfp_phase1b_chunk_benchmark.py`'s order/hashseed
invariance tests.

The frozen manifest lives at
`evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json` and records
the source artifact/E16 hashes, the selection formula, the 24 records,
the warm-up record, and two digests (`benchmark_record_digest`,
`canonical_manifest_digest`) that any loader independently recomputes.

## Warm-up record

Exactly one window, processed through the real production path with
the trial's own chunk size, **before** timing/memory measurement
begins. It never contributes to any timing summary or scientific-output
digest — it exists only to reach steady-state CUDA/cuDNN/allocator
state before the clock starts.

## Six-trial schedule

One committed SLURM allocation runs **six fresh child Python
processes**, strictly sequentially, in this exact counterbalanced
order (protects against first/last-execution ordering effects — warm
hardware/cache/clock-boost state):

| trial_index | repeat | chunk_size |
|---|---|---|
| 0 | 0 | 16 |
| 1 | 0 | 32 |
| 2 | 0 | 64 |
| 3 | 1 | 64 |
| 4 | 1 | 32 |
| 5 | 1 | 16 |

A fresh process is mandatory for every trial: host peak RSS is only
meaningful in a fresh process, and no CUDA/allocator/model state may
leak across chunk sizes.

## Timing boundaries

**Primary** (`time.perf_counter_ns`, integer nanoseconds): starts after
process creation, imports, identity verification, dataset/model
construction, checkpoint loading, the warm-up window, and the initial
CUDA synchronize; stops after all 24 measured records finish and a
final CUDA synchronize. Excludes startup, model loading, warm-up, JSON
serialization, and final artifact installation.

**Secondary** (chunk-sensitive/masked-forward-specific): recorded only
where an existing safe call boundary permits it without duplicating or
altering any scientific operation; recorded as unavailable otherwise.

**Per-record**: synchronized timing per canonical-image **group** only
(one `process_selected_image` call per image) — never divided among
that image's individual windows and presented as a measured per-window
time.

## Host/GPU memory definitions

GPU: `torch.cuda.reset_peak_memory_stats()` immediately after warm-up
and synchronization; then `gpu_baseline_allocated/reserved_bytes`,
`gpu_peak_allocated/reserved_bytes`, `gpu_end_allocated/reserved_bytes` —
synchronized before the final read. **These ARE scoped to the measured
24-window subset only** (reset after warm-up, read after the last
measured window).

Host: `resource.getrusage(...).ru_maxrss` (Linux KiB, converted to
bytes explicitly, field `host_process_lifetime_peak_rss_bytes`) plus
`/proc/self/status` `VmHWM` (field `host_process_lifetime_vmhwm_bytes`)
and `VmRSS` (field `host_current_rss_bytes`, a CURRENT sample, never a
peak). **`ru_maxrss`/`VmHWM` cannot be reset mid-process on Linux, so
they unavoidably cover the ENTIRE trial process** — imports, model
construction, checkpoint loading, and warm-up — **not the measured-
24-window subset alone.** This is structurally different from the GPU
fields above; the field names make the distinction explicit, and no
field may claim host memory is measured-subset-scoped. A model-setup-
dominated trial (large one-time allocation during checkpoint loading,
small measured-window allocations) still correctly reports the host
figure as the process-lifetime peak, not the smaller measured-subset
figure.

**Safety guard** (operational abort, not a scientific tolerance):
checked between measured image groups — fails cleanly and leaves the
trial incomplete if host RSS or GPU reserved memory exceeds 90% of the
80 GiB allocation. An aborted trial is never marked complete/final.
Every raw reading the guard compares against a threshold must be an
exact, finite, nonnegative integer byte count (`type(value) is int and
value >= 0`); a `bool`, `float` (including `NaN`/`±Infinity`), negative
integer, numeric string, `None`, or any other non-`int`-typed value is
rejected via the declared domain error before ever reaching the
comparison -- this is what prevents a corrupted/non-finite reading from
silently passing (`float('nan') > threshold` is always `False` in
Python, which would otherwise let a poisoned reading through
unnoticed).

`/proc/self/status`'s `VmRSS`/`VmHWM` fields are parsed strictly: each
must appear exactly once, as a plain decimal nonnegative integer
followed by the literal unit `kB`. The silent-fallback path (host
figures degrade to `ru_maxrss`-only) is taken **only** when
`/proc/self/status` is genuinely unavailable (file missing, permission
denied, no `/proc` at all) -- an ordinary filesystem-availability
`OSError` on the read itself. If the file **is** readable but its
content is malformed (a missing/duplicated field, wrong unit, decimal/
scientific-notation value, negative value, extra token, or invalid
UTF-8), parsing fails closed with the declared domain error -- it is
never silently dropped and never leaks a raw `ValueError`/`IndexError`/
`KeyError`.

Every trial document's `memory` field is itself closed-schema
validated (`MEMORY_FIELD_KEYS`): exactly the renamed, scope-explicit
fields above and the GPU fields, and nothing else -- in particular the
old, ambiguous `host_peak_rss_bytes` name (or any other unknown key)
is rejected outright rather than silently ignored, at both trial
construction and independent verification time.

## Benchmark-code provenance

E17 hash-binds the benchmark's own executable code -- the identity
loader (`src/dfp_phase1b_chunk_benchmark_identity.py`), the runner
(`src/dfp_phase1b_chunk_benchmark.py`), the CLI
(`diagnostics/run_dfp_phase1b_chunk_benchmark.py`), and the SLURM
launcher (`dfp_phase1b_chunk_benchmark_h100.sbatch`) -- not merely the
E16 scientific dependency chain. `run-trial` re-hashes all four files
live from disk (already checked against E17's own pinned values inside
`load_identity`) and records `identity_loader_sha256`, `runner_sha256`,
`cli_sha256`, `launcher_sha256`, `repository_commit`,
`required_ancestor_commit`, and `repository_worktree_clean` in every
trial document's `code_provenance` field. `merge_trials` requires all
six trials to agree exactly on `code_provenance`, and `verify-trial`/
`verify-result` independently re-verify it.

**Every dependency-binding path is an exact canonical repository-
relative path, not a freely-replaceable field.** Each of E17's
repository-file bindings (the twelve scientific/E16 dependencies, the
manifest, and the four benchmark-code files above) has a fixed,
pinned expected `relative_path` in `CANONICAL_DEPENDENCY_RELATIVE_PATHS`
(`src/dfp_phase1b_chunk_benchmark_identity.py`); `load_identity`
requires the E17 document's own `relative_path` for that binding to
equal it EXACTLY, checked entirely in-memory before any filesystem
access. This is what rejects swapping two otherwise-valid paths, or
substituting a different in-repository file whose hash is then
recomputed to match -- regardless of what that recomputed hash says.

**Path containment is checked after resolution, not by string
prefix.** The sole path through which any dependency-binding file is
ever opened for hashing is `resolve_repository_relative_path`: it
first rejects, lexically, any absolute path (POSIX or Windows drive/
UNC), backslash, `..`/`.`/empty path component, or NUL/whitespace --
entirely before touching the filesystem -- then resolves the candidate
against the FULLY RESOLVED (symlink-chased) repository root and
requires the FULLY RESOLVED (symlink-chased) target to remain
physically contained beneath that root (via `Path.relative_to`, never
a string-prefix check, which is unsafe for paths like
`/repo-root-other/file`) and be an existing regular file. Absolute-
path substitution, lexical traversal, and symlink escapes (through
either the final path entry or an intermediate directory) are all
rejected this way -- for every repository-file binding, not merely the
four benchmark-code files. A repository root itself reached through a
symlink is accepted; containment is still decided against its resolved
physical path.

**Ancestry, not equality.** `required_ancestor_commit` (E17's own
frozen prerequisite base commit) is required to be an ANCESTOR of (or
equal to) the executing `repository_commit` -- via Git's own commit
graph (`git merge-base --is-ancestor`), never exact equality. The
final benchmark implementation commit is legitimately a descendant of
that frozen base. At real execution time, `run-trial` additionally
requires the current worktree to be clean (including untracked files)
before any torch/CUDA/model import -- a dirty checkout is rejected
immediately. At later, independent verification time, the verifier's
own current HEAD is never required to equal the historical trial
commit; only the recorded ancestry relation and code hashes are
re-checked.

## Scientific-output digest comparison — no tolerance

For each of the 24 windows, a canonical record containing only the
four raw residual-family summaries (`rho_M`, `rho_0`, `rho_320`,
`rho_g`) plus canonical identity fields (never runtime, memory, device
UUID, timestamp, chunk size, or trial/repeat number) is SHA-256
digested; the 24 digests are combined into one ordered collection
digest.

- Repeat 0 and repeat 1 **must** match for the same chunk size.
- Chunk sizes 16, 32, and 64 **must** be bitwise/digest identical.
- **No numeric tolerance is allowed. No rounding is allowed before
  comparison.** A mismatch produces `SCIENTIFIC_OUTPUT_MISMATCH`,
  blocks any later chunk-size decision, and cannot be dismissed with an
  invented tolerance.

## No winner selection in Task 3

Neither the per-trial artifact nor the merged artifact contains a
winner, recommendation, or preferred-chunk-size field. The merged
artifact computes only descriptive summaries (elapsed ns per repeat,
median/min/max, GPU/host peak bytes, throughput over the fixed
24-window set) per chunk size.

`verify-result` relationally re-verifies the merged artifact's
persisted `descriptive_summaries_by_chunk_size` against the SAME
recomputation `merge_trials` already performs from the document's own
`raw_trials` -- using exact, deterministic, type-aware structural
equality (never plain `==`, which would let an integral-looking float
such as `600.0` silently pass in place of the int `600`, or a `bool`
pass in place of an `int`). A stored summary that disagrees with its
own recomputation -- whether hand-edited, corrupted, or containing a
non-standard JSON `NaN`/`Infinity` token -- is rejected with the
declared domain error before printing any PASS. This is a relational
check against the existing recomputation path, not a new summary
formula, and not a scientific digest: E17 continues to treat these
summaries as descriptive only.

## No extrapolation to 5,000 images

Nothing in this benchmark's schema, CLI, or launcher touches the held
5,000-image evaluation; no sharding/merging logic for it exists here.

## No authorization for full Phase 1B

A successful merged benchmark result does not authorize the full Phase
1B execution driver — that remains gated behind E16's own successor
authority (freezing a numeric conservation tolerance), entirely
untouched by this task.

## Expected output files

- `evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json` — the frozen
  24-window manifest (committed to the repository).
- Six per-trial artifacts (`<output_root>/trial_<trial_index>_repeat<repeat>_chunk<chunk_size>.json`),
  each independently verifiable via `verify-trial`.
- One merged artifact (`<output_root>/merged_result.json`),
  independently verifiable via `verify-result`.

## CLI

```text
diagnostics/run_dfp_phase1b_chunk_benchmark.py
    verify-protocol   verify E17: closed schema + relational hashes (no torch)
    build-manifest    build/install the 24-window manifest (no torch)
    verify-manifest   independently re-verify an installed manifest (no torch)
    preflight         full repository-readiness check: identity, manifest, git ancestry/clean-worktree (no torch, no CUDA)
    run-trial         run exactly one fresh-process trial (requires --device cuda)
    verify-trial      verify one trial artifact's schema/bindings (no torch)
    merge             merge exactly six trial artifacts (no torch)
    verify-result     independently re-verify a merged result (no torch)
```

`preflight` and `run-trial` both call the SAME shared
`perform_repository_readiness_check` function (never two
implementations that may drift): repository root resolution, E17
identity load (including canonical-path pinning and containment),
benchmark manifest re-verification, current-HEAD/required-ancestor
commit validity, the ancestor relation itself, full working-tree
cleanliness (unstaged tracked changes, staged changes, AND untracked
files), and -- when an output path is supplied -- that it is lexically
safe and either absent or `--force`d. All of this completes before any
torch/CUDA import, model/dataset construction, checkpoint loading, or
output creation. `run-trial` re-runs this exact check itself; it never
relies solely on the launcher having already run `preflight`.

Because the real implementation is still uncommitted (the eight files
listed at the top of this document remain untracked until this task's
commit), a readiness check run against the real repository's current
working tree correctly FAILS at the clean-worktree step -- this is
expected, intentional behavior, not a defect.

## Failure and cleanup behavior

Every domain error (`DfpPhase1bChunkBenchmarkError`,
`DfpPhase1bChunkBenchmarkIdentityError`) exits 2 with a clean,
prefixed message and no traceback.
`KeyboardInterrupt`/`SystemExit`/`MemoryError` always propagate
unmodified. No subcommand overwrites an existing output file unless
`--force` (or `--allow-overwrite`, for the merge) is passed explicitly.
Trial and merged artifacts are written atomically (temp file +
`os.replace`) — no partial file is ever left in place of an existing
one, and a failed write leaves no `.tmp` residue.

## Separate submission workflow

This implementation task never submits a SLURM job and never runs the
real benchmark. After an independent implementation-verification PASS
and a commit, submit the six-trial allocation separately with:

```bash
sbatch dfp_phase1b_chunk_benchmark_h100.sbatch
```

with the required environment variables exported first — see that
script's own header comment for the exact list (mirrors
`dfp_phase1b_calibration_coco_stuff_200.sbatch`'s established
convention).
