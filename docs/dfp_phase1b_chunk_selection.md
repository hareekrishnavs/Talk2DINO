# DFP Phase 1B Task 4 — Deterministic Chunk-Size Selection

```text
This is an operational engineering selection, not a scientific result.
It does not change scientific outputs, select a conservation tolerance,
authorize full Phase 1B, or execute the 5,000-image evaluation.
```

Authority: `evaluation_identities/e18_dfp_phase1b_chunk_selection.toml`
("E18"). E18 is an **additive successor** to E17: it binds, but never
modifies or reinterprets, E17, the completed and independently-verified
E17 benchmark output (the real `merged_result.json` and its six raw
trial files from SLURM job `21331697`, produced at committed HEAD
`e4caef84b98d048f160dc935282679ed20efcc99`), the benchmark manifest,
and the Task 3 runner/CLI. It never touches E13–E16 or the calibration
artifact/workload bundle those bind.

## What Task 3 measured

Task 3 ran six fresh-process trials — two repeats each of chunk sizes
16, 32, and 64, in the frozen counterbalanced schedule
`16,32,64,64,32,16` — over the same fixed 24-window benchmark subset,
on identical hardware (one NVIDIA H100 80GB HBM3), model checkpoint,
dataset, and precision configuration. It measured wall-clock elapsed
time (`timing.primary_elapsed_ns`) and GPU/host peak memory. It
selected no chunk size and froze no tolerance — see
`docs/dfp_phase1b_chunk_benchmark.md` for the complete account.

## Why chunk 16 was selected

Chunk 16 was **not** hard-coded as the answer. It is the output of a
deterministic rule (below), applied by `src/dfp_phase1b_chunk_selection.py`
to the real, independently-verified six-trial result, after
independently reconstructing every input from scratch. Chunk 16 is not
claimed to be scientifically superior — all three chunk sizes produced
**bitwise-identical scientific output** (a single shared digest,
`0ab0dcf6...`, across all six trials); chunk 16 is simply the fastest
**safe** candidate for this exact hardware/software/model binding.

## Exact selection rule

**Candidates:** `[16, 32, 64]`.

**Eligibility** — a candidate is safe only if *both* of its trials
satisfy every one of:

1. `complete` and `final` are both `true`.
2. The trial passes `src.dfp_phase1b_chunk_benchmark.verify_trial_document`
   (the committed Task 3 verifier) — schema, code-provenance, memory,
   and timing fields.
3. `(trial_index, repeat_index, chunk_size)` is a member of the frozen
   E17 trial schedule.
4. `len(window_identity) == 24`.
5. `timing.primary_elapsed_ns` is a finite, non-negative exact integer.
6. Every `memory.*` field is a finite, non-negative exact integer.
7. `abort_status.aborted` is `false`.
8. `device_record.gpu_model_name`/`device_type` match E17's required
   H100/`cuda` binding.
9. `scientific_output_digests.collection_digest` equals the common
   digest shared by all six trials.
10. `memory.host_process_lifetime_peak_rss_bytes` and
    `memory.gpu_peak_reserved_bytes` do not exceed E17's own
    `[measurement.safety_guard]` thresholds (`0.90 * 80 GiB` each,
    `usage > threshold` is unsafe) — **no new threshold is invented**;
    E18 re-derives these relationally from the live E17 document on
    every load.

**Primary statistic:** the arithmetic mean, in nanoseconds, of the two
complete trials' `timing.primary_elapsed_ns` values — exactly Task 3's
own frozen primary-timing field. This excludes warm-up, preflight,
merge/verification time, and model/dataset construction (all excluded
from `primary_elapsed_ns` by E17's own timing-boundary definition).

**Winner:** the eligible candidate with the smallest exact mean —
compared with no rounding and no epsilon.

**Tie-break sequence** (applied only on exact equality of the
statistic above):

1. Lower `max(host_process_lifetime_peak_rss_bytes)` across the
   candidate's two trials.
2. Lower `max(gpu_peak_reserved_bytes)` across the candidate's two
   trials.
3. Smaller numerical chunk size.

## Exact per-trial and aggregate results (job 21331697)

| chunk | repeat 0 elapsed | repeat 1 elapsed | mean elapsed | host peak RSS max | GPU reserved max |
|---|---|---|---|---|---|
| 16 | 532.762109651 s | 448.460212602 s | **490.611161126.5 ns** (490.611 s) | 15,295,516,672 B | 8,006,926,336 B |
| 32 | 609.678129698 s | 435.979034279 s | 522.828581988.5 ns (522.829 s) | 27,382,038,528 B | 7,998,537,728 B |
| 64 | 584.577861113 s | 464.970159718 s | 524.774010415.5 ns (524.774 s) | 51,819,261,952 B | 7,998,537,728 B |

All three candidates are **safe** (well under the ~77.3 GB host/GPU
safety thresholds). No tie occurred — chunk 16 is the unique minimum,
**6.567% faster than chunk 32 and 6.963% faster than chunk 64** in mean
elapsed time. These are descriptive differences over two repeats each,
**not a claim of statistical significance**.

## Symmetric trial ordering and host-memory scaling

The frozen schedule (`16,32,64,64,32,16`) places each chunk size once
in the first half and once in the second half of the six sequential
trials, so first/last-execution warm-hardware and cache-boost effects
are roughly balanced across chunk sizes — this **reduces but does not
eliminate** temporal/cache-order effects; it is not a substitute for
more repeats. Separately, host process-lifetime peak RSS scales
sharply with chunk size — chunk 32 uses roughly 1.8× chunk 16's peak
RSS, and chunk 64 roughly 3.4× — a real operational cost, even though
it did not end up binding the safety guard here.

## Limitations (frozen in E18's `[disclosure]` table)

- This is an **operational engineering selection**, not a scientific
  metric.
- The selection rule was **frozen before being applied**, but was
  written *after* observing that the benchmark existed — this is a
  **post-hoc** engineering decision, not a preregistered scientific
  hypothesis test.
- **Two repeats do not support a statistical-significance claim.**
- Symmetric trial order reduces, but does not eliminate,
  temporal/cache-order effects.
- The selection does not change any scientific result.
- The selection **does not** define the conservation tolerance.
- The selection **does not** authorize full Phase 1B execution.
- The selection **does not** authorize or implement the 5,000-image
  evaluation.

## Files

| File | Role |
|---|---|
| `evaluation_identities/e18_dfp_phase1b_chunk_selection.toml` | E18 authority: bindings, selection contract, safety contract, frozen selection result, authorization boundary, disclosure. |
| `src/dfp_phase1b_chunk_selection.py` | CPU-only selector/validator. Never imports torch. Independently re-verifies all six raw trials and the merged result via the real, committed Task 3 verifier/merger before ever computing a winner. |
| `diagnostics/analyze_dfp_phase1b_chunk_benchmark.py` | CLI: `verify-protocol`, `analyze`, `verify-selection`. |
| `tests/test_dfp_phase1b_chunk_selection.py` | Adversarial + real-artifact test suite. |

## CLI usage

```bash
module load opencv/4.14.0
source /scratch/haree/venv/talk2dino-a100/bin/activate

# Verify the E18 identity itself (closed schema + relational hashes,
# cross-checked against the live E17). No torch.
python diagnostics/analyze_dfp_phase1b_chunk_benchmark.py verify-protocol \
    --merged-result /scratch/haree/dfp_phase1b_chunk_benchmark/21331697/merged_result.json

# Independently re-verify the real benchmark output and recompute the
# selection from scratch. Reads the benchmark; runs no inference.
python diagnostics/analyze_dfp_phase1b_chunk_benchmark.py analyze \
    --output-dir /scratch/haree/dfp_phase1b_chunk_benchmark/21331697

# Full boundary: E18 + benchmark output re-verification + require the
# recomputed winner to equal E18's recorded selected_chunk_size.
python diagnostics/analyze_dfp_phase1b_chunk_benchmark.py verify-selection \
    --output-dir /scratch/haree/dfp_phase1b_chunk_benchmark/21331697
```

No subcommand accepts a GPU/device flag, submits a SLURM job, or offers
a 5,000-image option. Every expected failure exits 2 with no traceback.

## Output-path safety policy

`--output`/`--markdown-output` (and the direct API's `basename`
arguments) are **safe basenames only, never paths**. Both are resolved
directly under the explicit `--output-dir` (the same directory the six
real trials and `merged_result.json` are read from) — there is no
independent, arbitrary destination flag. This closes a finding from
independent verification: earlier revisions accepted an arbitrary
`--output <path>`, which a lexically-traversing value (`../escape.json`)
could use to write outside the intended directory.

Rejected: absolute paths, any `/`- or `\`-containing value (which
would make it a path rather than a bare name), the exact components
`.`/`..`, a Windows drive prefix, an empty/whitespace-only value, a NUL
byte, a directory target, an existing symlink at that name, and the
wrong required suffix (`.json` for `--output`, `.md` for
`--markdown-output`). The **substring** `..` inside an otherwise-safe
basename (e.g. `run..v2.json`) is explicitly permitted — only the exact
path *component* `..` is rejected. `--output-dir` itself may be any
absolute or relative operator-selected directory (no repository- or
`/scratch`-only restriction); it is validated independently (no lexical
`..`, no NUL, must resolve to an existing directory) but is not itself
constrained to a fixed location.

The same validator backs both the CLI and the direct Python API
(`write_selection_artifact_atomically`, `write_markdown_report`,
`install_selection_bundle`) — the direct API independently rejects an
unsafe destination even when called without the CLI. When both a JSON
and a Markdown destination are requested together, `install_selection_bundle`
installs them as one atomic pair: both destinations are validated and
both payloads fully generated *before* either temporary file is
written, so a failure at any point leaves neither destination
installed or modified.

## Regenerating and verifying the selection artifact after commit

This implementation task deliberately does **not** install a durable
`phase1b_chunk_selection.json` under `/scratch` — only a preview,
generated by pointing `--output-dir` at a **copy** of the real
benchmark output under `/tmp` (never at the real `/scratch` directory
itself when only previewing):

```bash
cp /scratch/haree/dfp_phase1b_chunk_benchmark/21331697/{trial_*.json,merged_result.json} /tmp/<scratch-dir>/preview/
python diagnostics/analyze_dfp_phase1b_chunk_benchmark.py analyze \
    --output-dir /tmp/<scratch-dir>/preview \
    --output phase1b_chunk_selection.json \
    --markdown-output phase1b_chunk_selection_report.md
```

After independent re-verification and commit of E18, install the
durable artifact under its committed `/scratch` destination by
re-running the same `analyze --output ...` command with `--output-dir`
pointed at the real `/scratch/.../21331697` directory (never overwriting
an existing installed artifact by default — pass `--force` only
deliberately). To verify an installed artifact later, re-run
`verify-selection` — it never trusts the artifact's own recorded
winner, and now recomputes the *entire* recorded decision payload
(every candidate's both repeats, all memory maxima, safety flags and
reasons, the selected chunk, and the selected mean) from the six raw
trials, not merely the winner and the scientific digest.
