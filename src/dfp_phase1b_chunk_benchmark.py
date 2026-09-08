"""DFP Phase 1B Task 3: fixed component-chunk-size performance
benchmark (E17-authorized).

This module implements NO scientific mathematics of its own: window
selection is pure arithmetic over already-persisted E16 artifact
fields; per-trial execution reuses the already-committed
``src.dfp_phase1b_calibration_run.process_selected_image`` (and the
real production provider wiring in
``src.dfp_phase1b_calibration_production``) completely unmodified,
through its own existing dependency-injection seam
(``component_chunk_size`` is already a first-class parameter of that
function). This module never imports torch, mmcv, or any model/
dataset module at the top level -- every heavy import is local to a
function body, exactly matching the calibration runner's own
convention.

Operational only: selects no chunk size, freezes no tolerance,
authorizes no full Phase 1B or 5,000-image execution. See
``evaluation_identities/e17_dfp_phase1b_chunk_benchmark.toml``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from src.dfp_phase1b_chunk_benchmark_identity import (
    BENCHMARK_WINDOW_COUNT,
    CHUNK_SIZES,
    NODE_COUNT_CONSTANT,
    TRIAL_SCHEDULE,
    DfpPhase1bChunkBenchmarkIdentityError,
    MemoryGuardContract,
    verify_ancestor_relation,
)


class DfpPhase1bChunkBenchmarkError(RuntimeError):
    """Raised on any DFP Phase 1B chunk-benchmark invariant violation.
    Always fail closed. Never raised for, and never catches,
    KeyboardInterrupt, SystemExit, or MemoryError -- those always
    propagate unmodified."""


MANIFEST_SCHEMA_NAME = "talk2dino-dfp-phase1b-chunk-benchmark-manifest-v1"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_NAMESPACE = "talk2dino-dfp-phase1b-chunk-benchmark-manifest-v1"

TRIAL_SCHEMA_NAME = "talk2dino-dfp-phase1b-chunk-benchmark-trial-v1"
TRIAL_SCHEMA_VERSION = 1

MERGED_SCHEMA_NAME = "talk2dino-dfp-phase1b-chunk-benchmark-merged-v1"
MERGED_SCHEMA_VERSION = 1

REQUIRED_ARTIFACT_RELATIVE_PATH = "phase1b_conservation_calibration.json"


# ---------------------------------------------------------------------
# Canonical byte encoding -- mirrors E16's own convention exactly
# (`json.dumps(obj, sort_keys=True, separators=(',', ':'),
# ensure_ascii=True)`, UTF-8, no trailing newline) so the same
# `digest_over_records`-style length-prefixed concatenation formula
# produces reproducible, order-sensitive digests.
# ---------------------------------------------------------------------


def canonical_encode(obj: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(dict(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DfpPhase1bChunkBenchmarkError(f"record could not be canonically encoded: {exc}") from exc


def digest_over_records(encoded_records: Sequence[bytes]) -> str:
    hasher = hashlib.sha256()
    for record_bytes in encoded_records:
        if not isinstance(record_bytes, (bytes, bytearray)):
            raise DfpPhase1bChunkBenchmarkError("digest_over_records requires a sequence of bytes")
        hasher.update(len(record_bytes).to_bytes(8, "big"))
        hasher.update(bytes(record_bytes))
    return hasher.hexdigest()


# ---------------------------------------------------------------------
# Section: deterministic 24-window selection (Section 4 of the task
# spec). Pure arithmetic over already-persisted E16 artifact fields --
# never touches component count, runtime, memory, residual magnitude,
# filesystem order, or any GPU-observed quantity.
# ---------------------------------------------------------------------

_BENCHMARK_RECORD_FIELDS: tuple[str, ...] = (
    "run_mode", "image_id", "image_path", "image_manifest_ordinal",
    "inference_canvas_height", "inference_canvas_width", "canonical_window_ordinal",
    "top", "left", "bottom", "right", "crop_height", "crop_width",
    "patch_grid_height", "patch_grid_width",
    "canonical_candidate_index", "selection_rank", "selection_key",
    "query_class_count", "benchmark_slot",
)


def _query_class_count_of(row: Mapping[str, Any]) -> int:
    rho_0 = row["rho_0"]["sample_count"]
    rho_320 = row["rho_320"]["sample_count"]
    if rho_0 != rho_320:
        raise DfpPhase1bChunkBenchmarkError(
            f"row canonical_candidate_index={row['canonical_candidate_index']}: "
            f"rho_0.sample_count ({rho_0}) != rho_320.sample_count ({rho_320})"
        )
    if NODE_COUNT_CONSTANT <= 0 or rho_0 % NODE_COUNT_CONSTANT != 0:
        raise DfpPhase1bChunkBenchmarkError(
            f"row canonical_candidate_index={row['canonical_candidate_index']}: "
            f"rho_0.sample_count ({rho_0}) is not an exact multiple of node_count ({NODE_COUNT_CONSTANT})"
        )
    return rho_0 // NODE_COUNT_CONSTANT


def slot_position(i: int, *, source_count: int = 200, slot_count: int = BENCHMARK_WINDOW_COUNT) -> int:
    """``floor(((2*i + 1) * source_count) / (2 * slot_count))`` -- the
    frozen E17 slot-position formula."""
    return ((2 * i + 1) * source_count) // (2 * slot_count)


def select_benchmark_windows(selected_window_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Implements the frozen E17 selection algorithm (Section 4).
    ``selected_window_records`` must be the verified E16 artifact's own
    200 ``selected_window_records`` (any input order is accepted and
    ignored -- the algorithm re-sorts internally, so this is invariant
    to dictionary/filesystem/hash-seed order by construction). Returns
    a plain dict with ``benchmark_records`` (24, in slot order),
    ``warmup_record``, and the raw ``positions`` list -- never writes
    anything."""
    rows = list(selected_window_records)
    if len(rows) != 200:
        raise DfpPhase1bChunkBenchmarkError(f"selection requires exactly 200 source rows, got {len(rows)}")

    enriched = []
    for row in rows:
        qcc = _query_class_count_of(row)
        enriched.append({**row, "query_class_count": qcc})

    sorted_rows = sorted(enriched, key=lambda r: (r["query_class_count"], r["selection_rank"]))

    positions = [slot_position(i) for i in range(BENCHMARK_WINDOW_COUNT)]
    if len(set(positions)) != BENCHMARK_WINDOW_COUNT:
        raise DfpPhase1bChunkBenchmarkError("computed slot positions are not unique -- refusing to select")

    benchmark_records = []
    for slot, pos in enumerate(positions):
        row = sorted_rows[pos]
        record = {field: row[field] for field in _BENCHMARK_RECORD_FIELDS if field != "benchmark_slot"}
        record["benchmark_slot"] = slot
        benchmark_records.append(record)

    ranks = [r["selection_rank"] for r in benchmark_records]
    if len(set(ranks)) != BENCHMARK_WINDOW_COUNT:
        raise DfpPhase1bChunkBenchmarkError("selected benchmark records do not have unique selection_rank values")
    indices = [r["canonical_candidate_index"] for r in benchmark_records]
    if len(set(indices)) != BENCHMARK_WINDOW_COUNT:
        raise DfpPhase1bChunkBenchmarkError("selected benchmark records do not have unique canonical_candidate_index values")

    all_ranks = set(row["selection_rank"] for row in rows)
    warmup_candidates = sorted(all_ranks - set(ranks))
    if not warmup_candidates:
        raise DfpPhase1bChunkBenchmarkError("no selection_rank remains available for the warm-up window")
    warmup_rank = warmup_candidates[0]
    warmup_source = next(row for row in rows if row["selection_rank"] == warmup_rank)
    warmup_qcc = _query_class_count_of(warmup_source)
    warmup_record = {field: warmup_source[field] for field in _BENCHMARK_RECORD_FIELDS if field not in ("query_class_count", "benchmark_slot")}
    warmup_record["query_class_count"] = warmup_qcc

    return {"positions": positions, "benchmark_records": benchmark_records, "warmup_record": warmup_record}


def build_benchmark_manifest(
    *, calibration_document: Mapping[str, Any], calibration_artifact_sha256: str, e16_identity_sha256: str,
) -> dict[str, Any]:
    """Build the full, closed-schema
    ``dfp_phase1b_chunk_benchmark_24.json`` document (Section 4)."""
    if calibration_document.get("complete") is not True or calibration_document.get("final") is not True:
        raise DfpPhase1bChunkBenchmarkError("source calibration artifact must be complete=True and final=True")

    selection = select_benchmark_windows(calibration_document["selected_window_records"])
    benchmark_records = selection["benchmark_records"]
    warmup_record = selection["warmup_record"]

    encoded = [canonical_encode(r) for r in benchmark_records]
    benchmark_record_digest = digest_over_records(encoded)
    canonical_manifest_digest = digest_over_records(
        encoded + [canonical_encode(warmup_record)] + [canonical_encode({"positions": selection["positions"]})]
    )

    return {
        "schema_name": MANIFEST_SCHEMA_NAME,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_calibration_artifact_sha256": calibration_artifact_sha256,
        "e16_identity_sha256": e16_identity_sha256,
        "source_selected_windows_digest": calibration_document["selected_windows_digest"],
        "source_selection_trace_digest": calibration_document["selection_trace_digest"],
        "selection_algorithm": {
            "sort_key": "(query_class_count ascending, selection_rank ascending)",
            "slot_position_formula": "floor(((2*i + 1) * 200) / (2 * 24))",
            "slot_count": BENCHMARK_WINDOW_COUNT,
            "warmup_selection_rule": "smallest original selection_rank not present among the 24 measured records",
        },
        "positions": selection["positions"],
        "benchmark_records": benchmark_records,
        "warmup_record": warmup_record,
        "benchmark_record_digest": benchmark_record_digest,
        "canonical_manifest_digest": canonical_manifest_digest,
    }


MANIFEST_TOP_KEYS: frozenset[str] = frozenset(
    [
        "schema_name", "schema_version", "source_calibration_artifact_sha256", "e16_identity_sha256",
        "source_selected_windows_digest", "source_selection_trace_digest", "selection_algorithm",
        "positions", "benchmark_records", "warmup_record", "benchmark_record_digest", "canonical_manifest_digest",
    ]
)


def verify_benchmark_manifest(
    document: Mapping[str, Any], *, calibration_document: Mapping[str, Any] | None = None,
    calibration_artifact_sha256: str | None = None, e16_identity_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Closed-schema + relational validation of an installed benchmark
    manifest. When ``calibration_document`` is supplied, independently
    RE-DERIVES the 24-window selection from it and requires every
    digest to match -- never a mere internal self-consistency check."""
    actual_keys = frozenset(document.keys())
    if actual_keys != MANIFEST_TOP_KEYS:
        raise DfpPhase1bChunkBenchmarkError(
            f"benchmark manifest has the wrong key set -- missing={sorted(MANIFEST_TOP_KEYS - actual_keys)} "
            f"unknown={sorted(actual_keys - MANIFEST_TOP_KEYS)}"
        )
    if document["schema_name"] != MANIFEST_SCHEMA_NAME or document["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise DfpPhase1bChunkBenchmarkError("benchmark manifest schema_name/schema_version mismatch")

    records = document["benchmark_records"]
    if not isinstance(records, list) or len(records) != BENCHMARK_WINDOW_COUNT:
        raise DfpPhase1bChunkBenchmarkError(f"benchmark manifest must contain exactly {BENCHMARK_WINDOW_COUNT} benchmark_records")
    seen_slots, seen_ranks, seen_indices = set(), set(), set()
    for position, record in enumerate(records):
        actual_field_set = frozenset(record.keys())
        if actual_field_set != frozenset(_BENCHMARK_RECORD_FIELDS):
            raise DfpPhase1bChunkBenchmarkError(f"benchmark_records[{position}] has the wrong field set")
        if record["benchmark_slot"] != position:
            raise DfpPhase1bChunkBenchmarkError(f"benchmark_records[{position}].benchmark_slot must equal its own list position")
        seen_slots.add(record["benchmark_slot"])
        seen_ranks.add(record["selection_rank"])
        seen_indices.add(record["canonical_candidate_index"])
    if len(seen_slots) != BENCHMARK_WINDOW_COUNT or len(seen_ranks) != BENCHMARK_WINDOW_COUNT or len(seen_indices) != BENCHMARK_WINDOW_COUNT:
        raise DfpPhase1bChunkBenchmarkError("benchmark_records contain duplicate slots, ranks, or candidate indices")

    warmup = document["warmup_record"]
    if warmup["selection_rank"] in seen_ranks:
        raise DfpPhase1bChunkBenchmarkError("warmup_record.selection_rank must not be one of the 24 measured ranks")

    encoded = [canonical_encode(r) for r in records]
    recomputed_record_digest = digest_over_records(encoded)
    if recomputed_record_digest != document["benchmark_record_digest"]:
        raise DfpPhase1bChunkBenchmarkError("benchmark manifest benchmark_record_digest does not recompute")
    recomputed_manifest_digest = digest_over_records(
        encoded + [canonical_encode(warmup)] + [canonical_encode({"positions": document["positions"]})]
    )
    if recomputed_manifest_digest != document["canonical_manifest_digest"]:
        raise DfpPhase1bChunkBenchmarkError("benchmark manifest canonical_manifest_digest does not recompute")

    if calibration_artifact_sha256 is not None and document["source_calibration_artifact_sha256"] != calibration_artifact_sha256:
        raise DfpPhase1bChunkBenchmarkError("benchmark manifest source_calibration_artifact_sha256 disagrees with the supplied artifact")
    if e16_identity_sha256 is not None and document["e16_identity_sha256"] != e16_identity_sha256:
        raise DfpPhase1bChunkBenchmarkError("benchmark manifest e16_identity_sha256 disagrees with the supplied E16 identity")

    if calibration_document is not None:
        rederived = select_benchmark_windows(calibration_document["selected_window_records"])
        if rederived["benchmark_records"] != records:
            raise DfpPhase1bChunkBenchmarkError("benchmark manifest benchmark_records do not match a fresh re-derivation from the calibration artifact")
        if rederived["warmup_record"] != warmup:
            raise DfpPhase1bChunkBenchmarkError("benchmark manifest warmup_record does not match a fresh re-derivation")

    return document


# ---------------------------------------------------------------------
# Section: transparent instrumentation wrapper (Section 7). Calls the
# original function exactly once, with the exact original arguments,
# and returns the exact original result object unmodified -- collects
# metadata only around the call, never inside it.
# ---------------------------------------------------------------------


class _InstrumentedCall:
    __slots__ = ("component_ids", "query_class_indices", "window_index", "component_id_start", "component_id_stop")

    def __init__(self, *, component_ids, query_class_indices, window_index, component_id_start, component_id_stop):
        self.component_ids = list(component_ids)
        self.query_class_indices = list(query_class_indices)
        self.window_index = window_index
        self.component_id_start = component_id_start
        self.component_id_stop = component_id_stop


def instrument_masked_forward_module(masked_forward_module: Any) -> tuple[Callable[[], None], list[_InstrumentedCall]]:
    """Wrap ``masked_forward_module.build_masked_seed_chunk`` with a
    transparent instrumentation shim that records one
    ``_InstrumentedCall`` per invocation (component-chunk boundaries
    and the window each call belongs to) and otherwise behaves
    identically to the original. Returns ``(restore, calls)`` --
    callers MUST call ``restore()`` (a plain function, safe to call
    from a ``finally`` block) exactly once, even on exception, to put
    the module back exactly as it was. Never used to alter dtype,
    device, shape, order, or contents of any returned value; never
    swallows ``KeyboardInterrupt``/``SystemExit``/``MemoryError``."""
    original = masked_forward_module.build_masked_seed_chunk
    calls: list[_InstrumentedCall] = []

    def wrapper(unary_for_query, membership_slice, component_ids, query_class_indices, *, contract, window_index):
        calls.append(
            _InstrumentedCall(
                component_ids=component_ids, query_class_indices=query_class_indices, window_index=window_index,
                component_id_start=component_ids[0] if component_ids else None,
                component_id_stop=(component_ids[-1] + 1) if component_ids else None,
            )
        )
        # Call the ORIGINAL exactly once, with the EXACT original
        # arguments, and return the EXACT original result -- no
        # mutation, no re-derivation.
        return original(unary_for_query, membership_slice, component_ids, query_class_indices, contract=contract, window_index=window_index)

    masked_forward_module.build_masked_seed_chunk = wrapper

    def restore() -> None:
        masked_forward_module.build_masked_seed_chunk = original

    return restore, calls


def summarize_chunk_calls(calls: Sequence[_InstrumentedCall], *, window_index: int) -> dict[str, Any]:
    """Reduce the raw instrumented calls belonging to one window into
    the workload fields Section 7 requires: chunk count, per-chunk
    widths, full-vs-partial split, and (since every call for the same
    window shares the same image-level component_count) the implied
    component_count for that window's image."""
    window_calls = [c for c in calls if c.window_index == window_index]
    if not window_calls:
        return {
            "actual_component_chunk_count": 0, "component_chunk_widths": [], "full_chunk_count": 0,
            "partial_chunk_count": 0, "image_level_component_count": 0,
        }
    widths = [c.component_id_stop - c.component_id_start for c in window_calls]
    component_chunk_size_observed = max(widths) if widths else 0
    full = sum(1 for w in widths if w == component_chunk_size_observed)
    partial = sum(1 for w in widths if w != component_chunk_size_observed)
    image_level_component_count = window_calls[-1].component_id_stop
    return {
        "actual_component_chunk_count": len(window_calls),
        "component_chunk_widths": widths,
        "full_chunk_count": full,
        "partial_chunk_count": partial,
        "image_level_component_count": image_level_component_count,
    }


# ---------------------------------------------------------------------
# Section: scientific-output invariance digest (Section 9).
# ---------------------------------------------------------------------

_SCIENTIFIC_DIGEST_FIELDS: tuple[str, ...] = (
    "canonical_candidate_index", "canonical_image_id", "canonical_window_ordinal", "selection_rank",
    "rho_M", "rho_0", "rho_320", "rho_g",
)


def scientific_output_record(*, canonical_candidate_index: int, canonical_image_id: str, canonical_window_ordinal: int,
                              selection_rank: int, family_summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "canonical_candidate_index": canonical_candidate_index, "canonical_image_id": canonical_image_id,
        "canonical_window_ordinal": canonical_window_ordinal, "selection_rank": selection_rank,
        "rho_M": dict(family_summaries["rho_M"]), "rho_0": dict(family_summaries["rho_0"]),
        "rho_320": dict(family_summaries["rho_320"]), "rho_g": dict(family_summaries["rho_g"]),
    }


def compute_scientific_digest(record: Mapping[str, Any]) -> str:
    if frozenset(record.keys()) != frozenset(_SCIENTIFIC_DIGEST_FIELDS):
        raise DfpPhase1bChunkBenchmarkError("scientific-output record has the wrong field set")
    return hashlib.sha256(canonical_encode(record)).hexdigest()


def compute_collection_digest(records_in_slot_order: Sequence[Mapping[str, Any]]) -> str:
    if len(records_in_slot_order) != BENCHMARK_WINDOW_COUNT:
        raise DfpPhase1bChunkBenchmarkError(f"scientific-output collection must contain exactly {BENCHMARK_WINDOW_COUNT} records")
    return digest_over_records([canonical_encode(r) for r in records_in_slot_order])


def max_absolute_difference(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Descriptive evidence only -- never used to declare equivalence.
    Returns ``None`` (not 0.0) when either sequence is empty, so an
    absent comparison is never confused with a proven-zero one."""
    if len(a) != len(b) or len(a) == 0:
        return None
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def strict_structural_equal(actual: Any, expected: Any) -> bool:
    """Exact, deterministic, type-aware structural equality -- used to
    relationally verify a persisted document against a freshly
    recomputed one (Finding A repair: merged-result descriptive-summary
    verification). Plain Python ``==``/``!=`` is NOT sufficient here:
    ``600 == 600.0`` and ``True == 1`` are both ``True`` in Python,
    which would let an integral-looking float or a bool silently pass
    as its corresponding int. This function requires ``type(actual) is
    type(expected)`` at every level (dict, list, and scalar), recurses
    into dicts (requiring an identical key set) and lists (requiring
    identical length AND identical element order -- reordering a
    semantically-ordered list is a mismatch), and falls back to plain
    ``==`` only once both operands are already confirmed to be the
    exact same type. No numeric tolerance anywhere. Never raises
    ``KeyError``/``TypeError``/``ValueError`` -- any shape mismatch is
    reported as ``False``, never an exception."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        if frozenset(actual.keys()) != frozenset(expected.keys()):
            return False
        return all(strict_structural_equal(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list):
        if len(actual) != len(expected):
            return False
        return all(strict_structural_equal(a, e) for a, e in zip(actual, expected))
    return actual == expected


# ---------------------------------------------------------------------
# Section: per-trial and merged result schemas (Section 10).
# ---------------------------------------------------------------------

TRIAL_TOP_KEYS: frozenset[str] = frozenset(
    [
        "schema_name", "schema_version", "authority_sha256", "manifest_sha256",
        "trial_index", "repeat_index", "chunk_size", "complete", "final",
        "device_record", "software_record", "window_identity", "timing", "workload_records",
        "memory", "scientific_output_digests", "abort_status", "code_provenance",
    ]
)

CODE_PROVENANCE_KEYS: frozenset[str] = frozenset(
    [
        "identity_loader_sha256", "runner_sha256", "cli_sha256", "launcher_sha256",
        "repository_commit", "required_ancestor_commit", "repository_worktree_clean",
    ]
)

# The closed set of trial-document memory fields (Finding 4 repair):
# renamed, scope-explicit host fields plus the GPU fields, and nothing
# else -- in particular NEVER the old ambiguous ``host_peak_rss_bytes``
# name. Enforced both at construction (``build_trial_document``) and
# at verification (``verify_trial_document``) so a hand-edited/
# tampered trial JSON reintroducing the old key, or any other unknown
# key, is rejected rather than silently ignored.
MEMORY_FIELD_KEYS: frozenset[str] = frozenset(
    [
        "gpu_baseline_allocated_bytes", "gpu_baseline_reserved_bytes",
        "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes",
        "gpu_end_allocated_bytes", "gpu_end_reserved_bytes",
        "host_process_lifetime_peak_rss_bytes", "host_process_lifetime_vmhwm_bytes", "host_current_rss_bytes",
    ]
)


def _verify_memory_mapping(memory: Any) -> None:
    """The SOLE persisted-memory validator (repair: the prior
    ``_verify_memory_field_keys`` checked only the key set, never the
    values -- a negative or non-finite byte count in any of the nine
    ``MEMORY_FIELD_KEYS`` could silently survive verification and flow
    into a merged descriptive summary). Requires ``memory`` to be a
    mapping with EXACTLY the nine frozen keys, and every one of those
    nine values to be an exact, nonnegative ``int`` byte count
    (``type(value) is int and value >= 0``) -- the identical standard
    the live safety-guard check already enforces on the specific
    readings it consumes (see ``_validate_memory_reading``), now
    extended to the full persisted trial-document schema. Rejects
    ``bool`` (a ``bool`` is not ``int`` under ``type() is``), ``float``
    (including integral-looking floats, ``NaN``, and ``+-Infinity``),
    negative integers, numeric strings, ``None``, complex numbers, and
    any other value whose exact type is not ``int`` -- regardless of
    whether it merely LOOKS numeric. Never raises a raw
    ``TypeError``/``KeyError``/``AttributeError``: always the declared
    domain error, naming the exact offending field
    (``memory.<field_name>``)."""
    if not isinstance(memory, Mapping):
        raise DfpPhase1bChunkBenchmarkError(f"memory must be a mapping, got {type(memory).__name__}")
    actual_keys = frozenset(memory.keys())
    if actual_keys != MEMORY_FIELD_KEYS:
        raise DfpPhase1bChunkBenchmarkError(
            f"trial document memory has the wrong key set -- missing={sorted(MEMORY_FIELD_KEYS - actual_keys)} "
            f"unknown={sorted(actual_keys - MEMORY_FIELD_KEYS)}"
        )
    for field in sorted(MEMORY_FIELD_KEYS):
        value = memory[field]
        if type(value) is not int:
            raise DfpPhase1bChunkBenchmarkError(
                f"memory.{field} must be an exact nonnegative int byte count, got {value!r} (type {type(value).__name__})"
            )
        if value < 0:
            raise DfpPhase1bChunkBenchmarkError(f"memory.{field} must be nonnegative, got {value!r}")


def _verify_timing_fields(timing: Mapping[str, Any]) -> None:
    """Re-validate ``timing.primary_elapsed_ns``/``secondary_elapsed_ns``
    at VERIFICATION time (not just at construction time), so a loaded/
    tampered trial JSON with a non-finite, negative, boolean, or
    wrongly-typed value (e.g. a stray JSON ``NaN``/``Infinity`` token,
    which Python's ``json`` module parses by default) is rejected
    rather than silently propagating into ``merge_trials``'s
    aggregation."""
    primary = timing.get("primary_elapsed_ns")
    if type(primary) is not int or primary < 0:
        raise DfpPhase1bChunkBenchmarkError(f"trial document timing.primary_elapsed_ns must be a non-negative exact integer, got {primary!r}")
    secondary = timing.get("secondary_elapsed_ns")
    if secondary is not None and (type(secondary) is not int or secondary < 0):
        raise DfpPhase1bChunkBenchmarkError(f"trial document timing.secondary_elapsed_ns must be null or a non-negative exact integer, got {secondary!r}")


def build_code_provenance(
    *, identity_loader_sha256: str, runner_sha256: str, cli_sha256: str, launcher_sha256: str,
    repository_commit: str, required_ancestor_commit: str, repository_worktree_clean: bool,
) -> dict[str, Any]:
    """Build the closed ``code_provenance`` mapping recorded in every
    trial document (Finding 3 repair). ``repository_worktree_clean``
    must already have been independently confirmed by the caller
    (``require_clean_worktree``) -- this function only assembles and
    strictly type-checks the record, it does not re-check Git state
    itself."""
    for name, value in (("repository_commit", repository_commit), ("required_ancestor_commit", required_ancestor_commit)):
        if type(value) is not str or len(value) != 40:
            raise DfpPhase1bChunkBenchmarkError(f"code_provenance.{name} must be a 40-hex-character commit id, got {value!r}")
    for name, value in (
        ("identity_loader_sha256", identity_loader_sha256), ("runner_sha256", runner_sha256),
        ("cli_sha256", cli_sha256), ("launcher_sha256", launcher_sha256),
    ):
        if type(value) is not str or len(value) != 64:
            raise DfpPhase1bChunkBenchmarkError(f"code_provenance.{name} must be a 64-hex-character SHA-256, got {value!r}")
    if type(repository_worktree_clean) is not bool:
        raise DfpPhase1bChunkBenchmarkError("code_provenance.repository_worktree_clean must be an exact bool")
    return {
        "identity_loader_sha256": identity_loader_sha256, "runner_sha256": runner_sha256,
        "cli_sha256": cli_sha256, "launcher_sha256": launcher_sha256,
        "repository_commit": repository_commit, "required_ancestor_commit": required_ancestor_commit,
        "repository_worktree_clean": repository_worktree_clean,
    }


def verify_code_provenance(record: Mapping[str, Any]) -> None:
    actual_keys = frozenset(record.keys())
    if actual_keys != CODE_PROVENANCE_KEYS:
        raise DfpPhase1bChunkBenchmarkError(
            f"code_provenance has the wrong key set -- missing={sorted(CODE_PROVENANCE_KEYS - actual_keys)} "
            f"unknown={sorted(actual_keys - CODE_PROVENANCE_KEYS)}"
        )
    if record["repository_worktree_clean"] is not True:
        raise DfpPhase1bChunkBenchmarkError("code_provenance.repository_worktree_clean must be true for a complete/final trial")


@dataclass(frozen=True)
class TrialRecord:
    """A single benchmark trial's complete, closed-schema result. Never
    contains a winner/recommendation field."""

    document: dict[str, Any]


def build_trial_document(
    *, authority_sha256: str, manifest_sha256: str, trial_index: int, repeat_index: int, chunk_size: int,
    device_record: Mapping[str, Any], software_record: Mapping[str, Any], window_identity: Sequence[Mapping[str, Any]],
    primary_elapsed_ns: int, secondary_elapsed_ns: int | None, per_record_timing: Sequence[Mapping[str, Any]],
    workload_records: Sequence[Mapping[str, Any]], memory: Mapping[str, Any],
    scientific_output_digests: Mapping[str, Any], complete: bool, final: bool,
    code_provenance: Mapping[str, Any],
    abort_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (trial_index, repeat_index, chunk_size) not in TRIAL_SCHEDULE:
        raise DfpPhase1bChunkBenchmarkError(
            f"(trial_index={trial_index}, repeat_index={repeat_index}, chunk_size={chunk_size}) is not a member of the frozen trial schedule"
        )
    if type(primary_elapsed_ns) is not int or primary_elapsed_ns < 0:
        raise DfpPhase1bChunkBenchmarkError("primary_elapsed_ns must be a non-negative exact integer")
    verify_code_provenance(code_provenance)
    _verify_memory_mapping(memory)
    document = {
        "schema_name": TRIAL_SCHEMA_NAME, "schema_version": TRIAL_SCHEMA_VERSION,
        "authority_sha256": authority_sha256, "manifest_sha256": manifest_sha256,
        "trial_index": trial_index, "repeat_index": repeat_index, "chunk_size": chunk_size,
        "complete": bool(complete), "final": bool(final),
        "device_record": dict(device_record), "software_record": dict(software_record),
        "window_identity": [dict(w) for w in window_identity],
        "timing": {
            "primary_elapsed_ns": primary_elapsed_ns,
            "secondary_elapsed_ns": secondary_elapsed_ns,
            "per_record_timing": [dict(r) for r in per_record_timing],
        },
        "workload_records": [dict(w) for w in workload_records],
        "memory": dict(memory),
        "scientific_output_digests": dict(scientific_output_digests),
        "abort_status": dict(abort_status) if abort_status is not None else {"aborted": False, "reason": None},
        "code_provenance": dict(code_provenance),
    }
    if frozenset(document.keys()) != TRIAL_TOP_KEYS:
        raise DfpPhase1bChunkBenchmarkError("internal error: trial document key set drifted from TRIAL_TOP_KEYS")
    return document


def verify_trial_document(document: Mapping[str, Any], *, expected_authority_sha256: str | None = None,
                           expected_manifest_sha256: str | None = None,
                           verify_repo_root: Path | None = None) -> Mapping[str, Any]:
    actual_keys = frozenset(document.keys())
    if actual_keys != TRIAL_TOP_KEYS:
        raise DfpPhase1bChunkBenchmarkError(
            f"trial document has the wrong key set -- missing={sorted(TRIAL_TOP_KEYS - actual_keys)} "
            f"unknown={sorted(actual_keys - TRIAL_TOP_KEYS)}"
        )
    if document["schema_name"] != TRIAL_SCHEMA_NAME or document["schema_version"] != TRIAL_SCHEMA_VERSION:
        raise DfpPhase1bChunkBenchmarkError("trial document schema_name/schema_version mismatch")
    key = (document["trial_index"], document["repeat_index"], document["chunk_size"])
    if key not in TRIAL_SCHEDULE:
        raise DfpPhase1bChunkBenchmarkError(f"trial document (trial_index, repeat_index, chunk_size) {key} is not in the frozen schedule")
    if expected_authority_sha256 is not None and document["authority_sha256"] != expected_authority_sha256:
        raise DfpPhase1bChunkBenchmarkError("trial document authority_sha256 disagrees with the expected E17 identity")
    if expected_manifest_sha256 is not None and document["manifest_sha256"] != expected_manifest_sha256:
        raise DfpPhase1bChunkBenchmarkError("trial document manifest_sha256 disagrees with the expected benchmark manifest")
    if document["abort_status"].get("aborted"):
        if document["complete"] or document["final"]:
            raise DfpPhase1bChunkBenchmarkError("an aborted trial must never be marked complete/final")
    verify_code_provenance(document["code_provenance"])
    _verify_memory_mapping(document["memory"])
    _verify_timing_fields(document["timing"])
    if verify_repo_root is not None:
        # Later artifact-verification time (Section 5): recheck the
        # ancestry relation using Git's own commit graph; never require
        # the verifier's own current HEAD to equal the historical trial
        # commit.
        try:
            verify_ancestor_relation(
                verify_repo_root, required_ancestor_commit=document["code_provenance"]["required_ancestor_commit"],
                repository_commit=document["code_provenance"]["repository_commit"],
            )
        except DfpPhase1bChunkBenchmarkIdentityError as exc:
            raise DfpPhase1bChunkBenchmarkError(f"trial document code_provenance ancestry re-verification failed: {exc}") from exc
    return document


def merge_trials(trials: Sequence[Mapping[str, Any]], *, verify_repo_root: Path | None = None) -> dict[str, Any]:
    """CPU-safe merger/verifier (Section 10). Requires all six frozen
    ``(repeat, chunk_size)`` combinations exactly once, rejects
    duplicates, verifies identical hardware/software/manifest/code-
    provenance bindings and scientific-output digest equality across
    all six trials, and computes only descriptive summaries. Selects
    no winner, binds no tolerance. When ``verify_repo_root`` is
    supplied, independently re-verifies the shared code_provenance's
    ancestor relation via Git's own commit graph (never requiring the
    caller's current HEAD to equal the historical trial commit)."""
    if len(trials) != len(TRIAL_SCHEDULE):
        raise DfpPhase1bChunkBenchmarkError(f"merge requires exactly {len(TRIAL_SCHEDULE)} trials, got {len(trials)}")

    seen_keys = set()
    by_key: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    for trial in trials:
        verify_trial_document(trial, verify_repo_root=verify_repo_root)
        key = (trial["trial_index"], trial["repeat_index"], trial["chunk_size"])
        if key in seen_keys:
            raise DfpPhase1bChunkBenchmarkError(f"duplicate trial for {key}")
        seen_keys.add(key)
        by_key[key] = trial
    expected_keys = set(TRIAL_SCHEDULE)
    if seen_keys != expected_keys:
        raise DfpPhase1bChunkBenchmarkError(
            f"trial set does not match the frozen schedule -- missing={sorted(expected_keys - seen_keys)} "
            f"unexpected={sorted(seen_keys - expected_keys)}"
        )
    for trial in trials:
        if not (trial["complete"] and trial["final"]) or trial["abort_status"].get("aborted"):
            raise DfpPhase1bChunkBenchmarkError(f"trial {(trial['trial_index'], trial['repeat_index'], trial['chunk_size'])} is not complete/final -- refusing to merge")

    first = trials[0]
    for trial in trials[1:]:
        if trial["authority_sha256"] != first["authority_sha256"]:
            raise DfpPhase1bChunkBenchmarkError("trials disagree on authority_sha256")
        if trial["manifest_sha256"] != first["manifest_sha256"]:
            raise DfpPhase1bChunkBenchmarkError("trials disagree on manifest_sha256")
        if trial["device_record"]["gpu_model_name"] != first["device_record"]["gpu_model_name"]:
            raise DfpPhase1bChunkBenchmarkError("trials disagree on device_record.gpu_model_name")
        if trial["software_record"] != first["software_record"]:
            raise DfpPhase1bChunkBenchmarkError("trials disagree on software_record")
        if trial["window_identity"] != first["window_identity"]:
            raise DfpPhase1bChunkBenchmarkError("trials disagree on window_identity")
        if trial["code_provenance"] != first["code_provenance"]:
            raise DfpPhase1bChunkBenchmarkError("trials disagree on code_provenance -- not all six trials executed the same benchmark code/repository state")

    # Scientific-output digest equality: zero tolerance, no rounding.
    digest_by_key = {key: by_key[key]["scientific_output_digests"]["collection_digest"] for key in by_key}
    reference_digest = digest_by_key[(0, 0, 16)]
    for chunk_size in CHUNK_SIZES:
        repeat0_key = next(k for k in by_key if k[1] == 0 and k[2] == chunk_size)
        repeat1_key = next(k for k in by_key if k[1] == 1 and k[2] == chunk_size)
        if digest_by_key[repeat0_key] != digest_by_key[repeat1_key]:
            raise DfpPhase1bChunkBenchmarkError(f"SCIENTIFIC_OUTPUT_MISMATCH: repeat 0 != repeat 1 for chunk_size={chunk_size}")
        if digest_by_key[repeat0_key] != reference_digest:
            raise DfpPhase1bChunkBenchmarkError(f"SCIENTIFIC_OUTPUT_MISMATCH: chunk_size={chunk_size} digest disagrees with chunk_size=16")

    summaries: dict[str, Any] = {}
    for chunk_size in CHUNK_SIZES:
        keys = [k for k in by_key if k[2] == chunk_size]
        times = sorted(by_key[k]["timing"]["primary_elapsed_ns"] for k in keys)
        gpu_peak_alloc = [by_key[k]["memory"]["gpu_peak_allocated_bytes"] for k in keys if by_key[k]["memory"].get("gpu_peak_allocated_bytes") is not None]
        gpu_peak_reserved = [by_key[k]["memory"]["gpu_peak_reserved_bytes"] for k in keys if by_key[k]["memory"].get("gpu_peak_reserved_bytes") is not None]
        host_peak = [by_key[k]["memory"]["host_process_lifetime_peak_rss_bytes"] for k in keys if by_key[k]["memory"].get("host_process_lifetime_peak_rss_bytes") is not None]
        summaries[str(chunk_size)] = {
            "primary_elapsed_ns_per_repeat": times,
            "median_of_two_primary_times_ns": sum(times) / len(times) if times else None,
            "min_ns": min(times) if times else None, "max_ns": max(times) if times else None,
            "gpu_peak_allocated_bytes": gpu_peak_alloc, "gpu_peak_reserved_bytes": gpu_peak_reserved,
            "host_process_lifetime_peak_rss_bytes": host_peak,
            "throughput_windows_per_second_over_fixed_24_window_set": [
                BENCHMARK_WINDOW_COUNT / (t / 1e9) for t in times if t > 0
            ],
        }

    return {
        "schema_name": MERGED_SCHEMA_NAME, "schema_version": MERGED_SCHEMA_VERSION,
        "authority_sha256": first["authority_sha256"], "manifest_sha256": first["manifest_sha256"],
        "code_provenance": dict(first["code_provenance"]),
        "trial_count": len(trials),
        "scientific_output_digest": reference_digest,
        "scientific_output_digest_equal_across_all_trials": True,
        "descriptive_summaries_by_chunk_size": summaries,
        "raw_trials": [dict(t) for t in trials],
    }


__all__ = [
    "DfpPhase1bChunkBenchmarkError",
    "MANIFEST_SCHEMA_NAME", "MANIFEST_SCHEMA_VERSION", "MANIFEST_TOP_KEYS",
    "TRIAL_SCHEMA_NAME", "TRIAL_SCHEMA_VERSION", "TRIAL_TOP_KEYS",
    "MERGED_SCHEMA_NAME", "MERGED_SCHEMA_VERSION",
    "REQUIRED_ARTIFACT_RELATIVE_PATH",
    "canonical_encode", "digest_over_records", "slot_position",
    "select_benchmark_windows", "build_benchmark_manifest", "verify_benchmark_manifest",
    "instrument_masked_forward_module", "summarize_chunk_calls",
    "scientific_output_record", "compute_scientific_digest", "compute_collection_digest", "max_absolute_difference",
    "strict_structural_equal",
    "build_trial_document", "verify_trial_document", "merge_trials",
    "CODE_PROVENANCE_KEYS", "build_code_provenance", "verify_code_provenance",
    "MEMORY_FIELD_KEYS",
]


# ---------------------------------------------------------------------
# Section: single-trial orchestration (Sections 5-8). Every heavy
# dependency (dataset/model resources, image preparer, inference,
# device/software record collectors, the masked-forward module to
# instrument, the scientific execution function itself, the clock, and
# the memory providers) is dependency-injected -- this function never
# imports torch/mmcv/CUDA at module scope, and CPU-only tests supply
# small fakes for every one of these seams (see
# tests/test_dfp_phase1b_chunk_benchmark.py). A real GPU trial (driven
# by the CLI's ``run-trial`` command) supplies the real production
# builders from ``src.dfp_phase1b_calibration_production`` and the
# real ``src.dfp_phase1b_calibration_run.process_selected_image``,
# completely unmodified.
# ---------------------------------------------------------------------


def _group_by_image_preserving_first_slot(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, list[Mapping[str, Any]]]]:
    order: list[str] = []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        image_id = record["image_id"]
        if image_id not in grouped:
            grouped[image_id] = []
            order.append(image_id)
        grouped[image_id].append(record)
    return [(image_id, grouped[image_id]) for image_id in order]


class MemoryProvider(Protocol):
    def baseline(self) -> Mapping[str, Any]: ...
    def reset_peak(self) -> None: ...
    def peak(self) -> Mapping[str, Any]: ...
    def host_current(self) -> Mapping[str, Any]: ...


def run_trial(
    *, chunk_size: int, trial_index: int, repeat_index: int,
    manifest: Mapping[str, Any], authority_sha256: str, manifest_sha256: str,
    build_resources: Callable[[], Any], build_image_preparer: Callable[[Any], Callable[[str], Any]],
    build_inference: Callable[[Any], Any], get_device_record: Callable[[], Mapping[str, Any]],
    get_software_record: Callable[[], Mapping[str, Any]], masked_forward_module: Any,
    process_selected_image_fn: Callable[..., Mapping[int, Mapping[str, Mapping[str, Any]]]],
    memory_provider: MemoryProvider, memory_guard_contract: MemoryGuardContract, code_provenance: Mapping[str, Any],
    cuda_synchronize: Callable[[], None] = lambda: None,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Run exactly one fresh-process benchmark trial (Sections 5-8) and
    return its closed-schema trial document. Never writes to disk
    itself -- callers install the returned document atomically (see
    the CLI's ``run-trial`` command). ``memory_guard_contract`` MUST be
    built via ``memory_guard_contract_from_identity`` from an already-
    verified E17 document -- this function supplies no Python default
    of its own (Finding 2 repair). ``code_provenance`` MUST be built
    via ``build_code_provenance`` from state collected BEFORE this
    call (Finding 3 repair)."""
    if not isinstance(memory_guard_contract, MemoryGuardContract):
        raise DfpPhase1bChunkBenchmarkError("memory_guard_contract must be a MemoryGuardContract built from the verified E17 identity -- no default is supplied")
    verify_code_provenance(code_provenance)
    if chunk_size not in CHUNK_SIZES:
        raise DfpPhase1bChunkBenchmarkError(f"chunk_size must be one of {CHUNK_SIZES}, got {chunk_size}")
    if (trial_index, repeat_index, chunk_size) not in TRIAL_SCHEDULE:
        raise DfpPhase1bChunkBenchmarkError(
            f"(trial_index={trial_index}, repeat_index={repeat_index}, chunk_size={chunk_size}) is not in the frozen trial schedule"
        )

    benchmark_records = manifest["benchmark_records"]
    warmup_record = manifest["warmup_record"]

    resources = build_resources()
    image_preparer = build_image_preparer(resources)
    inference = build_inference(resources)

    def process_one_image(image_id: str, ordinals: Sequence[int]) -> Mapping[int, Mapping[str, Mapping[str, Any]]]:
        prepared = image_preparer(image_id)
        result = process_selected_image_fn(
            inference, prepared, image_id, selected_windows_for_image=sorted(ordinals),
            protocol_identity=resources.protocol_identity, contracts=resources.contracts,
            class_count=resources.class_count, affinity_power=resources.affinity_power,
            component_chunk_size=chunk_size,
        )
        del prepared
        return result

    # --- warm-up: real production path, SAME chunk size, never timed,
    # never contributes to scientific-output digests. ---
    process_one_image(warmup_record["image_id"], [warmup_record["canonical_window_ordinal"]])
    cuda_synchronize()
    memory_provider.reset_peak()
    baseline_memory = memory_provider.baseline()

    grouped = _group_by_image_preserving_first_slot(benchmark_records)
    restore_instrumentation, instrumented_calls = instrument_masked_forward_module(masked_forward_module)

    per_image_group_timing: list[dict[str, Any]] = []
    family_summaries_by_index: dict[int, Mapping[str, Mapping[str, Any]]] = {}
    index_by_ordinal_for_image: dict[str, dict[int, int]] = {}
    for record in benchmark_records:
        index_by_ordinal_for_image.setdefault(record["image_id"], {})[record["canonical_window_ordinal"]] = record["canonical_candidate_index"]

    abort_reason: str | None = None
    start_ns = clock()
    try:
        for image_id, records in grouped:
            ordinals = [r["canonical_window_ordinal"] for r in records]
            group_start = clock()
            per_window = process_one_image(image_id, ordinals)
            group_stop = clock()
            per_image_group_timing.append(
                {"canonical_image_id": image_id, "window_count": len(ordinals), "elapsed_ns": group_stop - group_start}
            )
            for ordinal, summaries in per_window.items():
                index = index_by_ordinal_for_image[image_id][ordinal]
                family_summaries_by_index[index] = summaries
            # Operational abort guard (Section 8) -- checked between
            # measured image groups, never a scientific tolerance.
            abort_reason = check_memory_safety_guard(memory_provider, memory_guard_contract)
            if abort_reason is not None:
                break
        cuda_synchronize()
        stop_ns = clock()
    finally:
        restore_instrumentation()

    primary_elapsed_ns = stop_ns - start_ns
    peak_memory = memory_provider.peak()
    aborted = abort_reason is not None

    # --- assemble per-window scientific-output + workload records, in
    # frozen manifest slot order. An aborted trial only includes the
    # windows actually completed before the guard tripped, and is
    # never marked complete/final. ---
    scientific_records = []
    workload_records = []
    window_identity = []
    for record in benchmark_records:
        index = record["canonical_candidate_index"]
        if aborted and index not in family_summaries_by_index:
            continue
        summaries = family_summaries_by_index[index]
        scientific_records.append(
            scientific_output_record(
                canonical_candidate_index=index, canonical_image_id=record["image_id"],
                canonical_window_ordinal=record["canonical_window_ordinal"], selection_rank=record["selection_rank"],
                family_summaries=summaries,
            )
        )
        chunk_summary = summarize_chunk_calls(instrumented_calls, window_index=record["canonical_window_ordinal"])
        workload_records.append({
            "canonical_image_id": record["image_id"], "canonical_candidate_index": index,
            "selection_rank": record["selection_rank"], "benchmark_slot": record["benchmark_slot"],
            "query_class_count": record["query_class_count"], **chunk_summary,
        })
        window_identity.append({
            "canonical_image_id": record["image_id"], "canonical_candidate_index": index,
            "selection_rank": record["selection_rank"], "benchmark_slot": record["benchmark_slot"],
        })

    completed_slots = {r["benchmark_slot"] for r in benchmark_records if r["canonical_candidate_index"] in family_summaries_by_index}
    if aborted:
        collection_digest = None
        per_window_digests = {
            str(record["benchmark_slot"]): compute_scientific_digest(sci)
            for record, sci in zip(
                [r for r in benchmark_records if r["benchmark_slot"] in completed_slots], scientific_records,
            )
        }
    else:
        collection_digest = compute_collection_digest(scientific_records)
        per_window_digests = {
            str(record["benchmark_slot"]): compute_scientific_digest(sci)
            for record, sci in zip(benchmark_records, scientific_records)
        }

    document = build_trial_document(
        authority_sha256=authority_sha256, manifest_sha256=manifest_sha256,
        trial_index=trial_index, repeat_index=repeat_index, chunk_size=chunk_size,
        device_record=get_device_record(), software_record=get_software_record(), window_identity=window_identity,
        primary_elapsed_ns=primary_elapsed_ns, secondary_elapsed_ns=None, per_record_timing=per_image_group_timing,
        workload_records=workload_records,
        memory={
            "gpu_baseline_allocated_bytes": baseline_memory.get("allocated_bytes"),
            "gpu_baseline_reserved_bytes": baseline_memory.get("reserved_bytes"),
            "gpu_peak_allocated_bytes": peak_memory.get("allocated_bytes"),
            "gpu_peak_reserved_bytes": peak_memory.get("reserved_bytes"),
            "gpu_end_allocated_bytes": peak_memory.get("end_allocated_bytes"),
            "gpu_end_reserved_bytes": peak_memory.get("end_reserved_bytes"),
            # Host figures unavoidably cover the ENTIRE trial process
            # (imports, model construction, checkpoint loading, warm-up)
            # -- ru_maxrss/VmHWM cannot be reset mid-process on Linux,
            # unlike the GPU fields above which ARE measured-24-window-
            # subset-scoped (reset after warm-up). Field names make this
            # explicit; see measurement.host_memory.scope_note in E17.
            "host_process_lifetime_peak_rss_bytes": memory_provider.host_current().get("ru_maxrss_bytes"),
            "host_process_lifetime_vmhwm_bytes": memory_provider.host_current().get("vmhwm_bytes"),
            "host_current_rss_bytes": memory_provider.host_current().get("vmrss_bytes"),
        },
        scientific_output_digests={"collection_digest": collection_digest, "per_window_digests": per_window_digests},
        complete=(not aborted), final=(not aborted),
        abort_status={"aborted": aborted, "reason": abort_reason},
        code_provenance=code_provenance,
    )
    return document


def write_trial_document_atomically(document: Mapping[str, Any], output_path: Path, *, allow_overwrite: bool = False) -> None:
    output_path = Path(output_path)
    if output_path.exists() and not allow_overwrite:
        raise DfpPhase1bChunkBenchmarkError(f"refusing to overwrite existing trial artifact {output_path} (allow_overwrite=False)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    text = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    import os

    os.replace(tmp_path, output_path)


def write_merged_artifact_atomically(document: Mapping[str, Any], output_path: Path, *, allow_overwrite: bool = False) -> None:
    write_trial_document_atomically(document, output_path, allow_overwrite=allow_overwrite)


__all__.extend([
    "run_trial", "write_trial_document_atomically", "write_merged_artifact_atomically", "MemoryProvider",
])


class GpuHostMemoryProvider:
    """Real GPU + host memory provider (Section 8). Deferred torch/
    resource imports -- only constructed inside ``run-trial``, after
    preflight has already succeeded. ``reset_peak()`` calls
    ``torch.cuda.reset_peak_memory_stats()`` exactly as required
    immediately after warm-up; ``peak()`` synchronizes CUDA before
    reading final values, and IS scoped to the measured 24-window
    subset. Host RSS uses ``resource.getrusage(...).ru_maxrss``
    (Linux KiB, converted to bytes explicitly) plus ``/proc/self/
    status`` ``VmRSS``/``VmHWM`` where available -- ``ru_maxrss`` and
    ``VmHWM`` are UNAVOIDABLY process-lifetime (they cannot be reset
    mid-process on Linux, so they include imports/model-construction/
    checkpoint-loading/warm-up, NOT just the measured subset); ``VmRSS``
    is a CURRENT sample, never labeled a peak. ``run_trial`` maps these
    into the explicitly-named ``host_process_lifetime_peak_rss_bytes``/
    ``host_process_lifetime_vmhwm_bytes``/``host_current_rss_bytes``
    trial-document fields -- see their own naming for the scope
    distinction from the GPU fields above."""

    def __init__(self) -> None:
        import torch

        self._torch = torch

    def baseline(self) -> dict[str, Any]:
        torch = self._torch
        return {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }

    def reset_peak(self) -> None:
        self._torch.cuda.reset_peak_memory_stats()

    def peak(self) -> dict[str, Any]:
        torch = self._torch
        torch.cuda.synchronize()
        return {
            "allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.max_memory_reserved(),
            "end_allocated_bytes": torch.cuda.memory_allocated(),
            "end_reserved_bytes": torch.cuda.memory_reserved(),
        }

    def host_current(self) -> dict[str, Any]:
        import resource

        ru_maxrss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KiB
        if type(ru_maxrss_kb) is not int or ru_maxrss_kb < 0:
            raise DfpPhase1bChunkBenchmarkError(
                f"resource.getrusage(RUSAGE_SELF).ru_maxrss returned an invalid value: {ru_maxrss_kb!r}"
            )
        result: dict[str, Any] = {"ru_maxrss_kb": ru_maxrss_kb, "ru_maxrss_bytes": ru_maxrss_kb * 1024}
        try:
            raw_bytes = Path("/proc/self/status").read_bytes()
        except OSError:
            # Genuinely unavailable (no /proc, permission denied,
            # container/namespace without it, ...) -- fall back to
            # ru_maxrss-only. This is the ONLY case that degrades
            # silently (Section 8 repair).
            return result
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DfpPhase1bChunkBenchmarkError(f"/proc/self/status is not valid UTF-8: {exc}") from exc
        parsed = parse_proc_self_status_kib_fields(text, required_fields=("VmRSS", "VmHWM"))
        result["vmrss_bytes"] = parsed["VmRSS"]
        result["vmhwm_bytes"] = parsed["VmHWM"]
        return result


def _validate_memory_reading(value: Any, *, label: str) -> int:
    """Strict byte-count validation (Section 7 repair): requires
    ``type(value) is int and value >= 0`` exactly. Rejects ``bool``
    (a ``bool`` is not ``int`` under ``type() is``, even though
    ``isinstance(True, int)`` is true in Python), ``float`` (including
    ``NaN``/``+-Infinity`` -- their type is ``float``, never ``int``),
    negative integers, numeric strings, ``None``, and any other
    numeric-like object whose exact type is not ``int``. This is what
    prevents a corrupted/non-finite reading from silently passing a
    Python ``>`` comparison (``float('nan') > x`` is always ``False``)."""
    if type(value) is not int:
        raise DfpPhase1bChunkBenchmarkError(
            f"{label} must be an exact nonnegative int byte count, got {value!r} (type {type(value).__name__})"
        )
    if value < 0:
        raise DfpPhase1bChunkBenchmarkError(f"{label} must be nonnegative, got {value!r}")
    return value


def _select_memory_reading(readings: Mapping[str, Any], *, keys_in_priority_order: Sequence[str], label: str) -> int:
    """Pick the first AVAILABLE reading among ``keys_in_priority_order``
    (a key simply absent from ``readings`` means "unavailable", falls
    through to the next candidate, and ultimately to ``0`` if none are
    available); a key that IS present but explicitly ``None`` is an
    invalid reading, not an absent one, and is rejected outright.
    Every reading that is actually used is strictly validated via
    ``_validate_memory_reading``."""
    for key in keys_in_priority_order:
        if key in readings:
            value = readings[key]
            if value is None:
                raise DfpPhase1bChunkBenchmarkError(f"{label}.{key} is explicitly null -- invalid reading")
            return _validate_memory_reading(value, label=f"{label}.{key}")
    return 0


_PROC_STATUS_DECIMAL_RE = re.compile(r"^[0-9]+$")


def parse_proc_self_status_kib_fields(text: str, *, required_fields: Sequence[str]) -> dict[str, int]:
    """Strictly parse ``FieldName:  <decimal nonnegative int> kB``
    lines out of an already-read ``/proc/self/status`` text blob
    (Section 8 repair). Fails closed with the declared domain error --
    never a raw ``ValueError``/``IndexError``/``KeyError`` -- on any
    malformed value, wrong/missing unit, duplicated field, or a
    required field missing from an otherwise-readable file. This is
    "file was readable but content was invalid", distinct from "file
    genuinely unavailable" (which callers handle separately via a
    caught ``OSError`` on the read itself, before this function is
    ever called)."""
    seen: dict[str, int] = {}
    for raw_line in text.splitlines():
        for field in required_fields:
            prefix = f"{field}:"
            if not raw_line.startswith(prefix):
                continue
            if field in seen:
                raise DfpPhase1bChunkBenchmarkError(f"/proc/self/status field {field!r} appears more than once")
            remainder = raw_line[len(prefix):].strip()
            tokens = remainder.split()
            if len(tokens) != 2:
                raise DfpPhase1bChunkBenchmarkError(
                    f"/proc/self/status field {field!r} does not have exactly one value and one unit token: {raw_line!r}"
                )
            number_token, unit_token = tokens
            if unit_token != "kB":
                raise DfpPhase1bChunkBenchmarkError(
                    f"/proc/self/status field {field!r} has unit {unit_token!r}, expected exactly 'kB': {raw_line!r}"
                )
            if not _PROC_STATUS_DECIMAL_RE.match(number_token):
                raise DfpPhase1bChunkBenchmarkError(
                    f"/proc/self/status field {field!r} value {number_token!r} is not a plain decimal nonnegative integer: {raw_line!r}"
                )
            seen[field] = int(number_token)
    missing = [f for f in required_fields if f not in seen]
    if missing:
        raise DfpPhase1bChunkBenchmarkError(f"/proc/self/status is missing required field(s) {missing} in an otherwise-readable file")
    out: dict[str, int] = {}
    for field, kib in seen.items():
        bytes_value = kib * 1024
        if bytes_value < 0 or bytes_value // 1024 != kib:
            raise DfpPhase1bChunkBenchmarkError(f"/proc/self/status field {field!r} KiB-to-byte conversion is not exact: {kib}")
        out[field] = bytes_value
    return out


def check_memory_safety_guard(memory_provider: MemoryProvider, contract: MemoryGuardContract) -> str | None:
    """Operational abort guard (Section 8), not a scientific tolerance.
    ``contract`` MUST be built via ``memory_guard_contract_from_identity``
    from an already-verified E17 document -- this function supplies NO
    independent default of its own (Finding 2 repair: previously
    ``host_fraction``/``gpu_fraction``/``allocated_*_gib`` defaulted to
    hardcoded Python literals here, disconnected from E17's own frozen
    ``measurement.safety_guard`` values; that independent-default path
    has been removed entirely). Returns a human-readable abort reason
    if either threshold is exceeded (strictly greater than, matching
    E17's own frozen ``comparison_operator``), else ``None``. Callers
    must check this between measured image groups and abort cleanly
    (never mark the trial complete). Every raw memory reading is
    strictly validated (exact nonnegative int) before being compared
    against a threshold -- a non-finite/negative/wrongly-typed reading
    is rejected via the declared domain error rather than silently
    failing a Python ``>`` comparison (Section 7 repair)."""
    if not isinstance(contract, MemoryGuardContract):
        raise DfpPhase1bChunkBenchmarkError("check_memory_safety_guard requires a MemoryGuardContract -- no default is supplied")
    host = memory_provider.host_current()
    host_bytes = _select_memory_reading(host, keys_in_priority_order=("vmrss_bytes", "ru_maxrss_bytes"), label="host memory reading")
    host_limit = contract.host_limit_bytes()
    if host_bytes > host_limit:
        return f"host RSS {host_bytes} bytes exceeded {contract.host_abort_fraction:.0%} of the allocated {contract.allocated_host_gib} GiB"
    peak = memory_provider.peak()
    gpu_reserved = _select_memory_reading(peak, keys_in_priority_order=("reserved_bytes",), label="GPU reserved memory reading")
    gpu_limit = contract.gpu_limit_bytes()
    if gpu_reserved > gpu_limit:
        return f"GPU reserved memory {gpu_reserved} bytes exceeded {contract.gpu_abort_fraction:.0%} of the allocated {contract.allocated_gpu_gib} GiB"
    return None


__all__.extend([
    "GpuHostMemoryProvider", "check_memory_safety_guard",
    "parse_proc_self_status_kib_fields",
])
