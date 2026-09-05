"""Deterministic candidate encoding, 200-window selection algorithm,
digest functions, and closed-schema validation for
`phase1b_conservation_calibration.json`, exactly per
`evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml`.

This module implements NO runner: it never imports torch, never
initializes CUDA, never accesses a real dataset/model/checkpoint, and
never selects, infers, or stores a numeric conservation tolerance. It
is pure, deterministic, CPU-only bookkeeping over caller-supplied
candidate records and (for verification) caller-supplied artifact
documents.

Always fail closed: never silently substitute a default, coerce a
type, drop an unknown field, or supply a missing one.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class DfpPhase1bCalibrationArtifactError(ValueError):
    """Raised on any DFP Phase 1B conservation-calibration candidate,
    selection, or artifact invariant violation. Always fail closed.
    Never raised for, and never catches, KeyboardInterrupt, SystemExit,
    or MemoryError -- those always propagate unmodified."""


# ---------------------------------------------------------------------
# Frozen constants (verbatim from e16's own pinned values).
# ---------------------------------------------------------------------

WINDOW_SELECTION_NAMESPACE = "talk2dino-dfp-phase1b-calibration-window-selection-v1"
WINDOW_COUNT = 200
REQUIRED_ARTIFACT_RELATIVE_PATH = "phase1b_conservation_calibration.json"
ARTIFACT_SCHEMA_NAME = "talk2dino-dfp-phase1b-conservation-calibration-v1"
ARTIFACT_SCHEMA_VERSION = 1

CANDIDATE_INTEGER_FIELDS: tuple[str, ...] = (
    "image_manifest_ordinal", "inference_canvas_height", "inference_canvas_width",
    "canonical_window_ordinal", "top", "left", "bottom", "right",
    "crop_height", "crop_width", "patch_grid_height", "patch_grid_width",
)
CANDIDATE_STRING_FIELDS: tuple[str, ...] = ("run_mode", "image_id", "image_path")
CANDIDATE_FIELDS: tuple[str, ...] = (
    "run_mode", "image_id", "image_path", "image_manifest_ordinal",
    "inference_canvas_height", "inference_canvas_width", "canonical_window_ordinal",
    "top", "left", "bottom", "right", "crop_height", "crop_width",
    "patch_grid_height", "patch_grid_width",
)
_CANDIDATE_FIELD_SET = frozenset(CANDIDATE_FIELDS)

_TRACE_RECORD_FIELDS = frozenset({"canonical_candidate_index", "selection_key", "selection_rank"})

RESIDUAL_FAMILIES: tuple[str, ...] = ("rho_M", "rho_0", "rho_320", "rho_g")
RESIDUAL_SUMMARY_FIELDS: tuple[str, ...] = (
    "sample_count", "finite_count", "non_finite_count",
    "signed_minimum", "signed_maximum",
    "maximum_absolute_residual", "mean_absolute_residual", "root_mean_square_residual",
    "absolute_residual_sum", "squared_residual_sum",
)
_RESIDUAL_SUMMARY_FIELD_SET = frozenset(RESIDUAL_SUMMARY_FIELDS)
_RESIDUAL_SUMMARY_FLOAT_FIELDS = frozenset(
    {"signed_minimum", "signed_maximum", "maximum_absolute_residual", "mean_absolute_residual",
     "root_mean_square_residual", "absolute_residual_sum", "squared_residual_sum"}
)
_RESIDUAL_SUMMARY_COUNT_FIELDS = frozenset({"sample_count", "finite_count", "non_finite_count"})

_FORBIDDEN_FIELD_NAME_FRAGMENTS: tuple[str, ...] = (
    "tolerance", "epsilon", "threshold", "multiplier", "safety_factor", "percentile_cutoff",
    "pass", "fail", "acceptable", "verdict",
)

SELECTED_WINDOW_RECORD_FIELDS: tuple[str, ...] = CANDIDATE_FIELDS + (
    "canonical_candidate_index", "selection_rank", "selection_key", "completed",
) + RESIDUAL_FAMILIES
_SELECTED_WINDOW_RECORD_FIELD_SET = frozenset(SELECTED_WINDOW_RECORD_FIELDS)

TOP_LEVEL_ARTIFACT_FIELDS: tuple[str, ...] = (
    "schema_name", "schema_version", "complete", "final",
    "calibration_identity_sha256", "bound_e15_protocol_sha256", "bound_e15_artifacts_sha256",
    "bound_masked_forward_sha256", "repository_commit", "run_mode",
    "bound_subset_manifest_sha256", "selected_ids_digest",
    "candidate_window_count", "expected_selected_window_count", "completed_selected_window_count",
    "candidate_universe_digest", "selected_windows_digest", "selection_trace_digest",
    "device_record", "software_record", "propagation_precision_record",
    "selected_window_records", "aggregate_residual_summaries",
)
_TOP_LEVEL_ARTIFACT_FIELD_SET = frozenset(TOP_LEVEL_ARTIFACT_FIELDS)

DEVICE_RECORD_FIELDS = frozenset({
    "device_type", "gpu_model_name", "cuda_compute_capability",
    "total_device_memory_bytes", "cuda_driver_version", "cuda_runtime_version",
})
SOFTWARE_RECORD_FIELDS = frozenset({
    "repository_commit", "python_version", "torch_version", "numpy_version",
    "cuda_version", "cudnn_version",
    "deterministic_algorithms_enabled", "tf32_enabled", "autocast_enabled",
})
PROPAGATION_PRECISION_RECORD_FIELDS = frozenset({
    "graph_storage_dtype", "canonical_recurrence_dtype", "masked_recurrence_dtype",
    "residual_accumulation_dtype", "alpha", "steps", "k", "chunk_size",
})
# Exact frozen values from the E16 TOML's own [execution_contract] block
# (graph_storage_dtype/canonical_recurrence_dtype/masked_recurrence_dtype/
# residual_accumulation_dtype) -- never a second, independently-invented
# spelling convention.
_PROPAGATION_PRECISION_EXACT_DTYPES: dict[str, str] = {
    "graph_storage_dtype": "float32",
    "canonical_recurrence_dtype": "float64",
    "masked_recurrence_dtype": "float64",
    "residual_accumulation_dtype": "float64",
}
# The frozen recurrence alpha. E16's own [execution_contract].alpha_source
# is prose, not a literal numeric field -- this constant is the same kind
# of hardcoded scientific-constant pin already used throughout this
# codebase (e.g. src/dfp_phase1a_identity.py's own `!= 0.98` check),
# transitively bound via e15_protocol's own authority_bindings to
# e13_dfp_phase1a_protocol.toml's [propagation].alpha, never redefined
# independently here.
EXPECTED_ALPHA = 0.98


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DfpPhase1bCalibrationArtifactError(f"{label} must be a mapping")
    if set(value) != expected_keys:
        missing = expected_keys - set(value)
        unknown = set(value) - expected_keys
        raise DfpPhase1bCalibrationArtifactError(
            f"{label} has an unexpected schema (missing={sorted(missing)}, unknown={sorted(unknown)})"
        )
    return value


def _require_exact_string(value: Any, label: str) -> str:
    if type(value) is not str:
        raise DfpPhase1bCalibrationArtifactError(f"{label} must be an exact string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1bCalibrationArtifactError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise DfpPhase1bCalibrationArtifactError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise DfpPhase1bCalibrationArtifactError(f"{label} must be >= {minimum}, observed {value}")
    return value


def _require_finite_float_or_null(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if type(value) is not float:
        raise DfpPhase1bCalibrationArtifactError(f"{label} must be an exact float or JSON null")
    if not math.isfinite(value):
        raise DfpPhase1bCalibrationArtifactError(f"{label} must be finite (no NaN/Infinity), observed {value!r}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    text = _require_exact_string(value, label)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise DfpPhase1bCalibrationArtifactError(f"{label} must be a lowercase 64-character SHA-256")
    return text


def _require_git_commit(value: Any, label: str) -> str:
    text = _require_exact_string(value, label)
    if len(text) != 40 or any(c not in "0123456789abcdef" for c in text):
        raise DfpPhase1bCalibrationArtifactError(f"{label} must be a full lowercase 40-character Git commit id")
    return text


def _require_clean_nonempty_string(value: Any, label: str) -> str:
    """A non-empty string that is byte-preserving-canonical: no leading/
    trailing whitespace, not entirely whitespace, and no embedded NUL.
    Never strips and accepts the normalized value -- a noncanonical
    input is rejected outright, not silently repaired."""
    text = _require_exact_string(value, label)
    if text == "":
        raise DfpPhase1bCalibrationArtifactError(f"{label} must be non-empty")
    if "\x00" in text:
        raise DfpPhase1bCalibrationArtifactError(f"{label} must not contain an embedded NUL byte")
    if text.strip() == "":
        raise DfpPhase1bCalibrationArtifactError(f"{label} must not be whitespace-only")
    if text != text.strip():
        raise DfpPhase1bCalibrationArtifactError(f"{label} must not have leading or trailing whitespace")
    return text


def _require_posix_relative_path(value: Any, label: str) -> str:
    text = _require_clean_nonempty_string(value, label)
    if text.startswith("/") or "\\" in text or ".." in text.split("/") or "" in text.split("/"):
        raise DfpPhase1bCalibrationArtifactError(f"{label} must be a safe, posix-relative path with no traversal")
    return text


# ---------------------------------------------------------------------
# Candidate record: validation, canonical encoding, canonical order.
# ---------------------------------------------------------------------


def validate_candidate_record(record: Any) -> Mapping[str, Any]:
    _require_closed_mapping(record, _CANDIDATE_FIELD_SET, "candidate record")
    for field in CANDIDATE_STRING_FIELDS:
        _require_exact_string(record[field], f"candidate record.{field}")
    for field in CANDIDATE_INTEGER_FIELDS:
        _require_exact_int(record[field], f"candidate record.{field}", minimum=0)
    if record["run_mode"] != "coco_stuff_500":
        raise DfpPhase1bCalibrationArtifactError(
            f"candidate record.run_mode must be 'coco_stuff_500', observed {record['run_mode']!r}"
        )
    _require_clean_nonempty_string(record["image_id"], "candidate record.image_id")
    _require_posix_relative_path(record["image_path"], "candidate record.image_path")
    if record["top"] >= record["bottom"]:
        raise DfpPhase1bCalibrationArtifactError("candidate record.top must be strictly less than .bottom")
    if record["left"] >= record["right"]:
        raise DfpPhase1bCalibrationArtifactError("candidate record.left must be strictly less than .right")
    if record["bottom"] > record["inference_canvas_height"]:
        raise DfpPhase1bCalibrationArtifactError("candidate record.bottom must not exceed .inference_canvas_height")
    if record["right"] > record["inference_canvas_width"]:
        raise DfpPhase1bCalibrationArtifactError("candidate record.right must not exceed .inference_canvas_width")
    if record["crop_height"] <= 0 or record["crop_width"] <= 0:
        raise DfpPhase1bCalibrationArtifactError("candidate record.crop_height/.crop_width must be positive")
    if record["patch_grid_height"] <= 0 or record["patch_grid_width"] <= 0:
        raise DfpPhase1bCalibrationArtifactError("candidate record.patch_grid_height/.patch_grid_width must be positive")
    return record


def encode_candidate_record(record: Mapping[str, Any]) -> bytes:
    """The frozen `talk2dino-dfp-phase1b-calibration-candidate-record-v1`
    byte encoding: `json.dumps(record, sort_keys=True, separators=(',',
    ':'), ensure_ascii=True)`, UTF-8, no trailing newline. Accepts any
    real `collections.abc.Mapping`, including a read-only
    `types.MappingProxyType` -- validates it, then builds and
    serializes a FRESH plain `dict` from the validated field values,
    never the caller's own mapping object directly (which `json.dumps`
    cannot always serialize on its own, e.g. a `mappingproxy`)."""
    validated = validate_candidate_record(record)
    plain = {field: validated[field] for field in CANDIDATE_FIELDS}
    try:
        return json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DfpPhase1bCalibrationArtifactError(f"candidate record could not be serialized: {exc}") from exc


def canonical_candidate_order_key(record: Mapping[str, Any]) -> tuple[int, int]:
    return (record["image_manifest_ordinal"], record["canonical_window_ordinal"])


def sort_candidate_universe(records: Sequence[Any]) -> list[Mapping[str, Any]]:
    """Validate every candidate record, reject duplicates (both by full
    identity and by (image_manifest_ordinal, canonical_window_ordinal)
    order key), and return the universe in canonical candidate order.
    Never depends on the input sequence's own order."""
    validated = [validate_candidate_record(r) for r in records]
    seen_identity: set[tuple] = set()
    for record in validated:
        identity_key = tuple(record[field] for field in CANDIDATE_FIELDS)
        if identity_key in seen_identity:
            raise DfpPhase1bCalibrationArtifactError(f"duplicate candidate record: {dict(record)!r}")
        seen_identity.add(identity_key)
    ordered = sorted(validated, key=canonical_candidate_order_key)
    order_keys = [canonical_candidate_order_key(r) for r in ordered]
    if len(set(order_keys)) != len(order_keys):
        raise DfpPhase1bCalibrationArtifactError(
            "duplicate (image_manifest_ordinal, canonical_window_ordinal) pair in candidate universe"
        )
    return ordered


def digest_over_records(encoded_records: Sequence[bytes]) -> str:
    """sha256(concat(len(record).to_bytes(8, 'big') + record for record
    in encoded_records)).hexdigest() -- length-prefixing removes any
    concatenation-boundary ambiguity."""
    hasher = hashlib.sha256()
    for record_bytes in encoded_records:
        if not isinstance(record_bytes, (bytes, bytearray)):
            raise DfpPhase1bCalibrationArtifactError("digest_over_records requires a sequence of bytes")
        hasher.update(len(record_bytes).to_bytes(8, "big"))
        hasher.update(bytes(record_bytes))
    return hasher.hexdigest()


def compute_selection_key(namespace: str, encoded_record: bytes) -> str:
    if type(namespace) is not str or not namespace:
        raise DfpPhase1bCalibrationArtifactError("selection namespace must be a non-empty exact string")
    return hashlib.sha256(namespace.encode("utf-8") + b"\x00" + encoded_record).hexdigest()


def _encode_trace_record(trace_record: Mapping[str, Any]) -> bytes:
    _require_closed_mapping(trace_record, _TRACE_RECORD_FIELDS, "selection trace record")
    _require_exact_int(trace_record["canonical_candidate_index"], "selection trace record.canonical_candidate_index", minimum=0)
    _require_exact_int(trace_record["selection_rank"], "selection trace record.selection_rank", minimum=0)
    _require_sha256(trace_record["selection_key"], "selection trace record.selection_key")
    return json.dumps(trace_record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


@dataclass(frozen=True)
class SelectionResult:
    """The deterministic outcome of selecting `window_count` windows
    from a candidate universe. `selected_records`/
    `selected_canonical_indices` are in CANONICAL CANDIDATE ORDER (not
    selection-rank order); `selection_trace` is in selection-RANK order
    (0..window_count-1), exactly as the frozen digest formula requires."""

    selected_records: tuple[Mapping[str, Any], ...]
    selected_canonical_indices: tuple[int, ...]
    selection_trace: tuple[dict[str, Any], ...]
    candidate_universe_digest: str
    selected_windows_digest: str
    selection_trace_digest: str
    candidate_universe_size: int


def select_calibration_windows(
    candidate_universe: Sequence[Any], *, namespace: str = WINDOW_SELECTION_NAMESPACE, window_count: int = WINDOW_COUNT,
) -> SelectionResult:
    """The frozen deterministic selection algorithm: enumerate the
    candidate universe in canonical candidate order, hash each encoded
    record with the namespace, rank ascending by (selection_key_hex,
    canonical_candidate_index), select the first `window_count`. Fails
    closed if the universe is smaller than `window_count`. Never
    depends on filesystem/dict/PYTHONHASHSEED order, predictions, GT,
    or residual magnitude -- it only ever reads the candidate records
    themselves."""
    ordered = sort_candidate_universe(candidate_universe)
    if len(ordered) < window_count:
        raise DfpPhase1bCalibrationArtifactError(
            f"candidate universe has {len(ordered)} candidates, fewer than the required {window_count}"
        )
    encoded = [encode_candidate_record(r) for r in ordered]
    keys = [compute_selection_key(namespace, e) for e in encoded]
    ranked_indices = sorted(range(len(ordered)), key=lambda i: (keys[i], i))
    selected_by_rank = ranked_indices[:window_count]

    selection_trace = tuple(
        {"canonical_candidate_index": index, "selection_key": keys[index], "selection_rank": rank}
        for rank, index in enumerate(selected_by_rank)
    )

    selected_canonical_indices = tuple(sorted(selected_by_rank))
    selected_records = tuple(ordered[i] for i in selected_canonical_indices)

    candidate_universe_digest = digest_over_records(encoded)
    selected_windows_digest = digest_over_records([encoded[i] for i in selected_canonical_indices])
    selection_trace_digest = digest_over_records([_encode_trace_record(t) for t in selection_trace])

    return SelectionResult(
        selected_records=selected_records,
        selected_canonical_indices=selected_canonical_indices,
        selection_trace=selection_trace,
        candidate_universe_digest=candidate_universe_digest,
        selected_windows_digest=selected_windows_digest,
        selection_trace_digest=selection_trace_digest,
        candidate_universe_size=len(ordered),
    )


def verify_selection_against_universe(
    candidate_universe: Sequence[Any],
    claimed_selected_window_records: Sequence[Mapping[str, Any]],
    *,
    claimed_candidate_universe_digest: str,
    claimed_selected_windows_digest: str,
    claimed_selection_trace_digest: str,
    namespace: str = WINDOW_SELECTION_NAMESPACE,
    window_count: int = WINDOW_COUNT,
) -> None:
    """Independently RE-DERIVE the selection from the frozen candidate
    universe -- never merely check hashes carried inside the claimed
    document against each other. Raises on any disagreement, including
    a tampered per-row `selection_key`/`selection_rank`: those two
    fields are independently recomputed here from the candidate
    universe (via the same trusted `select_calibration_windows` call
    that produces the three global digests) and compared exactly
    against each claimed row -- never accepted merely because the
    row's OWN candidate identity and the document's aggregate digests
    happen to still be correct."""
    recomputed = select_calibration_windows(candidate_universe, namespace=namespace, window_count=window_count)
    trace_by_index = {trace["canonical_candidate_index"]: trace for trace in recomputed.selection_trace}

    if len(claimed_selected_window_records) != len(recomputed.selected_records):
        raise DfpPhase1bCalibrationArtifactError(
            f"claimed selected-window count {len(claimed_selected_window_records)} disagrees with "
            f"re-derived count {len(recomputed.selected_records)}"
        )
    for position, (claimed, recomputed_record, recomputed_index) in enumerate(
        zip(claimed_selected_window_records, recomputed.selected_records, recomputed.selected_canonical_indices)
    ):
        claimed_candidate_fields = {field: claimed[field] for field in CANDIDATE_FIELDS}
        if claimed_candidate_fields != dict(recomputed_record):
            raise DfpPhase1bCalibrationArtifactError(
                f"row {position}: claimed candidate identity disagrees with the frozen selection re-derived "
                "from the candidate universe"
            )
        if claimed["canonical_candidate_index"] != recomputed_index:
            raise DfpPhase1bCalibrationArtifactError(
                f"row {position}: claimed canonical_candidate_index {claimed['canonical_candidate_index']} "
                f"disagrees with re-derived index {recomputed_index}"
            )
        expected_trace = trace_by_index[recomputed_index]
        if claimed["selection_key"] != expected_trace["selection_key"]:
            raise DfpPhase1bCalibrationArtifactError(
                f"row {position}: claimed selection_key disagrees with the selection key independently "
                f"recomputed for canonical_candidate_index {recomputed_index} from the candidate universe "
                f"and the frozen namespace"
            )
        if claimed["selection_rank"] != expected_trace["selection_rank"]:
            raise DfpPhase1bCalibrationArtifactError(
                f"row {position}: claimed selection_rank {claimed['selection_rank']!r} disagrees with the "
                f"selection rank {expected_trace['selection_rank']!r} independently recomputed for "
                f"canonical_candidate_index {recomputed_index} from the complete candidate universe"
            )
    if claimed_candidate_universe_digest != recomputed.candidate_universe_digest:
        raise DfpPhase1bCalibrationArtifactError(
            "claimed candidate_universe_digest disagrees with the digest re-derived from the supplied universe"
        )
    if claimed_selected_windows_digest != recomputed.selected_windows_digest:
        raise DfpPhase1bCalibrationArtifactError(
            "claimed selected_windows_digest disagrees with the digest re-derived from the supplied universe"
        )
    if claimed_selection_trace_digest != recomputed.selection_trace_digest:
        raise DfpPhase1bCalibrationArtifactError(
            "claimed selection_trace_digest disagrees with the digest re-derived from the supplied universe"
        )


# ---------------------------------------------------------------------
# Residual summary validation.
# ---------------------------------------------------------------------


def validate_residual_summary(summary: Any, *, label: str) -> Mapping[str, Any]:
    _require_closed_mapping(summary, _RESIDUAL_SUMMARY_FIELD_SET, label)
    for field in _RESIDUAL_SUMMARY_COUNT_FIELDS:
        _require_exact_int(summary[field], f"{label}.{field}", minimum=0)
    sample_count = summary["sample_count"]
    finite_count = summary["finite_count"]
    non_finite_count = summary["non_finite_count"]
    if finite_count + non_finite_count != sample_count:
        raise DfpPhase1bCalibrationArtifactError(
            f"{label}: finite_count + non_finite_count must equal sample_count "
            f"({finite_count} + {non_finite_count} != {sample_count})"
        )
    if non_finite_count != 0:
        raise DfpPhase1bCalibrationArtifactError(f"{label}.non_finite_count must be 0 in a valid measurement, observed {non_finite_count}")

    floats: dict[str, float | None] = {}
    for field in _RESIDUAL_SUMMARY_FLOAT_FIELDS:
        floats[field] = _require_finite_float_or_null(summary[field], f"{label}.{field}")

    if sample_count == 0:
        for field, value in floats.items():
            if value is not None:
                raise DfpPhase1bCalibrationArtifactError(
                    f"{label}.{field} must be JSON null when sample_count is 0, observed {value!r}"
                )
        return summary

    for field, value in floats.items():
        if value is None:
            raise DfpPhase1bCalibrationArtifactError(f"{label}.{field} must not be null when sample_count > 0")

    if floats["signed_minimum"] > floats["signed_maximum"]:
        raise DfpPhase1bCalibrationArtifactError(f"{label}.signed_minimum must not exceed .signed_maximum")
    expected_max_abs = max(abs(floats["signed_minimum"]), abs(floats["signed_maximum"]))
    if floats["maximum_absolute_residual"] != expected_max_abs:
        raise DfpPhase1bCalibrationArtifactError(
            f"{label}.maximum_absolute_residual must equal max(abs(signed_minimum), abs(signed_maximum)) exactly"
        )
    if floats["absolute_residual_sum"] < 0:
        raise DfpPhase1bCalibrationArtifactError(f"{label}.absolute_residual_sum must be non-negative")
    if floats["squared_residual_sum"] < 0:
        raise DfpPhase1bCalibrationArtifactError(f"{label}.squared_residual_sum must be non-negative")
    expected_mean_abs = floats["absolute_residual_sum"] / sample_count
    if floats["mean_absolute_residual"] != expected_mean_abs:
        raise DfpPhase1bCalibrationArtifactError(
            f"{label}.mean_absolute_residual must equal absolute_residual_sum / sample_count exactly"
        )
    expected_rms = math.sqrt(floats["squared_residual_sum"] / sample_count)
    if floats["root_mean_square_residual"] != expected_rms:
        raise DfpPhase1bCalibrationArtifactError(
            f"{label}.root_mean_square_residual must equal sqrt(squared_residual_sum / sample_count) exactly"
        )
    return summary


def aggregate_residual_summaries(per_window_summaries: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
    """Exactly reconstruct one aggregate residual summary from a
    sequence of per-window summaries for the SAME residual family,
    using only their additive sum/count fields -- never a pooled mean
    or RMS computed any other way."""
    total_sample_count = 0
    total_finite_count = 0
    total_non_finite_count = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    signed_min: float | None = None
    signed_max: float | None = None
    for window_summary in per_window_summaries:
        validate_residual_summary(window_summary, label=label)
        total_sample_count += window_summary["sample_count"]
        total_finite_count += window_summary["finite_count"]
        total_non_finite_count += window_summary["non_finite_count"]
        if window_summary["sample_count"] == 0:
            continue
        absolute_sum += window_summary["absolute_residual_sum"]
        squared_sum += window_summary["squared_residual_sum"]
        window_min = window_summary["signed_minimum"]
        window_max = window_summary["signed_maximum"]
        signed_min = window_min if signed_min is None else min(signed_min, window_min)
        signed_max = window_max if signed_max is None else max(signed_max, window_max)

    if total_sample_count == 0:
        return {
            "sample_count": 0, "finite_count": 0, "non_finite_count": 0,
            "signed_minimum": None, "signed_maximum": None,
            "maximum_absolute_residual": None, "mean_absolute_residual": None, "root_mean_square_residual": None,
            "absolute_residual_sum": None, "squared_residual_sum": None,
        }
    return {
        "sample_count": total_sample_count,
        "finite_count": total_finite_count,
        "non_finite_count": total_non_finite_count,
        "signed_minimum": signed_min,
        "signed_maximum": signed_max,
        "maximum_absolute_residual": max(abs(signed_min), abs(signed_max)),
        "mean_absolute_residual": absolute_sum / total_sample_count,
        "root_mean_square_residual": math.sqrt(squared_sum / total_sample_count),
        "absolute_residual_sum": absolute_sum,
        "squared_residual_sum": squared_sum,
    }


# ---------------------------------------------------------------------
# Full artifact validation.
# ---------------------------------------------------------------------


def validate_selected_window_record(record: Any, *, label: str) -> Mapping[str, Any]:
    _require_closed_mapping(record, _SELECTED_WINDOW_RECORD_FIELD_SET, label)
    validate_candidate_record({field: record[field] for field in CANDIDATE_FIELDS})
    _require_exact_int(record["canonical_candidate_index"], f"{label}.canonical_candidate_index", minimum=0)
    _require_exact_int(record["selection_rank"], f"{label}.selection_rank", minimum=0)
    _require_sha256(record["selection_key"], f"{label}.selection_key")
    _require_exact_bool(record["completed"], f"{label}.completed")
    for family in RESIDUAL_FAMILIES:
        validate_residual_summary(record[family], label=f"{label}.{family}")
    return record


def _validate_device_record(value: Any, *, label: str) -> None:
    _require_closed_mapping(value, DEVICE_RECORD_FIELDS, label)
    if value["device_type"] != "cuda":
        raise DfpPhase1bCalibrationArtifactError(f"{label}.device_type must be 'cuda', observed {value['device_type']!r}")
    _require_exact_string(value["gpu_model_name"], f"{label}.gpu_model_name")
    _require_exact_string(value["cuda_compute_capability"], f"{label}.cuda_compute_capability")
    _require_exact_int(value["total_device_memory_bytes"], f"{label}.total_device_memory_bytes", minimum=1)
    _require_exact_string(value["cuda_driver_version"], f"{label}.cuda_driver_version")
    _require_exact_string(value["cuda_runtime_version"], f"{label}.cuda_runtime_version")


def _validate_software_record(value: Any, *, label: str) -> None:
    _require_closed_mapping(value, SOFTWARE_RECORD_FIELDS, label)
    _require_git_commit(value["repository_commit"], f"{label}.repository_commit")
    for field in ("python_version", "torch_version", "numpy_version", "cuda_version", "cudnn_version"):
        _require_exact_string(value[field], f"{label}.{field}")
    for field in ("deterministic_algorithms_enabled", "tf32_enabled", "autocast_enabled"):
        _require_exact_bool(value[field], f"{label}.{field}")


def _validate_propagation_precision_record(value: Any, *, label: str) -> None:
    _require_closed_mapping(value, PROPAGATION_PRECISION_RECORD_FIELDS, label)
    for field, expected in _PROPAGATION_PRECISION_EXACT_DTYPES.items():
        observed = _require_exact_string(value[field], f"{label}.{field}")
        if observed != expected:
            raise DfpPhase1bCalibrationArtifactError(
                f"{label}.{field} must be exactly {expected!r} (the frozen E16 execution_contract value), "
                f"observed {observed!r}"
            )
    if value["k"] != 12:
        raise DfpPhase1bCalibrationArtifactError(f"{label}.k must be exactly 12, observed {value['k']!r}")
    if value["steps"] != 320:
        raise DfpPhase1bCalibrationArtifactError(f"{label}.steps must be exactly 320, observed {value['steps']!r}")
    if type(value["alpha"]) is not float:
        raise DfpPhase1bCalibrationArtifactError(f"{label}.alpha must be an exact float")
    if not math.isfinite(value["alpha"]):
        raise DfpPhase1bCalibrationArtifactError(f"{label}.alpha must be finite, observed {value['alpha']!r}")
    if value["alpha"] != EXPECTED_ALPHA:
        raise DfpPhase1bCalibrationArtifactError(
            f"{label}.alpha must be exactly {EXPECTED_ALPHA!r} (the frozen recurrence alpha, transitively bound "
            f"via e15_protocol to e13_dfp_phase1a_protocol.toml's own [propagation].alpha), observed {value['alpha']!r}"
        )
    _require_exact_int(value["chunk_size"], f"{label}.chunk_size", minimum=1)


def validate_calibration_artifact(
    document: Any, *, candidate_universe: Sequence[Any] | None = None,
) -> Mapping[str, Any]:
    """Full closed-schema validation of a `phase1b_conservation_
    calibration.json` document. When `candidate_universe` is supplied,
    additionally RE-DERIVES the selection from it and confirms every
    row/digest agrees -- this is the independent verification path,
    never a mere internal-hash cross-check."""
    _require_closed_mapping(document, _TOP_LEVEL_ARTIFACT_FIELD_SET, "calibration artifact")

    if document["schema_name"] != ARTIFACT_SCHEMA_NAME:
        raise DfpPhase1bCalibrationArtifactError(
            f"calibration artifact.schema_name must be {ARTIFACT_SCHEMA_NAME!r}, observed {document['schema_name']!r}"
        )
    if type(document["schema_version"]) is not int or document["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise DfpPhase1bCalibrationArtifactError(
            f"calibration artifact.schema_version must be the exact int {ARTIFACT_SCHEMA_VERSION}, "
            f"observed {document['schema_version']!r}"
        )
    complete = _require_exact_bool(document["complete"], "calibration artifact.complete")
    final = _require_exact_bool(document["final"], "calibration artifact.final")

    for field in ("calibration_identity_sha256", "bound_e15_protocol_sha256", "bound_e15_artifacts_sha256",
                  "bound_masked_forward_sha256", "bound_subset_manifest_sha256",
                  "selected_ids_digest", "candidate_universe_digest", "selected_windows_digest", "selection_trace_digest"):
        _require_sha256(document[field], f"calibration artifact.{field}")
    _require_git_commit(document["repository_commit"], "calibration artifact.repository_commit")
    if document["run_mode"] != "coco_stuff_500":
        raise DfpPhase1bCalibrationArtifactError(
            f"calibration artifact.run_mode must be 'coco_stuff_500', observed {document['run_mode']!r}"
        )
    candidate_window_count = _require_exact_int(document["candidate_window_count"], "calibration artifact.candidate_window_count", minimum=WINDOW_COUNT)
    expected_selected_window_count = _require_exact_int(
        document["expected_selected_window_count"], "calibration artifact.expected_selected_window_count",
    )
    if expected_selected_window_count != WINDOW_COUNT:
        raise DfpPhase1bCalibrationArtifactError(
            f"calibration artifact.expected_selected_window_count must be exactly {WINDOW_COUNT}, "
            f"observed {expected_selected_window_count}"
        )
    completed_selected_window_count = _require_exact_int(
        document["completed_selected_window_count"], "calibration artifact.completed_selected_window_count", minimum=0,
    )
    if completed_selected_window_count > expected_selected_window_count:
        raise DfpPhase1bCalibrationArtifactError(
            "calibration artifact.completed_selected_window_count must not exceed expected_selected_window_count"
        )

    _validate_device_record(document["device_record"], label="calibration artifact.device_record")
    _validate_software_record(document["software_record"], label="calibration artifact.software_record")
    _validate_propagation_precision_record(document["propagation_precision_record"], label="calibration artifact.propagation_precision_record")

    rows = document["selected_window_records"]
    if not isinstance(rows, list):
        raise DfpPhase1bCalibrationArtifactError("calibration artifact.selected_window_records must be a list")
    if len(rows) > expected_selected_window_count:
        raise DfpPhase1bCalibrationArtifactError(
            f"calibration artifact.selected_window_records has {len(rows)} rows, exceeding "
            f"expected_selected_window_count={expected_selected_window_count}"
        )
    for position, row in enumerate(rows):
        validate_selected_window_record(row, label=f"calibration artifact.selected_window_records[{position}]")

    seen_indices: set[int] = set()
    seen_ranks: set[int] = set()
    seen_keys: set[str] = set()
    for position, row in enumerate(rows):
        index, rank, key = row["canonical_candidate_index"], row["selection_rank"], row["selection_key"]
        if index in seen_indices:
            raise DfpPhase1bCalibrationArtifactError(f"duplicate canonical_candidate_index {index} in selected_window_records")
        if rank in seen_ranks:
            raise DfpPhase1bCalibrationArtifactError(f"duplicate selection_rank {rank} in selected_window_records")
        if key in seen_keys:
            raise DfpPhase1bCalibrationArtifactError(f"duplicate selection_key {key} in selected_window_records")
        seen_indices.add(index)
        seen_ranks.add(rank)
        seen_keys.add(key)
    order_keys = [canonical_candidate_order_key(row) for row in rows]
    if order_keys != sorted(order_keys):
        raise DfpPhase1bCalibrationArtifactError(
            "calibration artifact.selected_window_records must be stored in canonical candidate order "
            "(image_manifest_ordinal ascending, then canonical_window_ordinal ascending), not selection-rank order"
        )

    completed_count = sum(1 for row in rows if row["completed"])
    if completed_count != completed_selected_window_count:
        raise DfpPhase1bCalibrationArtifactError(
            f"calibration artifact.completed_selected_window_count ({completed_selected_window_count}) disagrees "
            f"with the actual number of completed rows ({completed_count})"
        )

    if complete or final:
        if len(rows) != expected_selected_window_count:
            raise DfpPhase1bCalibrationArtifactError(
                f"a complete/final calibration artifact must have exactly {expected_selected_window_count} rows, "
                f"observed {len(rows)}"
            )
        if completed_selected_window_count != expected_selected_window_count:
            raise DfpPhase1bCalibrationArtifactError(
                "a complete/final calibration artifact must have completed_selected_window_count == "
                "expected_selected_window_count"
            )
        for position, row in enumerate(rows):
            if not row["completed"]:
                raise DfpPhase1bCalibrationArtifactError(
                    f"a complete/final calibration artifact must have every row completed; row {position} is not"
                )
            for family in RESIDUAL_FAMILIES:
                if row[family]["non_finite_count"] != 0:
                    raise DfpPhase1bCalibrationArtifactError(
                        f"a complete/final calibration artifact must have non_finite_count == 0 everywhere "
                        f"(row {position}, {family})"
                    )

    aggregates = document["aggregate_residual_summaries"]
    if not isinstance(aggregates, Mapping) or set(aggregates) != frozenset(RESIDUAL_FAMILIES):
        raise DfpPhase1bCalibrationArtifactError(
            "calibration artifact.aggregate_residual_summaries must contain exactly the four residual families"
        )
    for family in RESIDUAL_FAMILIES:
        validate_residual_summary(aggregates[family], label=f"calibration artifact.aggregate_residual_summaries.{family}")
        completed_family_summaries = [row[family] for row in rows if row["completed"]]
        expected_aggregate = aggregate_residual_summaries(completed_family_summaries, label=f"reconstruction.{family}")
        if dict(aggregates[family]) != expected_aggregate:
            raise DfpPhase1bCalibrationArtifactError(
                f"calibration artifact.aggregate_residual_summaries.{family} is not exactly reconstructable "
                "from the per-window absolute_residual_sum/squared_residual_sum/sample_count fields"
            )

    for field_name in TOP_LEVEL_ARTIFACT_FIELDS + SELECTED_WINDOW_RECORD_FIELDS + RESIDUAL_SUMMARY_FIELDS:
        lowered = field_name.lower()
        if any(fragment in lowered for fragment in _FORBIDDEN_FIELD_NAME_FRAGMENTS):
            raise DfpPhase1bCalibrationArtifactError(
                f"internal schema error: field name {field_name!r} resembles a forbidden tolerance/verdict field"
            )

    if candidate_universe is not None:
        universe_ordered = sort_candidate_universe(candidate_universe)
        if candidate_window_count != len(universe_ordered):
            raise DfpPhase1bCalibrationArtifactError(
                f"calibration artifact.candidate_window_count ({candidate_window_count}) disagrees with the "
                f"supplied candidate universe size ({len(universe_ordered)})"
            )
        recomputed = select_calibration_windows(universe_ordered)
        if document["candidate_universe_digest"] != recomputed.candidate_universe_digest:
            raise DfpPhase1bCalibrationArtifactError(
                "calibration artifact.candidate_universe_digest disagrees with the digest re-derived from the "
                "supplied candidate universe"
            )
        if len(rows) == WINDOW_COUNT:
            verify_selection_against_universe(
                universe_ordered,
                [
                    {field: row[field] for field in CANDIDATE_FIELDS + ("canonical_candidate_index", "selection_key", "selection_rank")}
                    for row in rows
                ],
                claimed_candidate_universe_digest=document["candidate_universe_digest"],
                claimed_selected_windows_digest=document["selected_windows_digest"],
                claimed_selection_trace_digest=document["selection_trace_digest"],
            )

    return document


__all__ = [
    "ARTIFACT_SCHEMA_NAME",
    "ARTIFACT_SCHEMA_VERSION",
    "CANDIDATE_FIELDS",
    "DfpPhase1bCalibrationArtifactError",
    "REQUIRED_ARTIFACT_RELATIVE_PATH",
    "RESIDUAL_FAMILIES",
    "RESIDUAL_SUMMARY_FIELDS",
    "SELECTED_WINDOW_RECORD_FIELDS",
    "SelectionResult",
    "TOP_LEVEL_ARTIFACT_FIELDS",
    "WINDOW_COUNT",
    "WINDOW_SELECTION_NAMESPACE",
    "aggregate_residual_summaries",
    "canonical_candidate_order_key",
    "compute_selection_key",
    "digest_over_records",
    "encode_candidate_record",
    "select_calibration_windows",
    "sort_candidate_universe",
    "validate_calibration_artifact",
    "validate_candidate_record",
    "validate_residual_summary",
    "validate_selected_window_record",
    "verify_selection_against_universe",
]
