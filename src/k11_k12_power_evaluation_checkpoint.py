"""Authoritative strict JSON loading and checkpoint-invariant validation for
the matched k11/k12 finite-step power evaluator.

This is the single, shared place both the evaluator's own resume path
(``diagnostics/run_matched_k11_k12_evaluation.py``) and the standalone
``verify_k11_k12_power_evaluation.py verify-checkpoint`` command use --
neither may reimplement or maintain a parallel, weaker version of any
check defined here. A checkpoint that fails any of these invariants must
never be resumed: the caller must fail closed before any dataset index is
skipped, duplicated, or reordered.

Validation is split into two phases, matching the order in which
information actually becomes available:

Phase A (:func:`validate_checkpoint_structure`,
:func:`validate_checkpoint_against_artifact`) needs only the checkpoint
JSON and the per-image-stats artifact already on disk -- no dataset, no
model, no CUDA. This is exactly what can (and must) be checked before
``_build_inference`` runs.

Phase B (:func:`validate_checkpoint_against_canonical_order`) needs the
freshly-resolved canonical dataset image order (cheap -- ``img_infos``
only, never runs the pipeline) and so can only run after the dataset
object exists.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.k11_k12_power_evaluation_identity import K11K12PowerEvaluationError, RUN_MODE_IMAGE_COUNT_KEYS


CHECKPOINT_SCHEMA_NAME = "talk2dino-k11-k12-power-evaluation-checkpoint-v1"

TOP_CHECKPOINT_KEYS = frozenset(
    {
        "schema", "run_mode", "identity", "identity_sha256", "matched_identity_sha256",
        "stability_result_sha256", "finite_step_kernel_sha256", "git_commit", "class_count",
        "image_count_expected", "image_order_digest", "next_dataset_index", "completed_image_ids",
        "images_completed_count", "windows_processed_total", "complete", "created_at_utc", "updated_at_utc",
    }
)

_PER_IMAGE_ARRAY_KEYS = ("label", "intersect_k11", "union_k11", "pred_k11", "intersect_k12", "union_k12", "pred_k12")
_VARIANT_ARRAY_KEYS = {
    "k11": ("intersect_k11", "union_k11", "pred_k11"),
    "k12": ("intersect_k12", "union_k12", "pred_k12"),
}


# ---------------------------------------------------------------------------
# Strict JSON loading -- the one shared loader for every JSON document this
# evaluator reads or writes (checkpoint, result, per-image-stats manifest,
# and -- via re-export -- the stability-gate result it binds against).
# ---------------------------------------------------------------------------


def _closed_object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise K11K12PowerEvaluationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_non_finite(token: str) -> Any:
    raise K11K12PowerEvaluationError(f"document contains non-finite JSON constant {token}")


def parse_strict_json_document(path: Path, *, label: str) -> Mapping[str, Any]:
    """Read and strictly parse one JSON document, failing closed (with a
    concise :class:`K11K12PowerEvaluationError`, never an uncaught
    traceback) on every malformed-input case this evaluator must handle:
    a missing file, a permission error, a path that is a directory,
    invalid UTF-8, an empty file, truncated JSON, trailing garbage,
    duplicate object keys, ``NaN``/``Infinity``/``-Infinity``, and a
    non-object JSON root (a bare list, string, or number).

    Floats are parsed as :class:`~decimal.Decimal` (never a lossy
    intermediate float64 round-trip) and converted to plain ``float`` only
    after every structural check has already passed, mirroring the same
    strict-parsing contract already used for the stability-gate result.
    """
    path = Path(path)
    try:
        raw_bytes = path.read_bytes()
    except IsADirectoryError as error:
        raise K11K12PowerEvaluationError(f"cannot read {label} {path}: path is a directory") from error
    except PermissionError as error:
        raise K11K12PowerEvaluationError(f"cannot read {label} {path}: permission denied") from error
    except FileNotFoundError as error:
        raise K11K12PowerEvaluationError(f"cannot read {label} {path}: file does not exist") from error
    except OSError as error:
        raise K11K12PowerEvaluationError(f"cannot read {label} {path}: {error}") from error

    try:
        text = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise K11K12PowerEvaluationError(f"{label} {path} is not valid UTF-8: {error}") from error

    if not text.strip():
        raise K11K12PowerEvaluationError(f"{label} {path} is empty")

    try:
        value = json.loads(
            text,
            parse_float=Decimal,
            object_pairs_hook=_closed_object_pairs_hook,
            parse_constant=_reject_non_finite,
        )
    except K11K12PowerEvaluationError:
        raise
    except json.JSONDecodeError as error:
        raise K11K12PowerEvaluationError(f"cannot parse {label} {path}: {error}") from error

    if not isinstance(value, Mapping):
        raise K11K12PowerEvaluationError(
            f"{label} {path} must contain one JSON object at its root, observed {type(value).__name__}"
        )
    return _decimals_to_float(value)


def _decimals_to_float(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Mapping):
        return {key: _decimals_to_float(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decimals_to_float(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Primitive type validators (deliberately duplicated in spirit -- not
# imported -- from the sibling identity/report modules, matching this
# repository's established convention of small, self-contained validator
# modules; the checkpoint INVARIANT LOGIC itself, unlike these one-line
# primitives, is never duplicated -- see the module docstring).
# ---------------------------------------------------------------------------


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise K11K12PowerEvaluationError(f"{label} has an unexpected schema")
    return value


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise K11K12PowerEvaluationError(f"{label} must be an exact non-empty string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise K11K12PowerEvaluationError(f"{label} must be an exact boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise K11K12PowerEvaluationError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise K11K12PowerEvaluationError(f"{label} must be at least {minimum}, observed {value}")
    if maximum is not None and value > maximum:
        raise K11K12PowerEvaluationError(f"{label} must be at most {maximum}, observed {value}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise K11K12PowerEvaluationError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 40 or any(c not in "0123456789abcdef" for c in token):
        raise K11K12PowerEvaluationError(f"{label} must be a full Git identity")
    return token


def _require_string_list(value: Any, label: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise K11K12PowerEvaluationError(f"{label} must be a list of exact strings")
    return value


# ---------------------------------------------------------------------------
# Phase A: pure structural validation of the checkpoint document itself
# ---------------------------------------------------------------------------


def validate_checkpoint_structure(
    checkpoint: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    identity_sha256: str,
    run_mode: str | None = None,
    stability_result_sha256: str | None = None,
    finite_step_kernel_sha256: str | None = None,
    class_count: int | None = None,
) -> None:
    """Validate everything that can be checked from the checkpoint document
    alone (never requires the dataset, model, or CUDA).

    This is the SOLE authority for the checkpoint schema/type/binding/
    completeness invariants -- both the evaluator's resume path and
    ``verify-checkpoint`` call this exact function; neither may reimplement
    any part of it. ``run_mode``/``stability_result_sha256``/
    ``finite_step_kernel_sha256``/``class_count`` are optional external
    cross-checks: the evaluator's resume path (which already knows
    ``--run-mode`` and has just computed the live stability/kernel/class-
    count bindings) always supplies all four for a full cross-check; the
    standalone ``verify-checkpoint`` command, which has no such external
    context, omits them and still gets the full *internal* self-consistency
    validation -- including the checkpoint's own recorded
    ``image_count_expected`` cross-checked against what the identity
    registers for the checkpoint's own recorded ``run_mode``.
    """
    _require_closed_mapping(checkpoint, TOP_CHECKPOINT_KEYS, "checkpoint")

    if _require_exact_string(checkpoint["schema"], "checkpoint.schema") != CHECKPOINT_SCHEMA_NAME:
        raise K11K12PowerEvaluationError("checkpoint.schema mismatch")
    if _require_exact_string(checkpoint["identity"], "checkpoint.identity") != identity["identity"]["name"]:
        raise K11K12PowerEvaluationError("checkpoint.identity mismatch")
    if _require_sha256(checkpoint["identity_sha256"], "checkpoint.identity_sha256") != identity_sha256:
        raise K11K12PowerEvaluationError("checkpoint.identity_sha256 does not match the loaded identity file")
    if checkpoint["matched_identity_sha256"] != identity["parent_identity"]["matched_identity_sha256"]:
        raise K11K12PowerEvaluationError("checkpoint.matched_identity_sha256 disagrees with the identity's parent")
    recorded_stability_sha = _require_sha256(checkpoint["stability_result_sha256"], "checkpoint.stability_result_sha256")
    if stability_result_sha256 is not None and recorded_stability_sha != stability_result_sha256:
        raise K11K12PowerEvaluationError("checkpoint.stability_result_sha256 disagrees with --stability-result")
    recorded_kernel_sha = _require_sha256(checkpoint["finite_step_kernel_sha256"], "checkpoint.finite_step_kernel_sha256")
    if finite_step_kernel_sha256 is not None and recorded_kernel_sha != finite_step_kernel_sha256:
        raise K11K12PowerEvaluationError("checkpoint.finite_step_kernel_sha256 disagrees with the current kernel source")
    _require_git_identity(checkpoint["git_commit"], "checkpoint.git_commit")

    recorded_run_mode = _require_exact_string(checkpoint["run_mode"], "checkpoint.run_mode")
    if recorded_run_mode not in RUN_MODE_IMAGE_COUNT_KEYS:
        raise K11K12PowerEvaluationError(f"checkpoint.run_mode must be one of {sorted(RUN_MODE_IMAGE_COUNT_KEYS)}")
    if run_mode is not None and recorded_run_mode != run_mode:
        raise K11K12PowerEvaluationError("checkpoint.run_mode disagrees with --run-mode")
    registered_image_count = identity["run_modes"][RUN_MODE_IMAGE_COUNT_KEYS[recorded_run_mode]]
    expected_image_count = _require_int(
        checkpoint["image_count_expected"], "checkpoint.image_count_expected", minimum=1
    )
    if expected_image_count != registered_image_count:
        raise K11K12PowerEvaluationError(
            f"checkpoint.image_count_expected ({expected_image_count}) disagrees with the registered "
            f"{recorded_run_mode} image count ({registered_image_count})"
        )
    recorded_class_count = _require_int(checkpoint["class_count"], "checkpoint.class_count", minimum=1)
    if class_count is not None and recorded_class_count != class_count:
        raise K11K12PowerEvaluationError("checkpoint.class_count disagrees with the live inference class count")

    _require_sha256(checkpoint["image_order_digest"], "checkpoint.image_order_digest")

    next_index = _require_int(
        checkpoint["next_dataset_index"], "checkpoint.next_dataset_index",
        minimum=0, maximum=expected_image_count,
    )
    completed = _require_string_list(checkpoint["completed_image_ids"], "checkpoint.completed_image_ids")
    if len(set(completed)) != len(completed):
        raise K11K12PowerEvaluationError("checkpoint.completed_image_ids contains a duplicate image ID")
    images_completed_count = _require_int(
        checkpoint["images_completed_count"], "checkpoint.images_completed_count", minimum=0
    )
    if images_completed_count != len(completed):
        raise K11K12PowerEvaluationError("checkpoint.images_completed_count disagrees with len(completed_image_ids)")
    # THE key invariant this module exists to enforce: next_dataset_index
    # must equal the number of images actually completed. A checkpoint
    # where these disagree previously caused the evaluator to silently
    # resume from the wrong dataset index -- skipping (next_dataset_index
    # too high) or reprocessing/duplicating (too low) images.
    if next_index != len(completed):
        raise K11K12PowerEvaluationError(
            f"checkpoint.next_dataset_index ({next_index}) must equal the number of completed images "
            f"({len(completed)}) -- refusing to resume a checkpoint that could skip or duplicate an image"
        )

    complete = _require_bool(checkpoint["complete"], "checkpoint.complete")
    # One-directional, deliberately not a biconditional: complete=true
    # REQUIRES next_dataset_index == expected_image_count (this is the
    # safety-critical direction -- it is what rejects a checkpoint that
    # falsely claims completion). The converse is legitimate and must NOT
    # be rejected: the evaluator writes an intermediate, complete=false
    # checkpoint immediately after its very last per-image update (where
    # next_dataset_index already equals expected_image_count), before its
    # separate post-loop completion contract/self-verification has run and
    # promoted the checkpoint to complete=true -- that is a well-defined,
    # safely-resumable state (resuming it immediately satisfies the loop's
    # exit condition and proceeds straight to the completion contract).
    if complete and next_index != expected_image_count:
        raise K11K12PowerEvaluationError(
            "checkpoint.complete is true but next_dataset_index does not equal the expected image count "
            f"(next_dataset_index={next_index}, expected={expected_image_count})"
        )

    _require_int(checkpoint["windows_processed_total"], "checkpoint.windows_processed_total", minimum=0)
    _require_exact_string(checkpoint["created_at_utc"], "checkpoint.created_at_utc")
    _require_exact_string(checkpoint["updated_at_utc"], "checkpoint.updated_at_utc")


# ---------------------------------------------------------------------------
# Phase A, continued: cross-check the checkpoint against the per-image-
# stats artifact already on disk
# ---------------------------------------------------------------------------


def validate_checkpoint_against_artifact(
    checkpoint: Mapping[str, Any],
    *,
    stats_manifest: Mapping[str, Any],
    stats_arrays: Mapping[str, Any],
) -> None:
    """Cross-check the checkpoint's own claims against the per-image
    sufficient-statistics artifact (manifest + NPZ arrays) already written
    to disk. Requires :func:`validate_checkpoint_structure` to have already
    passed (so ``next_dataset_index``/``completed_image_ids``/``class_count``
    are already known-well-formed). ``class_count`` is deliberately not a
    parameter here: it is read from ``checkpoint["class_count"]`` itself --
    the live ``inference.num_classes`` cross-check happens separately, once
    the model exists (see the evaluator CLI's Phase B), since this function
    must remain callable before any model/CUDA construction."""
    next_index = checkpoint["next_dataset_index"]
    completed = checkpoint["completed_image_ids"]
    class_count = checkpoint["class_count"]

    manifest_class_count = _require_int(stats_manifest["class_count"], "per-image-stats manifest.class_count", minimum=1)
    if manifest_class_count != class_count:
        raise K11K12PowerEvaluationError(
            "per-image-stats manifest.class_count disagrees with checkpoint.class_count"
        )

    recorded_ids = _require_string_list(stats_manifest["image_ids"], "per-image-stats manifest.image_ids")
    if recorded_ids != completed:
        raise K11K12PowerEvaluationError(
            "per-image-stats artifact image order disagrees with the checkpoint; refusing to resume"
        )
    recorded_indices = stats_manifest["dataset_indices"]
    if list(recorded_indices) != list(range(next_index)):
        raise K11K12PowerEvaluationError(
            "per-image-stats artifact dataset indices are not the exact canonical prefix "
            f"[0, ..., {next_index - 1}] -- no gap, reordering, skip, or future index is permitted, "
            f"observed {list(recorded_indices)!r}"
        )

    array_row_counts: dict[str, int] = {}
    for key in _PER_IMAGE_ARRAY_KEYS:
        if key not in stats_arrays:
            raise K11K12PowerEvaluationError(f"per-image-stats artifact is missing array {key!r}")
        array = stats_arrays[key]
        if array.ndim != 2:
            raise K11K12PowerEvaluationError(f"per-image-stats artifact array {key!r} must be 2-dimensional")
        array_row_counts[key] = int(array.shape[0])
        if array.shape[0] != next_index:
            raise K11K12PowerEvaluationError(
                f"per-image-stats artifact array {key!r} has {array.shape[0]} rows, expected {next_index} "
                "(one row per completed image)"
            )
        if array.shape[1] != class_count:
            raise K11K12PowerEvaluationError(
                f"per-image-stats artifact array {key!r} has {array.shape[1]} columns, expected class_count={class_count}"
            )

    # k11 and k12 per-image row counts identical
    for variant, keys in _VARIANT_ARRAY_KEYS.items():
        counts = {array_row_counts[key] for key in keys}
        if len(counts) != 1:
            raise K11K12PowerEvaluationError(f"per-image-stats artifact {variant} arrays have mismatched row counts: {counts}")
    if array_row_counts["intersect_k11"] != array_row_counts["intersect_k12"]:
        raise K11K12PowerEvaluationError(
            "per-image-stats artifact k11 and k12 row counts disagree "
            f"({array_row_counts['intersect_k11']} vs {array_row_counts['intersect_k12']})"
        )

    # GT (label) is architecturally shared, never duplicated per variant --
    # confirm the artifact never carries a separate label_k11/label_k12.
    if "label_k11" in stats_arrays or "label_k12" in stats_arrays:
        raise K11K12PowerEvaluationError(
            "per-image-stats artifact must store one shared GT ('label') array, never per-variant GT arrays"
        )

    # Basic finiteness/non-negativity sanity on the aggregate arrays --
    # "aggregate statistics equal the sum of completed rows" is trivially
    # true by construction (there is no separately-stored aggregate; the
    # rows themselves ARE what gets summed at metric time), so this
    # instead validates the rows are internally sane sufficient statistics.
    import numpy as np

    for key in _PER_IMAGE_ARRAY_KEYS:
        array = stats_arrays[key]
        if array.size and (not np.isfinite(array).all() or (array < 0).any()):
            raise K11K12PowerEvaluationError(f"per-image-stats artifact array {key!r} contains a negative or non-finite value")
    for variant, (intersect_key, union_key, pred_key) in _VARIANT_ARRAY_KEYS.items():
        intersect = stats_arrays[intersect_key]
        union = stats_arrays[union_key]
        pred = stats_arrays[pred_key]
        label = stats_arrays["label"]
        if (intersect > union).any():
            raise K11K12PowerEvaluationError(f"per-image-stats artifact {variant}: intersect exceeds union")
        if (intersect > pred).any():
            raise K11K12PowerEvaluationError(f"per-image-stats artifact {variant}: intersect exceeds predicted-pixel count")
        if (intersect > label).any():
            raise K11K12PowerEvaluationError(f"per-image-stats artifact {variant}: intersect exceeds ground-truth pixel count")


# ---------------------------------------------------------------------------
# Phase B: dataset-dependent canonical-order validation
# ---------------------------------------------------------------------------


def validate_checkpoint_against_canonical_order(
    checkpoint: Mapping[str, Any], expected_image_ids: Sequence[str], *, image_order_digest: str
) -> None:
    """Validate the checkpoint against the freshly-resolved canonical
    dataset image order. Requires the real dataset object (cheap:
    ``img_infos`` only, never runs the pipeline) so this can only run
    after ``_build_inference``."""
    if checkpoint["image_order_digest"] != image_order_digest:
        raise K11K12PowerEvaluationError(
            "checkpoint.image_order_digest disagrees with the freshly-resolved dataset order"
        )
    next_index = checkpoint["next_dataset_index"]
    expected_prefix = list(expected_image_ids[:next_index])
    if checkpoint["completed_image_ids"] != expected_prefix:
        raise K11K12PowerEvaluationError(
            "checkpoint.completed_image_ids does not match the canonical dataset image-ID prefix "
            f"for the first {next_index} images; refusing to resume against a mismatched image order"
        )


def resume_dataset_index(checkpoint: Mapping[str, Any]) -> int:
    if checkpoint["complete"] is True:
        raise K11K12PowerEvaluationError("checkpoint is already complete; refusing to resume it as though incomplete")
    return checkpoint["next_dataset_index"]


__all__ = [
    "CHECKPOINT_SCHEMA_NAME",
    "TOP_CHECKPOINT_KEYS",
    "parse_strict_json_document",
    "resume_dataset_index",
    "validate_checkpoint_against_artifact",
    "validate_checkpoint_against_canonical_order",
    "validate_checkpoint_structure",
]
