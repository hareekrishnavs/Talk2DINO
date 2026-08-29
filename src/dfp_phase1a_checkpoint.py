"""Checkpoint schema/invariant validation, per-image compute-projection
statistics NPZ schema/validation, ordered resume validation, and atomic
generation/install for DFP Phase 1A diagnostic runs.

Sole authority for the checkpoint's schema/type/binding/completeness
checks and for the per-image-stats NPZ member schema -- never
reimplemented at any call site (the evaluator's own resume path and the
standalone ``verify_dfp_phase1a_artifacts.py`` CLI both import this
module rather than maintaining a parallel, weaker version of any check
defined here). A checkpoint or NPZ that fails any invariant must never be
resumed: callers fail closed before any dataset index is skipped,
duplicated, or reordered.

Validation is split to match when information actually becomes available:
Phase A (:func:`validate_resume_state`, composing the ten functions below
it) needs only the checkpoint JSON and the per-image-stats artifact
already on disk -- no dataset, no model, no CUDA, no torch import beyond
what NumPy itself requires. This is exactly what can (and must) run
before any dataset/model/CUDA entry point.
"""

from __future__ import annotations

import secrets
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.dfp_phase1a_artifact_schema import (
    ALL_NPZ_MEMBER_NAMES,
    CHECKPOINT_SCHEMA_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    FLOAT64_MEMBER_NAMES,
    HORIZON_ORDER,
    INT64_MEMBER_NAMES,
    PER_IMAGE_STATS_SCHEMA_NAME,
    PER_IMAGE_STATS_SCHEMA_VERSION,
    RUN_MODE_NAMES,
    DfpPhase1aArtifactSchemaError,
    parse_strict_json_document,
    require_safe_basename,
    sha256_file,
    write_json_atomically,
)


class DfpPhase1aCheckpointError(ValueError):
    """Raised on any checkpoint, per-image-stats NPZ, or resume-state
    invariant violation. Always fail closed."""


TOP_CHECKPOINT_KEYS = frozenset(
    {
        "schema", "schema_version", "artifact_schema_identity_sha256", "protocol_identity_sha256",
        "subset_manifest_sha256", "selected_ids_digest", "run_mode", "image_count_expected", "class_count",
        "class_names_digest", "horizon_order", "source_commit", "required_ancestor_commit",
        "bridge_checkpoint_sha256", "next_dataset_index", "completed_image_ids", "images_completed_count",
        "total_windows_processed", "per_image_stats_manifest_relative_path", "per_image_stats_manifest_sha256",
        "complete", "final_result_produced", "generation_nonce", "created_at_utc", "updated_at_utc",
    }
)
assert len(TOP_CHECKPOINT_KEYS) == 25

PER_IMAGE_STATS_MANIFEST_KEYS = frozenset(
    {
        "schema", "schema_version", "npz_filename", "npz_sha256", "run_mode", "class_count",
        "class_names_digest", "image_count", "image_ids",
    }
)


# ---------------------------------------------------------------------------
# Primitive validators.
# ---------------------------------------------------------------------------


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        missing = expected_keys - set(value if isinstance(value, Mapping) else {})
        unknown = set(value if isinstance(value, Mapping) else {}) - expected_keys
        raise DfpPhase1aCheckpointError(f"{label} has an unexpected schema (missing={sorted(missing)}, unknown={sorted(unknown)})")
    return value


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise DfpPhase1aCheckpointError(f"{label} must be an exact non-empty string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1aCheckpointError(f"{label} must be an exact boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise DfpPhase1aCheckpointError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise DfpPhase1aCheckpointError(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise DfpPhase1aCheckpointError(f"{label} must be at most {maximum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise DfpPhase1aCheckpointError(f"{label} must be a lowercase 64-hex-character SHA-256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 40 or any(c not in "0123456789abcdef" for c in token):
        raise DfpPhase1aCheckpointError(f"{label} must be a full lowercase 40-hex-character Git commit id")
    return token


def _require_string_list(value: Any, label: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise DfpPhase1aCheckpointError(f"{label} must be a list of exact strings")
    return value


def _require_int_tuple(value: Any, label: str) -> tuple[int, ...]:
    if type(value) is not list or any(type(item) is not int for item in value):
        raise DfpPhase1aCheckpointError(f"{label} must be a list of exact non-boolean integers")
    return tuple(value)


# ---------------------------------------------------------------------------
# Checkpoint structure/invariants.
# ---------------------------------------------------------------------------


def validate_checkpoint_structure(
    checkpoint: Mapping[str, Any], *, artifact_schema_identity_sha256: str, protocol_identity_sha256: str,
    subset_manifest_sha256: str, bridge_checkpoint_sha256: str, required_ancestor_commit: str,
    run_mode: str | None = None, image_count_expected: int | None = None, class_count: int | None = None,
    class_names_digest: str | None = None,
) -> None:
    """Sole authority for this checkpoint's schema/type/binding/
    completeness checks. Rejects a checkpoint bound to a different
    artifact-schema/protocol/subset-manifest/bridge-checkpoint identity,
    or (when supplied) a different run_mode/image_count/class_count/
    class_names_digest. Never mutates ``checkpoint``."""
    _require_closed_mapping(checkpoint, TOP_CHECKPOINT_KEYS, "checkpoint")

    if _require_exact_string(checkpoint["schema"], "checkpoint.schema") != CHECKPOINT_SCHEMA_NAME:
        raise DfpPhase1aCheckpointError(f"checkpoint.schema must be exactly {CHECKPOINT_SCHEMA_NAME!r}")
    if _require_int(checkpoint["schema_version"], "checkpoint.schema_version", minimum=1) != CHECKPOINT_SCHEMA_VERSION:
        raise DfpPhase1aCheckpointError(f"checkpoint.schema_version must be exactly {CHECKPOINT_SCHEMA_VERSION}")
    if _require_sha256(checkpoint["artifact_schema_identity_sha256"], "checkpoint.artifact_schema_identity_sha256") != artifact_schema_identity_sha256:
        raise DfpPhase1aCheckpointError("checkpoint.artifact_schema_identity_sha256 disagrees with the live artifact-schema identity")
    if _require_sha256(checkpoint["protocol_identity_sha256"], "checkpoint.protocol_identity_sha256") != protocol_identity_sha256:
        raise DfpPhase1aCheckpointError("checkpoint.protocol_identity_sha256 disagrees with the live protocol identity")
    recorded_subset_manifest_sha256 = _require_sha256(checkpoint["subset_manifest_sha256"], "checkpoint.subset_manifest_sha256")
    if recorded_subset_manifest_sha256 != subset_manifest_sha256:
        raise DfpPhase1aCheckpointError("checkpoint.subset_manifest_sha256 disagrees with the currently-supplied subset manifest")
    _require_sha256(checkpoint["selected_ids_digest"], "checkpoint.selected_ids_digest")
    if _require_sha256(checkpoint["bridge_checkpoint_sha256"], "checkpoint.bridge_checkpoint_sha256") != bridge_checkpoint_sha256:
        raise DfpPhase1aCheckpointError("checkpoint.bridge_checkpoint_sha256 disagrees with the live artifact-schema binding")
    if _require_git_identity(checkpoint["required_ancestor_commit"], "checkpoint.required_ancestor_commit") != required_ancestor_commit:
        raise DfpPhase1aCheckpointError("checkpoint.required_ancestor_commit disagrees with the live protocol identity")
    _require_git_identity(checkpoint["source_commit"], "checkpoint.source_commit")

    recorded_run_mode = _require_exact_string(checkpoint["run_mode"], "checkpoint.run_mode")
    if recorded_run_mode not in RUN_MODE_NAMES:
        raise DfpPhase1aCheckpointError(f"checkpoint.run_mode must be one of {list(RUN_MODE_NAMES)}")
    if run_mode is not None and recorded_run_mode != run_mode:
        raise DfpPhase1aCheckpointError("checkpoint.run_mode disagrees with the currently-supplied run mode")

    expected_count = _require_int(checkpoint["image_count_expected"], "checkpoint.image_count_expected", minimum=0)
    if image_count_expected is not None and expected_count != image_count_expected:
        raise DfpPhase1aCheckpointError("checkpoint.image_count_expected disagrees with the currently-supplied subset manifest")

    recorded_class_count = _require_int(checkpoint["class_count"], "checkpoint.class_count", minimum=1)
    if class_count is not None and recorded_class_count != class_count:
        raise DfpPhase1aCheckpointError("checkpoint.class_count disagrees with the currently-supplied class count")
    _require_sha256(checkpoint["class_names_digest"], "checkpoint.class_names_digest")
    if class_names_digest is not None and checkpoint["class_names_digest"] != class_names_digest:
        raise DfpPhase1aCheckpointError("checkpoint.class_names_digest disagrees with the currently-supplied class names digest")

    horizons = _require_int_tuple(checkpoint["horizon_order"], "checkpoint.horizon_order")
    if horizons != HORIZON_ORDER:
        raise DfpPhase1aCheckpointError(f"checkpoint.horizon_order must be exactly {list(HORIZON_ORDER)}")

    next_index = _require_int(checkpoint["next_dataset_index"], "checkpoint.next_dataset_index", minimum=0, maximum=expected_count)
    completed = _require_string_list(checkpoint["completed_image_ids"], "checkpoint.completed_image_ids")
    if len(set(completed)) != len(completed):
        raise DfpPhase1aCheckpointError("checkpoint.completed_image_ids contains a duplicate image ID")
    images_completed_count = _require_int(checkpoint["images_completed_count"], "checkpoint.images_completed_count", minimum=0)
    if images_completed_count != len(completed):
        raise DfpPhase1aCheckpointError("checkpoint.images_completed_count disagrees with len(completed_image_ids)")
    if next_index != len(completed):
        raise DfpPhase1aCheckpointError(
            f"checkpoint.next_dataset_index ({next_index}) must equal the number of completed images "
            f"({len(completed)}) -- refusing to resume a checkpoint that could skip or duplicate an image"
        )

    total_windows_processed = _require_int(checkpoint["total_windows_processed"], "checkpoint.total_windows_processed", minimum=0)
    if images_completed_count > 0 and total_windows_processed < images_completed_count:
        raise DfpPhase1aCheckpointError("checkpoint.total_windows_processed must be at least images_completed_count")

    try:
        require_safe_basename(checkpoint["per_image_stats_manifest_relative_path"], "checkpoint.per_image_stats_manifest_relative_path")
    except DfpPhase1aArtifactSchemaError as error:
        raise DfpPhase1aCheckpointError(str(error)) from error
    _require_sha256(checkpoint["per_image_stats_manifest_sha256"], "checkpoint.per_image_stats_manifest_sha256")

    complete = _require_bool(checkpoint["complete"], "checkpoint.complete")
    if complete and next_index != expected_count:
        raise DfpPhase1aCheckpointError("checkpoint.complete is true but next_dataset_index does not equal image_count_expected")

    final_result_produced = _require_bool(checkpoint["final_result_produced"], "checkpoint.final_result_produced")
    if final_result_produced and not complete:
        raise DfpPhase1aCheckpointError("checkpoint.final_result_produced is true but checkpoint.complete is false -- an incomplete checkpoint may never carry a final-result flag")

    _require_exact_string(checkpoint["generation_nonce"], "checkpoint.generation_nonce")
    _require_exact_string(checkpoint["created_at_utc"], "checkpoint.created_at_utc")
    _require_exact_string(checkpoint["updated_at_utc"], "checkpoint.updated_at_utc")


def validate_checkpoint_against_canonical_order(
    checkpoint: Mapping[str, Any], selected_ids: Sequence[str], *, selected_ids_digest: str,
) -> None:
    """Bind the checkpoint's completed-ID prefix to the frozen manifest's
    own canonical selected-ID order. Requires no skip/reorder/duplicate
    relative to that canonical order."""
    if checkpoint["selected_ids_digest"] != selected_ids_digest:
        raise DfpPhase1aCheckpointError("checkpoint.selected_ids_digest disagrees with the freshly-recomputed subset-manifest digest")
    next_index = checkpoint["next_dataset_index"]
    expected_prefix = list(selected_ids[:next_index])
    if list(checkpoint["completed_image_ids"]) != expected_prefix:
        raise DfpPhase1aCheckpointError(
            "checkpoint.completed_image_ids does not match the canonical selected-ID prefix for the first "
            f"{next_index} images; refusing to resume against a mismatched image order"
        )


def resume_dataset_index(checkpoint: Mapping[str, Any]) -> int:
    if checkpoint["complete"] is True:
        raise DfpPhase1aCheckpointError("checkpoint is already complete; refusing to resume it as though incomplete")
    return checkpoint["next_dataset_index"]


def validate_checkpoint_against_per_image_stats(checkpoint: Mapping[str, Any], stats_manifest: Mapping[str, Any]) -> None:
    """Bind a per-image-stats manifest to an already-structurally-validated
    checkpoint (validate_checkpoint_structure has already bound the
    checkpoint to the artifact-schema/protocol/subset-manifest/run-mode/
    class bindings) -- this only needs to bind the STATS artifact to the
    CHECKPOINT, transitively inheriting every other binding. CPU-only;
    must run before any dataset/model/CUDA work. Never mutates either
    argument."""
    if stats_manifest["schema"] != PER_IMAGE_STATS_SCHEMA_NAME:
        raise DfpPhase1aCheckpointError(f"per-image-stats manifest schema disagrees with the authoritative {PER_IMAGE_STATS_SCHEMA_NAME!r}")
    if stats_manifest["schema_version"] != PER_IMAGE_STATS_SCHEMA_VERSION:
        raise DfpPhase1aCheckpointError("per-image-stats manifest schema_version disagrees with the authoritative version")
    if stats_manifest["run_mode"] != checkpoint["run_mode"]:
        raise DfpPhase1aCheckpointError("per-image-stats manifest run_mode disagrees with the checkpoint")
    if stats_manifest["class_count"] != checkpoint["class_count"]:
        raise DfpPhase1aCheckpointError("per-image-stats manifest class_count disagrees with the checkpoint")
    if stats_manifest["class_names_digest"] != checkpoint["class_names_digest"]:
        raise DfpPhase1aCheckpointError("per-image-stats manifest class_names_digest disagrees with the checkpoint")

    next_index = checkpoint["next_dataset_index"]
    if stats_manifest["image_count"] != next_index:
        raise DfpPhase1aCheckpointError(
            f"per-image-stats manifest image_count ({stats_manifest['image_count']}) disagrees with "
            f"checkpoint.next_dataset_index ({next_index})"
        )
    if list(stats_manifest["image_ids"]) != list(checkpoint["completed_image_ids"]):
        raise DfpPhase1aCheckpointError("per-image-stats manifest image_ids disagrees with checkpoint.completed_image_ids")


def validate_per_image_stats_manifest_structure(manifest: Mapping[str, Any]) -> None:
    _require_closed_mapping(manifest, PER_IMAGE_STATS_MANIFEST_KEYS, "per-image-stats manifest")
    if _require_exact_string(manifest["schema"], "per-image-stats manifest.schema") != PER_IMAGE_STATS_SCHEMA_NAME:
        raise DfpPhase1aCheckpointError(f"per-image-stats manifest.schema must be exactly {PER_IMAGE_STATS_SCHEMA_NAME!r}")
    if _require_int(manifest["schema_version"], "per-image-stats manifest.schema_version", minimum=1) != PER_IMAGE_STATS_SCHEMA_VERSION:
        raise DfpPhase1aCheckpointError(f"per-image-stats manifest.schema_version must be exactly {PER_IMAGE_STATS_SCHEMA_VERSION}")
    try:
        require_safe_basename(manifest["npz_filename"], "per-image-stats manifest.npz_filename")
    except DfpPhase1aArtifactSchemaError as error:
        raise DfpPhase1aCheckpointError(str(error)) from error
    _require_sha256(manifest["npz_sha256"], "per-image-stats manifest.npz_sha256")
    run_mode = _require_exact_string(manifest["run_mode"], "per-image-stats manifest.run_mode")
    if run_mode not in RUN_MODE_NAMES:
        raise DfpPhase1aCheckpointError(f"per-image-stats manifest.run_mode must be one of {list(RUN_MODE_NAMES)}")
    _require_int(manifest["class_count"], "per-image-stats manifest.class_count", minimum=1)
    _require_sha256(manifest["class_names_digest"], "per-image-stats manifest.class_names_digest")
    image_count = _require_int(manifest["image_count"], "per-image-stats manifest.image_count", minimum=0)
    image_ids = _require_string_list(manifest["image_ids"], "per-image-stats manifest.image_ids")
    if len(image_ids) != image_count:
        raise DfpPhase1aCheckpointError("per-image-stats manifest.image_ids length disagrees with image_count")
    if len(set(image_ids)) != len(image_ids):
        raise DfpPhase1aCheckpointError("per-image-stats manifest.image_ids contains a duplicate image ID")


# ---------------------------------------------------------------------------
# Per-image compute-projection-statistics NPZ: exact closed member schema.
# ---------------------------------------------------------------------------

_NATIVE_BYTEORDERS = ("=", "<")  # this platform (x86_64) is little-endian; '=' is native, '|' n/a for int/float


def validate_per_image_stats_array(name: str, array: Any, *, image_count: int) -> None:
    """Sole authority for the per-image-stats NPZ array dtype/rank/shape
    contract. Invoked identically by the atomic writer (before
    serialization) and by the reader (after loading) so the two can never
    drift apart. Never casts, reshapes, squeezes, truncates, or pads an
    array that fails the contract -- always rejected, never coerced."""
    if not isinstance(array, np.ndarray):
        raise DfpPhase1aCheckpointError(f"per-image-stats array {name!r} must be a numpy.ndarray")
    if name not in ALL_NPZ_MEMBER_NAMES:
        raise DfpPhase1aCheckpointError(f"per-image-stats array {name!r} is not part of the authoritative schema")
    if array.dtype.byteorder == ">":
        raise DfpPhase1aCheckpointError(f"per-image-stats array {name!r} must use native (non-big-endian) byte order")

    if name in INT64_MEMBER_NAMES:
        if array.dtype != np.dtype(np.int64):
            raise DfpPhase1aCheckpointError(f"per-image-stats array {name!r} must have exact dtype int64, got {array.dtype}")
    elif name in FLOAT64_MEMBER_NAMES:
        if array.dtype != np.dtype(np.float64):
            raise DfpPhase1aCheckpointError(f"per-image-stats array {name!r} must have exact dtype float64, got {array.dtype}")
        if array.size and not np.isfinite(array).all():
            raise DfpPhase1aCheckpointError(f"per-image-stats array {name!r} contains a non-finite value (NaN or Infinity)")

    if array.ndim != 1:
        raise DfpPhase1aCheckpointError(f"per-image-stats array {name!r} must be exactly 1-dimensional, got ndim={array.ndim}")
    if array.shape[0] != image_count:
        raise DfpPhase1aCheckpointError(
            f"per-image-stats array {name!r} has {array.shape[0]} rows, expected exactly image_count={image_count}"
        )
    if name in INT64_MEMBER_NAMES and array.size and array.min() < 0:
        raise DfpPhase1aCheckpointError(f"per-image-stats array {name!r} must be entirely non-negative")


def _validate_percentile_monotonicity(arrays: Mapping[str, np.ndarray]) -> None:
    """Cheap, mathematically-guaranteed content sanity check: percentiles
    of |Q_w| are monotonically non-decreasing by construction
    (median <= p90 <= p95 <= p99 <= maximum). A violation indicates
    corrupted or mismatched content, independent of dtype/shape checks."""
    order = ("q_w_median", "q_w_p90", "q_w_p95", "q_w_p99", "q_w_maximum")
    if not all(name in arrays for name in order):
        return
    stacked = np.stack([arrays[name] for name in order], axis=0)
    if stacked.shape[1] == 0:
        return
    if not np.all(np.diff(stacked, axis=0) >= 0):
        raise DfpPhase1aCheckpointError("per-image-stats percentile arrays violate required monotonicity (median <= p90 <= p95 <= p99 <= maximum)")
    if arrays["dispute_pixel_count"].size and np.any(arrays["dispute_pixel_count"] > arrays["evaluated_pixel_count"]):
        raise DfpPhase1aCheckpointError("per-image-stats dispute_pixel_count exceeds evaluated_pixel_count for at least one image")
    if arrays["zero_dispute_window_count"].size and np.any(arrays["zero_dispute_window_count"] > arrays["window_count"]):
        raise DfpPhase1aCheckpointError("per-image-stats zero_dispute_window_count exceeds window_count for at least one image")


def _check_zip_member_hygiene(npz_path: Path) -> None:
    """Inspect the NPZ as a raw ZIP container before trusting numpy's own
    loader: reject duplicate member names, directory entries, and
    path-traversal member names. ``numpy.load`` silently tolerates a
    duplicate ZIP member (returning only the last one), so this check
    must run independently first."""
    try:
        with zipfile.ZipFile(npz_path) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names):
                raise DfpPhase1aCheckpointError("per-image-stats NPZ contains a duplicate ZIP member name")
            for name in names:
                if name.endswith("/"):
                    raise DfpPhase1aCheckpointError(f"per-image-stats NPZ contains a directory entry {name!r}")
                if ".." in Path(name).parts or Path(name).is_absolute():
                    raise DfpPhase1aCheckpointError(f"per-image-stats NPZ member name {name!r} is unsafe (path traversal)")
    except zipfile.BadZipFile as error:
        raise DfpPhase1aCheckpointError(f"per-image-stats NPZ {npz_path} is not a valid ZIP archive: {error}") from error


def load_per_image_stats_npz(npz_path: Path, *, expected_sha256: str, image_count: int) -> dict[str, np.ndarray]:
    """Load and fully validate every member of the per-image-stats NPZ.
    Forces every member to be loaded and validated (never assumes reading
    ``data.files`` alone validates malformed or object-dtype members).
    Rejects a missing/inaccessible file, digest mismatch, malformed/
    truncated archive, an unexpected/missing/duplicate member, an
    object-dtype or structured-dtype array, and any dtype/rank/shape
    disagreement -- always as a clean :class:`DfpPhase1aCheckpointError`,
    never a raw traceback."""
    npz_path = Path(npz_path)
    try:
        observed_sha256 = sha256_file(npz_path)
    except DfpPhase1aArtifactSchemaError as error:
        raise DfpPhase1aCheckpointError(str(error)) from error
    if observed_sha256 != expected_sha256:
        raise DfpPhase1aCheckpointError(f"per-image-stats NPZ {npz_path} SHA256 mismatch: expected {expected_sha256}, observed {observed_sha256}")

    _check_zip_member_hygiene(npz_path)

    try:
        with np.load(npz_path, allow_pickle=False) as data:
            observed_names = frozenset(data.files)
            if observed_names != ALL_NPZ_MEMBER_NAMES:
                missing = sorted(ALL_NPZ_MEMBER_NAMES - observed_names)
                unexpected = sorted(observed_names - ALL_NPZ_MEMBER_NAMES)
                raise DfpPhase1aCheckpointError(f"per-image-stats NPZ member set disagrees with the authoritative schema (missing={missing}, unexpected={unexpected})")
            arrays = {key: data[key] for key in data.files}
    except DfpPhase1aCheckpointError:
        raise
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        raise DfpPhase1aCheckpointError(f"cannot load per-image-stats NPZ {npz_path}: {error}") from error

    for name, array in arrays.items():
        validate_per_image_stats_array(name, array, image_count=image_count)
    if not np.array_equal(arrays["dataset_indices"], np.arange(image_count, dtype=np.int64)):
        raise DfpPhase1aCheckpointError("per-image-stats NPZ dataset_indices is not the exact canonical 0..image_count-1 sequence")
    _validate_percentile_monotonicity(arrays)
    return arrays


def build_zero_row_arrays() -> dict[str, np.ndarray]:
    """Zero-image checkpoint support: every member as a correctly-shaped,
    correctly-typed zero-row array -- never via ``np.stack([])`` (which
    raises on an empty sequence)."""
    arrays: dict[str, np.ndarray] = {}
    for name in INT64_MEMBER_NAMES:
        arrays[name] = np.zeros((0,), dtype=np.int64)
    for name in FLOAT64_MEMBER_NAMES:
        arrays[name] = np.zeros((0,), dtype=np.float64)
    return arrays


# ---------------------------------------------------------------------------
# Ordered Phase-A resume validation -- every step here runs before dataset
# construction, the first dataset sample, torch import (where avoidable),
# model construction, checkpoint weight loading, CUDA availability checks,
# and .cuda()/.to("cuda").
# ---------------------------------------------------------------------------


def validate_resume_state(*, checkpoints_dir: Path, checkpoint_filename: str, repo_root: Path) -> dict[str, Any]:
    """Run the full ordered Phase-A validation: (1) artifact-schema
    identity, (2) scientific protocol identity, (3) frozen subset
    manifest, (4) checkpoint JSON structure, (5) canonical completed-ID
    prefix, (6) per-image manifest, (7) NPZ existence/SHA-256, (8) NPZ
    closed member schema, (9) array shape/dtype, (10) checkpoint/array
    reconciliation, (11) complete/resumable state. Returns a summary dict
    on success; raises :class:`DfpPhase1aCheckpointError` (chained from
    any dependency error) on the first violation. CPU-only: no dataset,
    model, or CUDA entry point is reachable from this function."""
    import src.dfp_phase1a_artifact_schema as artifact_schema

    repo_root = Path(repo_root)
    checkpoints_dir = Path(checkpoints_dir)

    # 1. artifact-schema identity
    try:
        schema_summary = artifact_schema.validate_static_configuration(repo_root=repo_root)
        artifact_identity = artifact_schema.load_identity(repo_root=repo_root)
    except DfpPhase1aArtifactSchemaError as error:
        raise DfpPhase1aCheckpointError(f"artifact-schema identity validation failed: {error}") from error

    # 2. scientific protocol identity (already loaded/validated transitively
    # by validate_static_configuration above; re-confirmed here explicitly)
    from src.dfp_phase1a_identity import DfpPhase1aIdentityError
    from src.dfp_phase1a_identity import load_identity as load_protocol_identity

    try:
        protocol = load_protocol_identity(repo_root / artifact_identity["protocol_binding"]["protocol_identity_relative_path"], repo_root=repo_root)
    except DfpPhase1aIdentityError as error:
        raise DfpPhase1aCheckpointError(f"scientific protocol identity validation failed: {error}") from error

    checkpoint_path = checkpoints_dir / checkpoint_filename
    try:
        checkpoint = parse_strict_json_document(checkpoint_path, label="checkpoint")
    except DfpPhase1aArtifactSchemaError as error:
        raise DfpPhase1aCheckpointError(str(error)) from error

    run_mode = checkpoint.get("run_mode") if isinstance(checkpoint, Mapping) else None
    if run_mode not in RUN_MODE_NAMES:
        raise DfpPhase1aCheckpointError(f"checkpoint.run_mode must be one of {list(RUN_MODE_NAMES)}")
    manifest_relative_path = (
        artifact_identity["subset_manifest_binding"]["coco_stuff_manifest_relative_path"]
        if run_mode == "coco_stuff_500"
        else artifact_identity["subset_manifest_binding"]["ade20k_manifest_relative_path"]
    )
    manifest_sha256_key = "coco_stuff_manifest_sha256" if run_mode == "coco_stuff_500" else "ade20k_manifest_sha256"
    class_count_key = "coco_stuff_class_count" if run_mode == "coco_stuff_500" else "ade20k_class_count"
    class_names_digest_key = "coco_stuff_class_names_digest" if run_mode == "coco_stuff_500" else "ade20k_class_names_digest"

    # 3. frozen subset manifest
    try:
        subset_manifest = parse_strict_json_document(repo_root / manifest_relative_path, label="subset manifest")
    except DfpPhase1aArtifactSchemaError as error:
        raise DfpPhase1aCheckpointError(str(error)) from error

    # 4. checkpoint JSON structure
    validate_checkpoint_structure(
        checkpoint,
        artifact_schema_identity_sha256=artifact_schema.sha256_file(repo_root / artifact_schema.IDENTITY_RELATIVE_PATH),
        protocol_identity_sha256=schema_summary["protocol_identity_sha256"],
        subset_manifest_sha256=schema_summary[manifest_sha256_key],
        bridge_checkpoint_sha256=schema_summary["bridge_checkpoint_sha256"],
        required_ancestor_commit=protocol["identity"]["required_ancestor_commit"],
        run_mode=run_mode,
        image_count_expected=len(subset_manifest["selected_ids"]),
        class_count=artifact_identity["dataset_binding"][class_count_key],
        class_names_digest=artifact_identity["dataset_binding"][class_names_digest_key],
    )

    # 5. canonical completed-ID prefix
    validate_checkpoint_against_canonical_order(
        checkpoint, subset_manifest["selected_ids"], selected_ids_digest=subset_manifest["selected_ids_digest"],
    )

    # 6. per-image manifest
    stats_manifest_path = checkpoints_dir / checkpoint["per_image_stats_manifest_relative_path"]
    try:
        stats_manifest = parse_strict_json_document(stats_manifest_path, label="per-image-stats manifest")
    except DfpPhase1aArtifactSchemaError as error:
        raise DfpPhase1aCheckpointError(str(error)) from error
    validate_per_image_stats_manifest_structure(stats_manifest)
    observed_manifest_sha256 = sha256_file(stats_manifest_path)
    if observed_manifest_sha256 != checkpoint["per_image_stats_manifest_sha256"]:
        raise DfpPhase1aCheckpointError("per-image-stats manifest SHA256 mismatch against the checkpoint's binding")
    validate_checkpoint_against_per_image_stats(checkpoint, stats_manifest)

    # 7. NPZ existence and SHA-256; 8. NPZ closed member schema; 9. array
    # shape/dtype; 10. checkpoint/array reconciliation (image_count already
    # bound to checkpoint.next_dataset_index in step 6)
    npz_path = checkpoints_dir / stats_manifest["npz_filename"]
    arrays = load_per_image_stats_npz(npz_path, expected_sha256=stats_manifest["npz_sha256"], image_count=stats_manifest["image_count"])

    # 11. complete/resumable state
    if checkpoint["complete"]:
        resumable = False
        next_index = checkpoint["next_dataset_index"]
    else:
        next_index = resume_dataset_index(checkpoint)
        resumable = True

    return {
        "checkpoint": checkpoint,
        "stats_manifest": stats_manifest,
        "arrays": arrays,
        "next_dataset_index": next_index,
        "resumable": resumable,
        "run_mode": run_mode,
    }


# ---------------------------------------------------------------------------
# Atomic generation-and-install: build a new per-image-stats NPZ +
# manifest under unpredictable names, self-verify the complete candidate,
# then atomically install the checkpoint pointer last. Preserves the
# previous valid checkpoint until the new candidate is fully verified;
# cleans up orphaned generation files on any failure; never mutates the
# caller's input checkpoint fields.
# ---------------------------------------------------------------------------


def generate_and_install_checkpoint(
    *, checkpoints_dir: Path, checkpoint_filename: str, checkpoint_fields: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray], image_ids: Sequence[str],
) -> Path:
    """Write a new per-image-stats NPZ + manifest generation under
    unpredictable task-specific names, self-verify the complete candidate
    (including a full round-trip re-load), then atomically install the
    checkpoint document referencing it last. ``checkpoint_fields`` must
    contain every :data:`TOP_CHECKPOINT_KEYS` field EXCEPT
    ``per_image_stats_manifest_relative_path``, ``per_image_stats_manifest_sha256``,
    and ``generation_nonce`` (filled in here). Returns the installed
    checkpoint path. On any PRE-replacement failure (including a failure
    of the final checkpoint-pointer write itself, before its own
    ``os.replace`` durably succeeds), leaves the previous valid
    checkpoint (and its own referenced generation files) byte-identical
    and untouched, with no orphaned temp/generation file left behind. On
    the narrow POST-replacement failure case (the pointer write's own
    ``os.replace`` already durably succeeded, but something in its
    post-replace bookkeeping still raised), reconciles by strictly
    re-reading the installed checkpoint's ``generation_nonce`` rather
    than inferring installation from file existence: if it already
    matches this attempt's own nonce, the new generation is left fully
    installed and referenced (never deleted); otherwise -- or if
    reconciliation itself cannot read the installed checkpoint -- the
    new generation is conservatively either removed (proven not
    installed) or preserved (state indeterminate), and the ORIGINAL
    exception always propagates unchanged, never masked by a
    reconciliation failure. Never removes a pre-existing/superseded
    generation in any case."""
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / checkpoint_filename

    image_count = len(image_ids)
    for name, array in arrays.items():
        validate_per_image_stats_array(name, array, image_count=image_count)
    if set(arrays) != ALL_NPZ_MEMBER_NAMES:
        missing = ALL_NPZ_MEMBER_NAMES - set(arrays)
        unexpected = set(arrays) - ALL_NPZ_MEMBER_NAMES
        raise DfpPhase1aCheckpointError(f"generation arrays disagree with the authoritative schema (missing={sorted(missing)}, unexpected={sorted(unexpected)})")
    _validate_percentile_monotonicity(dict(arrays))

    next_index = checkpoint_fields["next_dataset_index"]
    run_mode = checkpoint_fields["run_mode"]
    generation_nonce = secrets.token_hex(16)
    stem = f"dfp_phase1a_per_image_stats_{run_mode}_{next_index:06d}_{generation_nonce}"
    npz_filename = f"{stem}.npz"
    manifest_filename = f"{stem}.json"
    npz_path = checkpoints_dir / npz_filename
    manifest_path = checkpoints_dir / manifest_filename

    generated_paths: list[Path] = []
    pointer_install_attempted = False
    try:
        temp_npz = checkpoints_dir / f".tmp-{generation_nonce}.npz"
        np.savez(temp_npz, allow_pickle=False, **{name: arrays[name] for name in ALL_NPZ_MEMBER_NAMES})
        written = temp_npz if temp_npz.suffix == ".npz" else temp_npz.with_suffix(temp_npz.suffix + ".npz")
        # Track the temp file itself until os.replace consumes it: if
        # os.replace raises, `written` still exists on disk and must be
        # cleaned up like any other orphan from this attempt.
        generated_paths.append(written)
        import os

        os.replace(written, npz_path)
        generated_paths.remove(written)
        generated_paths.append(npz_path)

        # Self-verify: re-hash and fully re-load the just-installed NPZ.
        npz_sha256 = sha256_file(npz_path)
        load_per_image_stats_npz(npz_path, expected_sha256=npz_sha256, image_count=image_count)

        stats_manifest = {
            "schema": PER_IMAGE_STATS_SCHEMA_NAME, "schema_version": PER_IMAGE_STATS_SCHEMA_VERSION,
            "npz_filename": npz_filename, "npz_sha256": npz_sha256, "run_mode": run_mode,
            "class_count": checkpoint_fields["class_count"], "class_names_digest": checkpoint_fields["class_names_digest"],
            "image_count": image_count, "image_ids": list(image_ids),
        }
        validate_per_image_stats_manifest_structure(stats_manifest)
        write_json_atomically(manifest_path, stats_manifest, label="per-image-stats manifest")
        generated_paths.append(manifest_path)
        manifest_sha256 = sha256_file(manifest_path)

        new_checkpoint = dict(checkpoint_fields)
        new_checkpoint["per_image_stats_manifest_relative_path"] = manifest_filename
        new_checkpoint["per_image_stats_manifest_sha256"] = manifest_sha256
        new_checkpoint["generation_nonce"] = generation_nonce
        _require_closed_mapping(new_checkpoint, TOP_CHECKPOINT_KEYS, "generated checkpoint")

        # Self-verify the complete candidate against the exact same
        # validators a reader/resumer would use, BEFORE touching the
        # installed checkpoint pointer.
        validate_checkpoint_structure(
            new_checkpoint,
            artifact_schema_identity_sha256=new_checkpoint["artifact_schema_identity_sha256"],
            protocol_identity_sha256=new_checkpoint["protocol_identity_sha256"],
            subset_manifest_sha256=new_checkpoint["subset_manifest_sha256"],
            bridge_checkpoint_sha256=new_checkpoint["bridge_checkpoint_sha256"],
            required_ancestor_commit=new_checkpoint["required_ancestor_commit"],
        )
        validate_checkpoint_against_per_image_stats(new_checkpoint, stats_manifest)

        # Only now: atomically install the checkpoint pointer last, INSIDE
        # this same cleanup scope. write_json_atomically's own os.replace
        # is atomic and the last statement able to affect `path` under
        # normal operation, so a failure BEFORE it means `path` was not
        # replaced. However, its own post-replace bookkeeping (stat()-ing
        # /unlinking its already-consumed temp file) could in principle
        # still raise (e.g. MemoryError) even though the replace already
        # durably succeeded -- so a raised exception here does NOT, by
        # itself, prove the pointer was not installed. The except clause
        # below reconciles this explicitly rather than assuming either
        # outcome.
        pointer_install_attempted = True
        write_json_atomically(checkpoint_path, new_checkpoint, label="checkpoint")
    except Exception as original_error:
        if not pointer_install_attempted:
            # Pre-replacement failure: the checkpoint pointer write was
            # never even attempted, so it cannot have taken effect. The
            # current attempt's own new generation files -- and only
            # those -- are safely unreferenced orphans. Never remove a
            # pre-existing file: generated_paths contains exclusively
            # paths created earlier in *this* call.
            for path in generated_paths:
                path.unlink(missing_ok=True)
            raise

        # Pointer installation was attempted and then raised. Never infer
        # whether it took effect from file existence alone: strictly
        # re-read the installed checkpoint and compare its
        # generation_nonce to this attempt's own. Any failure while
        # reconciling (the checkpoint is unreadable, missing, malformed,
        # etc.) must not replace or mask the ORIGINAL exception, and must
        # not cause the new generation to be deleted -- referential
        # integrity takes priority over orphan cleanup whenever the
        # post-replacement state cannot be proven.
        try:
            if checkpoint_path.exists():
                installed_checkpoint = parse_strict_json_document(checkpoint_path, label="checkpoint")
                installed_nonce = installed_checkpoint.get("generation_nonce")
            else:
                # The destination does not exist at all -- this is sound,
                # unambiguous proof of non-installation (a successful
                # os.replace always leaves its destination present), NOT
                # a forbidden "existence implies installation" inference:
                # only ABSENCE is being used as a positive signal here.
                installed_nonce = None
        except Exception:
            # The destination exists but its installed state cannot be
            # proven (unreadable/malformed/permission error/etc.):
            # conservatively preserve the new generation rather than risk
            # deleting a file the (possibly successfully installed)
            # checkpoint now references. Re-raise the ORIGINAL exception,
            # never the reconciliation failure.
            raise original_error

        if installed_nonce == generation_nonce:
            # The pointer WAS durably installed referencing this
            # attempt's new generation before the exception fired. Do
            # not delete or otherwise modify anything -- the installed
            # checkpoint and its generation are valid and fully
            # referenced. Only the original exception propagates.
            raise original_error

        # The installed checkpoint still references the old/different
        # generation (or the field is absent/unrecognized) -- this
        # attempt's pointer write did not take effect. Clean up only
        # this attempt's own new files, never a pre-existing one.
        for path in generated_paths:
            path.unlink(missing_ok=True)
        raise original_error

    return checkpoint_path


__all__ = [
    "DfpPhase1aCheckpointError",
    "TOP_CHECKPOINT_KEYS",
    "PER_IMAGE_STATS_MANIFEST_KEYS",
    "validate_checkpoint_structure",
    "validate_checkpoint_against_canonical_order",
    "resume_dataset_index",
    "validate_checkpoint_against_per_image_stats",
    "validate_per_image_stats_manifest_structure",
    "validate_per_image_stats_array",
    "load_per_image_stats_npz",
    "build_zero_row_arrays",
    "validate_resume_state",
    "generate_and_install_checkpoint",
]
