"""Final-result document schema/validation and atomic staging-directory
installation for DFP Phase 1A diagnostic runs.

A result may be labeled ``complete=true`` / ``final=true`` only if the
bound checkpoint is itself complete at the exact expected image count,
its completed IDs equal the complete frozen selected-ID list, every
referenced artifact is present and hash-verified, and no unresolved
partial/temporary state remains. This module implements no scientific
metric, gate outcome, or oracle definition -- see
``evaluation_identities/e13_dfp_phase1a_artifacts.toml`` ``[result_schema]``
for the explicit list of what this step's result document excludes and
why (no Phase 1A diagnostic driver exists yet to compute GT-dependent
metrics or gate outcomes)."""

from __future__ import annotations

import os
import secrets
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.dfp_phase1a_artifact_schema import (
    RESULT_SCHEMA_NAME,
    RESULT_SCHEMA_VERSION,
    RUN_MODE_NAMES,
    DfpPhase1aArtifactSchemaError,
    parse_strict_json_document,
    require_safe_basename,
    sha256_file,
    write_json_atomically,
)
from src.dfp_phase1a_checkpoint import (
    DfpPhase1aCheckpointError,
    load_per_image_stats_npz,
    validate_checkpoint_against_per_image_stats,
    validate_checkpoint_structure,
    validate_per_image_stats_manifest_structure,
)


class DfpPhase1aReportError(ValueError):
    """Raised on any final-result document or bundle-installation
    invariant violation. Always fail closed."""


TOP_RESULT_KEYS = frozenset(
    {
        "schema", "schema_version", "artifact_schema_identity_sha256", "protocol_identity_sha256",
        "subset_manifest_sha256", "selected_ids_digest", "run_mode", "image_count_expected",
        "checkpoint_sha256", "per_image_stats_manifest_sha256", "complete", "final", "created_at_utc",
    }
)

RESULT_BUNDLE_MEMBER_NAMES = ("result.json", "checkpoint.json")  # plus the per-image-stats manifest + NPZ, named per their own generation


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        missing = expected_keys - set(value if isinstance(value, Mapping) else {})
        unknown = set(value if isinstance(value, Mapping) else {}) - expected_keys
        raise DfpPhase1aReportError(f"{label} has an unexpected schema (missing={sorted(missing)}, unknown={sorted(unknown)})")
    return value


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise DfpPhase1aReportError(f"{label} must be an exact non-empty string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1aReportError(f"{label} must be an exact boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise DfpPhase1aReportError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise DfpPhase1aReportError(f"{label} must be at least {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise DfpPhase1aReportError(f"{label} must be a lowercase 64-hex-character SHA-256")
    return token


def _checkpoint_authority_field(checkpoint: Any, key: str) -> Any:
    """Best-effort extraction of one of ``checkpoint``'s own fields, used
    ONLY as a fallback authority for :func:`validate_checkpoint_structure`
    below when ``result`` carries no corresponding field of its own --
    safe even when ``checkpoint`` is not yet known to be a well-formed
    mapping, since ``validate_checkpoint_structure``'s own closed-mapping
    check (its first statement) rejects any malformed checkpoint --
    missing, renamed, extra, or wrong-typed top-level key -- before ever
    comparing an authority value against anything. Never raises
    ``KeyError``; never mutates ``checkpoint``."""
    return checkpoint.get(key) if isinstance(checkpoint, Mapping) else None


def validate_result_structure(result: Mapping[str, Any], *, checkpoint: Mapping[str, Any], checkpoint_sha256: str) -> None:
    """Sole authority for the final-result document's schema/binding/
    completeness checks. A result may declare ``complete=true`` /
    ``final=true`` only if the bound checkpoint is itself complete at the
    exact expected image count and its completed IDs equal the checkpoint's
    own recorded ``image_count_expected``-length selected-ID list. Never
    mutates either argument."""
    _require_closed_mapping(result, TOP_RESULT_KEYS, "result")

    if _require_exact_string(result["schema"], "result.schema") != RESULT_SCHEMA_NAME:
        raise DfpPhase1aReportError(f"result.schema must be exactly {RESULT_SCHEMA_NAME!r}")
    if _require_int(result["schema_version"], "result.schema_version", minimum=1) != RESULT_SCHEMA_VERSION:
        raise DfpPhase1aReportError(f"result.schema_version must be exactly {RESULT_SCHEMA_VERSION}")

    # The embedded checkpoint must receive the SAME structural validation
    # a standalone checkpoint gets -- closed key set, required fields,
    # unknown-field rejection, exact types, SHA/Git-identity formats,
    # counts, horizons, class bindings, state flags -- before the first
    # direct `checkpoint[...]` access below. This function never reloads
    # a live identity file or an NPZ: the relational authorities
    # "available at this layer" are `result`'s own already-closed-key-set
    # -checked fields (the closed-mapping check just above guarantees
    # every one of these keys exists, so plain indexing is safe even
    # though their individual values are not yet format-checked) for
    # every authority `result` itself carries (artifact-schema/protocol/
    # subset-manifest identity, run_mode, image_count_expected); for the
    # two authorities `result`'s own schema never carries at all
    # (bridge_checkpoint_sha256, required_ancestor_commit -- Step 10
    # deliberately excludes these from the result document), the
    # checkpoint's own field is the only authority available at this
    # layer, used purely to exercise the structural/format checks for
    # those two. Agreement between `result` and `checkpoint`'s
    # corresponding fields remains the unchanged, separate relational
    # check performed explicitly below (now redundant-but-harmless for
    # the fields already cross-checked here).
    try:
        validate_checkpoint_structure(
            checkpoint,
            artifact_schema_identity_sha256=result["artifact_schema_identity_sha256"],
            protocol_identity_sha256=result["protocol_identity_sha256"],
            subset_manifest_sha256=result["subset_manifest_sha256"],
            bridge_checkpoint_sha256=_checkpoint_authority_field(checkpoint, "bridge_checkpoint_sha256"),
            required_ancestor_commit=_checkpoint_authority_field(checkpoint, "required_ancestor_commit"),
            run_mode=result["run_mode"] if type(result["run_mode"]) is str else None,
            image_count_expected=result["image_count_expected"] if type(result["image_count_expected"]) is int and type(result["image_count_expected"]) is not bool else None,
        )
    except DfpPhase1aCheckpointError as error:
        raise DfpPhase1aReportError(f"embedded checkpoint is invalid: {error}") from error

    _require_sha256(result["artifact_schema_identity_sha256"], "result.artifact_schema_identity_sha256")
    if result["artifact_schema_identity_sha256"] != checkpoint["artifact_schema_identity_sha256"]:
        raise DfpPhase1aReportError("result.artifact_schema_identity_sha256 disagrees with the bound checkpoint")
    if result["protocol_identity_sha256"] != checkpoint["protocol_identity_sha256"]:
        raise DfpPhase1aReportError("result.protocol_identity_sha256 disagrees with the bound checkpoint")
    if result["subset_manifest_sha256"] != checkpoint["subset_manifest_sha256"]:
        raise DfpPhase1aReportError("result.subset_manifest_sha256 disagrees with the bound checkpoint")
    if result["selected_ids_digest"] != checkpoint["selected_ids_digest"]:
        raise DfpPhase1aReportError("result.selected_ids_digest disagrees with the bound checkpoint")

    run_mode = _require_exact_string(result["run_mode"], "result.run_mode")
    if run_mode not in RUN_MODE_NAMES:
        raise DfpPhase1aReportError(f"result.run_mode must be one of {list(RUN_MODE_NAMES)}")
    if run_mode != checkpoint["run_mode"]:
        raise DfpPhase1aReportError("result.run_mode disagrees with the bound checkpoint -- a pilot/subset result must never be labeled as another dataset/run")

    expected_count = _require_int(result["image_count_expected"], "result.image_count_expected", minimum=0)
    if expected_count != checkpoint["image_count_expected"]:
        raise DfpPhase1aReportError("result.image_count_expected disagrees with the bound checkpoint")

    if _require_sha256(result["checkpoint_sha256"], "result.checkpoint_sha256") != checkpoint_sha256:
        raise DfpPhase1aReportError("result.checkpoint_sha256 does not match the freshly-hashed checkpoint document")
    if result["per_image_stats_manifest_sha256"] != checkpoint["per_image_stats_manifest_sha256"]:
        raise DfpPhase1aReportError("result.per_image_stats_manifest_sha256 disagrees with the bound checkpoint")

    complete = _require_bool(result["complete"], "result.complete")
    final = _require_bool(result["final"], "result.final")
    if complete or final:
        if not checkpoint["complete"]:
            raise DfpPhase1aReportError("result declares complete/final but the bound checkpoint is not itself complete")
        if checkpoint["next_dataset_index"] != checkpoint["image_count_expected"]:
            raise DfpPhase1aReportError("result declares complete/final but the bound checkpoint has not reached the exact expected image count")
        if len(checkpoint["completed_image_ids"]) != checkpoint["image_count_expected"]:
            raise DfpPhase1aReportError("result declares complete/final but completed_image_ids does not equal the complete frozen selected-ID list length")
    if final and not complete:
        raise DfpPhase1aReportError("result.final is true but result.complete is false")

    _require_exact_string(result["created_at_utc"], "result.created_at_utc")


def install_result_bundle(
    *, destination_dir: Path, checkpoints_dir: Path, checkpoint_filename: str, repo_root: Path,
) -> Path:
    """Build the complete result bundle in a fresh sibling staging
    directory, verify every file and every cross-file hash, verify final
    state, then atomically rename the staging directory into
    ``destination_dir``. Rejects cleanly (never partially overwrites) if
    ``destination_dir`` already exists. On any failure the staging
    directory is removed and no partial result appears at
    ``destination_dir``."""
    import datetime

    destination_dir = Path(destination_dir)
    checkpoints_dir = Path(checkpoints_dir)
    if destination_dir.exists():
        raise DfpPhase1aReportError(f"result destination {destination_dir} already exists; refusing to overwrite an existing result")

    checkpoint_path = checkpoints_dir / checkpoint_filename
    try:
        checkpoint = parse_strict_json_document(checkpoint_path, label="checkpoint")
    except DfpPhase1aArtifactSchemaError as error:
        raise DfpPhase1aReportError(str(error)) from error
    checkpoint_sha256 = sha256_file(checkpoint_path)

    if not checkpoint.get("complete") or checkpoint.get("next_dataset_index") != checkpoint.get("image_count_expected"):
        raise DfpPhase1aReportError("checkpoint is not complete at the exact expected image count; refusing to finalize a result")

    stats_manifest_path = checkpoints_dir / checkpoint["per_image_stats_manifest_relative_path"]
    try:
        stats_manifest = parse_strict_json_document(stats_manifest_path, label="per-image-stats manifest")
    except DfpPhase1aArtifactSchemaError as error:
        raise DfpPhase1aReportError(str(error)) from error
    try:
        validate_per_image_stats_manifest_structure(stats_manifest)
        validate_checkpoint_against_per_image_stats(checkpoint, stats_manifest)
    except DfpPhase1aCheckpointError as error:
        raise DfpPhase1aReportError(str(error)) from error
    if sha256_file(stats_manifest_path) != checkpoint["per_image_stats_manifest_sha256"]:
        raise DfpPhase1aReportError("per-image-stats manifest SHA256 mismatch against the checkpoint's binding")

    npz_path = checkpoints_dir / stats_manifest["npz_filename"]
    try:
        load_per_image_stats_npz(npz_path, expected_sha256=stats_manifest["npz_sha256"], image_count=stats_manifest["image_count"])
    except DfpPhase1aCheckpointError as error:
        raise DfpPhase1aReportError(str(error)) from error

    result_document = {
        "schema": RESULT_SCHEMA_NAME, "schema_version": RESULT_SCHEMA_VERSION,
        "artifact_schema_identity_sha256": checkpoint["artifact_schema_identity_sha256"],
        "protocol_identity_sha256": checkpoint["protocol_identity_sha256"],
        "subset_manifest_sha256": checkpoint["subset_manifest_sha256"],
        "selected_ids_digest": checkpoint["selected_ids_digest"], "run_mode": checkpoint["run_mode"],
        "image_count_expected": checkpoint["image_count_expected"], "checkpoint_sha256": checkpoint_sha256,
        "per_image_stats_manifest_sha256": checkpoint["per_image_stats_manifest_sha256"], "complete": True, "final": True,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    validate_result_structure(result_document, checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha256)

    staging_dir = destination_dir.parent / f"{destination_dir.name}.staging-{secrets.token_hex(8)}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    try:
        staging_dir.mkdir(parents=True)
        write_json_atomically(staging_dir / "result.json", result_document, label="result")
        write_json_atomically(staging_dir / "checkpoint.json", checkpoint, label="checkpoint")
        write_json_atomically(staging_dir / stats_manifest_path.name, stats_manifest, label="per-image-stats manifest")
        shutil.copyfile(npz_path, staging_dir / npz_path.name)

        # Full self-verification of the staged bundle before install: every
        # file present, every hash reconciles, exactly as a fresh reader
        # would independently re-check.
        staged_checkpoint_sha256 = sha256_file(staging_dir / "checkpoint.json")
        if staged_checkpoint_sha256 != checkpoint_sha256:
            raise DfpPhase1aReportError("staged checkpoint.json does not hash-match the source checkpoint")
        staged_manifest_sha256 = sha256_file(staging_dir / stats_manifest_path.name)
        if staged_manifest_sha256 != checkpoint["per_image_stats_manifest_sha256"]:
            raise DfpPhase1aReportError("staged per-image-stats manifest does not hash-match the checkpoint's binding")
        staged_npz_sha256 = sha256_file(staging_dir / npz_path.name)
        if staged_npz_sha256 != stats_manifest["npz_sha256"]:
            raise DfpPhase1aReportError("staged per-image-stats NPZ does not hash-match its manifest")
        load_per_image_stats_npz(staging_dir / npz_path.name, expected_sha256=staged_npz_sha256, image_count=stats_manifest["image_count"])
        staged_result = parse_strict_json_document(staging_dir / "result.json", label="staged result")
        validate_result_structure(staged_result, checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha256)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    try:
        os.rename(staging_dir, destination_dir)
    except OSError as error:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise DfpPhase1aReportError(f"cannot install result bundle at {destination_dir}: {error}") from error

    return destination_dir


def verify_result_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Read-only re-verification of an installed result bundle: every
    file present, every hash reconciles, and the result document itself
    validates against its bound checkpoint. Never mutates anything."""
    bundle_dir = Path(bundle_dir)
    result_path = bundle_dir / "result.json"
    checkpoint_path = bundle_dir / "checkpoint.json"
    try:
        result = parse_strict_json_document(result_path, label="result")
        checkpoint = parse_strict_json_document(checkpoint_path, label="checkpoint")
    except DfpPhase1aArtifactSchemaError as error:
        raise DfpPhase1aReportError(str(error)) from error

    checkpoint_sha256 = sha256_file(checkpoint_path)
    validate_result_structure(result, checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha256)

    manifest_relative_path = require_safe_basename(checkpoint["per_image_stats_manifest_relative_path"], "checkpoint.per_image_stats_manifest_relative_path")
    manifest_path = bundle_dir / manifest_relative_path
    try:
        stats_manifest = parse_strict_json_document(manifest_path, label="per-image-stats manifest")
    except DfpPhase1aArtifactSchemaError as error:
        raise DfpPhase1aReportError(str(error)) from error
    try:
        validate_per_image_stats_manifest_structure(stats_manifest)
        validate_checkpoint_against_per_image_stats(checkpoint, stats_manifest)
    except DfpPhase1aCheckpointError as error:
        raise DfpPhase1aReportError(str(error)) from error
    if sha256_file(manifest_path) != checkpoint["per_image_stats_manifest_sha256"]:
        raise DfpPhase1aReportError("bundled per-image-stats manifest SHA256 mismatch against the checkpoint's binding")

    npz_path = bundle_dir / require_safe_basename(stats_manifest["npz_filename"], "per-image-stats manifest.npz_filename")
    try:
        arrays = load_per_image_stats_npz(npz_path, expected_sha256=stats_manifest["npz_sha256"], image_count=stats_manifest["image_count"])
    except DfpPhase1aCheckpointError as error:
        raise DfpPhase1aReportError(str(error)) from error

    return {"result": result, "checkpoint": checkpoint, "stats_manifest": stats_manifest, "arrays": arrays}


__all__ = [
    "DfpPhase1aReportError",
    "TOP_RESULT_KEYS",
    "validate_result_structure",
    "install_result_bundle",
    "verify_result_bundle",
]
