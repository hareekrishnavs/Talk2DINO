"""DFP Phase 1B: strict artifact validators/loaders and atomic writers
for the BALP development checkpoint and development-result artifacts,
implementing exactly the schemas frozen in
`evaluation_identities/e22_dfp_phase1b_lightweight_boundary_artifacts.toml`
(`[checkpoint_schema]`, `[development_result_schema]`) and the write
policy frozen in `[artifact_write_policy]`.

On-disk layout (a design choice E22 leaves open -- E22's own artifacts
identity is explicitly `storage_representation_only = True`, governing
only the logical checkpoint DOCUMENT's closed key set and relational
rules, never a physical file layout; this module's layout is documented
here transparently, never fabricated as "the one true E22 layout"):

Each checkpoint generation is one IMMUTABLE directory containing a
`checkpoint.json` metadata file and a sibling `checkpoint_arrays.npz`
bulk-numeric file, bound by a SHA-256 recorded in the JSON's
`identity_bindings.development_digest` field:

    output_root/
      checkpoint_generations/
        generation_<N>/
          checkpoint.json
          checkpoint_arrays.npz
      current_checkpoint.json
      development_result/
        result.json

A generation directory is written into a private staging directory,
self-verified there, and made visible under its real name via exactly
ONE atomic directory rename (`os.replace`) -- the same single-syscall
atomicity `install_development_result` already relies on below. This
means a generation directory is either fully absent or fully complete;
there is no observable intermediate state with only one of its two
member files. Once installed and re-verified from disk, the small
`current_checkpoint.json` pointer (itself replaced via exactly one
atomic single-file `os.replace`) is the ONLY thing that ever changes to
make a new generation current -- so a crash or failure at any point
before that final pointer replace leaves the previous current
generation completely untouched and still loadable. Generation
directories are otherwise never overwritten or mutated in place.

This module imports no torch, no model, no dataset, and performs no
inference. It validates and persists already-computed scientific
arrays only.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import src.dfp_phase1b_lightweight_boundary_identity as e22
from src.dfp_phase1a_artifact_schema import DfpPhase1aArtifactSchemaError, write_json_atomically


class DfpPhase1bLightweightBoundaryArtifactError(ValueError):
    """Raised on any BALP development-artifact invariant violation,
    including a corrupt/truncated/unreadable NPZ or JSON member -- the
    original exception (e.g. `zipfile.BadZipFile`) is always preserved
    as `__cause__`. Always fail closed. Never raised for, and never
    catches, KeyboardInterrupt / SystemExit / MemoryError -- those
    always propagate unmodified."""


CANONICAL_FILENAMES: dict[str, str] = {
    "checkpoint": "phase1b_lightweight_boundary_checkpoint.json",
    "checkpoint_arrays": "phase1b_lightweight_boundary_checkpoint_arrays.npz",
    "development_result": "phase1b_lightweight_boundary_development_result.json",
    "development_result_arrays": "phase1b_lightweight_boundary_development_result_arrays.npz",
}

CHECKPOINT_GENERATIONS_DIRNAME = "checkpoint_generations"
CURRENT_CHECKPOINT_POINTER_FILENAME = "current_checkpoint.json"
# Storage-layer-only tag identifying THIS module's pointer-file shape.
# Not an E22 schema_name -- E22 does not govern a physical pointer file
# at all (see module docstring); this exists purely so a pointer file
# from an incompatible future storage-layer revision fails closed
# instead of being silently misread.
CURRENT_POINTER_FORMAT = "dfp-phase1b-lightweight-boundary-current-checkpoint-pointer-v1"
_CURRENT_POINTER_KEYS = frozenset({
    "format", "current_generation", "generation_directory",
    "checkpoint_json_sha256", "checkpoint_arrays_sha256",
})

PER_IMAGE_STAT_SCALAR_FIELDS: tuple[str, ...] = (
    "candidate_versus_t320_changed_pixel_count", "preservation_mask_pixel_count",
    "boundary_preservation_pixel_count", "non_boundary_preservation_pixel_count",
    "evaluated_pixel_count", "ignored_pixel_count",
)

# Expected data-corruption/parsing failures from opening or reading a
# `.npz` file -- a truncated file, random bytes, a valid zip missing a
# required member (raised by our own membership checks, not this set),
# or a valid zip whose internal `.npy` payload is malformed. Never
# includes KeyboardInterrupt/SystemExit/MemoryError or any other
# BaseException subclass, and this is never a bare `except:` or
# `except Exception:` -- only these specific, narrowly-justified types.
_NPZ_CORRUPTION_EXCEPTIONS = (zipfile.BadZipFile, OSError, ValueError, EOFError)


def _wrap_json_error(exc: DfpPhase1aArtifactSchemaError, label: str) -> DfpPhase1bLightweightBoundaryArtifactError:
    return DfpPhase1bLightweightBoundaryArtifactError(f"{label}: {exc}")


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise DfpPhase1bLightweightBoundaryArtifactError(f"cannot hash {path}: {exc}") from exc


def _read_npz_flat(path: Path) -> dict[str, np.ndarray]:
    """Opens and fully reads a `.npz` file, wrapping every expected
    corruption/parsing failure (truncated zip, random bytes, a
    corrupted internal `.npy` payload) in the single artifact domain
    exception with the original exception preserved as `__cause__`.
    Fatal BaseException subclasses (KeyboardInterrupt, SystemExit,
    MemoryError, ...) are never caught here and propagate unmodified."""
    try:
        with np.load(path, allow_pickle=False) as reloaded:
            return {k: reloaded[k] for k in reloaded.files}
    except _NPZ_CORRUPTION_EXCEPTIONS as exc:
        raise DfpPhase1bLightweightBoundaryArtifactError(f"checkpoint_arrays.npz at {path} is corrupt or unreadable: {exc}") from exc


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (json.JSONDecodeError, OSError) as exc:
        raise DfpPhase1bLightweightBoundaryArtifactError(f"cannot read/parse JSON at {path}: {exc}") from exc


# =======================================================================
# Section 1: in-memory checkpoint-array skeleton (aggregate_confusion +
# per_image_reconstruction_statistics), independent of any file format.
# =======================================================================


def empty_checkpoint_arrays(*, variant_ids: Sequence[str], class_count: int) -> dict[str, Any]:
    """A zero-image checkpoint arrays skeleton: aggregate_confusion is
    the [class_count,class_count] zero matrix per variant; every
    per-image field is an empty list per variant."""
    agg: dict[str, list[list[int]]] = {}
    per_image: dict[str, dict[str, list]] = {}
    for v in variant_ids:
        agg[v] = [[0] * class_count for _ in range(class_count)]
        per_image[v] = {f: [] for f in PER_IMAGE_STAT_SCALAR_FIELDS}
        per_image[v]["per_image_confusion_matrix"] = []
    return {"aggregate_confusion": agg, "per_image_reconstruction_statistics": per_image}


def _matrix_add(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def append_image_to_checkpoint_arrays(
    arrays: Mapping[str, Any], *, variant_ids: Sequence[str],
    per_variant_confusion: Mapping[str, list[list[int]]],
    per_variant_scalars: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Returns a NEW arrays dict with one more completed image appended
    (append-only; never mutates `arrays` in place)."""
    new_agg = {v: _matrix_add(arrays["aggregate_confusion"][v], per_variant_confusion[v]) for v in variant_ids}
    new_per_image: dict[str, dict[str, list]] = {}
    for v in variant_ids:
        old = arrays["per_image_reconstruction_statistics"][v]
        entry = {f: list(old[f]) + [int(per_variant_scalars[v][f])] for f in PER_IMAGE_STAT_SCALAR_FIELDS}
        entry["per_image_confusion_matrix"] = list(old["per_image_confusion_matrix"]) + [per_variant_confusion[v]]
        new_per_image[v] = entry
    return {"aggregate_confusion": new_agg, "per_image_reconstruction_statistics": new_per_image}


# =======================================================================
# Section 2: NPZ <-> nested-dict conversion for the bulk arrays.
# =======================================================================


def checkpoint_arrays_to_flat_npz(arrays: Mapping[str, Any], *, variant_ids: Sequence[str]) -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}
    for v in variant_ids:
        flat[f"agg__{v}"] = np.asarray(arrays["aggregate_confusion"][v], dtype=np.int64)
        per_image = arrays["per_image_reconstruction_statistics"][v]
        for f in PER_IMAGE_STAT_SCALAR_FIELDS:
            flat[f"pi__{v}__{f}"] = np.asarray(per_image[f], dtype=np.int64)
        cm = per_image["per_image_confusion_matrix"]
        class_count = len(arrays["aggregate_confusion"][v])
        flat[f"pi__{v}__per_image_confusion_matrix"] = (
            np.asarray(cm, dtype=np.int64) if cm else np.zeros((0, class_count, class_count), dtype=np.int64)
        )
    return flat


def _require_npz_ndarray(flat: Mapping[str, Any], key: str) -> np.ndarray:
    """A corrupted `.npy` payload inside an otherwise-valid zip
    container is not always caught by `np.load` itself -- numpy can
    return the member's raw bytes instead of raising. This guards every
    member access against that, so a malformed member always fails
    through the artifact domain exception rather than a raw
    AttributeError from code that assumed it already had an ndarray."""
    if key not in flat:
        raise DfpPhase1bLightweightBoundaryArtifactError(f"checkpoint_arrays.npz missing member {key!r}")
    value = flat[key]
    if not isinstance(value, np.ndarray):
        raise DfpPhase1bLightweightBoundaryArtifactError(f"checkpoint_arrays.npz member {key!r} is not a valid array, observed {type(value).__name__}")
    return value


def flat_npz_to_checkpoint_arrays(flat: Mapping[str, np.ndarray], *, variant_ids: Sequence[str]) -> dict[str, Any]:
    agg: dict[str, list] = {}
    per_image: dict[str, dict[str, list]] = {}
    for v in variant_ids:
        key = f"agg__{v}"
        agg_arr = _require_npz_ndarray(flat, key)
        if agg_arr.dtype != np.int64 or agg_arr.ndim != 2:
            raise DfpPhase1bLightweightBoundaryArtifactError(f"{key} must be a 2-D int64 array, observed dtype={agg_arr.dtype} ndim={agg_arr.ndim}")
        agg[v] = agg_arr.tolist()
        entry: dict[str, list] = {}
        for f in PER_IMAGE_STAT_SCALAR_FIELDS:
            k = f"pi__{v}__{f}"
            arr = _require_npz_ndarray(flat, k)
            if arr.dtype != np.int64 or arr.ndim != 1:
                raise DfpPhase1bLightweightBoundaryArtifactError(f"{k} must be a 1-D int64 array, observed dtype={arr.dtype} ndim={arr.ndim}")
            entry[f] = arr.tolist()
        k_cm = f"pi__{v}__per_image_confusion_matrix"
        cm_arr = _require_npz_ndarray(flat, k_cm)
        if cm_arr.dtype != np.int64 or cm_arr.ndim != 3:
            raise DfpPhase1bLightweightBoundaryArtifactError(f"{k_cm} must be a 3-D int64 array, observed dtype={cm_arr.dtype} ndim={cm_arr.ndim}")
        entry["per_image_confusion_matrix"] = cm_arr.tolist()
        per_image[v] = entry
    return {"aggregate_confusion": agg, "per_image_reconstruction_statistics": per_image}


# =======================================================================
# Section 3: crash-atomic immutable checkpoint generations + a single
# atomically-replaceable current-generation pointer. See module
# docstring for the on-disk layout and the atomicity argument.
# =======================================================================


def _generation_dir_name(generation: int) -> str:
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise DfpPhase1bLightweightBoundaryArtifactError(f"generation must be a non-negative int, observed {generation!r}")
    return f"generation_{generation}"


def _safe_generation_dir(output_root: Path, generation_directory: Any) -> Path:
    """Resolves a `generation_directory` name (as recorded in the
    current pointer) to a real path strictly inside
    `output_root/checkpoint_generations/`. Rejects a non-string, an
    empty string, `.`/`..`, any path separator (path traversal or
    absolute-path substitution), a symlink, and any resolved path
    escaping the generations root."""
    if not isinstance(generation_directory, str) or not generation_directory:
        raise DfpPhase1bLightweightBoundaryArtifactError(f"generation_directory must be a non-empty string, observed {generation_directory!r}")
    if "/" in generation_directory or "\\" in generation_directory or generation_directory in (".", ".."):
        raise DfpPhase1bLightweightBoundaryArtifactError(f"generation_directory must be a single path component, observed {generation_directory!r}")
    generations_root = Path(output_root) / CHECKPOINT_GENERATIONS_DIRNAME
    candidate = generations_root / generation_directory
    if candidate.is_symlink():
        raise DfpPhase1bLightweightBoundaryArtifactError(f"generation directory {candidate} is a symlink -- rejected")
    if not candidate.is_dir():
        raise DfpPhase1bLightweightBoundaryArtifactError(f"generation directory {candidate} does not exist or is not a directory")
    resolved_root = generations_root.resolve()
    resolved = candidate.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise DfpPhase1bLightweightBoundaryArtifactError(f"generation directory {candidate} escapes {resolved_root}")
    return resolved


def _write_checkpoint_arrays_npz_into(dest_dir: Path, arrays: Mapping[str, Any], *, variant_ids: Sequence[str]) -> tuple[Path, str]:
    """Writes `checkpoint_arrays.npz` directly into `dest_dir` (a
    private staging directory not yet visible under its final name),
    closes it, reopens and structurally reverifies the staged bytes,
    and returns (path, sha256-of-bytes). Never performs an `os.replace`
    itself -- the caller installs the whole containing directory
    atomically in one step once every staged member is verified."""
    flat = checkpoint_arrays_to_flat_npz(arrays, variant_ids=variant_ids)
    dest_path = dest_dir / CANONICAL_FILENAMES["checkpoint_arrays"]
    try:
        np.savez(dest_path, allow_pickle=False, **flat)
    except OSError as exc:
        raise DfpPhase1bLightweightBoundaryArtifactError(f"cannot write checkpoint arrays into staging: {exc}") from exc
    written = dest_path if dest_path.suffix == ".npz" else dest_path.with_suffix(dest_path.suffix + ".npz")
    staged_flat = _read_npz_flat(written)
    flat_npz_to_checkpoint_arrays(staged_flat, variant_ids=variant_ids)  # raises on structural violation
    return written, _sha256_file(written)


def _write_checkpoint_json_into(dest_dir: Path, metadata_doc: Mapping[str, Any]) -> Path:
    dest_path = dest_dir / CANONICAL_FILENAMES["checkpoint"]
    try:
        write_json_atomically(dest_path, metadata_doc, label="BALP development checkpoint")
    except DfpPhase1aArtifactSchemaError as exc:
        raise _wrap_json_error(exc, "checkpoint write") from exc
    reloaded = _read_json(dest_path)
    if reloaded != dict(metadata_doc):
        raise DfpPhase1bLightweightBoundaryArtifactError("staged checkpoint.json reopened content disagrees with the just-written document")
    return dest_path


def build_checkpoint_metadata(
    *, identity_bindings: Mapping[str, str], next_development_index: int, completed_image_ids: Sequence[str],
    generation: int, complete: bool, final: bool, final_result_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_name": e22.CHECKPOINT_SCHEMA_NAME, "schema_version": e22.CHECKPOINT_SCHEMA_VERSION,
        "identity_bindings": dict(identity_bindings),
        "progress": {
            "expected_development_image_count": e22.DEVELOPMENT_COUNT,
            "completed_image_ids": list(completed_image_ids),
            "next_development_index": next_development_index,
            "generation": generation,
        },
        "complete": complete, "final": final,
        "final_result_binding": dict(final_result_binding) if final_result_binding is not None else None,
    }


def assemble_full_checkpoint_document(metadata_doc: Mapping[str, Any], arrays: Mapping[str, Any]) -> dict[str, Any]:
    full = dict(metadata_doc)
    full["aggregate_confusion"] = arrays["aggregate_confusion"]
    full["per_image_reconstruction_statistics"] = arrays["per_image_reconstruction_statistics"]
    return full


def validate_full_checkpoint(
    full_doc: Mapping[str, Any], *, development_ids: Sequence[str], confirmation_ids: Sequence[str],
) -> None:
    """The single, public validation entry point for an assembled BALP
    checkpoint document -- delegates entirely to the real, already-
    verified `e22.validate_checkpoint_single`, never a second
    reimplementation. Re-raises E22's own domain exception as this
    module's domain exception so every public function in this module
    fails closed with exactly one exception type."""
    try:
        e22.validate_checkpoint_single(full_doc, development_ids=development_ids, confirmation_ids=confirmation_ids)
    except e22.DfpPhase1bLightweightBoundaryIdentityError as exc:
        raise DfpPhase1bLightweightBoundaryArtifactError(str(exc)) from exc


def load_current_pointer(output_root: Path) -> dict[str, Any] | None:
    """Reopens the current-checkpoint pointer file. Returns None if no
    checkpoint has ever been installed (fresh-start case) -- including
    when `output_root` itself does not yet exist."""
    pointer_path = Path(output_root) / CURRENT_CHECKPOINT_POINTER_FILENAME
    if not pointer_path.exists():
        return None
    doc = _read_json(pointer_path)
    if not isinstance(doc, dict):
        raise DfpPhase1bLightweightBoundaryArtifactError(f"current checkpoint pointer must be a JSON object, observed {type(doc).__name__}")
    actual = frozenset(doc.keys())
    if actual != _CURRENT_POINTER_KEYS:
        raise DfpPhase1bLightweightBoundaryArtifactError(
            f"current checkpoint pointer has the wrong key set -- missing={sorted(_CURRENT_POINTER_KEYS - actual)} unknown={sorted(actual - _CURRENT_POINTER_KEYS)}"
        )
    if doc["format"] != CURRENT_POINTER_FORMAT:
        raise DfpPhase1bLightweightBoundaryArtifactError(f"current checkpoint pointer format must be {CURRENT_POINTER_FORMAT!r}, observed {doc.get('format')!r}")
    gen = doc["current_generation"]
    if isinstance(gen, bool) or not isinstance(gen, int) or gen < 0:
        raise DfpPhase1bLightweightBoundaryArtifactError(f"current_generation must be a non-negative int, observed {gen!r}")
    for key in ("checkpoint_json_sha256", "checkpoint_arrays_sha256"):
        if not isinstance(doc[key], str) or not doc[key]:
            raise DfpPhase1bLightweightBoundaryArtifactError(f"current checkpoint pointer {key} must be a non-empty string, observed {doc[key]!r}")
    return doc


def load_full_checkpoint(output_root: Path, *, variant_ids: Sequence[str]) -> dict[str, Any] | None:
    """Resolves the current pointer to its immutable generation
    directory, reopens both member files from disk, and reassembles
    the full logical document -- verifying the pointer's own recorded
    digests, the JSON/NPZ sibling digest, and that the JSON's own
    `progress.generation` agrees with the pointer (catching a JSON from
    one generation ever being paired with an NPZ from another). Returns
    None if no checkpoint has ever been installed (fresh-start case)."""
    output_root = Path(output_root)
    pointer = load_current_pointer(output_root)
    if pointer is None:
        return None

    generation_dir = _safe_generation_dir(output_root, pointer["generation_directory"])
    json_path = generation_dir / CANONICAL_FILENAMES["checkpoint"]
    npz_path = generation_dir / CANONICAL_FILENAMES["checkpoint_arrays"]
    if not json_path.is_file() or not npz_path.is_file():
        raise DfpPhase1bLightweightBoundaryArtifactError(
            f"current generation directory {generation_dir} is incomplete (missing checkpoint.json or checkpoint_arrays.npz)"
        )

    metadata_doc = _read_json(json_path)
    flat = _read_npz_flat(npz_path)
    arrays = flat_npz_to_checkpoint_arrays(flat, variant_ids=variant_ids)

    actual_json_sha = _sha256_file(json_path)
    actual_npz_sha = _sha256_file(npz_path)
    if actual_json_sha != pointer["checkpoint_json_sha256"]:
        raise DfpPhase1bLightweightBoundaryArtifactError(
            f"current pointer checkpoint_json_sha256 mismatch: pointer={pointer['checkpoint_json_sha256']!r} actual={actual_json_sha!r}"
        )
    if actual_npz_sha != pointer["checkpoint_arrays_sha256"]:
        raise DfpPhase1bLightweightBoundaryArtifactError(
            f"current pointer checkpoint_arrays_sha256 mismatch: pointer={pointer['checkpoint_arrays_sha256']!r} actual={actual_npz_sha!r}"
        )

    recorded_digest = metadata_doc.get("identity_bindings", {}).get("development_digest") if isinstance(metadata_doc, dict) else None
    if recorded_digest != actual_npz_sha:
        raise DfpPhase1bLightweightBoundaryArtifactError(
            f"checkpoint sibling digest mismatch: identity_bindings.development_digest={recorded_digest!r} but "
            f"checkpoint_arrays.npz sha256={actual_npz_sha!r}"
        )

    recorded_generation = metadata_doc.get("progress", {}).get("generation") if isinstance(metadata_doc, dict) else None
    if recorded_generation != pointer["current_generation"]:
        raise DfpPhase1bLightweightBoundaryArtifactError(
            f"current pointer generation {pointer['current_generation']} disagrees with checkpoint.json's own "
            f"progress.generation {recorded_generation!r} -- possible mixed-generation JSON/NPZ pairing"
        )

    return assemble_full_checkpoint_document(metadata_doc, arrays)


def install_checkpoint_generation(
    output_root: Path, *, identity_bindings_without_digest: Mapping[str, str], variant_ids: Sequence[str],
    arrays: Mapping[str, Any], completed_image_ids: Sequence[str], generation: int, complete: bool, final: bool,
    final_result_binding: Mapping[str, Any] | None, development_ids: Sequence[str], confirmation_ids: Sequence[str],
) -> dict[str, Any]:
    """Installs ONE new, immutable checkpoint generation and -- only
    after it has been fully installed and reverified on disk -- calls
    `_advance_current_pointer` to atomically make it current.

    Crash-atomicity: `checkpoint.json`/`checkpoint_arrays.npz` are
    written into a private staging directory, self-verified there, and
    made visible as `checkpoint_generations/generation_<N>/` via
    exactly ONE atomic directory rename (`os.replace`) -- a single
    filesystem-level rename that can never expose a half-written
    generation directory. Only once that installed generation has been
    reopened from disk and revalidated does the small current pointer
    get atomically replaced to reference it (see
    `_advance_current_pointer`). Any failure before the pointer replace
    leaves the PREVIOUS current generation exactly as it was; the new
    generation, whether unwritten, partially staged, or fully installed
    but not yet current, never becomes current.

    An installed-but-unreferenced generation directory (the new
    generation was written and renamed into place successfully, but a
    later step -- post-install reverification or the pointer advance --
    failed) is deliberately left on disk rather than deleted: E22 does
    not govern this storage layer at all, so there is no schema
    obligation either way, and leaving it (a) is harmless (an
    unreferenced generation can never become current by itself), (b)
    avoids adding a second filesystem operation, with its own failure
    modes, on top of an already-successful install, and (c) preserves
    exactly the forensic evidence needed to diagnose what happened. A
    failure while still staging (before the directory rename) always
    removes the staging directory -- that residue carries no forensic
    value, since nothing under its real name was ever installed."""
    output_root = Path(output_root)
    generations_root = output_root / CHECKPOINT_GENERATIONS_DIRNAME
    generations_root.mkdir(parents=True, exist_ok=True)
    next_index = len(completed_image_ids)

    # Pre-stage self-verification (placeholder digest; every OTHER relational rule checked now).
    provisional_metadata = build_checkpoint_metadata(
        identity_bindings={**identity_bindings_without_digest, "development_digest": "0" * 64},
        next_development_index=next_index, completed_image_ids=completed_image_ids, generation=generation,
        complete=complete, final=final, final_result_binding=final_result_binding,
    )
    provisional_full = assemble_full_checkpoint_document(provisional_metadata, arrays)
    _validate_ignoring_identity_digest(provisional_full, development_ids=development_ids, confirmation_ids=confirmation_ids)

    generation_dir_name = _generation_dir_name(generation)
    generation_dir = generations_root / generation_dir_name
    if generation_dir.exists():
        raise DfpPhase1bLightweightBoundaryArtifactError(
            f"generation directory {generation_dir} already exists -- immutable generations are never overwritten"
        )

    staging = generations_root / f".staging-{generation_dir_name}-{secrets.token_hex(16)}"
    staging.mkdir(parents=True)
    try:
        npz_path, npz_sha = _write_checkpoint_arrays_npz_into(staging, arrays, variant_ids=variant_ids)
        final_metadata = build_checkpoint_metadata(
            identity_bindings={**identity_bindings_without_digest, "development_digest": npz_sha},
            next_development_index=next_index, completed_image_ids=completed_image_ids, generation=generation,
            complete=complete, final=final, final_result_binding=final_result_binding,
        )
        json_path = _write_checkpoint_json_into(staging, final_metadata)

        # Reload the STAGED files and run full structural + relational + public validation
        # before this generation is ever made visible under its real name.
        staged_flat = _read_npz_flat(npz_path)
        staged_arrays = flat_npz_to_checkpoint_arrays(staged_flat, variant_ids=variant_ids)
        staged_metadata = _read_json(json_path)
        staged_full = assemble_full_checkpoint_document(staged_metadata, staged_arrays)
        validate_full_checkpoint(staged_full, development_ids=development_ids, confirmation_ids=confirmation_ids)

        os.replace(staging, generation_dir)  # single atomic rename: fully absent or fully present, never partial
    except (OSError, DfpPhase1bLightweightBoundaryArtifactError):
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    # Post-install: reopen the INSTALLED generation (never staging, which no longer exists under that name)
    # and publicly reverify it, still WITHOUT touching the current pointer.
    installed_json_path = generation_dir / CANONICAL_FILENAMES["checkpoint"]
    installed_npz_path = generation_dir / CANONICAL_FILENAMES["checkpoint_arrays"]
    installed_metadata = _read_json(installed_json_path)
    installed_flat = _read_npz_flat(installed_npz_path)
    installed_arrays = flat_npz_to_checkpoint_arrays(installed_flat, variant_ids=variant_ids)
    installed_full = assemble_full_checkpoint_document(installed_metadata, installed_arrays)
    validate_full_checkpoint(installed_full, development_ids=development_ids, confirmation_ids=confirmation_ids)

    installed_json_sha = _sha256_file(installed_json_path)
    installed_npz_sha = _sha256_file(installed_npz_path)
    _advance_current_pointer(
        output_root, generation=generation, generation_directory=generation_dir_name,
        checkpoint_json_sha256=installed_json_sha, checkpoint_arrays_sha256=installed_npz_sha,
    )

    # Reopen the current pointer + resolve it, proving the reference resolves to the new valid generation.
    reopened = load_full_checkpoint(output_root, variant_ids=variant_ids)
    if reopened is None:
        raise DfpPhase1bLightweightBoundaryArtifactError(
            "post-advance reopen failed: current checkpoint pointer not resolvable immediately after install"
        )
    validate_full_checkpoint(reopened, development_ids=development_ids, confirmation_ids=confirmation_ids)
    return reopened


def _advance_current_pointer(
    output_root: Path, *, generation: int, generation_directory: str,
    checkpoint_json_sha256: str, checkpoint_arrays_sha256: str,
) -> None:
    """Atomically replaces the single current-checkpoint pointer file.
    Called ONLY after the target generation has already been installed
    (via one atomic directory rename) and independently reverified from
    disk -- this function's own write is therefore the single
    remaining step that can make a new generation current, and it is
    itself exactly one atomic single-file `os.replace` (performed
    inside `write_json_atomically`), so it can never leave a
    half-written pointer: either the previous pointer is untouched, or
    the new one is fully in place."""
    new_pointer = {
        "format": CURRENT_POINTER_FORMAT,
        "current_generation": generation,
        "generation_directory": generation_directory,
        "checkpoint_json_sha256": checkpoint_json_sha256,
        "checkpoint_arrays_sha256": checkpoint_arrays_sha256,
    }
    pointer_path = Path(output_root) / CURRENT_CHECKPOINT_POINTER_FILENAME
    try:
        write_json_atomically(pointer_path, new_pointer, label="BALP current checkpoint pointer")
    except DfpPhase1aArtifactSchemaError as exc:
        raise _wrap_json_error(exc, "current checkpoint pointer write") from exc


def reconcile_orphan_successor(
    output_root: Path, *, current: Mapping[str, Any] | None, variant_ids: Sequence[str],
    identity_bindings_without_digest: Mapping[str, str], development_ids: Sequence[str], confirmation_ids: Sequence[str],
) -> dict[str, Any] | None:
    """Repairs the one remaining crash window: a new generation
    directory installed successfully (atomic rename succeeded) but the
    current pointer never advanced to it, before `install_checkpoint_
    generation` could return -- e.g. a crash during post-install
    reverification or during the pointer write itself. Without this,
    every subsequent attempt re-derives the SAME already-occupied
    generation number and fails forever with "generation directory ...
    already exists".

    Must be called BEFORE any dataset/model/graph/recurrence/image-
    processor/GT-provider access, immediately after loading the current
    checkpoint (or establishing there is none). Inspects ONLY the
    single canonical expected successor directory `generation_<N+1>`
    (or `generation_0` when `current` is None) -- never guesses among
    multiple candidates, never picks "the newest" by mtime or name
    sort, and never recomputes, rewrites, deletes, or renames anything.

    Returns the newly-current full checkpoint document if a valid
    orphan was found and adopted (Case B); returns None if there was
    nothing to reconcile (Case A). Raises
    `DfpPhase1bLightweightBoundaryArtifactError` if a successor exists
    but fails any validation (Case C), or if the filesystem state at
    the successor slot is ambiguous -- more than one non-historical
    entry, or a present entry whose name disagrees with the expected
    one (Case D)."""
    output_root = Path(output_root)
    current_generation = current["progress"]["generation"] if current is not None else None
    expected_generation = 0 if current_generation is None else int(current_generation) + 1
    expected_name = _generation_dir_name(expected_generation)

    generations_root = output_root / CHECKPOINT_GENERATIONS_DIRNAME
    if not generations_root.is_dir():
        return None  # Case A: nothing has ever been installed.

    historical_names = {_generation_dir_name(g) for g in range(expected_generation)}
    present_names = [p.name for p in generations_root.iterdir() if not p.name.startswith(".")]
    non_historical = sorted(n for n in present_names if n not in historical_names)

    if not non_historical:
        return None  # Case A: no successor at all.

    if non_historical != [expected_name]:
        # Case D: more than one candidate, a gap, or an unexpected alternate name.
        # Never select "the newest" or otherwise guess -- fail closed.
        raise DfpPhase1bLightweightBoundaryArtifactError(
            f"ambiguous checkpoint-generation state at {generations_root}: expected at most the single "
            f"canonical successor {expected_name!r}, observed {non_historical} -- refusing to guess which "
            "is the valid interrupted commit"
        )

    # Exactly one candidate, with the canonically-expected name. Validate it COMPLETELY
    # before it is ever treated as current -- never partially trust it.
    candidate_dir = _safe_generation_dir(output_root, expected_name)  # path safety, symlink escape
    json_path = candidate_dir / CANONICAL_FILENAMES["checkpoint"]
    npz_path = candidate_dir / CANONICAL_FILENAMES["checkpoint_arrays"]
    if not json_path.is_file() or not npz_path.is_file():
        raise DfpPhase1bLightweightBoundaryArtifactError(
            f"orphan successor {expected_name!r} is incomplete (missing checkpoint.json or checkpoint_arrays.npz) -- refusing to adopt"
        )

    metadata_doc = _read_json(json_path)
    if not isinstance(metadata_doc, dict):
        raise DfpPhase1bLightweightBoundaryArtifactError(f"orphan successor {expected_name!r} checkpoint.json is not a JSON object")
    flat = _read_npz_flat(npz_path)
    arrays = flat_npz_to_checkpoint_arrays(flat, variant_ids=variant_ids)  # NPZ member/shape/dtype validation
    candidate_full = assemble_full_checkpoint_document(metadata_doc, arrays)

    actual_npz_sha = _sha256_file(npz_path)
    recorded_digest = metadata_doc.get("identity_bindings", {}).get("development_digest")
    if recorded_digest != actual_npz_sha:
        raise DfpPhase1bLightweightBoundaryArtifactError(
            f"orphan successor {expected_name!r} sibling digest mismatch: recorded={recorded_digest!r} actual={actual_npz_sha!r} -- refusing to adopt"
        )

    recorded_generation = metadata_doc.get("progress", {}).get("generation")
    if recorded_generation != expected_generation:
        raise DfpPhase1bLightweightBoundaryArtifactError(
            f"orphan successor {expected_name!r} internal generation mismatch: directory name implies "
            f"{expected_generation}, but checkpoint.json progress.generation={recorded_generation!r} -- refusing to adopt"
        )

    candidate_bindings = dict(metadata_doc.get("identity_bindings", {}))
    candidate_bindings_without_digest = {k: v for k, v in candidate_bindings.items() if k != "development_digest"}
    if candidate_bindings_without_digest != dict(identity_bindings_without_digest):
        raise DfpPhase1bLightweightBoundaryArtifactError(
            f"orphan successor {expected_name!r} identity_bindings disagree with this run's own bindings -- "
            "refusing to adopt a checkpoint from a different run"
        )

    # Full closed-schema + E22 relational validation: schema name/version, exact key sets,
    # frozen 350-image split prefix (this alone rejects an invalid completed-ID prefix or any
    # confirmation/non-development ID), candidate/control ordering, aggregate == sum(per-image),
    # complete/final legality, and every other checkpoint_schema.relational_validation rule.
    validate_full_checkpoint(candidate_full, development_ids=development_ids, confirmation_ids=confirmation_ids)

    if current is not None:
        # Predecessor/current relationship: every checkpoint_schema.cross_generation_invariants
        # rule (exact-prefix completed_image_ids, unchanged already-completed rows, aggregate ==
        # old aggregate plus the newly appended per-image confusion matrices, unchanged identity
        # bindings/ordering), independently re-derived by E22's own committed validator -- never
        # a second, divergent reimplementation.
        # e22.validate_checkpoint_resume_pair compares `identity_bindings` for byte-identical
        # equality, but `development_digest` is INTENTIONALLY different for every generation (it
        # is that generation's own NPZ hash) -- already checked separately above. Neutralize just
        # that one field to a shared placeholder in both probe copies so the comparison exercises
        # every OTHER invariant (exact-prefix completed_image_ids, unchanged already-completed
        # rows, aggregate == old aggregate plus newly appended per-image matrices, unchanged
        # ordering) meaningfully, without ever mutating the real documents being adopted.
        current_probe = dict(current)
        current_probe["identity_bindings"] = {**current["identity_bindings"], "development_digest": "0" * 64}
        candidate_probe = dict(candidate_full)
        candidate_probe["identity_bindings"] = {**candidate_full["identity_bindings"], "development_digest": "0" * 64}
        try:
            e22.validate_checkpoint_resume_pair(current_probe, candidate_probe)
        except e22.DfpPhase1bLightweightBoundaryIdentityError as exc:
            raise DfpPhase1bLightweightBoundaryArtifactError(
                f"orphan successor {expected_name!r} fails cross-generation validation against the current "
                f"generation {current_generation}: {exc}"
            ) from exc
        current_next_index = int(current["progress"]["next_development_index"])
        candidate_next_index = candidate_full["progress"]["next_development_index"]
        is_normal_image_extension = candidate_next_index == current_next_index + 1
        # A finalize-only transition (the generation `_finalize_only` installs) adds no NEW
        # completed image -- it keeps the exact same 350 completed images and flips
        # complete/final from (True, False) to (True, True), binding the installed result.
        # `next_development_index` is therefore unchanged, not incremented.
        is_finalize_only_transition = (
            candidate_next_index == current_next_index
            and bool(current.get("complete")) and not bool(current.get("final"))
            and bool(candidate_full.get("complete")) and bool(candidate_full.get("final"))
        )
        if not (is_normal_image_extension or is_finalize_only_transition):
            raise DfpPhase1bLightweightBoundaryArtifactError(
                f"orphan successor {expected_name!r} is neither a one-image extension nor a legal "
                f"finalize-only transition of the current checkpoint (current next_development_index="
                f"{current_next_index}, complete={current.get('complete')}, final={current.get('final')}; "
                f"successor next_development_index={candidate_next_index}, complete={candidate_full.get('complete')}, "
                f"final={candidate_full.get('final')}) -- refusing to adopt"
            )
    else:
        if candidate_full["progress"]["next_development_index"] != 1:
            raise DfpPhase1bLightweightBoundaryArtifactError(
                f"orphan successor {expected_name!r} (first-ever generation) must represent exactly 1 "
                f"completed image, observed next_development_index={candidate_full['progress']['next_development_index']} -- refusing to adopt"
            )

    # Every check passed: this is a uniquely valid, already-installed generation. Adopt it by
    # atomically advancing the pointer -- never recomputing or rewriting its contents -- then
    # reopen through the public loader and revalidate the reopened bytes before returning.
    _advance_current_pointer(
        output_root, generation=expected_generation, generation_directory=expected_name,
        checkpoint_json_sha256=_sha256_file(json_path), checkpoint_arrays_sha256=actual_npz_sha,
    )
    reopened = load_full_checkpoint(output_root, variant_ids=variant_ids)
    if reopened is None:
        raise DfpPhase1bLightweightBoundaryArtifactError(
            f"post-adoption reopen failed: current pointer not resolvable immediately after adopting {expected_name!r}"
        )
    validate_full_checkpoint(reopened, development_ids=development_ids, confirmation_ids=confirmation_ids)
    return reopened


def _validate_ignoring_identity_digest(full_doc: Mapping[str, Any], *, development_ids, confirmation_ids) -> None:
    """Pre-install staged validation cannot yet know the NPZ's final
    hash (chicken-and-egg: the hash is only known once the NPZ has been
    written) -- so this checks every OTHER relational rule via the real
    validator against a copy carrying a syntactically-valid placeholder
    digest, never skipping validation entirely."""
    probe = dict(full_doc)
    probe["identity_bindings"] = dict(full_doc["identity_bindings"])
    try:
        e22.validate_checkpoint_single(probe, development_ids=development_ids, confirmation_ids=confirmation_ids)
    except e22.DfpPhase1bLightweightBoundaryIdentityError as exc:
        raise DfpPhase1bLightweightBoundaryArtifactError(str(exc)) from exc


# =======================================================================
# Section 5: development result -- a `multi_file_artifacts` bundle
# (schema-name-carrying JSON + a sibling NPZ for per_class_iou/
# classwise_gain_loss arrays), installed exactly once via one atomic
# directory rename into an absent destination.
# =======================================================================

DEVELOPMENT_RESULT_SCHEMA_NAME = e22._CANONICAL_ARTIFACTS["development_result_schema.schema_name"]
DEVELOPMENT_RESULT_SCHEMA_VERSION = e22._CANONICAL_ARTIFACTS["development_result_schema.schema_version"]
DEVELOPMENT_RESULT_TOP_KEYS = frozenset(e22._CANONICAL_ARTIFACTS["development_result_schema.top_keys.list"])
CONTROL_RESULT_MEMBER_KEYS = frozenset(e22._CANONICAL_ARTIFACTS["development_result_schema.control_result_member.keys.list"])
CANDIDATE_RESULT_MEMBER_KEYS = frozenset(e22._CANONICAL_ARTIFACTS["development_result_schema.candidate_result_member.keys.list"])


def build_development_result_document(
    *, identity_bindings: Mapping[str, str], control_results: Mapping[str, Mapping[str, Any]],
    candidate_results: Mapping[str, Mapping[str, Any]], scientific_output_digest: str,
    complete: bool, final: bool,
) -> dict[str, Any]:
    for cid, block in control_results.items():
        if frozenset(block.keys()) != CONTROL_RESULT_MEMBER_KEYS:
            raise DfpPhase1bLightweightBoundaryArtifactError(f"control_results[{cid}] has the wrong key set")
    for cid, block in candidate_results.items():
        if frozenset(block.keys()) != CANDIDATE_RESULT_MEMBER_KEYS:
            raise DfpPhase1bLightweightBoundaryArtifactError(f"candidate_results[{cid}] has the wrong key set")
    return {
        "schema_name": DEVELOPMENT_RESULT_SCHEMA_NAME, "schema_version": DEVELOPMENT_RESULT_SCHEMA_VERSION,
        "run_mode": "coco_stuff_500", "development_image_count": e22.DEVELOPMENT_COUNT,
        "identity_bindings": dict(identity_bindings),
        "control_results": dict(control_results), "candidate_results": dict(candidate_results),
        "scientific_output_digest": scientific_output_digest, "complete": complete, "final": final,
    }


def validate_development_result(doc: Mapping[str, Any], *, variant_ids: Sequence[str]) -> None:
    """Strict, closed-schema, relational validation for a development
    result document -- schema name/version, exact top-level and nested
    key sets, exact control/candidate identity, and the frozen
    dataset-level (never per-image-mean) aggregation convention are all
    checked; a malformed/unhashable schema-version pair raises the
    domain exception, never a raw TypeError/KeyError."""
    E = DfpPhase1bLightweightBoundaryArtifactError
    if not isinstance(doc, dict):
        raise E(f"development result must be a mapping, observed {type(doc).__name__}")
    actual = frozenset(doc.keys())
    if actual != DEVELOPMENT_RESULT_TOP_KEYS:
        raise E(f"development result has the wrong key set -- missing={sorted(DEVELOPMENT_RESULT_TOP_KEYS-actual)} unknown={sorted(actual-DEVELOPMENT_RESULT_TOP_KEYS)}")
    try:
        name_ok = type(doc.get("schema_name")) is str and doc["schema_name"] == DEVELOPMENT_RESULT_SCHEMA_NAME
        version_ok = type(doc.get("schema_version")) is int and doc["schema_version"] == DEVELOPMENT_RESULT_SCHEMA_VERSION
    except TypeError as exc:
        raise E(f"malformed/unhashable schema_name or schema_version: {exc}") from exc
    if not name_ok:
        raise E(f"schema_name must be {DEVELOPMENT_RESULT_SCHEMA_NAME!r}, observed {doc.get('schema_name')!r}")
    if not version_ok:
        raise E(f"schema_version must be {DEVELOPMENT_RESULT_SCHEMA_VERSION!r}, observed {doc.get('schema_version')!r}")
    if doc["development_image_count"] != e22.DEVELOPMENT_COUNT:
        raise E(f"development_image_count must be {e22.DEVELOPMENT_COUNT}, observed {doc['development_image_count']}")
    control_ids = [v for v in variant_ids if v in e22.CANONICAL_CONTROLS]
    candidate_ids = [v for v in variant_ids if v not in e22.CANONICAL_CONTROLS]
    if set(doc["control_results"].keys()) != set(control_ids):
        raise E("control_results keys disagree with the frozen 5-control family")
    if set(doc["candidate_results"].keys()) != set(candidate_ids):
        raise E("candidate_results keys disagree with the frozen 16-candidate family")
    for cid, block in doc["control_results"].items():
        if frozenset(block.keys()) != CONTROL_RESULT_MEMBER_KEYS:
            raise E(f"control_results[{cid}] has the wrong key set")
        if block["control_id"] != cid:
            raise E(f"control_results[{cid}].control_id disagrees with its own key")
    for cid, block in doc["candidate_results"].items():
        if frozenset(block.keys()) != CANDIDATE_RESULT_MEMBER_KEYS:
            raise E(f"candidate_results[{cid}] has the wrong key set")
        if block["candidate_id"] != cid:
            raise E(f"candidate_results[{cid}].candidate_id disagrees with its own key")
        s, r = e22.parse_candidate_id(cid)
        if block["stability_start"] != s or block["radius"] != r:
            raise E(f"candidate_results[{cid}] stability_start/radius disagree with its own id")
    if not isinstance(doc["complete"], bool) or not isinstance(doc["final"], bool):
        raise E("complete/final must be booleans")


def install_development_result(
    output_root: Path, *, doc: Mapping[str, Any], variant_ids: Sequence[str],
) -> Path:
    """Multi-file atomic install: stage `result.json` in a uniquely-
    named sibling staging directory, self-verify staged bytes, then
    install via exactly ONE atomic directory rename into an ABSENT
    `development_result/` destination -- never an unguarded sequential
    per-file install. After the rename, the INSTALLED directory is
    reopened from disk and the public validator re-run against those
    installed bytes; only then does this function return."""
    output_root = Path(output_root)
    final_dir = output_root / "development_result"
    if final_dir.exists():
        raise DfpPhase1bLightweightBoundaryArtifactError(
            f"development result destination {final_dir} already exists -- E22 authorizes no overwrite"
        )
    validate_development_result(doc, variant_ids=variant_ids)  # pre-install staged self-verification

    staging = output_root / f".staging-development-result-{secrets.token_hex(16)}"
    staging.mkdir(parents=True)
    try:
        result_path = staging / "result.json"
        try:
            write_json_atomically(result_path, doc, label="BALP development result")
        except DfpPhase1aArtifactSchemaError as exc:
            raise _wrap_json_error(exc, "development result staging write") from exc
        staged_doc = _read_json(result_path)
        validate_development_result(staged_doc, variant_ids=variant_ids)
        os.replace(staging, final_dir)
    except (OSError, DfpPhase1bLightweightBoundaryArtifactError):
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    # Post-install: reopen the INSTALLED path (never the staging path, which no longer exists) and reverify.
    installed_path = final_dir / "result.json"
    installed_doc = _read_json(installed_path)
    validate_development_result(installed_doc, variant_ids=variant_ids)
    return final_dir
