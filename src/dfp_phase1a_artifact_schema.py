"""Strict loading/validation for the additive DFP Phase 1A storage-only
artifact-schema authority (`evaluation_identities/e13_dfp_phase1a_artifacts.toml`),
plus the shared strict JSON I/O and safe-path primitives that
``dfp_phase1a_checkpoint`` and ``dfp_phase1a_report`` build on.

This module is a closed schema AUTHORITY, not an evaluator: it defines
representation and validation for future DFP Phase 1A checkpoint/result
artifacts. It never rewrites or re-binds the committed
`e13-dfp-phase1a-protocol` identity -- both frozen Step-2 subset manifests
pin that file's exact SHA-256 -- and it defines no scientific quantity:
D, Q_w, R_w, horizons, recurrence, components, memberships, compute
formulas, oracle definitions, gates, and metrics all remain exactly as
already frozen by the protocol and by Steps 3-9.

Every relational hash bound here is checked against the LIVE file content
(never merely format-checked), following this project's own established
Step 6/7 lesson: a syntactically valid SHA-256-shaped string that was
never actually recomputed against the live authority is not a real check.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tomllib
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping


class DfpPhase1aArtifactSchemaError(ValueError):
    """Raised on any artifact-schema authority or shared I/O-primitive
    invariant violation. Always fail closed."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e13_dfp_phase1a_artifacts.toml")
PROTOCOL_IDENTITY_RELATIVE_PATH = "evaluation_identities/e13_dfp_phase1a_protocol.toml"

IDENTITY_TOP_KEYS = frozenset(
    {
        "format_version", "identity", "protocol_binding", "subset_manifest_binding",
        "parent_identity_binding", "dataset_binding", "checkpoint_schema", "per_image_stats_schema",
        "result_schema", "path_policy", "atomic_write_policy",
    }
)

SECTION_KEYS: dict[str, frozenset[str]] = {
    "identity": frozenset({"name", "schema_version", "description", "storage_representation_only"}),
    "protocol_binding": frozenset({"protocol_identity_relative_path", "protocol_identity_sha256"}),
    "subset_manifest_binding": frozenset(
        {
            "coco_stuff_manifest_relative_path", "coco_stuff_manifest_sha256",
            "ade20k_manifest_relative_path", "ade20k_manifest_sha256",
        }
    ),
    "parent_identity_binding": frozenset(
        {
            "matched_k11_k12_identity_sha256", "rwr_identity_sha256", "ade20k_source_identity_sha256",
            "ade20k_evaluator_identity_sha256", "bridge_checkpoint_relative_path", "bridge_checkpoint_sha256",
            "required_ancestor_commit",
        }
    ),
    "dataset_binding": frozenset(
        {
            "coco_stuff_class_count", "coco_stuff_class_names_digest", "ade20k_class_count",
            "ade20k_class_names_digest", "horizon_order", "run_mode_names",
        }
    ),
    "checkpoint_schema": frozenset(
        {
            "schema_name", "schema_version", "top_level_key_count", "resume_prefix_rule",
            "progress_invariant", "zero_row_representation", "state_transitions",
            "completion_forbids_further_resume",
        }
    ),
    "per_image_stats_schema": frozenset(
        {
            "schema_name", "schema_version", "row_axis", "int64_member_names", "float64_member_names",
            "excluded_by_design", "excluded_by_design_reason",
        }
    ),
    "result_schema": frozenset({"schema_name", "schema_version", "final_requires", "excludes_in_this_step", "excludes_reason"}),
    "path_policy": frozenset({"allowed_path_kinds", "forbidden", "canonical_image_id_distinct_from_physical_filename"}),
    "atomic_write_policy": frozenset({"checkpoint_install", "result_install"}),
}


# ---------------------------------------------------------------------------
# Primitive validators (deliberately self-contained, matching this
# repository's established convention -- see e.g.
# src/k11_k12_power_evaluation_checkpoint.py's own module docstring -- of
# small one-line primitives duplicated per module while INVARIANT LOGIC is
# never duplicated).
# ---------------------------------------------------------------------------


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str:
        raise DfpPhase1aArtifactSchemaError(f"{label} must be an exact string")
    if nonempty and not value.strip():
        raise DfpPhase1aArtifactSchemaError(f"{label} must be a non-empty, non-whitespace string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1aArtifactSchemaError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise DfpPhase1aArtifactSchemaError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise DfpPhase1aArtifactSchemaError(f"{label} must be at least {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise DfpPhase1aArtifactSchemaError(f"{label} must be a lowercase 64-hex-character SHA-256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 40 or any(c not in "0123456789abcdef" for c in token):
        raise DfpPhase1aArtifactSchemaError(f"{label} must be a full lowercase 40-hex-character Git commit id")
    return token


def _require_string_list(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    if type(value) is not list or (nonempty and not value):
        raise DfpPhase1aArtifactSchemaError(f"{label} must be a non-empty exact list of strings" if nonempty else f"{label} must be an exact list of strings")
    for item in value:
        if type(item) is not str:
            raise DfpPhase1aArtifactSchemaError(f"{label} elements must be exact strings")
    return value


def _require_int_list(value: Any, label: str) -> list[int]:
    if type(value) is not list or not value:
        raise DfpPhase1aArtifactSchemaError(f"{label} must be a non-empty exact list of integers")
    for item in value:
        if type(item) is not int:
            raise DfpPhase1aArtifactSchemaError(f"{label} elements must be exact non-boolean integers")
    return value


def require_safe_relative_path(value: Any, label: str) -> str:
    """Reject absolute paths, `..` traversal, empty/`.`-only components,
    drive-qualified paths, backslash-based traversal, and NUL bytes. Never
    normalizes or repairs -- an unsafe path is always rejected, never
    silently corrected."""
    token = _require_exact_string(value, label)
    if "\x00" in token:
        raise DfpPhase1aArtifactSchemaError(f"{label} must not contain a NUL byte")
    if "\\" in token:
        raise DfpPhase1aArtifactSchemaError(f"{label} must not contain a backslash")
    if token.startswith("/"):
        raise DfpPhase1aArtifactSchemaError(f"{label} must be a repository-relative path, not absolute")
    if len(token) >= 2 and token[1] == ":":
        raise DfpPhase1aArtifactSchemaError(f"{label} must not be drive-qualified")
    # Split manually on "/" rather than using pathlib.Path().parts: pathlib
    # silently collapses a bare "." component during parsing, which would
    # let "a/./b" pass through unnoticed -- this validator must always
    # reject an unsafe path, never silently normalize/repair it.
    parts = token.split("/")
    if parts == [""]:
        raise DfpPhase1aArtifactSchemaError(f"{label} must not be empty")
    for part in parts:
        if part == "":
            raise DfpPhase1aArtifactSchemaError(f"{label} must not contain an empty path component")
        if part == "..":
            raise DfpPhase1aArtifactSchemaError(f"{label} must not contain a parent-traversal component")
        if part == ".":
            raise DfpPhase1aArtifactSchemaError(f"{label} must not contain a '.' component")
    return token


def require_safe_basename(value: Any, label: str) -> str:
    """Reject anything with a directory component at all (only a bare
    filename is a safe basename)."""
    token = require_safe_relative_path(value, label)
    if len(Path(token).parts) != 1:
        raise DfpPhase1aArtifactSchemaError(f"{label} must be a bare basename with no directory component")
    return token


def resolve_within_root(root: Path, relative: str, *, label: str) -> Path:
    """Resolve ``relative`` under ``root`` and require the resolved path to
    still be located inside ``root`` (rejecting an unexpected symlink
    escape), never following it if it is not."""
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise DfpPhase1aArtifactSchemaError(f"{label} resolves outside the artifact root") from error
    return candidate


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except IsADirectoryError as error:
        raise DfpPhase1aArtifactSchemaError(f"cannot hash {path}: path is a directory") from error
    except PermissionError as error:
        raise DfpPhase1aArtifactSchemaError(f"cannot hash {path}: permission denied") from error
    except FileNotFoundError as error:
        raise DfpPhase1aArtifactSchemaError(f"cannot hash {path}: file does not exist") from error
    except OSError as error:
        raise DfpPhase1aArtifactSchemaError(f"cannot hash {path}: {error}") from error


# ---------------------------------------------------------------------------
# Strict JSON I/O -- the one shared reader/writer every DFP Phase 1A
# artifact document (checkpoint, per-image-stats manifest, result) uses.
# ---------------------------------------------------------------------------


def _closed_object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DfpPhase1aArtifactSchemaError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_non_finite(token: str) -> Any:
    raise DfpPhase1aArtifactSchemaError(f"document contains non-finite JSON constant {token}")


def _decimals_to_float(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Mapping):
        return {key: _decimals_to_float(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decimals_to_float(item) for item in value]
    return value


def parse_strict_json_document(path: Path, *, label: str) -> Mapping[str, Any]:
    """Read and strictly parse one JSON document: rejects a missing file,
    a permission error, a directory-as-file, invalid UTF-8, an empty file,
    truncated JSON, trailing garbage, duplicate object keys,
    ``NaN``/``Infinity``/``-Infinity``, and a non-object JSON root. Never
    mutates anything on disk. Floats are parsed via :class:`Decimal` first
    (never a lossy float64 round-trip) and converted to ``float`` only
    after every structural check has already passed."""
    path = Path(path)
    try:
        raw_bytes = path.read_bytes()
    except IsADirectoryError as error:
        raise DfpPhase1aArtifactSchemaError(f"cannot read {label} {path}: path is a directory") from error
    except PermissionError as error:
        raise DfpPhase1aArtifactSchemaError(f"cannot read {label} {path}: permission denied") from error
    except FileNotFoundError as error:
        raise DfpPhase1aArtifactSchemaError(f"cannot read {label} {path}: file does not exist") from error
    except OSError as error:
        raise DfpPhase1aArtifactSchemaError(f"cannot read {label} {path}: {error}") from error

    try:
        text = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise DfpPhase1aArtifactSchemaError(f"{label} {path} is not valid UTF-8: {error}") from error

    if not text.strip():
        raise DfpPhase1aArtifactSchemaError(f"{label} {path} is empty")

    try:
        value = json.loads(
            text, parse_float=Decimal, object_pairs_hook=_closed_object_pairs_hook, parse_constant=_reject_non_finite,
        )
    except DfpPhase1aArtifactSchemaError:
        raise
    except json.JSONDecodeError as error:
        raise DfpPhase1aArtifactSchemaError(f"cannot parse {label} {path}: {error}") from error

    if not isinstance(value, Mapping):
        raise DfpPhase1aArtifactSchemaError(f"{label} {path} must contain one JSON object at its root, observed {type(value).__name__}")
    return _decimals_to_float(value)


def write_json_atomically(path: Path, document: Mapping[str, Any], *, label: str) -> None:
    """Write ``document`` to ``path`` via write-temp-then-``os.replace``:
    a crash mid-write never leaves a corrupt/partial document at ``path``,
    and the previous valid document at ``path`` (if any) is left
    byte-identical until the replace succeeds. ``allow_nan=False``, keys
    sorted, UTF-8, exactly one trailing newline. Never mutates
    ``document``. The temp filename is unpredictable (a random hex nonce
    combined with the writing process's PID) so two concurrent writers
    targeting the same destination cannot collide."""
    try:
        text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise DfpPhase1aArtifactSchemaError(f"cannot serialize {label} to JSON: {error}") from error

    path = Path(path)
    temp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except OSError as error:
        raise DfpPhase1aArtifactSchemaError(f"cannot write {label} to {path}: {error}") from error
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Schema constants -- the authoritative names/versions/member sets
# validate_checkpoint_structure / validate_per_image_stats_array /
# validate_result_structure bind against.
# ---------------------------------------------------------------------------

CHECKPOINT_SCHEMA_NAME = "talk2dino-dfp-phase1a-checkpoint-v1"
CHECKPOINT_SCHEMA_VERSION = 1
PER_IMAGE_STATS_SCHEMA_NAME = "talk2dino-dfp-phase1a-per-image-stats-v1"
PER_IMAGE_STATS_SCHEMA_VERSION = 1
RESULT_SCHEMA_NAME = "talk2dino-dfp-phase1a-result-v1"
RESULT_SCHEMA_VERSION = 1

INT64_MEMBER_NAMES: tuple[str, ...] = (
    "dataset_indices", "dispute_pixel_count", "evaluated_pixel_count", "window_count",
    "zero_dispute_window_count", "total_node_count", "total_nonzero_count",
    "total_baseline_recurrence_flops", "total_projected_recurrence_flops",
    "total_chunk_invocations", "total_projected_width", "maximum_projected_width",
    "maximum_peak_planned_bytes",
)
FLOAT64_MEMBER_NAMES: tuple[str, ...] = ("q_w_mean", "q_w_median", "q_w_p90", "q_w_p95", "q_w_p99", "q_w_maximum")
ALL_NPZ_MEMBER_NAMES = frozenset(INT64_MEMBER_NAMES) | frozenset(FLOAT64_MEMBER_NAMES)

HORIZON_ORDER: tuple[int, ...] = (0, 16, 32, 64, 320)
RUN_MODE_NAMES: tuple[str, ...] = ("coco_stuff_500", "ade20k_500")


# ---------------------------------------------------------------------------
# Identity loading/validation.
# ---------------------------------------------------------------------------


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DfpPhase1aArtifactSchemaError(f"cannot load artifact-schema authority {source}: {error}") from error

    if set(identity) != IDENTITY_TOP_KEYS:
        missing = IDENTITY_TOP_KEYS - set(identity)
        unknown = set(identity) - IDENTITY_TOP_KEYS
        raise DfpPhase1aArtifactSchemaError(f"identity has an unexpected top-level schema (missing={sorted(missing)}, unknown={sorted(unknown)})")
    for section, keys in SECTION_KEYS.items():
        block = identity.get(section)
        if not isinstance(block, Mapping) or set(block) != keys:
            missing = keys - set(block or {})
            unknown = set(block or {}) - keys
            raise DfpPhase1aArtifactSchemaError(f"identity.{section} has an unexpected schema (missing={sorted(missing)}, unknown={sorted(unknown)})")

    _validate_identity_block(identity["identity"])
    _validate_protocol_binding_block(identity["protocol_binding"])
    _validate_subset_manifest_binding_block(identity["subset_manifest_binding"])
    _validate_parent_identity_binding_block(identity["parent_identity_binding"])
    _validate_dataset_binding_block(identity["dataset_binding"])
    _validate_checkpoint_schema_block(identity["checkpoint_schema"])
    _validate_per_image_stats_schema_block(identity["per_image_stats_schema"])
    _validate_result_schema_block(identity["result_schema"])
    _validate_path_policy_block(identity["path_policy"])
    _validate_atomic_write_policy_block(identity["atomic_write_policy"])
    return identity


def _validate_identity_block(block: Mapping[str, Any]) -> None:
    if _require_exact_string(block["name"], "identity.name") != "e13-dfp-phase1a-artifacts":
        raise DfpPhase1aArtifactSchemaError("identity.name must be exactly e13-dfp-phase1a-artifacts")
    if _require_exact_string(block["schema_version"], "identity.schema_version") != "talk2dino-dfp-phase1a-artifacts-identity-v1":
        raise DfpPhase1aArtifactSchemaError("identity.schema_version must be exactly talk2dino-dfp-phase1a-artifacts-identity-v1")
    _require_exact_string(block["description"], "identity.description")
    if _require_exact_bool(block["storage_representation_only"], "identity.storage_representation_only") is not True:
        raise DfpPhase1aArtifactSchemaError("identity.storage_representation_only must be true")


def _validate_protocol_binding_block(block: Mapping[str, Any]) -> None:
    path = require_safe_relative_path(block["protocol_identity_relative_path"], "protocol_binding.protocol_identity_relative_path")
    if path != PROTOCOL_IDENTITY_RELATIVE_PATH:
        raise DfpPhase1aArtifactSchemaError("protocol_binding.protocol_identity_relative_path must be exactly the frozen protocol identity path")
    _require_sha256(block["protocol_identity_sha256"], "protocol_binding.protocol_identity_sha256")


def _validate_subset_manifest_binding_block(block: Mapping[str, Any]) -> None:
    require_safe_relative_path(block["coco_stuff_manifest_relative_path"], "subset_manifest_binding.coco_stuff_manifest_relative_path")
    _require_sha256(block["coco_stuff_manifest_sha256"], "subset_manifest_binding.coco_stuff_manifest_sha256")
    require_safe_relative_path(block["ade20k_manifest_relative_path"], "subset_manifest_binding.ade20k_manifest_relative_path")
    _require_sha256(block["ade20k_manifest_sha256"], "subset_manifest_binding.ade20k_manifest_sha256")


def _validate_parent_identity_binding_block(block: Mapping[str, Any]) -> None:
    _require_sha256(block["matched_k11_k12_identity_sha256"], "parent_identity_binding.matched_k11_k12_identity_sha256")
    _require_sha256(block["rwr_identity_sha256"], "parent_identity_binding.rwr_identity_sha256")
    _require_sha256(block["ade20k_source_identity_sha256"], "parent_identity_binding.ade20k_source_identity_sha256")
    _require_sha256(block["ade20k_evaluator_identity_sha256"], "parent_identity_binding.ade20k_evaluator_identity_sha256")
    require_safe_relative_path(block["bridge_checkpoint_relative_path"], "parent_identity_binding.bridge_checkpoint_relative_path")
    _require_sha256(block["bridge_checkpoint_sha256"], "parent_identity_binding.bridge_checkpoint_sha256")
    _require_git_identity(block["required_ancestor_commit"], "parent_identity_binding.required_ancestor_commit")


def _validate_dataset_binding_block(block: Mapping[str, Any]) -> None:
    _require_exact_int(block["coco_stuff_class_count"], "dataset_binding.coco_stuff_class_count", minimum=1)
    _require_sha256(block["coco_stuff_class_names_digest"], "dataset_binding.coco_stuff_class_names_digest")
    _require_exact_int(block["ade20k_class_count"], "dataset_binding.ade20k_class_count", minimum=1)
    _require_sha256(block["ade20k_class_names_digest"], "dataset_binding.ade20k_class_names_digest")
    horizons = _require_int_list(block["horizon_order"], "dataset_binding.horizon_order")
    if tuple(horizons) != HORIZON_ORDER:
        raise DfpPhase1aArtifactSchemaError(f"dataset_binding.horizon_order must be exactly {list(HORIZON_ORDER)}")
    run_modes = _require_string_list(block["run_mode_names"], "dataset_binding.run_mode_names", nonempty=True)
    if tuple(run_modes) != RUN_MODE_NAMES:
        raise DfpPhase1aArtifactSchemaError(f"dataset_binding.run_mode_names must be exactly {list(RUN_MODE_NAMES)}")


def _validate_checkpoint_schema_block(block: Mapping[str, Any]) -> None:
    if _require_exact_string(block["schema_name"], "checkpoint_schema.schema_name") != CHECKPOINT_SCHEMA_NAME:
        raise DfpPhase1aArtifactSchemaError(f"checkpoint_schema.schema_name must be exactly {CHECKPOINT_SCHEMA_NAME!r}")
    if _require_exact_int(block["schema_version"], "checkpoint_schema.schema_version", minimum=1) != CHECKPOINT_SCHEMA_VERSION:
        raise DfpPhase1aArtifactSchemaError(f"checkpoint_schema.schema_version must be exactly {CHECKPOINT_SCHEMA_VERSION}")
    _require_exact_int(block["top_level_key_count"], "checkpoint_schema.top_level_key_count", minimum=1)
    _require_exact_string(block["resume_prefix_rule"], "checkpoint_schema.resume_prefix_rule")
    _require_exact_string(block["progress_invariant"], "checkpoint_schema.progress_invariant")
    _require_exact_string(block["zero_row_representation"], "checkpoint_schema.zero_row_representation")
    _require_string_list(block["state_transitions"], "checkpoint_schema.state_transitions", nonempty=True)
    if _require_exact_bool(block["completion_forbids_further_resume"], "checkpoint_schema.completion_forbids_further_resume") is not True:
        raise DfpPhase1aArtifactSchemaError("checkpoint_schema.completion_forbids_further_resume must be true")


def _validate_per_image_stats_schema_block(block: Mapping[str, Any]) -> None:
    if _require_exact_string(block["schema_name"], "per_image_stats_schema.schema_name") != PER_IMAGE_STATS_SCHEMA_NAME:
        raise DfpPhase1aArtifactSchemaError(f"per_image_stats_schema.schema_name must be exactly {PER_IMAGE_STATS_SCHEMA_NAME!r}")
    if _require_exact_int(block["schema_version"], "per_image_stats_schema.schema_version", minimum=1) != PER_IMAGE_STATS_SCHEMA_VERSION:
        raise DfpPhase1aArtifactSchemaError(f"per_image_stats_schema.schema_version must be exactly {PER_IMAGE_STATS_SCHEMA_VERSION}")
    _require_exact_string(block["row_axis"], "per_image_stats_schema.row_axis")
    int64_names = _require_string_list(block["int64_member_names"], "per_image_stats_schema.int64_member_names", nonempty=True)
    if tuple(int64_names) != INT64_MEMBER_NAMES:
        raise DfpPhase1aArtifactSchemaError("per_image_stats_schema.int64_member_names disagrees with the authoritative Python member list")
    float64_names = _require_string_list(block["float64_member_names"], "per_image_stats_schema.float64_member_names", nonempty=True)
    if tuple(float64_names) != FLOAT64_MEMBER_NAMES:
        raise DfpPhase1aArtifactSchemaError("per_image_stats_schema.float64_member_names disagrees with the authoritative Python member list")
    _require_string_list(block["excluded_by_design"], "per_image_stats_schema.excluded_by_design", nonempty=True)
    _require_exact_string(block["excluded_by_design_reason"], "per_image_stats_schema.excluded_by_design_reason")


def _validate_result_schema_block(block: Mapping[str, Any]) -> None:
    if _require_exact_string(block["schema_name"], "result_schema.schema_name") != RESULT_SCHEMA_NAME:
        raise DfpPhase1aArtifactSchemaError(f"result_schema.schema_name must be exactly {RESULT_SCHEMA_NAME!r}")
    if _require_exact_int(block["schema_version"], "result_schema.schema_version", minimum=1) != RESULT_SCHEMA_VERSION:
        raise DfpPhase1aArtifactSchemaError(f"result_schema.schema_version must be exactly {RESULT_SCHEMA_VERSION}")
    _require_string_list(block["final_requires"], "result_schema.final_requires", nonempty=True)
    _require_string_list(block["excludes_in_this_step"], "result_schema.excludes_in_this_step", nonempty=True)
    _require_exact_string(block["excludes_reason"], "result_schema.excludes_reason")


def _validate_path_policy_block(block: Mapping[str, Any]) -> None:
    _require_string_list(block["allowed_path_kinds"], "path_policy.allowed_path_kinds", nonempty=True)
    _require_string_list(block["forbidden"], "path_policy.forbidden", nonempty=True)
    if _require_exact_bool(block["canonical_image_id_distinct_from_physical_filename"], "path_policy.canonical_image_id_distinct_from_physical_filename") is not True:
        raise DfpPhase1aArtifactSchemaError("path_policy.canonical_image_id_distinct_from_physical_filename must be true")


def _validate_atomic_write_policy_block(block: Mapping[str, Any]) -> None:
    _require_exact_string(block["checkpoint_install"], "atomic_write_policy.checkpoint_install")
    _require_exact_string(block["result_install"], "atomic_write_policy.result_install")


def validate_static_configuration(*, repo_root: Path, identity_path: Path | None = None) -> dict[str, Any]:
    """Load the artifact-schema authority AND relationally verify every
    bound hash against the LIVE file content (protocol identity, both
    subset manifests, both parent identities the protocol itself binds,
    and the bridge checkpoint weights) -- never merely format-checked."""
    root = Path(repo_root)
    identity = load_identity(identity_path, repo_root=root)

    protocol_path = root / identity["protocol_binding"]["protocol_identity_relative_path"]
    observed_protocol_sha256 = sha256_file(protocol_path)
    if observed_protocol_sha256 != identity["protocol_binding"]["protocol_identity_sha256"]:
        raise DfpPhase1aArtifactSchemaError(
            f"protocol identity {protocol_path} SHA256 mismatch: artifact schema declares "
            f"{identity['protocol_binding']['protocol_identity_sha256']}, observed {observed_protocol_sha256}"
        )

    from src.dfp_phase1a_identity import load_identity as load_protocol_identity

    protocol = load_protocol_identity(protocol_path, repo_root=root)

    coco_manifest_path = root / identity["subset_manifest_binding"]["coco_stuff_manifest_relative_path"]
    observed_coco_manifest_sha256 = sha256_file(coco_manifest_path)
    if observed_coco_manifest_sha256 != identity["subset_manifest_binding"]["coco_stuff_manifest_sha256"]:
        raise DfpPhase1aArtifactSchemaError("coco_stuff subset manifest SHA256 mismatch against the artifact schema's binding")

    ade_manifest_path = root / identity["subset_manifest_binding"]["ade20k_manifest_relative_path"]
    observed_ade_manifest_sha256 = sha256_file(ade_manifest_path)
    if observed_ade_manifest_sha256 != identity["subset_manifest_binding"]["ade20k_manifest_sha256"]:
        raise DfpPhase1aArtifactSchemaError("ade20k subset manifest SHA256 mismatch against the artifact schema's binding")

    parent = identity["parent_identity_binding"]
    protocol_parents = protocol["parent_identities"]
    if parent["matched_k11_k12_identity_sha256"] != protocol_parents["matched_k11_k12_identity_sha256"]:
        raise DfpPhase1aArtifactSchemaError("artifact schema's matched_k11_k12_identity_sha256 disagrees with the live protocol identity's own binding")
    if parent["rwr_identity_sha256"] != protocol_parents["rwr_identity_sha256"]:
        raise DfpPhase1aArtifactSchemaError("artifact schema's rwr_identity_sha256 disagrees with the live protocol identity's own binding")
    if parent["ade20k_source_identity_sha256"] != protocol_parents["ade20k_source_identity_sha256"]:
        raise DfpPhase1aArtifactSchemaError("artifact schema's ade20k_source_identity_sha256 disagrees with the live protocol identity's own binding")
    if parent["ade20k_evaluator_identity_sha256"] != protocol_parents["ade20k_evaluator_identity_sha256"]:
        raise DfpPhase1aArtifactSchemaError("artifact schema's ade20k_evaluator_identity_sha256 disagrees with the live protocol identity's own binding")
    if parent["required_ancestor_commit"] != protocol["identity"]["required_ancestor_commit"]:
        raise DfpPhase1aArtifactSchemaError("artifact schema's required_ancestor_commit disagrees with the live protocol identity")

    bridge_path = root / parent["bridge_checkpoint_relative_path"]
    observed_bridge_sha256 = sha256_file(bridge_path)
    if observed_bridge_sha256 != parent["bridge_checkpoint_sha256"]:
        raise DfpPhase1aArtifactSchemaError("bridge checkpoint weights SHA256 mismatch against the artifact schema's binding")
    if observed_bridge_sha256 != protocol["bridge_checkpoint"]["sha256"]:
        raise DfpPhase1aArtifactSchemaError("bridge checkpoint weights SHA256 disagrees with the live protocol identity's own binding")

    dataset = identity["dataset_binding"]
    if dataset["coco_stuff_class_count"] != protocol["coco_stuff_dataset"]["class_count"]:
        raise DfpPhase1aArtifactSchemaError("artifact schema's coco_stuff_class_count disagrees with the live protocol identity")
    if dataset["coco_stuff_class_names_digest"] != protocol["coco_stuff_dataset"]["class_names_digest"]:
        raise DfpPhase1aArtifactSchemaError("artifact schema's coco_stuff_class_names_digest disagrees with the live protocol identity")
    if dataset["ade20k_class_count"] != protocol["ade20k_dataset"]["class_count"]:
        raise DfpPhase1aArtifactSchemaError("artifact schema's ade20k_class_count disagrees with the live protocol identity")
    if dataset["ade20k_class_names_digest"] != protocol["ade20k_dataset"]["class_names_digest"]:
        raise DfpPhase1aArtifactSchemaError("artifact schema's ade20k_class_names_digest disagrees with the live protocol identity")

    return {
        "identity_name": identity["identity"]["name"],
        "protocol_identity_sha256": observed_protocol_sha256,
        "coco_stuff_manifest_sha256": observed_coco_manifest_sha256,
        "ade20k_manifest_sha256": observed_ade_manifest_sha256,
        "bridge_checkpoint_sha256": observed_bridge_sha256,
    }


__all__ = [
    "DfpPhase1aArtifactSchemaError",
    "IDENTITY_RELATIVE_PATH",
    "PROTOCOL_IDENTITY_RELATIVE_PATH",
    "IDENTITY_TOP_KEYS",
    "SECTION_KEYS",
    "CHECKPOINT_SCHEMA_NAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "PER_IMAGE_STATS_SCHEMA_NAME",
    "PER_IMAGE_STATS_SCHEMA_VERSION",
    "RESULT_SCHEMA_NAME",
    "RESULT_SCHEMA_VERSION",
    "INT64_MEMBER_NAMES",
    "FLOAT64_MEMBER_NAMES",
    "ALL_NPZ_MEMBER_NAMES",
    "HORIZON_ORDER",
    "RUN_MODE_NAMES",
    "repository_root",
    "sha256_file",
    "require_safe_relative_path",
    "require_safe_basename",
    "resolve_within_root",
    "parse_strict_json_document",
    "write_json_atomically",
    "load_identity",
    "validate_static_configuration",
]
