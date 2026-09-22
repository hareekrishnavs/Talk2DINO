"""Torch-free, CUDA-free loader/validator for the T320 target-source
authority (research_specs/semantic_evidence_t320_source_v1.toml) and its
companion source manifest (research_specs/semantic_evidence_t320_source_manifest_v1.json).

This module implements NO graph construction, NO recurrence, NO stitching,
NO model inference, and NO GT access. It only loads, strictly validates, and
independently re-derives (from Git objects, via argument-vector subprocess
calls) the provenance and scientific-semantics claims frozen by those two
files. It must never import Torch, MMCV, any model/dataset code from this
repository, or OpenCV, and never checks out a branch.

Fatal exceptions (KeyboardInterrupt, SystemExit, MemoryError) are never
swallowed by this module: every broad except-clause below re-raises them.

TASK: FREEZE_T320_TARGET_SOURCE_AUTHORITY. This module authorizes NOTHING
beyond what [authorization_boundary]/[status] in the TOML itself declares;
every authorization-boundary flag defaults closed (False) except the single
flag permitting a future, separately-verified isolated implementation task.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import subprocess
import tomllib
from pathlib import Path
from typing import Any


class T320SourceAuthorityError(Exception):
    """Raised for any structural, value, hash, provenance, or authorization
    violation encountered while loading or verifying this authority."""


# Exception types that must never be caught by a broad handler in this
# module; every `except Exception` below is written to re-raise these.
_FATAL = (KeyboardInterrupt, SystemExit, MemoryError)


def _reraise_if_fatal(exc: BaseException) -> None:
    if isinstance(exc, _FATAL):
        raise exc


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def _reject_unsafe_relative_path(relative_path: str) -> None:
    if not isinstance(relative_path, str) or relative_path == "":
        raise T320SourceAuthorityError(f"path must be a non-empty string, got {relative_path!r}")
    if os.path.isabs(relative_path):
        raise T320SourceAuthorityError(f"path must be repository-relative, got absolute path: {relative_path!r}")
    if relative_path.startswith("-"):
        raise T320SourceAuthorityError(f"option-like path rejected: {relative_path!r}")
    parts = relative_path.split("/")
    for part in parts:
        if part in ("..", ""):
            raise T320SourceAuthorityError(f"lexical path traversal or empty segment rejected: {relative_path!r}")
        if part == ".":
            raise T320SourceAuthorityError(f"lexical '.' segment rejected: {relative_path!r}")
        if part.startswith("-"):
            raise T320SourceAuthorityError(f"option-like path segment rejected: {relative_path!r}")


def resolve_repo_relative_path(repo_root: Path, relative_path: str) -> Path:
    """Resolve a repository-relative path, rejecting lexical '..', absolute
    paths, option-like paths, and any symlink that would escape repo_root."""
    _reject_unsafe_relative_path(relative_path)
    repo_root_real = Path(os.path.realpath(repo_root))
    candidate = repo_root / relative_path
    candidate_real = Path(os.path.realpath(candidate))
    try:
        candidate_real.relative_to(repo_root_real)
    except ValueError as exc:
        raise T320SourceAuthorityError(
            f"path escapes repository root after symlink resolution: {relative_path!r}"
        ) from exc
    return candidate


_FULL_HEX40 = frozenset("0123456789abcdef")


def _require_full_commit_hex(commit: Any, label: str) -> str:
    if not isinstance(commit, str):
        raise T320SourceAuthorityError(f"{label} must be a string, got {type(commit).__name__}")
    if len(commit) != 40:
        raise T320SourceAuthorityError(f"{label} must be a full 40-hex-character commit id, got {commit!r} (length {len(commit)})")
    if any(ch not in _FULL_HEX40 for ch in commit.lower()) or commit != commit.lower():
        raise T320SourceAuthorityError(f"{label} must be lowercase hex, got {commit!r}")
    return commit


# ---------------------------------------------------------------------------
# Strict TOML / JSON loading
# ---------------------------------------------------------------------------

def _reject_nan_inf(token: str) -> float:
    raise T320SourceAuthorityError(f"NaN/Infinity is not permitted in this JSON document, got token {token!r}")


def load_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise T320SourceAuthorityError(f"failed to parse TOML at {path}: {exc}") from exc


def load_json_strict(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        return json.loads(text, parse_constant=_reject_nan_inf)
    except T320SourceAuthorityError:
        raise
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise T320SourceAuthorityError(f"failed to parse JSON at {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Git access (argument-vector subprocess only; never shell=True; never a
# checkout)
# ---------------------------------------------------------------------------

def _run_git_text(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise T320SourceAuthorityError(f"git command failed: git -C {repo_root} {' '.join(args)}: {exc}") from exc
    return result.stdout.strip()


def _run_git_bytes(repo_root: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            check=True,
        )
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise T320SourceAuthorityError(f"git command failed: git -C {repo_root} {' '.join(args)}: {exc}") from exc
    return result.stdout


def git_blob_id(repo_root: Path, commit: str, path: str) -> str:
    _require_full_commit_hex(commit, "commit")
    _reject_unsafe_relative_path(path)
    return _run_git_text(repo_root, "rev-parse", f"{commit}:{path}")


def git_blob_sha256(repo_root: Path, commit: str, path: str) -> str:
    _require_full_commit_hex(commit, "commit")
    _reject_unsafe_relative_path(path)
    content = _run_git_bytes(repo_root, "show", f"{commit}:{path}")
    return hashlib.sha256(content).hexdigest()


def git_commit_exists(repo_root: Path, commit: str) -> bool:
    _require_full_commit_hex(commit, "commit")
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise T320SourceAuthorityError(f"failed to check commit existence for {commit}: {exc}") from exc


def git_tree_id(repo_root: Path, commit: str) -> str:
    _require_full_commit_hex(commit, "commit")
    return _run_git_text(repo_root, "rev-parse", f"{commit}^{{tree}}")


def git_is_ancestor(repo_root: Path, ancestor_commit: str, descendant_commit: str) -> bool:
    _require_full_commit_hex(ancestor_commit, "ancestor_commit")
    _require_full_commit_hex(descendant_commit, "descendant_commit")
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", ancestor_commit, descendant_commit],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise T320SourceAuthorityError(f"failed to check ancestor relationship: {exc}") from exc


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# AST-only (no import, no execution) required-symbol locator grammar
#
# Closed grammar, exactly two forms (no others accepted):
#
#   A. Bare symbol locator:      SymbolName
#      Resolves to exactly one `ClassDef`/`FunctionDef`/`AsyncFunctionDef`
#      named `SymbolName`, found either at module level or as a direct
#      method of some class (one level of class nesting) -- the two
#      categories are pooled and must total exactly one match. A nested
#      function (a function defined inside another function), a name that
#      only appears in a comment/docstring/string, or an imported-but-not-
#      locally-defined name never counts as a match.
#
#   B. Instance-attribute locator: ClassName.attribute_name
#      `ClassName` must resolve to exactly one top-level class. Within it,
#      exactly one `self.attribute_name = <value>` assignment must exist,
#      and it must be a *direct* statement of that class's own `__init__`
#      method (not a nested helper defined inside `__init__`, not any other
#      method, not class-level scope, not a property setter, not an
#      annotation without a value). No other assignment to the same
#      attribute may exist anywhere else in the class. `__init__` is the
#      only enclosing method this grammar currently supports -- the sole
#      instance-attribute locator in this authority's manifest
#      (`DINOTextSegInference.align_corners`) requires it, and the grammar
#      is extended no further than that single existing entry requires.
#
# No wildcard/glob syntax, call syntax, indexing syntax, relative/module
# import syntax, or more than one dot is ever accepted.
# ---------------------------------------------------------------------------

ALIGN_CORNERS_LOCATOR_CLASS = "DINOTextSegInference"
ALIGN_CORNERS_LOCATOR_ATTRIBUTE = "align_corners"
ALIGN_CORNERS_LOCATOR_REQUIRED_VALUE = False
_ATTRIBUTE_LOCATOR_FROZEN_ENCLOSING_METHOD = "__init__"

# locator -> the literal value that must be assigned, for the (currently
# single) instance-attribute locator that carries a frozen expected value.
# A dotted locator not listed here is verified structurally (resolves
# exactly once, in `__init__`) without an additional value constraint.
_ATTRIBUTE_LOCATOR_REQUIRED_VALUES: dict[str, Any] = {
    f"{ALIGN_CORNERS_LOCATOR_CLASS}.{ALIGN_CORNERS_LOCATOR_ATTRIBUTE}": ALIGN_CORNERS_LOCATOR_REQUIRED_VALUE,
}


def _parse_symbol_locator(locator: str, context: str) -> tuple[str, str | None]:
    """Split a locator string into (name, None) for a bare symbol, or
    (class_name, attribute_name) for an instance-attribute locator. Rejects
    every malformed form: empty string, leading/trailing dot, empty
    segment, more than one dot, whitespace, or any non-identifier
    character (no call/index/glob/import syntax)."""
    if not isinstance(locator, str) or locator == "":
        raise T320SourceAuthorityError(f"empty or non-string required-symbol locator in {context}")
    if locator != locator.strip():
        raise T320SourceAuthorityError(f"required-symbol locator {locator!r} in {context} has leading/trailing whitespace")
    parts = locator.split(".")
    if len(parts) > 2:
        raise T320SourceAuthorityError(
            f"required-symbol locator {locator!r} in {context} has unsupported dotted depth "
            f"({len(parts) - 1} dots; only a single dot is supported)"
        )
    for part in parts:
        if part == "" or not part.isidentifier():
            raise T320SourceAuthorityError(
                f"required-symbol locator {locator!r} in {context} contains an invalid segment {part!r}"
            )
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[1]


class _DefinitionCollector(ast.NodeVisitor):
    """Collects every ClassDef/FunctionDef/AsyncFunctionDef in the module,
    tagging each with whether it sits at module level or as a direct method
    of some class (one level of class nesting) -- and nothing deeper."""

    def __init__(self) -> None:
        self.module_level: dict[str, list[ast.AST]] = {}
        self.method_level: dict[str, list[ast.AST]] = {}
        self._class_depth = 0
        self._function_depth = 0

    def _record(self, node: ast.AST, name: str) -> None:
        if self._function_depth > 0:
            return  # nested inside a function: never a valid locator target
        if self._class_depth == 0:
            self.module_level.setdefault(name, []).append(node)
        elif self._class_depth == 1:
            self.method_level.setdefault(name, []).append(node)
        # class_depth >= 2 (nested class): not supported by this grammar, ignored

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node, node.name)
        self._class_depth += 1
        self.generic_visit(node)
        self._class_depth -= 1

    def _visit_function_like(self, node: ast.AST, name: str) -> None:
        self._record(node, name)
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_like(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_like(node, node.name)


def verify_bare_symbol_locator_source(source: str, name: str, context: str) -> None:
    """Verify a bare (undotted) required-symbol locator resolves to exactly
    one module-level or method-level class/function definition."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise T320SourceAuthorityError(f"source at {context} failed to parse: {exc}") from exc
    collector = _DefinitionCollector()
    collector.visit(tree)
    matches = collector.module_level.get(name, []) + collector.method_level.get(name, [])
    if not matches:
        raise T320SourceAuthorityError(f"required symbol {name!r} not found (as a definition) in {context}")
    if len(matches) > 1:
        raise T320SourceAuthorityError(
            f"required symbol {name!r} is ambiguous in {context}: {len(matches)} definitions found"
        )


class _SelfAttributeAssignmentCollector(ast.NodeVisitor):
    """Collects every `self.<attr> = <value>` assignment, tagging each with
    its immediately enclosing class name/depth and immediately enclosing
    function name/depth, so a caller can require the assignment to be a
    *direct* statement of one specific class's one specific method."""

    def __init__(self) -> None:
        self.assignments: list[dict[str, Any]] = []
        self._class_stack: list[str] = []
        self._function_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def _visit_function_like(self, node: ast.AST, name: str) -> None:
        self._function_stack.append(name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_like(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_like(node, node.name)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                self.assignments.append({
                    "attr": target.attr,
                    "class_name": self._class_stack[-1] if self._class_stack else None,
                    "class_depth": len(self._class_stack),
                    "function_name": self._function_stack[-1] if self._function_stack else None,
                    "function_depth": len(self._function_stack),
                    "value_node": node.value,
                    "lineno": node.lineno,
                })
        self.generic_visit(node)


def verify_attribute_locator_source(
    source: str, class_name: str, attribute_name: str, context: str,
    *, enclosing_method: str = _ATTRIBUTE_LOCATOR_FROZEN_ENCLOSING_METHOD,
    required_value: Any = None,
) -> None:
    """Verify an instance-attribute locator `ClassName.attribute_name`:
    class exists exactly once (top level); the sole `self.attribute_name =
    ...` assignment anywhere in that class is a direct statement of its own
    `enclosing_method` (never a nested helper, another method, class-level
    scope, or a property setter); and, if `required_value` is not None,
    that assignment's value is exactly that literal."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise T320SourceAuthorityError(f"source at {context} failed to parse: {exc}") from exc

    class_defs = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name]
    if not class_defs:
        raise T320SourceAuthorityError(f"class {class_name} not found in {context}")
    if len(class_defs) > 1:
        raise T320SourceAuthorityError(f"class {class_name} defined more than once in {context} -- ambiguous locator")

    collector = _SelfAttributeAssignmentCollector()
    collector.visit(tree)
    class_matches = [
        a for a in collector.assignments
        if a["class_name"] == class_name and a["class_depth"] == 1 and a["attr"] == attribute_name
    ]
    if not class_matches:
        raise T320SourceAuthorityError(
            f"no assignment to self.{attribute_name} found anywhere in class {class_name} in {context}"
        )
    if len(class_matches) > 1:
        raise T320SourceAuthorityError(
            f"multiple ({len(class_matches)}) assignments to self.{attribute_name} found in class "
            f"{class_name} in {context} -- ambiguous locator, no single reachable assignment"
        )
    match = class_matches[0]
    if not (match["function_depth"] == 1 and match["function_name"] == enclosing_method):
        where = (
            f"function {match['function_name']!r} (depth {match['function_depth']})"
            if match["function_name"] is not None else "class-level scope (no enclosing function)"
        )
        raise T320SourceAuthorityError(
            f"the sole self.{attribute_name} assignment in {context} is not a direct statement of "
            f"{class_name}.{enclosing_method} -- found in {where} at line {match['lineno']}"
        )

    if required_value is not None:
        value_node = match["value_node"]
        if not (isinstance(value_node, ast.Constant) and value_node.value is required_value):
            raise T320SourceAuthorityError(
                f"self.{attribute_name} assignment in {context} is not the literal constant "
                f"{required_value!r} (non-literal or wrong-value assignment rejected): got {ast.dump(value_node)}"
            )


def verify_align_corners_locator_source(source: str, context: str = "<source>") -> None:
    """Backward-compatible name: verifies the sole
    `DINOTextSegInference.align_corners` instance-attribute locator, now
    implemented on top of the generic `verify_attribute_locator_source`,
    including the `__init__`-specificity requirement."""
    verify_attribute_locator_source(
        source, ALIGN_CORNERS_LOCATOR_CLASS, ALIGN_CORNERS_LOCATOR_ATTRIBUTE, context,
        enclosing_method=_ATTRIBUTE_LOCATOR_FROZEN_ENCLOSING_METHOD,
        required_value=ALIGN_CORNERS_LOCATOR_REQUIRED_VALUE,
    )


def verify_align_corners_locator(repo_root: Path, commit: str, path: str) -> None:
    """Fetch the historical blob at `commit:path` via a read-only Git object
    read (never an import, never a checkout) and verify it with
    `verify_align_corners_locator_source`."""
    _require_full_commit_hex(commit, "commit")
    _reject_unsafe_relative_path(path)
    try:
        source = _run_git_bytes(repo_root, "show", f"{commit}:{path}").decode("utf-8")
    except UnicodeDecodeError as exc:
        raise T320SourceAuthorityError(f"source at {commit}:{path} is not valid UTF-8: {exc}") from exc
    verify_align_corners_locator_source(source, context=f"{commit}:{path}")


def verify_required_symbols_source(source: str, required_symbols: list, context: str) -> None:
    """Verify every locator in `required_symbols` against already-fetched
    source text, dispatching each to the bare-symbol or instance-attribute
    branch of the closed grammar. Every symbol in the list must resolve --
    not merely one representative symbol."""
    for locator in required_symbols:
        name, attribute = _parse_symbol_locator(locator, context)
        if attribute is None:
            verify_bare_symbol_locator_source(source, name, context)
        else:
            required_value = _ATTRIBUTE_LOCATOR_REQUIRED_VALUES.get(locator)
            verify_attribute_locator_source(
                source, name, attribute, context,
                enclosing_method=_ATTRIBUTE_LOCATOR_FROZEN_ENCLOSING_METHOD,
                required_value=required_value,
            )


def verify_required_symbols(repo_root: Path, commit: str, path: str, required_symbols: list) -> None:
    """Fetch the historical blob at `commit:path` via a read-only Git object
    read and verify every required-symbol locator against it."""
    _require_full_commit_hex(commit, "commit")
    _reject_unsafe_relative_path(path)
    try:
        source = _run_git_bytes(repo_root, "show", f"{commit}:{path}").decode("utf-8")
    except UnicodeDecodeError as exc:
        raise T320SourceAuthorityError(f"source at {commit}:{path} is not valid UTF-8: {exc}") from exc
    verify_required_symbols_source(source, required_symbols, context=f"{commit}:{path}")


# ---------------------------------------------------------------------------
# Canonical, order/whitespace/locale-independent source-closure digest
# ---------------------------------------------------------------------------

def _encode_length_prefixed_utf8(value: str) -> bytes:
    if not isinstance(value, str):
        raise T320SourceAuthorityError(f"canonical encoding requires a string, got {type(value).__name__}")
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(8, "big") + encoded


def _encode_reason_required(reason: str) -> bytes:
    if not isinstance(reason, str):
        raise T320SourceAuthorityError(f"reason_required must be a string, got {type(reason).__name__}")
    if reason.strip() == "":
        raise T320SourceAuthorityError("reason_required must not be empty or whitespace-only")
    if "\x00" in reason:
        raise T320SourceAuthorityError("reason_required must not contain a NUL byte")
    # Encoded as ordinary well-formed Unicode text (NFC is not forced: the
    # frozen prose is authored once, by us, as plain ASCII/Unicode text, and
    # this function's determinism guarantee is about byte-for-byte UTF-8
    # encoding of whatever string is supplied, not about Unicode normalization).
    return _encode_length_prefixed_utf8(reason)


def _encode_record(record: dict) -> bytes:
    required = record["required"]
    if not isinstance(required, bool):
        raise T320SourceAuthorityError(f"required must be a bool, got {type(required).__name__}: {required!r}")
    parts = [
        _encode_length_prefixed_utf8(str(record["ordinal"])),
        _encode_length_prefixed_utf8(record["role"]),
        _encode_length_prefixed_utf8(record["path"]),
        _encode_length_prefixed_utf8(record["source_commit"]),
        _encode_length_prefixed_utf8(record["blob_id"]),
        _encode_length_prefixed_utf8(record["sha256"]),
        _encode_length_prefixed_utf8(record["classification"]),
        _encode_length_prefixed_utf8(str(len(record["required_symbols"]))),
    ]
    for symbol in record["required_symbols"]:
        parts.append(_encode_length_prefixed_utf8(symbol))
    # `required` has exactly one canonical representation per boolean value:
    # the length-prefixed UTF-8 string "true" or "false" (the same scalar
    # encoding used throughout this scheme), never a raw byte flag, so a
    # `required=false` record (schema-invalid on its own, but still a value
    # this function must encode deterministically) is unambiguously distinct
    # from `required=true`.
    parts.append(_encode_length_prefixed_utf8("true" if required else "false"))
    parts.append(_encode_reason_required(record["reason_required"]))
    return b"".join(parts)


def compute_canonical_source_closure_digest(manifest: dict) -> str:
    """Recompute the canonical closure digest directly from manifest content,
    independent of JSON whitespace, key order, filesystem order, locale, and
    PYTHONHASHSEED. See [source_manifest_binding].canonical_encoding_definition
    in the TOML for the exact prose specification this function implements.

    Covers every field of each file record: ordinal, role, path,
    source_commit, blob_id, sha256, classification, required_symbols (in
    order), required, and reason_required -- changing any one of them
    changes the digest."""
    digest_input = b"".join(_encode_record(record) for record in manifest["file_records"])
    return hashlib.sha256(digest_input).hexdigest()


# ---------------------------------------------------------------------------
# Closed schemas
# ---------------------------------------------------------------------------

_TOML_TOP_KEYS = frozenset({
    "identity", "prerequisite", "status", "historical_source", "source_manifest_binding",
    "t0_contract", "graph_contract", "recurrence_contract", "geometry_contract",
    "internal_upsample_contract", "final_rescale_contract", "stitching_contract",
    "prediction_contract", "gt_isolation", "historical_350_disclosure",
    "future_implementation_boundary", "authorization_boundary", "closure",
})

_STATUS_KEYS = frozenset({
    "source_authority_complete", "source_implementation_present_on_current_branch",
    "implementation_performed_in_this_task",
    "isolated_future_implementation_authorized_after_commit_and_independent_verification",
    "existing_e3_scientific_file_modification_authorized", "cli_implementation_authorized",
    "dataset_access_authorized", "model_construction_authorized",
    "checkpoint_tensor_loading_authorized", "t320_execution_authorized",
    "intervention_execution_authorized", "gt_access_authorized",
    "feasibility_96_execution_authorized", "training_authorized",
    "confirmation_execution_authorized", "full_5000_evaluation_authorized",
    "existence_of_this_authority_implies_execution_authorization",
})

_STATUS_MUST_BE_TRUE = frozenset({
    "source_authority_complete",
    "isolated_future_implementation_authorized_after_commit_and_independent_verification",
})
# every other key in _STATUS_KEYS must be False.

_AUTHORIZATION_BOUNDARY_KEYS = frozenset({
    "t320_execution_authorized", "graph_construction_authorized",
    "recurrence_execution_authorized", "cross_branch_source_use_authorized",
    "ignored_bytecode_is_authority", "gt_access_authorized", "model_construction_authorized",
    "checkpoint_tensor_loading_authorized", "dataset_access_authorized",
    "feasibility_96_execution_authorized", "training_authorized",
    "confirmation_execution_authorized", "full_5000_evaluation_authorized",
    "existing_e3_scientific_file_modification_authorized",
    "isolated_future_implementation_authorized_after_commit_and_independent_verification",
})

# Keys in _AUTHORIZATION_BOUNDARY_KEYS whose frozen value must be True; every
# other key in that set must be False.
_AUTHORIZATION_BOUNDARY_MUST_BE_TRUE = frozenset({
    "isolated_future_implementation_authorized_after_commit_and_independent_verification",
})

_MANIFEST_TOP_KEYS = frozenset({
    "schema", "created_at_utc", "task", "source_branch_informational", "source_commit",
    "source_tree_id", "reviewed_e3_base_commit", "file_records",
    "canonical_source_closure_digest", "canonical_encoding", "excluded_source",
    "complete", "final",
})

_FILE_RECORD_KEYS = frozenset({
    "ordinal", "role", "path", "source_commit", "blob_id", "sha256", "required_symbols",
    "classification", "required", "reason_required",
})

_VALID_CLASSIFICATIONS = frozenset({
    "scientific_kernel", "scientific_adapter", "geometry_adapter", "e3_model_boundary",
    "prediction_extraction",
})

# The loader implements exactly one canonical encoding version and never
# selects encoding behavior dynamically from an untrusted manifest label;
# `manifest.canonical_encoding` must equal this exact string.
CANONICAL_ENCODING_VERSION = "length_prefixed_utf8_v2"

_EXPECTED_FILE_COUNT = 7
_EXPECTED_RELATIVE_PATHS = (
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/graph.py",
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/finite_step_regime.py",
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/matched_power_evaluator.py",
    "src/open_vocabulary_segmentation/segmentation/evaluation/sliding_window_geometry.py",
    "src/open_vocabulary_segmentation/models/dinotext/dinotext.py",
    "src/open_vocabulary_segmentation/models/dinotext/masker.py",
    "src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_seg.py",
)
_WINDOW_CACHE_PATH = "src/open_vocabulary_segmentation/segmentation/evaluation/window_cache.py"
_DINOTEXT_SEG_PATH = "src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_seg.py"
_ALIGN_CORNERS_LOCATOR_SYMBOL = "DINOTextSegInference.align_corners"
_BROAD_DINOTEXT_SEG_CLASS_SYMBOL = "DINOTextSegInference"


def _require_keys(data: dict, expected: frozenset, label: str) -> None:
    if not isinstance(data, dict):
        raise T320SourceAuthorityError(f"{label} must be a table, got {type(data).__name__}")
    actual = frozenset(data.keys())
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise T320SourceAuthorityError(f"{label} is missing required keys: {sorted(missing)}")
    if extra:
        raise T320SourceAuthorityError(f"{label} has unexpected keys: {sorted(extra)}")


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise T320SourceAuthorityError(f"{label} must be a bool, got {type(value).__name__}: {value!r}")
    return value


def _require_exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise T320SourceAuthorityError(f"{label} must be an int (not bool), got {type(value).__name__}: {value!r}")
    return value


def _require_finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise T320SourceAuthorityError(f"{label} must be a number, got {type(value).__name__}: {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise T320SourceAuthorityError(f"{label} must be finite, got {value!r}")
    return value


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise T320SourceAuthorityError(f"{label} must be a string, got {type(value).__name__}: {value!r}")
    return value


# ---------------------------------------------------------------------------
# TOML authority validation
# ---------------------------------------------------------------------------

_SUBTABLE_KEYS: dict[str, frozenset] = {
    "identity": frozenset({"created_at_utc", "purpose", "schema_name", "schema_version", "task"}),
    "prerequisite": frozenset({
        "feasibility_contract_commit", "feasibility_contract_docs_sha256", "feasibility_contract_files_count",
        "feasibility_contract_identity_py_sha256", "feasibility_contract_states_96_image_execution_authorized",
        "feasibility_contract_states_t0", "feasibility_contract_states_t320", "feasibility_contract_subset_sha256",
        "feasibility_contract_tests_sha256", "feasibility_contract_toml_sha256", "note",
        "prior_audit_report_path", "prior_audit_report_verdict", "prior_completion_report_path",
        "prior_completion_report_verdict", "prior_independent_verification_report_path",
        "prior_independent_verification_verdict", "required_branch", "required_head",
        "reviewed_e3_base_commit",
    }),
    "historical_source": frozenset({
        "abbreviated_commit_rejected", "branch_ref_currently_contains_commit", "checkout_of_source_branch_used",
        "e3_base_is_ancestor_of_source_commit", "full_40_hex_commit_required", "read_mechanism",
        "rejected_lineage", "reviewed_e3_base_commit", "scientific_identity_is_commit_tree_blob_closure_not_branch_name",
        "source_branch_informational", "source_commit", "source_commit_object_type", "source_tree_id",
        "superseded_lineage",
    }),
    "historical_source.rejected_lineage": frozenset({"branch", "reason"}),
    "historical_source.superseded_lineage": frozenset({"branch", "reason"}),
    "source_manifest_binding": frozenset({
        "canonical_encoding", "canonical_encoding_exact_match_required", "canonical_encoding_note",
        "canonical_encoding_definition", "canonical_source_closure_digest",
        "excluded_source_count", "manifest_file_sha256", "manifest_path", "manifest_schema_name",
        "manifest_schema_version", "required_symbol_locator_grammar",
        "required_symbols_ast_verified_for_all_seven_records",
        "self_referential_hash_in_manifest", "source_file_count",
    }),
    "t0_contract": frozenset({
        "bound_section", "bound_to_identity_py", "bound_to_identity_py_sha256", "bound_to_toml",
        "bound_to_toml_sha256", "class_ordering", "identity_with_query_1", "mechanisms_excluded",
        "predicted_class_rule", "redefinition_by_this_authority_authorized", "score_domain", "view",
    }),
    "graph_contract": frozenset({
        "affinity_power", "affinity_power_application", "deterministic_neighbor_tie_breaking",
        "device_preserving", "directed_symmetric_behavior", "dtype", "k", "matmul_dtype_behavior",
        "matmul_orientation", "negative_affinity_handling", "node_ordering", "node_source",
        "normalization_direction", "patch_grid_ordering", "required_symbol", "self_loop_policy",
        "similarity_formula", "source_manifest_ordinal", "source_path", "zero_degree_behavior",
    }),
    "recurrence_contract": frozenset({
        "accumulation_dtype", "accumulation_dtype_confirmed_by", "alpha", "alpha_type_requirement",
        "deterministic_behavior_requirement", "formula", "initialization", "matrix_multiplication_orientation",
        "no_adaptive_step_count", "no_cgls_gmres_dense_solve_or_fallback", "no_convergence_test",
        "no_early_stopping", "no_tolerance", "non_finite_failure_policy", "propagation_dtype",
        "propagation_dtype_note", "required_symbol", "restart_coefficient", "restart_coefficient_derivation",
        "s0_mutation", "snapshot_at_step_320_required", "snapshot_step_320_is_final", "source_manifest_ordinal",
        "source_path", "steps", "steps_type_requirement", "tensor_axes", "update_order",
    }),
    "geometry_contract": frozenset({
        "align_corners_stencil_note", "crop", "crop_clipping_padding_behavior", "flush_bottom_behavior",
        "flush_right_behavior", "image_canvas_coordinate_mapping", "module_is_stdlib_only", "module_torch_free",
        "overlap_membership", "patch_grid_dimensions", "patch_size", "required_symbols",
        "source_manifest_ordinal", "source_path", "stride", "window_enumeration_source", "window_ordering",
    }),
    "internal_upsample_contract": frozenset({
        "align_corners", "align_corners_is_hardcoded", "input_domain", "interpolation_mode", "name",
        "operation_order", "output_domain", "source_function", "source_manifest_ordinal", "source_path",
        "target_size_source",
    }),
    "final_rescale_contract": frozenset({
        "align_corners_frozen_runtime_value", "align_corners_source", "align_corners_source_of_false",
        "internal_upsample_and_final_rescale_are_distinct", "interpolation_mode",
        "mutating_either_value_must_fail_validation", "name", "operation_order", "source_function",
        "source_manifest_ordinal", "source_path", "target_size_source",
        "align_corners_locator_class", "align_corners_locator_attribute",
        "align_corners_locator_assignment_site", "align_corners_locator_required_value",
        "align_corners_locator_verified_by_ast", "align_corners_locator_enclosing_method_enforced",
        "rwr_branch_present_in_historical_file",
        "rwr_branch_used_by_t320_path", "rwr_branch_reachable_from_align_corners_locator",
        "rwr_branch_note", "apply_rwr_to_e3_snapshot_in_required_closure",
        "rwr_branch_providing_sources_excluded", "future_implementation_must_not_import_rwr_branch",
    }),
    "stitching_contract": frozenset({
        "accumulation_formula", "denominator_count_handling", "final_canvas_shape", "overlap_weighting",
        "required_symbol", "rescale_order", "source_manifest_ordinal", "uncovered_pixel_behavior",
    }),
    "prediction_contract": frozenset({
        "argmax_call", "argmax_implementation", "byte_identical_to_350_image_prediction_extraction_claimed",
        "byte_identical_to_350_image_stitching_claimed", "class_axis", "lowest_index_tie_behavior_guaranteed",
        "lowest_index_tie_behavior_note", "output_dtype", "output_shape", "prediction_extraction_function",
        "required_symbol", "source_manifest_ordinal",
    }),
    "gt_isolation": frozenset({
        "gt_may_be_loaded_only_by", "t320_depends_only_on", "t320_must_not_depend_on",
        "verified_gt_isolation_in_source_closure",
    }),
    "historical_350_disclosure": frozenset({
        "artifact_path_symbolic", "contextual_evidence_only", "divergence", "execution_commit",
        "gain_percentage_points", "matched_evidence_for_this_authority", "same_job_computation",
        "scientific_digest", "statement", "t0_miou", "t320_miou",
    }),
    "historical_350_disclosure.divergence": frozenset({
        "complete_end_to_end_identity", "final_rescale", "geometry_relation", "graph_relation",
        "prediction_extraction", "recurrence_relation", "stitching", "unary_kernel_relation",
    }),
    "historical_350_disclosure.statement": frozenset({
        "expected_t320_miou_frozen_by_this_authority", "future_96_image_study_must_measure_its_own_t0_t320_behavior",
        "gain_must_not_be_cited_as_matched_empirical_performance",
        "mutation_marking_350_evidence_as_matched_must_be_rejected_by_loader",
        "the_350_image_run_used_align_corners", "this_authority_uses_final_align_corners",
    }),
    "future_implementation_boundary": frozenset({
        "authorized_after", "future_filenames_chosen_by_this_task", "must_add_new_isolated_source_files",
        "must_expose_pure_testable_functions_for", "must_fail_closed_on_non_finite_values_and_shape_mismatch",
        "must_generate_t0_t320_targets_without_gt",
        "must_include_differential_tests_against_independently_reconstructed_source_behavior",
        "must_leave_existing_e3_scientific_files_unchanged", "must_not_yet_include",
        "must_reproduce_exact_historical_source_contract", "must_reuse_e3_unary_inference_read_only",
    }),
    "closure": frozenset({"complete", "final", "note", "self_referential_hash_created"}),
}


def _validate_all_subtable_schemas(toml_data: dict) -> None:
    for name, expected in _SUBTABLE_KEYS.items():
        parts = name.split(".")
        node = toml_data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                raise T320SourceAuthorityError(f"missing required table: {name}")
            node = node[part]
        _require_keys(node, expected, name)


def validate_source_manifest_binding(toml_data: dict) -> None:
    binding = toml_data["source_manifest_binding"]
    if binding["canonical_encoding"] != CANONICAL_ENCODING_VERSION:
        raise T320SourceAuthorityError(
            f"source_manifest_binding.canonical_encoding must be exactly {CANONICAL_ENCODING_VERSION!r}, "
            f"got {binding['canonical_encoding']!r}"
        )
    if _require_bool(
        binding["canonical_encoding_exact_match_required"],
        "source_manifest_binding.canonical_encoding_exact_match_required",
    ) is not True:
        raise T320SourceAuthorityError(
            "source_manifest_binding.canonical_encoding_exact_match_required must be True"
        )
    if _require_bool(
        binding["required_symbols_ast_verified_for_all_seven_records"],
        "source_manifest_binding.required_symbols_ast_verified_for_all_seven_records",
    ) is not True:
        raise T320SourceAuthorityError(
            "source_manifest_binding.required_symbols_ast_verified_for_all_seven_records must be True"
        )
    grammar = binding["required_symbol_locator_grammar"]
    if "__init__" not in grammar or "module level or as a direct method" not in grammar:
        raise T320SourceAuthorityError(
            "source_manifest_binding.required_symbol_locator_grammar must describe both the bare-symbol "
            "and instance-attribute locator forms"
        )


def validate_authority_schema(toml_data: dict) -> None:
    _require_keys(toml_data, _TOML_TOP_KEYS, "authority")
    _require_keys(toml_data["status"], _STATUS_KEYS, "status")
    _require_keys(toml_data["authorization_boundary"], _AUTHORIZATION_BOUNDARY_KEYS, "authorization_boundary")
    _validate_all_subtable_schemas(toml_data)

    for key in _STATUS_KEYS:
        _require_bool(toml_data["status"][key], f"status.{key}")
    for key in _AUTHORIZATION_BOUNDARY_KEYS:
        _require_bool(toml_data["authorization_boundary"][key], f"authorization_boundary.{key}")

    for key in _AUTHORIZATION_BOUNDARY_KEYS:
        expected = key in _AUTHORIZATION_BOUNDARY_MUST_BE_TRUE
        actual = toml_data["authorization_boundary"][key]
        if actual != expected:
            raise T320SourceAuthorityError(
                f"authorization_boundary.{key} must be {expected}, got {actual}"
            )
    for key in _STATUS_KEYS:
        expected = key in _STATUS_MUST_BE_TRUE
        actual = toml_data["status"][key]
        if actual != expected:
            raise T320SourceAuthorityError(f"status.{key} must be {expected}, got {actual}")


def validate_graph_recurrence_geometry_constants(toml_data: dict) -> None:
    graph = toml_data["graph_contract"]
    k = _require_exact_int(graph["k"], "graph_contract.k")
    if k != 12:
        raise T320SourceAuthorityError(f"graph_contract.k must be 12, got {k}")
    affinity_power = _require_finite_float(graph["affinity_power"], "graph_contract.affinity_power")
    if affinity_power != 3.0:
        raise T320SourceAuthorityError(f"graph_contract.affinity_power must be 3.0, got {affinity_power}")

    recurrence = toml_data["recurrence_contract"]
    alpha = _require_finite_float(recurrence["alpha"], "recurrence_contract.alpha")
    if alpha != 0.98:
        raise T320SourceAuthorityError(f"recurrence_contract.alpha must be 0.98, got {alpha}")
    restart = _require_finite_float(recurrence["restart_coefficient"], "recurrence_contract.restart_coefficient")
    if abs((1.0 - alpha) - restart) > 1e-9:
        raise T320SourceAuthorityError(
            f"recurrence_contract.restart_coefficient ({restart}) is not consistent with 1 - alpha ({1.0 - alpha})"
        )
    steps = _require_exact_int(recurrence["steps"], "recurrence_contract.steps")
    if steps != 320:
        raise T320SourceAuthorityError(f"recurrence_contract.steps must be 320, got {steps}")
    for flag in (
        "no_convergence_test", "no_early_stopping", "no_tolerance", "no_adaptive_step_count",
        "no_cgls_gmres_dense_solve_or_fallback",
    ):
        if recurrence[flag] is not True:
            raise T320SourceAuthorityError(f"recurrence_contract.{flag} must be True, got {recurrence[flag]!r}")
    if recurrence["propagation_dtype"] != "float32":
        raise T320SourceAuthorityError("recurrence_contract.propagation_dtype must be 'float32'")
    if recurrence["accumulation_dtype"] != "float32":
        raise T320SourceAuthorityError("recurrence_contract.accumulation_dtype must be 'float32'")

    geometry = toml_data["geometry_contract"]
    for field, expected in (("crop", [448, 448]), ("stride", [224, 224]), ("patch_size", [14, 14])):
        value = geometry[field]
        if not isinstance(value, list) or len(value) != 2:
            raise T320SourceAuthorityError(f"geometry_contract.{field} must be a 2-element list, got {value!r}")
        for element in value:
            _require_exact_int(element, f"geometry_contract.{field} element")
        if value != expected:
            raise T320SourceAuthorityError(f"geometry_contract.{field} must be {expected}, got {value}")

    directed = graph["directed_symmetric_behavior"]
    if "directed" not in directed.lower() or "not symmetrize" not in directed.lower():
        raise T320SourceAuthorityError(
            "graph_contract.directed_symmetric_behavior must state the graph is directed and NOT symmetrized"
        )
    normalization = graph["normalization_direction"]
    if "row" not in normalization.lower() or "sums to 1" not in normalization.lower():
        raise T320SourceAuthorityError(
            "graph_contract.normalization_direction must state row-wise normalization summing to 1"
        )


def validate_two_distinct_interpolation_operations(toml_data: dict) -> None:
    internal = toml_data["internal_upsample_contract"]
    final = toml_data["final_rescale_contract"]
    if internal["interpolation_mode"] != "bilinear":
        raise T320SourceAuthorityError("internal_upsample_contract.interpolation_mode must be 'bilinear'")
    if final["interpolation_mode"] != "bilinear":
        raise T320SourceAuthorityError("final_rescale_contract.interpolation_mode must be 'bilinear'")
    if "inference.align_corners" not in final["align_corners_source"]:
        raise T320SourceAuthorityError(
            "final_rescale_contract.align_corners_source must reference 'inference.align_corners'"
        )
    internal_align = _require_bool(internal["align_corners"], "internal_upsample_contract.align_corners")
    final_align = _require_bool(
        final["align_corners_frozen_runtime_value"], "final_rescale_contract.align_corners_frozen_runtime_value"
    )
    if internal_align is not True:
        raise T320SourceAuthorityError("internal_upsample_contract.align_corners must be True")
    if final_align is not False:
        raise T320SourceAuthorityError("final_rescale_contract.align_corners_frozen_runtime_value must be False")
    if internal_align == final_align:
        raise T320SourceAuthorityError(
            "internal upsample and final rescale align_corners values must be distinct, "
            f"got internal={internal_align}, final={final_align}"
        )
    if final["internal_upsample_and_final_rescale_are_distinct"] is not True:
        raise T320SourceAuthorityError(
            "final_rescale_contract.internal_upsample_and_final_rescale_are_distinct must be True"
        )
    if final["mutating_either_value_must_fail_validation"] is not True:
        raise T320SourceAuthorityError(
            "final_rescale_contract.mutating_either_value_must_fail_validation must be True"
        )
    if internal["align_corners_is_hardcoded"] is not True:
        raise T320SourceAuthorityError("internal_upsample_contract.align_corners_is_hardcoded must be True")

    if final["align_corners_locator_class"] != ALIGN_CORNERS_LOCATOR_CLASS:
        raise T320SourceAuthorityError(
            f"final_rescale_contract.align_corners_locator_class must be {ALIGN_CORNERS_LOCATOR_CLASS!r}"
        )
    if final["align_corners_locator_attribute"] != ALIGN_CORNERS_LOCATOR_ATTRIBUTE:
        raise T320SourceAuthorityError(
            f"final_rescale_contract.align_corners_locator_attribute must be {ALIGN_CORNERS_LOCATOR_ATTRIBUTE!r}"
        )
    if _require_bool(
        final["align_corners_locator_required_value"], "final_rescale_contract.align_corners_locator_required_value"
    ) is not False:
        raise T320SourceAuthorityError("final_rescale_contract.align_corners_locator_required_value must be False")
    if _require_bool(
        final["align_corners_locator_verified_by_ast"], "final_rescale_contract.align_corners_locator_verified_by_ast"
    ) is not True:
        raise T320SourceAuthorityError("final_rescale_contract.align_corners_locator_verified_by_ast must be True")
    if _require_bool(
        final["align_corners_locator_enclosing_method_enforced"],
        "final_rescale_contract.align_corners_locator_enclosing_method_enforced",
    ) is not True:
        raise T320SourceAuthorityError(
            "final_rescale_contract.align_corners_locator_enclosing_method_enforced must be True"
        )
    if final["align_corners_locator_assignment_site"] != "class initialization path (__init__)":
        raise T320SourceAuthorityError(
            "final_rescale_contract.align_corners_locator_assignment_site must state the __init__ "
            "enclosing method exactly"
        )

    # RWR-branch disclosure: present but unreachable, never smuggled into the closure.
    if _require_bool(
        final["rwr_branch_present_in_historical_file"], "final_rescale_contract.rwr_branch_present_in_historical_file"
    ) is not True:
        raise T320SourceAuthorityError("final_rescale_contract.rwr_branch_present_in_historical_file must be True")
    if _require_bool(
        final["rwr_branch_used_by_t320_path"], "final_rescale_contract.rwr_branch_used_by_t320_path"
    ) is not False:
        raise T320SourceAuthorityError("final_rescale_contract.rwr_branch_used_by_t320_path must be False")
    if _require_bool(
        final["rwr_branch_reachable_from_align_corners_locator"],
        "final_rescale_contract.rwr_branch_reachable_from_align_corners_locator",
    ) is not False:
        raise T320SourceAuthorityError(
            "final_rescale_contract.rwr_branch_reachable_from_align_corners_locator must be False"
        )
    if _require_bool(
        final["apply_rwr_to_e3_snapshot_in_required_closure"],
        "final_rescale_contract.apply_rwr_to_e3_snapshot_in_required_closure",
    ) is not False:
        raise T320SourceAuthorityError(
            "final_rescale_contract.apply_rwr_to_e3_snapshot_in_required_closure must be False"
        )
    if _require_bool(
        final["future_implementation_must_not_import_rwr_branch"],
        "final_rescale_contract.future_implementation_must_not_import_rwr_branch",
    ) is not True:
        raise T320SourceAuthorityError(
            "final_rescale_contract.future_implementation_must_not_import_rwr_branch must be True"
        )
    excluded_rwr_sources = final["rwr_branch_providing_sources_excluded"]
    expected_rwr_sources = [
        "src/open_vocabulary_segmentation/models/dinotext/cover_dr/inference.py",
        "src/open_vocabulary_segmentation/models/dinotext/cover_dr/rwr.py",
    ]
    if list(excluded_rwr_sources) != expected_rwr_sources:
        raise T320SourceAuthorityError(
            f"final_rescale_contract.rwr_branch_providing_sources_excluded must be exactly {expected_rwr_sources}"
        )


_EXPECTED_350_EXECUTION_COMMIT = "804a774588b97e43e0347343089a0209ae28a5c9"
_EXPECTED_350_SCIENTIFIC_DIGEST = "6d75c370bcb03e10698429467053c3a19bc42f1c7405ceb9fccff83e97c8af80"
_EXPECTED_350_T0_MIOU = 28.993948261452340
_EXPECTED_350_T320_MIOU = 30.934420103426838
_EXPECTED_350_GAIN = 1.940471841974498


def validate_historical_350_disclosure(toml_data: dict) -> None:
    disclosure = toml_data["historical_350_disclosure"]
    if disclosure["execution_commit"] != _EXPECTED_350_EXECUTION_COMMIT:
        raise T320SourceAuthorityError(
            f"historical_350_disclosure.execution_commit must be {_EXPECTED_350_EXECUTION_COMMIT}"
        )
    if disclosure["scientific_digest"] != _EXPECTED_350_SCIENTIFIC_DIGEST:
        raise T320SourceAuthorityError(
            f"historical_350_disclosure.scientific_digest must be {_EXPECTED_350_SCIENTIFIC_DIGEST}"
        )
    for field, expected in (
        ("t0_miou", _EXPECTED_350_T0_MIOU),
        ("t320_miou", _EXPECTED_350_T320_MIOU),
        ("gain_percentage_points", _EXPECTED_350_GAIN),
    ):
        actual = _require_finite_float(disclosure[field], f"historical_350_disclosure.{field}")
        if abs(actual - expected) > 1e-6:
            raise T320SourceAuthorityError(
                f"historical_350_disclosure.{field} must be {expected}, got {actual}"
            )
    if disclosure["contextual_evidence_only"] is not True:
        raise T320SourceAuthorityError("historical_350_disclosure.contextual_evidence_only must be True")
    if disclosure["matched_evidence_for_this_authority"] is not False:
        raise T320SourceAuthorityError("historical_350_disclosure.matched_evidence_for_this_authority must be False")
    if disclosure["same_job_computation"] is not True:
        raise T320SourceAuthorityError("historical_350_disclosure.same_job_computation must be True")

    # [historical_350_disclosure.divergence] and [.statement] are nested
    # dotted-table keys under [historical_350_disclosure], as tomllib parses them.
    div = toml_data["historical_350_disclosure"]
    if "divergence" not in div:
        raise T320SourceAuthorityError("historical_350_disclosure.divergence subtable is required")
    sub_divergence = div["divergence"]
    if sub_divergence["complete_end_to_end_identity"] is not False:
        raise T320SourceAuthorityError(
            "historical_350_disclosure.divergence.complete_end_to_end_identity must be False"
        )
    for field in ("unary_kernel_relation", "graph_relation", "geometry_relation"):
        if sub_divergence[field] != "byte-identical":
            raise T320SourceAuthorityError(f"historical_350_disclosure.divergence.{field} must be 'byte-identical'")
    if sub_divergence["recurrence_relation"] != "byte-identical kernel":
        raise T320SourceAuthorityError(
            "historical_350_disclosure.divergence.recurrence_relation must be 'byte-identical kernel'"
        )

    sub_statement = div["statement"]
    if sub_statement["this_authority_uses_final_align_corners"] is not False:
        raise T320SourceAuthorityError(
            "historical_350_disclosure.statement.this_authority_uses_final_align_corners must be False"
        )
    if sub_statement["the_350_image_run_used_align_corners"] is not True:
        raise T320SourceAuthorityError(
            "historical_350_disclosure.statement.the_350_image_run_used_align_corners must be True"
        )
    if sub_statement["gain_must_not_be_cited_as_matched_empirical_performance"] is not True:
        raise T320SourceAuthorityError(
            "historical_350_disclosure.statement.gain_must_not_be_cited_as_matched_empirical_performance must be True"
        )
    if sub_statement["expected_t320_miou_frozen_by_this_authority"] is not False:
        raise T320SourceAuthorityError(
            "historical_350_disclosure.statement.expected_t320_miou_frozen_by_this_authority must be False"
        )
    if sub_statement["mutation_marking_350_evidence_as_matched_must_be_rejected_by_loader"] is not True:
        raise T320SourceAuthorityError(
            "historical_350_disclosure.statement.mutation_marking_350_evidence_as_matched_must_be_rejected_by_loader must be True"
        )


def validate_gt_isolation(toml_data: dict) -> None:
    gt = toml_data["gt_isolation"]
    forbidden = {
        "annotations", "GT labels", "ignore masks", "beneficial/harmful categories",
        "correctness targets", "discovery/holdout outcomes", "intervention scores",
        "future evaluation metrics",
    }
    declared_forbidden = set(gt["t320_must_not_depend_on"])
    if declared_forbidden != forbidden:
        raise T320SourceAuthorityError(
            f"gt_isolation.t320_must_not_depend_on does not match the required closed set; "
            f"missing={forbidden - declared_forbidden}, extra={declared_forbidden - forbidden}"
        )


def validate_prediction_contract(toml_data: dict) -> None:
    prediction = toml_data["prediction_contract"]
    if prediction["byte_identical_to_350_image_stitching_claimed"] is not False:
        raise T320SourceAuthorityError(
            "prediction_contract.byte_identical_to_350_image_stitching_claimed must be False"
        )
    if prediction["byte_identical_to_350_image_prediction_extraction_claimed"] is not False:
        raise T320SourceAuthorityError(
            "prediction_contract.byte_identical_to_350_image_prediction_extraction_claimed must be False"
        )
    if prediction["lowest_index_tie_behavior_guaranteed"] is not False:
        raise T320SourceAuthorityError("prediction_contract.lowest_index_tie_behavior_guaranteed must be False")


# ---------------------------------------------------------------------------
# Manifest validation (schema + Git-object re-derivation)
# ---------------------------------------------------------------------------

def validate_manifest_schema(manifest: dict) -> None:
    _require_keys(manifest, _MANIFEST_TOP_KEYS, "manifest")
    _require_keys(manifest["schema"], frozenset({"name", "version"}), "manifest.schema")

    if manifest["schema"]["name"] != "semantic_evidence_t320_source_manifest":
        raise T320SourceAuthorityError("manifest.schema.name mismatch")
    if manifest["complete"] is not True or manifest["final"] is not True:
        raise T320SourceAuthorityError("manifest.complete and manifest.final must both be True")

    encoding = _require_str(manifest["canonical_encoding"], "manifest.canonical_encoding")
    if encoding != CANONICAL_ENCODING_VERSION:
        raise T320SourceAuthorityError(
            f"manifest.canonical_encoding must be exactly {CANONICAL_ENCODING_VERSION!r}, got {encoding!r}; "
            f"the loader implements only {CANONICAL_ENCODING_VERSION!r} and does not select encoding "
            f"behavior dynamically from an untrusted label"
        )

    _require_full_commit_hex(manifest["source_commit"], "manifest.source_commit")
    _require_full_commit_hex(manifest["reviewed_e3_base_commit"], "manifest.reviewed_e3_base_commit")

    records = manifest["file_records"]
    if not isinstance(records, list):
        raise T320SourceAuthorityError("manifest.file_records must be a list")
    if len(records) != _EXPECTED_FILE_COUNT:
        raise T320SourceAuthorityError(
            f"manifest.file_records must contain exactly {_EXPECTED_FILE_COUNT} records, got {len(records)}"
        )

    seen_ordinals: set[int] = set()
    seen_paths: set[str] = set()
    for index, record in enumerate(records):
        _require_keys(record, _FILE_RECORD_KEYS, f"manifest.file_records[{index}]")
        ordinal = _require_exact_int(record["ordinal"], f"manifest.file_records[{index}].ordinal")
        if ordinal != index + 1:
            raise T320SourceAuthorityError(
                f"manifest.file_records must be in strict ordinal order 1..{_EXPECTED_FILE_COUNT}; "
                f"position {index} has ordinal {ordinal}"
            )
        if ordinal in seen_ordinals:
            raise T320SourceAuthorityError(f"duplicate ordinal {ordinal} in manifest.file_records")
        seen_ordinals.add(ordinal)

        path = _require_str(record["path"], f"manifest.file_records[{index}].path")
        _reject_unsafe_relative_path(path)
        if path in seen_paths:
            raise T320SourceAuthorityError(f"duplicate path {path!r} in manifest.file_records")
        seen_paths.add(path)
        if path == _WINDOW_CACHE_PATH:
            raise T320SourceAuthorityError(
                "window_cache.py must never appear in manifest.file_records; it is excluded source"
            )

        _require_full_commit_hex(record["source_commit"], f"manifest.file_records[{index}].source_commit")
        _require_str(record["blob_id"], f"manifest.file_records[{index}].blob_id")
        sha = _require_str(record["sha256"], f"manifest.file_records[{index}].sha256")
        if len(sha) != 64 or any(ch not in _FULL_HEX40 for ch in sha.lower()):
            raise T320SourceAuthorityError(f"manifest.file_records[{index}].sha256 malformed: {sha!r}")

        classification = _require_str(record["classification"], f"manifest.file_records[{index}].classification")
        if classification not in _VALID_CLASSIFICATIONS:
            raise T320SourceAuthorityError(
                f"manifest.file_records[{index}].classification {classification!r} not in {sorted(_VALID_CLASSIFICATIONS)}"
            )
        if _require_bool(record["required"], f"manifest.file_records[{index}].required") is not True:
            raise T320SourceAuthorityError(f"manifest.file_records[{index}].required must be True (all 7 are required)")

        symbols = record["required_symbols"]
        if not isinstance(symbols, list) or not symbols or not all(isinstance(s, str) and s for s in symbols):
            raise T320SourceAuthorityError(
                f"manifest.file_records[{index}].required_symbols must be a non-empty list of non-empty strings"
            )
        _require_str(record["reason_required"], f"manifest.file_records[{index}].reason_required")

        if path == _DINOTEXT_SEG_PATH:
            if _BROAD_DINOTEXT_SEG_CLASS_SYMBOL in symbols and _ALIGN_CORNERS_LOCATOR_SYMBOL not in symbols:
                raise T320SourceAuthorityError(
                    f"manifest.file_records[{index}] ({path}) must use the narrow locator "
                    f"{_ALIGN_CORNERS_LOCATOR_SYMBOL!r}, not the broad class-level symbol "
                    f"{_BROAD_DINOTEXT_SEG_CLASS_SYMBOL!r}"
                )
            if symbols != [_ALIGN_CORNERS_LOCATOR_SYMBOL]:
                raise T320SourceAuthorityError(
                    f"manifest.file_records[{index}] ({path}).required_symbols must be exactly "
                    f"[{_ALIGN_CORNERS_LOCATOR_SYMBOL!r}], got {symbols!r}"
                )

    if tuple(record["path"] for record in records) != _EXPECTED_RELATIVE_PATHS:
        raise T320SourceAuthorityError(
            "manifest.file_records paths do not match the audit-recovered seven-file closure; "
            f"got {tuple(record['path'] for record in records)}"
        )

    excluded = manifest["excluded_source"]
    if not isinstance(excluded, list) or not excluded:
        raise T320SourceAuthorityError("manifest.excluded_source must be a non-empty list")
    excluded_paths = {entry["path"] for entry in excluded}
    if _WINDOW_CACHE_PATH not in excluded_paths:
        raise T320SourceAuthorityError("manifest.excluded_source must list window_cache.py")
    if excluded_paths & seen_paths:
        raise T320SourceAuthorityError(
            f"manifest.excluded_source and manifest.file_records must be disjoint; overlap: {excluded_paths & seen_paths}"
        )


def verify_manifest_against_git(repo_root: Path, manifest: dict) -> dict:
    """Independently re-derive every commit/tree/blob/sha256/digest claim in
    the manifest directly from Git objects. Raises on any mismatch."""
    source_commit = manifest["source_commit"]
    if not git_commit_exists(repo_root, source_commit):
        raise T320SourceAuthorityError(f"source commit {source_commit} does not exist as a commit object")
    actual_tree = git_tree_id(repo_root, source_commit)
    if actual_tree != manifest["source_tree_id"]:
        raise T320SourceAuthorityError(
            f"source tree id mismatch: manifest says {manifest['source_tree_id']}, git says {actual_tree}"
        )
    if not git_is_ancestor(repo_root, manifest["reviewed_e3_base_commit"], source_commit):
        raise T320SourceAuthorityError(
            f"reviewed_e3_base_commit {manifest['reviewed_e3_base_commit']} is not an ancestor of source_commit {source_commit}"
        )

    per_file_results = []
    for record in manifest["file_records"]:
        actual_blob = git_blob_id(repo_root, record["source_commit"], record["path"])
        if actual_blob != record["blob_id"]:
            raise T320SourceAuthorityError(
                f"blob id mismatch for {record['path']}: manifest says {record['blob_id']}, git says {actual_blob}"
            )
        actual_sha256 = git_blob_sha256(repo_root, record["source_commit"], record["path"])
        if actual_sha256 != record["sha256"]:
            raise T320SourceAuthorityError(
                f"sha256 mismatch for {record['path']}: manifest says {record['sha256']}, git recomputation says {actual_sha256}"
            )
        verify_required_symbols(repo_root, record["source_commit"], record["path"], record["required_symbols"])
        per_file_results.append({"path": record["path"], "blob_id": actual_blob, "sha256": actual_sha256})

    recomputed_digest = compute_canonical_source_closure_digest(manifest)
    if recomputed_digest != manifest["canonical_source_closure_digest"]:
        raise T320SourceAuthorityError(
            "canonical_source_closure_digest mismatch: manifest says "
            f"{manifest['canonical_source_closure_digest']}, recomputation says {recomputed_digest}"
        )

    return {
        "source_commit": source_commit,
        "source_tree_id": actual_tree,
        "per_file": per_file_results,
        "canonical_source_closure_digest": recomputed_digest,
    }


def verify_current_repository_prerequisite(repo_root: Path, toml_data: dict) -> dict:
    """Verify the CURRENT repository (branch, HEAD, ancestry, cleanliness,
    committed feasibility-contract file hashes) against [prerequisite]."""
    prereq = toml_data["prerequisite"]

    actual_branch = _run_git_text(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    if actual_branch != prereq["required_branch"]:
        raise T320SourceAuthorityError(
            f"branch mismatch: required {prereq['required_branch']!r}, got {actual_branch!r}"
        )
    actual_head = _run_git_text(repo_root, "rev-parse", "HEAD")
    if actual_head != prereq["required_head"]:
        raise T320SourceAuthorityError(
            f"HEAD mismatch: required {prereq['required_head']!r}, got {actual_head!r}"
        )
    if not git_is_ancestor(repo_root, prereq["reviewed_e3_base_commit"], actual_head):
        raise T320SourceAuthorityError("reviewed_e3_base_commit is not an ancestor of current HEAD")

    status_output = _run_git_text(repo_root, "status", "--short")
    if status_output != "":
        raise T320SourceAuthorityError(f"working tree is not clean: {status_output!r}")

    feasibility_files = {
        "research_specs/semantic_evidence_feasibility_v1.toml": prereq["feasibility_contract_toml_sha256"],
        "evaluation_subsets/semantic_evidence_feasibility_96.json": prereq["feasibility_contract_subset_sha256"],
        "src/semantic_evidence_feasibility_identity.py": prereq["feasibility_contract_identity_py_sha256"],
        "tests/test_semantic_evidence_feasibility_identity.py": prereq["feasibility_contract_tests_sha256"],
        "docs/semantic_evidence_feasibility_protocol.md": prereq["feasibility_contract_docs_sha256"],
    }
    for relative_path, expected_sha in feasibility_files.items():
        resolved = resolve_repo_relative_path(repo_root, relative_path)
        actual_sha = sha256_file(resolved)
        if actual_sha != expected_sha:
            raise T320SourceAuthorityError(
                f"feasibility-contract file changed: {relative_path} expected sha256={expected_sha}, got {actual_sha}"
            )

    return {"branch": actual_branch, "head": actual_head, "working_tree_clean": True}


def verify_manifest_file_binding(repo_root: Path, toml_data: dict) -> str:
    """Verify the on-disk manifest JSON file's own sha256 matches the TOML's
    recorded [source_manifest_binding].manifest_file_sha256."""
    binding = toml_data["source_manifest_binding"]
    manifest_path = resolve_repo_relative_path(repo_root, binding["manifest_path"])
    actual = sha256_file(manifest_path)
    if actual != binding["manifest_file_sha256"]:
        raise T320SourceAuthorityError(
            f"manifest file sha256 mismatch: TOML says {binding['manifest_file_sha256']}, got {actual}"
        )
    return actual


def verify_toml_binds_same_digest_as_manifest(toml_data: dict, manifest: dict) -> None:
    toml_digest = toml_data["source_manifest_binding"]["canonical_source_closure_digest"]
    manifest_digest = manifest["canonical_source_closure_digest"]
    if toml_digest != manifest_digest:
        raise T320SourceAuthorityError(
            f"TOML canonical_source_closure_digest ({toml_digest}) does not match "
            f"manifest canonical_source_closure_digest ({manifest_digest})"
        )


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------

def load_authority(toml_path: Path) -> dict:
    data = load_toml(toml_path)
    validate_authority_schema(data)
    validate_source_manifest_binding(data)
    validate_graph_recurrence_geometry_constants(data)
    validate_two_distinct_interpolation_operations(data)
    validate_historical_350_disclosure(data)
    validate_gt_isolation(data)
    validate_prediction_contract(data)
    return data


def load_manifest(manifest_path: Path) -> dict:
    data = load_json_strict(manifest_path)
    validate_manifest_schema(data)
    return data


def verify_all(repo_root: Path, toml_path: Path, manifest_path: Path) -> dict:
    """Full independent verification pipeline: schema, Git-object
    re-derivation, canonical-digest reconstruction, cross-binding, and
    current-repository prerequisite checks. Raises T320SourceAuthorityError
    on any violation; never accesses images, annotations, a model, or a
    checkpoint tensor; never imports torch."""
    toml_data = load_authority(toml_path)
    manifest = load_manifest(manifest_path)

    prerequisite_result = verify_current_repository_prerequisite(repo_root, toml_data)
    manifest_file_sha256 = verify_manifest_file_binding(repo_root, toml_data)
    verify_toml_binds_same_digest_as_manifest(toml_data, manifest)
    git_result = verify_manifest_against_git(repo_root, manifest)

    if toml_data["historical_source"]["source_commit"] != manifest["source_commit"]:
        raise T320SourceAuthorityError("TOML and manifest disagree on source_commit")
    if toml_data["historical_source"]["source_tree_id"] != manifest["source_tree_id"]:
        raise T320SourceAuthorityError("TOML and manifest disagree on source_tree_id")

    return {
        "prerequisite": prerequisite_result,
        "manifest_file_sha256": manifest_file_sha256,
        "git": git_result,
        "source_authority_complete": toml_data["status"]["source_authority_complete"],
        "t320_execution_authorized": toml_data["status"]["t320_execution_authorized"],
        "isolated_future_implementation_authorized_after_commit_and_independent_verification": toml_data["status"][
            "isolated_future_implementation_authorized_after_commit_and_independent_verification"
        ],
    }


def is_t320_execution_authorized(toml_data: dict) -> bool:
    return bool(toml_data["status"]["t320_execution_authorized"])


def report_t320_source_authority(toml_data: dict, manifest: dict) -> dict:
    """A read-only, non-authorizing summary for external reporting."""
    return {
        "source_branch_informational": toml_data["historical_source"]["source_branch_informational"],
        "source_commit": toml_data["historical_source"]["source_commit"],
        "source_tree_id": toml_data["historical_source"]["source_tree_id"],
        "closure_file_count": len(manifest["file_records"]),
        "canonical_source_closure_digest": manifest["canonical_source_closure_digest"],
        "internal_upsample_align_corners": toml_data["internal_upsample_contract"]["align_corners"],
        "final_rescale_align_corners": toml_data["final_rescale_contract"]["align_corners_frozen_runtime_value"],
        "t320_execution_authorized": toml_data["status"]["t320_execution_authorized"],
        "isolated_future_implementation_authorized_after_commit_and_independent_verification": toml_data["status"][
            "isolated_future_implementation_authorized_after_commit_and_independent_verification"
        ],
    }
