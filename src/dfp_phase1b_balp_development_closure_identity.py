"""Strict loading/validation for the DFP Phase 1B BALP development-
closure identity
(`evaluation_identities/e24_dfp_phase1b_balp_development_closure.toml`,
"E24").

E24 freezes the DETERMINISTIC INTERPRETATION of one already-completed,
already-independently-verified real execution (SLURM job 22071646,
authorized by E23). It introduces no new scientific formula, no new
metric, no new threshold, and no relaxed eligibility rule -- E22's own
frozen strict eligibility rule is re-pinned by reference and applied
exactly as written. E24 authorizes no execution of any kind: it runs
no inference, submits no GPU job, constructs no model, and re-derives
no confusion matrix or metric the source checkpoint does not already
contain.

This module implements no method, runs no inference, and performs no
torch/CUDA/model/dataset access -- pure TOML/filesystem/hash
validation and exact comparison only.

`KeyboardInterrupt`, `SystemExit`, and `MemoryError` always propagate
unmodified; ordinary parse/type/path/relational failures are wrapped in
`DfpPhase1bBalpDevelopmentClosureIdentityError`.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping


class DfpPhase1bBalpDevelopmentClosureIdentityError(ValueError):
    """Raised on any E24 identity invariant violation. Always fail
    closed. Never raised for, and never catches, KeyboardInterrupt /
    SystemExit / MemoryError -- those always propagate unmodified."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e24_dfp_phase1b_balp_development_closure.toml")
SCHEMA_NAME = "talk2dino-dfp-phase1b-balp-development-closure-identity-v1"
SCHEMA_VERSION = 1
REQUIRED_ANCESTOR_COMMIT = "804a774588b97e43e0347343089a0209ae28a5c9"
EXPECTED_WORKING_BRANCH = "dfp-phase1a-diagnostics"
EXPECTED_SUCCESSOR_OF = "e23-dfp-phase1b-balp-development-execution"

SLURM_JOB_ID = "22071646"
SCIENTIFIC_OUTPUT_DIGEST = "6d75c370bcb03e10698429467053c3a19bc42f1c7405ceb9fccff83e97c8af80"
DEVELOPMENT_RESULT_SHA256 = "cfc722402f2df12e711337284d7dbc4b24025ece1e8ab18164dac69ecda319e5"
CHECKPOINT_JSON_SHA256 = "e060126df5ff3d94588b525c1ccc9aa2fc6f6d9969103df0f9ccda5f85b1a3c1"
CHECKPOINT_ARRAYS_SHA256 = "3d2062f3acb6e6baab0c89b2b5276efd7709e4870b124cbcc45626c53e52a6d7"

DEVELOPMENT_IMAGES = 350
DEVELOPMENT_WINDOWS = 769
EVALUATED_PIXELS = 91454742
CONTROLS = 5
CANDIDATES = 16

ELIGIBILITY_RULE = "development_miou_strictly_exceeds_t320"
ELIGIBLE_CANDIDATE_COUNT = 0
DEVELOPMENT_OUTCOME = "NO_BALP_CANDIDATE_PROCEEDS"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT40 = re.compile(r"^[0-9a-f]{40}$")

_REQUIRED_PROHIBITED = (
    "new_threshold", "relaxed_greater_equal_rule", "candidate_selection", "replacement_candidate",
    "confirmation_execution", "full_evaluation_exception", "fabricated_profiling", "post_hoc_scientific_method_change",
)


# ---------------------------------------------------------------------
# Small closed-schema primitives -- the same convention every other
# E-identity loader in this repository uses.
# ---------------------------------------------------------------------


def _require_string(value: Any, label: str) -> str:
    if type(value) is not str:
        raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"{label} must be a string, observed {type(value).__name__}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    s = _require_string(value, label)
    if not _HEX64.match(s):
        raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"{label} must be lowercase 64-hex, observed {value!r}")
    return s


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"{label} must be a mapping, observed {type(value).__name__}")
    return value


def _require_closed(node: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    node = _require_mapping(node, label)
    actual = frozenset(node.keys())
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
    if missing or unknown:
        raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"{label} has the wrong key set -- missing={missing} unknown={unknown}")
    return node


def _require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"{label} must be an int (not bool/float), observed {type(value).__name__}: {value!r}")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"{label} must be a bool, observed {type(value).__name__}: {value!r}")
    return value


def _require_exact_value(observed: Any, expected: Any, label: str) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise DfpPhase1bBalpDevelopmentClosureIdentityError(
            f"{label}: frozen value drift -- expected {expected!r} ({type(expected).__name__}), observed {observed!r} ({type(observed).__name__})"
        )


_IDENTITY_KEYS = frozenset({
    "name", "schema_name", "schema_version", "description", "required_ancestor_commit",
    "working_branch", "successor_of", "mutates_or_version_bumps_earlier_identity",
})


def _verify_identity_block(node: Any) -> None:
    node = _require_closed(node, _IDENTITY_KEYS, "identity")
    _require_exact_value(_require_string(node["name"], "identity.name"), "e24-dfp-phase1b-balp-development-closure", "identity.name")
    _require_exact_value(_require_string(node["schema_name"], "identity.schema_name"), SCHEMA_NAME, "identity.schema_name")
    _require_exact_value(_require_exact_int(node["schema_version"], "identity.schema_version"), SCHEMA_VERSION, "identity.schema_version")
    _require_string(node["description"], "identity.description")
    commit = _require_string(node["required_ancestor_commit"], "identity.required_ancestor_commit")
    if not _COMMIT40.match(commit):
        raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"identity.required_ancestor_commit must be 40-hex, observed {commit!r}")
    _require_exact_value(commit, REQUIRED_ANCESTOR_COMMIT, "identity.required_ancestor_commit")
    _require_exact_value(_require_string(node["working_branch"], "identity.working_branch"), EXPECTED_WORKING_BRANCH, "identity.working_branch")
    _require_exact_value(_require_string(node["successor_of"], "identity.successor_of"), EXPECTED_SUCCESSOR_OF, "identity.successor_of")
    _require_exact_value(_require_bool(node["mutates_or_version_bumps_earlier_identity"], "identity.mutates_or_version_bumps_earlier_identity"), False, "identity.mutates_or_version_bumps_earlier_identity")


_ORDINARY_BINDING_KEYS = frozenset({"relative_path", "sha256", "binds"})
_WINDOW_UNIVERSE_BINDING_KEYS = frozenset({"relative_path", "sha256", "image_count", "window_count", "canonical_record_digest", "binds"})
_ORDINARY_BINDING_NAMES = frozenset({"e22_preservation", "e22_artifacts", "e23", "balp_split_artifact"})
_WINDOW_UNIVERSE_BINDING_NAME = "development_window_universe"
_ALL_BINDING_NAMES = _ORDINARY_BINDING_NAMES | {_WINDOW_UNIVERSE_BINDING_NAME}


def _resolve_repo_relative(root: Path, relative_path: str, *, label: str) -> Path:
    candidate = (root / relative_path).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"{label}: relative_path escapes the repository root: {relative_path!r}") from exc
    return candidate


def _verify_dependency_bindings(root: Path, node: Any) -> None:
    node = _require_closed(node, _ALL_BINDING_NAMES, "dependency_bindings")
    for name in _ORDINARY_BINDING_NAMES:
        binding = _require_closed(node[name], _ORDINARY_BINDING_KEYS, f"dependency_bindings.{name}")
        rel = _require_string(binding["relative_path"], f"dependency_bindings.{name}.relative_path")
        expected_hash = _require_sha256(binding["sha256"], f"dependency_bindings.{name}.sha256")
        _require_string(binding["binds"], f"dependency_bindings.{name}.binds")
        target = _resolve_repo_relative(root, rel, label=f"dependency_bindings.{name}")
        try:
            observed_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as exc:
            raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"dependency_bindings.{name}: cannot read bound file {rel}: {exc}") from exc
        if observed_hash != expected_hash:
            raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"dependency_bindings.{name}: bound file {rel} sha256 drift -- expected {expected_hash}, observed {observed_hash}")

    wu = _require_closed(node[_WINDOW_UNIVERSE_BINDING_NAME], _WINDOW_UNIVERSE_BINDING_KEYS, f"dependency_bindings.{_WINDOW_UNIVERSE_BINDING_NAME}")
    rel = _require_string(wu["relative_path"], "dependency_bindings.development_window_universe.relative_path")
    expected_hash = _require_sha256(wu["sha256"], "dependency_bindings.development_window_universe.sha256")
    target = _resolve_repo_relative(root, rel, label="dependency_bindings.development_window_universe")
    try:
        observed_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError as exc:
        raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"dependency_bindings.development_window_universe: cannot read {rel}: {exc}") from exc
    if observed_hash != expected_hash:
        raise DfpPhase1bBalpDevelopmentClosureIdentityError("dependency_bindings.development_window_universe: bound file sha256 drift")
    _require_exact_value(_require_exact_int(wu["image_count"], "..image_count"), DEVELOPMENT_IMAGES, "dependency_bindings.development_window_universe.image_count")
    _require_exact_value(_require_exact_int(wu["window_count"], "..window_count"), DEVELOPMENT_WINDOWS, "dependency_bindings.development_window_universe.window_count")
    _require_sha256(wu["canonical_record_digest"], "dependency_bindings.development_window_universe.canonical_record_digest")
    _require_string(wu["binds"], "dependency_bindings.development_window_universe.binds")


_SOURCE_EXECUTION_KEYS = frozenset({
    "slurm_job_id", "slurm_job_state", "slurm_exit_code", "slurm_elapsed",
    "development_result_sha256", "checkpoint_json_sha256", "checkpoint_arrays_sha256", "scientific_output_digest",
})


def _verify_source_execution(node: Any) -> None:
    node = _require_closed(node, _SOURCE_EXECUTION_KEYS, "source_execution")
    _require_exact_value(_require_string(node["slurm_job_id"], "..slurm_job_id"), SLURM_JOB_ID, "source_execution.slurm_job_id")
    _require_exact_value(_require_string(node["slurm_job_state"], "..slurm_job_state"), "COMPLETED", "source_execution.slurm_job_state")
    _require_exact_value(_require_string(node["slurm_exit_code"], "..slurm_exit_code"), "0:0", "source_execution.slurm_exit_code")
    _require_exact_value(_require_string(node["slurm_elapsed"], "..slurm_elapsed"), "03:22:29", "source_execution.slurm_elapsed")
    _require_exact_value(_require_sha256(node["development_result_sha256"], "..development_result_sha256"), DEVELOPMENT_RESULT_SHA256, "source_execution.development_result_sha256")
    _require_exact_value(_require_sha256(node["checkpoint_json_sha256"], "..checkpoint_json_sha256"), CHECKPOINT_JSON_SHA256, "source_execution.checkpoint_json_sha256")
    _require_exact_value(_require_sha256(node["checkpoint_arrays_sha256"], "..checkpoint_arrays_sha256"), CHECKPOINT_ARRAYS_SHA256, "source_execution.checkpoint_arrays_sha256")
    _require_exact_value(_require_sha256(node["scientific_output_digest"], "..scientific_output_digest"), SCIENTIFIC_OUTPUT_DIGEST, "source_execution.scientific_output_digest")


_POPULATION_KEYS = frozenset({"development_images", "development_windows", "evaluated_pixels", "controls", "candidates"})


def _verify_population(node: Any) -> None:
    node = _require_closed(node, _POPULATION_KEYS, "population")
    _require_exact_value(_require_exact_int(node["development_images"], "..development_images"), DEVELOPMENT_IMAGES, "population.development_images")
    _require_exact_value(_require_exact_int(node["development_windows"], "..development_windows"), DEVELOPMENT_WINDOWS, "population.development_windows")
    _require_exact_value(_require_exact_int(node["evaluated_pixels"], "..evaluated_pixels"), EVALUATED_PIXELS, "population.evaluated_pixels")
    _require_exact_value(_require_exact_int(node["controls"], "..controls"), CONTROLS, "population.controls")
    _require_exact_value(_require_exact_int(node["candidates"], "..candidates"), CANDIDATES, "population.candidates")


_FROZEN_DECISION_KEYS = frozenset({
    "eligibility_rule", "eligible_candidate_count", "development_outcome",
    "confirmation_execution_authorized", "full_5000_balp_execution_authorized",
    "profiling_status", "stage_bottleneck_claim_authorized",
})


def _verify_frozen_decision(node: Any) -> None:
    node = _require_closed(node, _FROZEN_DECISION_KEYS, "frozen_decision")
    _require_exact_value(_require_string(node["eligibility_rule"], "..eligibility_rule"), ELIGIBILITY_RULE, "frozen_decision.eligibility_rule")
    _require_exact_value(_require_exact_int(node["eligible_candidate_count"], "..eligible_candidate_count"), ELIGIBLE_CANDIDATE_COUNT, "frozen_decision.eligible_candidate_count")
    _require_exact_value(_require_string(node["development_outcome"], "..development_outcome"), DEVELOPMENT_OUTCOME, "frozen_decision.development_outcome")
    _require_exact_value(_require_bool(node["confirmation_execution_authorized"], "..confirmation_execution_authorized"), False, "frozen_decision.confirmation_execution_authorized")
    _require_exact_value(_require_bool(node["full_5000_balp_execution_authorized"], "..full_5000_balp_execution_authorized"), False, "frozen_decision.full_5000_balp_execution_authorized")
    _require_exact_value(_require_string(node["profiling_status"], "..profiling_status"), "NOT_RECORDED", "frozen_decision.profiling_status")
    _require_exact_value(_require_bool(node["stage_bottleneck_claim_authorized"], "..stage_bottleneck_claim_authorized"), False, "frozen_decision.stage_bottleneck_claim_authorized")


def verify_repository_gate(repo_root: Path, repository_commit: str) -> None:
    commit = _require_string(repository_commit, "repository_commit")
    if not _COMMIT40.match(commit):
        raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"repository_commit must be a 40-hex commit, observed {commit!r}")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", REQUIRED_ANCESTOR_COMMIT, commit],
            capture_output=True, text=True,
        )
    except OSError as exc:
        raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"cannot invoke git for ancestor verification: {exc}") from exc
    if result.returncode != 0:
        raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"required_ancestor_commit {REQUIRED_ANCESTOR_COMMIT} is not an ancestor of (or equal to) {commit}")


_TOP_LEVEL_KEYS = frozenset({
    "format_version", "identity", "dependency_bindings", "source_execution", "population", "frozen_decision", "prohibited",
})


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load, closed-schema-validate, exact-value-validate, and
    relationally hash-verify the E24 BALP development-closure
    identity. Never mutates or re-derives any bound dependency -- only
    re-verifies SHA-256/content against current bytes. Performs no
    torch/CUDA/model/dataset access."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with open(source, "rb") as fh:
            document = tomllib.load(fh)
    except OSError as exc:
        raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"cannot read E24 identity {source}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"E24 identity {source} is not valid TOML: {exc}") from exc

    _require_closed(document, _TOP_LEVEL_KEYS, "<top level>")
    _require_exact_value(
        _require_string(document["format_version"], "format_version"),
        "talk2dino-dfp-phase1b-balp-development-closure-identity-v1", "format_version",
    )
    _verify_identity_block(document["identity"])
    _verify_dependency_bindings(root, document["dependency_bindings"])
    _verify_source_execution(document["source_execution"])
    _verify_population(document["population"])
    _verify_frozen_decision(document["frozen_decision"])

    prohibited = _require_closed(document["prohibited"], frozenset({"list"}), "prohibited")
    prohibited_list = prohibited["list"]
    if not isinstance(prohibited_list, list) or not all(type(x) is str for x in prohibited_list):
        raise DfpPhase1bBalpDevelopmentClosureIdentityError("prohibited.list must be a list of strings")
    if tuple(prohibited_list) != _REQUIRED_PROHIBITED:
        raise DfpPhase1bBalpDevelopmentClosureIdentityError(f"prohibited.list disagrees with the frozen required list, observed {prohibited_list!r}")

    return document
