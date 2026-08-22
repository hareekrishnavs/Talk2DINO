"""Static preflight and strict result/checkpoint verification for the
bounded k11/k12 finite-step stability gate.

This module deliberately avoids importing MMCV, Torch, model code, or
dataset code. All diagnostic-only gate settings (sample-selection window
count, reference window counts, snapshot steps, tolerances, gate
thresholds) live in the authoritative TOML at
:data:`IDENTITY_RELATIVE_PATH`; this module never hardcodes one of its own
-- it only defines schema names, key sets, and validation logic, and
relationally validates the gate identity against the parent matched
k11/k12 identity it hashes and references.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tomllib
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from src.matched_k11_k12_identity import load_identity as _load_matched_identity


IDENTITY_RELATIVE_PATH = Path(
    "evaluation_identities/e12_k11_k12_stability_gate.toml"
)
RESULT_SCHEMA_NAME = "talk2dino-k11-k12-stability-gate-v1"
CHECKPOINT_SCHEMA_NAME = "talk2dino-k11-k12-stability-checkpoint-v1"
SUPPORTED_PROPAGATION_METHOD = "finite_power_iteration"
SUPPORTED_COMPUTE_DTYPE = "float32"
SUPPORTED_REFERENCE_DTYPE = "float64"
SUPPORTED_INITIAL_ITERATE = "S0"
SUPPORTED_SAMPLE_SELECTION_RULE = "first_n_canonical_windows_in_canonical_image_then_window_order"
SUPPORTED_IMAGE_ORDER_SOURCE = "coco_stuff_164k_validation_canonical_dataset_order"
SUPPORTED_WINDOW_ORDER_SOURCE = "sliding_window_geometry_slidingwindowplan_build_row_major_flat_index"
SUPPORTED_REGIME_CLASSIFICATIONS = (
    "CLEAR_MATCHED_SIGNAL",
    "NUMERICALLY_STABLE_BUT_EFFECT_NEAR_NOISE",
    "NUMERICALLY_STABLE_BUT_REGIME_INCONCLUSIVE",
    "TRUNCATION_SENSITIVE",
    "INVALID",
)
# The implemented bounded harness protocol: these are not arbitrary
# positive integers, they define what the gate *is*. Changing any of them
# defines a different gate and must be a deliberate code change, never a
# silent TOML edit -- the authoritative TOML must still register the same
# values (this is checked relationally in ``load_identity`` below).
SUPPORTED_CANONICAL_WINDOW_COUNT = 100
SUPPORTED_FP64_FINITE_STEP_REFERENCE_WINDOW_COUNT = 20
SUPPORTED_DENSE_EQUILIBRIUM_REFERENCE_WINDOW_COUNT = 20
SUPPORTED_CONDITION_NUMBER_WINDOW_COUNT = 10
PROHIBITED_LIST = frozenset(
    {
        "method_efficacy_claim", "full_validation_run", "cover_dr",
        "t4_semantic_repair", "dcr", "sur", "equilibrium_cgls_reproduction",
        "dataset_miou_claim", "random_sampling", "class_based_selection",
        "gt_based_selection", "runtime_based_selection",
        "difficult_window_skipping", "synthetic_window_substitution",
        "independently_selected_k11_graph", "artificial_similarity_perturbation",
        "early_termination", "solver_fallback", "dense_solve_in_production_path",
    }
)

IDENTITY_TOP_KEYS = frozenset(
    {
        "format_version", "identity", "parent_identity", "sample_selection",
        "reference_windows", "snapshots", "propagation", "tolerances",
        "gate_thresholds", "result_contract", "prohibited",
    }
)
IDENTITY_SECTION_KEYS = {
    "identity": frozenset({"name", "schema_version", "description", "required_ancestor_commit"}),
    "parent_identity": frozenset(
        {"matched_identity_path", "matched_identity_name", "matched_identity_sha256", "required_relationship"}
    ),
    "sample_selection": frozenset(
        {"rule", "canonical_window_count", "image_order_source", "window_order_source", "allow_partial_final_image"}
    ),
    "reference_windows": frozenset(
        {
            "fp64_finite_step_reference_window_count",
            "dense_equilibrium_reference_window_count",
            "condition_number_window_count",
        }
    ),
    "snapshots": frozenset({"steps", "initial_iterate"}),
    "propagation": frozenset({"method", "alpha", "compute_dtype", "reference_dtype"}),
    "tolerances": frozenset(
        {
            "fp32_fp64_relative_frobenius_error_max", "fp32_fp64_max_absolute_error_max",
            "fp32_fp64_argmax_disagreement_rate_max", "dense_equilibrium_residual_relative_max",
            "t_snapshot_relative_frobenius_change_max", "t_snapshot_argmax_disagreement_rate_max",
            "delta_stability_relative_max", "determinism_replay_count",
            "determinism_exact_bytes_required",
        }
    ),
    "gate_thresholds": frozenset(
        {
            "effect_near_noise_relative_delta_norm_max", "truncation_sensitive_relative_change_min",
            "clear_signal_relative_delta_norm_min", "epsilon_denominator_floor",
        }
    ),
    "result_contract": frozenset({"schema_name", "checkpoint_schema_name", "regime_classifications"}),
    "prohibited": frozenset({"list"}),
}


class K11K12StabilityGateError(ValueError):
    """Raised when the gate identity, a result, or a checkpoint fails closed."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise K11K12StabilityGateError(f"{label} must be an exact non-empty string")
    return value


def _require_supported_value(value: Any, label: str, expected: str) -> str:
    token = _require_exact_string(value, label)
    if token != expected:
        raise K11K12StabilityGateError(f"{label} must be exactly {expected!r}, observed {token!r}")
    return token


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise K11K12StabilityGateError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise K11K12StabilityGateError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise K11K12StabilityGateError(f"{label} must be at least {minimum}")
    return value


def _require_exact_float(value: Any, label: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if type(value) is not float:
        raise K11K12StabilityGateError(f"{label} must be an exact float")
    if not math.isfinite(value):
        raise K11K12StabilityGateError(f"{label} must be finite")
    if minimum is not None and value < minimum:
        raise K11K12StabilityGateError(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise K11K12StabilityGateError(f"{label} must be at most {maximum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise K11K12StabilityGateError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", token) is None:
        raise K11K12StabilityGateError(f"{label} must be a full Git identity")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token:
        raise K11K12StabilityGateError(f"{label} must be a safe repository-relative path")
    return token


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise K11K12StabilityGateError(f"{label} has an unexpected schema")
    return value


def _require_exact_int_list(value: Any, label: str) -> tuple[int, ...]:
    if type(value) is not list or not value:
        raise K11K12StabilityGateError(f"{label} must be a non-empty exact list")
    if any(type(item) is not int for item in value):
        raise K11K12StabilityGateError(f"{label} elements must be exact integers")
    return tuple(value)


def _require_exact_string_list(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise K11K12StabilityGateError(f"{label} must be a non-empty exact list")
    if any(type(item) is not str for item in value):
        raise K11K12StabilityGateError(f"{label} elements must be exact strings")
    return tuple(value)


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise K11K12StabilityGateError(
            f"cannot load k11/k12 stability gate identity {source}: {error}"
        ) from error

    if set(identity) != IDENTITY_TOP_KEYS:
        raise K11K12StabilityGateError("stability gate identity has an unexpected top-level schema")
    _require_exact_string(identity["format_version"], "format_version")
    if identity["format_version"] != "talk2dino-k11-k12-stability-gate-identity-v1":
        raise K11K12StabilityGateError("unsupported stability gate identity format_version")
    for section, keys in IDENTITY_SECTION_KEYS.items():
        _require_closed_mapping(identity.get(section), keys, f"identity.{section}")

    block = identity["identity"]
    _require_exact_string(block["name"], "identity.name")
    _require_exact_string(block["schema_version"], "identity.schema_version")
    if block["schema_version"] != identity["format_version"]:
        raise K11K12StabilityGateError("identity.schema_version disagrees with format_version")
    _require_exact_string(block["description"], "identity.description")
    _require_git_identity(block["required_ancestor_commit"], "identity.required_ancestor_commit")

    parent = identity["parent_identity"]
    _require_relative_path(parent["matched_identity_path"], "parent_identity.matched_identity_path")
    _require_exact_string(parent["matched_identity_name"], "parent_identity.matched_identity_name")
    _require_sha256(parent["matched_identity_sha256"], "parent_identity.matched_identity_sha256")
    _require_exact_string(parent["required_relationship"], "parent_identity.required_relationship")

    sample = identity["sample_selection"]
    _require_supported_value(sample["rule"], "sample_selection.rule", SUPPORTED_SAMPLE_SELECTION_RULE)
    _require_exact_int(sample["canonical_window_count"], "sample_selection.canonical_window_count", minimum=1)
    if sample["canonical_window_count"] != SUPPORTED_CANONICAL_WINDOW_COUNT:
        raise K11K12StabilityGateError(
            "sample_selection.canonical_window_count must be exactly "
            f"{SUPPORTED_CANONICAL_WINDOW_COUNT} (the implemented bounded harness protocol), "
            f"observed {sample['canonical_window_count']!r}"
        )
    _require_supported_value(
        sample["image_order_source"], "sample_selection.image_order_source", SUPPORTED_IMAGE_ORDER_SOURCE
    )
    _require_supported_value(
        sample["window_order_source"], "sample_selection.window_order_source", SUPPORTED_WINDOW_ORDER_SOURCE
    )
    _require_exact_bool(sample["allow_partial_final_image"], "sample_selection.allow_partial_final_image")

    refs = identity["reference_windows"]
    supported_reference_counts = {
        "fp64_finite_step_reference_window_count": SUPPORTED_FP64_FINITE_STEP_REFERENCE_WINDOW_COUNT,
        "dense_equilibrium_reference_window_count": SUPPORTED_DENSE_EQUILIBRIUM_REFERENCE_WINDOW_COUNT,
        "condition_number_window_count": SUPPORTED_CONDITION_NUMBER_WINDOW_COUNT,
    }
    for name, supported_value in supported_reference_counts.items():
        count = _require_exact_int(refs[name], f"reference_windows.{name}", minimum=1)
        if count > sample["canonical_window_count"]:
            raise K11K12StabilityGateError(
                f"reference_windows.{name} must not exceed sample_selection.canonical_window_count"
            )
        if count != supported_value:
            raise K11K12StabilityGateError(
                f"reference_windows.{name} must be exactly {supported_value} "
                f"(the implemented bounded harness protocol), observed {count!r}"
            )

    snapshots = identity["snapshots"]
    steps = _require_exact_int_list(snapshots["steps"], "snapshots.steps")
    if steps != (160, 320, 640):
        raise K11K12StabilityGateError("snapshots.steps must be exactly [160, 320, 640]")
    _require_supported_value(snapshots["initial_iterate"], "snapshots.initial_iterate", SUPPORTED_INITIAL_ITERATE)

    propagation = identity["propagation"]
    _require_supported_value(propagation["method"], "propagation.method", SUPPORTED_PROPAGATION_METHOD)
    alpha = _require_exact_float(propagation["alpha"], "propagation.alpha", minimum=0.0)
    if not 0 <= alpha < 1:
        raise K11K12StabilityGateError("propagation.alpha must satisfy 0 <= alpha < 1")
    _require_supported_value(propagation["compute_dtype"], "propagation.compute_dtype", SUPPORTED_COMPUTE_DTYPE)
    _require_supported_value(propagation["reference_dtype"], "propagation.reference_dtype", SUPPORTED_REFERENCE_DTYPE)

    tolerances = identity["tolerances"]
    for name in (
        "fp32_fp64_relative_frobenius_error_max", "fp32_fp64_max_absolute_error_max",
        "fp32_fp64_argmax_disagreement_rate_max", "dense_equilibrium_residual_relative_max",
        "t_snapshot_relative_frobenius_change_max", "t_snapshot_argmax_disagreement_rate_max",
        "delta_stability_relative_max",
    ):
        _require_exact_float(tolerances[name], f"tolerances.{name}", minimum=0.0)
    _require_exact_int(tolerances["determinism_replay_count"], "tolerances.determinism_replay_count", minimum=2)
    _require_exact_bool(tolerances["determinism_exact_bytes_required"], "tolerances.determinism_exact_bytes_required")

    thresholds = identity["gate_thresholds"]
    for name in (
        "effect_near_noise_relative_delta_norm_max", "truncation_sensitive_relative_change_min",
        "clear_signal_relative_delta_norm_min", "epsilon_denominator_floor",
    ):
        _require_exact_float(thresholds[name], f"gate_thresholds.{name}", minimum=0.0)
    # The gap between these two thresholds intentionally defines the honest
    # "inconclusive" interval (see classify_regime): equal or reversed
    # thresholds would erase that interval and force every effect size into
    # a near-noise or clear-signal claim, so this is rejected relationally
    # rather than left to whichever branch happens to run first.
    if not (
        thresholds["effect_near_noise_relative_delta_norm_max"]
        < thresholds["clear_signal_relative_delta_norm_min"]
    ):
        raise K11K12StabilityGateError(
            "gate_thresholds.effect_near_noise_relative_delta_norm_max must be strictly less than "
            "gate_thresholds.clear_signal_relative_delta_norm_min, observed "
            f"{thresholds['effect_near_noise_relative_delta_norm_max']!r} >= "
            f"{thresholds['clear_signal_relative_delta_norm_min']!r}"
        )

    result_contract = identity["result_contract"]
    _require_supported_value(result_contract["schema_name"], "result_contract.schema_name", RESULT_SCHEMA_NAME)
    _require_supported_value(
        result_contract["checkpoint_schema_name"], "result_contract.checkpoint_schema_name", CHECKPOINT_SCHEMA_NAME
    )
    classifications = _require_exact_string_list(
        result_contract["regime_classifications"], "result_contract.regime_classifications"
    )
    if classifications != SUPPORTED_REGIME_CLASSIFICATIONS:
        raise K11K12StabilityGateError(
            f"result_contract.regime_classifications must be exactly {list(SUPPORTED_REGIME_CLASSIFICATIONS)}"
        )

    prohibited_list = _require_exact_string_list(identity["prohibited"]["list"], "prohibited.list")
    if set(prohibited_list) != PROHIBITED_LIST:
        raise K11K12StabilityGateError("prohibited.list does not match the supported prohibited-practice vocabulary")

    return identity


def _check_git_ancestry(root: Path, identity: Mapping[str, Any]) -> None:
    command = [
        "git", "merge-base", "--is-ancestor",
        str(identity["identity"]["required_ancestor_commit"]), "HEAD",
    ]
    try:
        result = subprocess.run(
            command, cwd=root, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except OSError as error:
        raise K11K12StabilityGateError(f"cannot execute Git ancestry check: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or "required ancestor is not an ancestor of HEAD"
        raise K11K12StabilityGateError(
            f"Git ancestry check failed for {identity['identity']['required_ancestor_commit']}: {detail}"
        )


def _validate_parent_identity(root: Path, identity: Mapping[str, Any]) -> Mapping[str, Any]:
    parent = identity["parent_identity"]
    matched_path = root / parent["matched_identity_path"]
    try:
        matched_bytes = matched_path.read_bytes()
    except OSError as error:
        raise K11K12StabilityGateError(f"cannot read matched parent identity {matched_path}: {error}") from error
    if hashlib.sha256(matched_bytes).hexdigest() != parent["matched_identity_sha256"]:
        raise K11K12StabilityGateError("matched parent identity file SHA256 mismatch")
    matched_identity = _load_matched_identity(matched_path, repo_root=root)
    if matched_identity["identity"]["name"] != parent["matched_identity_name"]:
        raise K11K12StabilityGateError("matched parent identity name mismatch")
    # Relational: the gate's alpha must match the parent experiment's alpha
    # (the gate probes the same finite-step recurrence the experiment uses).
    if identity["propagation"]["alpha"] != matched_identity["propagation"]["alpha"]:
        raise K11K12StabilityGateError(
            "gate propagation.alpha disagrees with the parent matched identity's propagation.alpha"
        )
    return matched_identity


def validate_static_configuration(
    *, repo_root: Path | None = None, identity_path: Path | None = None, check_git: bool = True
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    if check_git:
        _check_git_ancestry(root, identity)
    matched_identity = _validate_parent_identity(root, identity)

    from src.matched_k11_k12_identity import validate_static_configuration as _matched_preflight

    matched_result = _matched_preflight(repo_root=root, check_git=check_git)

    return {
        "identity_name": identity["identity"]["name"],
        "matched_identity": matched_result["identity_name"],
        "required_ancestor_commit": identity["identity"]["required_ancestor_commit"],
        "canonical_window_count": identity["sample_selection"]["canonical_window_count"],
    }


def _parse_strict_json(text: str, label: str) -> Mapping[str, Any]:
    def closed_object(pairs):
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise K11K12StabilityGateError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            parse_float=Decimal,
            object_pairs_hook=closed_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                K11K12StabilityGateError(f"{label} contains non-finite {token}")
            ),
        )
    except K11K12StabilityGateError:
        raise
    except (json.JSONDecodeError, TypeError) as error:
        raise K11K12StabilityGateError(f"cannot parse {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise K11K12StabilityGateError(f"{label} must contain one JSON object")
    return value


def parse_structured_document(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="strict")
    except OSError as error:
        raise K11K12StabilityGateError(f"cannot read {label} {path}: {error}") from error
    return _parse_strict_json(text, label)


__all__ = [
    "CHECKPOINT_SCHEMA_NAME",
    "IDENTITY_RELATIVE_PATH",
    "K11K12StabilityGateError",
    "PROHIBITED_LIST",
    "RESULT_SCHEMA_NAME",
    "SUPPORTED_CANONICAL_WINDOW_COUNT",
    "SUPPORTED_COMPUTE_DTYPE",
    "SUPPORTED_CONDITION_NUMBER_WINDOW_COUNT",
    "SUPPORTED_DENSE_EQUILIBRIUM_REFERENCE_WINDOW_COUNT",
    "SUPPORTED_FP64_FINITE_STEP_REFERENCE_WINDOW_COUNT",
    "SUPPORTED_IMAGE_ORDER_SOURCE",
    "SUPPORTED_INITIAL_ITERATE",
    "SUPPORTED_PROPAGATION_METHOD",
    "SUPPORTED_REFERENCE_DTYPE",
    "SUPPORTED_REGIME_CLASSIFICATIONS",
    "SUPPORTED_SAMPLE_SELECTION_RULE",
    "SUPPORTED_WINDOW_ORDER_SOURCE",
    "load_identity",
    "parse_structured_document",
    "repository_root",
    "validate_static_configuration",
]
