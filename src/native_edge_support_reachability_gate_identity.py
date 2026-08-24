"""Static identity loading/validation for the offline 20-image structural
reachability gate.

Deliberately avoids importing MMCV, Torch, model code, or dataset code --
this identity, and the gate it authorizes, are pure CPU/offline analysis
over an already-produced JSON result. Every threshold that belongs to the
native-edge-support-audit identity (alpha, steps, k, the decision
classification thresholds) is never redeclared here as a bare literal --
this identity only registers the gate's OWN facts: the parent binding, the
locked decision-to-roadmap-action mapping, the downstream stage list, and
the required-field contract the gate depends on.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e12_native_edge_support_reachability_gate.toml")

SUPPORTED_PARENT_RUN_MODE = "mechanics20"
SUPPORTED_PARENT_IMAGE_COUNT = 20
SUPPORTED_PARENT_SCHEMA = "talk2dino-native-edge-support-audit-mechanics20-result-v1"
SUPPORTED_DECISION_OUTCOMES = ("REACHABLE", "STRUCTURALLY_UNREACHABLE", "ALIGNMENT_LIMITED", "INCONCLUSIVE")
SUPPORTED_DECISION_MAPPING = {
    "REACHABLE": "PROCEED_TO_ELIGIBILITY_MATCHED_PRUNING",
    "STRUCTURALLY_UNREACHABLE": "STOP_STRUCTURAL_CONNECTIVITY_BRANCH",
    "ALIGNMENT_LIMITED": "STOP_NATIVE_SUPPORT_ALIGNMENT_LIMITED",
    "INCONCLUSIVE": "DO_NOT_PROCEED_INCONCLUSIVE",
}
SUPPORTED_STRUCTURAL_STAGES = (
    "feat: add eligibility-matched one-edge pruning variants",
    "test: add matched-budget graph and replay suite",
    "eval: add paired bootstrap and sample-size lock",
    "eval: run locked structural-connectivity efficacy pilot",
)
SUPPORTED_NEXT_STAGE_IF_REACHABLE = "feat: add eligibility-matched one-edge pruning variants"
SUPPORTED_NEXT_STAGE_IF_NOT_REACHABLE = "eval: add COCO-Object protocol confirmation"
SUPPORTED_DESIRED_CAUSE_CATEGORIES = (
    "single_window_no_alternative_view",
    "multi_window_no_overlapping_alternative",
    "candidate_observer_origin_unaligned",
    "source_maps_but_destination_outside_shared_overlap",
    "other_exact_geometry_failure",
    "support_defined",
)
SUPPORTED_COMBINED_FALLBACK_CATEGORY = "no_native_observer_cause_not_further_identifiable"
SUPPORTED_UNITS = ("count", "fraction_0_1", "percent_0_100", "dimensionless")
SUPPORTED_GATE_SCHEMA_NAME = "talk2dino-native-edge-support-reachability-gate-result-v1"

IDENTITY_TOP_KEYS = frozenset(
    {
        "format_version", "identity", "parent_identity", "parent_contract", "decision_vocabulary",
        "decision_mapping", "roadmap", "required_fields", "cause_decomposition", "units", "policy",
        "run_mode", "prohibited",
    }
)
IDENTITY_SECTION_KEYS = {
    "identity": frozenset({"name", "schema_version", "description", "required_ancestor_commit"}),
    "parent_identity": frozenset(
        {"native_audit_identity_path", "native_audit_identity_name", "native_audit_identity_sha256", "required_relationship"}
    ),
    "parent_contract": frozenset(
        {"required_run_mode", "required_image_count", "required_complete", "required_final", "required_schema"}
    ),
    "decision_vocabulary": frozenset({"outcomes"}),
    "decision_mapping": frozenset(SUPPORTED_DECISION_OUTCOMES),
    "roadmap": frozenset(
        {
            "structural_stages_gated_on_reachable", "next_stage_if_reachable", "next_stage_if_not_reachable",
            "skip_reason_template",
        }
    ),
    "required_fields": frozenset({"funnel_fields", "undefined_reason_fields", "cross_tab_tables"}),
    "cause_decomposition": frozenset(
        {
            "desired_categories", "combined_fallback_category", "infer_from_pair_counts",
            "fabricate_unavailable_subcategories", "rerun_audit_to_fill_gaps",
        }
    ),
    "units": frozenset({"allowed"}),
    "policy": frozenset(
        {
            "no_threshold_relaxation", "no_approximate_alignment_introduced", "no_pruning_implemented",
            "no_efficacy_computation", "deterministic_output", "cuda_required", "model_required", "dataset_required",
        }
    ),
    "run_mode": frozenset({"schema_name", "checkpoint_schema_name"}),
    "prohibited": frozenset({"list"}),
}


class NativeEdgeSupportReachabilityGateIdentityError(ValueError):
    """Raised when the reachability-gate identity, a gate result, or a
    parent-binding check fails closed. Always fail closed: never silently
    substitute a default decision mapping, downstream stage, or
    unverified assumption."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise NativeEdgeSupportReachabilityGateIdentityError(f"{label} must be an exact non-empty string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", token) is None:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"{label} must be a full Git identity")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"{label} must be a safe repository-relative path")
    return token


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"{label} has an unexpected schema")
    return value


def _require_exact_string_list(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"{label} must be a non-empty exact list")
    if any(type(item) is not str for item in value):
        raise NativeEdgeSupportReachabilityGateIdentityError(f"{label} elements must be exact strings")
    return tuple(value)


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise NativeEdgeSupportReachabilityGateIdentityError(
            f"cannot load reachability-gate identity {source}: {error}"
        ) from error

    if set(identity) != IDENTITY_TOP_KEYS:
        raise NativeEdgeSupportReachabilityGateIdentityError("reachability-gate identity has an unexpected top-level schema")
    _require_exact_string(identity["format_version"], "format_version")
    if identity["format_version"] != "talk2dino-native-edge-support-reachability-gate-identity-v1":
        raise NativeEdgeSupportReachabilityGateIdentityError("unsupported reachability-gate identity format_version")
    for section, keys in IDENTITY_SECTION_KEYS.items():
        _require_closed_mapping(identity.get(section), keys, f"identity.{section}")

    block = identity["identity"]
    _require_exact_string(block["name"], "identity.name")
    _require_exact_string(block["schema_version"], "identity.schema_version")
    if block["schema_version"] != identity["format_version"]:
        raise NativeEdgeSupportReachabilityGateIdentityError("identity.schema_version disagrees with format_version")
    _require_exact_string(block["description"], "identity.description")
    _require_git_identity(block["required_ancestor_commit"], "identity.required_ancestor_commit")

    parent = identity["parent_identity"]
    _require_relative_path(parent["native_audit_identity_path"], "parent_identity.native_audit_identity_path")
    _require_exact_string(parent["native_audit_identity_name"], "parent_identity.native_audit_identity_name")
    _require_sha256(parent["native_audit_identity_sha256"], "parent_identity.native_audit_identity_sha256")
    _require_exact_string(parent["required_relationship"], "parent_identity.required_relationship")

    contract = identity["parent_contract"]
    if contract["required_run_mode"] != SUPPORTED_PARENT_RUN_MODE:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"parent_contract.required_run_mode must be {SUPPORTED_PARENT_RUN_MODE!r}")
    if _require_exact_int(contract["required_image_count"], "parent_contract.required_image_count") != SUPPORTED_PARENT_IMAGE_COUNT:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"parent_contract.required_image_count must be exactly {SUPPORTED_PARENT_IMAGE_COUNT}")
    if _require_exact_bool(contract["required_complete"], "parent_contract.required_complete") is not True:
        raise NativeEdgeSupportReachabilityGateIdentityError("parent_contract.required_complete must be true")
    if _require_exact_bool(contract["required_final"], "parent_contract.required_final") is not False:
        # The native-edge-support audit's own schema never produces
        # final=true for mechanics20 (it is a structural audit, never
        # itself a dataset-level "final" measurement) -- requiring
        # anything else here would make every valid mechanics20 result
        # fail closed, which is the opposite of a correct binding check.
        raise NativeEdgeSupportReachabilityGateIdentityError("parent_contract.required_final must be false for mechanics20")
    if contract["required_schema"] != SUPPORTED_PARENT_SCHEMA:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"parent_contract.required_schema must be {SUPPORTED_PARENT_SCHEMA!r}")

    vocab = identity["decision_vocabulary"]
    if _require_exact_string_list(vocab["outcomes"], "decision_vocabulary.outcomes") != SUPPORTED_DECISION_OUTCOMES:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"decision_vocabulary.outcomes must be exactly {list(SUPPORTED_DECISION_OUTCOMES)}")

    mapping = identity["decision_mapping"]
    for outcome, expected_action in SUPPORTED_DECISION_MAPPING.items():
        observed = _require_exact_string(mapping[outcome], f"decision_mapping.{outcome}")
        if observed != expected_action:
            raise NativeEdgeSupportReachabilityGateIdentityError(
                f"decision_mapping.{outcome} must be exactly {expected_action!r}, observed {observed!r}"
            )

    roadmap = identity["roadmap"]
    if _require_exact_string_list(roadmap["structural_stages_gated_on_reachable"], "roadmap.structural_stages_gated_on_reachable") != SUPPORTED_STRUCTURAL_STAGES:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"roadmap.structural_stages_gated_on_reachable must be exactly {list(SUPPORTED_STRUCTURAL_STAGES)}")
    if roadmap["next_stage_if_reachable"] != SUPPORTED_NEXT_STAGE_IF_REACHABLE:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"roadmap.next_stage_if_reachable must be {SUPPORTED_NEXT_STAGE_IF_REACHABLE!r}")
    if roadmap["next_stage_if_not_reachable"] != SUPPORTED_NEXT_STAGE_IF_NOT_REACHABLE:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"roadmap.next_stage_if_not_reachable must be {SUPPORTED_NEXT_STAGE_IF_NOT_REACHABLE!r}")
    _require_exact_string(roadmap["skip_reason_template"], "roadmap.skip_reason_template")

    required = identity["required_fields"]
    _require_exact_string_list(required["funnel_fields"], "required_fields.funnel_fields")
    if _require_exact_string_list(required["undefined_reason_fields"], "required_fields.undefined_reason_fields") != ("single_window_image", "no_exactly_aligned_observer_covering_both_endpoints"):
        raise NativeEdgeSupportReachabilityGateIdentityError("required_fields.undefined_reason_fields does not match the parent audit's undefined-reason enum")
    if _require_exact_string_list(required["cross_tab_tables"], "required_fields.cross_tab_tables") != (
        "support_defined_vs_source_correct", "all_defined_vs_source_correct",
        "support_defined_vs_stitched_correct", "all_defined_vs_stitched_correct",
    ):
        raise NativeEdgeSupportReachabilityGateIdentityError("required_fields.cross_tab_tables does not match the parent audit's cross-tab schema")

    cause = identity["cause_decomposition"]
    if _require_exact_string_list(cause["desired_categories"], "cause_decomposition.desired_categories") != SUPPORTED_DESIRED_CAUSE_CATEGORIES:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"cause_decomposition.desired_categories must be exactly {list(SUPPORTED_DESIRED_CAUSE_CATEGORIES)}")
    if cause["combined_fallback_category"] != SUPPORTED_COMBINED_FALLBACK_CATEGORY:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"cause_decomposition.combined_fallback_category must be {SUPPORTED_COMBINED_FALLBACK_CATEGORY!r}")
    for flag in ("infer_from_pair_counts", "fabricate_unavailable_subcategories", "rerun_audit_to_fill_gaps"):
        if _require_exact_bool(cause[flag], f"cause_decomposition.{flag}") is not False:
            raise NativeEdgeSupportReachabilityGateIdentityError(f"cause_decomposition.{flag} must be false")

    units = identity["units"]
    if _require_exact_string_list(units["allowed"], "units.allowed") != SUPPORTED_UNITS:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"units.allowed must be exactly {list(SUPPORTED_UNITS)}")

    policy = identity["policy"]
    for flag in (
        "no_threshold_relaxation", "no_approximate_alignment_introduced", "no_pruning_implemented",
        "no_efficacy_computation", "deterministic_output",
    ):
        if _require_exact_bool(policy[flag], f"policy.{flag}") is not True:
            raise NativeEdgeSupportReachabilityGateIdentityError(f"policy.{flag} must be true")
    for flag in ("cuda_required", "model_required", "dataset_required"):
        if _require_exact_bool(policy[flag], f"policy.{flag}") is not False:
            raise NativeEdgeSupportReachabilityGateIdentityError(f"policy.{flag} must be false -- this gate is offline")

    run_mode = identity["run_mode"]
    if run_mode["schema_name"] != SUPPORTED_GATE_SCHEMA_NAME:
        raise NativeEdgeSupportReachabilityGateIdentityError(f"run_mode.schema_name must be {SUPPORTED_GATE_SCHEMA_NAME!r}")
    _require_exact_string(run_mode["checkpoint_schema_name"], "run_mode.checkpoint_schema_name")

    _require_exact_string_list(identity["prohibited"]["list"], "prohibited.list")

    return identity


def _check_git_ancestry(root: Path, commit: str, *, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"{label} is not an ancestor of HEAD"
        raise NativeEdgeSupportReachabilityGateIdentityError(f"Git ancestry check failed for {label} ({commit}): {detail}")


def _validate_parent_identity(root: Path, identity: Mapping[str, Any]) -> Mapping[str, Any]:
    from src.native_edge_support_identity import load_identity as _load_native_audit_identity

    parent = identity["parent_identity"]
    native_audit_path = root / parent["native_audit_identity_path"]
    try:
        native_audit_bytes = native_audit_path.read_bytes()
    except OSError as error:
        raise NativeEdgeSupportReachabilityGateIdentityError(
            f"cannot read native-edge-support-audit parent identity {native_audit_path}: {error}"
        ) from error
    if hashlib.sha256(native_audit_bytes).hexdigest() != parent["native_audit_identity_sha256"]:
        raise NativeEdgeSupportReachabilityGateIdentityError("native-edge-support-audit parent identity file SHA256 mismatch")
    native_audit_identity = _load_native_audit_identity(native_audit_path, repo_root=root)
    if native_audit_identity["identity"]["name"] != parent["native_audit_identity_name"]:
        raise NativeEdgeSupportReachabilityGateIdentityError("native-edge-support-audit parent identity name mismatch")

    return native_audit_identity


def validate_static_configuration(
    *, repo_root: Path | None = None, identity_path: Path | None = None, check_git: bool = True
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    if check_git:
        _check_git_ancestry(root, identity["identity"]["required_ancestor_commit"], label="identity.required_ancestor_commit")
    native_audit_identity = _validate_parent_identity(root, identity)

    return {
        "identity_name": identity["identity"]["name"],
        "native_audit_identity": native_audit_identity["identity"]["name"],
        "required_ancestor_commit": identity["identity"]["required_ancestor_commit"],
        "decision_outcomes": list(identity["decision_vocabulary"]["outcomes"]),
    }


__all__ = [
    "IDENTITY_RELATIVE_PATH",
    "NativeEdgeSupportReachabilityGateIdentityError",
    "SUPPORTED_COMBINED_FALLBACK_CATEGORY",
    "SUPPORTED_DECISION_MAPPING",
    "SUPPORTED_DECISION_OUTCOMES",
    "SUPPORTED_DESIRED_CAUSE_CATEGORIES",
    "SUPPORTED_GATE_SCHEMA_NAME",
    "SUPPORTED_NEXT_STAGE_IF_NOT_REACHABLE",
    "SUPPORTED_NEXT_STAGE_IF_REACHABLE",
    "SUPPORTED_PARENT_IMAGE_COUNT",
    "SUPPORTED_PARENT_RUN_MODE",
    "SUPPORTED_PARENT_SCHEMA",
    "SUPPORTED_STRUCTURAL_STAGES",
    "SUPPORTED_UNITS",
    "load_identity",
    "repository_root",
    "validate_static_configuration",
]
