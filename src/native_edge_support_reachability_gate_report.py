"""Structured result schema validation for the offline 20-image structural
reachability gate's own output artifact.

Never renders an efficacy verdict and never recomputes the parent audit's
science -- this only verifies a gate result is structurally complete,
internally consistent (recomputes the deterministic content digest and the
roadmap-authorization mapping rather than trusting the stored scalars), and
correctly bound to its own identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.native_edge_support_checkpoint import parse_strict_json_document
from src.native_edge_support_reachability_gate_identity import (
    SUPPORTED_DECISION_MAPPING,
    SUPPORTED_DESIRED_CAUSE_CATEGORIES,
    SUPPORTED_GATE_SCHEMA_NAME,
    SUPPORTED_NEXT_STAGE_IF_NOT_REACHABLE,
    SUPPORTED_NEXT_STAGE_IF_REACHABLE,
    SUPPORTED_STRUCTURAL_STAGES,
    NativeEdgeSupportReachabilityGateIdentityError,
    load_identity,
    repository_root,
)

TOP_GATE_RESULT_KEYS = frozenset(
    {
        "schema", "gate_identity", "gate_identity_sha256", "native_audit_identity_sha256", "parent_artifact",
        "validation_summary", "reconstructed_counts_ratios", "cause_decomposition", "reachability_summary",
        "parent_decision_reproduction", "roadmap_authorization", "stop_proceed_decision",
        "scientific_interpretation", "limitations", "content_digest", "audit_result_path_reference",
        "created_at_utc",
    }
)
PARENT_ARTIFACT_KEYS = frozenset(
    {"label", "sha256", "byte_size", "identity", "identity_sha256", "source_git_commit", "image_order_digest", "run_mode", "image_count_processed"}
)
VALIDATION_SUMMARY_KEYS = frozenset(
    {
        "strict_json", "schema_matches", "complete", "final", "run_mode_matches", "image_count_matches",
        "parent_verify_record_passed", "count_reconciliation_passed",
    }
)
DECISION_REPRODUCTION_KEYS = frozenset(
    {"parent_reported_decision", "gate_reproduced_decision", "match", "parent_decision_rationale", "gate_reproduced_rationale"}
)
ROADMAP_AUTHORIZATION_KEYS = frozenset({"roadmap_action", "stages", "next_authorized_stage"})
STAGE_KEYS = frozenset({"stage", "authorized", "reason"})
UNIT_VALUE_KEYS = frozenset({"value", "unit"})
ALLOWED_UNITS = ("count", "fraction_0_1", "percent_0_100", "dimensionless")


class NativeEdgeSupportReachabilityGateReportError(NativeEdgeSupportReachabilityGateIdentityError):
    """Raised when the reachability-gate's own result artifact fails
    validation. Always fail closed."""


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise NativeEdgeSupportReachabilityGateReportError(f"{label} has an unexpected schema")
    return value


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise NativeEdgeSupportReachabilityGateReportError(f"{label} must be an exact non-empty string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise NativeEdgeSupportReachabilityGateReportError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NativeEdgeSupportReachabilityGateReportError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise NativeEdgeSupportReachabilityGateReportError(f"{label} must be at least {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise NativeEdgeSupportReachabilityGateReportError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 40 or any(c not in "0123456789abcdef" for c in token):
        raise NativeEdgeSupportReachabilityGateReportError(f"{label} must be a full Git identity")
    return token


def _require_unit_value(value: Any, label: str, *, numeric_type: type | tuple[type, ...] = (int, float)) -> Any:
    mapping = _require_closed_mapping(value, UNIT_VALUE_KEYS, label)
    if mapping["unit"] not in ALLOWED_UNITS:
        raise NativeEdgeSupportReachabilityGateReportError(f"{label}.unit must be one of {ALLOWED_UNITS}")
    v = mapping["value"]
    if v is not None:
        if isinstance(v, bool) or not isinstance(v, numeric_type):
            raise NativeEdgeSupportReachabilityGateReportError(f"{label}.value must be numeric or null")
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            raise NativeEdgeSupportReachabilityGateReportError(f"{label}.value must be finite")
    return v


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def verify_record(record: Mapping[str, Any], gate_identity: Mapping[str, Any], *, gate_identity_sha256: str) -> str:
    if not hasattr(record, "get"):
        raise NativeEdgeSupportReachabilityGateReportError("structured gate result must be a mapping")
    _require_closed_mapping(record, TOP_GATE_RESULT_KEYS, "result")

    if _require_exact_string(record["schema"], "result.schema") != SUPPORTED_GATE_SCHEMA_NAME:
        raise NativeEdgeSupportReachabilityGateReportError(f"result.schema must be {SUPPORTED_GATE_SCHEMA_NAME!r}")
    if _require_exact_string(record["gate_identity"], "result.gate_identity") != gate_identity["identity"]["name"]:
        raise NativeEdgeSupportReachabilityGateReportError("result.gate_identity mismatch")
    if _require_sha256(record["gate_identity_sha256"], "result.gate_identity_sha256") != gate_identity_sha256:
        raise NativeEdgeSupportReachabilityGateReportError("result.gate_identity_sha256 does not match the loaded identity file")
    if record["native_audit_identity_sha256"] != gate_identity["parent_identity"]["native_audit_identity_sha256"]:
        raise NativeEdgeSupportReachabilityGateReportError("result.native_audit_identity_sha256 disagrees with the gate identity's parent")

    parent = _require_closed_mapping(record["parent_artifact"], PARENT_ARTIFACT_KEYS, "result.parent_artifact")
    _require_exact_string(parent["label"], "parent_artifact.label")
    _require_sha256(parent["sha256"], "parent_artifact.sha256")
    _require_exact_int(parent["byte_size"], "parent_artifact.byte_size", minimum=1)
    _require_exact_string(parent["identity"], "parent_artifact.identity")
    _require_sha256(parent["identity_sha256"], "parent_artifact.identity_sha256")
    _require_git_identity(parent["source_git_commit"], "parent_artifact.source_git_commit")
    _require_sha256(parent["image_order_digest"], "parent_artifact.image_order_digest")
    if parent["run_mode"] != "mechanics20":
        raise NativeEdgeSupportReachabilityGateReportError("parent_artifact.run_mode must be 'mechanics20'")
    if _require_exact_int(parent["image_count_processed"], "parent_artifact.image_count_processed") != 20:
        raise NativeEdgeSupportReachabilityGateReportError("parent_artifact.image_count_processed must be exactly 20")

    validation = _require_closed_mapping(record["validation_summary"], VALIDATION_SUMMARY_KEYS, "result.validation_summary")
    for key in VALIDATION_SUMMARY_KEYS - {"final"}:
        if _require_exact_bool(validation[key], f"validation_summary.{key}") is not True:
            raise NativeEdgeSupportReachabilityGateReportError(f"validation_summary.{key} must be true")
    if _require_exact_bool(validation["final"], "validation_summary.final") is not False:
        raise NativeEdgeSupportReachabilityGateReportError("validation_summary.final must be false")

    ratios = record["reconstructed_counts_ratios"]
    if not isinstance(ratios, Mapping) or not ratios:
        raise NativeEdgeSupportReachabilityGateReportError("result.reconstructed_counts_ratios must be a non-empty mapping")
    for key, value in ratios.items():
        if key == "funnel_survival":
            if not isinstance(value, Mapping):
                raise NativeEdgeSupportReachabilityGateReportError("reconstructed_counts_ratios.funnel_survival must be a mapping")
            for sub_key, sub_value in value.items():
                _require_unit_value(sub_value, f"reconstructed_counts_ratios.funnel_survival.{sub_key}")
        else:
            _require_unit_value(value, f"reconstructed_counts_ratios.{key}")

    causes = record["cause_decomposition"]
    if _require_exact_bool(causes["mutually_exclusive"], "cause_decomposition.mutually_exclusive") is not True:
        raise NativeEdgeSupportReachabilityGateReportError("cause_decomposition.mutually_exclusive must be true")
    categories = causes["categories"]
    if set(categories) != set(SUPPORTED_DESIRED_CAUSE_CATEGORIES) | {"no_native_observer_cause_not_further_identifiable"}:
        raise NativeEdgeSupportReachabilityGateReportError("cause_decomposition.categories has an unexpected key set")
    for name, entry in categories.items():
        available = _require_exact_bool(entry["available"], f"cause_decomposition.categories.{name}.available")
        if available:
            _require_exact_int(entry["count"], f"cause_decomposition.categories.{name}.count", minimum=0)
        elif entry["count"] is not None:
            raise NativeEdgeSupportReachabilityGateReportError(f"cause_decomposition.categories.{name}.count must be null when unavailable")
    _require_exact_string(causes["limitations"], "cause_decomposition.limitations")

    decision = _require_closed_mapping(record["parent_decision_reproduction"], DECISION_REPRODUCTION_KEYS, "result.parent_decision_reproduction")
    if decision["parent_reported_decision"] not in SUPPORTED_DECISION_MAPPING:
        raise NativeEdgeSupportReachabilityGateReportError("parent_decision_reproduction.parent_reported_decision is not a recognized outcome")
    if decision["gate_reproduced_decision"] != decision["parent_reported_decision"]:
        raise NativeEdgeSupportReachabilityGateReportError("parent_decision_reproduction: gate-reproduced decision disagrees with the parent's reported decision")
    if _require_exact_bool(decision["match"], "parent_decision_reproduction.match") is not True:
        raise NativeEdgeSupportReachabilityGateReportError("parent_decision_reproduction.match must be true")

    roadmap = _require_closed_mapping(record["roadmap_authorization"], ROADMAP_AUTHORIZATION_KEYS, "result.roadmap_authorization")
    expected_action = SUPPORTED_DECISION_MAPPING[decision["parent_reported_decision"]]
    if roadmap["roadmap_action"] != expected_action:
        raise NativeEdgeSupportReachabilityGateReportError("roadmap_authorization.roadmap_action disagrees with the locked decision mapping")
    if record["stop_proceed_decision"] != expected_action:
        raise NativeEdgeSupportReachabilityGateReportError("result.stop_proceed_decision disagrees with the locked decision mapping")
    stages = roadmap["stages"]
    if not isinstance(stages, list) or [s["stage"] for s in stages] != list(SUPPORTED_STRUCTURAL_STAGES):
        raise NativeEdgeSupportReachabilityGateReportError("roadmap_authorization.stages must list exactly the 4 locked structural stages in order")
    authorized_expected = expected_action == "PROCEED_TO_ELIGIBILITY_MATCHED_PRUNING"
    for stage_entry in stages:
        _require_closed_mapping(stage_entry, STAGE_KEYS, "roadmap_authorization.stages[]")
        if _require_exact_bool(stage_entry["authorized"], "stages[].authorized") is not authorized_expected:
            raise NativeEdgeSupportReachabilityGateReportError("a stage's authorized flag disagrees with the locked decision mapping")
        if authorized_expected and stage_entry["reason"] is not None:
            raise NativeEdgeSupportReachabilityGateReportError("an authorized stage must not carry a skip reason")
        if not authorized_expected and not stage_entry["reason"]:
            raise NativeEdgeSupportReachabilityGateReportError("a skipped stage must carry a non-empty skip reason")
    expected_next = SUPPORTED_NEXT_STAGE_IF_REACHABLE if authorized_expected else SUPPORTED_NEXT_STAGE_IF_NOT_REACHABLE
    if roadmap["next_authorized_stage"] != expected_next:
        raise NativeEdgeSupportReachabilityGateReportError("roadmap_authorization.next_authorized_stage disagrees with the locked mapping")

    _require_exact_string(record["scientific_interpretation"], "result.scientific_interpretation")
    limitations = record["limitations"]
    if type(limitations) is not list or any(type(item) is not str for item in limitations):
        raise NativeEdgeSupportReachabilityGateReportError("result.limitations must be a list of exact strings")

    # recompute the deterministic content digest independently
    deterministic_keys = TOP_GATE_RESULT_KEYS - {"content_digest", "audit_result_path_reference", "created_at_utc"}
    deterministic_payload = {key: record[key] for key in deterministic_keys}
    recomputed_digest = hashlib.sha256(_canonical_json_bytes(deterministic_payload)).hexdigest()
    if _require_sha256(record["content_digest"], "result.content_digest") != recomputed_digest:
        raise NativeEdgeSupportReachabilityGateReportError("result.content_digest does not match the recomputed deterministic content digest")

    _require_exact_string(record["audit_result_path_reference"], "result.audit_result_path_reference")
    _require_exact_string(record["created_at_utc"], "result.created_at_utc")

    return (
        f"NATIVE EDGE SUPPORT REACHABILITY GATE RESULT PASS parent_decision={decision['parent_reported_decision']} "
        f"roadmap_action={roadmap['roadmap_action']} next_stage={roadmap['next_authorized_stage']!r}"
    )


def verify_result(path: Path, *, identity_path: Path | None = None, repo_root: Path | None = None) -> str:
    root = Path(repo_root) if repo_root is not None else repository_root()
    gate_identity = load_identity(identity_path, repo_root=root)
    resolved_identity_path = (
        Path(identity_path) if identity_path is not None
        else root / "evaluation_identities/e12_native_edge_support_reachability_gate.toml"
    )
    gate_identity_sha256 = hashlib.sha256(Path(resolved_identity_path).read_bytes()).hexdigest()
    record = parse_strict_json_document(Path(path), label="reachability-gate result")
    return verify_record(record, gate_identity, gate_identity_sha256=gate_identity_sha256)


__all__ = [
    "NativeEdgeSupportReachabilityGateReportError",
    "TOP_GATE_RESULT_KEYS",
    "verify_record",
    "verify_result",
]
