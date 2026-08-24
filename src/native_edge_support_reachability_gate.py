"""Core logic for the offline, CPU-only 20-image structural reachability
gate: parent-artifact binding/validation, independent aggregate
reconciliation, undefined-support cause decomposition, parent-decision
reproduction, and roadmap authorization.

This module never recomputes the native-edge-support audit's own alignment
rule, support definition, or decision thresholds -- it reuses
:func:`src.native_edge_support_report.classify_reachability` (the parent's
own classifier) verbatim against the parent's own already-produced funnel,
and never introduces a new empirical threshold after observing the result.
No pruning, no efficacy computation, and (deliberately) no torch/CUDA/
model/dataset import anywhere in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from src.native_edge_support_checkpoint import parse_strict_json_document
from src.native_edge_support_identity import NativeEdgeSupportAuditIdentityError
from src.native_edge_support_identity import load_identity as load_native_audit_identity
from src.native_edge_support_report import classify_reachability, verify_record
from src.native_edge_support_reachability_gate_identity import (
    SUPPORTED_COMBINED_FALLBACK_CATEGORY,
    SUPPORTED_DECISION_MAPPING,
    SUPPORTED_DESIRED_CAUSE_CATEGORIES,
    SUPPORTED_NEXT_STAGE_IF_NOT_REACHABLE,
    SUPPORTED_NEXT_STAGE_IF_REACHABLE,
    SUPPORTED_PARENT_IMAGE_COUNT,
    SUPPORTED_PARENT_RUN_MODE,
    SUPPORTED_PARENT_SCHEMA,
    SUPPORTED_STRUCTURAL_STAGES,
    NativeEdgeSupportReachabilityGateIdentityError,
)


class NativeEdgeSupportReachabilityGateError(RuntimeError):
    """Raised on any reachability-gate invariant violation. Always fail
    closed: never silently substitute a default parent artifact,
    threshold, or decision mapping."""


# ---------------------------------------------------------------------------
# Section 3: parent-artifact discovery and binding
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_and_validate_parent_artifact(
    path: Path, *, repo_root: Path, gate_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Load and fully validate exactly one explicitly-supplied parent
    mechanics20 result. Never scans a directory or picks a file by mtime
    -- the caller must supply ``path`` directly (Section 3: "Do not
    silently scan and choose a file in production operation").

    Validates via the EXISTING native-audit verifier
    (:func:`src.native_edge_support_report.verify_record`) first -- this
    is the sole authority for the parent schema, including the parent's
    own decision reproduction -- then applies the additional gate-level
    binding constraints (run_mode, image count, complete/final, schema
    name, parent identity/source hash agreement)."""
    path = Path(path)
    if not path.is_file():
        raise NativeEdgeSupportReachabilityGateError(f"--audit-result path does not exist or is not a file: {path}")

    original_bytes = path.read_bytes()
    original_mtime_ns = path.stat().st_mtime_ns
    file_sha256 = hashlib.sha256(original_bytes).hexdigest()
    byte_size = len(original_bytes)

    record = parse_strict_json_document(path, label="parent mechanics20 result")

    parent_identity_section = gate_identity["parent_identity"]
    native_audit_identity_path = repo_root / parent_identity_section["native_audit_identity_path"]
    native_audit_identity = load_native_audit_identity(native_audit_identity_path, repo_root=repo_root)
    native_audit_identity_sha256 = sha256_file(native_audit_identity_path)
    if native_audit_identity_sha256 != parent_identity_section["native_audit_identity_sha256"]:
        raise NativeEdgeSupportReachabilityGateError(
            "native-edge-support-audit identity file SHA256 disagrees with the gate identity's recorded value"
        )

    # sole authority: reuse the parent's own verifier -- this independently
    # re-derives the parent's decision_output from its own funnel, checks
    # every histogram/cross-tab reconciliation the audit itself defines,
    # and confirms strict schema/type closure. Never re-implemented here.
    try:
        verify_record(record, native_audit_identity, identity_sha256=native_audit_identity_sha256)
    except NativeEdgeSupportAuditIdentityError as error:
        raise NativeEdgeSupportReachabilityGateError(f"parent result failed native-audit verification: {error}") from error

    contract = gate_identity["parent_contract"]
    if record["run_mode"] != contract["required_run_mode"] or record["run_mode"] != SUPPORTED_PARENT_RUN_MODE:
        raise NativeEdgeSupportReachabilityGateError(
            f"parent result run_mode must be {SUPPORTED_PARENT_RUN_MODE!r}, observed {record['run_mode']!r}"
        )
    if record["image_count_processed"] != contract["required_image_count"] or record["image_count_processed"] != SUPPORTED_PARENT_IMAGE_COUNT:
        raise NativeEdgeSupportReachabilityGateError(
            f"parent result image_count_processed must be exactly {SUPPORTED_PARENT_IMAGE_COUNT}, observed {record['image_count_processed']!r}"
        )
    if record["complete"] is not True:
        raise NativeEdgeSupportReachabilityGateError("parent result complete must be true")
    if record["final"] is not False:
        raise NativeEdgeSupportReachabilityGateError("parent result final must be false for mechanics20")
    if record["schema"] != contract["required_schema"] or record["schema"] != SUPPORTED_PARENT_SCHEMA:
        raise NativeEdgeSupportReachabilityGateError(f"parent result schema must be {SUPPORTED_PARENT_SCHEMA!r}")
    if record["identity"] != native_audit_identity["identity"]["name"]:
        raise NativeEdgeSupportReachabilityGateError("parent result identity name disagrees with the bound native-audit identity")
    if record["identity_sha256"] != native_audit_identity_sha256:
        raise NativeEdgeSupportReachabilityGateError("parent result identity_sha256 disagrees with the bound native-audit identity file")

    # never modify the parent file: verify byte-for-byte and mtime are unchanged
    if path.read_bytes() != original_bytes or path.stat().st_mtime_ns != original_mtime_ns:
        raise NativeEdgeSupportReachabilityGateError("parent artifact bytes or mtime changed during validation -- refusing to proceed")

    return {
        "record": record,
        "native_audit_identity": native_audit_identity,
        "native_audit_identity_sha256": native_audit_identity_sha256,
        "sha256": file_sha256,
        "byte_size": byte_size,
    }


# ---------------------------------------------------------------------------
# Section 4: independent aggregate reconciliation
# ---------------------------------------------------------------------------


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    value = numerator / denominator
    if not math.isfinite(value):
        raise NativeEdgeSupportReachabilityGateError("computed ratio is non-finite")
    return value


def reconcile_aggregate_ratios(record: Mapping[str, Any]) -> dict[str, Any]:
    """Independently recompute every headline ratio from raw counts and
    verify every required count invariant. Never trusts a precomputed
    ratio from the parent record without reproducing it from counts."""
    funnel = record["funnel"]
    cross_tabs = record["correctness_cross_tabs"]
    ignored = record["ignored_gt_count"]

    total_edges = funnel["directed_edges"]
    defined_edges = funnel["edges_with_observer"]
    undefined_edges = total_edges - defined_edges
    if defined_edges + undefined_edges != total_edges:
        raise NativeEdgeSupportReachabilityGateError("defined + undefined edge counts do not reconcile with total directed edges")
    if undefined_edges < 0 or defined_edges < 0:
        raise NativeEdgeSupportReachabilityGateError("negative edge count encountered during reconciliation")

    reasons = record["undefined_reason_counts"]
    if sum(reasons.values()) != undefined_edges:
        raise NativeEdgeSupportReachabilityGateError("undefined_reason_counts does not sum to the reconciled undefined edge count")
    if reasons["single_window_image"] > undefined_edges:
        raise NativeEdgeSupportReachabilityGateError("single_window_image count exceeds undefined edges")

    misclassified = funnel["misclassified_rows"]
    misclassified_defined = funnel["misclassified_rows_any_defined"]
    if misclassified_defined > misclassified:
        raise NativeEdgeSupportReachabilityGateError("misclassified_rows_any_defined exceeds misclassified_rows")

    for table_name in (
        "support_defined_vs_source_correct", "all_defined_vs_source_correct",
        "support_defined_vs_stitched_correct", "all_defined_vs_stitched_correct",
    ):
        table = cross_tabs[table_name]
        total = sum(table.values())
        expected_total = funnel["graph_rows"] - ignored
        if total != expected_total:
            raise NativeEdgeSupportReachabilityGateError(
                f"correctness_cross_tabs.{table_name} does not reconcile with graph_rows - ignored_gt_count"
            )

    if funnel["rows_all_defined"] > funnel["rows_any_defined"]:
        raise NativeEdgeSupportReachabilityGateError("rows_all_defined exceeds rows_any_defined")
    if funnel["rows_any_defined_zero_support"] > funnel["rows_any_defined"]:
        raise NativeEdgeSupportReachabilityGateError("rows_any_defined_zero_support exceeds rows_any_defined")
    if funnel["rows_with_unique_least_support"] + funnel["rows_tied_for_least_support"] != funnel["rows_any_defined"]:
        raise NativeEdgeSupportReachabilityGateError("unique + tied least-support rows do not reconcile with rows_any_defined")
    if funnel["clamped_windows"] + funnel["non_clamped_windows"] != funnel["windows"]:
        raise NativeEdgeSupportReachabilityGateError("clamped + non_clamped windows do not reconcile with windows")

    support_defined_valid_gt_source = cross_tabs["support_defined_vs_source_correct"]["defined_and_correct"] + cross_tabs["support_defined_vs_source_correct"]["defined_and_misclassified"]
    support_defined_wrong_fraction = _ratio(cross_tabs["support_defined_vs_source_correct"]["defined_and_misclassified"], support_defined_valid_gt_source)

    aligned_pairs = funnel["aligned_window_pairs"]
    unaligned_pairs = funnel["unaligned_window_pairs"]
    eligible_pairs = aligned_pairs + unaligned_pairs

    def unit(value: float | None) -> dict[str, Any]:
        return {"value": value, "unit": "fraction_0_1"}

    return {
        "total_directed_edges": {"value": total_edges, "unit": "count"},
        "support_defined_edges": {"value": defined_edges, "unit": "count"},
        "undefined_edges": {"value": undefined_edges, "unit": "count"},
        "undefined_edge_fraction": unit(_ratio(undefined_edges, total_edges)),
        "defined_edge_fraction": unit(_ratio(defined_edges, total_edges)),
        "wrong_row_reachability": unit(_ratio(misclassified_defined, misclassified)),
        "support_defined_wrong_fraction": unit(support_defined_wrong_fraction),
        "support_defined_wrong_fraction_denominator": {"value": support_defined_valid_gt_source, "unit": "count"},
        "clamped_window_fraction": unit(_ratio(funnel["clamped_windows"], funnel["windows"])),
        "aligned_pair_fraction": unit(_ratio(aligned_pairs, eligible_pairs)),
        "aligned_pair_fraction_denominator": {"value": eligible_pairs, "unit": "count"},
        "funnel_survival": {
            "rows_with_other_crop_of_graph_rows": unit(_ratio(funnel["rows_with_other_crop"], funnel["graph_rows"])),
            "rows_with_aligned_observer_of_rows_with_other_crop": unit(_ratio(funnel["rows_with_aligned_observer"], funnel["rows_with_other_crop"])),
            "rows_any_defined_of_rows_with_aligned_observer": unit(_ratio(funnel["rows_any_defined"], funnel["rows_with_aligned_observer"])),
            "rows_all_defined_of_rows_any_defined": unit(_ratio(funnel["rows_all_defined"], funnel["rows_any_defined"])),
        },
        "ignored_gt_fraction_of_graph_rows": unit(_ratio(ignored, funnel["graph_rows"])),
    }


# ---------------------------------------------------------------------------
# Section 5: undefined-support cause decomposition
# ---------------------------------------------------------------------------


def decompose_undefined_causes(record: Mapping[str, Any]) -> dict[str, Any]:
    """Report the most precise evidence the parent artifact actually
    supports -- never inferring finer subcategories from aggregate
    pair-level counts, never fabricating unavailable subcategories, and
    never rerunning the audit to fill a descriptive gap. The parent's
    ``undefined_reason_counts`` only distinguishes
    ``single_window_image`` from a single combined
    ``no_exactly_aligned_observer_covering_both_endpoints`` reason, so
    every case in Section 5's desired 6-category scheme that would
    require a finer per-edge cause code is marked unavailable and folded
    into the combined fallback category."""
    funnel = record["funnel"]
    reasons = record["undefined_reason_counts"]
    total_edges = funnel["directed_edges"]

    single_window = reasons["single_window_image"]
    combined_no_observer = reasons["no_exactly_aligned_observer_covering_both_endpoints"]
    support_defined = funnel["edges_with_observer"]

    partition_total = single_window + combined_no_observer + support_defined
    if partition_total != total_edges:
        raise NativeEdgeSupportReachabilityGateError(
            "cause-decomposition categories do not exactly partition total_directed_edges -- "
            f"got {partition_total}, expected {total_edges}"
        )

    def frac(count: int) -> float | None:
        return _ratio(count, total_edges)

    available_categories = {
        "single_window_no_alternative_view": {
            "count": single_window, "fraction_0_1": frac(single_window), "available": True,
            "provenance": "native_edge_support_audit.undefined_reason_counts.single_window_image",
        },
        SUPPORTED_COMBINED_FALLBACK_CATEGORY: {
            "count": combined_no_observer, "fraction_0_1": frac(combined_no_observer), "available": True,
            "provenance": "native_edge_support_audit.undefined_reason_counts.no_exactly_aligned_observer_covering_both_endpoints",
        },
        "support_defined": {
            "count": support_defined, "fraction_0_1": frac(support_defined), "available": True,
            "provenance": "native_edge_support_audit.funnel.edges_with_observer",
        },
    }
    unavailable_categories = [
        c for c in SUPPORTED_DESIRED_CAUSE_CATEGORIES
        if c not in available_categories and c != SUPPORTED_COMBINED_FALLBACK_CATEGORY
    ]
    # the 4 finer subcategories the parent cannot distinguish
    for name in (
        "multi_window_no_overlapping_alternative", "candidate_observer_origin_unaligned",
        "source_maps_but_destination_outside_shared_overlap", "other_exact_geometry_failure",
    ):
        available_categories[name] = {
            "count": None, "fraction_0_1": None, "available": False,
            "provenance": None,
            "reason": (
                "the parent audit records only a single combined "
                "'no_exactly_aligned_observer_covering_both_endpoints' undefined reason; "
                "it does not separately track whether the failure was a missing overlapping "
                "window, an unaligned candidate observer origin, or a source-maps/"
                "destination-does-not-map asymmetry. Folded into "
                f"'{SUPPORTED_COMBINED_FALLBACK_CATEGORY}'."
            ),
        }

    return {
        "mutually_exclusive": True,
        "categories": available_categories,
        "unavailable_categories": unavailable_categories,
        "limitations": (
            "Finer decomposition of the combined 'no observer' reason was not computed by the "
            "mechanics20 audit and is not inferred here from aggregate window/pair counts "
            "(doing so would misattribute individual edge failures to clamped-origin misalignment "
            "without per-edge evidence). Obtaining it would require re-instrumenting and rerunning "
            "the GPU audit, which this offline gate does not do."
        ),
    }


# ---------------------------------------------------------------------------
# Section 6: reachability gate decision
# ---------------------------------------------------------------------------


def reproduce_parent_decision(record: Mapping[str, Any], native_audit_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce the parent's own decision by calling the parent's own
    classifier (never a second, independently-derived threshold) against
    the parent's own funnel and ignored_gt_count."""
    reported = record["decision_output"]
    reproduced, rationale = classify_reachability(
        record["funnel"], native_audit_identity, ignored_gt_count=record["ignored_gt_count"],
    )
    if reproduced != reported:
        raise NativeEdgeSupportReachabilityGateError(
            f"gate-reproduced decision {reproduced!r} disagrees with the parent's own reported decision {reported!r}"
        )
    return {
        "parent_reported_decision": reported,
        "gate_reproduced_decision": reproduced,
        "match": reproduced == reported,
        "parent_decision_rationale": record["decision_rationale"],
        "gate_reproduced_rationale": rationale,
    }


def map_roadmap_authorization(decision_output: str, gate_identity: Mapping[str, Any]) -> dict[str, Any]:
    if decision_output not in SUPPORTED_DECISION_MAPPING:
        raise NativeEdgeSupportReachabilityGateError(f"unknown parent decision {decision_output!r}; refusing to map to a roadmap action")
    roadmap_action = gate_identity["decision_mapping"][decision_output]
    if roadmap_action != SUPPORTED_DECISION_MAPPING[decision_output]:
        raise NativeEdgeSupportReachabilityGateError("gate identity's decision_mapping disagrees with the locked mapping")

    authorized = roadmap_action == "PROCEED_TO_ELIGIBILITY_MATCHED_PRUNING"
    skip_reason = gate_identity["roadmap"]["skip_reason_template"]
    stages = [
        {
            "stage": stage,
            "authorized": authorized,
            "reason": None if authorized else skip_reason,
        }
        for stage in SUPPORTED_STRUCTURAL_STAGES
    ]
    next_stage = SUPPORTED_NEXT_STAGE_IF_REACHABLE if authorized else SUPPORTED_NEXT_STAGE_IF_NOT_REACHABLE

    return {
        "roadmap_action": roadmap_action,
        "stages": stages,
        "next_authorized_stage": next_stage,
    }


# ---------------------------------------------------------------------------
# Section 7: scientific wording
# ---------------------------------------------------------------------------

SCIENTIFIC_INTERPRETATION_ALIGNMENT_LIMITED = (
    "Native exact-alignment cross-view edge support is geometrically unavailable for most directed "
    "edges in the mechanics20 sample, and only a minority of currently misclassified rows are "
    "reachable. Under the preregistered mechanics decision, native-support pruning is not authorized."
)

REQUIRED_LIMITATIONS = (
    "this is not a pruning-efficacy result",
    "it does not prove all cross-view structural signals are useless",
    "it does not evaluate approximate alignment",
    "it does not authorize nearest-neighbour, bilinear or learned feature transport",
    "it does not say undefined edges are harmful or useful",
    "it is specific to the native exact-alignment definition and tested protocol",
    "the 20-image sample is a mechanics gate, not a dataset-level performance estimate",
)


def scientific_interpretation_for(decision_output: str) -> str:
    if decision_output == "ALIGNMENT_LIMITED":
        return SCIENTIFIC_INTERPRETATION_ALIGNMENT_LIMITED
    if decision_output == "REACHABLE":
        return (
            "Native exact-alignment cross-view edge support is geometrically defined for a meaningful "
            "fraction of currently misclassified rows in the mechanics20 sample. Under the preregistered "
            "mechanics decision, the roadmap may proceed to eligibility-matched structural pruning; this "
            "is still not itself a pruning-efficacy result."
        )
    if decision_output == "STRUCTURALLY_UNREACHABLE":
        return (
            "Native exact-alignment cross-view edge support is geometrically defined mainly where "
            "predictions are already correct, or too few misclassified rows are reachable, in the "
            "mechanics20 sample. Under the preregistered mechanics decision, native-support pruning is "
            "not authorized."
        )
    return (
        "The mechanics20 sample was insufficient for a conclusive native cross-view edge-support "
        "reachability read. Under the preregistered mechanics decision, native-support pruning is not "
        "authorized pending a conclusive result."
    )


# ---------------------------------------------------------------------------
# Section 8: deterministic report assembly
# ---------------------------------------------------------------------------


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def build_gate_report(
    *, gate_identity: Mapping[str, Any], gate_identity_sha256: str, parent: Mapping[str, Any],
    audit_result_path_reference: str, created_at_utc: str,
) -> dict[str, Any]:
    record = parent["record"]
    native_audit_identity = parent["native_audit_identity"]

    ratios = reconcile_aggregate_ratios(record)
    causes = decompose_undefined_causes(record)
    decision_repro = reproduce_parent_decision(record, native_audit_identity)
    roadmap = map_roadmap_authorization(decision_repro["parent_reported_decision"], gate_identity)
    interpretation = scientific_interpretation_for(decision_repro["parent_reported_decision"])

    parent_label = f"mechanics20-{record['git_commit'][:12]}-{record['image_order_digest'][:12]}"

    deterministic_payload = {
        "schema": gate_identity["run_mode"]["schema_name"],
        "gate_identity": gate_identity["identity"]["name"],
        "gate_identity_sha256": gate_identity_sha256,
        "native_audit_identity_sha256": parent["native_audit_identity_sha256"],
        "parent_artifact": {
            "label": parent_label,
            "sha256": parent["sha256"],
            "byte_size": parent["byte_size"],
            "identity": record["identity"],
            "identity_sha256": record["identity_sha256"],
            "source_git_commit": record["git_commit"],
            "image_order_digest": record["image_order_digest"],
            "run_mode": record["run_mode"],
            "image_count_processed": record["image_count_processed"],
        },
        "validation_summary": {
            "strict_json": True,
            "schema_matches": True,
            "complete": record["complete"],
            "final": record["final"],
            "run_mode_matches": True,
            "image_count_matches": True,
            "parent_verify_record_passed": True,
            "count_reconciliation_passed": True,
        },
        "reconstructed_counts_ratios": ratios,
        "cause_decomposition": causes,
        "reachability_summary": {
            "misclassified_rows": {"value": record["funnel"]["misclassified_rows"], "unit": "count"},
            "misclassified_rows_any_defined": {"value": record["funnel"]["misclassified_rows_any_defined"], "unit": "count"},
            "wrong_row_reachability_percent": {
                "value": None if ratios["wrong_row_reachability"]["value"] is None else round(ratios["wrong_row_reachability"]["value"] * 100.0, 6),
                "unit": "percent_0_100",
            },
        },
        "parent_decision_reproduction": decision_repro,
        "roadmap_authorization": roadmap,
        "stop_proceed_decision": roadmap["roadmap_action"],
        "scientific_interpretation": interpretation,
        "limitations": list(REQUIRED_LIMITATIONS),
    }
    content_digest = hashlib.sha256(_canonical_json_bytes(deterministic_payload)).hexdigest()

    report = dict(deterministic_payload)
    report["content_digest"] = content_digest
    report["audit_result_path_reference"] = audit_result_path_reference
    report["created_at_utc"] = created_at_utc
    return report


__all__ = [
    "NativeEdgeSupportReachabilityGateError",
    "REQUIRED_LIMITATIONS",
    "build_gate_report",
    "decompose_undefined_causes",
    "map_roadmap_authorization",
    "reconcile_aggregate_ratios",
    "reproduce_parent_decision",
    "scientific_interpretation_for",
    "select_and_validate_parent_artifact",
    "sha256_file",
]
