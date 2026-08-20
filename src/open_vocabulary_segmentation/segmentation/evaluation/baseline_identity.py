"""Typed, versioned identity/manifest for the stitching/k-sweep/DCR/SUR
baseline suite (Section 5).

This manifest is the sole authority for scientific baseline choices; tests
and the evaluation harness consume it rather than duplicating literal
values. Every scientific field (crop, stride, alpha, top_k, affinity_power,
solver tolerances, class count, checkpoint hash, ...) is derived from the
already-verified canonical RWR/E3 identities, never hand-typed here.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .stitching_baselines import REQUIRED_SCORE_SOURCE_MODE_MATRIX, STITCH_MODES
from .consensus_replacement import REPLACEMENT_KINDS
from .graph_degree_sweep import DEFAULT_K_VALUES


class BaselineIdentityError(ValueError):
    """Raised when the baseline manifest schema or types are invalid."""


BASELINE_MANIFEST_SCHEMA_VERSION = "talk2dino-baseline-suite-manifest-v1"

STAGE_SIGMOID_PROBABILITY = "post_rwr_post_sigmoid_pre_upsample"
UNIT_PERCENT_0_TO_100 = "percent_0_to_100"
UNIT_PERCENTAGE_POINTS = "percentage_points"
UNIT_FRACTION_0_TO_1 = "fraction_0_to_1"
UNIT_COUNT = "count"

_TOP_LEVEL_KEYS = frozenset(
    {
        "format_version",
        "identity_name",
        "e3_identity_name",
        "e3_identity_sha256",
        "rwr_identity_name",
        "rwr_identity_sha256",
        "canonical_config_sha256",
        "checkpoint_sha256",
        "dataset",
        "evaluation",
        "rwr",
        "solver",
        "k_sweep",
        "stitching",
        "dcr",
        "sur",
        "strict_safe",
        "t4_target_definition",
        "metric_units",
        "runtime",
        "bootstrap",
    }
)


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise BaselineIdentityError(f"{label} must be an exact non-empty string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise BaselineIdentityError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise BaselineIdentityError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise BaselineIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_exact_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    if type(value) is not float:
        raise BaselineIdentityError(f"{label} must be an exact float")
    if not math.isfinite(value):
        raise BaselineIdentityError(f"{label} must be finite")
    if minimum is not None and value < minimum:
        raise BaselineIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise BaselineIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_int_tuple(value: Any, label: str, *, strictly_increasing: bool = False) -> tuple[int, ...]:
    if type(value) is not tuple or not value:
        raise BaselineIdentityError(f"{label} must be a non-empty exact tuple")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise BaselineIdentityError(f"{label} elements must be positive exact integers")
    if len(set(value)) != len(value):
        raise BaselineIdentityError(f"{label} must not contain duplicates")
    if strictly_increasing and list(value) != sorted(value):
        raise BaselineIdentityError(f"{label} must be strictly increasing")
    return value


def validate_baseline_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed schema/type validation. Rejects unknown fields and wrong
    exact types; never coerces (e.g. bool is not accepted where int/float
    is required, matching the RWR/E3 identity validators' convention)."""
    if not isinstance(manifest, Mapping):
        raise BaselineIdentityError("baseline manifest must be a mapping")
    if set(manifest) != _TOP_LEVEL_KEYS:
        unknown = set(manifest) - _TOP_LEVEL_KEYS
        missing = _TOP_LEVEL_KEYS - set(manifest)
        raise BaselineIdentityError(
            f"baseline manifest has an unexpected schema (unknown={sorted(unknown)}, "
            f"missing={sorted(missing)})"
        )
    _require_exact_string(manifest["format_version"], "format_version")
    if manifest["format_version"] != BASELINE_MANIFEST_SCHEMA_VERSION:
        raise BaselineIdentityError("unsupported baseline manifest format_version")
    _require_exact_string(manifest["identity_name"], "identity_name")
    _require_exact_string(manifest["e3_identity_name"], "e3_identity_name")
    _require_sha256(manifest["e3_identity_sha256"], "e3_identity_sha256")
    _require_exact_string(manifest["rwr_identity_name"], "rwr_identity_name")
    _require_sha256(manifest["rwr_identity_sha256"], "rwr_identity_sha256")
    _require_sha256(manifest["canonical_config_sha256"], "canonical_config_sha256")
    _require_sha256(manifest["checkpoint_sha256"], "checkpoint_sha256")

    dataset = manifest["dataset"]
    if not isinstance(dataset, Mapping) or set(dataset) != {"name", "images", "classes", "ignore_label"}:
        raise BaselineIdentityError("dataset section has an unexpected schema")
    _require_exact_string(dataset["name"], "dataset.name")
    _require_exact_int(dataset["images"], "dataset.images", minimum=1)
    _require_exact_int(dataset["classes"], "dataset.classes", minimum=1)
    _require_exact_int(dataset["ignore_label"], "dataset.ignore_label", minimum=0)

    evaluation = manifest["evaluation"]
    if not isinstance(evaluation, Mapping) or set(evaluation) != {"crop", "stride", "score_stage"}:
        raise BaselineIdentityError("evaluation section has an unexpected schema")
    for name in ("crop", "stride"):
        pair = evaluation[name]
        if type(pair) is not list or len(pair) != 2 or any(type(v) is not int for v in pair):
            raise BaselineIdentityError(f"evaluation.{name} must be an exact two-element integer list")
    _require_exact_string(evaluation["score_stage"], "evaluation.score_stage")
    if evaluation["score_stage"] != STAGE_SIGMOID_PROBABILITY:
        raise BaselineIdentityError("evaluation.score_stage must be the canonical post-sigmoid stage")

    rwr = manifest["rwr"]
    if not isinstance(rwr, Mapping) or set(rwr) != {"alpha", "affinity_power"}:
        raise BaselineIdentityError("rwr section has an unexpected schema")
    _require_exact_float(rwr["alpha"], "rwr.alpha", minimum=0.0)
    if not rwr["alpha"] < 1:
        raise BaselineIdentityError("rwr.alpha must satisfy 0 <= alpha < 1")
    _require_exact_float(rwr["affinity_power"], "rwr.affinity_power", minimum=0.0)
    if rwr["affinity_power"] <= 0:
        raise BaselineIdentityError("rwr.affinity_power must be strictly positive")

    solver = manifest["solver"]
    if not isinstance(solver, Mapping) or set(solver) != {"method", "rtol", "atol", "max_iterations"}:
        raise BaselineIdentityError("solver section has an unexpected schema")
    _require_exact_string(solver["method"], "solver.method")
    if solver["method"] != "cgls":
        raise BaselineIdentityError("solver.method must be the verified cgls solver")
    _require_exact_float(solver["rtol"], "solver.rtol", minimum=0.0)
    _require_exact_float(solver["atol"], "solver.atol", minimum=0.0)
    _require_exact_int(solver["max_iterations"], "solver.max_iterations", minimum=1)

    k_sweep = manifest["k_sweep"]
    if not isinstance(k_sweep, Mapping) or set(k_sweep) != {"canonical_k", "values"}:
        raise BaselineIdentityError("k_sweep section has an unexpected schema")
    _require_exact_int(k_sweep["canonical_k"], "k_sweep.canonical_k", minimum=1)
    values = _require_int_tuple(k_sweep["values"], "k_sweep.values", strictly_increasing=True)
    if k_sweep["canonical_k"] not in values:
        raise BaselineIdentityError("k_sweep.values must include canonical_k")

    stitching = manifest["stitching"]
    if not isinstance(stitching, Mapping) or set(stitching) != {"modes", "score_source_mode_matrix"}:
        raise BaselineIdentityError("stitching section has an unexpected schema")
    modes = stitching["modes"]
    if type(modes) is not tuple or set(modes) != set(STITCH_MODES):
        raise BaselineIdentityError("stitching.modes must exactly match the implemented stitch modes")
    matrix = stitching["score_source_mode_matrix"]
    if type(matrix) is not tuple or set(matrix) != set(REQUIRED_SCORE_SOURCE_MODE_MATRIX):
        raise BaselineIdentityError(
            "stitching.score_source_mode_matrix must exactly match the required section-7.5 matrix"
        )

    dcr = manifest["dcr"]
    if not isinstance(dcr, Mapping) or set(dcr) != {"variants", "target_population"}:
        raise BaselineIdentityError("dcr section has an unexpected schema")
    if type(dcr["variants"]) is not tuple or set(dcr["variants"]) != {"dcr_hard", "dcr_jury_mean"}:
        raise BaselineIdentityError("dcr.variants must be exactly {dcr_hard, dcr_jury_mean}")
    _require_exact_string(dcr["target_population"], "dcr.target_population")
    if dcr["target_population"] != "strict_t4":
        raise BaselineIdentityError("dcr.target_population must default to strict_t4")

    sur = manifest["sur"]
    if not isinstance(sur, Mapping) or set(sur) != {"definition", "target_population"}:
        raise BaselineIdentityError("sur section has an unexpected schema")
    _require_exact_string(sur["definition"], "sur.definition")
    _require_exact_string(sur["target_population"], "sur.target_population")
    if sur["target_population"] != "strict_t4":
        raise BaselineIdentityError("sur.target_population must default to strict_t4")

    strict_safe = manifest["strict_safe"]
    if not isinstance(strict_safe, Mapping) or set(strict_safe) != {
        "acceptance_rule", "candidate_order", "protected_set_definition",
    }:
        raise BaselineIdentityError("strict_safe section has an unexpected schema")
    for name in ("acceptance_rule", "candidate_order", "protected_set_definition"):
        _require_exact_string(strict_safe[name], f"strict_safe.{name}")

    _require_exact_string(manifest["t4_target_definition"], "t4_target_definition")

    metric_units = manifest["metric_units"]
    if not isinstance(metric_units, Mapping) or set(metric_units) != {"metrics", "gains", "fractions"}:
        raise BaselineIdentityError("metric_units section has an unexpected schema")
    if metric_units["metrics"] != UNIT_PERCENT_0_TO_100:
        raise BaselineIdentityError("metric_units.metrics must be percent_0_to_100")
    if metric_units["gains"] != UNIT_PERCENTAGE_POINTS:
        raise BaselineIdentityError("metric_units.gains must be percentage_points")
    if metric_units["fractions"] != UNIT_FRACTION_0_TO_1:
        raise BaselineIdentityError("metric_units.fractions must be fraction_0_to_1")

    runtime = manifest["runtime"]
    if not isinstance(runtime, Mapping) or set(runtime) != {"device_preference", "gate"}:
        raise BaselineIdentityError("runtime section has an unexpected schema")
    _require_exact_string(runtime["device_preference"], "runtime.device_preference")
    _require_exact_string(runtime["gate"], "runtime.gate")
    if runtime["gate"] not in {"A", "B", "C", "D", "E"}:
        raise BaselineIdentityError("runtime.gate must be one of A, B, C, D, E")

    bootstrap = manifest["bootstrap"]
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != {"reused", "resamples", "seed"}:
        raise BaselineIdentityError("bootstrap section has an unexpected schema")
    _require_exact_bool(bootstrap["reused"], "bootstrap.reused")
    if bootstrap["reused"]:
        _require_exact_int(bootstrap["resamples"], "bootstrap.resamples", minimum=1)
        _require_exact_int(bootstrap["seed"], "bootstrap.seed", minimum=0)
    else:
        if bootstrap["resamples"] is not None or bootstrap["seed"] is not None:
            raise BaselineIdentityError("bootstrap.resamples/seed must be null when reused is false")

    return dict(manifest)


def build_baseline_manifest(
    *,
    rwr_identity: Mapping[str, Any],
    canonical_config_sha256: str,
    checkpoint_sha256: str,
    identity_name: str = "stitching-kdcr-baseline-suite",
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    gate: str = "A",
) -> dict[str, Any]:
    """Derive the baseline manifest's scientific fields directly from the
    already-verified canonical RWR identity -- never hand-typed here."""
    manifest = {
        "format_version": BASELINE_MANIFEST_SCHEMA_VERSION,
        "identity_name": identity_name,
        "e3_identity_name": rwr_identity["identity_name"],
        "e3_identity_sha256": canonical_config_sha256,
        "rwr_identity_name": rwr_identity["identity_name"],
        "rwr_identity_sha256": canonical_config_sha256,
        "canonical_config_sha256": canonical_config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset": {
            "name": rwr_identity["dataset"]["name"],
            "images": rwr_identity["dataset"]["images"],
            "classes": rwr_identity["dataset"]["classes"],
            "ignore_label": 255,
        },
        "evaluation": {
            "crop": list(rwr_identity["evaluation"]["crop"]),
            "stride": list(rwr_identity["evaluation"]["stride"]),
            "score_stage": STAGE_SIGMOID_PROBABILITY,
        },
        "rwr": {
            "alpha": rwr_identity["rwr"]["alpha"],
            "affinity_power": rwr_identity["rwr"]["affinity_power"],
        },
        "solver": {
            "method": rwr_identity["solver"]["method"],
            "rtol": rwr_identity["solver"]["rtol"],
            "atol": rwr_identity["solver"]["atol"],
            "max_iterations": rwr_identity["solver"]["max_iterations"],
        },
        "k_sweep": {
            "canonical_k": rwr_identity["rwr"]["top_k"],
            "values": tuple(sorted(set(k_values) | {rwr_identity["rwr"]["top_k"]})),
        },
        "stitching": {
            "modes": tuple(STITCH_MODES),
            "score_source_mode_matrix": tuple(REQUIRED_SCORE_SOURCE_MODE_MATRIX),
        },
        "dcr": {
            "variants": tuple(sorted(k for k in REPLACEMENT_KINDS if k != "sur")),
            "target_population": "strict_t4",
        },
        "sur": {
            "definition": "sigmoid(S0_w(i)) replacing only the source post-RWR probability row",
            "target_population": "strict_t4",
        },
        "strict_safe": {
            "acceptance_rule": "E(G_after) proper-subset E(G_before) over the protected T2 anchor set",
            "candidate_order": "(image_id, source_window_ordinal, source_node_index, replacement_kind)",
            "protected_set_definition": "every T2-or-higher ConsensusObservation from the original T4 audit signal",
        },
        "t4_target_definition": (
            "segmentation.evaluation.t4_audit.ConsensusObservation.t4 "
            "(strict T4: actionable and source_unary_label == consensus_label)"
        ),
        "metric_units": {
            "metrics": UNIT_PERCENT_0_TO_100,
            "gains": UNIT_PERCENTAGE_POINTS,
            "fractions": UNIT_FRACTION_0_TO_1,
        },
        "runtime": {"device_preference": "cuda_if_available_else_cpu", "gate": gate},
        "bootstrap": {"reused": False, "resamples": None, "seed": None},
    }
    return validate_baseline_manifest(manifest)


__all__ = [
    "BaselineIdentityError",
    "BASELINE_MANIFEST_SCHEMA_VERSION",
    "STAGE_SIGMOID_PROBABILITY",
    "UNIT_PERCENT_0_TO_100",
    "UNIT_PERCENTAGE_POINTS",
    "UNIT_FRACTION_0_TO_1",
    "UNIT_COUNT",
    "validate_baseline_manifest",
    "build_baseline_manifest",
]
